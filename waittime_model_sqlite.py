import argparse, sqlite3, sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

DB_PATH_DEFAULT = 'full.db'
Q_MAX = 10  # maximum inverse-respend bin

def stream_sql(path, conn):
    """
    Incrementally execute the SQL statements in a .sql dump file.
    This reads line-by-line, buffers until it sees a semicolon,
    skips transaction-control statements (BEGIN/COMMIT/PRAGMA),
    and executes each complete statement via conn.executescript().
    """
    buf = []
    with open(path, 'r') as f:
        for line in f:
            up = line.strip().upper()
            # skip dump file transaction markers
            if up.startswith(('BEGIN', 'COMMIT', 'PRAGMA')):
                continue
            buf.append(line)
            if line.strip().endswith(';'):
                stmt = ''.join(buf)
                buf.clear()
                try:
                    conn.executescript(stmt)
                except Exception as e:
                    # warn but continue on errors
                    print(f"failed to exec statement (skipping): {stmt[:50]!r}… → {e}",
                          file=sys.stderr)
    # execute any remaining buffered SQL
    if buf:
        stmt = ''.join(buf)
        up = stmt.strip().upper()
        if not up.startswith(('BEGIN', 'COMMIT', 'PRAGMA')):
            try:
                conn.executescript(stmt)
            except Exception as e:
                print(f"failed trailing statement: {e}", file=sys.stderr)

def load_data(db_path, tx_dump, rbf_dump):
    """
    Attempt to load transactions and RBF flags directly from an on-disk SQLite file.
    If that fails (e.g. missing tables or corrupted), rebuild an in-memory DB from
    the provided SQL dump files, then re-load.
    """
    try:
        conn = sqlite3.connect(db_path)
        # select only rows with a known mined_at timestamp; child_txid replaced parent_txid
        txdf = pd.read_sql_query("""
            SELECT
              tx_id,
              inputs_hash,
              found_at,
              mined_at,
              child_txid
            FROM transactions
            WHERE mined_at IS NOT NULL
        """, conn)
        # load the list of RBF-marked inputs
        rbfd = pd.read_sql_query("SELECT inputs_hash FROM rbf;", conn)
        conn.close()
        print(f"Direct load: {len(txdf)} tx & {len(rbfd)} rbf rows from {db_path}")
    except Exception as e:
        # fallback: rebuild in-memory from dump files
        print(f"Direct read failed: {e}", file=sys.stderr)
        print("Rebuilding in-memory DB from SQL dumps…", file=sys.stderr)
        conn = sqlite3.connect(":memory:", isolation_level=None)
        stream_sql(tx_dump, conn)
        stream_sql(rbf_dump, conn)
        # re-run the same queries on the in-memory DB
        txdf = pd.read_sql_query("""
            SELECT
              tx_id,
              inputs_hash,
              found_at,
              mined_at,
              child_txid
            FROM transactions
            WHERE mined_at IS NOT NULL
        """, conn)
        rbfd = pd.read_sql_query("SELECT inputs_hash FROM rbf;", conn)
        conn.close()
        print(f"In-memory rebuild: {len(txdf)} tx & {len(rbfd)} rbf rows",
              file=sys.stderr)

    # mark RBF transactions: merge and fill missing flags with 0
    rbfd['rbf_flag'] = 1
    txdf = txdf.merge(rbfd[['inputs_hash','rbf_flag']],
                      on='inputs_hash', how='left')
    txdf['rbf_flag'] = txdf['rbf_flag'].fillna(0).astype(int)
    return txdf

def compute_metrics(txdf, delta):
    """
    Compute various per-transaction metrics:
      1) Treat found_at == 0 (unseen in mempool) as instant confirmation.
      2) Calculate actual wait-time in minutes between found_at and mined_at.
      3) Identify first-child-pays-for-parent (CPFP) transactions.
      4) Compute re-spend lag and its inverse.
      5) Bin the inverse respend into discrete quantile bins 1..Q_MAX.
    """
    # Instant confirmation when no mempool timestamp, do not know if this is actually right
    txdf['found_at'] = txdf['found_at'].mask(
        txdf['found_at'] == 0,
        txdf['mined_at']
    )

    # convert Unix seconds to datetime, then compute wait-time (minutes)
    txdf['found_dt'] = pd.to_datetime(txdf['found_at'], unit='s', origin='unix')
    txdf['mined_dt'] = pd.to_datetime(txdf['mined_at'], unit='s', origin='unix')
    txdf['waittime'] = (txdf['mined_dt'] - txdf['found_dt']).dt.total_seconds() / 60.0

    # CPFP: find the earliest child transaction per parent
    parent = txdf[['tx_id','mined_at']].rename(
        columns={'tx_id':'parent_tx','mined_at':'parent_mined'}
    )
    child  = txdf[['child_txid','mined_at']].rename(
        columns={'child_txid':'parent_tx','mined_at':'child_mined'}
    )
    merged = child.merge(parent, on='parent_tx', how='inner')
    fc = (
        merged.groupby('parent_tx')['child_mined']
              .min()
              .reset_index()
              .rename(columns={'parent_tx':'tx_id','child_mined':'first_child_mined'})
    )
    txdf = txdf.merge(fc, on='tx_id', how='left')

    # censor non-spenders, compute re-spend lag
    max_mined = txdf['mined_at'].max()
    txdf['cpfp_flag'] = txdf['first_child_mined'].notnull().astype(int)
    txdf['respend_time'] = np.where(
        txdf['cpfp_flag']==1,
        txdf['first_child_mined'] - txdf['mined_at'],
        max_mined - txdf['mined_at']
    )
    # avoid division by zero
    txdf['inverse_respend'] = 1.0 / txdf['respend_time'].replace(0, np.nan)

    # quantile-bin the inverse respend values into [1..Q_MAX]
    raw_q = np.ceil(txdf['inverse_respend'] / delta).fillna(0)
    txdf['bin'] = raw_q.clip(lower=1, upper=Q_MAX).astype(int)

    return txdf

def compute_epoch_stats(txdf, bins):
    """
    Aggregate per-(epoch,bin) statistics:
      - total tx per epoch → congestion
      - fraction per bin → sigma
      - upper-tail fraction F
    Then merge those back into txdf for modeling.
    """
    txdf['epoch'] = pd.cut(txdf['found_at'], bins=bins, labels=False)

    # # transactions per epoch
    tot = txdf.groupby('epoch')['tx_id'].size().rename('total').reset_index()
    cong = tot.rename(columns={'total':'congestion'})

    # count per (epoch,bin)
    cnt = txdf.groupby(['epoch','bin'])['tx_id'] \
             .size().rename('count').reset_index()
    # compute sigma = count / total
    sigma = cnt.merge(tot, on='epoch')
    sigma['sigma'] = sigma['count'] / sigma['total']
    sigma = sigma.sort_values(['epoch','bin'])
    # cumulative upper-tail fraction F
    sigma['F'] = sigma.groupby('epoch')['sigma'] \
                      .transform(lambda x: x.iloc[::-1].cumsum().iloc[::-1])

    # merge congestion and sigma, F back into txdf
    txdf = txdf.merge(cong, on='epoch')
    txdf = txdf.merge(sigma[['epoch','bin','sigma','F']],
                      on=['epoch','bin'], how='left')
    return txdf

def run_regression(txdf, zoom=None):
    """
    Fit a simple linear regression of waittime on:
      [ congestion, F, rbf_flag, cpfp_flag ]
    Then plot observed vs. predicted wait-times.
    """
    X = txdf[['congestion','F','rbf_flag','cpfp_flag']]
    y = txdf['waittime']
    model = LinearRegression().fit(X, y)
    r2    = model.score(X, y)

    # print coefficients
    print("\nRegression results:")
    print(f"  Intercept: {model.intercept_:.4f}")
    for feat, coef in zip(X.columns, model.coef_):
        print(f"Beta_{feat}: {coef:.4f}")
    print(f"R² = {r2:.4f}\n")

    # scatter plot observed vs. predicted
    yhat = model.predict(X)
    plt.figure(figsize=(6,6))
    plt.scatter(y, yhat, alpha=0.3)
    plt.xlabel('Observed wait-time (mins)')
    plt.ylabel('Predicted wait-time (mins)')
    plt.title(f'Wait-time Model (R²={r2:.2f})')
    plt.grid(True)
    if zoom is not None:
        plt.xlim(0, zoom)
        plt.ylim(0, zoom)
    plt.tight_layout()
    plt.show()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db-path',  default=DB_PATH_DEFAULT)
    p.add_argument('--tx-dump',  required=True)
    p.add_argument('--rbf-dump', required=True)
    p.add_argument('--bins',     type=int,   default=10)
    p.add_argument('--delta',    type=float, default=1.0)
    p.add_argument('--zoom',     type=float)
    args = p.parse_args()

    print(f"RUNNING: {sys.argv[0]}")
    # load data (possibly from dumps)
    txdf = load_data(args.db_path, args.tx_dump, args.rbf_dump)
    # compute per-tx metrics
    txdf = compute_metrics(txdf, delta=args.delta)
    # compute per-epoch/bin aggregates
    txdf = compute_epoch_stats(txdf, bins=args.bins)
    # fit model and plot
    run_regression(txdf, zoom=args.zoom)

if __name__=='__main__':
    main()