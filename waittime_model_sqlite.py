import argparse, sqlite3, sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

DB_PATH_DEFAULT = 'full.db'
Q_MAX = 10  # maximum inverse-respend bin

def stream_sql(path, conn):
    buf = []
    with open(path, 'r') as f:
        for line in f:
            up = line.strip().upper()
            if up.startswith(('BEGIN', 'COMMIT', 'PRAGMA')):
                continue
            buf.append(line)
            if line.strip().endswith(';'):
                stmt = ''.join(buf); buf.clear()
                try:
                    conn.executescript(stmt)
                except Exception as e:
                    print(f"skipped bad stmt: {e}", file=sys.stderr)
    if buf:
        stmt = ''.join(buf)
        up = stmt.strip().upper()
        if not up.startswith(('BEGIN','COMMIT','PRAGMA')):
            try:
                conn.executescript(stmt)
            except Exception as e:
                print(f"failed trailing stmt: {e}", file=sys.stderr)

def load_data(db_path, tx_dump, rbf_dump):
    try:
        conn = sqlite3.connect(db_path)
        txdf = pd.read_sql_query("""
            SELECT tx_id, inputs_hash, found_at, mined_at, parent_txid
              FROM transactions
             WHERE mined_at IS NOT NULL
        """, conn)
        rbfd = pd.read_sql_query("SELECT inputs_hash FROM rbf;", conn)
        conn.close()
        print(f"Loaded {len(txdf)} tx & {len(rbfd)} rbf rows from {db_path}")
    except Exception as e:
        print(f"Direct read failed: {e}", file=sys.stderr)
        print("Rebuilding in memory from SQL dumps…", file=sys.stderr)
        conn = sqlite3.connect(":memory:", isolation_level=None)
        stream_sql(tx_dump, conn)
        stream_sql(rbf_dump, conn)
        txdf = pd.read_sql_query("""
            SELECT tx_id, inputs_hash, found_at, mined_at, parent_txid
              FROM transactions
             WHERE mined_at IS NOT NULL
        """, conn)
        rbfd = pd.read_sql_query("SELECT inputs_hash FROM rbf;", conn)
        conn.close()
        print(f"In-memory rebuild: {len(txdf)} tx & {len(rbfd)} rbf rows", file=sys.stderr)

    # mark replace missing found_at with mined_at (instant confirm)
    txdf['found_at'] = txdf['found_at'].mask(txdf['found_at']==0, txdf['mined_at'])

    # merge in rbf_flag
    rbfd['rbf_flag'] = 1
    txdf = txdf.merge(rbfd[['inputs_hash','rbf_flag']], on='inputs_hash', how='left')
    txdf['rbf_flag'] = txdf['rbf_flag'].fillna(0).astype(int)
    return txdf

def compute_metrics(txdf, delta):
    # compute true wait-time in minutes
    txdf['found_dt'] = pd.to_datetime(txdf['found_at'], unit='s', origin='unix')
    txdf['mined_dt'] = pd.to_datetime(txdf['mined_at'], unit='s', origin='unix')
    txdf['waittime'] = (txdf['mined_dt'] - txdf['found_dt']).dt.total_seconds() / 60.0

    # identify first-child-pays-for-parent (CPFP)
    parent = txdf[['tx_id','mined_at']].rename(columns={'tx_id':'parent_tx','mined_at':'parent_mined'})
    child  = txdf[['parent_txid','mined_at']].rename(columns={'parent_txid':'parent_tx','mined_at':'child_mined'})
    merged = child.merge(parent, on='parent_tx', how='inner')
    fc = (
        merged.groupby('parent_tx')['child_mined']
              .min()
              .reset_index()
              .rename(columns={'parent_tx':'tx_id','child_mined':'first_child_mined'})
    )
    txdf = txdf.merge(fc, on='tx_id', how='left')

    # censor non-spenders & compute inverse_respend
    max_mined = txdf['mined_at'].max()
    txdf['cpfp_flag'] = txdf['first_child_mined'].notnull().astype(int)
    txdf['respend_time'] = np.where(
        txdf['cpfp_flag']==1,
        txdf['first_child_mined'] - txdf['mined_at'],
        max_mined - txdf['mined_at']
    )
    txdf['inverse_respend'] = 1.0 / txdf['respend_time'].replace(0, np.nan)

    # bin inverse_respend into 1..Q_MAX
    raw_q = np.ceil(txdf['inverse_respend']/delta).fillna(0)
    txdf['bin'] = raw_q.clip(lower=1, upper=Q_MAX).astype(int)

    return txdf

def compute_epoch_stats(txdf, bins):
    txdf['epoch'] = pd.cut(txdf['found_at'], bins=bins, labels=False)

    tot  = txdf.groupby('epoch')['tx_id'].size().rename('total').reset_index()
    cong = tot.rename(columns={'total':'congestion'})

    cnt = txdf.groupby(['epoch','bin'])['tx_id'].size().rename('count').reset_index()
    sigma = cnt.merge(tot, on='epoch')
    sigma['sigma'] = sigma['count']/sigma['total']
    sigma = sigma.sort_values(['epoch','bin'])
    sigma['F'] = sigma.groupby('epoch')['sigma'] \
                      .transform(lambda x: x.iloc[::-1].cumsum().iloc[::-1])

    txdf = txdf.merge(cong, on='epoch')
    txdf = txdf.merge(sigma[['epoch','bin','sigma','F']], on=['epoch','bin'], how='left')
    return txdf

def plot_distribution(txdf):
    plt.figure(figsize=(6,4))
    plt.hist(txdf['waittime'], bins=50)
    plt.yscale('log')
    plt.xlabel('Wait-time (mins)')
    plt.ylabel('Frequency (log scale)')
    plt.title('Wait-time Distribution')
    plt.tight_layout()
    plt.show()

def run_regression(txdf, zoom=None):
    features = ['congestion','F','rbf_flag','cpfp_flag']
    X = txdf[features]
    y = txdf['waittime']
    model = LinearRegression().fit(X, y)
    r2    = model.score(X, y)

    print("\nRegression results:")
    print(f"  Intercept: {model.intercept_:.4f}")
    for f,c in zip(features, model.coef_):
        print(f"  Beta_{f}: {c:.4f}")
    print(f"  R² = {r2:.4f}\n")

    # cell-level scatter
    yhat = model.predict(X)
    plt.figure(figsize=(5,5))
    plt.scatter(y, yhat, alpha=0.3)
    plt.xlabel('Observed wait-time (mins)')
    plt.ylabel('Predicted wait-time (mins)')
    plt.title(f'Cell-level Observed vs Predicted (R²={r2:.2f})')
    plt.grid(True)
    if zoom: plt.xlim(0,zoom); plt.ylim(0,zoom)
    plt.tight_layout()
    plt.show()

    # binned (epoch,bin) aggregate scatter
    agg = (
        txdf
        .groupby(['epoch','bin'])
        .agg({
            'waittime':'mean',
            **{f:'mean' for f in features}
        })
        .reset_index()
    )
    agg['pred'] = model.predict(agg[features])
    plt.figure(figsize=(5,5))
    plt.scatter(agg['waittime'], agg['pred'], alpha=0.7)
    plt.xlabel('Mean Observed Wait (mins)')
    plt.ylabel('Mean Predicted Wait (mins)')
    plt.title('Binned Observed vs Predicted')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db-path',  default=DB_PATH_DEFAULT)
    p.add_argument('--tx-dump',  required=True)
    p.add_argument('--rbf-dump', required=True)
    p.add_argument('--bins',     type=int,   default=10)
    p.add_argument('--delta',    type=float, default=1.0)
    p.add_argument('--zoom',     type=float, help="limit axes to [0,zoom]")
    args = p.parse_args()

    print(f"RUNNING: {sys.argv[0]}")
    txdf = load_data(args.db_path, args.tx_dump, args.rbf_dump)
    txdf = compute_metrics(txdf, delta=args.delta)
    txdf = compute_epoch_stats(txdf, bins=args.bins)

    # show distribution
    plot_distribution(txdf)

    # fit & plot
    run_regression(txdf, zoom=args.zoom)

if __name__=='__main__':
    main()
