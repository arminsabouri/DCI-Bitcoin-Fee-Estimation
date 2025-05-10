import argparse, sqlite3, sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split

matplotlib.use('TkAgg')
pd.set_option('display.max_colwidth', 50)


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
            WHERE mined_at IS NOT NULL AND pruned_at IS NULL AND found_at != mined_at
        """, conn)
        # load the list of RBF-marked inputs
        rbfd = pd.read_sql_query("SELECT inputs_hash FROM rbf;", conn)
        
        mempool_df = pd.read_sql_query("""
            SELECT
                tx_id,
                created_at,
                size,
                tx_count,
                block_height,
                block_hash
            FROM mempool
        """, conn)
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
    
    return txdf, mempool_df

def compute_metrics(txdf, mempool_df, delta):
    # treat found_at == 0 as instant (set equal to mined_at)
    # txdf['found_at'] = txdf['found_at'].mask(
    #     txdf['found_at'] == 0,
    #     txdf['mined_at']
    # )
    # Remove rows where found_at is 0
    # txdf = txdf[txdf['found_at'] != 0]

    # compute actual wait-time in minutes
    # txdf['found_at'] = pd.to_datetime(txdf['found_at'], unit='s', origin='unix')
    # txdf['mined_at'] = pd.to_datetime(txdf['mined_at'], unit='s', origin='unix')
    
    # Compute mempool createat as datetime
    # mempool_df['created_at'] = pd.to_datetime(mempool_df['created_at'], unit='s', origin='unix')
    
    txdf['waittime'] = (txdf['mined_at'] - txdf['found_at'])
    
    # Remove waittime outliers
    txdf = txdf[txdf['waittime'] < 1000]
    
    # Print average waittime
    print(f"Average waittime Seconds: {txdf['waittime'].mean()}")
    
    # total number of txs
    print(f"Total number of txs: {len(txdf)}")

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
    txdf['respend_time'] = txdf['respend_time'].replace(0, np.nan)
    # txdf['inverse_respend'] = 1.0 / txdf['respend_time']

    # bin the inverse_respend into 1..Q_MAX
    # raw_q = np.ceil(txdf['respend_time'] / delta).fillna(0)
    # txdf['bin'] = raw_q.clip(lower=1, upper=Q_MAX).astype(int)
    
    # Add mempool size to txdf where mined_at is within 60 seconds 
    # txdf['mempool_size'] = mempool_df['size'].loc[txdf['mined_at'].isin(mempool_df['created_at'])]
    # Create a time window for each mempool timestamp
    mempool_df['window_start'] = mempool_df['created_at'] - 60
    mempool_df['window_end'] = mempool_df['created_at'] + 60

    # For each transaction, find matching mempool window and get size
    def get_mempool_size(mined_at):
        matching = mempool_df[
            (mempool_df['window_start'] <= mined_at) & 
            (mempool_df['window_end'] >= mined_at)
        ]
        return matching['size'].iloc[0] if not matching.empty else None

    txdf['mempool_size'] = txdf['mined_at'].apply(get_mempool_size)
    print(f"Average mempool size: {txdf['mempool_size'].mean()}")
    
    print(txdf['mempool_size'].head())
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
    # Print all the first 5 rows of txdf with no truncation
    for index, row in txdf.tail().iterrows():
        print(row)
        
        
    # Plot mempool size histogram
    plt.figure(figsize=(10, 6))
    plt.hist(txdf['mempool_size'], bins=100, edgecolor='black')
    plt.xlabel('Mempool Size')
    plt.ylabel('Frequency')
    plt.title('Mempool Size Distribution')
    plt.show()
    
    # Plot waittime histogram
    plt.figure(figsize=(10, 6))
    plt.hist(txdf['waittime'], bins=100, edgecolor='black')
    plt.xlabel('Wait Time (seconds)')
    plt.ylabel('Frequency')
    plt.title('Wait Time Distribution')
    plt.show()
        
    # Plot mempool size vs waittime
    plt.figure(figsize=(10, 6))
    plt.scatter(txdf['mempool_size'], txdf['waittime'], alpha=0.3)
    plt.xlabel('Mempool Size')
    plt.ylabel('Wait Time (seconds)')
    plt.title('Mempool Size vs Wait Time')
    plt.show()
    
    X = txdf[['mempool_size']]
    y = txdf['waittime']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LinearRegression().fit(X_train, y_train)
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)

    # print coefficients
    print("\nRegression results:")
    print(f"  Intercept: {model.intercept_:.4f}")
    for feat, coef in zip(X.columns, model.coef_):
        print(f"Beta_{feat}: {coef:.4f}")
    print(f"Train R² = {train_r2:.4f}")
    print(f"Test R² = {test_r2:.4f}")

    # scatter plot observed vs. predicted
    yhat = model.predict(X)
    plt.figure(figsize=(6,6))
    plt.scatter(y, yhat, alpha=0.3)
    plt.xlabel('Observed wait-time (seconds)')
    plt.ylabel('Predicted wait-time (seconds)')
    plt.title(f'Wait-time Model (Train R²={train_r2:.2f}, Test R²={test_r2:.2f})')
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
    txdf, mempool_df = load_data(args.db_path, args.tx_dump, args.rbf_dump)
    txdf = compute_metrics(txdf, mempool_df, delta=args.delta)
    # txdf = compute_epoch_stats(txdf, bins=args.bins)
    run_regression(txdf, zoom=args.zoom)

if __name__=='__main__':
    main()
