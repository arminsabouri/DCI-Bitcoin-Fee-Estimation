import os
print("RUNNING:", os.path.abspath(__file__))

import pandas as pd
import numpy as np
import sqlite3
import argparse
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Default path to your SQLite DB
DB_PATH_DEFAULT = '/Users/matth410/Desktop/DCI/mempool-tracker.db'
Q_MAX = 10  # max inverse-respend bin

def load_data(db_path):
    conn = sqlite3.connect(db_path)
    txdf = pd.read_sql_query("""
        SELECT tx_id, inputs_hash, found_at, mined_at, parent_txid
        FROM transactions
        WHERE found_at IS NOT NULL AND mined_at IS NOT NULL
    """, conn)
    rbfd = pd.read_sql_query("SELECT inputs_hash FROM rbf;", conn)
    conn.close()

    # mark RBF transactions
    rbfd['rbf_flag'] = 1
    txdf = txdf.merge(rbfd[['inputs_hash','rbf_flag']],
                      on='inputs_hash', how='left')
    txdf['rbf_flag'] = txdf['rbf_flag'].fillna(0).astype(int)
    return txdf

def compute_metrics(txdf, delta):
    # 1) confirmation delay in minutes
    txdf['waittime'] = (txdf['mined_at'] - txdf['found_at']) / 60.0

    # 2) parent-child mining times
    parent = txdf[['tx_id','mined_at']].rename(
        columns={'tx_id':'parent_tx','mined_at':'parent_mined'}
    )
    child = txdf[['parent_txid','mined_at']].rename(
        columns={'parent_txid':'parent_tx','mined_at':'child_mined'}
    )
    merged = child.merge(parent, on='parent_tx', how='inner')

    # 3) first mined child per parent
    fc = (merged.groupby('parent_tx')['child_mined']
                .min()
                .reset_index()
                .rename(columns={'parent_tx':'tx_id',
                                 'child_mined':'first_child_mined'}))
    txdf = txdf.merge(fc, on='tx_id', how='left')

    # 4) censor non-spenders and build inverse-respend bins
    max_mined = txdf['mined_at'].max()
    txdf['cpfp_flag'] = txdf['first_child_mined'].notnull().astype(int)
    txdf['respend_time'] = np.where(
        txdf['cpfp_flag']==1,
        txdf['first_child_mined'] - txdf['mined_at'],
        max_mined - txdf['mined_at']
    )
    txdf['inverse_respend'] = 1.0 / txdf['respend_time'].replace(0, np.nan)
    raw_q = np.ceil(txdf['inverse_respend'] / delta).fillna(0)
    txdf['bin'] = raw_q.clip(lower=1, upper=Q_MAX).astype(int)

    return txdf

def compute_epoch_stats(txdf, bins):
    # assign epochs by found_at
    txdf['epoch'] = pd.cut(txdf['found_at'], bins=bins, labels=False)
    # congestion = count per epoch
    ep_counts = txdf.groupby('epoch')['tx_id'].size()
    cong = ep_counts.rename('congestion')
    total = ep_counts.rename('total')
    # sigma_q = share in each bin
    count_by_bin = txdf.groupby(['epoch','bin'])['tx_id']\
                      .size().rename('count')
    sigma = count_by_bin.reset_index().merge(total.reset_index(), on='epoch')
    sigma['sigma'] = sigma['count'] / sigma['total']
    # F = reverse-cumsum of sigma
    sigma.sort_values(['epoch','bin'], inplace=True)
    sigma['F'] = (sigma.groupby('epoch')['sigma']
                       .transform(lambda x: x.iloc[::-1].cumsum().iloc[::-1]))
    # merge back
    txdf = txdf.merge(cong.reset_index(), on='epoch')
    txdf = txdf.merge(sigma[['epoch','bin','sigma','F']],
                      on=['epoch','bin'], how='left')
    return txdf

def diagnostic_waittime(txdf):
    # Print & plot the actual confirmation-delay distribution
    print("\nConfirmation-delay (waittime) distribution:")
    print(txdf['waittime'].describe())
    print("Unique confirmation delays:", txdf['waittime'].nunique())

    # Histogram of those under 500 mins
    subset = txdf[txdf['waittime'] < 500]
    plt.figure()
    subset['waittime'].hist(bins=50)
    plt.title("Wait-time < 500 mins")
    plt.xlabel("Wait-time (mins)")
    plt.ylabel("Frequency")
    plt.show()

def run_regression(txdf, zoom_thresh):
    # regression target is the actual waittime
    X = txdf[['congestion','F','rbf_flag','cpfp_flag']]
    y = txdf['waittime'].values
    model = LinearRegression().fit(X, y)

    print("\nRegression results:")
    print(f"  Intercept: {model.intercept_:.4f}")
    for feat, coef in zip(X.columns, model.coef_):
        print(f"  Beta_{feat}: {coef:.4f}")
    r2_full = model.score(X, y)
    print(f"  R² = {r2_full:.4f}")

    # full-range scatter
    y_pred = model.predict(X)
    plt.figure(figsize=(6,6))
    plt.scatter(y, y_pred, alpha=0.3)
    plt.xlabel('Observed wait-time (mins)')
    plt.ylabel('Predicted wait-time (mins)')
    plt.title(f'Wait-time Model (R²={r2_full:.2f})')
    plt.grid(True)
    plt.show()

    # zoomed scatter on [0, zoom_thresh]
    mask = (txdf['waittime'] >= 0) & (txdf['waittime'] <= zoom_thresh)
    if mask.sum() > 0:
        y_sub = txdf.loc[mask, 'waittime']
        X_sub = X.loc[mask]
        y_pred_sub = model.predict(X_sub)
        r2_sub = model.score(X_sub, y_sub)
        plt.figure(figsize=(6,6))
        plt.scatter(y_sub, y_pred_sub, alpha=0.3)
        plt.xlabel('Observed wait-time (mins)')
        plt.ylabel('Predicted wait-time (mins)')
        plt.title(f'Zoomed Wait-time Model (R²={r2_sub:.2f})')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db-path', default=DB_PATH_DEFAULT,
                   help='Path to SQLite DB')
    p.add_argument('--bins',   type=int,   default=10,
                   help='Number of epochs for congestion')
    p.add_argument('--delta',  type=float, default=1.0,
                   help='Width of inverse-respend bins')
    p.add_argument('--zoom',   type=float, default=200.0,
                   help='Max wait-time (mins) for zoom plot')
    args = p.parse_args()

    print(f"\nLoading data from: {args.db_path}")
    txdf = load_data(args.db_path)
    txdf = compute_metrics(txdf, delta=args.delta)
    txdf = compute_epoch_stats(txdf, bins=args.bins)

    diagnostic_waittime(txdf)
    run_regression(txdf, zoom_thresh=args.zoom)

if __name__ == '__main__':
    main()
