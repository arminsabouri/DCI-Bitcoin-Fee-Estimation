import argparse, sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Now points at your pre-processed pickle instead of any SQLite DB
PKL_DEFAULT = '/Users/matth410/Desktop/DCI/transactions.pkl'
Q_MAX = 10  # maximum inverse-respend bin

def load_data(pkl_path):
    """
    Load the fully-processed transactions DataFrame from a pickle.
    If the pickle lacks an 'rbf_flag' column, add it (all zeros).
    """
    try:
        txdf = pd.read_pickle(pkl_path)
    except Exception as e:
        print(f"Failed to load pickle {pkl_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # if pickle didn't already have RBF flags, add a column of zeros
    if 'rbf_flag' not in txdf.columns:
        txdf['rbf_flag'] = 0

    print(f"Loaded {len(txdf)} transactions from {pkl_path}")
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
    # instant for unseen entries
    txdf['found_at'] = txdf['found_at'].mask(
        txdf['found_at'] == 0, txdf['mined_at']
    )

    # wait-time in minutes
    txdf['found_dt'] = pd.to_datetime(txdf['found_at'], unit='s', origin='unix')
    txdf['mined_dt'] = pd.to_datetime(txdf['mined_at'], unit='s', origin='unix')
    txdf['waittime'] = (txdf['mined_dt'] - txdf['found_dt']) \
                        .dt.total_seconds() / 60.0

    # CPFP detection
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

    # censor non-spenders, compute respend_time & inverse_respend
    max_mined = txdf['mined_at'].max()
    txdf['cpfp_flag'] = txdf['first_child_mined'].notnull().astype(int)
    txdf['respend_time'] = np.where(
        txdf['cpfp_flag']==1,
        txdf['first_child_mined'] - txdf['mined_at'],
        max_mined - txdf['mined_at']
    )
    txdf['inverse_respend'] = 1.0 / txdf['respend_time'].replace(0, np.nan)

    # cin into 1..Q_MAX
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

    tot  = txdf.groupby('epoch')['tx_id'] \
              .size().rename('total').reset_index()
    cong = tot.rename(columns={'total':'congestion'})

    cnt   = txdf.groupby(['epoch','bin'])['tx_id'] \
               .size().rename('count').reset_index()
    sigma = cnt.merge(tot, on='epoch')
    sigma['sigma'] = sigma['count'] / sigma['total']
    sigma = sigma.sort_values(['epoch','bin'])
    sigma['F'] = sigma.groupby('epoch')['sigma'] \
                      .transform(lambda x: x.iloc[::-1].cumsum().iloc[::-1])

    txdf = txdf.merge(cong, on='epoch')
    txdf = txdf.merge(
        sigma[['epoch','bin','sigma','F']],
        on=['epoch','bin'], how='left'
    )
    return txdf

def run_regression(txdf, zoom=None):
    """
    Fit waittime ~ congestion + F + rbf_flag + cpfp_flag,
    then plot observed vs. predicted (nonzero waits only).
    """
    X = txdf[['congestion','F','rbf_flag','cpfp_flag']]
    y = txdf['waittime']
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)

    print("\nRegression results:")
    print(f"Intercept: {model.intercept_:.4f}")
    for feat,coef in zip(X.columns, model.coef_):
        print(f"Beta_{feat}: {coef:.4f}")
    print(f"R² = {r2:.4f}\n")

    mask = y > 0
    yhat = model.predict(X.loc[mask])
    plt.figure(figsize=(6,6))
    plt.scatter(y.loc[mask], yhat, alpha=0.3)
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
    p.add_argument('--pkl',   default=PKL_DEFAULT,
                   help="path to transactions.pkl")
    p.add_argument('--bins',  type=int, default=10,
                   help="number of epochs (time bins)")
    p.add_argument('--delta', type=float, default=1.0,
                   help="Δ for binning inverse-respend")
    p.add_argument('--zoom',  type=float,
                   help="max axis value for scatter zoom")
    args = p.parse_args()

    print(f"RUNNING: {sys.argv[0]}")
    txdf = load_data(args.pkl)
    txdf = compute_metrics(txdf, delta=args.delta)
    txdf = compute_epoch_stats(txdf, bins=args.bins)
    run_regression(txdf, zoom=args.zoom)

if __name__=='__main__':
    main()
