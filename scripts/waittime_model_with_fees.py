import os
import sys
import argparse
from load_from_sqlite import load_data_from_sqlite
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def compute_mempool_congestion(txdf, bins):
    txdf['epoch_id'] = pd.cut(txdf['found_at'], bins=bins, labels=False)
    congestion = txdf.groupby('epoch_id').size().rename('congestion')
    return txdf.merge(congestion, on='epoch_id')

def compute_inverse_respend(txdf):
    txdf['waittime'] = (txdf['mined_at'] - txdf['found_at']) / 60
    txdf['inverse_respend'] = 1 / txdf['waittime'].replace(0, np.nan)
    return txdf.dropna(subset=['inverse_respend'])

def estimate_Wqt(txdf, quantiles, delta=1):
    results = []
    for epoch in sorted(txdf['epoch_id'].unique()):
        epoch_df = txdf[txdf['epoch_id'] == epoch]
        rho = epoch_df['congestion'].iloc[0]
        for q in quantiles:
            lower = q * delta
            upper = (q + 1) * delta
            bin_df = epoch_df[(epoch_df['inverse_respend'] > lower) & (epoch_df['inverse_respend'] <= upper)]
            if not bin_df.empty:
                Wqt = bin_df['waittime'].mean()
                Ft_q = (epoch_df['inverse_respend'] > upper).mean()
                results.append({'epoch_id': epoch, 'quantile': q, 'Wqt': Wqt, 'Ft_q': Ft_q, 'rho': rho})
    return pd.DataFrame(results)

def build_riemann_sum(txdf, Wqt_df, quantiles, delta=1):
    def compute_sum(row):
        epoch = row['epoch_id']
        max_q = int(np.ceil(row['inverse_respend'] / delta))
        total = 0
        for q in range(max_q):
            match = Wqt_df[(Wqt_df['epoch_id'] == epoch) & (Wqt_df['quantile'] == q)]
            if not match.empty:
                sigma = 1 / len(quantiles)
                W_hat = match['W_hat'].values[0]
                total += sigma * q * delta * W_hat
        return row['congestion'] * total

    txdf['riemann_sum'] = txdf.apply(compute_sum, axis=1)
    return txdf

def run_final_regression(txdf):
    X = txdf[['riemann_sum']].fillna(0)
    y = txdf['fee'].fillna(0)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = model.score(X, y)

    plt.scatter(y, y_pred, alpha=0.3)
    plt.xlabel('Actual Fee (Satoshis)')
    plt.ylabel('Predicted Fee')
    plt.title(f'Transaction Fee Prediction (R² = {r2:.2f})')
    plt.grid(True)
    plt.show()

    return model, r2

def train_model(fees_path, bins=10, quantiles=None):
    # Ensure the fees CSV exists
    if not os.path.isfile(fees_path):
        sys.exit(f"Error: fees file '{fees_path}' not found. Run fetch_fees.py to generate it or provide the correct path with --fees-csv.")
    fees = pd.read_csv(fees_path)

    txdf, _ = load_data_from_sqlite()
    txdf = txdf.merge(fees, on="tx_id", how="inner")
    txdf.dropna(subset=['found_at', 'mined_at', 'fee'], inplace=True)
    txdf['waittime'] = (txdf['mined_at'] - txdf['found_at']) / 60

    # Default quantiles if not provided
    if quantiles is None:
        quantiles = list(range(16))

    txdf = compute_mempool_congestion(txdf, bins=bins)
    txdf = compute_inverse_respend(txdf)

    # First-stage: estimate Wqt and tail F
    Wqt_df = estimate_Wqt(txdf, quantiles)
    fs_df = Wqt_df.dropna(subset=['Wqt', 'Ft_q', 'rho'])
    X_fs = fs_df[['rho', 'Ft_q']]
    y_fs = fs_df['Wqt']
    fs_model = LinearRegression()
    fs_model.fit(X_fs, y_fs)
    b0 = fs_model.intercept_
    b_rho, b_Ft = fs_model.coef_
    print(f"First-stage coefficients: intercept={b0:.4f}, beta_rho={b_rho:.4f}, beta_Ft={b_Ft:.4f}")

    Wqt_df['W_hat'] = b0 + b_rho * Wqt_df['rho'] + b_Ft * Wqt_df['Ft_q']

    # Build structural term and run final regression
    txdf = build_riemann_sum(txdf, Wqt_df, quantiles)
    model, r2 = run_final_regression(txdf)
    print(f"Model R² score: {r2:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Estimate transaction wait time model with fees.")
    parser.add_argument('--fees-csv', default='fees_fetched.csv', help='Path to fees CSV file')
    parser.add_argument('--bins', type=int, default=10, help='Number of epochs for congestion')
    args = parser.parse_args()
    train_model(args.fees_csv, bins=args.bins)