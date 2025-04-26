import pandas as pd
import numpy as np
import sqlite3
import argparse
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Default path to the recovered (uncorrupted) database
DB_PATH_DEFAULT = '/Users/matth410/Desktop/DCI/recovered.db'


def load_data(db_path):
    """
    Load transactions and RBF flags from the specified SQLite database.
    """
    conn = sqlite3.connect(db_path)
    # Load only the needed columns
    txdf = pd.read_sql_query(
        """
        SELECT tx_id, inputs_hash, found_at, mined_at, parent_txid
        FROM transactions
        WHERE found_at IS NOT NULL AND mined_at IS NOT NULL
        """,
        conn
    )
    # Load RBF indicator
    rbfd = pd.read_sql_query("SELECT inputs_hash FROM rbf;", conn)
    conn.close()

    # Flag RBF transactions
    rbfd['rbf_flag'] = 1
    txdf = txdf.merge(rbfd[['inputs_hash','rbf_flag']], on='inputs_hash', how='left')
    txdf['rbf_flag'] = txdf['rbf_flag'].fillna(0).astype(int)
    return txdf


def compute_respend_and_flags(txdf):
    """
    Compute re-spend times (in seconds) and child_flag for every transaction.
    Non-spenders are right-censored at the latest mined_at.
    """
    # Prepare parent and child mining times
    parent = txdf[['tx_id','mined_at']].rename(columns={'tx_id':'parent_tx_id','mined_at':'parent_mined'})
    child = txdf[['parent_txid','mined_at']].rename(columns={'parent_txid':'parent_tx_id','mined_at':'child_mined'})
    merged = child.merge(parent, on='parent_tx_id', how='inner')

    # Get earliest child mined time per parent
    first_child = (
        merged.groupby('parent_tx_id')['child_mined']
              .min()
              .reset_index()
              .rename(columns={'parent_tx_id':'tx_id','child_mined':'first_child_mined'})
    )
    txdf = txdf.merge(first_child, on='tx_id', how='left')

    # Censor non-spenders at max mined_at
    max_mined = txdf['mined_at'].max()
    txdf['child_flag'] = txdf['first_child_mined'].notnull().astype(int)
    txdf['respend_time'] = np.where(
        txdf['child_flag']==1,
        txdf['first_child_mined'] - txdf['mined_at'],
        max_mined - txdf['mined_at']
    )
    return txdf


def compute_congestion(txdf, bins):
    """
    Compute mempool congestion (#transactions per epoch) from found_at.
    """
    txdf['epoch'] = pd.cut(txdf['found_at'], bins=bins, labels=False)
    cong = txdf.groupby('epoch').size().rename('congestion')
    return txdf.merge(cong, on='epoch')


def run_regression(txdf):
    """
    Fit linear model: respend_time ~ congestion + rbf_flag + child_flag
    """
    X = txdf[['congestion','rbf_flag','child_flag']]
    y = txdf['respend_time']
    model = LinearRegression().fit(X, y)

    # Print results
    print("Regression results:")
    print(f"  Intercept: {model.intercept_:.4f}")
    for feat, coef in zip(X.columns, model.coef_):
        print(f"  Beta_{feat}: {coef:.4f}")
    print(f"  R² = {model.score(X, y):.4f}\n")

    # Plot observed vs predicted
    y_pred = model.predict(X)
    plt.figure(figsize=(6,6))
    plt.scatter(y, y_pred, alpha=0.4)
    plt.xlabel('Observed re-spend time (sec)')
    plt.ylabel('Predicted re-spend time (sec)')
    plt.title(f'Re-spend Regression (R²={model.score(X,y):.2f})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Re-spend regression with RBF & child flags')
    parser.add_argument('--db-path', default=DB_PATH_DEFAULT, help='Path to recovered SQLite DB')
    parser.add_argument('--bins', type=int, default=10, help='Epoch bins for congestion')
    args = parser.parse_args()

    print(f"Loading data from: {args.db_path}")
    txdf = load_data(args.db_path)
    txdf = compute_respend_and_flags(txdf)
    txdf = compute_congestion(txdf, bins=args.bins)
    run_regression(txdf)

if __name__ == '__main__':
    main()