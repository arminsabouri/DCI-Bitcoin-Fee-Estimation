#! /usr/bin/env python3

# This sub-package will parse sqlite db tables: transactions, mempool, and rbf.
# And create one pandas dataframe where rows are transactions and columns are features.
# features from mempool and RBF will be added as new columns.

import pandas as pd
import sqlite3
import argparse
import pickle
from bitcoin.core import CTransaction
from io import BytesIO


def connect_to_db(db_path):
    conn = sqlite3.connect(db_path)
    return conn


def load_data(conn):
    # Get base transactions first
    transactions = pd.read_sql_query("""
        SELECT 
            transactions.*,
            (mined_at - found_at) AS waittime
        FROM transactions
        WHERE mined_at IS NOT NULL AND found_at IS NOT NULL
        ORDER BY found_at DESC
    """, conn)
    
    # Get RBF data separately
    rbf_data = pd.read_sql_query("""
        SELECT inputs_hash, MAX(fee_total) as rbf_fee_total
        FROM rbf
        GROUP BY inputs_hash
    """, conn)
    
    # Get mempool data separately (pre-aggregate by hour)
    mempool_data = pd.read_sql_query("""
        SELECT 
            strftime('%Y-%m-%d %H:00:00', datetime(created_at, 'unixepoch')) as hour,
            AVG(size) as mempool_size,
            AVG(tx_count) as mempool_tx_count
        FROM mempool
        GROUP BY hour
    """, conn)
    
    # Merge in Python (much faster than SQL JOINs)
    transactions = transactions.merge(rbf_data, on='inputs_hash', how='left')
    
    # Add hour column for mempool matching
    transactions['hour'] = pd.to_datetime(transactions['found_at'], unit='s').dt.floor('H')
    mempool_data['hour'] = pd.to_datetime(mempool_data['hour'])
    transactions = transactions.merge(mempool_data, on='hour', how='left')
    
    return transactions.drop('hour', axis=1)


def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.10)
    Q3 = df[column].quantile(0.90)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]


def get_tx_weight(tx_hex: str) -> int:
    tx_bytes = bytes.fromhex(tx_hex)
    stream = BytesIO(tx_bytes)
    tx = CTransaction.stream_deserialize(stream)
    return tx.calc_weight()


def compute_metrics(transactions):
    print(f"total transactions length: {len(transactions)}")
    # Ensure there are no null values or 0 values for found_at or mined_at
    assert transactions[transactions['found_at'] >
                        0].shape[0] == transactions.shape[0], "found_at has null values"
    assert transactions[transactions['mined_at'] >
                        0].shape[0] == transactions.shape[0], "mined_at has null values"

    # Remove outliers from waittime
    transactions = remove_outliers_iqr(transactions, 'waittime')
    print(f"transactions after removing outliers: {len(transactions)}")

    def get_weight_and_size(tx_hex):
        tx_bytes = bytes.fromhex(tx_hex)
        stream = BytesIO(tx_bytes)
        tx = CTransaction.stream_deserialize(stream)
        return tx.calc_weight(), len(tx_bytes)

    transactions[['weight', 'size']] = transactions['tx_data'].apply(
        lambda tx_hex: pd.Series(get_weight_and_size(tx_hex))
    )

    # We can drop tx_data. We should extract any data we can from it and then drop it.
    transactions = transactions.drop(columns=['tx_data'])
    return transactions


def output_data(transactions):
    with open('transactions.pkl', 'wb') as f:
        pickle.dump(transactions, f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db-path', required=True)
    args = p.parse_args()

    conn = connect_to_db(args.db_path)
    try:
        transactions = load_data(conn)
        transactions = compute_metrics(transactions)
        output_data(transactions)
        # Pretty print the first 5 rows with all columns using pandas built-in formatting
        # for index, row in transactions.head(15).iterrows():
        #     print(row['tx_id'], row['found_at'], row['mined_at'], row['wait_time'], row['rbf_fee_total'], row['mempool_size'], row['mempool_tx_count'])
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
