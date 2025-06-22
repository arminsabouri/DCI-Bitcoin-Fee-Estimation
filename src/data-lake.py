#! /usr/bin/env python3

# This sub-package will parse sqlite db tables: transactions, mempool, and rbf.
# And create one pandas dataframe where rows are transactions and columns are features.
# features from mempool and RBF will be added as new columns.

import pandas as pd
import sqlite3
import argparse
import pickle


def connect_to_db(db_path):
    conn = sqlite3.connect(db_path)
    return conn


def load_data(conn):
    # Load transactions table
    transactions = pd.read_sql_query("""
        SELECT 
            transactions.*,
            (transactions.mined_at - transactions.found_at) AS wait_time,
            MAX(rbf.fee_total) AS rbf_fee_total,
            AVG(mempool.size) AS mempool_size,
            AVG(mempool.tx_count) AS mempool_tx_count
        FROM transactions
        JOIN rbf ON transactions.inputs_hash = rbf.inputs_hash
        LEFT JOIN mempool 
            ON transactions.found_at BETWEEN mempool.created_at - 3600 AND mempool.created_at + 3600
        WHERE transactions.mined_at IS NOT NULL 
          AND transactions.found_at IS NOT NULL
        GROUP BY transactions.tx_id
        ORDER BY transactions.found_at DESC
        LIMIT 100
    """, conn)

    return transactions


def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.10)
    Q3 = df[column].quantile(0.90)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

def compute_metrics(transactions):
    print(f"total transactions length: {len(transactions)}")
    # Ensure there are no null values or 0 values for found_at or mined_at
    assert transactions[transactions['found_at'] >
                        0].shape[0] == transactions.shape[0], "found_at has null values"
    assert transactions[transactions['mined_at'] >
                        0].shape[0] == transactions.shape[0], "mined_at has null values"

    # Remove outliers from waittime
    transactions = remove_outliers_iqr(transactions, 'wait_time')
    print(f"transactions after removing outliers: {len(transactions)}")

    # We can drop tx_data. We should extract any deata we can from it and then drop it.
    # TODO Need to extract weight and size from tx_data before dropping it 
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
