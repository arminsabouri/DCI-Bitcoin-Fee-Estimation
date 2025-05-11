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
    transactions = pd.read_sql_query("SELECT * FROM transactions WHERE mined_at IS NOT NULL", conn)

    # Load mempool table
    mempool = pd.read_sql_query("SELECT * FROM mempool", conn)

    # Load rbf table
    rbf = pd.read_sql_query("SELECT * FROM rbf", conn)

    return transactions, mempool, rbf

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.10)
    Q3 = df[column].quantile(0.90)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]


def compute_metrics(transactions, mempool, rbf):
    print(f"total transactions length: {len(transactions)}")
    # Ensure there are no null values or 0 values for found_at or mined_at
    assert transactions[transactions['found_at'] > 0].shape[0] == transactions.shape[0] , "found_at has null values"
    assert transactions[transactions['mined_at'] > 0].shape[0] == transactions.shape[0] , "mined_at has null values"
    
    # Compute waittime for each transaction as the difference between mined_at and found_at
    # Timestamps are in seconds since UNIX epoch
    transactions['waittime'] = transactions['mined_at'] - transactions['found_at']
    
    # Remove outliers from waittime
    print(f"total transactions length: {len(transactions)}")
    transactions = remove_outliers_iqr(transactions, 'waittime')
    print(f"transactions after removing outliers: {len(transactions)}")
    mempool['window_start'] = mempool['created_at'] - 60
    mempool['window_end'] = mempool['created_at'] + 60

    # For each transaction, find matching mempool window and get size
    def get_mempool_size(found_at):
        matching = mempool[
            (mempool['window_start'] <= found_at) & 
            (mempool['window_end'] >= found_at)
        ]
        return matching['size'].iloc[0] if not matching.empty else None

    transactions['mempool_size'] = transactions['found_at'].apply(get_mempool_size)
    
    # For each transaction, find matching mempool window and get tx_count
    def get_mempool_tx_count(found_at):
        matching = mempool[
            (mempool['window_start'] <= found_at) & 
            (mempool['window_end'] >= found_at)
        ]
        return matching['tx_count'].iloc[0] if not matching.empty else None
        
    transactions['mempool_tx_count'] = transactions['found_at'].apply(get_mempool_tx_count)
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
        transactions, mempool, rbf = load_data(conn)
        transactions = compute_metrics(transactions, mempool, rbf)
        output_data(transactions)
        print(f"Output data to transactions.pkl")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()
        
        
if __name__ == "__main__":
    main()  
    