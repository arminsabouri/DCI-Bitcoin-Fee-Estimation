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
from bitcoinrpc import BitcoinRPC
import asyncio


def connect_to_rpc(rpc_user, rpc_password, rpc_host, rpc_port):
    host = f"http://{rpc_host}:{rpc_port}"
    rpc = BitcoinRPC.from_config(host, (rpc_user, rpc_password))
    return rpc


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
    transactions['hour'] = pd.to_datetime(
        transactions['found_at'], unit='s').dt.floor('H')
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


async def compute_metrics(transactions, rpc: BitcoinRPC):
    print(f"total transactions length: {len(transactions)}")
    # Ensure there are no null values or 0 values for found_at or mined_at
    assert transactions[transactions['found_at'] >
                        0].shape[0] == transactions.shape[0], "found_at has null values"
    assert transactions[transactions['mined_at'] >
                        0].shape[0] == transactions.shape[0], "mined_at has null values"

    # Remove outliers from waittime
    transactions = remove_outliers_iqr(transactions, 'waittime')
    print(f"transactions after removing outliers: {len(transactions)}")

    def ser_tx(tx_hex: str) -> CTransaction:
        tx_bytes = bytes.fromhex(tx_hex)
        stream = BytesIO(tx_bytes)
        tx = CTransaction.stream_deserialize(stream)
        return tx

    def get_weight_and_size(tx_hex: str):
        tx_bytes = bytes.fromhex(tx_hex)
        tx = ser_tx(tx_hex)
        return tx.calc_weight(), len(tx_bytes)

    async def get_min_respend_time(tx_hex: str):
        tx = ser_tx(tx_hex)
        txid = tx.GetTxid().hex()
        print(f"Computing min_respend_time for txid {txid}")
        try:
            confs = await asyncio.wait_for(
                asyncio.gather(
                    *[rpc.getrawtransaction(str(vin.prevout).split(':')[0], True) for vin in tx.vin]
                ),
                timeout=10
            )
        except asyncio.TimeoutError:
            print(f"Timeout occurred while fetching transaction data for txid {txid}")
            confs = []
        confs = [prev_tx['confirmations'] for prev_tx in confs if 'confirmations' in prev_tx]
        return min(confs, default=0)
    print("Computing weight and size")
    transactions[['weight', 'size']] = transactions['tx_data'].apply(
        lambda tx_hex: pd.Series(get_weight_and_size(tx_hex))
    )

    print("Computing min_respend_time")
    # Compute min_respend_time for each transaction asynchronously
    min_respend_times = await asyncio.gather(
        *[get_min_respend_time(tx_hex) for tx_hex in transactions['tx_data']]
    )
    transactions['min_respend_time'] = min_respend_times

    # We can drop tx_data. We should extract any data we can from it and then drop it.
    transactions = transactions.drop(columns=['tx_data'])
    return transactions


def output_data(transactions, output_path):
    transactions.to_hdf(
        output_path,
        key="mempool_transactions",
        mode="a",
        format="table",
        append=True
    )


async def main():
    print("Starting data lake")
    p = argparse.ArgumentParser()
    # TODO output destination should be configurable
    p.add_argument('--db-path', required=True)
    p.add_argument('--rpc-user', required=True)
    p.add_argument('--rpc-password', required=True)
    p.add_argument('--rpc-host', required=True)
    p.add_argument('--rpc-port', required=True)
    p.add_argument('--output-path', required=False, default="data-lake.h5")

    args = p.parse_args()

    conn = connect_to_db(args.db_path)
    rpc = connect_to_rpc(args.rpc_user, args.rpc_password,
                         args.rpc_host, args.rpc_port)
    try:
        print("Loading data")
        transactions = load_data(conn)
        print("Computing metrics")
        transactions = await compute_metrics(transactions, rpc)
        print(f"outputting data to {args.output_path}")
        output_data(transactions, args.output_path)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()


def _entrypoint():
    asyncio.run(main())


if __name__ == "__main__":
    _entrypoint()
