#!/usr/bin/env python3

# This sub-package will parse sqlite db tables: transactions, mempool, and rbf.
# And create one pandas dataframe where rows are transactions and columns are features.
# features from mempool and RBF will be added as new columns.

import pandas as pd
import sqlite3
import argparse
import pickle
from bitcoin.core import CTransaction
from io import BytesIO
from io import StringIO
from bitcoinrpc import BitcoinRPC
import asyncio
import json


async def connect_to_rpc(rpc_user, rpc_password, rpc_host, rpc_port):
    host = f"http://{rpc_host}:{rpc_port}"
    try:
        rpc = BitcoinRPC.from_config(
            host, (rpc_user, rpc_password), timeout=10)
        # Test the connection
        test_result = await rpc.getblockchaininfo()
        print(
            f"Successfully connected to bitcoind. Block height: {test_result['blocks']}")
        return rpc
    except Exception as e:
        print(f"Failed to connect to bitcoind: {e}")
        print(f"Connection details: {host}")
        print(f"Username: {rpc_user}")
        raise


def connect_to_db(db_path):
    conn = sqlite3.connect(db_path)
    return conn


def load_rbf(conn):
    rbf_data = pd.read_sql_query("""
        SELECT inputs_hash, MAX(fee_total) as rbf_fee_total
        FROM rbf
        GROUP BY inputs_hash
        ORDER BY created_at DESC
    """, conn)
    return rbf_data


def load_mempool(conn):
    mempool_data = pd.read_sql_query("""
        SELECT
            strftime('%Y-%m-%d %H:00:00', datetime(created_at, 'unixepoch')) as hour,
            AVG(size) as mempool_size,
            AVG(tx_count) as mempool_tx_count
        FROM mempool
        GROUP BY hour
        ORDER BY created_at DESC
    """, conn)
    return mempool_data


def merge_datasets(transactions, rbf_data, mempool_data):
    # merge the rbf data into the transactions dataframe
    transactions = transactions.merge(rbf_data, on='inputs_hash', how='left')

    # add the hour column to the transactions dataframe
    transactions['hour'] = pd.to_datetime(
        transactions['found_at'], unit='s').dt.floor('H')
    mempool_data['hour'] = pd.to_datetime(mempool_data['hour'])

    transactions = transactions.merge(mempool_data, on='hour', how='left')
    # after merge, drop the hour column
    transactions = transactions.drop('hour', axis=1)
    return transactions


def load_transactions(conn, limit: int):
    last_rowid = 0
    while True:
        query = """
            SELECT
                filtered.*,
                (filtered.mined_at - filtered.found_at) AS waittime,
                (SELECT parent.tx_id 
                 FROM transactions AS parent 
                 WHERE parent.child_txid = filtered.tx_id 
                 LIMIT 1) as parent_txid
            FROM (
                SELECT transactions.*, rowid
                FROM transactions
                WHERE
                    mined_at IS NOT NULL
                    AND found_at IS NOT NULL
                    AND pruned_at IS NULL
                    AND rowid > ?
                ORDER BY rowid ASC
                LIMIT ?
            ) AS filtered
        """
        transactions = pd.read_sql_query(
            query, conn, params=(last_rowid, limit))

        print(f"transactions: {len(transactions)}")
        if transactions.empty:
            break

        last_rowid = int(transactions['rowid'].max())
        print(f"last_rowid: {last_rowid}")

        # Sort here to avoid pagination headaches in sql
        transactions = transactions.sort_values(["found_at"], ascending=False)
        transactions = transactions.drop('rowid', axis=1)
        print(f"transactions: {transactions.columns}")

        yield transactions


def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.10)
    Q3 = df[column].quantile(0.90)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]


block_hash_to_height = {}


async def get_block_height(block_hash: str, rpc: BitcoinRPC):
    if block_hash in block_hash_to_height:
        return block_hash_to_height[block_hash]
    block = await rpc.acall('getblock', [block_hash])
    block_hash_to_height[block_hash] = block['height']
    return block['height']


async def compute_metrics(transactions, rpc: BitcoinRPC, debug: bool):
    print(f"total transactions length: {len(transactions)}")
    # Ensure there are no null values or 0 values for found_at or mined_at
    assert transactions[transactions['found_at'] >
                        0].shape[0] == transactions.shape[0], "found_at has null values"
    assert transactions[transactions['mined_at'] >
                        0].shape[0] == transactions.shape[0], "mined_at has null values"

    def ser_tx(tx_hex: str) -> CTransaction:
        tx_bytes = bytes.fromhex(tx_hex)
        stream = BytesIO(tx_bytes)
        tx = CTransaction.stream_deserialize(stream)
        return tx

    def get_tx_stats(tx_hex: str) -> (int, int, [int], int, [int]):
        """
        Returns:
            weight: int
            size: int
            output_amounts: list[int]
            total_output_amount: int
            output_weights: list[int]
        """
        tx_bytes = bytes.fromhex(tx_hex)
        tx = ser_tx(tx_hex)
        output_amounts = [vout.nValue for vout in tx.vout]

        # Calculate the individual weights of the outputs.
        # Bitcoin Core does not attribute a "weight" per output directly, but we can approximate
        # each vout's serialized size and its WEIGHT as 4x its byte size (witness discount doesn't apply to vouts).
        output_weights = []
        for vout in tx.vout:
            buf = BytesIO()
            vout.stream_serialize(buf)
            vout_len = buf.tell()
            output_weights.append(vout_len * 4)

        return tx.calc_weight(), len(tx_bytes), output_amounts, sum(output_amounts), output_weights

    # allow only 8 concurrent RPCs requests
    sem = asyncio.Semaphore(8)

    # Return type (min respend time, conf blockhash)
    async def get_min_respend_blocks(txid: str) -> (int, str):
        async with sem:
            try:
                print(f"Computing min_respend_blocks for txid {txid}")
                confs = []
                ver_tx = await rpc.acall('getrawtransaction', [txid, 2])
                if 'blockhash' not in ver_tx:
                    print(f"No blockhash found for txid {txid}")
                    return -1, ''
                conf_block_hash = ver_tx['blockhash']
                conf_height = await get_block_height(ver_tx['blockhash'], rpc)
                if debug:
                    print(f"Block height: {conf_height}")

                for i, vin in enumerate(ver_tx['vin']):
                    if 'prevout' in vin and 'height' in vin['prevout']:
                        prev_height = vin['prevout']['height']
                        conf_diff = conf_height - prev_height
                        confs.append(conf_diff)

                    else:
                        print(f"Vin {i}: No prevout or height found")
                for i, vout in enumerate(ver_tx['vout']):
                    if 'scriptPubKey' in vout and 'addresses' in vout['scriptPubKey']:
                        vout_address = vout['scriptPubKey']['address']

                if len(confs) == 0:
                    print(f"No valid prevout heights found for txid {txid}")
                    return -1, conf_block_hash

                return min(confs, default=0), conf_block_hash
            except Exception as e:
                print(f"Error fetching transaction {txid}: {e}")
                return -1, ''

    print("Computing weight and size")
    transactions[['weight', 'size', 'output_amounts', 'total_output_amount', 'output_weights']] = transactions['tx_data'].apply(
        lambda tx_hex: pd.Series(get_tx_stats(tx_hex))
    )

    transactions['output_amounts'] = transactions['output_amounts'].apply(
        lambda x: json.dumps(x))
    transactions['output_weights'] = transactions['output_weights'].apply(
        lambda x: json.dumps(x))

    # Convert to string because hdf5 will treat this as a variable length column
    transactions['output_amounts'] = transactions['output_amounts'].astype(
        'string')
    transactions['output_weights'] = transactions['output_weights'].astype(
        'string')

    print("Computing min_respend_blocks")
    # Compute min_respend_time for each transaction asynchronously
    min_respend_blocks = await asyncio.gather(
        *[get_min_respend_blocks(txid) for txid in transactions['tx_id']]
    )

    transactions['min_respend_blocks'] = [x[0] for x in min_respend_blocks]
    transactions['conf_block_hash'] = [x[1] for x in min_respend_blocks]
    transactions = transactions[transactions['min_respend_blocks'] != -1]

    return transactions


def output_data(transactions, conn):
    for col in ["found_at", "mined_at", "pruned_at"]:
        if col in transactions.columns:
            # Convert unix timestamp to human-readable datetime string
            transactions[col] = pd.to_datetime(
                transactions[col], unit='s', errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
    if "seen_in_mempool" in transactions.columns:
        transactions["seen_in_mempool"] = transactions["seen_in_mempool"].astype(
            int)
    # TODO: it would be great to skip rows that violate the UNIQUE constraint
    transactions.to_sql('mempool_transactions', conn,
                        if_exists='append', index=False)


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS mempool_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tx_id TEXT UNIQUE,
            inputs_hash TEXT UNIQUE,
            child_txid TEXT,
            parent_txid TEXT,

            tx_data TEXT,
            output_amounts TEXT,
            output_weights TEXT,
            conf_block_hash TEXT,

            found_at DATETIME,
            mined_at DATETIME,
            pruned_at DATETIME,

            rbf_fee_total INTEGER,
            min_respend_blocks INTEGER,
            absolute_fee INTEGER,
            fee_rate REAL,
            version INTEGER,
            seen_in_mempool INTEGER,
            waittime INTEGER,
            weight INTEGER,
            size INTEGER,
            total_output_amount INTEGER,
            mempool_size INTEGER,
            mempool_tx_count INTEGER
        );
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_mined_at ON mempool_transactions(mined_at);")

    conn.commit()
    return conn


async def main():
    print("Starting data lake")
    p = argparse.ArgumentParser()
    p.add_argument('--db-path', required=True)
    p.add_argument('--rpc-user', required=True)
    p.add_argument('--rpc-password', required=True)
    p.add_argument('--rpc-host', required=True)
    p.add_argument('--rpc-port', required=True)
    p.add_argument('--output-db', required=False, default="data-lake.db")
    p.add_argument('--limit', required=False, default=10_000)
    p.add_argument('--debug', required=False, default=False)

    args = p.parse_args()

    conn = connect_to_db(args.db_path)
    rpc = await connect_to_rpc(args.rpc_user, args.rpc_password,
                               args.rpc_host, args.rpc_port)
    db = init_db(args.output_db)

    try:
        print("Loading data")
        # Store rbf and mempool data in memory
        rbf_data = load_rbf(conn)
        print(f"Loaded {len(rbf_data)} rbf data")
        mempool_data = load_mempool(conn)
        print(f"Loaded {len(mempool_data)} mempool data")
        for transactions in load_transactions(conn, args.limit):
            print(f"Processing {len(transactions)} transactions")
            transactions = merge_datasets(transactions, rbf_data, mempool_data)
            print(f"Merged {len(transactions)} transactions")
            transactions = await compute_metrics(transactions, rpc, args.debug)
            output_data(transactions, db)
            print(f"Outputted {len(transactions)} transactions")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()


def _entrypoint():
    asyncio.run(main())


if __name__ == "__main__":
    _entrypoint()
