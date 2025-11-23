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


def load_data(conn, limit):
    # Get base transactions first
    transactions = pd.read_sql_query("""
        SELECT
            transactions.*,
            (mined_at - found_at) AS waittime
        FROM transactions
        WHERE
            mined_at IS NOT NULL
            AND found_at IS NOT NULL
            AND PRUNED_AT IS NULL
        ORDER BY found_at DESC
        LIMIT ?
    """, conn, params=(limit,))

    # Get RBF data separately
    rbf_data = pd.read_sql_query("""
        SELECT inputs_hash, MAX(fee_total) as rbf_fee_total
        FROM rbf
        GROUP BY inputs_hash
        ORDER BY created_at DESC
        LIMIT ?
    """, conn, params=(limit,))

    # Get mempool data separately (pre-aggregate by hour)
    mempool_data = pd.read_sql_query("""
        SELECT
            strftime('%Y-%m-%d %H:00:00', datetime(created_at, 'unixepoch')) as hour,
            AVG(size) as mempool_size,
            AVG(tx_count) as mempool_tx_count
        FROM mempool
        GROUP BY hour
        ORDER BY created_at DESC
        LIMIT ?
    """, conn, params=(limit,))

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


block_hash_to_height = {}


async def get_block_height(block_hash: str, rpc: BitcoinRPC):
    if block_hash in block_hash_to_height:
        return block_hash_to_height[block_hash]
    block = await rpc.acall('getblock', [block_hash])
    block_hash_to_height[block_hash] = block['height']
    return block['height']


async def compute_metrics(transactions, rpc: BitcoinRPC, debug: bool, exchange_addresses: pd.DataFrame):
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
    # TODO: disabled is from and too exchange until we have a better source of exchange addresses
    async def get_min_respend_time(txid: str) -> (int, str):
        async with sem:
            try:
                print(f"Computing min_respend_time for txid {txid}")
                confs = []
                exchange_is_sender = False
                exchange_is_receiver = False
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

                        # Get the address of the previous output
                        if 'scriptPubKey' in vin['prevout'] and 'addresses' in vin['prevout']['scriptPubKey']:
                            prev_address = vin['prevout']['scriptPubKey']['address']
                            if prev_address in exchange_addresses['Address']:
                                exchange_is_sender = True
                    else:
                        print(f"Vin {i}: No prevout or height found")

                for i, vout in enumerate(ver_tx['vout']):
                    if 'scriptPubKey' in vout and 'addresses' in vout['scriptPubKey']:
                        vout_address = vout['scriptPubKey']['address']
                        if vout_address in exchange_addresses['Address']:
                            exchange_is_receiver = True

                if len(confs) == 0:
                    print(f"No valid prevout heights found for txid {txid}")
                    return -1, conf_block_hash

                if debug:
                    print(
                        f"exchange_is_sender: {exchange_is_sender}, exchange_is_receiver: {exchange_is_receiver}")

                return min(confs, default=0), conf_block_hash
            except Exception as e:
                print(f"Error fetching transaction {txid}: {e}")
                return -1, ''

    print("Computing weight and size")
    transactions[['weight', 'size', 'output_amounts', 'total_output_amount', 'output_weights']] = transactions['tx_data'].apply(
        lambda tx_hex: pd.Series(get_tx_stats(tx_hex))
    )

    transactions['output_amounts'] = transactions['output_amounts'].apply(lambda x: json.dumps(x))
    transactions['output_weights'] = transactions['output_weights'].apply(lambda x: json.dumps(x))

    print("Computing min_respend_time")
    # Compute min_respend_time for each transaction asynchronously
    min_respend_times = await asyncio.gather(
        *[get_min_respend_time(txid) for txid in transactions['tx_id']]
    )

    transactions['min_respend_time'] = [x[0] for x in min_respend_times]
    transactions['conf_block_hash'] = [x[1] for x in min_respend_times]
    # TODO: disabled is from and too exchange until we have a better source of exchange addresses
    # transactions['exchange_is_sender'] = [x[1] for x in min_respend_times]
    # transactions['exchange_is_receiver'] = [x[2] for x in min_respend_times]
    transactions = transactions[transactions['min_respend_time'] != -1]

    return transactions


def output_data(transactions, output_path):
    transactions.to_hdf(
        output_path,
        key="mempool_transactions",
        mode="a",
        format="table",
        append=True
    )


def load_exchange_addresses():
    csv = """
    Rank,Address,Label/Notes
1,34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo,Binance - Cold Wallet
3,3M219KR5vEneNb47ewrPfWyb5jQ2DjxRP6,Binance - Cold Wallet
4,bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwv97,Bitfinex - Cold Wallet
11,3LYJfcfHPXYJreMsASk2jkn69LWEYKzexb,Binance - BTCB Reserve
15,3MgEAFWu1HKSnZ5ZsC8qf61ZW18xrP5pgd,OKEx
23,bc1qk4m9zv5tnxf2pddd565wugsjrkqkfn90aa0wypj2530f4f7tjwrqntpens,BitMEX - Cold Wallet
38,3JZq4atUahhuA9rLhXLMhhTo133J9rF97j,Bitfinex - Cold Wallet
44,bc1qx2x5cqhymfcnjtg902ky6u5t5htmt7fvqztdsm028hkrvxcl4t2sjtpd9l,Bitbank - Cold Wallet
47,bc1qr4dl5wa7kl8yu792dceg9z5knl2gkn220lk7a9,Crypto.com - Cold Wallet
49,1PJiGp2yDLvUgqeBsuZVCBADArNsk6XEiw,Binance - Cold Wallet
50,3FM9vDYsN2iuMPKWjAcqgyahdwdrUxhbJ3,OKEx
52,34HpHYiyQwg69gFmCq2BGHjF1DZnZnBeBP,Binance - Cold Wallet
54,38UmuUqPCrFmQo4khkomQwZ4VbY2nZMJ67,OKEx
59,bc1qchctnvmdva5z9vrpxkkxck64v7nmzdtyxsrq64,BitMEX
82,1DcT5Wij5tfb3oVViF8mA8p4WrG98ahZPT,OKX
99,bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h,Binance
205,1LnoZawVFFQihU8d8ntxLMpYheZUfyeVAK,OKX
244,3DVJfEsDTPkGDvqPCLC41X85L1B1DQWDyh,OKEx
268,bc1quhruqrghgcca950rvhtrg7cpd7u8k6svpzgzmrjy8xyukacl5lkq0r8l2d,OKX - Hot Wallet
293,3H5JTt42K7RmZtromfTSefcMEFMMe18pMD,Poloniex - Cold Wallet
334,3E5EPMGRL5PC6YDCLcHLVu9ayC3DysMpau,OKEx
336,bc1q4c8n5t00jmj8temxdgcc3t32nkg2wjwz24lywv,Crypto.com
341,bc1q7ramrn7krmgl8ja8vjm9g25a5t98l6kfyqgewe,Coinshares
378,1Kr6QSydW9bFQG1mXiPNNu6WpJGmUa9i1g,Bitfinex
    """
    exchange_addresses = pd.read_csv(StringIO(csv))
    return exchange_addresses


async def main():
    print("Starting data lake")
    p = argparse.ArgumentParser()
    p.add_argument('--db-path', required=True)
    p.add_argument('--rpc-user', required=True)
    p.add_argument('--rpc-password', required=True)
    p.add_argument('--rpc-host', required=True)
    p.add_argument('--rpc-port', required=True)
    p.add_argument('--output-path', required=False, default="data-lake.h5")
    p.add_argument('--limit', required=False, default=10_000)
    p.add_argument('--debug', required=False, default=False)

    args = p.parse_args()

    conn = connect_to_db(args.db_path)
    rpc = await connect_to_rpc(args.rpc_user, args.rpc_password,
                               args.rpc_host, args.rpc_port)
    exchange_addresses = load_exchange_addresses()

    print(f"Loading data from {args.db_path} with limit {args.limit}")
    try:
        print("Loading data")
        transactions = load_data(conn, args.limit)
        print("Computing metrics")
        transactions = await compute_metrics(transactions, rpc, args.debug, exchange_addresses)
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
