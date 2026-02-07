from __future__ import annotations

from dataclasses import dataclass
from typing import Union, List, Dict
from collections import Counter

from bitcoin.core import CTransaction, x
from bitcoin.core.script import CScript, OP_RETURN

# Inscription envelope: OP_0 (0x00) OP_IF (0x63)
OP_0 = 0x00
OP_IF = 0x63
_INSCRIPTION_ENVELOPE = bytes([OP_0, OP_IF])

# BIP-141 segwit marker and flag
_WITNESS_MARKER = b"\x00\x01"

import argparse
import asyncio
import sqlite3
import struct

import pandas as pd
from bitcoinrpc import BitcoinRPC


BytesLike = Union[bytes, bytearray, memoryview]
HexOrBytes = Union[str, BytesLike]


@dataclass
class TxFeatures:
    n_in: int
    n_out: int
    out_values: List[int]              # sats
    max_equal_group: int
    unique_out_values: int
    has_op_return: bool
    has_inscription: bool


def parse_tx(tx_data: HexOrBytes) -> CTransaction:
    if isinstance(tx_data, str):
        raw = x(tx_data)  # hex -> bytes
    else:
        raw = bytes(tx_data)
    return CTransaction.deserialize(raw)


def outputs_sats(tx: CTransaction) -> List[int]:
    return [int(vout.nValue) for vout in tx.vout]


def max_equal_value_group(values: List[int], tolerance_sats: int = 1000) -> int:
    if not values:
        return 0
    buckets = [round(v / tolerance_sats) for v in values]
    counts = Counter(buckets)
    return max(counts.values())


# -----------------------------
# OP_RETURN detection
# -----------------------------
def has_op_return_output(tx: CTransaction) -> bool:
    """
    True if any output scriptPubKey begins with OP_RETURN (standard null-data).
    """
    for vout in tx.vout:
        spk = CScript(vout.scriptPubKey)
        # Robust check: first opcode is OP_RETURN
        try:
            first = next(iter(spk))
        except StopIteration:
            continue
        if first == OP_RETURN:
            return True
    return False


# -----------------------------
# Inscription detection (heuristic)
# -----------------------------

def has_inscription_heuristic(tx: CTransaction, raw_tx_bytes: bytes | None = None) -> bool:
    """
    Heuristic detection for Taproot inscriptions:
    - Look in witness stack for the opcode sequence OP_0 OP_IF anywhere in the
      serialized script (ordinals-style envelope). We search raw bytes so we
      find it even when it appears after other opcodes (e.g. in Taproot script).
    """
    wit = getattr(tx, "wit", None)
    if wit is None:
        return False
    vtxinwit = getattr(wit, "vtxinwit", None)
    if vtxinwit is None:
        return False
    for in_wit in vtxinwit:
        script_witness = getattr(in_wit, "scriptWitness", None)
        if script_witness is None:
            continue
        stack = getattr(script_witness, "stack", None)
        if not stack:
            continue
        for item in stack:
            if isinstance(item, (bytes, bytearray)) and _INSCRIPTION_ENVELOPE in bytes(item):
                return True
    return False


def extract_features(
    tx: CTransaction,
    tolerance_sats: int = 1000,
    raw_tx_bytes: bytes | None = None,
) -> TxFeatures:
    out_vals = outputs_sats(tx)
    return TxFeatures(
        n_in=len(tx.vin),
        n_out=len(tx.vout),
        out_values=out_vals,
        max_equal_group=max_equal_value_group(out_vals, tolerance_sats=tolerance_sats),
        unique_out_values=len(set(out_vals)),
        has_op_return=has_op_return_output(tx),
        has_inscription=has_inscription_heuristic(tx, raw_tx_bytes=raw_tx_bytes),
    )


# Heuristics: consolidation / batch / coinjoin (equal-output style) ---

def is_consolidation(feat: TxFeatures) -> bool:
    if feat.n_in < 3:
        return False
    # Ideally we check for outputs of the same script type
    if feat.n_out == 1 or feat.n_out == 2:
        return True
    if feat.n_out == 3:
        mn = min(feat.out_values)
        mx = max(feat.out_values)
        if mn > 0 and mx > 5 * mn:
            return True
    return False


def is_batch_payment(feat: TxFeatures) -> bool:
    if feat.n_out < 5:
        return False
    if feat.n_in > 3:
        return False
    if feat.unique_out_values < int(0.8 * feat.n_out):
        return False
    return True


def is_coinjoin_equal_output(feat: TxFeatures) -> bool:
    if feat.n_in < 5 or feat.n_out < 5:
        return False
    if feat.max_equal_group < 5:
        return False
    if feat.max_equal_group < 0.4 * feat.n_out:
        return False
    return True


def classify_tx(tx_data: HexOrBytes, tolerance_sats: int = 1000) -> Dict[str, object]:
    raw = bytes(x(tx_data) if isinstance(tx_data, str) else tx_data)
    tx = parse_tx(tx_data)
    feat = extract_features(tx, tolerance_sats=tolerance_sats, raw_tx_bytes=raw)

    # Core structural label (order matters)
    if is_coinjoin_equal_output(feat):
        label = "coinjoin"
    elif is_consolidation(feat):
        label = "consolidation"
    elif is_batch_payment(feat):
        label = "batch_payment"
    else:
        label = "normal"

    # Extra tags layered on top
    tags = []
    if feat.has_op_return:
        tags.append("op_return")
    if feat.has_inscription:
        tags.append("inscription")

    # Normal tx that carries data (op_return or inscription) -> datacarrying
    if label == "normal" and tags:
        label = "datacarrying"

    return {
        "label": label,
        "tags": tags,
        "n_in": feat.n_in,
        "n_out": feat.n_out,
        "max_equal_group": feat.max_equal_group,
        "unique_out_values": feat.unique_out_values,
        "tolerance_sats": tolerance_sats,
    }


async def connect_to_rpc(rpc_user: str, rpc_password: str, rpc_host: str, rpc_port: str):
    """Connect to Bitcoin Core RPC (same pattern as main.py)."""
    host = f"http://{rpc_host}:{rpc_port}"
    rpc = BitcoinRPC.from_config(host, (rpc_user, rpc_password), timeout=10)
    test_result = await rpc.getblockchaininfo()
    print(f"Connected to bitcoind. Block height: {test_result['blocks']}")
    return rpc


async def main_async() -> None:
    """
    Read tx_id list from main.py output DB, fetch full tx (with witness) via
    Bitcoin Core RPC, annotate with classify_tx, and write results to pickle.
    """
    p = argparse.ArgumentParser(
        description="Annotate mempool_transactions: fetch full tx from Bitcoin RPC and classify (incl. inscription)."
    )
    p.add_argument("--db-path", default="data-lake.db", help="Path to output SQLite DB from main.py")
    p.add_argument("--rpc-user", required=True, help="Bitcoin RPC user")
    p.add_argument("--rpc-password", required=True, help="Bitcoin RPC password")
    p.add_argument("--rpc-host", required=True, help="Bitcoin RPC host")
    p.add_argument("--rpc-port", required=True, help="Bitcoin RPC port")
    p.add_argument("--limit", type=int, default=None, help="Max rows to process (default: all)")
    p.add_argument("--tolerance-sats", type=int, default=1000, help="Tolerance in sats for equal-value grouping")
    p.add_argument(
        "--output", "-o",
        default="mempool_transactions_annotated.pkl",
        help="Output path for pickled dataframe",
    )
    args = p.parse_args()

    conn = sqlite3.connect(args.db_path)
    query = "SELECT tx_id FROM mempool_transactions"
    if args.limit is not None:
        query += f" LIMIT {args.limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("No rows in mempool_transactions.")
        return

    tx_ids = df["tx_id"].tolist()
    rpc = await connect_to_rpc(args.rpc_user, args.rpc_password, args.rpc_host, args.rpc_port)
    sem = asyncio.Semaphore(16)

    async def fetch_and_annotate(tx_id: str) -> dict:
        async with sem:
            try:
                # verbose=0 returns raw hex including witness (BIP-141)
                raw_hex = await rpc.acall("getrawtransaction", [tx_id, False])
                result = classify_tx(raw_hex, tolerance_sats=args.tolerance_sats)
                result["tx_id"] = tx_id
                return result
            except Exception as e:
                print(f"Error annotating tx {tx_id}: {e}")
                return {
                    "tx_id": tx_id,
                    "label": "error",
                    "tags": [],
                    "n_in": -1,
                    "n_out": -1,
                    "max_equal_group": -1,
                    "unique_out_values": -1,
                    "tolerance_sats": args.tolerance_sats,
                    "_error": str(e),
                }

    print(f"Fetching and annotating {len(tx_ids)} transactions via RPC...")
    results = await asyncio.gather(*[fetch_and_annotate(tx_id) for tx_id in tx_ids])
    annotations = pd.DataFrame(results)
    annotations.to_pickle(args.output)
    print(f"Wrote {len(annotations)} rows to {args.output}.")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
