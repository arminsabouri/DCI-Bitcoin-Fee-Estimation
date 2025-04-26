from load_from_sqlite import load_data_from_sqlite
import time
import requests
import json

def fetch_transaction(txid):
    url = f'https://blockstream.info/api/tx/{txid}'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None

def is_nft_transaction(tx_data):
    for vout in tx_data.get('vout', []):
        scriptpubkey = vout.get('scriptpubkey', '')
        if scriptpubkey.startswith('6a'):
            op_return_data = scriptpubkey[2:]
            try:
                data_bytes = bytes.fromhex(op_return_data)
                data_str = data_bytes.decode('utf-8', errors='ignore')
                if 'nft' in data_str.lower() or data_str.startswith('ORDINAL'):
                    return True
            except:
                continue
    return False

def main():
    txdf, _ = load_data_from_sqlite()
    nft_txids = []
    for i, txid in enumerate(txdf['tx_id'].head(50)):
        print(f"Checking {txid} ({i+1})")
        tx_data = fetch_transaction(txid)
        if tx_data and is_nft_transaction(tx_data):
            print(f"NFT Transaction: {txid}")
            nft_txids.append(txid)
    print("\nDetected NFT Transactions:")
    for txid in nft_txids:
        print(txid)

if __name__ == "__main__":
    main()
