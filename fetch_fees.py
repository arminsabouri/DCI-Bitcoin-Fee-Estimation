from load_from_sqlite import load_data_from_sqlite
import requests
import time
import json
import pandas as pd

def fetch_transaction(txid):
    url = f"https://blockstream.info/testnet/api/tx/{txid}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch {txid}, status code: {response.status_code}")
    except Exception as e:
        print(f"Exception while fetching {txid}: {e}")
    return None

def compute_fee(tx_data):
    try:
        value_in = sum(vin['prevout']['value'] for vin in tx_data['vin'] if 'prevout' in vin)
        value_out = sum(vout['value'] for vout in tx_data['vout'])
        return value_in - value_out
    except Exception as e:
        print(f"Failed to compute fee for tx: {e}")
        return None

def main():
    txdf, _ = load_data_from_sqlite()
    fee_data = []

    for i, txid in enumerate(txdf['tx_id'].dropna().unique()[:100]): 
        print(f"Fetching {i+1}/100: {txid}")
        tx_data = fetch_transaction(txid)
        if tx_data:
            fee = compute_fee(tx_data)
            if fee is not None:
                fee_data.append({'tx_id': txid, 'fee': fee})
                print(f"  → Fee: {fee}")
            else:
                print(f"  → Fee not computable for {txid}")
        else:
            print(f"  → No tx data for {txid}")
        time.sleep(0.5)

    if fee_data:
        fee_df = pd.DataFrame(fee_data)
        fee_df.to_csv("fees_fetched.csv", index=False)
        print("Fee data saved to fees_fetched.csv")
    else:
        print("⚠️ No fee data collected — check connection and txid validity.")

if __name__ == "__main__":
    main()