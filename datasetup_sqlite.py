from load_from_sqlite import load_data_from_sqlite
import pandas as pd

def setup():
    txdf, rbfd = load_data_from_sqlite()
    txdf.dropna(subset=['found_at', 'mined_at'], inplace=True)
    # using this calculation as waittime because I do not know the actual one
    txdf['waittime'] = (txdf['mined_at'] - txdf['found_at']) / 60
    print(txdf[['tx_id', 'waittime']].head())

if __name__ == "__main__":
    setup()
