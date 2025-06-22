from load_from_sqlite import load_data_from_sqlite
import pandas as pd
import matplotlib.pyplot as plt

def test_distribution():
    txdf, _ = load_data_from_sqlite()
    txdf.dropna(subset=['found_at', 'mined_at'], inplace=True)
    txdf['waittime'] = (txdf['mined_at'] - txdf['found_at']) / 60
    txdf['waittime'].hist(bins=50)
    plt.title("Distribution of Wait Time (mins)")
    plt.xlabel("Wait Time")
    plt.ylabel("Frequency")
    plt.show()

if __name__ == "__main__":
    test_distribution()
