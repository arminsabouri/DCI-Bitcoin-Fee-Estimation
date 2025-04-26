import sqlite3
import pandas as pd

def load_data_from_sqlite(db_path="/Users/matth410/Desktop/DCI/mempool-tracker.db"):
    conn = sqlite3.connect(db_path)
    txdf = pd.read_sql_query("SELECT * FROM transactions", conn)
    rbfd = pd.read_sql_query("SELECT * FROM rbf", conn)
    conn.close()
    return txdf, rbfd
