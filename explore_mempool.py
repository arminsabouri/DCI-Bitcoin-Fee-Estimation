import sqlite3

def print_table_sample(db_file, table_name, limit=5):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Get column names
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = [col[1] for col in cursor.fetchall()]
    print(f"\n--- {table_name.upper()} ---")
    print(" | ".join(columns))

    # Get sample rows
    cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit};")
    rows = cursor.fetchall()
    for row in rows:
        print(" | ".join(str(cell) if cell is not None else 'NULL' for cell in row))

    conn.close()

if __name__ == "__main__":
    db_path = "/Users/matth410/Desktop/DCI/mempool-tracker.db"  # Adjust path if needed
    tables = ["transactions", "rbf"]

    for table in tables:
        print_table_sample(db_path, table)
