#! /usr/bin/env python3

import sqlite3

MAIN_DB_PATH = 'mempool-tracker.db'

def len_found_at(main_conn, , limit=500):
    cursor = main_conn.execute("""
        SELECT fo
        FROM transactions
        WHERE found_at = 0
    """)
    rows = cursor.fetchall()
    print(f"Copying {len(rows)} rows to backup DB...")

def main():
    main_conn = sqlite3.connect(MAIN_DB_PATH)

    len_found_at(main_conn)
    main_conn.close()

if __name__ == "__main__":
    main()
