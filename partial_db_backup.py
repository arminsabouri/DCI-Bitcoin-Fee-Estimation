#! /usr/bin/env python3

import sqlite3

MAIN_DB_PATH = 'mempool-tracker.db'
BACKUP_DB_PATH = 'mempool-tracker-backup.db'

def copy_schema(main_conn, backup_conn):
    cursor = main_conn.execute("""
        SELECT sql FROM sqlite_master
        WHERE type IN ('table', 'index')
          AND name NOT LIKE 'sqlite_%'
          AND sql IS NOT NULL
    """)
    schemas = cursor.fetchall()
    print(f"Found {len(schemas)} schema statements to execute.")
    for (schema_sql,) in schemas:
        print(f"Creating: {schema_sql.splitlines()[0]}...")
        backup_conn.execute(schema_sql)
    backup_conn.commit()
    print("Schema copy complete.")

def copy_transactions(main_conn, backup_conn, limit=500):
    cursor = main_conn.execute("""
        SELECT inputs_hash, tx_id, tx_data, found_at, mined_at, pruned_at, child_txid, version
        FROM transactions
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()

    print(f"Copying {len(rows)} rows to backup DB...")

    backup_conn.executemany("""
        INSERT INTO transactions (
            inputs_hash, tx_id, tx_data, found_at, mined_at, pruned_at, child_txid, version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    backup_conn.commit()
    print("Backup complete.")
    
def copy_rbf(main_conn, backup_conn, limit=500):
    cursor = main_conn.execute("""
        SELECT inputs_hash, created_at, fee_total, version
        FROM rbf
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()

    print(f"Copying {len(rows)} RBF rows to backup DB...")

    backup_conn.executemany("""
        INSERT INTO rbf (
            inputs_hash, created_at, fee_total, version
        ) VALUES (?, ?, ?, ?)
    """, rows)
    backup_conn.commit()
    print("RBF backup complete.")
    
def copy_mempool(main_conn, backup_conn, limit=500):
    cursor = main_conn.execute("""
        SELECT tx_id, created_at, size, tx_count, block_height, block_hash, version
        FROM mempool
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()

    print(f"Copying {len(rows)} mempool rows to backup DB...")

    backup_conn.executemany("""
        INSERT INTO mempool (
            tx_id, created_at, size, tx_count, block_height, block_hash, version
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, rows)
    backup_conn.commit()
    print("Mempool backup complete.")

def main():
    main_conn = sqlite3.connect(MAIN_DB_PATH)
    backup_conn = sqlite3.connect(BACKUP_DB_PATH)

    # Dynamically copy the schema
    copy_schema(main_conn, backup_conn)

    # Copy 500 rows from transactions
    copy_transactions(main_conn, backup_conn)
    copy_rbf(main_conn, backup_conn)
    copy_mempool(main_conn, backup_conn)

    main_conn.close()
    backup_conn.close()

if __name__ == "__main__":
    main()
