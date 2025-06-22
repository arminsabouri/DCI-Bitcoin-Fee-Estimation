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
        ORDER BY found_at DESC
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

    # Verify copy
    main_cursor = main_conn.execute("""
        SELECT inputs_hash, found_at, mined_at
        FROM transactions
        LIMIT ?
    """, (limit,))
    backup_cursor = backup_conn.execute("SELECT inputs_hash, found_at, mined_at FROM transactions")

    main_data = {row[0]: (row[1], row[2]) for row in main_cursor}
    backup_data = {row[0]: (row[1], row[2]) for row in backup_cursor}

    # Compare and report differences
    mismatches = 0
    for inputs_hash, (main_found, main_mined) in main_data.items():
        backup_found, backup_mined = backup_data[inputs_hash]
        if main_found != backup_found:
            mismatches += 1
            print(f"Mismatch for transaction {inputs_hash[:8]}...")
            print(f"  Main DB:    found_at={main_found}, mined_at={main_mined}")
            print(f"  Backup DB:  found_at={backup_found}, mined_at={backup_mined}")

    print(f"\nFound {mismatches} mismatches out of {len(rows)} rows")

def copy_rbf(main_conn, backup_conn, limit=500):
    cursor = main_conn.execute("""
        SELECT inputs_hash, created_at, fee_total, version
        FROM rbf
        ORDER BY created_at DESC
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
        ORDER BY created_at DESC
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

def dump_to_sql(backup_db_path, sql_path):
    conn = sqlite3.connect(backup_db_path)
    
    with open(sql_path, 'w') as f:
        # Get schema
        for line in conn.iterdump():
            f.write(f'{line}\n')
            
    conn.close()
    print(f"Successfully dumped database to {sql_path}")


def main():
    main_conn = sqlite3.connect(MAIN_DB_PATH)
    backup_conn = sqlite3.connect(BACKUP_DB_PATH)

    # Dynamically copy the schema
    copy_schema(main_conn, backup_conn)

    # Copy 500 rows from transactions
    copy_transactions(main_conn, backup_conn, limit=100_000)
    copy_rbf(main_conn, backup_conn, limit=100_000)
    copy_mempool(main_conn, backup_conn, limit=100_000)

    main_conn.close()
    backup_conn.close()

    # Dump the backup DB to a SQL file
    dump_to_sql(BACKUP_DB_PATH, 'mempool-tracker-backup.sqlite')
if __name__ == "__main__":
    main()
