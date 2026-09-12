import os

import pandas as pd
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), 'stocks_data.db')

create_login_table_query = f"""
    CREATE TABLE IF NOT EXISTS login_token (
    token TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """

create_scan_table_query = f"""
    CREATE TABLE IF NOT EXISTS stocks_scans (
        symbol TEXT NOT NULL,
        scan_date DATE NOT NULL,
        name TEXT,
        sector TEXT,
        industry TEXT,
        close REAL,
        volume REAL,
        PRIMARY KEY (symbol, scan_date)
    )
    """

def create_stocks_table():
    """
    Create the stocks table in the database.
    """
    try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute(create_login_table_query)
            conn.execute(create_scan_table_query)
            conn.close()
    except Exception as e:
        print(f"Error creating tables: {e}")

if __name__ == "__main__":
    create_stocks_table()