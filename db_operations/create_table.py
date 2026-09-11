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

create_master_table_query = f"""
    CREATE TABLE IF NOT EXISTS stocks_metadata (
        symbol TEXT PRIMARY KEY,
        name TEXT,
        sector TEXT,
        industry TEXT,
        exchange TEXT,
        sharia_complaint TEXT,
        mkt_cap TEXT
    )
    """

def create_stocks_login_token_table():
    """
    Create the stocks_login_token table in the database.
    """
    try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute(create_login_table_query)
            conn.execute(create_master_table_query)
            conn.close()
    except Exception as e:
        print(f"Error creating create_login_table table: {e}")

if __name__ == "__main__":
    create_stocks_login_token_table()