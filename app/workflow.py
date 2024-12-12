def save_database(df, db_name='stock_data.db', table_name='stocks'):

    import sqlite3

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Drop the table if it exists to avoid duplicates
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

    # Create the table with appropriate columns
    df.to_sql(table_name, conn, if_exists='replace', index=False)

    conn.commit()
    conn.close()

    print(f"Data saved to {db_name} as table '{table_name}'.")