from config_file import user_GM_DB, pass_GM_DB, postgis_pwd

import pandas as pd
import psycopg2.extras

psycopg2.extras.register_uuid()


def connect_db(database:str, user: str, pwd: str, host: str, port: int):
    print("\nConnecting to database")
    return psycopg2.connect(database=database, user=user, password=pwd, host=host, port=port) # , sslmode='enable'
    
def disconnect_db(conn, cursor):
    if conn:
        cursor.close()
        conn.close()
        print("database connection is closed")

def query_db(database: str, user: str, pwd: str, host: str, port: int, callback):
    conn = connect_db(database, user, pwd, host, port)
    cur = conn.cursor()
    res = callback(conn, cur)
    disconnect_db(conn, cur)
    return res


def query_postgres(query, callback=None):
    def cb(conn, cur):
        return pd.read_sql(query, conn, coerce_float=True)
    return query_db("postgres", user_GM_DB, pass_GM_DB, "liradb.compute.dtu.dk", 5435, cb)