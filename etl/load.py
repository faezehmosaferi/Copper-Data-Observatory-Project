import psycopg2
import os

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS copper_prices (
    id          SERIAL PRIMARY KEY,
    recorded_at DATE NOT NULL,
    price_usd   NUMERIC(10, 4) NOT NULL,
    unit        TEXT,
    source      TEXT,
    fetched_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (recorded_at, source)
);

CREATE TABLE IF NOT EXISTS macro_indicators (
    date DATE NOT NULL,
    series_id VARCHAR(30) NOT NULL,
    value NUMERIC,
    source TEXT,
    PRIMARY KEY (date, series_id)
);
"""

INSERT_PRICE = """
INSERT INTO copper_prices (recorded_at, price_usd, unit, source)
VALUES (%(recorded_at)s, %(price_usd)s, %(unit)s, %(source)s)
ON CONFLICT (recorded_at, source) DO UPDATE
SET price_usd = EXCLUDED.price_usd,
    source = EXCLUDED.source;
"""
INSERT_MACRO_IND = """
INSERT INTO macro_indicators (date, series_id, value)
VALUES %s
ON CONFLICT (date, series_id)
DO UPDATE SET value = EXCLUDED.value;
"""

def get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", 5432),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )

def ensure_table(conn):
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE)
    conn.commit()


def load_to_copper_prices(conn, table_name, records):
    placeholders = ",".join(["%s"] * len(records[0]))
    cols = ",".join(records[0].keys())
    query = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"
    with conn.cursor() as cur:
        cur.executemany(query, [tuple(r.values()) for r in records])
    conn.commit()

def load_to_macro_indicators(conn, table_name, records):
    placeholders = ",".join(["%s"] * len(records[0]))
    cols = "date, series_id, value, source"
    query = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"
    with conn.cursor() as cur:
        cur.executemany(query, records)
    conn.commit()

def insert_price(conn, records):
    cursor = conn.cursor()
    cursor.executemany(INSERT_PRICE, records)
    conn.commit()
    cursor.close()


def insert_macro_ind(conn, records):
    cursor = conn.cursor()
    cursor.executemany(INSERT_MACRO_IND, records)
    conn.commit()
    cursor.close()
