import sys
from dotenv import load_dotenv
load_dotenv()

from extract import fetch_alphavantage, fetch_fred, fetch_yahoofinance, fetch_fred_series
from transform import parse_alphavantage_latest, parse_fred_latest, parse_fred_all, transform_yf_copper_price
from load import get_conn, ensure_table, insert_price, insert_macro_ind, load_to_copper_prices, load_to_macro_indicators

FRED_SERIES = {
    "FEDFUNDS": "Federal Funds Rate",
    "CPIAUCSL": "CPI Inflation",
    "INDPRO": "Industrial Production",
    "CSUSHPISA": "Housing Price Index",
    "MANEMP": "Manufacturing Employment",
}

def main():
    conn = get_conn()
    try:
    # Alpha Vantage

#        raw_av = fetch_alphavantage()
#        if raw_av:
#             record_av = parse_alphavantage_latest(raw_av)
#             if record_av:
#                 ensure_table(conn)
#                 insert_price(conn, record_av)
#                 print(f"[AV] inserted: {record_av['recorded_at']} → {record_av['price_usd']}")

    # FRED

        raw_fred = fetch_fred()

        raw_fred_series = []
        for series_id in FRED_SERIES.keys():
            raw_fred_series.extend(fetch_fred_series(series_id))
        if raw_fred:
            records_fred = parse_fred_all(raw_fred)               
            if records_fred and raw_fred_series:
               ensure_table(conn)
               load_to_copper_prices(conn, 'copper_prices', records_fred)
               load_to_macro_indicators(conn, 'macro_indicators', raw_fred_series)
#               insert_price(conn, records_fred)
#               insert_macro_ind=(conn, raw_fred_series)
               print(f"[FRED] inserted {len(records_fred)} records")
               print(f"[FRED] inserted {len(records_fred)} macro-ind records")

     # yahoofinance

#        raw = fetch_yahoofinance(start_date="2020-01-01")
#        rows = transform_yf_copper_price(raw)
#        print(f"Loaded {len(rows)} copper price records from yahoofinance.")

    except Exception as e: 
        print(f"erorr: {e}", file=sys.stderr)
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
