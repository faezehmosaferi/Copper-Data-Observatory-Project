import sys
from dotenv import load_dotenv
load_dotenv()

from extract import fetch_alphavantage, fetch_fred, fetch_yahoofinance, fetch_fred_series, scrape_caixin_pmi
from transform import parse_alphavantage_latest, parse_fred_latest, parse_fred_all, transform_yf_copper_price
from load import get_conn, ensure_table, insert_price, insert_macro_ind, load_to_copper_prices, load_to_macro_indicators
from read_excel import  build_tuples

FRED_SERIES = {
    "FEDFUNDS": "Federal Funds Rate",
    "PIORECRUSDM":"Global price of ore",
    "CPIAUCSL": "CPI Inflation",
    "INDPRO": "Industrial Production",
    "CSUSHPISA": "Housing Price Index",
    "MANEMP": "Manufacturing Employment",
    "XTIMVA01CNM657S" : "Imported commodities for china",
    "COCHNZ335" : "Electrical equipment china",
    "COCHNZ333" : "Machinary manufacturing  china",
    "CCRETT01CNM661N" : "REER china"
}
UNITS = { "Percent", "USD/Ton", "index 1982-1984=100", "index 2017=100", "index 2000=100",
         "thousands of persons", "growth rate over previous period", "index 2012-Jun=100", "index 2012-Jun=100",
         "index 2015=100"
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

#        raw_fred = fetch_fred()
        copper_records = fetch_fred()
#        caixin_pmi_records = scrape_caixin_pmi()
        lme_copper_records = build_tuples()
        fred_series_records = []
        for series_id, unit in zip(FRED_SERIES.keys(), UNITS):
            fred_series_records.extend(fetch_fred_series(series_id, unit))

#        if raw_fred:
#            records_fred = parse_fred_all(raw_fred)               
        if copper_records and fred_series_records:
               ensure_table(conn)
#               load_to_copper_prices(conn, 'copper_prices',  copper_records)
               load_to_macro_indicators(conn, 'macro_indicators', copper_records)
               load_to_macro_indicators(conn, 'macro_indicators', fred_series_records)
               load_to_macro_indicators(conn, 'macro_indicators', lme_copper_records) 
#               if caixin_pmi_records:
#                   load_to_macro_indicators(conn, 'macro_indicators', caixin_pmi_records)


#               insert_price(conn, records_fred)
#               insert_macro_ind=(conn, raw_fred_series)
               print(f"[FRED] inserted {len(copper_records)} records")
               print(f"[FRED] inserted {len(fred_series_records)} macro-ind records")
               print(f"[westmetal.com] inserted {len(lme_copper_records)} lme_copper records")
#               print(f"[investing.com] inserted {len(caixin_pmi_records)} caixin_pmi_records")



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
