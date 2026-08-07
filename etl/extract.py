import requests
import os
import yfinance as yf
from datetime import datetime, timedelta
import pandas_datareader as pdr
import pandas_datareader.data as web
#from pandas_datareader.stooq import StooqDailyReader
import pandas as pd


FRED_API_KEY = os.getenv("FRED_API_KEY")

def fetch_alphavantage() -> dict:
    api_key = os.getenv("ALPHA_VANTAGE_KEY")
    url = (
        "https://www.alphavantage.co/query"
        f"?function=COPPER&interval=monthly&apikey={api_key}"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()

def fetch_fred(limit=10000):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "PCOPPUSDM",
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit,
        "observation_start": "2000-01-01",
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()["observations"]

FRED_SERIES = {
    "FEDFUNDS": "Federal Funds Rate",
    "CPIAUCSL": "CPI Inflation",
    "INDPRO": "Industrial Production",
    "CSUSHPISA": "Housing Price Index",
    "MANEMP": "Manufacturing Employment",
}

def fetch_fred_series(series_id: str) -> list[tuple]:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
            "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": "2000-01-01",
    }
    obs = requests.get(url, params=params).json()["observations"]
    return [
        (row["date"], series_id, float(row["value"]), "FRED")
        for row in obs if row["value"] != "."
    ]


def fetch_yahoofinance(start_date=None, end_date=None):
    """
    Fetch daily copper futures prices from Yahoo Finance (COMEX HG=F)
    Returns list of dicts with date and price in USD per pound.
    """
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365*5)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    ticker = yf.Ticker("HG=F")
    df = ticker.history(start=start_date, end=end_date)
#    df = pdr.get_data_stooq("HG.F", start="2000-01-01")
#    df.index.name = "date"
#    df = yf.download("HG=F", start="2020-01-01", auto_adjust=True)
    items = []
    for date, price in df["Close"].items():
        items.append({
            "date": str(date),
            "price_usd": round(price, 4),
        }) 
    return items    
#    df = pdr.DataReader("HG.F", "stooq", start="2020-01-01")
#    df = df.sort_index()
#    df = StooqDailyReader("HG.F", start="2020-01-01").read()
#    df = web.DataReader('CU', 'stooq')
#    df = df.sort_index()
#    close = df["Close"]
#    if isinstance(close, pd.DataFrame):
#        close = close.iloc[:, 0]  
#
#    copper_dict = {
#    str(date): round(float(price), 4)
#    for date, price in close.items()
#}
##    return items
#
#    return copper_dict
























