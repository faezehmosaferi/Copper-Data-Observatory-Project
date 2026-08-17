import requests
import os
import yfinance as yf
from datetime import datetime, timedelta
import pandas_datareader as pdr
import pandas_datareader.data as web
#from pandas_datareader.stooq import StooqDailyReader
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv
load_dotenv()


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
        "observation_start": "2012-06-01",
        "observation_end": "2026-05-01",
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    obs = r.json()["observations"]
    return [
        (row["date"], 'PCOPPUSDM', float(row["value"]), "USD/Ton", "FRED")
        for row in obs if row["value"] != "."]
#    return r.json()["observations"]

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

def fetch_fred_series(series_id: str, unit) -> list[tuple]:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
            "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": "2012-06-01",
        "observation_end": "2026-05-01",
    }
    resp = requests.get(url, params=params).json()
    if "observations" not in resp:
        raise ValueError(f"FRED API error for {series_id}: {resp}")
    obs = resp["observations"]
    return [
        (row["date"], series_id, float(row["value"]), unit, "FRED")
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


#HEADERS = {
#    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
#    "Accept-Language": "en-US,en;q=0.9",
#    "Referer": "https://www.investing.com/",
#}
#def scrape_caixin_pmi(url: str) -> list[tuple]:
#    with sync_playwright() as p:
#        browser = p.chromium.launch(headless=True)
#        page = browser.new_page()
#        page.set_extra_http_headers({
#    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
#    "Accept-Language": "en-US,en;q=0.9",
#})
#        page.goto(url, wait_until="networkidle")
#        page.wait_for_selector("table.your-selector", timeout=30000)
#        soup = BeautifulSoup(page.content(), "html.parser")
#        headers = [th.get_text(strip=True) for th in soup.select("table thead th")]
#        print(headers)
#        browser.close()
#
#        for tr in soup.select("table tbody tr"):
#          cols = [td.get_text(strip=True) for td in tr.find_all("td")]
#          if len(cols) >= 2:
#             try:
#                 rows.append((cols[0], "CAIXIN_PMI", float(cols[1]), "investing.com"))
#             except (ValueError, TypeError):
#                 value = none
#
#    rows.sort(key=lambda r: r[0])
#    return rows

API_URL = "https://endpoints.investing.com/pd-instruments/v1/calendars/economic/events/753/occurrences?domain_id=1&limit=1000"

def scrape_caixin_pmi() -> list[tuple]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ))
        page = context.new_page()

        response = page.request.get(API_URL, headers={"Accept-Language": "en-US,en;q=0.9"})
        data = response.json()
        print(response.status)
        print(response.text()[:500])
        browser.close()

    rows = []
    print(response.status)
    print(response.text()[:500])

    print(type(data))
    for item in data.get("data", []):
        print(type(data.get("data")), data.get("data")[:3])
        if not isinstance(item, dict):
           continue
        date = item.get("date", "")[:10]
        try:
            actual = float(item["actual"]) if item.get("actual") not in (None, "") else None
        except (ValueError, TypeError):
            actual = None
        if date and actual is not None:
            rows.append((date, "CAIXIN_PMI", actual, "investing.com"))

    rows.sort(key=lambda r: r[0])
    return rows




















