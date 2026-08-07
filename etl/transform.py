import pandas as pd
from datetime import datetime

def parse_alphavantage_latest(raw: dict) -> dict:
    latest = raw["data"][0]  # جدیدترین رکورد
    return {
        "recorded_at": datetime.strptime(latest["date"], "%Y-%m-%d").date(),
        "price_usd": float(latest["value"]),
        "unit": raw.get("unit", "usd per metric ton"),
        "source": "alphavantage",
    }

def parse_fred_latest(observations):
    latest = observations[0]  # چون sort_order=desc
    return {
        "recorded_at": datetime.strptime(latest["date"], "%Y-%m-%d").date(),
        "price_usd": float(latest["value"]),
        "unit": "usd per metric ton",
        "source": "fred",
    }

def parse_fred_all(observations):
    records = []
    for obs in observations:
        records.append({
            "recorded_at": datetime.strptime(obs["date"], "%Y-%m-%d").date(),
            "price_usd": float(obs["value"]),
            "unit": "usd per metric ton",
            "source": "fred",
        })
    return records

def transform_yf_copper_price(items):
    """
    Convert USD/lb to USD/metric ton (1 metric ton = 2204.62 lb)
    """
    rows = []
    for item in items:
        rows.append({
            "recorded_at": datetime.strptime(item["date"],"%Y-%m-%d").date(),
            "price_usd": round(item["price"] * 2204.62, 2),
            "unit": "usd per metric ton",
            "source": "yahoo_finance",
        })
    return rows


















