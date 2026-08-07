import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "copper_db"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}

# Analysis start date
START_DATE = '2025-12-01'
conn = psycopg2.connect(**DB_CONFIG)


# === 1. Copper Price Data (Monthly) ===
copper_df = pd.read_sql("""
    SELECT recorded_at AS date, price_usd AS close
    FROM copper_prices
    WHERE recorded_at >= %s
    ORDER BY recorded_at
""", conn, params=(START_DATE,))

copper_df['date'] = pd.to_datetime(copper_df['date'])
copper_df.set_index('date', inplace=True)
copper_df.rename(columns={'close': 'copper_price'}, inplace=True)
# Ensure monthly index (Month Start)
copper_df.index = copper_df.index.to_period('M').to_timestamp()

# === 2. FRED Economic Indicators (Monthly) ===
fred_df = pd.read_sql("""
    SELECT date, series_id, value
    FROM macro_indicators
    WHERE date >= %s
    ORDER BY date
""", conn, params=(START_DATE,))
fred_df['date'] = pd.to_datetime(fred_df['date'])
fred_pivot = fred_df.pivot(index='date', columns='series_id', values='value')
fred_pivot.index = fred_pivot.index.to_period('M').to_timestamp()

# === 3. News Sentiment (Daily → Aggregate to Monthly) ===
news_df = pd.read_sql("""
    SELECT published_at::date as date, sentiment_score
    FROM news_articles 
    WHERE published_at >= %s 
      AND sentiment_score IS NOT NULL
""", conn, params=(START_DATE,))
news_df['date'] = pd.to_datetime(news_df['date'])
news_df.set_index('date', inplace=True)

# Resample daily sentiment to monthly aggregates
news_monthly = news_df.resample('MS').agg(
    avg_sentiment=('sentiment_score', 'mean'),
    news_count=('sentiment_score', 'count')
)

conn.close()

# === Monthly Integration ===
df = copper_df.join(fred_pivot, how='left').join(news_monthly, how='left')
df['avg_sentiment'].fillna(0, inplace=True)
df['news_count'].fillna(0, inplace=True)

# === Monthly Price Change Percentage ===
df['price_change_pct'] = df['copper_price'].pct_change() * 100

# === Price Regime (3-month Moving Average instead of 20-day) ===
df['ma_3'] = df['copper_price'].rolling(3).mean()
df['regime'] = np.where(df['copper_price'] > df['ma_3'], 'Bullish', 'Bearish')

# === Lagged Correlation (In Months) ===
lags = [1, 2, 3]
correlations = {}
for lag in lags:
    df[f'price_change_lag_{lag}'] = df['price_change_pct'].shift(-lag)
    correlations[f'Lag {lag} month'] = df['avg_sentiment'].corr(df[f'price_change_lag_{lag}'])

print("=== Lagged Correlation: Sentiment vs Future Price Change (Monthly) ===")
for lag, corr in correlations.items():
    print(f"{lag}: {corr:.4f}")

merged_df = pd.merge(copper_df, news_monthly, on='date', how='inner').reset_index()

# === Visualization 1 ====    
fig, ax1 = plt.subplots(figsize=(12, 6))

color_sent = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Avg Sentiment Score', color=color_sent)
ax1.plot(merged_df['date'], merged_df['avg_sentiment'], color=color_sent, marker='o', label='Sentiment')
ax1.tick_params(axis='y', labelcolor=color_sent)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
color_price = 'tab:orange'
ax2.set_ylabel('Copper Price (USD/lb)', color=color_price)
ax2.plot(merged_df['date'], merged_df['copper_price'], color=color_price, marker='s', label='Price')
ax2.tick_params(axis='y', labelcolor=color_price)

fig.tight_layout()
plt.title('Sentiment Score vs Copper Price (Monthly)')
plt.savefig('sentiment_vs_price.png', dpi=300)
plt.show()

# === Visualization ===
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

# Copper price plot
axes[0].plot(df.index, df['copper_price'], marker='o', color='black')
axes[0].set_ylabel('Copper Price ($)')
axes[0].grid(True, alpha=0.3)

# Average monthly sentiment
axes[1].bar(df.index, df['avg_sentiment'], width=15, color='steelblue')
axes[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
axes[1].set_ylabel('Avg Monthly Sentiment')
axes[1].grid(True, alpha=0.3)

# Monthly news count
axes[2].bar(df.index, df['news_count'], width=15, color='coral')
axes[2].set_ylabel('Monthly News Count')
axes[2].set_xlabel('Month')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('integrated_analysis_monthly.png', dpi=150, bbox_inches='tight')
print("✅ Saved: integrated_analysis_monthly.png")

df.to_csv('integrated_data_monthly.csv')
print("✅ Monthly data saved: integrated_data_monthly.csv")

