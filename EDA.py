import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
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

conn = psycopg2.connect(**DB_CONFIG)

# 1. Load copper price data
prices = pd.read_sql("""
    SELECT recorded_at AS date, price_usd AS price, source
    FROM copper_prices
    ORDER BY recorded_at
""", conn, parse_dates=['date'])

# 2. Load economic indicators (pivot to convert into separate columns)
indicators = pd.read_sql("""
    SELECT date, series_id, value 
    FROM macro_indicators 
    WHERE series_id IN ('FEDFUNDS','CPIAUCSL','INDPRO','CSUSHPISA','MANEMP')
    ORDER BY date
""", conn, parse_dates=['date'])

indicators_pivot = indicators.pivot_table(
    index='date', columns='series_id', values='value'
).reset_index()

# Rename columns for readability
SERIES_NAMES = {
    "FEDFUNDS": "fed_rate",
    "CPIAUCSL": "cpi",
    "INDPRO": "industrial_prod",
    "CSUSHPISA": "housing_idx",
    "MANEMP": "mfg_employment",
}
indicators_pivot.rename(columns=SERIES_NAMES, inplace=True)

# 3. Load news sentiment (daily average)
sentiment = pd.read_sql("""
    SELECT DATE(published_at) AS date, 
           AVG(sentiment_score) AS avg_sentiment,
           COUNT(*) AS news_count
    FROM news_articles
    WHERE sentiment_score IS NOT NULL
    GROUP BY DATE(published_at)
    ORDER BY date
""", conn, parse_dates=['date'])

conn.close()

# 4. Merge all datasets
df = prices.merge(indicators_pivot, on='date', how='left')
df = df.merge(sentiment, on='date', how='left')

# Fill missing values (forward-fill for monthly indicators)
econ_cols = list(SERIES_NAMES.values())
df[econ_cols] = df[econ_cols].ffill()
df['avg_sentiment'] = df['avg_sentiment'].fillna(0)

df.set_index('date', inplace=True)

# 5. Lagged Correlation Analysis
lags = range(1, 31)
lag_corrs = {}

for col in econ_cols + ['avg_sentiment']:
    lag_corrs[col] = [df['price'].corr(df[col].shift(lag)) for lag in lags]

lag_df = pd.DataFrame(lag_corrs, index=lags)

# Plot lag correlation chart
plt.figure(figsize=(14, 6))
lag_df.plot(ax=plt.gca(), marker='o', markersize=3)
plt.axhline(0, color='k', lw=0.5, linestyle='--')
plt.xlabel('Lag (days)')
plt.ylabel('Correlation with Copper Price')
plt.title('Lagged Correlation: Economic Indicators & Sentiment vs Copper Price')
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('lagged_correlation1.png', dpi=150)
plt.show()

# Monthly Correlation
df_monthly = df.resample('ME').last()
lag_corrs_monthly = {col: [df_monthly['price'].corr(df_monthly[col].shift(lag)) 
                           for lag in range(1, 13)] 
                     for col in econ_cols + ['avg_sentiment']}
pd.DataFrame(lag_corrs_monthly, index=range(1,13)).plot(figsize=(12,5), title='Monthly Lagged Correlation')
plt.xlabel('Lag (months)')
plt.savefig('lagged_correlation2.png', dpi=150)
plt.show()



# 6. Price Regime Analysis (Bullish/Bearish)
df['price_regime'] = df['price'].pct_change(5).apply(
    lambda x: 'Bullish' if x > 0.02 else ('Bearish' if x < -0.02 else 'Neutral')
)

regime_stats = df.groupby('price_regime')[econ_cols + ['avg_sentiment']].mean()
print("\n=== Average Indicator Values Across Price Regimes ===")
print(regime_stats)

# 7. Correlation Heatmap
corr_matrix = df[['price'] + econ_cols + ['avg_sentiment']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix: Copper Price, Indicators & Sentiment')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()

print("\n✅ EDA analysis complete. PNG files saved.")


