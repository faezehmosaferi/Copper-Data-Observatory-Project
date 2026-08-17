import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.iolib.smpickle import load_pickle

results = load_pickle('var_results.pkl')
data = pd.read_pickle('var_data.pkl')
df = pd.read_pickle('df_original.pkl')

# --------------------------------------------------
# Settings
# --------------------------------------------------
price_level_col = 'Global_price_of_copper'
price_dlog_col = 'dlog_Global_price_of_copper'
p = results.k_ar   # same lag order as your selected VAR model

# Sort data
data = data.sort_index()
df = df.sort_index()

# Align actual level price with differenced VAR data
price_level = df[price_level_col].reindex(data.index)

if price_level.isna().any():
    raise ValueError(
        "Some dates in data were not found in df for copper price. "
        "Please check the date index alignment."
    )

# --------------------------------------------------
# Rolling one-step-ahead forecast
# --------------------------------------------------
forecast_dates = []
forecast_prices = []
actual_prices = []

# We start after enough observations for estimation
start_forecast = max(24, p + 1)  
# You can increase 24 if you want a longer initial training window

for i in range(start_forecast, len(data)):
    train_data = data.iloc[:i].copy()
    test_date = data.index[i]

    model = VAR(train_data)
    res = model.fit(maxlags=p, trend='c')

    # 1-step-ahead forecast in dlog space
    fc = res.forecast(
        y=train_data.values[-res.k_ar:],
        steps=1
    )

    idx_price = train_data.columns.get_loc(price_dlog_col)
    dlog_fc = fc[0, idx_price]

    # Last actual observed price before forecast date
    P0 = price_level.iloc[i - 1]

    # Convert forecast from dlog to level
    price_fc = P0 * np.exp(dlog_fc)

    forecast_dates.append(test_date)
    forecast_prices.append(price_fc)
    actual_prices.append(price_level.loc[test_date])

# Build dataframe
rolling_eval_df = pd.DataFrame({
    'Actual_Price': actual_prices,
    'Forecast_Price': forecast_prices
}, index=forecast_dates)

rolling_eval_df['Error'] = (
    rolling_eval_df['Actual_Price'] - rolling_eval_df['Forecast_Price']
)

rolling_eval_df['Absolute_Error'] = rolling_eval_df['Error'].abs()

rolling_eval_df['APE_percent'] = (
    rolling_eval_df['Absolute_Error'] / rolling_eval_df['Actual_Price']
) * 100

# --------------------------------------------------
# Metrics
# --------------------------------------------------
rmse = np.sqrt(np.mean(rolling_eval_df['Error'] ** 2))
mae = np.mean(rolling_eval_df['Absolute_Error'])
mape = np.mean(rolling_eval_df['APE_percent'])

print("=" * 45)
print("Rolling one-step forecast evaluation")
print("=" * 45)
print(f"RMSE : {rmse:,.2f}")
print(f"MAE  : {mae:,.2f}")
print(f"MAPE : {mape:.2f}%")

# --------------------------------------------------
# Plot
# --------------------------------------------------
plt.figure(figsize=(14, 7))

plt.plot(
    price_level.index,
    price_level.values,
    label='Actual copper price',
    color='steelblue',
    linewidth=2
)

plt.plot(
    rolling_eval_df.index,
    rolling_eval_df['Forecast_Price'],
    label='Rolling 1-step VAR forecast',
    color='crimson',
    linestyle='--',
    linewidth=1.8
)

plt.title('Copper Price: Actual vs Rolling One-Step VAR Forecast (Full Period)')
plt.xlabel('Date')
plt.ylabel('Copper price')
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig('actual_vs_rolling_forecast', dpi=300, bbox_inches='tight')

