import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.iolib.smpickle import load_pickle

results = load_pickle('var_results.pkl')
data = pd.read_pickle('var_data.pkl')
df = pd.read_pickle('df_original.pkl')

# --------------------------------------------------
#  Settings
# --------------------------------------------------
price_level_col = 'Global_price_of_copper'
price_dlog_col = 'dlog_Global_price_of_copper'

n_test = 12   

data = data.sort_index()
df = df.sort_index()

# --------------------------------------------------
#  1) Align price levels with the VAR data index
# --------------------------------------------------
# Since data becomes shorter than df after differencing,
# we align the price series using the same dates as data.
price_level = df['Global_price_of_copper'].reindex(data.index)

# Check for missing values
if price_level.isna().any():
    raise ValueError(
        "Some dates in data were not found in df for copper price. "
        "Please check the date index of both dataframes."
    )

# --------------------------------------------------
#  2) Train/Test split
# --------------------------------------------------
train_data = data.iloc[:-n_test].copy()
test_data = data.iloc[-n_test:].copy()

# Actual copper price during the test period
actual_price = price_level.iloc[-n_test:].copy()

print("Train period:", train_data.index.min(), "to", train_data.index.max())
print("Test period: ", test_data.index.min(), "to", test_data.index.max())

# --------------------------------------------------
#  3) Refit VAR only on the training data
# --------------------------------------------------
# Use the same lag length as the previous model.
# If your previous results were VAR(1), then p = 1.
p = results.k_ar

var_train = VAR(train_data)
results_train = var_train.fit(maxlags=p, trend='c')

print("\nSelected lag:", results_train.k_ar)
print("Model stable?:", results_train.is_stable())

# --------------------------------------------------
# 4) Forecast dlog values for the test period
# --------------------------------------------------
forecast_dlog_all = results_train.forecast(
    y=train_data.values[-results_train.k_ar:],
    steps=n_test
)

# Find the column position of the copper price dlog series
idx_price = train_data.columns.get_loc('dlog_Global_price_of_copper')

# Keep only the forecasted dlog copper price
dlog_price_forecast = forecast_dlog_all[:, idx_price]

# --------------------------------------------------
#  5) Convert dlog forecast to price level
# --------------------------------------------------
# Last observed actual price before the test period starts
P0 = price_level.iloc[-n_test - 1]

# Cumulative sum of predicted log differences
cum_log_diff = np.cumsum(dlog_price_forecast)

# Correct conversion from dlog to price level
forecast_price = P0 * np.exp(cum_log_diff)

# Build evaluation table
evaluation_df = pd.DataFrame({
    'Actual_Price': actual_price.values,
    'Forecast_Price': forecast_price,
    'Error': actual_price.values - forecast_price,
    'Absolute_Error': np.abs(actual_price.values - forecast_price),
    'APE_percent': (
        np.abs(actual_price.values - forecast_price)
        / actual_price.values
    ) * 100
}, index=test_data.index)

print("\nEvaluation table:")
print(evaluation_df.round(2))

# --------------------------------------------------
# 6) Evaluation metrics
# --------------------------------------------------
rmse = np.sqrt(
    np.mean((evaluation_df['Actual_Price'] -
             evaluation_df['Forecast_Price']) ** 2)
)

mae = np.mean(evaluation_df['Absolute_Error'])

mape = np.mean(evaluation_df['APE_percent'])

print("\n" + "=" * 45)
print("Forecast evaluation metrics")
print("=" * 45)
print(f"RMSE : {rmse:,.2f}")
print(f"MAE  : {mae:,.2f}")
print(f"MAPE : {mape:.2f}%")


plt.figure(figsize=(12, 6))

# Last training observation date, used to connect forecast to the actual series
train_last_date = train_data.index[-1]

# Last actual train price + forecasted prices
forecast_plot_dates = pd.DatetimeIndex(
    [train_last_date]
).append(test_data.index)

forecast_plot_values = np.concatenate(
    [[P0], forecast_price]
)

# Actual series: a few points before the split until the end of the test period
history_for_plot = price_level.iloc[-n_test - 13:]

plt.plot(
    history_for_plot.index,
    history_for_plot.values,
    label='Actual copper price',
    color='steelblue',
    linewidth=2
)

plt.plot(
    forecast_plot_dates,
    forecast_plot_values,
    label='VAR forecast',
    color='crimson',
    linestyle='--',
    marker='o',
    linewidth=2
)

# Vertical line marking the start of the test period
plt.axvline(
    x=train_last_date,
    color='gray',
    linestyle=':',
    linewidth=1.5,
    label='Start of test period'
)

plt.title('Copper Price: Actual vs VAR Forecast')
plt.xlabel('Date')
plt.ylabel('Copper price')
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig('var_backtest_actual_vs_forecast.png', dpi=300, bbox_inches='tight')


