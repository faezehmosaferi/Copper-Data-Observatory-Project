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

# Sort by date
data = data.sort_index()
df = df.sort_index()

# Align actual price level with VAR data index
price_level = df[price_level_col].reindex(data.index)

if price_level.isna().any():
    raise ValueError(
        "Some dates in data were not found in df for copper price. "
        "Please check the date index alignment."
    )

# --------------------------------------------------
# Fit VAR on full sample
# --------------------------------------------------
p = results.k_ar   # or set manually, e.g. p = 1

var_model = VAR(data)
results_full = var_model.fit(maxlags=p, trend='c')

print("Selected lag:", results_full.k_ar)
print("Model stable?:", results_full.is_stable())

# --------------------------------------------------
# Get fitted values (in dlog space)
# --------------------------------------------------
fitted_dlog = results_full.fittedvalues.copy()

# Keep only fitted dlog copper price
fitted_dlog_price = fitted_dlog[price_dlog_col]

# --------------------------------------------------
# Convert fitted dlog back to price level
# --------------------------------------------------
# Starting actual price just before first fitted observation
start_idx = fitted_dlog_price.index[0]
start_pos = price_level.index.get_loc(start_idx)

if start_pos == 0:
    raise ValueError("Not enough observations to reconstruct fitted price series.")

P0 = price_level.iloc[start_pos - 1]

cum_log_diff_fitted = np.cumsum(fitted_dlog_price.values)
fitted_price_level = P0 * np.exp(cum_log_diff_fitted)

fitted_price_series = pd.Series(
    fitted_price_level,
    index=fitted_dlog_price.index,
    name='Fitted_Price'
)

# Actual price for the same fitted period
actual_price_fitted_period = price_level.loc[fitted_price_series.index]

# --------------------------------------------------
# Plot full sample actual vs fitted
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
    fitted_price_series.index,
    fitted_price_series.values,
    label='VAR fitted price (in-sample)',
    color='crimson',
    linestyle='--',
    linewidth=2
)

plt.title('Copper Price: Actual vs VAR Fitted Values (Full Sample)')
plt.xlabel('Date')
plt.ylabel('Copper price')
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig('actual_vs_fitted.png', dpi=300, bbox_inches='tight')

