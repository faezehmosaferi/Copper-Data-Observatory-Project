import numpy as np
from statsmodels.tsa.vector_ar.var_model import VARResults
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.iolib.smpickle import load_pickle

results = load_pickle('var_results.pkl')
data = pd.read_pickle('var_data.pkl')
df = pd.read_pickle('df_original.pkl')

idx_price = data.columns.get_loc('dlog_Global_price_of_copper')

n_ahead = 1
forecast = results.forecast(y=data.values[-results.k_ar:], steps=n_ahead)
log_diff_forecast = forecast[:, idx_price]

cum_log_diff = np.cumsum(log_diff_forecast)

P0 = df['Global_price_of_copper'].iloc[-1]
price_path = P0 * np.exp(cum_log_diff)


last_date = df.index[-1]
freq = pd.infer_freq(df.index)
forecast_index = pd.date_range(start=last_date, periods=n_ahead+1, freq=freq)[1:]


plot_forecast = np.concatenate([[P0], price_path])
plot_index = pd.date_range(start=last_date, periods=n_ahead+1, freq=freq)
plot_index = pd.to_datetime(plot_index)
historical_index = pd.to_datetime(df.index[-60:])

plt.figure(figsize=(10, 6))
plt.plot(historical_index, df['Global_price_of_copper'].iloc[-60:], label='Historical Price', color='blue')
plt.plot(plot_index, plot_forecast, label='Forecast', color='red', linestyle='--')

plt.axvline(x=pd.Timestamp(last_date), color='gray', linestyle=':', alpha=0.7)
plt.title('Copper Price: Historical vs VAR Forecast for one month')
plt.xlabel('Date')
plt.ylabel('USD/ton')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('price_copper_forecast.png')
plt.show()
