import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
load_dotenv()
import numpy as np
from statsmodels.tsa.api import VAR

# --- Configuration ---
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "copper_db"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}

conn = psycopg2.connect(**DB_CONFIG)

# 1. Load macro indicators data
indicators = pd.read_sql("""
    SELECT date, series_id, value
    FROM macro_indicators
    ORDER BY date
""", conn, parse_dates=['date'])

indicators_pivot = indicators.pivot_table(
    index='date', columns='series_id', values='value'
)

# Rename columns for readability
SERIES_NAMES ={

    "PCOPPUSDM" : "Global_price_of_copper",
    "FEDFUNDS": "Federal_Funds_Rate",
    "PIORECRUSDM":"Global_price_of_ore",
    "CPIAUCSL": "CPI_Inflation",
    "INDPRO": "Industrial_Production",
    "CSUSHPISA": "Housing_Price_Index",
    "MANEMP": "Manufacturing_Employment",
    "XTIMVA01CNM657S" : "Imported_commodities_for_china",
    "COCHNZ335" : "Electrical_equipment_china",
    "COCHNZ333" : "Machinary_manufacturing_china",
#    "CCRETT01CNM661N" : "REER_china",
    "LME-COPPER-STOCK" : "lme-copper-stock",
}
indicators_pivot.rename(columns=SERIES_NAMES, inplace=True)

Series = SERIES_NAMES.copy()
Series.pop('FEDFUNDS')
Series_1 = list(Series.values())
    
# log returns / monthly growth
for col in Series_1:
    indicators_pivot[f'dlog_{col}'] = np.log(indicators_pivot[col]).diff()

indicators_pivot['dlog_Federal_Funds_Rate'] = indicators_pivot['Federal_Funds_Rate'].diff()
print("index:", pd.infer_freq(indicators_pivot.index))

var_columns = [
    "dlog_Global_price_of_copper",
    "dlog_Global_price_of_ore",
    "dlog_lme-copper-stock",
#    "dlog_Industrial_Production",
#    "dlog_Federal_Funds_Rate",        # علیت قوی در سیستم
#    "dlog_Manufacturing_Employment",  # کانال اشتغال
]

var_data = indicators_pivot[var_columns].dropna()

var_model = VAR(var_data)

lag_order = var_model.select_order(maxlags=6)

print(lag_order.summary())

selected_lag = lag_order.selected_orders["bic"]
if selected_lag == 0:
    selected_lag = 1

var_result = var_model.fit(4, trend="c")

var_result.save('var_results.pkl')
var_data.to_pickle('var_data.pkl')
indicators_pivot.to_pickle('df_original.pkl')

print(f"Selected lag by BIC: {selected_lag}")
print(var_result.summary())

print(var_result.test_whiteness(nlags=10).summary())

print("Stable VAR:", var_result.is_stable(verbose=True))

import matplotlib.pyplot as plt

resid = var_result.resid

fig, axes = plt.subplots(4, 1, figsize=(12, 10))
for i, col in enumerate(var_data.columns):
    axes[i].plot(resid.index, resid.iloc[:, i])
    axes[i].set_title(f"Residuals: {col}")
    axes[i].axhline(0, color='r', linestyle='--')
plt.tight_layout()

plt.savefig("residuals.png", dpi=300, bbox_inches="tight")

plt.close(fig)  

start_date = '2012-01-01'
end_date = '2019-12-01'
var_data.index = pd.to_datetime(var_data.index)
subset_data = var_data.loc[start_date:end_date, var_columns]

print(len(subset_data))

model_sub = VAR(subset_data)
var_sub_result = model_sub.fit(2, trend="c")
print(var_sub_result.summary())

print(var_sub_result.test_whiteness(nlags=10).summary())

#covid_dummy = pd.DataFrame(0, index=var_data.index, columns=['covid_shock'])

#covid_dummy.loc['2020-03-01':'2020-05-01', 'covid_shock'] = 1

#var_model = VAR(var_data, exog=covid_dummy)
#var_dummy_result = var_model.fit(maxlags=2, trend='c')

#print(var_dummy_result.summary())

#print(var_dummy_result.test_whiteness(nlags=10).summary())

shock_dummies = pd.DataFrame(0, index=var_data.index, columns=['covid_shock', 'supply_chain_2021'])

shock_dummies.loc['2020-03':'2020-05', 'covid_shock'] = 1

shock_dummies.loc['2021-07':'2021-12', 'supply_chain_2021'] = 1

var_model_full = VAR(var_data, exog=shock_dummies)
var_result_full = var_model_full.fit(maxlags=2, trend='c')

print(var_result_full.summary())

print(var_result_full.test_whiteness(nlags=10).summary())

from statsmodels.stats.diagnostic import het_arch

resid_full = var_result_full.resid

arch_test = het_arch(resid_full.iloc[:, 0], nlags=12)
print(f"LM statistic: {arch_test[0]:.3f}")
print(f"p-value: {arch_test[1]:.4f}")

from statsmodels.stats.diagnostic import het_arch

resid_full = var_result_full.resid

for col in resid_full.columns:
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(resid_full[col], nlags=12)
    print(f"{col}: LM-p = {lm_pvalue:.4f} | F-p = {f_pvalue:.4f}")


