import pandas as pd
import numpy as np
from extract import fetch_fred_series, fetch_fred
from statsmodels.tsa.stattools import adfuller, kpss
import os
from dotenv import load_dotenv
load_dotenv()
import matplotlib.pyplot as plt
import seaborn as sns

LEVEL_VARS = ['PCOPPUSDM', 'FEDFUNDS','PIORECRUSDM', 'CPIAUCSL', 'CSUSHPISA', 'MANEMP']

def run_tests(series, name):
    adf_stat, adf_p, adf_lags, _, adf_crit, _ = adfuller(series.dropna(), autolag='AIC')
    kpss_stat, kpss_p, _, kpss_crit = kpss(series.dropna(), regression='c', nlags='auto')
#    kpss_stat, kpss_p, _, kpss_crit = kpss(series.dropna(), regression='ct', nlags='auto')
    stationary = (adf_p < 0.05) and (kpss_p > 0.05)
    print(f"\n{name}:")
    print(f"  ADF  p={adf_p:.4f} {'✅' if adf_p < 0.05 else '❌'}")
    print(f"  KPSS p={kpss_p:.4f} {'✅' if kpss_p > 0.05 else '❌'}")
    print(f"  → {'stationary' if stationary else 'non-stationary'}")
    return stationary

FRED_SERIES = {
    "FEDFUNDS": "Federal Funds Rate",
    "PIORECRUSDM":"Global price of ore",
    "CPIAUCSL": "CPI Inflation",
    "INDPRO": "Industrial Production",
    "CSUSHPISA": "Housing Price Index",
    "MANEMP": "Manufacturing Employment",
    "XTIMVA01CNM657S" : "Imported commodities for china",
}
records = []
for series_id in FRED_SERIES.keys():
     records.extend(fetch_fred_series(series_id))

records.extend(fetch_fred())

df_long = pd.DataFrame(records, columns=["date", "series_id", "value", "source"])
dups = df_long[df_long.duplicated(subset=["date", "series_id"], keep=False)]
print(dups.sort_values(["series_id", "date"]))


df_wide = df_long.pivot(index="date", columns="series_id", values="value")
transformed = {}
for var in LEVEL_VARS:
#for var in ['CSUSHPISA', 'MANEMP']:
          transformed[var] = np.log(df_wide[var]).diff()
#       transformed[var] = np.log(df_long[var]).diff()
transformed['CSUSHPISA'] = transformed['CSUSHPISA'].diff()

transformed['XTIMVA01CNM657S'] = df_wide['XTIMVA01CNM657S']  
transformed['FEDFUNDS'] = df_wide['FEDFUNDS'].diff()
#transformed['XTIMVA01CNM657S'] = df_long['XTIMVA01CNM657S']
#transformed['FEDFUNDS'] = df_long['FEDFUNDS'].diff()

df_transformed = pd.DataFrame(transformed).dropna()
#print(len(df_transformed))
df_lme = pd.read_excel("copper-lme-stock.xlsx", sheet_name="Sheet1")
df_lme.columns = ['date', 'LME-copper']
print(df_lme.head())
df_lme = df_lme.set_index('date')

df_transformed.index = pd.to_datetime(df_transformed.index).to_period('M').to_timestamp()
df_lme.index = pd.to_datetime(df_lme.index).to_period('M').to_timestamp()
#df_transformed['date'] = pd.to_datetime(df_transformed.index).to_period('M').to_timestamp()
#df_lme['date'] = pd.to_datetime(df_lme['date']).dt.to_period('M').dt.to_timestamp()
df_merged = df_transformed.join(df_lme[['LME-copper']], how = 'left')
#df_merged = pd.merge(df_transformed, dflme, on='date', how='inner')
for col in df_merged.columns:
    run_tests(df_merged[col], col)

def plot_correlation_matrix(df_merged: pd.DataFrame):
    corr = df_merged.corr()
    mask = pd.DataFrame(False, index=corr.index, columns=corr.columns)
    # flag pairs with |r| > 0.9
    high_corr = [(i, j) for i in corr.columns for j in corr.columns
                 if i < j and abs(corr.loc[i, j]) > 0.9]

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Matrix (Transformed Variables)")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png", dpi=150)
    plt.show()

    if high_corr:
        print("⚠ High correlation pairs (|r| > 0.9):")
        for i, j in high_corr:
            print(f"  {i} — {j}: {corr.loc[i,j]:.3f}")
    else:
        print("✓ No pairs with |r| > 0.9")

    return corr

from statsmodels.tsa.api import VAR

#model = VAR(df_transformed)
#lag_selection = model.select_order(maxlags=12)
#print(lag_selection.summary())
#
#optimal_lag_bic = lag_selection.bic
#optimal_lag_aic = lag_selection.aic
#
#
#results_p1 = model.fit(1)
#results_p2 = model.fit(2)
#
#lr_stat = 2 * (results_p2.llf - results_p1.llf)
#df_diff = results_p2.df_model - results_p1.df_model
#from scipy.stats import chi2
#p_value = 1 - chi2.cdf(lr_stat, df_diff)
#print(f"LR stat: {lr_stat:.2f}, p-value: {p_value:.4f}")


