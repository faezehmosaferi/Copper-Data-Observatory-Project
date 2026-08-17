import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
load_dotenv()
import numpy as np

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

for col in SERIES_NAMES.values():
    s = indicators_pivot[col].dropna()
    print("1", col, "| n =", len(s), "|", s.index.min(), "→", s.index.max())

indicators0 = indicators_pivot.dropna()
print("len-0 : ", len(indicators0))
#SERIES_NAMES.pop("CCRETT01CNM661N")
# log returns / monthly growth
for col in Series_1:
    indicators_pivot[f'dlog_{col}'] = 100 * np.log(indicators_pivot[col]).diff()
    s = indicators_pivot[col].dropna()
    print("2", col, "| n =", len(s), "|", s.index.min(), "→", s.index.max())

indicators_pivot['dlog_Federal_Funds_Rate'] = indicators_pivot['Federal_Funds_Rate'].diff()


Series_dlog= {}
for key in SERIES_NAMES:
    Series_dlog[key] = "dlog_" + SERIES_NAMES[key]

dlog_serie_plus = []
dlog_serie_plus = list(Series_dlog.values())

Series_dlog.pop("PCOPPUSDM")

dlog_serie = []
dlog_serie = list(Series_dlog.values())

corr_with_copper = (
    indicators_pivot[["dlog_Global_price_of_copper"] + dlog_serie]
    .corr()
    .loc[Series_dlog.values(), 'dlog_Global_price_of_copper']
    .sort_values(ascending=False)
)

print("Correlation with copper returns:")
print(corr_with_copper)


#def cross_corr_table(df, x, y, max_lag=12):
#    rows = []
#    for lag in range(-max_lag, max_lag + 1):
#        if lag > 0:
#            corr = df[x].shift(lag).corr(df[y])
#        else:
#            corr = df[x].corr(df[y].shift(-lag))
#        rows.append({'lag': lag, 'corr': corr})
#    return pd.DataFrame(rows)
#
#
#results = {}
#for s in dlog_serie:
#    tbl = cross_corr_table(indicators_pivot.dropna(), s, 'dlog_Global_price_of_copper', max_lag=12)
#    results[s] = tbl
#    best = tbl.iloc[tbl['corr'].abs().argmax()]
#    print(f"{s}: best lag={best['lag']}, corr={best['corr']:.3f}")
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

def cross_corr_table(df, x, y, max_lag=12, min_obs=12):
    rows = []

    for lag in range(-max_lag, max_lag + 1):

        if lag > 0:
            x_shifted = df[x].shift(lag)
            y_aligned = df[y]
        elif lag < 0:
            x_shifted = df[x]
            y_aligned = df[y].shift(-lag)
        else:
            x_shifted = df[x]
            y_aligned = df[y]

        tmp = pd.concat([x_shifted, y_aligned], axis=1).dropna()
        tmp.columns = [x, y]

        n_obs = len(tmp)

        if n_obs >= 3:
            corr, p_value = pearsonr(tmp[x], tmp[y])
        else:
            corr, p_value = np.nan, np.nan

        rows.append({
            "lag": lag,
            "corr": corr,
            "p_value": p_value,
            "n_obs": n_obs
        })

    out = pd.DataFrame(rows)

    valid = out[(out["n_obs"] >= min_obs) & (out["corr"].notna())].copy()

    if not valid.empty:
        best_idx = valid["corr"].abs().idxmax()
        best_valid = valid.loc[best_idx].to_dict()
    else:
        best_valid = None

    return out, valid, best_valid

results = {}
best_lags = []

for s in dlog_serie:
    indicators_pivot[[s, 'dlog_Global_price_of_copper']].dropna()
    tbl_all, tbl_valid, best = cross_corr_table(
        indicators_pivot,
        x=s,
        y='dlog_Global_price_of_copper',
        max_lag=12,
        min_obs=12
    )

    results[s] = {
        "all_lags": tbl_all,
        "valid_lags": tbl_valid,
        "best": best
    }

    if best is not None:
        best_lags.append({
            "series": s,
            "best_lag": best["lag"],
            "corr": best["corr"],
            "p_value": best["p_value"],
            "n_obs": best["n_obs"]
        })

        print(
            f"{s}: best lag={int(best['lag'])}, "
            f"corr={best['corr']:.3f}, "
            f"p={best['p_value']:.4f}, "
            f"n={int(best['n_obs'])}"
        )
    else:
        print(f"{s}: no valid lag found")

best_lags_df = pd.DataFrame(best_lags).sort_values(
    by="corr",
    key=lambda s: s.abs(),
    ascending=False
)

print(best_lags_df)

from itertools import permutations
from statsmodels.tsa.stattools import grangercausalitytests

rows = []
for cause, effect in permutations(dlog_serie_plus, 2):
    pair = indicators_pivot[[effect, cause]].dropna()
    res = grangercausalitytests(pair, maxlag=3, verbose=False)
    for lag in range(1, 3):
        f_stat = res[lag][0]["ssr_ftest"][0]
        p_val = res[lag][0]["ssr_ftest"][1]
        rows.append({"cause": cause, "effect": effect,
                     "lag": lag, "f_stat": f_stat, "p_value": p_val})

df_granger = pd.DataFrame(rows)

pd.set_option("display.max_rows", None) 
pd.set_option("display.max_columns", None)  
pd.set_option("display.width", 200)     

print(df_granger.sort_values(['p_value', 'lag']))
print(df_granger[df_granger['cause'] == 'dlog_Global_price_of_copper'].sort_values(['p_value', 'lag']))
print(df_granger[df_granger['effect'] == 'dlog_Global_price_of_copper'].sort_values(['p_value', 'lag']))
import matplotlib.pyplot as plt

def df_to_image(df, filename="df_granger.png", title="Granger Causality Results"):
    fig, ax = plt.subplots(figsize=(14, 0.5 * len(df) + 1.5))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,          
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
        colColours=["#2c3e50"] * len(df.columns),
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:  # هدر
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f2f2f2")

    ax.set_title(title, fontsize=13, weight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved: {filename}")

df_to_image(df_granger)


#from statsmodels.tsa.stattools import grangercausalitytests
#
#def granger_summary(df, cause, effect, maxlag=3):
#    data = df[[effect, cause]].dropna()
#    res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
#
#    rows = []
#    for lag in range(1, maxlag + 1):
#        pval = res[lag][0]['ssr_ftest'][1]
#        rows.append({'lag': lag, 'pvalue': pval})
#    out = pd.DataFrame(rows)
#    out['cause'] = cause
#    out['effect'] = effect
#    return out[['cause', 'effect', 'lag', 'pvalue']]
#
#all_granger_copper = []
#for s in dlog_serie:
#    out = granger_summary(indicators_pivot, s, 'dlog_Global_price_of_copper', maxlag=3)
#    all_granger_copper.append(out)
#
#granger_copper_df = pd.concat(all_granger_copper, ignore_index=True)
#print(granger_copper_df.sort_values(['pvalue', 'lag']))

from statsmodels.stats.multitest import multipletests

df = df_granger.copy()

reject_fdr_all, p_adj_fdr_all, _, _ = multipletests(
    df["p_value"].values, alpha=0.05, method="fdr_bh"
)

reject_bonf_all, p_adj_bonf_all, _, _ = multipletests(
    df["p_value"].values, alpha=0.05, method="bonferroni"
)

df["pvalue_adj_fdr_all"] = p_adj_fdr_all
df["significant_fdr_all"] = reject_fdr_all

df["pvalue_adj_bonf_all"] = p_adj_bonf_all
df["significant_bonf_all"] = reject_bonf_all

print("=== Significant after FDR on ALL tests ===")
print(
    df[df["significant_fdr_all"]]
    .sort_values("pvalue_adj_fdr_all")[["cause", "effect", "lag", "p_value", "pvalue_adj_fdr_all"]]
)

print("\n=== Significant after Bonferroni on ALL tests ===")
print(
    df[df["significant_bonf_all"]]
    .sort_values("pvalue_adj_bonf_all")[["cause", "effect", "lag", "p_value", "pvalue_adj_bonf_all"]]
)

