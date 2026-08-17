import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.iolib.smpickle import load_pickle

results = load_pickle('var_results.pkl')
data = pd.read_pickle('var_data.pkl')
df = pd.read_pickle('df_original.pkl')

# ==================================================
# 1) choosing level columns & generating log series
# ==================================================

level_cols = [
    'Global_price_of_copper',
    'Global_price_of_ore',
    'lme-copper-stock',
]

vecm_levels = df[level_cols].copy()

if (vecm_levels <= 0).any().any():
    invalid_cols = vecm_levels.columns[(vecm_levels <= 0).any()].tolist()
    raise ValueError(
        f"These columns contain zero or negative values and cannot be logged: {invalid_cols}"
    )

log_levels = np.log(vecm_levels).dropna()

print("Observations used in Johansen test:", len(log_levels))
print(log_levels.head())

# ==================================================
# 2) Johansen Cointegration Test
# ==================================================

var_lag_order = results.k_ar   # lag order from your selected VAR model
k_ar_diff = max(var_lag_order - 1, 0)

johansen_result = coint_johansen(
    endog=log_levels,
    det_order=0,        # constant in cointegration relation, no deterministic trend
    k_ar_diff=k_ar_diff
)

print(f"VAR lag order used as reference: {var_lag_order}")
print(f"VECM k_ar_diff: {k_ar_diff}")

# ==================================================
# 3) Display Johansen test results
# ==================================================

# Critical values at 90%, 95%, 99%
trace_table = pd.DataFrame({
    'H0: Cointegration rank ≤ r': range(len(level_cols)),
    'Trace Statistic': johansen_result.lr1,
    'Critical Value (90%)': johansen_result.cvt[:, 0],
    'Critical Value (95%)': johansen_result.cvt[:, 1],
    'Critical Value (99%)': johansen_result.cvt[:, 2],
    'Reject H0 at 5%?': johansen_result.lr1 > johansen_result.cvt[:, 1]
})

maxeig_table = pd.DataFrame({
    'H0: Cointegration rank ≤ r': range(len(level_cols)),
    'Max-Eigen Statistic': johansen_result.lr2,
    'Critical Value (90%)': johansen_result.cvm[:, 0],
    'Critical Value (95%)': johansen_result.cvm[:, 1],
    'Critical Value (99%)': johansen_result.cvm[:, 2],
    'Reject H0 at 5%?': johansen_result.lr2 > johansen_result.cvm[:, 1]
})

print("=" * 70)
print("Johansen Cointegration Test — Trace Statistic")
print("=" * 70)
print(trace_table.round(3))

print("=" * 70)
print("Johansen Cointegration Test — Maximum Eigenvalue Statistic")
print("=" * 70)
print(maxeig_table.round(3))

# ==================================================
# 4) Determine cointegration rank at 5% significance
# ==================================================

# Rank based on Trace statistic
trace_rank = sum(johansen_result.lr1 > johansen_result.cvt[:, 1])

# Rank based on Maximum Eigenvalue statistic
maxeig_rank = sum(johansen_result.lr2 > johansen_result.cvm[:, 1])

print("=" * 55)
print("Suggested cointegration rank at 5% significance")
print("=" * 55)
print(f"Trace-test rank       : r = {trace_rank}")
print(f"Max-eigenvalue rank   : r = {maxeig_rank}")

n_variables = len(level_cols)

if trace_rank == 0:
    print(
        "\nInterpretation (Trace test): "
        "No cointegration detected at 5%. "
        "Keep using VAR on log-differenced variables."
    )

elif 0 < trace_rank < n_variables:
    print(
        f"\nInterpretation (Trace test): "
        f"Cointegration detected with rank r = {trace_rank}. "
        "Estimating a VECM is justified."
    )

else:
    print(
        "\nInterpretation (Trace test): "
        f"Full rank (r = {trace_rank}) was detected. "
        "Review stationarity tests and deterministic-term specification."
    )

