# Copper Data Observatory — Forecasting Copper Prices with VAR

An econometric study and data pipeline to model global copper prices using macroeconomic indicators from FRED, evaluate time-series properties, and test the predictive power of Vector Autoregression (VAR) models.

## Overview

This project investigates whether global copper prices can be forecasted using macroeconomic and market fundamentals. It implements a full econometric workflow: data extraction via FRED API, dual stationarity testing, feature selection via Granger causality, post-estimation diagnostic iterative refining, and multi-horizon out-of-sample forecast evaluation.

---

## Data Sources

Macroeconomic and commodity indicators fetched via the **FRED API**:

| Series ID | Description |
|---|---|
| `PIORECRUSDM` | Global Price of Ore |
| `INDPRO` | US Industrial Production Index |
| `FEDFUNDS` | Federal Funds Effective Rate |
| `CPIAUCSL` | Consumer Price Index (Inflation) |
| `CSUSHPISA` | S&P/Case-Shiller US Home Price Index |
| `MANEMP` | Manufacturing Employment |
| `XTIMVA01CNM657S` | China Imported Commodities |
| `COCHNZ335` | China Electrical Equipment Production |
| `COCHNZ333` | China Machinery Manufacturing |
| *(External)* | LME Copper Warehouse Stocks |

---

## Econometric Methodology & Workflow

### 1. Pre-processing & Stationarity
- Applied **first-difference of natural logs** ($\Delta \ln X_t$) to obtain growth rates/returns.
- Dual testing framework:
  - **ADF Test**: $H_0 =$ Unit root (non-stationary)
  - **KPSS Test**: $H_0 =$ Stationarity
  - Both confirmed integration order $I(1)$ for levels and $I(0)$ for differenced series.

### 2. Feature Selection
- Evaluated Pearson correlation, cross-correlation at various lags, and **pairwise Granger causality tests**.
- Selected 4 primary candidate drivers:
  1. `PIORECRUSDM` (Global Ore Price)
  2. `INDPRO` (Industrial Production)
  3. LME Copper Warehouse Stock
  4. Global Copper Spot/Benchmark Price

### 3. Iterative VAR Specification & Diagnostics
- **Specification 1 (4 Variables):**
  - Residual diagnostics failed: significant **autocorrelation** was detected in the residuals.
  - ARCH test revealed severe **conditional heteroskedasticity and structural breaks in `INDPRO`** around **2019 and 2021** (trade tensions & COVID-19 macroeconomic shocks).
- **Specification 2 (Refitted 3-Variable VAR):**
  - Excluded `INDPRO` to avoid contamination from unmodeled structural breaks.
  - Passed all diagnostic checks: **eigenvalue stability** condition (roots inside the unit circle) and **Portmanteau / LM tests for no residual autocorrelation**.

### 4. Cointegration Check
- Conducted the **Johansen Cointegration Test** (Trace and Maximum Eigenvalue statistics).
- Confirmed **no cointegrating vectors** at standard significance levels; thus, fitting an unconstrained VAR in first differences was econometrically valid (no VECM needed).

### 5. Forecasting & Findings
- **12-Month Out-of-Sample Evaluation:** Point forecasts deteriorated rapidly over the annual horizon.
- **Rolling-Window Forecast Analysis:** The model exhibits acceptable predictive power only in the **very short run ($h=1$ month)**.
- **Economic Takeaway:** Copper returns are heavily driven by unanticipated structural shocks (supply disruptions, policy shifts) that simple linear VAR models cannot propagate over medium-to-long horizons.

---

## Setup & Execution
```bash
# Clone the repository
git clone https://github.com/USERNAME/copper-data-observatory.git
cd copper-data-observatory

# Install dependencies
pip install -r requirements.txt

# Configure FRED API Key
export FRED_API_KEY="your_api_key_here"
