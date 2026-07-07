"""Mixed-frequency dynamic factor model for 7 firms + US macro.

Series (66 total)
-----------------
- 7 monthly log stock prices                    -> National + firm factor
- 56 quarterly financial measures (8 x 7 firms) -> National + firm factor
- 3 macro series: y, i, inflation               -> National factor ONLY

Model
-----
Each series_t = (national factor) + (its firm's factor, if any) + idiosyncratic_t.
The national factor and each firm-specific factor follow an AR(2) process; the
idiosyncratic terms follow AR(1). Estimated with statsmodels' DynamicFactorMQ
(monthly base frequency; quarterly series enter via the standard 3-month
aggregation), which supports block/grouped factor loadings.

Identification note
-------------------
DynamicFactorMQ requires (#factors <= #monthly series). We have 8 factors
(1 national + 7 firm), so we need >= 8 monthly series. The 7 prices are monthly;
`i` and inflation are therefore taken at MONTHLY frequency (both are natively
monthly on FRED), giving 9 monthly series. `y` (real GDP per capita) stays
quarterly. Macro loads on the national factor only, exactly as specified.

Output
------
data/firm_specific_factors.csv : monthly smoothed firm-specific factors
    (columns: AAPL, MSFT, AMZN, NVDA, GOOGL, TSLA, META)

Run:  python mixed_frequency_dfm.py
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- config
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))
PANEL_CSV = os.path.join(DATA_DIR, "firm_financials_panel", "firm_financials_long.csv")
PRICES_CSV = os.path.join(DATA_DIR, "monthly_stock_prices.csv")
OUT_CSV = os.path.join(DATA_DIR, "firm_specific_factors.csv")

TICKERS = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "TSLA", "META"]
MEASURES = ["revenue", "cost_of_revenue", "gross_profit", "operating_income",
            "rd_expense", "net_income", "total_assets", "stockholders_equity"]
# Log these (strictly positive); leave income measures (can be <= 0) in levels.
POS_MEASURES = {"revenue", "cost_of_revenue", "gross_profit",
                "rd_expense", "total_assets", "stockholders_equity"}

SAMPLE_START = "2015-01"
SAMPLE_END = "2026-04"


# ------------------------------------------------------------------------- helpers
def fred_series(series_id: str) -> pd.Series:
    """Download a single FRED series (public CSV endpoint, no API key)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    s = pd.read_csv(url, parse_dates=["observation_date"], index_col="observation_date")[series_id]
    return pd.to_numeric(s, errors="coerce").rename(series_id)


def build_monthly() -> pd.DataFrame:
    """Monthly block: 7 log stock prices + monthly i + monthly inflation."""
    prices = pd.read_csv(PRICES_CSV, parse_dates=[0], index_col=0)
    logp = np.log(prices)
    logp.columns = [f"{t}_logprice" for t in logp.columns]

    tbill = fred_series("TB3MS").resample("MS").mean()               # % per year, monthly
    cpi = fred_series("CPILFESL").resample("MS").mean()              # core CPI index, monthly
    infl = 1200.0 * np.log(cpi).diff()                              # annualized m/m, % per year

    macro_m = pd.concat([tbill.rename("i"), infl.rename("inflation")], axis=1)

    monthly = pd.concat([logp, macro_m], axis=1).loc[SAMPLE_START:SAMPLE_END]
    monthly.index = monthly.index.to_period("M")
    return monthly


def build_quarterly() -> pd.DataFrame:
    """Quarterly block: 56 normalized financial measures + quarterly y."""
    panel = pd.read_csv(PANEL_CSV, parse_dates=["date"])

    # Normalize when appropriate: log positive-only measures, level otherwise.
    panel["mval"] = panel["value"]
    logmask = panel["measure"].isin(POS_MEASURES) & (panel["value"] > 0)
    panel.loc[logmask, "mval"] = np.log(panel.loc[logmask, "value"])

    # Fiscal-period-end date -> calendar quarter (one obs per firm/measure/quarter).
    panel["quarter"] = panel["date"].dt.to_period("Q")
    panel["series"] = panel["ticker"] + "_" + panel["measure"]
    wide = (panel.pivot_table(index="quarter", columns="series", values="mval", aggfunc="last")
            .reindex(columns=[f"{t}_{m}" for t in TICKERS for m in MEASURES]))

    # Quarterly y = log real GDP per capita.
    y = np.log(fred_series("A939RX0Q048SBEA")).rename("y")
    y.index = y.index.to_period("Q")
    y = y[~y.index.duplicated()]

    quarterly = wide.join(y, how="outer")
    quarterly = quarterly.loc[pd.Period(SAMPLE_START, "Q"):pd.Period(SAMPLE_END, "Q")]
    return quarterly


def build_factor_map() -> dict:
    """Block loadings: firm series -> [National, firm]; macro -> [National]."""
    factors = {}
    for t in TICKERS:
        factors[f"{t}_logprice"] = ["National", t]
        for m in MEASURES:
            factors[f"{t}_{m}"] = ["National", t]
    for macro in ["i", "inflation", "y"]:
        factors[macro] = ["National"]
    return factors


def main(maxiter: int = 1000):
    from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

    monthly = build_monthly()
    quarterly = build_quarterly()
    factors = build_factor_map()

    # Drop empty/too-sparse series (e.g. Amazon files no R&D tag -> all NaN) and
    # prune the factor map to match; the firm factor still loads on the rest.
    MIN_OBS = 4
    for block in (monthly, quarterly):
        drop = [c for c in block.columns if block[c].notna().sum() < MIN_OBS]
        if drop:
            print("dropping sparse series:", drop)
            block.drop(columns=drop, inplace=True)
    keep = set(monthly.columns) | set(quarterly.columns)
    factors = {k: v for k, v in factors.items() if k in keep}

    print(f"monthly block:   {monthly.shape[1]} series x {monthly.shape[0]} months")
    print(f"quarterly block: {quarterly.shape[1]} series x {quarterly.shape[0]} quarters")
    print(f"factors: 1 National + {len(TICKERS)} firm = {1 + len(TICKERS)} "
          f"(<= {monthly.shape[1]} monthly series, OK)")

    mod = DynamicFactorMQ(
        endog=monthly,
        endog_quarterly=quarterly,
        factors=factors,          # block structure (national + firm-specific)
        factor_orders=2,          # AR(2) for every factor block
        idiosyncratic_ar1=True,   # AR(1) idiosyncratic terms
        standardize=True,         # z-score each series internally
    )
    print("\nfitting (EM)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mod.fit(maxiter=maxiter, disp=20)

    # Extract the smoothed firm-specific factors (monthly frequency).
    firm_factors = res.factors.smoothed[TICKERS].copy()
    firm_factors.index = firm_factors.index.to_timestamp()
    firm_factors.index.name = "date"
    firm_factors.to_csv(OUT_CSV)
    print(f"\nsaved {OUT_CSV}  ({firm_factors.shape[0]} months x {firm_factors.shape[1]} firm factors)")
    print(firm_factors.tail())
    return res


if __name__ == "__main__":
    main()
