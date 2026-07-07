# Composite Index of Firm-Specific Risk

A demo that builds a **composite index** of firm-specific
(idiosyncratic) factors for seven large U.S. technology firms (Apple, Microsoft, Amazon, Nvidia, Alphabet, Tesla, and Meta) from public data.

The full methodology and results are written up in **[`composite_index.pdf`](composite_index.pdf)**.

## What it does

1. **Collect data** (all public): monthly stock prices, eight quarterly financial
   measures per firm from SEC filings, and U.S. macro series (real output per
   capita, interest rate, inflation) from FRED.
2. **Principal component analysis** of the log stock prices, and extraction of each
   firm's idiosyncratic price residual.
3. **Mixed-frequency dynamic factor model** (a national factor plus firm-specific
   factors, AR(2) dynamics) estimated with `statsmodels`, yielding a firm-specific
   factor for each firm.
4. **Composite index** built from those factors following OECD best practice —
   orientation, normalization, and weighting/aggregation (equal, PCA-based, and
   geometric schemes compared).

## Repository layout

```
composite_index.pdf              Write-up (methodology, figures, results)
src/
  01_public_firm_data.ipynb      Extract SEC financials (test quarter)
  02_macro_data_stock_pca.ipynb  FRED macro, monthly prices, PCA, residuals
  03_composite_index.ipynb       Construct the composite index
  build_firm_financials_panel.py Build the 2015Q1–2026Q1 SEC financials panel
  mixed_frequency_dfm.py         Mixed-frequency dynamic factor model
data/                            Derived datasets (macro, prices, panels, factors, index)
graph/                           Generated figures
```

Note: raw SEC bulk downloads (`data/sec/`) are large and **not** committed — they
are re-downloaded on demand by `build_firm_financials_panel.py`. The small derived
datasets are included, so the analysis is reproducible without them.

## Reproduce

Requires Python 3.11 with `pandas`, `numpy`, `matplotlib`, `scikit-learn`, and
`statsmodels`.

Run in this order (later steps depend on earlier outputs):

1. Notebook `02_macro_data_stock_pca.ipynb` — builds the monthly prices and macro
   series, and runs the PCA. (`01_public_firm_data.ipynb` is an optional
   single-quarter SEC test.)
2. `python src/build_firm_financials_panel.py` — downloads SEC data and builds the
   2015Q1–2026Q1 financials panel.
3. `python src/mixed_frequency_dfm.py` — estimates the firm-specific factors
   (needs the prices from step 1 and the panel from step 2).
4. Notebook `03_composite_index.ipynb` — constructs the composite index.

Figures are written to `graph/` and the composite index to
`data/composite_index.csv`.

## Data sources

- **Stock prices** — weekly closing prices, averaged to monthly.
- **Firm financials** — [SEC DERA Financial Statement Data Sets](https://www.sec.gov/dera/data/financial-statement-data-sets).
- **Macro** — [FRED](https://fred.stlouisfed.org/): real GDP per capita
  (`A939RX0Q048SBEA`), 3-month T-bill rate (`TB3MS`), core CPI (`CPILFESL`).
