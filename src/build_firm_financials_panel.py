"""Build a quarterly panel of firm financials from SEC DERA Financial Statement Data Sets.

Loops the quarterly zip files from 2015Q1 through 2026Q1, extracts consolidated
financials for a fixed set of large-cap firms, derives clean *quarterly* series, and
writes a tidy long panel.

Output (data/firm_financials_panel/):
    firm_financials_long.csv   columns: date, ticker, measure, value
    firm_financials_wide.csv   date, ticker, one column per measure

    date    : fiscal period end (quarter end), YYYY-MM-DD
    ticker  : AAPL, MSFT, AMZN, NVDA, GOOGL, TSLA, META
    measure : revenue, cost_of_revenue, gross_profit, operating_income,
              rd_expense, net_income, total_assets, stockholders_equity
    value   : USD

Method
------
- Consolidated only: keep coreg == "" and segments == "" (company-level totals).
- Current period only: keep facts whose ddate == the filing period (drops the
  prior-year comparatives that filings also carry).
- Tag coalescing: each measure maps to a priority list of candidate XBRL tags,
  covering tag changes over 2015-2026 (e.g. ASC 606 revenue tags).
- Flows (income statement): interim quarters Q1-Q3 come from 10-Q facts (qtrs == 1);
  fiscal Q4 is derived as annual (qtrs == 4) minus the sum of the three interim
  quarters of the same fiscal year.
- Stocks (balance sheet): taken at each period end (qtrs == 0).
- Restatements: dedup on (cik, measure, qtrs, ddate), keeping the preferred tag and
  the latest-filed value.

Run:  python build_firm_financials_panel.py
"""

from __future__ import annotations

import os
import zipfile

import pandas as pd
import requests

# --------------------------------------------------------------------------- config
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))
ZIP_DIR = os.path.join(DATA_DIR, "sec", "zips")          # cached raw quarterly zips
OUT_DIR = os.path.join(DATA_DIR, "firm_financials_panel")  # new output sub-directory

BASE_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets"
HEADERS = {"User-Agent": "Xiangtao M xiangtaom@gmail.com"}  # SEC requires a contact

START = (2015, 1)
END = (2026, 1)

FIRMS = pd.DataFrame(
    [
        ("AAPL", 320193, "Apple"),
        ("MSFT", 789019, "Microsoft"),
        ("AMZN", 1018724, "Amazon"),
        ("NVDA", 1045810, "Nvidia"),
        ("GOOGL", 1652044, "Alphabet"),
        ("TSLA", 1318605, "Tesla"),
        ("META", 1326801, "Meta"),
    ],
    columns=["ticker", "cik", "company"],
)
CIK2TICKER = dict(zip(FIRMS["cik"], FIRMS["ticker"]))
CIKS = set(FIRMS["cik"])

# measure -> candidate XBRL tags, in priority order (first available wins)
CONCEPTS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "cost_of_revenue": [
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
        "CostOfGoodsSold",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "rd_expense": ["ResearchAndDevelopmentExpense"],
    "net_income": ["NetIncomeLoss"],
    "total_assets": ["Assets"],
    "stockholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
}
STOCK_MEASURES = {"total_assets", "stockholders_equity"}

TAG2MEASURE = {tag: m for m, tags in CONCEPTS.items() for tag in tags}
TAG_PRIORITY = {tag: i for tags in CONCEPTS.values() for i, tag in enumerate(tags)}
USED_TAGS = set(TAG2MEASURE)


# ------------------------------------------------------------------------- helpers
def quarters(start=START, end=END):
    """Yield 'YYYYqN' strings from start to end inclusive."""
    y, q = start
    while (y, q) <= end:
        yield f"{y}q{q}"
        q += 1
        if q == 5:
            y, q = y + 1, 1


def get_zip(quarter: str) -> str:
    """Return local path to the quarter's zip, downloading (and caching) if needed."""
    os.makedirs(ZIP_DIR, exist_ok=True)
    path = os.path.join(ZIP_DIR, f"{quarter}.zip")
    if not os.path.exists(path):
        url = f"{BASE_URL}/{quarter}.zip"
        print(f"  downloading {url}")
        resp = requests.get(url, headers=HEADERS, timeout=180)
        resp.raise_for_status()
        with open(path, "wb") as fh:
            fh.write(resp.content)
    return path


def read_quarter(quarter: str) -> pd.DataFrame:
    """Read one quarterly data set and return consolidated current-period facts
    for our firms and measures. Columns:
    cik, ticker, fy, fp, measure, tag_prio, qtrs, ddate, value, filed.
    """
    path = get_zip(quarter)
    with zipfile.ZipFile(path) as zf:
        sub = pd.read_csv(
            zf.open("sub.txt"),
            sep="\t",
            low_memory=False,
            keep_default_na=False,
            usecols=["adsh", "cik", "period", "fy", "fp", "filed"],
        )
        sub = sub[sub["cik"].isin(CIKS)]
        if sub.empty:
            return pd.DataFrame()
        adsh_set = set(sub["adsh"])

        # num.txt is large: read in chunks and keep only our filings + measures.
        parts = []
        for chunk in pd.read_csv(
            zf.open("num.txt"),
            sep="\t",
            low_memory=False,
            keep_default_na=False,
            usecols=["adsh", "tag", "ddate", "qtrs", "uom", "segments", "coreg", "value"],
            chunksize=500_000,
        ):
            chunk = chunk[
                chunk["adsh"].isin(adsh_set)
                & chunk["tag"].isin(USED_TAGS)
                & (chunk["coreg"] == "")
                & (chunk["segments"] == "")
                & (chunk["uom"] == "USD")
            ]
            if not chunk.empty:
                parts.append(chunk)
    if not parts:
        return pd.DataFrame()

    num = pd.concat(parts, ignore_index=True)
    df = num.merge(sub, on="adsh", how="inner")

    # Numeric coercion.
    for col in ("value", "ddate", "period", "qtrs", "filed"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["fy"] = pd.to_numeric(df["fy"], errors="coerce")

    # Current period only (drop prior-year comparatives carried in the same filing).
    df = df[df["ddate"] == df["period"]].dropna(subset=["value", "ddate"])

    df["ticker"] = df["cik"].map(CIK2TICKER)
    df["measure"] = df["tag"].map(TAG2MEASURE)
    df["tag_prio"] = df["tag"].map(TAG_PRIORITY)
    return df[
        ["cik", "ticker", "fy", "fp", "measure", "tag_prio", "qtrs", "ddate", "value", "filed"]
    ]


def build_panel(facts: pd.DataFrame) -> pd.DataFrame:
    """From all quarters' facts, build the tidy long quarterly panel."""
    # One observation per (cik, measure, qtrs, ddate): prefer the higher-priority tag,
    # then the latest-filed value (handles restatements and duplicate tags).
    facts = facts.sort_values(["tag_prio", "filed"], ascending=[True, False])
    facts = facts.drop_duplicates(["cik", "measure", "qtrs", "ddate"], keep="first")

    is_stock = facts["measure"].isin(STOCK_MEASURES)

    # Balance-sheet stocks: point-in-time values at each period end.
    stocks = facts[is_stock & (facts["qtrs"] == 0)][["ticker", "measure", "ddate", "value"]]

    flows = facts[~is_stock]
    interim = flows[flows["qtrs"] == 1]  # Q1-Q3 single-quarter income-statement values

    # Fiscal Q4 = annual - sum(interim quarters) within the same fiscal year.
    annual = (
        flows[flows["qtrs"] == 4]
        .sort_values("ddate")
        .drop_duplicates(["cik", "ticker", "measure", "fy"], keep="last")
        .rename(columns={"value": "annual", "ddate": "fye"})
    )
    isum = (
        interim.groupby(["cik", "ticker", "measure", "fy"])
        .agg(n=("value", "size"), interim_sum=("value", "sum"))
        .reset_index()
    )
    q4 = annual.merge(isum, on=["cik", "ticker", "measure", "fy"], how="inner")
    q4 = q4[q4["n"] == 3].copy()  # need all three interim quarters to back out Q4
    q4["value"] = q4["annual"] - q4["interim_sum"]
    q4 = q4.rename(columns={"fye": "ddate"})[["ticker", "measure", "ddate", "value"]]

    flow_panel = pd.concat(
        [interim[["ticker", "measure", "ddate", "value"]], q4], ignore_index=True
    )

    panel = pd.concat([stocks, flow_panel], ignore_index=True)
    panel["date"] = pd.to_datetime(panel["ddate"].astype("Int64").astype(str), format="%Y%m%d")
    panel = (
        panel[["date", "ticker", "measure", "value"]]
        .drop_duplicates(["date", "ticker", "measure"])
    )

    # Gross profit isn't tagged by every firm -> fall back to revenue - cost_of_revenue.
    wide = panel.pivot_table(index=["date", "ticker"], columns="measure", values="value")
    if {"revenue", "cost_of_revenue"}.issubset(wide.columns):
        gp = wide.get("gross_profit")
        derived = (wide["revenue"] - wide["cost_of_revenue"]).where(gp.isna() if gp is not None else True)
        derived = derived.dropna().reset_index()
        derived["measure"] = "gross_profit"
        derived = derived.rename(columns={0: "value"})
        panel = pd.concat([panel, derived[["date", "ticker", "measure", "value"]]], ignore_index=True)

    panel = (
        panel.drop_duplicates(["date", "ticker", "measure"])
        .sort_values(["ticker", "measure", "date"])
        .reset_index(drop=True)
    )
    return panel


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_facts = []
    for q in quarters():
        print(f"[{q}]")
        f = read_quarter(q)
        if not f.empty:
            all_facts.append(f)
    facts = pd.concat(all_facts, ignore_index=True)
    print(f"\n{len(facts):,} raw consolidated facts across all quarters")

    panel = build_panel(facts)
    long_path = os.path.join(OUT_DIR, "firm_financials_long.csv")
    panel.to_csv(long_path, index=False)
    print(f"wrote {long_path}  ({len(panel):,} rows)")

    wide = panel.pivot_table(index=["date", "ticker"], columns="measure", values="value").reset_index()
    wide.columns.name = None
    wide_path = os.path.join(OUT_DIR, "firm_financials_wide.csv")
    wide.to_csv(wide_path, index=False)
    print(f"wrote {wide_path}  ({len(wide):,} rows)")

    print(f"\ndate range: {panel['date'].min().date()} .. {panel['date'].max().date()}")
    print(panel.groupby(["ticker", "measure"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
