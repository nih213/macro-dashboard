# %%
import os
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

# %%
# --- CONFIG ---
# Loads FRED_API_KEY from the .env file in the project root.
# The .env file is gitignored — never committed to GitHub.
load_dotenv()
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
fred = Fred(api_key=FRED_API_KEY)

START_DATE = "1950-01-01"

# %%
# --- SERIES TO FETCH ---
# Each entry: friendly_name -> FRED series ID
SERIES = {
    # Yield curve: computed as GS10 - TB3MS (10Y-3M spread) — starts 1953
    # Using 10Y-3M rather than 10Y-2Y: original Estrella & Mishkin (1998) spec, starts earlier
    "gs10":        "GS10",            # 10-Year Treasury Constant Maturity Rate
    "tb3ms":       "TB3MS",           # 3-Month Treasury Bill Secondary Market Rate
    # Credit spread: Moody's Baa yield minus 10Y Treasury — starts 1953 (replaces HY spread from 1996)
    "baa":         "BAA",             # Moody's Baa Corporate Bond Yield
    "indpro":      "INDPRO",          # Industrial production index (business cycle) — starts 1919
    "commodity":   "PPIACO",          # PPI: All Commodities (global demand proxy, replaces copper) — starts 1913
    "employment":  "CE16OV",          # Civilian employment level — for VCI calculation — starts 1948
    "population":  "CNP16OV",         # Civilian noninstitutional population — for VCI — starts 1948
    "lfpr":        "CIVPART",         # Labor force participation rate — for VCI — starts 1948
    "permits":     "PERMIT",          # Building permits (housing leading indicator) — starts 1960
    "sp500":       "DJIA",            # Dow Jones Industrial Average (replaces NASDAQ/SP500) — starts 1928
    "payrolls":    "PAYEMS",          # Total nonfarm payrolls (replaces initial claims) — starts 1939
    "real_pi":     "W875RX1",         # Real personal income excl. transfer receipts (CEI component) — starts 1959
    "mfg_trade":   "CMRMTSPL",        # Real mfg & trade industries sales (CEI component) — starts 1967
    "sentiment":   "UMCSENT",         # University of Michigan consumer sentiment — starts 1952
    "fedfunds":    "FEDFUNDS",        # Federal Funds Rate — monetary policy stance — starts 1954
    "cpi":         "CPIAUCSL",        # CPI All Items — for real FFR and inflation feature — starts 1947
    "unrate":      "UNRATE",          # Unemployment rate — for outcome calculator — starts 1948
    "recession":   "USREC",           # NBER recession indicator: 1 = recession, 0 = expansion
}

# %%
# --- FETCH ---
def fetch_all() -> dict:
    """Pull each series from FRED and return as a dict of named Series."""
    import time
    result = {}
    for name, series_id in SERIES.items():
        print(f"Fetching {name} ({series_id})...")
        for attempt in range(1, 5):
            try:
                s = fred.get_series(series_id, observation_start=START_DATE)
                result[name] = s.rename(name)
                break
            except Exception as e:
                if attempt == 4:
                    raise
                wait = 15 * attempt
                print(f"  FRED error ({e}), retrying in {wait}s (attempt {attempt}/3)...")
                time.sleep(wait)
    print("Done.")
    return result

# %%
# Run this cell to test — you should see 5 series with recent dates and values
# The if __name__ == "__main__" guard means this block only runs when you
# execute fetch.py directly. It is skipped when another file imports fetch.py.
if __name__ == "__main__":
    data = fetch_all()

    for name, series in data.items():
        latest_date = series.dropna().index[-1].date()
        latest_val  = series.dropna().iloc[-1]
        print(f"  {name:<15} {len(series):>5} obs   latest: {latest_date}  value: {latest_val:.3f}")
