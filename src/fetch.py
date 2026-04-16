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

START_DATE = "1990-01-01"

# %%
# --- SERIES TO FETCH ---
# Each entry: friendly_name -> FRED series ID
SERIES = {
    "yield_spread":  "T10Y2Y",        # 10Y minus 2Y Treasury yield (the yield curve)
    "unemployment":  "UNRATE",         # US unemployment rate (monthly)
    "credit_spread": "BAMLH0A0HYM2",  # High-yield credit spread (risk appetite)
    "indpro":        "INDPRO",         # Industrial production index (business cycle)
    "recession":     "USREC",          # NBER recession indicator: 1 = recession, 0 = expansion
}

# %%
# --- FETCH ---
def fetch_all() -> dict:
    """Pull each series from FRED and return as a dict of named Series."""
    result = {}
    for name, series_id in SERIES.items():
        print(f"Fetching {name} ({series_id})...")
        s = fred.get_series(series_id, observation_start=START_DATE)
        result[name] = s.rename(name)
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
