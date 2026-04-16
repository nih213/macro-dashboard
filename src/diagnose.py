"""
Diagnoses which FRED series is limiting the model's start date.
Run with: python src/diagnose.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from fetch import fetch_all
from model import build_dataset, FEATURES

# --- RAW SERIES ---
print("=== RAW SERIES (first/last date) ===")
data = fetch_all()
for name, s in data.items():
    clean = s.dropna()
    print(f"  {name:<15} {clean.index[0].date()} → {clean.index[-1].date()}")

# --- FEATURES BEFORE DROPNA ---
# Temporarily rebuild without dropna to see each column's first valid date
print("\n=== BUILT FEATURES (first valid date each column) ===")
monthly = {name: s.resample("ME").last() for name, s in data.items()}
df_raw = pd.DataFrame(monthly)

# Re-run feature engineering manually (mirrors build_dataset)
df_raw["yield_spread"]       = df_raw["gs10"] - df_raw["tb3ms"]
df_raw["credit_spread"]      = df_raw["baa"] - df_raw["gs10"]
df_raw["indpro_chg"]         = df_raw["indpro"].pct_change(12) * 100
df_raw["commodity_chg"]      = df_raw["commodity"].pct_change(12) * 100
df_raw["commodity_ma_ratio"] = df_raw["commodity"] / df_raw["commodity"].rolling(12).mean()
df_raw["permits_chg"]        = df_raw["permits"].pct_change(12) * 100
df_raw["sp500_chg"]          = df_raw["sp500"].pct_change(12) * 100
df_raw["payrolls_chg"]       = df_raw["payrolls"].pct_change(12) * 100
df_raw["sentiment_chg"]      = df_raw["sentiment"].diff(12)
df_raw["financial_stress"]   = df_raw["credit_spread"] - df_raw["yield_spread"]
df_raw["real_activity"]      = (df_raw["indpro_chg"] + df_raw["permits_chg"] + df_raw["commodity_chg"]) / 3
df_raw["yield_momentum"]     = df_raw["yield_spread"].diff(3)
lfpr_pct    = df_raw["lfpr"] / 100
lfpr_5yr_ma = lfpr_pct.rolling(60).mean()
vci         = 1 - (df_raw["employment"] / (df_raw["population"] * lfpr_5yr_ma))
vci_3m      = vci.rolling(3).mean()
df_raw["vci_signal"] = (vci_3m - vci_3m.rolling(12).min()) * 100
df_raw["stress_breadth"] = (
    (df_raw["yield_spread"] < 0).astype(int) +
    (df_raw["credit_spread"] > df_raw["credit_spread"].expanding().mean()).astype(int) +
    (df_raw["indpro_chg"] < 0).astype(int) +
    (df_raw["sp500_chg"] < 0).astype(int) +
    (df_raw["payrolls_chg"] < 0).astype(int) +
    (df_raw["permits_chg"] < 0).astype(int) +
    (df_raw["sentiment_chg"] < 0).astype(int)
)

all_cols = FEATURES + ["vci_signal"]
for col in all_cols:
    if col in df_raw.columns:
        first = df_raw[col].first_valid_index()
        print(f"  {col:<22} first valid: {first.date() if first else 'NEVER'}")

print(f"\n=== AFTER DROPNA ===")
df = build_dataset(data)
print(f"  Dataset runs: {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} rows)")
print(f"\n  Bottleneck = whichever feature above has the latest first-valid date")
