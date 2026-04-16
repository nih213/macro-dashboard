# %%
import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Allow importing fetch.py from the same src/ folder
sys.path.insert(0, os.path.dirname(__file__))
from fetch import fetch_all

# %%
# --- BUILD DATASET ---
# We align all series to monthly frequency, then engineer features.
# Raw series like unemployment and industrial production are more useful
# as *changes* than as levels — a rising unemployment rate signals stress
# regardless of whether it starts from 3% or 6%.

def build_dataset(data: dict) -> pd.DataFrame:
    # Resample everything to month-end frequency
    monthly = {name: s.resample("ME").last() for name, s in data.items()}
    df = pd.DataFrame(monthly)

    # Feature engineering
    df["yield_spread"]     = df["yield_spread"]              # level: negative = inverted curve
    df["unemployment_chg"] = df["unemployment"].diff(12)     # 12m change: rising = stress
    df["credit_spread"]    = df["credit_spread"]             # level: high = risk-off
    df["indpro_chg"]       = df["indpro"].pct_change(12) * 100  # 12m % change: falling = contraction

    # TARGET: shift recession indicator back 3 months.
    # At each row (month t), the target = "will we be in recession at t+3?"
    # This is what makes the model a 3-month-ahead forecast.
    df["target"] = df["recession"].shift(-3)

    df = df.drop(columns=["unemployment", "indpro", "recession"])
    df = df.dropna()
    return df


# %%
# --- TRAIN ---
FEATURES = ["yield_spread", "unemployment_chg", "credit_spread", "indpro_chg"]

def train(df: pd.DataFrame):
    """Fit a logistic regression. Returns (scaler, model)."""
    X = df[FEATURES]
    y = df["target"]

    scaler = StandardScaler()   # logistic regression needs scaled inputs
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)
    return scaler, model


# %%
# --- EVALUATE ---
# ROC-AUC measures how well the model separates recession from expansion months.
# 0.5 = no better than a coin flip. 0.8+ = genuinely useful.

def evaluate(df: pd.DataFrame, scaler, model):
    X_scaled = scaler.transform(df[FEATURES])
    y = df["target"]

    proba = model.predict_proba(X_scaled)[:, 1]
    auc   = roc_auc_score(y, proba)
    print(f"ROC-AUC: {auc:.3f}  (0.5 = random | 0.8+ = good | 1.0 = perfect)")
    print()

    print("Feature coefficients (positive = raises recession probability):")
    for feat, coef in zip(FEATURES, model.coef_[0]):
        bar = "+" * int(abs(coef) * 5)
        sign = "+" if coef > 0 else "-"
        print(f"  {feat:<22} {sign}{abs(coef):.3f}  {bar}")


# %%
# --- CURRENT READING ---
def current_probability(df: pd.DataFrame, scaler, model) -> float:
    """Recession probability based on the latest available data."""
    latest  = df[FEATURES].iloc[[-1]]
    X_scaled = scaler.transform(latest)
    prob    = model.predict_proba(X_scaled)[0, 1]
    return prob


# %%
# --- RUN ALL ---
# Run this cell to train and evaluate the model end-to-end.
data          = fetch_all()
df            = build_dataset(data)
scaler, model = train(df)

evaluate(df, scaler, model)

prob = current_probability(df, scaler, model)
latest_date = df.index[-1].strftime("%B %Y")
print(f"\nAs of {latest_date}:")
print(f"  3-month recession probability: {prob:.1%}")
