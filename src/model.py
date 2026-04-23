# %%
import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import TimeSeriesSplit
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
    df["yield_spread"]      = df["gs10"] - df["tb3ms"]                   # 10Y-3M spread: negative = inverted curve
    df["credit_spread"]     = df["baa"] - df["gs10"]                     # Baa-Treasury: high = credit stress
    df["indpro_chg"]        = df["indpro"].pct_change(12) * 100          # 12m % change: falling = contraction
    df["commodity_chg"]     = df["commodity"].pct_change(12) * 100       # 12m % change in PPI: falling = demand slump
    df["commodity_ma_ratio"] = df["commodity"] / df["commodity"].rolling(12).mean()  # ratio to 12m MA: <1 = below trend
    df["permits_chg"]       = df["permits"].pct_change(12) * 100         # 12m % change: falling = housing slowdown
    df["sp500_chg"]         = df["sp500"].pct_change(12) * 100           # 12m % change: used in stress_breadth only (DJIA/SP500 not available pre-2017 on FRED)
    df["payrolls_chg"]      = df["payrolls"].pct_change(3) * 400          # annualized 3m rate: captures recent hiring pace; negative = shedding jobs
    # --- CEI COMPONENTS (Conference Board Coincident Economic Index) ---
    # These four series are the NBER's primary recession dating inputs
    df["real_pi_chg"]       = df["real_pi"].pct_change(12) * 100         # 12m % change: personal income ex-transfers
    df["mfg_trade_chg"]     = df["mfg_trade"].pct_change(12) * 100       # 12m % change: mfg & trade sales
    df["sentiment_chg"]     = df["sentiment"].diff(12)                   # 12m change: falling = demand weakness
    # --- MONETARY POLICY FEATURES (professor improvement) ---
    # Federal Funds Rate change: captures active tightening cycles (Bernanke & Blinder 1992)
    df["fedfunds_chg"]      = df["fedfunds"].diff(12)                    # 12m change: rising = tightening = stress
    # Real Federal Funds Rate: nominal FFR minus CPI inflation.
    # Captures genuine monetary restrictiveness — high nominal rates with high inflation
    # are not actually tight (e.g. 1970s). Volcker's 1980 hike was devastating precisely
    # because real rates turned sharply positive.
    cpi_yoy                 = df["cpi"].pct_change(12) * 100
    df["cpi_yoy"]           = cpi_yoy                                    # kept for real_fedfunds; not a model feature (collinear with real_fedfunds)
    df["real_fedfunds"]     = df["fedfunds"] - cpi_yoy                   # positive = restrictive policy
    df["cpi_accel"]         = cpi_yoy.diff(3)                            # 3m change in YoY inflation: positive = accelerating → Fed pressure → recession risk

    # --- COMPOSITE FEATURES ---
    # Individual features are linear inputs; composites capture interactions and breadth
    # that a linear model cannot learn from separate features alone.

    # Combined financial tightening: both components flipped so higher = more stress
    df["financial_stress"] = df["credit_spread"] - df["yield_spread"]

    # Real activity composite: average of three output/demand indicators reduces noise
    df["real_activity"]    = (df["indpro_chg"] + df["permits_chg"] + df["commodity_chg"]) / 3

    # Yield curve momentum: negative = curve is actively inverting (most dangerous phase)
    df["yield_momentum"]   = df["yield_spread"].diff(3)

    # Stress breadth: how many indicators are simultaneously in stress territory (0–7)
    # Recessions almost always have 5+ signals firing at once; expansions typically 1–2
    df["stress_breadth"] = (
        (df["yield_spread"] < 0).astype(int) +
        (df["credit_spread"] > df["credit_spread"].expanding().mean()).astype(int) +
        (df["indpro_chg"] < 0).astype(int) +
        (df["sp500_chg"] < 0).astype(int) +
        (df["payrolls_chg"] < 0).astype(int) +
        (df["permits_chg"] < 0).astype(int) +
        (df["sentiment_chg"] < 0).astype(int)
    )

    # --- VICIOUS CYCLE INDEX (Zandi) ---
    # Standard unemployment = 1 - (employment / (population * lfpr))
    # VCI replaces current lfpr with its 5-year (60-month) moving average.
    # This removes the distortion from participation rate swings (e.g. immigration
    # surges in 2024 that pushed up unemployment without a true labor market slump).
    lfpr_pct        = df["lfpr"] / 100
    lfpr_5yr_ma     = lfpr_pct.rolling(60).mean()
    vci             = 1 - (df["employment"] / (df["population"] * lfpr_5yr_ma))

    # Apply Zandi's signal logic to VCI:
    # Signal = 3m MA of VCI minus its minimum over the past 12 months.
    # > 1pp triggers Zandi's recession warning (VCI is a lagging indicator).
    # Included for dashboard display only — not used as a model feature.
    vci_3m          = vci.rolling(3).mean()
    df["vci_signal"] = (vci_3m - vci_3m.rolling(12).min()) * 100

    # --- POLICY & UNCERTAINTY FEATURES (alternative model only) ---
    # Both columns are always computed and kept in df so the alternative model can use them.
    # The main FEATURES list does not include them; they appear only in FEATURES_ALT.
    df["epu_news_level"]  = df["epu_news"].rolling(3, min_periods=1).mean()
    # Trade Policy Uncertainty starts 1985; fill pre-1985 with long-run mean so the
    # standardiser maps those rows to ≈ 0 ("average"), preserving full training history.
    trade_mean = df["epu_trade"].dropna().mean()
    df["epu_trade_level"] = df["epu_trade"].rolling(3, min_periods=1).mean().fillna(trade_mean)

    # TARGET columns: one model per horizon (direct multi-step forecasting).
    # At each row t, target_h = "will we be in recession at t+h?"
    df["target"]     = df["recession"].shift(-3)
    df["target_6m"]  = df["recession"].shift(-6)
    df["target_12m"] = df["recession"].shift(-12)

    df = df.drop(columns=["gs10", "tb3ms", "baa", "indpro", "commodity", "employment", "population", "lfpr", "permits", "sp500", "sp500_chg", "payrolls", "real_pi", "mfg_trade", "sentiment", "fedfunds", "cpi", "unrate", "recession", "epu_news", "epu_trade"])
    # Forward-fill feature columns to handle lagged FRED releases:
    # if a series hasn't published yet for the latest month(s), carry forward
    # the most recent available reading rather than dropping the whole row.
    feature_cols = [c for c in df.columns if not c.startswith("target")]
    df[feature_cols] = df[feature_cols].ffill()
    # Drop rows where features are still NaN (start of series, rolling windows not filled yet).
    # Keep rows where only target is NaN — those are the most recent months we predict on.
    df = df.dropna(subset=feature_cols)
    return df


# %%
# --- TRAIN ---
# financial_stress (= credit_spread - yield_spread) and real_activity
# (= mean of indpro_chg, permits_chg, commodity_chg) are exact linear combinations
# of features already in this list. Including them alongside their components
# creates rank deficiency; coefficients become arbitrary under any linear solver.
FEATURES = [
    "yield_spread", "credit_spread", "indpro_chg", "commodity_chg", "commodity_ma_ratio",
    "permits_chg", "payrolls_chg", "real_pi_chg", "mfg_trade_chg",
    "sentiment_chg", "fedfunds_chg", "real_fedfunds",
    "yield_momentum", "stress_breadth",
    "cpi_accel",
]

# Alternative model adds news-based and trade-specific policy uncertainty.
FEATURES_ALT = FEATURES + ["epu_news_level", "epu_trade_level"]

def train(df: pd.DataFrame, horizon: int = 3, features: list = None):
    """Fit a regularisation-tuned logistic regression via time-series CV.

    Uses LogisticRegressionCV with TimeSeriesSplit so the penalty strength C is
    selected on held-out data that respects temporal ordering (no future leakage).
    A gap of 6 months between each train/test split reduces autocorrelation
    inflation of the CV score.
    Returns (scaler, model).
    """
    if features is None:
        features = FEATURES
    target_col = "target" if horizon == 3 else f"target_{horizon}m"
    df_train = df[df[target_col].notna()]
    X = df_train[features]
    y = df_train[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=5, gap=6)
    model = LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 20),
        cv=tscv,
        scoring="roc_auc",
        penalty="l2",
        random_state=42,
        max_iter=2000,
        n_jobs=-1,
    )
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
# --- FEATURE IMPORTANCES ---
def feature_importances(model, features: list = None) -> dict:
    """Relative importance of each feature as % of total absolute coefficient mass."""
    if features is None:
        features = FEATURES
    abs_coefs = np.abs(model.coef_[0])
    return {feat: abs_coefs[i] / abs_coefs.sum() * 100
            for i, feat in enumerate(features)}


# %%
# --- WALK-FORWARD OUT-OF-SAMPLE BACKTESTING ---
# For each month t, trains only on data strictly before t, then predicts t.
# This is the academically honest approach (Stock & Watson 1999): no look-ahead bias.
# In-sample metrics overstate performance because the model has seen the answers.

def walk_forward_predict(df: pd.DataFrame, min_train: int = 60, horizon: int = 3,
                         features: list = None) -> pd.Series:
    """Expanding-window OOS predictions. min_train = minimum months before first prediction."""
    if features is None:
        features = FEATURES
    target_col = "target" if horizon == 3 else f"target_{horizon}m"
    results = {}
    for i in range(min_train, len(df)):
        if pd.isna(df[target_col].iloc[i]):
            continue
        train_df = df.iloc[:i][df.iloc[:i][target_col].notna()]
        if train_df[target_col].sum() < 5:
            continue
        sc = StandardScaler()
        X_tr = sc.fit_transform(train_df[features])
        m = LogisticRegression(random_state=42, max_iter=1000)
        m.fit(X_tr, train_df[target_col])
        X_pr = sc.transform(df.iloc[[i]][features])
        results[df.index[i]] = m.predict_proba(X_pr)[0, 1] * 100
    return pd.Series(results, name=f"oos_prob_{horizon}m")


# %%
# --- RUN ALL ---
# Run this cell to train and evaluate the model end-to-end.
if __name__ == "__main__":
    data          = fetch_all()
    df            = build_dataset(data)
    scaler, model = train(df)

    evaluate(df, scaler, model)

    prob = current_probability(df, scaler, model)
    latest_date = df.index[-1].strftime("%B %Y")
    print(f"\nAs of {latest_date}:")
    print(f"  3-month recession probability: {prob:.1%}")
