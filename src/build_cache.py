"""
Fetches FRED data, trains the model, and saves all dashboard inputs to data/cache.pkl.
Run: python src/build_cache.py
GitHub Actions runs this daily and commits the result so the Streamlit app loads instantly.
"""
import sys
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score

sys.path.insert(0, os.path.dirname(__file__))
from fetch import fetch_all
from model import build_dataset, train, FEATURES, feature_importances, walk_forward_predict

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CACHE_PATH   = os.path.join(PROJECT_ROOT, "data", "cache.pkl")


def compute_analogs(df, scaler, prob_series, recession, n=3):
    """Find the n historical months most similar to today's macro environment."""
    X_scaled = pd.DataFrame(
        scaler.transform(df[FEATURES]),
        index=df.index, columns=FEATURES
    )
    current = X_scaled.iloc[-1].values
    # Exclude last 6 months to avoid near-trivial self-similarity
    past = X_scaled.iloc[:-6]
    dists = np.sqrt(((past.values - current) ** 2).sum(axis=1))
    dist_series = pd.Series(dists, index=past.index)
    top_dates = dist_series.nsmallest(n).index

    analogs = []
    for date in top_dates:
        window = recession[
            (recession.index > date) &
            (recession.index <= date + pd.DateOffset(months=12))
        ]
        analogs.append({
            "date": date.strftime("%B %Y"),
            "distance": round(float(dist_series[date]), 2),
            "recession_12m": bool(window.sum() > 0),
            "prob_then": round(float(prob_series.get(date, 0)), 1),
        })

    return analogs


def build():
    # Read previous probability before overwriting cache (for threshold-crossing alerts)
    prev_prob = None
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                old = pickle.load(f)
            prev_prob = float(old["prob_series"].iloc[-1])
        except Exception:
            pass

    print("Fetching data...")
    data = fetch_all()

    print("Building dataset...")
    df = build_dataset(data)

    print("Training model...")
    scaler, model = train(df)

    print("In-sample predictions...")
    X_scaled    = scaler.transform(df[FEATURES])
    proba       = model.predict_proba(X_scaled)[:, 1] * 100
    prob_series = pd.Series(proba, index=df.index, name="prob")

    print("Walk-forward OOS predictions (this takes a minute)...")
    oos_series = walk_forward_predict(df)

    print("Computing performance metrics...")
    targets = df["target"].reindex(oos_series.index)
    rows = []
    for t in [10, 20, 30, 50]:
        pred = (oos_series >= t).astype(int)
        p = precision_score(targets, pred, zero_division=0)
        r = recall_score(targets, pred, zero_division=0)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        rows.append({"Threshold": f"{t}%", "Precision": f"{p:.0%}",
                     "Recall": f"{r:.0%}", "F1": f"{f1:.0%}"})
    perf_df = pd.DataFrame(rows).set_index("Threshold")

    candidate_thresholds = np.arange(5, 85, 1)
    f1s = [
        2 * precision_score(targets, (oos_series >= t).astype(int), zero_division=0)
          * recall_score(targets, (oos_series >= t).astype(int), zero_division=0)
        / max(precision_score(targets, (oos_series >= t).astype(int), zero_division=0)
              + recall_score(targets, (oos_series >= t).astype(int), zero_division=0), 1e-9)
        for t in candidate_thresholds
    ]
    danger_threshold = int(candidate_thresholds[np.argmax(f1s)])

    recession    = data["recession"].resample("ME").last().dropna()
    latest       = df.iloc[-1]
    credit_mean  = df["credit_spread"].mean()
    importances  = feature_importances(model)
    coefs        = dict(zip(FEATURES, model.coef_[0]))
    intercept    = float(model.intercept_[0])

    # Feature contributions: scaled_value × coefficient → contribution to log-odds
    # Positive = pushing probability up; negative = pulling it down
    current_scaled = scaler.transform(df[FEATURES].iloc[[-1]])[0]
    contributions = {FEATURES[i]: float(current_scaled[i] * model.coef_[0][i])
                     for i in range(len(FEATURES))}

    # Historical analogs: 3 past macro environments most similar to today
    print("Computing historical analogs...")
    analogs = compute_analogs(df, scaler, prob_series, recession, n=3)

    # Yield-curve-only model: approximates the NY Fed probit spec (Estrella & Mishkin 1998)
    # Uses only the 10Y–3M spread — useful as a simpler benchmark on the main chart
    df_train = df[df["target"].notna()]
    sc_yc = StandardScaler()
    X_yc_train = sc_yc.fit_transform(df_train[["yield_spread"]])
    m_yc  = LogisticRegression(random_state=42, max_iter=1000)
    m_yc.fit(X_yc_train, df_train["target"])
    nyfed_series = pd.Series(
        m_yc.predict_proba(sc_yc.transform(df[["yield_spread"]]))[:, 1] * 100,
        index=df.index, name="nyfed_approx"
    )

    # Scaler parameters (mean + std per feature) — used for what-if tool and factor attribution
    scaler_params = {feat: {"mean": float(scaler.mean_[i]), "scale": float(scaler.scale_[i])}
                     for i, feat in enumerate(FEATURES)}

    # Data freshness: last observation date per raw FRED series
    data_freshness = {name: s.dropna().index[-1].strftime("%b %Y") for name, s in data.items()}

    cache = dict(
        prob_series=prob_series,
        oos_series=oos_series,
        recession=recession,
        latest=latest,
        credit_mean=credit_mean,
        importances=importances,
        perf_df=perf_df,
        targets=targets,
        danger_threshold=danger_threshold,
        coefs=coefs,
        intercept=intercept,
        df_history=df,
        scaler=scaler,
        contributions=contributions,
        analogs=analogs,
        nyfed_series=nyfed_series,
        prev_prob=prev_prob,
        scaler_params=scaler_params,
        data_freshness=data_freshness,
    )

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

    prob = prob_series.iloc[-1]
    print(f"Cache saved to {CACHE_PATH}")
    print(f"  Dataset: {df.index[0].date()} to {df.index[-1].date()}  ({len(df)} rows)")
    print(f"  Current recession probability: {prob:.1f}%")
    if prev_prob is not None:
        print(f"  Previous probability: {prev_prob:.1f}%  (change: {prob - prev_prob:+.1f} pp)")
    print(f"  Top analog: {analogs[0]['date']}  (recession followed: {analogs[0]['recession_12m']})")


if __name__ == "__main__":
    build()
