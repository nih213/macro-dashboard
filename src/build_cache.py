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
from sklearn.metrics import precision_score, recall_score

sys.path.insert(0, os.path.dirname(__file__))
from fetch import fetch_all
from model import build_dataset, train, FEATURES, feature_importances, walk_forward_predict

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CACHE_PATH   = os.path.join(PROJECT_ROOT, "data", "cache.pkl")


def build():
    print("Fetching data...")
    data = fetch_all()

    print("Building dataset...")
    df = build_dataset(data)

    print("Training model...")
    scaler, model = train(df)

    print("In-sample predictions...")
    from sklearn.preprocessing import StandardScaler
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

    recession   = data["recession"].resample("ME").last().dropna()
    latest      = df.iloc[-1]
    credit_mean = df["credit_spread"].mean()
    importances = feature_importances(model)
    coefs       = dict(zip(FEATURES, model.coef_[0]))
    intercept   = float(model.intercept_[0])

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
    )

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

    print(f"Cache saved to {CACHE_PATH}")
    print(f"  Dataset: {df.index[0].date()} to {df.index[-1].date()}  ({len(df)} rows)")
    prob = prob_series.iloc[-1]
    print(f"  Current recession probability: {prob:.1f}%")


if __name__ == "__main__":
    build()
