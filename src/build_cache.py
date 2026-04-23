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
from fetch import fetch_all, fetch_state_unemployment
from model import build_dataset, train, FEATURES, FEATURES_ALT, feature_importances, walk_forward_predict

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CACHE_PATH   = os.path.join(PROJECT_ROOT, "data", "cache.pkl")


def compute_state_data(data, df, coefs, intercept, scaler_params):
    """
    Per-state recession probabilities.

    Method: take the national model's log-odds, remove the national payrolls_chg
    contribution, and substitute a state-specific equivalent derived from the
    state's 12-month unemployment rate change via a historically-fitted Okun
    relationship (payrolls_chg ~ f(unrate_chg), estimated from national data).

    Returns {"probs": {state: float}, "ur_latest": {state: {"ur": float, "chg": float}}}
    """
    from sklearn.linear_model import LinearRegression

    # Fit national UR-change → payrolls_chg conversion (Okun's law, data-driven)
    unrate_m = data["unrate"].resample("ME").last()
    ur_chg   = unrate_m.diff(3).reindex(df.index)   # 3m change to match annualized-3m payrolls
    aligned  = pd.DataFrame({"ur": ur_chg, "pay": df["payrolls_chg"]}).dropna()
    if len(aligned) >= 10:
        lr = LinearRegression().fit(aligned[["ur"]], aligned["pay"])
        ur_to_pay_coef = float(lr.coef_[0])
        ur_to_pay_int  = float(lr.intercept_)
    else:
        ur_to_pay_coef, ur_to_pay_int = -2.0, 1.5

    # Base log-odds: all national features except payrolls_chg
    pay_coef  = coefs.get("payrolls_chg", 0)
    pay_mean  = scaler_params["payrolls_chg"]["mean"]
    pay_scale = scaler_params["payrolls_chg"]["scale"]

    base_lo = intercept
    for feat in FEATURES:
        if feat == "payrolls_chg":
            continue
        z = (float(df[feat].iloc[-1]) - scaler_params[feat]["mean"]) / scaler_params[feat]["scale"]
        base_lo += coefs.get(feat, 0) * z

    print("Fetching state unemployment rates (50 series)...")
    state_ur = fetch_state_unemployment()

    probs     = {}
    ur_latest = {}
    for state, s in state_ur.items():
        m   = s.resample("ME").last().dropna()
        chg = m.diff(3).dropna()   # 3m change, matching annualized-3m payrolls feature
        if len(chg) == 0:
            continue
        state_ur_chg = float(chg.iloc[-1])
        pay_equiv    = ur_to_pay_coef * state_ur_chg + ur_to_pay_int
        state_z      = (pay_equiv - pay_mean) / pay_scale
        lo           = base_lo + pay_coef * state_z
        probs[state]     = round(float(1 / (1 + np.exp(-lo)) * 100), 1)
        ur_latest[state] = {"ur": round(float(m.iloc[-1]), 1), "chg": round(state_ur_chg, 2)}

    print(f"  State probabilities computed for {len(probs)} states.")
    return {"probs": probs, "ur_latest": ur_latest}


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

    print("Training models (3m / 6m / 12m) with CV-tuned regularisation...")
    scaler,     model     = train(df, horizon=3)
    scaler_6m,  model_6m  = train(df, horizon=6)
    scaler_12m, model_12m = train(df, horizon=12)
    print(f"  Best C — 3m: {model.C_[0]:.4f}  |  6m: {model_6m.C_[0]:.4f}  |  12m: {model_12m.C_[0]:.4f}")

    print("In-sample predictions (all horizons)...")
    X_scaled    = scaler.transform(df[FEATURES])
    proba       = model.predict_proba(X_scaled)[:, 1] * 100
    prob_series = pd.Series(proba, index=df.index, name="prob")

    prob_series_6m  = pd.Series(
        model_6m.predict_proba(scaler_6m.transform(df[FEATURES]))[:, 1] * 100,
        index=df.index, name="prob_6m",
    )
    prob_series_12m = pd.Series(
        model_12m.predict_proba(scaler_12m.transform(df[FEATURES]))[:, 1] * 100,
        index=df.index, name="prob_12m",
    )

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

    # --- OUTCOME CALCULATOR ---
    # For each probability bucket, compute average forward outcomes across 3/6/12 months.
    # Uses DJIA (already fetched), payrolls (already fetched), and unemployment rate (new).
    print("Computing outcome calculator data...")
    OUTCOME_BUCKETS = [(0,  3,  "Very Low (0–3%)"),
                       (3,  7,  "Low (3–7%)"),
                       (7,  10, "Guarded (7–10%)"),
                       (10, 20, "Moderate (10–20%)"),
                       (20, 40, "Elevated (20–40%)"),
                       (40, 100,"High (40%+)")]
    djia_m   = data["sp500"].resample("ME").last()
    unrate_m = data["unrate"].resample("ME").last()
    pay_m    = data["payrolls"].resample("ME").last()

    outcome_data = pd.DataFrame({"prob": prob_series}, index=df.index)
    for h in [3, 6, 12]:
        # Forward h-month return from each date
        outcome_data[f"stocks_{h}m"]   = (djia_m.pct_change(h).shift(-h) * 100).reindex(df.index)
        # Forward change in unemployment rate
        outcome_data[f"unrate_{h}m"]   = unrate_m.diff(h).shift(-h).reindex(df.index)
        # Average monthly payroll change (series in 000s of jobs)
        outcome_data[f"payrolls_{h}m"] = (pay_m.diff(h) / h).shift(-h).reindex(df.index)

    outcome_summary = {}
    for lo, hi, label in OUTCOME_BUCKETS:
        mask = (outcome_data["prob"] >= lo) & (outcome_data["prob"] < hi)
        sub  = outcome_data[mask]
        entry = {"n": int(mask.sum())}
        for h in [3, 6, 12]:
            for metric in ["stocks", "unrate", "payrolls"]:
                col  = f"{metric}_{h}m"
                vals = sub[col].dropna()
                entry[col] = {
                    "mean": round(float(vals.mean()), 1) if len(vals) > 0 else None,
                    "p25":  round(float(vals.quantile(0.25)), 1) if len(vals) > 0 else None,
                    "p75":  round(float(vals.quantile(0.75)), 1) if len(vals) > 0 else None,
                }
        outcome_summary[label] = entry

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

    # Bootstrap 90% confidence interval on the current probability
    # 500 resamples of the training data — captures parameter uncertainty
    print("Bootstrap confidence interval (500 iterations)...")
    df_train_ci = df[df["target"].notna()]
    rng = np.random.default_rng(42)
    boot_probs = []
    X_latest = scaler.transform(df[FEATURES].iloc[[-1]])   # current features, main scaler
    for _ in range(500):
        idx = rng.integers(0, len(df_train_ci), size=len(df_train_ci))
        X_b = df_train_ci.iloc[idx][FEATURES].values
        y_b = df_train_ci.iloc[idx]["target"].values
        sc_b = StandardScaler()
        m_b  = LogisticRegression(random_state=None, max_iter=1000)
        m_b.fit(sc_b.fit_transform(X_b), y_b)
        boot_probs.append(
            m_b.predict_proba(sc_b.transform(df[FEATURES].iloc[[-1]]))[0, 1] * 100
        )
    ci_lower = float(np.percentile(boot_probs, 5))
    ci_upper = float(np.percentile(boot_probs, 95))
    print(f"  90% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]")

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

    print("Computing state-level recession probabilities...")
    state_data = compute_state_data(data, df, coefs, intercept, scaler_params)

    # Data freshness: last observation date per raw FRED series
    data_freshness = {name: s.dropna().index[-1].strftime("%b %Y") for name, s in data.items()}

    # --- ALTERNATIVE MODEL (EPU-augmented) ---
    print("Training alternative model with EPU features (3m / 6m / 12m)...")
    scaler_alt,     model_alt     = train(df, horizon=3,  features=FEATURES_ALT)
    scaler_alt_6m,  model_alt_6m  = train(df, horizon=6,  features=FEATURES_ALT)
    scaler_alt_12m, model_alt_12m = train(df, horizon=12, features=FEATURES_ALT)
    print(f"  Alt C — 3m: {model_alt.C_[0]:.4f}  |  6m: {model_alt_6m.C_[0]:.4f}  |  12m: {model_alt_12m.C_[0]:.4f}")

    prob_series_alt = pd.Series(
        model_alt.predict_proba(scaler_alt.transform(df[FEATURES_ALT]))[:, 1] * 100,
        index=df.index, name="prob_alt",
    )
    prob_series_alt_6m = pd.Series(
        model_alt_6m.predict_proba(scaler_alt_6m.transform(df[FEATURES_ALT]))[:, 1] * 100,
        index=df.index, name="prob_alt_6m",
    )
    prob_series_alt_12m = pd.Series(
        model_alt_12m.predict_proba(scaler_alt_12m.transform(df[FEATURES_ALT]))[:, 1] * 100,
        index=df.index, name="prob_alt_12m",
    )

    coefs_alt     = dict(zip(FEATURES_ALT, model_alt.coef_[0]))
    intercept_alt = float(model_alt.intercept_[0])
    importances_alt = feature_importances(model_alt, FEATURES_ALT)
    scaler_params_alt = {feat: {"mean": float(scaler_alt.mean_[i]), "scale": float(scaler_alt.scale_[i])}
                         for i, feat in enumerate(FEATURES_ALT)}
    current_scaled_alt = scaler_alt.transform(df[FEATURES_ALT].iloc[[-1]])[0]
    contributions_alt  = {FEATURES_ALT[i]: float(current_scaled_alt[i] * model_alt.coef_[0][i])
                          for i in range(len(FEATURES_ALT))}

    _main_prob = float(prob_series.iloc[-1])
    print(f"  Alt model 3m: {prob_series_alt.iloc[-1]:.1f}%  "
          f"(vs main: {_main_prob:.1f}%,  Δ = {prob_series_alt.iloc[-1] - _main_prob:+.1f} pp)")

    cache = dict(
        prob_series=prob_series,
        prob_series_6m=prob_series_6m,
        prob_series_12m=prob_series_12m,
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
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        last_built=pd.Timestamp.now().strftime("%Y-%m-%d"),
        outcome_summary=outcome_summary,
        state_data=state_data,
        # Alternative model entries
        prob_series_alt=prob_series_alt,
        prob_series_alt_6m=prob_series_alt_6m,
        prob_series_alt_12m=prob_series_alt_12m,
        coefs_alt=coefs_alt,
        intercept_alt=intercept_alt,
        importances_alt=importances_alt,
        scaler_params_alt=scaler_params_alt,
        contributions_alt=contributions_alt,
    )

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

    prob = prob_series.iloc[-1]
    print(f"Cache saved to {CACHE_PATH}")
    print(f"  Dataset: {df.index[0].date()} to {df.index[-1].date()}  ({len(df)} rows)")
    print(f"  Recession probability — 3m: {prob:.1f}%  |  "
          f"6m: {prob_series_6m.iloc[-1]:.1f}%  |  "
          f"12m: {prob_series_12m.iloc[-1]:.1f}%")
    if prev_prob is not None:
        print(f"  Previous probability: {prev_prob:.1f}%  (change: {prob - prev_prob:+.1f} pp)")
    print(f"  Top analog: {analogs[0]['date']}  (recession followed: {analogs[0]['recession_12m']})")


if __name__ == "__main__":
    build()
