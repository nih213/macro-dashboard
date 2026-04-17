import sys
import os
import base64
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import precision_score, recall_score

sys.path.insert(0, os.path.dirname(__file__))
from fetch import fetch_all
from model import build_dataset, train, FEATURES, feature_importances, walk_forward_predict

st.set_page_config(page_title="US Recession Probability", layout="wide")

st.markdown("""
<style>
@media (max-width: 768px) {
    div[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        min-width: calc(50% - 8px) !important;
        flex: 1 1 calc(50% - 8px) !important;
    }
}
@media (max-width: 480px) {
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }
}
</style>
""", unsafe_allow_html=True)


import pickle

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_CACHE_PATH   = os.path.join(_PROJECT_ROOT, "data", "cache.pkl")


@st.cache_data(show_spinner="Loading dashboard data…")
def load(mtime=0):   # mtime as cache key: new file → cache miss → fresh load
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH, "rb") as f:
            c = pickle.load(f)
        return (c["prob_series"], c["oos_series"], c["recession"], c["latest"],
                c["credit_mean"], c["importances"], c["perf_df"], c["targets"],
                c["danger_threshold"], c["coefs"], c["intercept"], c["df_history"],
                c.get("contributions"), c.get("analogs"), c.get("nyfed_series"), c.get("prev_prob"),
                c.get("scaler_params"), c.get("data_freshness"),
                c.get("ci_lower"), c.get("ci_upper"), c.get("last_built"),
                c.get("outcome_summary"), c.get("state_data"))

    # Fallback: compute live (local dev without a cache file)
    st.warning("No cache found — computing live. Run `python src/build_cache.py` to pre-build.")
    data     = fetch_all()
    df       = build_dataset(data)
    scaler, model = train(df)

    X_scaled    = scaler.transform(df[FEATURES])
    proba       = model.predict_proba(X_scaled)[:, 1] * 100
    prob_series = pd.Series(proba, index=df.index, name="prob")

    oos_series  = walk_forward_predict(df)

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

    current_scaled = scaler.transform(df[FEATURES].iloc[[-1]])[0]
    contributions_fb = {FEATURES[i]: float(current_scaled[i] * model.coef_[0][i])
                        for i in range(len(FEATURES))}

    current_scaled = scaler.transform(df[FEATURES].iloc[[-1]])[0]
    scaler_params_fb = {FEATURES[i]: {"mean": float(scaler.mean_[i]), "scale": float(scaler.scale_[i])}
                        for i in range(len(FEATURES))}

    return (prob_series, oos_series, recession, latest, credit_mean, importances,
            perf_df, targets, danger_threshold, coefs, intercept, df,
            contributions_fb, None, None, None, scaler_params_fb, None, None, None, None, None, None)


def recession_periods(rec_series):
    periods, in_rec, start = [], False, None
    for date, val in rec_series.items():
        if val == 1 and not in_rec:
            in_rec, start = True, date
        elif val == 0 and in_rec:
            in_rec = False
            periods.append((start, date))
    if in_rec:
        periods.append((start, rec_series.index[-1]))
    return periods


def add_recession_shading(fig, recession, label_first=True):
    for i, (start, end) in enumerate(recession_periods(recession)):
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="rgba(180,180,180,0.35)", line_width=0,
            annotation_text="NBER recession" if (i == 0 and label_first) else "",
            annotation_position="top left",
            annotation_font_size=11, annotation_font_color="gray",
        )


def subscribe_email(email: str, token: str, repo: str) -> str:
    """Append email to data/subscribers.txt in the repo via GitHub API.
    Returns: 'success' | 'already_subscribed' | 'error'
    """
    url     = f"https://api.github.com/repos/{repo}/contents/subscribers.txt"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}

    r = requests.get(url, headers=headers, timeout=10)
    if r.status_code == 200:
        current = base64.b64decode(r.json()["content"]).decode()
        sha     = r.json()["sha"]
    elif r.status_code == 404:
        current, sha = "", None
    else:
        return "error"

    emails = [e.strip().lower() for e in current.splitlines() if e.strip()]
    if email in emails:
        return "already_subscribed"

    emails.append(email)
    payload = {
        "message": f"Subscribe: {email}",
        "content": base64.b64encode(("\n".join(emails) + "\n").encode()).decode(),
    }
    if sha:
        payload["sha"] = sha

    r = requests.put(url, json=payload, headers=headers, timeout=10)
    return "success" if r.status_code in (200, 201) else "error"


FEATURE_LABELS = {
    "yield_spread":      "Yield Spread (10Y–3M)",
    "credit_spread":     "Credit Spread (Baa–10Y)",
    "indpro_chg":        "Industrial Production Δ",
    "commodity_chg":     "Commodity Prices Δ",
    "commodity_ma_ratio":"Commodity vs. 12m Avg",
    "permits_chg":       "Building Permits Δ",
    "payrolls_chg":      "Nonfarm Payrolls Δ",
    "real_pi_chg":       "Personal Income (ex-transfers) Δ",
    "mfg_trade_chg":     "Mfg & Trade Sales Δ",
    "sentiment_chg":     "Consumer Sentiment Δ",
    "fedfunds_chg":      "Fed Funds Rate Δ",
    "real_fedfunds":     "Real Fed Funds Rate",
    "financial_stress":  "Financial Stress Index",
    "real_activity":     "Real Activity Composite",
    "yield_momentum":    "Yield Curve Momentum",
    "stress_breadth":    "Stress Breadth (0–7)",
    "vci_signal":        "VCI Signal (Zandi)",
    "cpi_accel":         "CPI Inflation Acceleration (3m)",
}

SIGNAL_THRESHOLDS = {
    "yield_spread":       (0,           "Inversion threshold (0 pp)"),
    "credit_spread":      (None,        ""),   # credit_mean set at runtime; handled in dialog
    "indpro_chg":         (0,           "Zero growth"),
    "commodity_chg":      (0,           "Zero growth"),
    "commodity_ma_ratio": (1,           "At 12-month trend"),
    "permits_chg":        (0,           "Zero growth"),
    "payrolls_chg":       (0,           "Zero growth"),
    "real_pi_chg":        (0,           "Zero growth"),
    "mfg_trade_chg":      (0,           "Zero growth"),
    "sentiment_chg":      (0,           "No change"),
    "fedfunds_chg":       (0,           "No change"),
    "real_fedfunds":      (2,           "Restrictive threshold (~2%)"),
    "financial_stress":   (0,           "Neutral"),
    "real_activity":      (0,           "No growth"),
    "yield_momentum":     (0,           "No change"),
    "stress_breadth":     (5,           "High stress (5+ signals)"),
    "vci_signal":         (1,           "Zandi threshold (1 pp)"),
    "cpi_accel":          (0,           "Inflation accelerating"),
}


@st.dialog("Signal History", width="large")
def signal_dialog(feature_key):
    if feature_key not in df_history.columns:
        st.warning("No historical data available for this signal.")
        return

    series = df_history[feature_key].dropna()
    title  = FEATURE_LABELS.get(feature_key, feature_key)

    thresh, thresh_label = SIGNAL_THRESHOLDS.get(feature_key, (None, ""))
    # credit_spread threshold is dynamic (historical mean)
    if feature_key == "credit_spread":
        thresh, thresh_label = credit_mean, f"Historical avg ({credit_mean:.2f} pp)"

    sf = go.Figure()
    add_recession_shading(sf, recession, label_first=True)
    sf.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode="lines", name=title,
        line=dict(color="#2563eb", width=2),
        hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>",
    ))
    if thresh is not None:
        sf.add_hline(y=thresh, line_dash="dash", line_color="rgba(220,50,50,0.6)",
                     annotation_text=thresh_label, annotation_position="right",
                     annotation_font_color="rgba(220,50,50,0.8)")
    sf.add_trace(go.Scatter(
        x=[series.index[-1]], y=[series.iloc[-1]],
        mode="markers", name="Current",
        marker=dict(color="#2563eb", size=8),
        hovertemplate=f"Latest: {series.iloc[-1]:.2f}<extra></extra>",
    ))
    sf.update_layout(
        height=380, title=dict(text=title, font=dict(size=14)),
        yaxis=dict(gridcolor="rgba(0,0,0,0.07)"),
        xaxis=dict(showgrid=False, range=[recession.index[0], recession.index[-1]]),
        hovermode="x unified", plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=0, r=80, t=40, b=0), showlegend=False,
    )
    st.plotly_chart(sf, use_container_width=True)


def signal_card(col, feature_key, label, value_str, stress: bool, importance: float, note=""):
    border_color = "#ef4444" if stress else "#22c55e"
    status_text  = "Stress" if stress else "Normal"
    with col:
        st.markdown(
            f"<div style='padding:14px 16px; border-radius:8px; background:#f8fafc;"
            f"border-left:4px solid {border_color}; height:100px'>"
            f"<div style='font-size:11px; color:#94a3b8; text-transform:uppercase;"
            f"letter-spacing:0.05em; margin-bottom:2px'>{label}</div>"
            f"<div style='font-size:21px; font-weight:600; color:#1e293b; line-height:1.2'>{value_str}</div>"
            f"<div style='display:flex; justify-content:space-between; margin-top:4px'>"
            f"<span style='font-size:11px; color:{border_color}'>{status_text}"
            f"{(' · ' + note) if note else ''}</span>"
            f"<span style='font-size:11px; color:#94a3b8'>{importance:.0f}% weight</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
        if st.button("View history", key=f"sig_{feature_key}", use_container_width=True):
            signal_dialog(feature_key)


_mtime = os.path.getmtime(_CACHE_PATH) if os.path.exists(_CACHE_PATH) else 0
prob_series, oos_series, recession, latest, credit_mean, importances, perf_df, oos_targets, danger_threshold, coefs, intercept, df_history, contributions, analogs, nyfed_series, prev_prob, scaler_params, data_freshness, ci_lower, ci_upper, last_built, outcome_summary, state_data = load(_mtime)

current_prob  = prob_series.iloc[-1]
latest_date   = prob_series.index[-1]
forecast_date = latest_date + pd.DateOffset(months=3)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("U.S. Recession Probability")
st.caption(
    f"Logistic regression · "
    f"Data as of {latest_date.strftime('%B %Y')} · "
    f"Probability of recession by {forecast_date.strftime('%B %Y')}"
)

# ── PROBABILITY HERO ──────────────────────────────────────────────────────────
prob_color = "#22c55e" if current_prob < 20 else "#eab308" if current_prob < 50 else "#ef4444"
risk_label = "Low risk" if current_prob < 20 else "Elevated risk" if current_prob < 50 else "High risk"

gauge_col, summary_col = st.columns([1, 2])

with gauge_col:
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number" + ("+delta" if prev_prob is not None else ""),
        value=current_prob,
        **({"delta": {"reference": prev_prob, "suffix": " pp",
                      "relative": False, "valueformat": ".1f"}} if prev_prob is not None else {}),
        number={"suffix": "%", "font": {"size": 52, "color": prob_color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8",
                     "tickvals": [0, 25, 50, 75, 100]},
            "bar": {"color": prob_color, "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 20],              "color": "rgba(34,197,94,0.1)"},
                {"range": [20, danger_threshold],"color": "rgba(234,179,8,0.1)"},
                {"range": [danger_threshold, 60],"color": "rgba(249,115,22,0.1)"},
                {"range": [60, 100],             "color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {
                "line": {"color": "rgba(249,115,22,0.8)", "width": 2},
                "thickness": 0.75,
                "value": danger_threshold,
            },
        },
        title={"text": f"Recession Probability by {forecast_date.strftime('%b %Y')}<br><span style='font-size:13px;color:#94a3b8'>{risk_label}</span>",
               "font": {"size": 14, "color": "#64748b"}},
    ))
    gauge_fig.update_layout(
        height=280, margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="white",
    )
    st.plotly_chart(gauge_fig, use_container_width=True)
    if ci_lower is not None and ci_upper is not None:
        st.markdown(
            f"<div style='text-align:center; font-size:13px; color:#64748b; margin-top:-16px'>"
            f"90% confidence interval: <strong>{ci_lower:.1f}% – {ci_upper:.1f}%</strong>"
            f"</div>",
            unsafe_allow_html=True,
        )

with summary_col:
    # 12-month probability trend sparkline with color-coded segments
    recent = prob_series.tail(13)
    spark = go.Figure()

    def _spark_color(v):
        return "#ef4444" if v >= 40 else "#eab308" if v >= 20 else "#22c55e"

    # One trace per segment so each can have its own color
    for i in range(len(recent) - 1):
        v1, v2 = float(recent.values[i]), float(recent.values[i + 1])
        spark.add_trace(go.Scatter(
            x=[recent.index[i], recent.index[i + 1]],
            y=[v1, v2],
            mode="lines",
            line=dict(color=_spark_color(max(v1, v2)), width=2.5),
            showlegend=False, hoverinfo="skip",
        ))
    # Markers with hover, colored per point
    spark.add_trace(go.Scatter(
        x=recent.index, y=recent.values,
        mode="markers",
        marker=dict(size=5, color=[_spark_color(v) for v in recent.values]),
        showlegend=False,
        hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>",
    ))
    spark.add_hline(y=danger_threshold, line_dash="dash",
                    line_color="rgba(249,115,22,0.5)",
                    annotation_text=f"Danger zone ({danger_threshold}%)",
                    annotation_position="right",
                    annotation_font_size=10,
                    annotation_font_color="rgba(249,115,22,0.8)")
    spark.update_layout(
        height=180,
        margin=dict(l=0, r=70, t=4, b=0),
        xaxis=dict(showgrid=False, tickformat="%b '%y", tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(range=[0, max(40, float(recent.max()) * 1.3)], showgrid=False,
                   ticksuffix="%", tickfont=dict(size=10)),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
    )
    st.markdown("<div style='font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:2px'>12-Month Trend</div>", unsafe_allow_html=True)
    st.plotly_chart(spark, use_container_width=True)

    # Auto-generated plain-English summary
    if contributions:
        change = current_prob - prev_prob if prev_prob is not None else 0
        if abs(change) < 0.5:
            trend_phrase = "remained broadly stable"
        elif change > 0:
            trend_phrase = f"increased {change:.1f} pp"
        else:
            trend_phrase = f"decreased {abs(change):.1f} pp"

        top_up   = max(contributions, key=lambda k: contributions[k])
        top_down = min(contributions, key=lambda k: contributions[k])
        if contributions[top_up] > 0:
            driver_phrase = f"{FEATURE_LABELS.get(top_up, top_up)} is the primary upward pressure"
        else:
            driver_phrase = f"All signals are currently below baseline, led by {FEATURE_LABELS.get(top_down, top_down).lower()}"

        if current_prob < 15:
            context_phrase = "Conditions remain broadly consistent with continued expansion."
        elif current_prob < danger_threshold:
            context_phrase = f"Some stress signals are visible, but the probability remains below the historical danger zone ({danger_threshold}%)."
        elif current_prob < 60:
            context_phrase = f"At {current_prob:.1f}%, the model is above its historical danger zone — conditions warrant close monitoring."
        else:
            context_phrase = "Multiple indicators are simultaneously stressed — historically a strong recession precursor."

        st.markdown(
            f"<div style='padding:16px; border-radius:8px; background:#f8fafc; border:1px solid #e2e8f0; margin-bottom:16px'>"
            f"<div style='font-size:12px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px'>Monthly Update</div>"
            f"<div style='font-size:14px; color:#334155; line-height:1.6'>"
            f"Since last month, the 3-month recession probability has <strong>{trend_phrase}</strong>. "
            f"{driver_phrase}. {context_phrase}"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    # Key drivers: top 3 signals by absolute contribution to current reading
    if contributions:
        top3 = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        st.markdown(
            "<div style='font-size:12px; color:#94a3b8; text-transform:uppercase; "
            "letter-spacing:0.05em; margin-bottom:8px'>Top Drivers Right Now</div>",
            unsafe_allow_html=True,
        )
        driver_cols = st.columns(3)
        for i, (feat, contrib) in enumerate(top3):
            arrow  = "▲" if contrib > 0 else "▼"
            color  = "#ef4444" if contrib > 0 else "#22c55e"
            label  = FEATURE_LABELS.get(feat, feat)
            with driver_cols[i]:
                st.markdown(
                    f"<div style='padding:10px 12px; border-radius:8px; background:#f8fafc; "
                    f"border-left:3px solid {color}; text-align:center'>"
                    f"<div style='font-size:11px; color:#94a3b8; margin-bottom:4px'>{label}</div>"
                    f"<div style='font-size:18px; font-weight:700; color:{color}'>"
                    f"{arrow} {abs(contrib):.2f}</div>"
                    f"<div style='font-size:10px; color:#94a3b8'>log-odds contribution</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

# ── SHARE ─────────────────────────────────────────────────────────────────────
share_text = (f"US recession probability: {current_prob:.1f}% by "
              f"{forecast_date.strftime('%B %Y')} — recession-chance.com")
share_col, _ = st.columns([3, 2])
with share_col:
    st.markdown(
        "<div style='font-size:11px; color:#94a3b8; text-transform:uppercase; "
        "letter-spacing:0.05em; margin-bottom:4px'>Share this reading</div>",
        unsafe_allow_html=True,
    )
    st.code(share_text, language=None)

# ── SCENARIO EXPLORER ─────────────────────────────────────────────────────────
st.divider()
st.subheader("Scenario Explorer")
st.caption(
    "How would recession probability change under different macro environments? "
    "Pick a named scenario or build your own."
)

if scaler_params:
    def _scenario_prob(overrides: dict) -> float:
        log_odds = intercept
        for feat in FEATURES:
            val = overrides.get(feat, float(latest[feat]))
            z   = (val - scaler_params[feat]["mean"]) / scaler_params[feat]["scale"]
            log_odds += coefs.get(feat, 0) * z
        return 1 / (1 + np.exp(-log_odds)) * 100

    _SCENARIOS = {
        "Boom": {
            "description": "Goldilocks: strong growth, low inflation",
            "overrides": {
                "yield_spread": 1.5, "credit_spread": 1.2, "indpro_chg": 4.0,
                "payrolls_chg": 2.5, "fedfunds_chg": -1.0, "real_fedfunds": 0.5,
                "cpi_accel": 0.0, "sentiment_chg": 12.0, "financial_stress": -0.3,
                "real_activity": 4.0, "stress_breadth": 0, "permits_chg": 8.0,
            },
        },
        "Soft Landing": {
            "description": "Fed tames inflation without triggering recession",
            "overrides": {
                "yield_spread": 0.5, "credit_spread": 1.8, "indpro_chg": 1.5,
                "payrolls_chg": 1.2, "fedfunds_chg": -1.5, "real_fedfunds": 1.0,
                "cpi_accel": -0.5, "sentiment_chg": 5.0, "financial_stress": 1.3,
                "real_activity": 1.5, "stress_breadth": 1, "permits_chg": 3.0,
            },
        },
        "Stagflation": {
            "description": "High inflation + stagnant growth, Fed stuck",
            "overrides": {
                "yield_spread": 0.0, "credit_spread": 2.5, "indpro_chg": -1.5,
                "payrolls_chg": 0.3, "fedfunds_chg": 3.0, "real_fedfunds": -1.5,
                "cpi_accel": 1.5, "sentiment_chg": -20.0, "financial_stress": 2.5,
                "real_activity": 0.2, "stress_breadth": 4, "permits_chg": -5.0,
            },
        },
        "Mild Recession": {
            "description": "Unemployment rises to ~5.5%, shallow contraction",
            "overrides": {
                "yield_spread": -0.5, "credit_spread": 2.8, "indpro_chg": -3.0,
                "payrolls_chg": -0.8, "fedfunds_chg": -2.0, "real_fedfunds": -0.5,
                "cpi_accel": -0.5, "sentiment_chg": -15.0, "financial_stress": 3.3,
                "real_activity": -3.3, "stress_breadth": 5, "permits_chg": -10.0,
            },
        },
        "Severe Recession": {
            "description": "2008-style crisis, unemployment peaks above 8%",
            "overrides": {
                "yield_spread": -1.2, "credit_spread": 4.5, "indpro_chg": -9.0,
                "payrolls_chg": -3.5, "fedfunds_chg": -4.0, "real_fedfunds": -2.0,
                "cpi_accel": -2.0, "sentiment_chg": -30.0, "financial_stress": 5.7,
                "real_activity": -9.7, "stress_breadth": 7, "permits_chg": -25.0,
            },
        },
    }

    sc_cols = st.columns(len(_SCENARIOS))
    for col, (sc_name, sc_data) in zip(sc_cols, _SCENARIOS.items()):
        sc_prob  = _scenario_prob(sc_data["overrides"])
        sc_delta = sc_prob - current_prob
        sc_color = "#22c55e" if sc_prob < 20 else "#eab308" if sc_prob < 50 else "#ef4444"
        d_color  = "#ef4444" if sc_delta > 0 else "#22c55e"
        with col:
            st.markdown(
                f"<div style='padding:14px; border-radius:8px; background:#f8fafc; "
                f"border:1px solid #e2e8f0; text-align:center'>"
                f"<div style='font-size:12px; font-weight:600; color:#1e293b; margin-bottom:4px'>{sc_name}</div>"
                f"<div style='font-size:28px; font-weight:800; color:{sc_color}; line-height:1.1'>{sc_prob:.0f}%</div>"
                f"<div style='font-size:12px; color:{d_color}; margin-bottom:6px'>"
                f"{'▲' if sc_delta > 0 else '▼'} {abs(sc_delta):.1f} pp vs current</div>"
                f"<div style='font-size:11px; color:#94a3b8'>{sc_data['description']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    with st.expander("Build your own scenario", expanded=False):
        st.caption("Adjust the five most impactful variables and see the model respond in real time.")
        byo_c1, byo_c2 = st.columns(2)
        byo_overrides = {}
        with byo_c1:
            byo_overrides["yield_spread"]  = st.slider(
                "Yield curve (10Y–3M spread, pp)", -3.0, 3.0, float(latest["yield_spread"]), 0.1)
            byo_overrides["cpi_accel"]     = st.slider(
                "Inflation acceleration (3m change in CPI YoY, pp)", -3.0, 3.0,
                float(latest["cpi_accel"]) if "cpi_accel" in latest.index else 0.0, 0.1)
            byo_overrides["payrolls_chg"]  = st.slider(
                "Payroll growth (YoY %)", -4.0, 4.0, float(latest["payrolls_chg"]), 0.1)
        with byo_c2:
            byo_overrides["fedfunds_chg"]  = st.slider(
                "Fed rate change (YoY pp)", -5.0, 5.0, float(latest["fedfunds_chg"]), 0.1)
            byo_overrides["credit_spread"] = st.slider(
                "Credit spread (Baa–10Y, pp)", 0.5, 6.0, float(latest["credit_spread"]), 0.1)

        byo_prob  = _scenario_prob(byo_overrides)
        byo_delta = byo_prob - current_prob
        byo_color = "#22c55e" if byo_prob < 20 else "#eab308" if byo_prob < 50 else "#ef4444"
        byo_d_col = "#ef4444" if byo_delta > 0 else "#22c55e"

        byo_r1, byo_r2, _ = st.columns([1, 1, 2])
        with byo_r1:
            st.markdown(
                f"<div style='padding:16px; border-radius:8px; background:#f8fafc; "
                f"border:1px solid #e2e8f0; text-align:center'>"
                f"<div style='font-size:11px; color:#94a3b8; text-transform:uppercase; margin-bottom:4px'>Your Scenario</div>"
                f"<div style='font-size:40px; font-weight:800; color:{byo_color}'>{byo_prob:.1f}%</div>"
                f"<div style='font-size:13px; color:{byo_d_col}'>{'▲' if byo_delta > 0 else '▼'} {abs(byo_delta):.1f} pp vs current</div>"
                f"</div>", unsafe_allow_html=True,
            )
        with byo_r2:
            st.markdown(
                f"<div style='padding:16px; border-radius:8px; background:#f8fafc; "
                f"border:1px solid #e2e8f0; text-align:center'>"
                f"<div style='font-size:11px; color:#94a3b8; text-transform:uppercase; margin-bottom:4px'>Current</div>"
                f"<div style='font-size:40px; font-weight:800; color:{prob_color}'>{current_prob:.1f}%</div>"
                f"<div style='font-size:13px; color:#94a3b8'>baseline</div>"
                f"</div>", unsafe_allow_html=True,
            )
        byo_share = (
            f"Scenario — yield curve: {byo_overrides['yield_spread']:+.1f}pp, "
            f"CPI accel: {byo_overrides['cpi_accel']:+.1f} pp, "
            f"payrolls: {byo_overrides['payrolls_chg']:+.1f}%, "
            f"Fed change: {byo_overrides['fedfunds_chg']:+.1f}pp → "
            f"recession probability = {byo_prob:.1f}% — recession-chance.com"
        )
        st.markdown("<div style='font-size:11px; color:#94a3b8; text-transform:uppercase; "
                    "letter-spacing:0.05em; margin-top:12px; margin-bottom:4px'>Share this scenario</div>",
                    unsafe_allow_html=True)
        st.code(byo_share, language=None)
else:
    st.info("Rebuild the cache to enable the scenario explorer.")

# ── WHAT THIS MEANS FOR YOU ───────────────────────────────────────────────────
if outcome_summary:
    st.divider()
    st.subheader("What does this probability mean in practice?")

    OUTCOME_BUCKETS = [(0,  3,  "Very Low (0–3%)"),
                       (3,  7,  "Low (3–7%)"),
                       (7,  10, "Guarded (7–10%)"),
                       (10, 20, "Moderate (10–20%)"),
                       (20, 40, "Elevated (20–40%)"),
                       (40, 100,"High (40%+)")]
    BUCKET_COLORS   = {"Very Low (0–3%)":    "#16a34a",
                       "Low (3–7%)":         "#22c55e",
                       "Guarded (7–10%)":    "#86efac",
                       "Moderate (10–20%)":  "#eab308",
                       "Elevated (20–40%)":  "#f97316",
                       "High (40%+)":        "#ef4444"}
    current_bucket  = next((lbl for lo, hi, lbl in OUTCOME_BUCKETS if lo <= current_prob < hi),
                           "High (40%+)")
    cb           = outcome_summary.get(current_bucket, {})
    bucket_color = BUCKET_COLORS[current_bucket]

    st.markdown(
        f"<div style='padding:12px 16px; border-radius:8px; border-left:4px solid {bucket_color}; "
        f"background:#f8fafc; margin-bottom:16px'>"
        f"<span style='font-size:13px; color:#64748b'>Current environment: </span>"
        f"<strong style='color:{bucket_color}'>{current_bucket}</strong>"
        f"<span style='font-size:12px; color:#94a3b8'> &nbsp;·&nbsp; "
        f"Based on {cb.get('n', 0)} similar historical months</span></div>",
        unsafe_allow_html=True,
    )
    st.caption("Average outcomes in similar past environments (DJIA returns, unemployment change, monthly payroll gains):")

    h_cols = st.columns(3)
    for col, h, hlabel in zip(h_cols, [3, 6, 12], ["3 months out", "6 months out", "12 months out"]):
        sv = cb.get(f"stocks_{h}m",   {}).get("mean")
        uv = cb.get(f"unrate_{h}m",   {}).get("mean")
        pv = cb.get(f"payrolls_{h}m", {}).get("mean")
        sc = "#22c55e" if sv is not None and sv > 0 else "#ef4444" if sv is not None else "#94a3b8"
        uc = "#ef4444" if uv is not None and uv > 0.05 else "#22c55e" if uv is not None else "#94a3b8"
        pc = "#22c55e" if pv is not None and pv > 0 else "#ef4444" if pv is not None else "#94a3b8"
        with col:
            st.markdown(
                f"<div style='padding:16px; border-radius:8px; background:#f8fafc; border:1px solid #e2e8f0'>"
                f"<div style='font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:12px'>{hlabel}</div>"
                f"<div style='margin-bottom:10px'>"
                f"  <div style='font-size:11px; color:#64748b'>Stock market (DJIA)</div>"
                f"  <div style='font-size:20px; font-weight:700; color:{sc}'>{f'{sv:+.1f}%' if sv is not None else '—'}</div>"
                f"</div>"
                f"<div style='margin-bottom:10px'>"
                f"  <div style='font-size:11px; color:#64748b'>Unemployment change</div>"
                f"  <div style='font-size:20px; font-weight:700; color:{uc}'>{f'{uv:+.2f} pp' if uv is not None else '—'}</div>"
                f"</div>"
                f"<div>"
                f"  <div style='font-size:11px; color:#64748b'>Monthly job gains</div>"
                f"  <div style='font-size:20px; font-weight:700; color:{pc}'>{f'{pv:+.0f}k/mo' if pv is not None else '—'}</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    # Cross-bucket comparison at 12-month horizon
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.caption("12-month outcomes across all risk levels (historical averages):")

    bucket_names = [lbl for _, _, lbl in OUTCOME_BUCKETS]
    metrics_cfg  = [
        ("Stock market",      "stocks_12m",   True,  "%"),
        ("Unemployment",      "unrate_12m",   False, " pp"),
        ("Monthly job gains", "payrolls_12m", True,  "k/mo"),
    ]
    th = "padding:8px 12px; font-size:11px; text-transform:uppercase; letter-spacing:0.05em; color:#94a3b8; border-bottom:2px solid #e2e8f0"
    td = "padding:8px 12px; font-size:14px; font-weight:600; text-align:center; border-bottom:1px solid #f1f5f9"
    tbl  = "<table style='width:100%; border-collapse:collapse'><tr>"
    tbl += f"<th style='{th}; text-align:left'>Metric</th>"
    for bname in bucket_names:
        bc     = BUCKET_COLORS[bname]
        is_cur = bname == current_bucket
        bg     = "background:#f8fafc" if is_cur else ""
        name_part, range_part = bname.split(" (") if " (" in bname else (bname, "")
        range_str = f"({range_part}" if range_part else ""
        tbl += (f"<th style='{th}; text-align:center; {bg}'>"
                f"<span style='color:{bc}'>{name_part}</span>"
                f"<br><span style='font-size:10px; color:#94a3b8'>{range_str}</span></th>")
    tbl += "</tr>"
    for mlabel, mkey, pos_good, unit in metrics_cfg:
        tbl += f"<tr><td style='{td}; text-align:left; color:#475569; font-weight:400'>{mlabel}</td>"
        for bname in bucket_names:
            val    = outcome_summary.get(bname, {}).get(mkey, {}).get("mean")
            is_cur = bname == current_bucket
            bg     = "background:#f8fafc" if is_cur else ""
            if val is not None:
                good  = val > 0 if pos_good else val < 0
                color = "#22c55e" if good else "#ef4444"
                tbl  += f"<td style='{td}; {bg}'><span style='color:{color}'>{val:+.1f}{unit}</span></td>"
            else:
                tbl += f"<td style='{td}; {bg}'>—</td>"
        tbl += "</tr>"
    tbl += "</table>"
    st.markdown(tbl, unsafe_allow_html=True)

# ── STATE MAP ────────────────────────────────────────────────────────────────
_STATE_NAMES = {
    "AL": "Alabama",       "AK": "Alaska",        "AZ": "Arizona",       "AR": "Arkansas",
    "CA": "California",    "CO": "Colorado",      "CT": "Connecticut",   "DE": "Delaware",
    "FL": "Florida",       "GA": "Georgia",       "HI": "Hawaii",        "ID": "Idaho",
    "IL": "Illinois",      "IN": "Indiana",       "IA": "Iowa",          "KS": "Kansas",
    "KY": "Kentucky",      "LA": "Louisiana",     "ME": "Maine",         "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan",      "MN": "Minnesota",     "MS": "Mississippi",
    "MO": "Missouri",      "MT": "Montana",       "NE": "Nebraska",      "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey",    "NM": "New Mexico",    "NY": "New York",
    "NC": "North Carolina","ND": "North Dakota",  "OH": "Ohio",          "OK": "Oklahoma",
    "OR": "Oregon",        "PA": "Pennsylvania",  "RI": "Rhode Island",  "SC": "South Carolina",
    "SD": "South Dakota",  "TN": "Tennessee",     "TX": "Texas",         "UT": "Utah",
    "VT": "Vermont",       "VA": "Virginia",      "WA": "Washington",    "WV": "West Virginia",
    "WI": "Wisconsin",     "WY": "Wyoming",
}

_state_probs   = (state_data or {}).get("probs", {})
_state_ur_data = (state_data or {}).get("ur_latest", {})

if _state_probs:
    st.divider()
    st.subheader("State-Level Recession Signals")
    st.caption(
        "Each state's recession probability using its own unemployment trend as the "
        "labor market input, holding all national indicators (yield curve, credit spreads, "
        "monetary policy, etc.) fixed at current readings."
    )

    _states = sorted(_state_probs.keys())
    _probs  = [_state_probs[s] for s in _states]
    _hover  = []
    for s in _states:
        ur = _state_ur_data.get(s, {})
        trend = ("▲ rising" if ur.get("chg", 0) > 0.1 else
                 "▼ falling" if ur.get("chg", 0) < -0.1 else "→ stable")
        _hover.append(
            f"<b>{_STATE_NAMES.get(s, s)}</b><br>"
            f"Recession probability: <b>{_state_probs[s]:.1f}%</b><br>"
            f"Unemployment rate: {ur.get('ur', '?'):.1f}%<br>"
            f"12-month change: {ur.get('chg', 0):+.2f} pp ({trend})"
        )

    _map_fig = go.Figure(go.Choropleth(
        locations=_states,
        z=_probs,
        locationmode="USA-states",
        colorscale=[
            [0.00, "#16a34a"],
            [0.10, "#4ade80"],
            [0.25, "#fde047"],
            [0.50, "#f97316"],
            [1.00, "#dc2626"],
        ],
        zmin=0, zmax=60,
        colorbar=dict(title="Recession<br>prob (%)", thickness=14, len=0.65, ticksuffix="%"),
        hoverinfo="text",
        text=_hover,
        marker_line_color="white",
        marker_line_width=0.5,
    ))
    _map_fig.update_layout(
        geo=dict(scope="usa", projection_type="albers usa",
                 showlakes=False, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(_map_fig, use_container_width=True)

    # Top 5 most stressed states
    _sorted_states = sorted(_state_probs.items(), key=lambda x: x[1], reverse=True)
    _top5 = _sorted_states[:5]
    _bot5 = _sorted_states[-5:]
    _col_hi, _col_lo = st.columns(2)
    with _col_hi:
        st.markdown("**Most stressed states**")
        for abbr, p in _top5:
            chg = _state_ur_data.get(abbr, {}).get("chg", 0)
            c = "#ef4444" if p >= 40 else "#f97316" if p >= 20 else "#eab308"
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; padding:4px 0; "
                f"border-bottom:1px solid #f1f5f9'>"
                f"<span style='color:#475569'>{_STATE_NAMES.get(abbr, abbr)}</span>"
                f"<span style='font-weight:700; color:{c}'>{p:.1f}% "
                f"<span style='font-size:11px; color:#94a3b8'>({chg:+.2f} pp UR)</span></span>"
                f"</div>", unsafe_allow_html=True)
    with _col_lo:
        st.markdown("**Least stressed states**")
        for abbr, p in reversed(_bot5):
            chg = _state_ur_data.get(abbr, {}).get("chg", 0)
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; padding:4px 0; "
                f"border-bottom:1px solid #f1f5f9'>"
                f"<span style='color:#475569'>{_STATE_NAMES.get(abbr, abbr)}</span>"
                f"<span style='font-weight:700; color:#22c55e'>{p:.1f}% "
                f"<span style='font-size:11px; color:#94a3b8'>({chg:+.2f} pp UR)</span></span>"
                f"</div>", unsafe_allow_html=True)

# ── INTRODUCTION & MODEL FORMULA ─────────────────────────────────────────────
st.markdown(
    """
    This dashboard estimates the probability that the U.S. economy will enter a recession
    within the next three months, using NBER business cycle dates as the target variable.
    The model is a logistic regression trained on monthly data from 1961 to present,
    combining leading indicators across four transmission channels: financial conditions
    (yield curve, credit spreads), real activity (industrial production, housing, commodities),
    labor markets (payrolls), and monetary policy (Federal Funds Rate).
    Probabilities are recomputed each time the data is refreshed.
    """
)

st.latex(r"""
P\!\left(\text{recession}_{t+3}\right)
= \frac{1}{1 + e^{-\left(\beta_0
    +\, \beta_1\,\text{YieldSpread}_t
    +\, \beta_2\,\text{CreditSpread}_t
    +\, \beta_3\,\Delta\text{IndustrialProd}_t
    +\, \beta_4\,\Delta\text{Commodity}_t
    +\, \cdots
    +\, \beta_{14}\,\text{StressBreadth}_t
\right)}}
""")
st.caption(
    "All 17 inputs are standardised (zero mean, unit variance) before fitting, "
    "so coefficients are directly comparable in magnitude. "
    "β parameters are estimated by maximum likelihood on the full 1961–present sample."
)

sorted_coefs = sorted(coefs.items(), key=lambda x: x[1])
labels  = [FEATURE_LABELS.get(f, f) for f, _ in sorted_coefs]
values  = [v for _, v in sorted_coefs]
colors  = ["#ef4444" if v > 0 else "#2563eb" for v in values]
hover   = [f"β = {v:+.3f}" for v in values]

coef_fig = go.Figure(go.Bar(
    x=values, y=labels, orientation="h",
    marker_color=colors, hovertext=hover, hoverinfo="text",
    text=[f"{v:+.3f}" for v in values], textposition="outside",
    textfont=dict(size=11),
))
coef_fig.add_vline(x=0, line_color="rgba(0,0,0,0.2)", line_width=1)
coef_fig.update_layout(
    height=420,
    xaxis=dict(title="Standardised coefficient (β)", zeroline=False, showgrid=True,
               gridcolor="rgba(0,0,0,0.07)"),
    yaxis=dict(showgrid=False),
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=0, r=60, t=10, b=0),
    showlegend=False,
    annotations=[dict(
        x=0.99, y=1.02, xref="paper", yref="paper", showarrow=False,
        text=f"Intercept β₀ = {intercept:+.3f}",
        font=dict(size=11, color="#94a3b8"), xanchor="right",
    )]
)
st.plotly_chart(coef_fig, use_container_width=True)
st.caption("Red = raises recession probability · Blue = lowers recession probability · Sorted by coefficient value")

st.divider()

# ── HISTORICAL CHART ──────────────────────────────────────────────────────────
st.subheader("Full Model History (1967 – Present)")
st.caption(
    "Recession probability since the model's start date. Grey shading = NBER recessions. "
    "Key events annotated. Dotted line = out-of-sample predictions trained only on past data."
)
fig = go.Figure()
add_recession_shading(fig, recession)

if nyfed_series is not None:
    fig.add_trace(go.Scatter(
        x=nyfed_series.index, y=nyfed_series.values,
        mode="lines", name="Yield-curve only",
        line=dict(color="rgba(16,185,129,0.5)", width=1.5, dash="dot"),
        hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra>Yield-curve only</extra>",
    ))
fig.add_trace(go.Scatter(
    x=oos_series.index, y=oos_series.values,
    mode="lines", name="OOS prediction (honest backtest)",
    line=dict(color="rgba(100,100,100,0.5)", width=1.5, dash="dot"),
    hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra>OOS</extra>",
))
fig.add_trace(go.Scatter(
    x=prob_series.index, y=prob_series.values,
    mode="lines", name="Model prediction",
    line=dict(color="#2563eb", width=2),
    fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
    hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>",
))
fig.add_hline(y=danger_threshold, line_dash="dash", line_color="rgba(249,115,22,0.7)",
              annotation_text=f"Danger zone ({danger_threshold}%)",
              annotation_position="right",
              annotation_font_color="rgba(249,115,22,0.9)")
fig.add_hline(y=50, line_dash="dash", line_color="rgba(220,50,50,0.5)",
              annotation_text="50%", annotation_position="right",
              annotation_font_color="rgba(220,50,50,0.7)")

# Key event annotations
_EVENTS = {
    pd.Timestamp("1973-11-01"): "Oil Crisis",
    pd.Timestamp("1980-01-01"): "Volcker",
    pd.Timestamp("2001-03-01"): "Dot-com",
    pd.Timestamp("2007-12-01"): "GFC",
    pd.Timestamp("2020-02-01"): "COVID-19",
}
for _ev_date, _ev_label in _EVENTS.items():
    if prob_series.index[0] <= _ev_date <= prob_series.index[-1]:
        fig.add_shape(
            type="line",
            x0=_ev_date.isoformat(), x1=_ev_date.isoformat(),
            y0=0, y1=1, yref="paper",
            line=dict(color="rgba(100,100,100,0.25)", dash="dot", width=1),
        )
        fig.add_annotation(
            x=_ev_date.isoformat(), y=0.98, yref="paper",
            text=_ev_label, showarrow=False,
            font=dict(size=9, color="rgba(100,100,100,0.65)"),
            xanchor="left", yanchor="top", textangle=-90,
        )

fig.update_layout(
    height=420,
    yaxis=dict(title="Probability (%)", range=[0, 100], gridcolor="rgba(0,0,0,0.07)"),
    xaxis=dict(showgrid=False, range=[recession.index[0], recession.index[-1]]),
    hovermode="x unified", plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=0, r=40, t=10, b=0),
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

# Custom legend below the chart
nyfed_legend = (
    "<span><span style='display:inline-block; width:28px; height:0px; "
    "border-top:2px dashed rgba(16,185,129,0.8); vertical-align:middle; margin-right:6px'></span>"
    "Yield-curve only model</span>"
) if nyfed_series is not None else ""

st.markdown(
    "<div style='display:flex; gap:28px; font-size:13px; color:#475569; margin-top:-8px; flex-wrap:wrap'>"
    "<span><span style='display:inline-block; width:28px; height:0px; "
    "border-top:2px dashed #64748b; vertical-align:middle; margin-right:6px'></span>"
    "Out-of-sample prediction</span>"
    "<span><span style='display:inline-block; width:28px; height:3px; background:#2563eb; "
    "vertical-align:middle; margin-right:6px'></span>"
    "Full model (in-sample)</span>"
    + nyfed_legend +
    "<span><span style='display:inline-block; width:16px; height:12px; "
    "background:rgba(180,180,180,0.5); vertical-align:middle; margin-right:6px'></span>"
    "NBER recession</span>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='margin-top:12px; padding:12px 16px; border-radius:8px; background:#f8fafc; "
    "border:1px solid #e2e8f0; font-size:13px; color:#475569'>"
    "<strong>Out-of-sample prediction (grey dotted).</strong> "
    "The solid blue line is trained on all available data — it has seen the entire history "
    "including past recessions. The grey dotted line is a stricter test: for each month, "
    "the model was re-trained using only data that existed <em>at that point in time</em>. "
    "Close agreement between the two indicates the model generalises well and is not overfitting."
    "</div>",
    unsafe_allow_html=True,
)

if nyfed_series is not None:
    # Compute how much the full model adds over the yield-curve-only benchmark right now
    current_nyfed = nyfed_series.iloc[-1]
    gap           = current_prob - current_nyfed
    gap_phrase    = (
        f"currently {gap:+.1f} pp <strong>above</strong> the yield-curve benchmark, "
        "meaning the additional indicators are adding meaningful recession signal beyond the spread alone."
        if gap > 2 else
        f"currently {gap:+.1f} pp <strong>below</strong> the yield-curve benchmark, "
        "meaning the broader indicator set is actually dampening the yield-curve's recession signal."
        if gap < -2 else
        "currently tracking <strong>close to the yield-curve benchmark</strong>, "
        "meaning the yield spread is the dominant driver and additional indicators are adding little."
    )
    st.markdown(
        "<div style='margin-top:8px; padding:12px 16px; border-radius:8px; background:#f0fdf4; "
        "border:1px solid #bbf7d0; font-size:13px; color:#475569'>"
        "<strong>Yield-curve only model (green dotted).</strong> "
        "This single-feature benchmark approximates the NY Fed's published probit model "
        "(Estrella &amp; Mishkin 1998), which uses only the 10Y–3M Treasury spread. "
        f"The full 17-feature model is {gap_phrase} "
        "Historically, the gap widens most sharply when credit markets, labour, or monetary "
        "policy diverge from what the yield curve alone would imply — for example, "
        "2006–07, when the curve was inverted but credit spreads were still benign, "
        "or 2022–23, when the curve inverted sharply but the labour market remained strong."
        "</div>",
        unsafe_allow_html=True,
    )

# Recession-by-recession performance table
with st.expander("Model accuracy at each recession", expanded=False):
    st.caption("How much warning did the model give before each NBER recession?")
    _rec_rows = []
    for _rs, _re in recession_periods(recession):
        if _rs < prob_series.index[0]:
            continue
        _lb = prob_series[
            (prob_series.index >= _rs - pd.DateOffset(months=18)) &
            (prob_series.index < _rs)
        ]
        if len(_lb) == 0:
            continue
        _peak     = _lb.max()
        _peak_dt  = _lb.idxmax()
        _above    = _lb[_lb >= danger_threshold]
        _warn     = round((_rs - _above.index[0]).days / 30.5) if len(_above) > 0 else None
        _duration = max(1, round((_re - _rs).days / 30.5))
        _rec_rows.append({
            "Recession":            f"{_rs.strftime('%b %Y')} – {_re.strftime('%b %Y')}",
            "Duration":             f"{_duration}m",
            "Peak prob before start": f"{_peak:.0f}%",
            "Peak reached":         _peak_dt.strftime("%b %Y"),
            f"Warning (>{danger_threshold}% threshold)": f"{_warn}m before" if _warn else "No signal",
        })
    if _rec_rows:
        st.dataframe(pd.DataFrame(_rec_rows), use_container_width=True, hide_index=True)

st.divider()

# ── WHAT-IF SCENARIO TOOL ─────────────────────────────────────────────────────
with st.expander("What-If Scenario Analysis", expanded=False):
    st.caption(
        "Adjust the five most influential indicators and see how the recession probability "
        "would change, holding all other signals at their current readings."
    )
    if scaler_params:
        top5 = sorted(FEATURES, key=lambda f: abs(coefs.get(f, 0)), reverse=True)[:5]
        sl_cols = st.columns(5)
        overrides = {}
        for i, feat in enumerate(top5):
            lo  = float(df_history[feat].quantile(0.05))
            hi  = float(df_history[feat].quantile(0.95))
            cur = float(max(lo, min(hi, latest[feat])))
            stp = round((hi - lo) / 100, 3) or 0.001
            with sl_cols[i]:
                overrides[feat] = st.slider(
                    FEATURE_LABELS.get(feat, feat),
                    min_value=lo, max_value=hi, value=cur, step=stp,
                    key=f"wi_{feat}",
                )

        log_odds = intercept
        for feat in FEATURES:
            val = overrides.get(feat, float(latest[feat]))
            z   = (val - scaler_params[feat]["mean"]) / scaler_params[feat]["scale"]
            log_odds += coefs.get(feat, 0) * z
        wi_prob  = 1 / (1 + np.exp(-log_odds)) * 100
        wi_color = "#22c55e" if wi_prob < 20 else "#eab308" if wi_prob < 50 else "#ef4444"
        delta    = wi_prob - current_prob

        wc1, wc2, _ = st.columns([1, 1, 2])
        with wc1:
            st.markdown(
                f"<div style='padding:16px; border-radius:8px; background:#f8fafc; "
                f"border:1px solid #e2e8f0; text-align:center'>"
                f"<div style='font-size:11px; color:#94a3b8; text-transform:uppercase; margin-bottom:4px'>Scenario</div>"
                f"<div style='font-size:40px; font-weight:800; color:{wi_color}'>{wi_prob:.1f}%</div>"
                f"<div style='font-size:13px; color:{'#ef4444' if delta > 0 else '#22c55e' if delta < 0 else '#94a3b8'}'>"
                f"{'▲' if delta > 0 else '▼' if delta < 0 else '—'} {abs(delta):.1f} pp vs current</div>"
                f"</div>", unsafe_allow_html=True,
            )
        with wc2:
            st.markdown(
                f"<div style='padding:16px; border-radius:8px; background:#f8fafc; "
                f"border:1px solid #e2e8f0; text-align:center'>"
                f"<div style='font-size:11px; color:#94a3b8; text-transform:uppercase; margin-bottom:4px'>Current</div>"
                f"<div style='font-size:40px; font-weight:800; color:{prob_color}'>{current_prob:.1f}%</div>"
                f"<div style='font-size:13px; color:#94a3b8'>baseline</div>"
                f"</div>", unsafe_allow_html=True,
            )
    else:
        st.info("Rebuild the cache to enable scenario analysis.")

st.divider()

# ── MODEL PERFORMANCE ────────────────────────────────────────────────────────
st.subheader("Historical Model Performance")

left, right = st.columns([1, 1])

with left:
    st.markdown("**Threshold analysis** (out-of-sample predictions)")
    st.caption("Precision = of months flagged, how many were recessions. Recall = of actual recession months, how many were caught.")
    st.dataframe(perf_df, use_container_width=True)

with right:
    st.markdown("**Predicted probability distribution**")
    st.caption("Where does the model sit during recessions vs normal times?")

    rec_probs = oos_series[oos_targets == 1].dropna()
    exp_probs = oos_series[oos_targets == 0].dropna()

    dist_fig = go.Figure()
    dist_fig.add_trace(go.Histogram(
        x=exp_probs, name="Expansion", nbinsx=20,
        marker_color="rgba(37,99,235,0.6)", opacity=0.8,
    ))
    dist_fig.add_trace(go.Histogram(
        x=rec_probs, name="Recession", nbinsx=20,
        marker_color="rgba(239,68,68,0.7)", opacity=0.8,
    ))
    dist_fig.update_layout(
        barmode="overlay", height=260,
        xaxis=dict(title="Predicted probability (%)", range=[0, 100]),
        yaxis=dict(title="Months"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(x=0.7, y=0.95),
    )
    st.plotly_chart(dist_fig, use_container_width=True)

    if len(rec_probs) > 0:
        st.caption(
            f"During recession months: median predicted probability = **{rec_probs.median():.0f}%**"
        )

# Calibration plot — full width below the two columns
st.markdown("**Calibration** — are the predicted probabilities reliable?")
st.caption("Each dot = one 10pp bin. Points on the diagonal = perfectly calibrated. Dot size = number of months in that bin.")

bin_edges = np.arange(0, 101, 10)
bin_mids, actuals, n_counts = [], [], []
for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
    mask = (oos_series >= lo) & (oos_series < hi)
    if mask.sum() >= 3:
        bin_mids.append(float((lo + hi) / 2))
        actuals.append(float(oos_targets[mask].mean() * 100))
        n_counts.append(int(mask.sum()))

cal_fig = go.Figure()
cal_fig.add_trace(go.Scatter(
    x=[0, 100], y=[0, 100], mode="lines",
    line=dict(dash="dash", color="rgba(0,0,0,0.2)", width=1),
    showlegend=False, hoverinfo="skip",
))
cal_fig.add_trace(go.Scatter(
    x=bin_mids, y=actuals, mode="markers+lines",
    marker=dict(size=[max(7, c // 3) for c in n_counts], color="#2563eb",
                line=dict(color="white", width=1)),
    line=dict(color="#2563eb", width=2),
    hovertemplate="Predicted: %{x:.0f}%<br>Actual: %{y:.1f}%<br>n = %{text}<extra></extra>",
    text=n_counts, showlegend=False,
))
cal_fig.update_layout(
    height=280,
    xaxis=dict(title="Predicted probability (%)", range=[0, 100], gridcolor="rgba(0,0,0,0.07)"),
    yaxis=dict(title="Actual recession frequency (%)", range=[0, 100], gridcolor="rgba(0,0,0,0.07)"),
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(cal_fig, use_container_width=True)

st.divider()

# ── FACTOR ATTRIBUTION ────────────────────────────────────────────────────────
if scaler_params:
    st.subheader("What Has Been Driving the Probability?")
    st.caption(
        "Log-odds contribution of each macro channel over the past 24 months. "
        "Bars above zero push the probability up; bars below push it down."
    )

    FEATURE_GROUPS = {
        "Yield curve":     ["yield_spread", "yield_momentum"],
        "Credit markets":  ["credit_spread", "financial_stress"],
        "Real activity":   ["indpro_chg", "commodity_chg", "commodity_ma_ratio", "permits_chg", "mfg_trade_chg", "real_activity"],
        "Labour":          ["payrolls_chg", "real_pi_chg", "stress_breadth"],
        "Consumer":        ["sentiment_chg"],
        "Inflation":       ["cpi_accel"],
        "Monetary policy": ["fedfunds_chg", "real_fedfunds"],
    }
    GROUP_COLORS = {
        "Yield curve":     "#2563eb",
        "Credit markets":  "#ef4444",
        "Real activity":   "#eab308",
        "Labour":          "#8b5cf6",
        "Consumer":        "#06b6d4",
        "Inflation":       "#ec4899",
        "Monetary policy": "#f97316",
    }

    recent = df_history[FEATURES].tail(24)
    attr_fig = go.Figure()
    for group_name, feats in FEATURE_GROUPS.items():
        contrib = pd.Series(0.0, index=recent.index)
        for feat in feats:
            if feat in recent.columns and feat in scaler_params:
                z = (recent[feat] - scaler_params[feat]["mean"]) / scaler_params[feat]["scale"]
                contrib += z * coefs.get(feat, 0)
        attr_fig.add_trace(go.Bar(
            x=recent.index, y=contrib.values,
            name=group_name, marker_color=GROUP_COLORS[group_name],
            hovertemplate="%{x|%b %Y}: %{y:+.2f}<extra>" + group_name + "</extra>",
        ))
    attr_fig.update_layout(
        barmode="relative", height=340,
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Log-odds contribution", gridcolor="rgba(0,0,0,0.07)", zeroline=True,
                   zerolinecolor="rgba(0,0,0,0.2)"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=-0.18, x=0),
        hovermode="x unified",
    )
    st.plotly_chart(attr_fig, use_container_width=True)

st.divider()

# ── HISTORICAL ANALOGS ────────────────────────────────────────────────────────
if analogs:
    st.subheader("Historical Analogs")
    st.caption(
        "The three past months whose macro environment — across all 17 indicators — "
        "most closely resembled today's. Lower distance = more similar."
    )
    acols = st.columns(3)
    for i, analog in enumerate(analogs):
        rec_color = "#ef4444" if analog["recession_12m"] else "#22c55e"
        rec_text  = "Recession followed within 12m" if analog["recession_12m"] else "No recession within 12m"
        with acols[i]:
            st.markdown(
                f"<div style='padding:16px; border-radius:8px; background:#f8fafc; "
                f"border:1px solid #e2e8f0; text-align:center; height:160px'>"
                f"<div style='font-size:18px; font-weight:700; color:#1e293b; margin-bottom:4px'>{analog['date']}</div>"
                f"<div style='font-size:13px; color:#64748b; margin-bottom:8px'>Similarity distance: {analog['distance']:.2f}</div>"
                f"<div style='font-size:22px; font-weight:700; color:#2563eb; margin-bottom:6px'>{analog['prob_then']:.1f}%</div>"
                f"<div style='font-size:11px; color:#94a3b8; margin-bottom:8px'>model probability then</div>"
                f"<div style='font-size:12px; font-weight:600; color:{rec_color}'>{rec_text}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

st.divider()

# ── SIGNAL DASHBOARD ─────────────────────────────────────────────────────────
st.subheader("Current Signal Readings")

# Sum importances for composite features into their parent signals where relevant
imp = importances

cols = st.columns(4)
signal_card(cols[0], "yield_spread",    "Yield Spread (10Y–3M)",  f"{latest['yield_spread']:+.2f} pp",
            stress=latest["yield_spread"] < 0, importance=imp.get("yield_spread", 0),
            note="Inverted" if latest["yield_spread"] < 0 else "Normal")
signal_card(cols[1], "credit_spread",   "Credit Spread (Baa–10Y)", f"{latest['credit_spread']:.2f} pp",
            stress=latest["credit_spread"] > credit_mean, importance=imp.get("credit_spread", 0),
            note=f"Avg {credit_mean:.2f} pp")
signal_card(cols[2], "indpro_chg",      "Industrial Production",   f"{latest['indpro_chg']:+.1f}% YoY",
            stress=latest["indpro_chg"] < 0, importance=imp.get("indpro_chg", 0))
signal_card(cols[3], "commodity_chg",   "Commodity Prices (PPI)",  f"{latest['commodity_chg']:+.1f}% YoY",
            stress=latest["commodity_chg"] < 0, importance=imp.get("commodity_chg", 0))

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
cols2 = st.columns(4)
signal_card(cols2[0], "permits_chg",    "Building Permits",       f"{latest['permits_chg']:+.1f}% YoY",
            stress=latest["permits_chg"] < 0, importance=imp.get("permits_chg", 0))
signal_card(cols2[1], "payrolls_chg",   "Nonfarm Payrolls",        f"{latest['payrolls_chg']:+.1f}% YoY",
            stress=latest["payrolls_chg"] < 0, importance=imp.get("payrolls_chg", 0))
signal_card(cols2[2], "sentiment_chg",  "Consumer Sentiment",      f"{latest['sentiment_chg']:+.1f} pts YoY",
            stress=latest["sentiment_chg"] < 0, importance=imp.get("sentiment_chg", 0))
signal_card(cols2[3], "vci_signal",     "VCI Signal (Zandi)",      f"{latest['vci_signal']:.2f} pp",
            stress=latest["vci_signal"] > 1.0, importance=0,
            note="Threshold 1.0 pp")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
cols3 = st.columns(4)
signal_card(cols3[0], "fedfunds_chg",   "Fed Funds Rate Δ",        f"{latest['fedfunds_chg']:+.2f} pp YoY",
            stress=latest["fedfunds_chg"] > 0, importance=imp.get("fedfunds_chg", 0),
            note="Tightening" if latest["fedfunds_chg"] > 0 else "Easing")
signal_card(cols3[1], "real_fedfunds",  "Real Fed Funds Rate",     f"{latest['real_fedfunds']:+.1f}%",
            stress=latest["real_fedfunds"] > 2.0, importance=imp.get("real_fedfunds", 0),
            note="Restrictive" if latest["real_fedfunds"] > 2.0 else "Neutral/Easy")
signal_card(cols3[2], "stress_breadth", "Stress Breadth",          f"{latest['stress_breadth']:.0f} / 7 signals",
            stress=latest["stress_breadth"] >= 5, importance=imp.get("stress_breadth", 0),
            note="Multiple signals" if latest["stress_breadth"] >= 5 else "Few signals")
signal_card(cols3[3], "financial_stress","Financial Stress Index",  f"{latest['financial_stress']:+.2f} pp",
            stress=latest["financial_stress"] > 0, importance=imp.get("financial_stress", 0),
            note="Tight" if latest["financial_stress"] > 0 else "Loose")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
cols4 = st.columns(4)
signal_card(cols4[0], "real_pi_chg",    "Personal Income (ex-transfers)", f"{latest['real_pi_chg']:+.1f}% YoY",
            stress=latest["real_pi_chg"] < 0, importance=imp.get("real_pi_chg", 0),
            note="CEI component")
signal_card(cols4[1], "mfg_trade_chg",  "Mfg & Trade Sales",   f"{latest['mfg_trade_chg']:+.1f}% YoY",
            stress=latest["mfg_trade_chg"] < 0, importance=imp.get("mfg_trade_chg", 0),
            note="CEI component")
signal_card(cols4[2], "cpi_accel",       "CPI Inflation",
            f"{latest['cpi_yoy']:+.1f}% YoY · accel {latest['cpi_accel']:+.1f} pp" if "cpi_accel" in latest.index else f"{latest['cpi_yoy']:+.1f}% YoY",
            stress="cpi_accel" in latest.index and latest["cpi_accel"] > 0,
            importance=imp.get("cpi_accel", 0),
            note="Rising" if "cpi_accel" in latest.index and latest["cpi_accel"] > 0 else "Falling/stable")

st.divider()

# ── DATA FRESHNESS ────────────────────────────────────────────────────────────
if data_freshness:
    from datetime import datetime as _dt
    _now = _dt.now()

    SERIES_LABELS = {
        "gs10": "10Y Treasury", "tb3ms": "3M T-Bill", "baa": "Moody's Baa",
        "indpro": "Industrial Prod.", "commodity": "PPI Commodities",
        "employment": "Employment", "population": "Population",
        "lfpr": "Labor Force Partic.", "permits": "Building Permits",
        "sp500": "DJIA", "payrolls": "Nonfarm Payrolls",
        "sentiment": "Consumer Sentiment", "fedfunds": "Fed Funds Rate",
        "cpi": "CPI", "recession": "NBER Recession",
    }

    with st.expander("Data freshness"):
        st.caption("Last observation date for each FRED series. Green = current month or last month; yellow = 2–3 months old; orange = older.")
        fcols = st.columns(5)
        for idx, (series, date_str) in enumerate(data_freshness.items()):
            try:
                dt = _dt.strptime(date_str, "%b %Y")
                months_old = (_now.year - dt.year) * 12 + (_now.month - dt.month)
                dot = "#22c55e" if months_old <= 1 else "#eab308" if months_old <= 3 else "#f97316"
            except Exception:
                dot = "#94a3b8"
            lbl = SERIES_LABELS.get(series, series)
            with fcols[idx % 5]:
                st.markdown(
                    f"<div style='padding:8px 10px; border-radius:6px; background:#f8fafc; "
                    f"border:1px solid #e2e8f0; margin-bottom:6px'>"
                    f"<div style='display:flex; align-items:center; gap:6px'>"
                    f"<span style='width:8px; height:8px; border-radius:50%; background:{dot}; "
                    f"display:inline-block; flex-shrink:0'></span>"
                    f"<span style='font-size:11px; color:#64748b; font-weight:500'>{lbl}</span>"
                    f"</div>"
                    f"<div style='font-size:12px; color:#94a3b8; margin-top:2px; padding-left:14px'>{date_str}</div>"
                    f"</div>", unsafe_allow_html=True,
                )

st.divider()

# ── METHODOLOGY ──────────────────────────────────────────────────────────────
with st.expander("About the model & improvements applied"):
    # Methodology disclaimer
    from datetime import datetime as _dt
    last_built_str = last_built or _dt.fromtimestamp(_mtime).strftime("%Y-%m-%d")
    st.markdown(
        f"<div style='padding:14px 16px; border-radius:8px; background:#fefce8; "
        f"border:1px solid #fde68a; margin-bottom:16px; font-size:13px; color:#78350f'>"
        f"<strong>Methodology disclaimer</strong><br>"
        f"This is an independent analytical tool, not an official forecast. "
        f"The model is a logistic regression, which assumes a <em>linear relationship</em> between "
        f"each standardised indicator and the log-odds of recession — non-linear interactions "
        f"are not captured. The 90% confidence interval reflects parameter uncertainty via "
        f"bootstrap resampling (500 draws) but does not account for model misspecification or "
        f"data revisions. Training window: January 1961 – present. "
        f"Recession dates: NBER Business Cycle Dating Committee. "
        f"Last model build: <strong>{last_built_str}</strong>."
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("""
**Model:** Logistic regression trained on monthly U.S. macroeconomic data from 1961 onwards.
Target variable: will the economy be in recession 3 months from now (NBER definition)?


1. **Federal Funds Rate features** — Two monetary policy signals added based on
   Bernanke & Blinder (1992) and Taylor (1993):
   - *FFR change (12m)*: captures active tightening cycles. Rising rates tighten
     credit and compress margins across the economy.
   - *Real FFR* = nominal FFR minus CPI inflation. Nominal rates alone are misleading —
     a 10% rate with 12% inflation is actually loose. Volcker's 1980 hike was devastating
     precisely because real rates turned sharply positive (~5–6%) for the first time in a decade.

2. **Out-of-sample walk-forward validation** (Stock & Watson 1999) — The dotted line on the
   chart shows *honest* predictions: for each month, the model was trained only on prior data.
   In-sample metrics are always too optimistic because the model has seen the answers.
   The threshold table uses OOS predictions only.

3. **10Y–3M yield spread** (Estrella & Mishkin 1998, 2,059 citations) — Used instead of the
   more common 10Y–2Y spread. The 3-month T-bill is more directly controlled by the Fed,
   making the 10Y–3M spread a cleaner signal of monetary tightness vs. long-run growth expectations.

4. **Moody's Baa–Treasury credit spread** — Replaces the high-yield spread (only available
   from 1996). Baa-rated bonds are investment grade but near the boundary — their spread
   over Treasuries is sensitive to credit cycle turning points while going back to 1953.

5. **Yield-curve only benchmark** — The green dotted line on the main chart is a logistic
   regression using only the 10Y–3M yield spread. This approximates the NY Fed's published
   probit model (Estrella & Mishkin 1998). Comparing it to the full 14-feature model shows
   how much information the additional indicators add beyond the yield curve alone.

6. **Historical analogs** — Euclidean distance across all 17 standardised features identifies
   the past months most similar to the current macro configuration. When those periods were
   followed by recessions, it provides additional context beyond the probability number alone.
    """)

st.divider()

# ── EMAIL ALERTS ──────────────────────────────────────────────────────────────
st.subheader("Get Email Alerts")
st.caption(
    f"Enter your email to receive an alert when the recession probability "
    f"crosses the historical danger zone ({danger_threshold}%)."
)

with st.form("email_subscribe", clear_on_submit=True):
    email_input = st.text_input("Email address", placeholder="you@example.com", label_visibility="collapsed")
    submitted   = st.form_submit_button("Subscribe", use_container_width=False)

if submitted:
    email_clean = email_input.strip().lower()
    if not email_clean or "@" not in email_clean or "." not in email_clean.split("@")[-1]:
        st.warning("Please enter a valid email address.")
    else:
        token = st.secrets.get("GITHUB_TOKEN", "")
        repo  = st.secrets.get("SUBSCRIBER_REPO", "")
        if not token or not repo:
            st.info("Alert subscriptions are not enabled on this deployment.")
        else:
            try:
                result = subscribe_email(email_clean, token, repo)
                if result == "success":
                    st.success("Subscribed. You'll receive an email when the recession probability crosses the alert threshold.")
                elif result == "already_subscribed":
                    st.info("That address is already subscribed.")
                else:
                    st.error("Something went wrong — please try again later.")
            except Exception:
                st.error("Could not connect. Please try again later.")
