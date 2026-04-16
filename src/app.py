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


import pickle

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_CACHE_PATH   = os.path.join(_PROJECT_ROOT, "data", "cache.pkl")


@st.cache_data(show_spinner="Loading dashboard data…")
def load():
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH, "rb") as f:
            c = pickle.load(f)
        return (c["prob_series"], c["oos_series"], c["recession"], c["latest"],
                c["credit_mean"], c["importances"], c["perf_df"], c["targets"],
                c["danger_threshold"], c["coefs"], c["intercept"], c["df_history"],
                c.get("contributions"), c.get("analogs"), c.get("nyfed_series"), c.get("prev_prob"))

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

    return (prob_series, oos_series, recession, latest, credit_mean, importances,
            perf_df, targets, danger_threshold, coefs, intercept, df,
            contributions_fb, None, None, None)


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
    url     = f"https://api.github.com/repos/{repo}/contents/data/subscribers.txt"
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
    "sentiment_chg":     "Consumer Sentiment Δ",
    "fedfunds_chg":      "Fed Funds Rate Δ",
    "real_fedfunds":     "Real Fed Funds Rate",
    "financial_stress":  "Financial Stress Index",
    "real_activity":     "Real Activity Composite",
    "yield_momentum":    "Yield Curve Momentum",
    "stress_breadth":    "Stress Breadth (0–7)",
    "vci_signal":        "VCI Signal (Zandi)",
}

SIGNAL_THRESHOLDS = {
    "yield_spread":       (0,           "Inversion threshold (0 pp)"),
    "credit_spread":      (None,        ""),   # credit_mean set at runtime; handled in dialog
    "indpro_chg":         (0,           "Zero growth"),
    "commodity_chg":      (0,           "Zero growth"),
    "commodity_ma_ratio": (1,           "At 12-month trend"),
    "permits_chg":        (0,           "Zero growth"),
    "payrolls_chg":       (0,           "Zero growth"),
    "sentiment_chg":      (0,           "No change"),
    "fedfunds_chg":       (0,           "No change"),
    "real_fedfunds":      (2,           "Restrictive threshold (~2%)"),
    "financial_stress":   (0,           "Neutral"),
    "real_activity":      (0,           "No growth"),
    "yield_momentum":     (0,           "No change"),
    "stress_breadth":     (5,           "High stress (5+ signals)"),
    "vci_signal":         (1,           "Zandi threshold (1 pp)"),
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


prob_series, oos_series, recession, latest, credit_mean, importances, perf_df, oos_targets, danger_threshold, coefs, intercept, df_history, contributions, analogs, nyfed_series, prev_prob = load()

current_prob = prob_series.iloc[-1]
latest_date  = prob_series.index[-1]

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("U.S. Recession Probability")
st.caption(
    f"3-month ahead forecast · Logistic regression · "
    f"Data as of {latest_date.strftime('%B %Y')}"
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
        title={"text": f"3-Month Recession Probability<br><span style='font-size:13px;color:#94a3b8'>{risk_label}</span>",
               "font": {"size": 14, "color": "#64748b"}},
    ))
    gauge_fig.update_layout(
        height=280, margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="white",
    )
    st.plotly_chart(gauge_fig, use_container_width=True)

with summary_col:
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
    "All 14 inputs are standardised (zero mean, unit variance) before fitting, "
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
        f"The full 14-feature model is {gap_phrase} "
        "Historically, the gap widens most sharply when credit markets, labour, or monetary "
        "policy diverge from what the yield curve alone would imply — for example, "
        "2006–07, when the curve was inverted but credit spreads were still benign, "
        "or 2022–23, when the curve inverted sharply but the labour market remained strong."
        "</div>",
        unsafe_allow_html=True,
    )

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
            f"During recession months: median predicted probability = **{rec_probs.median():.0f}%** · "
        )

st.divider()

# ── HISTORICAL ANALOGS ────────────────────────────────────────────────────────
if analogs:
    st.subheader("Historical Analogs")
    st.caption(
        "The three past months whose macro environment — across all 14 indicators — "
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

st.divider()

# ── METHODOLOGY ──────────────────────────────────────────────────────────────
with st.expander("About the model & improvements applied"):
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

6. **Historical analogs** — Euclidean distance across all 14 standardised features identifies
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
        repo  = st.secrets.get("GITHUB_REPO", "")
        if not token or not repo:
            st.info("Alert subscriptions are not enabled on this deployment.")
        else:
            try:
                result = subscribe_email(email_clean, token, repo)
                if result == "success":
                    st.success("Subscribed. You'll receive an email when the probability crosses the danger zone.")
                elif result == "already_subscribed":
                    st.info("That address is already subscribed.")
                else:
                    st.error("Something went wrong — please try again later.")
            except Exception:
                st.error("Could not connect. Please try again later.")
