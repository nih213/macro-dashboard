import sys
import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model import FEATURES_ALT

st.set_page_config(page_title="Alternative Data Model", layout="wide")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_CACHE_PATH   = os.path.join(_PROJECT_ROOT, "data", "cache.pkl")

_FEATURE_LABELS = {
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
    "yield_momentum":    "Yield Curve Momentum",
    "stress_breadth":    "Stress Breadth (0–7)",
    "cpi_accel":         "CPI Inflation Acceleration (3m)",
    "epu_news_level":    "Policy Uncertainty (News-based EPU)",
    "epu_trade_level":   "Trade Policy Uncertainty",
}


@st.cache_data(show_spinner="Loading alternative model data…")
def load(mtime=0):
    if not os.path.exists(_CACHE_PATH):
        return None
    with open(_CACHE_PATH, "rb") as f:
        c = pickle.load(f)
    return c


_mtime = os.path.getmtime(_CACHE_PATH) if os.path.exists(_CACHE_PATH) else 0
cache = load(_mtime)

if cache is None or cache.get("prob_series_alt") is None:
    st.title("Alternative Data Model")
    st.warning(
        "No alternative model data found in the cache. "
        "Run `python src/build_cache.py` (or trigger the GitHub Action) to build it."
    )
    st.stop()

prob_series     = cache["prob_series"]
prob_series_alt = cache["prob_series_alt"]
prob_series_alt_6m  = cache.get("prob_series_alt_6m")
prob_series_alt_12m = cache.get("prob_series_alt_12m")
prob_series_6m  = cache.get("prob_series_6m")
prob_series_12m = cache.get("prob_series_12m")
recession       = cache["recession"]
df_history      = cache["df_history"]
latest          = cache["latest"]
coefs_alt       = cache.get("coefs_alt", {})
intercept_alt   = cache.get("intercept_alt", 0.0)
importances_alt = cache.get("importances_alt", {})
contributions_alt = cache.get("contributions_alt", {})
danger_threshold = cache["danger_threshold"]

main_prob  = float(prob_series.iloc[-1])
alt_prob   = float(prob_series_alt.iloc[-1])
delta      = alt_prob - main_prob
latest_date = prob_series.index[-1]

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("Alternative Data Model")
st.caption(
    "Same logistic regression framework as the main model, with two additional features: "
    "**news-based Economic Policy Uncertainty** (Baker, Bloom & Davis — back to 1900) and "
    "**Trade Policy Uncertainty** (back to 1985). "
    "This page answers: does adding policy uncertainty shift the recession probability, "
    "and by how much?"
)
st.divider()

# ── MODEL COMPARISON ──────────────────────────────────────────────────────────
st.subheader("Model Comparison — Current Reading")

def _prob_color(p):
    return "#22c55e" if p < 20 else "#eab308" if p < 50 else "#ef4444"

delta_color  = "#ef4444" if delta > 0 else "#22c55e" if delta < 0 else "#94a3b8"
delta_arrow  = "▲" if delta > 0 else "▼" if delta < 0 else "→"
delta_phrase = (
    f"Policy uncertainty is adding <strong>{abs(delta):.1f} pp</strong> to the recession probability."
    if delta > 1 else
    f"Policy uncertainty is subtracting <strong>{abs(delta):.1f} pp</strong> from the recession probability."
    if delta < -1 else
    "Policy uncertainty is having <strong>minimal effect</strong> on the recession probability."
)

cmp_cols = st.columns(3)
for col, (label, p, sublabel) in zip(cmp_cols, [
    ("Main Model",              main_prob, "15 macro indicators"),
    ("Alternative Model (EPU)", alt_prob,  "15 macro + 2 uncertainty"),
    ("Difference",              abs(delta), "EPU contribution"),
]):
    c = _prob_color(p) if label != "Difference" else delta_color
    val_str = f"{p:.1f}%" if label != "Difference" else f"{delta_arrow} {abs(delta):.1f} pp"
    with col:
        st.markdown(
            f"<div style='padding:18px 16px; border-radius:10px; background:#f8fafc; "
            f"border:1px solid #e2e8f0; text-align:center'>"
            f"<div style='font-size:11px; color:#94a3b8; text-transform:uppercase; "
            f"letter-spacing:0.05em; margin-bottom:6px'>{label}</div>"
            f"<div style='font-size:38px; font-weight:800; color:{c}; line-height:1.1'>"
            f"{val_str}</div>"
            f"<div style='font-size:11px; color:#94a3b8; margin-top:6px'>{sublabel}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown(
    f"<div style='margin-top:10px; padding:10px 14px; border-radius:8px; background:#f8fafc; "
    f"border-left:3px solid {delta_color}; font-size:13px; color:#475569; line-height:1.5'>"
    f"{delta_phrase}</div>",
    unsafe_allow_html=True,
)

# ── FORECAST PATH ─────────────────────────────────────────────────────────────
if prob_series_alt_6m is not None and prob_series_alt_12m is not None:
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:11px; color:#94a3b8; text-transform:uppercase; "
        "letter-spacing:0.05em; margin-bottom:8px'>Alternative Model — Forecast Path</div>",
        unsafe_allow_html=True,
    )

    alt_6m  = float(prob_series_alt_6m.iloc[-1])
    alt_12m = float(prob_series_alt_12m.iloc[-1])

    d_6_3   = alt_6m  - alt_prob
    d_12_6  = alt_12m - alt_6m

    def _arrow(d):
        if d > 5:  return "↗", "#ef4444"
        if d < -5: return "↘", "#22c55e"
        return "→", "#94a3b8"

    fp_cols = st.columns([6, 1, 6, 1, 6])
    for idx, (label, p, hdate) in enumerate([
        ("3 months out",  alt_prob, (latest_date + pd.DateOffset(months=3)).strftime("%b %Y")),
        ("6 months out",  alt_6m,   (latest_date + pd.DateOffset(months=6)).strftime("%b %Y")),
        ("12 months out", alt_12m,  (latest_date + pd.DateOffset(months=12)).strftime("%b %Y")),
    ]):
        c = _prob_color(p)
        with fp_cols[idx * 2]:
            st.markdown(
                f"<div style='padding:16px; border-radius:10px; background:#f8fafc; "
                f"border:1px solid #e2e8f0; text-align:center'>"
                f"<div style='font-size:11px; color:#94a3b8; text-transform:uppercase; "
                f"letter-spacing:0.05em; margin-bottom:4px'>{label}</div>"
                f"<div style='font-size:34px; font-weight:800; color:{c}'>{p:.1f}%</div>"
                f"<div style='font-size:11px; color:#94a3b8; margin-top:4px'>by {hdate}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        if idx < 2:
            sym, col = _arrow(d_6_3 if idx == 0 else d_12_6)
            diff = d_6_3 if idx == 0 else d_12_6
            with fp_cols[idx * 2 + 1]:
                st.markdown(
                    f"<div style='text-align:center; padding-top:28px'>"
                    f"<div style='font-size:20px; color:{col}'>{sym}</div>"
                    f"<div style='font-size:10px; color:#94a3b8'>{diff:+.1f} pp</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

st.divider()

# ── HISTORICAL COMPARISON CHART ───────────────────────────────────────────────
st.subheader("Historical Probability — Main vs Alternative Model")
st.caption(
    "Solid blue = main model (15 features). "
    "Dashed orange = alternative model (+EPU). "
    "The gap between the two lines shows how much policy uncertainty is shifting the probability at each point in history."
)

def _recession_periods(rec):
    periods, in_rec, start = [], False, None
    for date, val in rec.items():
        if val == 1 and not in_rec:
            in_rec, start = True, date
        elif val == 0 and in_rec:
            in_rec = False
            periods.append((start, date))
    if in_rec:
        periods.append((start, rec.index[-1]))
    return periods

comp_fig = go.Figure()
for i, (s, e) in enumerate(_recession_periods(recession)):
    comp_fig.add_vrect(
        x0=s, x1=e, fillcolor="rgba(180,180,180,0.3)", line_width=0,
        annotation_text="NBER recession" if i == 0 else "",
        annotation_position="top left",
        annotation_font_size=10, annotation_font_color="gray",
    )

comp_fig.add_trace(go.Scatter(
    x=prob_series_alt.index, y=prob_series_alt.values,
    mode="lines", name="Alternative (EPU)",
    line=dict(color="rgba(249,115,22,0.7)", width=1.5, dash="dash"),
    hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra>Alt model</extra>",
))
comp_fig.add_trace(go.Scatter(
    x=prob_series.index, y=prob_series.values,
    mode="lines", name="Main model",
    line=dict(color="#2563eb", width=2),
    fill="tozeroy", fillcolor="rgba(37,99,235,0.06)",
    hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra>Main model</extra>",
))
comp_fig.add_hline(y=danger_threshold, line_dash="dash",
                   line_color="rgba(249,115,22,0.5)",
                   annotation_text=f"Danger zone ({danger_threshold}%)",
                   annotation_position="right",
                   annotation_font_color="rgba(249,115,22,0.8)")
comp_fig.update_layout(
    height=400,
    yaxis=dict(title="Probability (%)", range=[0, 100], gridcolor="rgba(0,0,0,0.07)"),
    xaxis=dict(showgrid=False, range=[recession.index[0], recession.index[-1]]),
    hovermode="x unified", plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=0, r=80, t=10, b=0),
    legend=dict(orientation="h", y=-0.12),
)
st.plotly_chart(comp_fig, use_container_width=True)

st.divider()

# ── EPU SIGNAL READINGS ───────────────────────────────────────────────────────
st.subheader("Policy Uncertainty Signals")
st.caption(
    "Current readings for the two additional features in the alternative model. "
    "Orange = above historical average (elevated uncertainty). "
    "Click 'View history' to see the full time series."
)

epu_cols = st.columns(2)
for col, feat, label in zip(epu_cols,
    ["epu_news_level", "epu_trade_level"],
    ["Policy Uncertainty (News-based EPU)", "Trade Policy Uncertainty"],
):
    if feat not in df_history.columns or feat not in latest.index:
        continue
    val     = float(latest[feat])
    hist    = df_history[feat].dropna()
    mean    = float(hist.mean())
    pct     = float((hist <= val).mean() * 100)
    stress  = val > mean
    bc      = "#ef4444" if stress else "#22c55e"
    note    = f"Above avg ({mean:.0f})" if stress else f"Below avg ({mean:.0f})"

    with col:
        st.markdown(
            f"<div style='padding:18px 16px; border-radius:10px; background:#f8fafc; "
            f"border-left:4px solid {bc}'>"
            f"<div style='font-size:11px; color:#94a3b8; text-transform:uppercase; "
            f"letter-spacing:0.05em; margin-bottom:6px'>{label}</div>"
            f"<div style='font-size:34px; font-weight:800; color:{bc}'>{val:.0f}</div>"
            f"<div style='font-size:11px; color:{bc}; margin-top:4px'>{note}</div>"
            f"<div style='font-size:11px; color:#94a3b8; margin-top:4px'>"
            f"Higher than {pct:.0f}% of historical readings</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# EPU history charts
epu_chart_cols = st.columns(2)
for col, feat, label in zip(epu_chart_cols,
    ["epu_news_level", "epu_trade_level"],
    ["Policy Uncertainty (News-based EPU)", "Trade Policy Uncertainty"],
):
    if feat not in df_history.columns:
        continue
    series = df_history[feat].dropna()
    mean   = float(series.mean())

    fig = go.Figure()
    for i, (s, e) in enumerate(_recession_periods(recession)):
        fig.add_vrect(x0=s, x1=e, fillcolor="rgba(180,180,180,0.3)", line_width=0)
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode="lines", line=dict(color="#2563eb", width=1.5),
        hovertemplate="%{x|%b %Y}: %{y:.0f}<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(y=mean, line_dash="dash", line_color="rgba(249,115,22,0.6)",
                  annotation_text=f"Avg ({mean:.0f})", annotation_position="right",
                  annotation_font_color="rgba(249,115,22,0.8)")
    fig.add_trace(go.Scatter(
        x=[series.index[-1]], y=[series.iloc[-1]],
        mode="markers", marker=dict(color="#2563eb", size=7),
        hovertemplate=f"Latest: {series.iloc[-1]:.0f}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        height=220, title=dict(text=label, font=dict(size=13)),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0.07)"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=0, r=80, t=36, b=0),
    )
    with col:
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── COEFFICIENT CHART ─────────────────────────────────────────────────────────
st.subheader("Alternative Model — Feature Coefficients")
st.caption(
    "Standardised coefficients for all 17 features. "
    "Red = raises recession probability · Blue = lowers it. "
    "The two EPU features are highlighted."
)

sorted_coefs = sorted(coefs_alt.items(), key=lambda x: x[1])
labels  = [_FEATURE_LABELS.get(f, f) for f, _ in sorted_coefs]
values  = [v for _, v in sorted_coefs]
feats   = [f for f, _ in sorted_coefs]
colors  = []
for f, v in sorted_coefs:
    if f in ("epu_news_level", "epu_trade_level"):
        colors.append("#f97316" if v > 0 else "#8b5cf6")  # orange/purple to highlight EPU
    else:
        colors.append("#ef4444" if v > 0 else "#2563eb")

coef_fig = go.Figure(go.Bar(
    x=values, y=labels, orientation="h",
    marker_color=colors,
    hovertext=[f"β = {v:+.3f}" for v in values], hoverinfo="text",
    text=[f"{v:+.3f}" for v in values], textposition="outside",
    textfont=dict(size=11),
))
coef_fig.add_vline(x=0, line_color="rgba(0,0,0,0.2)", line_width=1)
coef_fig.update_layout(
    height=460,
    xaxis=dict(title="Standardised coefficient (β)", zeroline=False,
               gridcolor="rgba(0,0,0,0.07)"),
    yaxis=dict(showgrid=False),
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=0, r=70, t=10, b=0),
    showlegend=False,
    annotations=[
        dict(x=0.99, y=1.02, xref="paper", yref="paper", showarrow=False,
             text=f"Intercept β₀ = {intercept_alt:+.3f}",
             font=dict(size=11, color="#94a3b8"), xanchor="right"),
        dict(x=0.99, y=0.98, xref="paper", yref="paper", showarrow=False,
             text="Orange/purple = EPU features",
             font=dict(size=11, color="#f97316"), xanchor="right"),
    ],
)
st.plotly_chart(coef_fig, use_container_width=True)

st.divider()

# ── METHODOLOGY ───────────────────────────────────────────────────────────────
with st.expander("About the alternative model"):
    st.markdown("""
**Why add policy uncertainty?**

Standard macro models assume that the *level* of variables like the yield curve or credit spreads
captures all relevant information. But uncertainty about future policy — trade tariffs, executive
actions, monetary policy shifts — can depress investment and hiring *before* it shows up in hard
data, making it a useful leading indicator.

**USEPUNEWSINDXM — News-based EPU (Baker, Bloom & Davis 2016)**
Constructed by counting the frequency of articles in major US newspapers containing terms related
to "economy", "uncertainty", and "policy". Goes back to January 1900 using newspaper archives,
so no imputation is needed over the 1961–present training window. The 3-month rolling average is
used to smooth month-to-month noise.

**EPUTRADE — Trade Policy Uncertainty**
A categorical sub-index of EPU specifically tracking uncertainty about trade policy (tariffs,
import duties, trade agreements). Starts January 1985; pre-1985 months are filled with the
long-run series mean, which maps to approximately zero after standardisation — equivalent to
assuming "average" trade uncertainty for that era.

**Model spec**
Identical to the main model: logistic regression with L2 penalty strength selected via
`TimeSeriesSplit` cross-validation (`n_splits=5, gap=6`). The same 3m/6m/12m direct
multi-step forecasting structure is used.
    """)
