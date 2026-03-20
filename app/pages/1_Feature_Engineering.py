from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st
import plotly.express as px
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in [str(ROOT), str(ROOT / "app"), str(ROOT / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from shared import load_css, tip_header, summary_banner, what_it_means, load_train_df, footer
from src.modeling import DEFAULT_FEATURES

st.set_page_config(page_title="Feature Engineering — Pastry Prediction", page_icon="🛠️", layout="wide")
load_css("fe")

df = load_train_df()

_TOOLTIPS = {
    "build": "**What this section does:** It shows how raw data is turned into useful signals that make the forecast more accurate.",
    "lag": "**Why this chart matters:** It shows whether recent sales help explain what happens next.",
    "rolling": "**Why this chart matters:** It shows how smoothing the series helps reveal the underlying demand pattern.",
    "feature_set": "**Why this list matters:** It shows which inputs the final model actually uses to make decisions.",
}

st.title("🛠️ Feature Engineering & Validation")
summary_banner(
    "Feature Layer — Turn Patterns Into Better Forecasts",
    "This section explains how the model moves beyond raw data. "
    "Instead of treating each day in isolation, we add features that reflect recent demand patterns and recurring calendar behavior.",
    border="#00A389",
)

if df.empty:
    st.warning("Missing `data/raw/train.csv`. Place the file in `data/raw/` to render the interactive feature-engineering charts.")
else:
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="fe-step-badge">STEP 1 — CREATE BETTER INPUTS</div>', unsafe_allow_html=True)
        tip_header("Engineer time-aware features", _TOOLTIPS["build"], "fe")
        st.caption("Build features that reflect how demand really behaves.")
    with b:
        st.markdown('<div class="fe-step-badge">STEP 2 — CHECK THEY MAKE SENSE</div>', unsafe_allow_html=True)
        tip_header("Validate the feature behavior", _TOOLTIPS["lag"], "fe")
        st.caption("A feature should be understandable before it is trusted.")
    with c:
        st.markdown('<div class="fe-step-badge">STEP 3 — PASS THEM TO MODELING</div>', unsafe_allow_html=True)
        tip_header("Use the final feature set", _TOOLTIPS["feature_set"], "fe")
        st.caption("The final list is transparent and easy to review.")

    tab1, tab2, tab3 = st.tabs(["Lag Features", "Rolling Features", "Final Feature Set"])

    with tab1:
        tip_header("Figure 8.1 — Sales vs Lag (t-1)", _TOOLTIPS["lag"], "fe")
        lag_df = df.dropna(subset=["sales_lag_1", "sales"]).copy()
        fig = px.scatter(lag_df, x="sales_lag_1", y="sales", opacity=0.3)
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
        what_it_means(
            "Sales from the previous day are a strong clue for what may happen next. This helps the forecast respond to short-term changes in demand more quickly."
        )

    with tab2:
        tip_header("Figure 8.2 — Sales vs Rolling Mean (7 Days)", _TOOLTIPS["rolling"], "fe")
        roll_df = df.dropna(subset=["sales_rolling_mean_7"]).groupby("date", as_index=False)[["sales", "sales_rolling_mean_7"]].mean()
        fig = px.line(roll_df, x="date", y=["sales", "sales_rolling_mean_7"])
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        what_it_means(
            "The rolling average makes the overall demand pattern easier to see by reducing daily noise. This helps the forecast stay more stable and avoids reacting too strongly to one unusual day."
        )

    with tab3:
        tip_header("Final Feature Set Used in Modeling", _TOOLTIPS["feature_set"], "fe")
        st.dataframe(pd.DataFrame({"Feature": DEFAULT_FEATURES}), use_container_width=True, hide_index=True)
        what_it_means(
            "The final model uses a mix of business context and recent sales behavior. Together, these inputs help produce a forecast that is more practical for day-to-day planning."
        )

footer("Feature Engineering page · dynamic charts · readable explanations")
