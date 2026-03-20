from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parents[2]
for p in [str(ROOT), str(ROOT / "app")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from shared import load_css, tip_header, summary_banner, what_it_means, load_train_df, footer

st.set_page_config(page_title="EDA — Pastry Prediction", page_icon="🔍", layout="wide")
load_css("eda")

df = load_train_df()

_TOOLTIPS = {
    "scope": "**What this section does:** It shows how sales behave before we build any models, so we know which patterns the business should care about.",
    "sales_time": "**Why this chart matters:** It shows whether demand is steady or unpredictable and whether the business should expect clear ups and downs over time.",
    "sales_dist": "**Why this chart matters:** It shows whether most days look similar or whether there are occasional spikes that could affect planning.",
    "weekday": "**Why this chart matters:** It shows whether some days are consistently busier than others, which is useful for staffing and production.",
    "month": "**Why this chart matters:** It shows whether demand changes across the year, which helps with seasonal planning.",
    "heatmap": "**Why this chart matters:** It highlights which factors move more closely with sales, helping us focus on the inputs that are most useful.",
}

st.title("🔍 Exploratory Data Analysis")
summary_banner(
    "EDA — Understand Demand Before Building the Forecast",
    "This section focuses on what the sales data is telling us. "
    "Before choosing a model, we need to understand whether demand is stable, seasonal, or highly variable, because that affects how the business should plan production.",
    border="#7B1FA2",
)

if df.empty:
    st.warning("Missing `data/raw/train.csv`. Place the file in `data/raw/` to render the interactive EDA charts.")
else:
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="eda-step-badge">STEP 1 — UNDERSTAND THE TARGET</div>', unsafe_allow_html=True)
        tip_header("Describe the series first", _TOOLTIPS["scope"], "eda")
        st.caption("Start with the big picture: how sales behave over time.")
    with b:
        st.markdown('<div class="eda-step-badge">STEP 2 — CHECK CALENDAR EFFECTS</div>', unsafe_allow_html=True)
        tip_header("Look for repeating patterns", _TOOLTIPS["weekday"], "eda")
        st.caption("See whether demand is higher on certain days or months.")
    with c:
        st.markdown('<div class="eda-step-badge">STEP 3 — INSPECT DRIVERS</div>', unsafe_allow_html=True)
        tip_header("See which factors matter more", _TOOLTIPS["heatmap"], "eda")
        st.caption("This helps us decide what to keep in the model.")

    tab1, tab2, tab3 = st.tabs(["Trend & Distribution", "Calendar Effects", "Correlation Structure"])

    with tab1:
        tip_header("Figure 6.1 — Pastry Sales Over Time", _TOOLTIPS["sales_time"], "eda")
        ts = df.groupby("date", as_index=False)["sales"].mean()
        fig = px.line(ts, x="date", y="sales")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
        what_it_means(
            "Sales do not stay at the same level throughout the period. Demand rises and falls over time, which means production cannot rely on a fixed daily average. The forecast needs to adjust to changing demand patterns."
        )

        tip_header("Figure 6.2 — Distribution of Sales", _TOOLTIPS["sales_dist"], "eda")
        fig = px.histogram(df, x="sales", nbins=35)
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
        what_it_means(
            "Most days fall within a normal sales range, but there are also occasional high-demand days. This matters because the business needs to prepare for both everyday demand and busy periods without overproducing."
        )

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            tip_header("Figure 6.3 — Average Sales by Weekday", _TOOLTIPS["weekday"], "eda")
            weekday_df = df.groupby("day_of_week", as_index=False)["sales"].mean()
            fig = px.bar(weekday_df, x="day_of_week", y="sales")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            tip_header("Figure 6.4 — Average Sales by Month", _TOOLTIPS["month"], "eda")
            month_df = df.groupby("month", as_index=False)["sales"].mean()
            fig = px.bar(month_df, x="month", y="sales")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        what_it_means(
            "Demand is not evenly spread across the week or across the year. Some days and months are naturally stronger than others, which means production planning should reflect these repeating patterns rather than treating all periods the same."
        )

    with tab3:
        tip_header("Figure 6.5 — Correlation Heatmap", _TOOLTIPS["heatmap"], "eda")
        corr_cols = [
            c for c in [
                "temperature_max","temperature_min","temperature_mean","sunshine_sum","precipitation_sum","sales",
                "unsold","ordered","store_num","year","month","day","day_of_week","week_of_year","is_weekend",
                "is_state_holiday_num","is_school_holiday_num","is_special_day_num","is_holiday","is_holiday_binary",
                "precipitation_sum_boxcox","unsold_boxcox","ordered_boxcox","sales_lag_1","sales_lag_7",
                "sales_rolling_mean_7","sales_rolling_mean_14"
            ] if c in df.columns
        ]
        corr = df[corr_cols].corr(numeric_only=True)
        fig = px.imshow(corr, aspect="auto", color_continuous_scale="Viridis")
        fig.update_layout(height=850, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
        what_it_means(
            "Some factors are more closely linked to sales than others, especially recent sales patterns. This helps us focus on the inputs that are most useful for forecasting and avoid relying too heavily on weaker signals."
        )

footer("EDA page · dynamic charts · final-report-aligned narrative")
