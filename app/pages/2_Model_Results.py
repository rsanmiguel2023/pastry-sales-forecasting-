from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parents[2]
for p in [str(ROOT), str(ROOT / "app")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from shared import load_css, tip_header, summary_banner, what_it_means, safe_csv, footer

st.set_page_config(page_title="Model Results — Pastry Prediction", page_icon="🤖", layout="wide")
load_css("mr")

results_df = safe_csv("time_based_model_comparison.csv")
pred_df = safe_csv("model_predictions.csv")
best = results_df.iloc[0] if not results_df.empty else None
best_model = best["model"] if best is not None else None

_TOOLTIPS = {
    "champion": "**Champion Model:** This is the model that performed best on the holdout period and is the strongest option for forecasting future sales.",
    "mse": "**Best Test MSE:** This measures how far predictions were from actual sales. Lower is better.",
    "r2": "**Best Test R²:** This shows how well the model captured the overall sales pattern. Higher is better.",
    "mse_plot": "**Why this chart matters:** It compares prediction error across all models so we can see which one was most accurate.",
    "r2_plot": "**Why this chart matters:** It compares how well each model explained the movement in sales.",
    "pred_actual": "**Why this chart matters:** It shows how closely predicted sales matched actual sales.",
    "residual": "**Why this chart matters:** It shows whether the prediction errors stayed balanced or whether the model regularly missed in one direction.",
}

st.title("🤖 Model Results & Diagnostics")
summary_banner(
    "Model Evaluation — Which Forecast Performed Best?",
    "This section compares the candidate models and then checks whether the top model behaves in a trustworthy way. "
    "The goal is not only to find the best score, but to choose a result that the business can rely on.",
    border="#7986CB",
)

if results_df.empty:
    st.warning("Missing `reports/time_based_model_comparison.csv` or `reports/model_predictions.csv`.")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Champion Model", best["model"], help=_TOOLTIPS["champion"])
    c2.metric("Best Test MSE", f"{best['test_mse']:.4f}", help=_TOOLTIPS["mse"])
    c3.metric("Best Test R²", f"{best['test_r2']:.4f}", help=_TOOLTIPS["r2"])
    c4.metric("Features Used", str(int(best["n_features"])), help="Number of predictors passed into the final trained model.")

    tab1, tab2, tab3 = st.tabs(["Overview", "Model Comparison", "Diagnostics"])

    with tab1:
        a, b, c = st.columns(3)
        with a:
            st.markdown('<div class="mr-step-badge">STEP 1 — TEST ON FUTURE-LIKE DATA</div>', unsafe_allow_html=True)
            tip_header("Use a realistic validation setup", "**Why this matters:** Forecasting should be tested on later dates, not mixed dates, so the result feels closer to real business use.", "mr")
            st.caption("The evaluation setup is designed to be fair.")
        with b:
            st.markdown('<div class="mr-step-badge">STEP 2 — CHOOSE THE BEST MODEL</div>', unsafe_allow_html=True)
            tip_header("Rank models by holdout performance", "**Why this matters:** The best model is the one that stays closest to actual sales on unseen later periods.", "mr")
            st.caption(f"{best_model} currently ranks first.")
        with c:
            st.markdown('<div class="mr-step-badge">STEP 3 — CHECK IF IT IS TRUSTWORTHY</div>', unsafe_allow_html=True)
            tip_header("Review the diagnostic charts", "**Why this matters:** Scores alone are not enough. We also want to see whether the predictions and errors behave sensibly.", "mr")
            st.caption("The diagnostics help turn scores into confidence.")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        what_it_means(
            "This section shows which model gave the most reliable forecast. We first compare overall performance, then check whether the top model behaves in a way that is trustworthy enough to support planning decisions."
        )

    with tab2:
        x1, x2 = st.columns(2)
        with x1:
            tip_header("Figure 9.1 — Model Comparison (Test MSE)", _TOOLTIPS["mse_plot"], "mr")
            order = results_df.sort_values("test_mse")
            fig = px.bar(order, x="model", y="test_mse")
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with x2:
            tip_header("Figure 9.2 — Model Comparison (Test R²)", _TOOLTIPS["r2_plot"], "mr")
            order = results_df.sort_values("test_r2", ascending=False)
            fig = px.bar(order, x="model", y="test_r2")
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        what_it_means(
            "The stronger models produced more accurate forecasts than the simpler ones. This suggests pastry demand is influenced by patterns that are too complex for a basic linear approach."
        )

    with tab3:
        best_pred = pred_df[pred_df["model"] == best_model].copy() if not pred_df.empty else pd.DataFrame()
        y1, y2 = st.columns(2)
        with y1:
            tip_header("Figure 9.3 — Predicted vs Actual", _TOOLTIPS["pred_actual"], "mr")
            fig = px.scatter(best_pred, x="actual", y="predicted", opacity=0.5)
            fig.update_layout(height=500, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with y2:
            tip_header("Figure 9.4 — Residual Plot", _TOOLTIPS["residual"], "mr")
            fig = px.scatter(best_pred, x="predicted", y="residual", opacity=0.5)
            fig.add_hline(y=0, line_dash="dash")
            fig.update_layout(height=500, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        what_it_means(
            "The top model’s predictions are generally close to actual sales, and its errors stay reasonably balanced. This gives more confidence that the forecast can support real production planning."
        )
        st.subheader("Prediction Sample")
        st.dataframe(best_pred.head(50), use_container_width=True, hide_index=True)

footer("Model Results page · dynamic comparison and diagnostics · business-facing explanations")
