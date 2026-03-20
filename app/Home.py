from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
for p in [str(ROOT), str(ROOT / "src"), str(ROOT / "app")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from shared import load_css, tip_header, summary_banner, safe_csv, footer

st.set_page_config(page_title="Pastry Prediction Analytics", page_icon="🥐", layout="wide", initial_sidebar_state="expanded")
load_css("home")

results_df = safe_csv("time_based_model_comparison.csv")
best = results_df.iloc[0] if not results_df.empty else None

_TOOLTIPS = {
    "champion_model": "**Champion Model:** This is the model that performed best on the holdout period, so it is the model we would trust most for forecasting future demand.",
    "best_test_mse": "**Best Test MSE:** This measures prediction error. Lower values mean the forecast stayed closer to actual sales.",
    "best_test_r2": "**Best Test R²:** This shows how well the model captured the overall movement in sales. Higher values mean a better fit.",
    "features_used": "**Features Used:** This is the number of inputs the final model used, such as calendar effects, weather variables, and recent sales patterns.",
    "overview": "**Project goal:** Predict daily pastry demand so the business can prepare the right amount of product and reduce waste.",
    "validation": "**Why this matters:** We test the models on later dates, not mixed dates, so the results better reflect how the model would perform in real planning.",
    "dashboard": "**How to read this dashboard:** It follows the same story as the final report: understand the data, explain the key patterns, compare the models, and judge whether the top result is reliable.",
}

st.title("🥐 Pastry Prediction Analytics Dashboard")
st.markdown(
    """
**Forecast daily pastry sales to improve production planning and reduce waste**

*Recruiter-ready forecasting repository · time-based validation · interactive Streamlit presentation*
"""
)
st.divider()

if best is not None:
    summary_banner(
        "Project Overview — Forecasting Framework & Best Result",
        f"<strong style='color:#ffffff;'>{best['model']}</strong> is the current top model with "
        f"<strong style='color:#ffffff;'>Test MSE = {best['test_mse']:.4f}</strong> and "
        f"<strong style='color:#ffffff;'>Test R² = {best['test_r2']:.4f}</strong>. "
        "The workflow is designed around the final report: first understand demand patterns, then build better features, and finally choose the model that gives the most reliable forecast.",
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Champion Model", best["model"] if best is not None else "N/A", help=_TOOLTIPS["champion_model"])
c2.metric("Best Test MSE", f"{best['test_mse']:.4f}" if best is not None else "N/A", help=_TOOLTIPS["best_test_mse"])
c3.metric("Best Test R²", f"{best['test_r2']:.4f}" if best is not None else "N/A", help=_TOOLTIPS["best_test_r2"])
c4.metric("Features Used", str(int(best["n_features"])) if best is not None else "N/A", help=_TOOLTIPS["features_used"])

st.divider()
tab1, tab2, tab3 = st.tabs(["Overview", "Repository Structure", "How to Run"])

with tab1:
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="home-step-badge">STEP 1 — DEFINE THE PROBLEM</div>', unsafe_allow_html=True)
        tip_header("Forecast pastry demand", _TOOLTIPS["overview"], "home")
        st.caption("The business question is simple: how much should we prepare each day?")
        st.markdown(
            """
- Target: `sales`
- Grain: one row per store per date
- Goal: improve planning and reduce waste
- Setting: future demand forecasting
"""
        )
    with b:
        st.markdown('<div class="home-step-badge">STEP 2 — TEST IT FAIRLY</div>', unsafe_allow_html=True)
        tip_header("Use time-based validation", _TOOLTIPS["validation"], "home")
        st.caption("A good forecasting model should work on future dates, not just on past data.")
        st.markdown(
            """
- later dates are held out
- metrics reflect future-like performance
- ranking is based on holdout error
"""
        )
    with c:
        st.markdown('<div class="home-step-badge">STEP 3 — EXPLAIN THE RESULTS</div>', unsafe_allow_html=True)
        tip_header("Make the findings easy to understand", _TOOLTIPS["dashboard"], "home")
        st.caption("The dashboard is built for presentation, not just technical review.")
        st.markdown(
            """
- interactive charts
- concise metric cards
- readable tooltips
- clear business explanations
"""
        )

with tab2:
    st.subheader("Repository Structure")
    repo_df = pd.DataFrame({
        "Folder": ["src/", "notebooks/", "app/", "reports/", "docs/"],
        "Purpose": [
            "Reusable modules for cleaning, feature engineering, modeling, I/O, and chart support",
            "Narrative notebook that runs the full report-style workflow",
            "Presentation-first Streamlit dashboard",
            "Saved model outputs used by the dashboard",
            "Technical documentation and portfolio-facing explanations",
        ]
    })
    st.dataframe(repo_df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Execution Order")
    st.code("pip install -r requirements.txt", language="bash")
    st.markdown(
        """
1. Place `train.csv`, `test.csv`, and `sample_submission.csv` into `data/raw/`  
2. Run `notebooks/pastry_pipeline.ipynb`  
3. Review outputs in `reports/` and `docs/`  
4. Launch the dashboard with `streamlit run app/Home.py`
"""
    )

footer("Home page · final-report-aligned dashboard · portfolio presentation style")
