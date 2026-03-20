"""Streamlit dashboard for the pastry forecasting project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"


def configure_page() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(page_title="Pastry Sales Forecasting", layout="wide")


def read_csv_safe(path: Path) -> pd.DataFrame:
    """Read a CSV if it exists, otherwise return an empty DataFrame."""
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def render_image_if_exists(file_name: str, caption: str) -> None:
    """Render a saved image if it exists."""
    image_path = FIGURES_DIR / file_name
    if image_path.exists():
        st.image(str(image_path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing figure: {file_name}")


def main() -> None:
    """Run the Streamlit dashboard."""
    configure_page()

    st.title("Pastry Sales Forecasting Dashboard")
    st.caption("The dashboard reads saved outputs from the notebook-driven analysis pipeline.")

    results_df = read_csv_safe(REPORTS_DIR / "model_comparison.csv")
    predictions_df = read_csv_safe(REPORTS_DIR / "model_predictions.csv")
    time_results_df = read_csv_safe(REPORTS_DIR / "time_based_model_comparison.csv")
    submission_df = read_csv_safe(REPORTS_DIR / "submission.csv")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Executive Summary", "EDA", "Feature Validation", "Model Results", "Competition Output"]
    )

    with tab1:
        st.subheader("Executive Summary")
        if not time_results_df.empty:
            best = time_results_df.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Best Time-Based Model", best["model"])
            c2.metric("Best Validation MSE", f'{best["test_mse"]:.4f}')
            c3.metric("Best Validation R²", f'{best["test_r2"]:.4f}')
        elif not results_df.empty:
            best = results_df.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Best Model", best["model"])
            c2.metric("Best Test MSE", f'{best["test_mse"]:.4f}')
            c3.metric("Best Test R²", f'{best["test_r2"]:.4f}')
        else:
            st.info("Run the notebook pipeline first to populate the dashboard.")

    with tab2:
        st.subheader("Baseline EDA")
        for file_name, caption in [
            ("sales_over_time.png", "Sales over time"),
            ("sales_distribution.png", "Sales distribution"),
            ("sales_by_weekday.png", "Average sales by weekday"),
            ("sales_by_month.png", "Average sales by month"),
            ("correlation_heatmap.png", "Correlation heatmap"),
        ]:
            render_image_if_exists(file_name, caption)

    with tab3:
        st.subheader("Feature Validation")
        for file_name, caption in [
            ("sales_lag_1_relationship.png", "Lag feature relationship"),
            ("sales_rolling_mean_7_comparison.png", "Rolling mean comparison"),
        ]:
            render_image_if_exists(file_name, caption)

    with tab4:
        st.subheader("Model Results")
        if not time_results_df.empty:
            st.write("Time-based validation results")
            st.dataframe(time_results_df, use_container_width=True)
            for file_name, caption in [
                ("time_based_model_comparison_test_mse.png", "Time-based model comparison - Test MSE"),
                ("time_based_model_comparison_test_r2.png", "Time-based model comparison - Test R²"),
                ("time_based_predicted_vs_actual.png", "Time-based predicted vs actual"),
                ("time_based_residual_plot.png", "Time-based residual plot"),
            ]:
                render_image_if_exists(file_name, caption)
        elif not results_df.empty:
            st.dataframe(results_df, use_container_width=True)
            for file_name, caption in [
                ("model_comparison_test_mse.png", "Model comparison - Test MSE"),
                ("model_comparison_test_r2.png", "Model comparison - Test R²"),
                ("predicted_vs_actual.png", "Predicted vs Actual for best model"),
                ("residual_plot.png", "Residual plot for best model"),
            ]:
                render_image_if_exists(file_name, caption)

        if not predictions_df.empty:
            st.dataframe(predictions_df.head(50), use_container_width=True)

    with tab5:
        st.subheader("Competition Output")
        if not submission_df.empty:
            st.dataframe(submission_df.head(50), use_container_width=True)
            render_image_if_exists("submission_prediction_distribution.png", "Distribution of submission predictions")
        else:
            st.info("Run the submission step in the notebook to generate reports/submission.csv.")


if __name__ == "__main__":
    main()
