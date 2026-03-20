"""Visualization utilities for pastry forecasting outputs.

All plotting functions now:
- display inline in the notebook
- save automatically in `figures/`
- use report-aligned file names for easier reference
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR


REPORT_FIGURE_NAMES = {
    "sales_over_time": "fig_6_1_sales_over_time.png",
    "sales_distribution": "fig_6_2_sales_distribution.png",
    "sales_by_weekday": "fig_6_3_sales_by_weekday.png",
    "sales_by_month": "fig_6_4_sales_by_month.png",
    "correlation_heatmap": "fig_6_5_correlation_heatmap.png",
    "sales_lag_1_relationship": "fig_8_1_sales_lag_1_relationship.png",
    "sales_rolling_mean_7_comparison": "fig_8_2_sales_rolling_mean_7_comparison.png",
    "model_comparison_test_mse": "fig_9_1_time_based_model_comparison_test_mse.png",
    "model_comparison_test_r2": "fig_9_2_time_based_model_comparison_test_r2.png",
    "predicted_vs_actual": "fig_9_3_time_based_predicted_vs_actual.png",
    "residual_plot": "fig_9_4_time_based_residual_plot.png",
    "submission_prediction_distribution": "fig_10_1_submission_prediction_distribution.png",
}


def set_plot_style() -> None:
    """Apply a simple consistent matplotlib style across the project."""
    plt.style.use("default")
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.rcParams["axes.grid"] = True


def _resolve_figure_name(key_or_file_name: str) -> str:
    """Resolve a report key or raw file name into the final file name."""
    return REPORT_FIGURE_NAMES.get(key_or_file_name, key_or_file_name)


def save_and_show_figure(fig, file_name: str) -> Path:
    """Save the figure, display it inline, and close it cleanly."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / _resolve_figure_name(file_name)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return output_path


def plot_sales_over_time(df: pd.DataFrame, date_col: str = "date", target_col: str = "sales") -> Path:
    set_plot_style()
    plot_df = df.sort_values(date_col)

    fig, ax = plt.subplots()
    ax.plot(plot_df[date_col], plot_df[target_col])
    ax.set_title("Pastry Sales Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")

    return save_and_show_figure(fig, "sales_over_time")


def plot_sales_distribution(df: pd.DataFrame, target_col: str = "sales", bins: int = 40) -> Path:
    set_plot_style()

    fig, ax = plt.subplots()
    ax.hist(df[target_col].dropna(), bins=bins)
    ax.set_title("Distribution of Sales")
    ax.set_xlabel("Sales")
    ax.set_ylabel("Frequency")

    return save_and_show_figure(fig, "sales_distribution")


def plot_sales_by_weekday(df: pd.DataFrame, weekday_col: str = "day_of_week", target_col: str = "sales") -> Path:
    set_plot_style()
    plot_df = df.groupby(weekday_col, as_index=False)[target_col].mean()

    fig, ax = plt.subplots()
    ax.bar(plot_df[weekday_col], plot_df[target_col])
    ax.set_title("Average Sales by Weekday")
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Average Sales")

    return save_and_show_figure(fig, "sales_by_weekday")


def plot_sales_by_month(df: pd.DataFrame, month_col: str = "month", target_col: str = "sales") -> Path:
    set_plot_style()
    plot_df = df.groupby(month_col, as_index=False)[target_col].mean()

    fig, ax = plt.subplots()
    ax.bar(plot_df[month_col], plot_df[target_col])
    ax.set_title("Average Sales by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Sales")

    return save_and_show_figure(fig, "sales_by_month")


def plot_correlation_heatmap(df: pd.DataFrame, columns: list[str] | None = None) -> Path:
    set_plot_style()
    plot_df = df.copy()
    if columns is not None:
        plot_df = plot_df[columns]

    numeric_df = plot_df.select_dtypes(include=["number"])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(corr, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    ax.set_title("Correlation Heatmap")
    fig.colorbar(im, ax=ax)

    return save_and_show_figure(fig, "correlation_heatmap")


def plot_lag_relationship(df: pd.DataFrame, lag_col: str = "sales_lag_1", target_col: str = "sales") -> Path:
    set_plot_style()
    plot_df = df.dropna(subset=[lag_col, target_col])

    fig, ax = plt.subplots()
    ax.scatter(plot_df[lag_col], plot_df[target_col], alpha=0.4)
    ax.set_title(f"{target_col} vs {lag_col}")
    ax.set_xlabel(lag_col)
    ax.set_ylabel(target_col)

    return save_and_show_figure(fig, f"{lag_col}_relationship")


def plot_rolling_feature_comparison(
    df: pd.DataFrame,
    target_col: str = "sales",
    rolling_col: str = "sales_rolling_mean_7",
    date_col: str = "date",
) -> Path:
    set_plot_style()
    plot_df = df.dropna(subset=[rolling_col]).sort_values(date_col)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(plot_df[date_col], plot_df[target_col], label="Actual Sales")
    ax.plot(plot_df[date_col], plot_df[rolling_col], label=rolling_col)
    ax.set_title(f"{target_col} vs {rolling_col}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()

    return save_and_show_figure(fig, f"{rolling_col}_comparison")


def plot_model_comparison(results_df: pd.DataFrame, file_name: str = "model_comparison_test_mse") -> Path:
    set_plot_style()
    plot_df = results_df.sort_values("test_mse")

    fig, ax = plt.subplots()
    ax.bar(plot_df["model"], plot_df["test_mse"])
    ax.set_title("Model Comparison - Test MSE")
    ax.set_xlabel("Model")
    ax.set_ylabel("Test MSE")
    ax.tick_params(axis="x", rotation=20)

    return save_and_show_figure(fig, file_name)


def plot_model_r2_comparison(results_df: pd.DataFrame, file_name: str = "model_comparison_test_r2") -> Path:
    set_plot_style()
    plot_df = results_df.sort_values("test_r2", ascending=False)

    fig, ax = plt.subplots()
    ax.bar(plot_df["model"], plot_df["test_r2"])
    ax.set_title("Model Comparison - Test R²")
    ax.set_xlabel("Model")
    ax.set_ylabel("Test R²")
    ax.tick_params(axis="x", rotation=20)

    return save_and_show_figure(fig, file_name)


def plot_predictions_vs_actual(
    predictions_df: pd.DataFrame,
    model_name: str | None = None,
    file_name: str = "predicted_vs_actual",
) -> Path:
    set_plot_style()
    plot_df = predictions_df.copy()
    if model_name:
        plot_df = plot_df[plot_df["model"] == model_name]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(plot_df["actual"], plot_df["predicted"], alpha=0.4)
    ax.set_title(f"Predicted vs Actual{f' - {model_name}' if model_name else ''}")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")

    return save_and_show_figure(fig, file_name)


def plot_residuals(
    predictions_df: pd.DataFrame,
    model_name: str | None = None,
    file_name: str = "residual_plot",
) -> Path:
    set_plot_style()
    plot_df = predictions_df.copy()
    if model_name:
        plot_df = plot_df[plot_df["model"] == model_name]

    fig, ax = plt.subplots()
    ax.scatter(plot_df["predicted"], plot_df["residual"], alpha=0.4)
    ax.axhline(0, linestyle="--")
    ax.set_title(f"Residual Plot{f' - {model_name}' if model_name else ''}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")

    return save_and_show_figure(fig, file_name)


def plot_submission_distribution(submission_df: pd.DataFrame, target_col: str = "sales") -> Path:
    set_plot_style()

    fig, ax = plt.subplots()
    ax.hist(submission_df[target_col].dropna(), bins=40)
    ax.set_title("Distribution of Submission Predictions")
    ax.set_xlabel("Predicted Sales")
    ax.set_ylabel("Frequency")

    return save_and_show_figure(fig, "submission_prediction_distribution")


def generate_eda_figures(df: pd.DataFrame) -> list[Path]:
    return [
        plot_sales_over_time(df),
        plot_sales_distribution(df),
        plot_sales_by_weekday(df),
        plot_sales_by_month(df),
        plot_correlation_heatmap(df),
    ]


def generate_feature_validation_figures(df: pd.DataFrame) -> list[Path]:
    outputs = []
    if "sales_lag_1" in df.columns:
        outputs.append(plot_lag_relationship(df, lag_col="sales_lag_1"))
    if "sales_rolling_mean_7" in df.columns:
        outputs.append(plot_rolling_feature_comparison(df, rolling_col="sales_rolling_mean_7"))
    return outputs


def generate_model_result_figures(results_df: pd.DataFrame, predictions_df: pd.DataFrame) -> list[Path]:
    outputs = [
        plot_model_comparison(results_df, file_name="model_comparison_test_mse"),
        plot_model_r2_comparison(results_df, file_name="model_comparison_test_r2"),
    ]
    if not results_df.empty and not predictions_df.empty:
        best_model = results_df.iloc[0]["model"]
        outputs.append(plot_predictions_vs_actual(predictions_df, model_name=best_model, file_name="predicted_vs_actual"))
        outputs.append(plot_residuals(predictions_df, model_name=best_model, file_name="residual_plot"))
    return outputs


def generate_time_based_model_figures(results_df: pd.DataFrame, predictions_df: pd.DataFrame) -> list[Path]:
    outputs = [
        plot_model_comparison(results_df, file_name="model_comparison_test_mse"),
        plot_model_r2_comparison(results_df, file_name="model_comparison_test_r2"),
    ]
    if not results_df.empty and not predictions_df.empty:
        best_model = results_df.iloc[0]["model"]
        outputs.append(plot_predictions_vs_actual(predictions_df, model_name=best_model, file_name="predicted_vs_actual"))
        outputs.append(plot_residuals(predictions_df, model_name=best_model, file_name="residual_plot"))
    return outputs
