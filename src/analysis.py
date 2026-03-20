"""High-level analysis orchestration for pastry sales forecasting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR, REPORTS_DIR
from src.data_cleaning import clean_pastry_data, clean_pastry_test_data
from src.feature_engineering import build_competition_features, build_features
from src.io_utils import read_table, write_csv, write_json, write_parquet
from src.modeling import (
    build_summary_payload,
    train_and_evaluate,
    train_and_evaluate_time_based,
    train_full_model,
)


def load_raw_data(raw_path: str | Path) -> pd.DataFrame:
    """Load a raw dataset from disk."""
    return read_table(raw_path)


def prepare_dataset(raw_path: str | Path) -> pd.DataFrame:
    """Load raw train data, clean it, and engineer features."""
    df = load_raw_data(raw_path)
    df = clean_pastry_data(df)
    df = build_features(df)
    return df


def prepare_competition_datasets(
    train_path: str | Path,
    test_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare train and test data for competition-style forecasting."""
    raw_train = load_raw_data(train_path)
    raw_test = load_raw_data(test_path)

    clean_train = clean_pastry_data(raw_train)
    clean_test = clean_pastry_test_data(raw_test, reference_df=clean_train)

    train_features, test_features = build_competition_features(clean_train, clean_test)
    return train_features, test_features


def save_processed_data(df: pd.DataFrame, file_name: str = "pastry_processed.parquet") -> Path:
    """Save the processed modeling dataset to parquet."""
    output_path = PROCESSED_DATA_DIR / file_name
    return write_parquet(df, output_path)


def export_outputs(results_df: pd.DataFrame, predictions_df: pd.DataFrame) -> None:
    """Export tabular outputs for reports and Streamlit."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(results_df, REPORTS_DIR / "model_comparison.csv")
    write_csv(predictions_df, REPORTS_DIR / "model_predictions.csv")
    write_json(build_summary_payload(results_df), REPORTS_DIR / "summary.json")


def export_submission(submission_df: pd.DataFrame, file_name: str = "submission.csv") -> Path:
    """Export a competition submission file to the reports folder."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / file_name
    return write_csv(submission_df, output_path)


def run_analysis(raw_path: str | Path):
    """Run the complete random-split analysis pipeline end to end."""
    df = prepare_dataset(raw_path)
    save_processed_data(df)

    results_df, predictions_df, trained_models, features = train_and_evaluate(df)
    export_outputs(results_df, predictions_df)

    return df, results_df, predictions_df, trained_models, features


def run_time_based_analysis(raw_path: str | Path):
    """Run the complete time-based analysis pipeline end to end."""
    df = prepare_dataset(raw_path)
    save_processed_data(df)

    results_df, predictions_df, trained_models, features, train_df, valid_df = train_and_evaluate_time_based(df)
    export_outputs(results_df, predictions_df)

    return df, results_df, predictions_df, trained_models, features, train_df, valid_df


def build_submission_from_paths(
    train_path: str | Path,
    test_path: str | Path,
    submission_file_name: str = "submission.csv",
):
    """Train on the full competition train set and create Kaggle-style predictions."""
    train_df, test_df = prepare_competition_datasets(train_path, test_path)

    save_processed_data(train_df, file_name="pastry_train_processed.parquet")
    write_parquet(test_df, PROCESSED_DATA_DIR / "pastry_test_processed.parquet")

    best_model_name, best_model, features, time_results_df = train_full_model(train_df)

    test_model_df = test_df.dropna(subset=features).copy()
    preds = best_model.predict(test_model_df[features])

    submission_df = test_model_df[["row_id"]].copy()
    submission_df["sales"] = preds

    export_submission(submission_df, file_name=submission_file_name)
    write_csv(time_results_df, REPORTS_DIR / "time_based_model_comparison.csv")

    return submission_df, best_model_name, features, time_results_df
