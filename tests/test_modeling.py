"""Unit tests for the modeling module."""

import pandas as pd

from src.modeling import select_features, build_summary_payload, time_based_split


def test_select_features_returns_only_existing_columns():
    df = pd.DataFrame(
        {
            "sales": [1, 2],
            "temperature_mean": [10, 12],
            "sunshine_sum": [5, 6],
            "store_num": [1, 1],
            "unknown": [0, 0],
        }
    )
    features = select_features(df)
    assert "temperature_mean" in features
    assert "sunshine_sum" in features
    assert "unknown" not in features


def test_build_summary_payload_returns_best_model():
    results_df = pd.DataFrame(
        [
            {"model": "Random Forest", "test_mse": 0.15, "test_r2": 0.82},
            {"model": "XGBoost", "test_mse": 0.17, "test_r2": 0.80},
        ]
    )
    summary = build_summary_payload(results_df)
    assert summary["best_model"] == "Random Forest"
    assert summary["best_test_mse"] == 0.15


def test_time_based_split_respects_time_order():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "sales": [1, 2, 3, 4],
            "temperature_mean": [10, 11, 12, 13],
        }
    )
    features = ["temperature_mean"]
    X_train, X_valid, y_train, y_valid, train_df, valid_df = time_based_split(df, features=features)
    assert train_df["date"].max() <= valid_df["date"].min()
