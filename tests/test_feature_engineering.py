"""Unit tests for feature engineering utilities."""

import pandas as pd

from src.feature_engineering import build_features, build_competition_features


def test_build_features_creates_expected_columns():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "sales": [10, 15, 12],
            "store_id": ["A", "A", "A"],
            "temperature_mean": [1.0, 2.0, 3.0],
            "sunshine_sum": [2.0, 3.0, 4.0],
            "precipitation_sum": [0.1, 0.2, 0.3],
            "unsold": [1.0, 2.0, 3.0],
            "ordered": [11.0, 17.0, 15.0],
            "is_state_holiday": ["normal_day", "state_holiday", "normal_day"],
            "is_school_holiday": ["normal_day", "normal_day", "school_holiday"],
            "is_special_day": ["normal_day", "special_day", "normal_day"],
        }
    )
    result = build_features(df)
    expected_cols = {
        "day_of_week",
        "month",
        "is_holiday_binary",
        "precipitation_sum_boxcox",
        "unsold_boxcox",
        "ordered_boxcox",
    }
    assert expected_cols.issubset(set(result.columns))


def test_build_competition_features_passes_history_to_test():
    train_df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "store_id": ["A", "A"],
            "sales": [10.0, 20.0],
            "temperature_mean": [1.0, 2.0],
            "sunshine_sum": [2.0, 3.0],
            "precipitation_sum": [0.1, 0.2],
            "unsold": [1.0, 2.0],
            "ordered": [11.0, 12.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "row_id": [1],
            "date": ["2024-01-03"],
            "store_id": ["A"],
            "temperature_mean": [3.0],
            "sunshine_sum": [4.0],
            "precipitation_sum": [0.3],
            "unsold": [None],
            "ordered": [None],
        }
    )
    train_out, test_out = build_competition_features(train_df, test_df)
    assert "sales_lag_1" in test_out.columns
