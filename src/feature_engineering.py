"""Feature engineering utilities for the pastry forecasting project."""

from __future__ import annotations

import numpy as np
import pandas as pd


STATE_HOLIDAY_MAP = {
    "normal_day": 0,
    "state_holiday": 1,
    "day_before": 1,
    "day_after": 1,
}
SCHOOL_HOLIDAY_MAP = {
    "normal_day": 0,
    "school_holiday": 2,
}
SPECIAL_DAY_MAP = {
    "normal_day": 0,
    "special_day": 3,
    "day_before": 3,
}


def safe_log_transform(series: pd.Series, eps: float = 1e-6) -> pd.Series:
    """Apply a safe log transform after shifting values if needed."""
    shifted = series.astype(float).copy()
    min_value = shifted.min()

    if pd.notna(min_value) and min_value <= 0:
        shifted = shifted + abs(min_value) + 1.0

    return np.log1p(shifted + eps)


def add_date_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Create calendar-based features from a date column."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])

    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["day"] = out[date_col].dt.day
    out["day_of_week"] = out[date_col].dt.dayofweek
    out["week_of_year"] = out[date_col].dt.isocalendar().week.astype(int)
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)

    return out


def encode_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode holiday text columns into numeric features."""
    out = df.copy()

    if "is_state_holiday" in out.columns:
        out["is_state_holiday_num"] = (
            out["is_state_holiday"].map(STATE_HOLIDAY_MAP).fillna(0).astype(int)
        )

    if "is_school_holiday" in out.columns:
        out["is_school_holiday_num"] = (
            out["is_school_holiday"].map(SCHOOL_HOLIDAY_MAP).fillna(0).astype(int)
        )

    if "is_special_day" in out.columns:
        out["is_special_day_num"] = (
            out["is_special_day"].map(SPECIAL_DAY_MAP).fillna(0).astype(int)
        )

    holiday_num_cols = [
        col
        for col in ["is_state_holiday_num", "is_school_holiday_num", "is_special_day_num"]
        if col in out.columns
    ]

    if holiday_num_cols:
        out["is_holiday"] = out[holiday_num_cols].sum(axis=1)
        out["is_holiday_binary"] = (out["is_holiday"] > 0).astype(int)

    return out


def add_transformed_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create transformed features for skewed variables."""
    out = df.copy()

    for col in ["precipitation_sum", "unsold", "ordered"]:
        if col in out.columns:
            out[f"{col}_boxcox"] = safe_log_transform(out[col])

    return out


def add_lag_features(
    df: pd.DataFrame,
    group_col: str = "store_id",
    target_col: str = "sales",
    lags: tuple[int, ...] = (1, 7),
) -> pd.DataFrame:
    """Create lag features for the sales target."""
    out = df.copy()

    if target_col not in out.columns:
        return out

    if group_col in out.columns:
        out = out.sort_values([group_col, "date"]).copy()
        for lag in lags:
            out[f"{target_col}_lag_{lag}"] = out.groupby(group_col)[target_col].shift(lag)
    else:
        out = out.sort_values("date").copy()
        for lag in lags:
            out[f"{target_col}_lag_{lag}"] = out[target_col].shift(lag)

    return out


def add_rolling_features(
    df: pd.DataFrame,
    group_col: str = "store_id",
    target_col: str = "sales",
    windows: tuple[int, ...] = (7, 14),
) -> pd.DataFrame:
    """Create rolling mean features for recent sales behavior."""
    out = df.copy()

    if target_col not in out.columns:
        return out

    if group_col in out.columns:
        out = out.sort_values([group_col, "date"]).copy()
        for window in windows:
            out[f"{target_col}_rolling_mean_{window}"] = (
                out.groupby(group_col)[target_col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )
    else:
        out = out.sort_values("date").copy()
        for window in windows:
            out[f"{target_col}_rolling_mean_{window}"] = (
                out[target_col].shift(1).rolling(window, min_periods=1).mean()
            )

    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline on a single frame."""
    out = df.copy()
    out = add_date_features(out)
    out = encode_holiday_features(out)
    out = add_transformed_features(out)
    out = add_lag_features(out)
    out = add_rolling_features(out)
    return out


def build_competition_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_indicator_col: str = "row_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build train/test features jointly for competition-style forecasting.

    The train and test frames are concatenated before lag and rolling
    features are created so the test rows can inherit historical context
    from the end of the training period.
    """
    train_part = train_df.copy()
    test_part = test_df.copy()

    train_part["_dataset_split"] = "train"
    test_part["_dataset_split"] = "test"

    combined = pd.concat([train_part, test_part], ignore_index=True, sort=False)
    combined = add_date_features(combined)
    combined = encode_holiday_features(combined)
    combined = add_transformed_features(combined)
    combined = add_lag_features(combined)
    combined = add_rolling_features(combined)

    train_features = combined[combined["_dataset_split"] == "train"].drop(columns=["_dataset_split"])
    test_features = combined[combined["_dataset_split"] == "test"].drop(columns=["_dataset_split"])

    return train_features, test_features
