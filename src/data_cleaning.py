"""Data cleaning utilities for pastry sales forecasting.

This module performs light-touch preprocessing before feature engineering.
It standardizes column names, converts date fields, encodes stores, and
fills selected missing values while preserving the original analytical grain.
"""

from __future__ import annotations

import pandas as pd


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names for consistent downstream usage."""
    out = df.copy()
    out.columns = [col.strip().lower().replace(" ", "_") for col in out.columns]
    return out


def convert_date_columns(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Convert the date column to pandas datetime."""
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    return out


def encode_store_column(df: pd.DataFrame) -> pd.DataFrame:
    """Create a stable numeric store identifier."""
    out = df.copy()

    if "store" in out.columns and "store_id" not in out.columns:
        out = out.rename(columns={"store": "store_id"})

    if "store_id" in out.columns and "store_num" not in out.columns:
        out["store_num"] = pd.factorize(out["store_id"])[0] + 1

    return out


def fill_selected_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill selected numeric columns with median values from the same frame."""
    out = df.copy()

    for col in ["unsold", "ordered"]:
        if col in out.columns:
            out[col] = out[col].fillna(out[col].median())

    return out


def fill_selected_missing_values_with_reference(
    df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """Fill selected numeric columns using medians from a reference DataFrame.

    This is useful for test-set preparation where train-set medians should
    be reused to avoid leakage.
    """
    out = df.copy()

    for col in ["unsold", "ordered"]:
        if col in out.columns and col in reference_df.columns:
            out[col] = out[col].fillna(reference_df[col].median())

    return out


def sort_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """Sort the dataset in time order."""
    out = df.copy()

    if "store_id" in out.columns and "date" in out.columns:
        out = out.sort_values(["store_id", "date"]).reset_index(drop=True)
    elif "date" in out.columns:
        out = out.sort_values("date").reset_index(drop=True)

    return out


def clean_pastry_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the baseline cleaning workflow."""
    out = df.copy()
    out = standardize_column_names(out)
    out = convert_date_columns(out)
    out = encode_store_column(out)
    out = fill_selected_missing_values(out)
    out = sort_time_series(out)
    return out


def clean_pastry_test_data(df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    """Clean the test dataset using train-based reference statistics."""
    out = df.copy()
    out = standardize_column_names(out)
    out = convert_date_columns(out)
    out = encode_store_column(out)
    out = fill_selected_missing_values_with_reference(out, reference_df=reference_df)
    out = sort_time_series(out)
    return out
