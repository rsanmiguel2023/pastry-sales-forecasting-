"""Unit tests for the analysis module."""

import pandas as pd

from src.analysis import select_features


def make_sample_analysis_df() -> pd.DataFrame:
    """Create a small sample DataFrame for analysis tests."""
    return pd.DataFrame(
        {
            "sales": [1, 2],
            "temperature_mean": [10, 12],
            "sunshine_sum": [5, 6],
            "store_num": [1, 1],
            "unused": [0, 0],
        }
    )


def test_select_features_returns_only_available_columns():
    """Ensure only configured and available features are selected."""
    df = make_sample_analysis_df()
    features = select_features(df)

    assert "temperature_mean" in features
    assert "sunshine_sum" in features
    assert "unused" not in features
