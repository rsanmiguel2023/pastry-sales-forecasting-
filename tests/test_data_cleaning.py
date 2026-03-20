"""Unit tests for the data cleaning module."""

import pandas as pd

from src.data_cleaning import clean_pastry_data


def test_clean_pastry_data_encodes_store_and_parses_date():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "store": ["A", "B"],
            "sales": [1.0, 2.0],
            "unsold": [None, 1.5],
            "ordered": [2.0, None],
        }
    )

    result = clean_pastry_data(df)

    assert "store_id" in result.columns
    assert "store_num" in result.columns
    assert str(result["date"].dtype).startswith("datetime64")
    assert result["unsold"].isna().sum() == 0
    assert result["ordered"].isna().sum() == 0
