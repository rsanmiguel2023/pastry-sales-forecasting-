"""Input/output helpers for the pastry forecasting project.

This module centralizes file reading and writing so the notebook and the
analysis pipeline can remain focused on workflow orchestration.
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import pandas as pd


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory for a file path if it does not exist.

    Args:
        path: File path that will be written.

    Returns:
        Normalized Path object.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or parquet table based on file extension.

    Args:
        path: Input file path.

    Returns:
        Loaded pandas DataFrame.
    """
    path = Path(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file type: {path.suffix}")


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to CSV.

    Args:
        df: DataFrame to save.
        path: Output CSV path.

    Returns:
        Output Path.
    """
    path = ensure_parent_dir(path)
    df.to_csv(path, index=False)
    return path


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to parquet.

    Args:
        df: DataFrame to save.
        path: Output parquet path.

    Returns:
        Output Path.
    """
    path = ensure_parent_dir(path)
    df.to_parquet(path, index=False)
    return path


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Write a dictionary to JSON.

    Args:
        payload: Dictionary payload.
        path: Output JSON path.

    Returns:
        Output Path.
    """
    path = ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
