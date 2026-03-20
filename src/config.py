"""Project-wide path configuration.

This module centralizes folder paths so the rest of the codebase does not
hard-code directory locations in multiple files.
"""

from pathlib import Path


# Resolve the project root from the current file location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define standard project folders.
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"
