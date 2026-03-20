# Analytics Module

## Role of `src/analysis.py`

`analysis.py` orchestrates the end-to-end workflow:
- load raw train/test files
- clean and engineer features
- run time-based model evaluation
- export saved report files
- generate a competition-style submission

## Main outputs

- `reports/time_based_model_comparison.csv`
- `reports/model_predictions.csv`
- `reports/submission.csv`
