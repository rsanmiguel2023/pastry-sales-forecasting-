# Data Description

## Project context

This project uses a competition-style pastry sales forecasting dataset with the core files:
- `train.csv`
- `test.csv`
- `sample_submission.csv`

The analytical unit is **one row per store per date**. The prediction target is `sales`, which appears in the training set but not in the test set.

## Why the dataset matters

The data structure supports a complete forecasting workflow:
- descriptive EDA
- feature engineering
- time-based validation
- final competition-style submission generation

## Key takeaway

This is not a generic regression dataset. It is a **future-facing forecasting problem**, which is why the repository uses time-based validation instead of relying only on random train/test splitting.
