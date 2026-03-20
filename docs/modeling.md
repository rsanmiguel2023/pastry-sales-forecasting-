# Modeling

## Modeling objective

Compare candidate regressors under a **time-based validation design** and identify the strongest forecasting model for the pastry sales problem.

## Candidate models

- Linear Regression
- Decision Tree
- Random Forest
- XGBoost

## Current saved result

- Champion model: **XGBoost**
- Best Test MSE: **0.025968**
- Best Test R²: **0.959253**

## Interpretation

Tree-based models outperform the linear baseline, which suggests the demand process includes non-linear effects and short-run temporal structure.

## Why time-based validation matters

Because this is a forecasting problem, later observations should be held out from training.  
That makes the evaluation more realistic than a random split.
