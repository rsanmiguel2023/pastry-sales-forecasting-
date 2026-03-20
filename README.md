# Pastry Prediction Analytics

A recruiter-ready, portfolio-grade forecasting project built around a **store-day pastry sales prediction problem**.  
The repository combines a modular Python pipeline, a narrative notebook, technical documentation, and a polished **Streamlit dashboard** designed to mirror the flow of the final report rather than a competition-submission page.

## What makes this project strong

This is not a toy machine-learning notebook. The repository shows the full lifecycle of an applied forecasting problem:

- descriptive EDA that motivates the modeling choices
- engineered temporal features such as lags and rolling means
- time-based validation instead of relying only on a random split
- model comparison and diagnostics
- a presentation-first dashboard that explains the analytical story clearly

## Current best result

From the latest saved time-based results:
- **Champion model:** XGBoost
- **Best Test MSE:** 0.025968
- **Best Test R²:** 0.959253
- **Features used:** 16

## Repository structure

```text
pastry_prediction_analytics/
├── app/
│   ├── Home.py
│   ├── shared.py
│   └── pages/
│       ├── 0_EDA.py
│       ├── 1_Feature_Engineering.py
│       └── 2_Model_Results.py
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── notebooks/
│   └── pastry_pipeline.ipynb
├── reports/
├── src/
└── tests/
```

## Dashboard structure

The Streamlit app follows the final-report flow:

- **Home** — project framing, KPIs, execution logic
- **EDA** — descriptive trend, distribution, calendar, and correlation views
- **Feature Engineering** — lag and rolling-feature validation
- **Model Results** — comparison, champion model, and diagnostics

The dashboard uses:
- KPI cards
- readable inline tooltips
- expandable “What does this mean?” interpretation blocks
- interactive Plotly charts instead of static saved figures

## How to run

1. Put `train.csv`, `test.csv`, and `sample_submission.csv` into `data/raw/`
2. Run the notebook: `notebooks/pastry_pipeline.ipynb`
3. Launch the dashboard:
   ```bash
   streamlit run app/Home.py
   ```

## Why the evaluation is credible

The project uses **time-based validation**. That matters because forecasting is a future-facing problem: a model should be tested on later dates, not on randomly shuffled rows that mix past and future patterns.

## Portfolio value

This project is built to show:
- forecasting workflow design
- feature-engineering discipline
- model comparison and diagnostics
- technical documentation quality
- executive-friendly presentation

It is meant to be easy for both hiring managers and technical reviewers to understand.
