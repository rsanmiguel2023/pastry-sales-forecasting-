# Pastry Sales Forecasting for Demand Optimization

## Overview

This project focuses on forecasting daily pastry sales across multiple stores using historical sales, weather, and holiday data. The goal is to improve production planning, reduce food waste, and optimize inventory decisions.

## Objective

* Predict daily sales for each store
* Identify key drivers of demand (weather, holidays, seasonality)
* Build accurate forecasting models to support operational decisions

## Dataset

* Source: Kaggle – Pastry Sales Forecasting Competition
* Time Range: August 2021 – May 2024
* Target Variable: Daily pastry sales (scaled)

### Key Features:

* Date (time series)
* Store ID
* Weather data (temperature, sunshine, precipitation)
* Holiday indicators (public, school, special days)

## Methodology

### 1. Exploratory Data Analysis (EDA)

* Time series trends and seasonality
* Store-level sales patterns
* Impact of holidays and weather on demand

### 2. Feature Engineering

* Lag features (previous day/week sales)
* Rolling averages (7-day, 14-day trends)
* Time-based features (day of week, month)
* Weather interactions

### 3. Models Used

* Linear Regression
* Decision Tree
* Random Forest
* XGBoost

### 4. Evaluation Metric

* Mean Squared Error (MSE)

## Results

* XGBoost achieved the best performance
* Weather and holiday features significantly improved predictions
* Weekly seasonality was a strong predictor of demand

## Business Impact

* Helps bakeries reduce overproduction and waste
* Improves staffing and inventory planning
* Enables data-driven decision-making for daily operations

## Tools and Technologies

* Python (Pandas, NumPy, Scikit-learn, XGBoost)
* Google Colab
* Matplotlib / Seaborn

## Sample Visualizations

(Add screenshots here: time series trends, feature importance, predictions vs actuals)

## How to Run

```bash
pip install -r requirements.txt
python src/train_model.py
```

## Future Improvements

* Incorporate deep learning models (LSTM)
* Hyperparameter tuning (GridSearch / Optuna)
* Deploy as a real-time forecasting app
