# Data Cleaning

## Purpose

The cleaning layer standardizes raw competition files before feature engineering.

## Main steps

1. normalize column names  
2. convert `date` to datetime  
3. encode the store column into a stable numeric identifier  
4. fill selected missing values  
5. sort observations in time order  

## Why this matters

A forecasting model is only as reliable as its time ordering and schema consistency. The cleaning layer ensures both before any engineered features are added.
