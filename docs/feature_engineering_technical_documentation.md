# Feature Engineering Technical Documentation

## Objective

The feature layer converts raw store-day observations into a forecasting-ready design matrix.

## Feature families

- calendar features
- holiday encodings
- transformed operational/weather variables
- lag features
- rolling mean features

## Validation figures

### Figure 8.1 — Sales vs Lag (t-1)
Confirms that recent sales contain strong predictive signal.

### Figure 8.2 — Sales vs Rolling Mean (7)
Shows that rolling means smooth noise and preserve the local demand signal.

## Final model feature count

The current saved best model uses **16 features**.
