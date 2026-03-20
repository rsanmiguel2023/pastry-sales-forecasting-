# Exploratory Data Analysis Technical Documentation

## Objective

The EDA stage is descriptive only. Its goal is to establish the structure of the target and identify the main recurring demand patterns before model development.

## Figures used

### Figure 6.1 — Sales Over Time
Shows daily volatility, spikes, and broader level shifts.

### Figure 6.2 — Distribution of Sales
Shows a right-skewed target with a long upper tail.

### Figure 6.3 — Average Sales by Weekday
Shows strong weekly signal, especially higher weekend demand.

### Figure 6.4 — Average Sales by Month
Shows broader seasonal movement and a visible mid-year dip.

### Figure 6.5 — Correlation Heatmap
Shows where numeric features overlap and where engineered temporal features align with the target.

## EDA conclusion

The EDA justifies:
- calendar features
- lag features
- rolling mean features
- non-linear models
