# Visualization

## Design principle

The repository now uses two different presentation layers:

### Notebook / pipeline layer
Plots can still be saved as report-aligned figures for documentation and artifact tracking.

### Dashboard layer
The Streamlit app **does not display saved figure files**.  
Instead, it recreates the visuals dynamically with Plotly from the available data and report outputs.

## Why this is better for the dashboard

Dynamic charts make the app:
- less static
- easier to explore
- more polished for portfolio review
- easier to maintain when the underlying data changes

## Dashboard chart groups

### EDA
- sales over time
- sales distribution
- average sales by weekday
- average sales by month
- correlation heatmap

### Feature Engineering
- sales vs lag (t-1)
- sales vs 7-day rolling mean

### Model Results
- model comparison by test MSE
- model comparison by test R²
- predicted vs actual
- residual plot
