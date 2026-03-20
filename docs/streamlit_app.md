# Streamlit App

## Design goal

The dashboard follows the **final report flow** and is written for presentation use.  
The language is intentionally business-friendly: clear enough for a manager, but still precise enough for a technical reviewer.

## Main design choices

- dynamic Plotly charts instead of displaying saved image files
- short and readable tooltip text
- expandable **“What does this mean?”** sections written in plain business language
- executive summary banners and step-badge logic blocks
- separate pages for EDA, feature engineering, and model results

## Why this matters

The app is meant to answer business-facing questions such as:
- what demand looks like
- which patterns matter most
- which model is strongest
- whether the forecast is reliable enough to support planning
