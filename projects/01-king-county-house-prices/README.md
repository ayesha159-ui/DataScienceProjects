# King County, WA — House Price Prediction
**Goal:** build a clean, reproducible baseline that predicts sale price from property features, with clear business takeaways.

## Dataset
King County (Seattle area) house sales dataset (public).  
File: `data/kc_final.csv` (see “Data” below).

## What I deliver
- Exploratory analysis (pricing trends, feature correlations, geospatial hints).
- Clean feature pipeline (dates → age, bath/bed ratios, sqft per room, zipcode dummies).
- Baseline & tuned models (Linear Regression, Extra Tree Regressor, GradientBoostingRegressor etc).
- Explainability (per-feature importances + SHAP summary if enabled).
- Metrics: RMSE / MAE on a strict hold-out split + cross-val.
- Practical insights: top price drivers, where models underperform, how to use it.

## How to run
```bash
# Option A: conda
conda env create -f environment.yml
conda activate kc-prices
jupyter lab

# Option B: pip
pip install -r requirements.txt
jupyter lab

