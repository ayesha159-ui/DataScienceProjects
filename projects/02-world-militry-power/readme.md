# Clustering the “Military Power” Profiles of Countries
**Goal:** group countries by *capability profiles* (resources, equipment, readiness proxies) using unsupervised learning—clarify *types* rather than rank “power”.

> ⚠️ **Framing note:** “Military power” is ambiguous and value-laden. This project clusters **features** (inputs) and reports uncertainty; it does **not** claim a universal ranking.

## Data
Bring a tidy CSV to `data/country_military_features.csv` with one row per country (ISO3 code recommended). Example columns:
- `iso3`, `country`
- **Resources/inputs:** `milspend_usd_bn`, `milspend_gdp_pct`, `personnel_active_k`
- **Hardware proxies:** `tanks`, `afvs`, `artillery`, `combat_aircraft`, `helicopters`, `submarines`, `destroyers`
- **Readiness/logistics (optional):** `defense_imports_index`, `defense_exports_index`, `conscription` (0/1)
- **Context (optional):** `gdp_usd_bn`, `pop_m`

> Keep units consistent; document each column in `data/README.md`.

## Method (short)
1. **Clean & harmonize** country IDs → ISO3; impute few missing values.
2. **Scale** features (StandardScaler).
3. **Dimensionality**: PCA for visualization/explained variance (retain info, avoid dominance by scale).
4. **Clustering**: start with **K-Means** (k chosen via elbow + silhouette), then test **HDBSCAN** for shape-free clusters.
5. **Validation**: silhouette score, Davies–Bouldin; stability across seeds.
6. **Explain clusters**: per-feature z-score profiles, top differentiators.
7. **Visualize**: PCA scatter with cluster colors; optional choropleth by cluster.

## What you’ll see
- Clear cluster count rationale and scores.
- Per-cluster “fingerprints” (spider/heatmap).
- Country lists per cluster + notable outliers.

## How to run
```bash
# conda
conda env create -f environment.yml
conda activate mil-cluster
jupyter lab
# open notebook.ipynb and "Run All"
