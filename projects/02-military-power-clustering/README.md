# Military Power Clustering (Countries)

**Goal:** Cluster countries by capability profiles (budget, personnel, hardware proxies) using unsupervised learning.  
**Notebook:** `World Military Power.ipynb`

## Data
- File: `data/World military power.xlsx`` (see “Data” below).
## What I Deliver
- **Clear EDA**: quick stats, missingness, and correlations to understand drivers.
- **Clean features**: parsed numerics, light imputation, and scaled inputs ready for modeling.
- **Clustering**: K-Means (k≈4) with elbow/silhouette justification + a hierarchical dendrogram preview.
- **Explainability**: per-cluster feature profiles (boxplots/heatmaps) and concise takeaways.
- **Artifacts**: cluster assignments table in the notebook (optional CSV export to `results/`).
- **Reproducibility**: environment file, simple run steps, and git-ignored data/artifacts.
- **Limitations noted**: “power” is feature-dependent; results are descriptive, not a ranking.

## How to run
```bash
# Option A: conda
conda env create -f environment.yml
conda activate kc-prices
jupyter lab

# Option B: pip
pip install -r requirements.txt
jupyter lab
