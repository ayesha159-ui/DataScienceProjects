# Military Power Clustering (Countries)

**Goal:** Cluster countries by capability profiles (budget, personnel, hardware proxies) using unsupervised learning.  
**Notebook:** `World Military Power.ipynb`

## Data
- Put the Excel file at: `data/World military power.xlsx`
- In the notebook, it’s read with:  
  ```python
  df = pd.read_excel('data/World military power.xlsx', header=1)
