# 📉 Customer Churn Prediction with PySpark (GBTClassifier)

This project predicts **customer churn** using **Apache Spark’s MLlib** and a **Gradient Boosted Tree (GBTClassifier)** model.  
It is designed to handle large datasets efficiently and can be deployed with a Streamlit interface for real-time predictions.

---

## 🧠 Overview

Customer churn prediction helps companies identify which customers are likely to discontinue their services.  
By analyzing customer behavior, billing information, and contract details, this system builds a scalable predictive model that enables proactive customer retention strategies.

This project leverages **PySpark** for distributed data processing and **Streamlit** for an interactive user interface.

---

## ⚙️ Key Features

✅ End-to-end machine learning pipeline using **PySpark MLlib**  
✅ Scalable training on large datasets  
✅ Automatic categorical encoding & feature vector assembly  
✅ Gradient Boosted Tree classifier for high performance  
✅ Evaluation using ROC-AUC and accuracy metrics  ]

---

## 🧰 Tech Stack

**Languages & Frameworks**
- Python 3.x
- Apache Spark (PySpark 4.0.1)
- Streamlit

**Libraries Used**
- pyspark  
- findspark  
- pandas  
- numpy  
- matplotlib, seaborn  
- joblib (for saving additional files, optional)

---
## 📈 Example Results
---
| Model                     | ROC-AUC  | Accuracy | 
| ------------------------- | -------- | -------- | 
| Logistic Regression       | 0.83     | 0.81     | 
| Random Forest             | 0.84     | 0.84     | 
| **GBTClassifier (final)** | **0.86** | **0.87** | 

The GBT model achieved the best ROC-AUC and generalization.

## 🚀 How to Run
```bash
# Option A: conda
conda env create -f environment.yml
conda activate kc-prices
jupyter lab

# Option B: pip
pip install -r requirements.txt
jupyter lab

