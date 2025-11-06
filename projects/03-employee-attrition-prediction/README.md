# Employee Attrition Prediction-(Classification)
This project focuses on predicting employee attrition (whether an employee will leave the company or not) using machine learning techniques. It’s an end-to-end data science project that covers data preprocessing, visualization, model training, and performance evaluation.

## Project Overview
Employee attrition is one of the key challenges faced by organizations. Identifying employees who are likely to leave can help companies take proactive measures to improve retention.
In this notebook, we build and evaluate several machine learning models to predict attrition using employee-related data.

## Workflow
1. **Data Loading & Exploration**
   - Import dataset (CSV file)
   - Explore data types, missing values, and distributions
2. **Data Preprocessing**
   - Handle categorical and numerical features
   - Encode categorical variables
   - Normalize/scale features
   - Handle class imbalance using SMOTE (if applicable)
3. **Exploratory Data Analysis (EDA)**
   - Correlation analysis
   - Visualizations (attrition vs. various features)
4. **Model Development**
   - Train-Test Split  
   - Models used:
     - Logistic Regression  
     - Random Forest Classifier  
     - Decision Tree  
     - Support Vector Machine (SVM)
   - Hyperparameter tuning (GridSearchCV)
5. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-score  
   - Confusion Matrix  
   - ROC-AUC Curve
6. **Results & Insights**
   - Comparison of model performances  
   - Key features influencing attrition
     
## Sample Results

| Model                    | Accuracy  | Precision  | Recall  | F1-score  |
|--------------------------|-----------|------------|---------|-----------|
| Logistic Regression      | 84%       | 0.78       | 0.65    | 0.71      |
| Random Forest Classifier | 89%       | 0.82       | 0.74    | 0.78      |
| Decision Tree Classifier | 85%       | 0.79       | 0.68    | 0.73      |

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/ayesha159-ui/DataScienceProjects/projects/03-employee-attrition-prediction.git
   cd EmployeeAttritionPrediction
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
3. Launch the notebook
   ```bash
   jupyter notebook EmployeeAttritionPrediction.ipynb
4. Run all cells sequentially to reproduce results. 

