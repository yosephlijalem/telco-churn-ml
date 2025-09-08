# telco-churn-ml
Predicting customer churn with ML (XGBoost, SHAP, explainability)
# Telco Customer Churn Project

This project predicts whether a telecom customer will leave (churn) using machine learning.  
The focus is not only on accuracy but also on understanding why customers churn through explainable AI methods.

---

## Project Goals
- Predict churn with machine learning models.
- Compare a baseline (Logistic Regression) with a stronger model (XGBoost).
- Use SHAP values to explain which factors drive churn.
- Present results with clear visuals (confusion matrix, ROC curve, feature importance).

---

## Project Structure
- `data/` → raw and processed datasets  
- `notebooks/` → Jupyter notebooks for analysis and modeling  
- `src/` → Python scripts for data prep and training  
- `reports/` → figures and final PDF report  
- `models/` → saved ML models  
- `scripts/` → helper scripts (e.g., dataset download)  

---

## Techniques Used
- Data cleaning and preprocessing
- Logistic Regression (baseline model)
- Gradient Boosting with XGBoost
- Cross-validation for reliable metrics
- Metrics: ROC-AUC, F1, Precision/Recall
- Explainability with SHAP
