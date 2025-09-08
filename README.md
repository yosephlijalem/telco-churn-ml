# telco-churn-ml
Predicting customer churn with ML (XGBoost, SHAP, explainability)

## 📊 Results Summary

| Model                | ROC-AUC | Accuracy | Precision (Churn=1) | Recall (Churn=1) | F1 (Churn=1) |
|-----------------------|---------|----------|----------------------|------------------|--------------|
| Logistic Regression   | 0.84    | 0.81     | 0.66                 | 0.56             | 0.61         |
| Random Forest         | 0.83    | 0.78     | 0.62                 | 0.48             | 0.54         |
| XGBoost               | 0.83    | 0.75     | 0.53                 | **0.75**         | 0.62         |

- Logistic regression: Strong balanced baseline.  
- Random forest: More robust, but struggled with churn recall.  
- XGBoost: Best **recall (75%)**, making it the most effective for catching churners.  


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


## ⚙️ How to Run the Project

Follow these steps to reproduce my results:

1. **Clone this repository**  
   Open a terminal and run:
   ```bash
   git clone https://github.com/yosephlijalem/telco-churn-ml.git
   cd telco-churn-ml


2. **Install dependencies**  
   Make sure Python is installed, then run:
   ```bash
   pip install -r requirements.txt


3. **Launch Jupyter Notebook**  
   ```bash
   jupyter notebook





