# Telco Customer Churn — Project Report

## 1. Business goal
Predict which customers will churn so a team can contact them and reduce revenue loss.

## 2. Dataset
Public “Telco Customer Churn” (Kaggle). Target column: `Churn` (Yes/No).  
Important features: tenure, contract type, monthly charges, services.

## 3. Plan
- Explore data (types, missing values, class balance).
- Build a **baseline** (Logistic Regression).
- Build a **stronger model** (XGBoost).
- Evaluate with **ROC-AUC, Precision/Recall, F1**.
- Tune the **decision threshold** for business needs.
- Explain predictions with **SHAP** (global + per-customer).

## 4. Why these choices (short math/logic)
- **Logistic Regression baseline**: probability model with sigmoid  
  \( \sigma(z) = \frac{1}{1+e^{-z}} \)  
  Simple, interpretable starting point.
- **Cross-validation**: stratified K=5 to estimate generalization reliably.
- **XGBoost**: gradient boosting trees handle mixed features and non-linearities well.
- **Metrics**:  
  - Precision = TP/(TP+FP)  
  - Recall = TP/(TP+FN)  
  - F1 = 2·(Precision·Recall)/(Precision+Recall)  
  - ROC-AUC summarizes TPR vs FPR.
- **Threshold tuning**: choose probability cut-off \(p^*\) to balance retention cost vs. saved revenue.

## 5. Results (fill after training)
- CV ROC-AUC: **__** ± **__**
- Holdout: ROC-AUC **__**, F1 **__**, Recall@\(p^*\) **__**  
- Figures saved in `reports/figures/`: confusion matrix, ROC, SHAP.

## 6. Error analysis & next steps
Common FP/FN patterns, probability calibration, cost-based thresholding, deployment idea (API).
