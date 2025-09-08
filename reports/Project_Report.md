# Telco Customer Churn — Project Report

## 1. Business goal
Predict which customers will churn so a team can contact them and reduce revenue loss.

## 2. Dataset
Public “Telco Customer Churn” (Kaggle). Target column: `Churn` (Yes/No).  
Important features: tenure, contract type, monthly charges, services.

### Exploration findings
- No missing values detected across all 21 columns.
- Dataset has 7,043 customers (rows).
- Target variable (`Churn`) is imbalanced:
  - No: ~73% of customers
  - Yes: ~27% of customers
- This imbalance means accuracy alone is not a good metric; now i  will focus on ROC-AUC, Precision, Recall, and F1 instead.

### Numerical feature distributions
- **SeniorCitizen**: ~84% of customers are not senior, ~16% are senior.
- **Tenure**: Many customers are new (tenure < 12 months) or long-term (close to 70 months).
- **MonthlyCharges**: Two clusters appear — low (~$20) and higher ($70–$100).
- Implies different customer groups may churn for different reasons (new customers vs long-term).


### Categorical feature insights
- Contract type is the strongest churn signal: **month-to-month churn >> one-year > two-year**.
- **Electronic check** customers churn more than **auto-pay** (credit card/bank transfer).
- **PaperlessBilling** shows higher churn than mailed billing.
- Missing add-on services (**OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport**) correlate with higher churn.
- **InternetService**: fiber optic customers churn more than DSL or no internet.
- Weak/near-neutral effects: **gender, phone service, multiple lines, streaming**.
- **TotalCharges** will be treated as a numeric feature in modeling.




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


## Baseline Modeling Results

I trained a baseline Logistic Regression model using one-hot encoding and scaled numerical features.  
Evaluation on the test set gave the following results:

- **ROC-AUC**: 0.84
- **Accuracy**: 0.81
- **Churn (class = 1)**: Precision = 0.66, Recall = 0.56, F1 = 0.61
- **Non-Churn (class = 0)**: Precision = 0.85, Recall = 0.89, F1 = 0.87

### Interpretation
- The model is much better at predicting **non-churners** than churners.
- Recall for churn (56%) indicates some churners are missed.
- Still, ROC-AUC of 0.84 shows good separation ability.
- This baseline will be improved using more advanced models (e.g., Random Forest, XGBoost) and imbalance handling techniques (SMOTE, class weights).

### Random Forest Results

A Random Forest model with class-weight balancing was trained for comparison.

- **ROC-AUC**: 0.83
- **Accuracy**: 0.78
- **Churn (class = 1)**: Precision = 0.62, Recall = 0.48, F1 = 0.54
- **Non-Churn (class = 0)**: Precision = 0.83, Recall = 0.89, F1 = 0.86

### Interpretation
- Random Forest performs similarly to logistic regression in ROC-AUC.
- It achieves slightly higher precision for churn but recall remains low.
- Accuracy is lower than logistic regression, showing a trade-off.
- Logistic regression remains competitive and interpretable, but Random Forest adds value as a non-linear baseline.

### XGBoost Results

An XGBoost classifier was trained with class-weight balancing.  

- **ROC-AUC**: 0.83  
- **Accuracy**: 0.75  
- **Churn (class = 1)**: Precision = 0.53, Recall = 0.75, F1 = 0.62  
- **Non-Churn (class = 0)**: Precision = 0.89, Recall = 0.76, F1 = 0.82  

### Interpretation
- XGBoost achieves the **best recall for churn (75%)**, meaning it identifies more customers likely to churn.
- Precision for churn is lower (53%), so some non-churners are misclassified, but this trade-off is acceptable in churn prediction.
- Compared to logistic regression and random forest, XGBoost balances performance and recall better, making it a stronger candidate for business use.





