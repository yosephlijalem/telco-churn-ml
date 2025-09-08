# Model Card — Telco Churn Prediction

## Intended Use
Help a telecom business identify customers at risk of churning, so they can be targeted with retention campaigns.

## Data
- Dataset: Telco Customer Churn (public Kaggle dataset)
- Target: `Churn` (Yes/No)
- Features: customer demographics, contract type, tenure, monthly charges, services used

## Training
- Models: Logistic Regression (baseline), XGBoost (main)
- Validation: Stratified 5-fold cross-validation
- Metrics: ROC-AUC, Precision, Recall, F1-score

## Explainability
- Feature importance: SHAP global beeswarm plot
- Local explanations: SHAP force plot for individual customers

## Limitations
- Dataset may not generalize to all telecom providers
- Cost assumptions for threshold tuning are business-dependent
- Bias risk: demographic features may create unintended disparities

## Next Steps
- Add cost-sensitive evaluation
- Test with real business data
- Package model for deployment (API)
