# AI Driven Business Risk Intelligence
# Customer Risk Assessment & Churn Prediction
## Project Report

---

## Chapter 1 — Introduction
### 1.1 Problem Statement
Businesses lose crores of rupees every year because
they discover customer risk and churn too late.

### 1.2 Objectives
1. Build multi dimensional customer risk scoring
2. Predict churn across 30/60/90 day horizons
3. Analyze customer sentiment from feedback
4. Provide explainable AI with SHAP values
5. Build What-If action simulator
6. Calculate revenue at risk in rupees

### 1.3 Scope
Applicable to Telecom, Banking, E-commerce industries

---

## Chapter 2 — Literature Review
### 2.1 Existing Systems
- Salesforce Einstein Analytics
- SAS Customer Intelligence
- IBM Watson Customer Experience

### 2.2 Limitations of Existing Systems
- Very expensive (lakhs per year)
- Not open source
- No What-If simulation
- No combined risk + churn view

### 2.3 Research Gap
No existing open source tool combines fraud risk,
churn prediction, sentiment analysis, and revenue
impact in one unified platform.

---

## Chapter 3 — System Design
### 3.1 Architecture
Data Pipeline → ML Models → Explainability →
Action Simulator → Revenue Calculator → Dashboard

### 3.2 Technology Stack
- Python 3.14
- XGBoost, Scikit-learn
- SHAP, LIME
- Streamlit, Plotly
- Pandas, NumPy

### 3.3 Dataset
- Telco Customer Churn (Kaggle) — 7,043 records
- Bank Customer Dataset (Kaggle)
- E-commerce Dataset (Kaggle)

---

## Chapter 4 — Implementation
### 4.1 Data Pipeline
- Data loading and cleaning
- Customer 360 profile building
- Feature engineering

### 4.2 Risk Scoring Engine
- 5 dimensional risk scoring
- Combined risk score formula

### 4.3 Churn Prediction
- Triple horizon prediction
- SMOTE for imbalanced data
- XGBoost models

### 4.4 Explainable AI
- SHAP values
- Auto English reports

### 4.5 What-If Simulator
- 7 business actions simulated
- Revenue impact calculated

### 4.6 Dashboard
- 6 page Streamlit dashboard
- Deployed on Streamlit Cloud

---

## Chapter 5 — Results
### 5.1 Model Performance
| Model | Accuracy | AUC-ROC |
|---|---|---|
| 30-Day Churn | 77.86% | 82.69% |
| 60-Day Churn | 77.50% | 83.23% |
| 90-Day Churn | 77.86% | 83.09% |

### 5.2 Business Impact
- Total customers analyzed: 7,043
- Critical risk customers: 952 (13.5%)
- Total revenue at risk: ₹17,74,373/year

---

## Chapter 6 — Conclusion
### 6.1 Summary
Successfully built an AI powered business risk
intelligence platform with 7 unique innovations.

### 6.2 Future Scope
- Real time social media integration
- Mobile app development
- Multi language support
- More industry adapters

---

## References
1. Kaggle Telco Customer Churn Dataset
2. XGBoost: A Scalable Tree Boosting System
3. SHAP: A Unified Approach to Explaining ML Models
4. Streamlit Documentation
5. Scikit-learn Documentation