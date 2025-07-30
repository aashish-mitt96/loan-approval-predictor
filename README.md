# 🏦 Loan Approval Prediction App

A machine learning-powered web application built using **Streamlit** to predict whether a loan application will be approved or rejected based on user input like income, loan amount, CIBIL score, and assets.

## 📌 Features

- 🔢 Predicts loan approval using a trained ML model (`loan_model.pkl`)
- 📈 Inputs include annual income, CIBIL score, loan amount, and asset values
- 📊 Automatically calculates financial ratios like debt-to-income and loan-to-asset
- ✅ Provides instant approval or rejection feedback
- 🧠 Trained on real-world loan data
- 💡 User-friendly UI built with Streamlit

---

## 🧠 Model Details

- **Algorithm**: Logistic Regression
- **Preprocessing**: Scaler used to normalize features (loaded via `scaler.pkl`)
- **Input Features**:
  - Number of Dependents
  - Education Level
  - Self Employment Status
  - Annual Income
  - Loan Amount
  - Loan Term
  - CIBIL Score
  - Residential, Commercial, Luxury, and Bank Assets
  - Debt-to-Income Ratio
  - Loan-to-Assets Ratio
