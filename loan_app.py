import streamlit as st
import numpy as np
import joblib as dump

# Load model and scaler
model = dump.load("loan_model.pkl")
scaler = dump.load("scaler.pkl")

st.set_page_config(page_title="Loan Prediction App", page_icon="🏦", layout="centered")

st.title("🏦 Loan Approval Prediction App")
st.markdown("Fill in the details below to check your **loan eligibility**.")

st.markdown("---")
st.header("👤 Applicant Details")

col1, col2 = st.columns(2)
with col1:
    no_of_dependents = st.slider("Number of Dependents", 0, 10, 1)
    education = st.selectbox("Education", ("Not Graduate", "Graduate"))
    self_employed = st.selectbox("Self Employed", ("No", "Yes"))
with col2:
    income_annum = st.number_input("Annual Income (₹)", min_value=0, format="%d")
    cibil_score = st.slider("CIBIL Score", 300, 900, 650)

st.markdown("---")
st.header("💰 Loan Details")

col3, col4 = st.columns(2)
with col3:
    loan_amount = st.number_input("Loan Amount (₹)", min_value=0, format="%d")
    loan_term = st.number_input("Loan Term (in months)", min_value=1)
with col4:
    res_assets = st.number_input("Residential Asset Value (₹)", min_value=0, format="%d")
    comm_assets = st.number_input("Commercial Asset Value (₹)", min_value=0, format="%d")

lux_assets = st.number_input("Luxury Asset Value (₹)", min_value=0, format="%d")
bank_assets = st.number_input("Bank Asset Value (₹)", min_value=0, format="%d")

st.markdown("---")

# Submit Button
if st.button("🚀 Submit"):
    # Encoding categorical fields
    education_val = 1 if education == "Graduate" else 0
    self_employed_val = 1 if self_employed == "Yes" else 0

    total_assets = res_assets + comm_assets + lux_assets + bank_assets
    debt_to_income = loan_amount / (income_annum + 1)
    loan_to_assets = loan_amount / (total_assets + 1)

    input_data = np.array([
        no_of_dependents,
        education_val,
        self_employed_val,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        total_assets,
        debt_to_income,
        loan_to_assets
    ]).reshape(1, -1)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.markdown("---")
    if prediction == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Rejected.")
