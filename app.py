import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Enterprise Churn Intelligence", page_icon="📈", layout="wide")

st.title("📈 Customer Churn Risk & Retention Intelligence")
st.write("Predict churn risk scores, uncover primary risk drivers via SHAP, and receive retention strategy recommendations.")

@st.cache_resource
def load_artifacts():
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler, encoders, feature_names

model, scaler, encoders, feature_names = load_artifacts()

# Sidebar Input Form
st.sidebar.header("📋 Customer Profile Settings")

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 6)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 18.0, 150.0, 85.0)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

# Feature Calculations
total_charges = tenure * monthly_charges
avg_cost_month = total_charges / tenure if tenure > 0 else monthly_charges
services_list = [online_security, tech_support]
total_services = sum([1 for s in services_list if s == 'Yes'])
has_protection = 1 if (online_security == 'Yes' or tech_support == 'Yes') else 0

input_dict = {
    'gender': 'Male', 'SeniorCitizen': 0, 'Partner': partner, 'Dependents': dependents,
    'tenure': tenure, 'PhoneService': 'Yes', 'MultipleLines': 'No',
    'InternetService': internet_service, 'OnlineSecurity': online_security,
    'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': tech_support,
    'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': contract,
    'PaperlessBilling': paperless_billing, 'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
    'AvgCostPerMonth': avg_cost_month, 'TotalServices': total_services,
    'HasProtection': has_protection
}

input_df = pd.DataFrame([input_dict])

# Encode Categoricals
for col, le in encoders.items():
    if col in input_df.columns:
        input_df[col] = le.transform(input_df[col])

input_df = input_df[feature_names]

# Main Dashboard Layout
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("🎯 Risk Assessment")
    churn_proba = model.predict_proba(input_df)[0][1] * 100
    
    st.metric(label="Churn Probability Score", value=f"{churn_proba:.1f}%")
    
    if churn_proba >= 65:
        st.error("🚨 **High Risk Tier** - Immediate Action Required")
    elif churn_proba >= 35:
        st.warning("⚠️ **Medium Risk Tier** - Monitor & Engage")
    else:
        st.success("✅ **Low Risk Tier** - Healthy Account")

    st.subheader("💡 Retention Strategy")
    if contract == "Month-to-month" and churn_proba > 40:
        st.write("• **Offer Incentive:** Transition to 1-Year Contract with a 15% discount.")
    if internet_service == "Fiber optic" and tech_support == "No":
        st.write("• **Bundle Tech Support:** Add complimentary 3-month Tech Support package.")
    if churn_proba <= 40:
        st.write("• **Upsell Target:** Eligible for long-term loyalty reward program.")

with col_right:
    st.subheader("🔍 Explainable AI (Feature Impact)")
    
    # SHAP Explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_df)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)