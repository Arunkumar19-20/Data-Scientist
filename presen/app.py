import streamlit as st
import numpy as np
import joblib
import os

# Page setup
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# Load Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))


# Feature names (from Credit Card Dataset)
feature_names = [
    "Transaction Time (seconds)",
    "Transaction Velocity",
    "Location Risk Score",
    "Merchant Risk Level",
    "Transaction Frequency",
    "Device Trust Score",
    "IP Risk Level",
    "Geo Distance",
    "Spending Pattern Score",
    "Amount Deviation",
    "User Behavior Score",
    "Transaction Consistency",
    "Account Age Impact",
    "Fraud History Score",
    "Stability Index",
    "Velocity Anomaly",
    "Usage Pattern Shift",
    "Login Risk Score",
    "Merchant Trust Index",
    "Device Change Score",
    "Session Risk",
    "Browser Fingerprint Score",
    "Purchase Type Risk",
    "Region Risk Index",
    "Spending Trend",
    "Credit Utilization",
    "Network Risk Score",
    "Anomaly Probability",
    "Transaction Amount"
]


# Title
st.title("💳 Credit Card Fraud Detection using SVM")

st.write("""
This application predicts whether a credit card transaction is
**Fraudulent or Normal** using a trained SVM model.
""")


# Input Section
st.header("📥 Enter Transaction Details")

inputs = []

for name in feature_names:
    value = st.number_input(
        name,
        value=0.0,
        format="%.6f"
    )
    inputs.append(value)


# Predict Button
if st.button("🔍 Predict"):

    # Convert to numpy array
    input_data = np.array(inputs).reshape(1, -1)

    # Prediction
    prediction = model.predict(input_data)[0]

    # Output
    st.header("📤 Prediction Result")

    if prediction == 1:
        st.error("⚠️ Fraud Transaction Detected")
    else:
        st.success("✅ Normal Transaction")


# Footer
st.markdown("---")
st.markdown("Developed using SVM & Streamlit")