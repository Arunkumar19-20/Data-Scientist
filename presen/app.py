import streamlit as st
import joblib
import numpy as np


# Load Model
model = joblib.load("svm_creditcard_model.pkl")


# App Settings
st.set_page_config(page_title="Credit Card Fraud Detection")

st.title("💳 Credit Card Fraud Detection (SVM)")
st.write("Enter transaction details to predict fraud")


# Input Features (Renamed)
st.subheader("Enter Transaction Details")

Time = st.number_input(
    "Transaction Time (Seconds)",
    value=10000.0
)

Pattern_Score = st.number_input(
    "Transaction Pattern Score",
    value=0.0
)

Behavior_Index = st.number_input(
    "Spending Behavior Index",
    value=0.0
)

Risk_Level = st.number_input(
    "Risk Signal Level",
    value=0.0
)

Amount = st.number_input(
    "Transaction Amount (₹)",
    value=100.0
)


# Predict Button
if st.button("Predict"):

    # Input in SAME order as training
    input_data = np.array([[
        Time,
        Pattern_Score,   # V1
        Behavior_Index,  # V2
        Risk_Level,      # V3
        Amount
    ]])

    # Predict
    prediction = model.predict(input_data)[0]

    # Result
    if prediction == 0:
        st.success("✅ Transaction is NORMAL (Not Fraud)")
    else:
        st.error("🚨 Transaction is FRAUD")