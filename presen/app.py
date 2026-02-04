import streamlit as st
import joblib
import numpy as np
import os


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Credit Card Fraud Detection")


# -----------------------------
# Load Model Safely
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "svm_creditcard_model.pkl")


if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file 'svm_creditcard_model.pkl' not found in project folder!")
    st.stop()


model = joblib.load(MODEL_PATH)


# -----------------------------
# App UI
# -----------------------------
st.title("💳 Credit Card Fraud Detection (SVM)")
st.write("Enter transaction details to predict fraud")


st.subheader("Enter Transaction Details")


# Feature Inputs
Time = st.number_input(
    "Transaction Time (Seconds)",
    value=10000.0
)

Pattern_Score = st.number_input(
    "Transaction Pattern Score (V1)",
    value=0.0
)

Behavior_Index = st.number_input(
    "Spending Behavior Index (V2)",
    value=0.0
)

Risk_Level = st.number_input(
    "Risk Signal Level (V3)",
    value=0.0
)

Amount = st.number_input(
    "Transaction Amount (₹)",
    value=100.0
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    # Arrange input (same order as training)
    input_data = np.array([[
        Time,
        Pattern_Score,
        Behavior_Index,
        Risk_Level,
        Amount
    ]])

    try:
        prediction = model.predict(input_data)[0]

        if prediction == 0:
            st.success("✅ Transaction is NORMAL (Not Fraud)")
        else:
            st.error("🚨 Transaction is FRAUD")

    except Exception as e:
        st.error(f"Prediction failed: {e}")