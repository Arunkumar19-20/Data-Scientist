import os
import streamlit as st
import joblib
import numpy as np


# -----------------------------
# Load Model Safely
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "knn_model.pkl")

knn = joblib.load(model_path)


# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="Cancer Diagnosis Predictor",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------

st.title("🩺 Cancer Diagnosis Prediction (KNN)")
st.write("Enter patient details to predict diagnosis")


# -----------------------------
# User Inputs
# -----------------------------

bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)

smoking = st.selectbox("Smoking (0 = No, 1 = Yes)", [0, 1])

genetic_risk = st.selectbox("Genetic Risk (0 = No, 1 = Yes)", [0, 1])

physical_activity = st.number_input(
    "Physical Activity (hours/week)", min_value=0.0, max_value=50.0, step=0.1
)

alcohol_intake = st.number_input(
    "Alcohol Intake (units/week)", min_value=0.0, max_value=50.0, step=0.1
)

cancer_history = st.selectbox("Family Cancer History (0 = No, 1 = Yes)", [0, 1])


# -----------------------------
# Predict Button
# -----------------------------

if st.button("Predict Diagnosis"):

    # Combine input
    user_data = np.array([[
        bmi,
        smoking,
        genetic_risk,
        physical_activity,
        alcohol_intake,
        cancer_history
    ]])

    # Predict (NO scaling)
    prediction = knn.predict(user_data)[0]

    # Result
    if prediction == 1:
        st.error("⚠️ High Risk: Cancer Detected")
    else:
        st.success("✅ Low Risk: No Cancer Detected")


# -----------------------------
# Footer
# -----------------------------

st.markdown("---")
st.caption("Developed by Arun | KNN ML Project 🚀")