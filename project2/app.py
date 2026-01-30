import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os


# =========================
# Load Model Safely
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "desics.pkl")

model = joblib.load(MODEL_PATH)


# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction App")
st.write("Enter Patient Details to Predict 10-Year Heart Disease Risk")


# =========================
# Input Fields
# =========================

male = st.selectbox("Gender (1 = Male, 0 = Female)", [0, 1])

age = st.number_input("Age", min_value=1, max_value=120, value=40)

currentSmoker = st.selectbox("Current Smoker (1 = Yes, 0 = No)", [0, 1])

cigsPerDay = st.number_input("Cigarettes Per Day", min_value=0, max_value=100, value=0)

BPMeds = st.selectbox("BP Medicine (1 = Yes, 0 = No)", [0, 1])

prevalentStroke = st.selectbox("Previous Stroke (1 = Yes, 0 = No)", [0, 1])

prevalentHyp = st.selectbox("Hypertension (1 = Yes, 0 = No)", [0, 1])

diabetes = st.selectbox("Diabetes (1 = Yes, 0 = No)", [0, 1])

totChol = st.number_input("Total Cholesterol", min_value=100, max_value=500, value=200)

sysBP = st.number_input("Systolic BP", min_value=80, max_value=250, value=120)

diaBP = st.number_input("Diastolic BP", min_value=50, max_value=150, value=80)

BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0)

heartRate = st.number_input("Heart Rate", min_value=40, max_value=200, value=75)

glucose = st.number_input("Glucose Level", min_value=50, max_value=400, value=100)


# =========================
# Create Input DataFrame
# =========================

input_data = pd.DataFrame([[
    male,
    age,
    currentSmoker,
    cigsPerDay,
    BPMeds,
    prevalentStroke,
    prevalentHyp,
    diabetes,
    totChol,
    sysBP,
    diaBP,
    BMI,
    heartRate,
    glucose
]], columns=[

    'male',
    'age',
    'currentSmoker',
    'cigsPerDay',
    'BPMeds',
    'prevalentStroke',
    'prevalentHyp',
    'diabetes',
    'totChol',
    'sysBP',
    'diaBP',
    'BMI',
    'heartRate',
    'glucose'
])


# =========================
# Ensure No NaN Values
# =========================

input_data = input_data.astype(float)
input_data.fillna(0, inplace=True)


# =========================
# Prediction
# =========================

if st.button("Predict"):

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.write(f"Risk Probability: {probability[0][1]*100:.2f}%")


# =========================
# Footer
# =========================

st.markdown("---")
st.markdown("Developed by Arun | Heart Disease ML Predictor")