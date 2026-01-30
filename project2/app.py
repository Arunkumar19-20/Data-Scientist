import streamlit as st
import joblib
import pandas as pd
import numpy as np


# =========================
# Load Model
# =========================

model = joblib.load("desics.pkl")


# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction App")
st.write("Enter Patient Details")


# =========================
# Input Fields
# =========================

male = st.selectbox("Male (1 = Yes, 0 = No)", [0, 1])

age = st.number_input("Age", 1, 120, 40)

currentSmoker = st.selectbox("Current Smoker (1 = Yes, 0 = No)", [0, 1])

cigsPerDay = st.number_input("Cigarettes Per Day", 0, 100, 0)

BPMeds = st.selectbox("BP Medicine (1 = Yes, 0 = No)", [0, 1])

prevalentStroke = st.selectbox("Previous Stroke (1 = Yes, 0 = No)", [0, 1])

prevalentHyp = st.selectbox("Hypertension (1 = Yes, 0 = No)", [0, 1])

diabetes = st.selectbox("Diabetes (1 = Yes, 0 = No)", [0, 1])

totChol = st.number_input("Total Cholesterol", 100, 500, 200)

sysBP = st.number_input("Systolic BP", 80, 250, 120)

diaBP = st.number_input("Diastolic BP", 50, 150, 80)

BMI = st.number_input("BMI", 10.0, 60.0, 22.0)

heartRate = st.number_input("Heart Rate", 40, 200, 75)

glucose = st.number_input("Glucose", 50, 400, 100)


# =========================
# Create DataFrame
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
# Safety Check (No NaN)
# =========================

# Force convert to float
input_data = input_data.astype(float)

# Fill any accidental NaN with 0
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
st.markdown("Developed by Arun | Heart Disease ML App")