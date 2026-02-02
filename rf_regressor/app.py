import streamlit as st
import joblib
import numpy as np
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# -----------------------------
# Load Models
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

linear_path = os.path.join(BASE_DIR, "linear_model.pkl")
rf_path = os.path.join(BASE_DIR, "Randomforest.pkl")

@st.cache_resource
def load_models():
    linear_model = joblib.load(linear_path)
    rf_model = joblib.load(rf_path)
    return linear_model, rf_model


linear_model, rf_model = load_models()

# -----------------------------
# App Title
# -----------------------------
st.title("🏠 House Price Prediction App")
st.markdown("Predict house price using Machine Learning models")

st.divider()

# -----------------------------
# Model Selection
# -----------------------------
model_choice = st.selectbox(
    "Choose Model",
    ["Linear Regression", "Random Forest"]
)

# -----------------------------
# Input Fields
# -----------------------------
area = st.number_input(
    "Area (Square Feet)",
    min_value=100,
    max_value=10000,
    value=1000
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=2
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price 🚀"):

    # Input in same order as training
    input_data = np.array([[area, bedrooms]])

    # Select model
    if model_choice == "Linear Regression":
        prediction = linear_model.predict(input_data)[0]
    else:
        prediction = rf_model.predict(input_data)[0]

    # Display Result
    st.success(f"💰 Predicted House Price: ₹ {prediction:.2f} Lakhs")