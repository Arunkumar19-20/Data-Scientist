import streamlit as st
import joblib
import numpy as np
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Interest Prediction",
    page_icon="🎯",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

# -----------------------------
# Reverse Map
# -----------------------------
reverse_map = {
    1: "Animation",
    2: "Action",
    3: "Drama"
}

# -----------------------------
# Title
# -----------------------------
st.title("🎯 Interest Prediction App")
st.divider()

# -----------------------------
# Inputs (ONLY 2 FEATURES)
# -----------------------------

age = st.number_input("Age", min_value=5, max_value=100, value=25)

gender = st.selectbox("Gender", ["Male", "Female"])

# Encode gender (same as training!)
gender_val = 1 if gender == "Male" else 0

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict 🚀"):

    # EXACTLY 2 features
    input_data = np.array([[age, gender_val]])

    prediction = model.predict(input_data)[0]

    pred_int = int(round(prediction))

    result = reverse_map.get(pred_int, "Unknown")

    st.success(f"✅ Predicted Interest: **{result}**")