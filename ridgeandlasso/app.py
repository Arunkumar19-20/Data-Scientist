import streamlit as st
import numpy as np
import joblib

# =====================================
# Page Config
# =====================================
st.set_page_config(
    page_title="Health Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# =====================================
# Custom CSS (Premium UI)
# =====================================
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
}

h1 {
    color: white;
    text-align: center;
}

.stNumberInput label, .stSlider label, .stSelectbox label {
    font-weight: 600 !important;
}

.predict-btn button {
    width: 100%;
    height: 3.2em;
    font-size: 18px;
    border-radius: 12px;
    background: linear-gradient(to right, #11998e, #38ef7d);
    color: white;
    border: none;
}

.result-card {
    background-color: white;
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

# =====================================
# Title
# =====================================
st.title("🩺 Health Risk Score Predictor")
st.markdown("<br>", unsafe_allow_html=True)

# =====================================
# Load Model
# =====================================
model = joblib.load("ridge.pkl")

# =====================================
# Layout Columns
# =====================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Patient Details")

    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0)
    blood_pressure = st.number_input("Blood Pressure", min_value=60, max_value=250, value=120)
    cholesterol = st.number_input("Cholesterol", min_value=100, max_value=500, value=180)
    glucose = st.number_input("Glucose", min_value=50, max_value=400, value=100)
    insulin = st.number_input("Insulin", min_value=0.0, max_value=500.0, value=80.0)

with col2:
    st.subheader("❤️ Lifestyle Information")

    heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=72)
    activity_level = st.slider("Activity Level (1-10)", 1, 10, 5)
    diet_quality = st.slider("Diet Quality (1-10)", 1, 10, 5)

    smoking_status = st.selectbox(
        "Smoking Status",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    alcohol_intake = st.slider("Alcohol Intake (0-10)", 0, 10, 2)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================
# Prediction Button
# =====================================
if st.button("🔍 Predict Health Risk Score"):

    input_data = np.array([[
        age,
        bmi,
        blood_pressure,
        cholesterol,
        glucose,
        insulin,
        heart_rate,
        activity_level,
        diet_quality,
        smoking_status,
        alcohol_intake
    ]])

    prediction = model.predict(input_data)[0]

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
        <div class="result-card">
            <h2>📊 Predicted Health Risk Score</h2>
    """, unsafe_allow_html=True)

    st.markdown(
        f"<h1 style='color:#2c5364;'>{prediction:.2f}</h1>",
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # Risk Interpretation
    if prediction < 30:
        st.success("🟢 Low Health Risk")
    elif prediction < 70:
        st.warning("🟡 Moderate Health Risk")
    else:
        st.error("🔴 High Health Risk")

