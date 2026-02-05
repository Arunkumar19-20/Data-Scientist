import os
import streamlit as st
import numpy as np
import joblib


# ====================================
# Page Config
# ====================================
st.set_page_config(
    page_title="Diabetes Prediction AI",
    page_icon="🩺",
    layout="wide"
)


# ====================================
# Load Model
# ====================================
@st.cache_resource
def load_model():

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "gb_model.pkl")

    return joblib.load(model_path)


model = load_model()


# ====================================
# Feature Labels (NO DUPLICATES)
# ====================================
FEATURES = {
    "bmi": "Body Mass Index (BMI)",

    "bp": "Blood Pressure (BP)",          # Real BP

    "s1": "Total Cholesterol (s1)",

    "s3": "HDL Cholesterol (s3)",

    "s4": "Cholesterol / HDL Ratio (s4)", # Not BP

    "s5": "Triglycerides (s5)",

    "s6": "Blood Sugar (s6)"
}


# ====================================
# Custom CSS
# ====================================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#020024,#090979,#00d4ff);
}

.block-container {
    padding: 2rem 3rem;
    max-width: 100%;
}

.app-title {
    font-size: 46px;
    font-weight: 700;
    color: white;
    text-align: center;
}

.app-sub {
    text-align: center;
    color: #e0f2fe;
    margin-bottom: 35px;
}

.main-card {
    background: rgba(255,255,255,0.12);
    padding: 35px;
    border-radius: 22px;
    backdrop-filter: blur(15px);
    box-shadow: 0px 15px 40px rgba(0,0,0,0.4);
}

.section-title {
    font-size: 22px;
    color: white;
    margin: 20px 0 12px;
    border-bottom: 1px solid rgba(255,255,255,0.3);
}

label {
    color: white !important;
    font-weight: 500;
}

.stNumberInput input {
    background: white !important;
    color: black !important;
    border-radius: 7px;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(90deg,#ff512f,#dd2476);
    color: white;
    font-size: 18px;
    font-weight: 600;
    padding: 14px;
    border-radius: 14px;
    border: none;
    margin-top: 25px;
}

.stButton>button:hover {
    transform: scale(1.03);
    transition: 0.2s;
}

.result-box {
    margin-top: 25px;
    padding: 22px;
    border-radius: 14px;
    text-align: center;
    font-size: 26px;
    font-weight: 700;
}

.good {
    background: rgba(34,197,94,0.25);
    color: #22c55e;
}

.bad {
    background: rgba(239,68,68,0.25);
    color: #ef4444;
}

.footer {
    text-align: center;
    color: #e0f2fe;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# ====================================
# Header
# ====================================
st.markdown("""
<div class="app-title">🩺 Diabetes Prediction System</div>
<div class="app-sub">AI Powered Health Risk Analyzer</div>
""", unsafe_allow_html=True)


# ====================================
# Main Card
# ====================================
#st.markdown('<div class="main-card">', unsafe_allow_html=True)


# ====================================
# Input Form
# ====================================
with st.form("prediction_form"):

    st.markdown(
        '<div class="section-title">📋 Enter Medical Details</div>',
        unsafe_allow_html=True
    )

    inputs = {}

    cols = st.columns(3)

    for i, (key, label) in enumerate(FEATURES.items()):

        with cols[i % 3]:
            inputs[key] = st.number_input(
                label,
                min_value=0.0,
                value=0.0,
                step=0.1
            )


    submit = st.form_submit_button("🔍 Predict Health Risk")


# Close Card
st.markdown('</div>', unsafe_allow_html=True)


# ====================================
# Prediction
# ====================================
if submit:

    input_data = np.array([[
        inputs[f] for f in FEATURES.keys()
    ]])


    prediction = model.predict(input_data)[0]


    st.divider()
    st.subheader("📊 Prediction Result")


    if prediction >= 150:

        st.markdown(
            f'<div class="result-box bad">⚠️ High Risk Detected<br>Score: {prediction:.2f}</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f'<div class="result-box good">✅ Low Risk<br>Score: {prediction:.2f}</div>',
            unsafe_allow_html=True
        )


# ====================================
# Footer
# ====================================
st.markdown("""
<div class="footer">
© 2026 | Diabetes AI System by Arun Kumar
</div>
""", unsafe_allow_html=True)
