import streamlit as st
import numpy as np
import joblib
import os


# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Revenue Predictor",
    page_icon="📊",
    layout="centered"
)


# ---------------- New UI Theme ----------------
def set_new_theme():

    st.markdown("""
    <style>

    /* Background */
    .stApp {
        background: linear-gradient(
            120deg,
            #ff9a9e,
            #fad0c4,
            #a1c4fd,
            #c2e9fb
        );
        background-size: 300% 300%;
        animation: bgFlow 18s ease infinite;
    }

    @keyframes bgFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glass Card */
    section.main > div {
        background: rgba(255,255,255,0.88);
        backdrop-filter: blur(12px);
        padding: 32px;
        border-radius: 25px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.25);
    }

    /* Title */
    h1 {
        text-align: center;
        color: #2f2f4f;
        font-weight: 900;
        letter-spacing: 1px;
    }

    h3 {
        text-align: center;
        color: #5f6caf;
    }

    /* Input Boxes */
    input, select {
        border-radius: 12px !important;
        border: 2px solid #c2d6ff !important;
    }

    /* Button Animation */
    .stButton>button {
        background: linear-gradient(to right, #667eea, #ff758c);
        color: white;
        border-radius: 30px;
        height: 52px;
        width: 100%;
        font-size: 18px;
        font-weight: 700;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 8px 18px rgba(0,0,0,0.25);
        animation: pulse 2s infinite;
    }

    /* Button Hover */
    .stButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(to right, #43e97b, #38f9d7);
        box-shadow: 0 12px 28px rgba(0,0,0,0.35);
    }

    /* Button Click */
    .stButton>button:active {
        transform: scale(0.95);
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(102,126,234,0.6); }
        70% { box-shadow: 0 0 0 15px rgba(102,126,234,0); }
        100% { box-shadow: 0 0 0 0 rgba(102,126,234,0); }
    }

    /* Result Box */
    .result-box {
        background: linear-gradient(to right, #fbc2eb, #a6c1ee);
        padding: 22px;
        border-radius: 18px;
        text-align: center;
        font-size: 23px;
        font-weight: bold;
        color: #3a1c71;
        margin-top: 22px;
        animation: slideUp 0.5s ease;
    }

    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    </style>
    """, unsafe_allow_html=True)


# ---------------- Apply Theme ----------------
set_new_theme()


# ---------------- Title ----------------
st.markdown("<h1>📈 Revenue Forecast System</h1>", unsafe_allow_html=True)
st.markdown("<h3>Smart Business Analytics</h3>", unsafe_allow_html=True)

st.divider()


# ---------------- Load Model ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

try:
    model = joblib.load(MODEL_PATH)
except:
    st.error("❌ model.pkl not found in this folder")
    st.stop()


# ---------------- Input Section ----------------
st.subheader("📝 Enter Business Data")

col1, col2 = st.columns(2)

with col1:
    ad_spend = st.number_input(
        "💰 Ad Spend ($)",
        min_value=0.0,
        value=2500.0,
        step=100.0
    )

with col2:
    season = st.selectbox(
        "🌤️ Season",
        ["Monsoon", "Summer", "Winter"]
    )


# ---------------- Season Encoding ----------------
season_monsoon = 0
season_summer = 0
season_winter = 0

if season == "Monsoon":
    season_monsoon = 1
elif season == "Summer":
    season_summer = 1
elif season == "Winter":
    season_winter = 1


# ---------------- Predict ----------------
st.write("")
predict_btn = st.button("🚀 Predict Revenue Now")


# ---------------- Prediction ----------------
if predict_btn:

    input_data = np.array([
        [ad_spend, season_monsoon, season_summer, season_winter]
    ])

    try:
        prediction = model.predict(input_data)[0]

        st.markdown(f"""
        <div class="result-box">
            🎯 Estimated Revenue <br>
            💵 ${prediction:,.2f}
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")


# ---------------- Footer ----------------
st.divider()

st.markdown("""
<center>
🌟 Powered by Machine Learning & Streamlit <br>
Designed for Business Growth
</center>
""", unsafe_allow_html=True)