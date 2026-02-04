import streamlit as st
import joblib
import numpy as np
import os


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Fraud Detection AI",
    page_icon="💳",
    layout="wide"
)


# -----------------------------
# Advanced CSS
# -----------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: radial-gradient(circle at top, #020617, #000000);
    color: white;
}


/* Main Title */
.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 900;
    background: linear-gradient(to right,#38bdf8,#22c55e,#38bdf8);
    -webkit-background-clip: text;
    color: transparent;
    animation: glow 3s infinite alternate;
}

@keyframes glow {
    from { text-shadow: 0 0 10px #38bdf8; }
    to { text-shadow: 0 0 25px #22c55e; }
}


/* Subtitle */
.sub-title {
    text-align: center;
    color: #cbd5f5;
    font-size: 18px;
    margin-bottom: 40px;
}


/* Glass Container */
.glass-box {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(18px);
    border-radius: 25px;
    padding: 40px;
    border: 2px solid rgba(56,189,248,0.4);
    box-shadow: 0 0 40px rgba(56,189,248,0.3);
    transition: all 0.4s ease;
}

.glass-box:hover {
    transform: translateY(-8px);
    box-shadow: 0 0 60px rgba(34,197,94,0.6);
}


/* Section Heading */
.section-title {
    font-size: 28px;
    font-weight: bold;
    margin-bottom: 25px;
    color: #38bdf8;
    border-left: 5px solid #22c55e;
    padding-left: 15px;
}


/* Button */
.stButton button {
    background: linear-gradient(90deg,#38bdf8,#22c55e);
    border: none;
    height: 60px;
    border-radius: 15px;
    font-size: 20px;
    font-weight: bold;
    color: black;
    transition: 0.3s;
}

.stButton button:hover {
    transform: scale(1.1);
    box-shadow: 0 0 25px #38bdf8;
}


/* Result Cards */
.safe-box {
    background: linear-gradient(135deg,#22c55e,#15803d);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
    box-shadow: 0 0 30px #22c55e;
}

.fraud-box {
    background: linear-gradient(135deg,#ef4444,#7f1d1d);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
    box-shadow: 0 0 30px #ef4444;
}


/* Inputs */
input {
    border-radius: 12px !important;
    background: #020617 !important;
    color: white !important;
    border: 1px solid #38bdf8 !important;
}


/* Input Labels */
label {
    color: #38bdf8 !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    text-shadow: 0 0 8px rgba(56,189,248,0.8);
}


/* Focus Effect */
input:focus {
    box-shadow: 0 0 12px #38bdf8 !important;
    border: 1px solid #22c55e !important;
}


/* Hover Effect */
input:hover {
    box-shadow: 0 0 10px rgba(56,189,248,0.6);
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Load Model Files
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "svm_creditcard_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "category_encoder.pkl"))


# -----------------------------
# Header
# -----------------------------
st.markdown('<h1 class="main-title">💳 Fraud Detection AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Next-Gen Secure Transaction Analyzer</p>', unsafe_allow_html=True)


# -----------------------------
# MAIN GLASS BOX
# -----------------------------
container = st.container()

with container:

    st.markdown("""
    <div class="glass-box">
        <div class="section-title">📝 Transaction Details</div>
    """, unsafe_allow_html=True)


    col1, col2 = st.columns(2)

    with col1:
        amt = st.number_input("💰 Amount (₹)", min_value=0.0, value=500.0)

        unix_time = st.number_input("⏱ Unix Time", value=1650000000)

        city_pop = st.number_input(
            "🏙 City Population",
            min_value=0,
            value=100000
        )

        category = st.selectbox(
            "🏪 Merchant Category",
            encoder.classes_
        )

    with col2:
        merch_lat = st.number_input("📍 Merchant Latitude", value=40.0)

        merch_long = st.number_input("📍 Merchant Longitude", value=-80.0)


    st.markdown("<br>", unsafe_allow_html=True)


    center = st.columns([1,2,1])

    with center[1]:
        predict = st.button("🚀 Detect Fraud", use_container_width=True)


    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Prediction
# -----------------------------
if predict:

    try:

        # Encode category
        cat_val = encoder.transform([category])[0]


        # Arrange input (MUST match training order)
        data = np.array([[
            amt,
            unix_time,
            city_pop,
            cat_val,
            merch_lat,
            merch_long
        ]])


        # Scale
        data = scaler.transform(data)


        with st.spinner("⚡ AI Analyzing Transaction..."):
            result = model.predict(data)[0]
            proba = model.predict_proba(data)[0][1] * 100


        st.markdown("<br>", unsafe_allow_html=True)


        if result == 1:

            st.markdown(f"""
            <div class="fraud-box">
                🚨 FRAUD DETECTED <br><br>
                🔴 Fraud Risk: {proba:.2f}%
            </div>
            """, unsafe_allow_html=True)

            st.warning("⚠ High risk transaction!")


        else:

            st.markdown(f"""
            <div class="safe-box">
                ✅ TRANSACTION SAFE <br><br>
                🔵 Fraud Risk: {proba:.2f}%
            </div>
            """, unsafe_allow_html=True)

            st.balloons()


    except Exception as e:

        st.error("❌ Prediction Error")
        st.write(e)


# -----------------------------
# Footer
# -----------------------------
st.markdown("<br><hr><br>", unsafe_allow_html=True)

st.caption("© 2026 Fraud Detection AI | Developed by Arun Kumar")