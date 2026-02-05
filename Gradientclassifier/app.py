import streamlit as st
import numpy as np
import joblib


# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Mushroom AI",
    page_icon="🍄",
    layout="wide"
)


# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    return joblib.load("deci.pkl")

model = load_model()


# =========================
# Custom CSS (FIXED)
# =========================
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}

/* Keep natural padding */
.block-container {
    max-width: 100% !important;
    padding: 2rem 3rem !important;
}

/* Header */
.app-title {
    font-size: 44px;
    font-weight: 700;
    color: white;
    margin-bottom: 4px;
}

.app-sub {
    color: #cbd5e1;
    margin-bottom: 25px;
}

/* Main Card */
.main-card {
    width: 100%;
    background: rgba(255,255,255,0.12);
    padding: 35px 40px;
    border-radius: 20px;
    backdrop-filter: blur(14px);
    box-shadow: 0px 15px 40px rgba(0,0,0,0.35);
}

/* Section Title */
.section-title {
    color: #ffffff;
    font-size: 22px;
    margin: 25px 0 12px 0;
    border-bottom: 1px solid rgba(255,255,255,0.25);
    padding-bottom: 6px;
}

/* Labels */
label {
    color: #f8fafc !important;
    font-weight: 500;
}

/* Inputs */
.stNumberInput input,
.stSelectbox select {
    background: #ffffff !important;
    color: #000000 !important;
    border-radius: 6px;
}

/* Button */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg,#ff512f,#dd2476);
    color: white;
    font-size: 20px;
    font-weight: 600;
    padding: 14px;
    border-radius: 12px;
    border: none;
    margin-top: 20px;
}

/* Result */
.result-box {
    margin-top: 25px;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 24px;
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

/* Footer */
.footer {
    text-align: center;
    color: #cbd5e1;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)


# =========================
# Header
# =========================
st.markdown("""
<div class="app-title">🍄 Mushroom Classification System</div>
<div class="app-sub">AI Powered Prediction Platform</div>
""", unsafe_allow_html=True)


# =========================
# Main Card
# =========================
#st.markdown('<div class="main-card">', unsafe_allow_html=True)


# =========================
# Form
# =========================
with st.form("predict_form"):

    st.markdown('<div class="section-title">📏 Measurements</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        cap_diameter = st.number_input("Cap Diameter", 0.0, 50.0, step=0.1)

    with c2:
        stem_height = st.number_input("Stem Height", 0.0, 50.0, step=0.1)

    with c3:
        stem_width = st.number_input("Stem Width", 0.0, 20.0, step=0.1)


    st.markdown('<div class="section-title">🧬 Biological Features</div>', unsafe_allow_html=True)

    c4, c5 = st.columns(2)

    with c4:
        cap_shape = st.selectbox(
            "Cap Shape",
            ["bell","conical","convex","flat","sunken","spherical"]
        )

        gill_attachment = st.selectbox(
            "Gill Attachment",
            ["free","attached","descending","notched"]
        )

        gill_color = st.selectbox(
            "Gill Color",
            ["white","brown","yellow","pink","gray","black"]
        )


    with c5:
        stem_color = st.selectbox(
            "Stem Color",
            ["white","brown","yellow","gray","black"]
        )

        season = st.selectbox(
            "Season",
            ["spring","summer","autumn","winter"]
        )


    submit = st.form_submit_button("🔍 Predict Mushroom")


# Close main card
st.markdown('</div>', unsafe_allow_html=True)


# =========================
# Encoding
# =========================
cap_shape_map = {
    "bell":0,"conical":1,"convex":2,"flat":3,"sunken":4,"spherical":5
}

gill_attach_map = {
    "free":0,"attached":1,"descending":2,"notched":3
}

gill_color_map = {
    "white":0,"brown":1,"yellow":2,"pink":3,"gray":4,"black":5
}

stem_color_map = {
    "white":0,"brown":1,"yellow":2,"gray":3,"black":4
}

season_map = {
    "spring":0,"summer":1,"autumn":2,"winter":3
}


# =========================
# Prediction
# =========================
if submit:

    X = np.array([[
        cap_diameter,
        cap_shape_map[cap_shape],
        gill_attach_map[gill_attachment],
        gill_color_map[gill_color],
        stem_height,
        stem_width,
        stem_color_map[stem_color],
        season_map[season]
    ]])


    pred = model.predict(X)[0]


    if pred == 1:

        st.markdown(
            '<div class="result-box good">✅ Edible Mushroom</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="result-box bad">☠️ Poisonous Mushroom</div>',
            unsafe_allow_html=True
        )


# =========================
# Footer
# =========================
st.markdown("""
<div class="footer">
© 2026 | Mushroom AI by Arun
</div>
""", unsafe_allow_html=True)
