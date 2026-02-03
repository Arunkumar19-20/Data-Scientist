import streamlit as st
import joblib
import numpy as np
import os
import time


# =================================
# Page Config
# =================================
st.set_page_config(
    page_title="Titanic AI Dashboard",
    page_icon="📊",
    layout="wide"
)


# =================================
# Load Model
# =================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "titanic_rf_model.pkl")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# =================================
# Corporate Theme CSS
# =================================
st.markdown("""
<style>

/* Background */
.main {
    background-color: #f4f7fb;
}

/* Header */
.header {
    background: #0b3c5d;
    padding: 25px;
    border-radius: 12px;
    color: white;
    margin-bottom: 20px;
}

.header h1 {
    margin: 0;
    font-size: 36px;
}

.header p {
    margin: 0;
    color: #dbe7f1;
}

/* Cards */
.card {
    background: white;
    padding: 22px;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    margin-bottom: 18px;
}

/* KPI */
.kpi {
    text-align: center;
    padding: 15px;
}

.kpi h2 {
    color: #0b3c5d;
    margin: 0;
}

.kpi p {
    margin: 0;
    color: gray;
}

/* Result Badge */
.success {
    background: #e6f4ea;
    color: #137333;
    padding: 12px;
    border-radius: 8px;
    font-weight: bold;
}

.danger {
    background: #fce8e6;
    color: #c5221f;
    padding: 12px;
    border-radius: 8px;
    font-weight: bold;
}

/* Button */
.stButton>button {
    width: 100%;
    background: #0b3c5d;
    color: white;
    border-radius: 8px;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)


# =================================
# Header Section
# =================================
st.markdown("""
<div class="header">
    <h1>📊 Titanic Survival Prediction System</h1>
    <p>Enterprise Machine Learning Dashboard | Random Forest Model</p>
</div>
""", unsafe_allow_html=True)


# =================================
# KPI Section
# =================================
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown("""
    <div class="card kpi">
        <h2>85%</h2>
        <p>Model Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown("""
    <div class="card kpi">
        <h2>Random Forest</h2>
        <p>Algorithm</p>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown("""
    <div class="card kpi">
        <h2>Titanic</h2>
        <p>Dataset</p>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown("""
    <div class="card kpi">
        <h2>Live</h2>
        <p>Status</p>
    </div>
    """, unsafe_allow_html=True)


# =================================
# Layout
# =================================
left, center, right = st.columns([1.2, 2, 1.2])


# =================================
# Input Panel
# =================================
with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("👤 Passenger Information")

    pclass = st.selectbox("Ticket Class", [1,2,3])
    sex = st.radio("Gender", ["Male","Female"])
    age = st.slider("Age", 1, 80, 30)
    sibsp = st.number_input("Siblings/Spouse", 0, 8, 0)
    parch = st.number_input("Parents/Children", 0, 6, 0)
    fare = st.slider("Fare", 0.0, 520.0, 50.0)
    embarked = st.selectbox("Embarked From", ["Southampton","Cherbourg","Queenstown"])

    predict = st.button("Run Prediction")

    st.markdown('</div>', unsafe_allow_html=True)


# =================================
# Data Processing
# =================================
sex_val = 0 if sex == "Male" else 1
emb_val = {"Southampton":0,"Cherbourg":1,"Queenstown":2}[embarked]

input_data = np.array([[pclass, sex_val, age, sibsp, parch, fare, emb_val]])


# =================================
# Prediction Panel
# =================================
with center:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Prediction Analysis")

    if predict:

        with st.spinner("Processing data..."):
            time.sleep(1.2)

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]

        survive = round(prob[1]*100,2)
        death = round(prob[0]*100,2)

        if pred == 1:
            st.markdown(f"""
            <div class="success">
                ✅ Passenger Likely to Survive ({survive}%)
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="danger">
                ❌ Passenger Unlikely to Survive ({death}%)
            </div>
            """, unsafe_allow_html=True)

        st.metric("Survival Probability", f"{survive}%")
        st.metric("Risk Probability", f"{death}%")

        st.bar_chart({
            "Survival": survive,
            "Death": death
        })

    else:
        st.info("Enter passenger data and click Run Prediction")

    st.markdown('</div>', unsafe_allow_html=True)


# =================================
# Info Panel
# =================================
with right:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📌 System Overview")

    st.write("""
    ✔ Machine Learning Model  
    ✔ Random Forest Classifier  
    ✔ Trained on Titanic Dataset  
    ✔ Real-time Prediction  
    ✔ Enterprise Dashboard UI  
    """)

    st.write("👨‍💻 Developer: Arun Kumar")
    st.write("📅 Year: 2026")

    st.markdown('</div>', unsafe_allow_html=True)


# =================================
# Footer
# =================================
st.markdown("""
<hr>
<center style="color:gray;">
© 2026 Titanic AI System | Enterprise Edition
</center>
""", unsafe_allow_html=True)