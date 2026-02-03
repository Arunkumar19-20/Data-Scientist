import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("iris_model")

# Page config
st.set_page_config(page_title="Iris Prediction App", page_icon="🌸")

st.title("🌸 Iris Flower Prediction (SVM Model)")
st.write("Enter flower measurements to predict the species.")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

# Predict button
if st.button("Predict 🌼"):

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = model.predict(input_data)[0]

    # Map output
    if prediction == 0:
        result = "Setosa 🌱"
    elif prediction == 1:
        result = "Versicolor 🌼"
    else:
        result = "Virginica 🌺"

    st.success(f"Prediction: {result}")