import streamlit as st
import numpy as np
import joblib
import os
from PIL import Image


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Digit Recognition",
    page_icon="✍️",
    layout="centered"
)


# -----------------------------
# Load Model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Poly.pkl")   # Your model file


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except:
    st.error("❌ Model file not found!")
    st.stop()


# -----------------------------
# Title
# -----------------------------
st.title("✍️ Handwritten Digit Recognition")
st.write("Upload a digit image (0–9) to predict")

st.markdown("---")


# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)


# -----------------------------
# Preprocess (FIXED)
# -----------------------------
def preprocess(img):

    # Convert to grayscale
    img = img.convert("L")
    img = np.array(img)

    # Invert colors
    img = 255 - img

    # Remove noise
    img[img < 40] = 0

    # -------------------------
    # Crop digit (Bounding Box)
    # -------------------------
    coords = np.column_stack(np.where(img > 0))

    if coords.size == 0:
        return np.zeros((1, 64))

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    img = img[y0:y1+1, x0:x1+1]

    # -------------------------
    # Make square (Centering)
    # -------------------------
    h, w = img.shape
    size = max(h, w)

    square = np.zeros((size, size), dtype=np.uint8)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2

    square[y_offset:y_offset+h, x_offset:x_offset+w] = img

    # -------------------------
    # Resize to 8x8
    # -------------------------
    square = Image.fromarray(square).resize((8, 8))
    square = np.array(square)

    # -------------------------
    # Scale to 0–16
    # -------------------------
    square = (square / 255.0) * 16.0

    # Flatten
    square = square.reshape(1, -1)

    return square


# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width=200)

    if st.button("Predict"):

        with st.spinner("Predicting..."):

            data = preprocess(image)

            prediction = model.predict(data)[0]

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(data)[0]
                confidence = round(max(prob) * 100, 2)
            else:
                confidence = "N/A"


        st.markdown("---")

        st.success(f"✅ Predicted Digit: {prediction}")

        st.info(f"📊 Confidence: {confidence}%")


else:

    st.warning("Please upload an image first")


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Simple Digit Recognition App using Streamlit")