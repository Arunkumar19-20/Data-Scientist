import streamlit as st
import joblib
import numpy as np
import cv2
from PIL import Image


# Load Model
model = joblib.load("digit_blur_model.pkl")


# Image Preprocessing (NO SCALER)
def preprocess_image(img):

    # Convert PIL → OpenCV
    img = np.array(img)

    # Convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize
    img = cv2.resize(img, (28, 28))

    # Normalize (0–1)
    img = img / 255.0

    # Flatten
    img = img.flatten().reshape(1, -1)

    return img


# UI
st.set_page_config(page_title="Digit Recognition")

st.title("✍️ Handwritten Digit Recognition")

st.write("Upload a clear or blurred digit image (0–9)")


# Upload Image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "png", "jpeg"]
)


if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width=200)

    # Preprocess
    img = preprocess_image(image)

    # Predict
    prediction = model.predict(img)[0]

    st.success(f"✅ Predicted Digit: {prediction}")