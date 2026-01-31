import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vector.pkl")

# Page title
st.set_page_config(page_title="Spam Detector", layout="centered")

st.title("📩 SMS Spam Detection App")
st.write("Enter a message to check whether it is **Spam** or **Not Spam (Ham)**")

# Text input
message = st.text_area("Enter your SMS message here:")

# Predict button
if st.button("Predict"):

    if message.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        # Transform input text
        data = vectorizer.transform([message])

        # Predict
        prediction = model.predict(data)[0]

        # Show result
        if prediction == "spam" or prediction == 1:
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is NOT Spam (Ham)")