# app.py

import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("phish_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(page_title="Phishing Email Detector", page_icon="📧")

st.title("📧 Phishing / Spam Email Detector")
st.write("This AI model predicts whether an email message is phishing/spam or safe.")

user_input = st.text_area("Enter an email text:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform input and predict
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)[0]
        result = "🚨 Phishing / Spam Email Detected!" if prediction == 1 else "✅ This email seems safe."
        st.subheader(result)
