# app.py
import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load('spam_detector_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to predict spam
def predict_spam(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Streamlit application layout
st.title("Email Spam Detector")
st.write("This application classifies an email as Spam or Not Spam based on its content.")

# Input text area for the user to enter email text
input_text = st.text_area("Enter the email content")

# When the user clicks the button, show prediction
if st.button("Check for Spam"):
    if input_text:
        result = predict_spam(input_text)
        st.success(f"The email is: **{result}**")
    else:
        st.warning("Please enter some text to analyze.")
