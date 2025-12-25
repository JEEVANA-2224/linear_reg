import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ----------------- Page Config -----------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Predictor")
st.write("Predict the final score of a student based on their performance features.")

# ----------------- Load Model and Scaler -----------------
@st.cache_resource
def load_model():
    with open("student_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# ----------------- User Input -----------------
st.header("Enter Student Details")

# Replace these with your actual selected features
features = [
    "Hours_Studied",
    "Assignments_Score",
    "Attendance",
    "Participation",
    "Quiz1_Score",
    "Quiz2_Score",
    "Project_Score",
    "Midterm_Score",
    "Extra_Credit",
    "Sleep_Hours"
]

user_input = {}
for feature in features:
    user_input[feature] = st.number_input(f"{feature}:", value=0)

input_df = pd.DataFrame([user_input])

# ----------------- Predict Button -----------------
if st.button("Predict Final Score"):
    # Scale the input
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted Final Score: {prediction:.2f}")

# ----------------- Optional: Show User Input -----------------
if st.checkbox("Show Input Data"):
    st.write(input_df)
