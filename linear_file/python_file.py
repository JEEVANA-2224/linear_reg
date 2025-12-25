import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

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
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    st.error(f"BASE_DIR = {BASE_DIR}")
    st.error(f"FILES = {os.listdir(BASE_DIR)}")

    model_path = os.path.join(BASE_DIR, "student_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "student_scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("❌ Model or Scaler file not found in the app directory.")
        st.stop()

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


# 🔥 IMPORTANT: CALL THE FUNCTION HERE
model, scaler = load_model()

# ----------------- User Input -----------------
st.header("Enter Student Details")

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
    user_input[feature] = st.number_input(
        f"{feature}",
        min_value=0.0,
        value=0.0,
        step=1.0
    )

input_df = pd.DataFrame([user_input])

# ----------------- Predict Button -----------------
if st.button("Predict Final Score"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.success(f"🎯 Predicted Final Score: **{prediction:.2f}**")

# ----------------- Optional: Show User Input -----------------
if st.checkbox("Show Input Data"):
    st.write(input_df)
