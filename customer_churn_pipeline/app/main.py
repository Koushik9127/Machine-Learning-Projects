
import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = "models/model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Please train first.")
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("📊 Customer Churn Prediction")

if model:
    st.write("Enter customer details:")

    # Example schema – adjust according to your dataset
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    monthly = st.number_input("Monthly Charges", 0.0, 500.0, 70.0)
    tenure = st.slider("Tenure (months)", 0, 72, 12)

    if st.button("Predict"):
        df = pd.DataFrame(
            [[gender, senior, monthly, tenure]],
            columns=["gender", "SeniorCitizen", "MonthlyCharges", "tenure"]
        )
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0][1]
        st.success(f"Prediction: {'Churn' if pred else 'No Churn'}  (Probability: {proba:.2%})")
