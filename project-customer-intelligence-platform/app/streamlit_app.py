import streamlit as st
import pandas as pd

st.title("📊 Customer Intelligence Dashboard")

uploaded_file = st.file_uploader("Upload CSV")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())
    st.write("Shape:", df.shape)
