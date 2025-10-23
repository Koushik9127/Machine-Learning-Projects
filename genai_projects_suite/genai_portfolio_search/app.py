import streamlit as st
import os

st.title("🔍 GenAI Portfolio README Search")
query = st.text_input("Search across project READMEs")
if query:
    st.write(f"Searching for '{query}' ... (mock output)")