import streamlit as st
import tempfile
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.qa_chain import load_resume_text, build_qa_chain

st.set_page_config(page_title="Resume Q&A Bot", layout="centered")
st.title("Resume Q&A Bot")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        resume_path = tmp_file.name

    st.success("Resume uploaded successfully ")
    resume_text = load_resume_text(resume_path)
    qa_chain = build_qa_chain(resume_text)

    question = st.text_input("Ask a question about the resume:")
    if question:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(question)
            st.write("🧠 Answer:", answer)

