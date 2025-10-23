import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from PyPDF2 import PdfReader

st.title("📄 Chat Over PDF")
pdf = st.file_uploader("Upload PDF", type="pdf")
if pdf:
    reader = PdfReader(pdf)
    text = "".join([page.extract_text() for page in reader.pages])
    embeddings = OpenAIEmbeddings()
    # Mock FAISS vectorstore usage
    st.success("PDF loaded and ready for QnA")
    query = st.text_input("Ask something about the PDF")
    if query:
        st.write("🔍 Answer coming soon...")