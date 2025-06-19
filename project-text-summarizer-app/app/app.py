import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

@st.cache_resource
def load_model():
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    return model, tokenizer

model, tokenizer = load_model()

st.title("🧠 Text Summarizer App")
text_input = st.text_area("Enter text to summarize")

if st.button("Summarize"):
    inputs = tokenizer.encode("summarize: " + text_input, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.success(summary)
