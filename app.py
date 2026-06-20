import streamlit as st

st.set_page_config(page_title="AI Study Assistant", layout="wide")

st.title("AI Student Study Assistant")
st.write("Upload lecture notes or PDFs and ask questions from them.")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully.")

question = st.text_input("Ask a question from your uploaded notes")

if question:
    st.write("Question:", question)