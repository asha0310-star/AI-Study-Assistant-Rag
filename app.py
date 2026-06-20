import os
import tempfile

import streamlit as st

from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="AI Study Assistant", layout="wide")


# --- Helpers ---

def save_uploaded_files(uploaded_files):
    """Save uploaded PDFs to a temp folder and return that folder path."""
    temp_dir = tempfile.mkdtemp()
    for uf in uploaded_files:
        path = os.path.join(temp_dir, uf.name)
        with open(path, "wb") as f:
            f.write(uf.getbuffer())
    return temp_dir


@st.cache_resource
def get_pipeline():
    """Create the RAG pipeline once and reuse it across reruns."""
    return RAGPipeline()


# --- App state ---

pipeline = get_pipeline()

if "ingested" not in st.session_state:
    st.session_state.ingested = False


# --- UI ---

st.title("AI Student Study Assistant")
st.write("Upload lecture notes or PDFs and ask questions from them.")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded.")
    if st.button("Process files"):
        with st.spinner("Reading and indexing your notes..."):
            pdf_dir = save_uploaded_files(uploaded_files)
            pipeline.ingest_pdfs(pdf_dir)
            st.session_state.ingested = True
        st.success("Files processed. You can now ask questions.")


st.divider()

question = st.text_input("Ask a question from your uploaded notes")

if question:
    if not st.session_state.ingested:
        st.warning("Please upload and process some PDFs first.")
    else:
        with st.spinner("Searching your notes..."):
            answer = pipeline.query(question)
        st.subheader("Answer")
        st.write(answer)

        with st.expander("Generate a quiz from these notes"):
            if st.button("Make quiz"):
                quiz = pipeline.generate_quiz(question)
                for i, qa in enumerate(quiz, start=1):
                    st.markdown(f"**Q{i}.** {qa['question']}")
                    st.markdown(f"*Answer:* {qa['answer']}")