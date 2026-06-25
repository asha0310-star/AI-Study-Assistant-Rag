import streamlit as st
from dotenv import load_dotenv

from src.chunker import chunk_pages
from src.pdf_loader import extract_text_from_pdfs
from src.vector_store import search_chunks, store_chunks


load_dotenv()

st.set_page_config(
    page_title="AI Student Study Assistant RAG",
    page_icon="📚",
    layout="centered",
)

st.title("📚 AI Student Study Assistant RAG")
st.write(
    "Upload one or more PDF study files. This first MVP extracts text, "
    "counts the pages, and shows a short preview."
)

with st.sidebar:
    st.header("MVP Features")
    st.write("1. Upload PDF files")
    st.write("2. Extract text from each page")
    st.write("3. Show page and chunk counts")
    st.write("4. Store chunks in ChromaDB")
    st.write("5. Search retrieved chunks")
    st.info("Gemini answers will be added in a later version.")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    with st.spinner("Extracting text from your PDFs..."):
        pages, combined_text = extract_text_from_pdfs(uploaded_files)
        chunks = chunk_pages(pages)
        chunks_saved = store_chunks(chunks)

    st.success("PDFs processed successfully.")

    st.subheader("Extraction summary")
    st.write(f"Pages extracted: {len(pages)}")
    st.write(f"Chunks created: {len(chunks)}")
    st.write(f"Chunks saved to ChromaDB: {chunks_saved}")

    st.subheader("Uploaded files")
    for uploaded_file in uploaded_files:
        st.write(f"- {uploaded_file.name}")

    st.subheader("Text preview")
    if combined_text.strip():
        preview_text = combined_text[:1000]
        st.text_area(
            "First 1,000 characters",
            value=preview_text,
            height=250,
            disabled=True,
        )
    else:
        st.warning("No readable text was found in the uploaded PDF files.")

    st.subheader("Ask a question")
    question = st.text_input("Search your uploaded notes")

    if question:
        retrieved_chunks = search_chunks(question)

        st.subheader("Retrieved chunks")
        if retrieved_chunks:
            for index, chunk in enumerate(retrieved_chunks, start=1):
                metadata = chunk["metadata"]
                with st.expander(
                    f"Chunk {index}: {metadata['file_name']} "
                    f"page {metadata['page_number']}"
                ):
                    st.write(chunk["text"])
                    st.caption(f"Distance: {chunk['distance']:.4f}")
        else:
            st.warning("No chunks were found. Upload PDFs before searching.")
else:
    st.info("Upload one or more PDF files to begin.")
