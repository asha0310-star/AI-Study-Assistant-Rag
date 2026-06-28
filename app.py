import streamlit as st
from dotenv import load_dotenv

from src.chunker import chunk_pages
from src.pdf_loader import extract_text_from_pdfs
from src.rag_pipeline import answer_question, get_source_references
from src.summary_generator import SUMMARY_TYPES, generate_summary
from src.vector_store import search_chunks, store_chunks
from src.quiz_generator import generate_quiz_from_chunks, SUPPORTED_TYPES as QUIZ_TYPES


load_dotenv()


def initialize_session_state():
    """Create the session state values used by the app."""
    if "pages" not in st.session_state:
        st.session_state.pages = []
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "combined_text" not in st.session_state:
        st.session_state.combined_text = ""
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "processed_file_names" not in st.session_state:
        st.session_state.processed_file_names = []
    if "summary_text" not in st.session_state:
        st.session_state.summary_text = ""
    if "summary_type" not in st.session_state:
        st.session_state.summary_type = "Exam-focused summary"
    if "quiz" not in st.session_state:
        st.session_state.quiz = []
    if "quiz_type" not in st.session_state:
        st.session_state.quiz_type = "MCQ"
    if "quiz_n" not in st.session_state:
        st.session_state.quiz_n = 5


def get_uploaded_file_names(uploaded_files):
    """Return a simple list of uploaded file names."""
    return [uploaded_file.name for uploaded_file in uploaded_files]

st.set_page_config(
    page_title="AI Student Study Assistant RAG",
    page_icon="📚",
    layout="centered",
)

initialize_session_state()

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
    st.write("6. Generate a Gemini answer")
    st.write("7. Generate a summary")
    st.info("Answers use only the retrieved document chunks.")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    uploaded_file_names = get_uploaded_file_names(uploaded_files)

    st.subheader("Uploaded files")
    for uploaded_file_name in uploaded_file_names:
        st.write(f"- {uploaded_file_name}")

    if st.session_state.processed and st.session_state.processed_file_names != uploaded_file_names:
        st.session_state.processed = False
        st.session_state.pages = []
        st.session_state.chunks = []
        st.session_state.combined_text = ""
        st.session_state.processed_file_names = []

    process_clicked = st.button("Process PDFs")

    if process_clicked:
        with st.spinner("Extracting text from your PDFs..."):
            pages, combined_text = extract_text_from_pdfs(uploaded_files)
            chunks = chunk_pages(pages)
            store_chunks(chunks)

        st.session_state.pages = pages
        st.session_state.chunks = chunks
        st.session_state.combined_text = combined_text
        st.session_state.processed = True
        st.session_state.processed_file_names = uploaded_file_names

        st.success("PDFs processed successfully.")

    if st.session_state.processed:
        pages = st.session_state.pages
        chunks = st.session_state.chunks
        combined_text = st.session_state.combined_text

        st.subheader("Extraction summary")
        st.write(f"Pages extracted: {len(pages)}")
        st.write(f"Chunks created: {len(chunks)}")

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

        st.subheader("Generate Summary")
        summary_type = st.selectbox(
            "Choose a summary style",
            list(SUMMARY_TYPES.keys()),
            index=list(SUMMARY_TYPES.keys()).index(st.session_state.summary_type)
            if st.session_state.summary_type in SUMMARY_TYPES
            else 1,
        )

        if st.button("Generate Summary"):
            with st.spinner("Generating summary from your processed chunks..."):
                st.session_state.summary_text = generate_summary(chunks, summary_type)
                st.session_state.summary_type = summary_type

        if st.session_state.summary_text:
            st.markdown(f"**{st.session_state.summary_type}**")
            st.write(st.session_state.summary_text)

        st.subheader("Generate Quiz")
        quiz_type = st.selectbox(
            "Quiz type",
            QUIZ_TYPES,
            index=QUIZ_TYPES.index(st.session_state.quiz_type) if st.session_state.quiz_type in QUIZ_TYPES else 0,
        )

        quiz_n = st.number_input("Number of questions", min_value=1, max_value=20, value=st.session_state.quiz_n)

        use_gemini_quiz = st.checkbox("Use Gemini to improve distractors & explanations", value=False)

        if st.button("Generate Quiz"):
            with st.spinner("Generating quiz from processed chunks..."):
                st.session_state.quiz = generate_quiz_from_chunks(
                    chunks, n=int(quiz_n), qtype=quiz_type, use_gemini=use_gemini_quiz
                )
                st.session_state.quiz_type = quiz_type
                st.session_state.quiz_n = int(quiz_n)

        if st.session_state.quiz:
            st.markdown(f"**Quiz ({st.session_state.quiz_type})**")
            for i, q in enumerate(st.session_state.quiz, start=1):
                st.write(f"{i}. {q['question']}")
                if q.get("type") == "MCQ":
                    for opt in q.get("options", []):
                        st.write(f"- {opt}")
                with st.expander("Show answer & explanation"):
                    st.write(f"**Answer:** {q.get('answer')}")
                    st.write(f"**Explanation:** {q.get('explanation')}")

        st.subheader("Ask a question")
        beginner_mode = st.toggle("Beginner mode", value=True)
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

            if st.button("Generate answer with Gemini"):
                with st.spinner("Generating answer from retrieved chunks..."):
                    answer = answer_question(question, retrieved_chunks, beginner_mode)

                st.subheader("Final answer")
                st.write(answer)

                st.subheader("Source references")
                source_references = get_source_references(retrieved_chunks)
                if source_references:
                    for source in source_references:
                        st.write(f"- {source['file_name']} — page {source['page_number']}")
                else:
                    st.write("No source references available.")
    else:
        st.info("Click 'Process PDFs' to extract text, create chunks, and save them to ChromaDB.")
else:
    st.info("Upload one or more PDF files to begin.")
