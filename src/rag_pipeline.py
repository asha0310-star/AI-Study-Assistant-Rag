import os

from dotenv import load_dotenv


NOT_FOUND_MESSAGE = "I could not find this clearly in the uploaded documents."
GEMINI_MODEL = "gemini-1.5-flash"


load_dotenv()


def build_context(retrieved_chunks):
    """Turn retrieved chunks into one context string for Gemini."""
    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk["metadata"]
        context_parts.append(
            f"Source {index}: {metadata['file_name']}, "
            f"page {metadata['page_number']}\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(context_parts)


def get_source_references(retrieved_chunks):
    """Return unique source references from retrieved chunks."""
    sources = []
    seen_sources = set()

    for chunk in retrieved_chunks:
        metadata = chunk["metadata"]
        source = (metadata["file_name"], metadata["page_number"])

        if source not in seen_sources:
            sources.append(
                {
                    "file_name": metadata["file_name"],
                    "page_number": metadata["page_number"],
                }
            )
            seen_sources.add(source)

    return sources


def answer_question(question, retrieved_chunks, beginner_mode=False):
    """Answer a question using only the retrieved document chunks."""
    if not retrieved_chunks:
        return NOT_FOUND_MESSAGE

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Gemini API key is missing. Add GEMINI_API_KEY to your .env file."

    import google.generativeai as genai

    context = build_context(retrieved_chunks)
    beginner_instruction = "Explain the answer in simple beginner-friendly language."
    normal_instruction = "Answer clearly and concisely."
    style_instruction = beginner_instruction if beginner_mode else normal_instruction

    prompt = f"""
You are an AI study assistant.
Use only the context below to answer the question.
If the answer is not clearly found in the context, say exactly:
{NOT_FOUND_MESSAGE}

{style_instruction}

Context:
{context}

Question:
{question}
"""

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)

    if not response.text:
        return NOT_FOUND_MESSAGE

    return response.text.strip()
