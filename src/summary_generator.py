"""Gemini-powered summary generation helpers for the study assistant."""

import os

from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()


SUMMARY_MODEL = "gemini-2.5-flash"
SUMMARY_TYPES = {
    "Beginner summary": {
        "title": "Beginner summary",
        "instruction": "Explain the material in simple language for a new learner.",
    },
    "Exam-focused summary": {
        "title": "Exam-focused summary",
        "instruction": (
            "Focus on high-yield concepts, likely exam questions, definitions, "
            "formulae, and important relationships."
        ),
    },
    "Bullet-point summary": {
        "title": "Bullet-point summary",
        "instruction": "Return the summary as concise bullet points with clear headings.",
    },
}


def get_api_key():
    """Read the Gemini API key from the environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to your .env file before generating summaries."
        )
    return api_key


def build_context(chunks, max_chunks=12, max_chars=18000):
    """Turn processed chunks into a single context string for Gemini."""
    context_parts = []
    total_chars = 0

    for index, chunk in enumerate(chunks[:max_chunks], start=1):
        metadata = chunk["metadata"]
        chunk_text = (
            f"Chunk {index}: {metadata['file_name']}, page {metadata['page_number']}, "
            f"chunk {metadata['chunk_id']}\n{chunk['text']}"
        )
        context_parts.append(chunk_text)
        total_chars += len(chunk_text)

        if total_chars >= max_chars:
            break

    return "\n\n".join(context_parts)


def generate_summary(chunks, summary_type="Exam-focused summary"):
    """Generate a Gemini summary from the already processed chunks."""
    if not chunks:
        return "No processed chunks are available yet. Click 'Process PDFs' first."

    summary_config = SUMMARY_TYPES.get(summary_type, SUMMARY_TYPES["Exam-focused summary"])
    context = build_context(chunks)

    prompt = f"""
You are an AI study assistant.
Use only the context below.
Create a {summary_config['title']}.

Requirements:
- {summary_config['instruction']}
- Keep the summary practical and easy to study from.
- Do not add facts that are not in the context.

Context:
{context}
"""

    genai.configure(api_key=get_api_key())
    model = genai.GenerativeModel(SUMMARY_MODEL)
    response = model.generate_content(prompt)

    if not response.text:
        return "Gemini did not return a summary. Please try again."

    return response.text.strip()
