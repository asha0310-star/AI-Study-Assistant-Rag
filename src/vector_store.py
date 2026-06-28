import os
import hashlib
import math
import re

import chromadb
import google.generativeai as genai
from google.api_core.exceptions import NotFound


CHROMA_DB_FOLDER = "chroma_db"
COLLECTION_NAME = "study_notes"
# Default embedding model; change if your account uses a different name/version.
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"


def get_api_key():
    """Read the Gemini API key from the environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to your .env file before using embeddings."
        )
    return api_key


def _local_hash_embedding(text, size=128):
    """Local fallback embedding used only if Gemini fails.

    Keeps a simple vector so the app continues to work offline.
    """
    vector = [0.0] * size
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    for word in words:
        word_hash = hashlib.md5(word.encode("utf-8")).hexdigest()
        index = int(word_hash, 16) % size
        vector[index] += 1.0

    length = math.sqrt(sum(value * value for value in vector))
    if length > 0:
        vector = [value / length for value in vector]

    return vector


def create_embedding(text, task_type="retrieval_document"):
    """Create an embedding for the given text using Gemini.

    If the configured Gemini embedding model is not available, a helpful
    RuntimeError is raised that suggests listing available models. For other
    runtime failures we fall back to a small local hashing embedding so the
    app remains usable.
    """
    api_key = get_api_key()
    genai.configure(api_key=api_key)

    try:
        result = genai.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            content=text,
            task_type=task_type,
        )
        return result["embedding"]

    except NotFound as nf:
        # Model not available for this API version; give an actionable message.
        msg = (
            f"Embedding model '{GEMINI_EMBEDDING_MODEL}' was not found or is not supported for embedContent. "
            "Run the following short script to list available models and their names:\n\n"
            "python - <<'PY'\nfrom google.ai import generativelanguage_v1beta as gl\nclient = gl.GenerativeServiceClient()\nfor m in client.list_models():\n    print(m.name)\nPY\n\n"
            "Then update `GEMINI_EMBEDDING_MODEL` in `src/vector_store.py` to one of the listed model names."
        )
        raise RuntimeError(msg) from nf

    except Exception:
        # For other errors, fall back to local embedding so the user can continue.
        return _local_hash_embedding(text)


def get_collection():
    """Open the local ChromaDB collection for this study assistant."""
    client = chromadb.PersistentClient(path=CHROMA_DB_FOLDER)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def store_chunks(chunks):
    """Save chunk text and metadata in the local ChromaDB folder."""
    collection = get_collection()

    existing_items = collection.get()
    existing_ids = existing_items.get("ids", [])
    if existing_ids:
        collection.delete(ids=existing_ids)

    if not chunks:
        return 0

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]
        chunk_id = (
            f"{metadata['file_name']}-"
            f"page-{metadata['page_number']}-"
            f"chunk-{metadata['chunk_id']}-"
            f"{index}"
        )

        ids.append(chunk_id)
        documents.append(chunk["text"])
        metadatas.append(metadata)
        embeddings.append(create_embedding(chunk["text"], task_type="retrieval_document"))

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(ids)


def search_chunks(question, top_k=5):
    """Search ChromaDB and return the top matching chunks for a question."""
    collection = get_collection()

    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[create_embedding(question, task_type="retrieval_query")],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    matches = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(documents, metadatas, distances):
        matches.append(
            {
                "text": document,
                "metadata": metadata,
                "distance": distance,
            }
        )

    return matches
