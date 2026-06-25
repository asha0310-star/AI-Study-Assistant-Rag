import hashlib
import math
import re

import chromadb


CHROMA_DB_FOLDER = "chroma_db"
COLLECTION_NAME = "study_notes"


def create_embedding(text, size=128):
    """Create a small local embedding without calling any external API."""
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
        embeddings.append(create_embedding(chunk["text"]))

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
        query_embeddings=[create_embedding(question)],
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
