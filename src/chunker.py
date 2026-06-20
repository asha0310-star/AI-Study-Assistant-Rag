from typing import List


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks (by words).

    Args:
        text: input string
        chunk_size: approximate chunk size in words
        overlap: overlap between chunks in words
    """
    if not text:
        return []
    tokens = text.split()
    if chunk_size <= 0:
        return [text]
    chunks = []
    i = 0
    n = len(tokens)
    while i < n:
        chunk = tokens[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += max(1, chunk_size - overlap)
    return chunks
