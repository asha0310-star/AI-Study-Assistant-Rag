from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def chunk_pages(pages):
    """Split extracted PDF pages into smaller text chunks.

    Each returned chunk keeps the PDF file name, page number, and a chunk ID.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = []

    for page in pages:
        page_chunks = text_splitter.split_text(page["text"])

        for chunk_index, chunk_text in enumerate(page_chunks, start=1):
            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        "file_name": page["file_name"],
                        "page_number": page["page_number"],
                        "chunk_id": chunk_index,
                    },
                }
            )

    return chunks
