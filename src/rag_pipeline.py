from .pdf_loader import load_pdfs
from .chunker import chunk_text
from .vector_store import VectorStore
from .summary_generator import summarize_text
from .quiz_generator import generate_quiz_from_text


class RAGPipeline:
    """High-level pipeline that connects loader, chunker, vector store, and generators."""

    def __init__(self, persist_dir: str = "chroma_db"):
        self.vs = VectorStore(persist_dir=persist_dir)

    def ingest_pdfs(self, pdf_dir: str):
        items = load_pdfs(pdf_dir)
        texts = []
        metas = []
        for it in items:
            path = it.get("path")
            txt = it.get("text", "")
            if not txt:
                continue
            chunks = chunk_text(txt)
            for i, c in enumerate(chunks):
                texts.append(c)
                metas.append({"source": path, "chunk": i})
        if texts:
            self.vs.add_texts(texts, metadatas=metas)

    def query(self, question: str, top_k: int = 3) -> str:
        """Return a short summary of the top retrieved documents for the question."""
        res = self.vs.search(question, top_k=top_k)
        # Normalize Chroma vs in-memory response shapes
        docs = []
        if isinstance(res, dict):
            # Chroma-style response
            docs = res.get("documents", [[]])
            # `documents` is a list-of-lists; take first query
            if isinstance(docs, list) and len(docs) > 0 and isinstance(docs[0], list):
                docs = docs[0]
            metas = res.get("metadatas", [[]])
            if isinstance(metas, list) and len(metas) > 0 and isinstance(metas[0], list):
                metas = metas[0]
        else:
            docs = [r.get("document") for r in res]
            metas = [r.get("metadata") for r in res]
        combined = "\n\n---\n\n".join([d for d in docs if d])
        if not combined:
            return "No relevant documents found."
        summary = summarize_text(combined)
        return summary

    def generate_quiz(self, question: str, top_k: int = 3, n: int = 5):
        res = self.vs.search(question, top_k=top_k)
        if isinstance(res, dict):
            docs = res.get("documents", [[]])[0]
            combined = "\n".join([d for d in docs if d])
        else:
            combined = "\n".join([r.get("document", "") for r in res])
        return generate_quiz_from_text(combined, n=n)
