from typing import List, Optional

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except Exception:
    chromadb = None
    CHROMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    _EMB_MODEL = None

import numpy as np


class VectorStore:
    """Simple vector store. Uses Chroma if installed, otherwise an in-memory store.

    Methods:
      - add_texts(texts, metadatas)
      - search(query, top_k)
    """

    def __init__(self, persist_dir: str = "chroma_db"):
        self.persist_dir = persist_dir
        if CHROMA_AVAILABLE:
            self.client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir))
            self.col = self.client.create_collection(name="documents", exist_ok=True)
        else:
            self.documents: List[str] = []
            self.metadatas: List[dict] = []
            self.embeddings: Optional[np.ndarray] = None

    def _embed(self, texts: List[str]) -> np.ndarray:
        if _EMB_MODEL is not None:
            emb = _EMB_MODEL.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return np.asarray(emb)
        # simple fallback: lightweight numeric vectors based on character sums
        arr = []
        for t in texts:
            s = sum(ord(c) for c in t) % 100000
            v = np.full((128,), (s / 100000.0), dtype=float)
            arr.append(v)
        return np.vstack(arr)

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        metadatas = metadatas or [{}] * len(texts)
        if CHROMA_AVAILABLE:
            emb = self._embed(texts).tolist()
            self.col.add(documents=texts, metadatas=metadatas, embeddings=emb)
            try:
                # persist if supported
                self.client.persist()
            except Exception:
                pass
            return
        # in-memory
        emb = self._embed(texts)
        if self.embeddings is None:
            self.embeddings = emb.copy()
        else:
            self.embeddings = np.vstack([self.embeddings, emb])
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)

    def search(self, query: str, top_k: int = 3):
        """Return top_k results. If using Chroma, return the raw dict response. Otherwise return list of dicts.
        """
        if CHROMA_AVAILABLE:
            qemb = self._embed([query]).tolist()[0]
            res = self.col.query(query_embeddings=[qemb], n_results=top_k, include=["documents", "metadatas", "distances"]) or {}
            return res
        if self.embeddings is None or len(self.documents) == 0:
            return []
        qemb = self._embed([query])[0]
        # squared L2 distances
        dists = np.sum((self.embeddings - qemb) ** 2, axis=1)
        idx = np.argsort(dists)[:top_k]
        results = []
        for i in idx:
            results.append({
                "document": self.documents[int(i)],
                "metadata": self.metadatas[int(i)],
                "distance": float(dists[int(i)]),
            })
        return results
