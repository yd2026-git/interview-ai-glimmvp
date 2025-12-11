# rag.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class RAGIndex:
    def __init__(self, index_dir="saved_index"):
        os.makedirs(index_dir, exist_ok=True)
        self.index_dir = index_dir
        self.embedder = SentenceTransformer(MODEL_NAME)
        self.index = None
        self.docs = []  # store (id, text, meta)
        self._index_file = os.path.join(index_dir, "faiss.index")
        self._meta_file = os.path.join(index_dir, "meta.pkl")
        if os.path.exists(self._meta_file) and os.path.exists(self._index_file):
            self._load()

    def _load(self):
        with open(self._meta_file, "rb") as f:
            self.docs = pickle.load(f)
        d = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.read_index(self._index_file)

    def save(self):
        # save meta and index
        with open(self._meta_file, "wb") as f:
            pickle.dump(self.docs, f)
        faiss.write_index(self.index, self._index_file)

    def build_from_texts(self, list_of_docs):
        """
        list_of_docs = [{'id': 'doc1', 'text': '...', 'meta': {...}}, ...]
        """
        self.docs = list_of_docs
        texts = [d['text'] for d in self.docs]
        embeddings = self.embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product (cosine after normalization)
        # normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.save()

    def add_doc(self, doc):
        """Append single doc and add to index"""
        self.docs.append(doc)
        emb = self.embedder.encode([doc['text']], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        if self.index is None:
            d = emb.shape[1]
            self.index = faiss.IndexFlatIP(d)
        self.index.add(emb)
        self.save()

    def query(self, text, top_k=5):
        emb = self.embedder.encode([text], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        D, I = self.index.search(emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.docs):
                continue
            results.append({"score": float(score), "doc": self.docs[idx]})
        return results
