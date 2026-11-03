"""Embeddings + vector search over feedback. One index, two uses:
  1. semantic issue grouping (aggregation_service)
  2. retrieval for grounded Q&A (rag_service)

Embedder: local sentence-transformers `all-MiniLM-L6-v2` (free, no API key,
384 dims). If it isn't installed, falls back to a hashed bag-of-words vector so
the app still runs; `backend_name()` reports which one is active.

Vector store: vectors live in the `feedback_embeddings` table and search is a
numpy cosine scan, which is plenty for thousands of rows. Production upgrade:
Supabase `pgvector` (same table, `vector(384)` column, `<=>` operator).
"""
import hashlib
import re

import numpy as np

from .. import config

_model = None
_backend = None
DIM_FALLBACK = 512


def _load():
    global _model, _backend
    if _backend:
        return
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
        _backend = config.EMBEDDING_MODEL
    except Exception:
        _backend = "hashed-bow-fallback"


def backend_name() -> str:
    _load()
    return _backend


def _hash_embed(text: str) -> np.ndarray:
    v = np.zeros(DIM_FALLBACK, dtype=np.float32)
    words = re.findall(r"[a-z]+", text.lower())
    for w in words + [a + "_" + b for a, b in zip(words, words[1:])]:
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % DIM_FALLBACK] += 1.0 if h & 1 else -1.0
    n = np.linalg.norm(v)
    return v / n if n else v


def embed(texts: list[str]) -> np.ndarray:
    _load()
    if _model is not None:
        return np.asarray(_model.encode(texts, normalize_embeddings=True), dtype=np.float32)
    return np.stack([_hash_embed(t) for t in texts])


def top_k(query_vec: np.ndarray, matrix: np.ndarray, k: int = 8) -> list[tuple[int, float]]:
    """Indices and cosine scores of the k nearest rows (vectors are pre-normalised)."""
    if matrix.size == 0:
        return []
    sims = matrix @ query_vec
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in idx]
