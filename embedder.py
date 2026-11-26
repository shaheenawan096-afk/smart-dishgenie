"""
services/embedder.py
--------------------
Sentence Transformer based semantic embedder.
Used for RECIPE RETRIEVAL — not ingredient normalization.

How it works:
  1. User's normalized ingredients  → embed as vector
  2. Each recipe's ingredients      → embed as vector (cached)
  3. Cosine similarity              → find closest recipes

Model: all-mpnet-base-v2 (~80MB, downloads once)
Cache: embedding_cache.pkl (so model runs once per recipe DB)
"""

import os
import pickle
import numpy as np

_BASE       = os.path.dirname(os.path.abspath(__file__))
_CACHE_FILE = os.path.join(_BASE, "embedding_cache.pkl")

_model    = None
_db_embs  = None   # cached recipe embeddings
_db_texts = None   # cached recipe ingredient strings


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-mpnet-base-v2")
    return _model


def _load_or_build_cache(recipe_texts: list[str]) -> np.ndarray:
    """Load embeddings from cache, or build + save if missing."""
    global _db_embs, _db_texts

    if os.path.exists(_CACHE_FILE):
        with open(_CACHE_FILE, "rb") as f:
            cached = pickle.load(f)
            _db_embs  = cached["embeddings"]
            _db_texts = cached["texts"]
            # If recipe DB changed, rebuild
            if _db_texts == recipe_texts:
                return _db_embs

    model = _get_model()
    _db_embs  = model.encode(recipe_texts, show_progress_bar=False)
    _db_texts = recipe_texts

    with open(_CACHE_FILE, "wb") as f:
        pickle.dump({"embeddings": _db_embs, "texts": _db_texts}, f)

    return _db_embs


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string into a vector."""
    return _get_model().encode([text])[0]


def get_recipe_embeddings(recipe_texts: list[str]) -> np.ndarray:
    """Get (cached) embeddings for all recipe ingredient strings."""
    return _load_or_build_cache(recipe_texts)


def cosine_similarity_scores(query_vec: np.ndarray, db_vecs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between one query vector and all DB vectors.
    Returns array of similarity scores (0.0 - 1.0).
    """
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity([query_vec], db_vecs)[0]
