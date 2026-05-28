"""Embedding index and semantic search for filo.

Loads the fine-tuned model from ``models/filo-recommender/`` (falling back to the
base ``all-MiniLM-L6-v2`` if no checkpoint exists), pre-computes and caches
embeddings for every venue in ``data/venues.json``, and exposes a ``search``
function that ranks venues against a query by cosine similarity.

The embedding matrix is cached to disk so the API starts fast on subsequent
runs. The cache is invalidated automatically when the model or venue file
changes (tracked via a fingerprint stored alongside the cache).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]
VENUES_PATH = REPO_ROOT / "data" / "venues.json"
FINETUNED_DIR = REPO_ROOT / "models" / "filo-recommender"
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CACHE_DIR = Path(__file__).resolve().parent
EMB_CACHE_PATH = CACHE_DIR / ".embeddings_cache.npy"
META_CACHE_PATH = CACHE_DIR / ".embeddings_cache.json"


def resolve_model_path() -> tuple[str, str]:
    """Decide which model to load.

    Returns:
        (path_or_name, source_label) where source_label is "finetuned" or
        "base", used for logging and cache fingerprinting.
    """
    if FINETUNED_DIR.exists() and (FINETUNED_DIR / "config.json").exists():
        return str(FINETUNED_DIR), "finetuned"
    return BASE_MODEL, "base"


def load_venues() -> list[dict[str, Any]]:
    """Load the venue catalog.

    Returns:
        A list of venue dicts.

    Raises:
        FileNotFoundError: if venues.json has not been generated yet.
    """
    if not VENUES_PATH.exists():
        raise FileNotFoundError(
            f"{VENUES_PATH} not found. Run `python data/process.py` first."
        )
    return json.loads(VENUES_PATH.read_text())


def _fingerprint(model_source: str, n_venues: int) -> dict[str, Any]:
    """Build a cache fingerprint from model source and venue file state."""
    return {
        "model_source": model_source,
        "n_venues": n_venues,
        "venues_mtime": VENUES_PATH.stat().st_mtime if VENUES_PATH.exists() else 0,
    }


class EmbeddingIndex:
    """In-memory cosine-similarity index over venue embeddings."""

    def __init__(self) -> None:
        """Load the model, venues and (cached) normalized embedding matrix."""
        model_path, source = resolve_model_path()
        print(f"[embeddings] loading model: {model_path} ({source})")
        self.model = SentenceTransformer(model_path)
        self.source = source
        self.venues = load_venues()
        self.embeddings = self._load_or_compute_embeddings()

    def _load_or_compute_embeddings(self) -> np.ndarray:
        """Return a normalized (n_venues, dim) embedding matrix, using cache.

        Recomputes and rewrites the cache if the fingerprint has changed.

        Returns:
            A float32 numpy array of L2-normalized venue embeddings.
        """
        fp = _fingerprint(self.source, len(self.venues))
        if EMB_CACHE_PATH.exists() and META_CACHE_PATH.exists():
            try:
                cached_fp = json.loads(META_CACHE_PATH.read_text())
                if cached_fp == fp:
                    print("[embeddings] using cached embeddings.")
                    return np.load(EMB_CACHE_PATH)
            except (ValueError, OSError):
                pass  # fall through and recompute

        print(f"[embeddings] computing embeddings for {len(self.venues)} venues ...")
        texts = [v["description"] for v in self.venues]
        emb = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        np.save(EMB_CACHE_PATH, emb)
        META_CACHE_PATH.write_text(json.dumps(fp))
        return emb

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the top_k venues most similar to ``query``.

        Args:
            query: natural-language search string.
            top_k: number of results to return.

        Returns:
            A list of venue dicts (name, description, neighborhood, category,
            lat, lng, price_tier) each augmented with a float ``score`` in
            [0, 1], sorted by descending score.
        """
        if not query or not query.strip():
            return []

        q_emb = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)[0]

        scores = self.embeddings @ q_emb  # cosine similarity (both normalized)
        k = max(1, min(top_k, len(self.venues)))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        results: list[dict[str, Any]] = []
        for idx in top_idx:
            venue = self.venues[int(idx)]
            results.append(
                {
                    "id": venue["id"],
                    "name": venue["name"],
                    "description": venue["description"],
                    "neighborhood": venue["neighborhood"],
                    "category": venue["category"],
                    "lat": venue["lat"],
                    "lng": venue["lng"],
                    "price_tier": venue["price_tier"],
                    "score": round(float(scores[idx]), 4),
                }
            )
        return results


@lru_cache(maxsize=1)
def get_index() -> EmbeddingIndex:
    """Return a process-wide singleton EmbeddingIndex (lazily constructed)."""
    return EmbeddingIndex()


def search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Module-level convenience wrapper around EmbeddingIndex.search.

    Args:
        query: natural-language search string.
        top_k: number of results to return.

    Returns:
        Ranked venue dicts with similarity scores.
    """
    return get_index().search(query, top_k)


if __name__ == "__main__":
    for q in ["romantic wine bar with candlelight", "late night ramen", "rooftop drinks"]:
        print(f"\nQUERY: {q}")
        for r in search(q, top_k=3):
            print(f"  {r['score']:.3f}  {r['name']} ({r['category']}, {r['neighborhood']})")
