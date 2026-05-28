"""filo FastAPI backend.

Exposes a small semantic-search API over the NYC venue catalog:

    GET  /health     -> {"status": "ok"}
    POST /recommend  -> ranked venues for a natural-language query

Run:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    # Works when run as a module: `uvicorn api.main:app`
    from api.embeddings import get_index, search
except ImportError:  # pragma: no cover - allows `uvicorn main:app` from api/
    from embeddings import get_index, search

app = FastAPI(
    title="filo",
    description="NYC venue recommendation engine — semantic search over venues.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    """Request body for POST /recommend."""

    query: str = Field(..., description="Natural-language description of the desired venue.")
    top_k: int = Field(5, ge=1, le=50, description="Number of venues to return.")


class VenueResult(BaseModel):
    """A single ranked venue returned by the recommender."""

    id: str
    name: str
    description: str
    neighborhood: str
    category: str
    lat: float
    lng: float
    price_tier: int
    score: float


@app.on_event("startup")
def warm_up() -> None:
    """Eagerly build the embedding index so the first request is fast."""
    index = get_index()
    print(f"[api] ready — {len(index.venues)} venues indexed using '{index.source}' model.")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe.

    Returns:
        {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/recommend", response_model=list[VenueResult])
def recommend(req: RecommendRequest) -> list[dict[str, Any]]:
    """Return the venues best matching the query.

    Args:
        req: the recommendation request (query + top_k).

    Returns:
        A list of ranked venue objects with similarity scores.
    """
    return search(req.query, req.top_k)
