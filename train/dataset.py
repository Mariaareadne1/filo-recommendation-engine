"""Training dataset construction for the filo recommender.

Builds (query, positive_venue_description) training pairs from the sample
queries and venue catalog, plus hard negatives — venues that sound similar to
the query but belong to the wrong category. Exposes a PyTorch Dataset and a
helper that returns sentence-transformers ``InputExample`` lists.

Run standalone to sanity-check the pairs:
    python train/dataset.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
VENUES_PATH = DATA_DIR / "venues.json"
QUERIES_PATH = DATA_DIR / "sample_queries.json"

# Categories that are easy to confuse — used to mine *hard* negatives so the
# model must learn fine-grained distinctions rather than coarse topic matching.
CONFUSABLE: dict[str, list[str]] = {
    "wine_bar": ["cocktail_bar", "bar"],
    "cocktail_bar": ["wine_bar", "speakeasy", "bar"],
    "speakeasy": ["cocktail_bar", "bar"],
    "bar": ["cocktail_bar", "wine_bar"],
    "rooftop_bar": ["cocktail_bar", "bar"],
    "date_spot": ["restaurant", "wine_bar"],
    "restaurant": ["brunch_spot", "date_spot"],
    "brunch_spot": ["cafe", "restaurant"],
    "cafe": ["brunch_spot"],
    "ramen_shop": ["restaurant", "pizza_spot"],
    "pizza_spot": ["restaurant", "ramen_shop"],
    "jazz_club": ["cocktail_bar", "bar"],
}


def load_venues() -> list[dict[str, Any]]:
    """Load the venue catalog produced by data/process.py.

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


def load_queries() -> list[dict[str, Any]]:
    """Load the sample queries produced by data/process.py.

    Returns:
        A list of {"query", "expected_categories"} dicts.
    """
    if not QUERIES_PATH.exists():
        raise FileNotFoundError(
            f"{QUERIES_PATH} not found. Run `python data/process.py` first."
        )
    return json.loads(QUERIES_PATH.read_text())


def _index_by_category(venues: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group venues by their category label."""
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for v in venues:
        by_cat.setdefault(v["category"], []).append(v)
    return by_cat


def build_pairs(
    venues: list[dict[str, Any]] | None = None,
    queries: list[dict[str, Any]] | None = None,
    positives_per_query: int = 4,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build (query, positive, hard_negative) training triplets.

    For each sample query we draw several positives from the expected
    categories and one hard negative from a confusable-but-wrong category.

    Args:
        venues: venue catalog; loaded from disk if None.
        queries: sample queries; loaded from disk if None.
        positives_per_query: number of positive venues to pair per query.
        seed: RNG seed for reproducibility.

    Returns:
        A list of {"query", "positive", "negative", "positive_category"} dicts.
    """
    rng = random.Random(seed)
    venues = venues if venues is not None else load_venues()
    queries = queries if queries is not None else load_queries()
    by_cat = _index_by_category(venues)

    pairs: list[dict[str, Any]] = []
    for q in queries:
        query_text = q["query"]
        expected = [c for c in q["expected_categories"] if c in by_cat]
        if not expected:
            continue

        # Positives: sample from the expected categories.
        positive_pool: list[dict[str, Any]] = []
        for cat in expected:
            positive_pool.extend(by_cat[cat])
        rng.shuffle(positive_pool)
        positives = positive_pool[:positives_per_query]

        # Hard negatives: confusable categories that are NOT expected.
        neg_cats: list[str] = []
        for cat in expected:
            neg_cats.extend(c for c in CONFUSABLE.get(cat, []) if c not in expected)
        neg_cats = [c for c in dict.fromkeys(neg_cats) if c in by_cat]

        for pos in positives:
            neg_venue = None
            if neg_cats:
                neg_cat = rng.choice(neg_cats)
                neg_venue = rng.choice(by_cat[neg_cat])
            pairs.append(
                {
                    "query": query_text,
                    "positive": pos["description"],
                    "negative": neg_venue["description"] if neg_venue else None,
                    "positive_category": pos["category"],
                }
            )
    rng.shuffle(pairs)
    return pairs


def split_queries(
    queries: list[dict[str, Any]], val_every: int = 4
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split sample queries into disjoint train / validation sets.

    The split happens at the *query* level, before pairs are built, so no
    validation query (or its exact text) is ever seen during training.
    Splitting pairs instead would leak: every query contributes several pairs,
    so the same query text would land on both sides.

    Args:
        queries: sample queries.
        val_every: every Nth query (deterministic) goes to validation.

    Returns:
        (train_queries, val_queries), disjoint by query text.
    """
    val = queries[::val_every]
    val_text = {q["query"] for q in val}
    train = [q for q in queries if q["query"] not in val_text]
    return train, val


def train_val_split(
    pairs: list[dict[str, Any]], val_frac: float = 0.15, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split pairs into train/validation sets.

    Args:
        pairs: list of training pairs.
        val_frac: fraction held out for validation.
        seed: RNG seed.

    Returns:
        (train_pairs, val_pairs)
    """
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]


class FiloPairDataset(Dataset):
    """PyTorch Dataset of (anchor, positive, negative) text triplets.

    Each item is a dict with string fields ``anchor``, ``positive`` and
    ``negative`` (negative may be an empty string when none was mined). This is
    consumed by the sentence-transformers training loop via a collate step in
    finetune.py.
    """

    def __init__(self, pairs: list[dict[str, Any]]):
        """Initialize the dataset.

        Args:
            pairs: list of pair dicts from build_pairs().
        """
        self.pairs = pairs

    def __len__(self) -> int:
        """Return the number of training pairs."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, str]:
        """Return the triplet at ``idx`` as a dict of strings.

        Args:
            idx: index into the pair list.

        Returns:
            {"anchor", "positive", "negative"} string dict.
        """
        p = self.pairs[idx]
        return {
            "anchor": p["query"],
            "positive": p["positive"],
            "negative": p["negative"] or "",
        }


if __name__ == "__main__":
    train_queries, val_queries = split_queries(load_queries())
    all_pairs = build_pairs(queries=train_queries)
    print(f"queries: {len(train_queries)} train / {len(val_queries)} val (disjoint)")
    print(f"training pairs (train queries only): {len(all_pairs)}")
    print("\nexample pair:")
    print(json.dumps(all_pairs[0], indent=2, ensure_ascii=False))
