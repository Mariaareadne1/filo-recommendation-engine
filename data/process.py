"""filo data pipeline.

Fetches ~500 NYC venues from the Foursquare Places API and builds natural
language descriptions for each one. If the Foursquare API is unavailable (no
key, rate limited, or network error) it falls back to generating a realistic
synthetic dataset using hardcoded NYC neighborhood and category data.

Outputs:
    data/venues.json         - list of venues with embeddings-ready descriptions
    data/sample_queries.json - 50 example queries paired with expected venue types

Run:
    python data/process.py
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional at runtime
    pass

DATA_DIR = Path(__file__).resolve().parent
VENUES_PATH = DATA_DIR / "venues.json"
QUERIES_PATH = DATA_DIR / "sample_queries.json"

TARGET_VENUE_COUNT = 500
FOURSQUARE_API_KEY = os.getenv("FOURSQUARE_API_KEY", "")
FOURSQUARE_SEARCH_URL = "https://api.foursquare.com/v3/places/search"

# Foursquare category ids mapped to filo's internal category labels.
# https://location.foursquare.com/places/docs/categories
FOURSQUARE_CATEGORIES: dict[str, str] = {
    "13065": "restaurant",      # Restaurant
    "13003": "bar",             # Bar
    "13032": "cafe",            # Cafe / Coffee Shop
    "13018": "wine_bar",        # Wine Bar
    "13009": "cocktail_bar",    # Cocktail Bar
    "13008": "rooftop_bar",     # Beer/lounge (used loosely for rooftop)
    "13073": "jazz_club",       # Jazz Club
    "13016": "speakeasy",       # Lounge (used for speakeasy vibe)
    "13263": "ramen_shop",      # Ramen Restaurant
    "13064": "pizza_spot",      # Pizzeria
    "13002": "brunch_spot",     # Breakfast Spot
}

# NYC bounding boxes used to bias Foursquare search toward the five boroughs.
NYC_LL = "40.7128,-74.0060"  # Manhattan-ish centroid


# --------------------------------------------------------------------------- #
# Foursquare path
# --------------------------------------------------------------------------- #
def fetch_foursquare_category(category_id: str, label: str, limit: int = 50) -> list[dict[str, Any]]:
    """Fetch venues for a single Foursquare category id near NYC.

    Args:
        category_id: Foursquare category id to query.
        label: filo internal category label to attach to results.
        limit: max number of venues to request (Foursquare caps at 50).

    Returns:
        A list of normalized venue dicts. Empty on any error.
    """
    headers = {"Authorization": FOURSQUARE_API_KEY, "accept": "application/json"}
    params = {
        "ll": NYC_LL,
        "radius": 15000,
        "categories": category_id,
        "limit": min(limit, 50),
        "fields": "fsq_id,name,location,categories,geocodes,price,description",
    }
    try:
        resp = requests.get(FOURSQUARE_SEARCH_URL, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except (requests.RequestException, ValueError) as exc:
        print(f"  [foursquare] failed for {label}: {exc}")
        return []

    venues: list[dict[str, Any]] = []
    for item in results:
        loc = item.get("location", {})
        geo = item.get("geocodes", {}).get("main", {})
        neighborhood = (
            loc.get("neighborhood", [None])[0]
            if isinstance(loc.get("neighborhood"), list)
            else loc.get("neighborhood")
        ) or loc.get("locality") or "Manhattan"
        venues.append(
            {
                "id": item.get("fsq_id"),
                "name": item.get("name", "Unknown"),
                "neighborhood": neighborhood,
                "category": label,
                "lat": geo.get("latitude"),
                "lng": geo.get("longitude"),
                "price_tier": int(item.get("price", random.randint(1, 4))),
                "_tags": [c.get("name") for c in item.get("categories", [])],
                "_blurb": item.get("description", ""),
            }
        )
    return venues


def fetch_all_foursquare() -> list[dict[str, Any]]:
    """Fetch venues across all configured categories from Foursquare.

    Returns:
        A combined list of normalized venues, or an empty list if the API key
        is missing or every category request fails.
    """
    if not FOURSQUARE_API_KEY:
        print("[foursquare] no FOURSQUARE_API_KEY set; skipping live fetch.")
        return []

    per_category = max(1, TARGET_VENUE_COUNT // len(FOURSQUARE_CATEGORIES))
    all_venues: list[dict[str, Any]] = []
    for category_id, label in FOURSQUARE_CATEGORIES.items():
        print(f"[foursquare] fetching {label} ...")
        all_venues.extend(fetch_foursquare_category(category_id, label, per_category))
        time.sleep(0.5)  # be polite to the free tier
    print(f"[foursquare] fetched {len(all_venues)} venues total.")
    return all_venues


# --------------------------------------------------------------------------- #
# Synthetic fallback
# --------------------------------------------------------------------------- #
NEIGHBORHOODS: dict[str, list[tuple[str, float, float]]] = {
    "Manhattan": [
        ("Lower East Side", 40.7150, -73.9843),
        ("West Village", 40.7358, -74.0036),
        ("East Village", 40.7265, -73.9815),
        ("SoHo", 40.7233, -74.0030),
        ("Tribeca", 40.7163, -74.0086),
        ("Chelsea", 40.7465, -74.0014),
        ("Greenwich Village", 40.7336, -74.0027),
        ("Upper West Side", 40.7870, -73.9754),
        ("Upper East Side", 40.7736, -73.9566),
        ("Harlem", 40.8116, -73.9465),
        ("Midtown", 40.7549, -73.9840),
        ("Nolita", 40.7224, -73.9955),
        ("Hell's Kitchen", 40.7638, -73.9918),
        ("Financial District", 40.7075, -74.0113),
    ],
    "Brooklyn": [
        ("Williamsburg", 40.7081, -73.9571),
        ("Bushwick", 40.6944, -73.9213),
        ("Greenpoint", 40.7304, -73.9510),
        ("Park Slope", 40.6710, -73.9814),
        ("Dumbo", 40.7033, -73.9881),
        ("Bedford-Stuyvesant", 40.6872, -73.9418),
        ("Cobble Hill", 40.6873, -73.9961),
        ("Fort Greene", 40.6920, -73.9740),
        ("Crown Heights", 40.6694, -73.9442),
        ("Red Hook", 40.6751, -74.0099),
    ],
    "Queens": [
        ("Astoria", 40.7644, -73.9235),
        ("Long Island City", 40.7447, -73.9485),
        ("Flushing", 40.7675, -73.8331),
        ("Jackson Heights", 40.7557, -73.8831),
        ("Ridgewood", 40.7000, -73.9050),
        ("Sunnyside", 40.7434, -73.9196),
    ],
}

# category -> (adjectives, signature tags, name fragments)
CATEGORY_PROFILE: dict[str, dict[str, list[str]]] = {
    "restaurant": {
        "adjectives": ["bustling", "refined", "neighborhood", "buzzy", "seasonal"],
        "tags": ["farm-to-table", "tasting menu", "shareable plates", "open kitchen", "natural wine"],
        "names": ["The Quarter", "Lila", "Saffron House", "Maison Olive", "Cervena", "The Larder"],
    },
    "bar": {
        "adjectives": ["laid-back", "lively", "no-frills", "friendly", "divey"],
        "tags": ["craft beer", "happy hour", "sports on TV", "back patio", "pool table"],
        "names": ["The Anchor", "Hudson Tap", "Local 9", "The Watering Hole", "Brass Monkey"],
    },
    "cafe": {
        "adjectives": ["cozy", "sunlit", "minimalist", "charming", "quiet"],
        "tags": ["single-origin coffee", "pastries", "laptop-friendly", "oat milk", "matcha"],
        "names": ["Daybreak", "Ember Coffee", "The Reading Room", "Saint Espresso", "Foglia"],
    },
    "wine_bar": {
        "adjectives": ["intimate", "candlelit", "romantic", "rustic", "elegant"],
        "tags": ["natural wine", "small plates", "cheese board", "by-the-glass", "low-lit"],
        "names": ["Vino & Veritas", "The Cellar", "Decant", "Terroir House", "Amaro"],
    },
    "cocktail_bar": {
        "adjectives": ["sleek", "moody", "inventive", "upscale", "stylish"],
        "tags": ["craft cocktails", "house infusions", "bartender's choice", "amaro list", "date night"],
        "names": ["The Alchemist", "Bitter & Sweet", "Gold Leaf", "The Gilded Olive", "Lantern"],
    },
    "rooftop_bar": {
        "adjectives": ["breezy", "scenic", "see-and-be-seen", "sunset", "panoramic"],
        "tags": ["skyline views", "frozen cocktails", "DJ sets", "open-air", "golden hour"],
        "names": ["Skyline Social", "The Heights", "Cloud Terrace", "Westview Rooftop", "Altitude"],
    },
    "jazz_club": {
        "adjectives": ["historic", "smoky", "soulful", "dim", "storied"],
        "tags": ["live jazz", "late-night sets", "vintage vibe", "table service", "standards"],
        "names": ["Blue Note Lounge", "The Velvet Room", "Smalls Club", "Birdcage", "Minton's Corner"],
    },
    "speakeasy": {
        "adjectives": ["hidden", "cozy", "secretive", "intimate", "dimly-lit"],
        "tags": ["password entry", "creative drinks", "behind-the-door", "no signage", "vintage decor"],
        "names": ["Attaboy", "The Back Room", "Please Don't Tell", "Phone Booth", "The Hidden Key"],
    },
    "ramen_shop": {
        "adjectives": ["steamy", "casual", "no-reservations", "slurp-worthy", "comforting"],
        "tags": ["tonkotsu broth", "handmade noodles", "spicy miso", "counter seating", "chashu"],
        "names": ["Ippudo Annex", "Noodle Republic", "Tonkotsu House", "Slurp Lab", "Menya East"],
    },
    "pizza_spot": {
        "adjectives": ["classic", "blistered-crust", "old-school", "wood-fired", "no-frills"],
        "tags": ["coal oven", "thin crust", "by-the-slice", "fresh mozzarella", "Neapolitan"],
        "names": ["Lucia Pizza", "Coal & Char", "Slice Society", "Forno Rosso", "Pomodoro"],
    },
    "brunch_spot": {
        "adjectives": ["sunny", "leisurely", "bottomless", "popular", "easygoing"],
        "tags": ["bottomless mimosas", "weekend brunch", "ricotta pancakes", "outdoor seating", "eggs benedict"],
        "names": ["Sunday Social", "The Egg & Spoon", "Maple & Thyme", "Brunch Club", "Daisy's"],
    },
    "date_spot": {
        "adjectives": ["romantic", "candlelit", "intimate", "swoon-worthy", "atmospheric"],
        "tags": ["low lighting", "shared plates", "great wine list", "tucked away", "special occasion"],
        "names": ["L'Amour", "The Corner Table", "Lumiere", "Petite Maison", "The Nook"],
    },
}

SYNTHETIC_CATEGORY_WEIGHTS: dict[str, int] = {
    "restaurant": 120,
    "bar": 70,
    "cafe": 60,
    "wine_bar": 35,
    "cocktail_bar": 45,
    "rooftop_bar": 25,
    "jazz_club": 15,
    "speakeasy": 20,
    "ramen_shop": 25,
    "pizza_spot": 35,
    "brunch_spot": 25,
    "date_spot": 25,
}

PRICE_WORDS = {1: "budget-friendly", 2: "moderately priced", 3: "upscale", 4: "splurge-worthy"}


def _jitter(value: float, amount: float = 0.01) -> float:
    """Return value perturbed by a small random offset for unique coordinates."""
    return round(value + random.uniform(-amount, amount), 6)


def generate_synthetic_venues(count: int = TARGET_VENUE_COUNT) -> list[dict[str, Any]]:
    """Generate a realistic synthetic dataset of NYC venues.

    Args:
        count: total number of venues to generate.

    Returns:
        A list of normalized venue dicts spanning all categories and boroughs.
    """
    flat_neighborhoods = [
        (borough, name, lat, lng)
        for borough, hoods in NEIGHBORHOODS.items()
        for (name, lat, lng) in hoods
    ]

    # Expand category weights into a sampling pool.
    pool: list[str] = []
    for cat, weight in SYNTHETIC_CATEGORY_WEIGHTS.items():
        pool.extend([cat] * weight)

    venues: list[dict[str, Any]] = []
    used_names: set[str] = set()
    idx = 0
    while len(venues) < count:
        category = pool[idx % len(pool)] if idx < len(pool) else random.choice(pool)
        idx += 1
        profile = CATEGORY_PROFILE[category]
        borough, hood, lat, lng = random.choice(flat_neighborhoods)

        base_name = random.choice(profile["names"])
        name = base_name
        suffix = 1
        while name in used_names:
            suffix += 1
            name = f"{base_name} {hood.split()[0]}" if suffix == 2 else f"{base_name} {suffix}"
        used_names.add(name)

        price_tier = random.choices([1, 2, 3, 4], weights=[2, 4, 3, 1])[0]
        tags = random.sample(profile["tags"], k=min(3, len(profile["tags"])))

        venues.append(
            {
                "id": f"syn_{len(venues):04d}",
                "name": name,
                "neighborhood": hood,
                "borough": borough,
                "category": category,
                "lat": _jitter(lat),
                "lng": _jitter(lng),
                "price_tier": price_tier,
                "_tags": tags,
                "_adjective": random.choice(profile["adjectives"]),
                "_blurb": "",
            }
        )
    return venues


# --------------------------------------------------------------------------- #
# Description building (shared by both paths)
# --------------------------------------------------------------------------- #
CATEGORY_PHRASE = {
    "restaurant": "restaurant",
    "bar": "bar",
    "cafe": "cafe",
    "wine_bar": "wine bar",
    "cocktail_bar": "cocktail bar",
    "rooftop_bar": "rooftop bar",
    "jazz_club": "jazz club",
    "speakeasy": "speakeasy-style cocktail bar",
    "ramen_shop": "ramen shop",
    "pizza_spot": "pizza spot",
    "brunch_spot": "brunch spot",
    "date_spot": "date-night spot",
}


def build_description(venue: dict[str, Any]) -> str:
    """Construct a natural language description for a venue.

    Combines name, category, neighborhood, price tier and top tags into a
    single fluent sentence suitable for sentence-embedding models.

    Args:
        venue: a normalized venue dict.

    Returns:
        A human-readable description string.
    """
    name = venue["name"]
    cat_phrase = CATEGORY_PHRASE.get(venue["category"], venue["category"].replace("_", " "))
    hood = venue["neighborhood"]
    adjective = venue.get("_adjective") or random.choice(["popular", "well-loved", "go-to"])
    price = PRICE_WORDS.get(venue.get("price_tier", 2), "moderately priced")
    tags = venue.get("_tags") or []

    if venue.get("_blurb"):
        # Prefer a real Foursquare description when present, but still anchor it.
        base = f"{name} is a {cat_phrase} in {hood}. {venue['_blurb'].strip()}"
    else:
        base = f"{name} is a {adjective} {cat_phrase} in {hood}, NYC"

    if tags:
        tag_str = ", ".join(t for t in tags if t)
        base += f", known for {tag_str}"

    base += f". It's a {price} spot."
    return base


def finalize_venues(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach descriptions and strip private fields from raw venues.

    Args:
        raw: list of normalized venue dicts (synthetic or Foursquare).

    Returns:
        Clean venue dicts with the public schema only.
    """
    cleaned: list[dict[str, Any]] = []
    for v in raw:
        if not v.get("lat") or not v.get("lng"):
            continue
        description = build_description(v)
        cleaned.append(
            {
                "id": v["id"],
                "name": v["name"],
                "description": description,
                "neighborhood": v["neighborhood"],
                "category": v["category"],
                "lat": v["lat"],
                "lng": v["lng"],
                "price_tier": v["price_tier"],
            }
        )
    return cleaned


# --------------------------------------------------------------------------- #
# Sample queries
# --------------------------------------------------------------------------- #
def build_sample_queries() -> list[dict[str, Any]]:
    """Return 50 natural-language queries paired with expected venue categories.

    Returns:
        A list of {"query": str, "expected_categories": list[str]} dicts.
    """
    queries: list[tuple[str, list[str]]] = [
        ("romantic wine bar with candlelight", ["wine_bar", "date_spot"]),
        ("cozy speakeasy for a first date", ["speakeasy", "date_spot", "cocktail_bar"]),
        ("rooftop bar with skyline views at sunset", ["rooftop_bar"]),
        ("late night ramen after drinks", ["ramen_shop"]),
        ("classic New York slice of pizza", ["pizza_spot"]),
        ("bottomless mimosa brunch with friends", ["brunch_spot"]),
        ("quiet cafe to work on my laptop", ["cafe"]),
        ("live jazz and a martini", ["jazz_club", "cocktail_bar"]),
        ("hidden cocktail bar with creative drinks", ["speakeasy", "cocktail_bar"]),
        ("dive bar to watch the game", ["bar"]),
        ("intimate restaurant for an anniversary dinner", ["restaurant", "date_spot"]),
        ("natural wine and small plates", ["wine_bar"]),
        ("trendy spot to grab espresso", ["cafe"]),
        ("upscale tasting menu for a special occasion", ["restaurant", "date_spot"]),
        ("casual beer and burgers", ["bar", "restaurant"]),
        ("matcha latte and pastries in the morning", ["cafe", "brunch_spot"]),
        ("moody bar for craft cocktails", ["cocktail_bar"]),
        ("date night with low lighting and great wine", ["date_spot", "wine_bar"]),
        ("rooftop drinks with a DJ", ["rooftop_bar"]),
        ("spicy miso ramen in the East Village", ["ramen_shop"]),
        ("wood-fired Neapolitan pizza", ["pizza_spot"]),
        ("weekend brunch with eggs benedict", ["brunch_spot"]),
        ("soulful live music venue", ["jazz_club"]),
        ("secret bar with a password", ["speakeasy"]),
        ("happy hour after work", ["bar", "cocktail_bar"]),
        ("farm-to-table seasonal dinner", ["restaurant"]),
        ("cheese board and a glass of red", ["wine_bar"]),
        ("cute coffee shop in Williamsburg", ["cafe"]),
        ("fancy cocktails for a celebration", ["cocktail_bar"]),
        ("romantic dinner tucked away", ["date_spot", "restaurant"]),
        ("open-air drinks on a warm night", ["rooftop_bar"]),
        ("comforting bowl of noodles", ["ramen_shop"]),
        ("coal oven pizza old-school style", ["pizza_spot"]),
        ("leisurely Sunday brunch outdoors", ["brunch_spot"]),
        ("dim jazz lounge for late night", ["jazz_club"]),
        ("creative drinks behind an unmarked door", ["speakeasy"]),
        ("friendly neighborhood bar with a patio", ["bar"]),
        ("buzzy restaurant with shareable plates", ["restaurant"]),
        ("candlelit wine bar in the West Village", ["wine_bar", "date_spot"]),
        ("laptop-friendly cafe with oat milk", ["cafe"]),
        ("stylish cocktail spot for date night", ["cocktail_bar", "date_spot"]),
        ("scenic rooftop for golden hour", ["rooftop_bar"]),
        ("handmade tonkotsu ramen", ["ramen_shop"]),
        ("by-the-slice pizza late night", ["pizza_spot"]),
        ("ricotta pancakes and bottomless mimosas", ["brunch_spot"]),
        ("historic jazz club with standards", ["jazz_club"]),
        ("intimate speakeasy with vintage decor", ["speakeasy"]),
        ("upscale wine bar with by-the-glass pours", ["wine_bar"]),
        ("special occasion restaurant in Tribeca", ["restaurant", "date_spot"]),
        ("chill cafe to read a book", ["cafe"]),
    ]
    return [{"query": q, "expected_categories": cats} for q, cats in queries]


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    """Run the full data pipeline and write venues.json + sample_queries.json."""
    random.seed(42)

    raw = fetch_all_foursquare()
    source = "foursquare"
    if len(raw) < TARGET_VENUE_COUNT * 0.5:
        if raw:
            print(f"[pipeline] only {len(raw)} live venues; topping up with synthetic data.")
        else:
            print("[pipeline] using synthetic dataset.")
        synthetic = generate_synthetic_venues(TARGET_VENUE_COUNT - len(raw))
        raw = raw + synthetic
        source = "synthetic" if not FOURSQUARE_API_KEY else "hybrid"

    venues = finalize_venues(raw)
    VENUES_PATH.write_text(json.dumps(venues, indent=2, ensure_ascii=False))
    print(f"[pipeline] wrote {len(venues)} venues to {VENUES_PATH} (source: {source})")

    queries = build_sample_queries()
    QUERIES_PATH.write_text(json.dumps(queries, indent=2, ensure_ascii=False))
    print(f"[pipeline] wrote {len(queries)} sample queries to {QUERIES_PATH}")


if __name__ == "__main__":
    main()
