# filo — NYC venue recommendation engine

filo turns a natural-language vibe ("romantic wine bar with candlelight") into a
ranked list of real NYC venues. It pairs a fine-tuned sentence-embedding model
with a FastAPI semantic-search backend and a dark, NYC-inspired React frontend.

```
query ──▶ fine-tuned MiniLM ──▶ cosine similarity over venue embeddings ──▶ ranked venues
```

## Architecture

| Phase | Path | What it does |
|-------|------|--------------|
| 1. Data | `data/process.py` | Fetches ~500 NYC venues from Foursquare (synthetic fallback) and builds NL descriptions |
| 2. Train | `train/finetune.py` | Fine-tunes `all-MiniLM-L6-v2` with contrastive loss, logs to W&B |
| 3. API | `api/main.py` | FastAPI semantic search over cached venue embeddings |
| 4. UI | `frontend/` | Vite + React + Tailwind search experience |

Each phase is independently runnable.

## Setup

```bash
# 1. Python env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure keys (all optional — the pipeline degrades gracefully)
cp .env.example .env
# edit .env and add FOURSQUARE_API_KEY / WANDB_API_KEY / HF_TOKEN
```

> **Keys are optional.** Without `FOURSQUARE_API_KEY` the pipeline generates a
> realistic synthetic dataset. Without `WANDB_API_KEY` training runs in W&B
> offline mode. Without `HF_TOKEN` the model is saved locally only.

## Running each phase

```bash
# Phase 1 — build the dataset
python data/process.py
#   -> data/venues.json, data/sample_queries.json

# Phase 2 — fine-tune (CPU-friendly, ~3-5 epochs, < 10 min)
python train/finetune.py
#   -> models/filo-recommender/

# Phase 3 — start the API
uvicorn api.main:app --reload --port 8000
#   GET  http://localhost:8000/health
#   POST http://localhost:8000/recommend  {"query": "...", "top_k": 5}

# Phase 4 — start the frontend
cd frontend && npm install && npm run dev
#   -> http://localhost:3000
```

Or run everything in sequence:

```bash
bash scripts/run_all.sh
```

## API

`POST /recommend`

```json
{ "query": "cozy wine bar for a date", "top_k": 5 }
```

returns

```json
[
  {
    "name": "Decant",
    "description": "Decant is an intimate wine bar in the West Village ...",
    "neighborhood": "West Village",
    "category": "wine_bar",
    "score": 0.83
  }
]
```

## Notes

- The API falls back to base `all-MiniLM-L6-v2` if no fine-tuned checkpoint
  exists in `models/filo-recommender/`, so Phase 3 works before Phase 2 runs.
- Heavy ML dependencies (`torch`, `sentence-transformers`) are only needed for
  Phases 2 and 3. Phase 1 needs only `requests` + `python-dotenv`.
