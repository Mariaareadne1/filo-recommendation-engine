# filo-recommendation-engine

The ML backbone for [Filo](https://github.com/Mariaareadne1/Filo) — a NYC venue discovery app. This repo is a learning project built to go deeper into machine learning: fine-tuning sentence embeddings, building a PyTorch training loop from scratch, logging experiments with Weights & Biases, and serving predictions through a REST API.

Type a query like `"cozy wine bar for a date"` or `"late night ramen in Brooklyn"` and get back ranked NYC venues that actually match the vibe.

---

## What this is

Filo (the mobile app) helps people find places in NYC. This engine is the recommendation layer — it understands natural language queries and matches them to venues using semantic similarity rather than keyword matching.

Built across four phases as a hands-on ML learning project:

| Phase | What was built |
|---|---|
| 1 — Data pipeline | Foursquare fetch + synthetic fallback → 500 NYC venues |
| 2 — Fine-tuning | MiniLM-L6-v2 fine-tuned with contrastive loss, W&B logging |
| 3 — API | FastAPI endpoint with cached cosine similarity search |
| 4 — Frontend | React search UI with venue cards |

---

## How it works

A sentence-transformers model (`all-MiniLM-L6-v2`) is fine-tuned on pairs of natural language queries and venue descriptions using `MultipleNegativesRankingLoss`. At inference time, a query like `"jazz club with good cocktails"` is embedded into the same vector space as all 500 venue descriptions, and the top-k closest venues by cosine similarity are returned.

```
query: "jazz club with good cocktails"
   ↓  embed (fine-tuned MiniLM)
[0.23, -0.11, 0.87, ...]
   ↓  cosine similarity against 500 venue embeddings
top 5 → Smalls Jazz Club (0.81), Mezzrow (0.79), Fat Cat (0.76) ...
```

---

## Project structure

```
filo-recommendation-engine/
├── data/
│   ├── process.py          # Foursquare fetch + synthetic fallback
│   ├── venues.json         # 500 NYC venues (generated)
│   └── sample_queries.json # 50 query → category training pairs
├── train/
│   ├── dataset.py          # PyTorch Dataset: query/positive + hard negatives
│   └── finetune.py         # Training loop, W&B logging, checkpoint save
├── api/
│   ├── embeddings.py       # Load model, cache embeddings, cosine search
│   └── main.py             # FastAPI: GET /health, POST /recommend
├── frontend/
│   └── src/
│       ├── App.jsx
│       └── components/
│           ├── SearchBar.jsx
│           └── VenueCard.jsx
├── scripts/
│   └── run_all.sh          # One-shot: install → data → train → serve
├── .env.example
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/Mariaareadne1/filo-recommendation-engine
cd filo-recommendation-engine

cp .env.example .env
# Add your keys to .env (all optional — synthetic data works without them)

bash scripts/run_all.sh
```

Or run each phase manually:

```bash
# 1. Set up environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate venue data
python data/process.py

# 3. Fine-tune the model (~30s on CPU)
python train/finetune.py

# 4. Start the API
uvicorn api.main:app --reload --port 8000

# 5. Start the frontend (separate terminal)
cd frontend && npm install && npm run dev
```

---

## API

```
POST /recommend
{
  "query": "cozy wine bar for a date",
  "top_k": 5
}
```

Returns:
```json
[
  {
    "name": "Anfora",
    "neighborhood": "West Village",
    "category": "wine_bar",
    "description": "Anfora is a dimly lit wine bar in the West Village...",
    "score": 0.81
  },
  ...
]
```

Also: `GET /health` → `{"status": "ok"}`

---

## Environment variables

| Variable | Required | What it's for |
|---|---|---|
| `FOURSQUARE_API_KEY` | No | Real venue data (falls back to synthetic) |
| `WANDB_API_KEY` | No | Training metrics in W&B dashboard |
| `HF_TOKEN` | No | Push fine-tuned model to Hugging Face Hub |

---

## Training details

- **Base model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Loss:** `MultipleNegativesRankingLoss` (contrastive)
- **Training time:** ~30s on CPU, 3–5 epochs
- **Logged to W&B:** loss/step, MRR@5, learning rate, epoch
- **Checkpoint:** saved to `models/filo-recommender/` (gitignored)

Note: with synthetic data, MRR@5 reads 1.0 because descriptions contain category words directly. Against real Foursquare data the metric becomes meaningful — swap in a `FOURSQUARE_API_KEY` to see it in action.

---

## Stack

**ML:** PyTorch · sentence-transformers · Weights & Biases · Hugging Face  
**Backend:** FastAPI · NumPy (cosine similarity) · Pydantic  
**Frontend:** React 19 · Vite · Tailwind CSS v4  
**Data:** Foursquare Places API (+ synthetic fallback)

---

## Related

**[Filo (mobile app)](https://github.com/Mariaareadne1/Filo)** — the React Native / Expo app this engine is built for. Expo + Supabase + Mapbox, helping people discover NYC venues on their phone.

---

## Why I built this

Filo started as a mobile app project. This repo is the ML extension of it — a way to go hands-on with the parts of machine learning I wanted to understand better: how fine-tuning actually works, what a training loop looks like in raw PyTorch, how embeddings encode meaning, and how you take a trained model and turn it into something you can call from an app. Every phase was a new thing to figure out.
