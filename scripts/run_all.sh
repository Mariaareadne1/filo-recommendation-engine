#!/usr/bin/env bash
#
# run_all.sh — end-to-end filo pipeline.
#
# Installs dependencies, builds the venue dataset, fine-tunes the recommender,
# then starts the FastAPI backend. Designed to be run from anywhere; it resolves
# paths relative to the repo root.
#
# Usage:
#   bash scripts/run_all.sh              # full pipeline + start API
#   SKIP_TRAIN=1 bash scripts/run_all.sh # skip fine-tuning (use base model)
#   SKIP_API=1   bash scripts/run_all.sh # build data + train, but don't serve
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${VENV_DIR:-.venv}"
PORT="${PORT:-8000}"

echo "==> filo pipeline starting (repo: $REPO_ROOT)"

# --------------------------------------------------------------------------- #
# 1. Python environment + dependencies
# --------------------------------------------------------------------------- #
if [ ! -d "$VENV_DIR" ]; then
  echo "==> creating virtualenv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> installing requirements"
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt

# --------------------------------------------------------------------------- #
# 2. Data pipeline
# --------------------------------------------------------------------------- #
echo "==> [phase 1] building venue dataset"
python data/process.py

# --------------------------------------------------------------------------- #
# 3. Fine-tuning (optional)
# --------------------------------------------------------------------------- #
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
  echo "==> [phase 2] SKIP_TRAIN=1 set; skipping fine-tuning (API will use base MiniLM)"
else
  echo "==> [phase 2] fine-tuning the recommender"
  # train/finetune.py imports sibling modules, so run it from train/.
  ( cd train && python finetune.py )
fi

# --------------------------------------------------------------------------- #
# 4. API
# --------------------------------------------------------------------------- #
if [ "${SKIP_API:-0}" = "1" ]; then
  echo "==> [phase 3] SKIP_API=1 set; not starting the API."
  echo "==> done. Start it later with: uvicorn api.main:app --port $PORT"
  exit 0
fi

echo "==> [phase 3] starting FastAPI on http://localhost:$PORT"
echo "    health:    http://localhost:$PORT/health"
echo "    recommend: POST http://localhost:$PORT/recommend"
echo "    (frontend: cd frontend && npm install && npm run dev -> http://localhost:3000)"
exec uvicorn api.main:app --port "$PORT"
