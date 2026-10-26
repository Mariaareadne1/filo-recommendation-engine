"""Fine-tune all-MiniLM-L6-v2 for filo venue retrieval.

Uses a contrastive objective (MultipleNegativesRankingLoss) over
(query, positive_venue, hard_negative) triplets, with a custom PyTorch training
loop. Logs training loss per step, validation MRR@5, learning rate and epoch to
Weights & Biases (project ``filo-recommender``). Saves the best checkpoint by
MRR@5 to ``models/filo-recommender/`` and optionally pushes to the Hugging Face
Hub when ``HF_TOKEN`` is set.

Designed to run on CPU in well under 10 minutes (small model, small batch,
3-5 epochs).

Run:
    python train/finetune.py
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from dataset import (  # noqa: E402  (local module, after dotenv)
    build_pairs,
    load_queries,
    load_venues,
    split_queries,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "models" / "filo-recommender"

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 4
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WARMUP_FRAC = 0.1
MAX_SEQ_LENGTH = 128
SEED = 42


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch RNGs for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_wandb() -> Any:
    """Initialize a Weights & Biases run.

    Falls back to offline mode if no ``WANDB_API_KEY`` is configured so training
    never blocks on credentials.

    Returns:
        The wandb module (already ``init``-ed), or None if wandb is unavailable.
    """
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed; skipping experiment logging.")
        return None

    if not os.getenv("WANDB_API_KEY"):
        os.environ["WANDB_MODE"] = "offline"
        print("[wandb] no WANDB_API_KEY; running in offline mode.")

    wandb.init(
        project="filo-recommender",
        config={
            "base_model": BASE_MODEL,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "warmup_frac": WARMUP_FRAC,
            "loss": "MultipleNegativesRankingLoss",
        },
    )
    return wandb


def make_dataloader(pairs: list[dict[str, Any]], model: SentenceTransformer) -> DataLoader:
    """Build a DataLoader of InputExamples using the model's smart collate.

    When a hard negative is present each example is a triplet
    [anchor, positive, negative]; otherwise a pair [anchor, positive]. The
    sentence-transformers collate function tokenizes and batches these.

    Args:
        pairs: training pairs from build_pairs().
        model: the SentenceTransformer (provides the tokenizing collate fn).

    Returns:
        A torch DataLoader yielding (features, labels) batches.
    """
    examples: list[InputExample] = []
    for p in pairs:
        texts = [p["query"], p["positive"]]
        if p.get("negative"):
            texts.append(p["negative"])
        examples.append(InputExample(texts=texts))

    return DataLoader(
        examples,
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=model.smart_batching_collate,
        drop_last=True,
    )


@torch.no_grad()
def evaluate_mrr(
    model: SentenceTransformer,
    queries: list[dict[str, Any]],
    venues: list[dict[str, Any]],
    k: int = 5,
) -> float:
    """Compute MRR@k over the sample queries against the full venue corpus.

    A venue is relevant to a query if its category is in the query's
    ``expected_categories``. The reciprocal rank is the inverse position of the
    first relevant venue within the top-k (0 if none).

    Args:
        model: the (current) SentenceTransformer.
        queries: sample queries with expected_categories.
        venues: full venue catalog.
        k: cutoff for the metric.

    Returns:
        Mean reciprocal rank at k across all queries.
    """
    venue_texts = [v["description"] for v in venues]
    venue_cats = [v["category"] for v in venues]
    venue_emb = model.encode(
        venue_texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
    )
    query_texts = [q["query"] for q in queries]
    query_emb = model.encode(
        query_texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
    )

    sims = query_emb @ venue_emb.T  # cosine (normalized) similarity
    topk = torch.topk(sims, k=min(k, len(venues)), dim=1).indices.cpu().numpy()

    rr_total = 0.0
    for i, q in enumerate(queries):
        expected = set(q["expected_categories"])
        rr = 0.0
        for rank, idx in enumerate(topk[i], start=1):
            if venue_cats[idx] in expected:
                rr = 1.0 / rank
                break
        rr_total += rr
    return rr_total / max(1, len(queries))


def train() -> None:
    """Run the full fine-tuning loop and save the best checkpoint."""
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device: {device}")

    venues = load_venues()
    # Split at the query level *before* building pairs. Previously pairs were
    # built from every query and validation used queries[::4], so each
    # validation query had also been trained on and MRR@5 was leaked.
    train_queries, val_queries = split_queries(load_queries())
    overlap = {q["query"] for q in train_queries} & {q["query"] for q in val_queries}
    assert not overlap, f"validation queries leaked into training: {sorted(overlap)}"
    train_pairs = build_pairs(venues, train_queries)
    print(
        f"[train] {len(train_pairs)} train pairs from {len(train_queries)} queries, "
        f"{len(val_queries)} held-out val queries"
    )

    model = SentenceTransformer(BASE_MODEL, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH

    train_loader = make_dataloader(train_pairs, model)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(total_steps * WARMUP_FRAC)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            1.0,
            max(
                0.0,
                (step + 1) / max(1, warmup_steps)
                if step < warmup_steps
                else (total_steps - step) / max(1, total_steps - warmup_steps),
            ),
        ),
    )

    wandb = init_wandb()
    best_mrr = -1.0
    global_step = 0

    baseline_mrr = evaluate_mrr(model, val_queries, venues)
    print(f"[train] baseline (pre-finetune) MRR@5 = {baseline_mrr:.4f}")
    if wandb:
        wandb.log({"val/mrr@5": baseline_mrr, "epoch": 0}, step=0)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for features, labels in train_loader:
            features = [
                {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in f.items()}
                for f in features
            ]
            labels = labels.to(device)

            optimizer.zero_grad()
            loss_value = train_loss(features, labels)
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            step_loss = loss_value.item()
            epoch_loss += step_loss
            if wandb:
                wandb.log(
                    {
                        "train/loss": step_loss,
                        "train/lr": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    },
                    step=global_step,
                )

        avg_loss = epoch_loss / steps_per_epoch
        mrr = evaluate_mrr(model, val_queries, venues)
        print(f"[train] epoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}  MRR@5={mrr:.4f}")
        if wandb:
            wandb.log({"val/mrr@5": mrr, "train/epoch_loss": avg_loss, "epoch": epoch}, step=global_step)

        if mrr >= best_mrr:
            best_mrr = mrr
            OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(OUTPUT_DIR))
            print(f"[train] saved best checkpoint (MRR@5={best_mrr:.4f}) -> {OUTPUT_DIR}")

    # Ensure something is always saved even if MRR never improved.
    if not OUTPUT_DIR.exists():
        model.save(str(OUTPUT_DIR))

    print(f"[train] done. best MRR@5 = {best_mrr:.4f}")
    if wandb:
        wandb.summary["best_mrr@5"] = best_mrr
        wandb.finish()

    maybe_push_to_hub()


def maybe_push_to_hub() -> None:
    """Push the saved model to the Hugging Face Hub if HF_TOKEN is set."""
    token = os.getenv("HF_TOKEN")
    if not token:
        print("[hub] HF_TOKEN not set; skipping push to Hugging Face Hub.")
        return
    if not OUTPUT_DIR.exists():
        print("[hub] no checkpoint found; skipping push.")
        return
    try:
        model = SentenceTransformer(str(OUTPUT_DIR))
        repo_id = os.getenv("HF_REPO_ID", "filo-recommender")
        model.push_to_hub(repo_id, token=token, exist_ok=True)
        print(f"[hub] pushed model to {repo_id}")
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[hub] push failed: {exc}")


if __name__ == "__main__":
    train()
