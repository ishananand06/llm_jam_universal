"""
Compare Mistral-7B vs Llama-3.1-8B paraphrase classes.

Samples 10 random classes from each, prints them side by side, then
picks the model with higher mean within-class cosine similarity as the
winner. The winner is copied to:
  paraphrase_classes.json
  paraphrase_embeddings.npy
The loser's files are deleted.
"""
from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path

import numpy as np

DATA_DIR = Path("/home/ishana/scratch/data/classes")
SEED = 42


def load(tag: str) -> tuple[list[dict], np.ndarray]:
    with open(DATA_DIR / f"paraphrase_classes_{tag}.json") as f:
        classes = json.load(f)
    embs = np.load(DATA_DIR / f"paraphrase_embeddings_{tag}.npy")
    return classes, embs


def stats(classes: list[dict]) -> dict:
    sims = [c["within_class_similarity"] for c in classes]
    return {
        "mean": float(np.mean(sims)),
        "min": float(np.min(sims)),
        "max": float(np.max(sims)),
        "below_080": sum(1 for s in sims if s < 0.80),
        "below_085": sum(1 for s in sims if s < 0.85),
    }


def main():
    random.seed(SEED)

    mistral, mistral_embs = load("mistral")
    llama, llama_embs = load("llama")

    assert len(mistral) == len(llama), "Class count mismatch"
    n = len(mistral)

    sample_idx = sorted(random.sample(range(n), min(10, n)))

    # ── side-by-side random sample ─────────────────────────────────────────────
    print("=" * 80)
    print(f"RANDOM SAMPLE ({len(sample_idx)} of {n} classes)")
    print("=" * 80)
    for i, idx in enumerate(sample_idx, 1):
        m = mistral[idx]
        l = llama[idx]
        print(f"\n── Class {i} (idx={idx}) ───────────────────────────────────────────")
        print(f"  Original: {m['original_query']}")
        print(f"  {'MISTRAL (sim=%.3f)' % m['within_class_similarity']:40s}  {'LLAMA (sim=%.3f)' % l['within_class_similarity']}")
        for j in range(5):
            mp = m["paraphrases"][j] if j < len(m["paraphrases"]) else ""
            lp = l["paraphrases"][j] if j < len(l["paraphrases"]) else ""
            print(f"  {j+1}. {mp[:55]:<55}  {j+1}. {lp[:55]}")

    # ── aggregate stats ────────────────────────────────────────────────────────
    ms = stats(mistral)
    ls = stats(llama)

    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)
    print(f"{'Metric':<30} {'Mistral-7B':>12} {'Llama-3.1-8B':>14}")
    print("-" * 58)
    print(f"{'Mean within-class sim':<30} {ms['mean']:>12.4f} {ls['mean']:>14.4f}")
    print(f"{'Min within-class sim':<30} {ms['min']:>12.4f} {ls['min']:>14.4f}")
    print(f"{'Max within-class sim':<30} {ms['max']:>12.4f} {ls['max']:>14.4f}")
    print(f"{'Classes below 0.80':<30} {ms['below_080']:>12d} {ls['below_080']:>14d}")
    print(f"{'Classes below 0.85':<30} {ms['below_085']:>12d} {ls['below_085']:>14d}")

    # ── pick winner ────────────────────────────────────────────────────────────
    if ls["mean"] >= ms["mean"]:
        winner, loser = "llama", "mistral"
        winner_label = "Llama-3.1-8B-Instruct"
    else:
        winner, loser = "mistral", "llama"
        winner_label = "Mistral-7B-Instruct-v0.2"

    print("\n" + "=" * 80)
    print(f"WINNER: {winner_label}  (mean sim: {stats(mistral if winner=='mistral' else llama)['mean']:.4f})")
    print("=" * 80)

    # Copy winner → canonical filenames
    shutil.copy2(
        DATA_DIR / f"paraphrase_classes_{winner}.json",
        DATA_DIR / "paraphrase_classes.json",
    )
    shutil.copy2(
        DATA_DIR / f"paraphrase_embeddings_{winner}.npy",
        DATA_DIR / "paraphrase_embeddings.npy",
    )
    print(f"Copied {winner} → paraphrase_classes.json + paraphrase_embeddings.npy")

    # Delete loser
    (DATA_DIR / f"paraphrase_classes_{loser}.json").unlink()
    (DATA_DIR / f"paraphrase_embeddings_{loser}.npy").unlink()
    print(f"Deleted {loser} files")

    print(f"\nFinal files:")
    for f in sorted(DATA_DIR.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
