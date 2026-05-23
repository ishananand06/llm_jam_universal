"""
Step 0: Honest top-5 retrieval re-evaluation of Task 4 paraphrase blockers.

For each (blocker_doc, query) pair in task4_m2_paraphrase.csv, computes:
  - blocker_sim       : GTR-base cosine similarity between blocker and query
  - threshold_sim     : k-th neighbor score from the real NQ corpus (k=5)
  - retrieved_top5    : blocker_sim >= threshold_sim (true top-5 competition)
  - rank              : rank of blocker among top-5 when retrieved (1 = best)
  - sim_over_03       : blocker_sim >= 0.3 (old BBO threshold)
  - jammed_lenient    : jam_success (from Task 4 CSV) AND sim_over_03
  - jammed_honest     : jam_success AND retrieved_top5

Reports:
  ASR_lenient  = jammed_lenient rate (old 0.3-threshold + force-injection judging)
  ASR_honest   = jammed_honest rate (true top-5 AND judged not-answered)
  ASR_injected = raw jam_success rate (force-injection, no retrieval check — what
                 Task 4 reported)

Also computes McNemar test on honest labels (M2 vs Shafran per query pair).
Reconciles with M1's 40% retrieval result for the unoptimised blocker.

Output:
  /home/ishana/scratch/results/step0_paraphrase_rejudged_retrieval.csv

GPU: CUDA_VISIBLE_DEVICES=0  (retrieval only, no LLM)
"""
from __future__ import annotations

import csv
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

os.environ["HF_HOME"] = "/home/ishana/scratch/hf_cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
TASK4_CSV    = Path("/home/ishana/scratch/results/task4_m2_paraphrase.csv")
CKPT_FILE    = Path("/home/ishana/scratch/results/task4_m2_paraphrase_ckpt.pkl")
INDEX_DIR    = Path("/home/ishana/scratch/data/indices/nq/sentence-transformers__gtr-t5-base")
RESULTS_DIR  = Path("/home/ishana/scratch/results")
OUT_CSV      = RESULTS_DIR / "step0_paraphrase_rejudged_retrieval.csv"

GTR_MODEL    = "sentence-transformers/gtr-t5-base"
TOP_K        = 5
OLD_THRESHOLD = 0.3


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_task4_csv() -> list[dict]:
    rows = []
    with open(TASK4_CSV) as f:
        lines = [l for l in f if not l.startswith("#")]
    for row in csv.DictReader(lines):
        rows.append(dict(row))
    log.info("Loaded %d rows from task4 CSV", len(rows))
    return rows


def load_checkpoint() -> dict:
    with open(CKPT_FILE, "rb") as f:
        ckpt = pickle.load(f)
    log.info("Loaded checkpoint: %d classes", len(ckpt))
    return ckpt


def build_blocker_lookup(ckpt: dict) -> dict[tuple[str, str], str]:
    """Return {(class_id, variant): full_blocker_doc}."""
    lookup: dict[tuple[str, str], str] = {}
    for cid, entry in ckpt.items():
        lookup[(cid, "m2")]     = entry["m2_result"].final_doc
        lookup[(cid, "shafran")] = entry["shafran_result"].final_doc
    return lookup


# ── Retrieval check ───────────────────────────────────────────────────────────

def check_top5_retrieval(
    blocker_emb: np.ndarray,
    query: str,
    retriever,
    k: int = TOP_K,
) -> tuple[bool, int | None, float, float]:
    """
    Check whether a pre-embedded blocker would land in top-k for `query`.

    Returns (retrieved_top5, rank, blocker_sim, threshold_sim).
      rank: 1-based rank among top-k docs; None if not retrieved.
    """
    top_k_results = retriever.retrieve_with_scores(query, k=k)
    scores = [s for _, s in top_k_results]
    threshold = scores[-1] if scores else 0.0

    query_emb = retriever.embed_batch([query])[0]
    blocker_sim = float(np.dot(blocker_emb, query_emb))

    # Rank = #docs with strictly higher sim + 1  (rank 1 = closest to query)
    rank_in_top = sum(1 for s in scores if s > blocker_sim) + 1
    retrieved = blocker_sim >= threshold

    return retrieved, (rank_in_top if retrieved else None), blocker_sim, threshold


# ── McNemar test ──────────────────────────────────────────────────────────────

def mcnemar_test(m2_labels: list[bool], shafran_labels: list[bool]) -> float:
    """
    McNemar's test on paired binary labels.
    Returns two-sided p-value using chi-squared approximation (b+c > 20)
    or exact binomial otherwise.
    """
    from scipy.stats import binom, chi2
    b = sum(1 for m, s in zip(m2_labels, shafran_labels) if m and not s)
    c = sum(1 for m, s in zip(m2_labels, shafran_labels) if not m and s)
    log.info("McNemar: b=%d (M2-only), c=%d (Shafran-only)", b, c)
    if b + c == 0:
        return 1.0
    if b + c > 20:
        stat = (abs(b - c) - 1) ** 2 / (b + c)  # continuity correction
        p = float(chi2.sf(stat, df=1))
    else:
        # Exact two-sided binomial
        n = b + c
        p_obs = min(b, c) / n
        p = float(2 * binom.cdf(min(b, c), n, 0.5))
        p = min(p, 1.0)
    return p


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── Load data ─────────────────────────────────────────────────────────────
    rows = load_task4_csv()
    ckpt = load_checkpoint()
    blocker_lookup = build_blocker_lookup(ckpt)

    # ── Load retriever ────────────────────────────────────────────────────────
    log.info("Loading GTR-base retriever from %s", INDEX_DIR)
    from rag.retriever import Retriever
    retriever = Retriever(model_name=GTR_MODEL, score_function="cos_sim")
    retriever.load_index(INDEX_DIR)
    log.info("Index loaded: %d docs", retriever._index.ntotal)

    # ── Pre-embed all unique blockers once ────────────────────────────────────
    unique_keys = list({(r["class_id"], r["variant"]) for r in rows})
    log.info("Pre-embedding %d unique blocker docs...", len(unique_keys))
    blocker_embs: dict[tuple[str, str], np.ndarray] = {}
    for key in unique_keys:
        doc = blocker_lookup.get(key)
        if doc is None:
            log.warning("No blocker found in checkpoint for %s/%s", *key)
            continue
        blocker_embs[key] = retriever.embed_batch([doc])[0]
    log.info("Blocker embeddings ready.")

    # ── Evaluate each row ─────────────────────────────────────────────────────
    out_rows: list[dict] = []
    for i, row in enumerate(rows):
        cid     = row["class_id"]
        variant = row["variant"]
        query   = row["query_text"]
        jam     = int(row["jam_success"])

        key = (cid, variant)
        b_emb = blocker_embs.get(key)
        if b_emb is None:
            log.warning("Missing blocker embedding for %s/%s — skipping", cid, variant)
            continue

        retrieved, rank, blocker_sim, threshold_sim = check_top5_retrieval(
            b_emb, query, retriever
        )

        sim_over_03   = bool(blocker_sim >= OLD_THRESHOLD)
        jammed_lenient = bool(jam and sim_over_03)
        jammed_honest  = bool(jam and retrieved)

        out_rows.append({
            "class_id":        cid,
            "variant":         variant,
            "query_idx":       row["query_idx"],
            "query_text":      query,
            "blocker_sim":     round(blocker_sim, 5),
            "threshold_sim":   round(threshold_sim, 5),
            "retrieved_top5":  int(retrieved),
            "rank":            rank if rank is not None else "",
            "sim_over_03":     int(sim_over_03),
            "jam_success":     jam,
            "jammed_lenient":  int(jammed_lenient),
            "jammed_honest":   int(jammed_honest),
        })

        if (i + 1) % 40 == 0:
            log.info("  %d/%d rows processed...", i + 1, len(rows))

    log.info("Retrieval checks complete. %d rows.", len(out_rows))

    # ── Save CSV ──────────────────────────────────────────────────────────────
    fieldnames = [
        "class_id", "variant", "query_idx", "query_text",
        "blocker_sim", "threshold_sim", "retrieved_top5", "rank",
        "sim_over_03", "jam_success", "jammed_lenient", "jammed_honest",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    log.info("Saved %d rows → %s", len(out_rows), OUT_CSV)

    # ── Compute and print ASR summary ─────────────────────────────────────────
    _print_summary(out_rows)

    log.info("Total runtime: %.1f s", time.time() - t0)


def _print_summary(rows: list[dict]) -> None:
    from collections import defaultdict

    def rate(vals: list[int]) -> str:
        n = len(vals)
        s = sum(vals)
        return f"{s}/{n} = {100*s/n:.1f}%" if n else "0/0"

    # Per-variant accumulation
    buckets: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = r["variant"]
        buckets[v]["retrieved_top5"].append(int(r["retrieved_top5"]))
        buckets[v]["sim_over_03"].append(int(r["sim_over_03"]))
        buckets[v]["jam_success"].append(int(r["jam_success"]))
        buckets[v]["jammed_lenient"].append(int(r["jammed_lenient"]))
        buckets[v]["jammed_honest"].append(int(r["jammed_honest"]))

    print()
    print("=" * 70)
    print("STEP 0 — HONEST RETRIEVAL RE-EVALUATION SUMMARY")
    print("=" * 70)
    header = f"{'Metric':<28} {'M2':>18} {'Shafran':>18}"
    print(header)
    print("-" * 66)

    metrics = [
        ("retrieved_top5",  "Retrieved (real top-5)"),
        ("sim_over_03",     "Retrieved (sim >= 0.3, lenient)"),
        ("jam_success",     "ASR_injected (force-inject)"),
        ("jammed_lenient",  "ASR_lenient (0.3 thresh + judged)"),
        ("jammed_honest",   "ASR_honest (top-5 + judged)"),
    ]
    for key, label in metrics:
        m2_vals  = buckets["m2"][key]
        sh_vals  = buckets["shafran"][key]
        print(f"  {label:<26} {rate(m2_vals):>18} {rate(sh_vals):>18}")

    print("=" * 70)

    # Compute McNemar on honest labels
    m2_honest      = buckets["m2"]["jammed_honest"]
    shafran_honest = buckets["shafran"]["jammed_honest"]

    # Pair by (class_id, query_idx) — must sort both lists consistently
    m2_rows = sorted(
        [r for r in rows if r["variant"] == "m2"],
        key=lambda r: (r["class_id"], int(r["query_idx"]))
    )
    sh_rows = sorted(
        [r for r in rows if r["variant"] == "shafran"],
        key=lambda r: (r["class_id"], int(r["query_idx"]))
    )
    if len(m2_rows) == len(sh_rows):
        m2_labels = [bool(r["jammed_honest"]) for r in m2_rows]
        sh_labels = [bool(r["jammed_honest"]) for r in sh_rows]
        p = mcnemar_test(m2_labels, sh_labels)
        print(f"\nMcNemar test (honest labels, M2 vs Shafran): p = {p:.4f}")
        sig = "significant (p < 0.05)" if p < 0.05 else "not significant (p >= 0.05)"
        print(f"  → {sig}")
    else:
        print(f"\nWarning: M2 and Shafran row counts differ ({len(m2_rows)} vs {len(sh_rows)}); McNemar skipped.")

    # Inflation estimate
    print()
    for variant in ["m2", "shafran"]:
        b = buckets[variant]
        n = len(b["jam_success"])
        asr_inj  = 100 * sum(b["jam_success"]) / n
        asr_hon  = 100 * sum(b["jammed_honest"]) / n
        inflation = asr_inj - asr_hon
        print(
            f"  {variant:<10} ASR_injected={asr_inj:.1f}%  "
            f"ASR_honest={asr_hon:.1f}%  "
            f"inflation={inflation:+.1f} pp"
        )
    print()


if __name__ == "__main__":
    main()
