"""
M1 retrieval evaluation.

For each query class (paraphrase + entity), builds two retrieval sub-documents:
  - simple:   representative query (closest embedding to class centroid)
  - hotflip:  gradient-optimised tokens toward class centroid (BGE-large oracle)

Constructs the full blocker (d_r + 50 '!' suffix) and evaluates whether it
would appear in the top-5 retrieved documents for every query in the class.
Retrieval uses GTR-base (the RAG's actual retriever), NOT BGE-large — this
intentionally tests black-box transferability.

Output:
  /home/ishana/scratch/results/task3_m1_retrieval.csv
  (columns: class_id, class_type, variant, query_id, query_text,
            retrieved, rank, blocker_sim, threshold_sim)
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure src/ on path
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from attacks.m1_retrieval import (
    build_dr_representative,
    build_dr_hotflip,
    make_blocker,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
HF_HOME        = "/home/ishana/scratch/hf_cache"
DATA_DIR       = Path("/home/ishana/scratch/data/classes")
INDEX_DIR      = Path("/home/ishana/scratch/data/indices/nq/sentence-transformers__gtr-t5-base")
NQ_DATA_DIR    = Path("/home/ishana/scratch/data/nq")
RESULTS_DIR    = Path("/home/ishana/scratch/results")
OUT_CSV        = RESULTS_DIR / "task3_m1_retrieval.csv"

GTR_MODEL  = "sentence-transformers/gtr-t5-base"
BGE_MODEL  = "BAAI/bge-large-en-v1.5"
TOP_K      = 5
GPU_DEVICE = "cuda"

os.environ["HF_HOME"] = HF_HOME
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_classes(json_path: Path, npy_path: Path) -> list[dict]:
    with open(json_path) as f:
        classes = json.load(f)
    embs = np.load(npy_path)  # (N, max_size, D) for entity; (N, 6, D) for para
    for i, cls in enumerate(classes):
        n = len(cls.get("paraphrases", cls.get("queries", []))) + (
            1 if "paraphrases" in cls else 0
        )
        cls["_embeddings"] = embs[i, :n]  # (n_queries, D)
        # Collect queries as a flat list
        if "paraphrases" in cls:
            cls["_queries"] = [cls["original_query"]] + cls["paraphrases"]
        else:
            cls["_queries"] = cls["queries"]
        cls["_centroid"] = np.array(cls["centroid"], dtype=np.float32)
    return classes


def load_nq_corpus() -> dict[str, str]:
    corpus = {}
    with open(NQ_DATA_DIR / "corpus.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            text = f"{obj.get('title', '')} {obj['text']}".strip()
            corpus[obj["_id"]] = text
    log.info("Loaded NQ corpus: %d docs", len(corpus))
    return corpus


# ── Retrieval evaluation ──────────────────────────────────────────────────────

def eval_blocker_retrieval(
    blocker_text: str,
    queries: list[str],
    retriever,
    k: int = TOP_K,
) -> list[dict]:
    """
    Check whether `blocker_text` would be in top-k for each query.

    Does NOT modify the FAISS index. Instead, computes blocker's embedding and
    compares its similarity against the k-th neighbour score of each query.

    Returns list of dicts with keys: retrieved, rank, blocker_sim, threshold_sim.
    """
    blocker_emb = retriever.embed_batch([blocker_text])[0]  # (D,)
    results = []
    for query in queries:
        top_k_results = retriever.retrieve_with_scores(query, k=k)
        # top_k_results: [(doc_id, score), ...] in descending order
        scores = [s for _, s in top_k_results]
        threshold = scores[-1] if scores else 0.0

        query_emb = retriever.embed_batch([query])[0]
        blocker_sim = float(np.dot(blocker_emb, query_emb))

        # Rank blocker among top-k: count docs with higher sim + 1
        rank_among_top = sum(1 for s in scores if s > blocker_sim) + 1
        retrieved = blocker_sim >= threshold

        results.append({
            "retrieved": retrieved,
            "rank": rank_among_top if retrieved else None,
            "blocker_sim": round(blocker_sim, 5),
            "threshold_sim": round(threshold, 5),
        })
    return results


# ── BGE-large loader for HotFlip ──────────────────────────────────────────────

def load_bge_for_hotflip(device: str):
    from transformers import AutoModel, AutoTokenizer
    log.info("Loading BGE-large for HotFlip oracle...")
    tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL)
    model = AutoModel.from_pretrained(BGE_MODEL).to(device)
    model.eval()
    log.info("BGE-large loaded on %s", device)
    return model, tokenizer


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ── Load class data ───────────────────────────────────────────────────────
    para_classes = load_classes(
        DATA_DIR / "paraphrase_classes.json",
        DATA_DIR / "paraphrase_embeddings.npy",
    )
    entity_classes = load_classes(
        DATA_DIR / "entity_classes.json",
        DATA_DIR / "entity_embeddings.npy",
    )
    log.info(
        "Loaded %d paraphrase classes, %d entity classes",
        len(para_classes), len(entity_classes),
    )

    # ── Load GTR-base retriever (RAG retriever — evaluation model) ────────────
    log.info("Loading GTR-base retriever...")
    from rag.retriever import Retriever
    retriever = Retriever(model_name=GTR_MODEL, score_function="cos_sim")
    retriever.load_index(INDEX_DIR)
    log.info("GTR-base index loaded: %d docs", retriever._index.ntotal)

    # ── Load BGE-large for HotFlip oracle (kept in memory throughout) ─────────
    bge_model, bge_tokenizer = load_bge_for_hotflip(GPU_DEVICE)

    # ── CSV writer setup ──────────────────────────────────────────────────────
    fieldnames = [
        "class_id", "class_type", "variant",
        "query_idx", "query_text",
        "retrieved", "rank", "blocker_sim", "threshold_sim",
        "d_r_text",
    ]
    out_rows: list[dict] = []

    # ── Evaluate all classes ──────────────────────────────────────────────────
    all_classes = [
        (cls, "paraphrase") for cls in para_classes
    ] + [
        (cls, "entity") for cls in entity_classes
    ]

    for cls_idx, (cls, cls_type) in enumerate(all_classes):
        class_id  = cls.get("class_id", f"{cls_type}_{cls_idx:03d}")
        queries   = cls["_queries"]
        embs      = cls["_embeddings"]          # (n_queries, D) — BGE embeddings
        centroid  = cls["_centroid"]            # (D,)

        log.info(
            "[%d/%d] %s  class=%s  n_queries=%d",
            cls_idx + 1, len(all_classes), cls_type, class_id, len(queries),
        )

        # ── Simple variant ────────────────────────────────────────────────────
        d_r_simple = build_dr_representative(queries, embs, centroid)
        blocker_simple = make_blocker(d_r_simple)
        ret_simple = eval_blocker_retrieval(blocker_simple, queries, retriever)

        for q_idx, (query, ret) in enumerate(zip(queries, ret_simple)):
            out_rows.append({
                "class_id": class_id,
                "class_type": cls_type,
                "variant": "simple",
                "query_idx": q_idx,
                "query_text": query,
                "retrieved": ret["retrieved"],
                "rank": ret["rank"],
                "blocker_sim": ret["blocker_sim"],
                "threshold_sim": ret["threshold_sim"],
                "d_r_text": d_r_simple[:120],
            })

        # ── HotFlip variant ───────────────────────────────────────────────────
        d_r_hotflip = build_dr_hotflip(
            queries=queries,
            query_embeddings=embs,
            centroid=centroid,
            bge_model=bge_model,
            bge_tokenizer=bge_tokenizer,
            device=GPU_DEVICE,
        )
        blocker_hotflip = make_blocker(d_r_hotflip)
        ret_hotflip = eval_blocker_retrieval(blocker_hotflip, queries, retriever)

        for q_idx, (query, ret) in enumerate(zip(queries, ret_hotflip)):
            out_rows.append({
                "class_id": class_id,
                "class_type": cls_type,
                "variant": "hotflip",
                "query_idx": q_idx,
                "query_text": query,
                "retrieved": ret["retrieved"],
                "rank": ret["rank"],
                "blocker_sim": ret["blocker_sim"],
                "threshold_sim": ret["threshold_sim"],
                "d_r_text": d_r_hotflip[:120],
            })

        # Quick per-class summary
        s_rate = 100 * sum(r["retrieved"] for r in ret_simple) / len(ret_simple)
        h_rate = 100 * sum(r["retrieved"] for r in ret_hotflip) / len(ret_hotflip)
        log.info(
            "  simple=%.0f%%  hotflip=%.0f%%  |  d_r_simple=%r  d_r_hf=%r",
            s_rate, h_rate, d_r_simple[:50], d_r_hotflip[:50],
        )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    log.info("Saved %d rows → %s", len(out_rows), OUT_CSV)

    # ── Aggregate summary ─────────────────────────────────────────────────────
    _print_summary(out_rows)
    log.info("Total runtime: %.1f min", (time.time() - t_start) / 60)


def _print_summary(rows: list[dict]) -> None:
    from collections import defaultdict

    # Group by (class_type, variant)
    buckets: dict[tuple, list[bool]] = defaultdict(list)
    for r in rows:
        buckets[(r["class_type"], r["variant"])].append(bool(r["retrieved"]))

    print("\n" + "=" * 60)
    print("M1 RETRIEVAL EVALUATION — SUMMARY")
    print("=" * 60)
    print(f"{'Class type':<14} {'Variant':<10} {'Retrieved':>10} {'Total':>7} {'Rate':>8}")
    print("-" * 56)
    for (ctype, variant), vals in sorted(buckets.items()):
        n_ret = sum(vals)
        n_tot = len(vals)
        print(f"{ctype:<14} {variant:<10} {n_ret:>10} {n_tot:>7} {100*n_ret/n_tot:>7.1f}%")
    print("=" * 60)

    # Target thresholds
    print("\nTargets: paraphrase ≥90%, entity ≥70%")
    for ctype, target in [("paraphrase", 90.0), ("entity", 70.0)]:
        for variant in ["simple", "hotflip"]:
            vals = buckets.get((ctype, variant), [])
            if vals:
                rate = 100 * sum(vals) / len(vals)
                status = "✓ PASS" if rate >= target else "✗ FAIL"
                print(f"  {ctype}/{variant}: {rate:.1f}%  {status}")
    print()


if __name__ == "__main__":
    main()
