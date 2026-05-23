"""
Task 5: Single-query BBO (Shafran) on entity classes — HONEST top-5 evaluation.

Phases
------
1. BBO optimisation  (GTR-base retriever + vLLM on GPU 1)
   For each entity class:
     - q_star = representative query (closest BGE embedding to class centroid)
     - ShafranBBO.run(q_star) → optimised blocker
     - Checkpoint after each class (safe to resume from crash)

2. Honest retrieval + response generation  (vLLM still loaded)
   For each (class, query):
     - check_top5_retrieval() → retrieved_top5, rank, blocker_sim, threshold_sim
     - Build RAG prompt ONLY for retrieved queries; batch vLLM call

3. Judge  (vLLM closed, Gemma-2-9B loaded on same GPU)
   - Binary refusal judge applied to retrieved-only responses
   - Not-retrieved queries are automatically jammed_honest = False

Usage
-----
PILOT mode (6 representative classes, timing measurement):
    python eval_task5_entity_singlequery.py
    # PILOT_ONLY = True (default)

Full run (all 20 entity classes):
    python eval_task5_entity_singlequery.py --full
    # or set PILOT_ONLY = False below

Output
------
/home/ishana/scratch/results/task5_entity_singlequery_honest.csv
/home/ishana/scratch/results/task5_entity_singlequery_ckpt.pkl

Columns
-------
class_id, query_idx, query_text, within_class_sim, n_class_queries,
blocker_doc, blocker_sim, threshold_sim, retrieved_top5, rank,
response, jam_success, jammed_honest, q_star, d_r, final_loss, n_iterations, class_time_min
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# ── Env / path setup ──────────────────────────────────────────────────────────
os.environ["HF_HOME"]            = "/home/ishana/scratch/hf_cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("/home/ishana/scratch/data/classes")
INDEX_DIR   = Path("/home/ishana/scratch/data/indices/nq/sentence-transformers__gtr-t5-base")
RESULTS_DIR = Path("/home/ishana/scratch/results")
OUT_CSV     = RESULTS_DIR / "task5_entity_singlequery_honest.csv"
CKPT_FILE   = RESULTS_DIR / "task5_entity_singlequery_ckpt.pkl"

GTR_MODEL   = "sentence-transformers/gtr-t5-base"
LLM_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL = "google/gemma-2-9b-it"
TOP_K       = 5

# Pilot: indices chosen for spread in n_queries (3–8) and within-class sim (0.476–0.630)
# entity_18: n=3, sim=0.476 | entity_07: n=6, sim=0.478 | entity_12: n=4, sim=0.502
# entity_00: n=8, sim=0.529 | entity_08: n=6, sim=0.614 | entity_03: n=7, sim=0.630
PILOT_CLASS_INDICES = [0, 3, 7, 8, 12, 18]


# ── Config ────────────────────────────────────────────────────────────────────

def _build_cfg():
    cfg_dir = Path(__file__).resolve().parents[2] / "configs"
    base = OmegaConf.load(cfg_dir / "base.yaml")
    # Keep base hyperparameters: n=50 tokens, B=32, T=1000 iters, patience=100
    return base


# ── Data loading ──────────────────────────────────────────────────────────────

def load_entity_classes(indices: list[int] | None = None) -> list[dict]:
    with open(DATA_DIR / "entity_classes.json") as f:
        classes = json.load(f)
    embs = np.load(DATA_DIR / "entity_embeddings.npy")  # (20, 8, 1024)

    result = []
    sel = indices if indices is not None else list(range(len(classes)))
    for i in sel:
        cls = classes[i]
        queries = cls["queries"]
        n = len(queries)
        e = embs[i, :n]   # (n, 1024) — BGE embeddings

        # L2-normalize for cosine similarity
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        e_norm = e / np.where(norms < 1e-9, 1.0, norms)
        centroid = np.array(cls["centroid"], dtype=np.float32)
        c_norm = centroid / max(float(np.linalg.norm(centroid)), 1e-9)

        # Pairwise within-class similarity
        pw = e_norm @ e_norm.T
        pairs = [pw[r, c] for r in range(n) for c in range(r + 1, n)]
        within_sim = float(np.mean(pairs)) if pairs else 0.0

        cls["_queries"]     = queries
        cls["_embeddings"]  = e_norm
        cls["_centroid"]    = c_norm
        cls["_within_sim"]  = within_sim
        cls["_idx"]         = i
        result.append(cls)

    log.info("Loaded %d entity classes (indices %s)", len(result), sel)
    return result


# ── Representative query (M1 simple variant) ─────────────────────────────────

def find_representative_query(queries: list[str], embs: np.ndarray, centroid: np.ndarray) -> tuple[str, int]:
    """Return (q_star, q_star_idx): query whose BGE embedding is closest to centroid."""
    dists = np.linalg.norm(embs - centroid[np.newaxis, :], axis=1)
    best_idx = int(np.argmin(dists))
    return queries[best_idx], best_idx


# ── Honest retrieval check ────────────────────────────────────────────────────

def check_top5_retrieval(
    blocker_emb: np.ndarray,
    query: str,
    retriever,
    k: int = TOP_K,
) -> tuple[bool, int | None, float, float]:
    """
    Real top-k competition against the NQ FAISS index.
    Returns (retrieved, rank, blocker_sim, threshold_sim).
    """
    top_k_results = retriever.retrieve_with_scores(query, k=k)
    scores = [s for _, s in top_k_results]
    threshold = scores[-1] if scores else 0.0

    query_emb = retriever.embed_batch([query])[0]
    blocker_sim = float(np.dot(blocker_emb, query_emb))

    retrieved = blocker_sim >= threshold
    rank = sum(1 for s in scores if s > blocker_sim) + 1 if retrieved else None
    return retrieved, rank, blocker_sim, threshold


# ── Response generation for retrieved queries only ────────────────────────────

def build_retrieved_prompts(
    blocker_doc: str,
    queries: list[str],
    retrieval_results: list[dict],
    retriever,
    rag_prompt_template: str,
    k: int,
) -> tuple[list[str], list[int]]:
    """
    Build RAG prompts for queries where the blocker is retrieved.
    Returns (prompts, query_indices_that_were_retrieved).
    Blocker is placed at position 0; remaining k-1 slots filled from real corpus.
    """
    prompts: list[str] = []
    retrieved_idxs: list[int] = []

    for q_idx, (query, ret) in enumerate(zip(queries, retrieval_results)):
        if not ret["retrieved"]:
            continue
        top_k_ids = retriever.retrieve(query, k=k)
        base_docs = [retriever.get_doc_text(d) for d in top_k_ids]
        ctx_docs  = [blocker_doc] + base_docs[: k - 1]
        context   = "\n\n".join(ctx_docs)
        prompt    = rag_prompt_template.format(context=context, query=query)
        prompts.append(prompt)
        retrieved_idxs.append(q_idx)

    return prompts, retrieved_idxs


# ── Main ──────────────────────────────────────────────────────────────────────

def main(pilot_only: bool = True) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    indices = PILOT_CLASS_INDICES if pilot_only else None
    mode    = "PILOT (6 classes)" if pilot_only else "FULL (20 classes)"
    log.info("=" * 60)
    log.info("Task 5 entity single-query BBO — %s", mode)
    log.info("=" * 60)

    cfg = _build_cfg()

    classes = load_entity_classes(indices)

    # ── Load GTR retriever ────────────────────────────────────────────────────
    log.info("Loading GTR-base retriever...")
    from rag.retriever import Retriever
    retriever = Retriever(model_name=GTR_MODEL, score_function="cos_sim")
    retriever.load_index(INDEX_DIR)
    log.info("Retriever: %d docs in index", retriever._index.ntotal)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt: dict[str, dict] = {}
    if CKPT_FILE.exists():
        with open(CKPT_FILE, "rb") as f:
            ckpt = pickle.load(f)
        log.info("Resumed checkpoint: %d classes already done", len(ckpt))

    # ── Phase 1+2: BBO + retrieval check + response generation ───────────────
    log.info("Loading vLLM: %s", LLM_MODEL)
    from rag.generator import VLLMGenerator
    generator = VLLMGenerator(
        model_name=LLM_MODEL,
        temperature=float(cfg.attack.temperature),
        max_tokens=int(cfg.attack.max_response_len),
        gpu_memory_utilization=float(cfg.vllm.gpu_memory_utilization),
        dtype=str(cfg.vllm.dtype),
        max_model_len=int(cfg.vllm.max_model_len),
    )
    from utils.gpu_manager import GPUManager
    gpu_manager = GPUManager()
    from attacks.shafran_bbo import ShafranBBO
    attacker = ShafranBBO(cfg, retriever, generator, gpu_manager)

    rag_prompt    = cfg.rag_prompt
    retrieval_k   = int(cfg.retrieval.k)
    class_timings: dict[str, float] = {}

    for cls in classes:
        cid     = cls["class_id"]
        queries = cls["_queries"]
        embs    = cls["_embeddings"]
        centroid = cls["_centroid"]
        within_sim = cls["_within_sim"]

        if cid in ckpt:
            log.info("[SKIP] %s already in checkpoint", cid)
            continue

        log.info("=" * 55)
        log.info("Class %s | n=%d | within_sim=%.3f", cid, len(queries), within_sim)
        t_cls = time.time()

        # ── Representative query ──────────────────────────────────────────────
        q_star, q_star_idx = find_representative_query(queries, embs, centroid)
        log.info("  q_star[%d]: %r", q_star_idx, q_star[:70])

        # ── BBO optimisation on q_star ────────────────────────────────────────
        log.info("  Running ShafranBBO.run(q_star)...")
        t_bbo = time.time()
        result = attacker.run(query=q_star)
        bbo_min = (time.time() - t_bbo) / 60
        log.info(
            "  BBO done in %.1f min | loss=%.4f | iters=%d",
            bbo_min, result.final_loss, result.n_iterations,
        )

        # ── Honest top-5 retrieval check for every class query ────────────────
        blocker_emb = retriever.embed_batch([result.final_doc])[0]
        retrieval_results: list[dict] = []
        for query in queries:
            retr, rank, b_sim, t_sim = check_top5_retrieval(blocker_emb, query, retriever)
            retrieval_results.append({
                "retrieved": retr,
                "rank":       rank,
                "blocker_sim": round(b_sim, 5),
                "threshold_sim": round(t_sim, 5),
            })

        n_retrieved = sum(r["retrieved"] for r in retrieval_results)
        log.info(
            "  Retrieval: %d/%d queries → blocker in top-5",
            n_retrieved, len(queries),
        )

        # ── Build prompts and generate responses (retrieved queries only) ─────
        prompts, ret_idxs = build_retrieved_prompts(
            result.final_doc, queries, retrieval_results,
            retriever, rag_prompt, retrieval_k,
        )

        responses: list[str | None] = [None] * len(queries)
        if prompts:
            gen_responses = generator.generate(prompts)
            for q_idx, resp in zip(ret_idxs, gen_responses):
                responses[q_idx] = resp
        log.info("  Generated %d responses (retrieved only).", len(prompts))

        cls_min = (time.time() - t_cls) / 60
        class_timings[cid] = cls_min

        ckpt[cid] = {
            "class_id":          cid,
            "within_sim":        within_sim,
            "n_queries":         len(queries),
            "queries":           queries,
            "q_star":            q_star,
            "q_star_idx":        q_star_idx,
            "result":            result,
            "retrieval_results": retrieval_results,
            "responses":         responses,
            "class_time_min":    cls_min,
        }
        with open(CKPT_FILE, "wb") as f:
            pickle.dump(ckpt, f)
        log.info("  Checkpointed %s (%.1f min total)", cid, cls_min)

    # ── Close LLM, free VRAM ──────────────────────────────────────────────────
    log.info("Closing vLLM...")
    generator.close()
    del generator, attacker
    gc.collect()
    torch.cuda.empty_cache()
    log.info("vLLM closed.")

    # ── Phase 3: judge (retrieved responses only) ─────────────────────────────
    log.info("Loading judge: %s", JUDGE_MODEL)
    from judges.local_judge import LocalJudge
    judge = LocalJudge(model_name=JUDGE_MODEL)

    rows: list[dict] = []
    for cls in classes:
        cid   = cls["class_id"]
        entry = ckpt[cid]
        queries   = entry["queries"]
        responses = entry["responses"]
        ret_results = entry["retrieval_results"]
        result      = entry["result"]
        within_sim  = entry["within_sim"]

        for q_idx, (query, resp, ret) in enumerate(zip(queries, responses, ret_results)):
            retrieved  = bool(ret["retrieved"])
            jam_success: int | None = None
            jammed_honest = 0

            if retrieved and resp is not None:
                answered = judge.is_answered(query, resp)
                jam_success = int(not answered)
                jammed_honest = jam_success  # honest: retrieved AND not answered
            elif not retrieved:
                jam_success   = None   # not applicable (blocker not seen)
                jammed_honest = 0

            rows.append({
                "class_id":         cid,
                "query_idx":        q_idx,
                "query_text":       query,
                "within_class_sim": round(within_sim, 4),
                "n_class_queries":  entry["n_queries"],
                "blocker_doc":      result.final_doc[:300],
                "blocker_sim":      ret["blocker_sim"],
                "threshold_sim":    ret["threshold_sim"],
                "retrieved_top5":   int(retrieved),
                "rank":             ret["rank"] if ret["rank"] is not None else "",
                "response":         resp[:300] if resp else "",
                "jam_success":      jam_success if jam_success is not None else "",
                "jammed_honest":    jammed_honest,
                "q_star":           entry["q_star"][:120],
                "final_loss":       round(result.final_loss, 5),
                "n_iterations":     result.n_iterations,
                "class_time_min":   round(entry["class_time_min"], 2),
            })

    judge.close()

    # ── Save CSV ──────────────────────────────────────────────────────────────
    fieldnames = [
        "class_id", "query_idx", "query_text", "within_class_sim", "n_class_queries",
        "blocker_doc", "blocker_sim", "threshold_sim", "retrieved_top5", "rank",
        "response", "jam_success", "jammed_honest", "q_star",
        "final_loss", "n_iterations", "class_time_min",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Saved %d rows → %s", len(rows), OUT_CSV)

    _print_summary(rows, class_timings, mode)
    log.info("Total runtime: %.1f min", (time.time() - t_start) / 60)


def _print_summary(rows: list[dict], timings: dict[str, float], mode: str) -> None:
    from collections import defaultdict

    print()
    print("=" * 65)
    print(f"TASK 5 — ENTITY SINGLE-QUERY BBO HONEST ASR — {mode}")
    print("=" * 65)

    per_class: dict[str, dict] = defaultdict(lambda: {
        "n": 0, "retrieved": 0, "jammed_honest": 0,
        "within_sim": 0.0, "loss": 0.0, "iters": 0,
    })
    for r in rows:
        cid = r["class_id"]
        pc = per_class[cid]
        pc["n"] += 1
        pc["retrieved"] += int(r["retrieved_top5"])
        pc["jammed_honest"] += int(r["jammed_honest"])
        pc["within_sim"] = float(r["within_class_sim"])
        pc["loss"]  = float(r["final_loss"])
        pc["iters"] = int(r["n_iterations"])

    print(f"\n{'Class':<12} {'n':>3} {'sim':>6} {'ret%':>6} {'asr_h%':>7} {'loss':>7} {'iters':>6} {'min':>6}")
    print("-" * 60)
    all_ret = all_jammed = all_n = 0
    for cid, pc in sorted(per_class.items()):
        n = pc["n"]
        ret = pc["retrieved"]
        jam = pc["jammed_honest"]
        t   = timings.get(cid, 0.0)
        print(
            f"{cid:<12} {n:>3} {pc['within_sim']:>6.3f} "
            f"{100*ret/n:>5.0f}% {100*jam/n:>6.0f}% "
            f"{pc['loss']:>7.4f} {pc['iters']:>6} {t:>6.1f}"
        )
        all_n += n; all_ret += ret; all_jammed += jam

    print("-" * 60)
    print(
        f"{'TOTAL':<12} {all_n:>3}        "
        f"{100*all_ret/all_n:>5.0f}% {100*all_jammed/all_n:>6.0f}%"
    )

    print()
    print(f"  Retrieval rate (top-5): {all_ret}/{all_n} = {100*all_ret/all_n:.1f}%")
    print(f"  Honest ASR:             {all_jammed}/{all_n} = {100*all_jammed/all_n:.1f}%")

    # P(jam|retrieved)
    jam_given_ret = sum(int(r["jammed_honest"]) for r in rows if int(r["retrieved_top5"]))
    n_ret = sum(int(r["retrieved_top5"]) for r in rows)
    if n_ret:
        print(f"  P(jam | retrieved):     {jam_given_ret}/{n_ret} = {100*jam_given_ret/n_ret:.1f}%")

    # Timing
    if timings:
        total_t  = sum(timings.values())
        per_cls  = total_t / len(timings)
        remaining = 20 - len(timings)
        print(f"\n  Pilot timing: {total_t:.1f} min for {len(timings)} classes ({per_cls:.1f} min/class)")
        if remaining > 0:
            print(f"  Estimated remaining {remaining} classes: ~{remaining * per_cls:.0f} min "
                  f"({remaining * per_cls / 60:.1f} h)")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run all 20 classes (default: pilot 6)")
    args = parser.parse_args()
    main(pilot_only=not args.full)
