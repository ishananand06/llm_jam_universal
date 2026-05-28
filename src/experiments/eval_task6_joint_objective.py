"""
Task 6: Constrained-Joint BBO — honest evaluation across paraphrase + entity classes.

Compares ConstrainedJointBBO (rejects candidates that fail real top-5) against
the existing ShafranBBO baselines (which use a 0.3 proxy threshold).

For each cluster type (paraphrase, entity):
  1. Pick q* (representative query: BGE embedding closest to class centroid)
  2. Precompute real top-5 threshold for q*
  3. Run ConstrainedJointBBO.run(q*) -> blocker doc
  4. Honest top-5 retrieval check for every class member (real FAISS scores)
  5. Generate LLM responses only for queries where blocker was retrieved
  6. Judge with locked Gemma-2-9b-it

Resumability
------------
Per-class checkpoint after each completion. Safe to kill and restart.

CLI
---
  --cluster {paraphrase,entity,all}  which cluster set to run (default all)
  --num-paraphrase N                 paraphrase classes to run (default 10)
  --num-entity N                     entity classes to run (default 10)
  --smoke                            single-class smoke test (1 paraphrase + 1 entity)

Output
------
/home/ishana/scratch/results/task6_constrained_joint_honest.csv
/home/ishana/scratch/results/task6_constrained_joint_ckpt.pkl
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

# ── Env / path setup (set BEFORE importing torch-heavy modules) ──────────────
os.environ.setdefault("HF_HOME", "/home/ishana/scratch/hf_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("/home/ishana/scratch/data/classes")
INDEX_DIR   = Path("/home/ishana/scratch/data/indices/nq/sentence-transformers__gtr-t5-base")
RESULTS_DIR = Path("/home/ishana/scratch/results")
OUT_CSV     = RESULTS_DIR / "task6_constrained_joint_honest.csv"
CKPT_FILE   = RESULTS_DIR / "task6_constrained_joint_ckpt.pkl"

GTR_MODEL   = "sentence-transformers/gtr-t5-base"
LLM_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL = "google/gemma-2-9b-it"
TOP_K       = 5


# ── Config ───────────────────────────────────────────────────────────────────

def _build_cfg():
    cfg_dir = Path(__file__).resolve().parents[2] / "configs"
    base = OmegaConf.load(cfg_dir / "base.yaml")
    # Task 6 budget: shorter than full Shafran (T=500, patience=80) to fit
    # within ~7h across 20 classes.
    base.attack.num_iterations = 500
    base.attack.es_patience = 80
    return base


# ── Cluster loading ──────────────────────────────────────────────────────────

def _load_paraphrase_classes(n_classes: int) -> list[dict]:
    """Wrap the first n paraphrase classes with the fields the runner expects."""
    with open(DATA_DIR / "paraphrase_classes.json") as f:
        classes = json.load(f)
    embs = np.load(DATA_DIR / "paraphrase_embeddings.npy")  # (100, 6, 1024)

    result = []
    for i in range(min(n_classes, len(classes))):
        cls = classes[i]
        queries = [cls["original_query"]] + cls["paraphrases"]
        n = len(queries)
        e = embs[i, :n]
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        e_norm = e / np.where(norms < 1e-9, 1.0, norms)
        centroid = np.array(cls["centroid"], dtype=np.float32)
        c_norm = centroid / max(float(np.linalg.norm(centroid)), 1e-9)
        # within-class sim
        pw = e_norm @ e_norm.T
        pairs = [pw[r, c] for r in range(n) for c in range(r + 1, n)]
        within = float(np.mean(pairs)) if pairs else 0.0

        result.append({
            "class_id":     f"para_{cls['class_id']}",   # disambiguate from entity
            "class_type":   "paraphrase",
            "_queries":     queries,
            "_embeddings":  e_norm,
            "_centroid":    c_norm,
            "_within_sim":  within,
        })
    log.info("Loaded %d paraphrase classes", len(result))
    return result


def _load_entity_classes(n_classes: int) -> list[dict]:
    with open(DATA_DIR / "entity_classes.json") as f:
        classes = json.load(f)
    embs = np.load(DATA_DIR / "entity_embeddings.npy")   # (20, 8, 1024) zero-padded

    result = []
    for i in range(min(n_classes, len(classes))):
        cls = classes[i]
        queries = cls["queries"]
        n = len(queries)
        e = embs[i, :n]
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        e_norm = e / np.where(norms < 1e-9, 1.0, norms)
        centroid = np.array(cls["centroid"], dtype=np.float32)
        c_norm = centroid / max(float(np.linalg.norm(centroid)), 1e-9)
        pw = e_norm @ e_norm.T
        pairs = [pw[r, c] for r in range(n) for c in range(r + 1, n)]
        within = float(np.mean(pairs)) if pairs else 0.0

        result.append({
            "class_id":     cls["class_id"],   # already prefixed entity_
            "class_type":   "entity",
            "_queries":     queries,
            "_embeddings":  e_norm,
            "_centroid":    c_norm,
            "_within_sim":  within,
        })
    log.info("Loaded %d entity classes", len(result))
    return result


def find_representative_query(queries, embs, centroid):
    dists = np.linalg.norm(embs - centroid[np.newaxis, :], axis=1)
    best_idx = int(np.argmin(dists))
    return queries[best_idx], best_idx


# ── Honest retrieval check (real top-5) ──────────────────────────────────────

def check_top5_retrieval(blocker_emb, query, retriever, k=TOP_K):
    top_k_results = retriever.retrieve_with_scores(query, k=k)
    scores = [s for _, s in top_k_results]
    threshold = scores[-1] if scores else 0.0
    query_emb = retriever.embed_batch([query])[0]
    blocker_sim = float(np.dot(blocker_emb, query_emb))
    retrieved = blocker_sim >= threshold
    rank = sum(1 for s in scores if s > blocker_sim) + 1 if retrieved else None
    return retrieved, rank, blocker_sim, threshold


def build_retrieved_prompts(blocker_doc, queries, retrieval_results,
                            retriever, rag_prompt_template, k):
    prompts, retrieved_idxs = [], []
    for q_idx, (q, ret) in enumerate(zip(queries, retrieval_results)):
        if not ret["retrieved"]:
            continue
        top_k_ids = retriever.retrieve(q, k=k)
        base_docs = [retriever.get_doc_text(d) for d in top_k_ids]
        ctx_docs  = [blocker_doc] + base_docs[: k - 1]
        context   = "\n\n".join(ctx_docs)
        prompt    = rag_prompt_template.format(context=context, query=q)
        prompts.append(prompt)
        retrieved_idxs.append(q_idx)
    return prompts, retrieved_idxs


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", choices=["paraphrase", "entity", "all"],
                        default="all")
    parser.add_argument("--num-paraphrase", type=int, default=10)
    parser.add_argument("--num-entity", type=int, default=10)
    parser.add_argument("--smoke", action="store_true",
                        help="Single-class smoke test (1 para + 1 entity)")
    args = parser.parse_args()

    if args.smoke:
        args.num_paraphrase = 1
        args.num_entity = 1
        args.cluster = "all"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    log.info("=" * 65)
    log.info("Task 6: Constrained-Joint BBO  (cluster=%s, P=%d, E=%d, smoke=%s)",
             args.cluster, args.num_paraphrase, args.num_entity, args.smoke)
    log.info("=" * 65)

    # Build cluster list
    classes: list[dict] = []
    if args.cluster in ("paraphrase", "all"):
        classes.extend(_load_paraphrase_classes(args.num_paraphrase))
    if args.cluster in ("entity", "all"):
        classes.extend(_load_entity_classes(args.num_entity))
    log.info("Total classes to run: %d", len(classes))

    cfg = _build_cfg()

    # ── Retriever ─────────────────────────────────────────────────────────────
    log.info("Loading GTR-base retriever...")
    from rag.retriever import Retriever
    retriever = Retriever(model_name=GTR_MODEL, score_function="cos_sim")
    retriever.load_index(INDEX_DIR)
    log.info("Retriever: %d docs in index", retriever._index.ntotal)

    # ── Checkpoint resume ─────────────────────────────────────────────────────
    ckpt: dict[str, dict] = {}
    if CKPT_FILE.exists():
        with open(CKPT_FILE, "rb") as f:
            ckpt = pickle.load(f)
        log.info("Resumed checkpoint with %d classes already done", len(ckpt))

    # ── vLLM + attacker ───────────────────────────────────────────────────────
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
    from attacks.constrained_joint_bbo import ConstrainedJointBBO
    attacker = ConstrainedJointBBO(cfg, retriever, generator, gpu_manager)

    rag_prompt  = cfg.rag_prompt
    retrieval_k = int(cfg.retrieval.k)
    class_timings: dict[str, float] = {}

    # ── BBO + honest eval per class ───────────────────────────────────────────
    for cls in classes:
        cid = cls["class_id"]
        if cid in ckpt:
            log.info("[SKIP] %s already in checkpoint", cid)
            continue

        queries  = cls["_queries"]
        embs     = cls["_embeddings"]
        centroid = cls["_centroid"]
        within   = cls["_within_sim"]
        ctype    = cls["class_type"]

        log.info("=" * 55)
        log.info("Class %s (%s) | n=%d | within=%.3f", cid, ctype, len(queries), within)
        t_cls = time.time()

        q_star, q_star_idx = find_representative_query(queries, embs, centroid)
        log.info("  q*[%d]: %r", q_star_idx, q_star[:80])

        # ── Precompute real top-5 threshold for q* (Task 6 core constraint) ──
        real_thr = attacker.precompute_retrieval_threshold(q_star)

        # ── BBO with hard real-threshold constraint ──────────────────────────
        log.info("  Running ConstrainedJointBBO.run(q*) ...")
        t_bbo = time.time()
        result = attacker.run(query=q_star)
        bbo_min = (time.time() - t_bbo) / 60
        log.info(
            "  BBO done in %.1f min | loss=%.4f | iters=%d | "
            "cands_scored=%d | rejected=%d | accepted=%d",
            bbo_min, result.final_loss, result.n_iterations,
            attacker.n_candidates_scored,
            attacker.n_rejected_by_constraint,
            attacker.n_accepted_by_constraint,
        )

        constraint_stats = {
            "real_threshold":     real_thr,
            "n_cands_scored":     int(attacker.n_candidates_scored),
            "n_cands_rejected":   int(attacker.n_rejected_by_constraint),
            "n_cands_accepted":   int(attacker.n_accepted_by_constraint),
        }

        # ── Honest top-5 retrieval for every class member ────────────────────
        blocker_emb = retriever.embed_batch([result.final_doc])[0]
        retrieval_results = []
        for q in queries:
            retr, rank, b_sim, t_sim = check_top5_retrieval(blocker_emb, q, retriever)
            retrieval_results.append({
                "retrieved":     retr,
                "rank":          rank,
                "blocker_sim":   round(b_sim, 5),
                "threshold_sim": round(t_sim, 5),
            })
        n_ret = sum(r["retrieved"] for r in retrieval_results)
        log.info("  Retrieval: %d/%d queries -> blocker in top-5", n_ret, len(queries))

        # ── Generate responses (retrieved only) ──────────────────────────────
        prompts, ret_idxs = build_retrieved_prompts(
            result.final_doc, queries, retrieval_results,
            retriever, rag_prompt, retrieval_k,
        )
        responses: list[str | None] = [None] * len(queries)
        if prompts:
            gen_resps = generator.generate(prompts)
            for q_idx, resp in zip(ret_idxs, gen_resps):
                responses[q_idx] = resp
        log.info("  Generated %d responses (retrieved only).", len(prompts))

        cls_min = (time.time() - t_cls) / 60
        class_timings[cid] = cls_min

        ckpt[cid] = {
            "class_id":          cid,
            "class_type":        ctype,
            "within_sim":        within,
            "n_queries":         len(queries),
            "queries":           queries,
            "q_star":            q_star,
            "q_star_idx":        q_star_idx,
            "result":            result,
            "retrieval_results": retrieval_results,
            "responses":         responses,
            "class_time_min":    cls_min,
            "constraint_stats":  constraint_stats,
        }
        with open(CKPT_FILE, "wb") as f:
            pickle.dump(ckpt, f)
        log.info("  Checkpointed %s  (%.1f min)", cid, cls_min)

    # ── Close vLLM ───────────────────────────────────────────────────────────
    log.info("Closing vLLM...")
    generator.close()
    del generator, attacker
    gc.collect()
    torch.cuda.empty_cache()
    log.info("vLLM closed.")

    # ── Phase 3: judge (retrieved responses only) ────────────────────────────
    log.info("Loading judge: %s", JUDGE_MODEL)
    from judges.local_judge import LocalJudge
    judge = LocalJudge(model_name=JUDGE_MODEL)

    rows: list[dict] = []
    for cls in classes:
        cid = cls["class_id"]
        entry = ckpt[cid]
        queries     = entry["queries"]
        responses   = entry["responses"]
        ret_results = entry["retrieval_results"]
        result      = entry["result"]
        within      = entry["within_sim"]
        cstats      = entry["constraint_stats"]
        ctype       = entry["class_type"]

        for q_idx, (q, resp, ret) in enumerate(zip(queries, responses, ret_results)):
            retrieved = bool(ret["retrieved"])
            if retrieved and resp is not None:
                answered = judge.is_answered(q, resp)
                jam_success = int(not answered)
                jammed_honest = jam_success
            else:
                jam_success = None
                jammed_honest = 0

            rows.append({
                "class_id":         cid,
                "class_type":       ctype,
                "query_idx":        q_idx,
                "query_text":       q,
                "within_class_sim": round(within, 4),
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
                "real_threshold":   round(cstats["real_threshold"], 5),
                "cands_scored":     cstats["n_cands_scored"],
                "cands_rejected":   cstats["n_cands_rejected"],
                "cands_accepted":   cstats["n_cands_accepted"],
            })

    judge.close()

    fieldnames = [
        "class_id", "class_type", "query_idx", "query_text",
        "within_class_sim", "n_class_queries",
        "blocker_doc", "blocker_sim", "threshold_sim",
        "retrieved_top5", "rank",
        "response", "jam_success", "jammed_honest",
        "q_star", "final_loss", "n_iterations", "class_time_min",
        "real_threshold", "cands_scored", "cands_rejected", "cands_accepted",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Saved %d rows -> %s", len(rows), OUT_CSV)

    _print_summary(rows, class_timings)
    log.info("Total runtime: %.1f min", (time.time() - t_start) / 60)


def _print_summary(rows, timings):
    from collections import defaultdict

    print()
    print("=" * 75)
    print("TASK 6 — CONSTRAINED-JOINT BBO  HONEST ASR")
    print("=" * 75)

    # Group by class
    per_class: dict[str, dict] = defaultdict(lambda: {
        "n": 0, "retrieved": 0, "jammed_honest": 0,
        "within_sim": 0.0, "loss": 0.0, "iters": 0,
        "ctype": "?", "rej_frac": 0.0,
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
        pc["ctype"] = r["class_type"]
        cs = int(r["cands_scored"]) or 1
        pc["rej_frac"] = float(r["cands_rejected"]) / cs

    print(f"\n{'Class':<20} {'type':<10} {'n':>3} {'sim':>6} {'ret%':>6} "
          f"{'asr_h%':>7} {'rej%':>6} {'loss':>7} {'iters':>6}")
    print("-" * 80)

    # Totals split by type
    totals = defaultdict(lambda: {"n": 0, "ret": 0, "jam": 0})

    for cid, pc in sorted(per_class.items()):
        n = pc["n"]
        ret = pc["retrieved"]
        jam = pc["jammed_honest"]
        print(
            f"{cid:<20} {pc['ctype']:<10} {n:>3} {pc['within_sim']:>6.3f} "
            f"{100*ret/n:>5.0f}% {100*jam/n:>6.0f}% {100*pc['rej_frac']:>5.0f}% "
            f"{pc['loss']:>7.4f} {pc['iters']:>6}"
        )
        t = totals[pc["ctype"]]
        t["n"] += n; t["ret"] += ret; t["jam"] += jam

    print("-" * 80)
    grand_n = grand_ret = grand_jam = 0
    for ctype, t in totals.items():
        n = t["n"]
        print(f"  {ctype:<10}  ret={t['ret']}/{n} = {100*t['ret']/n:.1f}%  "
              f"asr_h={t['jam']}/{n} = {100*t['jam']/n:.1f}%")
        grand_n += n; grand_ret += t["ret"]; grand_jam += t["jam"]

    if grand_n:
        print(f"  {'TOTAL':<10}  ret={grand_ret}/{grand_n} = {100*grand_ret/grand_n:.1f}%  "
              f"asr_h={grand_jam}/{grand_n} = {100*grand_jam/grand_n:.1f}%")

    # P(jam | retrieved)
    jam_given_ret = sum(int(r["jammed_honest"]) for r in rows if int(r["retrieved_top5"]))
    n_ret = sum(int(r["retrieved_top5"]) for r in rows)
    if n_ret:
        print(f"  P(jam | retrieved) = {jam_given_ret}/{n_ret} = "
              f"{100*jam_given_ret/n_ret:.1f}%")

    if timings:
        total_t = sum(timings.values())
        per_cls = total_t / len(timings)
        print(f"\n  Timing: {total_t:.1f} min for {len(timings)} classes "
              f"({per_cls:.1f} min/class avg)")

    print("=" * 75)


if __name__ == "__main__":
    main()
