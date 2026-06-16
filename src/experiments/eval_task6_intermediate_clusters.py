"""
Run ConstrainedJointBBO on 7 hand-selected intermediate-similarity clusters
from the NQ answerable subset (output of cluster_intermediate.py).

These clusters span within-class cosine similarity 0.62–0.83, filling the
gap between entity classes (~0.56) and paraphrase classes (~0.90).

Same pipeline as eval_task6_joint_objective.py:
  - ConstrainedJointBBO with real top-5 threshold (not 0.3 proxy)
  - Honest top-5 retrieval check for every class member
  - Gemma-2-9b-it judge (locked)
  - Checkpoint resume

Output
------
/home/ishana/scratch/results/task6_intermediate_honest.csv
/home/ishana/scratch/results/task6_intermediate_ckpt.pkl
"""
from __future__ import annotations

import csv
import gc
import json
import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

os.environ["HF_HOME"] = "/home/ishana/scratch/hf_cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

_SRC = Path(__file__).resolve().parents[2] / "src"
import sys
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH     = Path("/home/ishana/scratch")
DATA_DIR    = SCRATCH / "data"
INDEX_DIR   = DATA_DIR / "indices/nq/sentence-transformers__gtr-t5-base"
RESULTS_DIR = SCRATCH / "results"
OUT_CSV     = RESULTS_DIR / "task6_intermediate_honest.csv"
CKPT_FILE   = RESULTS_DIR / "task6_intermediate_ckpt.pkl"

EMB_CACHE   = DATA_DIR / "classes/answerable_bge_embeddings.npy"
ID_CACHE    = DATA_DIR / "classes/answerable_bge_ids.json"
QUERIES_FILE = DATA_DIR / "nq/queries.jsonl"

GTR_MODEL   = "sentence-transformers/gtr-t5-base"
LLM_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL = "google/gemma-2-9b-it"
TOP_K       = 5

# ── Cluster definitions (hand-selected from intermediate_class_candidates.csv) ─
# class_id is unique; two "689" clusters are disambiguated by band suffix.
CLUSTERS = [
    {
        "class_id":   "inter_551_072",
        "cluster_id": 551,
        "target_band": "0.72",
        "within_sim": 0.83443,
        "query_ids":  ["test2391", "test2442", "test467"],
    },
    {
        "class_id":   "inter_504_072",
        "cluster_id": 504,
        "target_band": "0.72",
        "within_sim": 0.81056,
        "query_ids":  ["test1585", "test1624", "test2496"],
    },
    {
        "class_id":   "inter_1031_072",
        "cluster_id": 1031,
        "target_band": "0.72",
        "within_sim": 0.75892,
        "query_ids":  ["test1906", "test2368", "test2819", "test3380", "test721"],
    },
    {
        "class_id":   "inter_1298_072",
        "cluster_id": 1298,
        "target_band": "0.72",
        "within_sim": 0.74734,
        "query_ids":  ["test1796", "test2275", "test2563", "test490"],
    },
    {
        "class_id":   "inter_1060_072",
        "cluster_id": 1060,
        "target_band": "0.72",
        "within_sim": 0.73149,
        "query_ids":  ["test1928", "test1980", "test2467", "test596"],
    },
    {
        "class_id":   "inter_689_072",
        "cluster_id": 689,
        "target_band": "0.72",
        "within_sim": 0.71373,
        "query_ids":  ["test209", "test2936", "test3261", "test574"],
    },
    {
        "class_id":   "inter_689_065",
        "cluster_id": 689,
        "target_band": "0.65",
        "within_sim": 0.61860,
        "query_ids":  ["test1146", "test1375", "test843"],
    },
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_classes() -> list[dict]:
    """
    Resolve query texts and BGE embeddings for each cluster.
    Embeddings come from the cache built by cluster_intermediate.py.
    """
    # query_id → text
    with open(QUERIES_FILE) as f:
        id2text = {json.loads(l)["_id"]: json.loads(l)["text"] for l in f}

    # query_id → row index in EMB_CACHE
    cached_ids: list[str] = json.loads(ID_CACHE.read_text())
    id2idx = {qid: i for i, qid in enumerate(cached_ids)}

    all_embs = np.load(EMB_CACHE)  # (2279, 1024) L2-normalised

    result = []
    for cl in CLUSTERS:
        qids    = cl["query_ids"]
        queries = [id2text[qid] for qid in qids]
        idxs    = [id2idx[qid] for qid in qids]
        embs    = all_embs[idxs]  # (n, 1024)

        centroid_raw = embs.mean(axis=0)
        norm = float(np.linalg.norm(centroid_raw))
        centroid = centroid_raw / max(norm, 1e-9)

        result.append({
            **cl,
            "_queries":    queries,
            "_embeddings": embs,       # already L2-normalised
            "_centroid":   centroid,
        })
        log.info(
            "Loaded class %s: n=%d  within_sim=%.3f  queries=%s",
            cl["class_id"], len(queries), cl["within_sim"],
            [q[:40] for q in queries],
        )
    return result


def find_representative_query(queries, embs, centroid):
    dists = np.linalg.norm(embs - centroid[np.newaxis, :], axis=1)
    best_idx = int(np.argmin(dists))
    return queries[best_idx], best_idx


# ── Honest retrieval helpers (identical to task5 / task6) ────────────────────

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
        ctx_docs  = [blocker_doc] + base_docs[:k - 1]
        context   = "\n\n".join(ctx_docs)
        prompts.append(rag_prompt_template.format(context=context, query=q))
        retrieved_idxs.append(q_idx)
    return prompts, retrieved_idxs


# ── Config ────────────────────────────────────────────────────────────────────

def _build_cfg():
    cfg_dir = Path(__file__).resolve().parents[2] / "configs"
    base = OmegaConf.load(cfg_dir / "base.yaml")
    base.attack.num_iterations = 500
    base.attack.es_patience = 80
    return base


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    log.info("=" * 65)
    log.info("Task 6 — ConstrainedJointBBO on 7 intermediate clusters")
    log.info("=" * 65)

    classes = load_classes()
    cfg = _build_cfg()

    log.info("Loading GTR-base retriever...")
    from rag.retriever import Retriever
    retriever = Retriever(model_name=GTR_MODEL, score_function="cos_sim")
    retriever.load_index(INDEX_DIR)
    log.info("Retriever: %d docs in index", retriever._index.ntotal)

    ckpt: dict[str, dict] = {}
    if CKPT_FILE.exists():
        with open(CKPT_FILE, "rb") as f:
            ckpt = pickle.load(f)
        log.info("Resumed checkpoint: %d classes already done", len(ckpt))

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
        within   = cls["within_sim"]

        log.info("=" * 55)
        log.info("Class %s | n=%d | within_sim=%.3f", cid, len(queries), within)
        t_cls = time.time()

        q_star, q_star_idx = find_representative_query(queries, embs, centroid)
        log.info("  q*[%d]: %r", q_star_idx, q_star[:80])

        real_thr = attacker.precompute_retrieval_threshold(q_star)

        log.info("  Running ConstrainedJointBBO.run(q*) ...")
        t_bbo = time.time()
        result = attacker.run(query=q_star)
        bbo_min = (time.time() - t_bbo) / 60
        log.info(
            "  BBO done in %.1f min | loss=%.4f | iters=%d | "
            "rejected=%d accepted=%d",
            bbo_min, result.final_loss, result.n_iterations,
            attacker.n_rejected_by_constraint,
            attacker.n_accepted_by_constraint,
        )

        constraint_stats = {
            "real_threshold":   real_thr,
            "n_cands_scored":   int(attacker.n_candidates_scored),
            "n_cands_rejected": int(attacker.n_rejected_by_constraint),
            "n_cands_accepted": int(attacker.n_accepted_by_constraint),
        }

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
        log.info("  Retrieval: %d/%d queries → blocker in top-5", n_ret, len(queries))

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
            "cluster_id":        cls["cluster_id"],
            "target_band":       cls["target_band"],
            "within_sim":        within,
            "n_queries":         len(queries),
            "queries":           queries,
            "query_ids":         cls["query_ids"],
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
        log.info("  Checkpointed %s (%.1f min)", cid, cls_min)

    log.info("Closing vLLM...")
    generator.close()
    del generator, attacker
    gc.collect()
    torch.cuda.empty_cache()
    log.info("vLLM closed.")

    log.info("Loading judge: %s", JUDGE_MODEL)
    from judges.local_judge import LocalJudge
    judge = LocalJudge(model_name=JUDGE_MODEL)

    rows: list[dict] = []
    for cls in classes:
        cid   = cls["class_id"]
        entry = ckpt[cid]
        queries     = entry["queries"]
        responses   = entry["responses"]
        ret_results = entry["retrieval_results"]
        result      = entry["result"]
        within      = entry["within_sim"]
        cstats      = entry["constraint_stats"]

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
                "cluster_id":       entry["cluster_id"],
                "target_band":      entry["target_band"],
                "query_idx":        q_idx,
                "query_id":         entry["query_ids"][q_idx],
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
        "class_id", "cluster_id", "target_band",
        "query_idx", "query_id", "query_text",
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
    log.info("Saved %d rows → %s", len(rows), OUT_CSV)

    _print_summary(rows, class_timings)
    log.info("Total runtime: %.1f min", (time.time() - t_start) / 60)


def _print_summary(rows, timings):
    from collections import defaultdict

    print()
    print("=" * 70)
    print("TASK 6 — CONSTRAINED BBO — INTERMEDIATE CLUSTERS HONEST ASR")
    print("=" * 70)

    per_class = defaultdict(lambda: {
        "n": 0, "retrieved": 0, "jammed": 0,
        "within_sim": 0.0, "loss": 0.0, "iters": 0, "rej_frac": 0.0,
    })
    for r in rows:
        cid = r["class_id"]
        pc = per_class[cid]
        pc["n"] += 1
        pc["retrieved"] += int(r["retrieved_top5"])
        pc["jammed"] += int(r["jammed_honest"])
        pc["within_sim"] = float(r["within_class_sim"])
        pc["loss"] = float(r["final_loss"])
        pc["iters"] = int(r["n_iterations"])
        denom = int(r["cands_scored"]) or 1
        pc["rej_frac"] = float(r["cands_rejected"]) / denom

    print(f"\n{'Class':<22} {'n':>3} {'sim':>6} {'ret%':>6} {'asr%':>6} "
          f"{'rej%':>6} {'loss':>7} {'iters':>6} {'min':>6}")
    print("-" * 75)
    total_n = total_ret = total_jam = 0
    for cid, pc in sorted(per_class.items()):
        n, ret, jam = pc["n"], pc["retrieved"], pc["jammed"]
        t = timings.get(cid, 0.0)
        print(
            f"{cid:<22} {n:>3} {pc['within_sim']:>6.3f} "
            f"{100*ret/n:>5.0f}% {100*jam/n:>5.0f}% "
            f"{100*pc['rej_frac']:>5.0f}% "
            f"{pc['loss']:>7.4f} {pc['iters']:>6} {t:>6.1f}"
        )
        total_n += n; total_ret += ret; total_jam += jam

    print("-" * 75)
    print(f"{'TOTAL':<22} {total_n:>3}        "
          f"{100*total_ret/total_n:>5.0f}% {100*total_jam/total_n:>5.0f}%")

    n_ret = sum(int(r["retrieved_top5"]) for r in rows)
    jam_ret = sum(int(r["jammed_honest"]) for r in rows if int(r["retrieved_top5"]))
    if n_ret:
        print(f"\n  P(jam | retrieved): {jam_ret}/{n_ret} = {100*jam_ret/n_ret:.1f}%")

    if timings:
        total_t = sum(timings.values())
        print(f"  Total time: {total_t:.1f} min ({total_t/60:.1f} h) "
              f"for {len(timings)} classes")
    print("=" * 70)


if __name__ == "__main__":
    main()
