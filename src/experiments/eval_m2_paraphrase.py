"""
M2 class-averaged BBO evaluation — first 20 paraphrase classes.

Phases
------
1. BBO optimisation  (GTR-base retriever + vLLM + BGE-large oracle on GPU 1)
   For each of the first 20 paraphrase classes:
     - d_r  = build_dr_representative(queries, embs, centroid)
     - M2ClassBBO.run_class(queries, d_r) → optimised class blocker
     - ShafranBBO.run(q_star)            → single-query baseline blocker
     - Checkpoint to disk after each class (safe to resume from crash)

2. Response generation  (vLLM still loaded)
   Build all RAG prompts for final blockers × all queries; single-batch
   vLLM call to get all 20×2×n_queries responses.

3. Judge  (vLLM closed, Gemma-2-9B loaded on same GPU)
   Binary refusal judge; save final CSV.

Output: /home/ishana/scratch/results/task4_m2_paraphrase.csv
GPU:    CUDA_VISIBLE_DEVICES=1  (45 GB free)
"""
from __future__ import annotations

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

# ── Path / env setup ──────────────────────────────────────────────────────────
os.environ["HF_HOME"] = "/home/ishana/scratch/hf_cache"
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
OUT_CSV     = RESULTS_DIR / "task4_m2_paraphrase.csv"
CKPT_FILE   = RESULTS_DIR / "task4_m2_paraphrase_ckpt.pkl"

GTR_MODEL   = "sentence-transformers/gtr-t5-base"
LLM_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL = "google/gemma-2-9b-it"
N_CLASSES   = 20    # first 20 paraphrase classes


# ── Config ────────────────────────────────────────────────────────────────────

def _build_cfg():
    """Merge base.yaml with M2 overrides into an OmegaConf DictConfig."""
    cfg_dir = Path(__file__).resolve().parents[2] / "configs"
    base = OmegaConf.load(cfg_dir / "base.yaml")
    m2   = OmegaConf.load(cfg_dir / "m2_paraphrase.yaml")
    # Remove the 'defaults' key (Hydra-only) before merging
    m2_clean = OmegaConf.masked_copy(
        m2, [k for k in m2 if k != "defaults"]
    )
    return OmegaConf.merge(base, m2_clean)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_paraphrase_classes(n: int) -> list[dict]:
    """
    Load first `n` paraphrase classes with pre-computed BGE embeddings.
    Returns list of dicts with keys: class_id, _queries, _embeddings, _centroid.
    """
    with open(DATA_DIR / "paraphrase_classes.json") as f:
        classes = json.load(f)
    embs = np.load(DATA_DIR / "paraphrase_embeddings.npy")  # (N, 6, D)

    result = []
    for i, cls in enumerate(classes[:n]):
        queries = [cls["original_query"]] + cls["paraphrases"]
        n_q = len(queries)
        cls["_queries"]     = queries
        cls["_embeddings"]  = embs[i, :n_q]  # (n_queries, D)
        cls["_centroid"]    = np.array(cls["centroid"], dtype=np.float32)
        cls.setdefault("class_id", f"para_{i:03d}")
        result.append(cls)
    log.info("Loaded %d paraphrase classes", len(result))
    return result


# ── Response generation ───────────────────────────────────────────────────────

def generate_responses(
    blocker_docs: list[str],
    queries: list[str],
    retriever,
    generator,
    rag_prompt_template: str,
    retrieval_k: int,
) -> list[str]:
    """
    Inject each blocker as top-1 retrieved doc, generate one response per query.
    Returns responses aligned with queries (same length).
    """
    all_prompts: list[str] = []
    for blocker_doc, query in zip(blocker_docs, queries):
        top_k_ids = retriever.retrieve(query, k=retrieval_k)
        base_docs = [retriever.get_doc_text(d) for d in top_k_ids]
        ctx_docs  = [blocker_doc] + base_docs[: retrieval_k - 1]
        context   = "\n\n".join(ctx_docs)
        prompt    = rag_prompt_template.format(context=context, query=query)
        all_prompts.append(prompt)

    return generator.generate(all_prompts)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    cfg = _build_cfg()
    log.info("Config: num_iterations=%d, es_patience=%d, mini_batch_k=%d, batch_size=%d",
             cfg.attack.num_iterations, cfg.attack.es_patience,
             cfg.attack.get("mini_batch_k", 3), cfg.attack.batch_size)

    # ── Load classes ──────────────────────────────────────────────────────────
    classes = load_paraphrase_classes(N_CLASSES)

    # ── Load retriever (kept loaded all phases) ───────────────────────────────
    log.info("Loading GTR-base retriever...")
    from rag.retriever import Retriever
    retriever = Retriever(model_name=GTR_MODEL, score_function="cos_sim")
    retriever.load_index(INDEX_DIR)
    log.info("Retriever loaded: %d docs in index", retriever._index.ntotal)

    # ── Phase 1 & 2: BBO + response generation (LLM loaded) ──────────────────
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

    from attacks.m2_class_bbo import M2ClassBBO
    from attacks.shafran_bbo import ShafranBBO
    from attacks.m1_retrieval import build_dr_representative

    m2_attacker      = M2ClassBBO(cfg, retriever, generator, gpu_manager)
    shafran_attacker = ShafranBBO(cfg, retriever, generator, gpu_manager)

    # Load checkpoint if one exists (allows crash-resume)
    ckpt: dict[str, dict] = {}
    if CKPT_FILE.exists():
        with open(CKPT_FILE, "rb") as f:
            ckpt = pickle.load(f)
        log.info("Resumed from checkpoint: %d classes already done", len(ckpt))

    # ── BBO loop (phase 1) ────────────────────────────────────────────────────
    for cls in classes:
        cid     = cls["class_id"]
        queries = cls["_queries"]
        embs    = cls["_embeddings"]
        centroid = cls["_centroid"]

        if cid in ckpt:
            log.info("[SKIP] %s already in checkpoint", cid)
            continue

        log.info("=" * 60)
        log.info("Class %s | n_queries=%d", cid, len(queries))

        d_r   = build_dr_representative(queries, embs, centroid)
        q_star = d_r  # representative query doubles as d_r for simple M1

        # M2 optimisation
        log.info("  Running M2ClassBBO.run_class()...")
        t0 = time.time()
        m2_result = m2_attacker.run_class(
            queries=queries,
            d_r=d_r,
            rng_seed=int(cfg.attack.seed),
        )
        log.info("  M2 done in %.1f min | final_loss=%.4f | iters=%d",
                 (time.time() - t0) / 60, m2_result.final_loss, m2_result.n_iterations)

        # Shafran baseline (single-query BBO on q_star)
        log.info("  Running ShafranBBO.run(q_star)...")
        t0 = time.time()
        shafran_result = shafran_attacker.run(query=q_star)
        log.info("  Shafran done in %.1f min | final_loss=%.4f | iters=%d",
                 (time.time() - t0) / 60, shafran_result.final_loss, shafran_result.n_iterations)

        ckpt[cid] = {
            "queries":        queries,
            "d_r":            d_r,
            "m2_result":      m2_result,
            "shafran_result": shafran_result,
        }
        with open(CKPT_FILE, "wb") as f:
            pickle.dump(ckpt, f)
        log.info("  Checkpointed %s", cid)

    # ── Phase 2: generate responses for all final blockers ────────────────────
    log.info("=" * 60)
    log.info("Phase 2: generating responses for all final blockers...")

    rag_prompt = cfg.rag_prompt
    retrieval_k = int(cfg.retrieval.k)

    # Build flat list of (cid, variant, query_idx, query, blocker_doc)
    eval_records: list[tuple[str, str, int, str, str, float, int]] = []
    for cls in classes:
        cid = cls["class_id"]
        entry = ckpt[cid]
        queries = entry["queries"]
        for q_idx, query in enumerate(queries):
            eval_records.append((
                cid, "m2", q_idx, query,
                entry["m2_result"].final_doc,
                entry["m2_result"].final_loss,
                entry["m2_result"].n_iterations,
            ))
        for q_idx, query in enumerate(queries):
            eval_records.append((
                cid, "shafran", q_idx, query,
                entry["shafran_result"].final_doc,
                entry["shafran_result"].final_loss,
                entry["shafran_result"].n_iterations,
            ))

    blocker_docs = [r[4] for r in eval_records]
    query_texts  = [r[3] for r in eval_records]

    log.info("Generating %d responses...", len(eval_records))
    responses = generate_responses(
        blocker_docs, query_texts, retriever, generator, rag_prompt, retrieval_k
    )
    log.info("Response generation complete.")

    # ── Close LLM, free VRAM for judge ───────────────────────────────────────
    log.info("Closing vLLM to free VRAM for judge...")
    generator.close()
    del generator
    del m2_attacker
    del shafran_attacker
    gc.collect()
    torch.cuda.empty_cache()
    log.info("vLLM closed.")

    # ── Phase 3: judge ────────────────────────────────────────────────────────
    log.info("Loading judge: %s", JUDGE_MODEL)
    from judges.local_judge import LocalJudge
    judge = LocalJudge(model_name=JUDGE_MODEL)

    rows: list[dict] = []
    for i, (cid, variant, q_idx, query, blocker_doc, final_loss, n_iters) in enumerate(eval_records):
        rows.append({
            "class_id":    cid,
            "variant":     variant,
            "query_idx":   q_idx,
            "query_text":  query,
            "blocker_doc": blocker_doc[:300],
            "response":    responses[i],
            "final_loss":  round(final_loss, 5),
            "n_iterations": n_iters,
            "jam_success": None,  # filled by judge
        })

    log.info("Judging %d responses...", len(rows))
    for row in rows:
        # jam_success=1 means the blocker caused a refusal (attack succeeded)
        row["jam_success"] = int(not judge.is_answered(row["query_text"], row["response"]))

    # ── Save CSV ──────────────────────────────────────────────────────────────
    fieldnames = [
        "class_id", "variant", "query_idx", "query_text",
        "blocker_doc", "response", "final_loss", "n_iterations", "jam_success",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Saved %d rows → %s", len(rows), OUT_CSV)

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(rows)
    log.info("Total runtime: %.1f min", (time.time() - t_start) / 60)


def _print_summary(rows: list[dict]) -> None:
    from collections import defaultdict
    buckets: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        buckets[r["variant"]].append(int(r["jam_success"]))

    print("\n" + "=" * 60)
    print("M2 PARAPHRASE EVALUATION — SUMMARY")
    print("=" * 60)
    print(f"{'Variant':<12} {'Jammed':>8} {'Total':>7} {'ASR':>8}")
    print("-" * 40)
    for variant in ["m2", "shafran"]:
        vals = buckets.get(variant, [])
        if not vals:
            continue
        n_jammed = sum(vals)
        n_total  = len(vals)
        asr      = 100 * n_jammed / n_total
        print(f"{variant:<12} {n_jammed:>8} {n_total:>7} {asr:>7.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
