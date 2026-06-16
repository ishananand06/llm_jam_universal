"""
Experiment 2: Ablation — white-box GTR loss only, random-suffix init.

Isolates whether the white-box GTR joint-loss term, independent of the
vec2text-perturbed initialization, is responsible for the performance gain
observed in the original pilot.

Design:
  - Same 6 classes as the original white-box pilot:
      Paraphrase: para_test1, para_test2, para_test6
      Entity:     entity_00, entity_08, entity_09
  - Same white-box GTR joint loss: λ·(−GTR_cos) + (1−λ)·(−jam_cos)
  - Same λ=0.5 (pilot-optimal)
  - Same BBO budget: T=500, patience=50, n=50
  - DIFFERENT init: 50 '!' tokens (Shafran standard random-suffix init)
    instead of vec2text-perturbed init

Comparison:
  A. Original pilot (WB λ=0.5, vec2text init)   — loaded from existing ckpt
  B. This script     (WB λ=0.5, random '!' init) — new runs

If A ≈ B: the init did nothing; the GTR loss term is the driver.
If A >> B: the init contributed non-trivially; the causal story is more complex.

Output
------
/home/ishana/scratch/results/whitebox_ablation_init_honest.csv
/home/ishana/scratch/results/whitebox_ablation_init_ckpt.pkl
/home/ishana/scratch/results/whitebox_ablation_init_report.md
/home/ishana/scratch/results/whitebox_ablation_init_trajectory_<class>.csv

Run detached in tmux:
  tmux new-session -d -s wb_ablation \\
    '.venv/bin/python src/experiments/eval_whitebox_ablation_init.py \
       > /home/ishana/scratch/results/whitebox_ablation_init.log 2>&1'
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
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

# ── Env / path ────────────────────────────────────────────────────────────────
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

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("/home/ishana/scratch/data/classes")
INDEX_DIR   = Path("/home/ishana/scratch/data/indices/nq/sentence-transformers__gtr-t5-base")
RESULTS_DIR = Path("/home/ishana/scratch/results")
OUT_CSV     = RESULTS_DIR / "whitebox_ablation_init_honest.csv"
CKPT_FILE   = RESULTS_DIR / "whitebox_ablation_init_ckpt.pkl"

# The existing pilot checkpoint to load baseline results from
PILOT_CKPT  = RESULTS_DIR / "whitebox_pilot_ckpt.pkl"

GTR_MODEL   = "sentence-transformers/gtr-t5-base"
BGE_MODEL   = "BAAI/bge-large-en-v1.5"
LLM_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL = "google/gemma-2-9b-it"

TOP_K        = 5
BBO_T        = 500
BBO_PATIENCE = 50
N_CANDIDATES = 50
MIN_TOKENS   = 50
LAM          = 0.5
RUN_TIMEOUT  = 3600

FILLER_TOKEN = "!"

# Same 6 classes as the original pilot for direct comparison
PARA_CLASS_IDS   = ["test1", "test2", "test6"]
ENTITY_CLASS_IDS = ["entity_00", "entity_08", "entity_09"]

TARGET_RESPONSE = (
    "I don't know. The context does not provide enough information to "
    "answer the query."
)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def _build_cfg():
    cfg_dir = Path(__file__).resolve().parents[2] / "configs"
    base = OmegaConf.load(cfg_dir / "base.yaml")
    base.attack.num_iterations       = BBO_T
    base.attack.es_patience          = BBO_PATIENCE
    base.attack.batch_size           = N_CANDIDATES
    base.vllm.gpu_memory_utilization = 0.65
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Class loading (mirrors pilot)
# ─────────────────────────────────────────────────────────────────────────────

def _load_paraphrase_classes(class_ids):
    with open(DATA_DIR / "paraphrase_classes.json") as f:
        all_classes = json.load(f)
    embs_all = np.load(DATA_DIR / "paraphrase_embeddings.npy")
    id_to_idx = {c["class_id"]: i for i, c in enumerate(all_classes)}
    result = []
    for cid in class_ids:
        i = id_to_idx[cid]
        cls = all_classes[i]
        queries = [cls["original_query"]] + cls["paraphrases"]
        n = len(queries)
        e = embs_all[i, :n].astype(np.float32)
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        e_norm = e / np.where(norms < 1e-9, 1.0, norms)
        centroid = np.array(cls["centroid"], dtype=np.float32)
        c_norm = centroid / max(float(np.linalg.norm(centroid)), 1e-9)
        pw = e_norm @ e_norm.T
        pairs = [pw[r, c] for r in range(n) for c in range(r + 1, n)]
        within = float(np.mean(pairs)) if pairs else 0.0
        result.append({
            "class_id":    f"para_{cid}",
            "class_type":  "paraphrase",
            "_queries":    queries,
            "_embeddings": e_norm,
            "_centroid":   c_norm,
            "_within_sim": within,
        })
    return result


def _load_entity_classes(class_ids):
    with open(DATA_DIR / "entity_classes.json") as f:
        all_classes = json.load(f)
    embs_all = np.load(DATA_DIR / "entity_embeddings.npy")
    id_to_idx = {c["class_id"]: i for i, c in enumerate(all_classes)}
    result = []
    for cid in class_ids:
        i = id_to_idx[cid]
        cls = all_classes[i]
        queries = cls["queries"]
        n = len(queries)
        e = embs_all[i, :n].astype(np.float32)
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        e_norm = e / np.where(norms < 1e-9, 1.0, norms)
        centroid = np.array(cls["centroid"], dtype=np.float32)
        c_norm = centroid / max(float(np.linalg.norm(centroid)), 1e-9)
        pw = e_norm @ e_norm.T
        pairs = [pw[r, c] for r in range(n) for c in range(r + 1, n)]
        within = float(np.mean(pairs)) if pairs else 0.0
        result.append({
            "class_id":    cid,
            "class_type":  "entity",
            "_queries":    queries,
            "_embeddings": e_norm,
            "_centroid":   c_norm,
            "_within_sim": within,
        })
    return result


def find_qstar(queries, embs, centroid):
    dists = np.linalg.norm(embs - centroid[np.newaxis, :], axis=1)
    return queries[int(np.argmin(dists))], int(np.argmin(dists))


# ─────────────────────────────────────────────────────────────────────────────
# BGE helper
# ─────────────────────────────────────────────────────────────────────────────

def _embed_bge(oracle_embedder, texts):
    return oracle_embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Random '!' initialization (Shafran standard)
# ─────────────────────────────────────────────────────────────────────────────

def make_exclamation_init(tokenizer) -> list[int]:
    """Return MIN_TOKENS copies of the '!' token ID."""
    pad_tok = tokenizer.encode(FILLER_TOKEN, add_special_tokens=False)[0]
    return [pad_tok] * MIN_TOKENS


# ─────────────────────────────────────────────────────────────────────────────
# White-box joint-loss BBO (identical to scaleup script)
# ─────────────────────────────────────────────────────────────────────────────

def tokens_to_doc(q_star, tokens, tokenizer):
    return f"{q_star}. {tokenizer.decode(tokens, skip_special_tokens=True)}"


def score_candidates_joint(
    candidate_docs, q_star_emb_gtr, q_star_emb_bge, target_resp_emb,
    real_threshold, lam, retriever, oracle_embedder, generator,
    rag_prompt_template, retrieval_k, q_star,
):
    n = len(candidate_docs)
    losses    = np.full(n, np.inf, dtype=np.float32)
    gtr_sims  = np.zeros(n, dtype=np.float32)
    bge_sims  = np.zeros(n, dtype=np.float32)
    responses = [""] * n

    cand_embs_gtr = retriever.embed_batch(candidate_docs)
    gtr_sims_raw  = cand_embs_gtr @ q_star_emb_gtr
    gtr_sims      = gtr_sims_raw.copy()

    cand_embs_bge = _embed_bge(oracle_embedder, candidate_docs)
    bge_sims      = (cand_embs_bge @ q_star_emb_bge).copy()

    retrieved_indices = np.where(gtr_sims_raw >= real_threshold)[0].tolist()
    if not retrieved_indices:
        return losses, gtr_sims, bge_sims, responses

    top_k_ids = retriever.retrieve(q_star, k=retrieval_k)
    base_docs = [retriever.get_doc_text(d) for d in top_k_ids]
    prompts   = []
    for idx in retrieved_indices:
        ctx_docs = [candidate_docs[idx]] + base_docs[: retrieval_k - 1]
        prompts.append(rag_prompt_template.format(
            context="\n\n".join(ctx_docs), query=q_star))

    gen_resps = generator.generate(prompts)
    for i, idx in enumerate(retrieved_indices):
        responses[idx] = gen_resps[i]

    resp_embs = _embed_bge(oracle_embedder, gen_resps)
    for i, idx in enumerate(retrieved_indices):
        jam_sim = float(np.dot(resp_embs[i], target_resp_emb))
        losses[idx] = lam * (-float(gtr_sims_raw[idx])) + (1.0 - lam) * (-jam_sim)

    return losses, gtr_sims, bge_sims, responses


def get_rank_gtr(blocker_emb_gtr, q_star_emb_gtr, top5_scores_cache):
    blocker_sim = float(np.dot(blocker_emb_gtr, q_star_emb_gtr))
    if blocker_sim < top5_scores_cache[-1]:
        return -1
    return sum(1 for s in top5_scores_cache if s > blocker_sim) + 1


def run_bbo_whitebox_random_init(
    q_star, init_tokens, lam,
    q_star_emb_gtr, q_star_emb_bge, target_resp_emb,
    real_threshold, top5_scores_cache,
    candidate_vocab, tokenizer, retriever, oracle_embedder, generator,
    rag_prompt_template, retrieval_k, trajectory_csv, class_seed,
):
    rng = np.random.default_rng(class_seed + int(lam * 100))
    cur_tokens  = list(init_tokens)
    cur_doc     = tokens_to_doc(q_star, cur_tokens, tokenizer)
    blocker_len = len(cur_tokens)

    losses, gtr_s, bge_s, _ = score_candidates_joint(
        [cur_doc], q_star_emb_gtr, q_star_emb_bge, target_resp_emb,
        real_threshold, lam, retriever, oracle_embedder, generator,
        rag_prompt_template, retrieval_k, q_star,
    )
    cur_loss    = float(losses[0])
    cur_gtr_sim = float(gtr_s[0])
    cur_bge_sim = float(bge_s[0])
    best_loss   = cur_loss
    best_doc    = cur_doc
    best_tokens = list(cur_tokens)
    es_count    = 0
    n_iters     = 0
    cands_scored   = 0
    cands_accepted = 0

    traj_rows = []

    def _flush(rows):
        write_header = not trajectory_csv.exists()
        with open(trajectory_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "iter_num", "best_loss_so_far", "retrieval_term",
                "retrieval_term_bge", "joint_loss", "current_top5_rank_gtr",
                "cands_scored", "cands_accepted",
            ])
            if write_header:
                w.writeheader()
            w.writerows(rows)
        rows.clear()

    cur_blocker_emb = retriever.embed_batch([cur_doc])[0]
    cur_rank = get_rank_gtr(cur_blocker_emb, q_star_emb_gtr, top5_scores_cache)
    traj_rows.append({
        "iter_num": 0, "best_loss_so_far": round(cur_loss, 5),
        "retrieval_term": round(cur_gtr_sim, 4),
        "retrieval_term_bge": round(cur_bge_sim, 4),
        "joint_loss": round(cur_loss, 5),
        "current_top5_rank_gtr": cur_rank,
        "cands_scored": 0, "cands_accepted": 0,
    })
    _flush(traj_rows)

    t_start = time.time()

    for iteration in range(1, BBO_T + 1):
        n_iters = iteration
        pos     = int(rng.integers(0, blocker_len))
        sampled = rng.choice(candidate_vocab, size=N_CANDIDATES, replace=False)
        cand_token_lists = []
        for tok_id in sampled:
            c = list(cur_tokens)
            c[pos] = int(tok_id)
            cand_token_lists.append(c)
        cand_docs = [tokens_to_doc(q_star, c, tokenizer) for c in cand_token_lists]

        losses, gtr_s, bge_s, _ = score_candidates_joint(
            cand_docs, q_star_emb_gtr, q_star_emb_bge, target_resp_emb,
            real_threshold, lam, retriever, oracle_embedder, generator,
            rag_prompt_template, retrieval_k, q_star,
        )

        n_ret = int(np.sum(losses < np.inf))
        cands_scored   += N_CANDIDATES
        cands_accepted += n_ret

        best_idx  = int(np.argmin(losses))
        best_cand = float(losses[best_idx])

        if best_cand < cur_loss:
            cur_tokens  = cand_token_lists[best_idx]
            cur_doc     = cand_docs[best_idx]
            cur_loss    = best_cand
            cur_gtr_sim = float(gtr_s[best_idx])
            cur_bge_sim = float(bge_s[best_idx])
            es_count    = 0
            if cur_loss < best_loss:
                best_loss   = cur_loss
                best_doc    = cur_doc
                best_tokens = list(cur_tokens)
        else:
            es_count += 1

        cur_blocker_emb = retriever.embed_batch([cur_doc])[0]
        cur_rank = get_rank_gtr(cur_blocker_emb, q_star_emb_gtr, top5_scores_cache)

        if iteration % 10 == 0:
            log.info("  iter %d | loss=%.4f | gtr=%.4f | bge=%.4f | rank=%s | es=%d",
                     iteration, cur_loss, cur_gtr_sim, cur_bge_sim,
                     cur_rank if cur_rank > 0 else "OUT", es_count)

        traj_rows.append({
            "iter_num": iteration, "best_loss_so_far": round(best_loss, 5),
            "retrieval_term": round(cur_gtr_sim, 4),
            "retrieval_term_bge": round(cur_bge_sim, 4),
            "joint_loss": round(cur_loss, 5),
            "current_top5_rank_gtr": cur_rank,
            "cands_scored": cands_scored, "cands_accepted": cands_accepted,
        })
        if len(traj_rows) >= 50:
            _flush(traj_rows)

        if time.time() - t_start > RUN_TIMEOUT:
            log.warning("  Timeout at iter %d", iteration)
            break
        if es_count >= BBO_PATIENCE:
            log.info("  Early stop at iter %d", iteration)
            break

    if traj_rows:
        _flush(traj_rows)

    best_emb_gtr  = retriever.embed_batch([best_doc])[0]
    best_emb_bge  = _embed_bge(oracle_embedder, [best_doc])[0]

    return {
        "final_doc":      best_doc,
        "final_loss":     best_loss,
        "n_iterations":   n_iters,
        "cands_scored":   cands_scored,
        "cands_accepted": cands_accepted,
        "final_gtr_cos":  float(np.dot(best_emb_gtr, q_star_emb_gtr)),
        "final_bge_cos":  float(np.dot(best_emb_bge, q_star_emb_bge)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Honest retrieval check
# ─────────────────────────────────────────────────────────────────────────────

def check_top5_retrieval(blocker_emb, query, retriever):
    top_k_results = retriever.retrieve_with_scores(query, k=TOP_K)
    scores    = [s for _, s in top_k_results]
    threshold = scores[-1] if scores else 0.0
    q_emb     = retriever.embed_batch([query])[0]
    b_sim     = float(np.dot(blocker_emb, q_emb))
    retrieved = b_sim >= threshold
    rank      = sum(1 for s in scores if s > b_sim) + 1 if retrieved else None
    return retrieved, rank, b_sim, threshold


def build_prompts_for_retrieved(blocker_doc, queries, ret_results,
                                retriever, rag_prompt_template, k):
    prompts, ret_idxs = [], []
    for q_idx, (q, ret) in enumerate(zip(queries, ret_results)):
        if not ret["retrieved"]:
            continue
        top_k_ids = retriever.retrieve(q, k=k)
        base_docs = [retriever.get_doc_text(d) for d in top_k_ids]
        ctx_docs  = [blocker_doc] + base_docs[: k - 1]
        prompts.append(rag_prompt_template.format(
            context="\n\n".join(ctx_docs), query=q))
        ret_idxs.append(q_idx)
    return prompts, ret_idxs


# ─────────────────────────────────────────────────────────────────────────────
# Load pilot results for comparison
# ─────────────────────────────────────────────────────────────────────────────

def _load_pilot_asr() -> dict:
    """
    Returns {class_id: jammed_honest_count} for λ=0.5 from the existing pilot
    checkpoint. These are the 'WB + vec2text init' reference numbers.
    """
    if not PILOT_CKPT.exists():
        log.warning("Pilot checkpoint not found at %s", PILOT_CKPT)
        return {}
    with open(PILOT_CKPT, "rb") as f:
        pilot_ckpt = pickle.load(f)

    result = {}
    lam    = 0.5
    for cid_short in ["test1", "test2", "test6", "entity_00", "entity_08", "entity_09"]:
        # Pilot uses "para_test1" style for paraphrase, "entity_00" for entity
        full_cid = f"para_{cid_short}" if cid_short.startswith("test") else cid_short
        key      = f"eval_{full_cid}_lam{lam}"
        if key in pilot_ckpt:
            ev = pilot_ckpt[key]
            result[full_cid] = {
                "jammed_honest": ev.get("jammed_honest_total", 0),
                "n_queries":     ev.get("n_queries", 0),
            }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    log.info("=" * 70)
    log.info("Experiment 2: White-box Ablation — random '!' init vs vec2text init")
    log.info("  Para classes  : %s", PARA_CLASS_IDS)
    log.info("  Entity classes: %s", ENTITY_CLASS_IDS)
    log.info("  λ=%.1f | T=%d | patience=%d | n=%d", LAM, BBO_T, BBO_PATIENCE, N_CANDIDATES)
    log.info("  Init: %d × '%s' tokens (Shafran standard)", MIN_TOKENS, FILLER_TOKEN)
    log.info("=" * 70)

    para_classes   = _load_paraphrase_classes(PARA_CLASS_IDS)
    entity_classes = _load_entity_classes(ENTITY_CLASS_IDS)
    all_classes    = para_classes + entity_classes

    cfg = _build_cfg()

    ckpt: dict = {}
    if CKPT_FILE.exists():
        with open(CKPT_FILE, "rb") as f:
            ckpt = pickle.load(f)
        log.info("Resumed checkpoint (%d keys)", len(ckpt))

    def _save_ckpt():
        with open(CKPT_FILE, "wb") as f:
            pickle.dump(ckpt, f)

    # ── Retriever ────────────────────────────────────────────────────────────
    log.info("Loading GTR retriever …")
    from rag.retriever import Retriever
    retriever = Retriever(model_name=GTR_MODEL, score_function="cos_sim")
    retriever.load_index(INDEX_DIR)
    log.info("Retriever: %d docs", retriever._index.ntotal)

    # ── BGE oracle ───────────────────────────────────────────────────────────
    log.info("Loading BGE oracle …")
    from sentence_transformers import SentenceTransformer
    oracle_embedder = SentenceTransformer(
        BGE_MODEL, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    target_resp_emb = _embed_bge(oracle_embedder, [TARGET_RESPONSE])[0]

    # ── Tokenizer + vocab ────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

    from datasets import load_dataset
    log.info("Building candidate vocab …")
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        counts = np.zeros(len(tokenizer), dtype=np.int64)
        for ex in ds:
            for tid in tokenizer.encode(ex["text"], add_special_tokens=False):
                if 0 <= tid < len(counts):
                    counts[tid] += 1
        top_ids = np.argsort(counts)[::-1][:100]
        candidate_vocab = np.delete(np.arange(len(tokenizer)), top_ids)
    except Exception as e:
        log.warning("Wikitext failed (%s) — uniform vocab", e)
        candidate_vocab = np.arange(len(tokenizer))
    log.info("Candidate vocab: %d tokens", len(candidate_vocab))

    # Exclamation-mark init tokens (shared for all classes)
    excl_init_tokens = make_exclamation_init(tokenizer)
    log.info("Init: %d × token_id=%d ('%s')", len(excl_init_tokens),
             excl_init_tokens[0], FILLER_TOKEN)

    # ── Load vLLM ────────────────────────────────────────────────────────────
    log.info("=" * 60)
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
    rag_prompt = cfg.rag_prompt

    # ── Phase 2: White-box BBO with random '!' init ───────────────────────────
    log.info("=" * 60)
    log.info("PHASE 2: White-box GTR BBO (random '!' init, λ=%.1f)", LAM)

    for cls_idx, cls in enumerate(all_classes):
        cid    = cls["class_id"]
        key_bbo = f"bbo_{cid}"
        if key_bbo in ckpt:
            log.info("  [SKIP] %s", cid)
            continue
        queries, embs, centroid = cls["_queries"], cls["_embeddings"], cls["_centroid"]
        q_star, q_star_idx = find_qstar(queries, embs, centroid)
        log.info("Class %s | q*=%r", cid, q_star[:70])

        top5_scores_cache = [s for _, s in retriever.retrieve_with_scores(q_star, k=TOP_K)]
        real_threshold    = top5_scores_cache[-1]
        q_star_emb_gtr    = retriever.embed_batch([q_star])[0]
        q_star_emb_bge    = _embed_bge(oracle_embedder, [q_star])[0]

        log.info("  real_thr=%.4f | init: 50 × '!'", real_threshold)
        traj_csv = RESULTS_DIR / f"whitebox_ablation_init_trajectory_{cid}.csv"
        t0 = time.time()
        bbo_res = run_bbo_whitebox_random_init(
            q_star=q_star,
            init_tokens=list(excl_init_tokens),  # random '!' init — the ablation key
            lam=LAM,
            q_star_emb_gtr=q_star_emb_gtr,
            q_star_emb_bge=q_star_emb_bge,
            target_resp_emb=target_resp_emb,
            real_threshold=real_threshold,
            top5_scores_cache=top5_scores_cache,
            candidate_vocab=candidate_vocab,
            tokenizer=tokenizer,
            retriever=retriever,
            oracle_embedder=oracle_embedder,
            generator=generator,
            rag_prompt_template=rag_prompt,
            retrieval_k=TOP_K,
            trajectory_csv=traj_csv,
            class_seed=42 + cls_idx,
        )
        log.info("  BBO done %.1f min | loss=%.4f | gtr_cos=%.4f | bge_cos=%.4f",
                 (time.time() - t0) / 60, bbo_res["final_loss"],
                 bbo_res["final_gtr_cos"], bbo_res["final_bge_cos"])
        bbo_res["q_star"]     = q_star
        bbo_res["q_star_idx"] = q_star_idx
        bbo_res["real_threshold"] = real_threshold
        ckpt[key_bbo] = bbo_res
        _save_ckpt()

    # ── Close vLLM ───────────────────────────────────────────────────────────
    log.info("Closing vLLM …")
    generator.close()
    del generator
    gc.collect()
    torch.cuda.empty_cache()

    # ── Phase 3: Honest retrieval + generation ────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 3: Honest evaluation")
    from rag.generator import VLLMGenerator
    generator = VLLMGenerator(
        model_name=LLM_MODEL, temperature=0.0, max_tokens=128,
        gpu_memory_utilization=0.65, dtype="float16", max_model_len=4096,
    )

    for cls in all_classes:
        cid     = cls["class_id"]
        keyE    = f"eval_{cid}"
        key_bbo = f"bbo_{cid}"
        if keyE in ckpt:
            log.info("  [SKIP] %s (eval)", cid)
            continue
        queries = cls["_queries"]
        bbo_res = ckpt[key_bbo]
        final_doc = bbo_res["final_doc"]

        blocker_emb = retriever.embed_batch([final_doc])[0]
        ret_results = []
        for q in queries:
            retr, rank, b_sim, t_sim = check_top5_retrieval(blocker_emb, q, retriever)
            ret_results.append({"retrieved": retr, "rank": rank,
                                "blocker_sim": round(b_sim, 5),
                                "threshold_sim": round(t_sim, 5)})
        n_ret = sum(r["retrieved"] for r in ret_results)
        log.info("  %s: retrieval %d/%d", cid, n_ret, len(queries))

        prompts, ret_idxs = build_prompts_for_retrieved(
            final_doc, queries, ret_results, retriever, rag_prompt, TOP_K)
        responses: list[Optional[str]] = [None] * len(queries)
        if prompts:
            gen_resps = generator.generate(prompts)
            for qi, resp in zip(ret_idxs, gen_resps):
                responses[qi] = resp

        ckpt[keyE] = {"n_queries": len(queries), "n_retrieved": n_ret,
                      "responses": responses, "ret_results": ret_results}
        _save_ckpt()

    generator.close()
    del generator
    gc.collect()
    torch.cuda.empty_cache()

    # ── Judge ────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Loading judge: %s", JUDGE_MODEL)
    from judges.local_judge import LocalJudge
    judge = LocalJudge(model_name=JUDGE_MODEL)

    csv_rows: list[dict] = []
    ablation_asr: dict[str, dict] = {}

    for cls in all_classes:
        cid     = cls["class_id"]
        ctype   = cls["class_type"]
        queries = cls["_queries"]
        within  = cls["_within_sim"]
        bbo_res = ckpt[f"bbo_{cid}"]
        ev      = ckpt[f"eval_{cid}"]
        q_star  = bbo_res["q_star"]

        jam_total = 0
        for q_idx, (q, resp, ret) in enumerate(
            zip(queries, ev["responses"], ev["ret_results"])
        ):
            retrieved = bool(ret["retrieved"])
            if retrieved and resp is not None:
                jam_success = int(not judge.is_answered(q, resp))
                jammed_honest = jam_success
            else:
                jam_success   = None
                jammed_honest = 0
            jam_total += jammed_honest

            csv_rows.append({
                "class_id":         cid,
                "class_type":       ctype,
                "init_method":      "random_exclamation",
                "lambda":           LAM,
                "query_idx":        q_idx,
                "query_text":       q,
                "within_class_sim": round(within, 4),
                "n_class_queries":  len(queries),
                "blocker_doc":      bbo_res["final_doc"][:300],
                "blocker_sim":      ret["blocker_sim"],
                "threshold_sim":    ret["threshold_sim"],
                "retrieved_top5":   int(retrieved),
                "rank":             ret["rank"] if ret["rank"] is not None else "",
                "response":         resp[:300] if resp else "",
                "jam_success":      jam_success if jam_success is not None else "",
                "jammed_honest":    jammed_honest,
                "q_star":           q_star[:120],
                "final_loss":       round(bbo_res["final_loss"], 5),
                "n_iterations":     bbo_res["n_iterations"],
                "cands_scored":     bbo_res["cands_scored"],
                "cands_accepted":   bbo_res["cands_accepted"],
                "final_gtr_cos":    round(bbo_res["final_gtr_cos"], 5),
                "final_bge_cos":    round(bbo_res["final_bge_cos"], 5),
                "real_threshold":   round(bbo_res["real_threshold"], 5),
            })

        ablation_asr[cid] = {"jammed_honest": jam_total, "n": len(queries), "ctype": ctype}
        log.info("  %s: ablation ASR %d/%d", cid, jam_total, len(queries))

    judge.close()

    # Write CSV
    fieldnames = [
        "class_id", "class_type", "init_method", "lambda",
        "query_idx", "query_text", "within_class_sim", "n_class_queries",
        "blocker_doc", "blocker_sim", "threshold_sim",
        "retrieved_top5", "rank", "response", "jam_success", "jammed_honest",
        "q_star", "final_loss", "n_iterations",
        "cands_scored", "cands_accepted", "final_gtr_cos", "final_bge_cos",
        "real_threshold",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    log.info("Saved %d rows -> %s", len(csv_rows), OUT_CSV)

    # ── Report ────────────────────────────────────────────────────────────────
    _write_report(all_classes, ablation_asr)

    log.info("=" * 70)
    log.info("DONE. Total elapsed: %.1f min", (time.time() - t_total) / 60)
    log.info("=" * 70)


def _write_report(all_classes, ablation_asr):
    pilot_asr = _load_pilot_asr()
    report_path = RESULTS_DIR / "whitebox_ablation_init_report.md"
    lines = ["# Experiment 2: White-box Ablation — Init Method Comparison\n\n"]

    lines.append(
        "**Ablation question**: Does the vec2text-perturbed init contribute to the\n"
        "white-box GTR performance gain, or is the GTR loss term alone sufficient?\n\n"
        f"- **λ** = {LAM}  \n"
        f"- **Budget**: T={BBO_T}, patience={BBO_PATIENCE}, n={N_CANDIDATES}  \n"
        "- **Condition A** (pilot): WB GTR λ=0.5 + vec2text-perturbed init  \n"
        "- **Condition B** (this run): WB GTR λ=0.5 + random '!' init  \n\n"
    )

    lines.append("## Per-class results\n\n")
    lines.append(
        "| class_id | type | within_sim | "
        "A: WB+vec2text ASR | B: WB+random! ASR | Δ (A−B) |\n"
        "|----------|------|-----------|"
        "-------------------|-------------------|--------|\n"
    )

    p_a_jam, p_a_n = 0, 0
    p_b_jam, p_b_n = 0, 0
    e_a_jam, e_a_n = 0, 0
    e_b_jam, e_b_n = 0, 0

    for cls in all_classes:
        cid    = cls["class_id"]
        ctype  = cls["class_type"]
        within = round(cls["_within_sim"], 4)
        a      = pilot_asr.get(cid, {})
        b      = ablation_asr.get(cid, {})
        a_str  = f"{a.get('jammed_honest',0)}/{a.get('n_queries',0)}" if a else "—"
        b_str  = f"{b.get('jammed_honest',0)}/{b.get('n',0)}"

        if a and b.get("n"):
            delta = a.get("jammed_honest", 0) / a.get("n_queries", 1) - \
                    b.get("jammed_honest", 0) / b.get("n", 1)
            delta_str = f"{delta:+.0%}"
        else:
            delta_str = "—"

        lines.append(f"| {cid} | {ctype} | {within} | {a_str} | {b_str} | {delta_str} |\n")

        if ctype == "paraphrase":
            p_a_jam += a.get("jammed_honest", 0); p_a_n += a.get("n_queries", 0)
            p_b_jam += b.get("jammed_honest", 0); p_b_n += b.get("n", 0)
        else:
            e_a_jam += a.get("jammed_honest", 0); e_a_n += a.get("n_queries", 0)
            e_b_jam += b.get("jammed_honest", 0); e_b_n += b.get("n", 0)

    lines.append("\n## Aggregated comparison\n\n")
    lines.append(
        "| Condition | Paraphrase ASR | Entity ASR |\n"
        "|-----------|---------------|------------|\n"
    )

    def _fmt(jam, n):
        return f"{jam}/{n} ({100*jam/max(n,1):.0f}%)" if n else "—"

    lines.append(
        f"| A: WB GTR λ=0.5 + vec2text init (pilot) | "
        f"{_fmt(p_a_jam, p_a_n)} | {_fmt(e_a_jam, e_a_n)} |\n"
    )
    lines.append(
        f"| B: WB GTR λ=0.5 + random '!' init (ablation) | "
        f"{_fmt(p_b_jam, p_b_n)} | {_fmt(e_b_jam, e_b_n)} |\n\n"
    )

    lines.append("## Verdict\n\n")
    if p_b_n > 0 and p_a_n > 0:
        a_rate = p_a_jam / p_a_n
        b_rate = p_b_jam / p_b_n
        delta  = a_rate - b_rate
        if abs(delta) <= 0.05:
            lines.append(
                f"**Init is irrelevant.** Paraphrase ASR with vec2text init ({100*a_rate:.0f}%) "
                f"and with random '!' init ({100*b_rate:.0f}%) differ by only "
                f"{100*abs(delta):.0f}pp (≤5pp threshold). The GTR loss term alone "
                "drives the performance gain. The vec2text perturbation idea can be "
                "cleanly dropped from the paper's causal story.\n"
            )
        elif delta > 0.05:
            lines.append(
                f"**Init contributes non-trivially.** Vec2text init achieves {100*a_rate:.0f}% "
                f"vs {100*b_rate:.0f}% for random init (Δ={100*delta:.0f}pp > 5pp). "
                "The causal story is more complex — both the GTR loss term and the "
                "perturbed initialization contribute to the gain.\n"
            )
        else:
            lines.append(
                f"**Random init outperforms vec2text init.** {100*b_rate:.0f}% vs "
                f"{100*a_rate:.0f}% (Δ={100*abs(delta):.0f}pp). The vec2text init "
                "may have started the optimizer in a poor local region. The GTR loss "
                "term is the driver, and standard init is preferable.\n"
            )

    if e_b_n > 0:
        e_b_rate = e_b_jam / e_b_n
        lines.append(
            f"\n**Entity**: ablation ASR = {100*e_b_rate:.0f}% (vs pilot {100*e_a_jam/max(e_a_n,1):.0f}%). "
            "Entity classes remain bounded by geometry regardless of init method.\n"
        )

    with open(report_path, "w") as f:
        f.writelines(lines)
    log.info("Report -> %s", report_path)


if __name__ == "__main__":
    main()
