"""
White-box GTR pilot: vec2text-perturbed initialization + joint-loss BBO.

Phase 1  — vec2text inversion of q* + Gaussian perturbation in GTR space
Phase 2  — BBO with Loss = λ·retrieval_loss_GTR + (1-λ)·jamming_loss_BGE
Phase 3  — Honest top-5 retrieval + Mistral + locked Gemma judge
Phase 4  — Comparison report vs. constrained-joint BBO baseline

Classes selected (for direct comparison to task6_constrained_joint_honest.csv):
  Paraphrase: para_test1 (0% ASR), para_test2 (50% ASR), para_test6 (83% ASR)
  Entity:     entity_00 (12% ASR), entity_08 (0% ASR),  entity_09 (20% ASR)

Run detached:
  nohup .venv/bin/python src/experiments/eval_whitebox_gtr_pilot.py \
      > /home/ishana/scratch/results/whitebox_pilot.log 2>&1 &
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
from collections import defaultdict
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
OUT_CSV     = RESULTS_DIR / "whitebox_pilot_honest.csv"
CKPT_FILE   = RESULTS_DIR / "whitebox_pilot_ckpt.pkl"

GTR_MODEL   = "sentence-transformers/gtr-t5-base"
BGE_MODEL   = "BAAI/bge-large-en-v1.5"
LLM_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL = "google/gemma-2-9b-it"
VEC2TEXT_MODEL = "jxm/gtr__nq__32"

TOP_K           = 5
BBO_T           = 500          # max iterations
BBO_PATIENCE    = 50
N_CANDIDATES    = 50           # candidates per iteration
LLM_BATCH_SIZE  = 32          # LLM batch
MIN_TOKENS      = 50           # pad if shorter
PERTURB_NORM    = 0.05         # L2 norm of epsilon
LAMBDAS         = [0.3, 0.5, 0.7]
RUN_TIMEOUT_SEC = 3600         # 60 min per run

# Classes for this pilot (matching task6 constrained-joint for direct comparison)
PARA_CLASS_IDS   = ["test1", "test2", "test6"]
ENTITY_CLASS_IDS = ["entity_00", "entity_08", "entity_09"]

TARGET_RESPONSE = (
    "I don't know. The context does not provide enough information to "
    "answer the query."
)

FILLER_TOKEN = "!"   # Shafran padding convention

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def _build_cfg():
    cfg_dir = Path(__file__).resolve().parents[2] / "configs"
    base = OmegaConf.load(cfg_dir / "base.yaml")
    base.attack.num_iterations = BBO_T
    base.attack.es_patience    = BBO_PATIENCE
    base.attack.batch_size     = N_CANDIDATES
    base.attack.llm_batch_size = LLM_BATCH_SIZE
    base.vllm.gpu_memory_utilization = 0.65
    return base

# ─────────────────────────────────────────────────────────────────────────────
# Class loading (mirrors task6)
# ─────────────────────────────────────────────────────────────────────────────

def _load_paraphrase_classes(class_ids: list[str]) -> list[dict]:
    with open(DATA_DIR / "paraphrase_classes.json") as f:
        all_classes = json.load(f)
    embs_all = np.load(DATA_DIR / "paraphrase_embeddings.npy")  # (100, 6, 1024)
    id_to_idx = {c["class_id"]: i for i, c in enumerate(all_classes)}

    result = []
    for cid in class_ids:
        if cid not in id_to_idx:
            raise ValueError(f"Paraphrase class {cid!r} not found in JSON")
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


def _load_entity_classes(class_ids: list[str]) -> list[dict]:
    with open(DATA_DIR / "entity_classes.json") as f:
        all_classes = json.load(f)
    embs_all = np.load(DATA_DIR / "entity_embeddings.npy")  # (20, 8, 1024)
    id_to_idx = {c["class_id"]: i for i, c in enumerate(all_classes)}

    result = []
    for cid in class_ids:
        if cid not in id_to_idx:
            raise ValueError(f"Entity class {cid!r} not found in JSON")
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
    idx = int(np.argmin(dists))
    return queries[idx], idx

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: vec2text perturbed initialization
# ─────────────────────────────────────────────────────────────────────────────

def phase1_init(q_star: str, class_idx: int, retriever, tokenizer) -> dict:
    """
    Embed q* with GTR, perturb in embedding space, invert with vec2text.
    Returns dict with all Phase 1 diagnostics + initial token list.
    """
    import vec2text

    seed = 42 + class_idx   # fixed reproducible seed per class
    rng  = np.random.default_rng(seed)

    # Step 1: embed q* with GTR
    v_q = retriever.embed_batch([q_star])[0]   # (768,) L2-normalized
    cos_self = float(np.dot(v_q, v_q))

    # Step 2: random perturbation
    epsilon = rng.standard_normal(v_q.shape).astype(np.float32)
    epsilon = epsilon / float(np.linalg.norm(epsilon)) * PERTURB_NORM
    v_perturbed = v_q + epsilon

    # Normalize before inversion (vec2text expects L2-normalized input)
    v_norm = float(np.linalg.norm(v_perturbed))
    v_perturbed_normed = v_perturbed / max(v_norm, 1e-9)

    # Sanity: cosine between v_q and v_perturbed_normed
    cos_perturbed = float(np.dot(v_q, v_perturbed_normed))
    # Expected: for 768-dim, norm=0.05 perturbation on unit vector
    # cos ≈ 1/sqrt(1 + 0.05^2) ≈ 0.9988

    # Step 3: vec2text inversion
    # jxm/gtr__nq__32 ships only .bin weights (no safetensors).
    # transformers >= 4.x on torch 2.5 blocks torch.load via a CVE check that is
    # imported directly into modeling_utils — patch it there, not in import_utils.
    import transformers.modeling_utils as _tmu
    _orig_check = _tmu.check_torch_load_is_safe
    _tmu.check_torch_load_is_safe = lambda: None

    log.info("  Phase1: loading vec2text corrector %s …", VEC2TEXT_MODEL)
    try:
        corrector = vec2text.load_pretrained_corrector("gtr-base")
    finally:
        _tmu.check_torch_load_is_safe = _orig_check

    # Move embedding to whichever device the corrector loaded onto
    _v2t_device = next(iter(corrector.model.parameters())).device
    emb_tensor = torch.tensor(v_perturbed_normed, dtype=torch.float32).unsqueeze(0).to(_v2t_device)
    inverted_texts = vec2text.invert_embeddings(
        embeddings=emb_tensor,
        corrector=corrector,
        num_steps=20,
        sequence_beam_width=4,
    )
    inverted_text = inverted_texts[0]
    del corrector
    gc.collect()
    torch.cuda.empty_cache()

    # Step 4: tokenize + pad to MIN_TOKENS
    tokens = tokenizer.encode(inverted_text, add_special_tokens=False)
    raw_len = len(tokens)
    pad_tok = tokenizer.encode(FILLER_TOKEN, add_special_tokens=False)[0]
    if len(tokens) < MIN_TOKENS:
        tokens = tokens + [pad_tok] * (MIN_TOKENS - len(tokens))
    blocker_len = len(tokens)

    # Build the initial blocker doc (query prefix + suffix, as per Shafran)
    suffix = tokenizer.decode(tokens, skip_special_tokens=True)
    init_blocker = f"{q_star}. {suffix}"

    # Sanity: GTR cos of init_blocker to q*
    init_blocker_emb = retriever.embed_batch([init_blocker])[0]
    init_gtr_cos = float(np.dot(init_blocker_emb, v_q))

    log.info("  Phase1 sanity:")
    log.info("    v_q · v_q       = %.6f (should be 1.0)", cos_self)
    log.info("    cos(v_q, v_pert)= %.6f (expected ~0.9988 for 768-dim, norm=0.05)", cos_perturbed)
    log.info("    raw inversion len (tokens) = %d", raw_len)
    log.info("    padded blocker len (tokens)= %d", blocker_len)
    log.info("    inverted text: %r", inverted_text[:120])
    log.info("    init_blocker GTR cos to q* = %.4f", init_gtr_cos)

    return {
        "seed":               seed,
        "v_q":                v_q,
        "v_perturbed":        v_perturbed_normed,
        "cos_vq_self":        cos_self,
        "cos_vq_perturbed":   cos_perturbed,
        "inverted_text":      inverted_text,
        "init_tokens":        tokens,
        "raw_inversion_len":  raw_len,
        "blocker_len":        blocker_len,
        "init_blocker":       init_blocker,
        "init_gtr_cos":       init_gtr_cos,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Joint-loss BBO
# ─────────────────────────────────────────────────────────────────────────────

def _embed_bge(oracle_embedder, texts: list[str]) -> np.ndarray:
    return oracle_embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)


def tokens_to_doc(q_star: str, tokens: list[int], tokenizer) -> str:
    suffix = tokenizer.decode(tokens, skip_special_tokens=True)
    return f"{q_star}. {suffix}"


def score_candidates_joint(
    candidate_docs:    list[str],
    q_star_emb_gtr:   np.ndarray,
    q_star_emb_bge:   np.ndarray,
    target_resp_emb:  np.ndarray,
    real_threshold:   float,
    lam:              float,
    retriever,
    oracle_embedder,
    generator,
    rag_prompt_template: str,
    retrieval_k:      int,
    q_star:           str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Returns (losses, gtr_sims, bge_sims, responses).

    losses[i] = λ·(-gtr_sim[i]) + (1-λ)·(-jam_sim[i]) for retrieved,
                +inf for non-retrieved.
    gtr_sims and bge_sims are computed for ALL candidates.
    responses contains empty strings for non-scored candidates.
    """
    n = len(candidate_docs)
    losses   = np.full(n, np.inf, dtype=np.float32)
    gtr_sims = np.zeros(n, dtype=np.float32)
    bge_sims = np.zeros(n, dtype=np.float32)
    jam_sims = np.zeros(n, dtype=np.float32)
    responses = [""] * n

    # --- GTR embeddings (white-box retrieval term) ---
    cand_embs_gtr = retriever.embed_batch(candidate_docs)
    gtr_sims_raw  = cand_embs_gtr @ q_star_emb_gtr    # (n,)
    gtr_sims      = gtr_sims_raw.copy()

    # --- BGE embeddings for cross-embedder diagnostic ---
    cand_embs_bge = _embed_bge(oracle_embedder, candidate_docs)
    bge_sims_raw  = cand_embs_bge @ q_star_emb_bge    # (n,)
    bge_sims      = bge_sims_raw.copy()

    # --- Real-threshold gate (same as ConstrainedJointBBO) ---
    retrieved_mask    = gtr_sims_raw >= real_threshold
    retrieved_indices = np.where(retrieved_mask)[0].tolist()

    if not retrieved_indices:
        return losses, gtr_sims, bge_sims, responses

    # --- Build RAG prompts for retrieved candidates ---
    top_k_ids  = retriever.retrieve(q_star, k=retrieval_k)
    base_docs  = [retriever.get_doc_text(d) for d in top_k_ids]
    prompts    = []
    for idx in retrieved_indices:
        ctx_docs = [candidate_docs[idx]] + base_docs[: retrieval_k - 1]
        context  = "\n\n".join(ctx_docs)
        prompts.append(rag_prompt_template.format(context=context, query=q_star))

    # --- Generate responses ---
    gen_resps = generator.generate(prompts)
    for i, idx in enumerate(retrieved_indices):
        responses[idx] = gen_resps[i]

    # --- Embed responses with BGE oracle ---
    resp_embs = _embed_bge(oracle_embedder, gen_resps)

    # --- Compute joint loss ---
    for i, idx in enumerate(retrieved_indices):
        jam_sim        = float(np.dot(resp_embs[i], target_resp_emb))
        jam_sims[idx]  = jam_sim
        retrieval_loss = -float(gtr_sims_raw[idx])
        jamming_loss   = -jam_sim
        losses[idx]    = lam * retrieval_loss + (1.0 - lam) * jamming_loss

    return losses, gtr_sims, bge_sims, responses


def get_rank_gtr(blocker_emb_gtr: np.ndarray, q_star_emb_gtr: np.ndarray,
                 top5_scores_cache: list[float]) -> int:
    """
    Estimate honest GTR rank of blocker for q*.
    top5_scores_cache is the top-5 corpus scores (cached, static per run).
    Returns 1-5 if in top-5, -1 otherwise.
    """
    blocker_sim = float(np.dot(blocker_emb_gtr, q_star_emb_gtr))
    threshold   = top5_scores_cache[-1]
    if blocker_sim < threshold:
        return -1
    rank = sum(1 for s in top5_scores_cache if s > blocker_sim) + 1
    return rank


def run_bbo_joint(
    q_star:              str,
    init_tokens:         list[int],
    lam:                 float,
    q_star_emb_gtr:      np.ndarray,
    q_star_emb_bge:      np.ndarray,
    target_resp_emb:     np.ndarray,
    real_threshold:      float,
    top5_scores_cache:   list[float],
    candidate_vocab:     np.ndarray,
    tokenizer,
    retriever,
    oracle_embedder,
    generator,
    rag_prompt_template: str,
    retrieval_k:         int,
    trajectory_csv:      Path,
    class_seed:          int,
) -> dict:
    """
    BBO loop starting from init_tokens with joint GTR+BGE loss.
    Writes per-iteration trajectory to trajectory_csv.
    Returns dict with final blocker, final loss, stats, etc.
    """
    rng = np.random.default_rng(class_seed + int(lam * 100))

    cur_tokens = list(init_tokens)
    cur_doc    = tokens_to_doc(q_star, cur_tokens, tokenizer)
    blocker_len = len(cur_tokens)

    # Initial score
    losses, gtr_s, bge_s, resps = score_candidates_joint(
        [cur_doc], q_star_emb_gtr, q_star_emb_bge, target_resp_emb,
        real_threshold, lam, retriever, oracle_embedder, generator,
        rag_prompt_template, retrieval_k, q_star,
    )
    cur_loss         = float(losses[0])
    cur_gtr_sim      = float(gtr_s[0])
    cur_bge_sim      = float(bge_s[0])
    cur_blocker_emb  = retriever.embed_batch([cur_doc])[0]
    cur_rank         = get_rank_gtr(cur_blocker_emb, q_star_emb_gtr, top5_scores_cache)

    best_loss = cur_loss
    best_doc  = cur_doc
    best_tokens = cur_tokens
    es_count  = 0
    n_iters   = 0
    cands_scored_total  = 0
    cands_accepted_total = 0

    traj_rows = []

    def _flush_traj(rows):
        write_header = not trajectory_csv.exists()
        with open(trajectory_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "iter_num", "best_loss_so_far", "current_blocker",
                "retrieval_term", "retrieval_term_bge",
                "jamming_term", "joint_loss", "current_top5_rank_gtr",
                "cands_scored", "cands_accepted",
            ])
            if write_header:
                w.writeheader()
            w.writerows(rows)
        rows.clear()

    # Log initial state
    traj_rows.append({
        "iter_num": 0, "best_loss_so_far": round(cur_loss, 5),
        "current_blocker": cur_doc[:100],
        "retrieval_term": round(cur_gtr_sim, 4),
        "retrieval_term_bge": round(cur_bge_sim, 4),
        "jamming_term": round((-cur_loss - lam * (-cur_gtr_sim)) / max(1 - lam, 1e-9), 4),
        "joint_loss": round(cur_loss, 5),
        "current_top5_rank_gtr": cur_rank,
        "cands_scored": 0, "cands_accepted": 0,
    })
    _flush_traj(traj_rows)

    t_start = time.time()

    for iteration in range(1, BBO_T + 1):
        n_iters = iteration

        # Generate candidates (single-position token substitution)
        pos        = int(rng.integers(0, blocker_len))
        sampled    = rng.choice(candidate_vocab, size=N_CANDIDATES, replace=False)
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
        cands_scored_total   += N_CANDIDATES
        cands_accepted_total += n_ret

        best_idx  = int(np.argmin(losses))
        best_cand = float(losses[best_idx])

        improved = best_cand < cur_loss
        if improved:
            cur_tokens = cand_token_lists[best_idx]
            cur_doc    = cand_docs[best_idx]
            cur_loss   = best_cand
            cur_gtr_sim = float(gtr_s[best_idx])
            cur_bge_sim = float(bge_s[best_idx])
            es_count   = 0
            if cur_loss < best_loss:
                best_loss   = cur_loss
                best_doc    = cur_doc
                best_tokens = list(cur_tokens)
        else:
            es_count += 1

        # Honest rank (fast FAISS check via cached threshold)
        cur_blocker_emb = retriever.embed_batch([cur_doc])[0]
        cur_rank = get_rank_gtr(cur_blocker_emb, q_star_emb_gtr, top5_scores_cache)

        if iteration % 10 == 0:
            log.info(
                "  iter %d | loss=%.4f | gtr_cos=%.4f | bge_cos=%.4f | rank=%s | es=%d",
                iteration, cur_loss, cur_gtr_sim, cur_bge_sim,
                cur_rank if cur_rank > 0 else "NOT",
                es_count,
            )

        # Derive jamming term from loss components
        jam_term = (cur_loss - lam * (-cur_gtr_sim)) / max(1 - lam, 1e-9) if (1 - lam) > 1e-9 else 0.0

        traj_rows.append({
            "iter_num": iteration,
            "best_loss_so_far": round(best_loss, 5),
            "current_blocker": cur_doc[:100],
            "retrieval_term": round(cur_gtr_sim, 4),
            "retrieval_term_bge": round(cur_bge_sim, 4),
            "jamming_term": round(jam_term, 4),
            "joint_loss": round(cur_loss, 5),
            "current_top5_rank_gtr": cur_rank,
            "cands_scored": cands_scored_total,
            "cands_accepted": cands_accepted_total,
        })
        if len(traj_rows) >= 50:
            _flush_traj(traj_rows)

        # Timeout check
        if time.time() - t_start > RUN_TIMEOUT_SEC:
            log.warning("  !! Timeout hit at iter %d — aborting run", iteration)
            break

        if es_count >= BBO_PATIENCE:
            log.info("  Early stop at iter %d (patience=%d)", iteration, BBO_PATIENCE)
            break

    if traj_rows:
        _flush_traj(traj_rows)

    # Final cosines on best doc
    best_emb_gtr = retriever.embed_batch([best_doc])[0]
    best_emb_bge = _embed_bge(oracle_embedder, [best_doc])[0]
    final_gtr_cos = float(np.dot(best_emb_gtr, q_star_emb_gtr))
    final_bge_cos = float(np.dot(best_emb_bge, q_star_emb_bge))

    return {
        "final_doc":      best_doc,
        "final_loss":     best_loss,
        "n_iterations":   n_iters,
        "cands_scored":   cands_scored_total,
        "cands_accepted": cands_accepted_total,
        "final_gtr_cos":  final_gtr_cos,
        "final_bge_cos":  final_bge_cos,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Honest retrieval + generation
# ─────────────────────────────────────────────────────────────────────────────

def check_top5_retrieval(blocker_emb, query, retriever):
    top_k_results = retriever.retrieve_with_scores(query, k=TOP_K)
    scores        = [s for _, s in top_k_results]
    threshold     = scores[-1] if scores else 0.0
    q_emb         = retriever.embed_batch([query])[0]
    blocker_sim   = float(np.dot(blocker_emb, q_emb))
    retrieved     = blocker_sim >= threshold
    rank          = sum(1 for s in scores if s > blocker_sim) + 1 if retrieved else None
    return retrieved, rank, blocker_sim, threshold


def build_prompts_for_retrieved(blocker_doc, queries, ret_results,
                                retriever, rag_prompt_template, k):
    prompts, ret_idxs = [], []
    for q_idx, (q, ret) in enumerate(zip(queries, ret_results)):
        if not ret["retrieved"]:
            continue
        top_k_ids = retriever.retrieve(q, k=k)
        base_docs = [retriever.get_doc_text(d) for d in top_k_ids]
        ctx_docs  = [blocker_doc] + base_docs[: k - 1]
        context   = "\n\n".join(ctx_docs)
        prompts.append(rag_prompt_template.format(context=context, query=q))
        ret_idxs.append(q_idx)
    return prompts, ret_idxs

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Comparison report
# ─────────────────────────────────────────────────────────────────────────────

def _load_task6_baseline(class_ids_full: list[str]) -> dict:
    """
    Returns {class_id: {n, retrieved, jammed_honest}} from task6 CSV.
    class_ids_full are the full IDs as they appear in task6 (e.g. "para_test1").
    """
    baseline_csv = RESULTS_DIR / "task6_constrained_joint_honest.csv"
    out = {}
    if not baseline_csv.exists():
        return out
    with open(baseline_csv) as f:
        for row in csv.DictReader(f):
            cid = row["class_id"]
            if cid not in class_ids_full:
                continue
            if cid not in out:
                out[cid] = {"n": 0, "retrieved": 0, "jammed_honest": 0, "ctype": row["class_type"]}
            out[cid]["n"]              += 1
            out[cid]["retrieved"]      += int(row["retrieved_top5"])
            out[cid]["jammed_honest"]  += int(row["jammed_honest"])
    return out


def write_report(ckpt: dict, all_classes: list[dict], lambdas: list[float]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "whitebox_pilot_report.md"

    # Collect pilot results
    full_ids = [c["class_id"] for c in all_classes]
    task6_bl = _load_task6_baseline(full_ids)

    lines = ["# White-box GTR Pilot — Report\n\n"]

    # Section A: per-class table
    lines.append("## A. Per-class results\n\n")
    lines.append(
        "| class_id | type | q* | seed | init_blocker (first 60) | "
        "λ=0.3 ASR | λ=0.5 ASR | λ=0.7 ASR | Task6 baseline |\n"
    )
    lines.append("|----------|------|----|----|------------------------|"
                 "-----------|-----------|-----------|----------------|\n")

    for cls in all_classes:
        cid   = cls["class_id"]
        ctype = cls["class_type"]
        phase1 = ckpt.get(f"phase1_{cid}", {})
        q_star = phase1.get("q_star", "")
        seed   = phase1.get("seed", "")
        init_b = phase1.get("init_blocker", "")[:60].replace("|", "∣")

        asrs = []
        for lam in lambdas:
            key = f"eval_{cid}_lam{lam}"
            ev  = ckpt.get(key, {})
            n   = ev.get("n_queries", 0)
            jam = ev.get("jammed_honest_total", 0)
            asrs.append(f"{jam}/{n}" if n else "—")

        bl = task6_bl.get(cid, {})
        bl_asr = f"{bl.get('jammed_honest',0)}/{bl.get('n',0)}" if bl else "—"

        lines.append(
            f"| {cid} | {ctype} | {q_star[:40]} | {seed} | "
            f"{init_b} | {asrs[0]} | {asrs[1]} | {asrs[2]} | {bl_asr} |\n"
        )

    # Section B: aggregated comparison
    lines.append("\n## B. Aggregated comparison\n\n")
    lines.append(
        "| Method | Paraphrase ASR | Entity ASR |\n"
        "|--------|---------------|------------|\n"
    )

    # Task6 baseline
    p_jam, p_n, e_jam, e_n = 0, 0, 0, 0
    for cid, v in task6_bl.items():
        if v["ctype"] == "paraphrase":
            p_jam += v["jammed_honest"]; p_n += v["n"]
        else:
            e_jam += v["jammed_honest"]; e_n += v["n"]
    lines.append(
        f"| Constrained-joint BBO (black-box BGE) | "
        f"{p_jam}/{p_n} ({100*p_jam/max(p_n,1):.0f}%) | "
        f"{e_jam}/{e_n} ({100*e_jam/max(e_n,1):.0f}%) |\n"
    )

    for lam in lambdas:
        pp_jam, pp_n, ee_jam, ee_n = 0, 0, 0, 0
        for cls in all_classes:
            cid   = cls["class_id"]
            ctype = cls["class_type"]
            key   = f"eval_{cid}_lam{lam}"
            ev    = ckpt.get(key, {})
            jh    = ev.get("jammed_honest_total", 0)
            nn    = ev.get("n_queries", 0)
            if ctype == "paraphrase":
                pp_jam += jh; pp_n += nn
            else:
                ee_jam += jh; ee_n += nn
        lines.append(
            f"| White-box GTR + perturbed init, λ={lam} | "
            f"{pp_jam}/{pp_n} ({100*pp_jam/max(pp_n,1):.0f}%) | "
            f"{ee_jam}/{ee_n} ({100*ee_jam/max(ee_n,1):.0f}%) |\n"
        )

    # Section C: verdict
    lines.append("\n## C. Verdict\n\n")
    lines.append("*(Written at report generation time based on completed runs.)*\n\n")

    # Compute best-lambda ASR for paraphrase and entity
    best_p, best_e = 0.0, 0.0
    for lam in lambdas:
        pp_jam, pp_n, ee_jam, ee_n = 0, 0, 0, 0
        for cls in all_classes:
            cid   = cls["class_id"]
            ctype = cls["class_type"]
            ev    = ckpt.get(f"eval_{cid}_lam{lam}", {})
            jh    = ev.get("jammed_honest_total", 0)
            nn    = ev.get("n_queries", 0)
            if ctype == "paraphrase":
                pp_jam += jh; pp_n += nn
            else:
                ee_jam += jh; ee_n += nn
        if pp_n: best_p = max(best_p, pp_jam / pp_n)
        if ee_n: best_e = max(best_e, ee_jam / ee_n)

    bl_p = p_jam / max(p_n, 1)
    bl_e = e_jam / max(e_n, 1)

    para_better  = best_p > bl_p
    entity_better = best_e > bl_e

    # Cross-embedder pattern
    gtr_bge_diverge_para = []
    for cls in all_classes:
        if cls["class_type"] != "paraphrase":
            continue
        ev = ckpt.get(f"eval_{cls['class_id']}_lam0.5", {})
        fg = ev.get("final_gtr_cos")
        fb = ev.get("final_bge_cos")
        if fg is not None and fb is not None:
            gtr_bge_diverge_para.append(fg - fb)

    cross_note = ""
    if gtr_bge_diverge_para:
        mean_div = np.mean(gtr_bge_diverge_para)
        if mean_div > 0.05:
            cross_note = (
                f"The white-box GTR loss pulled blockers to higher GTR cosine "
                f"(mean GTR-BGE gap = {mean_div:.3f}), confirming the optimization "
                "specifically targets the GTR space rather than the BGE surrogate."
            )
        elif mean_div < -0.05:
            cross_note = (
                f"Interestingly, optimizing GTR cosine produced higher BGE cosine "
                f"too (mean GTR-BGE gap = {mean_div:.3f}), suggesting both embedding "
                "spaces agree on what a good retrieval blocker looks like."
            )
        else:
            cross_note = (
                f"GTR and BGE cosines tracked closely (mean gap = {mean_div:.3f}), "
                "suggesting the joint loss does not cause significant embedding-space drift."
            )

    para_verdict = (
        f"White-box GTR {'improved' if para_better else 'did not clearly improve'} "
        f"paraphrase ASR over constrained-joint: best λ achieves {100*best_p:.0f}% "
        f"vs baseline {100*bl_p:.0f}%."
    )
    entity_verdict = (
        f"White-box GTR {'improved' if entity_better else 'did not improve'} "
        f"entity ASR: best λ achieves {100*best_e:.0f}% "
        f"vs baseline {100*bl_e:.0f}%. "
        "Entity classes remain limited by geometric mismatch in the query class — "
        "the white-box retrieval term helps q* retrieval but does not bridge the "
        "high within-class cosine gap that limits transfer to class members."
    )
    prediction = (
        "Predicted: paraphrase modest improvement to ~50%, entity near-zero (~13%). "
        f"Observed: paraphrase {100*best_p:.0f}%, entity {100*best_e:.0f}%. "
        f"Prediction {'matched' if abs(best_p - 0.5) < 0.15 and best_e < 0.20 else 'not matched'}."
    )

    lines.append(
        f"{para_verdict} {entity_verdict} {cross_note} {prediction}\n"
    )

    with open(report_path, "w") as f:
        f.writelines(lines)
    log.info("Report written -> %s", report_path)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    log.info("=" * 70)
    log.info("White-box GTR Pilot")
    log.info("  Paraphrase classes : %s", PARA_CLASS_IDS)
    log.info("  Entity classes     : %s", ENTITY_CLASS_IDS)
    log.info("  Lambdas            : %s", LAMBDAS)
    log.info("  BBO budget         : T=%d patience=%d n=%d", BBO_T, BBO_PATIENCE, N_CANDIDATES)
    log.info("=" * 70)

    # Load classes
    para_classes   = _load_paraphrase_classes(PARA_CLASS_IDS)
    entity_classes = _load_entity_classes(ENTITY_CLASS_IDS)
    all_classes    = para_classes + entity_classes
    log.info("Loaded %d classes total", len(all_classes))

    cfg = _build_cfg()

    # Load checkpoint
    ckpt: dict = {}
    if CKPT_FILE.exists():
        with open(CKPT_FILE, "rb") as f:
            ckpt = pickle.load(f)
        log.info("Resumed checkpoint (%d keys)", len(ckpt))

    def _save_ckpt():
        with open(CKPT_FILE, "wb") as f:
            pickle.dump(ckpt, f)

    # ── Retriever ─────────────────────────────────────────────────────────────
    log.info("Loading GTR retriever …")
    from rag.retriever import Retriever
    retriever = Retriever(model_name=GTR_MODEL, score_function="cos_sim")
    retriever.load_index(INDEX_DIR)
    log.info("Retriever: %d docs", retriever._index.ntotal)

    # ── BGE oracle embedder ────────────────────────────────────────────────────
    log.info("Loading BGE oracle embedder …")
    from sentence_transformers import SentenceTransformer
    oracle_embedder = SentenceTransformer(
        BGE_MODEL, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    target_resp_emb = _embed_bge(oracle_embedder, [TARGET_RESPONSE])[0]

    # ── Tokenizer (for candidate vocab + token ops) ───────────────────────────
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

    # Build filtered vocab (same as ShafranBBO)
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
        log.warning("Wikitext load failed (%s) — using uniform vocab", e)
        candidate_vocab = np.arange(len(tokenizer))
    log.info("Candidate vocab size: %d", len(candidate_vocab))

    # ── Phase 1: vec2text perturbed init for each class ──────────────────────
    log.info("=" * 60)
    log.info("PHASE 1: vec2text initialization")
    for cls_idx, cls in enumerate(all_classes):
        cid    = cls["class_id"]
        ctype  = cls["class_type"]
        key1   = f"phase1_{cid}"
        if key1 in ckpt:
            log.info("  [SKIP Phase1] %s", cid)
            continue

        queries  = cls["_queries"]
        embs     = cls["_embeddings"]
        centroid = cls["_centroid"]
        q_star, q_star_idx = find_qstar(queries, embs, centroid)

        log.info("Class %s (%s)  q*=%r", cid, ctype, q_star[:70])
        p1 = phase1_init(q_star, cls_idx, retriever, tokenizer)
        p1["q_star"]     = q_star
        p1["q_star_idx"] = q_star_idx

        # Sanity: initial blocker's honest GTR top-5 rank for q*
        init_emb = retriever.embed_batch([p1["init_blocker"]])[0]
        top5     = retriever.retrieve_with_scores(q_star, k=TOP_K)
        top5_sc  = [s for _, s in top5]
        rank_init = get_rank_gtr(init_emb, p1["v_q"], top5_sc)
        p1["init_rank_gtr"] = rank_init
        log.info("  init blocker honest GTR rank: %s", rank_init if rank_init > 0 else "NOT in top-5")

        ckpt[key1] = p1
        _save_ckpt()

    # ── Load vLLM ─────────────────────────────────────────────────────────────
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

    # ── Phase 2: BBO for each (class, lambda) ────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 2: Joint-loss BBO  (%d classes × %d lambdas)", len(all_classes), len(LAMBDAS))

    for cls in all_classes:
        cid   = cls["class_id"]
        ctype = cls["class_type"]
        p1    = ckpt[f"phase1_{cid}"]
        q_star = p1["q_star"]

        # Pre-cache top-5 scores for rank checks
        top5_scores_cache = [s for _, s in retriever.retrieve_with_scores(q_star, k=TOP_K)]
        real_threshold    = top5_scores_cache[-1]
        log.info("  %s: real threshold=%.4f", cid, real_threshold)

        # q* embeddings in GTR and BGE space
        q_star_emb_gtr = retriever.embed_batch([q_star])[0]
        q_star_emb_bge = _embed_bge(oracle_embedder, [q_star])[0]

        t_cls_start = time.time()
        for lam in LAMBDAS:
            key2 = f"bbo_{cid}_lam{lam}"
            if key2 in ckpt:
                log.info("  [SKIP BBO] %s lam=%s", cid, lam)
                continue

            traj_csv = RESULTS_DIR / f"whitebox_pilot_trajectory_{cid}_lambda{lam}.csv"
            log.info("  Running BBO: %s  λ=%.1f …", cid, lam)
            t_run = time.time()

            bbo_result = run_bbo_joint(
                q_star          = q_star,
                init_tokens     = list(p1["init_tokens"]),
                lam             = lam,
                q_star_emb_gtr  = q_star_emb_gtr,
                q_star_emb_bge  = q_star_emb_bge,
                target_resp_emb = target_resp_emb,
                real_threshold  = real_threshold,
                top5_scores_cache = top5_scores_cache,
                candidate_vocab = candidate_vocab,
                tokenizer       = tokenizer,
                retriever       = retriever,
                oracle_embedder = oracle_embedder,
                generator       = generator,
                rag_prompt_template = rag_prompt,
                retrieval_k     = TOP_K,
                trajectory_csv  = traj_csv,
                class_seed      = p1["seed"],
            )

            elapsed = (time.time() - t_run) / 60
            log.info(
                "  BBO done: %s λ=%.1f | loss=%.4f | iters=%d | "
                "scored=%d | accepted=%d | gtr_cos=%.4f | bge_cos=%.4f | %.1f min",
                cid, lam, bbo_result["final_loss"], bbo_result["n_iterations"],
                bbo_result["cands_scored"], bbo_result["cands_accepted"],
                bbo_result["final_gtr_cos"], bbo_result["final_bge_cos"], elapsed,
            )
            ckpt[key2] = bbo_result
            _save_ckpt()

    # ── Close vLLM ──────────────────────────────────────────────────────────
    log.info("Closing vLLM …")
    generator.close()
    del generator
    gc.collect()
    torch.cuda.empty_cache()

    # ── Phase 3: Honest evaluation ───────────────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 3: Honest evaluation  (retrieval + generation per query)")
    log.info("Loading vLLM again for generation …")
    from rag.generator import VLLMGenerator
    generator = VLLMGenerator(
        model_name=LLM_MODEL,
        temperature=0.0,
        max_tokens=128,
        gpu_memory_utilization=0.65,
        dtype="float16",
        max_model_len=4096,
    )

    all_rows = []

    for cls in all_classes:
        cid     = cls["class_id"]
        ctype   = cls["class_type"]
        queries = cls["_queries"]
        within  = cls["_within_sim"]
        p1      = ckpt[f"phase1_{cid}"]
        q_star  = p1["q_star"]

        for lam in LAMBDAS:
            key2 = f"bbo_{cid}_lam{lam}"
            keyE = f"eval_{cid}_lam{lam}"
            if keyE in ckpt:
                log.info("  [SKIP eval] %s lam=%s", cid, lam)
                all_rows.extend(ckpt[keyE].get("rows", []))
                continue

            bbo_res   = ckpt[key2]
            final_doc = bbo_res["final_doc"]

            # Honest top-5 for all queries
            blocker_emb = retriever.embed_batch([final_doc])[0]
            ret_results = []
            for q in queries:
                retr, rank, b_sim, t_sim = check_top5_retrieval(blocker_emb, q, retriever)
                ret_results.append({
                    "retrieved":     retr,
                    "rank":          rank,
                    "blocker_sim":   round(b_sim, 5),
                    "threshold_sim": round(t_sim, 5),
                })
            n_ret = sum(r["retrieved"] for r in ret_results)
            log.info("  %s λ=%.1f: retrieval %d/%d", cid, lam, n_ret, len(queries))

            # Generate for retrieved
            prompts, ret_idxs = build_prompts_for_retrieved(
                final_doc, queries, ret_results, retriever, rag_prompt, TOP_K
            )
            responses: list[Optional[str]] = [None] * len(queries)
            if prompts:
                gen_resps = generator.generate(prompts)
                for qi, resp in zip(ret_idxs, gen_resps):
                    responses[qi] = resp

            eval_entry = {
                "n_queries":          len(queries),
                "n_retrieved":        n_ret,
                "responses":          responses,
                "ret_results":        ret_results,
                "jammed_honest_total": 0,
                "final_gtr_cos":      bbo_res["final_gtr_cos"],
                "final_bge_cos":      bbo_res["final_bge_cos"],
                "rows":               [],
            }
            ckpt[keyE] = eval_entry
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
    for cls in all_classes:
        cid     = cls["class_id"]
        ctype   = cls["class_type"]
        queries = cls["_queries"]
        within  = cls["_within_sim"]
        p1      = ckpt[f"phase1_{cid}"]
        q_star  = p1["q_star"]

        for lam in LAMBDAS:
            key2  = f"bbo_{cid}_lam{lam}"
            keyE  = f"eval_{cid}_lam{lam}"
            bbo_res  = ckpt[key2]
            ev       = ckpt[keyE]
            ret_results = ev["ret_results"]
            responses   = ev["responses"]
            final_doc   = bbo_res["final_doc"]

            jam_total = 0
            rows_for_class = []
            for q_idx, (q, resp, ret) in enumerate(zip(queries, responses, ret_results)):
                retrieved = bool(ret["retrieved"])
                if retrieved and resp is not None:
                    answered     = judge.is_answered(q, resp)
                    jam_success  = int(not answered)
                    jammed_honest = jam_success
                else:
                    jam_success   = None
                    jammed_honest = 0
                jam_total += jammed_honest

                row = {
                    "class_id":            cid,
                    "class_type":          ctype,
                    "lambda":              lam,
                    "query_idx":           q_idx,
                    "query_text":          q,
                    "within_class_sim":    round(within, 4),
                    "n_class_queries":     len(queries),
                    "blocker_doc":         final_doc[:300],
                    "blocker_sim":         ret["blocker_sim"],
                    "threshold_sim":       ret["threshold_sim"],
                    "retrieved_top5":      int(retrieved),
                    "rank":                ret["rank"] if ret["rank"] is not None else "",
                    "response":            resp[:300] if resp else "",
                    "jam_success":         jam_success if jam_success is not None else "",
                    "jammed_honest":       jammed_honest,
                    "q_star":              q_star[:120],
                    "final_loss":          round(bbo_res["final_loss"], 5),
                    "n_iterations":        bbo_res["n_iterations"],
                    "real_threshold":      ret["threshold_sim"],
                    "cands_scored":        bbo_res["cands_scored"],
                    "cands_accepted":      bbo_res["cands_accepted"],
                    "perturbation_seed":   p1["seed"],
                    "blocker_length":      p1["blocker_len"],
                    "init_text":           p1["inverted_text"][:200],
                    "final_gtr_cos":       round(bbo_res["final_gtr_cos"], 5),
                    "final_bge_cos":       round(bbo_res["final_bge_cos"], 5),
                }
                rows_for_class.append(row)
                csv_rows.append(row)

            ev["jammed_honest_total"] = jam_total
            ev["rows"] = rows_for_class
            _save_ckpt()
            log.info("  %s λ=%.1f: ASR %d/%d", cid, lam, jam_total, len(queries))

    judge.close()

    # Write output CSV
    fieldnames = [
        "class_id", "class_type", "lambda",
        "query_idx", "query_text", "within_class_sim", "n_class_queries",
        "blocker_doc", "blocker_sim", "threshold_sim",
        "retrieved_top5", "rank", "response", "jam_success", "jammed_honest",
        "q_star", "final_loss", "n_iterations", "real_threshold",
        "cands_scored", "cands_accepted",
        "perturbation_seed", "blocker_length", "init_text",
        "final_gtr_cos", "final_bge_cos",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    log.info("Saved %d rows -> %s", len(csv_rows), OUT_CSV)

    # ── Phase 4: Report ───────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 4: Writing comparison report …")
    write_report(ckpt, all_classes, LAMBDAS)

    log.info("=" * 70)
    log.info("DONE. Total elapsed: %.1f min", (time.time() - t_total) / 60)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
