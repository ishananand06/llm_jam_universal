"""
Experiment 1: Paraphrase scale-up for white-box GTR attack.

Runs white-box GTR + vec2text-perturbed init (λ=0.5 only, the pilot-optimal
value) on 10 NEW paraphrase classes that were NOT in the original 3-class
white-box pilot (test1, test2, test6).  Classes are drawn from indices 3–12
of paraphrase_classes.json (test9 through test22).

For each class we also run the constrained-joint BBO black-box baseline
(same budget, same q*) so the comparison is apples-to-apples within this
script — no dependency on the old task6 CSV.

Pipeline (mirrors eval_whitebox_gtr_pilot.py exactly):
  Phase 1  — vec2text inversion of q* + Gaussian perturbation in GTR space
  Phase 2a — White-box GTR joint-loss BBO at λ=0.5
  Phase 2b — Black-box ConstrainedJointBBO baseline on the same q*
  Phase 3  — Honest top-5 retrieval + LLM generation (both methods)
  Phase 4  — Gemma judge + comparison report

Output
------
/home/ishana/scratch/results/whitebox_para_scaleup_honest.csv
/home/ishana/scratch/results/whitebox_para_scaleup_ckpt.pkl
/home/ishana/scratch/results/whitebox_para_scaleup_report.md
/home/ishana/scratch/results/whitebox_para_scaleup_trajectory_<class>_wb.csv

Run detached in tmux:
  tmux new-session -d -s wb_scaleup \\
    '.venv/bin/python src/experiments/eval_whitebox_para_scaleup.py \
       > /home/ishana/scratch/results/whitebox_para_scaleup.log 2>&1'
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
OUT_CSV     = RESULTS_DIR / "whitebox_para_scaleup_honest.csv"
CKPT_FILE   = RESULTS_DIR / "whitebox_para_scaleup_ckpt.pkl"

GTR_MODEL      = "sentence-transformers/gtr-t5-base"
BGE_MODEL      = "BAAI/bge-large-en-v1.5"
LLM_MODEL      = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL    = "google/gemma-2-9b-it"
VEC2TEXT_MODEL = "jxm/gtr__nq__32"

TOP_K          = 5
BBO_T          = 500
BBO_PATIENCE   = 50
N_CANDIDATES   = 50
LLM_BATCH_SIZE = 32
MIN_TOKENS     = 50
PERTURB_NORM   = 0.05
LAM            = 0.5          # pilot-optimal λ; no sweep needed here
RUN_TIMEOUT    = 3600         # 60 min per run

# 10 new paraphrase classes (indices 3-12 in JSON, i.e. never run in the pilot)
PARA_CLASS_IDS = ["test9", "test10", "test11", "test12", "test13",
                  "test14", "test16", "test17", "test19", "test20"]

TARGET_RESPONSE = (
    "I don't know. The context does not provide enough information to "
    "answer the query."
)
FILLER_TOKEN = "!"


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def _build_cfg():
    cfg_dir = Path(__file__).resolve().parents[2] / "configs"
    base = OmegaConf.load(cfg_dir / "base.yaml")
    base.attack.num_iterations   = BBO_T
    base.attack.es_patience      = BBO_PATIENCE
    base.attack.batch_size       = N_CANDIDATES
    base.attack.llm_batch_size   = LLM_BATCH_SIZE
    base.vllm.gpu_memory_utilization = 0.65
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Class loading
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


def find_qstar(queries, embs, centroid):
    dists = np.linalg.norm(embs - centroid[np.newaxis, :], axis=1)
    return queries[int(np.argmin(dists))], int(np.argmin(dists))


# ─────────────────────────────────────────────────────────────────────────────
# BGE helper
# ─────────────────────────────────────────────────────────────────────────────

def _embed_bge(oracle_embedder, texts: list[str]) -> np.ndarray:
    return oracle_embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: vec2text perturbed initialization (identical to pilot)
# ─────────────────────────────────────────────────────────────────────────────

def phase1_init(q_star: str, class_idx: int, retriever, tokenizer) -> dict:
    import vec2text
    import transformers.modeling_utils as _tmu

    seed = 42 + class_idx
    rng  = np.random.default_rng(seed)

    v_q = retriever.embed_batch([q_star])[0]

    epsilon = rng.standard_normal(v_q.shape).astype(np.float32)
    epsilon = epsilon / float(np.linalg.norm(epsilon)) * PERTURB_NORM
    v_perturbed = v_q + epsilon
    v_norm = float(np.linalg.norm(v_perturbed))
    v_perturbed_normed = v_perturbed / max(v_norm, 1e-9)
    cos_perturbed = float(np.dot(v_q, v_perturbed_normed))

    # Patch torch.load safety check (jxm model uses .bin weights)
    _orig_check = _tmu.check_torch_load_is_safe
    _tmu.check_torch_load_is_safe = lambda: None
    log.info("  Phase1: loading vec2text corrector …")
    try:
        corrector = vec2text.load_pretrained_corrector("gtr-base")
    finally:
        _tmu.check_torch_load_is_safe = _orig_check

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

    tokens = tokenizer.encode(inverted_text, add_special_tokens=False)
    raw_len = len(tokens)
    pad_tok = tokenizer.encode(FILLER_TOKEN, add_special_tokens=False)[0]
    if len(tokens) < MIN_TOKENS:
        tokens = tokens + [pad_tok] * (MIN_TOKENS - len(tokens))

    suffix = tokenizer.decode(tokens, skip_special_tokens=True)
    init_blocker = f"{q_star}. {suffix}"
    init_blocker_emb = retriever.embed_batch([init_blocker])[0]
    init_gtr_cos = float(np.dot(init_blocker_emb, v_q))

    log.info("  cos(v_q, v_pert)=%.4f | raw_len=%d | init_gtr_cos=%.4f",
             cos_perturbed, raw_len, init_gtr_cos)
    log.info("  inverted: %r", inverted_text[:100])

    return {
        "seed": seed, "v_q": v_q, "v_perturbed": v_perturbed_normed,
        "cos_vq_perturbed": cos_perturbed, "inverted_text": inverted_text,
        "init_tokens": tokens, "raw_inversion_len": raw_len,
        "blocker_len": len(tokens), "init_blocker": init_blocker,
        "init_gtr_cos": init_gtr_cos,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2a: White-box joint-loss BBO
# ─────────────────────────────────────────────────────────────────────────────

def tokens_to_doc(q_star: str, tokens: list[int], tokenizer) -> str:
    return f"{q_star}. {tokenizer.decode(tokens, skip_special_tokens=True)}"


def score_candidates_joint(
    candidate_docs, q_star_emb_gtr, q_star_emb_bge, target_resp_emb,
    real_threshold, lam, retriever, oracle_embedder, generator,
    rag_prompt_template, retrieval_k, q_star,
):
    n = len(candidate_docs)
    losses   = np.full(n, np.inf, dtype=np.float32)
    gtr_sims = np.zeros(n, dtype=np.float32)
    bge_sims = np.zeros(n, dtype=np.float32)
    responses = [""] * n

    cand_embs_gtr = retriever.embed_batch(candidate_docs)
    gtr_sims_raw  = cand_embs_gtr @ q_star_emb_gtr
    gtr_sims      = gtr_sims_raw.copy()

    cand_embs_bge = _embed_bge(oracle_embedder, candidate_docs)
    bge_sims      = (cand_embs_bge @ q_star_emb_bge).copy()

    retrieved_mask    = gtr_sims_raw >= real_threshold
    retrieved_indices = np.where(retrieved_mask)[0].tolist()

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
    threshold   = top5_scores_cache[-1]
    if blocker_sim < threshold:
        return -1
    return sum(1 for s in top5_scores_cache if s > blocker_sim) + 1


def run_bbo_whitebox(
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
    cands_scored_total   = 0
    cands_accepted_total = 0

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
        cands_scored_total   += N_CANDIDATES
        cands_accepted_total += n_ret

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
            "cands_scored": cands_scored_total,
            "cands_accepted": cands_accepted_total,
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
    final_gtr_cos = float(np.dot(best_emb_gtr, q_star_emb_gtr))
    final_bge_cos = float(np.dot(best_emb_bge, q_star_emb_bge))

    return {
        "final_doc": best_doc, "final_loss": best_loss,
        "n_iterations": n_iters, "cands_scored": cands_scored_total,
        "cands_accepted": cands_accepted_total,
        "final_gtr_cos": final_gtr_cos, "final_bge_cos": final_bge_cos,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Honest retrieval check
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
        prompts.append(rag_prompt_template.format(
            context="\n\n".join(ctx_docs), query=q))
        ret_idxs.append(q_idx)
    return prompts, ret_idxs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    log.info("=" * 70)
    log.info("Experiment 1: White-box GTR Paraphrase Scale-up")
    log.info("  Classes  : %s", PARA_CLASS_IDS)
    log.info("  Lambda   : %.1f (pilot-optimal)", LAM)
    log.info("  BBO T=%d patience=%d n=%d", BBO_T, BBO_PATIENCE, N_CANDIDATES)
    log.info("=" * 70)

    classes = _load_paraphrase_classes(PARA_CLASS_IDS)
    log.info("Loaded %d classes", len(classes))

    cfg = _build_cfg()

    # Checkpoint
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
        log.warning("Wikitext load failed (%s) — uniform vocab", e)
        candidate_vocab = np.arange(len(tokenizer))
    log.info("Candidate vocab: %d tokens", len(candidate_vocab))

    # ── Phase 1: vec2text init for each class ────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 1: vec2text initialization")
    for cls_idx, cls in enumerate(classes):
        cid  = cls["class_id"]
        key1 = f"phase1_{cid}"
        if key1 in ckpt:
            log.info("  [SKIP] %s", cid)
            continue
        queries, embs, centroid = cls["_queries"], cls["_embeddings"], cls["_centroid"]
        q_star, q_star_idx = find_qstar(queries, embs, centroid)
        log.info("Class %s | q*=%r", cid, q_star[:70])
        p1 = phase1_init(q_star, cls_idx, retriever, tokenizer)
        p1["q_star"]     = q_star
        p1["q_star_idx"] = q_star_idx
        init_emb = retriever.embed_batch([p1["init_blocker"]])[0]
        top5     = retriever.retrieve_with_scores(q_star, k=TOP_K)
        top5_sc  = [s for _, s in top5]
        p1["init_rank_gtr"] = get_rank_gtr(init_emb, p1["v_q"], top5_sc)
        log.info("  init rank GTR: %s", p1["init_rank_gtr"] if p1["init_rank_gtr"] > 0 else "NOT top-5")
        ckpt[key1] = p1
        _save_ckpt()

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

    # ── Phase 2a: White-box BBO ───────────────────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 2a: White-box GTR joint-loss BBO  (λ=%.1f)", LAM)
    for cls in classes:
        cid    = cls["class_id"]
        key_wb = f"bbo_wb_{cid}"
        if key_wb in ckpt:
            log.info("  [SKIP] %s (white-box BBO)", cid)
            continue
        p1     = ckpt[f"phase1_{cid}"]
        q_star = p1["q_star"]
        top5_scores_cache = [s for _, s in retriever.retrieve_with_scores(q_star, k=TOP_K)]
        real_threshold    = top5_scores_cache[-1]
        q_star_emb_gtr    = retriever.embed_batch([q_star])[0]
        q_star_emb_bge    = _embed_bge(oracle_embedder, [q_star])[0]

        log.info("  %s | real_thr=%.4f", cid, real_threshold)
        traj_csv = RESULTS_DIR / f"whitebox_para_scaleup_trajectory_{cid}_wb.csv"
        t0 = time.time()
        bbo_res = run_bbo_whitebox(
            q_star=q_star, init_tokens=list(p1["init_tokens"]), lam=LAM,
            q_star_emb_gtr=q_star_emb_gtr, q_star_emb_bge=q_star_emb_bge,
            target_resp_emb=target_resp_emb, real_threshold=real_threshold,
            top5_scores_cache=top5_scores_cache, candidate_vocab=candidate_vocab,
            tokenizer=tokenizer, retriever=retriever, oracle_embedder=oracle_embedder,
            generator=generator, rag_prompt_template=rag_prompt,
            retrieval_k=TOP_K, trajectory_csv=traj_csv, class_seed=p1["seed"],
        )
        log.info("  WB BBO done %.1f min | loss=%.4f | gtr_cos=%.4f | bge_cos=%.4f",
                 (time.time() - t0) / 60, bbo_res["final_loss"],
                 bbo_res["final_gtr_cos"], bbo_res["final_bge_cos"])
        ckpt[key_wb] = bbo_res
        _save_ckpt()

    # ── Phase 2b: Constrained-joint BBO baseline (black-box) ─────────────────
    log.info("=" * 60)
    log.info("PHASE 2b: Constrained-joint BBO black-box baseline")
    from utils.gpu_manager import GPUManager
    from attacks.constrained_joint_bbo import ConstrainedJointBBO
    gpu_manager = GPUManager()
    attacker    = ConstrainedJointBBO(cfg, retriever, generator, gpu_manager)

    for cls in classes:
        cid    = cls["class_id"]
        key_bb = f"bbo_bb_{cid}"
        if key_bb in ckpt:
            log.info("  [SKIP] %s (black-box BBO)", cid)
            continue
        p1     = ckpt[f"phase1_{cid}"]
        q_star = p1["q_star"]

        log.info("  %s  q*=%r", cid, q_star[:60])
        real_thr = attacker.precompute_retrieval_threshold(q_star)
        t0 = time.time()
        result = attacker.run(query=q_star)
        log.info("  BB BBO done %.1f min | loss=%.4f | iters=%d",
                 (time.time() - t0) / 60, result.final_loss, result.n_iterations)
        ckpt[key_bb] = {
            "final_doc":      result.final_doc,
            "final_loss":     result.final_loss,
            "n_iterations":   result.n_iterations,
            "real_threshold": real_thr,
            "cands_scored":   int(attacker.n_candidates_scored),
            "cands_rejected": int(attacker.n_rejected_by_constraint),
            "cands_accepted": int(attacker.n_accepted_by_constraint),
        }
        _save_ckpt()

    # ── Close vLLM ───────────────────────────────────────────────────────────
    log.info("Closing vLLM …")
    generator.close()
    del generator, attacker
    gc.collect()
    torch.cuda.empty_cache()

    # ── Phase 3: Honest retrieval + generation (both methods) ────────────────
    log.info("=" * 60)
    log.info("PHASE 3: Honest evaluation (retrieval + generation)")
    from rag.generator import VLLMGenerator
    generator = VLLMGenerator(
        model_name=LLM_MODEL, temperature=0.0, max_tokens=128,
        gpu_memory_utilization=0.65, dtype="float16", max_model_len=4096,
    )

    for method in ("wb", "bb"):
        for cls in classes:
            cid    = cls["class_id"]
            keyE   = f"eval_{method}_{cid}"
            key_bbo = f"bbo_{method}_{cid}"
            if keyE in ckpt:
                log.info("  [SKIP] %s (%s eval)", cid, method)
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
            log.info("  %s [%s]: retrieval %d/%d", cid, method, n_ret, len(queries))

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
    summary: dict[str, dict] = {}

    for method in ("wb", "bb"):
        for cls in classes:
            cid     = cls["class_id"]
            ctype   = cls["class_type"]
            queries = cls["_queries"]
            within  = cls["_within_sim"]
            p1      = ckpt[f"phase1_{cid}"]
            q_star  = p1["q_star"]
            key_bbo = f"bbo_{method}_{cid}"
            keyE    = f"eval_{method}_{cid}"
            bbo_res = ckpt[key_bbo]
            ev      = ckpt[keyE]

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

                row = {
                    "class_id":         cid,
                    "class_type":       ctype,
                    "method":           method,
                    "lambda":           LAM if method == "wb" else "bb_baseline",
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
                    "cands_scored":     bbo_res.get("cands_scored", ""),
                    "cands_accepted":   bbo_res.get("cands_accepted", ""),
                    "final_gtr_cos":    round(bbo_res.get("final_gtr_cos", 0), 5),
                    "final_bge_cos":    round(bbo_res.get("final_bge_cos", 0), 5),
                    "init_text":        p1["inverted_text"][:200] if method == "wb" else "random_exclamation",
                    "perturbation_seed": p1["seed"] if method == "wb" else "",
                }
                csv_rows.append(row)

            key_sum = f"{method}_{cid}"
            summary[key_sum] = {"n": len(queries), "jammed_honest": jam_total, "method": method}
            log.info("  %s [%s]: ASR %d/%d", cid, method, jam_total, len(queries))

    judge.close()

    # Write CSV
    fieldnames = [
        "class_id", "class_type", "method", "lambda",
        "query_idx", "query_text", "within_class_sim", "n_class_queries",
        "blocker_doc", "blocker_sim", "threshold_sim",
        "retrieved_top5", "rank", "response", "jam_success", "jammed_honest",
        "q_star", "final_loss", "n_iterations",
        "cands_scored", "cands_accepted", "final_gtr_cos", "final_bge_cos",
        "init_text", "perturbation_seed",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    log.info("Saved %d rows -> %s", len(csv_rows), OUT_CSV)

    # ── Report ────────────────────────────────────────────────────────────────
    _write_report(classes, summary)

    log.info("=" * 70)
    log.info("DONE. Total elapsed: %.1f min", (time.time() - t_total) / 60)
    log.info("=" * 70)


def _write_report(classes, summary):
    report_path = RESULTS_DIR / "whitebox_para_scaleup_report.md"
    lines = ["# Experiment 1: White-box GTR Paraphrase Scale-up Report\n\n"]

    lines.append(
        f"**Classes**: {[c['class_id'] for c in classes]}  \n"
        f"**λ** = {LAM} (pilot-optimal)  \n"
        f"**Budget**: T={BBO_T}, patience={BBO_PATIENCE}, n={N_CANDIDATES}  \n\n"
    )

    lines.append("## Per-class results\n\n")
    lines.append(
        "| class_id | within_sim | WB λ=0.5 ASR | BB baseline ASR |\n"
        "|----------|-----------|-------------|----------------|\n"
    )

    wb_jam, wb_n, bb_jam, bb_n = 0, 0, 0, 0
    for cls in classes:
        cid    = cls["class_id"]
        within = round(cls["_within_sim"], 4)
        wb     = summary.get(f"wb_{cid}", {})
        bb     = summary.get(f"bb_{cid}", {})
        wb_str = f"{wb.get('jammed_honest',0)}/{wb.get('n',0)}"
        bb_str = f"{bb.get('jammed_honest',0)}/{bb.get('n',0)}"
        lines.append(f"| {cid} | {within} | {wb_str} | {bb_str} |\n")
        wb_jam += wb.get("jammed_honest", 0); wb_n += wb.get("n", 0)
        bb_jam += bb.get("jammed_honest", 0); bb_n += bb.get("n", 0)

    lines.append(
        f"| **TOTAL** | — | **{wb_jam}/{wb_n} "
        f"({100*wb_jam/max(wb_n,1):.0f}%)** | **{bb_jam}/{bb_n} "
        f"({100*bb_jam/max(bb_n,1):.0f}%)** |\n\n"
    )

    lines.append("## Comparison with original 3-class pilot (λ=0.5)\n\n")
    lines.append(
        "| Experiment | Paraphrase ASR |\n"
        "|-----------|---------------|\n"
        "| Original pilot (3 classes: test1, test2, test6) | 11/18 (61%) |\n"
        f"| Scale-up (10 new classes) — white-box λ=0.5 | {wb_jam}/{wb_n} "
        f"({100*wb_jam/max(wb_n,1):.0f}%) |\n"
        f"| Scale-up (10 new classes) — black-box baseline | {bb_jam}/{bb_n} "
        f"({100*bb_jam/max(bb_n,1):.0f}%) |\n\n"
    )

    lines.append("## Verdict\n\n")
    if wb_n > 0 and bb_n > 0:
        wb_rate = wb_jam / wb_n
        bb_rate = bb_jam / bb_n
        delta   = wb_rate - bb_rate
        if delta > 0.05:
            lines.append(
                f"White-box GTR improves paraphrase ASR by +{100*delta:.0f}pp "
                f"({100*wb_rate:.0f}% vs {100*bb_rate:.0f}%) on the 10-class scale-up, "
                "corroborating the pilot finding.\n"
            )
        elif delta < -0.05:
            lines.append(
                f"White-box GTR performs worse than the black-box baseline on the scale-up "
                f"({100*wb_rate:.0f}% vs {100*bb_rate:.0f}%), suggesting the pilot gain "
                "was class-specific.\n"
            )
        else:
            lines.append(
                f"White-box GTR and black-box baseline perform similarly on the scale-up "
                f"({100*wb_rate:.0f}% vs {100*bb_rate:.0f}%), with |Δ| < 5pp.\n"
            )

    with open(report_path, "w") as f:
        f.writelines(lines)
    log.info("Report -> %s", report_path)


if __name__ == "__main__":
    main()
