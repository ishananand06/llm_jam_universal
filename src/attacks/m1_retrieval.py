"""
M1: Centroid-targeted retrieval attack.

Builds a retrieval sub-document d_r that embeds near the centroid of a query
class, so a single blocker document is retrieved for every query in the class.

Two variants:
  - Simple: find Q* = argmin ||E(Q) - C|| among class queries. Return Q* as d_r.
  - HotFlip: gradient-based token substitution to push embed(d_r) → C.
    Uses BGE-large (attacker's oracle) for optimization.
    Evaluation intentionally uses GTR-base (RAG's retriever) to test
    black-box transferability — we never optimize against GTR directly.

Reference: Zhong et al. 2023, "Poisoning Retrieval Corpora by Injecting
Adversarial Passages," EMNLP. Algorithm §3.2 adapted for class centroids.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

log = logging.getLogger(__name__)

# HotFlip default hyperparameters
_HOTFLIP_ITERS = 200
_HOTFLIP_TOP_K = 32      # candidates scored per iteration
_HOTFLIP_MAX_LEN = 64    # max BGE token length for d_r


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dr_representative(
    queries: list[str],
    query_embeddings: np.ndarray,
    centroid: np.ndarray,
) -> str:
    """
    Simple variant: return the query whose embedding is closest to the centroid.

    Args:
        queries: list of query strings (len N)
        query_embeddings: L2-normalised BGE embeddings, shape (N, D)
        centroid: L2-normalised class centroid, shape (D,)

    Returns:
        The query string Q* = argmin_Q ||E(Q) - C||
    """
    dists = np.linalg.norm(query_embeddings - centroid[np.newaxis, :], axis=1)
    best_idx = int(np.argmin(dists))
    log.debug(
        "Representative query idx=%d, dist_to_centroid=%.4f: %r",
        best_idx, float(dists[best_idx]), queries[best_idx][:80],
    )
    return queries[best_idx]


def build_dr_hotflip(
    queries: list[str],
    query_embeddings: np.ndarray,
    centroid: np.ndarray,
    bge_model: "PreTrainedModel",
    bge_tokenizer: "PreTrainedTokenizerBase",
    num_iters: int = _HOTFLIP_ITERS,
    top_k_candidates: int = _HOTFLIP_TOP_K,
    device: str = "cuda",
    seed: int = 42,
) -> str:
    """
    HotFlip variant: gradient-based token substitution toward class centroid.

    Initializes from the representative query, then iteratively replaces one
    token at a time to minimise 1 - cosine_sim(embed(d_r), C) under BGE-large.

    Args:
        queries: class query strings
        query_embeddings: pre-computed BGE embeddings, shape (N, D)
        centroid: L2-normalised class centroid, shape (D,)
        bge_model: loaded AutoModel for BGE-large (must be on `device`)
        bge_tokenizer: BGE tokenizer
        num_iters: number of HotFlip substitution iterations
        top_k_candidates: vocabulary candidates evaluated per iteration
        device: cuda device string
        seed: RNG seed for reproducibility

    Returns:
        Optimised d_r text string.
    """
    # ── Initialise from representative query ──────────────────────────────────
    init_text = build_dr_representative(queries, query_embeddings, centroid)

    enc = bge_tokenizer(
        init_text,
        return_tensors="pt",
        truncation=True,
        max_length=_HOTFLIP_MAX_LEN,
        padding=False,
    )
    current_ids: list[int] = enc["input_ids"][0].tolist()  # includes [CLS], [SEP]
    n_tokens = len(current_ids)

    # Token positions eligible for substitution (skip [CLS]=0 and [SEP]=-1)
    opt_positions = list(range(1, n_tokens - 1))
    if not opt_positions:
        log.warning("HotFlip: rep query tokenised to ≤2 tokens; returning as-is.")
        return init_text

    # ── Setup ─────────────────────────────────────────────────────────────────
    centroid_t = torch.tensor(
        centroid, device=device, dtype=torch.float32
    ).unsqueeze(0)  # [1, D]
    centroid_norm = F.normalize(centroid_t, p=2, dim=-1)

    # Token embedding matrix — used for gradient scoring and batched eval
    E: torch.Tensor = bge_model.embeddings.word_embeddings.weight  # [V, D]

    # Special token IDs to exclude from replacement
    special_ids: set[int] = set(filter(None, [
        bge_tokenizer.pad_token_id,
        bge_tokenizer.unk_token_id,
        bge_tokenizer.cls_token_id,
        bge_tokenizer.sep_token_id,
        bge_tokenizer.mask_token_id,
    ]))

    rng = np.random.default_rng(seed)

    best_ids = list(current_ids)
    best_loss = _eval_loss(bge_model, E, current_ids, centroid_norm, device)

    log.debug("HotFlip init: loss=%.4f  text=%r", best_loss, init_text[:60])

    # ── Optimisation loop ─────────────────────────────────────────────────────
    for iteration in range(num_iters):
        pos = int(rng.choice(opt_positions))

        # ── Forward + backward to get gradient at `pos` ──────────────────────
        ids_t = torch.tensor([current_ids], device=device)           # [1, L]
        attn   = torch.ones_like(ids_t)

        # Embed current tokens as a leaf variable so we can get grad
        with torch.no_grad():
            all_embs = E[ids_t]                                       # [1, L, D]
        input_emb = all_embs.detach().requires_grad_(True)

        outputs = bge_model(inputs_embeds=input_emb, attention_mask=attn)
        cls_emb = F.normalize(outputs.last_hidden_state[:, 0, :], p=2, dim=-1)
        loss = 1.0 - (cls_emb * centroid_norm).sum()
        loss.backward()

        grad_pos = input_emb.grad[0, pos]                            # [D]

        # ── Score vocabulary by first-order Taylor approximation ─────────────
        # approx_delta[v] = grad[pos] · E[v]  (lower = better improvement)
        with torch.no_grad():
            approx_scores = E @ grad_pos                              # [V]
            # Mask specials and current token
            approx_scores[list(special_ids)] = float("inf")
            approx_scores[current_ids[pos]] = float("inf")

            top_k_ids = torch.topk(
                approx_scores, top_k_candidates, largest=False
            ).indices.tolist()

        # ── Evaluate top-k candidates in a single batched forward pass ────────
        batch_ids = []
        for cand_id in top_k_ids:
            test = list(current_ids)
            test[pos] = cand_id
            batch_ids.append(test)

        batch_t  = torch.tensor(batch_ids, device=device)            # [K, L]
        batch_attn = torch.ones_like(batch_t)

        with torch.no_grad():
            batch_emb  = E[batch_t]                                  # [K, L, D]
            batch_out  = bge_model(
                inputs_embeds=batch_emb, attention_mask=batch_attn
            )
            batch_cls  = F.normalize(
                batch_out.last_hidden_state[:, 0, :], p=2, dim=-1
            )                                                        # [K, D]
            batch_loss = 1.0 - (batch_cls * centroid_norm).squeeze(0)  # [K, D] → sum below
            batch_loss = (batch_loss).sum(dim=-1)                    # actually need scalar per row
            # Correct: loss per candidate = 1 - dot(cls_k, centroid_norm)
            batch_loss = 1.0 - (batch_cls @ centroid_norm.T).squeeze(-1)  # [K]

        best_k_idx = int(batch_loss.argmin())
        best_k_loss = float(batch_loss[best_k_idx])

        if best_k_loss < float(loss.item()):
            current_ids[pos] = top_k_ids[best_k_idx]
            if best_k_loss < best_loss:
                best_loss = best_k_loss
                best_ids = list(current_ids)

        if iteration % 50 == 0:
            log.debug(
                "HotFlip iter %3d | loss=%.4f (best=%.4f)",
                iteration, float(loss.item()), best_loss,
            )

    # ── Decode result (strip [CLS] and [SEP]) ─────────────────────────────────
    result = bge_tokenizer.decode(best_ids[1:-1], skip_special_tokens=True).strip()
    log.debug("HotFlip final: loss=%.4f  text=%r", best_loss, result[:80])
    return result if result else init_text


def make_blocker(d_r: str, num_excl: int = 50) -> str:
    """
    Build the full blocker document: retrieval prefix d_r followed by
    num_excl '!' tokens as an unoptimised jamming suffix.

    The '!' suffix matches Shafran's exclamation initialisation and keeps
    the blocker structure consistent with the BBO-optimised baselines.
    """
    suffix = " ".join(["!"] * num_excl)
    return f"{d_r}. {suffix}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _eval_loss(
    model: "PreTrainedModel",
    E: torch.Tensor,
    token_ids: list[int],
    centroid_norm: torch.Tensor,
    device: str,
) -> float:
    """Compute 1 - cosine_sim(CLS(doc), centroid) without grad."""
    ids_t = torch.tensor([token_ids], device=device)
    with torch.no_grad():
        emb = E[ids_t]
        out = model(inputs_embeds=emb, attention_mask=torch.ones_like(ids_t))
        cls = F.normalize(out.last_hidden_state[:, 0, :], p=2, dim=-1)
        loss = float(1.0 - (cls @ centroid_norm.T).squeeze())
    return loss
