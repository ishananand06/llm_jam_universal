"""
M2: Class-averaged BBO jamming attack.

Extends the Shafran BBO attack to optimise a single blocker document against
an entire class of semantically-related queries (paraphrase or entity).

Key changes vs ShafranBBO
--------------------------
* Scoring averages loss over a mini-batch of k queries (default k=3) sampled
  fresh each iteration, instead of a single fixed query.
* All k×B candidate prompts are batched into ONE vLLM call per iteration for
  throughput.
* The adversarial document prepends d_r (M1 retrieval prefix) rather than the
  raw query, so retrieval works for all class members simultaneously.
* Warm-starting: pass init_tokens to run_class() to start from a previously
  trained blocker for a semantically close class.

Cost analysis
-------------
Per iteration: k retrieval checks + 1 batched LLM call (≤ k×B prompts) + k
oracle embeddings.  With k=3, B=32 this is 3× the per-iteration cost of
Shafran, but the richer gradient signal means convergence typically needs
fewer iterations → net ~2× overhead vs single-query BBO.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from omegaconf import DictConfig

from .base import AttackResult
from .shafran_bbo import ShafranBBO, _RETRIEVAL_THRESHOLD

if TYPE_CHECKING:
    from rag.retriever import Retriever
    from rag.generator import VLLMGenerator
    from utils.gpu_manager import GPUManager

log = logging.getLogger(__name__)

_MINI_BATCH_K = 3  # queries sampled per BBO iteration (fixed per spec)


@dataclass
class M2AttackResult(AttackResult):
    """AttackResult with per-class M2 metadata."""
    d_r: str = ""
    class_queries: list[str] = field(default_factory=list)
    # Per-query losses evaluated at the final blocker
    per_query_final_losses: list[float] = field(default_factory=list)


class M2ClassBBO(ShafranBBO):
    """
    Class-averaged BBO: one blocker optimised against a mini-batch of class
    queries each iteration.

    Inherits everything from ShafranBBO (oracle setup, candidate generation,
    token vocabulary filtering).  Only scoring and the main loop differ.
    """

    def __init__(
        self,
        cfg: DictConfig,
        retriever: "Retriever",
        generator: "VLLMGenerator",
        gpu_manager: "GPUManager",
    ) -> None:
        super().__init__(cfg, retriever, generator, gpu_manager)
        self._mini_batch_k: int = int(
            cfg.attack.get("mini_batch_k", _MINI_BATCH_K)
        )
        log.info("M2ClassBBO ready (mini_batch_k=%d)", self._mini_batch_k)

    # ------------------------------------------------------------------
    # Document construction
    # ------------------------------------------------------------------

    def _make_class_adv_doc(self, d_r: str, tokens: list[int]) -> str:
        """Build blocker = [M1 retrieval prefix] + [BBO jamming suffix]."""
        suffix = self._tokens_to_text(tokens)
        return f"{d_r}. {suffix}"

    # ------------------------------------------------------------------
    # Multi-query scoring (core M2 logic)
    # ------------------------------------------------------------------

    def _score_candidates_class(
        self,
        candidate_docs: list[str],
        sampled_queries: list[str],
    ) -> np.ndarray:
        """
        Score each candidate by averaging loss over `sampled_queries`.

        For each query:
          1. Retrieval check: skip candidates whose cosine sim < threshold.
          2. Collect RAG prompts for retrieved candidates.
        All prompts for all queries are batched into a SINGLE vLLM call.
        Loss per candidate = -mean_k(sim(response_embed, target_embed))
        where non-retrieved queries contribute sim=0 (conservative).

        Returns losses of shape (n_candidates,); lower = better.
        """
        n = len(candidate_docs)
        k = len(sampled_queries)
        accumulated_sim = np.zeros(n, dtype=np.float32)

        # Embed all candidates once — reused across all k queries
        cand_embs = self.retriever.embed_batch(candidate_docs)  # (n, D_rag)

        # ── Build all prompts across all k queries ────────────────────────────
        # prompt_meta: list of (candidate_idx, query_idx) so we can unpack
        all_prompts: list[str] = []
        prompt_meta: list[tuple[int, int]] = []

        for q_idx, query in enumerate(sampled_queries):
            query_emb = self.retriever.embed_batch([query])[0]  # (D_rag,)
            retrieval_sims = cand_embs @ query_emb               # (n,)
            retrieved_mask = retrieval_sims >= _RETRIEVAL_THRESHOLD
            retrieved_indices = np.where(retrieved_mask)[0].tolist()

            if not retrieved_indices:
                continue

            top_k_ids = self.retriever.retrieve(query, k=self.cfg.retrieval.k)
            base_docs = [self.retriever.get_doc_text(d) for d in top_k_ids]

            for c_idx in retrieved_indices:
                ctx_docs = [candidate_docs[c_idx]] + base_docs[: self.cfg.retrieval.k - 1]
                context = "\n\n".join(ctx_docs)
                prompt = self._rag_prompt_template.format(
                    context=context, query=query
                )
                all_prompts.append(prompt)
                prompt_meta.append((c_idx, q_idx))

        if not all_prompts:
            # No candidate retrieved by any query — all losses are 0
            return -(accumulated_sim / k)

        # ── Single batched vLLM call for all prompts ──────────────────────────
        responses = self.generator.generate(all_prompts)

        # ── Oracle embedding + loss accumulation ──────────────────────────────
        resp_embs = self._embed_oracle(responses)  # (total_prompts, D_oracle)

        for i, (c_idx, _q_idx) in enumerate(prompt_meta):
            sim = float(np.dot(resp_embs[i], self._target_resp_emb))
            accumulated_sim[c_idx] += sim

        # Store last response for the final eval call (single-doc path)
        if len(all_prompts) == 1:
            self._last_response = responses[0]

        # loss = -mean_sim (non-retrieved queries contribute 0)
        return -(accumulated_sim / k)

    # ------------------------------------------------------------------
    # Per-query final loss (used after optimization for evaluation)
    # ------------------------------------------------------------------

    def _eval_final_losses(
        self,
        final_doc: str,
        queries: list[str],
    ) -> list[float]:
        """Compute single-query loss for each class member at the final doc."""
        losses = []
        for query in queries:
            query_emb = self.retriever.embed_batch([query])[0]
            loss_arr = self.score_candidates(
                [final_doc], query, query_emb
            )
            losses.append(float(loss_arr[0]))
        return losses

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_class(
        self,
        queries: list[str],
        d_r: str,
        init_tokens: list[int] | None = None,
        rng_seed: int | None = None,
    ) -> M2AttackResult:
        """
        Optimise a jamming suffix d_j so that d_r + d_j jams all queries.

        Args:
            queries:      All class query strings (N ≥ 1).
            d_r:          M1 retrieval prefix (already found to be retrieved).
            init_tokens:  Warm-start: token list for d_j initialisation.
                          If None, uses the default exclamation initialisation.
            rng_seed:     Override RNG seed (default: cfg.attack.seed).

        Returns:
            M2AttackResult with the optimised blocker and per-query losses.
        """
        cfg_a = self.cfg.attack
        seed = rng_seed if rng_seed is not None else cfg_a.seed
        rng = np.random.default_rng(seed)
        k = self._mini_batch_k

        # ── Initialise d_j token sequence ────────────────────────────────────
        if init_tokens is not None:
            cur_tokens = list(init_tokens)
            log.info("M2: warm-start from %d provided tokens", len(cur_tokens))
        else:
            cur_tokens = self._init_doc_tokens()

        cur_doc = self._make_class_adv_doc(d_r, cur_tokens)

        # Initial score: sample k queries and evaluate
        sample_k = min(k, len(queries))
        init_q_idx = rng.choice(len(queries), size=sample_k, replace=False).tolist()
        sampled_q = [queries[i] for i in init_q_idx]
        initial_scores = self._score_candidates_class([cur_doc], sampled_q)
        cur_loss = float(initial_scores[0])

        loss_history: list[float] = [cur_loss]
        es_count = 0
        n_iters = 0

        log.info(
            "M2 class BBO | d_r=%r | n_queries=%d | initial_loss=%.4f",
            d_r[:50], len(queries), cur_loss,
        )

        # ── BBO loop ──────────────────────────────────────────────────────────
        for iteration in range(cfg_a.num_iterations):
            n_iters = iteration + 1

            # Sample fresh mini-batch of k queries each iteration
            sample_k = min(k, len(queries))
            q_idx = rng.choice(len(queries), size=sample_k, replace=False).tolist()
            sampled_q = [queries[i] for i in q_idx]

            # Generate candidates (inherited from ShafranBBO — same token flips)
            candidates = self.generate_candidates(
                cur_tokens, queries[0], iteration, rng
            )
            cand_docs = [self._make_class_adv_doc(d_r, c) for c in candidates]

            # Score candidates with class-averaged loss
            scores = self._score_candidates_class(cand_docs, sampled_q)

            best_idx = int(np.argmin(scores))
            best_loss = float(scores[best_idx])

            if best_loss < cur_loss:
                cur_tokens = candidates[best_idx]
                cur_doc = cand_docs[best_idx]
                cur_loss = best_loss
                es_count = 0
            else:
                es_count += 1

            loss_history.append(cur_loss)

            if iteration % 50 == 0:
                log.debug(
                    "M2 iter %3d | loss=%.4f | es=%d",
                    iteration, cur_loss, es_count,
                )

            if es_count >= cfg_a.es_patience:
                log.info(
                    "M2 early stop at iter %d (patience=%d, loss=%.4f)",
                    iteration, cfg_a.es_patience, cur_loss,
                )
                break

        # ── Final per-query evaluation ────────────────────────────────────────
        per_query_losses = self._eval_final_losses(cur_doc, queries)

        log.info(
            "M2 done | iters=%d | final_loss=%.4f | per_query=%s",
            n_iters,
            cur_loss,
            [f"{l:.3f}" for l in per_query_losses],
        )

        return M2AttackResult(
            query=queries[0],          # representative query for compatibility
            final_doc=cur_doc,
            final_loss=cur_loss,
            loss_history=loss_history,
            n_iterations=n_iters,
            success=bool(cur_loss < 0),
            d_r=d_r,
            class_queries=queries,
            per_query_final_losses=per_query_losses,
        )

    # ------------------------------------------------------------------
    # Compatibility shim
    # ------------------------------------------------------------------

    def run(self, query: str) -> AttackResult:
        """Single-query fallback — delegates to ShafranBBO.run()."""
        return ShafranBBO.run(self, query)
