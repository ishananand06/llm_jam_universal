"""
Constrained-Joint BBO (Task 6).

Identical to ShafranBBO except for one change: a candidate is rejected
(loss = +inf) if it would fail real top-5 retrieval. The "real" threshold
is the k-th neighbor cosine similarity of the optimisation query (q*) against
the NQ FAISS index, computed ONCE per run before iteration starts.

Why
---
ShafranBBO uses a fixed 0.3 cosine proxy in its retrieval gate. Empirically:
  - Mean real top-5 threshold across NQ queries ≈ 0.755
  - The 0.3 proxy lets the optimiser freely mutate the suffix to reduce
    jamming loss, even when the mutation drags blocker_sim from ~0.77 to
    ~0.73 (still above 0.3, but below the real bar).
  - Result: ~50% real retrieval rate even though every candidate passed
    the proxy gate.

This class enforces the real bar in the BBO hot path, so the optimiser
can only accept edits that preserve real retrievability of q*.

Note on scope
-------------
This constraint protects retrieval of q* only. For paraphrase classes
(within-class sim ≈ 0.90) we expect this to transfer to class members.
For entity classes (within-class sim ≈ 0.54) class-member retrieval is
governed by a geometric mismatch the BBO cannot fix on its own; the
constraint at most prevents the optimiser from making q* retrieval worse.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from omegaconf import DictConfig

from .shafran_bbo import ShafranBBO

if TYPE_CHECKING:
    from rag.retriever import Retriever
    from rag.generator import VLLMGenerator
    from utils.gpu_manager import GPUManager

log = logging.getLogger(__name__)


class ConstrainedJointBBO(ShafranBBO):
    """
    BBO with a hard real-top-5 retrieval constraint.

    Differences from ShafranBBO:
      - precompute_retrieval_threshold(q_star) must be called before .run()
      - score_candidates() rejects candidates with blocker_sim < real_threshold
        (instead of < 0.3 proxy) and never calls the LLM on them
      - exposes counters for diagnostics (n_rejected_by_constraint, ...)
    """

    def __init__(
        self,
        cfg: DictConfig,
        retriever: Retriever,
        generator: VLLMGenerator,
        gpu_manager: GPUManager,
    ) -> None:
        super().__init__(cfg, retriever, generator, gpu_manager)
        self._real_threshold: float | None = None
        self._constraint_query: str | None = None
        # Diagnostics
        self.n_candidates_scored = 0
        self.n_rejected_by_constraint = 0
        self.n_accepted_by_constraint = 0

    def precompute_retrieval_threshold(self, q_star: str) -> float:
        """
        Compute the real top-k retrieval threshold for q_star against the NQ
        corpus. The threshold is the cosine similarity of the k-th neighbor
        (k from cfg.retrieval.k, e.g. 5). A candidate blocker must have
        cosine_sim(blocker, q_star) >= threshold to enter the real top-k.
        """
        k = int(self.cfg.retrieval.k)
        top_k_results = self.retriever.retrieve_with_scores(q_star, k=k)
        scores = [s for _, s in top_k_results]
        if not scores:
            raise RuntimeError(f"Retriever returned no neighbors for q*: {q_star!r}")
        self._real_threshold = float(scores[-1])
        self._constraint_query = q_star
        log.info(
            "Constrained-Joint BBO: real top-%d threshold for q* = %.4f (vs proxy 0.3)",
            k, self._real_threshold,
        )
        # Reset diagnostics on each new run
        self.n_candidates_scored = 0
        self.n_rejected_by_constraint = 0
        self.n_accepted_by_constraint = 0
        return self._real_threshold

    def score_candidates(
        self,
        candidate_docs: list[str],
        query: str,
        query_embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Same as ShafranBBO.score_candidates but uses self._real_threshold
        as the retrieval gate.

        Candidates with blocker_sim < real_threshold get loss = +inf and the
        LLM is never invoked on them. This is the "never accept an edit that
        pushes the doc out of the real top-5" rule.
        """
        if self._real_threshold is None:
            raise RuntimeError(
                "ConstrainedJointBBO: must call precompute_retrieval_threshold(q*) "
                "before scoring. The constraint requires a known target query."
            )

        n = len(candidate_docs)
        losses = np.full(n, np.inf, dtype=np.float32)

        # --- Step 1: real-threshold retrieval check ---
        cand_embs = self.retriever.embed_batch(candidate_docs)
        retrieval_sims = cand_embs @ query_embedding   # shape (n,)
        retrieved_mask = retrieval_sims >= self._real_threshold
        retrieved_indices = np.where(retrieved_mask)[0].tolist()

        # Diagnostics
        self.n_candidates_scored += n
        self.n_rejected_by_constraint += int(n - len(retrieved_indices))
        self.n_accepted_by_constraint += int(len(retrieved_indices))

        if not retrieved_indices:
            # All candidates fail the constraint; nothing to score this iteration
            return losses

        # --- Step 2: build RAG prompts for retrieved candidates ---
        top_k_ids = self.retriever.retrieve(query, k=self.cfg.retrieval.k)
        base_docs = [self.retriever.get_doc_text(d) for d in top_k_ids]

        prompts: list[str] = []
        for idx in retrieved_indices:
            # Inject adversarial doc at position 0, drop last to keep k docs
            context_docs = [candidate_docs[idx]] + base_docs[:self.cfg.retrieval.k - 1]
            context = "\n\n".join(context_docs)
            prompt = self._rag_prompt_template.format(context=context, query=query)
            prompts.append(prompt)

        # --- Step 3: generate responses ---
        responses = self.generator.generate(prompts)

        # --- Step 4: embed responses with oracle model ---
        if len(candidate_docs) == 1 and responses:
            self._last_response = responses[0]
        resp_embs = self._embed_oracle(responses)

        # --- Step 5: compute losses ---
        for i, idx in enumerate(retrieved_indices):
            sim = float(np.dot(resp_embs[i], self._target_resp_emb))
            losses[idx] = -sim

        return losses
