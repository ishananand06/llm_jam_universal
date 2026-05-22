"""
Shafran et al. single-query Black-Box Optimization (BBO) attack.

This is our baseline reproduction. Each query gets one adversarial document
optimized independently via greedy token substitution.

Algorithm (from Shafran attack.py, lines ~140-250):
  - Candidate generation: sample batch_size unique tokens from a filtered vocab
    (top-100 wikitext-frequency tokens excluded), place each at a random
    position in the current adversarial document.
  - Scoring:
      1. Embed candidates with the RAG model; skip if cosine sim < threshold.
      2. For retrieved candidates: build RAG prompt, get LLM response.
      3. Embed responses with oracle model; loss = -cosine_sim(resp, target_resp).
  - Selection: greedy (accept only if strict improvement).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from omegaconf import DictConfig

from .base import Attack, AttackResult

if TYPE_CHECKING:
    from rag.retriever import Retriever
    from rag.generator import VLLMGenerator
    from utils.gpu_manager import GPUManager

log = logging.getLogger(__name__)

# Retrieval threshold: candidate must exceed this cosine sim to query to be
# considered "retrieved". Shafran uses the k-th neighbor score as the threshold;
# we use a fixed conservative value to avoid FAISS lookups in the hot path.
_RETRIEVAL_THRESHOLD = 0.3

# Number of top-frequency wikitext tokens to exclude from the candidate pool.
# Shafran filters top-100 to avoid trivially common tokens dominating search.
_FILTER_TOP_N = 100


def _load_wikitext_token_probs(tokenizer) -> np.ndarray:
    """
    Build a unigram token frequency distribution from wikitext-2.
    Used to filter the most common tokens (Shafran's vocab filtering).

    Returns array of shape (vocab_size,) with normalized counts.
    Falls back to uniform if wikitext can't be loaded.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        counts = np.zeros(len(tokenizer), dtype=np.int64)
        for example in ds:
            ids = tokenizer.encode(example["text"], add_special_tokens=False)
            for tid in ids:
                if 0 <= tid < len(counts):
                    counts[tid] += 1
        return counts
    except Exception as e:
        log.warning("Could not build wikitext token probs (%s). Using uniform.", e)
        return np.ones(len(tokenizer), dtype=np.int64)


class ShafranBBO(Attack):
    """
    Reproduces the Shafran et al. BBO attack exactly, using local models.

    Key difference from the original: oracle embedding uses BAAI/bge-large-en-v1.5
    instead of OpenAI text-embedding-3-small (functionally equivalent).
    """

    def __init__(
        self,
        cfg: DictConfig,
        retriever: Retriever,
        generator: VLLMGenerator,
        gpu_manager: GPUManager,
    ) -> None:
        super().__init__(cfg, retriever, generator, gpu_manager)

        # Build filtered token vocabulary (exclude top-N by wikitext frequency)
        log.info("Building candidate token vocabulary...")
        token_counts = _load_wikitext_token_probs(self._tokenizer)
        vocab_size = len(self._tokenizer)
        all_ids = np.arange(vocab_size)
        top_ids = np.argsort(token_counts)[::-1][:_FILTER_TOP_N]
        self._candidate_vocab = np.delete(all_ids, top_ids)
        log.info("Candidate vocab size: %d (filtered %d)", len(self._candidate_vocab), _FILTER_TOP_N)

        # Oracle embedding model (for scoring LLM responses)
        from sentence_transformers import SentenceTransformer
        log.info("Loading oracle embedding model: %s", cfg.models.oracle_embed)
        self._oracle_embedder = SentenceTransformer(
            cfg.models.oracle_embed,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Pre-compute target response embedding
        self._target_resp_emb = self._embed_oracle([cfg.attack.target_response])[0]
        log.info("Target response embedding computed.")

        # Format string for the RAG prompt
        self._rag_prompt_template: str = cfg.rag_prompt

    def _embed_oracle(self, texts: list[str]) -> np.ndarray:
        embs = self._oracle_embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        return embs

    def generate_candidates(
        self,
        current_tokens: list[int],
        query: str,
        iteration: int,
        rng: np.random.Generator,
    ) -> list[list[int]]:
        """
        Sample batch_size replacement tokens, each placed at a random position.

        Follows Shafran: one position per iteration, batch_size unique token
        replacements at that position. No replacement=True (unique candidates).
        """
        batch_size = self.cfg.attack.batch_size
        n = len(current_tokens)

        # Random position to mutate (same position for entire batch)
        pos = int(rng.integers(0, n))

        # Sample unique replacement tokens
        sampled = rng.choice(self._candidate_vocab, size=batch_size, replace=False)

        candidates: list[list[int]] = []
        for tok_id in sampled:
            cand = list(current_tokens)
            cand[pos] = int(tok_id)
            candidates.append(cand)

        return candidates

    def score_candidates(
        self,
        candidate_docs: list[str],
        query: str,
        query_embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Score each candidate by: retrieval check → LLM response → oracle distance.

        Loss = -cosine_sim(oracle_embed(response), oracle_embed(target_response)).
        Candidates that are not retrieved get loss = +inf.
        """
        n = len(candidate_docs)
        losses = np.full(n, np.inf, dtype=np.float32)

        # --- Step 1: retrieval check ---
        cand_embs = self.retriever.embed_batch(candidate_docs)
        retrieval_sims = cand_embs @ query_embedding  # shape (n,)
        retrieved_mask = retrieval_sims >= _RETRIEVAL_THRESHOLD
        retrieved_indices = np.where(retrieved_mask)[0].tolist()

        if not retrieved_indices:
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
            # Loss = -(similarity to target): minimizing loss = maximizing similarity
            # i.e., pushing the response toward "I don't know"
            losses[idx] = -sim

        return losses
