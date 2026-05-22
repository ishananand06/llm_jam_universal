from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from omegaconf import DictConfig

if TYPE_CHECKING:
    from rag.retriever import Retriever
    from rag.generator import VLLMGenerator
    from utils.gpu_manager import GPUManager

log = logging.getLogger(__name__)


@dataclass
class AttackResult:
    query: str
    final_doc: str
    final_loss: float
    loss_history: list[float] = field(default_factory=list)
    response_history: list[str] = field(default_factory=list)
    n_iterations: int = 0
    success: bool = False


class Attack(ABC):
    """
    Abstract base class for all blocker-document attacks.

    Subclasses implement generate_candidates() and score_candidates().
    The BBO loop in run() is shared across all attack variants.

    Strategy pattern: only candidate generation and scoring differ across
    Shafran-BBO, M1, M2, M3, and Combined attacks.
    """

    def __init__(
        self,
        cfg: DictConfig,
        retriever: Retriever,
        generator: VLLMGenerator,
        gpu_manager: GPUManager,
    ) -> None:
        self.cfg = cfg
        self.retriever = retriever
        self.generator = generator
        self.gpu_manager = gpu_manager
        self._tokenizer = self._load_tokenizer()
        self._last_response: str = ""

    def _load_tokenizer(self):
        from transformers import AutoTokenizer
        log.info("Loading tokenizer for %s", self.cfg.models.target_llm)
        return AutoTokenizer.from_pretrained(self.cfg.models.target_llm)

    def _init_doc_tokens(self) -> list[int]:
        """Initialize adversarial document token sequence."""
        method = self.cfg.attack.doc_init
        n = self.cfg.attack.num_tokens
        if method == "exclamation":
            tok_id = self._tokenizer.encode("!", add_special_tokens=False)[0]
            return [tok_id] * n
        elif method == "random":
            vocab_size = len(self._tokenizer)
            return list(np.random.randint(0, vocab_size, size=n))
        else:
            raise ValueError(f"Unknown doc_init method: {method!r}")

    def _tokens_to_text(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=True)

    def _make_adv_doc(self, query: str, tokens: list[int]) -> str:
        """
        Prepend query text to the adversarial token sequence.

        This is Shafran's trick for guaranteeing retrieval: a document that
        starts with the query string will always have high cosine similarity
        to the query embedding, ensuring the adversarial document is in top-k.
        """
        suffix = self._tokens_to_text(tokens)
        return f"{query}. {suffix}"

    @abstractmethod
    def generate_candidates(
        self,
        current_tokens: list[int],
        query: str,
        iteration: int,
        rng: np.random.Generator,
    ) -> list[list[int]]:
        """
        Generate a batch of candidate token sequences for this iteration.

        Each candidate is a small perturbation of current_tokens.
        Returns a list of length cfg.attack.batch_size.
        """

    @abstractmethod
    def score_candidates(
        self,
        candidate_docs: list[str],
        query: str,
        query_embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Score each candidate adversarial document.

        Returns losses of shape (n_candidates,); lower = better attack.
        Unretrieved candidates should be assigned loss=+inf.
        """

    def run(self, query: str) -> AttackResult:
        """
        Main BBO optimization loop — identical across all Attack subclasses.

        Iteration structure:
            1. generate_candidates() → list of token sequences
            2. decode + prepend query → list of doc strings
            3. score_candidates() → per-candidate losses
            4. greedy accept: update if best loss < current loss
            5. early stop if patience exceeded or attack already succeeded
        """
        cfg_a = self.cfg.attack
        rng = np.random.default_rng(cfg_a.seed)

        # Initialize
        cur_tokens = self._init_doc_tokens()
        cur_doc = self._make_adv_doc(query, cur_tokens)
        query_emb = self.retriever.embed_batch([query])[0]

        # Score initial document
        initial_scores = self.score_candidates([cur_doc], query, query_emb)
        cur_loss = float(initial_scores[0])

        loss_history: list[float] = [cur_loss]
        response_history: list[str] = []
        es_count = 0
        n_iters = 0

        log.debug("Attack on query %r | initial_loss=%.4f", query[:60], cur_loss)

        for iteration in range(cfg_a.num_iterations):
            n_iters = iteration + 1

            candidates = self.generate_candidates(cur_tokens, query, iteration, rng)
            cand_docs = [self._make_adv_doc(query, c) for c in candidates]
            scores = self.score_candidates(cand_docs, query, query_emb)

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

            if iteration % 100 == 0:
                log.debug(
                    "iter %d | loss=%.4f | es=%d | doc=%r",
                    iteration, cur_loss, es_count, cur_doc[:60],
                )

            if es_count >= cfg_a.es_patience:
                log.debug("Early stop at iter %d (patience=%d)", iteration, cfg_a.es_patience)
                break

        # Final evaluation: get the actual response for this doc
        self._last_response = ""
        self.score_candidates([cur_doc], query, query_emb)
        if self._last_response:
            response_history = [self._last_response]

        success = bool(cur_loss < 0)

        return AttackResult(
            query=query,
            final_doc=cur_doc,
            final_loss=cur_loss,
            loss_history=loss_history,
            response_history=response_history,
            n_iterations=n_iters,
            success=success,
        )
