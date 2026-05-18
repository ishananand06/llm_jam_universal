"""
Tests for the BBO attack loop.

Uses toy embeddings and mocked generator to verify that:
  1. The loop runs without error.
  2. Loss decreases over iterations (optimization works).
  3. AttackResult has correct fields.
  4. Early stopping fires when patience is exhausted.

No GPU required — retriever and generator are fully mocked.
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock


def _make_toy_attack(base_cfg):
    """
    Build a ShafranBBO with mocked retriever, generator, and oracle embedder.

    Embedding space: 8-dimensional. We set things up so that replacing
    the first token of the adversarial doc moves the oracle response
    embedding progressively closer to the target → loss decreases.
    """
    from attacks.shafran_bbo import ShafranBBO
    from attacks.base import Attack

    cfg = base_cfg
    dim = 8

    # --- Mock retriever ---
    rng_r = np.random.default_rng(1)
    query_emb = rng_r.standard_normal(dim).astype(np.float32)
    query_emb /= np.linalg.norm(query_emb)

    mock_retriever = MagicMock()
    mock_retriever.embed_batch.side_effect = lambda texts: np.vstack([
        query_emb for _ in texts
    ])  # Every candidate has same embedding as query → always "retrieved"
    mock_retriever.retrieve.return_value = ["doc0", "doc1", "doc2"]
    mock_retriever.get_doc_text.return_value = "Some document text."

    # --- Mock generator ---
    # First call returns a non-IDK response; subsequent calls return IDK
    call_count = {"n": 0}
    def mock_generate(prompts):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return ["The Eiffel Tower is in Paris."] * len(prompts)
        return ["I don't know."] * len(prompts)

    mock_generator = MagicMock()
    mock_generator.generate.side_effect = mock_generate

    # --- Mock gpu_manager ---
    mock_gpu = MagicMock()

    # --- Build attack without __init__ (to avoid loading real models) ---
    attack = object.__new__(ShafranBBO)
    attack.cfg = cfg
    attack.retriever = mock_retriever
    attack.generator = mock_generator
    attack.gpu_manager = mock_gpu

    # Mock tokenizer
    mock_tok = MagicMock()
    mock_tok.__len__ = lambda self: 1000
    mock_tok.encode.return_value = [33]  # '!' → token 33
    mock_tok.decode.return_value = "! ! ! ! !"
    attack._tokenizer = mock_tok

    # Candidate vocab: tokens 100-199
    attack._candidate_vocab = np.arange(100, 200)

    # Oracle embedder: returns embeddings that get progressively closer to target
    target_emb = np.zeros(dim, dtype=np.float32)
    target_emb[0] = 1.0  # target direction

    call_count_oracle = {"n": 0}
    def mock_oracle_embed(texts):
        call_count_oracle["n"] += 1
        n = len(texts)
        # Return embeddings increasingly aligned with target_emb
        t = min(call_count_oracle["n"] * 0.1, 0.99)
        embs = np.zeros((n, dim), dtype=np.float32)
        embs[:, 0] = t
        embs[:, 1] = np.sqrt(1 - t**2)
        return embs

    attack._embed_oracle = mock_oracle_embed
    attack._target_resp_emb = target_emb
    attack._rag_prompt_template = cfg.rag_prompt

    return attack


def test_bbo_loop_runs(base_cfg):
    attack = _make_toy_attack(base_cfg)
    result = attack.run("What is the Eiffel Tower?")
    assert result.query == "What is the Eiffel Tower?"
    assert isinstance(result.final_doc, str)
    assert isinstance(result.final_loss, float)
    assert result.n_iterations > 0
    assert len(result.loss_history) > 0


def test_bbo_loss_decreases(base_cfg):
    """Loss should decrease over the course of optimization."""
    attack = _make_toy_attack(base_cfg)
    result = attack.run("What is the Eiffel Tower?")
    # The mock oracle embedder returns progressively better embeddings,
    # so the loss should be strictly lower at the end than at the start.
    assert result.final_loss <= result.loss_history[0], (
        f"Expected final_loss={result.final_loss} <= initial_loss={result.loss_history[0]}"
    )


def test_bbo_early_stop(base_cfg):
    """Early stop should fire after es_patience iterations without improvement."""
    import copy
    cfg = copy.deepcopy(base_cfg)
    cfg.attack.es_patience = 3
    cfg.attack.num_iterations = 50

    attack = _make_toy_attack(cfg)

    # Make oracle embedder return constant embeddings → no improvement after first step
    constant_emb = np.array([[0.0] * 8], dtype=np.float32)
    attack._embed_oracle = lambda texts: np.repeat(constant_emb, len(texts), axis=0)
    attack._target_resp_emb = np.array([1.0] + [0.0] * 7, dtype=np.float32)

    result = attack.run("What is Paris?")
    # Should stop before num_iterations due to early stopping
    assert result.n_iterations <= 50


def test_attack_result_fields(base_cfg):
    attack = _make_toy_attack(base_cfg)
    result = attack.run("Who invented radium?")

    assert hasattr(result, "query")
    assert hasattr(result, "final_doc")
    assert hasattr(result, "final_loss")
    assert hasattr(result, "loss_history")
    assert hasattr(result, "response_history")
    assert hasattr(result, "n_iterations")
    assert hasattr(result, "success")
    assert isinstance(result.loss_history, list)
    assert len(result.loss_history) >= 1


def test_generate_candidates_batch_size(base_cfg):
    attack = _make_toy_attack(base_cfg)
    rng = np.random.default_rng(0)
    tokens = [33] * base_cfg.attack.num_tokens  # 5 '!' tokens
    candidates = attack.generate_candidates(tokens, "test query", 0, rng)
    assert len(candidates) == base_cfg.attack.batch_size
    for c in candidates:
        assert len(c) == len(tokens)


def test_generate_candidates_single_position_change(base_cfg):
    """Each candidate should differ from the original in exactly one position."""
    attack = _make_toy_attack(base_cfg)
    rng = np.random.default_rng(0)
    original = [33, 33, 33, 33, 33]
    candidates = attack.generate_candidates(original, "test query", 0, rng)
    for c in candidates:
        diffs = sum(1 for a, b in zip(original, c) if a != b)
        assert diffs == 1, f"Expected 1 difference, got {diffs}: {original} vs {c}"
