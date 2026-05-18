"""
Tests for the FAISS-based retriever.

All tests use mocked embeddings to avoid loading real models or needing GPU.
Retriever tests that call build_index/save_index/load_index require faiss
to be installed (available in the llm_jam conda env). They are skipped
automatically in environments without faiss.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

faiss = pytest.importorskip("faiss", reason="faiss not installed — run in llm_jam conda env")


def _make_mock_retriever(tiny_corpus):
    """Build a Retriever with a mocked _embed() so no GPU/model is needed."""
    from rag.retriever import Retriever

    r = object.__new__(Retriever)
    r.model_name = "mock"
    r.score_function = "cos_sim"
    r.batch_size = 8
    r.device = "cpu"
    r._is_contriever = False
    r._index = None
    r._doc_ids = []
    r._corpus = {}

    dim = 8
    rng = np.random.default_rng(0)
    # Fixed embeddings: each doc gets a unique unit vector
    n = len(tiny_corpus)
    raw = rng.standard_normal((n + 10, dim)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    unit_vecs = raw / norms

    doc_ids = list(tiny_corpus.keys())
    _emb_map = {text: unit_vecs[i] for i, text in enumerate(tiny_corpus.values())}
    # Query embedding = same as doc0 so doc0 is always top-1
    _emb_map["query about Paris"] = unit_vecs[0]
    _emb_map["unrelated query xyz"] = unit_vecs[8]

    def mock_embed(texts):
        out = []
        for t in texts:
            if t in _emb_map:
                out.append(_emb_map[t])
            else:
                v = unit_vecs[hash(t) % len(unit_vecs)]
                out.append(v)
        return np.array(out, dtype=np.float32)

    r._embed = mock_embed
    r._st = None
    return r


def test_build_and_retrieve(tiny_corpus):
    r = _make_mock_retriever(tiny_corpus)
    r.build_index(tiny_corpus)

    assert r._index is not None
    assert len(r._doc_ids) == len(tiny_corpus)

    results = r.retrieve("query about Paris", k=1)
    assert len(results) == 1
    assert results[0] == "doc0"  # closest to Paris query embedding


def test_retrieve_returns_k_results(tiny_corpus):
    r = _make_mock_retriever(tiny_corpus)
    r.build_index(tiny_corpus)

    results = r.retrieve("query about Paris", k=3)
    assert len(results) == 3
    # All returned IDs should be in the corpus
    for doc_id in results:
        assert doc_id in tiny_corpus


def test_score_doc_range(tiny_corpus):
    r = _make_mock_retriever(tiny_corpus)
    r.build_index(tiny_corpus)

    score = r.score_doc("The Eiffel Tower is in Paris.", "query about Paris")
    # Cosine similarity of unit vectors is in [-1, 1]
    assert -1.0 <= score <= 1.0


def test_retrieve_with_scores(tiny_corpus):
    r = _make_mock_retriever(tiny_corpus)
    r.build_index(tiny_corpus)

    results = r.retrieve_with_scores("query about Paris", k=2)
    assert len(results) == 2
    ids, scores = zip(*results)
    # Scores should be in descending order (best first)
    assert scores[0] >= scores[1]


def test_save_load_index(tmp_path, tiny_corpus):
    import faiss

    r = _make_mock_retriever(tiny_corpus)
    r.build_index(tiny_corpus)

    r.save_index(tmp_path)
    assert (tmp_path / "index.faiss").exists()
    assert (tmp_path / "meta.pkl").exists()

    r2 = _make_mock_retriever(tiny_corpus)
    r2.load_index(tmp_path)
    assert r2._index.ntotal == len(tiny_corpus)
    assert r2._doc_ids == r._doc_ids
