"""
Shared fixtures for all tests.

All tests are designed to run without a GPU. Models are mocked.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

# Ensure src/ is importable from tests/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def base_cfg():
    return OmegaConf.create({
        "models": {
            "target_llm": "dummy/model",
            "rag_embed": "dummy/rag",
            "oracle_embed": "dummy/oracle",
            "judge": "dummy/judge",
            "surrogate_lm": "distilgpt2",
        },
        "retrieval": {"k": 3, "score_function": "cos_sim"},
        "attack": {
            "num_tokens": 5,
            "num_iterations": 20,
            "es_patience": 10,
            "batch_size": 4,
            "llm_batch_size": 4,
            "max_response_len": 32,
            "doc_init": "exclamation",
            "temperature": 0.0,
            "seed": 42,
            "num_queries": 5,
            "target_response": "I don't know.",
        },
        "data": {"dataset": "nq", "split": "test"},
        "paths": {"data_dir": "data/", "results_dir": "results/", "index_dir": "data/indices/"},
        "vllm": {"gpu_memory_utilization": 0.85, "dtype": "float16", "max_model_len": 4096},
        "rag_prompt": "Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and no other prior knowledge, answer the query. If the context does not provide enough information to answer the query, reply 'I don't know.'\nDo not use any prior knowledge that was not supplied in the context.\nQuery: {query}\nAnswer:",
    })


@pytest.fixture
def tiny_corpus():
    return {
        "doc0": "The Eiffel Tower is in Paris.",
        "doc1": "Python is a programming language.",
        "doc2": "Water boils at 100 degrees Celsius.",
        "doc3": "The Amazon river is the longest river.",
        "doc4": "Marie Curie discovered radium.",
    }


@pytest.fixture
def fixed_rng():
    return np.random.default_rng(42)
