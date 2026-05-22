"""
Evaluate true ASR using Gemma judge on saved attack results.

Re-runs each (query, final_doc) pair through the RAG pipeline to get the
actual final response, then judges with Gemma. Saves a new CSV with
judge_success column and prints the true ASR.

Usage:
    python src/experiments/eval_asr.py results_csv=/path/to/results.csv --config-name reproduction
"""
from __future__ import annotations

import csv
import json
import logging
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/ishana/scratch/hf_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

_src = Path(__file__).resolve().parents[2] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

log = logging.getLogger(__name__)


def _load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#")]
    for row in csv.DictReader(lines):
        rows.append(dict(row))
    return rows


def _save_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@hydra.main(config_path="../../configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    from rag.retriever import Retriever
    from rag.generator import VLLMGenerator
    from judges.local_judge import LocalJudge

    results_csv = cfg.get("results_csv", None)
    if results_csv is None:
        raise ValueError("Pass results_csv=/path/to/file.csv on the command line.")
    results_csv = Path(results_csv)
    log.info("Loading results from %s", results_csv)

    rows = _load_csv(results_csv)
    log.info("Loaded %d rows.", len(rows))

    # Cache path: avoid re-running Mistral if we crash between phases
    responses_cache = results_csv.with_suffix(".eval_responses.pkl")

    # --- Build retrieval index ---
    retriever = Retriever(
        model_name=cfg.models.rag_embed,
        score_function=cfg.retrieval.score_function,
    )
    index_dir = Path(cfg.paths.index_dir) / cfg.data.dataset / cfg.models.rag_embed.replace("/", "__")
    retriever.build_or_load_index(None, index_dir)

    # --- Phase 1: re-run final docs through Mistral ---
    if responses_cache.exists():
        log.info("Loading eval response cache from %s", responses_cache)
        with open(responses_cache, "rb") as f:
            responses: list[str] = pickle.load(f)
        log.info("Loaded %d cached responses.", len(responses))
    else:
        generator = VLLMGenerator(
            model_name=cfg.models.target_llm,
            temperature=0.0,
            max_tokens=cfg.attack.max_response_len,
            gpu_memory_utilization=cfg.vllm.gpu_memory_utilization,
            dtype=cfg.vllm.dtype,
            max_model_len=cfg.vllm.max_model_len,
        )

        prompts = []
        for row in rows:
            query = row["query"]
            final_doc = row["final_doc"]
            top_k_ids = retriever.retrieve(query, k=cfg.retrieval.k)
            base_docs = [retriever.get_doc_text(d) for d in top_k_ids]
            context_docs = [final_doc] + base_docs[: cfg.retrieval.k - 1]
            context = "\n\n".join(context_docs)
            prompt = cfg.rag_prompt.format(context=context, query=query)
            prompts.append(prompt)

        log.info("Running %d prompts through %s...", len(prompts), cfg.models.target_llm)
        responses = generator.generate(prompts)

        generator.close()
        log.info("Generator closed.")

        with open(responses_cache, "wb") as f:
            pickle.dump(responses, f)
        log.info("Eval responses cached to %s", responses_cache)

    # --- Phase 2: judge with Gemma ---
    log.info("Loading judge: %s", cfg.models.judge)
    judge = LocalJudge(model_name=cfg.models.judge)

    n_jammed = 0
    for i, (row, response) in enumerate(tqdm(zip(rows, responses), total=len(rows), desc="Judge")):
        answered = judge.is_answered(row["query"], response)
        jammed = not answered  # attack succeeded if model no longer answers
        row["judge_success"] = str(jammed)
        row["final_response"] = response
        row["judge_response"] = response
        if jammed:
            n_jammed += 1

    judge.close()
    responses_cache.unlink(missing_ok=True)

    true_asr = 100.0 * n_jammed / len(rows)
    log.info("True ASR (Gemma judge): %.1f%% (%d/%d)", true_asr, n_jammed, len(rows))

    out_path = results_csv.with_stem(results_csv.stem + "_judged")
    _save_csv(rows, out_path)
    log.info("Judged results saved to %s", out_path)


if __name__ == "__main__":
    main()
