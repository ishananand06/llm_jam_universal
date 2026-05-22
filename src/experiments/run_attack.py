"""
Main attack experiment runner.

Entry point for all attack experiments. Config-driven via Hydra.

Usage:
    python src/experiments/run_attack.py --config-name reproduction
    python src/experiments/run_attack.py --config-name reproduction attack.num_queries=5
    python src/experiments/run_attack.py --config-name base models.target_llm=Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone

os.environ.setdefault("HF_HOME", "/home/ishana/scratch/hf_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
from pathlib import Path

import pickle

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Ensure src/ is on the path when running as a script
_src = Path(__file__).resolve().parents[2] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

log = logging.getLogger(__name__)


def _build_attack(cfg: DictConfig, retriever, generator, gpu_manager):
    method = cfg.attack.get("method", "shafran_bbo")
    if method == "shafran_bbo":
        from attacks.shafran_bbo import ShafranBBO
        return ShafranBBO(cfg, retriever, generator, gpu_manager)
    else:
        raise ValueError(
            f"Unknown attack method: {method!r}. "
            "Implement the method and register it here."
        )


@hydra.main(config_path="../../configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    from utils.seed import set_seed
    from utils.gpu_manager import GPUManager
    from utils.io import save_results
    from data.nq_loader import load_nq
    from rag.retriever import Retriever
    from rag.generator import VLLMGenerator
    from rag.pipeline import RAGPipeline
    from judges.local_judge import LocalJudge
    from data.answerable_filter import filter_answerable

    set_seed(cfg.attack.seed)

    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    log.info(
        "Models: target=%s | rag_embed=%s | oracle=%s | judge=%s",
        cfg.models.target_llm, cfg.models.rag_embed,
        cfg.models.oracle_embed, cfg.models.judge,
    )

    gpu = GPUManager()

    # --- Load data ---
    if cfg.data.dataset == "nq":
        queries, corpus = load_nq(cfg)
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.dataset!r}")

    # --- Build retrieval index ---
    retriever = Retriever(
        model_name=cfg.models.rag_embed,
        score_function=cfg.retrieval.score_function,
    )
    index_dir = Path(cfg.paths.index_dir) / cfg.data.dataset / cfg.models.rag_embed.replace("/", "__")
    retriever.build_or_load_index(corpus, index_dir)

    # --- Load target LLM ---
    generator = VLLMGenerator(
        model_name=cfg.models.target_llm,
        temperature=cfg.attack.temperature,
        max_tokens=cfg.attack.max_response_len,
        gpu_memory_utilization=cfg.vllm.gpu_memory_utilization,
        dtype=cfg.vllm.dtype,
        max_model_len=cfg.vllm.max_model_len,
    )

    pipeline = RAGPipeline(retriever, generator, cfg)

    # --- Filter to answerable queries ---
    # Generator (vLLM) and judge (Gemma-2-9B) can't coexist on one GPU.
    # On cache hit: load IDs and skip both models.
    # On cache miss: phase 1 generates all responses (generator only),
    #               then generator is closed before judge loads.
    cache_path = Path(cfg.paths.data_dir) / f"answerable_{cfg.data.dataset}_{cfg.data.split}.pkl"

    if cache_path.exists():
        log.info("Loading answerable-filter cache from %s", cache_path)
        with open(cache_path, "rb") as f:
            answerable_ids: set[str] = pickle.load(f)
        queries = [q for q in queries if q.id in answerable_ids]
        log.info("Loaded %d answerable queries from cache.", len(queries))
    else:
        responses_path = Path(cfg.paths.data_dir) / f"filter_responses_{cfg.data.dataset}_{cfg.data.split}.pkl"

        if responses_path.exists():
            log.info("Loading phase-1 response cache from %s (skipping generation)", responses_path)
            with open(responses_path, "rb") as f:
                query_responses: list[tuple] = pickle.load(f)
            log.info("Loaded %d cached responses. Closing generator.", len(query_responses))
            generator.close()
        else:
            log.info("Answerable filter phase 1: running pipeline on %d queries...", len(queries))
            query_responses = [
                (q, pipeline.run(q.text).response)
                for q in tqdm(queries, desc="Filter/generate")
            ]
            responses_path.parent.mkdir(parents=True, exist_ok=True)
            with open(responses_path, "wb") as f:
                pickle.dump(query_responses, f)
            log.info("Phase-1 responses saved to %s", responses_path)

            log.info("Closing generator to free VRAM for judge...")
            generator.close()

        log.info("Answerable filter phase 2: loading judge...")
        judge = LocalJudge(model_name=cfg.models.judge)
        answerable_ids = set()
        for q, response in tqdm(query_responses, desc="Filter/judge"):
            if judge.is_answered(q.text, response):
                answerable_ids.add(q.id)
        judge.close()
        del judge

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(answerable_ids, f)
        log.info("Answerable filter: %d/%d answerable. Cache saved to %s",
                 len(answerable_ids), len(queries), cache_path)
        responses_path.unlink(missing_ok=True)

        queries = [q for q in queries if q.id in answerable_ids]

        log.info("Reloading generator for attack loop...")
        generator = VLLMGenerator(
            model_name=cfg.models.target_llm,
            temperature=cfg.attack.temperature,
            max_tokens=cfg.attack.max_response_len,
            gpu_memory_utilization=cfg.vllm.gpu_memory_utilization,
            dtype=cfg.vllm.dtype,
            max_model_len=cfg.vllm.max_model_len,
        )
        pipeline = RAGPipeline(retriever, generator, cfg)

    # --- Run attack ---
    attack = _build_attack(cfg, retriever, generator, gpu)
    queries_to_run = queries[: cfg.attack.num_queries]
    attack_method = cfg.attack.get("method", "shafran_bbo")
    checkpoint_path = (
        Path(cfg.paths.results_dir)
        / f"{attack_method}_{cfg.data.dataset}_checkpoint.csv"
    )
    log.info("Running attack on %d queries...", len(queries_to_run))

    results = []
    for q in tqdm(queries_to_run, desc="Attack"):
        result = attack.run(q.text)
        results.append(result)
        log.debug("Done: query=%r | success=%s | iters=%d",
                  q.text[:60], result.success, result.n_iterations)
        save_results(results, cfg, checkpoint_path)

    # --- Save results ---
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = (
        Path(cfg.paths.results_dir)
        / f"{attack_method}_{cfg.data.dataset}_{timestamp}.csv"
    )
    save_results(results, cfg, out_path)
    checkpoint_path.unlink(missing_ok=True)

    n_success = sum(r.success for r in results)
    log.info(
        "Done. ASR=%.1f%% (%d/%d). Results saved to %s",
        100 * n_success / max(len(results), 1), n_success, len(results), out_path,
    )

    generator.close()
    gpu.unload_all()


if __name__ == "__main__":
    main()
