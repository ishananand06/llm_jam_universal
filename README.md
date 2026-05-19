# llm_jam_universal

Research codebase extending **"Machine Against the RAG"** (Shafran et al., USENIX Security 2025, arXiv:2406.05870).

We extend single-query blocker documents to **universal** blockers that jam entire classes of related queries. Three contributions:

| | Method | Description |
|--|--------|-------------|
| **M1** | Centroid-targeted retrieval | One document retrieved across a query class |
| **M2** | Class-averaged jamming loss | One document triggers refusal across a class |
| **M3** | Naturalness-constrained search | Low-perplexity blockers that evade defenses |

## Setup

```bash
uv sync
```

## Reproduce Shafran baseline

```bash
# Full run (100 NQ queries, Mistral-7B)
bash scripts/reproduce_shafran.sh

# Smoke test (5 queries)
bash scripts/reproduce_shafran.sh attack.num_queries=5
```

## Run tests (no GPU required)

```bash
pytest tests/ -v
```

## Project structure

```
configs/        YAML configs — one per experiment variant
src/
  attacks/      Attack implementations (base + Shafran BBO + stubs for M1/M2/M3)
  rag/          Retriever (FAISS), Generator (vLLM), RAGPipeline
  judges/       Local Gemma-2-9B judge (replaces GPT-4)
  defenses/     Defense wrappers (stubs)
  data/         Dataset loaders + answerable query filter
  utils/        GPUManager, seed, logging, CSV I/O with metadata headers
  experiments/  run_attack.py (Hydra entry), run_defense_eval.py, analyze_results.py
scripts/        Shell wrappers for common experiment runs
tests/          pytest tests, all mocked (no GPU needed)
```

See `CONTRIBUTING.md` for how to add a new attack, defense, or dataset.

## Hardware

Single NVIDIA L40S (46 GB VRAM). GPU memory management is explicit via `GPUManager` — never load the target LLM and judge simultaneously.

## Citation

```bibtex
@inproceedings{shafran2025machine,
  title={Machine Against the {RAG}: Jamming Retrieval-Augmented Generation with Blocker Documents},
  author={Shafran, Avital and Perez-Etzioni, Roie and Goldstein, Amir and Zanella-B\'eguelin, Santiago and Mirman, Matthew and Schuster, Roei},
  booktitle={USENIX Security Symposium},
  year={2025}
}
```
