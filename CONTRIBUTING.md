# Contributing

This document is for anyone joining the project who wants to add a new attack, defense, or dataset. Read it before writing code.

## Adding a new attack

1. Create `src/attacks/your_attack.py`. Subclass `Attack` from `attacks/base.py`:

   ```python
   from attacks.base import Attack, AttackResult
   import numpy as np

   class YourAttack(Attack):
       def generate_candidates(self, current_tokens, query, iteration, rng):
           # Return list[list[int]] of length cfg.attack.batch_size
           ...

       def score_candidates(self, candidate_docs, query, query_embedding):
           # Return np.ndarray of shape (n_candidates,); lower = better attack
           # Assign +inf to candidates that fail retrieval
           ...
   ```

   The BBO loop in `Attack.run()` handles initialization, acceptance, early stopping, and result packaging automatically. You only implement the two methods above.

2. Register it in `src/experiments/run_attack.py` in `_build_attack()`:

   ```python
   elif method == "your_attack":
       from attacks.your_attack import YourAttack
       return YourAttack(cfg, retriever, generator, gpu_manager)
   ```

3. Add a config in `configs/your_attack.yaml`:

   ```yaml
   defaults:
     - base
   attack:
     method: your_attack
   ```

4. Add at least one test in `tests/test_attack_loop.py` using the toy mock pattern from the existing tests.

## Adding a new defense

1. Create `src/defenses/your_defense.py`. Implement one of two interfaces depending on what the defense does:

   - **Document filter** (remove suspicious docs before generation):
     ```python
     def filter_docs(docs: list[str], query: str, cfg: DictConfig) -> list[str]:
         """Return subset of docs that pass the defense."""
         ...
     ```

   - **Injection detector** (binary classifier per document):
     ```python
     def is_injection(doc: str, cfg: DictConfig) -> bool:
         """Return True if the document looks adversarial."""
         ...
     ```

2. Wire it into `src/experiments/run_defense_eval.py` once that file is implemented.

3. GPU memory rule: defenses that use a model must register with `GPUManager` and must not be loaded simultaneously with the target LLM or judge.

## Adding a new dataset

1. Create `src/data/your_dataset_loader.py`. Implement:

   ```python
   def load_your_dataset(cfg: DictConfig) -> tuple[list[Query], dict[str, str]]:
       """Returns (queries, corpus) where corpus is {doc_id: text}."""
       ...
   ```

2. Register it in `src/experiments/run_attack.py`:

   ```python
   elif cfg.data.dataset == "your_dataset":
       queries, corpus = load_your_dataset(cfg)
   ```

3. Export from `src/data/__init__.py`.

## GPU memory rules

The L40S has 46 GB shared with other users. Follow these rules to avoid OOM:

- **Never load the target LLM and the judge simultaneously.** The judge should be loaded only for the answerable-filter step, then closed (`judge.close()`) before the attack loop starts.
- **Always register large models with GPUManager.** Use `gpu.load(name, loader_fn)` and `gpu.unload(name)` so memory is tracked explicitly.
- **vLLM does not play nicely with `del model`.** Use `generator.close()` which calls the vLLM engine destructor properly, then `gc.collect()` + `torch.cuda.empty_cache()`.
- **Embedding models** (GTR-base, BAAI/bge-large-en-v1.5) can typically coexist with one 7-9B model but not two. If you add a new embedding model, verify VRAM usage in GPUManager logs.

## Reproducibility contract

Every experiment that writes results must call `utils.io.save_results()`. This function writes a metadata header (git commit, config, seed, timestamp) before the CSV rows. Results without this header will be rejected from paper tables.

To reproduce any result from a CSV, read the `# git_commit:` line, check out that commit, and run with the same config and seed.
