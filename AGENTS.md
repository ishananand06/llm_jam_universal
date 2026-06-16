# AGENTS.md

This file tells you how to orient quickly. Read it first, then follow the pointers.

---

## What this project is

Extending Shafran et al. "Machine Against the RAG" (USENIX Security 2025) from
single-query attacks to **class-level attacks**: one blocker document that jams a
whole class of related queries. The central thesis is that single-document jamming
is bounded by *retrieval geometry*, not by the LLM or the optimizer.

Victim retriever: GTR-T5-base, 768-dim, FAISS top-5, cosine.
Attacker oracle (black-box): BGE-large-en-v1.5, 1024-dim.
Victim LLM: Llama-3.1-8B-Instruct (Mistral-7B for Shafran baseline reproduction).
Judge (locked, never change): Gemma-2-9B-Instruct.
Dataset: Natural Questions, 2.68M Wikipedia passages, ~3,450 queries.

---

## The most important files to read before doing anything

**1. Current experiment state:**
```
SESSION_HANDOFF.md              ← always reflects the most recent session
```

**2. What has been tried and decided:**
```
EXPERIMENT_LOGBOOK.md           ← one entry per completed experiment; decisions and findings only
```

**3. Ground truth results (CSV + markdown reports):**
```
results/                        ← committed markdown reports, one per task
/home/ishana/scratch/results/   ← all CSVs, checkpoints, logs (not committed)
```

---

## Conventions you must follow

**Metrics — always use honest ASR, never force-injection ASR.**
`jammed_honest = retrieved_top5 AND judge_refused`. The old inflated ASR
(~75%) was force-injection and is wrong by ~35pp. Every script written after
Step 0 uses honest ASR. Do not revert this.

**GPU assignment.** All experiment scripts set `CUDA_VISIBLE_DEVICES=1` at the
top. GPU 0 is used by unrelated processes. Do not change this.

**Resumability.** Every experiment script checkpoints to a `_ckpt.pkl` file in
`/home/ishana/scratch/results/`. If a run dies, re-run the same script — it
will skip completed keys. Never delete a checkpoint unless starting fresh
intentionally.

**tmux.** Always run experiments in named tmux sessions, never in a foreground
shell. Use `nohup` or log redirection inside the session. Check progress with
`tmux capture-pane -t <session> -p | tail -20`.

**New experiment scripts** go in `src/experiments/`. Follow the 3-phase
structure of existing scripts: (1) BBO with vLLM loaded, (2) generation with
vLLM loaded, (3) judge with vLLM closed and Gemma loaded. Always close vLLM
before loading Gemma — they cannot coexist on GPU 1.

**Updating this file.** AGENTS.md describes structure and conventions and
rarely needs changing. When you finish a session, update SESSION_HANDOFF.md
(overwrite it entirely) and append one entry to EXPERIMENT_LOGBOOK.md.

---

## Codebase map (key files only)

```
configs/base.yaml                           all model names and hyperparams
src/attacks/shafran_bbo.py                  Shafran baseline BBO
src/attacks/constrained_joint_bbo.py        black-box baseline with real top-5 gate
src/experiments/eval_task6_joint_objective.py   task 6 black-box baseline runner
src/experiments/eval_whitebox_gtr_pilot.py  original 3-class white-box pilot
src/experiments/eval_whitebox_para_scaleup.py   Exp 1: 10-class scale-up (NEW)
src/experiments/eval_whitebox_ablation_init.py  Exp 2: init ablation (NEW)
src/rag/retriever.py                        GTR FAISS retriever
src/judges/local_judge.py                   Gemma-2-9B judge
```

---

## Data locations

```
/home/ishana/scratch/data/classes/paraphrase_classes.json   100 classes, 6 queries each
/home/ishana/scratch/data/classes/entity_classes.json       20 classes, 3-8 queries each
/home/ishana/scratch/data/indices/nq/sentence-transformers__gtr-t5-base/   FAISS index
/home/ishana/scratch/results/                               all experiment outputs
/home/ishana/scratch/hf_cache/                              HuggingFace model cache
```
