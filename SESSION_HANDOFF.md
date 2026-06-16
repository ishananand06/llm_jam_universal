# Session Handoff — RAG Jamming Experiments

**Written**: 2026-06-16 ~22:57  
**Project**: llm_jam_universal (`/home/ishana/projects/llm_jam_universal`)  
**Context**: Two new experiments were designed and launched to characterise the
geometric bound on single-document RAG jamming attacks (extending Shafran et al.,
USENIX Security 2025).

---

## What is currently running

Two tmux sessions are active. Both run on **GPU 1** (L40S, ~45 GB free).
GPU 0 is nearly full and is used by a separate unrelated process.

| tmux session | experiment | status |
|---|---|---|
| `wb_scaleup` | Exp 1 — paraphrase scale-up | **actively running** |
| `wb_ablation` | Exp 2 — init ablation | **waiting for Exp 1**, then auto-starts |

Check status at any time:
```bash
tmux capture-pane -t wb_scaleup -p | tail -20
tmux capture-pane -t wb_ablation -p | tail -5
tail -30 /home/ishana/scratch/results/whitebox_para_scaleup.log
tail -30 /home/ishana/scratch/results/whitebox_ablation_init.log  # only exists after Exp 2 starts
```

---

## Experiment 1: Paraphrase scale-up

**Purpose**: The original white-box GTR pilot ran on only 3 paraphrase classes
(test1, test2, test6) and the +17pp gain over baseline was driven entirely by
one class (test2). This experiment tests whether that gain generalises to 10 new
classes.

**Script**: `src/experiments/eval_whitebox_para_scaleup.py`

**Design**:
- 10 new paraphrase classes: `para_test9, para_test10, para_test11, para_test12,
  para_test13, para_test14, para_test16, para_test17, para_test19, para_test20`
  (indices 3–12 in `paraphrase_classes.json`; none were in the original pilot)
- White-box GTR joint-loss BBO at **λ=0.5 only** (pilot-optimal; no λ sweep)
- **Also runs constrained-joint BBO black-box baseline on the same classes** so
  comparison is self-contained (no dependency on old task6 CSV)
- Same budget as original pilot: T=500, patience=50, n=50 candidates/iter
- Vec2text-perturbed init (same as original pilot, L2=0.05)

**Pipeline phases**:
1. Phase 1: vec2text init for all 10 classes
2. Phase 2a: White-box BBO (λ=0.5) per class
3. Phase 2b: Black-box ConstrainedJointBBO per class
4. Phase 3: Honest top-5 retrieval + LLM generation (both methods)
5. Judge: Gemma-2-9B-it
6. Write report

**Progress as of handoff** (from checkpoint):
- Phase 1 complete: all 10 classes have vec2text init ✓
- Phase 2a (WB BBO) complete: `para_test9`, `para_test10` ✓ (8 remaining)
- Phase 2b (BB BBO): not started yet
- Phase 3 / judge: not started yet

**Output files**:
```
/home/ishana/scratch/results/whitebox_para_scaleup.log          # live log
/home/ishana/scratch/results/whitebox_para_scaleup_ckpt.pkl     # resumable checkpoint
/home/ishana/scratch/results/whitebox_para_scaleup_honest.csv   # final results (when done)
/home/ishana/scratch/results/whitebox_para_scaleup_report.md    # final report (when done)
/home/ishana/scratch/results/whitebox_para_scaleup_trajectory_<class>_wb.csv  # per-class BBO trace
```

**Estimated time remaining**: ~8–10h from handoff time (8 WB BBO runs + 10 BB BBO
runs + generation + judging; ~30–45 min per BBO run).

**Checkpoint is resumable**: if the run dies, re-run the same script — it will
skip all completed keys and pick up where it left off.

---

## Experiment 2: Init ablation

**Purpose**: Isolate whether the vec2text-perturbed init contributed to the
white-box GTR performance gain, or whether the GTR loss term alone is sufficient.
The original pilot changed two things at once (init + loss term); this ablation
holds the loss term fixed and changes only the init.

**Script**: `src/experiments/eval_whitebox_ablation_init.py`

**Design**:
- Same 6 classes as original pilot: `para_test1, para_test2, para_test6,
  entity_00, entity_08, entity_09`
- White-box GTR joint-loss BBO, **λ=0.5**
- **Init: 50 × `!` tokens** (Shafran standard random-suffix init)
  — the only difference from the original pilot
- Same budget: T=500, patience=50, n=50
- Loads pilot checkpoint (`whitebox_pilot_ckpt.pkl`) to fill in the
  "Condition A" (vec2text init) column of the comparison — no re-running

**Trigger**: auto-starts when `whitebox_para_scaleup_report.md` is created
(polled every 2 min in `wb_ablation` tmux session).

**Output files**:
```
/home/ishana/scratch/results/whitebox_ablation_init.log          # live log (after start)
/home/ishana/scratch/results/whitebox_ablation_init_ckpt.pkl     # resumable checkpoint
/home/ishana/scratch/results/whitebox_ablation_init_honest.csv   # final results
/home/ishana/scratch/results/whitebox_ablation_init_report.md    # final report
/home/ishana/scratch/results/whitebox_ablation_init_trajectory_<class>.csv  # per-class BBO trace
```

**Estimated time**: ~3–4h after Exp 1 finishes (6 classes × 1 BBO run each).

**Checkpoint is resumable**: same as Exp 1.

---

## What the results will tell you

### Exp 1 report (`whitebox_para_scaleup_report.md`)

The report auto-generates a table:

| class_id | within_sim | WB λ=0.5 ASR | BB baseline ASR |
|---|---|---|---|

And an aggregated comparison to the original 3-class pilot. The verdict section
will state one of three outcomes:
- **WB consistently better than BB** (+>5pp): white-box GTR generalises —
  the pilot finding was real, not lucky class selection. Strengthens the paper.
- **WB ≈ BB** (|Δ| ≤ 5pp): the pilot gain was class-specific. The bound is
  robust to attacker oracle quality even in the paraphrase regime.
- **WB worse than BB**: unexpected; would suggest vec2text init placed the
  optimiser in a poor region for these classes specifically.

Also check the per-class BB baseline ASR for the 10 new classes — this extends
the task6 dataset from 10 to 20 paraphrase classes, useful independent of the
white-box comparison.

### Exp 2 report (`whitebox_ablation_init_report.md`)

Three-way comparison table (per-class and aggregated):

| Condition | Paraphrase ASR | Entity ASR |
|---|---|---|
| A: WB GTR λ=0.5 + vec2text init (pilot) | e.g. 11/18 (61%) | e.g. 2/19 (11%) |
| B: WB GTR λ=0.5 + random ! init (ablation) | ? | ? |

The verdict section will state:
- **|A − B| ≤ 5pp**: init is irrelevant → GTR loss term alone drives the gain.
  Vec2text perturbation idea can be cleanly removed from the paper's causal story.
- **A >> B** (>5pp): init contributes non-trivially → both components matter.
- **B > A**: random init is better → vec2text init was actively harmful
  (started in a poor local region of embedding space).

Also check the trajectory CSVs (`whitebox_ablation_init_trajectory_<class>.csv`)
for GTR cosine at iter 0: for random init it should start near the real threshold
(~0.8 range) vs. the pilot's 0.821 for vec2text init. If both start similarly,
that's additional evidence the init didn't matter.

---

## Where all the data lives

### Class definitions
```
/home/ishana/scratch/data/classes/paraphrase_classes.json   # 100 paraphrase classes
/home/ishana/scratch/data/classes/paraphrase_embeddings.npy # (100, 6, 1024) BGE-large
/home/ishana/scratch/data/classes/entity_classes.json       # 20 entity classes
/home/ishana/scratch/data/classes/entity_embeddings.npy     # (20, 8, 1024) BGE-large
```

### FAISS index (GTR-base, NQ corpus, 2.68M docs)
```
/home/ishana/scratch/data/indices/nq/sentence-transformers__gtr-t5-base/
```

### Prior experiment results (for comparison baselines)
```
/home/ishana/scratch/results/task6_constrained_joint_honest.csv  # BB baseline, 10 para + 10 entity
/home/ishana/scratch/results/whitebox_pilot_honest.csv           # original 3-class pilot
/home/ishana/scratch/results/whitebox_pilot_ckpt.pkl             # original pilot checkpoint (Exp 2 reads this)
/home/ishana/scratch/results/whitebox_pilot_report.md            # original pilot report
```

### This session's new results
```
/home/ishana/scratch/results/whitebox_para_scaleup_honest.csv    # Exp 1 output
/home/ishana/scratch/results/whitebox_para_scaleup_report.md     # Exp 1 report
/home/ishana/scratch/results/whitebox_ablation_init_honest.csv   # Exp 2 output
/home/ishana/scratch/results/whitebox_ablation_init_report.md    # Exp 2 report
```

---

## Suggested next steps (after both experiments finish)

### Immediate (analysis)

1. **Read both reports** — they are auto-generated markdown and self-interpreting.
   Start with `whitebox_para_scaleup_report.md`, then `whitebox_ablation_init_report.md`.

2. **Update the paper's evidence table**. The project currently has 6 lines of
   evidence for the geometric bound. These experiments add:
   - Exp 1 → evidence line 7: "threat-model invariance at scale (10-class paraphrase)"
   - Exp 2 → evidence line 8 (or a clarification of line 6): "bound is not
     init-sensitive; GTR loss term is the sole driver"

3. **Characterise para_test1** (the 0/6 unreachable class from the pilot). Check
   whether it is also 0/N in Exp 1's BB baseline — if it's consistently 0, it
   warrants a brief structural analysis (embedding spread, distance to nearest
   retrievable doc). See `eval_whitebox_pilot.py:PARA_CLASS_IDS` for context.

4. **Check the trajectory CSVs** for Exp 2's random-init runs on test2 (the class
   that drove the pilot gain). Compare GTR cosine trajectories between:
   - `whitebox_pilot_trajectory_para_test2_lambda0.5.csv` (vec2text init)
   - `whitebox_ablation_init_trajectory_para_test2.csv` (random ! init)
   If both reach similar final GTR cosines, the init truly didn't matter.

### Potential follow-up experiments

5. **Larger λ sweep on the scale-up classes** — only needed if Exp 1 shows WB
   consistently better than BB (confirming the pilot finding). Then sweep
   λ ∈ {0.3, 0.7} on the same 10 classes to check if λ=0.5 remains optimal
   at scale. Use `eval_whitebox_para_scaleup.py` as a template; add the two
   lambdas to the `LAM` list.

6. **White-box Mistral experiment** (full threat-model relaxation). Relaxing
   retriever access is done; relaxing LLM access is the remaining dimension.
   A 6-class pilot (same as this ablation's classes) would complete the 2×2
   threat-model table. This is explicitly flagged as future work in the existing
   pilot design.

---

## Codebase quick reference

| File | Purpose |
|---|---|
| `src/experiments/eval_whitebox_para_scaleup.py` | Exp 1 script |
| `src/experiments/eval_whitebox_ablation_init.py` | Exp 2 script |
| `src/experiments/eval_whitebox_gtr_pilot.py` | Original 3-class pilot |
| `src/experiments/eval_task6_joint_objective.py` | BB baseline (task6) |
| `src/attacks/constrained_joint_bbo.py` | ConstrainedJointBBO implementation |
| `src/rag/retriever.py` | GTR FAISS retriever |
| `src/judges/local_judge.py` | Gemma-2-9B judge |
| `configs/base.yaml` | All model names, hyperparams |

All scripts set `CUDA_VISIBLE_DEVICES=1` and `HF_HOME=/home/ishana/scratch/hf_cache`
at the top. Do not change these — GPU 0 is occupied.

All scripts are **resumable**: delete the `_ckpt.pkl` file only if you want to
start completely fresh. Never needed under normal circumstances.
