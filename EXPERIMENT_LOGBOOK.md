# EXPERIMENT_LOGBOOK.md

One entry per completed experiment. Record the decision that motivated it, the
key finding, and what it implies for the next step. Do not re-summarise data
that already lives in `results/` — link to the report instead.

**Rule:** append one entry here at the end of every session, after updating
SESSION_HANDOFF.md.

---

## Task 1 — Class construction and validation
**Report:** `results/task1_class_validation.md`

**Decision:** Build two class types with different within-class similarity regimes
to characterise the full range of the geometric bound.

**Finding:** Paraphrase classes (Llama-generated) have mean within-class sim 0.897
(BGE-large); entity classes (spaCy NER grouping) have mean 0.559. Llama paraphrases
beat Mistral on quality (higher min and mean sim). Final counts: 100 paraphrase classes
(6 queries each), 20 entity classes (3–8 queries each).

**Decision made:** Keep Llama paraphrases as canonical. Both class sets are final and
will not be regenerated.

---

## Task 3 — M1 retrieval baseline (representative query as blocker prefix)
**Report:** `results/task3_m1_report.md`

**Decision:** Before running BBO, measure how often the naive blocker
(`representative_query. !!!…!!!`) appears in honest top-5 retrieval.

**Finding:** Paraphrase simple: 40%, HotFlip: 58%. Entity simple: 13.8%, HotFlip: 7.3%
(HotFlip hurts entity). The `!` suffix dilutes the embedding; optimised suffixes will do
better, but entity classes face a structural ~0.22 cosine gap between the blocker and
the retrieval threshold.

**Decision made:** Use simple representative query (not HotFlip) as the d_r prefix for
all subsequent BBO experiments. HotFlip adds cost and degrades entity performance.

---

## Task 4 — M2 class-averaged BBO vs Shafran baseline (paraphrase, force-injection ASR)
**Report:** `results/task4_m2_paraphrase_report.md`

**Finding:** M2 73.3% vs Shafran 75.0% ASR; McNemar p=0.856 — no significant
difference. M2 does not improve over single-query BBO on paraphrase classes.

**Critical caveat (discovered in Step 0):** These ASR numbers are inflated ~35pp by
force-injection evaluation. They measure P(LLM refuses | blocker injected), not honest
end-to-end attack success. Do not cite these numbers without qualification.

---

## Step 0 — Retrieval audit and honest ASR correction
**Report:** `results/step0_retrieval_report.md`

**Finding:** The Task 4 evaluation force-injected the blocker regardless of whether it
won real top-5 retrieval. Honest ASR (requires actual top-5 retrieval) is 39.2% for M2
and 38.3% for Shafran on 20 paraphrase classes — inflated by ~35pp. McNemar p=1.00.

**Decision made:** All subsequent scripts use `jammed_honest = retrieved_top5 AND
judge_refused` as the sole reported metric. The 0.3 proxy threshold in ShafranBBO is
acceptable for the BBO hot path only; final evaluation always uses real FAISS retrieval.

---

## Task 5 — Entity class honest ASR and collapse characterisation
**Report:** `results/task5_entity_collapse_report.md`

**Finding:** Entity honest ASR = 13.8% vs paraphrase 38.3% — a 24.5pp collapse.
Decomposition: P(retrieved) fell from 50.8% to 15.6% (−35pp); P(jam|retrieved) was
*higher* for entity (88.2%) than paraphrase (75.4%). **The collapse is entirely
retrieval-driven; the jamming component is unaffected by class diversity.**

**Decision made:** The geometric bound is confirmed. Improving retrieval (not jamming) is
the productive attack direction for entity classes. Joint retrieval+jamming BBO is the
next logical step.

---

## PCA structural analysis (Phase A)
**Report:** `results/pca_analysis_report.md`

**Decision:** Test whether entity query embeddings have a dominant secondary direction
(PC1) that a blocker could exploit to cover more than just the centroid.

**Finding:** After accounting for a class-size artifact (small classes mechanically inflate
PC1 variance), entity classes with n≥6 show median PC1 = 0.243 — below the 0.30
red-light threshold. PC1 is near-orthogonal to the centroid (|cos| < 0.05) for all 120
classes, which is expected in high-dimensional PCA and provides no exploitable signal.
Entity query embeddings scatter across 4–6 independent dimensions with no dominant
secondary direction.

**Decision made:** Phase B (PC1-aligned blocker attack) was not pursued — Phase A alone
is the paper-relevant result. It closes the "use PCA structure to escape the bound"
objection.

---

## Task 6 — Constrained-joint BBO black-box baseline
**Report:** none committed (results in `/home/ishana/scratch/results/task6_constrained_joint_honest.csv`)

**Decision:** Replace the 0.3 proxy retrieval gate in ShafranBBO with the real FAISS
top-5 threshold during BBO (ConstrainedJointBBO), and sweep paraphrase + entity classes.

**Finding:** Constrained-joint BBO (real threshold gate) yields paraphrase ASR ~44%,
entity ASR ~11% — marginally better than unconstrained Shafran on paraphrase, essentially
identical on entity. The real threshold gate slightly improves blocker quality by forcing
the optimizer to keep the blocker retrievable throughout training.

**Decision made:** Constrained-joint BBO is the canonical black-box single-document
baseline for all subsequent comparisons. This is the bar the white-box experiments
must beat.

---

## Hubness analysis (Experiment 1A/1B in context document)
**Report:** not yet committed to `results/`

**Finding (Part A — hubness magnitude):** Max N_5 = 4 across 2.68M corpus documents.
Skewness = 13.06 (GTR), 13.25 (BGE). High skewness reflects sparsity of the retrievable
region (99.37% of docs have N_5=0), not a hub concentration. Top-10 hub overlap between
GTR and BGE: 2/10. Hub identity is embedder-specific.

**Finding (Part B — cluster structure):** HDBSCAN (min_size=5) finds 337 clusters in
GTR space; 78.3% of retrievable docs are noise (not in any cluster). Mean cluster size
~11 docs. Of each hub's 20 nearest neighbors, only ~15% are themselves retrievable. No
connected "hot zone" exists.

**Decision made:** Hubness-exploitation objection is foreclosed. The bound is not
vulnerable to a hub-targeting attack variant.

---

## White-box GTR pilot (3-class, original)
**Report:** `/home/ishana/scratch/results/whitebox_pilot_report.md`
**Checkpoint:** `/home/ishana/scratch/results/whitebox_pilot_ckpt.pkl`
**Script:** `src/experiments/eval_whitebox_gtr_pilot.py`

**Decision:** Relax the threat model to white-box GTR access (keeping Mistral black-box)
to test whether the bound is optimization-constrained or geometry-constrained. Combined
with vec2text-perturbed initialization as a smarter starting point.

**Finding:**
- Paraphrase: baseline 8/18 (44%) → best (λ=0.5) 11/18 (61%), +17pp.
  **But** the entire gain comes from one class (para_test2: 3/6 → 6/6). One class
  (para_test1) is 0/6 under all conditions.
- Entity: 2/19 → 2–3/19 across all λ — within noise. The bound is invariant to
  white-box retriever access for diverse classes.
- Trajectory analysis: GTR cosine rose (0.821→0.954) while BGE cosine fell
  (0.854→0.806) over 500 iters. Mean GTR–BGE gap = +0.058. The BGE oracle would
  have rejected the tokens white-box GTR ultimately selected — direct evidence the
  oracle was suboptimal.
- The vec2text perturbation (L2=0.05) was too small: the inversion returned ~q*
  unchanged (init_gtr_cos=0.821, already rank 2 at iter 0). The perturbation idea
  did not contribute a novel starting point.

**Decision made:** The +17pp paraphrase gain needs verification across more classes
(Exp 1 scale-up). The init vs. loss-term contribution needs isolation (Exp 2 ablation).
λ=0.5 is the pilot-optimal value; no further λ sweep needed for these follow-ups.

---

## Exp 1 — White-box GTR paraphrase scale-up (10 new classes) — IN PROGRESS
**Script:** `src/experiments/eval_whitebox_para_scaleup.py`
**Output:** `/home/ishana/scratch/results/whitebox_para_scaleup_honest.csv`
**Report:** `/home/ishana/scratch/results/whitebox_para_scaleup_report.md` (when done)
**tmux:** `wb_scaleup`

**Decision:** The pilot's +17pp paraphrase gain was driven by one of three classes.
Test on 10 new classes (test9–test20, never seen in the pilot) at λ=0.5 only, with a
self-contained black-box baseline on the same classes.

**Status:** Running as of 2026-06-16. Phase 1 (vec2text init) complete for all 10
classes. Phase 2a (WB BBO) complete for test9, test10 as of handoff. See
SESSION_HANDOFF.md for live progress.

*(Append finding and decision here once complete.)*

---

## Exp 2 — Init ablation: white-box loss only, random `!` init — PENDING
**Script:** `src/experiments/eval_whitebox_ablation_init.py`
**Output:** `/home/ishana/scratch/results/whitebox_ablation_init_honest.csv`
**Report:** `/home/ishana/scratch/results/whitebox_ablation_init_report.md` (when done)
**tmux:** `wb_ablation` (auto-starts after Exp 1 completes)

**Decision:** The white-box pilot changed two things simultaneously: the init (vec2text
perturbed) and the loss (GTR joint objective). The trajectory analysis suggests the init
did nothing, but the causal claim needs a clean ablation. This run holds λ=0.5 and the
GTR loss term fixed; only the init changes (50 × `!` instead of vec2text).

**Status:** Waiting for Exp 1 to finish. Will auto-start when
`whitebox_para_scaleup_report.md` is created.

*(Append finding and decision here once complete.)*
