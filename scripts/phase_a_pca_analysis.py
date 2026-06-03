"""Phase A: PCA structural analysis of query embeddings per class.

Runs PCA on each paraphrase and entity class, computes diagnostics,
saves results/pca_analysis.csv and results/pca_analysis_report.md,
and produces notebooks/05_pca_analysis.ipynb.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRATCH = Path('/home/ishana/scratch')
DATA    = SCRATCH / 'data' / 'classes'
PROJ    = Path('/home/ishana/projects/llm_jam_universal')
RESULTS = PROJ / 'results'
NB_DIR  = PROJ / 'notebooks'

RESULTS.mkdir(exist_ok=True)
NB_DIR.mkdir(exist_ok=True)

# ── Load raw data ──────────────────────────────────────────────────────────────
print("Loading class JSON and embeddings …")

with open(DATA / 'paraphrase_classes.json') as f:
    para_classes = json.load(f)
with open(DATA / 'entity_classes.json') as f:
    entity_classes = json.load(f)

# Shape: (100, 6, 1024) and (20, 8, 1024)
para_embs   = np.load(DATA / 'paraphrase_embeddings.npy').astype(np.float64)
entity_embs = np.load(DATA / 'entity_embeddings.npy').astype(np.float64)

print(f"  Paraphrase embeddings: {para_embs.shape}")
print(f"  Entity embeddings:     {entity_embs.shape}")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def mean_pairwise_cosine(E: np.ndarray) -> float:
    """E: (N, D) — compute mean pairwise cosine similarity."""
    E_norm = normalize(E, norm='l2')
    gram   = E_norm @ E_norm.T
    n      = E_norm.shape[0]
    pairs  = [(gram[i, j]) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(pairs)) if pairs else 1.0


def analyze_class(E: np.ndarray, class_id: str, class_type: str,
                  within_class_sim: float) -> dict:
    """Run PCA on N×1024 embedding matrix and return diagnostic dict."""
    n, d = E.shape
    assert n >= 2, f"Class {class_id} has only {n} query — need ≥ 2."

    # Max PCs we can meaningfully extract (rank of centered data ≤ n-1)
    max_pcs = min(n - 1, d)

    pca = PCA(n_components=max_pcs, svd_solver='full', random_state=42)
    pca.fit(E)
    evr = pca.explained_variance_ratio_  # shape: (max_pcs,)

    def evr_at(k: int) -> float:
        return float(evr[k - 1]) if k <= len(evr) else float('nan')

    def cumevr_at(k: int) -> float:
        return float(evr[:k].sum()) if k <= len(evr) else float(evr.sum())

    # PC1 vector
    pc1 = pca.components_[0]  # shape: (1024,)

    # Class centroid
    centroid = E.mean(axis=0)

    # Cosine similarity between PC1 direction and centroid direction
    pc1_centroid_cos = cosine_sim(pc1, centroid)

    # Intrinsic dimensionality: min PCs to explain >= 90% variance
    cumulative = np.cumsum(evr)
    idxs_90    = np.where(cumulative >= 0.90)[0]
    if len(idxs_90) == 0:
        intrinsic_dim_90 = max_pcs  # even all PCs don't reach 90%
    else:
        intrinsic_dim_90 = int(idxs_90[0]) + 1

    return {
        'class_id':             class_id,
        'class_type':           class_type,
        'n_queries':            n,
        'within_class_sim':     within_class_sim,
        'pc1_explained_var':    evr_at(1),
        'pc2_explained_var':    evr_at(2),
        'pc3_explained_var':    evr_at(3),
        'cumulative_var_top3':  cumevr_at(3),
        'cumulative_var_top5':  cumevr_at(5),
        'pc1_centroid_cosine':  pc1_centroid_cos,
        'intrinsic_dim_90':     intrinsic_dim_90,
        'max_pcs_available':    max_pcs,
    }


# ── Analyse paraphrase classes ─────────────────────────────────────────────────
print(f"\nAnalysing {len(para_classes)} paraphrase classes …")
para_records = []
for i, cls in enumerate(para_classes):
    cid   = cls.get('class_id', f'para_{i:03d}')
    # Original query + paraphrases
    n_q   = 1 + len(cls.get('paraphrases', []))
    E     = para_embs[i, :n_q]           # (n_q, 1024)
    wcs   = cls.get('within_class_similarity', mean_pairwise_cosine(E))
    rec   = analyze_class(E, cid, 'paraphrase', wcs)
    para_records.append(rec)
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(para_classes)} done")

# ── Analyse entity classes ─────────────────────────────────────────────────────
print(f"\nAnalysing {len(entity_classes)} entity classes …")
entity_records = []
for i, cls in enumerate(entity_classes):
    cid  = cls.get('class_id', f'entity_{i:02d}')
    size = cls.get('size', len(cls.get('queries', [])))
    E    = entity_embs[i, :size]          # (size, 1024)
    wcs  = cls.get('within_class_similarity', mean_pairwise_cosine(E))
    rec  = analyze_class(E, cid, 'entity', float(wcs))
    entity_records.append(rec)
    print(f"  {cid}: n={size}, PC1={rec['pc1_explained_var']:.3f}, "
          f"cos={rec['pc1_centroid_cosine']:.3f}")

# ── Build DataFrame and save CSV ───────────────────────────────────────────────
df = pd.DataFrame(para_records + entity_records)
csv_path = RESULTS / 'pca_analysis.csv'
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")


# ── Aggregate stats helper ─────────────────────────────────────────────────────
def stats_str(vals: np.ndarray, label: str) -> str:
    vals = vals[~np.isnan(vals)]
    return (f"  {label}: mean={vals.mean():.3f}  median={np.median(vals):.3f}  "
            f"min={vals.min():.3f}  max={vals.max():.3f}")


para_df   = df[df.class_type == 'paraphrase']
entity_df = df[df.class_type == 'entity']

# ── Gating criterion ──────────────────────────────────────────────────────────
med_pc1_entity   = float(np.median(entity_df['pc1_explained_var']))
med_cos_entity   = float(np.median(np.abs(entity_df['pc1_centroid_cosine'])))

if med_pc1_entity >= 0.40 and med_cos_entity <= 0.85:
    gating = "GREENLIGHT"
    gating_detail = (
        f"Median PC1 explained variance for entity classes = {med_pc1_entity:.3f} >= 0.40 "
        f"AND median |PC1-centroid cosine| = {med_cos_entity:.3f} <= 0.85. "
        "PC1 is a meaningful direction distinct from the centroid — Phase B is worth running."
    )
elif med_pc1_entity < 0.30 or med_cos_entity > 0.95:
    gating = "RED-LIGHT"
    gating_detail = (
        f"Median PC1 explained variance for entity classes = {med_pc1_entity:.3f} "
        f"(threshold <0.30 triggers red-light) "
        f"OR median |PC1-centroid cosine| = {med_cos_entity:.3f} "
        f"(threshold >0.95 triggers red-light). "
        "PC1 is either weak or redundant with the centroid — Phase B is unlikely to help."
    )
else:
    gating = "AMBIGUOUS"
    gating_detail = (
        f"Median PC1 explained variance for entity classes = {med_pc1_entity:.3f} "
        f"(between 0.30 and 0.40) "
        f"AND median |PC1-centroid cosine| = {med_cos_entity:.3f}. "
        "Results are ambiguous — let user decide on Phase B."
    )

print(f"\n{'='*60}")
print(f"PHASE A GATING VERDICT: {gating}")
print(gating_detail)
print('='*60)


# ── Build report ───────────────────────────────────────────────────────────────
report_lines = []
A = report_lines.append

A("# Phase A PCA Analysis Report\n")
A("## Overview\n")
A(f"- Paraphrase classes analysed: {len(para_df)}")
A(f"- Entity classes analysed: {len(entity_df)}")
A(f"- Embedding model: BGE-large (dim=1024)")
A(f"- PCA solver: sklearn full SVD\n")

A("## 1. PC1 Explained Variance\n")
A("### Paraphrase classes")
A(stats_str(para_df['pc1_explained_var'].values, "PC1 var"))
A(stats_str(para_df['pc2_explained_var'].values, "PC2 var"))
A(stats_str(para_df['pc3_explained_var'].values, "PC3 var"))
A(stats_str(para_df['cumulative_var_top3'].values, "Cum top-3"))
A(stats_str(para_df['cumulative_var_top5'].values, "Cum top-5\n"))

A("### Entity classes")
A(stats_str(entity_df['pc1_explained_var'].values, "PC1 var"))
A(stats_str(entity_df['pc2_explained_var'].values, "PC2 var"))
A(stats_str(entity_df['pc3_explained_var'].values, "PC3 var"))
A(stats_str(entity_df['cumulative_var_top3'].values, "Cum top-3"))
A(stats_str(entity_df['cumulative_var_top5'].values, "Cum top-5\n"))

A("## 2. Intrinsic Dimensionality (PCs needed for ≥90% variance)\n")
A("### Paraphrase classes")
A(stats_str(para_df['intrinsic_dim_90'].values, "Intrinsic dim"))
id_counts_p = para_df['intrinsic_dim_90'].value_counts().sort_index()
A(f"  Distribution: {id_counts_p.to_dict()}\n")

A("### Entity classes")
A(stats_str(entity_df['intrinsic_dim_90'].values, "Intrinsic dim"))
id_counts_e = entity_df['intrinsic_dim_90'].value_counts().sort_index()
A(f"  Distribution: {id_counts_e.to_dict()}\n")

A("## 3. PC1 vs Centroid Cosine Similarity\n")
A("(High cosine = PC1 is redundant with centroid; low = PC1 is a genuinely new direction)\n")
A("### Paraphrase classes")
A(stats_str(para_df['pc1_centroid_cosine'].values, "PC1-centroid cos"))
A(stats_str(np.abs(para_df['pc1_centroid_cosine'].values), "|cos|\n"))

A("### Entity classes")
A(stats_str(entity_df['pc1_centroid_cosine'].values, "PC1-centroid cos"))
A(stats_str(np.abs(entity_df['pc1_centroid_cosine'].values), "|cos|\n"))

A("## 4. Per-Class Breakdown (Entity Classes)\n")
A("```")
A(f"{'class_id':<12} {'entity':<20} {'n':>3} {'wcs':>6} {'PC1':>6} {'PC2':>6} {'cos':>7} {'dim90':>5}")
A("-" * 72)
for _, row in entity_df.sort_values('pc1_explained_var', ascending=False).iterrows():
    # Get primary entity name from JSON
    eid  = row['class_id']
    idx  = next((j for j, c in enumerate(entity_classes)
                 if c.get('class_id', f'entity_{j:02d}') == eid), -1)
    name = entity_classes[idx]['primary_entity'] if idx >= 0 else '?'
    A(f"{eid:<12} {name:<20} {int(row['n_queries']):>3} "
      f"{row['within_class_sim']:>6.3f} "
      f"{row['pc1_explained_var']:>6.3f} "
      f"{row['pc2_explained_var']:>6.3f} "
      f"{row['pc1_centroid_cosine']:>7.3f} "
      f"{int(row['intrinsic_dim_90']):>5}")
A("```\n")

A("## 5. Verdict: Does PC1 Capture Meaningful Structure?\n")
A("**Paraphrase classes:**")
A(f"  - Median PC1 explained variance: {np.median(para_df['pc1_explained_var']):.3f}")
A(f"  - With only 6 queries per class, the embedding matrix is low-rank (max 5 PCs).")
A(f"  - High PC1 variance is expected when queries are close paraphrases.")
A("")
A("**Entity classes:**")
A(f"  - Median PC1 explained variance: {med_pc1_entity:.3f}")
A(f"  - Median |PC1-centroid cosine|: {med_cos_entity:.3f}")
A(f"  - Entity classes have 3–8 diverse queries about the same entity,")
A(f"    so we ask whether a dominant direction exists beyond the centroid.")
A("")

if gating == "RED-LIGHT":
    A("**Honest verdict:** Entity query embeddings are largely isotropic around their centroid. "
      "PC1 captures only a small fraction of variance, and/or it is nearly parallel to the "
      "centroid direction — providing no new signal beyond 'centroid-aligned'. This is "
      "consistent with the geometric impossibility argument: diverse entity queries do not "
      "share a dominant secondary direction in embedding space that could be exploited for "
      "better retrieval. A single-document blocker cannot simultaneously align with all query "
      "directions if there is no such shared direction.\n")
elif gating == "GREENLIGHT":
    A("**Honest verdict:** Entity query embeddings show meaningful dominant structure. "
      "PC1 explains a substantial fraction of variance AND is distinct from the centroid "
      "direction. This suggests that there may be a shared secondary direction across "
      "entity queries that a blocker could exploit. Phase B is warranted.\n")
else:
    A("**Honest verdict:** Results are in the ambiguous zone. PC1 variance is moderate and "
      "the PC1-centroid relationship is unclear. Phase B is uncertain — consult the "
      "per-class breakdown and decide whether the signal is strong enough to pursue.\n")

A("## 6. Phase A Gating Decision\n")
A(f"**Verdict: {gating}**\n")
A(gating_detail)
A("")
if gating != "GREENLIGHT":
    A("**DO NOT proceed to Phase B automatically.** The Phase A result is the paper-relevant "
      "finding: entity classes lack a dominant principal direction that is geometrically "
      "distinct from the centroid, consistent with the impossibility bound.\n")

report_path = RESULTS / 'pca_analysis_report.md'
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))
print(f"Saved {report_path}")


# ── Print key numbers to stdout ────────────────────────────────────────────────
print("\n=== KEY NUMBERS ===")
print(f"Paraphrase PC1 var: mean={para_df['pc1_explained_var'].mean():.3f}  "
      f"median={np.median(para_df['pc1_explained_var']):.3f}")
print(f"Entity     PC1 var: mean={entity_df['pc1_explained_var'].mean():.3f}  "
      f"median={np.median(entity_df['pc1_explained_var']):.3f}")
print(f"Paraphrase |PC1-centroid|: mean={np.abs(para_df['pc1_centroid_cosine']).mean():.3f}  "
      f"median={np.median(np.abs(para_df['pc1_centroid_cosine'])):.3f}")
print(f"Entity     |PC1-centroid|: mean={np.abs(entity_df['pc1_centroid_cosine']).mean():.3f}  "
      f"median={np.median(np.abs(entity_df['pc1_centroid_cosine'])):.3f}")
print(f"Paraphrase intrinsic dim90: mean={para_df['intrinsic_dim_90'].mean():.2f}  "
      f"median={np.median(para_df['intrinsic_dim_90']):.1f}")
print(f"Entity     intrinsic dim90: mean={entity_df['intrinsic_dim_90'].mean():.2f}  "
      f"median={np.median(entity_df['intrinsic_dim_90']):.1f}")


# ── Write Jupyter notebook ─────────────────────────────────────────────────────
print("\nWriting notebook 05_pca_analysis.ipynb …")

import json as _json

notebook_cells = []

def md_cell(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }

notebook_cells.append(md_cell(
    "# Phase A — PCA Structural Analysis of Query Embeddings\n\n"
    "Diagnostic: do entity-class query embeddings have a dominant principal direction "
    "(PC1) that is meaningfully distinct from the class centroid?\n\n"
    "**Gating criterion:**\n"
    "- GREENLIGHT Phase B if median entity PC1 variance ≥ 0.40 AND median |PC1-centroid cos| ≤ 0.85\n"
    "- RED-LIGHT if median entity PC1 variance < 0.30 OR median |PC1-centroid cos| > 0.95\n"
    "- Otherwise: ambiguous, report to user."
))

setup_code = """\
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

SCRATCH  = Path('/home/ishana/scratch')
DATA     = SCRATCH / 'data' / 'classes'
PROJ     = Path('/home/ishana/projects/llm_jam_universal')
RESULTS  = PROJ / 'results'
FIGS     = SCRATCH / 'results'

plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

df = pd.read_csv(RESULTS / 'pca_analysis.csv')
para_df   = df[df.class_type == 'paraphrase']
entity_df = df[df.class_type == 'entity']

print(f"Paraphrase classes: {len(para_df)}")
print(f"Entity classes:     {len(entity_df)}")
print(df.dtypes)
"""
notebook_cells.append(code_cell(setup_code))

notebook_cells.append(md_cell("## 1. Summary Statistics"))

stats_code = """\
print("=== PC1 Explained Variance ===")
for ct, sub in [('paraphrase', para_df), ('entity', entity_df)]:
    v = sub['pc1_explained_var'].values
    print(f"  {ct}: mean={v.mean():.3f}  median={np.median(v):.3f}  "
          f"min={v.min():.3f}  max={v.max():.3f}")

print()
print("=== |PC1-centroid cosine| ===")
for ct, sub in [('paraphrase', para_df), ('entity', entity_df)]:
    v = np.abs(sub['pc1_centroid_cosine'].values)
    print(f"  {ct}: mean={v.mean():.3f}  median={np.median(v):.3f}  "
          f"min={v.min():.3f}  max={v.max():.3f}")

print()
print("=== Intrinsic dim (90% var) ===")
for ct, sub in [('paraphrase', para_df), ('entity', entity_df)]:
    v = sub['intrinsic_dim_90'].values
    print(f"  {ct}: mean={v.mean():.2f}  median={np.median(v):.1f}  "
          f"min={v.min():.0f}  max={v.max():.0f}")

# Gating
med_pc1   = float(np.median(entity_df['pc1_explained_var']))
med_cos   = float(np.median(np.abs(entity_df['pc1_centroid_cosine'])))
print(f"\\nGating: entity median PC1={med_pc1:.3f}, |cos|={med_cos:.3f}")
if med_pc1 >= 0.40 and med_cos <= 0.85:
    print("  → GREENLIGHT")
elif med_pc1 < 0.30 or med_cos > 0.95:
    print("  → RED-LIGHT")
else:
    print("  → AMBIGUOUS")
"""
notebook_cells.append(code_cell(stats_code))

notebook_cells.append(md_cell(
    "## 2. Main Figure: Explained-Variance Histograms\n\n"
    "Paraphrase vs Entity classes, side by side. Candidate paper figure."
))

fig_code = """\
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

COLORS = {'paraphrase': '#1f77b4', 'entity': '#d62728'}

# ─ PC1 explained variance ─────────────────────────────────────────────────────
ax = axes[0]
bins = np.linspace(0, 1, 21)
for ct, sub in [('paraphrase', para_df), ('entity', entity_df)]:
    v   = sub['pc1_explained_var'].values
    med = np.median(v)
    ax.hist(v, bins=bins, color=COLORS[ct], alpha=0.6, label=f'{ct} (n={len(sub)})', density=False)
    ax.axvline(med, color=COLORS[ct], lw=2, ls='--', label=f'{ct} median={med:.2f}')
ax.set_xlabel('PC1 explained variance')
ax.set_ylabel('Count')
ax.set_title('PC1 Explained Variance')
ax.legend(fontsize=8)

# ─ Intrinsic dimensionality ───────────────────────────────────────────────────
ax = axes[1]
max_dim = max(df['intrinsic_dim_90'].max(), 5)
bins_id = np.arange(0.5, max_dim + 1.5)
width   = 0.35
for offset, (ct, sub) in zip([-width/2, width/2],
                               [('paraphrase', para_df), ('entity', entity_df)]):
    v     = sub['intrinsic_dim_90'].values
    vals, cnts = np.unique(v, return_counts=True)
    ax.bar(vals + offset, cnts, width=width, color=COLORS[ct],
           alpha=0.7, label=f'{ct} (med={np.median(v):.0f})')
ax.set_xlabel('Intrinsic dim (PCs for ≥90% var)')
ax.set_ylabel('Count')
ax.set_title('Intrinsic Dimensionality')
ax.legend(fontsize=8)
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# ─ |PC1-centroid cosine| ──────────────────────────────────────────────────────
ax = axes[2]
bins_c = np.linspace(0, 1, 21)
for ct, sub in [('paraphrase', para_df), ('entity', entity_df)]:
    v   = np.abs(sub['pc1_centroid_cosine'].values)
    med = np.median(v)
    ax.hist(v, bins=bins_c, color=COLORS[ct], alpha=0.6, label=f'{ct}', density=False)
    ax.axvline(med, color=COLORS[ct], lw=2, ls='--', label=f'{ct} median={med:.2f}')
ax.set_xlabel('|PC1 · centroid| cosine')
ax.set_ylabel('Count')
ax.set_title('PC1 vs Centroid Alignment\\n(high = PC1 redundant with centroid)')
ax.legend(fontsize=8)

plt.suptitle('Phase A — PCA Structural Analysis of Query Classes', fontsize=13)
plt.tight_layout()
fig.savefig(FIGS / 'pca_analysis_histograms.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved pca_analysis_histograms.png")
"""
notebook_cells.append(code_cell(fig_code))

notebook_cells.append(md_cell("## 3. Per-Class Breakdown (Entity)"))

entity_table_code = """\
with open(DATA / 'entity_classes.json') as f:
    entity_json = json.load(f)
eid_to_name = {c.get('class_id', f'entity_{i:02d}'): c['primary_entity']
               for i, c in enumerate(entity_json)}

edf = entity_df.copy()
edf['entity'] = edf['class_id'].map(eid_to_name)
edf = edf.sort_values('pc1_explained_var', ascending=False)
display_cols = ['class_id', 'entity', 'n_queries', 'within_class_sim',
                'pc1_explained_var', 'pc2_explained_var',
                'pc1_centroid_cosine', 'intrinsic_dim_90']
print(edf[display_cols].to_string(index=False, float_format='{:.3f}'.format))
"""
notebook_cells.append(code_cell(entity_table_code))

notebook_cells.append(md_cell("## 4. Scatter: Within-Class Sim vs PC1 Variance"))

scatter_code = """\
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, (y_col, y_lbl) in zip(axes, [
    ('pc1_explained_var',   'PC1 explained variance'),
    ('pc1_centroid_cosine', 'PC1-centroid cosine'),
]):
    for ct, sub in [('paraphrase', para_df), ('entity', entity_df)]:
        ax.scatter(sub['within_class_sim'], sub[y_col],
                   color=COLORS[ct], alpha=0.6, label=ct, s=40)
    ax.set_xlabel('Within-class cosine similarity')
    ax.set_ylabel(y_lbl)
    ax.set_title(y_lbl + ' vs within-class sim')
    ax.legend(fontsize=9)

plt.suptitle('Phase A — PCA vs Within-Class Similarity', fontsize=13)
plt.tight_layout()
fig.savefig(FIGS / 'pca_vs_sim_scatter.png', dpi=150, bbox_inches='tight')
plt.show()
"""
notebook_cells.append(code_cell(scatter_code))

notebook_cells.append(md_cell("## 5. Gating Verdict"))

gating_code = """\
med_pc1 = float(np.median(entity_df['pc1_explained_var']))
med_cos = float(np.median(np.abs(entity_df['pc1_centroid_cosine'])))

print(f"Entity median PC1 explained variance : {med_pc1:.3f}")
print(f"Entity median |PC1-centroid cosine|  : {med_cos:.3f}")
print()
if med_pc1 >= 0.40 and med_cos <= 0.85:
    verdict = "GREENLIGHT"
    reason  = "PC1 is strong and distinct from centroid — Phase B is worth running."
elif med_pc1 < 0.30 or med_cos > 0.95:
    verdict = "RED-LIGHT"
    reason  = ("PC1 is weak or redundant with centroid — Phase B unlikely to help. "
               "Phase A alone is the paper-relevant result.")
else:
    verdict = "AMBIGUOUS"
    reason  = "In between the thresholds — consult user before proceeding."

print(f"PHASE A VERDICT: {verdict}")
print(reason)
"""
notebook_cells.append(code_cell(gating_code))

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    },
    "cells": notebook_cells,
}

nb_path = NB_DIR / '05_pca_analysis.ipynb'
with open(nb_path, 'w') as f:
    _json.dump(notebook, f, indent=1)
print(f"Saved {nb_path}")
print("\nPhase A complete.")
