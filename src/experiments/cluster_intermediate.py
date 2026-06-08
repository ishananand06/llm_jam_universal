"""
Cluster the 2279-query NQ answerable subset to find intermediate-similarity
query classes that fill the gap between paraphrase (~0.90) and entity (~0.56).

Outputs
-------
/home/ishana/scratch/results/intermediate_class_candidates.md
/home/ishana/scratch/results/intermediate_class_candidates.csv
/home/ishana/scratch/data/classes/answerable_bge_embeddings.npy  (cache)
"""
from __future__ import annotations

import csv
import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

os.environ["HF_HOME"] = "/home/ishana/scratch/hf_cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCRATCH       = Path("/home/ishana/scratch")
DATA_DIR      = SCRATCH / "data"
RESULTS_DIR   = SCRATCH / "results"
PROJECT_RESULTS = Path("/home/ishana/projects/llm_jam_universal/results")

QUERIES_FILE  = DATA_DIR / "nq/queries.jsonl"
ANSWERABLE_PKL = DATA_DIR / "answerable_nq_test.pkl"
EMB_CACHE     = DATA_DIR / "classes/answerable_bge_embeddings.npy"
ID_CACHE      = DATA_DIR / "classes/answerable_bge_ids.json"

BGE_MODEL     = "BAAI/bge-large-en-v1.5"

# Target mean within-cluster cosine similarity bands
TARGET_BANDS  = [0.65, 0.72, 0.80]
SIZE_MIN, SIZE_MAX = 3, 8

# Distance threshold search range (cosine distance = 1 - cosine sim)
THRESH_RANGE  = np.arange(0.05, 0.55, 0.01)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_queries() -> tuple[list[str], list[str]]:
    """Return (query_ids, query_texts) for the 2279 answerable subset, sorted by ID."""
    with open(ANSWERABLE_PKL, "rb") as f:
        answerable_ids: set[str] = pickle.load(f)
    with open(QUERIES_FILE) as f:
        id2text = {json.loads(l)["_id"]: json.loads(l)["text"] for l in f}
    pairs = sorted(
        [(qid, id2text[qid]) for qid in answerable_ids if qid in id2text],
        key=lambda x: x[0],
    )
    ids   = [p[0] for p in pairs]
    texts = [p[1] for p in pairs]
    log.info("Loaded %d answerable queries", len(ids))
    return ids, texts


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_queries(texts: list[str], ids: list[str]) -> np.ndarray:
    """Return L2-normalised BGE-large embeddings, loading from cache if available."""
    if EMB_CACHE.exists() and ID_CACHE.exists():
        cached_ids = json.loads(ID_CACHE.read_text())
        if cached_ids == ids:
            log.info("Loading embeddings from cache: %s", EMB_CACHE)
            embs = np.load(EMB_CACHE)
            log.info("Cache hit — shape %s", embs.shape)
            return embs
        log.warning("Cache ID mismatch — re-embedding")

    log.info("Embedding %d queries with %s ...", len(texts), BGE_MODEL)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(BGE_MODEL)
    embs = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    log.info("Embedding done — shape %s", embs.shape)

    EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMB_CACHE, embs)
    ID_CACHE.write_text(json.dumps(ids))
    log.info("Embeddings cached → %s", EMB_CACHE)
    return embs


# ── Clustering ────────────────────────────────────────────────────────────────

def mean_pairwise_sim(embs: np.ndarray, idxs: list[int]) -> float:
    """Mean pairwise cosine similarity for a set of (L2-normalised) embeddings."""
    if len(idxs) < 2:
        return 1.0
    sub = embs[idxs]
    sims = sub @ sub.T
    n = len(idxs)
    upper = [sims[r, c] for r in range(n) for c in range(r + 1, n)]
    return float(np.mean(upper))


def cluster_at_threshold(Z: np.ndarray, n: int, thresh: float) -> dict[int, list[int]]:
    """Return {cluster_label: [row_indices]} for complete-linkage cut at thresh."""
    labels = fcluster(Z, t=thresh, criterion="distance")
    clusters: dict[int, list[int]] = {}
    for i, lbl in enumerate(labels):
        clusters.setdefault(lbl, []).append(i)
    return clusters


def sweep_thresholds(
    embs: np.ndarray,
    Z: np.ndarray,
    targets: list[float],
    size_min: int,
    size_max: int,
) -> dict[float, tuple[float, list[dict]]]:
    """
    For each target mean within-cluster similarity, find the threshold that
    produces kept clusters (size in [size_min, size_max]) with actual mean
    similarity closest to target.

    Returns {target_sim: (best_thresh, [cluster_dicts])}
    where cluster_dicts have keys: cluster_id, size, actual_sim, member_indices
    """
    n = embs.shape[0]

    # For each threshold, compute kept-cluster mean sim
    log.info("Sweeping %d thresholds ...", len(THRESH_RANGE))
    thresh_stats: list[tuple[float, float, list[dict]]] = []
    for thresh in THRESH_RANGE:
        raw = cluster_at_threshold(Z, n, thresh)
        kept = []
        for lbl, idxs in raw.items():
            if size_min <= len(idxs) <= size_max:
                sim = mean_pairwise_sim(embs, idxs)
                kept.append({
                    "cluster_id": int(lbl),
                    "size": len(idxs),
                    "actual_sim": sim,
                    "member_indices": idxs,
                })
        if kept:
            mean_sim = float(np.mean([c["actual_sim"] for c in kept]))
        else:
            mean_sim = 0.0
        thresh_stats.append((thresh, mean_sim, kept))
        log.debug("thresh=%.2f  kept=%d  mean_sim=%.3f", thresh, len(kept), mean_sim)

    results: dict[float, tuple[float, list[dict]]] = {}
    for target in targets:
        best = min(thresh_stats, key=lambda x: abs(x[1] - target))
        log.info(
            "Target sim=%.2f → best thresh=%.2f  actual_mean=%.3f  kept_clusters=%d",
            target, best[0], best[1], len(best[2]),
        )
        results[target] = (best[0], best[2])
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def ascii_histogram(values: list[float], bins: int = 10, width: int = 40) -> str:
    if not values:
        return "(no data)"
    lo, hi = min(values), max(values)
    if lo == hi:
        return f"All values = {lo:.3f}"
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        b = min(int((v - lo) / step), bins - 1)
        counts[b] += 1
    max_count = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        lo_b = lo + i * step
        hi_b = lo_b + step
        bar = "█" * int(c / max_count * width)
        lines.append(f"  {lo_b:.3f}–{hi_b:.3f} | {bar} {c}")
    return "\n".join(lines)


def write_markdown(
    band_results: dict[float, tuple[float, list[dict]]],
    ids: list[str],
    texts: list[str],
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Intermediate-Similarity Query Class Candidates")
    lines.append("")
    lines.append("Agglomerative clustering (cosine distance, complete linkage) of the")
    lines.append("2279-query NQ answerable subset. Three distance thresholds chosen to")
    lines.append("target mean within-cluster cosine similarity ≈ 0.65, 0.72, 0.80.")
    lines.append("Only clusters with size in [3, 8] are kept (matching entity class range).")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Target sim | Best thresh | Kept clusters | Actual mean sim | Sim range |")
    lines.append("|---|---|---|---|---|")
    for target in sorted(band_results):
        thresh, kept = band_results[target]
        if kept:
            sims = [c["actual_sim"] for c in kept]
            lines.append(
                f"| {target:.2f} | {thresh:.2f} | {len(kept)} | "
                f"{np.mean(sims):.3f} | {min(sims):.3f}–{max(sims):.3f} |"
            )
        else:
            lines.append(f"| {target:.2f} | {thresh:.2f} | **0** | — | — |")
    lines.append("")

    # Per-band detail
    total_clusters = 0
    for target in sorted(band_results):
        thresh, kept = band_results[target]
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"## Band: target sim ≈ {target:.2f}  (thresh={thresh:.2f})")
        lines.append("")

        if not kept:
            lines.append("*No clusters of size 3–8 found at this threshold.*")
            lines.append("")
            continue

        sims = [c["actual_sim"] for c in kept]
        lines.append(f"**{len(kept)} clusters kept** — "
                     f"actual sim {min(sims):.3f}–{max(sims):.3f}, "
                     f"mean {np.mean(sims):.3f}")
        lines.append("")

        # ASCII histogram
        lines.append("### Within-cluster similarity distribution")
        lines.append("")
        lines.append("```")
        lines.append(ascii_histogram(sims))
        lines.append("```")
        lines.append("")

        # Individual clusters, sorted by actual_sim descending
        lines.append("### Clusters")
        lines.append("")
        for c in sorted(kept, key=lambda x: -x["actual_sim"]):
            cid = c["cluster_id"]
            sim = c["actual_sim"]
            sz  = c["size"]
            midxs = c["member_indices"]
            lines.append(f"#### Cluster {cid} — actual within-sim {sim:.3f}, size {sz}")
            lines.append("")
            for rank, idx in enumerate(midxs, 1):
                lines.append(f"{rank}. {texts[idx]}")
            lines.append("")
            total_clusters += 1

    # Verdict placeholder
    lines.append("---")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"Total kept clusters across all bands: **{total_clusters}**.")
    lines.append("")
    lines.append("*(Manual inspection required — read each cluster above and judge whether*")
    lines.append("*the queries form a genuinely coherent 'related but not identical' group,*")
    lines.append("*or whether the clustering is mixing unrelated topics.)*")
    lines.append("")
    lines.append("Fill in after review:")
    lines.append("")
    lines.append("- Coherent clusters: TBD")
    lines.append("- Incoherent / mixed clusters: TBD")
    lines.append("- Overall verdict: TBD")

    out_path.write_text("\n".join(lines))
    log.info("Markdown written → %s", out_path)


def write_csv(
    band_results: dict[float, tuple[float, list[dict]]],
    ids: list[str],
    out_path: Path,
) -> None:
    fieldnames = ["cluster_id", "target_band", "best_thresh", "actual_sim", "size", "query_indices", "query_ids"]
    rows = []
    for target in sorted(band_results):
        thresh, kept = band_results[target]
        for c in kept:
            midxs = c["member_indices"]
            rows.append({
                "cluster_id":    c["cluster_id"],
                "target_band":   f"{target:.2f}",
                "best_thresh":   f"{thresh:.2f}",
                "actual_sim":    round(c["actual_sim"], 5),
                "size":          c["size"],
                "query_indices": ";".join(str(i) for i in midxs),
                "query_ids":     ";".join(ids[i] for i in midxs),
            })
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("CSV written → %s  (%d rows)", out_path, len(rows))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ids, texts = load_queries()
    embs = embed_queries(texts, ids)  # (N, 1024), L2-normalised

    # Build linkage matrix (complete linkage, cosine distance)
    log.info("Computing pairwise cosine distances for %d queries ...", len(ids))
    # cosine distance on L2-normalised vectors = 1 - dot product
    cos_dist = 1.0 - (embs @ embs.T)
    np.clip(cos_dist, 0, 2, out=cos_dist)  # numerical safety
    condensed = cos_dist[np.triu_indices(len(ids), k=1)]
    log.info("Building complete-linkage dendrogram ...")
    Z = linkage(condensed, method="complete")
    log.info("Dendrogram built.")

    band_results = sweep_thresholds(embs, Z, TARGET_BANDS, SIZE_MIN, SIZE_MAX)

    # Log per-band summary before writing
    for target, (thresh, kept) in sorted(band_results.items()):
        if kept:
            sims = [c["actual_sim"] for c in kept]
            sizes = [c["size"] for c in kept]
            log.info(
                "Band %.2f: %d clusters | sim %.3f–%.3f (mean %.3f) | "
                "sizes %d–%d",
                target, len(kept), min(sims), max(sims), np.mean(sims),
                min(sizes), max(sizes),
            )
        else:
            log.info("Band %.2f: 0 clusters found", target)

    md_path  = RESULTS_DIR / "intermediate_class_candidates.md"
    csv_path = RESULTS_DIR / "intermediate_class_candidates.csv"

    write_markdown(band_results, ids, texts, md_path)
    write_csv(band_results, ids, csv_path)

    # Also copy md to project results for git tracking
    project_md = PROJECT_RESULTS / "intermediate_class_candidates.md"
    project_md.write_text(md_path.read_text())
    log.info("Report also copied → %s", project_md)


if __name__ == "__main__":
    main()
