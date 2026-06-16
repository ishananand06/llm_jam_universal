"""
Hubness analysis on the NQ corpus retrieval space.

Runs top-5 retrieval for all NQ test queries across three settings:
  1. GTR-base  (victim retriever, existing FAISS index reused)
  2. BGE-large-en-v1.5  (attacker oracle, builds index if absent)
  3. Cross-space comparison  (post-hoc: do BGE hubs appear in GTR hub list?)

For each space, counts N_5(d) = number of times document d appears in any
query's top-5 result set.  Reports skewness, top-20 hubs, ASCII histogram.

Usage
-----
    nohup python hubness_test.py > /home/ishana/scratch/results/hubness.log 2>&1 &

Output
------
    /home/ishana/scratch/results/hubness_gtr.csv
    /home/ishana/scratch/results/hubness_gtr.md
    /home/ishana/scratch/results/hubness_bge.csv
    /home/ishana/scratch/results/hubness_bge.md
    /home/ishana/scratch/results/hubness_summary.md
"""
from __future__ import annotations

import csv
import gc
import json
import logging
import os
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

os.environ["HF_HOME"] = "/home/ishana/scratch/hf_cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH      = Path("/home/ishana/scratch")
DATA_DIR     = SCRATCH / "data"
RESULTS_DIR  = SCRATCH / "results"
NQ_DIR       = DATA_DIR / "nq"
GTR_IDX_DIR  = DATA_DIR / "indices/nq/sentence-transformers__gtr-t5-base"
BGE_IDX_DIR  = DATA_DIR / "indices/nq/BAAI__bge-large-en-v1.5"

GTR_MODEL = "sentence-transformers/gtr-t5-base"
BGE_MODEL = "BAAI/bge-large-en-v1.5"
TOP_K     = 5

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_queries() -> list[dict]:
    queries = []
    with open(NQ_DIR / "queries.jsonl") as f:
        for line in f:
            queries.append(json.loads(line))
    log.info("Loaded %d queries", len(queries))
    return queries


def load_corpus_texts() -> tuple[list[str], list[str]]:
    """Returns (doc_ids, doc_texts) in index order."""
    with open(GTR_IDX_DIR / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    doc_ids = meta["doc_ids"]
    corpus  = meta["corpus"]
    texts   = [corpus[d] for d in doc_ids]
    log.info("Corpus: %d documents", len(doc_ids))
    return doc_ids, texts


def embed_batch(model, texts: list[str], batch_size: int = 2048) -> np.ndarray:
    """Embed texts in batches; returns L2-normalised float32 array."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = model.encode(
            batch,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        all_embs.append(embs)
        if (i // batch_size) % 50 == 0:
            log.info("  embedded %d / %d", min(i + batch_size, len(texts)), len(texts))
    return np.vstack(all_embs)


def batch_faiss_search(index, query_embs: np.ndarray, k: int) -> np.ndarray:
    """Returns (n_queries, k) integer array of FAISS internal indices."""
    _, ids = index.search(query_embs, k)
    return ids  # shape (n_queries, k)


def count_hubness(ids: np.ndarray, n_docs: int) -> np.ndarray:
    """ids shape (n_queries, k). Returns N_5 array of length n_docs."""
    flat = ids.flatten()
    flat = flat[flat >= 0]           # remove FAISS -1 sentinels
    counts = np.bincount(flat.astype(np.int64), minlength=n_docs)
    return counts.astype(np.int64)


def skewness(arr: np.ndarray) -> float:
    """Standardized skewness (third standardized moment)."""
    mu  = arr.mean()
    sig = arr.std()
    if sig < 1e-12:
        return 0.0
    return float(np.mean(((arr - mu) / sig) ** 3))


def ascii_histogram(counts: np.ndarray, max_bins: int = 20, width: int = 50) -> str:
    nonzero = counts[counts > 0]
    if len(nonzero) == 0:
        return "(no non-zero counts)"
    lo, hi = int(nonzero.min()), int(nonzero.max())
    if hi == lo:
        return f"All non-zero counts = {lo}"
    bins = np.linspace(lo, hi + 1, max_bins + 1)
    hist, edges = np.histogram(nonzero, bins=bins)
    peak = max(hist)
    lines = []
    for count, left, right in zip(hist, edges[:-1], edges[1:]):
        bar = "#" * int(width * count / peak) if peak > 0 else ""
        lines.append(f"  [{int(left):>4}–{int(right)-1:>4}] {bar} ({count})")
    return "\n".join(lines)


def interpret_skew(sk: float) -> str:
    if sk < 1.5:
        return "LOW  (< 1.5): bound robust to hub-based attacks"
    elif sk < 3.0:
        return "MODERATE  (1.5–3): inspect top hubs — semantically generic (exploitable) or relevant (not)?"
    else:
        return "HIGH  (> 3): strong hubness; bound has a hub-shaped hole"


def run_hubness(
    space_name: str,
    query_embs: np.ndarray,
    index,
    doc_ids: list[str],
    corpus_texts: list[str],
    k: int = TOP_K,
) -> tuple[np.ndarray, list[dict]]:
    """
    Returns (N5_array, top20_rows).
    top20_rows: list of dicts with doc_id, count, text_preview.
    """
    t0 = time.time()
    log.info("[%s] Searching %d queries × top-%d ...", space_name, len(query_embs), k)
    ids = batch_faiss_search(index, query_embs, k)
    log.info("[%s] Search done in %.1f s", space_name, time.time() - t0)

    n5 = count_hubness(ids, len(doc_ids))
    log.info("[%s] Hubness counts computed", space_name)

    top20_idx = np.argsort(n5)[::-1][:20]
    top20 = []
    for rank, idx in enumerate(top20_idx, 1):
        top20.append({
            "rank":       rank,
            "doc_id":     doc_ids[idx],
            "n5":         int(n5[idx]),
            "text":       corpus_texts[idx][:200],
        })
    return n5, top20


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info("Saved %s", path)


def write_md(
    space_name: str,
    n5: np.ndarray,
    top20: list[dict],
    skew: float,
    n_queries: int,
    elapsed_min: float,
    path: Path,
) -> None:
    n_docs       = len(n5)
    n_nonzero    = int((n5 > 0).sum())
    mean_n5      = float(n5.mean())
    median_n5    = float(np.median(n5))
    max_n5       = int(n5.max())

    lines = [
        f"# Hubness Report — {space_name}",
        "",
        f"**Queries:** {n_queries}  |  **Corpus:** {n_docs:,} docs  |  **k:** {TOP_K}",
        f"**Runtime:** {elapsed_min:.1f} min",
        "",
        "## Statistics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Docs with N_5 > 0 | {n_nonzero:,} ({100*n_nonzero/n_docs:.2f}%) |",
        f"| Mean N_5 | {mean_n5:.4f} |",
        f"| Median N_5 | {median_n5:.1f} |",
        f"| Max N_5 | {max_n5} |",
        f"| Skewness | {skew:.3f} |",
        f"| Interpretation | {interpret_skew(skew)} |",
        "",
        "## N_5 Distribution (non-zero docs only)",
        "",
        "```",
        ascii_histogram(n5),
        "```",
        "",
        "## Top 20 Hubs",
        "",
        "| Rank | Doc ID | N_5 | Text (first 200 chars) |",
        "|------|--------|-----|------------------------|",
    ]
    for r in top20:
        safe = r["text"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {r['rank']} | {r['doc_id']} | {r['n5']} | {safe} |")

    path.write_text("\n".join(lines) + "\n")
    log.info("Saved %s", path)


def _gpu_index(cpu_index, device_id: int = 0):
    """Move a CPU FAISS index to GPU; returns gpu index."""
    import faiss
    res = faiss.StandardGpuResources()
    res.setTempMemory(256 * 1024 * 1024)
    return faiss.index_cpu_to_gpu(res, device_id, cpu_index)


def build_bge_corpus_index(doc_ids: list[str], corpus_texts: list[str]) -> None:
    """
    Embed full corpus with BGE-large on GPU, build CPU FAISS index, save to disk.

    batch_size=128 + max_length=256: avoids heavy padding from long NQ passages.
    At ~295 docs/s on L40S, 2.68M docs ≈ 151 min.
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    CHUNK = 20_000
    BATCH = 128
    MAX_LEN = 256

    log.info("Building BGE index for %d docs (batch=%d, max_len=%d) — no time limit ...",
             len(corpus_texts), BATCH, MAX_LEN)
    BGE_IDX_DIR.mkdir(parents=True, exist_ok=True)

    bge = SentenceTransformer(BGE_MODEL, device="cuda")
    bge.max_seq_length = MAX_LEN
    dim = bge[0].auto_model.config.hidden_size
    cpu_idx = faiss.IndexFlatIP(dim)

    t0 = time.time()
    for chunk_start in range(0, len(corpus_texts), CHUNK):
        chunk = corpus_texts[chunk_start : chunk_start + CHUNK]
        embs = bge.encode(
            chunk,
            batch_size=BATCH,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        cpu_idx.add(embs)
        del embs
        done = min(chunk_start + CHUNK, len(corpus_texts))
        elapsed = time.time() - t0
        rate = done / elapsed
        eta = (len(corpus_texts) - done) / rate / 60 if rate > 0 else 0
        log.info("  BGE embed %d / %d  (%.0f docs/s  ETA %.1f min)",
                 done, len(corpus_texts), rate, eta)

    elapsed = time.time() - t0
    log.info("BGE embedding + indexing done in %.1f min (%.0f docs/s)",
             elapsed / 60, len(corpus_texts) / elapsed)

    del bge
    gc.collect()
    torch.cuda.empty_cache()

    faiss.write_index(cpu_idx, str(BGE_IDX_DIR / "index.faiss"))
    with open(BGE_IDX_DIR / "meta.pkl", "wb") as f:
        pickle.dump({"doc_ids": doc_ids}, f)
    log.info("BGE index saved to %s  (%d vectors)", BGE_IDX_DIR, cpu_idx.ntotal)


def load_bge_index_to_gpu():
    """Load saved BGE FAISS index from disk and move to GPU."""
    import faiss
    t0 = time.time()
    cpu_idx = faiss.read_index(str(BGE_IDX_DIR / "index.faiss"))
    log.info("BGE index read from disk in %.1f s (%d vectors)", time.time() - t0, cpu_idx.ntotal)
    return _gpu_index(cpu_idx)


def write_summary(
    gtr_n5: np.ndarray,
    bge_n5: np.ndarray | None,
    gtr_top20: list[dict],
    bge_top20: list[dict] | None,
    doc_ids: list[str],
    path: Path,
) -> None:
    gtr_skew = skewness(gtr_n5)
    lines = [
        "# Hubness Analysis — Cross-Space Summary",
        "",
        "## Skewness Comparison",
        "",
        f"| Space | Skewness | Interpretation |",
        f"|-------|----------|----------------|",
        f"| GTR-base | {gtr_skew:.3f} | {interpret_skew(gtr_skew)} |",
    ]
    if bge_n5 is not None:
        bge_skew = skewness(bge_n5)
        lines.append(f"| BGE-large | {bge_skew:.3f} | {interpret_skew(bge_skew)} |")

        # Hub overlap: top-10 doc IDs in each space
        gtr_top10_ids = set(r["doc_id"] for r in gtr_top20[:10])
        bge_top10_ids = set(r["doc_id"] for r in bge_top20[:10])
        overlap = gtr_top10_ids & bge_top10_ids

        lines += [
            "",
            "## Top-10 Hub Overlap (BGE ∩ GTR)",
            "",
            f"GTR top-10 hubs: `{sorted(gtr_top10_ids)}`",
            "",
            f"BGE top-10 hubs: `{sorted(bge_top10_ids)}`",
            "",
            f"**Intersection ({len(overlap)}/10):** `{sorted(overlap) if overlap else '∅'}`",
            "",
            "## BGE Hub Performance in GTR Space",
            "",
            "| BGE Hub Rank | Doc ID | BGE N_5 | GTR N_5 |",
            "|---|---|---|---|",
        ]
        # Map doc_id → GTR N5
        id2gtr = {doc_ids[i]: int(gtr_n5[i]) for i in range(len(doc_ids))}
        id2bge = {doc_ids[i]: int(bge_n5[i]) for i in range(len(doc_ids))}
        for r in (bge_top20 or [])[:10]:
            gtr_cnt = id2gtr.get(r["doc_id"], 0)
            lines.append(f"| {r['rank']} | {r['doc_id']} | {r['n5']} | {gtr_cnt} |")

        lines += [
            "",
            "## Verdict",
            "",
        ]
        if len(overlap) >= 5:
            verdict = (
                f"Strong hub overlap ({len(overlap)}/10 top hubs shared between BGE and GTR space). "
                f"BGE-identified hubs transfer reliably to GTR retrieval, making hubness exploitation "
                f"viable under our oracle-free threat model: an attacker using only BGE can identify "
                f"high-footprint documents that also appear frequently in GTR top-5 lists. "
                f"GTR skewness {gtr_skew:.2f} and BGE skewness {bge_skew:.2f} "
                f"({'both indicate' if gtr_skew > 1.5 and bge_skew > 1.5 else 'suggest'}) "
                f"{'exploitable hub structure' if max(gtr_skew, bge_skew) > 1.5 else 'low hubness; bound is robust'}."
            )
        elif len(overlap) >= 2:
            verdict = (
                f"Partial hub overlap ({len(overlap)}/10). BGE hubs partially predict GTR hubs, "
                f"suggesting moderate transferability. GTR skewness = {gtr_skew:.2f}, "
                f"BGE skewness = {bge_skew:.2f}. Hub-based exploitation is possible but "
                f"imprecise — an attacker targeting BGE hubs would incidentally hit GTR hubs "
                f"in roughly {len(overlap)*10}% of cases."
            )
        else:
            verdict = (
                f"Minimal hub overlap ({len(overlap)}/10). BGE and GTR hub structures are largely "
                f"disjoint. An oracle-free attacker using BGE to identify target documents cannot "
                f"reliably predict which documents are hubs in GTR retrieval. "
                f"GTR skewness = {gtr_skew:.2f}, BGE skewness = {bge_skew:.2f}. "
                f"{'Hubness is a concern in GTR space alone' if gtr_skew > 1.5 else 'Hubness is low overall; bound is robust'}."
            )
        lines.append(verdict)
    else:
        lines += [
            "",
            "*(BGE results not available — BGE index build exceeded time budget)*",
            "",
            "## GTR-Only Verdict",
            "",
            f"GTR skewness = {gtr_skew:.2f}. {interpret_skew(gtr_skew)}. "
            f"Cross-space transfer cannot be assessed without BGE index.",
        ]

    path.write_text("\n".join(lines) + "\n")
    log.info("Summary saved to %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import faiss
    from sentence_transformers import SentenceTransformer

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    log.info("=" * 65)
    log.info("Hubness Test — NQ corpus (2.68M docs)")
    log.info("=" * 65)

    queries      = load_queries()
    query_texts  = [q["text"] for q in queries]
    doc_ids, corpus_texts = load_corpus_texts()

    # ── Setting 1: GTR-base ───────────────────────────────────────────────────
    # Step 1a: embed queries with GTR model on GPU, then unload model
    # Step 1b: load GTR FAISS index to GPU (8.2 GB), search, unload index
    # Never have both model and index on GPU simultaneously.
    log.info("")
    log.info("=== SETTING 1: GTR-base ===")
    t1 = time.time()

    log.info("1a. Embedding %d queries with GTR on GPU ...", len(query_texts))
    gtr_model = SentenceTransformer(GTR_MODEL, device="cuda")
    gtr_q_embs = embed_batch(gtr_model, query_texts, batch_size=512)
    del gtr_model
    gc.collect()
    torch.cuda.empty_cache()
    log.info("    GTR model freed. GPU free for index.")

    log.info("1b. Loading GTR FAISS index to GPU from %s ...", GTR_IDX_DIR)
    gtr_cpu_idx = faiss.read_index(str(GTR_IDX_DIR / "index.faiss"))
    gtr_index   = _gpu_index(gtr_cpu_idx)
    del gtr_cpu_idx

    gtr_n5, gtr_top20 = run_hubness(
        "GTR", gtr_q_embs, gtr_index, doc_ids, corpus_texts
    )
    gtr_skew = skewness(gtr_n5)

    gtr_elapsed = (time.time() - t1) / 60
    log.info("GTR done in %.1f min | skewness=%.3f | max_N5=%d",
             gtr_elapsed, gtr_skew, int(gtr_n5.max()))

    write_csv(gtr_top20, RESULTS_DIR / "hubness_gtr.csv")
    write_md("GTR-base", gtr_n5, gtr_top20, gtr_skew,
             len(query_texts), gtr_elapsed, RESULTS_DIR / "hubness_gtr.md")
    np.save(RESULTS_DIR / "hubness_gtr_n5.npy", gtr_n5)

    del gtr_index
    gc.collect()
    torch.cuda.empty_cache()
    log.info("GTR index freed.")

    # ── Setting 2: BGE-large ──────────────────────────────────────────────────
    # Step 2a: build BGE corpus index if missing (model on GPU, index on CPU)
    # Step 2b: embed queries with BGE model on GPU, unload model
    # Step 2c: load BGE FAISS index to GPU (11 GB), search, unload
    log.info("")
    log.info("=== SETTING 2: BGE-large-en-v1.5 ===")
    t2 = time.time()

    bge_n5 = bge_top20 = None

    try:
        # 2a: build index if needed
        if not (BGE_IDX_DIR / "index.faiss").exists():
            build_bge_corpus_index(doc_ids, corpus_texts)
        else:
            log.info("BGE index already on disk — skipping build.")

        # 2b: embed queries with BGE model on GPU, then unload
        log.info("2b. Embedding %d queries with BGE on GPU ...", len(query_texts))
        bge_model  = SentenceTransformer(BGE_MODEL, device="cuda")
        bge_q_embs = embed_batch(bge_model, query_texts, batch_size=512)
        del bge_model
        gc.collect()
        torch.cuda.empty_cache()
        log.info("    BGE model freed. GPU free for index.")

        # 2c: load BGE FAISS index to GPU, search
        log.info("2c. Loading BGE FAISS index to GPU ...")
        bge_index = load_bge_index_to_gpu()

        bge_n5, bge_top20 = run_hubness(
            "BGE", bge_q_embs, bge_index, doc_ids, corpus_texts
        )
        bge_skew = skewness(bge_n5)

        bge_elapsed = (time.time() - t2) / 60
        log.info("BGE done in %.1f min | skewness=%.3f | max_N5=%d",
                 bge_elapsed, bge_skew, int(bge_n5.max()))

        write_csv(bge_top20, RESULTS_DIR / "hubness_bge.csv")
        write_md("BGE-large-en-v1.5", bge_n5, bge_top20, bge_skew,
                 len(query_texts), bge_elapsed, RESULTS_DIR / "hubness_bge.md")
        np.save(RESULTS_DIR / "hubness_bge_n5.npy", bge_n5)

        del bge_index
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        log.error("BGE phase failed: %s", e, exc_info=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("=== SUMMARY ===")
    write_summary(gtr_n5, bge_n5, gtr_top20, bge_top20, doc_ids,
                  RESULTS_DIR / "hubness_summary.md")

    total_min = (time.time() - t_total) / 60
    log.info("Total runtime: %.1f min (%.1f h)", total_min, total_min / 60)
    log.info("Done.")


if __name__ == "__main__":
    main()
