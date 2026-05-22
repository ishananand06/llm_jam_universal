"""
Step 3: Build entity query classes for M2 evaluation.

Groups answerable NQ queries by their primary named entity (longest
PERSON/ORG/GPE/WORK_OF_ART span). Targets 20 classes of size 3–8.

Uses the full 2279-query answerable pool from answerable_nq_test.pkl —
no re-judging needed.

Outputs:
  /home/ishana/scratch/data/classes/entity_classes.json
  /home/ishana/scratch/data/classes/entity_embeddings.npy  (shape: [N, max_size, 1024])
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────
HF_HOME = "/home/ishana/scratch/hf_cache"
BGE_MODEL = "BAAI/bge-large-en-v1.5"
NQ_QUERIES = Path("/home/ishana/scratch/data/nq/queries.jsonl")
ANSWERABLE_PKL = Path("/home/ishana/scratch/data/answerable_nq_test.pkl")
OUT_DIR = Path("/home/ishana/scratch/data/classes")
TARGET_CLASSES = 20
MIN_SIZE = 3
MAX_SIZE = 8
# NER label types to extract primary entity from
ENTITY_TYPES = {"PERSON", "ORG", "GPE", "WORK_OF_ART"}

os.environ["HF_HOME"] = HF_HOME
# Use GPU 0 (BGE only, no vLLM needed here)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_answerable_queries() -> list[dict]:
    with open(ANSWERABLE_PKL, "rb") as f:
        answerable_ids: set[str] = pickle.load(f)

    queries = []
    with open(NQ_QUERIES) as f:
        for line in f:
            obj = json.loads(line)
            if obj["_id"] in answerable_ids:
                queries.append({"id": obj["_id"], "text": obj["text"]})

    log.info("Loaded %d answerable queries", len(queries))
    return queries


def extract_primary_entity(doc) -> str | None:
    """Return the longest matching entity span text, or None."""
    candidates = [ent for ent in doc.ents if ent.label_ in ENTITY_TYPES]
    if not candidates:
        return None
    # Longest span wins; title-case for consistent grouping
    longest = max(candidates, key=lambda e: len(e.text))
    return longest.text.strip().title()


def run_ner(queries: list[dict]) -> list[tuple[dict, str | None]]:
    import spacy

    log.info("Loading spaCy en_core_web_lg...")
    nlp = spacy.load("en_core_web_lg", disable=["parser", "lemmatizer"])

    log.info("Running NER on %d queries...", len(queries))
    t0 = time.time()
    results = []
    texts = [q["text"] for q in queries]
    for q, doc in zip(queries, nlp.pipe(texts, batch_size=256)):
        entity = extract_primary_entity(doc)
        results.append((q, entity))
    log.info("NER done in %.1fs", time.time() - t0)
    return results


def build_groups(
    ner_results: list[tuple[dict, str | None]],
    min_size: int,
    max_size: int,
    target: int,
) -> list[dict]:
    """
    Group queries by primary entity, filter to [min_size, max_size],
    keep at most `target` groups (prioritise largest groups first).
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for q, entity in ner_results:
        if entity:
            buckets[entity].append(q)

    valid = {
        entity: queries
        for entity, queries in buckets.items()
        if min_size <= len(queries) <= max_size
    }
    log.info(
        "Found %d entity buckets with %d–%d queries (before trim to %d)",
        len(valid), min_size, max_size, target,
    )

    # If not enough, relax max_size to include larger groups (truncated to MAX_SIZE)
    if len(valid) < target:
        log.info("Fewer than %d valid groups — expanding to include larger buckets (truncated to %d)", target, max_size)
        for entity, queries in buckets.items():
            if entity not in valid and len(queries) >= min_size:
                valid[entity] = queries[:max_size]
        log.info("After expansion: %d groups", len(valid))

    # Sort by group size descending, take top `target`
    sorted_groups = sorted(valid.items(), key=lambda kv: len(kv[1]), reverse=True)[:target]

    classes = []
    for entity, queries in sorted_groups:
        # Cap at MAX_SIZE
        qs = queries[:max_size]
        classes.append({
            "primary_entity": entity,
            "queries": [q["text"] for q in qs],
            "query_ids": [q["id"] for q in qs],
        })

    return classes


def mean_pairwise_cosine(embs: np.ndarray) -> float:
    n = len(embs)
    if n < 2:
        return 1.0
    dot = embs @ embs.T
    idx = np.triu_indices(n, k=1)
    return float(dot[idx].mean())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load + NER ──────────────────────────────────────────────────────────
    queries = load_answerable_queries()
    ner_results = run_ner(queries)

    # ── 2. Group by entity ─────────────────────────────────────────────────────
    classes = build_groups(ner_results, MIN_SIZE, MAX_SIZE, TARGET_CLASSES)
    log.info("Built %d entity classes", len(classes))
    if len(classes) < TARGET_CLASSES:
        log.warning("Only %d classes found (target was %d)", len(classes), TARGET_CLASSES)

    # ── 3. Embed with BGE-large ────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer

    log.info("Loading BGE-large-en-v1.5...")
    bge = SentenceTransformer(BGE_MODEL)

    all_texts: list[str] = []
    offsets: list[tuple[int, int]] = []
    for cls in classes:
        s = len(all_texts)
        all_texts.extend(cls["queries"])
        offsets.append((s, s + len(cls["queries"])))

    log.info("Embedding %d texts (%d classes)...", len(all_texts), len(classes))
    t0 = time.time()
    all_embs: np.ndarray = bge.encode(
        all_texts,
        normalize_embeddings=True,
        batch_size=128,
        show_progress_bar=True,
    ).astype(np.float32)
    log.info("Embedding done in %.1fs", time.time() - t0)

    # ── 4. Compute per-class stats ────────────────────────────────────────────
    max_size_actual = max(len(c["queries"]) for c in classes)
    embed_dim = all_embs.shape[1]
    # Padded array (zeros for unused slots)
    emb_array = np.zeros((len(classes), max_size_actual, embed_dim), dtype=np.float32)

    classes_out = []
    for i, (cls, (s, e)) in enumerate(zip(classes, offsets)):
        class_embs = all_embs[s:e]
        emb_array[i, : len(class_embs)] = class_embs

        centroid = class_embs.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        within_sim = mean_pairwise_cosine(class_embs)

        classes_out.append({
            "class_id": f"entity_{i:02d}",
            "primary_entity": cls["primary_entity"],
            "queries": cls["queries"],
            "query_ids": cls["query_ids"],
            "size": len(cls["queries"]),
            "centroid": centroid.tolist(),
            "within_class_similarity": round(within_sim, 4),
        })

    # ── 5. Save ────────────────────────────────────────────────────────────────
    out_json = OUT_DIR / "entity_classes.json"
    out_npy = OUT_DIR / "entity_embeddings.npy"

    with open(out_json, "w") as f:
        json.dump(classes_out, f, indent=2)
    np.save(out_npy, emb_array)

    sims = [c["within_class_similarity"] for c in classes_out]
    sizes = [c["size"] for c in classes_out]
    log.info("─" * 60)
    log.info("Saved %d entity classes", len(classes_out))
    log.info("Class sizes: min=%d  max=%d  mean=%.1f", min(sizes), max(sizes), np.mean(sizes))
    log.info("Within-class similarity: mean=%.3f  min=%.3f  max=%.3f", np.mean(sims), np.min(sims), np.max(sims))
    log.info("JSON  → %s", out_json)
    log.info("NPY   → %s  shape=%s", out_npy, emb_array.shape)
    log.info("─" * 60)

    # Print a preview
    log.info("Top 5 entity classes:")
    for c in classes_out[:5]:
        log.info("  [%s] %d queries, sim=%.3f", c["primary_entity"], c["size"], c["within_class_similarity"])
        for q in c["queries"]:
            log.info("    - %s", q)


if __name__ == "__main__":
    main()
