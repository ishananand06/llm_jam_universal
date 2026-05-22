"""
Step 2: Build paraphrase query classes for M2 evaluation.

Generates 5 paraphrases per query for 100 answerable NQ queries using
Llama-3.1-8B-Instruct via vLLM, then embeds all 6 queries per class with
BGE-large-en-v1.5.

Outputs:
  /home/ishana/scratch/data/classes/paraphrase_classes.json
  /home/ishana/scratch/data/classes/paraphrase_embeddings.npy  (shape: [N, 6, 1024])
"""
from __future__ import annotations

import gc
import json
import logging
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────
HF_HOME = "/home/ishana/scratch/hf_cache"
LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BGE_MODEL = "BAAI/bge-large-en-v1.5"
NQ_QUERIES = Path("/home/ishana/scratch/data/nq/queries.jsonl")
ANSWERABLE_PKL = Path("/home/ishana/scratch/data/answerable_nq_test.pkl")
OUT_DIR = Path("/home/ishana/scratch/data/classes")
NUM_CLASSES = 100
MAX_RETRIES = 3
GPU = "1"  # L40S with 45 GB free

os.environ["HF_HOME"] = HF_HOME
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

# ── paraphrase prompt ─────────────────────────────────────────────────────────
_SYS = (
    "You are a paraphrase generator. Your only task is to rephrase questions "
    "in different words while preserving the exact meaning."
)
_USER = """\
Generate exactly 5 paraphrases of the following question. Each paraphrase must:
- Ask the exact same thing in different words
- Be a complete, natural-sounding question
- Not add or remove information

Output ONLY the 5 paraphrases, numbered 1-5, one per line. No other text.

Question: {query}"""


def load_answerable_queries(n: int) -> list[dict]:
    with open(ANSWERABLE_PKL, "rb") as f:
        answerable_ids: set[str] = pickle.load(f)

    queries = []
    with open(NQ_QUERIES) as f:
        for line in f:
            obj = json.loads(line)
            if obj["_id"] in answerable_ids:
                queries.append({"id": obj["_id"], "text": obj["text"]})
            if len(queries) == n:
                break

    log.info("Selected %d answerable queries for paraphrase classes", len(queries))
    return queries


def parse_paraphrases(text: str) -> list[str] | None:
    lines = []
    for line in text.strip().splitlines():
        line = line.strip()
        # Match "1. text", "1) text", "1: text"
        m = re.match(r"^[1-5][.):\s]\s*(.+)$", line)
        if m:
            lines.append(m.group(1).strip())
    return lines[:5] if len(lines) >= 5 else None


def generate_all_paraphrases(
    llm, tokenizer, queries: list[dict]
) -> dict[str, list[str]]:
    from vllm import SamplingParams

    sampling = SamplingParams(temperature=0.7, max_tokens=350, n=1)

    pending = {q["id"]: q for q in queries}
    results: dict[str, list[str]] = {}
    skipped: list[str] = []

    for attempt in range(MAX_RETRIES + 1):
        if not pending:
            break

        ids = list(pending.keys())
        prompts = []
        for qid in ids:
            messages = [
                {"role": "system", "content": _SYS},
                {"role": "user", "content": _USER.format(query=pending[qid]["text"])},
            ]
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt_str)

        log.info(
            "Attempt %d/%d: generating paraphrases for %d queries...",
            attempt + 1, MAX_RETRIES + 1, len(ids),
        )
        t0 = time.time()
        outputs = llm.generate(prompts, sampling)
        log.info("  Done in %.1fs", time.time() - t0)

        still_pending = {}
        for qid, out in zip(ids, outputs):
            text = out.outputs[0].text
            paras = parse_paraphrases(text)
            if paras:
                results[qid] = paras
            else:
                log.warning(
                    "  Parse failed for %s (attempt %d): %r",
                    qid, attempt + 1, text[:150],
                )
                if attempt < MAX_RETRIES:
                    still_pending[qid] = pending[qid]
                else:
                    log.error("  Giving up on %s after %d retries", qid, MAX_RETRIES)
                    skipped.append(qid)

        pending = still_pending

    if skipped:
        log.warning("Skipped %d queries: %s", len(skipped), skipped)

    return results


def mean_pairwise_cosine(embs: np.ndarray) -> float:
    """Mean pairwise cosine sim for L2-normalised embeddings (shape [n, d])."""
    n = len(embs)
    if n < 2:
        return 1.0
    dot = embs @ embs.T  # [n, n]
    idx = np.triu_indices(n, k=1)
    return float(dot[idx].mean())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load answerable queries ─────────────────────────────────────────────
    queries = load_answerable_queries(NUM_CLASSES)

    # ── 2. Load tokenizer (before vLLM to apply chat template) ────────────────
    from transformers import AutoTokenizer

    log.info("Loading Llama tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)

    # ── 3. Load Llama via vLLM, generate paraphrases ──────────────────────────
    log.info("Loading Llama-3.1-8B-Instruct via vLLM...")
    from vllm import LLM

    llm = LLM(
        model=LLAMA_MODEL,
        dtype="float16",
        max_model_len=4096,
        gpu_memory_utilization=0.80,
    )

    paraphrase_map = generate_all_paraphrases(llm, tokenizer, queries)
    log.info("Paraphrase generation complete: %d/%d succeeded", len(paraphrase_map), len(queries))

    # ── 4. Tear down vLLM before loading BGE (single GPU) ─────────────────────
    log.info("Shutting down vLLM to free VRAM...")
    del llm
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
    log.info("VRAM freed")

    # ── 5. Build ordered text list for embedding ───────────────────────────────
    class_info = []
    all_texts: list[str] = []

    for q in queries:
        qid = q["id"]
        if qid not in paraphrase_map:
            continue
        six = [q["text"]] + paraphrase_map[qid]  # original + 5 paraphrases
        start = len(all_texts)
        all_texts.extend(six)
        class_info.append({
            "class_id": qid,
            "original_query": q["text"],
            "paraphrases": paraphrase_map[qid],
            "_embed_start": start,
        })

    # ── 6. Embed with BGE-large ────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer

    log.info("Loading BGE-large-en-v1.5...")
    bge = SentenceTransformer(BGE_MODEL)

    log.info("Embedding %d texts (%d classes × 6)...", len(all_texts), len(class_info))
    t0 = time.time()
    all_embs: np.ndarray = bge.encode(
        all_texts,
        normalize_embeddings=True,
        batch_size=128,
        show_progress_bar=True,
    ).astype(np.float32)
    log.info("Embedding done in %.1fs", time.time() - t0)

    # ── 7. Compute per-class stats ────────────────────────────────────────────
    n_classes = len(class_info)
    embed_dim = all_embs.shape[1]
    emb_array = np.zeros((n_classes, 6, embed_dim), dtype=np.float32)

    classes_out = []
    for i, info in enumerate(class_info):
        s = info["_embed_start"]
        class_embs = all_embs[s : s + 6]
        emb_array[i] = class_embs

        centroid = class_embs.mean(axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
        within_sim = mean_pairwise_cosine(class_embs)

        classes_out.append({
            "class_id": info["class_id"],
            "original_query": info["original_query"],
            "paraphrases": info["paraphrases"],
            "centroid": centroid_norm.tolist(),
            "within_class_similarity": round(within_sim, 4),
        })

    # ── 8. Save ────────────────────────────────────────────────────────────────
    out_json = OUT_DIR / "paraphrase_classes_llama.json"
    out_npy = OUT_DIR / "paraphrase_embeddings_llama.npy"

    with open(out_json, "w") as f:
        json.dump(classes_out, f, indent=2)
    np.save(out_npy, emb_array)

    sims = [c["within_class_similarity"] for c in classes_out]
    log.info("─" * 60)
    log.info("Saved %d classes", len(classes_out))
    log.info("Within-class similarity: mean=%.3f  min=%.3f  max=%.3f", np.mean(sims), np.min(sims), np.max(sims))
    log.info("JSON  → %s", out_json)
    log.info("NPY   → %s  shape=%s", out_npy, emb_array.shape)
    log.info("─" * 60)


if __name__ == "__main__":
    main()
