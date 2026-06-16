"""
ConstrainedJointBBO on 15 additional intermediate-similarity clusters.
Appends to the existing task6_intermediate_honest.csv (7 classes already there).

Deduplication note
------------------
Cluster 552 (0.72 band) and Cluster 744 (0.80 band) share identical queries
(Olympic medals); run once as inter_744_080 and skip inter_552_072.
Cluster 546 does not exist in the 0.80 band; using 0.72 band entry (NBA MVPs,
sim=0.807) per user intent of high-similarity clusters.
Cluster 638 is in the 0.72 band in the CSV despite user labelling it 0.65.

Net result: 14 unique clusters run (15 requested, 1 duplicate collapsed).

Output
------
Appended to: /home/ishana/scratch/results/task6_intermediate_honest.csv
Checkpoint:  /home/ishana/scratch/results/task6_intermediate_ext_ckpt.pkl
Log:         /home/ishana/projects/llm_jam_universal/logs/task6_intermediate_ext.log
"""
from __future__ import annotations

import csv
import gc
import json
import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

os.environ["HF_HOME"] = "/home/ishana/scratch/hf_cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

_SRC = Path(__file__).resolve().parents[2] / "src"
import sys
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "task6_intermediate_ext.log"),
    ],
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH     = Path("/home/ishana/scratch")
DATA_DIR    = SCRATCH / "data"
INDEX_DIR   = DATA_DIR / "indices/nq/sentence-transformers__gtr-t5-base"
RESULTS_DIR = SCRATCH / "results"
OUT_CSV     = RESULTS_DIR / "task6_intermediate_honest.csv"        # append target
CKPT_FILE   = RESULTS_DIR / "task6_intermediate_ext_ckpt.pkl"     # own checkpoint

EMB_CACHE   = DATA_DIR / "classes/answerable_bge_embeddings.npy"
ID_CACHE    = DATA_DIR / "classes/answerable_bge_ids.json"
QUERIES_FILE = DATA_DIR / "nq/queries.jsonl"

GTR_MODEL   = "sentence-transformers/gtr-t5-base"
LLM_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL = "google/gemma-2-9b-it"
TOP_K       = 5

# ── 14 unique clusters (15 requested; 552/744 duplicate collapsed to 744_080) ─
CLUSTERS = [
    # ── "0.80 band" group (user's labelling) ─────────────────────────────────
    # cluster 546 not in 0.80 band; use 0.72 entry (NBA MVPs, sim=0.807)
    dict(class_id="inter_546_072",  cluster_id=546,  target_band="0.72",
         within_sim=0.80726, query_ids=["test213","test2821","test645"]),
    # cluster 744 (0.80) == cluster 552 (0.72) same queries; run once as 744_080
    dict(class_id="inter_744_080",  cluster_id=744,  target_band="0.80",
         within_sim=0.80578, query_ids=["test2873","test2977","test3293"]),
    dict(class_id="inter_680_080",  cluster_id=680,  target_band="0.80",
         within_sim=0.81159, query_ids=["test1117","test112","test1283",
                                        "test1489","test2811","test764"]),
    dict(class_id="inter_1334_080", cluster_id=1334, target_band="0.80",
         within_sim=0.80301, query_ids=["test2368","test3380","test721"]),
    # ── 0.72 band ─────────────────────────────────────────────────────────────
    dict(class_id="inter_1177_072", cluster_id=1177, target_band="0.72",
         within_sim=0.76569, query_ids=["test1763","test1768","test2521","test2712"]),
    dict(class_id="inter_930_072",  cluster_id=930,  target_band="0.72",
         within_sim=0.75694, query_ids=["test115","test2556","test2569"]),
    dict(class_id="inter_1574_072", cluster_id=1574, target_band="0.72",
         within_sim=0.74747, query_ids=["test2004","test2315","test313"]),
    dict(class_id="inter_566_072",  cluster_id=566,  target_band="0.72",
         within_sim=0.74698, query_ids=["test1167","test1864","test2233","test3257"]),
    dict(class_id="inter_1062_072", cluster_id=1062, target_band="0.72",
         within_sim=0.73779, query_ids=["test1485","test3009","test497"]),
    dict(class_id="inter_493_072",  cluster_id=493,  target_band="0.72",
         within_sim=0.72539, query_ids=["test1363","test164","test1771","test388"]),
    # ── 0.65 band (cluster 638 is 0.72 in CSV; others are genuine 0.65) ───────
    dict(class_id="inter_638_072",  cluster_id=638,  target_band="0.72",
         within_sim=0.70283, query_ids=["test1183","test1590","test2286","test2511"]),
    dict(class_id="inter_442_065",  cluster_id=442,  target_band="0.65",
         within_sim=0.72889, query_ids=["test1938","test2850","test750"]),
    dict(class_id="inter_207_065",  cluster_id=207,  target_band="0.65",
         within_sim=0.70909, query_ids=["test1456","test1464","test2280",
                                        "test718","test85"]),
    dict(class_id="inter_422_065",  cluster_id=422,  target_band="0.65",
         within_sim=0.67259, query_ids=["test219","test2381","test2395",
                                        "test2863","test623"]),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_classes() -> list[dict]:
    with open(QUERIES_FILE) as f:
        id2text = {json.loads(l)["_id"]: json.loads(l)["text"] for l in f}
    cached_ids: list[str] = json.loads(ID_CACHE.read_text())
    id2idx = {qid: i for i, qid in enumerate(cached_ids)}
    all_embs = np.load(EMB_CACHE)  # (2279, 1024) L2-normalised

    result = []
    for cl in CLUSTERS:
        qids    = cl["query_ids"]
        queries = [id2text[qid] for qid in qids]
        idxs    = [id2idx[qid] for qid in qids]
        embs    = all_embs[idxs]
        centroid_raw = embs.mean(axis=0)
        centroid = centroid_raw / max(float(np.linalg.norm(centroid_raw)), 1e-9)
        result.append({**cl, "_queries": queries, "_embeddings": embs,
                        "_centroid": centroid})
    return result


def find_q_star(queries, embs, centroid):
    dists = np.linalg.norm(embs - centroid[np.newaxis, :], axis=1)
    idx = int(np.argmin(dists))
    return queries[idx], idx


# ── Honest retrieval helpers ──────────────────────────────────────────────────

def check_top5(blocker_emb, query, retriever):
    results = retriever.retrieve_with_scores(query, k=TOP_K)
    scores  = [s for _, s in results]
    thresh  = scores[-1] if scores else 0.0
    q_emb   = retriever.embed_batch([query])[0]
    b_sim   = float(np.dot(blocker_emb, q_emb))
    retr    = b_sim >= thresh
    rank    = sum(1 for s in scores if s > b_sim) + 1 if retr else None
    return retr, rank, round(b_sim, 5), round(thresh, 5)


def build_prompts(blocker_doc, queries, ret_results, retriever, tpl, k):
    prompts, idxs = [], []
    for qi, (q, r) in enumerate(zip(queries, ret_results)):
        if not r["retrieved"]:
            continue
        top_ids  = retriever.retrieve(q, k=k)
        base     = [retriever.get_doc_text(d) for d in top_ids]
        ctx      = "\n\n".join([blocker_doc] + base[:k - 1])
        prompts.append(tpl.format(context=ctx, query=q))
        idxs.append(qi)
    return prompts, idxs


# ── Config ────────────────────────────────────────────────────────────────────

def _cfg():
    cfg_dir = Path(__file__).resolve().parents[2] / "configs"
    base = OmegaConf.load(cfg_dir / "base.yaml")
    base.attack.num_iterations = 500
    base.attack.es_patience = 80
    return base


# ── CSV helpers ───────────────────────────────────────────────────────────────

FIELDNAMES = [
    "class_id", "cluster_id", "target_band",
    "query_idx", "query_id", "query_text",
    "within_class_sim", "n_class_queries",
    "blocker_doc", "blocker_sim", "threshold_sim",
    "retrieved_top5", "rank",
    "response", "jam_success", "jammed_honest",
    "q_star", "final_loss", "n_iterations", "class_time_min",
    "real_threshold", "cands_scored", "cands_rejected", "cands_accepted",
]


def existing_class_ids() -> set[str]:
    if not OUT_CSV.exists():
        return set()
    with open(OUT_CSV) as f:
        return {r["class_id"] for r in csv.DictReader(f)}


def append_rows(rows: list[dict]) -> None:
    write_header = not OUT_CSV.exists()
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerows(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    log.info("=" * 65)
    log.info("Task 6 ext — ConstrainedJointBBO on 14 intermediate clusters")
    log.info("Appending to %s", OUT_CSV)
    log.info("=" * 65)

    already_in_csv = existing_class_ids()
    log.info("Classes already in output CSV: %s", sorted(already_in_csv))

    classes = load_classes()
    for cls in classes:
        log.info("  %s  n=%d  sim=%.3f  q=[%s...]",
                 cls["class_id"], len(cls["_queries"]), cls["within_sim"],
                 cls["_queries"][0][:40])

    cfg = _cfg()

    log.info("Loading GTR retriever...")
    from rag.retriever import Retriever
    retriever = Retriever(model_name=GTR_MODEL, score_function="cos_sim")
    retriever.load_index(INDEX_DIR)
    log.info("Retriever: %d docs", retriever._index.ntotal)

    ckpt: dict[str, dict] = {}
    if CKPT_FILE.exists():
        with open(CKPT_FILE, "rb") as f:
            ckpt = pickle.load(f)
        log.info("Checkpoint resume: %d classes done", len(ckpt))

    log.info("Loading vLLM: %s", LLM_MODEL)
    from rag.generator import VLLMGenerator
    generator = VLLMGenerator(
        model_name=LLM_MODEL,
        temperature=float(cfg.attack.temperature),
        max_tokens=int(cfg.attack.max_response_len),
        gpu_memory_utilization=float(cfg.vllm.gpu_memory_utilization),
        dtype=str(cfg.vllm.dtype),
        max_model_len=int(cfg.vllm.max_model_len),
    )
    from utils.gpu_manager import GPUManager
    from attacks.constrained_joint_bbo import ConstrainedJointBBO
    attacker = ConstrainedJointBBO(cfg, retriever, generator, GPUManager())

    rag_tpl    = cfg.rag_prompt
    retrieval_k = int(cfg.retrieval.k)
    class_timings: dict[str, float] = {}

    for cls in classes:
        cid = cls["class_id"]

        # Sanity check: skip if already in the output CSV
        if cid in already_in_csv:
            log.info("[SKIP-CSV] %s already in output CSV", cid)
            continue
        if cid in ckpt:
            log.info("[SKIP-CKPT] %s already in checkpoint (will judge + append)", cid)
        else:
            queries  = cls["_queries"]
            embs     = cls["_embeddings"]
            centroid = cls["_centroid"]

            log.info("=" * 55)
            log.info("Class %s | n=%d | sim=%.3f", cid, len(queries), cls["within_sim"])
            t_cls = time.time()

            q_star, q_star_idx = find_q_star(queries, embs, centroid)
            log.info("  q*[%d]: %r", q_star_idx, q_star[:80])

            real_thr = attacker.precompute_retrieval_threshold(q_star)

            log.info("  Running ConstrainedJointBBO ...")
            t_bbo = time.time()
            result = attacker.run(query=q_star)
            bbo_min = (time.time() - t_bbo) / 60
            log.info("  BBO %.1f min | loss=%.4f | iters=%d | rej=%d acc=%d",
                     bbo_min, result.final_loss, result.n_iterations,
                     attacker.n_rejected_by_constraint,
                     attacker.n_accepted_by_constraint)

            blocker_emb = retriever.embed_batch([result.final_doc])[0]
            ret_results = []
            for q in queries:
                retr, rank, b_sim, t_sim = check_top5(blocker_emb, q, retriever)
                ret_results.append({"retrieved": retr, "rank": rank,
                                    "blocker_sim": b_sim, "threshold_sim": t_sim})
            n_ret = sum(r["retrieved"] for r in ret_results)
            log.info("  Retrieval: %d/%d in top-5", n_ret, len(queries))

            prompts, ret_idxs = build_prompts(
                result.final_doc, queries, ret_results, retriever, rag_tpl, retrieval_k)
            responses: list[str | None] = [None] * len(queries)
            if prompts:
                for qi, resp in zip(ret_idxs, generator.generate(prompts)):
                    responses[qi] = resp
            log.info("  Generated %d responses", len(prompts))

            cls_min = (time.time() - t_cls) / 60
            class_timings[cid] = cls_min

            ckpt[cid] = {
                "class_id":    cid, "cluster_id":   cls["cluster_id"],
                "target_band": cls["target_band"],  "within_sim":   cls["within_sim"],
                "n_queries":   len(queries),        "queries":      queries,
                "query_ids":   cls["query_ids"],    "q_star":       q_star,
                "q_star_idx":  q_star_idx,          "result":       result,
                "retrieval_results": ret_results,   "responses":    responses,
                "class_time_min":    cls_min,
                "constraint_stats": {
                    "real_threshold":   real_thr,
                    "n_cands_scored":   int(attacker.n_candidates_scored),
                    "n_cands_rejected": int(attacker.n_rejected_by_constraint),
                    "n_cands_accepted": int(attacker.n_accepted_by_constraint),
                },
            }
            with open(CKPT_FILE, "wb") as f:
                pickle.dump(ckpt, f)
            log.info("  Checkpointed %s (%.1f min total)", cid, cls_min)

            # Budget check: warn if on track to exceed 8h
            elapsed_h = (time.time() - t_start) / 3600
            done_so_far = len(class_timings)
            remaining = len(classes) - done_so_far - sum(
                1 for c in classes if c["class_id"] in already_in_csv)
            if done_so_far > 0 and remaining > 0:
                avg_min = sum(class_timings.values()) / done_so_far
                est_h   = (remaining * avg_min) / 60
                log.info("  Budget: %.1f h elapsed | ~%.1f h remaining for %d classes",
                         elapsed_h, est_h, remaining)
                if elapsed_h + est_h > 8.5:
                    log.warning("  Budget warning: projected total %.1f h > 8 h cap",
                                elapsed_h + est_h)

    log.info("Closing vLLM...")
    generator.close()
    del generator, attacker
    gc.collect()
    torch.cuda.empty_cache()

    log.info("Loading judge: %s", JUDGE_MODEL)
    from judges.local_judge import LocalJudge
    judge = LocalJudge(model_name=JUDGE_MODEL)

    new_rows: list[dict] = []
    for cls in classes:
        cid = cls["class_id"]
        if cid in already_in_csv:
            continue
        if cid not in ckpt:
            log.warning("Class %s not in checkpoint — skipped (BBO may not have run)", cid)
            continue

        entry   = ckpt[cid]
        cstats  = entry["constraint_stats"]

        for qi, (q, resp, ret) in enumerate(zip(
                entry["queries"], entry["responses"], entry["retrieval_results"])):
            retrieved = bool(ret["retrieved"])
            if retrieved and resp is not None:
                jam_success   = int(not judge.is_answered(q, resp))
                jammed_honest = jam_success
            else:
                jam_success   = None
                jammed_honest = 0

            new_rows.append({
                "class_id":         cid,
                "cluster_id":       entry["cluster_id"],
                "target_band":      entry["target_band"],
                "query_idx":        qi,
                "query_id":         entry["query_ids"][qi],
                "query_text":       q,
                "within_class_sim": round(entry["within_sim"], 4),
                "n_class_queries":  entry["n_queries"],
                "blocker_doc":      entry["result"].final_doc[:300],
                "blocker_sim":      ret["blocker_sim"],
                "threshold_sim":    ret["threshold_sim"],
                "retrieved_top5":   int(retrieved),
                "rank":             ret["rank"] if ret["rank"] is not None else "",
                "response":         resp[:300] if resp else "",
                "jam_success":      jam_success if jam_success is not None else "",
                "jammed_honest":    jammed_honest,
                "q_star":           entry["q_star"][:120],
                "final_loss":       round(entry["result"].final_loss, 5),
                "n_iterations":     entry["result"].n_iterations,
                "class_time_min":   round(entry["class_time_min"], 2),
                "real_threshold":   round(cstats["real_threshold"], 5),
                "cands_scored":     cstats["n_cands_scored"],
                "cands_rejected":   cstats["n_cands_rejected"],
                "cands_accepted":   cstats["n_cands_accepted"],
            })

    judge.close()

    append_rows(new_rows)
    log.info("Appended %d rows → %s", len(new_rows), OUT_CSV)

    _print_summary(new_rows, class_timings, already_in_csv)
    log.info("Total runtime: %.1f min", (time.time() - t_start) / 60)


def _print_summary(rows, timings, skipped_csv):
    from collections import defaultdict

    print()
    print("=" * 78)
    print("TASK 6 EXT — CONSTRAINED BBO — 14 INTERMEDIATE CLUSTERS HONEST ASR")
    print("=" * 78)

    if skipped_csv:
        print(f"\n  Skipped (already in CSV): {sorted(skipped_csv)}")

    per = defaultdict(lambda: {"n": 0, "ret": 0, "jam": 0, "sim": 0.0,
                                "loss": 0.0, "iters": 0})
    for r in rows:
        cid = r["class_id"]
        per[cid]["n"]    += 1
        per[cid]["ret"]  += int(r["retrieved_top5"])
        per[cid]["jam"]  += int(r["jammed_honest"])
        per[cid]["sim"]   = float(r["within_class_sim"])
        per[cid]["loss"]  = float(r["final_loss"])
        per[cid]["iters"] = int(r["n_iterations"])

    print(f"\n{'Class':<22} {'n':>3} {'sim':>6} {'ret/n':>7} {'jam|ret':>8} "
          f"{'asr%':>6} {'min':>6}")
    print("-" * 65)

    total_n = total_ret = total_jam = 0
    for cid, pc in sorted(per.items()):
        n, ret, jam = pc["n"], pc["ret"], pc["jam"]
        jam_given_ret = f"{jam}/{ret}" if ret > 0 else "—"
        t = timings.get(cid, 0.0)
        print(f"{cid:<22} {n:>3} {pc['sim']:>6.3f} {ret:>3}/{n:<3}  "
              f"{jam_given_ret:>7}   {100*jam/n:>5.1f}%  {t:>5.1f}")
        total_n += n; total_ret += ret; total_jam += jam

    print("-" * 65)
    jam_given_ret_total = f"{total_jam}/{total_ret}" if total_ret > 0 else "—"
    print(f"{'TOTAL':<22} {total_n:>3}        {total_ret:>3}/{total_n:<3}  "
          f"{jam_given_ret_total:>7}   {100*total_jam/total_n:>5.1f}%")

    n_ret = sum(int(r["retrieved_top5"]) for r in rows)
    jam_ret = sum(int(r["jammed_honest"]) for r in rows if int(r["retrieved_top5"]))
    if n_ret:
        print(f"\n  P(jam | retrieved): {jam_ret}/{n_ret} = {100*jam_ret/n_ret:.1f}%")
    print(f"  Retrieval rate:     {total_ret}/{total_n} = {100*total_ret/total_n:.1f}%")
    print(f"  Honest ASR:         {total_jam}/{total_n} = {100*total_jam/total_n:.1f}%")

    if timings:
        total_t = sum(timings.values())
        print(f"  Total BBO time:     {total_t:.1f} min ({total_t/60:.1f} h) "
              f"for {len(timings)} new classes")
    print("=" * 78)


if __name__ == "__main__":
    main()
