"""
Re-judge an existing attack CSV using only the Gemma judge (no Mistral re-run).

Reads the response from final_response or judge_response column, re-runs only
the judge, and writes a new CSV with updated judge_success / final_response.

Usage:
    uv run python src/experiments/rejudge.py \
        --config-name reproduction \
        +results_csv=/path/to/judged.csv
"""
from __future__ import annotations

import csv
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/ishana/scratch/hf_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

_src = Path(__file__).resolve().parents[2] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

log = logging.getLogger(__name__)


def _load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#")]
    for row in csv.DictReader(lines):
        rows.append(dict(row))
    return rows


def _save_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@hydra.main(config_path="../../configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    from judges.local_judge import LocalJudge

    results_csv = cfg.get("results_csv", None)
    if results_csv is None:
        raise ValueError("Pass +results_csv=/path/to/file.csv on the command line.")
    results_csv = Path(results_csv)

    rows = _load_csv(results_csv)
    log.info("Loaded %d rows from %s", len(rows), results_csv)

    # Prefer final_response; fall back to judge_response for older CSVs
    response_col = "final_response" if rows[0].get("final_response", "").strip() else "judge_response"
    log.info("Reading responses from column: %s", response_col)

    missing = sum(1 for r in rows if not r.get(response_col, "").strip())
    if missing:
        raise ValueError(f"{missing} rows have empty {response_col!r}. Run eval_asr.py first.")

    judge = LocalJudge(model_name=cfg.models.judge)

    n_jammed = 0
    for row in tqdm(rows, desc="Re-judge"):
        response = row[response_col]
        answered = judge.is_answered(row["query"], response)
        jammed = not answered
        row["judge_success"] = str(jammed)
        row["final_response"] = response
        if jammed:
            n_jammed += 1

    judge.close()

    asr = 100.0 * n_jammed / len(rows)
    log.info("ASR (re-judged): %.1f%% (%d/%d)", asr, n_jammed, len(rows))

    out_path = results_csv.with_stem(results_csv.stem + "_rejudged")
    _save_csv(rows, out_path)
    log.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
