from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from attacks.base import AttackResult


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def save_results(
    results: list[AttackResult],
    cfg: DictConfig,
    output_path: Path,
    config_path: str = "",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "git_commit": _git_commit(),
        "config": config_path or "unknown",
        "seed": cfg.attack.seed,
        "hardware": "L40S",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_llm": cfg.models.target_llm,
        "rag_embed": cfg.models.rag_embed,
        "oracle_embed": cfg.models.oracle_embed,
        "num_queries": len(results),
    }

    with output_path.open("w", newline="") as f:
        # Metadata header block — prefixed with # so pandas read_csv skips by default
        for k, v in metadata.items():
            f.write(f"# {k}: {v}\n")

        fieldnames = [
            "query", "final_doc", "final_loss", "n_iterations",
            "success", "loss_history_json", "final_response",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "query": r.query,
                "final_doc": r.final_doc,
                "final_loss": round(r.final_loss, 6),
                "n_iterations": r.n_iterations,
                "success": r.success,
                "loss_history_json": json.dumps(
                    [round(x, 6) for x in r.loss_history]
                ),
                "final_response": r.response_history[-1] if r.response_history else "",
            })


def load_results(path: Path) -> tuple[dict, list[dict]]:
    """Returns (metadata_dict, list_of_row_dicts). Skips # comment lines."""
    path = Path(path)
    metadata: dict = {}
    rows: list[dict] = []

    with path.open() as f:
        lines = f.readlines()

    data_lines: list[str] = []
    for line in lines:
        if line.startswith("#"):
            stripped = line.lstrip("# ").strip()
            if ": " in stripped:
                k, v = stripped.split(": ", 1)
                metadata[k.strip()] = v.strip()
        else:
            data_lines.append(line)

    reader = csv.DictReader(data_lines)
    for row in reader:
        rows.append(dict(row))

    return metadata, rows
