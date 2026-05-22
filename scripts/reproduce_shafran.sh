#!/usr/bin/env bash
# Reproduces one row of Shafran et al. Table 2 (NQ, Mistral-7B-Instruct-v0.2).
# Runs 100 queries with BBO baseline.
#
# Usage:
#   bash scripts/reproduce_shafran.sh
#   bash scripts/reproduce_shafran.sh attack.num_queries=10  # quick smoke test

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXTRA_ARGS="${@}"

export HF_HOME=/home/ishana/scratch/hf_cache
export CUDA_VISIBLE_DEVICES=1

uv run python src/experiments/run_attack.py \
    --config-name reproduction \
    $EXTRA_ARGS
