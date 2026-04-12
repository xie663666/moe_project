#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

DEVICE="${1:-cuda}"

log_ts() {
  echo "[$(date '+%F %T')] $*"
}

log_ts "bootstrap start"
bash bootstrap_server.sh

log_ts "[REF] running reference stage"
python scripts/run_stage.py --config-dir configs/generated/main_round/ref --device "${DEVICE}" --skip-existing --failed-list logs/failed_ref_runs.txt

log_ts "[CHECK] refs ready"
python scripts/preflight_check.py --project-root . --stage refs_ready

log_ts "[DYN] running target dynamic stage"
python scripts/run_stage.py --config-dir configs/generated/main_round/dynamic --device "${DEVICE}" --skip-existing --failed-list logs/failed_dynamic_runs.txt

log_ts "[CHECK] hybrid prerequisites"
python scripts/preflight_check.py --project-root . --stage hybrid_ready

log_ts "[HYB] running hybrid/fixed stage"
python scripts/run_stage.py --config-dir configs/generated/main_round/hybrid --device "${DEVICE}" --skip-existing --failed-list logs/failed_hybrid_runs.txt

log_ts "[ANALYSIS] aggregate + tables + plots"
python scripts/aggregate_results.py --results-root results/runs --out-dir results/summaries
python scripts/make_analysis_tables.py --summaries-dir results/summaries --out-dir results/analysis/tables
python scripts/plot_results.py --summaries-dir results/summaries --tables-dir results/analysis/tables --out-dir results/analysis/plots

log_ts "full pipeline finished"
