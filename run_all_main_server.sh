#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

DEVICE="${1:-cuda}"

bash bootstrap_server.sh

echo "[REF] Running reference stage..."
python scripts/run_stage.py --config-dir configs/generated/main_round/ref --device "${DEVICE}" --skip-existing --failed-list logs/failed_ref_runs.txt

echo "[CHECK] Reference outputs..."
python scripts/preflight_check.py --project-root . --stage refs_ready

echo "[DYN] Running target dynamic stage..."
python scripts/run_stage.py --config-dir configs/generated/main_round/dynamic --device "${DEVICE}" --skip-existing --failed-list logs/failed_dynamic_runs.txt

echo "[HYB] Running hybrid/fixed stage..."
python scripts/preflight_check.py --project-root . --stage hybrid_ready
python scripts/run_stage.py --config-dir configs/generated/main_round/hybrid --device "${DEVICE}" --skip-existing --failed-list logs/failed_hybrid_runs.txt

echo "[ANALYSIS] Aggregating, making tables, and plotting..."
python scripts/aggregate_results.py --results-root results/runs --out-dir results/summaries
python scripts/make_analysis_tables.py --summaries-dir results/summaries --out-dir results/analysis/tables
python scripts/plot_results.py --summaries-dir results/summaries --tables-dir results/analysis/tables --out-dir results/analysis/plots

echo "Full pipeline finished."
