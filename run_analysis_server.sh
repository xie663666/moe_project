#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
python scripts/aggregate_results.py --results-root results/runs --out-dir results/summaries
python scripts/make_analysis_tables.py --summaries-dir results/summaries --out-dir results/analysis/tables
python scripts/plot_results.py --summaries-dir results/summaries --tables-dir results/analysis/tables --out-dir results/analysis/plots
echo "Analysis finished."
