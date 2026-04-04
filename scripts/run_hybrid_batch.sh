#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
python scripts/run_stage.py --config-dir configs/generated/main_round/hybrid --device "${1:-cuda}" --skip-existing --failed-list logs/failed_hybrid_runs.txt
