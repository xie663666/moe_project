#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

for cfg in configs/generated/main_round/hybrid/*.yaml; do
  echo "Running ${cfg}"
  python train.py --config "${cfg}"
done
