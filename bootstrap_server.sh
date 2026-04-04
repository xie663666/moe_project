#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

echo "[1/6] Project root: ${ROOT_DIR}"

if command -v conda >/dev/null 2>&1; then
  echo "[2/6] Conda detected. Using current environment."
else
  echo "[2/6] Conda not detected. Using system python."
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "[3/6] Expanding directed task pairs..."
python scripts/expand_pairs.py --pairs configs/pairs/main_pairs.yaml --out configs/pairs/directed_main_pairs.yaml

echo "[4/6] Generating run configs..."
python scripts/gen_run_configs.py --round main --tasks configs/tasks/cifar100_superclass_tasks.yaml --pairs configs/pairs/directed_main_pairs.yaml --output configs/generated/main_round

echo "[5/6] Running smoke test..."
python scripts/smoke_test.py --project-root .

echo "[6/6] Key directory check..."
test -d configs/generated/main_round && echo "OK configs/generated/main_round"
test -d results && echo "OK results"
test -d logs && echo "OK logs"
test -d checkpoints && echo "OK checkpoints"

echo "Done."
