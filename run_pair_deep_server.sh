#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

DEVICE="${1:-cuda}"
EPOCHS="${2:-30}"
SOURCE_TASK="${3:-vehicles1}"
TARGET_TASK="${4:-vehicles2}"
SEED="${5:-1}"

PAIR_TAG="${SOURCE_TASK}_to_${TARGET_TASK}"
ROUND_TAG="deep_${PAIR_TAG}_ep${EPOCHS}"
CONFIG_DIR="configs/generated/${ROUND_TAG}"
FAILED_REF="logs/failed_ref_${PAIR_TAG}_ep${EPOCHS}.txt"
FAILED_DYN="logs/failed_dynamic_${PAIR_TAG}_ep${EPOCHS}.txt"
FAILED_HYB="logs/failed_hybrid_${PAIR_TAG}_ep${EPOCHS}.txt"

python -m pip install -r requirements.txt

python scripts/gen_pair_deep_configs.py \
  --source-task "${SOURCE_TASK}" \
  --target-task "${TARGET_TASK}" \
  --epochs "${EPOCHS}" \
  --output "${CONFIG_DIR}" \
  --seeds "${SEED}"

echo "[REF] Running focused reference stage for ${SOURCE_TASK} ..."
python scripts/run_stage.py --config-dir "${CONFIG_DIR}/ref" --device "${DEVICE}" --failed-list "${FAILED_REF}"

echo "[DYN] Running focused target dynamic stage for ${TARGET_TASK} ..."
python scripts/run_stage.py --config-dir "${CONFIG_DIR}/dynamic" --device "${DEVICE}" --failed-list "${FAILED_DYN}"

echo "[HYB] Running focused transfer stage ${SOURCE_TASK} -> ${TARGET_TASK} ..."
python scripts/run_stage.py --config-dir "${CONFIG_DIR}/hybrid" --device "${DEVICE}" --failed-list "${FAILED_HYB}"

echo "Focused deep run finished."
echo "Generated configs: ${CONFIG_DIR}"
echo "Results root: results/runs"
echo "Use run IDs containing: _ep${EPOCHS}_s${SEED}"
