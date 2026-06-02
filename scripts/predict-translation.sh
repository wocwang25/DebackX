#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/config-pipeline.json}"
CHECKPOINT="${CHECKPOINT:-models/mt-nllb-en-vi/best}"
SPLIT="${SPLIT:-test}"
OUTPUT="${OUTPUT:-outputs/mt/${SPLIT}.pred.vi.txt}"
INPUT="${INPUT:-}"

if [ -n "${INPUT}" ]; then
  PYTHONDONTWRITEBYTECODE=1 python3 src/training/predict_mt.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --split "${SPLIT}" \
    --output "${OUTPUT}" \
    --input "${INPUT}" \
    "$@"
else
  PYTHONDONTWRITEBYTECODE=1 python3 src/training/predict_mt.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --split "${SPLIT}" \
    --output "${OUTPUT}" \
    "$@"
fi
