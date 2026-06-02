#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/config-pipeline.json}"
CHECKPOINT="${CHECKPOINT:-models/ocr-trocr-en/best}"
SPLIT="${SPLIT:-test}"
OUTPUT="${OUTPUT:-outputs/ocr/en/${SPLIT}.pred.en.txt}"

PYTHONDONTWRITEBYTECODE=1 python3 src/training/predict_ocr.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --split "${SPLIT}" \
  --output "${OUTPUT}" \
  "$@"
