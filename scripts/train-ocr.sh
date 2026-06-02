#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/config-pipeline.json}"

sh scripts/prepare-ocr-dataset.sh
PYTHONDONTWRITEBYTECODE=1 python3 src/training/train_ocr_trocr.py --config "${CONFIG}" "$@"
