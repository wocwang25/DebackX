#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/config-pipeline-strong.json}"

PYTHONDONTWRITEBYTECODE=1 python3 src/training/train_mt_nllb.py --config "${CONFIG}" "$@"
