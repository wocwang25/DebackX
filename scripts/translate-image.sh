#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/config-pipeline-strong.json}"

PYTHONDONTWRITEBYTECODE=1 python3 src/pipeline/translate_real_image.py --config "${CONFIG}" "$@"
