#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/config-pipeline.json}"

python3 src/pipeline/render_translations.py --config "${CONFIG}" "$@"
