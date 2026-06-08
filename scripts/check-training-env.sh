#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/config-pipeline-strong.json}"

PYTHONDONTWRITEBYTECODE=1 python3 scripts/check-training-env.py --config "${CONFIG}" "$@"
