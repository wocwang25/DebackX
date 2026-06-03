#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/config-pipeline.json}"
SPLIT="${SPLIT:-test}"
PREDICTIONS="${PREDICTIONS:-outputs/mt/${SPLIT}.pred.vi.txt}"
OUTPUT="${OUTPUT:-outputs/mt/${SPLIT}.translation-errors.tsv}"
TOP="${TOP:-200}"

PYTHONDONTWRITEBYTECODE=1 python3 scripts/analyze-translation-errors.py \
  --config "${CONFIG}" \
  --split "${SPLIT}" \
  --predictions "${PREDICTIONS}" \
  --output "${OUTPUT}" \
  --top "${TOP}" \
  "$@"
