#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/config-pipeline-strong.json}"
SPLIT="${SPLIT:-test}"
PREDICTIONS="${PREDICTIONS:-outputs/mt/${SPLIT}.1p3b.pred.vi.txt}"
OUTPUT="${OUTPUT:-outputs/mt/${SPLIT}.1p3b.translation-errors.tsv}"
TOP="${TOP:-200}"

PYTHONDONTWRITEBYTECODE=1 python3 scripts/analyze-translation-errors.py \
  --config "${CONFIG}" \
  --split "${SPLIT}" \
  --predictions "${PREDICTIONS}" \
  --output "${OUTPUT}" \
  --top "${TOP}" \
  "$@"
