#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/config-pipeline.json}"
CHECKPOINT="${CHECKPOINT:-models/mt-nllb-en-vi/best}"
SPLIT="${SPLIT:-test}"
INPUT="${INPUT:-}"
TAG="${TAG:-variant}"
BEAMS="${BEAMS:-4 5 6}"
LENGTH_PENALTIES="${LENGTH_PENALTIES:-0.9 1.0 1.1}"

for BEAM in ${BEAMS}; do
  for LENGTH_PENALTY in ${LENGTH_PENALTIES}; do
    SAFE_LP="$(printf "%s" "${LENGTH_PENALTY}" | tr "." "p")"
    OUTPUT="outputs/mt/${SPLIT}.${TAG}.beam${BEAM}.lp${SAFE_LP}.pred.vi.txt"
    echo "predicting ${OUTPUT}"
    INPUT="${INPUT}" \
    CONFIG="${CONFIG}" \
    CHECKPOINT="${CHECKPOINT}" \
    SPLIT="${SPLIT}" \
    OUTPUT="${OUTPUT}" \
    NUM_BEAMS="${BEAM}" \
    LENGTH_PENALTY="${LENGTH_PENALTY}" \
    sh scripts/predict-translation.sh

    PREDICTIONS="${OUTPUT}" \
    OUTPUT="${OUTPUT%.txt}.errors.tsv" \
    CONFIG="${CONFIG}" \
    SPLIT="${SPLIT}" \
    TOP=50 \
    sh scripts/analyze-translation-errors.sh
  done
done
