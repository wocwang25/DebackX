#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/config-pipeline-strong.json}"
CHECKPOINT="${CHECKPOINT:-models/mt-nllb-1p3b-en-vi/best}"
SPLIT="${SPLIT:-test}"
OUTPUT="${OUTPUT:-outputs/mt/${SPLIT}.pred.vi.txt}"
INPUT="${INPUT:-}"
NUM_BEAMS="${NUM_BEAMS:-}"
LENGTH_PENALTY="${LENGTH_PENALTY:-}"
NO_REPEAT_NGRAM_SIZE="${NO_REPEAT_NGRAM_SIZE:-}"
REPETITION_PENALTY="${REPETITION_PENALTY:-}"

EXTRA_ARGS=""
if [ -n "${NUM_BEAMS}" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --num-beams ${NUM_BEAMS}"
fi
if [ -n "${LENGTH_PENALTY}" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --length-penalty ${LENGTH_PENALTY}"
fi
if [ -n "${NO_REPEAT_NGRAM_SIZE}" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --no-repeat-ngram-size ${NO_REPEAT_NGRAM_SIZE}"
fi
if [ -n "${REPETITION_PENALTY}" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --repetition-penalty ${REPETITION_PENALTY}"
fi

if [ -n "${INPUT}" ]; then
  PYTHONDONTWRITEBYTECODE=1 python3 src/training/predict_mt.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --split "${SPLIT}" \
    --output "${OUTPUT}" \
    --input "${INPUT}" \
    ${EXTRA_ARGS} \
    "$@"
else
  PYTHONDONTWRITEBYTECODE=1 python3 src/training/predict_mt.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --split "${SPLIT}" \
    --output "${OUTPUT}" \
    ${EXTRA_ARGS} \
    "$@"
fi
