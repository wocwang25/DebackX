#!/usr/bin/env sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

if [ -n "${PADDLEX_HOME:-}" ]; then
    PADDLEX_ROOT="${PADDLEX_HOME}"
elif [ -n "${SUDO_USER:-}" ] && [ -d "/home/${SUDO_USER}/.paddlex" ]; then
    PADDLEX_ROOT="/home/${SUDO_USER}/.paddlex"
else
    PADDLEX_ROOT="${HOME}/.paddlex"
fi

SOURCE_DIR="${PADDLEX_ROOT}/official_models"
TARGET_DIR="${ROOT_DIR}/vendor/paddlex/official_models"

for model_name in PP-OCRv5_server_det en_PP-OCRv5_mobile_rec; do
    if [ ! -d "${SOURCE_DIR}/${model_name}" ]; then
        echo "Missing ${SOURCE_DIR}/${model_name}" >&2
        echo "Run a normal image translation once, or set PADDLEX_HOME to the PaddleX cache directory." >&2
        exit 1
    fi
done

mkdir -p "${TARGET_DIR}"

for model_name in PP-OCRv5_server_det en_PP-OCRv5_mobile_rec; do
    rm -rf "${TARGET_DIR}/${model_name}"
    cp -a "${SOURCE_DIR}/${model_name}" "${TARGET_DIR}/"
done

du -sh "${TARGET_DIR}/PP-OCRv5_server_det" "${TARGET_DIR}/en_PP-OCRv5_mobile_rec"
echo "Prepared PaddleOCR release assets in ${TARGET_DIR}"
