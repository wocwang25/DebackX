#!/usr/bin/env sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "${ROOT_DIR}"

IMAGE_NAME="${IMAGE_NAME:-iimt-worker}"
TAG="${TAG:-release}"
PUSH="${PUSH:-0}"

if [ -n "${IMAGE:-}" ]; then
    TARGET_IMAGE="${IMAGE}"
elif [ -n "${DOCKERHUB_USER:-}" ]; then
    TARGET_IMAGE="${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"
else
    TARGET_IMAGE="${IMAGE_NAME}:${TAG}"
fi

if [ ! -f "${ROOT_DIR}/Dockerfile.release" ]; then
    echo "Missing ${ROOT_DIR}/Dockerfile.release" >&2
    exit 1
fi

if [ ! -d "${ROOT_DIR}/vendor/paddlex/official_models/PP-OCRv5_server_det" ] \
    || [ ! -d "${ROOT_DIR}/vendor/paddlex/official_models/en_PP-OCRv5_mobile_rec" ]; then
    echo "Missing PaddleOCR release assets." >&2
    echo "Run: sh scripts/prepare-release-assets.sh" >&2
    exit 1
fi

docker build -f "${ROOT_DIR}/Dockerfile.release" -t "${TARGET_IMAGE}" "${ROOT_DIR}"

echo "Built ${TARGET_IMAGE}"

if [ "${PUSH}" = "1" ]; then
    docker push "${TARGET_IMAGE}"
    echo "Pushed ${TARGET_IMAGE}"
fi
