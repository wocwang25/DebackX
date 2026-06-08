#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/config-pipeline-strong.json}"
HOST="${HOST:-$(PYTHONDONTWRITEBYTECODE=1 python3 -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("worker", {}).get("host", "0.0.0.0"))' "${CONFIG}")}"
PORT="${PORT:-$(PYTHONDONTWRITEBYTECODE=1 python3 -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("worker", {}).get("port", 8000))' "${CONFIG}")}"

export IIMT_CONFIG="${CONFIG}"

PYTHONDONTWRITEBYTECODE=1 python3 -m uvicorn src.worker.app:app --host "${HOST}" --port "${PORT}"
