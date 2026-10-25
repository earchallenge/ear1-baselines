#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="asr"
PYTHON_VERSION="${ASR_PYTHON_VERSION:-3.10}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name) ENV_NAME="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
ENV_NAME="${ASR_ENV_NAME:-${ENV_NAME}}"

command -v conda >/dev/null 2>&1 || {
  echo "conda is required to create the ${ENV_NAME} environment" >&2
  exit 1
}

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi
conda run --no-capture-output -n "${ENV_NAME}" python -m pip install -r "${ROOT}/requirements.txt"
echo "Environment ready: ${ENV_NAME}"
