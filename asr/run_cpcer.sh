#!/usr/bin/env bash
set -euo pipefail

usage() {
  while IFS= read -r line; do printf '%s\n' "$line"; done <<'USAGE'
Usage: bash run_cpcer.sh DATA_ROOT [OPTIONS]

Whisper large-v3 ASR on predicted DiariZen segments, followed by Dev cpCER.

Options:
  --split FILE        Official split (default: DATA_ROOT/split.lst)
  --rttm-root DIR     Predicted RTTMs: DIR/D1/<session>.rttm and DIR/R1/<session>.rttm
                      Default: ../diarization/results/infer/dev relative to asr/
  --model-path DIR    Local Whisper large-v3 (default: asr/models/whisper-large-v3)
  --tracks D1 R1      Track selection (default: both)
  --output-dir DIR    Manifest, resumable predictions and detailed scores
  --results-root DIR  Per-track reports under DIR/whisper-large-v3/
  --stage STAGE       prepare, infer, score, or all (default: all)
  --device DEVICE     PyTorch CUDA device (default: cuda)
  --dry-run           Validate/prepare inputs without loading Whisper
  -h, --help          Show this help

The active Conda environment is used; ASR_ENV_NAME overrides it. With no active
environment, an installed 'asr' environment is used when available. Otherwise
PYTHON (or python) is used. Matching successful ASR windows resume automatically.
USAGE
}

if [[ $# -eq 0 ]]; then usage >&2; exit 2; fi
if [[ "$1" == "--help" || "$1" == "-h" ]]; then usage; exit 0; fi
if [[ "$1" == --* ]]; then usage >&2; exit 2; fi
DATA_ROOT="$1"
shift
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ASR_ENV_NAME:-${CONDA_DEFAULT_ENV:-asr}}"
run_python() {
  if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda run --no-capture-output -n "$ENV_NAME" python "$@"
  else
    "${PYTHON:-python}" "$@"
  fi
}
# User-supplied relative paths retain the caller's working-directory semantics.
run_python "$ROOT/scripts/diarized_asr.py" --data-root "$DATA_ROOT" --resume "$@"
