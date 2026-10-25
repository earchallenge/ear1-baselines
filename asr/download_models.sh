#!/usr/bin/env bash
set -euo pipefail
MODEL="both"
MODEL_ROOT="models"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --model-root) MODEL_ROOT="$2"; shift 2 ;;
    --full) FULL=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$MODEL" in
  both|whisper-large-v2|whisper-large-v3) ;;
  *) echo "--model must be both, whisper-large-v2, or whisper-large-v3" >&2; exit 2 ;;
esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
ENV_NAME="${ASR_ENV_NAME:-${CONDA_DEFAULT_ENV:-asr}}"
if [[ "$MODEL_ROOT" != /* ]]; then MODEL_ROOT="$ROOT/$MODEL_ROOT"; fi
run_python() {
  if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda run --no-capture-output -n "${ENV_NAME}" python "$@"
  else
    "${PYTHON:-python}" "$@"
  fi
}
model_ready() {
  local path="$1"
  [[ -f "$path/config.json" && -f "$path/preprocessor_config.json" && \
     -f "$path/tokenizer.json" && -f "$path/tokenizer_config.json" ]] || return 1
  if find "$path" -maxdepth 1 -type f -name '*.index*.json' -print -quit | grep -q .; then
    run_python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); i=next(p.glob("*.index*.json")); d=json.loads(i.read_text()); raise SystemExit(0 if all((p/n).is_file() for n in set(d.get("weight_map",{}).values())) else 1)' "$path"
  else
    [[ -f "$path/model.safetensors" || -f "$path/pytorch_model.bin" ]]
  fi
}
if [[ "$MODEL" == "both" ]]; then
  if model_ready "$MODEL_ROOT/whisper-large-v2" && model_ready "$MODEL_ROOT/whisper-large-v3" && [[ -z "${FULL:-}" ]]; then
    echo "reuse whisper-large-v2: $MODEL_ROOT/whisper-large-v2"
    echo "reuse whisper-large-v3: $MODEL_ROOT/whisper-large-v3"
    exit 0
  fi
elif [[ -z "${FULL:-}" ]] && model_ready "$MODEL_ROOT/$MODEL"; then
  echo "reuse $MODEL: $MODEL_ROOT/$MODEL"
  exit 0
fi
CMD=(scripts/download_models.py --model "$MODEL" --model-root "$MODEL_ROOT")
if [[ -n "${FULL:-}" ]]; then CMD+=(--full); fi
run_python "${CMD[@]}"
