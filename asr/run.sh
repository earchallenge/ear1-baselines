#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run.sh /path/to/data [--split /path/to/split.lst] [--eval-only]" >&2
  exit 2
fi
DATA_ROOT="$1"
shift
EVAL_ONLY=""
MODEL="whisper-large-v3"
SPLIT_PATH=""
if [[ $# -gt 0 ]]; then
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --eval-only) EVAL_ONLY=1; shift ;;
      --model) [[ $# -ge 2 ]] || { echo "ERROR: --model requires a value" >&2; exit 2; }; MODEL="$2"; shift 2 ;;
      --split) [[ $# -ge 2 ]] || { echo "ERROR: --split requires a path" >&2; exit 2; }; SPLIT_PATH="$2"; shift 2 ;;
      *) echo "Usage: bash run.sh /path/to/data [--model whisper-large-v2|whisper-large-v3] [--split /path/to/split.lst] [--eval-only]" >&2; exit 2 ;;
    esac
  done
fi
CALLER_ROOT="$(pwd)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ "$DATA_ROOT" != /* ]]; then DATA_ROOT="$CALLER_ROOT/$DATA_ROOT"; fi
if [[ -n "$SPLIT_PATH" && "$SPLIT_PATH" != /* ]]; then SPLIT_PATH="$CALLER_ROOT/$SPLIT_PATH"; fi
SPLIT_ARGS=()
if [[ -n "$SPLIT_PATH" ]]; then SPLIT_ARGS=(--split "$SPLIT_PATH"); fi
ENV_NAME="${ASR_ENV_NAME:-${CONDA_DEFAULT_ENV:-asr}}"
run_python() {
  if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda run --no-capture-output -n "${ENV_NAME}" python "$@"
  else
    "${PYTHON:-python}" "$@"
  fi
}
case "$MODEL" in
  whisper-large-v2|whisper-large-v3) ;;
  *) echo "--model must be whisper-large-v2 or whisper-large-v3" >&2; exit 2 ;;
esac
MANIFEST="$ROOT/data/dev.jsonl"
OUTPUT="$ROOT/outputs/$MODEL/dev"

if [[ -z "$EVAL_ONLY" ]]; then
  echo "[1/3] Preparing development manifest"
  run_python scripts/prepare_dev.py "$DATA_ROOT" --manifest "$MANIFEST" "${SPLIT_ARGS[@]}"
else
  echo "[1/3] Checking existing development manifest"
fi

check_predictions() {
  run_python - "$MANIFEST" "$OUTPUT" "$MODEL" "$DATA_ROOT" <<'PY'
import json, sys
from pathlib import Path
manifest = Path(sys.argv[1])
output = Path(sys.argv[2])
model = sys.argv[3]
data_root = str(Path(sys.argv[4]).resolve())
predictions = output / "predictions.jsonl"
metadata = output / "run_metadata.json"
if not manifest.is_file(): raise SystemExit(f"ERROR: manifest not found: {manifest}")
if not predictions.is_file() or not metadata.is_file(): raise SystemExit("ERROR: existing predictions or metadata are missing")
records = [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
rows = [json.loads(x) for x in predictions.read_text(encoding="utf-8").splitlines() if x.strip()]
meta = json.loads(metadata.read_text(encoding="utf-8"))
actual_model = meta.get("model", "")
if actual_model not in {model, f"openai/{model}"}:
    raise SystemExit(f"ERROR: existing predictions do not match model={model}")
if meta.get("data_root") and str(Path(meta["data_root"]).resolve()) != data_root:
    raise SystemExit("ERROR: existing predictions do not match data_root")
manifest_ids = {x.get("id", x.get("utterance_id")) for x in records}
prediction_ids = {x.get("utterance_id", x.get("id")) for x in rows}
if len(rows) != len(records) or manifest_ids != prediction_ids:
    raise SystemExit("ERROR: prediction count or sample IDs do not match the manifest")
by_id = {x.get("id", x.get("utterance_id")): x for x in records}
for row in rows:
    uid = row.get("utterance_id", row.get("id"))
    ref = by_id[uid]
    for key in ("audio_path", "start", "end"):
        if row.get(key) != ref.get(key):
            raise SystemExit(f"ERROR: prediction {uid} does not match manifest field {key}")
    if row.get("status") != "SUCCESS" or "prediction" not in row or not isinstance(row["prediction"], str):
        raise SystemExit(f"ERROR: prediction {uid} is incomplete")
PY
}

if [[ -n "$EVAL_ONLY" ]]; then
  check_predictions
  echo "[2/3] SKIPPED (existing predictions)"
else
  if check_predictions 2>/dev/null; then
    echo "[2/3] SKIPPED (existing predictions)"
    echo "[3/3] Evaluating $MODEL CER"
    run_python scripts/evaluate.py --model "$MODEL" --manifest "$MANIFEST" --output-root "$ROOT/outputs"
    echo "Baseline finished"
    exit 0
  fi
  echo "[2/3] Running Whisper $MODEL ASR"
  run_python scripts/infer.py --model "$MODEL" --manifest "$MANIFEST" --data-root "$DATA_ROOT" --model-root "$ROOT/models" --output-root "$ROOT/outputs"
fi

echo "[3/3] Evaluating $MODEL CER"
run_python scripts/evaluate.py --model "$MODEL" --manifest "$MANIFEST" --output-root "$ROOT/outputs"
echo "Baseline finished"

