#!/usr/bin/env bash
# =============================================================================
# EAR-1 Challenge | DiariZen Speaker Diarization Baseline
# =============================================================================
# Workflow:
#   1. prepare : Convert per-speaker TextGrid annotations to reference RTTMs.
#   2. infer   : Run DiariZen separately on D1 and/or R1 recordings.
#   3. score   : Evaluate DER with pyannote.metrics and save CSV summaries.
#
# Only D1 is an evaluation input for the Fixed-Array track; only R1 is an
# evaluation input for the Moving-Array track. H* and A/P* audio are NOT used
# for inference in this baseline.
# =============================================================================
# Reference Results | EAR-1 Development Set
# =============================================================================
#
# Model:
#   BUT-FIT/diarizen-wavlm-large-s80-md-v2
#
# Evaluation:
#   Metric       : pyannote.metrics.DiarizationErrorRate
#   Collar       : 0.0 seconds
#   Skip overlap : False
#   Reference    : A/P*.TextGrid -> RTTM
#   Scoring UEM  : Full recording duration (track-specific)
#
# Expected DER (%) on the development set:
#
#   Session        Fixed-Array (D1)    Moving-Array (R1)
#   ---------------------------------------------------
#   20260908-1          25.74               28.60
#   20260908-2          29.17               29.77
#   20260923-1          22.67               46.93
#   20260923-2          42.53               40.58
#   20260923-5          14.89               15.43
#   20260924-2          16.79               14.84
#   20260924-5          45.89               48.48
#   ---------------------------------------------------
#   OVERALL             29.34               32.61
#
# Overall DER is micro-averaged across all seven development sessions.
#
# Notes:
#   1. Results were obtained with the corrected EAR-1 annotations.
#   2. D1 and R1 are evaluated independently using single-channel audio.
#   3. Reference timestamps are preserved without truncation.
#   4. Evaluation is restricted to the corresponding audio duration.
#   5. These are preliminary reproducibility results, NOT official
#      challenge baseline scores.
#   6. Minor differences may arise from software versions or scoring
#      configurations.
#
# =============================================================================
#
# Default usage:
#   bash run_baseline.sh
#
# Common overrides:
#   bash run_baseline.sh --tracks D1
#   bash run_baseline.sh --stage prepare
#   bash run_baseline.sh --stage infer --tracks R1 --overwrite
#   bash run_baseline.sh --stage score --collar 0.25
#   bash run_baseline.sh --data-root /path/to/EarChallenge --out-dir ./results
#
# Note: These are preliminary results. Re-check DER options against the
#       official challenge scoring script when it becomes available.
# =============================================================================

set -Eeuo pipefail

# =============================================================================
# Hugging Face settings
# =============================================================================
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"  # 1 for offline

# -----------------------------------------------------------------------------
# 1. Paths and default experiment settings
#    All values can be overridden by environment variables or CLI arguments.
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# Conda environment
# =============================================================================

CONDA_ENV="${DIAR_ENV_NAME:-diarizen}"

DATA_ROOT="${DATA_ROOT:-/Your_Path/EarChallenge}"
SPLIT_PATH="${SPLIT_PATH:-}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/results}"
PYTHON="${PYTHON:-python}"

MODEL_ID="${MODEL_ID:-BUT-FIT/diarizen-wavlm-large-s80-md-v2}"
TRACKS="${TRACKS:-D1 R1}"                  # D1, R1, or "D1 R1"
STAGE="${STAGE:-all}"                      # all, prepare, infer, score
COLLAR="${COLLAR:-0.0}"                     # pyannote DER collar, seconds
SKIP_OVERLAP="${SKIP_OVERLAP:-false}"       # true: exclude overlapping speech
OVERWRITE="${OVERWRITE:-false}"             # true: re-run existing hypothesis RTTMs

# -----------------------------------------------------------------------------
# 2. CLI / logging helpers
# -----------------------------------------------------------------------------
usage() {
    cat <<'USAGE'
Usage: bash run_baseline.sh [OPTIONS]

Options:
  --data-root DIR     EAR-1 dataset directory
  --split FILE        Official split file (default: DATA_ROOT/split.lst)
  --out-dir DIR       Experiment output directory
  --python CMD        Python interpreter (default: python)
  --model ID          Hugging Face DiariZen model ID
  --tracks TRACKS     "D1 R1" (default), D1, or R1
  --stage STAGE       all (default), prepare, infer, or score
  --collar SEC        DER collar in seconds (default: 0.0)
  --skip-overlap      Ignore overlap during DER evaluation
  --overwrite         Recompute hypothesis RTTMs even if they exist
  -h, --help          Show this help

Examples:
  bash run_baseline.sh --tracks D1
  bash run_baseline.sh --stage prepare
  bash run_baseline.sh --stage infer --tracks R1 --overwrite
  bash run_baseline.sh --stage score --collar 0.25
USAGE
}

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

trap 'printf "ERROR: command failed at line %s (exit status %s)\n" "$LINENO" "$?" >&2' ERR

# -----------------------------------------------------------------------------
# 3. Parse optional command-line overrides
# -----------------------------------------------------------------------------
while (( $# > 0 )); do
    case "$1" in
        --data-root)    (( $# >= 2 )) || die "Missing --data-root value"; DATA_ROOT="$2"; shift 2 ;;
        --split)        (( $# >= 2 )) || die "Missing --split value"; SPLIT_PATH="$2"; shift 2 ;;
        --out-dir)      (( $# >= 2 )) || die "Missing --out-dir value"; OUT_DIR="$2"; shift 2 ;;
        --python)       (( $# >= 2 )) || die "Missing --python value"; PYTHON="$2"; shift 2 ;;
        --model)        (( $# >= 2 )) || die "Missing --model value"; MODEL_ID="$2"; shift 2 ;;
        --tracks)       (( $# >= 2 )) || die "Missing --tracks value"; TRACKS="$2"; shift 2 ;;
        --stage)        (( $# >= 2 )) || die "Missing --stage value"; STAGE="$2"; shift 2 ;;
        --collar)       (( $# >= 2 )) || die "Missing --collar value"; COLLAR="$2"; shift 2 ;;
        --skip-overlap) SKIP_OVERLAP=true; shift ;;
        --overwrite)    OVERWRITE=true; shift ;;
        -h|--help)      usage; exit 0 ;;
        *)              die "Unknown argument: $1 (try --help)" ;;
    esac
done

# Activate Conda when available; otherwise use the selected Python environment.
if command -v conda >/dev/null 2>&1; then
    set +u
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
        conda activate "$CONDA_ENV"
    fi
    set -u
fi

# -----------------------------------------------------------------------------
# 4. Validate configuration before processing the dataset
# -----------------------------------------------------------------------------
[[ -d "$DATA_ROOT" ]] || die "Dataset directory does not exist: $DATA_ROOT"
[[ -f "$SCRIPT_DIR/ear1_baseline.py" ]] || die "Missing $SCRIPT_DIR/ear1_baseline.py"
command -v "$PYTHON" >/dev/null 2>&1 || die "Python interpreter not found: $PYTHON"

case "$STAGE" in
    all|prepare|infer|score) ;;
    *) die "Invalid stage '$STAGE' (expected all|prepare|infer|score)" ;;
esac

# Accept both TRACKS="D1 R1" and TRACKS="D1,R1".
read -r -a TRACK_ARRAY <<< "${TRACKS//,/ }"
(( ${#TRACK_ARRAY[@]} > 0 )) || die "At least one track is required"
for track in "${TRACK_ARRAY[@]}"; do
    case "$track" in
        D1|R1) ;;
        *) die "Invalid track '$track' (expected D1 or R1)" ;;
    esac
done

case "$SKIP_OVERLAP" in
    true|false) ;;
    *) die "SKIP_OVERLAP must be 'true' or 'false'" ;;
esac
case "$OVERWRITE" in
    true|false) ;;
    *) die "OVERWRITE must be 'true' or 'false'" ;;
esac

mkdir -p -- "$OUT_DIR"

log "EAR-1 baseline started"
log "Dataset : $DATA_ROOT"
log "Output  : $OUT_DIR"
log "Model   : $MODEL_ID"
log "Tracks  : ${TRACK_ARRAY[*]}"
log "Stage   : $STAGE"
log "Scoring : collar=${COLLAR}s, skip_overlap=${SKIP_OVERLAP}"

# Common arguments shared by all Python stages.
COMMON_ARGS=(
    --data "$DATA_ROOT"
    --out "$OUT_DIR"
    --tracks "${TRACK_ARRAY[@]}"
)
if [[ -n "$SPLIT_PATH" ]]; then
    COMMON_ARGS+=(--split "$SPLIT_PATH")
fi

# -----------------------------------------------------------------------------
# 5. Stage functions
# -----------------------------------------------------------------------------
run_prepare() {
    log "[1/3] PREPARE | TextGrid -> reference RTTM"
    "$PYTHON" -u "$SCRIPT_DIR/ear1_baseline.py" prepare \
        "${COMMON_ARGS[@]}"
}

run_infer() {
    log "[2/3] INFER | DiariZen on ${TRACK_ARRAY[*]}"

    local extra_args=()
    if [[ "$OVERWRITE" == true ]]; then
        extra_args+=(--overwrite)
    fi

    "$PYTHON" -u "$SCRIPT_DIR/ear1_baseline.py" infer \
        "${COMMON_ARGS[@]}" \
        --model "$MODEL_ID" \
        "${extra_args[@]}"
}

run_score() {
    log "[3/3] SCORE | pyannote.metrics DER"

    local extra_args=()
    if [[ "$SKIP_OVERLAP" == true ]]; then
        extra_args+=(--skip-overlap)
    fi

    "$PYTHON" -u "$SCRIPT_DIR/ear1_baseline.py" score \
        "${COMMON_ARGS[@]}" \
        --collar "$COLLAR" \
        "${extra_args[@]}"
}

# -----------------------------------------------------------------------------
# 6. Execute the requested stage(s)
# -----------------------------------------------------------------------------
case "$STAGE" in
    all)
        run_prepare
        run_infer
        run_score
        ;;
    prepare) run_prepare ;;
    infer)   run_infer   ;;
    score)   run_score   ;;
esac

log "EAR-1 baseline finished successfully. Results: $OUT_DIR"
