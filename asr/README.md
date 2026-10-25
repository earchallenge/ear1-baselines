# EAR-1 ASR Baseline

Whisper large-v3 ASR on the development set, with CER evaluation.

## Scripts

| File | Purpose |
|---|---|
| `scripts/download_models.py` | Download Whisper weights |
| `scripts/prepare_dev.py` | Prepare the development-set manifest |
| `scripts/infer.py` | Run ASR and save predictions |
| `scripts/evaluate.py` | Calculate CER and export results |
| `run.sh` | Run the baseline workflow |

## Environment

Use the environment setup script, if included:

```bash
bash setup_env.sh
```

Or create and activate a Conda environment manually:

```bash
ENV_NAME=asr
conda create -n "$ENV_NAME" python=3.10 -y
conda activate "$ENV_NAME"
pip install -r requirements.txt
```

After activation, `download_models.sh` and `run.sh` use the active Conda environment. If no environment is active, they use the default `asr` environment when it is available.

## Data and run

Place the development-set audio and TextGrid session directories anywhere accessible to the machine. The official train/development split is read from `DATA_ROOT/split.lst`, supplied with the dataset. The repository does not duplicate the split. If the file is stored elsewhere, pass `--split /path/to/split.lst` to `run.sh` or `scripts/prepare_dev.py`. The split uses `Train Set:` and `Dev Set:` headings (full-width colons are also accepted), followed by session directory names.

## Run

Download the default Whisper large-v3 model, then run from the project root:

```bash
bash download_models.sh
bash run.sh /path/to/dataset
```

The download command fetches both official Whisper large-v2 and large-v3 weights and reuses complete local models. The baseline workflow uses large-v3 by default. To use large-v2, select it explicitly:

```bash
bash run.sh /path/to/dataset --model whisper-large-v2
```

If predictions and matching metadata already exist, stage 2 is skipped automatically.

To recalculate CER from existing predictions only:

```bash
bash run.sh /path/to/dataset --eval-only
```

`--eval-only` reports an error if the manifest, prediction file, model, data root, sample IDs, audio boundaries, or utterance count do not match. It never runs ASR.

The workflow prepares the manifest, transcribes development audio, and calculates CER.

```bash
bash run.sh /path/to/Train\&Dev --split /path/to/split.lst
```

The Python scripts can also be run independently: `prepare_dev.py DATA_ROOT`, `infer.py --manifest ... --model-root ... --output-root ...`, and `evaluate.py --manifest ... --output-root ...`.

For users who prefer to run each stage directly, use:

```bash
# Prepare the development manifest from the dataset split
python scripts/prepare_dev.py /path/to/dataset --manifest data/dev.jsonl

# Run Whisper ASR
python scripts/infer.py \
  --model whisper-large-v3 \
  --manifest data/dev.jsonl \
  --model-root models \
  --output-root outputs

# Recalculate CER from the existing predictions
python scripts/evaluate.py \
  --model whisper-large-v3 \
  --manifest data/dev.jsonl \
  --output-root outputs
```

To use large-v2, replace `whisper-large-v3` in the inference and evaluation commands with `whisper-large-v2`. The Python commands should be run in the environment where the requirements are installed.

## Output

```text
outputs/<model>/dev/
├── predictions.jsonl              # Utterance-level ASR outputs
├── scores_by_session.csv          # CER for each session
├── scores_by_device.csv           # CER for each device
├── scores_by_session_device.csv   # CER for each session-device pair
└── scores_summary.json            # Overall statistics and excluded-pair reasons
```

`predictions.jsonl` contains the utterance ID, Session, device, audio path, segment boundaries, raw recognition text, and normalized recognition text. CER and edit statistics are written to the CSV and JSON summary files.

`Session-averaged CER (%)` averages the available scorable session CERs equally. Each session CER is computed from its accumulated character errors divided by its reference characters. `Overall CER (%)` divides total character errors by total reference characters across the complete development set. Neither metric is an utterance-level arithmetic mean.

This is an independent ASR Baseline for development-set reproduction and comparison. It does not include model fine-tuning or other challenge tracks, and the development-set CER is not an official competition ranking.

## Evaluation protocol and current scope

This component uses reference TextGrid utterance boundaries (oracle segmentation); it does not perform diarization or provide an end-to-end ESR-ON submission pipeline. It runs inference without model fine-tuning.

The current inventory discovers all paired audio/TextGrid recordings in the development sessions, including headset and H recordings where available. Per-device results are supplementary comparisons. Official evaluation audio is D1 only for Fixed-Array and R1 only for Moving-Array; these tracks are tested separately.

CER applies Unicode NFKC normalization, lowercase conversion, and removal of whitespace and Unicode punctuation. Traditional Chinese characters, digits, and spoken fillers are retained. Empty normalized references are excluded. Invalid intervals and segments longer than 30 seconds are excluded; an end-time excess of at most 0.01 seconds is clamped to the audio duration. Check the exported exclusion counts.

Apply both the timeline patch and the subsequent speaker-ID corrections described in the [repository README](../README.md#data-corrections) before preparing manifests. Use the updated 55/7 split and exclude 20260916-5. After changing audio or annotations, use a fresh output directory or remove the affected generated predictions before rerunning: cache checks do not fingerprint audio bytes.

No measured baseline scores or fully validated pinned environment are included in this initial source release.
