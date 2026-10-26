# EAR-1 ASR Baseline

Whisper large-v3 ASR on the development set, with CER evaluation.

An additional `run_cpcer.sh` workflow runs base Whisper large-v3 on **predicted
DiariZen RTTM segments**, then calculates character-level concatenated
minimum-permutation error rate (cpCER) for D1 and R1 independently. It uses the
same dataset split and dependencies as the CER baseline. Both workflows evaluate
D1 and R1 only. The CER commands, output formats and scoring rules are preserved.

## Scripts

| File | Purpose |
|---|---|
| `scripts/download_models.py` | Download Whisper weights |
| `scripts/prepare_dev.py` | Prepare the development-set manifest |
| `scripts/infer.py` | Run ASR and save predictions |
| `scripts/evaluate.py` | Calculate CER and export results |
| `run.sh` | Run the baseline workflow |
| `run_cpcer.sh` | Predicted-RTTM Whisper large-v3 ASR and cpCER |
| `scripts/diarized_asr.py` | Prepare RTTM windows, recognize, resume and score |
| `utils/cpcer.py` | Exact character edit distance and speaker assignment |
| `utils/cpcer_reports.py` | Per-session D1/R1 reports and micro averages |

## Environment

Use Linux, Bash and Python 3.10 with CUDA PyTorch. From this `asr/` directory, run:

```bash
bash setup_env.sh
conda activate asr
```

Or create and activate a Conda environment manually:

```bash
conda create -n asr python=3.10 -y
conda activate asr
pip install -r requirements.txt
```

After activation, `download_models.sh`, `run.sh` and `run_cpcer.sh` use the active Conda environment. If no environment is active, they use the default `asr` environment when it is available.

## Data and run

Apply all [organizer corrections](../README.md#data-corrections), including speaker
ID updates and removal of 20260916-5. Use the official split from the data package.
After corrections, regenerate manifests and affected predictions.

Place the development-set audio and TextGrid session directories anywhere accessible to the machine. The official train/development split is read from `DATA_ROOT/split.lst`, supplied with the dataset. The repository does not duplicate the split. If the file is stored elsewhere, pass `--split /path/to/split.lst` to `run.sh`, `run_cpcer.sh` or `scripts/prepare_dev.py`. The split uses `Train Set:` and `Dev Set:` headings (full-width colons are also accepted), followed by session directory names.

For cpCER, also provide predicted `D1/<session>.rttm` and `R1/<session>.rttm`
files from the [diarization baseline](../diarization/README.md). The default RTTM
root is `../diarization/results/infer/dev/`. Personal `A/P*.TextGrid` files supply
the reference transcripts.

## Run

Download the Whisper models, then run CER from this `asr/` directory:

```bash
bash download_models.sh
CUDA_VISIBLE_DEVICES=0 bash run.sh /path/to/dataset
```

The download command fetches both official Whisper large-v2 and large-v3 weights and reuses complete local models. The baseline workflow uses large-v3 by default. To use large-v2, select it explicitly:

```bash
bash run.sh /path/to/dataset --model whisper-large-v2
```

Replace `/path/to/...` with your paths and `CUDA_VISIBLE_DEVICES=0` with the GPU
to use. Quote paths containing spaces or `&`, for example `"/data/Train&Dev"`.
To download only large-v3, use `bash download_models.sh --model whisper-large-v3`.

If predictions and matching metadata already exist, stage 2 is skipped automatically.

To recalculate CER from existing predictions only:

```bash
bash run.sh /path/to/dataset --eval-only
```

`--eval-only` reports an error if the manifest, prediction file, model, data root, sample IDs, audio boundaries, or utterance count do not match. It never runs ASR.

After updating from an all-device CER run, rerun without `--eval-only` to rebuild
the D1/R1 manifest and predictions. An old all-device manifest is rejected by
`--eval-only`.

The workflow prepares the manifest, transcribes development audio, and calculates CER.

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh "/path/to/Train&Dev" --split /path/to/split.lst
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

To calculate cpCER using predicted RTTMs, activate the ASR environment and run:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_cpcer.sh /path/to/dataset \
  --split /path/to/official/split.lst \
  --rttm-root /path/to/predicted/infer/dev \
  --model-path /path/to/whisper-large-v3 \
  --output-dir outputs/whisper-large-v3/dev_diarized \
  --results-root results
```

`--rttm-root` is the directory containing predicted `D1/` and `R1/` RTTMs.
`--model-path` points to a complete local Whisper large-v3 model and defaults to
`models/whisper-large-v3/`. The output paths above are also the defaults.
If `split.lst` is in the data root, omit `--split`.

Append `--dry-run` to validate inputs before recognition. After
`INPUT CHECK COMPLETE`, remove it to run ASR and scoring. Both D1 and R1 are
selected by default; use `--tracks D1` or `--tracks R1` for one track.
Repeat the same command to resume, or append `--stage score` to recalculate
cpCER from completed predictions. Use a new `--output-dir` when changing inputs
or configuration, and a separate `--results-root` to retain previous reports.
See `bash run_cpcer.sh --help` for all options.

## Outputs and scoring

```text
outputs/<model>/dev/
├── predictions.jsonl              # D1/R1 utterance-level ASR outputs
├── scores_by_session.csv          # Per-session CER, pooling D1 and R1
├── scores_by_device.csv           # D1 and R1 CER separately, across Dev sessions
├── scores_by_session_device.csv   # Per-session CER for D1 and R1 separately
└── scores_summary.json            # Pooled D1/R1 statistics and excluded-pair reasons
```

`predictions.jsonl` contains the utterance ID, Session, device, audio path, segment boundaries, raw recognition text, and normalized recognition text. CER and edit statistics are written to the CSV and JSON summary files.

The standard `run.sh` workflow evaluates D1 and R1 only. Use
`scores_by_session_device.csv` for each session's separate D1/R1 scores and
`scores_by_device.csv` for their separate Dev totals. With seven fully scorable
Dev sessions, these files contain 14 and 2 result rows, respectively.

`scores_by_session.csv` pools D1/R1 character errors and reference characters
within each session. In `scores_summary.json`, `overall_cer` pools both tracks
across all Dev sessions, while `session_averaged_cer` averages the scorable
session CERs equally. These JSON CER values are ratios; CSV CER values are percentages.

The standalone `scripts/evaluate.py` also scores D1/R1 only. Other devices are
ignored in both predictions and device exclusion statistics; their prediction
count is recorded as `ignored_non_eval_records` in `scores_summary.json`.
If no D1/R1 predictions are present, evaluation stops with an error.

cpCER exports per-session results and an OVERALL row for each device:

```text
results/whisper-large-v3/
  prepare/dev/D1/<session>.diarized.jsonl
  prepare/dev/D1/<session>.cpcer_reference.jsonl
  prepare/dev/D1/<session>.uem
  prepare/dev/R1/                       # Same structure for R1
  infer/dev/D1/<session>.diarized.jsonl
  infer/dev/D1/result_cpcer.csv
  infer/dev/R1/<session>.diarized.jsonl
  infer/dev/R1/result_cpcer.csv
  cpcer_summary.json
  result_summary.md
```

Recognition progress, predictions, run metadata and detailed scores are saved
under `outputs/whisper-large-v3/dev_diarized/`. Reference exclusions are recorded
in `reference_exclusions.json`.

Both metrics normalize text with NFKC, lowercase, and removal of whitespace and
Unicode punctuation. Whisper uses Chinese transcription, 16 kHz audio, one beam
and maximum token length 448. In the D1/R1 results table below and the cpCER
per-track reports, **OVERALL is micro-averaged per track:** total character edit
errors / total reference characters across Dev sessions × 100.

CER scores reference TextGrid utterances. cpCER concatenates text chronologically
per speaker and minimizes total character edit errors over speaker assignments.
Predicted RTTM segments are clipped to recording duration and split into windows
of at most 30 seconds, retaining overlapping speech in the original mixture.

Reference end times allow 10 ms rounding tolerance. Invalid, empty or out-of-range
references are excluded; boundary-crossing reference text is excluded in full.
CER excludes reference intervals longer than 30 seconds; cpCER retains them.
Exclusions are recorded in the reports.

## Contributor-reported preliminary development results

CER values come from the contributor's existing `ASR_Baseline` output. cpCER
values come from the completed DiariZen + Whisper run (8,531 ASR windows).

| Session | D1 CER (%) | R1 CER (%) | D1 cpCER (%) | R1 cpCER (%) |
|---|---:|---:|---:|---:|
| 20260908-1 | 85.15 | 73.91 | 81.02 | 74.64 |
| 20260908-2 | 70.59 | 73.69 | 78.86 | 71.65 |
| 20260923-1 | 57.69 | 98.39 | 45.81 | 33.66 |
| 20260923-2 | 68.61 | 55.07 | 72.32 | 35.48 |
| 20260923-5 | 41.00 | 30.28 | 29.25 | 15.96 |
| 20260924-2 | 27.45 | 12.90 | 28.80 | 18.57 |
| 20260924-5 | 71.29 | 67.39 | 81.41 | 74.64 |
| OVERALL | 61.53 | 59.57 | 61.68 | 49.52 |

Versions and scoring settings can affect reproduction. Record model, environment
and data revisions with your results. Follow upstream licenses and cite
[Whisper](https://github.com/openai/whisper) and
[DiariZen](https://github.com/BUTSpeechFIT/DiariZen).
