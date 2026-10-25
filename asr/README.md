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

After updating from an all-device CER run, rerun without `--eval-only` to rebuild
the D1/R1 manifest and predictions. An old all-device manifest is rejected by
`--eval-only`.

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

The CER workflow uses reference TextGrid utterance boundaries (oracle segmentation). The additional cpCER workflow consumes predicted RTTM from the sibling diarization component. This ASR component does not itself perform diarization or provide a finalized official ESR-ON submission pipeline. Both workflows run inference without model fine-tuning.

The current inventory selects only D1 and R1 audio/TextGrid recordings in the development sessions. Headset, H and other devices are not evaluated. Official evaluation audio is D1 only for Fixed-Array and R1 only for Moving-Array; these tracks are reported separately in the per-device and per-session-device results.

CER applies Unicode NFKC normalization, lowercase conversion, and removal of whitespace and Unicode punctuation. Traditional Chinese characters, digits, and spoken fillers are retained. Empty normalized references are excluded. Invalid intervals and segments longer than 30 seconds are excluded; an end-time excess of at most 0.01 seconds is clamped to the audio duration. Check the exported exclusion counts.

Apply both the timeline patch and the subsequent speaker-ID corrections described in the [repository README](../README.md#data-corrections) before preparing manifests. Use the updated 55/7 split and exclude 20260916-5. After changing audio or annotations, use a fresh output directory or remove the affected generated predictions before rerunning: cache checks do not fingerprint audio bytes.

Contributor-reported preliminary CER and cpCER scores are listed below. A fully validated pinned environment is not included in this source release.

## Contributor-reported preliminary development results

### Reference-segmented CER

These Whisper large-v3 CER values were reported from the contributor's existing
`ASR_Baseline` run, using reference TextGrid utterance boundaries (oracle
segmentation). D1/R1 rows were extracted from
`outputs/whisper-large-v3/dev/scores_by_session_device.csv`. They have not been
independently reproduced with this packaged recipe.

| Session | D1 CER (%) | R1 CER (%) |
|---|---:|---:|
| 20260908-1 | 85.15 | 73.91 |
| 20260908-2 | 70.59 | 73.69 |
| 20260923-1 | 57.69 | 98.39 |
| 20260923-2 | 68.61 | 55.07 |
| 20260923-5 | 41.00 | 30.28 |
| 20260924-2 | 27.45 | 12.90 |
| 20260924-5 | 71.29 | 67.39 |
| OVERALL | 61.53 | 59.57 |

OVERALL is computed independently for each device as total character edit errors
divided by total reference characters across the seven Dev sessions, multiplied
by 100. It is not the arithmetic mean of the session percentages.

Reported decoding settings are Chinese transcription (`language=zh`,
`task=transcribe`), `num_beams=1`, and `max_length=448`. The reported normalization
is NFKC, lowercase, and removal of whitespace and Unicode punctuation, as
described above. Record model revision, package versions and data revision when
reproducing these results.

### Predicted-RTTM cpCER

The contributor also completed the packaged `run_cpcer.sh` workflow using
DiariZen-predicted RTTMs and base Whisper large-v3, reporting successful
recognition of all **8,531 ASR windows** across the seven Dev sessions and both
devices. The following values are transcribed from the completed run's console
output; independent reproduction and inspection of the full result artifacts
remain pending.

| Session | D1 cpCER (%) | R1 cpCER (%) |
|---|---:|---:|
| 20260908-1 | 81.02 | 74.64 |
| 20260908-2 | 78.86 | 71.65 |
| 20260923-1 | 45.81 | 33.66 |
| 20260923-2 | 72.32 | 35.48 |
| 20260923-5 | 29.25 | 15.96 |
| 20260924-2 | 28.80 | 18.57 |
| 20260924-5 | 81.41 | 74.64 |
| OVERALL | 61.68 | 49.52 |

OVERALL is micro-averaged independently per device using total character edit
errors divided by total reference characters across sessions. The run uses the
predicted-RTTM and reference-text policies documented below, including
intersection of predicted intervals with the available recording duration.
These cpCER values and the reference-segmented CER values above use different
recognition boundaries and reference-interval filtering; they are separate
comparisons, not a controlled measure of improvement over one another.

## Predicted-RTTM cpCER: complete workflow

This workflow requires the corrected dataset's D1/R1 recordings, personal
`A/P*.TextGrid` reference transcripts, the official `split.lst`, and predicted
RTTMs. It reads Dev session IDs from the supplied split, rather than hard-coding
a second list. The released corrected split has seven Dev sessions, giving 14
session/track pairs. The excluded session `20260916-5` is ignored even if present
in an older split.

Use Linux, Bash and an NVIDIA GPU with working CUDA PyTorch for recognition.
The input checks and scoring alone do not load a GPU model. cpCER adds no new
runtime packages beyond the existing `requirements.txt`; the separate DiariZen
environment is needed only when generating RTTMs.

### 1. Install ASR and download base Whisper large-v3

From this `asr/` directory:

```bash
bash setup_env.sh
conda activate asr
bash download_models.sh --model whisper-large-v3
```

The downloaded model goes to `asr/models/whisper-large-v3/`. The cpCER runner
loads only local model files and uses this directory by default. To reuse an
existing model, pass `--model-path /path/to/whisper-large-v3`.

### 2. Obtain predicted RTTMs

Install DiariZen following the [diarization README](../diarization/README.md).
From `asr/`, run its inference stage in the sibling component:

```bash
cd ../diarization
bash run_baseline.sh \
  --data-root /path/to/dataset \
  --split /path/to/official/split.lst \
  --tracks "D1 R1" --stage infer
cd ../asr
```

This produces:

```text
../diarization/results/infer/dev/D1/<session>.rttm
../diarization/results/infer/dev/R1/<session>.rttm
```

Use **predicted** RTTMs from `infer/dev`, not reference RTTMs from `prepare/dev`.
For each Dev session and selected track, one RTTM is required. Empty RTTM files
are permitted and score as missing speech; missing files fail validation.

### 3. Validate, recognize and score

Back in `asr/`, activate the ASR environment and validate all inputs:

```bash
conda activate asr
bash run_cpcer.sh /path/to/dataset \
  --split /path/to/official/split.lst \
  --dry-run
```

The default RTTM path is resolved relative to this code directory, so the
sibling `../diarization/results/infer/dev/` works even when the script is invoked
from another working directory. User-supplied relative paths are resolved from
the caller's directory. When the dataset contains `split.lst`, `--split` can be
omitted in all cpCER commands.

After `INPUT CHECK COMPLETE`, run:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_cpcer.sh /path/to/dataset \
  --split /path/to/official/split.lst
```

If predicted RTTMs are stored elsewhere, select their Dev parent directory:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_cpcer.sh /path/to/dataset \
  --split /path/to/official/split.lst \
  --rttm-root /path/to/predicted/infer/dev \
  --model-path /path/to/local/whisper-large-v3
```

Use `--tracks D1` or `--tracks R1` for one track. `--stage prepare`, `--stage
infer`, and `--stage score` run stages independently; the default `all` runs the
complete workflow. Repeat the same command to resume successful ASR windows and
retry failed/pending ones. With matching completed predictions, `--stage score`
recalculates cpCER and reports without loading Whisper. Keep the same input,
model, environment and decoding arguments when resuming or scoring.

`run_cpcer.sh` follows the existing ASR environment selection: the active Conda
environment, overridden by `ASR_ENV_NAME`; otherwise an installed `asr`
environment, or `PYTHON`/`python`. Use `bash run_cpcer.sh --help` for options.

### cpCER outputs

```text
results/whisper-large-v3/
  prepare/dev/D1/<session>.diarized.jsonl          # Predicted ASR windows
  prepare/dev/D1/<session>.cpcer_reference.jsonl  # Reference times and exclusion flags
  prepare/dev/D1/<session>.uem                    # 0 to the D1 recording duration
  prepare/dev/R1/                                # Same structure for R1
  infer/dev/D1/<session>.diarized.jsonl           # ASR text and predicted speaker IDs
  infer/dev/D1/result_cpcer.csv                   # Each Dev session + OVERALL
  infer/dev/R1/<session>.diarized.jsonl
  infer/dev/R1/result_cpcer.csv
  cpcer_summary.json
  result_summary.md                              # D1/R1 columns for README reporting
```

Resumable/internal files remain in `outputs/whisper-large-v3/dev_diarized/`:
`manifest.json`, `predictions.jsonl`, `run_metadata.json`,
`cpcer_by_session_device.csv`, `cpcer_details.json`, `cpcer_summary.json`,
`cpcer_results.md`, and `reference_exclusions.json`. The details include
concatenated speaker streams and the optimal assignment. Metadata records
source/manifest hashes, local model fingerprints, decoding settings and package
versions. Audio is checked by path, size, modification time, rate and duration;
full WAV content is not hashed. Changed inputs/configuration block reuse rather
than mixing experiments. Select a fresh `--output-dir` (and optionally
`--results-root`) for a different experiment.

### cpCER scoring protocol

- Only D1 and R1 Dev recordings are recognition inputs. Tracks are scored
  independently. Personal TextGrid files supply scoring references only.
- Recognition uses predicted RTTM boundaries and speaker IDs exclusively.
  Intervals longer than 30 seconds are split into contiguous windows without
  overlap or dropped audio. Cross-speaker overlap remains, with both windows
  receiving the original mixture; no source separation is performed.
- Whisper uses Chinese transcription, 16 kHz audio, FP16, one beam and maximum
  token length 448. Token-limit hits are counted in the summary.
- References retain nonempty normalized personal-speaker intervals fully inside
  the selected device recording, with 10 ms end-rounding tolerance. Original
  TextGrid timestamps are preserved. Out-of-range and boundary-crossing text
  intervals are excluded in full and explicitly reported with character counts;
  partial text alignment is not inferred. Reference intervals longer than
  30 seconds are retained. Predicted RTTM intervals are intersected with each
  device's actual recording duration, following the diarization recipe's
  full-recording UEM scope. Crossing intervals are clipped for recognition;
  intervals wholly after the recording are omitted and audited. Original RTTMs
  remain unchanged. There is no arbitrary 100/200 ms tail tolerance threshold.
- Normalization matches the CER baseline: NFKC, lowercase, remove whitespace
  and Unicode punctuation. There is no OpenCC, number rewriting or filler
  removal. Text is concatenated chronologically per speaker. Minimum character
  edit errors are found over a one-to-one speaker assignment, padding unequal
  stream counts with empty text. This follows the
  [minimum-permutation construction in MeetEval](https://github.com/fgnt/meeteval/blob/main/meeteval/wer/wer/cp.py),
  using Unicode characters rather than words. The implementation uses an exact
  bit-vector edit distance and SciPy's assignment solver.
- For each device, **OVERALL = total character edit errors / total reference
  characters across Dev sessions**. It is a micro average, not an arithmetic
  mean of session percentages. Both statistics are exported separately.
  cpCER may exceed 100%. Missing speech and extra recognized text remain in the
  score. Failed/incomplete recognition blocks scoring; it is never silently
  dropped. Substitution/deletion/insertion breakdowns are not computed.
- The `.uem` files document each device's full available recording window,
  including silence. cpCER has no temporal collar or time-constrained matching.
  Recognized text from predicted false alarms in silence contributes to errors.
  The reference-interval CER baseline cannot assess silence it never transcribes.

cpCER is an additional character-level Dev comparison. **It is not cpWER and
does not replace the official ESR-ON ranking metric or official scoring tools.**
Contributor-reported cpCER values are listed above. Preserve `cpcer_summary.json`,
per-track `result_cpcer.csv` files and run metadata when reporting new results.

The supplied diarization recipe (`ear1_baseline.py`) scores DER only. Its
`0.5`-second reference-overrun threshold triggers a warning, not an exclusion or
scoring tolerance. It does not specify ASR text normalization or how to assign
characters from a reference sentence crossing the recording boundary. The
normalization and 10 ms / whole-reference-interval exclusion policies above are
inherited/local ASR policies, not confirmed by that contributor's DER logs.
An ASR scoring script or organizer specification is still needed to establish
text-level protocol equivalence; DER's time clipping cannot align partial text.

