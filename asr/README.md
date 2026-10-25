# EAR-1 ASR Baseline

Base Whisper large-v3 development baseline. D1 and R1 are evaluated separately:
`run.sh` reports CER using reference TextGrid segments; `run_cpcer.sh` reports
cpCER using DiariZen-predicted RTTM segments and speaker IDs.

## Installation

Use Linux, Bash and Python 3.10 with CUDA PyTorch. From `asr/`:

```bash
bash setup_env.sh
conda activate asr
bash download_models.sh --model whisper-large-v3
```

The scripts use the active Conda environment; set `ASR_ENV_NAME` to override it.
Weights are stored in `models/whisper-large-v3/`. For cpCER, generate predicted
RTTMs using the separate environment and commands in the
[diarization README](../diarization/README.md).

## Data and commands

Apply the [data corrections](../README.md#data-corrections) and use the official
55/7 split, excluding `20260916-5`. The data root contains session folders.
Both scripts default to `DATA_ROOT/split.lst`; use `--split` for another path.
CER uses matching D1/R1 audio and TextGrids. cpCER uses D1/R1 WAVs, personal
`A/P*.TextGrid` references and predicted `D1/<session>.rttm` and
`R1/<session>.rttm` files under the RTTM root.

```bash
# Reference-segmented CER
CUDA_VISIBLE_DEVICES=0 bash run.sh /path/to/dataset \
  --split /path/to/official/split.lst

# Predicted-RTTM cpCER
CUDA_VISIBLE_DEVICES=0 bash run_cpcer.sh /path/to/dataset \
  --split /path/to/official/split.lst \
  --rttm-root /path/to/predicted/infer/dev
```

Change `CUDA_VISIBLE_DEVICES` to select a GPU. For cpCER, use `--model-path` to
select existing local weights, `--dry-run` to check inputs, and `--help` for
options. Its default RTTM root is `../diarization/results/infer/dev/`.

To score existing predictions, add `--eval-only` for CER or `--stage score` for
cpCER. Repeating the same cpCER command resumes completed windows. After changing
inputs or configuration, use a new `--output-dir` and `--results-root`.

## Outputs and scoring

Main result files:

```text
outputs/whisper-large-v3/dev/
  predictions.jsonl
  scores_by_session_device.csv
  scores_by_device.csv
  scores_summary.json
outputs/whisper-large-v3/dev_diarized/
  predictions.jsonl
  run_metadata.json
  cpcer_summary.json
  reference_exclusions.json
results/whisper-large-v3/
  infer/dev/D1/result_cpcer.csv
  infer/dev/R1/result_cpcer.csv
  result_summary.md
```

Both metrics apply NFKC, lowercase, and removal of whitespace and Unicode
punctuation. Chinese transcription uses 16 kHz audio, one beam and maximum token
length 448. **OVERALL is micro-averaged per device:** total character errors
divided by total reference characters across sessions, multiplied by 100.

CER accumulates utterance-level edit errors; invalid reference intervals and
those longer than 30 seconds are excluded. cpCER concatenates text per speaker
and minimizes total character edit errors over speaker assignments. Predicted
intervals are restricted to the actual recording duration and split into windows
of at most 30 seconds; overlapping speakers retain the original mixture.

Reference end times allow 10 ms rounding tolerance. For cpCER, crossing or
out-of-range reference intervals are excluded in full and reported; longer-than-
30-second reference intervals remain. cpCER has no temporal collar. These text
and reference-filtering policies belong to this ASR baseline, not the DER
scorer. CER and cpCER use different segmentation and are not directly comparable;
neither replaces official ESR-ON cpWER scoring.

## Contributor-reported preliminary development results

CER values come from the contributor's existing `ASR_Baseline` output. cpCER
values come from the completed DiariZen + Whisper run (8,531 ASR windows).
These preliminary values have not been independently reproduced.

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
