# EAR-1 ASR Baseline

Base Whisper large-v3 development baseline. D1 and R1 are evaluated separately:
`run.sh` reports CER using reference TextGrid segments; `run_cpcer.sh` reports
cpCER using DiariZen-predicted RTTM segments and speaker IDs.

## Installation

Use Linux, Bash and Python 3.10 with CUDA PyTorch. From `asr/`:

```bash
bash setup_env.sh
conda activate asr
```

Alternatively, create and activate the environment manually:

```bash
conda create -n asr python=3.10 -y
conda activate asr
pip install -r requirements.txt
```

Then download Whisper large-v3:

```bash
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

Run the following commands from `asr/` after activating the ASR environment.
Replace `/path/to/...` with your own paths and `CUDA_VISIBLE_DEVICES=0` with the
GPU to use. Quote paths containing spaces or `&`, for example
`"/data/Train&Dev"`. If the dataset contains `split.lst`, omit `--split`.

```bash
# Reference-segmented CER
CUDA_VISIBLE_DEVICES=0 bash run.sh /path/to/dataset \
  --split /path/to/official/split.lst

# Predicted-RTTM cpCER
CUDA_VISIBLE_DEVICES=0 bash run_cpcer.sh /path/to/dataset \
  --split /path/to/official/split.lst \
  --rttm-root /path/to/predicted/infer/dev \
  --model-path /path/to/whisper-large-v3 \
  --output-dir outputs/whisper-large-v3/dev_diarized \
  --results-root results
```

`--rttm-root` must contain the predicted `D1/` and `R1/` subdirectories.
`--model-path` points to a complete local Whisper large-v3 model;
`--output-dir` stores recognition progress, and reports go under
`--results-root/whisper-large-v3/`. The model/output arguments above are optional
when using the default locations. The default RTTM root is
`../diarization/results/infer/dev/`.

Before a first cpCER run, append `--dry-run` to the command above. After
`INPUT CHECK COMPLETE`, remove that flag to recognize and score. Both commands
evaluate D1 and R1; cpCER also accepts `--tracks D1` or `--tracks R1`.

To recalculate CER from completed predictions without running Whisper:

```bash
bash run.sh /path/to/dataset --eval-only
```

For cpCER, repeat your original command with `--stage score` appended to rescore,
or repeat it unchanged to resume successful windows and retry pending ones.
Keep the same paths and configuration. Changes require a new cpCER
`--output-dir` and `--results-root`. Use `bash run_cpcer.sh --help` for options.

<details>
<summary>Optional: run CER stages individually</summary>

```bash
python scripts/prepare_dev.py /path/to/dataset \
  --split /path/to/official/split.lst --manifest data/dev.jsonl

CUDA_VISIBLE_DEVICES=0 python scripts/infer.py --model whisper-large-v3 \
  --manifest data/dev.jsonl --data-root /path/to/dataset \
  --model-root models --output-root outputs

python scripts/evaluate.py --model whisper-large-v3 \
  --manifest data/dev.jsonl --output-root outputs
```

`--model-root` is the parent directory containing `whisper-large-v3/`.
For cpCER stages, append `--stage prepare`, `--stage infer`, or `--stage score`
to the same `run_cpcer.sh` command.

</details>

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

Text normalization applies NFKC, lowercase, and removal of whitespace and Unicode
punctuation. Whisper uses Chinese transcription, 16 kHz audio, one beam and
maximum token length 448. **OVERALL is micro-averaged per device:** total character
edit errors divided by total reference characters across sessions, multiplied by 100.

CER accumulates edit errors over reference utterances. cpCER concatenates text
chronologically per speaker and finds the speaker assignment with the fewest
total character edit errors. Predicted RTTM segments are clipped to the recording
duration and split into windows of at most 30 seconds; speaker overlap is retained.

Reference end times allow 10 ms rounding tolerance. Invalid or out-of-range
reference intervals are excluded. CER also excludes intervals longer than
30 seconds. cpCER retains long reference intervals and reports any whole
reference intervals excluded for crossing the recording boundary.

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
