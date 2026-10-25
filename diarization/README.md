# EAR-1 Speaker Diarization Baseline

Single-channel development recipe using `BUT-FIT/diarizen-wavlm-large-s80-md-v2`.
D1 and R1 are processed independently.

## Installation

Follow the installation instructions in the upstream
[DiariZen repository](https://github.com/BUTSpeechFIT/DiariZen), including its
bundled pyannote-audio, in a separate Python 3.10 environment named `diarizen`.
Do not combine it with the ASR environment; the Torch requirements differ.

The recipe requires `torch`, `diarizen`, `soundfile`, `pyannote.core`, and
`pyannote.metrics`. Check your installation:

```bash
conda activate diarizen
python -c "import torch, soundfile, pyannote.core, pyannote.metrics; from diarizen.pipelines.inference import DiariZenPipeline"
```

Inference requires a CUDA GPU. Model files are downloaded from Hugging Face on
first use; upstream model terms apply. `HF_HUB_OFFLINE=1` uses cached files.
Set `DIAR_ENV_NAME` if your Conda environment has another name. Without Conda,
the shell script uses the current environment or the `--python` interpreter.

## Data and commands

Apply all [organizer corrections](../README.md#data-corrections), including speaker
ID updates and removal of 20260916-5. The data root directly contains session
folders. Use the official split from the data package; no split copy is bundled.

```bash
bash run_baseline.sh --help
bash run_baseline.sh --data-root /path/to/TrainAndDev \
  --split /path/to/official/split.lst

# Run one stage / track, rebuilding existing hypotheses
bash run_baseline.sh --data-root /path/to/TrainAndDev \
  --split /path/to/official/split.lst --tracks R1 --stage infer --overwrite
```

Stages are `prepare`, `infer`, `score`, or `all` (default). Preparation converts
`A/P*.TextGrid` to reference RTTMs for official train/dev sessions. Inference and
scoring process development sessions only. The default split path is
`DATA_ROOT/split.lst`; 20260916-5 is excluded even if listed in an old split.

Direct Python commands (in the installed environment):

```bash
python ear1_baseline.py prepare --data /path/to/TrainAndDev \
  --split /path/to/official/split.lst --out results
python ear1_baseline.py infer --data /path/to/TrainAndDev \
  --split /path/to/official/split.lst --out results --tracks D1 R1
python ear1_baseline.py score --data /path/to/TrainAndDev \
  --split /path/to/official/split.lst --out results --tracks D1 R1
```

## Outputs and scoring

```text
results/
  prepare/train/<session>.rttm
  prepare/dev/<session>.rttm
  infer/dev/D1/<session>.rttm
  infer/dev/D1/result_collar0.csv
  infer/dev/R1/<session>.rttm
  infer/dev/R1/result_collar0.csv
```

DER uses `pyannote.metrics.DiarizationErrorRate`, zero collar, overlaps included,
and a track-specific **full-recording UEM** including silence. Reference timestamps
are preserved; scoring is restricted to available audio duration. Overall DER is
micro-averaged across sessions.

**Oracle VAD is not used during inference.** These results are not an
oracle-VAD-constrained ESD-ON result or finalized official challenge scores.
After corrections, regenerate preparation files and use `--overwrite` for inference.

## Contributor-reported preliminary development results

These values come from the supplied run script. The contributor reports using
corrected annotations; they have not been independently reproduced here.

| Session | D1 DER (%) | R1 DER (%) |
|---|---:|---:|
| 20260908-1 | 25.74 | 28.60 |
| 20260908-2 | 29.17 | 29.77 |
| 20260923-1 | 22.67 | 46.93 |
| 20260923-2 | 42.53 | 40.58 |
| 20260923-5 | 14.89 | 15.43 |
| 20260924-2 | 16.79 | 14.84 |
| 20260924-5 | 45.89 | 48.48 |
| OVERALL | 29.34 | 32.61 |

Versions and scoring settings can affect reproduction. Record the DiariZen commit
and environment with your results. Follow the upstream licenses and cite DiariZen.
