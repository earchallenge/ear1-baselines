# EAR-1 Baselines

Baseline components for the EAR-1 Challenge.

| Component | Contents | Status |
|---|---|---|
| [ASR](asr/README.md) | Whisper large-v3/large-v2 with oracle utterance segmentation and CER evaluation | Initial source package; full data/model run pending |
| [Speaker diarization](diarization/README.md) | DiariZen WavLM Large v2 and full-recording DER scoring | Preliminary development recipe |

## Getting started

Obtain the dataset through the challenge registration process and apply the official correction archive before running experiments. Dataset files and model weights are stored outside version control.

Use the official `split.lst` supplied with the dataset. This repository does not maintain a second split copy.

Use the updated **55 training / 7 development session** split and exclude
**20260916-5**. Both recipes accept an explicit `--split` path; their default is
`DATA_ROOT/split.lst`. The data root directly contains session folders.

```bash
cd asr
bash setup_env.sh
bash download_models.sh
bash run.sh /path/to/dataset
# If the split file is elsewhere:
bash run.sh /path/to/dataset --split /path/to/split.lst
```

See the [ASR documentation](asr/README.md) for environment activation, model selection, outputs, and CER policy.

For diarization, first install DiariZen in a **separate environment** as described
in the [diarization README](diarization/README.md), then run:

```bash
cd diarization  # from the repository root
bash run_baseline.sh --data-root /path/to/dataset --split /path/to/split.lst
```

## Challenge inputs

- Fixed-Array evaluation: D1 only.
- Moving-Array evaluation: R1 only.
- Tracks are evaluated separately. Evaluation speakers and rooms do not overlap with training or development.
- All released training/development audio may be used as auxiliary training material, including H1/H2/H3 and headset audio. A shared model may be jointly trained on D1 and R1 and submitted to both tracks.

The ASR component is an oracle-segmented development-set comparison, not a complete diarization or end-to-end challenge system.

ASR reports **CER**, not the ESR-ON ranking metric **cpWER**. The diarization
recipe does not use oracle VAD during inference and scores over the full audio
duration. It is not an oracle-VAD-constrained ESD-ON pipeline or a finalized
official scoring package. These limitations should be resolved before presenting
the recipes as complete challenge submission systems.

## Data corrections

Apply the correction archive distributed by the organizers for sessions 20260909-6, 20260911-1, 20260911-5, and 20260911-6. Access details are distributed to registered teams; they are not embedded in this code repository.

Also apply the subsequent speaker-ID/TextGrid corrections for **20260910-4,
20260924-2, and 20260924-8**, remove the original P040 entry in 20260924-8,
update speaker metadata, and exclude **20260916-5** entirely.
Regenerate intermediate files and predictions after corrections. Diarization
reuses existing hypotheses unless `--overwrite` is supplied; ASR caches do not
fingerprint audio bytes.

## Validation and third-party software

Packaging checks cover Python/Bash syntax and lightweight functional behavior;
full GPU/data reproduction has not been run here. Diarization results are
contributor-reported preliminary values.

Follow the licenses and citation requirements of
[Whisper](https://github.com/openai/whisper) and
[DiariZen](https://github.com/BUTSpeechFIT/DiariZen), including model terms.
A license for this repository's integration code has not yet been selected.

## Contact

EAR-1 Organising Team: ear.challenge.office@gmail.com
