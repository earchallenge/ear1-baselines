"""DiariZen RTTM -> base Whisper large-v3 -> development cpCER, D1/R1.

References are read only for scoring. Recognition boundaries and hypothesis
speaker labels come exclusively from predicted RTTM. No speech separation is
performed. Existing oracle-boundary CER outputs are not read or modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import soundfile as sf

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from utils.cpcer import cpcer  # noqa: E402
from utils.text_normalize import normalize  # noqa: E402
from utils.textgrid_utils import read_textgrid  # noqa: E402

NORMALIZATION = "NFKC; lowercase; remove whitespace and Unicode punctuation"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, allow_nan=False
    ).encode("utf-8")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(json_bytes(value) + b"\n")
    temporary.replace(path)


def one_audio(root, session, device):
    matches = [
        p
        for p in (root / session).iterdir()
        if p.is_file() and p.stem.upper() == device and p.suffix.lower() == ".wav"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {session}/{device} WAV; got {matches}")
    return matches[0].resolve()


def rttm_segments(path, session, duration, outside_segments=None):
    segments, seen = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if (
            len(fields) != 10
            or fields[0] != "SPEAKER"
            or fields[1] != session
            or fields[2] != "1"
        ):
            raise ValueError(f"Invalid RTTM/session/channel: {path}:{number}")
        start, length, speaker = float(fields[3]), float(fields[4]), fields[7]
        end = start + length
        if (
            not math.isfinite(start)
            or not math.isfinite(length)
            or not math.isfinite(end)
            or start < 0
            or length <= 0
            or speaker in {"<NA>", ""}
        ):
            raise ValueError(f"Invalid or out-of-range RTTM segment: {path}:{number}")
        key = (start, end, speaker)
        if key in seen:
            raise ValueError(f"Duplicate RTTM segment: {path}:{number}")
        seen.add(key)
        # Match the diarization recipe's full-recording UEM, without editing RTTM.
        # Wholly outside intervals have no audio to recognize; audit them explicitly.
        if start >= duration:
            if outside_segments is not None:
                outside_segments.append(dict(
                    start=start, end=end, speaker=speaker, rttm_line=number,
                    reason="predicted_rttm_outside_device_audio",
                ))
            continue
        segments.append(
            dict(
                start=start,
                end=min(end, duration),
                speaker=speaker,
                rttm_line=number,
                original_end=end,
            )
        )
    return sorted(
        segments, key=lambda r: (r["start"], r["end"], r["speaker"], r["rttm_line"])
    )


def read_reference(root, session):
    directory = root / session / "A"
    grids = sorted(
        p
        for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".textgrid"
        and re.fullmatch(r"P\d+", p.stem)
    )
    if not grids:
        raise ValueError(f"No personal reference TextGrids: {directory}")
    rows, fingerprints, speakers = [], {}, set()
    for path in grids:
        if path.stem in speakers:
            raise ValueError(f"Duplicate personal reference speaker: {path}")
        speakers.add(path.stem)
        intervals, metadata = read_textgrid(path)
        if [t["name"] for t in metadata["tiers"]] != [path.stem]:
            raise ValueError(f"Personal filename/tier mismatch: {path}")
        fingerprints[str(path.resolve())] = digest(path)
        for row in intervals:
            text = normalize(row["text"])
            if not text:
                continue
            if (
                not math.isfinite(row["start"])
                or not math.isfinite(row["end"])
                or row["start"] < 0
                or row["end"] <= row["start"]
            ):
                raise ValueError(f"Invalid nonempty reference interval: {path}")
            rows.append(
                dict(
                    start=row["start"],
                    end=row["end"],
                    speaker=path.stem,
                    text=text,
                    source=str(path.resolve()),
                    interval_index=row["interval_index"],
                )
            )
    return sorted(
        rows, key=lambda r: (r["start"], r["end"], r["speaker"], r["interval_index"])
    ), fingerprints


def read_dev_sessions(path):
    """Use the dataset's official split, including its corrected-session exclusion."""
    groups = {"train": [], "dev": []}
    current = None
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        heading = line.replace("：", ":").lower()
        if heading in {"train set:", "dev set:"}:
            current = heading.split()[0]
        elif current and re.fullmatch(r"\d{8}-[1-9]\d*", line):
            if any(line in values for values in groups.values()):
                raise ValueError(
                    f"Duplicate/cross-split session: {path}:{number}: {line}"
                )
            if line != "20260916-5":
                groups[current].append(line)
        else:
            raise ValueError(f"Invalid official split line: {path}:{number}: {line}")
    if not all(groups.values()):
        raise ValueError(f"Expected nonempty Train Set and Dev Set: {path}")
    return groups["dev"]


def build_plan(root, rttm_root, tracks=("D1", "R1"), sessions=None, split_path=None):
    root, rttm_root = root.resolve(strict=True), rttm_root.resolve(strict=True)
    pairs, segments, sources = [], [], {}
    if sessions is None:
        split_path = (
            split_path if split_path is not None else root / "split.lst"
        ).resolve(strict=True)
        sessions = read_dev_sessions(split_path)
        sources[str(split_path)] = digest(split_path)
    for session in sessions:
        reference, grids = read_reference(root, session)
        sources.update(grids)
        for device in tracks:
            audio = one_audio(root, session, device)
            info = sf.info(str(audio))
            if info.channels != 1 or not info.frames:
                raise ValueError(f"Expected nonempty mono WAV: {audio}")
            duration = info.frames / info.samplerate
            rttm = rttm_root / device / f"{session}.rttm"
            sources[str(rttm)] = digest(rttm)
            outside_predictions = []
            predicted = rttm_segments(rttm, session, duration, outside_predictions)
            kept = [
                r
                for r in reference
                if r["end"] <= duration + 0.01 and r["start"] < duration
            ]
            excluded = [
                dict(r, reason="reference_outside_device_audio")
                for r in reference
                if r["end"] > duration + 0.01 or r["start"] >= duration
            ]
            if not kept:
                raise ValueError(f"No scoreable reference: {session}/{device}")
            streams = defaultdict(str)
            for row in kept:
                streams[row["speaker"]] += row["text"]
            pair = dict(
                session=session,
                device=device,
                audio_path=str(audio),
                duration_sec=duration,
                audio_bytes=audio.stat().st_size,
                audio_mtime_ns=audio.stat().st_mtime_ns,
                sample_rate=info.samplerate,
                rttm_path=str(rttm),
                rttm_segments=len(predicted),
                rttm_clamped_segments=sum(
                    r["end"] != r["original_end"] for r in predicted
                ),
                rttm_outside_segments=outside_predictions,
                rttm_clipped_intervals=[
                    r for r in predicted if r["end"] != r["original_end"]
                ],
                reference_streams=dict(streams),
                reference_intervals=reference,
                reference_exclusions=excluded,
                reference_characters=sum(map(len, streams.values())),
            )
            pairs.append(pair)
            for row in predicted:
                # Equal non-overlapping subwindows, each <=30 seconds. All RTTM
                # speech is retained, including overlaps between speakers.
                pieces = max(1, math.ceil((row["end"] - row["start"]) / 30))
                for index in range(pieces):
                    start = row["start"] + (row["end"] - row["start"]) * index / pieces
                    end = (
                        row["start"]
                        + (row["end"] - row["start"]) * (index + 1) / pieces
                    )
                    segment = dict(
                        session=session,
                        device=device,
                        audio_path=str(audio),
                        speaker=row["speaker"],
                        start=start,
                        end=end,
                        rttm_line=row["rttm_line"],
                        chunk_index=index,
                    )
                    segment["id"] = hashlib.sha256(json_bytes(segment)).hexdigest()[:24]
                    segments.append(segment)
    sources.update(
        {
            str(PROJECT / "utils" / name): digest(PROJECT / "utils" / name)
            for name in (
                "audio.py",
                "text_normalize.py",
                "textgrid_utils.py",
                "textgrid_normalization.py",
            )
        }
    )
    return dict(
        schema="diarized_whisper_cpcer_v1",
        data_root=str(root),
        rttm_root=str(rttm_root),
        sessions=list(sessions),
        tracks=list(tracks),
        source_sha256=sources,
        normalization=NORMALIZATION,
        max_audio_segment_sec=30,
        rttm_time_policy="Intersect predicted intervals with [0, device audio duration]; audit clipped/wholly outside segments; original RTTM unchanged; no tail tolerance threshold",
        reference_policy="Personal A/P*.TextGrid, nonempty normalized intervals fully within device duration (+10ms tolerance); outside intervals explicitly reported; no 30s reference filter",
        recognition_policy="Predicted RTTM boundaries/speaker IDs only, intersected with available device audio; split >30s; retain speaker overlaps; no source separation",
        audio_identity_policy="Path, byte size, mtime, sample rate, duration; full WAV content not hashed",
        pairs=pairs,
        segments=segments,
    )


def prepare(plan, output):
    path = output / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != plan and any(
            (output / n).exists() for n in ("predictions.jsonl", "run_metadata.json")
        ):
            raise ValueError(
                "Sources/manifest changed. Preserve old results and select a new --output-dir."
            )
    write_json(path, plan)
    print(
        f"PREPARED: {len(plan['pairs'])} session-device pairs, {len(plan['segments'])} ASR windows",
        flush=True,
    )
    for pair in plan["pairs"]:
        print(
            f"  {pair['session']}/{pair['device']}: RTTM={pair['rttm_segments']}, clipped_RTTM={pair['rttm_clamped_segments']}, outside_RTTM={len(pair['rttm_outside_segments'])}, ref_chars={pair['reference_characters']}, excluded_ref_intervals={len(pair['reference_exclusions'])}",
            flush=True,
        )
    return digest(path)


def model_identity(path):
    path = path.resolve(strict=True)
    required = ("config.json", "preprocessor_config.json", "tokenizer_config.json")
    for name in required:
        if not (path / name).is_file():
            raise ValueError(f"Incomplete local Whisper model: {path / name}")
    files = sorted(p for p in path.iterdir() if p.is_file())
    weights = [p for p in files if p.suffix in {".bin", ".safetensors"}]
    if not weights:
        raise ValueError(f"No local Whisper weights: {path}")
    return dict(
        path=str(path),
        config_sha256={
            p.name: digest(p)
            for p in files
            if p.suffix in {".json", ".txt", ".model", ".tiktoken"}
        },
        weight_files={
            p.name: dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
            for p in weights
        },
    )


def prediction_rows(path, segments, repair_tail=False):
    if not path.exists():
        return {}
    expected = {r["id"]: r for r in segments}
    rows, successes, offset = {}, set(), 0
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            if repair_tail and index == len(lines) - 1 and not line.endswith(b"\n"):
                with path.open("r+b") as stream:
                    stream.truncate(offset)
                print("Recovered incomplete final prediction line", flush=True)
                break
            raise ValueError(f"Invalid prediction JSON: {path}:{index + 1}") from None
        uid = row.get("id")
        if uid not in expected or any(
            row.get(k) != v for k, v in expected[uid].items()
        ):
            raise ValueError(f"Prediction does not match manifest: {uid}")
        if row.get("status") not in {"SUCCESS", "FAILED"} or not isinstance(
            row.get("prediction"), str
        ):
            raise ValueError(f"Invalid prediction status/text: {uid}")
        if uid in successes:
            raise ValueError(f"Duplicate or superseded successful prediction: {uid}")
        if row["status"] == "SUCCESS":
            successes.add(uid)
        rows[uid] = row
        offset += len(line)
    if repair_tail and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with path.open("ab") as stream:
            stream.write(b"\n")
    return rows


def run_recognition(plan, output, identity, recognizer_factory, resume=False):
    metadata_path, predictions = (
        output / "run_metadata.json",
        output / "predictions.jsonl",
    )
    if predictions.exists() or metadata_path.exists():
        if not resume:
            raise ValueError(
                "Existing inference output; use --resume or a new --output-dir"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["inference_identity"] != identity:
            raise ValueError(
                "Model/manifest/decoding settings changed; refusing to reuse predictions"
            )
    rows = prediction_rows(predictions, plan["segments"], repair_tail=resume)
    pending = [
        r for r in plan["segments"] if rows.get(r["id"], {}).get("status") != "SUCCESS"
    ]
    metadata = dict(
        inference_identity=identity, status="running", total=len(plan["segments"])
    )
    write_json(metadata_path, metadata)
    started = time.perf_counter()
    try:
        recognizer = recognizer_factory() if pending else None
        from tqdm import tqdm

        with predictions.open("a", encoding="utf-8") as stream:
            for item in tqdm(
                pending,
                total=len(plan["segments"]),
                initial=len(plan["segments"]) - len(pending),
                desc="diarized large-v3",
                unit="segment",
            ):
                try:
                    text, details = recognizer(item)
                    if not isinstance(text, str):
                        raise TypeError("Recognition output must be text")
                    row = dict(
                        item, prediction=text, status="SUCCESS", error="", **details
                    )
                except Exception as exc:
                    row = dict(
                        item,
                        prediction="",
                        status="FAILED",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    stream.write(
                        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                    )
                    stream.flush()
                    rows[item["id"]] = row
                    raise
                stream.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
                stream.flush()
                rows[item["id"]] = row
    finally:
        metadata.update(
            success=sum(r["status"] == "SUCCESS" for r in rows.values()),
            failed=sum(r["status"] == "FAILED" for r in rows.values()),
            last_invocation_sec=time.perf_counter() - started,
        )
        metadata["status"] = (
            "complete" if metadata["success"] == metadata["total"] else "incomplete"
        )
        write_json(metadata_path, metadata)
    print(f"ASR COMPLETE: {metadata['success']}/{metadata['total']}", flush=True)


def whisper_factory(model_path, device):
    import numpy as np
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from utils.audio import read_segment

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable; use the era/asr environment with working PyTorch CUDA"
        )
    dtype = torch.float16
    processor = WhisperProcessor.from_pretrained(str(model_path), local_files_only=True)
    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            str(model_path), dtype=dtype, local_files_only=True
        )
    except TypeError:
        model = WhisperForConditionalGeneration.from_pretrained(
            str(model_path), torch_dtype=dtype, local_files_only=True
        )
    if (
        model.config.num_mel_bins,
        model.config.encoder_layers,
        model.config.decoder_layers,
    ) != (128, 32, 32):
        raise ValueError(
            "Expected full Whisper large-v3 architecture (128 mels, 32 encoder/decoder layers)"
        )
    model.to(device).eval()
    model.generation_config.forced_decoder_ids = None

    def recognize(item):
        audio = read_segment(item["audio_path"], item["start"], item["end"], 16000)
        if not audio.size or not np.isfinite(audio).all():
            raise ValueError("Empty or nonfinite audio")
        inputs = processor(
            audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
        )
        kwargs = dict(
            input_features=inputs.input_features.to(device=device, dtype=dtype),
            language="zh",
            task="transcribe",
            num_beams=1,
            max_length=448,
        )
        if inputs.get("attention_mask") is not None:
            kwargs["attention_mask"] = inputs["attention_mask"].to(device)
        with torch.inference_mode():
            ids = model.generate(**kwargs)
        text = processor.batch_decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return text, dict(
            generated_tokens=len(ids[0]),
            hit_token_limit=len(ids[0]) >= 448
            and int(ids[0, -1]) != model.config.eos_token_id,
        )

    return recognize


def score_predictions(plan, output, identity, results_root=None):
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata["inference_identity"] != identity:
        raise ValueError("Scoring metadata does not match current model/manifest")
    rows = prediction_rows(output / "predictions.jsonl", plan["segments"])
    if len(rows) != len(plan["segments"]) or any(
        r["status"] != "SUCCESS" for r in rows.values()
    ):
        raise ValueError(
            "Incomplete/failed ASR. Resume inference before scoring; no failed rows are dropped."
        )
    streams = defaultdict(lambda: defaultdict(str))
    for item in sorted(
        plan["segments"],
        key=lambda r: (
            r["session"],
            r["device"],
            r["speaker"],
            r["start"],
            r["end"],
            r["id"],
        ),
    ):
        streams[item["session"], item["device"]][item["speaker"]] += normalize(
            rows[item["id"]]["prediction"]
        )
    results = []
    for pair in plan["pairs"]:
        hypothesis = dict(streams[pair["session"], pair["device"]])
        result = cpcer(pair["reference_streams"], hypothesis)
        results.append(
            dict(
                session=pair["session"],
                device=pair["device"],
                **result,
                excluded_reference_intervals=len(pair["reference_exclusions"]),
                excluded_reference_characters=sum(
                    len(r["text"]) for r in pair["reference_exclusions"]
                ),
                reference_streams=pair["reference_streams"],
                hypothesis_streams=hypothesis,
            )
        )
        print(
            f"{pair['session']}/{pair['device']} cpCER={result['cpcer']:.2%}",
            flush=True,
        )

    def aggregate(items):
        errors = sum(r["errors"] for r in items)
        characters = sum(r["reference_characters"] for r in items)
        return dict(
            errors=errors,
            reference_characters=characters,
            cpcer=errors / characters,
            session_device_pairs=len(items),
            session_device_average_cpcer=sum(r["cpcer"] for r in items) / len(items),
        )

    summary = dict(
        metric="cpCER",
        model="Whisper large-v3",
        diarization="DiariZen predicted RTTM",
        overall=aggregate(results),
        by_device={
            d: aggregate([r for r in results if r["device"] == d])
            for d in plan["tracks"]
        },
        normalization=NORMALIZATION,
        reference_policy=plan["reference_policy"],
        recognition_policy=plan["recognition_policy"],
        rttm_time_policy=plan["rttm_time_policy"],
        rttm_clamped_segments=sum(p["rttm_clamped_segments"] for p in plan["pairs"]),
        rttm_outside_segments=sum(len(p["rttm_outside_segments"]) for p in plan["pairs"]),
        inference_identity=identity,
        reference_excluded_intervals=sum(
            r["excluded_reference_intervals"] for r in results
        ),
        hit_token_limit_segments=sum(
            bool(r.get("hit_token_limit")) for r in rows.values()
        ),
        note="Preliminary cpCER, not official challenge scoring; per-session/device optimal speaker assignment. Overall sums errors/reference characters; no time collar. Error-operation breakdown not computed.",
    )
    write_json(output / "cpcer_details.json", results)
    write_json(output / "cpcer_summary.json", summary)
    write_json(
        output / "reference_exclusions.json",
        [
            dict(
                session=p["session"],
                device=p["device"],
                intervals=p["reference_exclusions"],
            )
            for p in plan["pairs"]
        ],
    )
    with (output / "cpcer_by_session_device.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        keys = [
            "session",
            "device",
            "cpcer_pct",
            "errors",
            "reference_characters",
            "reference_speakers",
            "hypothesis_speakers",
            "excluded_reference_intervals",
            "excluded_reference_characters",
        ]
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    k: result["cpcer"] * 100 if k == "cpcer_pct" else result[k]
                    for k in keys
                }
            )
    lines = [
        "# DiariZen + Whisper large-v3 Dev cpCER",
        "",
        "| Session | " + " | ".join(f"{d} cpCER (%)" for d in plan["tracks"]) + " |",
        "|---|" + "---:|" * len(plan["tracks"]),
    ]
    lookup = {(r["session"], r["device"]): r["cpcer"] for r in results}
    for session in plan["sessions"]:
        lines.append(
            "| "
            + session
            + " | "
            + " | ".join(f"{lookup[session, d] * 100:.2f}" for d in plan["tracks"])
            + " |"
        )
    lines.append(
        "| OVERALL (micro) | "
        + " | ".join(
            f"{summary['by_device'][d]['cpcer'] * 100:.2f}" for d in plan["tracks"]
        )
        + " |"
    )
    lines += [
        "",
        f"Normalization: {NORMALIZATION}.",
        "",
        plan["reference_policy"],
        "",
        plan["recognition_policy"],
        "",
        summary["note"],
        "",
        f"Excluded reference intervals: {summary['reference_excluded_intervals']}; token-limit segments: {summary['hit_token_limit_segments']}.",
    ]
    (output / "cpcer_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if results_root is not None:
        from utils.cpcer_reports import export_reports

        export_reports(plan, results, rows, summary, results_root)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--split",
        type=Path,
        help="Official dataset split; default: DATA_ROOT/split.lst",
    )
    parser.add_argument(
        "--rttm-root",
        type=Path,
        default=PROJECT.parent / "diarization/results/infer/dev",
        help="Directory containing D1/*.rttm and R1/*.rttm",
    )
    parser.add_argument(
        "--tracks", nargs="+", choices=["D1", "R1"], default=["D1", "R1"]
    )
    parser.add_argument(
        "--model-path", type=Path, default=PROJECT / "models/whisper-large-v3"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "outputs/whisper-large-v3/dev_diarized",
    )
    parser.add_argument(
        "--stage", choices=["prepare", "infer", "score", "all"], default="all"
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT / "results",
        help="Track reports are written under ROOT/whisper-large-v3/",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare and validate inputs; no model loaded",
    )
    args = parser.parse_args()
    if len(args.tracks) != len(set(args.tracks)):
        parser.error("Duplicate --tracks")
    output = args.output_dir.resolve()
    results_root = args.results_root.resolve()
    protected = (
        args.data_root.resolve(),
        args.rttm_root.resolve(),
        args.model_path.resolve(),
    )
    if any(
        output.is_relative_to(p) or results_root.is_relative_to(p) for p in protected
    ):
        parser.error("Output must be outside data, RTTM and model directories")
    split_path = (
        args.split if args.split is not None else args.data_root / "split.lst"
    ).resolve()
    if not split_path.is_file():
        parser.error(
            f"Official split not found: {split_path}. Pass --split /path/to/split.lst."
        )
    if not args.rttm_root.is_dir():
        parser.error(
            f"Predicted RTTM directory not found: {args.rttm_root}. Run ../diarization first or pass --rttm-root."
        )
    plan = build_plan(
        args.data_root, args.rttm_root, args.tracks, split_path=split_path
    )
    plan_hash = prepare(plan, output)
    if args.dry_run or args.stage == "prepare":
        print("INPUT CHECK COMPLETE; no ASR or scoring performed", flush=True)
        return
    identity = dict(
        manifest_sha256=plan_hash,
        model=model_identity(args.model_path),
        asr_policy="whisper_large_v3_rttm_windows_v1",
        normalization=NORMALIZATION,
        inference_script_sha256=digest(Path(__file__)),
        environment={
            name: version(name)
            for name in ("torch", "transformers", "numpy", "soundfile", "scipy")
        },
        decoding=dict(
            language="zh",
            task="transcribe",
            sample_rate=16000,
            num_beams=1,
            max_length=448,
            dtype="float16",
            device=args.device,
        ),
    )
    if args.stage in {"infer", "all"}:
        run_recognition(
            plan,
            output,
            identity,
            lambda: whisper_factory(args.model_path.resolve(), args.device),
            args.resume,
        )
    if args.stage in {"score", "all"}:
        score_predictions(plan, output, identity, results_root)
        print("CPCER COMPLETE:", output / "cpcer_summary.json", flush=True)


if __name__ == "__main__":
    main()
