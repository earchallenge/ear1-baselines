from __future__ import annotations

import argparse
import json
import sys
import time
import hashlib
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.audio import read_segment
from utils.text_normalize import normalize


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/dev.jsonl"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=448)
    parser.add_argument("--model", required=True, choices=["whisper-large-v2", "whisper-large-v3"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    output = args.output_root.resolve() / args.model / "dev"
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "predictions.jsonl"
    done = set()
    if args.resume and prediction_path.is_file():
        for line in prediction_path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                if item.get("status") == "SUCCESS":
                    done.add(item.get("utterance_id", item.get("id")))
            except json.JSONDecodeError:
                continue
    model_path = args.model_root.resolve() / args.model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(str(model_path))
    try:
        model = WhisperForConditionalGeneration.from_pretrained(str(model_path), dtype=dtype)
    except TypeError:
        model = WhisperForConditionalGeneration.from_pretrained(str(model_path), torch_dtype=dtype)
    model.to(device).eval()
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    pending = [item for item in records if item["id"] not in done]
    mode = "a" if args.resume and prediction_path.is_file() else "w"
    with prediction_path.open(mode, encoding="utf-8") as handle:
        progress = tqdm(pending, total=len(records), initial=len(records) - len(pending),
                        desc=args.model, unit="utt")
        for item in progress:
            result = {
                "utterance_id": item.get("id"),
                "session_id": item.get("session"),
                "device_id": item.get("device"),
                "audio_path": item.get("audio_path"),
                "reference": item.get("reference_raw", item.get("reference_text", "")),
                "reference_normalized": item.get("reference_norm", ""),
                "start": item.get("start"),
                "end": item.get("end"),
                "overlap": item.get("overlap", False),
            }
            try:
                audio = read_segment(item["audio_path"], item["start"], item["end"], args.sample_rate)
                if not audio.size or not np.isfinite(audio).all():
                    raise ValueError("empty or invalid audio")
                inputs = processor(audio, sampling_rate=args.sample_rate,
                                   return_tensors="pt", return_attention_mask=True)
                features = inputs.input_features.to(device=device, dtype=dtype)
                attention_mask = inputs.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                with torch.inference_mode():
                    generate_args = dict(input_features=features,
                                         language=args.language,
                                         task=args.task,
                                         num_beams=args.num_beams,
                                         max_length=args.max_length)
                    if attention_mask is not None:
                        generate_args["attention_mask"] = attention_mask
                    ids = model.generate(**generate_args)
                prediction = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
                result.update(prediction=prediction, prediction_normalized=normalize(prediction), status="SUCCESS", error="")
            except Exception as exc:
                result.update(prediction="", prediction_normalized="", status="FAILED", error=f"{type(exc).__name__}: {exc}")
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            progress.set_postfix(status=result["status"])
    elapsed = time.perf_counter() - started
    reported_success = 0
    reported_failed = 0
    for line in prediction_path.read_text(encoding="utf-8").splitlines():
        try:
            status = json.loads(line).get("status")
        except json.JSONDecodeError:
            continue
        if status == "SUCCESS":
            reported_success += 1
        elif status == "FAILED":
            reported_failed += 1
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    metadata = {"model": args.model, "model_path": str(model_path), "data_root": str(args.data_root.resolve()) if args.data_root else None,
                "manifest_sha256": manifest_hash, "device": device, "dtype": str(dtype),
                "sample_rate": args.sample_rate, "language": args.language, "task": args.task,
                "num_beams": args.num_beams, "max_length": args.max_length,
                "records": len(records), "manifest": str(manifest), "success": reported_success,
                "failed": reported_failed, "elapsed_sec": elapsed}
    (output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ASR complete: success={reported_success}, failed={reported_failed}, elapsed={elapsed:.1f}s")
    if reported_failed:
        print(f"Warning: {reported_failed} utterance(s) failed; see {prediction_path}", file=sys.stderr)
    return 1 if reported_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


