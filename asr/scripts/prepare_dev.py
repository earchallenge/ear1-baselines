from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.dev_reader import build_inventory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "dev.jsonl"
DATA_CONFIG = {
    "train_devices": "auto",
    "eval_devices": ["D1", "R1"],
    "train_overlap_policy": "exclude_mixed",
    "eval_overlap_policy": "retain",
    "audio_extensions": [".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".m4a", ".aac", ".mp4", ".mov"],
    "reviews": {},
}


def stable_id(session: str, device: str, row: dict) -> str:
    value = f"{session}\0{device}\0{row.get('tier_index')}\0{row.get('interval_index')}\0{row.get('start')}\0{row.get('end')}"
    return hashlib.sha1(value.encode()).hexdigest()[:20]


def read_split(path: Path) -> dict:
    groups = {"Train Set：": "train_sessions", "Train Set:": "train_sessions",
              "Dev Set：": "eval_sessions", "Dev Set:": "eval_sessions"}
    split = {"train_sessions": [], "eval_sessions": [], "eval_role": "development_tuning"}
    current = None
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line in groups:
            current = groups[line]
        elif current and line != "20260916-5":
            split[current].append(line)
    if not split["train_sessions"] or not split["eval_sessions"]:
        raise ValueError(f"Invalid split file: {path}")
    return split


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, help="Official split file; default: DATA_ROOT/split.lst")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    split_path = (args.split if args.split else data_root / "split.lst").resolve()
    if not split_path.is_file():
        parser.error(f"Split file not found: {split_path}. Use --split /path/to/split.lst.")
    split = read_split(split_path)
    inventory, inspections = build_inventory(data_root, DATA_CONFIG, split)
    val_inventory = [r for r in inventory if r["session_role"] == "val"]
    records = []
    for device_row in val_inventory:
        key = (device_row["session"], device_row["device"])
        inspection = inspections.get(key)
        if not inspection or device_row["status"] != "READY":
            continue
        for row in inspection["rows"]:
            if row.get("excluded"):
                continue
            records.append({
                "id": stable_id(*key, row),
                "session": key[0], "device": key[1],
                "audio_path": inspection["audio_path"],
                "textgrid_path": inspection["textgrid_path"],
                "start": row["start"], "end": row["end"],
                "speaker": row.get("speaker"),
                "tier_index": row.get("tier_index"),
                "interval_index": row.get("interval_index"),
                "overlap": bool(row.get("overlap_flag", False)),
                "reference_text": row.get("reference_raw", ""),
                "reference_norm": row.get("reference_norm", ""),
            })
    records.sort(key=lambda r: (r["session"], r["device"], r["start"], r["id"]))
    if args.limit is not None:
        records = records[:args.limit]
    out = args.manifest.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    summary = {
        "data_root": str(data_root), "split_path": str(split_path), "split": split,
        "eval_devices": DATA_CONFIG["eval_devices"],
        "discovered_val_devices": len(val_inventory),
        "ready_val_devices": sum(r["status"] == "READY" for r in val_inventory),
        "records": len(records),
        "sessions": dict(Counter(r["session"] for r in records)),
        "devices": dict(Counter(r["device"] for r in records)),
        "excluded_or_unreadable_devices": sum(r["status"] != "READY" for r in val_inventory),
        "elapsed_sec": time.perf_counter() - started,
    }
    (out.parent / "dev_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out.parent / "dev_devices.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in val_inventory) + "\n", encoding="utf-8")
    print("Development set prepared")
    print(f"  Sessions: {len(split.get('eval_sessions', []))}")
    print(f"  Scorable devices: {summary['ready_val_devices']}")
    print(f"  Total segments: {summary['records']}")
    print(f"  Excluded devices: {summary['excluded_or_unreadable_devices']}")
    print(f"  Elapsed: {summary['elapsed_sec']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
