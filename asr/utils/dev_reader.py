from __future__ import annotations

from bisect import bisect_left
from collections import Counter
import math
from pathlib import Path

import soundfile as sf

from .textgrid_utils import prepare_records, read_textgrid


def discover_devices(data_root, session, cfg):
    root = Path(data_root).resolve()
    directory = root / session
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing session: {directory}")
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    extensions = {str(x).casefold() for x in cfg.get("audio_extensions", [])}
    grids = {}
    for path in files:
        if path.suffix.casefold() == ".textgrid":
            grids.setdefault(path.with_suffix("").as_posix().casefold(), []).append(path)
    reviews = {str(k).casefold(): v for k, v in cfg.get("reviews", {}).items()}
    result, used = [], set()
    for audio in files:
        if audio.suffix.casefold() not in extensions:
            continue
        relative = audio.relative_to(root).as_posix()
        decision = reviews.get(relative.casefold(), {}) or {}
        matches = grids.get(audio.with_suffix("").as_posix().casefold(), [])
        grid = matches[0] if len(matches) == 1 else None
        if decision.get("textgrid"):
            candidate = (root / decision["textgrid"]).resolve()
            grid = candidate if candidate.is_file() else None
        if grid:
            used.add(grid.resolve())
        status = decision.get("status", "include")
        result.append({"session": session, "device": audio.relative_to(directory).with_suffix("").as_posix(),
                       "audio_path": str(audio.resolve()), "textgrid_path": str(grid.resolve()) if grid else None,
                       "status": status, "reason": decision.get("reason", "") or ("missing_textgrid" if grid is None else ""),
                       "review": decision})
    for grid in files:
        if grid.suffix.casefold() == ".textgrid" and grid.resolve() not in used:
            result.append({"session": session, "device": grid.stem, "audio_path": None,
                           "textgrid_path": str(grid.resolve()), "status": "exclude",
                           "reason": "missing_audio", "review": {}})
    return result


def session_activity(devices):
    activity = {}
    for base in devices:
        if base["reason"] or not base["textgrid_path"]:
            continue
        try:
            intervals, _ = read_textgrid(base["textgrid_path"])
        except (OSError, ValueError):
            continue
        for row in intervals:
            if (math.isfinite(row["start"]) and math.isfinite(row["end"])
                    and row["start"] >= 0 and row["end"] > row["start"] and row["text"].strip()):
                activity[(row["speaker"], row["start"], row["end"])] = row
    ordered = sorted(activity.values(), key=lambda r: r["start"])
    return ordered, [r["start"] for r in ordered]


def build_inventory(data_root, cfg, split):
    inventory, inspections = [], {}
    for session in split["eval_sessions"]:
        role = "val"
        devices = discover_devices(data_root, session, cfg)
        activity, starts = session_activity(devices)
        for base in devices:
            row = {"session": session, "session_role": role, "device": base["device"],
                   "entry_type": "audio_device" if base["audio_path"] else "annotation_only",
                   "audio_path": base["audio_path"], "textgrid_path": base["textgrid_path"],
                   "review_status": base["status"], "review_reason": base["reason"],
                   "status": "EXCLUDED", "reason": base["reason"], "error": "",
                   "valid_segments": 0, "total_ref_chars": 0}
            if base["status"] == "exclude" or not base["audio_path"] or not base["textgrid_path"]:
                row["reason"] = row["reason"] or "missing_audio_or_textgrid"
                inventory.append(row)
                continue
            try:
                info = sf.info(base["audio_path"])
                intervals, _ = read_textgrid(base["textgrid_path"])
                records = prepare_records(intervals, info.frames / info.samplerate)
                stem = Path(base["audio_path"]).stem
                speakers = {r["speaker"] for r in intervals if r["text"].strip()}
                personal = base["review"].get("speaker") or next((s for s in speakers if s.casefold() == stem.casefold()), None)
                for record in records:
                    if personal and record["speaker"].casefold() != personal.casefold():
                        record.update(excluded=True, exclude_reason="non_target_speaker")
                    if personal and not record["excluded"]:
                        record["overlap_flag"] = record["overlap_flag"] or any(
                            other["speaker"].casefold() != personal.casefold() and other["end"] > record["start"]
                            for other in activity[:bisect_left(starts, record["end"])]
                        )
                valid = [r for r in records if not r["excluded"]]
                row.update(status="READY" if valid else "NO_VALID_SEGMENTS", reason="" if valid else "no_valid_segments",
                           valid_segments=len(valid), total_ref_chars=sum(len(r["reference_norm"]) for r in valid),
                           recording_type="personal" if personal else "mixed_or_unknown", personal_speaker=personal,
                           excluded_segments=len(records) - len(valid),
                           exclude_reason_counts=dict(Counter(r["exclude_reason"] for r in records if r["excluded"])))
                if valid:
                    inspections[session, base["device"]] = {"audio_path": base["audio_path"],
                        "textgrid_path": base["textgrid_path"], "rows": records}
            except (OSError, ValueError, RuntimeError) as exc:
                row.update(status="UNREADABLE", reason="unreadable_audio_or_textgrid",
                           error=f"{type(exc).__name__}: {exc}")
            inventory.append(row)
    return inventory, inspections
