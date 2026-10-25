"""Per-track cpCER reports; no coupling to the existing CER implementation."""

import csv
import json
from pathlib import Path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def export_reports(plan, details, predictions, summary, results_root):
    root = Path(results_root).resolve() / "whisper-large-v3"
    expected = {(s, d) for s in plan["sessions"] for d in plan["tracks"]}
    actual = {(r["session"], r["device"]) for r in details}
    if actual != expected or len(details) != len(expected):
        raise ValueError(
            "Incomplete or duplicate session-device scores; refusing partial track reports"
        )
    for pair in plan["pairs"]:
        session, device = pair["session"], pair["device"]
        segments = [
            r
            for r in plan["segments"]
            if r["session"] == session and r["device"] == device
        ]
        prepared, inferred = root / "prepare/dev" / device, root / "infer/dev" / device
        write_jsonl(prepared / f"{session}.diarized.jsonl", segments)
        write_jsonl(
            inferred / f"{session}.diarized.jsonl",
            [predictions[r["id"]] for r in segments],
        )
        refs = [
            dict(
                r,
                excluded=r["end"] > pair["duration_sec"] + 0.01
                or r["start"] >= pair["duration_sec"],
            )
            for r in pair["reference_intervals"]
        ]
        write_jsonl(prepared / f"{session}.cpcer_reference.jsonl", refs)
        (prepared / f"{session}.uem").write_text(
            f"{session} 1 0.000000 {pair['duration_sec']:.6f}\n", encoding="utf-8"
        )

    for device in plan["tracks"]:
        items = sorted(
            (r for r in details if r["device"] == device), key=lambda r: r["session"]
        )
        path = root / "infer/dev" / device / "result_cpcer.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        keys = [
            "session",
            "cpcer_pct",
            "errors",
            "reference_characters",
            "excluded_reference_intervals",
            "excluded_reference_characters",
        ]
        with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            for row in items:
                writer.writerow(
                    {
                        k: row["cpcer"] * 100 if k == "cpcer_pct" else row[k]
                        for k in keys
                    }
                )
            total = summary["by_device"][device]
            writer.writerow(
                dict(
                    session="OVERALL",
                    cpcer_pct=100 * total["cpcer"],
                    errors=total["errors"],
                    reference_characters=total["reference_characters"],
                    excluded_reference_intervals=sum(
                        r["excluded_reference_intervals"] for r in items
                    ),
                    excluded_reference_characters=sum(
                        r["excluded_reference_characters"] for r in items
                    ),
                )
            )
        temporary.replace(path)
        print(
            f"[cpCER] {device} OVERALL={summary['by_device'][device]['cpcer']:.2%} (micro): {path}",
            flush=True,
        )

    write_json(root / "cpcer_summary.json", summary)
    lines = [
        "# DiariZen + Whisper large-v3 development cpCER",
        "",
        "| Session | " + " | ".join(f"{d} cpCER (%)" for d in plan["tracks"]) + " |",
        "|---|" + "---:|" * len(plan["tracks"]),
    ]
    lookup = {(r["session"], r["device"]): r["cpcer"] for r in details}
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
        "OVERALL = total character edit errors / total reference characters across sessions, independently for each track.",
        "",
        f"Normalization: {summary['normalization']}.",
        "",
        summary["reference_policy"],
        "",
        summary["recognition_policy"],
        "",
        "The UEM files document full available recording windows. cpCER has no temporal collar. Recognized false-alarm text in silence remains in the hypothesis; overlap is retained. References and their original timestamps are never edited.",
        "",
        f"Reference intervals excluded outside device audio: {summary['reference_excluded_intervals']}. Token-limit segments: {summary['hit_token_limit_segments']}.",
        "",
        summary["note"],
    ]
    (root / "result_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
