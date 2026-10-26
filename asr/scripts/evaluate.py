from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.metrics import aggregate, score
from utils.text_normalize import normalize

EVAL_DEVICES = {"D1", "R1"}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.write_text("", encoding="utf-8")
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_file(path: Path, output: Path, expected_pairs: set[tuple[str, str]] | None = None,
                  pair_reasons: dict[str, str] | None = None, model_id: str = "",
                  decoding: dict | None = None, manifest_path: Path | None = None) -> dict:
    started = time.perf_counter()
    if expected_pairs is not None:
        expected_pairs = {(session, str(device).upper()) for session, device in expected_pairs
                          if str(device).upper() in EVAL_DEVICES}
    pair_reasons = {f"{pair.rsplit('/', 1)[0]}/{pair.rsplit('/', 1)[-1].upper()}": reason
                    for pair, reason in (pair_reasons or {}).items()
                    if pair.rsplit('/', 1)[-1].upper() in EVAL_DEVICES}
    scored = []
    ignored_records = 0
    manifest_rows = {}
    if manifest_path is not None:
        manifest_rows = {row.get("id", row.get("utterance_id")): row for row in
                         (json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip())}
    all_rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        device = str(item.get("device_id", item.get("device", ""))).upper()
        if device not in EVAL_DEVICES:
            ignored_records += 1
            continue
        uid = item.get("utterance_id", item.get("id"))
        manifest_item = manifest_rows.get(uid, {})
        reference = manifest_item.get("reference_raw", manifest_item.get("reference_text", item.get("reference", "")))
        prediction = item.get("prediction", item.get("prediction_text", ""))
        ref = normalize(manifest_item.get("reference_norm", reference))
        pred = normalize(item.get("prediction_normalized", item.get("prediction_norm", prediction)))
        item = dict(item)
        item.update(
            utterance_id=item.get("utterance_id", item.get("id")),
            session_id=item.get("session_id", item.get("session")),
            device_id=device,
            reference=reference,
            prediction=prediction,
            reference_normalized=ref,
            prediction_normalized=pred,
            status=item.get("status", "SUCCESS"),
            error=item.get("error", ""),
        )
        if item["status"] != "SUCCESS":
            item.update(substitutions=None, deletions=None, insertions=None, ref_units=None, cer=None)
        elif not ref:
            item["status"] = "EXCLUDED_EMPTY_REFERENCE"
            item.update(substitutions=None, deletions=None, insertions=None, ref_units=None, cer=None)
        else:
            details = score(ref, pred)
            item.update(details)
            scored.append(item)
        item["session"] = item["session_id"]
        item["device"] = item["device_id"]
        item["score"] = ({k: item[k] for k in ("substitutions", "deletions", "insertions", "ref_units", "errors", "cer")}
                         if item.get("cer") is not None else None)
        all_rows.append(item)
    if not all_rows:
        raise ValueError(f"No D1/R1 predictions found in {path}; prepare and recognize D1/R1 first.")
    overall = aggregate(scored)
    def grouped(key):
        groups = defaultdict(list)
        for row in scored:
            groups[row.get(f"{key}_id", row.get(key, "unknown"))].append(row)
        return [{key: group, **aggregate(rows)} for group, rows in sorted(groups.items())]
    grouped_session_device = defaultdict(list)
    for row in scored:
        grouped_session_device[(row.get("session_id", "unknown"), row.get("device_id", "unknown"))].append(row)
    session_devices = [{"session": session, "device": device, **aggregate(rows)}
                       for (session, device), rows in sorted(grouped_session_device.items())]
    sessions, devices = grouped("session"), grouped("device")
    session_cers = [row["overall_cer"] for row in sessions if row.get("overall_cer") is not None]
    session_averaged_cer = sum(session_cers) / len(session_cers) if session_cers else None
    corpus_micro_cer = overall.get("overall_cer")
    elapsed = time.perf_counter() - started
    scored_pairs = {(row.get("session_id", "unknown"), row.get("device_id", "unknown")) for row in scored}
    observed_pairs = {(row.get("session_id", "unknown"), row.get("device_id", "unknown")) for row in all_rows}
    pair_universe = expected_pairs or observed_pairs
    skipped_pairs = len(pair_universe - scored_pairs)
    summary = {
        "Session-averaged CER (%)": session_averaged_cer * 100 if session_averaged_cer is not None else None,
        "Overall CER (%)": corpus_micro_cer * 100 if corpus_micro_cer is not None else None,
        "# Utterances": len(scored),
        "# Reference Characters": overall.get("ref_units", 0),
        "# Scored Session-Device Pairs": len(scored_pairs),
        "# Skipped Session-Device Pairs": skipped_pairs,
    }
    scores_summary = {
        "model": model_id,
        "split": "dev",
        "eval_devices": sorted(EVAL_DEVICES),
        "ignored_non_eval_records": ignored_records,
        "session_averaged_cer": session_averaged_cer,
        "overall_cer": corpus_micro_cer,
        "utterances": len(scored),
        "reference_characters": overall.get("ref_units", 0),
        "substitutions": overall.get("substitutions", 0),
        "deletions": overall.get("deletions", 0),
        "insertions": overall.get("insertions", 0),
        "scored_session_device_pairs": len(scored_pairs),
        "skipped_session_device_pairs": skipped_pairs,
        "decoding": decoding or {},
        "text_normalization": "NFKC, lowercase, remove whitespace and Unicode punctuation",
        "excluded_session_device_pairs": pair_reasons or {},
    }
    session_report = [
        {"Session": row["session"], "# Utterances": row["segments"],
         "CER (%)": row["overall_cer"] * 100 if row.get("overall_cer") is not None else None,
         "# Reference Characters": row["ref_units"], "Substitutions": row["substitutions"],
         "Deletions": row["deletions"], "Insertions": row["insertions"], "Errors": row["errors"]}
        for row in sessions
    ]
    device_report = [
        {"Device": row["device"], "# Utterances": row["segments"],
         "CER (%)": row["overall_cer"] * 100 if row.get("overall_cer") is not None else None,
         "# Reference Characters": row["ref_units"], "Substitutions": row["substitutions"],
         "Deletions": row["deletions"], "Insertions": row["insertions"], "Errors": row["errors"]}
        for row in devices
    ]
    session_device_report = [
        {"Session": row["session"], "Device": row["device"], "# Utterances": row["segments"],
         "CER (%)": row["overall_cer"] * 100 if row.get("overall_cer") is not None else None,
         "# Reference Characters": row["ref_units"], "Substitutions": row["substitutions"],
         "Deletions": row["deletions"], "Insertions": row["insertions"], "Errors": row["errors"]}
        for row in session_devices
    ]
    overall = {
        **overall,
        "corpus_micro_cer": corpus_micro_cer,
        "session_averaged_cer": session_averaged_cer,
        "session_count": len(session_cers),
    }
    metrics = {"summary": summary, "overall": overall, "sessions": session_report,
               "devices": device_report, "session_devices": session_device_report,
               "scored_records": len(scored), "total_records": len(all_rows),
               "elapsed_sec": elapsed,
               "note": "Session-averaged CER is the equal-weight mean of Session CERs. Overall CER is total edit errors divided by total reference characters. Neither is the utterance-level mean CER."}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "scores_by_session.csv", session_report)
    write_csv(output / "scores_by_device.csv", device_report)
    write_csv(output / "scores_by_session_device.csv", session_device_report)
    (output / "scores_summary.json").write_text(json.dumps(scores_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if ignored_records:
        print(f"Ignored {ignored_records} prediction records outside D1/R1.")
    print("Session CER (D1/R1 only)")
    print(f"{'Session':<20} {'# Utterances':>13} {'CER (%)':>12}")
    print("-" * 50)
    for row in session_report:
        cer = row.get("CER (%)")
        cer_text = f"{cer:.2f}" if cer is not None else "N/A"
        print(f"{row['Session']:<20} {row['# Utterances']:>13,} {cer_text:>12}")
    print("-" * 50)
    session_average_text = f"{summary['Session-averaged CER (%)']:.2f}" if summary["Session-averaged CER (%)"] is not None else "N/A"
    overall_text = f"{summary['Overall CER (%)']:.2f}" if summary["Overall CER (%)"] is not None else "N/A"
    print(f"Session-averaged CER (%): {session_average_text}")
    print(f"Overall CER (%):          {overall_text}")
    print(f"# Utterances:             {summary['# Utterances']:,}")
    print(f"# Reference Characters:   {summary['# Reference Characters']:,}")
    print(f"# Scored Session-Device Pairs:  {summary['# Scored Session-Device Pairs']:,}")
    print(f"# Skipped Session-Device Pairs: {summary['# Skipped Session-Device Pairs']:,}")
    print(f"Elapsed: {elapsed:.1f}s")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["whisper-large-v2", "whisper-large-v3"])
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--manifest", type=Path, default=Path("data/dev.jsonl"))
    parser.add_argument("--comparison", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    if args.comparison:
        reports = {}
        for name in ("whisper-large-v2", "whisper-large-v3"):
            p = root / name / "dev" / "metrics.json"
            if p.is_file():
                report = json.loads(p.read_text(encoding="utf-8"))
                devices = {str(row.get("Device", "")).upper() for row in report.get("devices", [])}
                if not devices or not devices <= EVAL_DEVICES:
                    raise ValueError(f"Comparison requires a D1/R1-only report: {p}")
                reports[name] = report
        lines = ["# Whisper baseline comparison", "", "| Model | Session-averaged CER (%) | Overall CER (%) | # Utterances |", "|---|---:|---:|---:|"]
        for name, report in reports.items():
            values = report.get("summary", {})
            session_average = values.get("Session-averaged CER (%)")
            overall = values.get("Overall CER (%)")
            session_average_text = f"{session_average:.2f}" if session_average is not None else "N/A"
            overall_text = f"{overall:.2f}" if overall is not None else "N/A"
            lines.append(f"| {name} | {session_average_text} | {overall_text} | {values.get('# Utterances', 'N/A')} |")
        (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 0
    if not args.model:
        parser.error("--model is required unless --comparison is used")
    expected_pairs = set()
    pair_reasons = {}
    inventory_path = args.manifest.resolve().parent / "dev_devices.jsonl"
    if inventory_path.is_file():
        for line in inventory_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                pair = (item.get("session", "unknown"), item.get("device", "unknown"))
                expected_pairs.add(pair)
                if item.get("status") != "READY":
                    pair_reasons[f"{pair[0]}/{pair[1]}"] = item.get("error") or item.get("reason") or item.get("status", "excluded")
    model_id = f"openai/{args.model}"
    evaluate_file(root / args.model / "dev" / "predictions.jsonl", root / args.model / "dev",
                  expected_pairs or None, pair_reasons, model_id,
                  {"language": "zh", "task": "transcribe", "num_beams": 1, "max_length": 448},
                  args.manifest.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




