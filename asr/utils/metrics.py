from __future__ import annotations


def edit_counts(reference: str, prediction: str) -> tuple[int, int, int, int]:
    previous = [(0, 0, j) for j in range(len(prediction) + 1)]
    for i, ref_char in enumerate(reference, 1):
        current = [(0, i, 0)]
        for j, pred_char in enumerate(prediction, 1):
            s, d, ins = previous[j - 1]
            diagonal = (s + (ref_char != pred_char), d, ins)
            s, d, ins = previous[j]
            deletion = (s, d + 1, ins)
            s, d, ins = current[j - 1]
            insertion = (s, d, ins + 1)
            current.append(min((diagonal, deletion, insertion), key=sum))
        previous = current
    s, d, ins = previous[-1]
    return s, d, ins, len(reference)


def score(reference: str, prediction: str) -> dict:
    s, d, ins, n = edit_counts(reference, prediction)
    return {"substitutions": s, "deletions": d, "insertions": ins,
            "ref_units": n, "errors": s + d + ins,
            "cer": (s + d + ins) / n if n else None}


def tokens(text: str) -> list[str]:
    pieces = text.split()
    return pieces if len(pieces) > 1 else list(text)


def sequence_edit_counts(reference: list[str], prediction: list[str]) -> tuple[int, int, int, int]:
    previous = [(0, 0, j) for j in range(len(prediction) + 1)]
    for i, ref_token in enumerate(reference, 1):
        current = [(0, i, 0)]
        for j, pred_token in enumerate(prediction, 1):
            s, d, ins = previous[j - 1]
            diagonal = (s + (ref_token != pred_token), d, ins)
            s, d, ins = previous[j]
            deletion = (s, d + 1, ins)
            s, d, ins = current[j - 1]
            insertion = (s, d, ins + 1)
            current.append(min((diagonal, deletion, insertion), key=sum))
        previous = current
    s, d, ins = previous[-1]
    return s, d, ins, len(reference)


def wer_details(reference: str, prediction: str) -> dict:
    s, d, ins, n = sequence_edit_counts(tokens(reference), tokens(prediction))
    return {"substitutions": s, "deletions": d, "insertions": ins,
            "ref_units": n, "errors": s + d + ins,
            "wer": (s + d + ins) / n if n else None}


def wer(reference: str, prediction: str) -> float | None:
    return wer_details(reference, prediction)["wer"]


def aggregate(rows: list[dict]) -> dict:
    rows = [r for r in rows if r.get("score") and r["score"]["cer"] is not None]
    counts = {key: sum(r["score"][key] for r in rows) for key in ("substitutions", "deletions", "insertions", "ref_units")}
    errors = counts["substitutions"] + counts["deletions"] + counts["insertions"]
    return {**counts, "errors": errors, "segments": len(rows),
            "overall_cer": errors / counts["ref_units"] if counts["ref_units"] else None,
            "mean_utterance_cer": sum(r["score"]["cer"] for r in rows) / len(rows) if rows else None}
