"""Strict quote-aware Praat long/short TextGrid reader; no Praat dependency.

Empty/invalid intervals are preserved for filtering and auditing. Doubled quotes,
multiline labels, numeric exponents, UTF-8/BOM, UTF-16/BOM and GB18030 supported.
Point tiers are recorded in metadata, never treated as speech intervals.
"""
import math
import re
from pathlib import Path

from .textgrid_normalization import normalize_for_cer, asterisk_count

TOKEN = re.compile(r'"(?:""|[^"])*"|[^\s"\[\]=:]+|[\[\]=:]')


class Reader:
    def __init__(self, source):
        self.tokens = []
        pos = 0
        for match in TOKEN.finditer(source):
            if source[pos:match.start()].strip():
                raise ValueError("Malformed quoted TextGrid string")
            self.tokens.append(match.group())
            pos = match.end()
        if source[pos:].strip():
            raise ValueError("Malformed trailing TextGrid text")
        self.index = 0

    def take(self):
        if self.index >= len(self.tokens):
            raise ValueError("Unexpected end of TextGrid")
        value = self.tokens[self.index]
        self.index += 1
        return value

    def expect(self, *expected):
        for word in expected:
            actual = self.take()
            if actual != word:
                raise ValueError(f"TextGrid token {self.index}: expected {word!r}, got {actual!r}")

    def string(self):
        value = self.take()
        if not (value.startswith('"') and value.endswith('"')):
            raise ValueError(f"Expected quoted TextGrid string, got {value!r}")
        return value[1:-1].replace('""', '"')

    def number(self):
        return float(self.take())

    def count(self):
        value = self.number()
        if not math.isfinite(value) or value < 0 or value != int(value):
            raise ValueError(f"Invalid TextGrid item count: {value}")
        return int(value)


def read_textgrid(path):
    data = Path(path).read_bytes()
    encodings = ("utf-16",) if data.startswith((b"\xff\xfe", b"\xfe\xff")) else ("utf-8-sig", "gb18030")
    for encoding in encodings:
        try:
            source = data.decode(encoding)
            break
        except UnicodeError:
            pass
    else:
        raise ValueError(f"Cannot decode TextGrid: {path}")
    r = Reader(source)
    labeled = r.tokens[:2] == ["File", "type"]
    if labeled:
        r.expect("File", "type", "=")
    file_type = r.string()
    if labeled:
        r.expect("Object", "class", "=")
    if r.string() != "TextGrid" or file_type not in ("ooTextFile", "ooTextFile short"):
        raise ValueError("Only Praat text TextGrid (long/short) is supported")
    long = labeled and file_type == "ooTextFile"
    if long:
        r.expect("xmin", "=")
    xmin = r.number()
    if long:
        r.expect("xmax", "=")
    xmax = r.number()
    if long:
        r.expect("tiers?")
    r.expect("<exists>")
    if long:
        r.expect("size", "=")
    tier_count = r.count()
    if long:
        r.expect("item", "[", "]", ":")
    rows, tiers = [], []
    for tier_index in range(1, tier_count + 1):
        if long:
            r.expect("item", "[", str(tier_index), "]", ":", "class", "=")
        kind = r.string()
        if long:
            r.expect("name", "=")
        name = r.string()
        if long:
            r.expect("xmin", "=")
        tier_start = r.number()
        if long:
            r.expect("xmax", "=")
        tier_end = r.number()
        if kind not in ("IntervalTier", "TextTier"):
            raise ValueError(f"Unsupported tier class: {kind}")
        group = "intervals" if kind == "IntervalTier" else "points"
        if long:
            r.expect(group, ":", "size", "=")
        size = r.count()
        tiers.append(dict(tier_index=tier_index, name=name, kind=kind,
                          xmin=tier_start, xmax=tier_end, entry_count=size))
        for index in range(1, size + 1):
            if long:
                r.expect(group, "[", str(index), "]", ":",
                         "xmin" if kind == "IntervalTier" else "number", "=")
            start = r.number()
            if kind == "IntervalTier":
                if long:
                    r.expect("xmax", "=")
                end = r.number()
                if long:
                    r.expect("text", "=")
                rows.append(dict(speaker=name, tier_index=tier_index,
                                 interval_index=index, start=start, end=end, text=r.string()))
            else:
                if long:
                    r.expect("mark", "=")
                r.string()
    if r.index != len(r.tokens):
        raise ValueError("Unexpected tokens after TextGrid entries")
    if not any(t["kind"] == "IntervalTier" for t in tiers):
        raise ValueError("TextGrid contains no IntervalTier")
    return rows, dict(xmin=xmin, xmax=xmax, tiers=tiers, encoding=encoding)


def prepare_records(intervals, audio_duration):
    """Flag cross-speaker activity before duration filtering; keep >30s overlap.

    Empty/punctuation-only labels do not imply another speaker is active.
    Touching boundaries and duplicate tiers of the same speaker are not overlap.
    """
    rows = []
    for interval in intervals:
        row = {k: v for k, v in interval.items() if k != "text"}
        start, end = row["start"], row["end"]
        raw = interval["text"]
        row.update(reference_raw=raw, reference_norm=normalize_for_cer(raw),
                   reference_asterisk_count=asterisk_count(raw),
                   prediction_raw=None, prediction_norm=None, prediction_asterisk_count=None,
                   original_start=start, original_end=end, clamped_end=False,
                   substitutions=None, deletions=None, insertions=None,
                   ref_chars=len(normalize_for_cer(raw)), cer=None,
                   overlap_flag=False, excluded=False, exclude_reason=None)
        reason = None
        if not math.isfinite(start) or not math.isfinite(end):
            reason = "nonfinite_time"
            # JSON standard does not permit NaN/Infinity.
            for key in ("start", "end", "original_start", "original_end"):
                if not math.isfinite(row[key]):
                    row[key] = None
        elif start < 0:
            reason = "negative_start"
        elif end <= start:
            reason = "end_not_after_start"
        elif end > audio_duration:
            if end - audio_duration <= 0.01 + 1e-9:
                row["end"] = audio_duration
                row["clamped_end"] = True
                if audio_duration <= start:
                    reason = "start_out_of_bounds"
            else:
                reason = "end_out_of_bounds"
        if reason is None and not raw.strip():
            reason = "empty_text"
        if reason is None and not row["reference_norm"]:
            reason = "empty_normalized_text"
        row["duration"] = (row["end"] - row["start"]
                           if row["end"] is not None and row["start"] is not None else None)
        row["excluded"] = reason is not None
        row["exclude_reason"] = reason
        rows.append(row)
    active = []
    candidates = sorted((r for r in rows if not r["excluded"]),
                        key=lambda r: (r["start"], r["end"]))
    for row in candidates:
        active = [other for other in active if other["end"] > row["start"]]
        for other in active:
            if other["speaker"] != row["speaker"] and min(other["end"], row["end"]) > row["start"]:
                other["overlap_flag"] = row["overlap_flag"] = True
        active.append(row)
    for row in rows:
        if not row["excluded"] and row["duration"] > 30:
            row.update(excluded=True, exclude_reason="over_30_seconds")
    return rows
