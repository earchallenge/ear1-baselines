#!/usr/bin/env python3
"""EAR-1 preliminary single-channel D1/R1 speaker diarization baseline."""
import argparse
import csv
import re
import sys
from pathlib import Path

SESSION_RE = re.compile(r'^\d{8}-\d+$')
EXCLUDED = {'20260916-5'}
TIER_RE = re.compile(r'^\s*item\s*\[\d+\]\s*:\s*$', re.M)
NAME_RE = re.compile(r'^\s*name\s*=\s*"([^"]+)"\s*$', re.M)
INTERVAL_RE = re.compile(r'intervals\s*\[\d+\]\s*:\s*\n\s*xmin\s*=\s*([\d.eE+-]+)\s*\n\s*xmax\s*=\s*([\d.eE+-]+)\s*\n\s*text\s*=\s*"((?:[^"]|"")*)"', re.M)


def read_split(data_root, split_path=None):
    path = Path(split_path) if split_path else Path(data_root) / 'split.lst'
    groups = {'train': [], 'dev': []}
    current = None
    for raw in path.read_text(encoding='utf-8-sig').splitlines():
        line = raw.strip()
        if not line:
            continue
        heading = line.replace('：', ':').lower()
        if heading in ('train set:', 'dev set:'):
            current = heading.split()[0]
        elif current and SESSION_RE.fullmatch(line):
            if line in groups[current]:
                raise ValueError(f'Duplicate session in {path}: {line}')
            if line not in EXCLUDED:
                groups[current].append(line)
        else:
            raise ValueError(f'Invalid split line in {path}: {line}')
    if not all(groups.values()):
        raise ValueError(f'Empty train or dev split: {path}')
    if set(groups['train']) & set(groups['dev']):
        raise ValueError(f'Train/dev session overlap in {path}')
    return groups


def read_speech(textgrid):
    content = textgrid.read_text(encoding='utf-8-sig')
    pieces = TIER_RE.split(content)[1:]
    matches = TIER_RE.findall(content)
    if not pieces or not matches:
        raise ValueError(f'No TextGrid tiers: {textgrid}')
    segments = []
    for tier in pieces:
        m = NAME_RE.search(tier)
        if not m:
            raise ValueError(f'No tier name in {textgrid}')
        speaker = m.group(1)
        found = list(INTERVAL_RE.finditer(tier))
        declared = re.search(r'intervals:\s*size\s*=\s*(\d+)', tier)
        if declared and int(declared.group(1)) != len(found):
            raise ValueError(f'Interval count mismatch in {textgrid}: {len(found)} vs {declared.group(1)}')
        for interval in found:
            start, end = float(interval.group(1)), float(interval.group(2))
            # Non-empty transcription denotes oracle speech. Empty text is silence.
            txt = interval.group(3).replace('""', '"').strip()
            if txt and end > start:
                segments.append((start, end, speaker))
    return segments


def select_wav(session_dir, track):
    candidates = [p for p in session_dir.iterdir()
                  if p.is_file() and p.stem.upper() == track.upper() and p.suffix.lower() == '.wav']
    if len(candidates) != 1:
        raise FileNotFoundError(f'Expected one {track} WAV in {session_dir}; found {candidates}')
    return candidates[0]


def get_duration(path):
    import soundfile as sf
    info = sf.info(str(path))
    if info.channels != 1:
        raise ValueError(f'Expected mono audio: {path}, found {info.channels} channels')
    return info.frames / info.samplerate


def write_rttm(path, sid, segments):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for start, end, spk in sorted(segments):
            f.write(f'SPEAKER {sid} 1 {start:.6f} {end-start:.6f} <NA> <NA> {spk} <NA> <NA>\n')


def load_rttm(path, sid):
    from pyannote.core import Annotation, Segment
    ann = Annotation(uri=sid)
    with path.open(encoding='utf-8') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if not parts or parts[0] != 'SPEAKER':
                continue
            if len(parts) < 8 or parts[1] != sid:
                raise ValueError(f'Invalid RTTM line: {line}')
            start, dur = float(parts[3]), float(parts[4])
            if dur > 0:
                ann[Segment(start, start + dur), idx] = parts[7]
    return ann



def prepare(args):
    """Generate reference RTTMs for train and development sessions."""
    root, out = Path(args.data).resolve(), Path(args.out).resolve()

    splits = read_split(root, args.split)
    missing = [sid for ids in splits.values() for sid in ids
               if not (root / sid).is_dir()]
    if missing:
        raise FileNotFoundError(f'Missing sessions: {missing}')

    for split, session_ids in splits.items():
        print(f'[PREP] {split}: {len(session_ids)} sessions', flush=True)

        for sid in session_ids:
            sess = root / sid

            # Load oracle speech annotations from per-speaker TextGrid files.
            grids = sorted((sess / 'A').glob('P*.TextGrid'))
            grids = [tg for tg in grids if '.ear1.' not in tg.name]

            if not grids:
                raise FileNotFoundError(
                    f'No speaker TextGrid files in {sess / "A"}'
                )

            segments = []
            seen_spk = set()

            for tg in grids:
                rows = read_speech(tg)
                labels = {speaker for _, _, speaker in rows}

                if labels and labels != {tg.stem}:
                    raise ValueError(
                        f'Speaker mismatch in {tg}: {labels} != {tg.stem}'
                    )

                if tg.stem in seen_spk:
                    raise ValueError(f'Duplicate speaker {tg.stem} in {sid}')

                seen_spk.add(tg.stem)
                segments.extend(rows)

            # Check audio/annotation duration mismatches.
            # Preserve original TextGrid timestamps without truncation.
            ref_end = max((end for _, end, _ in segments), default=0.0)

            for track in args.tracks:
                try:
                    wav = select_wav(sess, track)
                except FileNotFoundError as exc:
                    print(
                        f'[WARN] {split}/{sid} {track}: {exc}',
                        file=sys.stderr,
                        flush=True,
                    )
                    continue

                duration = get_duration(wav)

                if ref_end > duration + 0.5:
                    print(
                        f'[WARN] {split}/{sid} {track}: '
                        f'reference speech ends at {ref_end:.2f}s, '
                        f'but audio ends at {duration:.2f}s '
                        f'(+{ref_end - duration:.2f}s). '
                        'Reference RTTM will not be truncated.',
                        file=sys.stderr,
                        flush=True,
                    )

            # Save reference RTTM to the corresponding data split.
            dest = out / 'prepare' / split / f'{sid}.rttm'
            write_rttm(dest, sid, segments)

            print(
                f'[PREP] {split}/{sid}: speakers={len(seen_spk)} '
                f'speech_intervals={len(segments)} -> {dest}',
                flush=True,
            )

    print('[PREP] train/dev reference RTTMs complete', flush=True)


def infer(args):
    import torch
    from diarizen.pipelines.inference import DiariZenPipeline
    from pyannote.core import Annotation
    root, out = Path(args.data).resolve(), Path(args.out).resolve()

    # Require CUDA for diarization inference.
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. GPU is required for inference.')

    device = torch.device('cuda:0')
    print(f'[DEVICE] Using GPU: {torch.cuda.get_device_name(device)}', flush=True)

    # Load the pretrained DiariZen model and move it to GPU.
    print(f'[MODEL] Loading {args.model}', flush=True)
    pipeline = DiariZenPipeline.from_pretrained(args.model)
    pipeline.to(device)
    for track in args.tracks:
        for sid in read_split(root, args.split)['dev']:
            wav = select_wav(root / sid, track)
            dest = out / 'infer' / 'dev' / track / f'{sid}.rttm'
            if dest.exists() and not args.overwrite:
                print(f'[SKIP] {dest}', flush=True)
                continue
            print(f'[INFER] {track} {sid} {wav}', flush=True)
            ann = pipeline(str(wav))
            if not isinstance(ann, Annotation):
                raise TypeError(f'Expected pyannote.core.Annotation, got {type(ann)}')
            segments = [(seg.start, seg.end, str(label)) for seg, _, label in ann.itertracks(yield_label=True)]
            write_rttm(dest, sid, segments)
            print(f'[INFER] saved {dest}: {len(segments)} segments', flush=True)


def score(args):
    from pyannote.core import Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate
    root, out = Path(args.data).resolve(), Path(args.out).resolve()
    for track in args.tracks:
        metric = DiarizationErrorRate(collar=args.collar, skip_overlap=args.skip_overlap)
        rows = []
        print(f'\n[SCORE] {track} (collar={args.collar}, skip_overlap={args.skip_overlap})')
        for sid in read_split(root, args.split)['dev']:
            ref_path = out / 'prepare' / 'dev' / f'{sid}.rttm'
            hyp_path = out / 'infer' / 'dev' / track / f'{sid}.rttm'
            if not ref_path.exists() or not hyp_path.exists():
                raise FileNotFoundError(f'Missing RTTM: {ref_path} or {hyp_path}')
            ref = load_rttm(ref_path, sid)
            hyp = load_rttm(hyp_path, sid)
            duration = get_duration(select_wav(root / sid, track))
            ref_end = max((segment.end for segment in ref.itersegments()), default=0.0)
            if ref_end > duration + 0.5:
                print(
                    f'[WARN] {track} {sid}: reference speech ends at {ref_end:.2f}s, '
                    f'but audio ends at {duration:.2f}s (+{ref_end - duration:.2f}s). '
                    'Scoring only the available audio range; reference RTTM is unchanged.',
                    file=sys.stderr, flush=True,
                )
            # Track-specific full-recording UEM: score only the audio time span,
            # including silence (to penalize false alarms). The reference RTTM
            # is never clipped or rewritten.
            uem = Timeline([Segment(0.0, duration)], uri=sid)
            d = metric(ref, hyp, uem=uem, detailed=True)
            row = {'session': sid, 'der_pct': 100 * d['diarization error rate'],
                   'miss_s': d['missed detection'], 'fa_s': d['false alarm'],
                   'conf_s': d['confusion'], 'ref_s': d['total']}
            rows.append(row)
            print(f"  {sid} DER={row['der_pct']:.2f}%  MISS={row['miss_s']:.2f}  FA={row['fa_s']:.2f}  CONF={row['conf_s']:.2f}")
        totals = metric[:]
        combined = 100 * abs(metric)
        print(f"  OVERALL DER={combined:.2f}% (micro-averaged); MISS={totals['missed detection']:.2f}s FA={totals['false alarm']:.2f}s CONF={totals['confusion']:.2f}s REF={totals['total']:.2f}s")
        # Save DER results alongside the inference RTTM files.
        score_dir = out / 'infer' / 'dev' / track
        score_dir.mkdir(parents=True, exist_ok=True)

        suffix = '_skip_overlap' if args.skip_overlap else ''
        path = score_dir / f'result_collar{args.collar:g}{suffix}.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
            writer.writerow({'session':'OVERALL', 'der_pct':combined,
                             'miss_s':totals['missed detection'], 'fa_s':totals['false alarm'],
                             'conf_s':totals['confusion'], 'ref_s':totals['total']})
        print(f'[SCORE] saved {path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'infer', 'score'])
    p.add_argument('--data', required=True, help='EarChallenge data directory')
    p.add_argument('--split', help='Official split file; default: DATA_ROOT/split.lst')
    p.add_argument('--out', required=True)
    p.add_argument('--tracks', nargs='+', default=['D1', 'R1'], choices=['D1', 'R1'])
    p.add_argument('--model', default='BUT-FIT/diarizen-wavlm-large-s80-md-v2')
    p.add_argument('--collar', type=float, default=0.)
    p.add_argument('--skip-overlap', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    {'prepare':prepare, 'infer':infer, 'score':score}[args.stage](args)

if __name__ == '__main__':
    main()
