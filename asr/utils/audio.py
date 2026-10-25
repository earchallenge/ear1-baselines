from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def read_segment(path: str | Path, start: float, end: float, sample_rate: int = 16000) -> np.ndarray:
    with sf.SoundFile(str(path)) as handle:
        source_rate = handle.samplerate
        first = max(0, round(start * source_rate))
        last = min(len(handle), round(end * source_rate))
        handle.seek(first)
        audio = handle.read(max(0, last - first), dtype="float32", always_2d=True)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    audio = audio.mean(axis=1)
    if source_rate != sample_rate:
        audio = resample_poly(audio, sample_rate, source_rate).astype(np.float32)
    return np.asarray(audio, dtype=np.float32)
