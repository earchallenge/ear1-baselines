"""Character-level concatenated minimum-permutation error rate.

Concatenate each speaker chronologically before calling this module. Pad missing
streams with empty strings and minimize character edit errors over a bijection,
as in MeetEval cpWER, but with Unicode characters as the scoring units.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def character_distance(reference: str, hypothesis: str) -> int:
    """Exact unit-cost Levenshtein distance using bit-vector dynamic programming."""
    if reference == hypothesis:
        return 0
    if len(reference) > len(hypothesis):
        reference, hypothesis = hypothesis, reference
    width = len(reference)
    if not width:
        return len(hypothesis)
    equal = {}
    for index, char in enumerate(reference):
        equal[char] = equal.get(char, 0) | (1 << index)
    mask = (1 << width) - 1
    highest = 1 << (width - 1)
    positive, negative, distance = mask, 0, width
    for char in hypothesis:
        matches = equal.get(char, 0)
        vertical = matches | negative
        horizontal = (((matches & positive) + positive) ^ positive) | matches
        increase = negative | ~(horizontal | positive)
        decrease = positive & horizontal
        distance += bool(increase & highest) - bool(decrease & highest)
        increase = (increase << 1) | 1
        decrease <<= 1
        positive = (decrease | ~(vertical | increase)) & mask
        negative = increase & vertical & mask
    return distance


def cpcer(reference: dict[str, str], hypothesis: dict[str, str]) -> dict:
    refs, hyps = sorted(reference), sorted(hypothesis)
    size = max(len(refs), len(hyps))
    ref_chars = sum(len(text) for text in reference.values())
    if not ref_chars:
        raise ValueError("cpCER requires nonempty reference text")
    costs = np.zeros((size, size), dtype=np.int64)
    for i in range(size):
        for j in range(size):
            costs[i, j] = character_distance(
                reference[refs[i]] if i < len(refs) else "",
                hypothesis[hyps[j]] if j < len(hyps) else "",
            )
    rows, cols = linear_sum_assignment(costs)
    errors = int(costs[rows, cols].sum())
    return dict(
        errors=errors,
        reference_characters=ref_chars,
        cpcer=errors / ref_chars,
        reference_speakers=len(refs),
        hypothesis_speakers=len(hyps),
        assignment=[
            dict(
                reference_speaker=refs[i] if i < len(refs) else None,
                hypothesis_speaker=hyps[j] if j < len(hyps) else None,
                errors=int(costs[i, j]),
            )
            for i, j in zip(rows, cols)
        ],
    )
