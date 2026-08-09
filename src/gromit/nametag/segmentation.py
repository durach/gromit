"""Cut a video's per-frame feature sequence into stable layout segments."""

from __future__ import annotations

import itertools
from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    start: int
    end: int  # inclusive
    representative: int


def _l1(a, b) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def segment_features(features, threshold: float = 0.08) -> list[Segment]:
    if not features:
        return []
    bounds = [0]
    for i in range(1, len(features)):
        if _l1(features[i], features[i - 1]) > threshold:
            bounds.append(i)
    bounds.append(len(features))
    segs = []
    for s, e in itertools.pairwise(bounds):
        last = e - 1
        segs.append(Segment(start=s, end=last, representative=(s + last) // 2))
    return segs
