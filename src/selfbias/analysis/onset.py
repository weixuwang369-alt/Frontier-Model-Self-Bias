"""Onset estimation L* (METRICS §5.3).

The onset length L* for a curve is the **smallest** length bin whose bootstrap 95% CI
excludes the null, with **all larger bins also excluding it** (the monotone-onset rule).
Null is chance for recognition/attribution and 1.0 for HSPP-R.
"""

from __future__ import annotations

from dataclasses import dataclass

from .bootstrap import CI


@dataclass
class BinCI:
    bin: int
    ci: CI


def estimate_onset(points: list[BinCI], null: float, side: str = "greater") -> int | None:
    """Return L* - the first bin from which every bin's CI excludes ``null`` - or None.

    ``points`` need not be sorted. ``side`` is "greater" (recognition/attribution/HSPP-R
    rising above the null) or "less".
    """

    ordered = sorted(points, key=lambda p: p.bin)
    n = len(ordered)
    for i, p in enumerate(ordered):
        if p.ci.excludes(null, side) and all(q.ci.excludes(null, side) for q in ordered[i + 1 : n]):
            return p.bin
    return None
