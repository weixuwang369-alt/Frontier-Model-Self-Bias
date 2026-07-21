"""Prompt-level bootstrap (METRICS §8): resample prompts, recompute the statistic.

All headline numbers get a prompt-level bootstrap CI (B ≥ 1000). Resampling is by
*prompt* (not row) so within-prompt correlation is respected. Seeded for reproducibility.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CI:
    point: float
    lo: float
    hi: float
    n_prompts: int

    def excludes(self, null: float, side: str = "greater") -> bool:
        """Whether the CI excludes ``null`` on the given side."""

        if np.isnan(self.lo) or np.isnan(self.hi):
            return False
        if side == "greater":
            return self.lo > null
        if side == "less":
            return self.hi < null
        return self.lo > null or self.hi < null  # two-sided


def bootstrap_over_prompts(
    df: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    prompt_col: str = "prompt",
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> CI:
    """Bootstrap ``statistic(df)`` by resampling whole prompts with replacement.

    Returns the point estimate on the full data plus a percentile CI. A statistic that
    returns NaN on a resample (e.g. a bin with no eligible data) is dropped from the CI.
    """

    point = float(statistic(df))
    prompts = df[prompt_col].unique()
    n = len(prompts)
    if n == 0:
        return CI(point=point, lo=float("nan"), hi=float("nan"), n_prompts=0)

    rng = np.random.default_rng(seed)
    # Pre-group once for fast reassembly.
    groups = {p: g for p, g in df.groupby(prompt_col)}
    stats: list[float] = []
    for _ in range(n_boot):
        pick = rng.choice(prompts, size=n, replace=True)
        sample = pd.concat([groups[p] for p in pick], ignore_index=True)
        try:
            val = float(statistic(sample))
        except Exception:  # noqa: BLE001 - a degenerate resample just doesn't count
            val = float("nan")
        if not np.isnan(val):
            stats.append(val)

    if not stats:
        return CI(point=point, lo=float("nan"), hi=float("nan"), n_prompts=n)
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return CI(point=point, lo=lo, hi=hi, n_prompts=n)
