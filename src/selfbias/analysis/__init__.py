"""Analysis layer - recompute METRICS.md from disk (no API calls).

Public entry point: :func:`analyze` builds the full report (length curves with bootstrap
CIs, onset L*, TF-IDF attributability, mechanism regression) and saves it under
``data/reports/``.
"""

from __future__ import annotations

from .bootstrap import CI, bootstrap_over_prompts
from .onset import BinCI, estimate_onset
from .report import analyze

__all__ = ["analyze", "estimate_onset", "bootstrap_over_prompts", "CI", "BinCI"]
