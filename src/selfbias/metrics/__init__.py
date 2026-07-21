"""Metrics package - implements ``docs/METRICS.md``.

Re-exports the Phase 0 core functions for a stable import surface
(``from selfbias.metrics import hspp_r_self``).
"""

from __future__ import annotations

from .core import (
    centered_delta_row,
    hspp_r_family,
    hspp_r_self,
    overestimation_rate_instance,
    overestimation_rate_rubric,
    recognition_accuracy,
    resolve_pairwise,
)

__all__ = [
    "resolve_pairwise",
    "overestimation_rate_instance",
    "overestimation_rate_rubric",
    "hspp_r_self",
    "hspp_r_family",
    "centered_delta_row",
    "recognition_accuracy",
]
