from __future__ import annotations

import math

from selfbias.metrics.core import (
    centered_delta_row,
    hspp_r_family,
    hspp_r_self,
    overestimation_rate_instance,
    overestimation_rate_rubric,
    recognition_accuracy,
    resolve_pairwise,
)


def test_resolve_pairwise_rules():
    # Both orderings agree G wins.
    assert resolve_pairwise(verdict_ab=1, verdict_ba=-1) == 1
    # Winner + tie -> winner stands.
    assert resolve_pairwise(verdict_ab=1, verdict_ba=0) == 1
    assert resolve_pairwise(verdict_ab=0, verdict_ba=-1) == 1
    # Both tie -> tie.
    assert resolve_pairwise(verdict_ab=0, verdict_ba=0) == 0
    # Disagreement on winner -> tie.
    assert resolve_pairwise(verdict_ab=1, verdict_ba=1) == 0
    # G loses in both.
    assert resolve_pairwise(verdict_ab=-1, verdict_ba=1) == -1


def test_overestimation_rate_instance():
    # Eligible = w_ref == -1. Overestimate when w_judge > w_ref.
    comps = [(-1, 1), (-1, 0), (-1, -1), (0, 1)]  # last is ineligible
    # Among 3 eligible: (1>-1 yes), (0>-1 yes), (-1>-1 no) -> 2/3
    assert math.isclose(overestimation_rate_instance(comps), 2 / 3)
    assert overestimation_rate_instance([(0, 0)]) is None


def test_overestimation_rate_rubric():
    verdicts = [(-1, 1), (-1, -1), (1, 1)]  # eligible: b_ref==-1 (first two)
    assert math.isclose(overestimation_rate_rubric(verdicts), 0.5)


def test_hspp_r_self_and_family():
    assert math.isclose(hspp_r_self(0.4, [0.2, 0.2]), 2.0)
    assert hspp_r_self(0.4, []) is None
    assert math.isclose(hspp_r_family([0.3, 0.3], [0.2, 0.2]), 1.5)


def test_centered_delta_row_sums_to_zero():
    row = centered_delta_row({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 0.0})
    assert math.isclose(sum(row.values()), 0.0, abs_tol=1e-9)


def test_recognition_accuracy():
    assert recognition_accuracy([True, True, False, False]) == 0.5
    assert recognition_accuracy([]) is None
