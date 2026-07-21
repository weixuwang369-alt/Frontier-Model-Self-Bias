"""Core metric functions - implement ``docs/METRICS.md`` definitions 1:1.

Pure functions over already-loaded outcomes (no disk, no API). Phase 0 implements the
headline pieces used by the dashboard demo and unit tests: PWC resolution, overestimation
rates, HSPP-Ratio (self/family), the centered score-delta matrix, and recognition
accuracy. The remaining definitions (EO-Bias, MIPA/MRA, calibration, onset, fingerprint
density) are scheduled with the phases that produce their inputs.
"""

from __future__ import annotations

from statistics import mean

# ---------------------------------------------------------------------------
# 1.1 PWC resolution (order-swapped pair -> single outcome for G vs G')
# ---------------------------------------------------------------------------


def resolve_pairwise(verdict_ab: int, verdict_ba: int) -> int:
    """Resolve two order-swapped verdicts into w(G, G') in {+1, 0, -1}.

    ``verdict_ab``: verdict when G is presented FIRST (+1 => first/G wins).
    ``verdict_ba``: verdict when G' is presented first (+1 => first/G' wins).

    Rule (METRICS §1.1): if one run yields a winner and the other a tie, the winner
    stands; if the runs disagree on the winner, or both are ties, the result is a tie.
    """

    a_from_ab = verdict_ab  # +1 G wins, 0 tie, -1 G loses
    a_from_ba = -verdict_ba  # flip: first-position win there means G' won
    winners = {a_from_ab, a_from_ba}
    if winners == {0}:  # both ties
        return 0
    non_tie = {w for w in winners if w != 0}
    if len(non_tie) == 1 and 0 in winners:  # one winner + one tie -> winner stands
        return next(iter(non_tie))
    if len(non_tie) == 1:  # both agree on the same winner
        return next(iter(non_tie))
    return 0  # disagreement on winner -> tie


# ---------------------------------------------------------------------------
# 2. Overestimation rates (Pombal)
# ---------------------------------------------------------------------------


def overestimation_rate_instance(comparisons: list[tuple[int, int]]) -> float | None:
    """O_inst(J, G): among comparisons where G should lose (w* = -1), the fraction where
    the judge ruled more favorably than warranted (w_J > w*).

    ``comparisons`` = list of (w_ref, w_judge), each in {+1, 0, -1}.
    Returns None when there are no eligible (w_ref == -1) comparisons.
    """

    eligible = [(wr, wj) for wr, wj in comparisons if wr == -1]
    if not eligible:
        return None
    return mean(1.0 if wj > wr else 0.0 for wr, wj in eligible)


def overestimation_rate_rubric(verdicts: list[tuple[int, int]]) -> float | None:
    """O_rub(J, G): among rubrics G objectively fails (b* = -1), the fraction the judge
    marked satisfied (b_J = +1). ``verdicts`` = list of (b_ref, b_judge) in {-1, +1}."""

    eligible = [(br, bj) for br, bj in verdicts if br == -1]
    if not eligible:
        return None
    return mean(1.0 if bj == 1 else 0.0 for br, bj in eligible)


# ---------------------------------------------------------------------------
# 3.1 / 3.2 HSPP-Ratio
# ---------------------------------------------------------------------------


def hspp_r_self(o_self: float, o_unrelated: list[float]) -> float | None:
    """HSPP-R_self(J) = O(J, J) / mean_{G in S_J} O(J, G). >1 => self-preference."""

    others = [o for o in o_unrelated if o is not None]
    if not others:
        return None
    denom = mean(others)
    if denom == 0:
        return float("inf") if o_self and o_self > 0 else None
    return o_self / denom


def hspp_r_family(o_family: list[float], o_unrelated: list[float]) -> float | None:
    """HSPP-R_fam(J) = mean_{G in F_J} O(J, G) / mean_{G in S_J} O(J, G)."""

    fam = [o for o in o_family if o is not None]
    others = [o for o in o_unrelated if o is not None]
    if not fam or not others:
        return None
    denom = mean(others)
    if denom == 0:
        return float("inf")
    return mean(fam) / denom


# ---------------------------------------------------------------------------
# 3.4 Centered score-delta matrix
# ---------------------------------------------------------------------------


def centered_delta_row(
    judge_scores: dict[str, float], ref_scores: dict[str, float]
) -> dict[str, float]:
    """One judge's row of Δ(J, G) = [score_J(G) - score*(G)] - mean_G[score_J(G) - score*(G)]."""

    raw = {g: judge_scores[g] - ref_scores[g] for g in judge_scores if g in ref_scores}
    if not raw:
        return {}
    m = mean(raw.values())
    return {g: v - m for g, v in raw.items()}


# ---------------------------------------------------------------------------
# 5.2 Self-recognition accuracy
# ---------------------------------------------------------------------------


def recognition_accuracy(correct_flags: list[bool]) -> float | None:
    """Pairwise recognition accuracy vs 50% chance. Returns None if no probes."""

    if not correct_flags:
        return None
    return mean(1.0 if c else 0.0 for c in correct_flags)
