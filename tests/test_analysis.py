"""Analysis layer: bootstrap, onset, and end-to-end report (mock-only)."""

from __future__ import annotations

import pandas as pd

from selfbias.analysis import CI, BinCI, bootstrap_over_prompts, estimate_onset
from selfbias.analysis.reference import build_rb_observations
from selfbias.analysis.report import analyze
from selfbias.pipeline import Pipeline
from selfbias.storage import default_data_paths


def _ci(lo, hi, point=None):
    return CI(point=point if point is not None else (lo + hi) / 2, lo=lo, hi=hi, n_prompts=10)


# --- bootstrap -------------------------------------------------------------


def test_bootstrap_of_constant_is_tight():
    df = pd.DataFrame({"prompt": [1, 2, 3, 4, 5] * 4, "v": [1.0] * 20})
    ci = bootstrap_over_prompts(df, lambda d: d["v"].mean(), n_boot=200, seed=1)
    assert ci.point == 1.0 and ci.lo == 1.0 and ci.hi == 1.0
    assert ci.n_prompts == 5


def test_bootstrap_ci_brackets_the_mean():
    df = pd.DataFrame({"prompt": list(range(20)), "v": [0.0, 1.0] * 10})
    ci = bootstrap_over_prompts(df, lambda d: d["v"].mean(), n_boot=500, seed=2)
    assert ci.lo <= ci.point <= ci.hi
    assert 0.0 <= ci.lo and ci.hi <= 1.0


# --- onset -----------------------------------------------------------------


def test_onset_detected_with_monotone_rule():
    pts = [
        BinCI(25, _ci(0.40, 0.60)),
        BinCI(50, _ci(0.45, 0.65)),
        BinCI(100, _ci(0.55, 0.75)),  # first to clear 0.5...
        BinCI(250, _ci(0.70, 0.90)),  # ...and all larger also clear
    ]
    assert estimate_onset(pts, 0.5) == 100


def test_onset_requires_all_larger_bins_to_clear():
    pts = [
        BinCI(100, _ci(0.55, 0.75)),  # clears 0.5
        BinCI(250, _ci(0.45, 0.65)),  # but this larger bin does NOT
    ]
    assert estimate_onset(pts, 0.5) is None


def test_onset_none_when_never_clears():
    assert estimate_onset([BinCI(50, _ci(0.40, 0.60))], 0.5) is None
    assert estimate_onset([], 0.5) is None


def test_ci_excludes_null_sides():
    assert _ci(1.2, 1.8).excludes(1.0, "greater")
    assert not _ci(0.9, 1.8).excludes(1.0, "greater")
    assert _ci(0.2, 0.4).excludes(0.5, "less")


# --- end-to-end ------------------------------------------------------------


def test_analyze_end_to_end_on_mock(mock_config, tmp_path):
    data_root = str(tmp_path / "data")
    Pipeline(mock_config, data_root=data_root, pricing_path="config/pricing.yaml").run()

    obs = build_rb_observations(default_data_paths(data_root))
    assert not obs.empty
    assert {"judge", "gen_model", "relation", "b_ref", "overest", "bin"} <= set(obs.columns)

    report = analyze(mock_config, data_root=data_root, n_boot=50)
    bins = mock_config.lengths.target_bins_tokens
    for key in ("hspp_r_self", "recognition_accuracy", "attribution_f1"):
        assert key in report["curves"] and len(report["curves"][key]) == len(bins)
        assert key in report["onsets"]
    assert report["hspp_table"]  # per-judge HSPP-R computed
    assert report["regression"]["status"] in ("ok",) or isinstance(
        report["regression"]["status"], str
    )
    # Report persisted to disk.
    import json
    from pathlib import Path

    saved = json.loads(Path(report["_path"]).read_text())
    assert saved["run_id"] == report["run_id"]
