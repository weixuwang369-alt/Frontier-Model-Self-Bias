"""Assemble the analysis report for a run and save it to ``data/reports/<run_id>.json``.

Recomputes everything from disk (no API calls). The dashboard Results page reads this
JSON to show real-run curves, onsets, and the mechanism regression.
"""

from __future__ import annotations

import json
import math

from ..config import ExperimentConfig
from ..storage import default_data_paths
from .attributability import attribution_curve, attribution_df
from .curves import (
    hspp_curve,
    per_judge_hspp,
    recognition_curve,
    recognition_df,
)
from .onset import BinCI, estimate_onset
from .reference import build_rb_observations, overestimation_matrix
from .regression import mechanism_regression


def _f(v) -> float | None:
    """JSON-safe float: NaN/inf → None, else rounded."""

    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, 4)


def _curve(points: list[BinCI], null: float) -> list[dict]:
    return [
        {"bin": p.bin, "point": _f(p.ci.point), "lo": _f(p.ci.lo), "hi": _f(p.ci.hi), "null": null}
        for p in points
    ]


def analyze(
    config: ExperimentConfig, data_root: str = "data", *, n_boot: int = 1000, seed: int = 0
) -> dict:
    paths = default_data_paths(data_root).ensure()
    bins = config.lengths.target_bins_tokens
    n_models = len(config.roster)
    chance_attr = 1.0 / n_models

    obs = build_rb_observations(paths)
    probe_df = recognition_df(paths)
    attr_df = attribution_df(paths)

    hspp = hspp_curve(obs, bins, n_boot=n_boot, seed=seed) if not obs.empty else []
    recog = recognition_curve(probe_df, bins, n_boot=n_boot, seed=seed)
    attr = attribution_curve(attr_df, bins, n_models, n_boot=n_boot, seed=seed)

    onsets = {
        "hspp_r_self": estimate_onset(hspp, 1.0) if hspp else None,
        "recognition_accuracy": estimate_onset(recog, 0.5),
        "attribution_f1": estimate_onset(attr, chance_attr),
    }

    hspp_table = []
    if not obs.empty:
        for r in per_judge_hspp(obs, bins[-1]).to_dict("records"):
            hspp_table.append(
                {
                    "judge": r["judge"],
                    "hspp_r_self": _f(r["hspp_r_self"]),
                    "hspp_r_fam": _f(r["hspp_r_fam"]),
                }
            )

    matrix = []
    if not obs.empty:
        for r in overestimation_matrix(obs).to_dict("records"):
            matrix.append(
                {
                    "judge": r["judge"],
                    "generator": r["gen_model"],
                    "relation": r["relation"],
                    "O": _f(r["O"]),
                }
            )

    report = {
        "run_id": config.run_id(),
        "run_name": config.run.name,
        "n_models": n_models,
        "length_bins": bins,
        "n_boot": n_boot,
        "nulls": {"hspp_r_self": 1.0, "recognition_accuracy": 0.5, "attribution_f1": chance_attr},
        "curves": {
            "hspp_r_self": _curve(hspp, 1.0),
            "recognition_accuracy": _curve(recog, 0.5),
            "attribution_f1": _curve(attr, chance_attr),
        },
        "onsets": onsets,
        "hspp_table": hspp_table,
        "overestimation_matrix": matrix,
        "regression": mechanism_regression(obs) if not obs.empty else {"status": "no observations"},
    }

    out = paths.reports / f"{config.run_id()}.json"
    out.write_text(json.dumps(report, indent=2))
    report["_path"] = str(out)
    return report
