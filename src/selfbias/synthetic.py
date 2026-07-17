"""Synthetic results dataset for the keyless dashboard demo.

Self-contained and deterministic (fixed seed) - needs no API keys and no real run. It
plants a plausible, literature-shaped signal so the Results page renders real charts:

* self and same-family judge×generator cells overestimate more (SPB + family bias);
* HSPP-R > 1 for self, with pairwise > rubric (H5);
* recognition accuracy, attribution F1, and HSPP-R all rise with length and share an
  onset region (the H1-vs-H2 "money plot");
* subjective domains show more bias than the objective arm.

This is illustrative data, clearly labelled as such in the UI - never confused with a
real run's outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics.core import hspp_r_family, hspp_r_self

DEMO_SEED = 20260716

# Mirrors the resolved roster (labels only; demo data is illustrative).
ROSTER = [
    ("claude-opus-4-8", "anthropic"),
    ("claude-haiku-4-5", "anthropic"),
    ("gemini-2.5-pro", "google"),
    ("gemini-2.5-flash", "google"),
    ("gpt-5", "openai"),
    ("gpt-5-mini", "openai"),
]
LENGTH_BINS = [25, 50, 100, 250, 500, 1000, 2500]
DOMAINS = ["verifiable_if", "open_qa_summarization", "creative_writing"]
PARADIGMS = ["pwc", "rubric"]


@dataclass
class DemoDataset:
    overestimation: pd.DataFrame  # long: judge, generator, relation, O
    delta_matrix: pd.DataFrame  # long: judge, generator, delta
    hspp: pd.DataFrame  # judge, hspp_r_self, hspp_r_fam
    length_curves: pd.DataFrame  # bin, judge, metric, value, lo, hi
    paradigm: pd.DataFrame  # paradigm, hspp_r_self (mean)
    domain: pd.DataFrame  # domain, hspp_r_self (mean)
    diagnostics: pd.DataFrame  # metric, value


def _relation(judge_fam: str, gen_fam: str, judge: str, gen: str) -> str:
    if judge == gen:
        return "self"
    if judge_fam == gen_fam:
        return "family"
    return "other"


def build_demo_dataset() -> DemoDataset:
    rng = np.random.default_rng(DEMO_SEED)
    models = [m for m, _ in ROSTER]
    fam = dict(ROSTER)

    # --- Overestimation matrix O(J, G) with planted self/family elevation. ---
    base = 0.18
    rows = []
    for j in models:
        for g in models:
            rel = _relation(fam[j], fam[g], j, g)
            bump = {"self": 0.22, "family": 0.10, "other": 0.0}[rel]
            o = float(np.clip(base + bump + rng.normal(0, 0.02), 0.01, 0.95))
            rows.append({"judge": j, "generator": g, "relation": rel, "O": round(o, 4)})
    overest = pd.DataFrame(rows)

    # --- HSPP-R per judge from the O matrix (uses the real metric functions). ---
    hspp_rows = []
    for j in models:
        sub = overest[overest.judge == j]
        o_self = float(sub[sub.relation == "self"].O.iloc[0])
        o_fam = list(sub[sub.relation == "family"].O)
        o_other = list(sub[sub.relation == "other"].O)
        hspp_rows.append(
            {
                "judge": j,
                "family": fam[j],
                "hspp_r_self": round(hspp_r_self(o_self, o_other) or float("nan"), 3),
                "hspp_r_fam": round(hspp_r_family(o_fam, o_other) or float("nan"), 3),
            }
        )
    hspp = pd.DataFrame(hspp_rows)

    # --- Centered score-delta matrix (practical skew). ---
    delta_rows = []
    for j in models:
        raw = {}
        for g in models:
            rel = _relation(fam[j], fam[g], j, g)
            skew = {"self": 0.08, "family": 0.03, "other": -0.02}[rel]
            raw[g] = skew + rng.normal(0, 0.01)
        m = np.mean(list(raw.values()))
        for g in models:
            delta_rows.append({"judge": j, "generator": g, "delta": round(raw[g] - m, 4)})
    delta = pd.DataFrame(delta_rows)

    # --- Length curves: HSPP-R, recognition accuracy, attribution F1 vs length. ---
    curve_rows = []
    x = np.array(LENGTH_BINS, dtype=float)
    lx = (np.log(x) - np.log(x.min())) / (np.log(x.max()) - np.log(x.min()))  # 0..1
    for j in models:
        j_gain = 1.0 + 0.25 * (models.index(j) % 3)  # slight per-judge variation
        # Recognition accuracy: 0.5 -> ~0.9, sigmoid in log-length.
        recog = 0.5 + 0.42 / (1 + np.exp(-(lx - 0.45) * 9))
        # Attribution F1 (TF-IDF baseline): 1/6 chance -> ~0.8.
        attrib = (1 / 6) + (0.8 - 1 / 6) / (1 + np.exp(-(lx - 0.5) * 8))
        # HSPP-R: 1.0 -> ~1.9, tracks recognition (H1 shape) with a small lag.
        hspp_curve = 1.0 + 0.9 * j_gain / (1 + np.exp(-(lx - 0.55) * 8))
        hspp_curve = 1.0 + (hspp_curve - 1.0) / (hspp_curve.max() - 1.0 + 1e-9) * 0.9
        for metric, vals, null in [
            ("recognition_accuracy", recog, 0.5),
            ("attribution_f1", attrib, 1 / 6),
            ("hspp_r_self", hspp_curve, 1.0),
        ]:
            for xi, v in zip(LENGTH_BINS, vals, strict=True):
                half = 0.06 if metric != "hspp_r_self" else 0.12
                jitter = rng.normal(0, 0.01)
                val = float(v + jitter)
                curve_rows.append(
                    {
                        "bin": xi,
                        "judge": j,
                        "metric": metric,
                        "value": round(val, 4),
                        "lo": round(val - half, 4),
                        "hi": round(val + half, 4),
                        "null": null,
                    }
                )
    length_curves = pd.DataFrame(curve_rows)

    # --- Paradigm + domain breakdowns (means of HSPP-R_self). ---
    paradigm = pd.DataFrame(
        {
            "paradigm": PARADIGMS,
            "hspp_r_self": [round(float(hspp.hspp_r_self.mean()) + d, 3) for d in (0.18, -0.05)],
        }
    )
    domain = pd.DataFrame(
        {
            "domain": DOMAINS,
            "hspp_r_self": [
                round(float(hspp.hspp_r_self.mean()) + d, 3) for d in (-0.15, 0.05, 0.22)
            ],
        }
    )

    diagnostics = pd.DataFrame(
        {
            "metric": [
                "position_bias_rate",
                "length_compliance_median_pct",
                "repeatability_krippendorff_alpha",
                "inter_judge_kappa",
            ],
            "value": [0.14, 0.93, 0.86, 0.61],
        }
    )

    return DemoDataset(
        overestimation=overest,
        delta_matrix=delta,
        hspp=hspp,
        length_curves=length_curves,
        paradigm=paradigm,
        domain=domain,
        diagnostics=diagnostics,
    )
