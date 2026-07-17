"""Mechanism regression v1 (RESEARCH_PLAN §5.5, METRICS §8).

Logistic regression of per-rubric overestimation on log realized length, authorship
relation (self / family / other), and domain, with **prompt-clustered** robust standard
errors. Reports odds ratios with 95% CIs - the self/family coefficients are the
self-preference effect.

This is the v1 (fixed-effects + clustered SE). The full mixed-effects GLMM with random
intercepts for prompt and judge is the Phase-2/3 refinement (logged in DECISIONS.md).
Fingerprint density (H3) enters in Phase 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def mechanism_regression(obs: pd.DataFrame) -> dict:
    """Fit the v1 overestimation model. Returns odds ratios or a status string."""

    d = obs[obs["eligible"]].copy()
    if d.empty:
        return {"status": "no eligible observations"}
    if d["overest"].nunique() < 2:
        return {"status": "no variation in overestimation (all 0 or all 1)"}

    d["log_len"] = np.log(d["realized_len"].clip(lower=1))
    formula = "overest ~ log_len + C(relation, Treatment('other'))"
    if d["domain"].nunique() > 1:
        formula += " + C(domain)"

    try:
        model = smf.logit(formula, data=d)
        res = model.fit(
            disp=False,
            maxiter=200,
            cov_type="cluster",
            cov_kwds={"groups": d["prompt"]},
        )
    except Exception as exc:  # noqa: BLE001 - separation / non-convergence on sparse data
        return {"status": f"fit failed: {type(exc).__name__}: {str(exc)[:120]}"}

    params = res.params
    conf = res.conf_int()
    terms: dict[str, dict] = {}
    for name in params.index:
        if name == "Intercept":
            continue
        label = (
            name.replace("C(relation, Treatment('other'))[T.", "relation=")
            .replace("C(domain)[T.", "domain=")
            .replace("]", "")
        )
        terms[label] = {
            "odds_ratio": round(float(np.exp(params[name])), 4),
            "ci_lo": round(float(np.exp(conf.loc[name, 0])), 4),
            "ci_hi": round(float(np.exp(conf.loc[name, 1])), 4),
            "p_value": round(float(res.pvalues[name]), 4),
        }
    return {
        "status": "ok",
        "n_obs": int(res.nobs),
        "n_prompts": int(d["prompt"].nunique()),
        "terms": terms,
    }
