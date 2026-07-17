"""Attributability curve - TF-IDF baseline (METRICS §5.1).

Per length bin, a TF-IDF + logistic-regression classifier predicts which model wrote a
text, with **prompt-grouped** train/test splits (so the model can't cheat via shared
prompt wording). We report 6-way macro-F1 (chance = 1/#models) with a prompt-level
bootstrap CI over out-of-fold predictions. This is the Phase-2 baseline; the Phase-3
fingerprint classifier plugs into the same curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from ..schemas import Generation
from ..storage import DataPaths, JsonlStore
from .bootstrap import CI, bootstrap_over_prompts
from .onset import BinCI


def attribution_df(paths: DataPaths) -> pd.DataFrame:
    """Controlled-length generations as (model, prompt, bin, text) for attribution."""

    gens = JsonlStore(paths.generations / "generations.jsonl").read_all(Generation)
    rows = [
        {"model": g.model, "prompt": g.task_id, "bin": g.target_tokens, "text": g.text}
        for g in gens
        if g.truncation_of is None and g.text.strip()
    ]
    return pd.DataFrame(rows)


def _oof_predictions(d: pd.DataFrame) -> pd.DataFrame | None:
    """Out-of-fold predictions with prompt-grouped folds; None if too little data."""

    n_groups = d["prompt"].nunique()
    classes = sorted(d["model"].unique())
    if n_groups < 2 or len(classes) < 2:
        return None
    n_splits = min(5, n_groups)
    X = d["text"].to_numpy()
    y = d["model"].to_numpy()
    groups = d["prompt"].to_numpy()

    preds = np.empty(len(d), dtype=object)
    gkf = GroupKFold(n_splits=n_splits)
    for train, test in gkf.split(X, y, groups=groups):
        clf = make_pipeline(
            TfidfVectorizer(min_df=1, ngram_range=(1, 2)),
            LogisticRegression(max_iter=1000),
        )
        clf.fit(X[train], y[train])
        preds[test] = clf.predict(X[test])
    return pd.DataFrame({"prompt": groups, "true": y, "pred": preds})


def attribution_curve(
    df: pd.DataFrame,
    bins: list[int],
    n_models: int,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> list[BinCI]:
    out: list[BinCI] = []
    for b in bins:
        d = df[df["bin"] == b] if not df.empty else df
        oof = _oof_predictions(d) if not d.empty else None
        if oof is None:
            out.append(BinCI(bin=b, ci=CI(float("nan"), float("nan"), float("nan"), 0)))
            continue
        labels = sorted(pd.unique(oof["true"]))

        def macro_f1(s: pd.DataFrame, labels=labels) -> float:
            return float(
                f1_score(s["true"], s["pred"], labels=labels, average="macro", zero_division=0)
            )

        ci = bootstrap_over_prompts(oof, macro_f1, n_boot=n_boot, seed=seed + b)
        out.append(BinCI(bin=b, ci=ci))
    return out
