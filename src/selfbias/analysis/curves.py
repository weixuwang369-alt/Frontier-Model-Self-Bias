"""Length-sweep curves with bootstrap CIs (METRICS §3, §5.1–5.2).

Per length bin: pooled HSPP-Ratio (self) and pooled self-recognition accuracy, each with
a prompt-level bootstrap CI. These are the curves whose onsets answer RQ2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..schemas import Generation, Probe
from ..storage import DataPaths, JsonlStore
from .bootstrap import CI, bootstrap_over_prompts
from .onset import BinCI


def pooled_hspp_self(obs: pd.DataFrame) -> float:
    """Mean over judges of HSPP-R_self(J) = O(J,J) / mean_{unrelated G} O(J,G).

    ``obs`` should already be the eligible rows (b_ref == -1). NaN if no judge has both a
    self and an unrelated overestimation rate.
    """

    d = obs[obs["eligible"]] if "eligible" in obs.columns else obs
    if d.empty:
        return float("nan")
    o = d.groupby(["judge", "gen_model", "relation"])["overest"].mean().reset_index()
    ratios: list[float] = []
    for _, sub in o.groupby("judge"):
        self_o = sub.loc[sub["relation"] == "self", "overest"]
        other_o = sub.loc[sub["relation"] == "other", "overest"]
        if self_o.empty or other_o.empty:
            continue
        denom = float(other_o.mean())
        if denom <= 0:
            continue
        ratios.append(float(self_o.mean()) / denom)
    return float(np.mean(ratios)) if ratios else float("nan")


def per_judge_hspp(obs: pd.DataFrame, length_bin: int | None = None) -> pd.DataFrame:
    """HSPP-R self and family per judge (point estimates), optionally at one bin."""

    d = obs[obs["eligible"]]
    if length_bin is not None:
        d = d[d["bin"] == length_bin]
    o = d.groupby(["judge", "gen_model", "relation"])["overest"].mean().reset_index()
    rows = []
    for judge, sub in o.groupby("judge"):
        self_o = sub.loc[sub["relation"] == "self", "overest"]
        fam_o = sub.loc[sub["relation"] == "family", "overest"]
        other_o = sub.loc[sub["relation"] == "other", "overest"]
        denom = float(other_o.mean()) if not other_o.empty else float("nan")
        rows.append(
            {
                "judge": judge,
                "hspp_r_self": (float(self_o.mean()) / denom)
                if (not self_o.empty and denom and denom > 0)
                else float("nan"),
                "hspp_r_fam": (float(fam_o.mean()) / denom)
                if (not fam_o.empty and denom and denom > 0)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def hspp_curve(
    obs: pd.DataFrame, bins: list[int], *, n_boot: int = 1000, seed: int = 0
) -> list[BinCI]:
    out: list[BinCI] = []
    for b in bins:
        d = obs[(obs["bin"] == b) & obs["eligible"]]
        ci = bootstrap_over_prompts(d, pooled_hspp_self, n_boot=n_boot, seed=seed + b)
        out.append(BinCI(bin=b, ci=ci))
    return out


# --- recognition -----------------------------------------------------------


def recognition_df(paths: DataPaths) -> pd.DataFrame:
    """Controlled-length recognition probes as (judge, prompt, bin, correct)."""

    gens = {
        g.gen_id: g
        for g in JsonlStore(paths.generations / "generations.jsonl").read_all(Generation)
    }
    probes = JsonlStore(paths.probes / "probes.jsonl").read_all(Probe)
    rows = []
    for p in probes:
        if p.series != "controlled" or not p.subject_gen_ids:
            continue
        own = gens.get(p.subject_gen_ids[0])
        if own is None or p.correct is None:
            continue
        rows.append(
            {
                "judge": p.judge_model,
                "prompt": own.task_id,
                "bin": own.target_tokens,
                "correct": 1 if p.correct else 0,
            }
        )
    return pd.DataFrame(rows)


def _mean_correct(df: pd.DataFrame) -> float:
    return float(df["correct"].mean()) if not df.empty else float("nan")


def recognition_curve(
    probe_df: pd.DataFrame, bins: list[int], *, n_boot: int = 1000, seed: int = 0
) -> list[BinCI]:
    out: list[BinCI] = []
    for b in bins:
        d = probe_df[probe_df["bin"] == b] if not probe_df.empty else probe_df
        ci = (
            bootstrap_over_prompts(d, _mean_correct, n_boot=n_boot, seed=seed + b)
            if not d.empty
            else CI(point=float("nan"), lo=float("nan"), hi=float("nan"), n_prompts=0)
        )
        out.append(BinCI(bin=b, ci=ci))
    return out
