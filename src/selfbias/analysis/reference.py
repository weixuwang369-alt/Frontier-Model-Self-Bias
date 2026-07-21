"""Reference outcomes + the rubric-level observation table (METRICS §1–2).

Loads a run's rubric-based judgments from disk and builds one row per
(judge, generation, rubric) with the judge's verdict ``b_J`` and the reference verdict
``b*``:

* objective (verifiable) tasks: ``b*`` is the programmatic constraint check (exact, free);
* subjective tasks: ``b*`` is the majority vote of all judges on that rubric (ties →
  unsatisfied, the conservative choice - METRICS §1.3).

The overestimation observation (Pombal, §2.2) is defined only where ``b* = -1`` (the
generator objectively fails the rubric): ``overest = 1`` iff the judge marked it satisfied.
Everything downstream (HSPP-R, curves, onset, regression) is computed from this table.
"""

from __future__ import annotations

import pandas as pd

from ..checkers import check
from ..schemas import Generation, Judgment, Paradigm, ReferenceSource, Task
from ..storage import DataPaths, JsonlStore


def _relation(judge_model: str, judge_family: str, gen_model: str, gen_family: str) -> str:
    if judge_model == gen_model:
        return "self"
    if judge_family == gen_family:
        return "family"
    return "other"


def build_rb_observations(
    paths: DataPaths, keep_gen_ids: set[str] | None = None
) -> pd.DataFrame:
    """One row per (judge, generation, rubric) with ``b_judge``, ``b_ref``, ``overest``.

    Columns: judge, judge_family, gen_model, gen_family, relation, prompt (task_id),
    domain, disclosure, bin (target_tokens), realized_len, rubric_id, b_judge, b_ref,
    eligible (b_ref == -1), overest (1 if judge said satisfied on an eligible rubric).
    Returns an empty DataFrame if there are no rubric judgments.

    ``keep_gen_ids`` restricts to one run's generations; judgments over other generations
    are skipped via the gens lookup, so no cross-run data leaks in.
    """

    tasks = {t.task_id: t for t in JsonlStore(paths.tasks / "tasks.jsonl").read_all(Task)}
    gens = {
        g.gen_id: g
        for g in JsonlStore(paths.generations / "generations.jsonl").read_all(Generation)
    }
    if keep_gen_ids is not None:
        gens = {gid: g for gid, g in gens.items() if gid in keep_gen_ids}
    judgments = JsonlStore(paths.judgments / "judgments.jsonl").read_all(Judgment)

    rows: list[dict] = []
    for j in judgments:
        if j.paradigm != Paradigm.rubric or not j.subject_gen_ids:
            continue
        gen = gens.get(j.subject_gen_ids[0])
        if gen is None:
            continue
        task = tasks.get(gen.task_id)
        if task is None:
            continue
        for pr in j.per_rubric:
            rows.append(
                {
                    "judge": j.judge_model,
                    "judge_family": j.judge_family,
                    "gen_model": gen.model,
                    "gen_family": gen.family,
                    "gen_id": gen.gen_id,
                    "prompt": gen.task_id,
                    "domain": task.domain,
                    "disclosure": j.disclosure.value,
                    "bin": gen.target_tokens,
                    "realized_len": gen.realized_tokens,
                    "ref_source": task.reference_source.value,
                    "rubric_id": pr.rubric_id,
                    "gen_text": gen.text,
                    "b_judge": 1 if pr.satisfied else -1,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Reference verdicts b*(gen, rubric).
    b_ref = _reference_verdicts(df, tasks, gens)
    df["b_ref"] = df.apply(lambda r: b_ref[(r["gen_id"], r["rubric_id"])], axis=1)
    df["relation"] = df.apply(
        lambda r: _relation(r["judge"], r["judge_family"], r["gen_model"], r["gen_family"]),
        axis=1,
    )
    df["eligible"] = df["b_ref"] == -1
    df["overest"] = ((df["b_ref"] == -1) & (df["b_judge"] == 1)).astype(int)
    return df.drop(columns=["gen_text"])


def _reference_verdicts(
    df: pd.DataFrame, tasks: dict[str, Task], gens: dict[str, Generation]
) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    # Programmatic tasks: check the constraint directly.
    constraints = {}
    for task in tasks.values():
        for c in task.constraints:
            constraints[(task.task_id, c.constraint_id)] = c

    for (gen_id, rubric_id), grp in df.groupby(["gen_id", "rubric_id"]):
        ref_source = grp["ref_source"].iloc[0]
        if ref_source == ReferenceSource.programmatic.value:
            gen = gens[gen_id]
            c = constraints.get((gen.task_id, rubric_id))
            out[(gen_id, rubric_id)] = 1 if (c is not None and check(c, gen.text)) else -1
        else:
            # Subjective: majority vote of judges; tie -> unsatisfied (conservative).
            votes = grp["b_judge"].tolist()
            out[(gen_id, rubric_id)] = 1 if sum(v == 1 for v in votes) > len(votes) / 2 else -1
    return out


def overestimation_matrix(obs: pd.DataFrame, length_bin: int | None = None) -> pd.DataFrame:
    """O_rub(J, G): mean overestimation over eligible rubrics, per (judge, generator).

    Optionally restricted to one length bin. Returns a tidy frame with columns
    judge, gen_model, relation, O, n (eligible-rubric count).
    """

    d = obs[obs["eligible"]]
    if length_bin is not None:
        d = d[d["bin"] == length_bin]
    if d.empty:
        return pd.DataFrame(columns=["judge", "gen_model", "relation", "O", "n"])
    g = (
        d.groupby(["judge", "gen_model", "relation"])
        .agg(O=("overest", "mean"), n=("overest", "size"))
        .reset_index()
    )
    return g
