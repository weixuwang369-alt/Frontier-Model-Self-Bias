"""Excel prompt library - import users' own prompts, export results.

Import (tiered schema - only ``prompt`` + ``domain`` are required):

| column        | required | meaning |
|---------------|----------|---------|
| ``prompt``    | yes      | the task text the models respond to |
| ``domain``    | yes      | free-form domain label (groups tasks) |
| ``reference`` | no       | ``ensemble`` (default) or ``programmatic`` |
| ``rubrics``   | no       | ``;``-separated criteria; append `` | negative`` to flip polarity |
| ``constraint``| no       | ``;``-separated ``checker:arg`` for the objective arm |

A row with any ``constraint`` (or ``reference=programmatic``) becomes an objective task
with programmatic ground truth; otherwise it's subjective (ensemble reference). Subjective
rows with no ``rubrics`` get a sensible default rubric set so rubric-based judging still
works out of the box.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .checkers import _REGISTRY as CHECKERS
from .config import ExperimentConfig
from .schemas import (
    Constraint,
    ReferenceSource,
    Rubric,
    RubricPolarity,
    Task,
    stable_hash,
)
from .tasks import _rubrics_for  # reuse the default rubric set for bare subjective rows
from .tokens import approx_tokens

REQUIRED_COLUMNS = ("prompt", "domain")
TEMPLATE_COLUMNS = ("prompt", "domain", "reference", "rubrics", "constraint")

# checker -> (param_name, caster) for parsing "checker:arg" cells.
_CHECKER_ARGS = {
    "must_include": ("substring", str),
    "max_words": ("max_words", int),
    "exact_sentence_count": ("count", int),
}


class ExcelImportError(ValueError):
    """Raised with a clear, row-referenced message when a sheet can't be parsed."""


def _parse_rubrics(cell: str) -> list[Rubric]:
    rubrics: list[Rubric] = []
    for i, chunk in enumerate(str(cell).split(";")):
        text = chunk.strip()
        if not text:
            continue
        polarity = RubricPolarity.positive
        if "|" in text:
            text, tag = (p.strip() for p in text.rsplit("|", 1))
            polarity = (
                RubricPolarity.negative
                if tag.lower().startswith("neg")
                else RubricPolarity.positive
            )
        rubrics.append(
            Rubric(
                rubric_id=stable_hash(text, "rubric", i),
                text=text,
                polarity=polarity,
                weight=1.0,
                n_tokens=approx_tokens(text),
            )
        )
    return rubrics


def _parse_constraints(cell: str, row: int) -> list[Constraint]:
    constraints: list[Constraint] = []
    for chunk in str(cell).split(";"):
        spec = chunk.strip()
        if not spec:
            continue
        if ":" not in spec:
            raise ExcelImportError(
                f"row {row}: constraint '{spec}' must be 'checker:arg' "
                f"(checkers: {', '.join(sorted(CHECKERS))})"
            )
        checker, arg = (p.strip() for p in spec.split(":", 1))
        if checker not in _CHECKER_ARGS:
            raise ExcelImportError(
                f"row {row}: unknown checker '{checker}' "
                f"(available: {', '.join(sorted(_CHECKER_ARGS))})"
            )
        param, caster = _CHECKER_ARGS[checker]
        try:
            value = caster(arg)
        except ValueError as exc:
            raise ExcelImportError(f"row {row}: bad argument for {checker}: '{arg}'") from exc
        constraints.append(
            Constraint(
                constraint_id=stable_hash(spec, row),
                text=spec,
                checker=checker,
                params={param: value},
            )
        )
    return constraints


def _cell(row, key) -> str:
    val = row.get(key)
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    return str(val).strip()


def import_tasks(path: str | Path, config: ExperimentConfig | None = None) -> list[Task]:
    """Read an ``.xlsx`` / ``.csv`` prompt library into validated :class:`Task`s."""

    path = Path(path)
    if not path.exists():
        raise ExcelImportError(f"prompt file not found: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)  # openpyxl engine
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ExcelImportError(
            f"missing required column(s): {missing}. Expected at least {list(REQUIRED_COLUMNS)}."
        )

    tasks: list[Task] = []
    seen: set[str] = set()
    for idx, raw in df.iterrows():
        rownum = int(idx) + 2  # 1-based + header row, matches what users see in Excel
        prompt = _cell(raw, "prompt")
        domain = _cell(raw, "domain")
        if not prompt or not domain:
            continue  # skip blank rows silently
        constraint_cell = _cell(raw, "constraint")
        ref_cell = _cell(raw, "reference").lower()
        is_objective = bool(constraint_cell) or ref_cell == "programmatic"

        task_id = Task.make_id(domain, prompt)
        if task_id in seen:
            continue  # de-dupe identical (domain, prompt)
        seen.add(task_id)

        if is_objective:
            constraints = _parse_constraints(constraint_cell, rownum)
            if not constraints:
                raise ExcelImportError(
                    f"row {rownum}: reference=programmatic needs at least one 'constraint'."
                )
            tasks.append(
                Task(
                    task_id=task_id,
                    domain=domain,
                    prompt=prompt,
                    reference_source=ReferenceSource.programmatic,
                    constraints=constraints,
                )
            )
        else:
            rubric_cell = _cell(raw, "rubrics")
            rubrics = (
                _parse_rubrics(rubric_cell)
                if rubric_cell
                else _default_rubrics(domain, prompt, config)
            )
            tasks.append(
                Task(
                    task_id=task_id,
                    domain=domain,
                    prompt=prompt,
                    reference_source=ReferenceSource.ensemble,
                    rubrics=rubrics,
                )
            )
    if not tasks:
        raise ExcelImportError("no usable rows found (need non-empty 'prompt' and 'domain').")
    return tasks


def _default_rubrics(domain: str, prompt: str, config: ExperimentConfig | None) -> list[Rubric]:
    # Reuse the built-in default rubric set so bare subjective rows still support RB.
    dcfg = None
    if config is not None:
        dcfg = next((d for d in config.domains if d.name == domain), None)
    if dcfg is None:
        from .config import DomainConfig

        dcfg = DomainConfig(name=domain, n_prompts=1, reference=ReferenceSource.ensemble)
    return _rubrics_for(prompt, dcfg)


def write_template(path: str | Path) -> Path:
    """Write a starter prompt-library workbook users can fill in."""

    path = Path(path)
    rows = [
        {
            "prompt": "Write a short scene: a lighthouse keeper receives an unexpected letter.",
            "domain": "creative_writing",
            "reference": "ensemble",
            "rubrics": "Stays on topic; Vivid and concrete; Avoids cliché | negative",
            "constraint": "",
        },
        {
            "prompt": "List steps to reset a home router. Include the word 'firmware'.",
            "domain": "verifiable_if",
            "reference": "programmatic",
            "rubrics": "",
            "constraint": "must_include:firmware; max_words:120",
        },
    ]
    df = pd.DataFrame(rows, columns=list(TEMPLATE_COLUMNS))
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_excel(path, index=False)
    return path


def export_results(
    data_root: str | Path, out_path: str | Path, run_id: str | None = None
) -> Path:
    """Export a run's persisted rows to a multi-sheet workbook (or CSVs for .csv)."""

    from .storage import default_data_paths

    paths = default_data_paths(data_root, run_id=run_id)
    frames = {
        "generations": paths.generations / "generations.jsonl",
        "judgments": paths.judgments / "judgments.jsonl",
        "probes": paths.probes / "probes.jsonl",
    }
    out_path = Path(out_path)
    loaded = {name: pd.read_json(fp, lines=True) for name, fp in frames.items() if fp.exists()}
    if not loaded:
        raise ExcelImportError(f"no run data found under {data_root} to export.")
    if out_path.suffix.lower() == ".csv":
        # One CSV per artifact, suffixed.
        for name, df in loaded.items():
            df.to_csv(out_path.with_name(f"{out_path.stem}_{name}.csv"), index=False)
    else:
        with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
            for name, df in loaded.items():
                # Drop bulky raw_response for readability; it stays on disk.
                df.drop(columns=[c for c in ("raw_response",) if c in df.columns]).to_excel(
                    xl, sheet_name=name, index=False
                )
    return out_path
