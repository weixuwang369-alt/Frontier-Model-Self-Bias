"""Excel prompt import/export (Phase 0.5, W3). Mock-only."""

from __future__ import annotations

import copy

import pytest

from selfbias.config import ExperimentConfig
from selfbias.excel import ExcelImportError, export_results, import_tasks, write_template
from selfbias.pipeline import Pipeline
from selfbias.schemas import ReferenceSource


def test_template_roundtrips_into_tasks(tmp_path):
    xlsx = write_template(tmp_path / "prompts.xlsx")
    tasks = import_tasks(xlsx)
    assert len(tasks) == 2
    by_ref = {t.reference_source: t for t in tasks}

    obj = by_ref[ReferenceSource.programmatic]
    checkers = {c.checker for c in obj.constraints}
    assert "must_include" in checkers and "max_words" in checkers
    assert any(c.params.get("substring") == "firmware" for c in obj.constraints)

    subj = by_ref[ReferenceSource.ensemble]
    assert subj.rubrics
    # The "| negative" rubric parsed as negative polarity.
    assert any(r.polarity.value == "negative" for r in subj.rubrics)


def test_csv_template_also_works(tmp_path):
    csv = write_template(tmp_path / "prompts.csv")
    tasks = import_tasks(csv)
    assert len(tasks) == 2


def test_missing_required_column_errors(tmp_path):
    import pandas as pd

    p = tmp_path / "bad.xlsx"
    pd.DataFrame({"prompt": ["hi"]}).to_excel(p, index=False)  # no 'domain'
    with pytest.raises(ExcelImportError, match="domain"):
        import_tasks(p)


def test_bad_checker_errors(tmp_path):
    import pandas as pd

    p = tmp_path / "bad2.xlsx"
    pd.DataFrame(
        {"prompt": ["x"], "domain": ["d"], "reference": ["programmatic"], "constraint": ["nope:1"]}
    ).to_excel(p, index=False)
    with pytest.raises(ExcelImportError, match="unknown checker"):
        import_tasks(p)


def test_end_to_end_run_from_excel(config_dict, tmp_path):
    xlsx = write_template(tmp_path / "prompts.xlsx")
    cfg = copy.deepcopy(config_dict)
    cfg["prompts"] = {"source": "excel", "excel_path": str(xlsx)}
    config = ExperimentConfig.model_validate(cfg)

    pipe = Pipeline(config, data_root=str(tmp_path / "data"), pricing_path="config/pricing.yaml")
    result = pipe.run()
    assert result.status.value == "completed"
    # 2 prompts from the sheet × 4 models × 2 bins controlled generations.
    assert result.n_tasks == 2
    assert result.n_generations >= 2 * len(config.roster) * len(config.lengths.target_bins_tokens)

    # Export produces a workbook with the run's rows.
    out = export_results(tmp_path / "data", tmp_path / "results.xlsx", run_id=config.run_id())
    assert out.exists()
    import pandas as pd

    sheets = pd.read_excel(out, sheet_name=None)
    assert "judgments" in sheets and len(sheets["judgments"]) == result.n_judgments


def test_excel_source_requires_path(config_dict):
    from pydantic import ValidationError

    cfg = copy.deepcopy(config_dict)
    cfg["prompts"] = {"source": "excel"}  # no excel_path
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(cfg)
