"""LLM-generated prompt mode (Phase 0.5, W4). Mock-only."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from selfbias.config import ExperimentConfig
from selfbias.costs import counts_by_stage, estimate_cost
from selfbias.pipeline import Pipeline
from selfbias.schemas import Task
from selfbias.storage import JsonlStore, default_data_paths
from selfbias.tasks import build_tasks


def _llm_config(config_dict, **prompts) -> ExperimentConfig:
    cfg = copy.deepcopy(config_dict)
    cfg["prompts"] = {"source": "llm_generated", "n_per_domain": 3, **prompts}
    return ExperimentConfig.model_validate(cfg)


def test_placeholders_are_count_accurate_and_add_promptgen_stage(config_dict, pricing):
    config = _llm_config(config_dict)
    tasks = build_tasks(config)  # estimate path → placeholders
    assert len(tasks) == 3 * len(config.domains)
    counts = counts_by_stage(config, tasks)
    assert counts["promptgen"] == len(config.domains)
    est = estimate_cost(config, tasks, pricing)
    assert "promptgen" in est.by_stage
    assert est.by_stage["promptgen"].calls == len(config.domains)


def test_end_to_end_llm_generated_run(config_dict, tmp_path):
    config = _llm_config(config_dict)
    pipe = Pipeline(config, data_root=str(tmp_path / "data"), pricing_path="config/pricing.yaml")
    result = pipe.run()

    assert result.status.value == "completed"
    assert result.n_tasks == 3 * len(config.domains)
    assert result.manifest.stages["promptgen"].calls_done == len(config.domains)

    tasks = JsonlStore(default_data_paths(tmp_path / "data").tasks / "tasks.jsonl").read_all(Task)
    assert len(tasks) == 3 * len(config.domains)
    assert all(t.prompt.strip() for t in tasks)
    # Downstream stages ran against the generated tasks.
    assert result.n_generations >= 3 * len(config.domains) * len(config.roster)


def test_llm_generated_resume_is_idempotent(config_dict, tmp_path):
    config = _llm_config(config_dict)
    r1 = Pipeline(
        config, data_root=str(tmp_path / "data"), pricing_path="config/pricing.yaml"
    ).run()
    r2 = Pipeline(
        config, data_root=str(tmp_path / "data"), pricing_path="config/pricing.yaml"
    ).run()
    assert (r1.n_generations, r1.n_judgments, r1.n_probes) == (
        r2.n_generations,
        r2.n_judgments,
        r2.n_probes,
    )


def test_generator_model_must_be_in_roster(config_dict):
    with pytest.raises(ValidationError):
        _llm_config(config_dict, generator_model="not-a-roster-model")


def test_default_generator_is_first_roster(config_dict):
    config = _llm_config(config_dict)
    assert config.prompt_generator().model == config.roster[0].model
