from __future__ import annotations

import pytest
from pydantic import ValidationError

from selfbias.config import ExperimentConfig
from selfbias.schemas import (
    Constraint,
    Generation,
    ReferenceSource,
    Task,
    stable_hash,
)


def test_stable_hash_deterministic_and_order_sensitive():
    assert stable_hash("a", 1) == stable_hash("a", 1)
    assert stable_hash("a", 1) != stable_hash(1, "a")
    assert len(stable_hash("x")) == 16
    assert len(stable_hash("x", length=8)) == 8


def test_task_and_generation_ids_are_deterministic():
    t1 = Task.make_id("creative_writing", "prompt")
    t2 = Task.make_id("creative_writing", "prompt")
    assert t1 == t2
    g = Generation.make_id(t1, "model-x", 100, 7)
    assert g == Generation.make_id(t1, "model-x", 100, 7)
    assert g != Generation.make_id(t1, "model-x", 250, 7)


def test_programmatic_task_requires_constraint():
    with pytest.raises(ValidationError):
        Task(
            task_id="t",
            domain="verifiable_if",
            prompt="p",
            reference_source=ReferenceSource.programmatic,
        )
    # With a constraint it is valid.
    Task(
        task_id="t",
        domain="verifiable_if",
        prompt="p",
        reference_source=ReferenceSource.programmatic,
        constraints=[
            Constraint(
                constraint_id="c", text="x", checker="must_include", params={"substring": "x"}
            )
        ],
    )


def test_roster_requires_unique_model_strings(config_dict):
    config_dict["roster"][1]["model"] = config_dict["roster"][0]["model"]
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(config_dict)


def test_judging_integrity_rules_enforced(config_dict):
    bad = dict(config_dict)
    bad["judging"] = dict(config_dict["judging"], order_swap_pwc=False)
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(bad)

    bad2 = dict(config_dict)
    bad2["judging"] = dict(config_dict["judging"], judge_isolation=False)
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(bad2)


def test_run_id_changes_with_config(mock_config, config_dict):
    other = ExperimentConfig.model_validate(
        {**config_dict, "run": {**config_dict["run"], "seed": 999}}
    )
    assert mock_config.run_id() != other.run_id()
    assert mock_config.run_id() == mock_config.run_id()
