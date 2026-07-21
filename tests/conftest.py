"""Shared fixtures. Tests use ONLY the mock provider - never a real API."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from selfbias.config import ExperimentConfig, Pricing, load_pricing

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Isolate tests from the machine's real credentials.

    ``Settings()`` loads ``.env`` into the process environment, so a developer with real
    keys on disk would otherwise flip key-presence assertions. Neutralize the .env load
    and clear known provider keys so these tests are deterministic everywhere.
    """

    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    for var in (
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "DASHSCOPE_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

# A tiny, all-mock config: 1 prompt/domain, 2 bins, 4 models across 2 families.
MOCK_CONFIG_DICT = {
    "run": {"name": "test-run", "seed": 123, "budget_usd": 100.0, "phase": 0},
    "roster": [
        {
            "slot": "a_frontier",
            "provider": "mock",
            "model": "mock-a-frontier",
            "family": "anthropic",
        },
        {"slot": "a_small", "provider": "mock", "model": "mock-a-small", "family": "anthropic"},
        {"slot": "o_frontier", "provider": "mock", "model": "mock-o-frontier", "family": "openai"},
        {"slot": "o_small", "provider": "mock", "model": "mock-o-small", "family": "openai"},
    ],
    "domains": [
        {"name": "verifiable_if", "n_prompts": 1, "reference": "programmatic"},
        {
            "name": "creative_writing",
            "n_prompts": 1,
            "reference": "ensemble",
            "rubrics": {
                "source": "llm_drafted_human_reviewed",
                "per_task_min": 4,
                "per_task_max": 4,
            },
        },
    ],
    "lengths": {
        "target_bins_tokens": [50, 100],
        "tolerance_pct": 20,
        "max_retries": 3,
        "truncation_series": True,
    },
    "generation": {"temperature": 0.7, "persist_raw": True},
    "judging": {
        "paradigms": ["pwc", "rubric"],
        "disclosure_arms": ["anonymous"],
        "order_swap_pwc": True,
        "temperature": 0.0,
        "elicit_confidence": True,
        "judge_isolation": True,
    },
    "probes": {
        "pairwise_recognition": True,
        "single_recognition": False,
        "on_truncation_series": True,
    },
    "diagnostics": {"repeatability_sample_pct": 2, "repeatability_n": 5},
    "analysis": {
        "bootstrap_iterations": 1000,
        "attribution_baseline": "tfidf",
        "mixed_effects": True,
    },
    "fingerprint": {"enabled": False},
}


@pytest.fixture
def mock_config() -> ExperimentConfig:
    return ExperimentConfig.model_validate(copy.deepcopy(MOCK_CONFIG_DICT))


@pytest.fixture
def config_dict() -> dict:
    return copy.deepcopy(MOCK_CONFIG_DICT)


@pytest.fixture
def pricing() -> Pricing:
    return load_pricing(REPO_ROOT / "config" / "pricing.yaml")
