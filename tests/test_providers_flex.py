"""Flexible-provider architecture (Phase 0.5, W1) - config validation + key resolution.

Still mock-only; no real adapters are constructed or called.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from selfbias.config import ExperimentConfig, RosterModel, Settings
from selfbias.schemas import Provider


def _cfg(config_dict, roster):
    return ExperimentConfig.model_validate({**config_dict, "roster": roster})


def test_openai_compatible_roster_validates(config_dict):
    # An open model (e.g. Qwen via Ollama) with a custom family + endpoint + key env.
    roster = [
        {"slot": "claude", "provider": "anthropic", "model": "claude-x", "family": "anthropic"},
        {
            "slot": "qwen",
            "provider": "openai_compatible",
            "model": "qwen2.5-72b",
            "family": "qwen",
            "base_url": "http://localhost:11434/v1",
            "api_key_env": "OLLAMA_KEY",
        },
    ]
    cfg = _cfg(config_dict, roster)
    q = cfg.roster[1]
    assert q.provider == Provider.openai_compatible
    assert q.family == "qwen"
    assert q.key_env() == "OLLAMA_KEY"


def test_openai_compatible_requires_base_url(config_dict):
    roster = [
        {"slot": "a", "provider": "anthropic", "model": "claude-x", "family": "anthropic"},
        {"slot": "b", "provider": "openai_compatible", "model": "qwen", "family": "qwen"},
    ]
    with pytest.raises(ValidationError):
        _cfg(config_dict, roster)


def test_roster_minimum_two(config_dict):
    roster = [{"slot": "only", "provider": "mock", "model": "m1", "family": "x"}]
    with pytest.raises(ValidationError):
        _cfg(config_dict, roster)


def test_default_key_env_by_provider():
    m = RosterModel(slot="a", provider=Provider.anthropic, model="claude-x", family="anthropic")
    assert m.key_env() == "ANTHROPIC_API_KEY"
    m2 = RosterModel(
        slot="b", provider=Provider.openai, model="gpt", family="openai", api_key_env="MY_KEY"
    )
    assert m2.key_env() == "MY_KEY"


def test_resolve_key_reads_named_env_var(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-router-123")
    s = Settings()
    assert s.resolve_key(Provider.openai_compatible, "OPENROUTER_API_KEY") == "sk-router-123"
    # mock never needs a key; unknown env resolves to None (keyless local is allowed).
    assert s.resolve_key(Provider.mock) == "mock-key"
    assert s.resolve_key(Provider.openai_compatible, "DOES_NOT_EXIST") is None


def test_key_status_reports_per_model(config_dict, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    roster = [
        {"slot": "a", "provider": "anthropic", "model": "claude-x", "family": "anthropic"},
        {
            "slot": "q",
            "provider": "openai_compatible",
            "model": "qwen",
            "family": "qwen",
            "base_url": "http://localhost:11434/v1",
        },
    ]
    cfg = _cfg(config_dict, roster)
    status = {s["model"]: s for s in Settings().key_status(cfg.roster)}
    assert status["claude-x"]["required"] and status["claude-x"]["present"]
    # openai_compatible is not "required" (local servers may need no key).
    assert status["qwen"]["required"] is False
