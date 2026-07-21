"""Connection test (Phase 0.5, W5). Mock-only; no real network."""

from __future__ import annotations

from selfbias.check import check_models
from selfbias.config import ExperimentConfig, Settings
from selfbias.providers.base import Provider as BaseProvider
from selfbias.schemas import Provider


class _FailingProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "boom"

    def _call(self, request):
        raise RuntimeError("401 Unauthorized")


def test_check_all_mock_ok(mock_config):
    results = check_models(mock_config)
    assert results and all(r.ok for r in results)
    assert all(r.latency_ms is not None for r in results)
    # Deduped by model string.
    assert len(results) == len({m.model for m in mock_config.roster})


def test_check_reports_missing_key(config_dict, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = {
        **config_dict,
        "roster": [
            {"slot": "a", "provider": "anthropic", "model": "claude-x", "family": "anthropic"},
            {"slot": "b", "provider": "mock", "model": "mock-b", "family": "other"},
        ],
    }
    config = ExperimentConfig.model_validate(cfg)
    results = {r.model: r for r in check_models(config, Settings())}
    assert results["claude-x"].skipped and not results["claude-x"].ok
    assert results["claude-x"].key_env == "ANTHROPIC_API_KEY"
    assert results["mock-b"].ok  # mock needs no key


def test_check_reports_call_failure(config_dict, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-bad")
    cfg = {
        **config_dict,
        "roster": [
            {"slot": "a", "provider": "anthropic", "model": "claude-x", "family": "anthropic"},
            {"slot": "b", "provider": "mock", "model": "mock-b", "family": "other"},
        ],
    }
    config = ExperimentConfig.model_validate(cfg)

    def _build(provider, key, base_url):
        if provider == Provider.mock:
            from selfbias.providers.mock import MockProvider

            return MockProvider()
        return _FailingProvider()

    results = {r.model: r for r in check_models(config, Settings(), build=_build)}
    assert not results["claude-x"].ok and not results["claude-x"].skipped
    assert "401" in results["claude-x"].error
    assert results["mock-b"].ok
