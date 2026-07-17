"""W2: fail-loud structured parsing + length-compliance retry, via fake providers."""

from __future__ import annotations

import random

from selfbias.pipeline import Pipeline
from selfbias.providers.base import Provider as BaseProvider
from selfbias.providers.mock import MockProvider, _lorem, _requested_length
from selfbias.schemas import LLMResponse, Provider, Usage
from selfbias.tasks import build_tasks
from selfbias.tokens import approx_tokens


class _BadJudgeProvider(BaseProvider):
    """Valid generations, but every structured (judge/probe) call returns non-JSON."""

    def __init__(self):
        self._mock = MockProvider()

    @property
    def name(self) -> str:
        return "mock"

    def _call(self, request):
        if request.response_schema is None:
            return self._mock._call(request)
        return LLMResponse(
            provider=Provider.mock,
            model=request.model,
            text="I think the first one is better, honestly.",  # prose, not JSON
            usage=Usage(input_tokens=10, output_tokens=10),
        )


class _LengthProvider(BaseProvider):
    """Returns 3x the target on the first attempt, on-target once nudged to retry."""

    def __init__(self):
        self._mock = MockProvider()

    @property
    def name(self) -> str:
        return "mock"

    def _call(self, request):
        if request.response_schema is not None:
            return self._mock._call(request)
        target = _requested_length(request) or 50
        nudged = any("previous attempt" in m.content for m in request.messages)
        realized = target if nudged else target * 3
        text = _lorem(realized, random.Random(1))
        return LLMResponse(
            provider=Provider.mock,
            model=request.model,
            text=text,
            usage=Usage(input_tokens=20, output_tokens=approx_tokens(text)),
        )


def _pipe(config, tmp_path, provider_factory):
    pipe = Pipeline(config, data_root=str(tmp_path / "data"), pricing_path="config/pricing.yaml")
    pipe.providers = {m.model: provider_factory() for m in config.roster}
    return pipe


def test_malformed_judge_output_flagged_and_skipped(mock_config, tmp_path):
    pipe = _pipe(mock_config, tmp_path, _BadJudgeProvider)
    result = pipe.run()
    assert result.status.value == "completed"
    # Generations succeed; judgments and probes are all skipped (never coerced to ties).
    assert result.n_generations > 0
    assert result.n_judgments == 0
    assert result.n_probes == 0
    parse_failures = sum(s.parse_failures for s in result.manifest.stages.values())
    assert parse_failures > 0


def test_length_compliance_retries_until_in_band(mock_config, tmp_path):
    pipe = _pipe(mock_config, tmp_path, _LengthProvider)
    result = pipe.run()
    assert result.status.value == "completed"

    from selfbias.schemas import Generation
    from selfbias.storage import JsonlStore, default_data_paths

    gens = JsonlStore(
        default_data_paths(tmp_path / "data").generations / "generations.jsonl"
    ).read_all(Generation)
    controlled = [g for g in gens if g.truncation_of is None]
    tol = mock_config.lengths.tolerance_pct / 100.0
    for g in controlled:
        assert abs(g.realized_tokens - g.target_tokens) <= g.target_tokens * tol

    # Each controlled generation took exactly two attempts (miss, then in-band).
    tasks = build_tasks(mock_config)
    n_controlled = (
        len(tasks) * len(mock_config.roster) * len(mock_config.lengths.target_bins_tokens)
    )
    assert result.manifest.stages["generate"].calls_done == 2 * n_controlled
