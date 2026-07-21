"""Connection test - verify each roster model's keys/endpoint actually work.

Makes ONE tiny real call per distinct model (max a few tokens), bypassing the cache so
it's a genuine round-trip. This is the "did my keys work?" check the first key-holder
runs before spending on a real sweep. Mock models always pass; models with no key are
reported as such without a call; anything else that fails reports the error verbatim.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

from .config import ExperimentConfig, Settings
from .providers import build_provider
from .schemas import LLMRequest, Message, Provider

# Injectable for tests; defaults to the real per-model transport builder (no cache).
BuildFn = Callable[[Provider, str | None, str | None], object]


@dataclass
class CheckResult:
    slot: str
    model: str
    provider: str
    key_env: str | None
    ok: bool
    latency_ms: float | None = None
    error: str | None = None
    skipped: bool = False  # skipped because a required key was missing


def check_models(
    config: ExperimentConfig,
    settings: Settings | None = None,
    build: BuildFn | None = None,
) -> list[CheckResult]:
    settings = settings or Settings()
    build = build or build_provider

    results: list[CheckResult] = []
    seen: set[str] = set()
    for m in config.roster:
        if m.model in seen:
            continue
        seen.add(m.model)

        key = settings.resolve_key(m.provider, m.key_env())
        needs_key = m.provider not in (Provider.mock, Provider.openai_compatible)
        base = dict(slot=m.slot, model=m.model, provider=m.provider.value, key_env=m.key_env())
        if needs_key and not key:
            results.append(CheckResult(**base, ok=False, skipped=True, error="no API key set"))
            continue

        request = LLMRequest(
            provider=m.provider,
            model=m.model,
            messages=(Message(role="user", content="Reply with the single word: ok"),),
            temperature=0.0,
            max_tokens=5,
            purpose="check",
        )
        try:
            provider = build(m.provider, key, m.base_url)
            t0 = perf_counter()
            resp = provider.generate(request)
            dt = (perf_counter() - t0) * 1000.0
            ok = resp is not None and resp.text is not None
            results.append(
                CheckResult(
                    **base, ok=ok, latency_ms=round(dt, 1), error=None if ok else "empty response"
                )
            )
        except Exception as exc:  # noqa: BLE001 - surface any failure to the user
            results.append(CheckResult(**base, ok=False, error=str(exc)[:200]))
    return results
