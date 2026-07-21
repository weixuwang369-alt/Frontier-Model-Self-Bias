"""Provider registry / factory.

``build_provider`` returns a transport provider for a given :class:`Provider` enum,
reading keys from :class:`Settings`. The mock provider needs no key and is always
available (Phase 0, tests, and the dashboard's keyless demo mode).

Pipeline code should obtain a *cache-first* provider via :func:`build_providers`, which
wraps each transport in :class:`selfbias.cache.CachingProvider`.
"""

from __future__ import annotations

from ..schemas import Provider as ProviderEnum
from .base import Provider, ProviderError
from .mock import MockProvider

__all__ = ["Provider", "ProviderError", "MockProvider", "build_provider", "build_providers"]


def build_provider(
    provider: ProviderEnum, api_key: str | None, base_url: str | None = None
) -> Provider:
    """Construct a single transport provider. Vendor adapters import SDKs lazily."""

    if provider == ProviderEnum.mock:
        return MockProvider()
    if provider == ProviderEnum.anthropic:
        from .anthropic import AnthropicProvider

        return AnthropicProvider(api_key or "")
    if provider == ProviderEnum.google:
        from .google import GoogleProvider

        return GoogleProvider(api_key or "")
    if provider == ProviderEnum.openai:
        from .openai import OpenAIProvider

        return OpenAIProvider(api_key or "")
    if provider == ProviderEnum.openai_compatible:
        from .openai_compatible import OpenAICompatibleProvider

        return OpenAICompatibleProvider(api_key, base_url or "")
    raise ValueError(f"unknown provider: {provider}")


def build_providers(roster, settings, cache):
    """Return ``{model_string: CachingProvider}`` for a roster.

    Keyed per model (not per provider kind) because each roster entry can carry its own
    ``base_url`` and key env var. ``settings`` resolves keys; ``cache`` is a
    :class:`selfbias.cache.ResponseCache`. Local import avoids a circular import.
    """

    from ..cache import CachingProvider

    out: dict[str, Provider] = {}
    for m in roster:
        if m.model in out:
            continue
        key = settings.resolve_key(m.provider, m.key_env())
        transport = build_provider(m.provider, key, m.base_url)
        out[m.model] = CachingProvider(transport, cache)
    return out
