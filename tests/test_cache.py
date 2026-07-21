from __future__ import annotations

from selfbias.cache import CachingProvider, ResponseCache, cache_key
from selfbias.providers.mock import MockProvider
from selfbias.schemas import LLMRequest, Message, Provider


def _req(seed=1, purpose="generate"):
    return LLMRequest(
        provider=Provider.mock,
        model="mock-model",
        messages=(Message(role="user", content="hello world"),),
        temperature=0.0,
        max_tokens=64,
        seed=seed,
        purpose=purpose,
    )


def test_cache_key_ignores_purpose_but_reflects_semantics():
    # purpose is debug-only and excluded from the key...
    assert cache_key(_req(purpose="generate")) == cache_key(_req(purpose="judge"))
    # ...but semantic fields change it.
    assert cache_key(_req(seed=1)) != cache_key(_req(seed=2))


def test_response_cache_roundtrip_sets_hit_flag(tmp_path):
    cache = ResponseCache(tmp_path / "cache")
    provider = MockProvider()
    req = _req()
    resp = provider.generate(req)
    assert resp.cache_hit is False
    key = cache_key(req)
    cache.put(key, resp)
    got = cache.get(key)
    assert got is not None
    assert got.cache_hit is True
    assert got.text == resp.text


def test_caching_provider_serves_second_call_from_cache(tmp_path):
    cache = ResponseCache(tmp_path / "cache")
    caching = CachingProvider(MockProvider(), cache)
    req = _req()
    first = caching.generate(req)
    assert first.cache_hit is False
    second = caching.generate(req)
    assert second.cache_hit is True
    assert second.text == first.text
