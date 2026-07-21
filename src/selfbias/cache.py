"""Content-addressed response cache.

Cache key = SHA-256 of the canonical-JSON request payload (provider, model, messages,
temperature, max_tokens, seed, response_schema). A re-run of an identical experiment
must cost ~$0, so every pipeline call goes through :class:`CachingProvider`, which
checks the cache before spending. Cache hits are flagged on the response and tallied in
the run manifest.

Stored bodies are the full raw response (never discarded) so metrics are recomputable
from disk without re-spending budget.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .providers.base import Provider as BaseProvider
from .schemas import LLMRequest, LLMResponse


def cache_key(request: LLMRequest) -> str:
    payload = json.dumps(
        request.cache_payload(), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    """Filesystem cache under ``data/cache/`` (one JSON file per key)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Shard by first two hex chars to keep directories small.
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> LLMResponse | None:
        path = self._path(key)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        resp = LLMResponse.model_validate(data)
        resp.cache_hit = True
        return resp

    def put(self, key: str, response: LLMResponse) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Persist with cache_hit=False; the flag is set to True only on read-back.
        to_store = response.model_copy(update={"cache_hit": False})
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(to_store.model_dump(mode="json"), fh, ensure_ascii=False)
        tmp.replace(path)  # atomic within a filesystem

    def has(self, key: str) -> bool:
        return self._path(key).exists()


class CachingProvider(BaseProvider):
    """Wraps any provider so calls are cache-first. Delegates transport to ``inner``."""

    def __init__(self, inner: BaseProvider, cache: ResponseCache) -> None:
        self.inner = inner
        self.cache = cache

    @property
    def name(self) -> str:
        return self.inner.name

    def _call(self, request: LLMRequest) -> LLMResponse:  # pragma: no cover - unused
        # CachingProvider overrides generate(); _call is never reached.
        raise NotImplementedError

    def generate(self, request: LLMRequest) -> LLMResponse:
        key = cache_key(request)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        response = self.inner.generate(request)
        self.cache.put(key, response)
        # Return a fresh read so callers see the same object shape as a cache hit,
        # but with cache_hit=False (this was a live/mock call that cost budget).
        return response
