"""Provider abstraction.

All LLM calls in the pipeline go through a :class:`Provider`. Adapters handle auth from
env, retries with exponential backoff, rate-limit handling, and usage extraction. Vendor
SDKs are imported lazily inside each adapter so Phase 0 (mock-only) needs none of them.

The cache lives *above* this layer (see :class:`selfbias.cache.ResponseCache` and
:class:`CachingProvider`) so every adapter can stay a thin transport with no cache logic.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from ..schemas import LLMRequest, LLMResponse


class ProviderError(RuntimeError):
    """Raised when a provider call fails after exhausting retries."""


class Provider(ABC):
    """Common interface: ``generate(request) -> Response``."""

    #: max attempts on transient/rate-limit errors
    max_retries: int = 5
    #: base seconds for exponential backoff (overridable in tests)
    backoff_base: float = 1.0

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider enum value, e.g. ``"anthropic"`` / ``"mock"``."""

    @abstractmethod
    def _call(self, request: LLMRequest) -> LLMResponse:
        """Single transport call. Adapters implement auth + parsing here."""

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Call ``_call`` with exponential backoff on transient failures."""

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._call(request)
            except _Retryable as exc:  # pragma: no cover - real providers only
                last_exc = exc
                self._sleep(self.backoff_base * (2**attempt))
        raise ProviderError(
            f"{self.name} call failed after {self.max_retries} attempts"
        ) from last_exc

    def _sleep(self, seconds: float) -> None:  # pragma: no cover - patched in tests
        time.sleep(seconds)


class _Retryable(Exception):
    """Internal marker adapters raise to request a backoff-and-retry."""
