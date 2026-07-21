"""Anthropic adapter. Not exercised in Phase 0 (mock-only, no real calls).

The vendor SDK is imported lazily inside ``_call`` so that Phase 0 and the test suite
need neither the SDK installed nor a key present. Wiring is left concrete enough that
Phase 1 is a matter of filling in message translation, not restructuring.
"""

from __future__ import annotations

from ..schemas import LLMRequest, LLMResponse, Provider, Usage
from .base import Provider as BaseProvider
from .base import _Retryable


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("AnthropicProvider requires an API key")
        self._api_key = api_key
        self._client = None  # lazily constructed on first use

    @property
    def name(self) -> str:
        return Provider.anthropic.value

    def _ensure_client(self):  # pragma: no cover - requires SDK + key (Phase 1+)
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "anthropic SDK not installed. Install the 'providers' extra "
                    "(uv sync --extra providers) before running real Anthropic calls."
                ) from exc
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _call(self, request: LLMRequest) -> LLMResponse:  # pragma: no cover - Phase 1+
        client = self._ensure_client()
        system = "\n".join(m.content for m in request.messages if m.role == "system")
        turns = [
            {"role": m.role, "content": m.content} for m in request.messages if m.role != "system"
        ]
        kwargs: dict = {
            "model": request.model,
            "messages": turns,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        # Only include ``system`` when there is content; the API rejects ``system=None``.
        if system:
            kwargs["system"] = system
        try:
            resp = client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - translate to retryable
            raise _Retryable(str(exc)) from exc
        text = "".join(block.text for block in resp.content if block.type == "text")
        usage = Usage(
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
        return LLMResponse(
            provider=Provider.anthropic,
            model=request.model,
            text=text,
            usage=usage,
            raw_response=resp.model_dump() if hasattr(resp, "model_dump") else {},
        )
