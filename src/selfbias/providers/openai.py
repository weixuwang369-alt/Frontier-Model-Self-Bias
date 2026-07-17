"""OpenAI adapter. Not exercised in Phase 0 (mock-only, no real calls).

SDK imported lazily; see the Anthropic adapter for the same pattern and rationale.
"""

from __future__ import annotations

from ..schemas import LLMRequest, LLMResponse, Provider, Usage
from .base import Provider as BaseProvider
from .base import _Retryable


class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("OpenAIProvider requires an API key")
        self._api_key = api_key
        self._client = None

    @property
    def name(self) -> str:
        return Provider.openai.value

    def _ensure_client(self):  # pragma: no cover - requires SDK + key (Phase 1+)
        if self._client is None:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "openai SDK not installed. Install the 'providers' extra before "
                    "running real OpenAI calls."
                ) from exc
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def _call(self, request: LLMRequest) -> LLMResponse:  # pragma: no cover - Phase 1+
        client = self._ensure_client()
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        kwargs = dict(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            seed=request.seed,
        )
        # Nudge JSON output when a structured verdict is expected; the universal parser
        # (selfbias.structured) still validates it.
        if request.response_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise _Retryable(str(exc)) from exc
        text = resp.choices[0].message.content or ""
        usage = Usage(
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
        )
        return LLMResponse(
            provider=Provider.openai,
            model=request.model,
            text=text,
            usage=usage,
            raw_response=resp.model_dump() if hasattr(resp, "model_dump") else {},
        )
