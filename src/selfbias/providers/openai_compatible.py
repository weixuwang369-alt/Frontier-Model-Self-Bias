"""OpenAI-compatible adapter - the flexibility workhorse.

Points the OpenAI SDK at a configurable ``base_url``, so any OpenAI-style endpoint works
with no new code: local **Ollama** / **vLLM** / **LM Studio**, or hosted **OpenRouter** /
**Together** / **Groq** / **Fireworks**, serving open models like **Qwen**, **Llama**,
**Mistral**, **DeepSeek**. A key is optional (local servers often need none).

Not exercised in Phase 0 (mock only); the SDK is imported lazily.
"""

from __future__ import annotations

from ..schemas import LLMRequest, LLMResponse, Provider, Usage
from .base import Provider as BaseProvider
from .base import _Retryable


class OpenAICompatibleProvider(BaseProvider):
    def __init__(self, api_key: str | None, base_url: str) -> None:
        if not base_url:
            raise ValueError("OpenAICompatibleProvider requires a base_url")
        # Many local servers accept any non-empty key; use a placeholder if none given.
        self._api_key = api_key or "not-needed"
        self._base_url = base_url
        self._client = None

    @property
    def name(self) -> str:
        return Provider.openai_compatible.value

    def _ensure_client(self):  # pragma: no cover - requires SDK (Phase 1+)
        if self._client is None:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "openai SDK not installed. Install the 'providers' extra before "
                    "running OpenAI-compatible calls."
                ) from exc
            self._client = openai.OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def _call(self, request: LLMRequest) -> LLMResponse:  # pragma: no cover - Phase 1+
        client = self._ensure_client()
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        kwargs = dict(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        if request.seed is not None:
            kwargs["seed"] = request.seed
        # Ask for structured output when the endpoint supports it; harmless otherwise.
        if request.response_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - translate to retryable
            raise _Retryable(str(exc)) from exc
        text = resp.choices[0].message.content or ""
        usage = Usage(
            input_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
        )
        return LLMResponse(
            provider=Provider.openai_compatible,
            model=request.model,
            text=text,
            usage=usage,
            raw_response=resp.model_dump() if hasattr(resp, "model_dump") else {},
        )
