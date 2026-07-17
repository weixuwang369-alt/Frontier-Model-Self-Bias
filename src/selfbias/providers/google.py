"""Google (Gemini) adapter. Not exercised in Phase 0 (mock-only, no real calls).

SDK imported lazily; see the Anthropic adapter for the same pattern and rationale.
"""

from __future__ import annotations

from ..schemas import LLMRequest, LLMResponse, Provider, Usage
from .base import Provider as BaseProvider
from .base import _Retryable


class GoogleProvider(BaseProvider):
    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("GoogleProvider requires an API key")
        self._api_key = api_key
        self._client = None

    @property
    def name(self) -> str:
        return Provider.google.value

    def _ensure_client(self):  # pragma: no cover - requires SDK + key (Phase 1+)
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "google-genai SDK not installed. Install the 'providers' extra "
                    "before running real Google calls."
                ) from exc
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _call(self, request: LLMRequest) -> LLMResponse:  # pragma: no cover - Phase 1+
        client = self._ensure_client()
        contents = "\n\n".join(m.content for m in request.messages)
        try:
            resp = client.models.generate_content(
                model=request.model,
                contents=contents,
                config={
                    "temperature": request.temperature,
                    "max_output_tokens": request.max_tokens,
                },
            )
        except Exception as exc:  # noqa: BLE001
            raise _Retryable(str(exc)) from exc
        um = getattr(resp, "usage_metadata", None)
        usage = Usage(
            input_tokens=getattr(um, "prompt_token_count", 0) or 0,
            output_tokens=getattr(um, "candidates_token_count", 0) or 0,
        )
        return LLMResponse(
            provider=Provider.google,
            model=request.model,
            text=resp.text or "",
            usage=usage,
            raw_response={},
        )
