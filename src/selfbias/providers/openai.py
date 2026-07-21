"""OpenAI adapter.

Targets the Chat Completions API. Newer families (GPT-5+) tightened their parameters:
they require ``max_completion_tokens`` instead of ``max_tokens`` and reject any
``temperature`` other than the default. Rather than hard-code per-model rules that go
stale, the adapter *self-heals*: on a 400 that names an unsupported parameter it drops or
swaps just that parameter and retries. Transient failures (rate limits, timeouts, 5xx)
bubble up as ``_Retryable`` for the base-class backoff; anything else (auth, unknown
model) surfaces verbatim instead of burning five silent retries.
"""

from __future__ import annotations

import re

from ..schemas import LLMRequest, LLMResponse, Provider, Usage
from .base import Provider as BaseProvider
from .base import _Retryable

# Optional params the model may reject; dropping one falls back to the model's default.
_DROPPABLE = ("temperature", "seed", "top_p", "response_format")

# Reasoning models spend hidden "thinking" tokens that also count against
# max_completion_tokens; a tight budget (small length bins) can leave no room to finish,
# returning an empty completion or an output-limit 400. Give every call headroom on top
# of the requested visible length. It's a ceiling, not a target - only tokens actually
# produced are billed, and visible length stays governed by the prompt + length retry.
_REASONING_HEADROOM = 2000
_TOKEN_CEILING = 24000


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
        import openai  # for exception types; module already imported by _ensure_client

        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        kwargs: dict = {
            "model": request.model,
            "messages": messages,
            # GPT-5+ requires max_completion_tokens; _heal swaps it back for older models.
            # Headroom so reasoning tokens don't starve the visible answer on small bins.
            "max_completion_tokens": min(request.max_tokens + _REASONING_HEADROOM, _TOKEN_CEILING),
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.seed is not None:
            kwargs["seed"] = request.seed
        # Nudge JSON output when a structured verdict is expected; selfbias.structured
        # still validates it. (The prompt itself also asks for JSON.)
        if request.response_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        # Retry to drop rejected params or raise the token ceiling; bounded so it ends.
        resp = None
        for _ in range(1 + len(_DROPPABLE) + 3):
            try:
                resp = client.chat.completions.create(**kwargs)
                break
            except openai.BadRequestError as exc:
                if _heal_params(kwargs, exc) or _raise_token_limit(kwargs, exc):
                    continue
                raise  # a genuine bad request - surface it, don't retry
            except (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
            ) as exc:
                raise _Retryable(str(exc)) from exc
        if resp is None:
            raise _Retryable("openai: could not satisfy the model's parameter constraints")

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


def _raise_token_limit(kwargs: dict, exc: Exception) -> bool:
    """Backstop for reasoning models that still exhaust the budget: the API asks for a
    higher limit, so raise the cap (up to the ceiling) and retry. Returns False when the
    error is unrelated or the ceiling is already hit."""

    msg = str(exc).lower()
    if "output limit" not in msg and "higher max_tokens" not in msg:
        return False
    key = "max_completion_tokens" if "max_completion_tokens" in kwargs else "max_tokens"
    cur = kwargs.get(key) or _REASONING_HEADROOM
    new = min(cur * 3, _TOKEN_CEILING)
    if new <= cur:
        return False
    kwargs[key] = new
    return True


def _heal_params(kwargs: dict, exc: Exception) -> bool:
    """Drop or swap the one parameter a 400 flags as unsupported.

    Returns True if ``kwargs`` changed (worth another attempt), False if the error is not
    about a tunable parameter (so the caller should re-raise it).
    """

    # The SDK exposes the offending parameter directly; body fields sit at the top level
    # (``{'message':..., 'param': 'temperature', ...}``), not nested under ``error``.
    param = getattr(exc, "param", None)
    if not param:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            param = body.get("param") or (body.get("error") or {}).get("param")
    if not param:  # last resort: the parameter named after "'param':" in the message
        m = re.search(r"param'?\s*:\s*'([A-Za-z_]+)'", str(exc))
        param = m.group(1) if m else None

    if param == "max_tokens" and "max_tokens" in kwargs:
        kwargs.setdefault("max_completion_tokens", kwargs.pop("max_tokens"))
        return True
    if param == "max_completion_tokens" and "max_completion_tokens" in kwargs:
        kwargs.setdefault("max_tokens", kwargs.pop("max_completion_tokens"))
        return True
    if param in _DROPPABLE and param in kwargs:
        kwargs.pop(param)
        return True
    return False
