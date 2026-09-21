"""Claude as the duck's brain, via the official `anthropic` SDK (optional extra).

Written against `anthropic` 1.x: adaptive thinking is the model's default on Claude Opus 5,
and the request asks for `display: "summarized"` because the default display leaves every
thinking block's text empty, which would make the trace's "what it thought" a blank line.
`display` changes what is shown, never what is thought or billed. A model that rejects the
`thinking` parameter (one older than Claude 4.6) gets one retry without it, and the run goes
on without thinking text. `tool_choice={"type": "any", "disable_parallel_tool_use": True}`
guarantees exactly one tool call per turn, images ride as base64 PNG blocks, and the
assistant's raw content blocks (including thinking) are replayed verbatim on the next turn.
Server-side refusal fallbacks are on by default and drop out automatically if the installed
SDK predates them.
"""

from __future__ import annotations

import base64
import os
import re
from typing import Any

from quackd.agent.providers.base import (
    Exchange,
    ProviderError,
    ProviderNotInstalled,
    ProviderTurn,
    ToolCall,
    Usage,
    picture_parts,
)
from quackd.agent.providers.catalogue import default_model_for

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _image_block(png: bytes) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.standard_b64encode(png).decode("ascii"),
        },
    }


def render_messages(history: list[Exchange]) -> list[dict[str, Any]]:
    """quackd's vendor-neutral history → Messages API `messages`."""
    messages: list[dict[str, Any]] = []
    for ex in history:
        obs = ex.observation
        pictures = picture_parts(obs, _image_block, lambda text: {"type": "text", "text": text})
        if obs.tool_call_id:
            inner: list[dict[str, Any]] = [{"type": "text", "text": obs.text}, *pictures]
            content: list[dict[str, Any]] = [
                {"type": "tool_result", "tool_use_id": obs.tool_call_id, "content": inner}
            ]
        else:
            content = [*pictures, {"type": "text", "text": obs.text}]
        messages.append({"role": "user", "content": content})
        if ex.decision is not None:
            if isinstance(ex.decision.raw, list) and ex.decision.raw:
                blocks = ex.decision.raw  # replay thinking + tool_use blocks unchanged
            else:
                tc = ex.decision.tool_call
                blocks = []
                if ex.decision.text:
                    blocks.append({"type": "text", "text": ex.decision.text})
                blocks.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                )
            messages.append({"role": "assistant", "content": blocks})
    return messages


def render_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
        for t in tools
    ]


def _block_to_dict(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        return block
    for attr in ("model_dump", "to_dict"):
        fn = getattr(block, attr, None)
        if callable(fn):
            out = fn()
            if isinstance(out, dict):
                return {k: v for k, v in out.items() if v is not None}
    return {"type": getattr(block, "type", "text"), "text": str(block)}


def parse_response(response: Any) -> ProviderTurn:
    tool_calls: list[ToolCall] = []
    texts: list[str] = []
    thoughts: list[str] = []
    for block in response.content:
        kind = getattr(block, "type", None)
        if kind == "tool_use":
            args = block.input if isinstance(block.input, dict) else dict(block.input)
            tool_calls.append(ToolCall(id=str(block.id), name=str(block.name), arguments=args))
        elif kind == "text":
            texts.append(block.text)
        elif kind == "thinking":
            # zero or more per response, and any of them can be empty (a progress block, or
            # the default display); an empty one renders nothing rather than a blank line
            thought = str(getattr(block, "thinking", "") or "").strip()
            if thought:
                thoughts.append(thought)
        elif kind == "redacted_thinking":
            thoughts.append("[redacted thinking]")
    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        explanation = getattr(details, "explanation", None) or "the model refused"
        texts.append(f"[refusal] {explanation}")
        tool_calls = []
    usage = getattr(response, "usage", None)
    # Anthropic is the one vendor that reports its three input buckets DISJOINT: `input_tokens`
    # is what was neither read from a cache nor written to one. quackd's `input_tokens` is the
    # whole prompt (`providers.base.Usage`), so the three are added back up here and the two
    # cache numbers ride along beside it for the rates they are actually billed at. Nothing in
    # quackd sets `cache_control` today, so both are 0 on every run so far, and this is the
    # arithmetic that stops being a no-op on the day one is set.
    cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    # `cache_creation_input_tokens`, not the `cache_creation` object beside it: that object is
    # a breakdown of this same number by cache lifetime (5 minute and 1 hour), and adding both
    # would charge every cached prompt twice.
    cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    # Thinking is inside `output_tokens`, which Anthropic calls "the inclusive, authoritative
    # total used for billing"; `output_tokens_details` is a read-only decomposition of it. So
    # this is recorded for the reader and never priced again (`providers.base.Usage`).
    out_details = getattr(usage, "output_tokens_details", None)
    return ProviderTurn(
        tool_calls=tool_calls,
        text="\n".join(texts) or None,
        usage=Usage(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0) + cache_read + cache_write,
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            reasoning_tokens=int(getattr(out_details, "thinking_tokens", 0) or 0),
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        ),
        stop_reason=stop_reason,
        raw=[_block_to_dict(b) for b in response.content],
        thinking="\n\n".join(thoughts) or None,
    )


def _api_message(e: Exception) -> str:
    """The API's own sentence. The SDK's `message` is `Error code: 400 - {whole body}`;
    `body["error"]["message"]` is the part that names the parameter path."""
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
        if isinstance(body.get("message"), str):
            return str(body["message"])
    return str(getattr(e, "message", None) or e)


def _rejects_thinking(e: Exception) -> bool:
    """A 400 about the top-level `thinking` parameter: a model too old for adaptive thinking.

    The match is anchored because a 400 about a *replayed* thinking block names a path
    (`messages.3.content.0.thinking.signature: Invalid signature`) rather than the parameter.
    Treating that as "this model has no thinking" would retry the identical request — the
    messages are what it objected to — fail again, and leave thinking off for the whole run.
    """
    if type(e).__name__ != "BadRequestError":
        return False
    return re.match(r"\s*thinking\b", _api_message(e)) is not None


class AnthropicProvider:
    name = "anthropic"
    supports_vision = True

    def __init__(
        self,
        model: str | None = None,
        *,
        client: Any = None,
        max_tokens: int | None = None,
        effort: str | None = None,
        fallbacks: bool | None = None,
        thinking_display: str | None = None,
        vision: bool | None = None,
    ) -> None:
        # No model means whatever the catalogue lists first for this vendor.
        self.model = model or default_model_for(self.name) or ""
        # `--no-vision` has to reach every vendor, not most of them: the catalogue promises
        # the flag overrides it in both directions, and a reader who declined the frames must
        # not be billed for them anyway. The catalogue's own per-model flag is not consulted
        # here, because every model this vendor lists takes an image.
        if vision is not None:
            self.supports_vision = vision
        self.max_tokens = max_tokens or int(os.environ.get("QUACKD_MAX_TOKENS", "16000"))
        self.effort = effort or os.environ.get("QUACKD_EFFORT", "medium")
        env_fb = os.environ.get("QUACKD_ANTHROPIC_FALLBACKS", "1") not in ("0", "false", "no")
        self.fallbacks = env_fb if fallbacks is None else fallbacks
        # `summarized` is what makes the thinking blocks carry text. `omitted` keeps the
        # request but hides the text; an empty value sends no `thinking` parameter at all.
        display = (
            os.environ.get("QUACKD_THINKING_DISPLAY", "summarized")
            if thinking_display is None
            else thinking_display
        )
        self.thinking_display: str | None = display.strip() or None
        self.calls = 0
        if client is None:
            try:
                import anthropic
            except ImportError as e:
                raise ProviderNotInstalled("anthropic", "anthropic") from e
            client = anthropic.AsyncAnthropic()  # resolves the key / `ant auth` profile itself
        self.client = client

    def _params(
        self, system: str, history: list[Exchange], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": render_messages(history),
            "tools": render_tools(tools),
            "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
        }
        if self.effort:
            params["output_config"] = {"effort": self.effort}
        if self.thinking_display:
            params["thinking"] = {"type": "adaptive", "display": self.thinking_display}
        return params

    async def _create(self, params: dict[str, Any]) -> Any:
        if self.fallbacks:
            try:
                return await self.client.beta.messages.create(
                    **params, betas=[FALLBACK_BETA], fallbacks="default"
                )
            except TypeError as e:
                # Only a TypeError naming one of the two keywords means "this SDK predates
                # server-side fallbacks". Any other one comes from inside the request (a
                # serialisation bug, a stub) and swallowing it would silently drop fallbacks
                # for the rest of the run and hide the real failure behind a second request.
                if "betas" not in str(e) and "fallbacks" not in str(e):
                    raise
                self.fallbacks = False  # use the plain endpoint from now on
        return await self.client.messages.create(**params)

    async def step(
        self, system: str, history: list[Exchange], tools: list[dict[str, Any]]
    ) -> ProviderTurn:
        params = self._params(system, history, tools)
        self.calls += 1
        # parse_response is inside the classifying try on purpose: a response whose content
        # blocks or usage fields are not the shape it expects would otherwise escape the
        # provider as a raw traceback — the CLI only catches TransportError and ProviderError.
        try:
            try:
                response = await self._create(params)
            except Exception as e:
                if not (self.thinking_display and _rejects_thinking(e)):
                    raise
                # a model that predates adaptive thinking: one retry without the parameter,
                # and every later turn goes without it too. The run loses the thinking text,
                # not itself.
                self.thinking_display = None
                params.pop("thinking", None)
                response = await self._create(params)
            return parse_response(response)
        except ProviderError:
            raise
        except Exception as e:
            raise _classify(e) from e


def _classify(e: Exception) -> ProviderError:
    """Map SDK exceptions (most specific first) to one ProviderError with a useful message."""
    name = type(e).__name__
    status = getattr(e, "status_code", None)
    if name == "AuthenticationError":
        return ProviderError(
            "anthropic: invalid or missing API key (ANTHROPIC_API_KEY, or `ant auth login`)"
        )
    if name == "PermissionDeniedError":
        return ProviderError("anthropic: this key lacks permission for the requested model")
    if name == "NotFoundError":
        return ProviderError("anthropic: model not found — check --model / QUACKD_MODEL")
    if name == "RateLimitError":
        retry = getattr(getattr(e, "response", None), "headers", {}).get("retry-after", "?")
        return ProviderError(f"anthropic: rate limited (retry-after {retry}s)")
    if name == "BadRequestError":
        return ProviderError(f"anthropic: bad request — {getattr(e, 'message', e)}")
    if name == "APIStatusError" or (status is not None and int(status) >= 500):
        return ProviderError(f"anthropic: API error {status}: {getattr(e, 'message', e)}")
    if name in ("APIConnectionError", "APITimeoutError"):
        return ProviderError(f"anthropic: network error — {e}")
    return ProviderError(f"anthropic: {name}: {e}")
