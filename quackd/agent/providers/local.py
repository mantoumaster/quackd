"""Local and open-source models as the duck's brain: llama.cpp, vLLM, Ollama, LM Studio.

They all speak OpenAI's Chat Completions format, so this is the OpenAI provider with the
knobs turned for servers that are pickier and models that are weaker at tool calling:
no key needed, a preset base URL per server, `tool_choice="auto"` and no
`parallel_tool_calls` field, vision off unless asked, model discovery from `/v1/models`,
and a text fallback that rescues a tool call a small model wrote as plain JSON.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from quackd.agent.providers.base import ProviderError, ProviderTurn, ToolCall
from quackd.agent.providers.openai import OpenAIProvider

PRESETS: dict[str, str | None] = {
    "local": None,  # needs --base-url or QUACKD_BASE_URL
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
    "llamacpp": "http://localhost:8080/v1",
    "lmstudio": "http://localhost:1234/v1",
}
LOCAL_NAMES = tuple(PRESETS)

TOOL_HINT = (
    "\n## If you cannot call tools natively\n"
    "Reply with exactly one JSON object and nothing else, in this shape:\n"
    '{"name": "<verb>", "arguments": {<parameters>}}\n'
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
_NAME_KEYS = ("name", "tool", "function", "verb")
_ARGS_KEYS = ("arguments", "parameters", "params", "args", "input")


def split_thinking(text: str) -> tuple[str | None, str]:
    """`<think>...</think>` out of a reply, as (thinking, the rest).

    A server that does not separate reasoning (llama.cpp without `--reasoning-format`, LM
    Studio with separation off, vLLM without a parser) leaves the model's thoughts inline in
    `content`. Left there they would be shown as the answer and replayed to the model next
    turn as something it said. An opening tag with no closing one is thinking to the end of
    the reply: a max_tokens cut mid-thought must not turn a half-finished thought into the
    answer."""
    thoughts = [m.group(1).strip() for m in _THINK_RE.finditer(text)]
    rest = _THINK_RE.sub("", text)
    if (cut := rest.find("<think>")) != -1:
        thoughts.append(rest[cut + len("<think>") :].strip())
        rest = rest[:cut]
    return "\n\n".join(t for t in thoughts if t) or None, rest.strip()


def _candidates(text: str) -> list[dict[str, Any]]:
    """Every JSON object we can find in the text: fenced blocks first, then bare ones."""
    found: list[dict[str, Any]] = []
    for m in _FENCE_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            found.append(obj)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj not in found:
            found.append(obj)
    return found


def _as_call(obj: dict[str, Any], tool_names: set[str]) -> ToolCall | None:
    fn = obj.get("function")
    if isinstance(fn, dict):  # OpenAI style {"function": {"name": .., "arguments": ..}}
        obj = {**obj, **fn}
    name = next((obj[k] for k in _NAME_KEYS if isinstance(obj.get(k), str)), None)
    if name is None or name not in tool_names:
        return None
    args: Any = next((obj[k] for k in _ARGS_KEYS if k in obj), {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return ToolCall(name=name, arguments=args)


def parse_tool_call_from_text(text: str, tool_names: set[str]) -> ToolCall | None:
    """A tool call a model wrote as JSON text instead of a native tool call, or None."""
    for obj in _candidates(text or ""):
        call = _as_call(obj, tool_names)
        if call is not None:
            return call
    return None


class LocalProvider(OpenAIProvider):
    supports_vision = False
    key_env = "LOCAL_API_KEY"
    default_tool_choice = "auto"
    send_parallel_flag = False
    prompt_hint = TOOL_HINT

    def __init__(
        self,
        model: str | None = None,
        *,
        preset: str = "local",
        base_url: str | None = None,
        client: Any = None,
        api_key: str | None = None,
        tool_choice: str | None = None,
        vision: bool | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        if preset not in PRESETS:
            raise ProviderError(f"unknown local preset {preset!r}; one of {', '.join(LOCAL_NAMES)}")
        self.name = preset
        self.preset = preset
        url = (
            base_url
            or os.environ.get("QUACKD_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or PRESETS[preset]
        )
        if not url:
            raise ProviderError(
                "--llm local needs the server address: --base-url http://host:port/v1 "
                "(or QUACKD_BASE_URL). Or use a preset: --llm ollama, vllm, llamacpp, lmstudio."
            )
        choice = tool_choice or os.environ.get("QUACKD_TOOL_CHOICE") or "auto"
        if vision is None:
            vision = os.environ.get("QUACKD_VISION", "0").lower() not in ("", "0", "false", "no")
        super().__init__(
            model or "",
            client=client,
            api_key=api_key,
            base_url=url,
            tool_choice=choice,
            vision=vision,
            extra_body=extra_body,
        )
        self.text_fallbacks = 0

    def _fallback_key(self) -> str | None:
        # local servers ignore the key but the SDK insists on a non-empty string
        return os.environ.get("OPENAI_API_KEY") or "not-needed"

    async def ensure_model(self) -> str:
        """Discover the served model when none was given (`/v1/models`, first entry)."""
        if self.model:
            return self.model
        try:
            page = await self.client.models.list()
        except Exception as e:
            raise ProviderError(
                f"{self.name}: cannot list models at {self.base_url} ({type(e).__name__}: {e}). "
                "Is the server running? Pass --llm <preset>:<model> to skip discovery."
            ) from e
        ids = [str(getattr(m, "id", "")) for m in (getattr(page, "data", None) or [])]
        ids = [i for i in ids if i]
        if not ids:
            raise ProviderError(
                f"{self.name}: the server at {self.base_url} lists no models. "
                "Load one (e.g. `ollama pull qwen3:8b`) or pass --llm <preset>:<model>."
            )
        self.model = ids[0]
        return self.model

    async def step(
        self, system: str, history: list[Any], tools: list[dict[str, Any]]
    ) -> ProviderTurn:
        await self.ensure_model()
        return await super().step(system, history, tools)

    def _normalise(self, turn: ProviderTurn) -> ProviderTurn:
        """Split inline `<think>` off before anything else reads the text.

        It runs here, not after `super().step()`, because the JSON text fallback reads
        `turn.text`: a model that weighs a verb in its reasoning and then rejects it
        (`<think>maybe {"name": "kick"} but no</think>I will wait.`) would otherwise have
        that call parsed out of the discarded thought and executed.
        """
        if turn.thinking is None and turn.text and "<think>" in turn.text:
            thinking, rest = split_thinking(turn.text)
            turn = turn.model_copy(update={"thinking": thinking, "text": rest or None})
        return turn

    def _fallback(self, turn: ProviderTurn, tools: list[dict[str, Any]]) -> ProviderTurn:
        call = parse_tool_call_from_text(turn.text or "", {t["name"] for t in tools})
        if call is None:
            return turn
        self.text_fallbacks += 1
        call = call.model_copy(update={"id": f"text-{self.text_fallbacks}"})
        return turn.model_copy(update={"tool_calls": [call], "stop_reason": "text_fallback"})
