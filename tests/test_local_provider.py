"""Local and open-source providers: presets, no key, relaxed tool calling, text fallback."""

from __future__ import annotations

import json
from types import SimpleNamespace as NS
from typing import Any

import pytest

from quackd.agent.providers.base import (
    Exchange,
    NamedPng,
    Observation,
    ProviderError,
    ToolCall,
)
from quackd.agent.providers.factory import LOCAL_NAMES, PROVIDER_NAMES, make_provider
from quackd.agent.providers.local import (
    PRESETS,
    LocalProvider,
    parse_tool_call_from_text,
    split_thinking,
)
from quackd.agent.providers.openai import OpenAIProvider

TOOLS = [
    {"name": "walk_to", "description": "d", "input_schema": {"type": "object", "properties": {}}},
    {"name": "kick", "description": "d", "input_schema": {"type": "object", "properties": {}}},
]
NAMES = {"walk_to", "kick"}


class FakeClient:
    def __init__(self, response: Any = None, models: list[str] | None = None) -> None:
        self.kwargs: dict[str, Any] = {}
        self.listed = 0

        async def create(**kwargs: Any) -> Any:
            self.kwargs = kwargs
            return response

        async def list_models() -> Any:
            self.listed += 1
            return NS(data=[NS(id=m) for m in (models or [])])

        self.chat = NS(completions=NS(create=create))
        self.models = NS(list=list_models)


def reply(text: str | None = None, tool_calls: list[Any] | None = None) -> Any:
    return NS(
        choices=[NS(message=NS(content=text, tool_calls=tool_calls), finish_reason="stop")],
        usage=NS(prompt_tokens=10, completion_tokens=5),
    )


def history() -> list[Exchange]:
    return [
        Exchange(observation=Observation(text="obs", images=[NamedPng(name="camera", png=b"png")]))
    ]


# ── presets and construction ────────────────────────────────────────────────────────────


def test_presets_and_names() -> None:
    assert set(LOCAL_NAMES) == set(PRESETS) and all(n in PROVIDER_NAMES for n in LOCAL_NAMES)
    assert LocalProvider("m", preset="ollama", client=FakeClient()).base_url.endswith(":11434/v1")
    assert LocalProvider("m", preset="vllm", client=FakeClient()).base_url.endswith(":8000/v1")
    assert LocalProvider("m", preset="llamacpp", client=FakeClient()).base_url.endswith(":8080/v1")
    assert LocalProvider("m", preset="lmstudio", client=FakeClient()).base_url.endswith(":1234/v1")


def test_local_needs_an_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QUACKD_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(ProviderError, match="--base-url"):
        LocalProvider("m", preset="local", client=FakeClient())
    p = LocalProvider("m", preset="local", base_url="http://gpu:8000/v1", client=FakeClient())
    assert p.base_url == "http://gpu:8000/v1" and p.name == "local"


def test_env_base_url_overrides_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUACKD_BASE_URL", "http://box:9999/v1")
    assert LocalProvider("m", preset="ollama", client=FakeClient()).base_url == "http://box:9999/v1"


def test_no_key_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCAL_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    p = LocalProvider("m", preset="ollama", client=FakeClient())
    assert p._fallback_key() == "not-needed"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        OpenAIProvider("m")  # cloud still requires a key


# ── request params ──────────────────────────────────────────────────────────────────────


async def test_local_sends_relaxed_tool_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QUACKD_TOOL_CHOICE", raising=False)
    client = FakeClient(reply(tool_calls=[NS(id="c1", function=NS(name="kick", arguments="{}"))]))
    turn = await LocalProvider("m", preset="ollama", client=client).step("S", history(), TOOLS)
    assert client.kwargs["tool_choice"] == "auto"
    assert "parallel_tool_calls" not in client.kwargs
    assert turn.tool_calls[0].name == "kick"


async def test_tool_choice_none_omits_the_field(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUACKD_TOOL_CHOICE", "none")
    client = FakeClient(reply(text="{}"))
    await LocalProvider("m", preset="ollama", client=client).step("S", history(), TOOLS)
    assert "tool_choice" not in client.kwargs


async def test_local_forwards_extra_body() -> None:
    """The presets are where #12 came from: a vLLM server wants a field in the body and
    `LocalProvider` has to carry it up to the base class, which is what sends it."""
    body = {"chat_template_kwargs": {"enable_thinking": False}}
    client = FakeClient(reply(text="{}"))
    await LocalProvider("m", preset="vllm", client=client, extra_body=body).step(
        "S", history(), TOOLS
    )
    assert client.kwargs["extra_body"] == body


async def test_cloud_openai_keeps_strict_params() -> None:
    client = FakeClient(reply(tool_calls=[NS(id="c1", function=NS(name="kick", arguments="{}"))]))
    await OpenAIProvider("gpt-5", client=client).step("S", history(), TOOLS)
    assert client.kwargs["tool_choice"] == "required"
    assert client.kwargs["parallel_tool_calls"] is False


# ── vision ──────────────────────────────────────────────────────────────────────────────


def test_vision_off_by_default_on_for_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QUACKD_VISION", raising=False)
    assert LocalProvider("m", preset="ollama", client=FakeClient()).supports_vision is False
    assert LocalProvider("m", preset="ollama", client=FakeClient(), vision=True).supports_vision
    monkeypatch.setenv("QUACKD_VISION", "1")
    assert LocalProvider("m", preset="ollama", client=FakeClient()).supports_vision is True


async def test_every_camera_reaches_a_local_server_in_one_message() -> None:
    """A local server speaks the OpenAI wire format, so it inherits multi-camera rendering
    without a line of its own here. That inheritance is the thing worth pinning: the one
    provider quackd cannot test against a real endpoint is the one most likely to be handed
    two pictures by somebody with a webcam on each side of an arm.

    It is also where the cost lands. Two cameras is two image parts in a single message, and
    `docs/local-llms.md` warns that a server or a model may accept only one. This says what
    quackd sends; whether a given server takes it is that server's business."""
    client = FakeClient(reply(text="{}"))
    p = LocalProvider("m", preset="ollama", client=client, vision=True)
    two = [
        Exchange(
            observation=Observation(
                text="obs",
                images=[NamedPng(name="top", png=b"a"), NamedPng(name="side", png=b"b")],
            )
        )
    ]
    await p.step("system", two, [])
    parts = client.kwargs["messages"][1]["content"]  # type: ignore[attr-defined]
    assert [part["type"] for part in parts] == ["text", "text", "image_url", "text", "image_url"]
    assert parts[1]["text"] == "camera top:" and parts[3]["text"] == "camera side:"

    # and one camera is the single part it has always been, with no label in front of it
    client_one = FakeClient(reply(text="{}"))
    await LocalProvider("m", preset="ollama", client=client_one, vision=True).step(
        "system", history(), []
    )
    one = client_one.kwargs["messages"][1]["content"]  # type: ignore[attr-defined]
    assert [part["type"] for part in one] == ["text", "image_url"], one

    # the case the two above cannot tell apart, and the only one that needed a change: ONE
    # picture from a body that has TWO cameras. Counting the images that arrived makes this
    # look exactly like the single-camera message above, and sends the side view out bare
    # under a detections line measured off the lens that died.
    client_lost = FakeClient(reply(text="{}"))
    lost = [
        Exchange(
            observation=Observation(
                text="obs",
                images=[NamedPng(name="side", png=b"b")],
                cameras=["top", "side"],
            )
        )
    ]
    await LocalProvider("m", preset="ollama", client=client_lost, vision=True).step(
        "system", lost, []
    )
    survivor = client_lost.kwargs["messages"][1]["content"]  # type: ignore[attr-defined]
    assert [part["type"] for part in survivor] == ["text", "text", "image_url"], survivor
    assert survivor[1]["text"] == "camera side:", "the one lens left went out unnamed"


# ── model discovery ─────────────────────────────────────────────────────────────────────


async def test_model_discovery_takes_first_served_model() -> None:
    client = FakeClient(reply(text="{}"), models=["qwen3:8b", "llama3.2"])
    p = LocalProvider(None, preset="ollama", client=client)
    await p.step("S", history(), TOOLS)
    assert p.model == "qwen3:8b" and client.kwargs["model"] == "qwen3:8b"
    await p.step("S", history(), TOOLS)
    assert client.listed == 1  # discovered once


async def test_model_discovery_with_empty_server_is_clear() -> None:
    p = LocalProvider(None, preset="ollama", client=FakeClient(reply(text="{}"), models=[]))
    with pytest.raises(ProviderError, match="lists no models"):
        await p.step("S", history(), TOOLS)


# ── text fallback ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        '{"name": "walk_to", "arguments": {"target": "ball"}}',
        'Sure! ```json\n{"name": "walk_to", "arguments": {"target": "ball"}}\n```',
        '{"tool": "walk_to", "parameters": {"target": "ball"}}',
        '{"function": {"name": "walk_to", "arguments": "{\\"target\\": \\"ball\\"}"}}',
        'I will approach. {"name":"walk_to","args":{"target":"ball"}} Then kick.',
    ],
)
def test_parse_tool_call_from_text_variants(text: str) -> None:
    call = parse_tool_call_from_text(text, NAMES)
    assert call == ToolCall(name="walk_to", arguments={"target": "ball"})


@pytest.mark.parametrize(
    "text",
    ["no json here", '{"name": "fly", "arguments": {}}', "{broken", "", '{"foo": 1}'],
)
def test_parse_tool_call_from_text_rejects(text: str) -> None:
    assert parse_tool_call_from_text(text, NAMES) is None


async def test_text_fallback_synthesizes_a_tool_call() -> None:
    text = 'Let me walk there.\n```json\n{"name": "walk_to", "arguments": {"target": "ball"}}\n```'
    p = LocalProvider("m", preset="llamacpp", client=FakeClient(reply(text=text)))
    turn = await p.step("S", history(), TOOLS)
    assert turn.stop_reason == "text_fallback" and p.text_fallbacks == 1
    assert turn.tool_calls == [ToolCall(id="text-1", name="walk_to", arguments={"target": "ball"})]


async def test_no_fallback_when_nothing_parses() -> None:
    p = LocalProvider("m", preset="llamacpp", client=FakeClient(reply(text="I am a duck.")))
    turn = await p.step("S", history(), TOOLS)
    assert turn.tool_calls == [] and turn.stop_reason == "stop"


async def test_a_tool_call_the_model_only_contemplated_in_its_thinking_is_not_executed() -> None:
    """A small model weighs a verb out loud and rejects it. The fallback scans plain text, so
    unless the reasoning is stripped first it finds the JSON in the discarded thought and the
    duck kicks."""
    text = '<think>maybe {"name": "kick", "arguments": {}} but no</think>I will wait.'
    p = LocalProvider("m", preset="llamacpp", client=FakeClient(reply(text=text)))
    turn = await p.step("S", history(), TOOLS)
    assert turn.tool_calls == [] and p.text_fallbacks == 0
    assert turn.text == "I will wait."
    assert turn.thinking == 'maybe {"name": "kick", "arguments": {}} but no'


async def test_the_json_fallback_reads_the_answer_after_the_thinking() -> None:
    text = '<think>kick? no</think>{"name": "walk_to", "arguments": {"target": "ball"}}'
    p = LocalProvider("m", preset="llamacpp", client=FakeClient(reply(text=text)))
    turn = await p.step("S", history(), TOOLS)
    assert turn.tool_calls == [ToolCall(id="text-1", name="walk_to", arguments={"target": "ball"})]
    assert turn.thinking == "kick? no"


async def test_an_unterminated_think_tag_is_thinking_to_the_end_not_the_answer() -> None:
    """max_tokens can cut a reply mid-thought, leaving no `</think>`. Everything after the
    opening tag is still reasoning: shown as the answer it would be replayed next turn as
    something the model said, and any verb it was still weighing would be executed."""
    assert split_thinking("<think>the ball is") == ("the ball is", "")
    assert split_thinking("ok<think>cut") == ("cut", "ok")
    assert split_thinking("<think>a</think>b<think>c") == ("a\n\nc", "b")
    text = '<think>I could {"name": "kick", "arguments": {}}'
    p = LocalProvider("m", preset="llamacpp", client=FakeClient(reply(text=text)))
    turn = await p.step("S", history(), TOOLS)
    assert turn.tool_calls == [] and turn.text is None
    assert turn.thinking == 'I could {"name": "kick", "arguments": {}}'


def test_prompt_hint_only_for_local() -> None:
    assert "JSON object" in LocalProvider("m", preset="ollama", client=FakeClient()).prompt_hint
    assert OpenAIProvider("gpt-5", client=FakeClient()).prompt_hint == ""


# ── factory ─────────────────────────────────────────────────────────────────────────────


def test_factory_builds_local_presets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("quackd.agent.providers.local.LocalProvider.__init__", _record_init)
    p = make_provider(
        "vllm",
        model="Qwen/Qwen3-8B",
        base_url="http://gpu:8000/v1",
        vision=True,
        extra_body='{"chat_template_kwargs": {"enable_thinking": false}}',
    )
    assert p.recorded == {  # type: ignore[attr-defined]
        "model": "Qwen/Qwen3-8B",
        "preset": "vllm",
        "base_url": "http://gpu:8000/v1",
        "api_key": None,
        "vision": True,
        # the flag arrives as text and the factory hands the provider the parsed object
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }


def _record_init(self: Any, model: Any = None, **kw: Any) -> None:
    self.recorded = {
        "model": model,
        **{k: kw.get(k) for k in ("preset", "base_url", "api_key", "vision", "extra_body")},
    }
    self.name = kw.get("preset", "local")
    self.model = model or ""
    self.supports_vision = bool(kw.get("vision"))


def test_openai_history_json_roundtrip_unchanged() -> None:
    # sanity: the local provider still renders OpenAI-shaped messages
    client = FakeClient(reply(text="{}"))
    p = LocalProvider("m", preset="ollama", client=client)
    params = p._params("SYS", history(), TOOLS)
    assert params["messages"][0] == {"role": "system", "content": "SYS"}
    assert json.dumps(params["tools"])  # serialisable
