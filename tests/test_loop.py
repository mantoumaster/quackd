"""The deliberation loop, end to end on the mock transport with the scripted provider."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from quackd.adapters.base import AdapterError
from quackd.agent.images import load_task_images
from quackd.agent.loop import AgentLoop, RunConfig, run_duck
from quackd.agent.providers.base import (
    Exchange,
    Observation,
    ProviderError,
    ProviderTurn,
    ToolCall,
    Usage,
)
from quackd.agent.providers.fake import FakeProvider
from quackd.agent.providers.openai import render_messages
from quackd.agent.transcript import Transcript
from quackd.duckfile.schema import Budgets, DuckFile
from quackd.transport.base import CameraFrame
from quackd.transport.mock import MockTransport
from quackd_lerobot import LeRobotAdapter
from quackd_lerobot.mock import REST, LeRobotMock
from quackd_lerobot.verbs import GRIPPER_OPEN, TOL_DEG, rest_goal
from quackd_microduck import MicroduckAdapter

# the verdict comes first on every run now: the scripted pilot answers it as a rule, and the
# duck's own three verbs follow exactly as they did
GOLDEN_HELLO = ["assess_task", "quack", "walk", "quack", "declare_success"]


async def test_run_start_records_the_extra_body(hello_duck: DuckFile, tmp_path: Path) -> None:
    """A run whose model was told not to think reads nothing like one that was, and the
    transcript is the only place a reader can tell which of the two they are holding. The
    emit is a `getattr`, which would record None for ever if the attribute were renamed."""
    body = {"chat_template_kwargs": {"enable_thinking": False}}
    provider = FakeProvider.for_duck(hello_duck.name)
    provider.extra_body = body  # type: ignore[attr-defined]
    result = await run_duck(
        RunConfig(duck=hello_duck, provider=provider, transport=MockTransport(), runs_dir=tmp_path)
    )
    start = Transcript.read(result.run_dir / "transcript.jsonl")[0]
    assert start["kind"] == "run_start" and start["extra_body"] == body

    # a provider with no such attribute records None rather than raising
    plain = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=FakeProvider.for_duck(hello_duck.name),
            transport=MockTransport(),
            runs_dir=tmp_path / "plain",
        )
    )
    assert Transcript.read(plain.run_dir / "transcript.jsonl")[0]["extra_body"] is None


async def test_hello_world_golden(hello_duck: DuckFile, tmp_path: Path) -> None:
    transport = MockTransport()
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=FakeProvider.for_duck("hello-world"),
            transport=transport,
            runs_dir=tmp_path,
        )
    )
    assert result.outcome == "success", result.reason
    # hello-world allows 5 llm calls and the verdict is the fifth: it fits, exactly
    assert result.steps == 3 and result.llm_calls == 5
    events = Transcript.read(result.run_dir / "transcript.jsonl")
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    assert {"observation", "llm", "verb", "declare"} <= set(kinds)
    calls = [tc["name"] for e in events if e["kind"] == "llm" for tc in e["tool_calls"]]
    assert calls == GOLDEN_HELLO
    assert (result.run_dir / "summary.json").exists()
    assert [i.kind for i in transport.intents if i.kind != "stop"] == ["sound"] + ["move"] * 10 + [
        "sound"
    ]
    assert transport.intents[-1].kind == "stop"  # the loop always stops the duck on exit
    assert not transport.connected  # and closes the transport
    assert (result.run_dir / "frames").is_dir()


class NoToolProvider:
    name = "no-tool"
    model = "x"
    supports_vision = False

    async def step(
        self, system: str, history: list[Exchange], tools: list[dict[str, Any]]
    ) -> ProviderTurn:
        return ProviderTurn(tool_calls=[], text="I would rather talk.")


class ClockAdvancingProvider:
    name = "clock-advancing"
    model = "test"
    supports_vision = False

    def __init__(
        self, transport: MockTransport, seconds: float, declaration: str = "declare_success"
    ) -> None:
        self.transport = transport
        self.seconds = seconds
        self.declaration = declaration

    async def step(
        self, system: str, history: list[Exchange], tools: list[dict[str, Any]]
    ) -> ProviderTurn:
        await self.transport.sleep(self.seconds)
        return ProviderTurn(
            tool_calls=[ToolCall(name=self.declaration, arguments={"reason": "provider response"})]
        )


async def test_no_tool_call_is_reprompted_once_then_failure(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    result = await run_duck(
        RunConfig(
            duck=hello_duck, provider=NoToolProvider(), transport=MockTransport(), runs_dir=tmp_path
        )
    )
    assert result.outcome == "failure" and "no tool call" in result.reason
    assert result.llm_calls == 2


@pytest.mark.parametrize("declaration", ["declare_success", "declare_failure"])
async def test_time_budget_wins_over_late_declaration(
    hello_duck: DuckFile, tmp_path: Path, declaration: str
) -> None:
    hello_duck.frontmatter.budgets = Budgets(max_minutes=0.1)
    transport = MockTransport()
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=ClockAdvancingProvider(transport, 7, declaration),
            transport=transport,
            runs_dir=tmp_path,
        )
    )

    assert result.outcome == "budget"
    assert result.reason == "max_minutes (0.1) exceeded"
    assert transport.now() == 7
    events = Transcript.read(result.run_dir / "transcript.jsonl")
    kinds = [event["kind"] for event in events]
    assert "llm" in kinds and "declare" not in kinds


async def test_provider_response_before_time_budget_is_processed(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    hello_duck.frontmatter.budgets = Budgets(max_minutes=0.1, max_llm_calls=1)
    transport = MockTransport()
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=ClockAdvancingProvider(transport, 5),
            transport=transport,
            runs_dir=tmp_path,
        )
    )

    assert result.outcome == "success"
    assert result.reason == "provider response"
    assert result.llm_calls == 1
    assert transport.now() == 5
    events = Transcript.read(result.run_dir / "transcript.jsonl")
    assert "declare" in [event["kind"] for event in events]


async def test_budget_ends_the_run(hello_duck: DuckFile, tmp_path: Path) -> None:
    forever = FakeProvider(script=[ToolCall(name="quack", arguments={})])
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=forever,
            transport=MockTransport(),
            runs_dir=tmp_path,
            max_steps=2,
        )
    )
    assert result.outcome == "budget" and "max_steps" in result.reason
    assert result.steps == 2


async def test_a_max_steps_override_reaches_the_prompt_as_well_as_the_observation_header(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    """`--max-steps` changed the budget and not the sentence about it.

    On 2026-09-15 a `--max-steps 10` run on the SO-101 arm was handed a system prompt saying
    `Budgets: 40 steps` above observations that said `step 0/10`, because the override was
    copied into the contract the executor enforces and the prompt was built from the task
    file's own. A pilot told it has four times the budget it has plans differently, and it
    cannot tell which number is real. All three readings of it have to agree."""
    result = await run_duck(
        RunConfig(
            duck=hello_duck,  # its own file says five
            provider=FakeProvider.for_duck("hello-world"),
            transport=MockTransport(),
            runs_dir=tmp_path,
            max_steps=7,
        )
    )
    assert result.outcome == "success", result.reason
    events = Transcript.read(result.run_dir / "transcript.jsonl")
    start = events[0]
    assert start["contract"]["budgets"]["max_steps"] == 7, "the contract kept the override"
    assert "Budgets: 7 steps" in start["system_prompt"], "the prompt did not"
    assert "Budgets: 5 steps" not in start["system_prompt"], "it still says the file's own"
    header = next(e for e in events if e["kind"] == "observation")["text"]
    assert header.startswith("[step 0/7 "), header


async def test_disallowed_verb_is_feedback(hello_duck: DuckFile, tmp_path: Path) -> None:
    naughty = FakeProvider(
        script=[
            ToolCall(name="kick", arguments={}),
            ToolCall(name="declare_failure", arguments={"reason": "refused"}),
        ]
    )
    transport = MockTransport()
    result = await run_duck(
        RunConfig(duck=hello_duck, provider=naughty, transport=transport, runs_dir=tmp_path)
    )
    assert result.outcome == "failure"
    events = Transcript.read(result.run_dir / "transcript.jsonl")
    verb_events = [e for e in events if e["kind"] == "verb"]
    assert verb_events[0]["name"] == "kick" and not verb_events[0]["ok"]
    assert "allowlist" in verb_events[0]["summary"]
    assert transport.intents_of("do") == []


async def test_heartbeat_failure_aborts_the_run(hello_duck: DuckFile, tmp_path: Path) -> None:
    transport = MockTransport(fail_heartbeat_after=0)
    forever = FakeProvider(script=[ToolCall(name="walk", arguments={"duration_s": 2.0})])
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=forever,
            transport=transport,
            runs_dir=tmp_path,
            heartbeat_period_s=0.001,
        )
    )
    assert result.outcome == "aborted"
    assert "heartbeat" in result.reason
    assert transport.stops >= 1


async def test_dry_run_touches_nothing(hello_duck: DuckFile, tmp_path: Path) -> None:
    transport = MockTransport()
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=FakeProvider.for_duck("hello-world"),
            transport=transport,
            runs_dir=tmp_path,
            dry_run=True,
        )
    )
    assert result.outcome == "success"
    assert [i.kind for i in transport.intents] == ["stop"]  # only the final safety stop


# ── the trace: the run narrates itself ──────────────────────────────────────────────────


class ThinkingProvider:
    """A model that reasons out loud, which no scripted strategy does."""

    name = "thinker"
    model = "test"
    supports_vision = False

    def __init__(self, *calls: ToolCall) -> None:
        self.script = list(calls)
        self.calls = 0

    async def step(
        self, system: str, history: list[Exchange], tools: list[dict[str, Any]]
    ) -> ProviderTurn:
        call = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return ProviderTurn(
            tool_calls=[call],
            text="on it",
            thinking=f"turn {self.calls}: I will {call.name}",
            usage=Usage(input_tokens=100, output_tokens=10),
            stop_reason="tool_use",
        )


async def test_the_transcript_carries_the_whole_conversation(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    """Every step the run takes on the model's behalf is a line: what was asked, what it
    thought, what it answered, what the executor decided, what went to the robot."""
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=ThinkingProvider(
                ToolCall(name="quack", arguments={"text": "hi"}),
                ToolCall(name="declare_success", arguments={"reason": "quacked"}),
            ),
            transport=MockTransport(),
            runs_dir=tmp_path,
        )
    )
    assert result.outcome == "success"
    events = Transcript.read(result.run_dir / "transcript.jsonl")
    kinds = {e["kind"] for e in events}
    assert {"llm_request", "verb_start", "intent", "verb_end"} <= kinds

    llm = next(e for e in events if e["kind"] == "llm")
    assert llm["thinking"] == "turn 1: I will quack"
    assert llm["latency_s"] >= 0 and llm["usage_total"]["input_tokens"] == 100

    request = next(e for e in events if e["kind"] == "llm_request")
    assert request["messages"] == 1 and request["reprompt"] is False

    start = next(e for e in events if e["kind"] == "verb_start")
    assert start["name"] == "quack" and start["source"] == "agent" and start["nested"] is False

    intent = next(e for e in events if e["kind"] == "intent")
    assert intent["intent"] == "sound" and intent["accepted"] is True

    end = next(e for e in events if e["kind"] == "verb_end")
    assert end["outcome"] == "ok" and end["intents"] == {"sound": 1} and end["elapsed_s"] >= 0
    # the loop's own `verb` record is unchanged, so everything that reads it still can
    verb = next(e for e in events if e["kind"] == "verb")
    assert verb["name"] == "quack" and verb["ok"] is True


async def test_the_final_safety_stop_is_in_the_trace_too(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    """The intents that matter most are the ones sent because something went wrong."""
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=FakeProvider.for_duck("hello-world"),
            transport=MockTransport(),
            runs_dir=tmp_path,
        )
    )
    events = Transcript.read(result.run_dir / "transcript.jsonl")
    stops = [e for e in events if e["kind"] == "intent" and e["intent"] == "stop"]
    assert stops, "the run always stops the robot on the way out, and must say so"


async def test_a_provider_that_fails_says_so_instead_of_exiting_unexpectedly(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    """A bad key, a 429 or a dropped connection is what a first real run hits. The run used
    to end with `loop exited unexpectedly` and no record of the call that failed."""

    class Failing:
        name, model, supports_vision = "failing", "test", False

        async def step(self, system: str, history: Any, tools: Any) -> ProviderTurn:
            raise ProviderError("anthropic: rate limited (retry-after 7s)")

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(ProviderError):
        await run_duck(
            RunConfig(
                duck=hello_duck,
                provider=Failing(),
                transport=MockTransport(),
                run_dir=run_dir,
                runs_dir=tmp_path,
            )
        )
    events = Transcript.read(run_dir / "transcript.jsonl")
    failed = next(e for e in events if e["kind"] == "llm")
    assert "rate limited" in failed["error"] and failed["latency_s"] >= 0
    end = next(e for e in events if e["kind"] == "run_end")
    assert end["outcome"] == "error" and "rate limited" in end["reason"]


async def test_a_cancelled_run_ends_as_an_abort_that_still_stops_and_records(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    """`KeyboardInterrupt` and `CancelledError` are not `Exception`, so neither reached the
    error branch and `run_end` kept its default, `loop exited unexpectedly` — the very string
    the trace work claimed to have removed. The CLI's second Ctrl-C is this path."""

    class Stalling:
        name, model, supports_vision = "stalling", "test", False

        async def step(self, system: str, history: Any, tools: Any) -> ProviderTurn:
            await asyncio.sleep(10)
            raise AssertionError("never reached")

    run_dir = tmp_path / "cancelled"
    run_dir.mkdir()
    transport = MockTransport()
    task = asyncio.create_task(
        run_duck(
            RunConfig(
                duck=hello_duck,
                provider=Stalling(),
                transport=transport,
                run_dir=run_dir,
                runs_dir=tmp_path,
            )
        )
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    events = Transcript.read(run_dir / "transcript.jsonl")
    end = next(e for e in events if e["kind"] == "run_end")
    assert end["outcome"] == "aborted" and "CancelledError" in end["reason"]
    assert any(e["kind"] == "note" and "interrupted" in e["text"] for e in events)
    assert transport.intents[-1].kind == "stop" and not transport.connected
    assert (run_dir / "summary.json").exists()


async def test_a_record_that_fails_at_run_end_still_gets_its_summary_and_is_closed(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    """A disk that fills at the last line used to skip summary.json, leak the handle and
    replace the run's own outcome with an OSError."""
    from quackd.agent.loop import AgentLoop

    run_dir = tmp_path / "full"
    run_dir.mkdir()
    loop = AgentLoop(
        RunConfig(
            duck=hello_duck,
            provider=FakeProvider.for_duck("hello-world"),
            transport=MockTransport(),
            run_dir=run_dir,
            runs_dir=tmp_path,
        )
    )
    good = loop.transcript.sink

    def record(event: Any) -> None:
        if event.kind == "run_end":
            raise OSError("disk full")
        good(event)

    loop.tracer.record = record
    with pytest.raises(OSError, match="disk full"):
        await loop.run()
    assert (run_dir / "summary.json").exists()
    assert loop.transcript._fh.closed


def test_writing_to_a_closed_transcript_is_a_no_op(tmp_path: Path) -> None:
    """A verb task cancelled during teardown narrates its last intent after the record has
    closed; that must not raise inside a task nobody awaits."""
    from quackd.trace import TraceEvent

    t = Transcript(tmp_path)
    t.close()
    t.write("intent", intent="stop")
    t.sink(TraceEvent("intent", 0.0, {"intent": "stop"}))
    assert t.events == 0


async def test_a_console_sees_the_run_as_it_happens(hello_duck: DuckFile, tmp_path: Path) -> None:
    seen: list[str] = []
    await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=FakeProvider.for_duck("hello-world"),
            transport=MockTransport(),
            runs_dir=tmp_path,
            trace=lambda event: seen.append(event.kind),
        )
    )
    assert seen[0] == "run_start" and seen[-1] == "run_end"
    assert {"observation", "llm", "verb_start", "intent", "verb_end", "declare"} <= set(seen)


async def test_a_broken_console_never_ends_a_run(hello_duck: DuckFile, tmp_path: Path) -> None:
    """A terminal that cannot print is not a reason to stop a robot mid-task."""

    def broken(_event: Any) -> None:
        raise RuntimeError("the terminal went away")

    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=FakeProvider.for_duck("hello-world"),
            transport=MockTransport(),
            runs_dir=tmp_path,
            trace=broken,
        )
    )
    assert result.outcome == "success"
    assert Transcript.read(result.run_dir / "transcript.jsonl")  # the record is unaffected


async def test_the_summary_counts_the_events_a_broken_console_dropped(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    """A console that raises on every event produced a silent trace, an unchanged exit code
    and no line anywhere saying events had been dropped."""

    def broken(_event: Any) -> None:
        raise RuntimeError("the terminal went away")

    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=FakeProvider.for_duck("hello-world"),
            transport=MockTransport(),
            runs_dir=tmp_path,
            trace=broken,
        )
    )
    assert result.trace_dropped > 0
    summary = json.loads((result.run_dir / "summary.json").read_text(encoding="utf-8"))
    end = next(
        e for e in Transcript.read(result.run_dir / "transcript.jsonl") if e["kind"] == "run_end"
    )
    # the record's own count is one short of the run's, and can only ever be: it is taken
    # while the summary is built, and emitting `run_end` with it is one more event to drop.
    # The CLI prints the result's, which is complete.
    assert summary["trace_dropped"] == end["trace_dropped"] == result.trace_dropped - 1


async def test_thinking_on_the_reprompt_turn_is_recorded(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    """The re-prompt is a second call to the model in the same step, and nothing asserted
    that the request was marked as one or that its answer's reasoning was kept."""

    class Dithering:
        name, model, supports_vision = "dithering", "test", False

        def __init__(self) -> None:
            self.calls = 0

        async def step(self, system: str, history: Any, tools: Any) -> ProviderTurn:
            self.calls += 1
            if self.calls == 1:
                return ProviderTurn(tool_calls=[], text="hmm", thinking="turn 1: still deciding")
            return ProviderTurn(
                tool_calls=[ToolCall(name="declare_success", arguments={"reason": "done"})],
                thinking="turn 2: it wants exactly one tool",
            )

    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=Dithering(),
            transport=MockTransport(),
            runs_dir=tmp_path,
        )
    )
    assert result.outcome == "success"
    events = Transcript.read(result.run_dir / "transcript.jsonl")
    requests = [e for e in events if e["kind"] == "llm_request"]
    assert [r["reprompt"] for r in requests] == [False, True]
    assert [e["thinking"] for e in events if e["kind"] == "llm"][1] == (
        "turn 2: it wants exactly one tool"
    )
    enforce = next(e for e in events if e["kind"] == "enforce")
    assert enforce["text"] == "You must call exactly one tool. Choose now."


async def test_a_composite_is_traced_as_nested_pairs_with_the_parents_tally(
    kick_duck: DuckFile, tmp_path: Path
) -> None:
    """A composite sends nothing itself: every intent `approach_and` reports came from the
    `go_to` and the `kick` it ran. The tally is a chain of `ContextVar` frames, so a
    regression there is silent — the parent would keep reporting a number, just a smaller
    one, and a reader would believe it."""
    from quackd.perception.color_blob import ColorBlobDetector
    from quackd.transport.sim2d import Sim2DTransport

    kick_duck.frontmatter.verbs.allow = [*kick_duck.frontmatter.verbs.allow, "approach_and"]
    seen: list[Any] = []
    result = await run_duck(
        RunConfig(
            duck=kick_duck,
            provider=FakeProvider(
                script=[
                    ToolCall(name="search_scan", arguments={"target": "ball"}),
                    ToolCall(
                        name="approach_and",
                        arguments={"target": "ball", "stop_distance": 0.22, "then": "kick"},
                    ),
                    ToolCall(name="declare_success", arguments={"reason": "kicked"}),
                ]
            ),
            transport=Sim2DTransport(seed=6),
            detector=ColorBlobDetector(),
            runs_dir=tmp_path,
            trace=seen.append,
        )
    )
    assert result.outcome == "success", result.reason

    opened = next(
        i for i, e in enumerate(seen) if e.kind == "verb_start" and e.data["name"] == "approach_and"
    )
    closed = next(
        i for i, e in enumerate(seen) if e.kind == "verb_end" and e.data["name"] == "approach_and"
    )
    assert seen[opened].data["nested"] is False and seen[closed].data["nested"] is False

    inside = seen[opened + 1 : closed]
    assert all(e.data["nested"] is True for e in inside if e.kind in ("verb_start", "verb_end"))
    children = [e for e in inside if e.kind == "verb_end"]
    assert [e.data["name"] for e in children] == ["go_to", "kick"]

    sent = sum(sum(e.data["intents"].values()) for e in children)
    assert sent > 0, "the children really did drive the robot"
    assert sum(seen[closed].data["intents"].values()) >= sent


async def test_the_log_callback_still_gets_the_lines_that_only_it_had(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    """`log` is a contract other callers rely on: the flock's member records, the MCP
    logger, and tests that assert on what a run said. The trace observes it, never replaces
    it."""
    # a v1 task may allow more than it needs; a verb this body lacks is dropped with a line
    hello_duck.frontmatter.duck = 1
    hello_duck.frontmatter.requires = ["quack"]
    hello_duck.frontmatter.verbs.allow = ["quack", "walk", "stop", "fly"]
    lines: list[str] = []
    seen: list[Any] = []
    await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=FakeProvider.for_duck("hello-world"),
            transport=MockTransport(),
            runs_dir=tmp_path,
            log=lines.append,
            trace=seen.append,
        )
    )
    assert any("does not have fly" in line for line in lines)
    notes = [e.data["text"] for e in seen if e.kind == "note"]
    assert any("does not have fly" in note for note in notes)


def test_intents_are_buffered_until_the_next_event_flushes_them(tmp_path: Path) -> None:
    """The flush was a syscall on the event loop between two deadman resends of a steering
    verb. Intents ride the buffer; anything else, `verb_end` included, puts them on disk."""
    transcript = Transcript(tmp_path)
    path = transcript.path
    try:
        for _ in range(20):
            transcript.write("intent", intent="move", accepted=True)
        assert path.read_text(encoding="utf-8") == "", "an intent must not reach the disk alone"
        transcript.write("verb_end", name="walk", outcome="ok")
        assert len(Transcript.read(path)) == 21, "the verb ending must flush every intent"
    finally:
        transcript.close()


class CapturingProvider:
    """Records the system prompt and every observation, then declares success."""

    name = "capturing"
    model = "test"
    supports_vision = False

    def __init__(self) -> None:
        self.systems: list[str] = []
        self.observations: list[str] = []

    async def step(
        self, system: str, history: list[Exchange], tools: list[dict[str, Any]]
    ) -> ProviderTurn:
        self.systems.append(system)
        self.observations.append(history[-1].observation.text)
        return ProviderTurn(
            tool_calls=[ToolCall(name="declare_success", arguments={"reason": "done"})]
        )


async def test_the_stand_ins_a_robot_declares_are_told_to_the_model(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    """`extras.assumptions` is what a backend says quackd is standing in for. It reached the
    transcript and `FakeProvider`, and no further: every real provider sends `obs.text`, which
    is built from `state.summary()`, and neither had a branch for it. So the docs said a
    transcript never implies more than happened while the model was told nothing at all.
    """
    from quackd.transport.base import DuckState

    stand_ins = [
        "kick is a scripted impulse, not the robot's own kick policy",
        "a fall is recovered by standing the model up; upstream ships no get-up policy",
    ]
    transport = MockTransport(
        states=[DuckState(policy="mock", posture="standing", extras={"assumptions": stand_ins})]
    )
    provider = CapturingProvider()
    await run_duck(
        RunConfig(duck=hello_duck, provider=provider, transport=transport, runs_dir=tmp_path)
    )
    system = provider.systems[0]
    assert "## What is a stand-in on this robot" in system
    for sentence in stand_ins:
        assert sentence in system, "verbatim, in the robot's own words"
    # and the observation points at them, for a pilot with no system prompt at all (MCP)
    assert "stand-ins=2-listed-in-extras.assumptions" in provider.observations[0]


async def test_a_robot_that_claims_no_stand_ins_gets_no_such_section(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    provider = CapturingProvider()
    await run_duck(
        RunConfig(duck=hello_duck, provider=provider, transport=MockTransport(), runs_dir=tmp_path)
    )
    assert "stand-in" not in provider.systems[0]
    assert "stand-ins=" not in provider.observations[0]


ARM_DUCK = """\
---
duck: {version}
name: lift-the-mug
description: Pick up the mug.
verbs:
  allow: [observe, report_state, stop]
success: [The mug is up.]
{block}---
# Task
Pick it up.
"""
_OVERRIDE = """datasheet:
  payload_kg: {value: 0.3, confidence: measured, source: weighed with the printed gripper}
"""


async def test_a_task_files_datasheet_reaches_the_prompt_the_executor_and_the_record(
    tmp_path: Path,
) -> None:
    from quackd.adapters.factory import make_adapter
    from quackd.duckfile.parser import parse_duck_text

    provider = CapturingProvider()
    adapter = make_adapter("lerobot:mock")
    loop = AgentLoop(
        RunConfig(
            duck=parse_duck_text(ARM_DUCK.format(version=2, block=_OVERRIDE)),
            provider=provider,
            transport=adapter,
            runs_dir=tmp_path,
        )
    )
    result = await loop.run()
    assert result.outcome == "success", result.reason

    assert (
        "0.3 kg (measured: the task file, weighed with the printed gripper)" in provider.systems[0]
    )
    assert "0.5 kg (estimate" not in provider.systems[0], "the vendor figure was corrected"
    sheet = loop.executor.manifest.datasheet if loop.executor.manifest else None
    assert sheet is not None and sheet.payload_kg is not None and sheet.payload_kg.value == 0.3
    start = Transcript.read(result.run_dir / "transcript.jsonl")[0]
    assert start["robot"]["datasheet"]["payload_kg"]["source"].startswith("the task file")


async def test_without_a_correction_the_body_speaks_for_itself(tmp_path: Path) -> None:
    from quackd.adapters.factory import make_adapter
    from quackd.duckfile.parser import parse_duck_text

    provider = CapturingProvider()
    result = await run_duck(
        RunConfig(
            duck=parse_duck_text(ARM_DUCK.format(version=1, block="")),
            provider=provider,
            transport=make_adapter("lerobot:mock"),
            runs_dir=tmp_path,
        )
    )
    assert result.outcome == "success", result.reason
    assert "0.5 kg (estimate: one vendor's listing)" in provider.systems[0]
    assert "task file" not in provider.systems[0]


def _verdict_call(word: str, reason: str, **extra: Any) -> ToolCall:
    return ToolCall(name="assess_task", arguments={"verdict": word, "reason": reason, **extra})


async def test_the_rule_answers_the_gate_before_its_own_first_verb(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    """The scripted pilot has no judgement of a body, so it says so and goes on. Without
    that, every keyless run in the README would stop at the gate."""
    transport = MockTransport()
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=FakeProvider(
                script=[
                    ToolCall(name="walk", arguments={"vx": 0.1, "duration_s": 1.0}),
                    ToolCall(name="declare_success", arguments={"reason": "walked"}),
                ]
            ),
            transport=transport,
            runs_dir=tmp_path,
        )
    )
    assert result.outcome == "success", result.reason
    events = Transcript.read(result.run_dir / "transcript.jsonl")
    calls = [tc["name"] for e in events if e["kind"] == "llm" for tc in e["tool_calls"]]
    assert calls == ["assess_task", "walk", "declare_success"]
    assert [i.kind for i in transport.intents if i.kind == "move"]
    assessed = next(e for e in events if e["kind"] == "assess")
    assert assessed["verdict"] == "feasible"
    assert "a rule has no judgement of the body" in assessed["reason"]


async def test_a_verb_before_any_verdict_is_refused_and_the_pilot_is_told_why(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    class Impatient:
        """A pilot that reaches for a leg before it has judged the task."""

        name = "impatient"
        model = "test"
        supports_vision = False

        def __init__(self) -> None:
            self.calls = 0

        async def step(
            self, system: str, history: list[Exchange], tools: list[dict[str, Any]]
        ) -> ProviderTurn:
            self.calls += 1
            call = (
                ToolCall(name="walk", arguments={"vx": 0.1, "duration_s": 1.0})
                if self.calls == 1
                else ToolCall(name="declare_failure", arguments={"reason": "refused"})
            )
            return ProviderTurn(tool_calls=[call])

    transport = MockTransport()
    result = await run_duck(
        RunConfig(duck=hello_duck, provider=Impatient(), transport=transport, runs_dir=tmp_path)
    )
    assert result.outcome == "failure"
    events = Transcript.read(result.run_dir / "transcript.jsonl")
    verb = next(e for e in events if e["kind"] == "verb")
    assert verb["name"] == "walk" and verb["ok"] is False
    assert "moves the body" in verb["summary"] and "assess_task" in verb["summary"]
    gate = next(e for e in events if e["kind"] == "gate" and e.get("gate") == "verdict")
    assert gate["outcome"] == "refused"
    assert [i.kind for i in transport.intents if i.kind == "move"] == []


async def test_an_infeasible_verdict_ends_the_run_before_anything_moves(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    transport = MockTransport()
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=FakeProvider(
                script=[
                    _verdict_call(
                        "infeasible",
                        "the basket looks like 3 kg of clothes and this body has no arms",
                        limits_consulted=["manipulator", "payload_kg"],
                        estimates=[
                            {
                                "object": "laundry basket",
                                "quantity": "mass_kg",
                                "value": 3.0,
                                "basis": "image",
                                "confidence": "medium",
                            }
                        ],
                        needs={"payload_kg": 3.0, "manipulator": "gripper"},
                    ),
                    ToolCall(name="walk", arguments={"vx": 0.1, "duration_s": 1.0}),
                ]
            ),
            transport=transport,
            runs_dir=tmp_path,
        )
    )
    assert result.outcome == "infeasible"
    assert result.steps == 0 and result.llm_calls == 1
    assert not result.ok
    assert "3 kg of clothes" in result.reason
    assert "No robot installed here meets needs" in result.reason, "the hint names what could"
    assert [i.kind for i in transport.intents if i.kind != "stop"] == []

    events = Transcript.read(result.run_dir / "transcript.jsonl")
    assessed = next(e for e in events if e["kind"] == "assess")
    assert assessed["verdict"] == "infeasible" and assessed["ends_run"] is True
    assert assessed["estimates"][0]["object"] == "laundry basket"
    assert assessed["needs"] == {"payload_kg": 3.0, "manipulator": "gripper"}
    assert assessed["limits_consulted"] == ["manipulator", "payload_kg"]
    summary = json.loads((result.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["outcome"] == "infeasible"


async def test_an_uncertain_verdict_asks_the_person_in_the_room(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    asked: list[str] = []

    def no(why: str) -> bool:
        asked.append(why)
        return False

    transport = MockTransport()
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=FakeProvider(
                script=[
                    _verdict_call(
                        "uncertain",
                        "the basket is out of frame, so its weight is a guess",
                        needs={"payload_kg": 2.0},
                    ),
                    ToolCall(name="walk", arguments={"vx": 0.1, "duration_s": 1.0}),
                ]
            ),
            transport=transport,
            runs_dir=tmp_path,
            decide=no,
        )
    )
    assert result.outcome == "aborted", "a person stopping the run is the kill switch's kind"
    assert "the human said no" in result.reason
    assert "out of frame" in result.reason
    assert [i.kind for i in transport.intents if i.kind != "stop"] == []
    assert len(asked) == 1
    assert "payload_kg=2" in asked[0] and "not sure" in asked[0]
    events = Transcript.read(result.run_dir / "transcript.jsonl")
    assert next(e for e in events if e["kind"] == "assess")["human"] == "no_go"


async def test_a_person_who_says_go_clears_the_gate(hello_duck: DuckFile, tmp_path: Path) -> None:
    transport = MockTransport()
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=FakeProvider(
                script=[
                    _verdict_call("uncertain", "cannot see the thing from here"),
                    ToolCall(name="walk", arguments={"vx": 0.1, "duration_s": 1.0}),
                    ToolCall(name="declare_success", arguments={"reason": "walked"}),
                ]
            ),
            transport=transport,
            runs_dir=tmp_path,
            decide=lambda _why: True,
        )
    )
    assert result.outcome == "success", result.reason
    assert [i.kind for i in transport.intents if i.kind == "move"]
    events = Transcript.read(result.run_dir / "transcript.jsonl")
    assert next(e for e in events if e["kind"] == "assess")["human"] == "go"


async def test_with_nobody_to_ask_the_pilot_is_told_to_decide_itself(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    transport = MockTransport()
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=FakeProvider(
                script=[
                    _verdict_call("uncertain", "cannot see the thing from here"),
                    ToolCall(name="walk", arguments={"vx": 0.1, "duration_s": 1.0}),
                    _verdict_call("feasible", "looked again: it is a tennis ball"),
                    ToolCall(name="walk", arguments={"vx": 0.1, "duration_s": 1.0}),
                    ToolCall(name="declare_success", arguments={"reason": "walked"}),
                ]
            ),
            transport=transport,
            runs_dir=tmp_path,
            decide=None,
        )
    )
    assert result.outcome == "success", result.reason
    events = Transcript.read(result.run_dir / "transcript.jsonl")
    observations = [e["text"] for e in events if e["kind"] == "observation"]
    assert any("decide yourself" in text for text in observations)
    refused = [e for e in events if e["kind"] == "verb" and not e["ok"]]
    assert refused and "assess_task" in refused[0]["summary"]
    assert [e["human"] for e in events if e["kind"] == "assess"] == [None, None]


async def test_a_later_verdict_can_still_end_the_run(hello_duck: DuckFile, tmp_path: Path) -> None:
    transport = MockTransport()
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=FakeProvider(
                script=[
                    _verdict_call("feasible", "looks light from here"),
                    ToolCall(name="quack", arguments={}),
                    _verdict_call(
                        "infeasible",
                        "close up it is a full crate, not a box",
                        needs={"payload_kg": 5.0},
                    ),
                ]
            ),
            transport=transport,
            runs_dir=tmp_path,
        )
    )
    assert result.outcome == "infeasible"
    assert result.steps == 1, "the quack ran before the pilot changed its mind"
    assert "full crate" in result.reason


async def test_a_feasible_verdict_is_held_to_its_own_datasheet(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    """A pilot may not move its body on a need its own sheet does not meet.

    `missing_needs` already held another robot's bid to its datasheet at the coordinator,
    and nothing held a pilot's verdict about its own body to its own sheet, so a `needs`
    naming a figure nobody published passed straight through. The Microduck's endurance is
    not published, and a run that says the task needs 45 minutes of it is saying, in its own
    two fields, both that it depends on that number and that the body is fine. The pilot is
    told which need is unmet and can assess again, the same way a verdict carrying `human`
    is refused rather than quietly stripped.
    """
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=FakeProvider(
                script=[
                    _verdict_call("feasible", "it can patrol", needs={"endurance_min": 45}),
                    _verdict_call("infeasible", "endurance is not published"),
                ]
            ),
            transport=MicroduckAdapter(MockTransport()),
            runs_dir=tmp_path,
        )
    )
    assert result.outcome == "infeasible", result.reason
    events = Transcript.read(result.run_dir / "transcript.jsonl")
    assessed = [e for e in events if e["kind"] == "assess"]
    assert assessed[0]["ok"] is False
    assert "endurance_min >= 45 (not published)" in assessed[0]["summary"]
    # the second one ends the run, which assess records the way infeasible always does
    assert assessed[1]["verdict"] == "infeasible"
    assert "endurance is not published" in assessed[1]["summary"]


async def test_a_need_this_body_meets_still_passes(hello_duck: DuckFile, tmp_path: Path) -> None:
    """The check refuses what the sheet does not cover, and nothing else.

    Without this, a check that refused every `needs` would pass the test above and break
    every run that fills the field in honestly. The Microduck is legged and rated for a flat
    indoor floor, so a verdict asking for exactly that is the sheet agreeing with itself.
    """
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=FakeProvider(
                script=[
                    _verdict_call(
                        "feasible",
                        "it can walk there",
                        needs={"mobility": "legged", "terrain": "indoor_flat"},
                    ),
                    ToolCall(name="declare_success", arguments={"reason": "done"}),
                ]
            ),
            transport=MicroduckAdapter(MockTransport()),
            runs_dir=tmp_path,
        )
    )
    assert result.outcome == "success", result.reason
    events = Transcript.read(result.run_dir / "transcript.jsonl")
    assessed = [e for e in events if e["kind"] == "assess"]
    assert assessed[0]["ok"] is True


async def test_a_refused_verdict_shuts_the_gate_an_earlier_one_opened(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    """A refusal that left an earlier `feasible` standing refused the words and not the motion.

    The check runs before the verdict is recorded, the way the `human` and validation refusals
    do, so a pilot already cleared for one reading of the task could name a need this body
    cannot meet and go on moving on the older verdict, with the newer and better informed one
    thrown away. That is the exact failure #24 exists to stop, one re-assessment later. So the
    refusal withdraws what was standing: nothing moves until the pilot answers again.
    """
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=FakeProvider(
                script=[
                    _verdict_call("feasible", "it walks", needs={"mobility": "legged"}),
                    ToolCall(name="quack", arguments={"text": "off we go"}),
                    _verdict_call("feasible", "a 45 minute patrol", needs={"endurance_min": 45}),
                    ToolCall(name="walk", arguments={"vx": 0.1, "duration_s": 0.1}),
                    ToolCall(name="declare_failure", arguments={"reason": "cannot judge it"}),
                ]
            ),
            transport=MicroduckAdapter(MockTransport()),
            runs_dir=tmp_path,
        )
    )
    events = Transcript.read(result.run_dir / "transcript.jsonl")
    assessed = [e for e in events if e["kind"] == "assess"]
    assert assessed[0]["ok"] is True, "the first verdict cleared the gate"
    assert assessed[1]["ok"] is False and "endurance_min >= 45" in assessed[1]["summary"]
    # and the row describes the call that was refused, not the verdict that was standing:
    # before, a refused re-assessment was written down with the earlier verdict's own word,
    # reason and needs, and read as though that one had been refused
    assert assessed[1]["verdict"] is None
    assert assessed[1]["reason"] == "a 45 minute patrol"
    assert assessed[0]["reason"] == "it walks"
    refused = [
        e
        for e in events
        if e["kind"] == "gate" and e.get("gate") == "verdict" and e.get("name") == "walk"
    ]
    assert refused, "the walk after the refusal was allowed by the withdrawn verdict"
    assert "no feasibility verdict has been recorded" in str(refused[0]["reason"])


async def test_a_refused_assessment_is_recorded_as_itself_not_as_the_standing_verdict(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    """Every refusal in `_assess` returns before recording, and the transcript row was built
    from whatever verdict happened to be standing, so a refused re-assessment was written down
    with the earlier verdict's own word, reason and needs. Read back, the row said that the
    earlier verdict had been refused, which is a different and false story. The row describes
    the call now.
    """
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=FakeProvider(
                script=[
                    _verdict_call("feasible", "it walks"),
                    ToolCall(name="assess_task", arguments={"verdict": "maybe", "reason": "hm"}),
                    ToolCall(name="declare_success", arguments={"reason": "done"}),
                ]
            ),
            transport=MicroduckAdapter(MockTransport()),
            runs_dir=tmp_path,
        )
    )
    assessed = [
        e for e in Transcript.read(result.run_dir / "transcript.jsonl") if e["kind"] == "assess"
    ]
    assert assessed[0]["ok"] is True and assessed[0]["verdict"] == "feasible"
    assert assessed[1]["ok"] is False and "invalid assess_task" in assessed[1]["summary"]
    assert assessed[1]["verdict"] is None, "the row claimed the standing verdict was refused"
    assert assessed[1]["reason"] == "hm", "and it carried the standing verdict's reason"
    # the standing verdict is untouched by an invalid re-assessment: the run went on
    assert result.outcome == "success", result.reason


async def test_yes_clears_the_doubt_a_refused_feasible_became(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    """What the check costs and does not cost under `--yes`, because docs/safety.md says so.

    A refused `feasible` leaves the pilot three answers, and `uncertain` is one of them. At a
    terminal `--yes` answers that with go, on purpose and documented, so the same unmet need
    reaches the body one word later. This is not a hole the check should close: `--yes` is a
    person saying they have read the contract, and ADR-0032 puts a reachable human above a
    flag. What the check buys here is the record. The refusal and the doubt it became are both
    in the transcript, where a silent `feasible` left nothing at all.
    """
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=FakeProvider(
                script=[
                    _verdict_call("feasible", "it can patrol", needs={"endurance_min": 45}),
                    _verdict_call(
                        "uncertain", "endurance is not published", needs={"endurance_min": 45}
                    ),
                    ToolCall(name="quack", arguments={"text": "hi"}),
                    ToolCall(name="declare_success", arguments={"reason": "done"}),
                ]
            ),
            transport=MicroduckAdapter(MockTransport()),
            runs_dir=tmp_path,
            decide=lambda _why: True,  # what `--yes` passes (cli.py `_yes_to_go`)
        )
    )
    assert result.outcome == "success", result.reason
    assessed = [
        e for e in Transcript.read(result.run_dir / "transcript.jsonl") if e["kind"] == "assess"
    ]
    assert assessed[0]["ok"] is False and "endurance_min >= 45" in assessed[0]["summary"]
    assert assessed[1]["ok"] is True and assessed[1]["human"] == "go"
    assert "uncertain" in assessed[1]["summary"]


async def test_an_invalid_verdict_is_refused_and_the_run_goes_on(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=FakeProvider(
                script=[
                    ToolCall(name="assess_task", arguments={"verdict": "maybe", "reason": "hm"}),
                    ToolCall(
                        name="assess_task",
                        arguments={"verdict": "feasible", "reason": "fine", "human": "go"},
                    ),
                    _verdict_call("feasible", "fine"),
                    ToolCall(name="declare_success", arguments={"reason": "done"}),
                ]
            ),
            transport=MockTransport(),
            runs_dir=tmp_path,
        )
    )
    assert result.outcome == "success", result.reason
    events = Transcript.read(result.run_dir / "transcript.jsonl")
    assessed = [e for e in events if e["kind"] == "assess"]
    assert assessed[0]["ok"] is False and "invalid assess_task" in assessed[0]["summary"]
    assert assessed[1]["ok"] is False and "only a person sets it" in assessed[1]["summary"]
    assert assessed[2]["ok"] is True


async def test_the_prompt_offers_the_tool_and_states_the_rule(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    provider = CapturingProvider()
    result = await run_duck(
        RunConfig(duck=hello_duck, provider=provider, transport=MockTransport(), runs_dir=tmp_path)
    )
    events = Transcript.read(result.run_dir / "transcript.jsonl")
    assert "assess_task" in events[0]["tools"]
    system = provider.systems[0]
    assert "Before the first verb that moves the body, call " in system
    assert "assess_task" in system
    assert "the run ends, nothing moves" in system
    # read off this duck's own allowlist, not a fixed list: hello-world allows quack, walk
    # and stop, so `observe` and the head verbs have no business in its rule line
    assert "Until then only `quack` and `stop` run." in system
    assert "`observe`" not in system.split("Until then")[1].split("Assess again")[0]


def test_the_rule_line_names_a_bodys_own_read_only_verb() -> None:
    """The gate honours `Verb.read_only` since #26, so the sentence that tells a pilot what
    runs before the verdict has to be read off the body. A third-party `locate` that only
    looks belongs in it, the `reach` beside it does not, and `stop` is there whether or not
    the contract listed it."""
    from quackd.agent.prompts import before_verdict_clause, build_system_prompt
    from quackd.duckfile.parser import parse_duck_text
    from quackd.verbs.registry import NoParams, Verb, VerbResult

    async def noop(_ctx: object, _p: object) -> VerbResult:
        return VerbResult.success("ok")

    verbs = [
        Verb("locate", "where a thing is", noop, NoParams, read_only=True),
        Verb("reach", "move a hand to it", noop, NoParams),
    ]
    assert before_verdict_clause(verbs) == "only `locate` and `stop` run"
    assert before_verdict_clause([]) == "only `stop` runs", "the brake is never gated"

    duck = parse_duck_text(
        """---
duck: 0
name: t
description: d
verbs:
  allow: [locate, reach]
success: [x]
---
# Task
x
"""
    )
    rule = next(
        line
        for line in build_system_prompt(duck, verbs, "mock").splitlines()
        if "Until then" in line
    )
    assert "Until then only `locate` and `stop` run." in rule
    assert "`reach`" not in rule.split("Until then")[1]


# ── the rest pose: the arm is put down however the run ended ────────────────────────────


ARM_REST = {
    "shoulder_pan": 45.0,
    "shoulder_lift": -40.0,
    "elbow_flex": 20.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
}
"""Somewhere the mock arm does not already start, so every rest move here is a real move
rather than a reading of `already`."""


def _arm_duck() -> DuckFile:
    """A task an arm can run: it looks, it reads its own state, it stops."""
    from quackd.duckfile.parser import parse_duck_text

    return parse_duck_text(ARM_DUCK.format(version=1, block=""))


async def test_a_task_the_arm_cannot_run_still_puts_the_arm_down_before_letting_go(
    tmp_path: Path,
) -> None:
    """The window between the connect and the first step, which the run's own `finally` does
    not cover because it does not exist yet.

    A task that needs a verb this build does not have is refused after the arm is connected
    and holding. Before this the process exited there with the arm energised, wherever it
    happened to be standing, and nothing said so: the exact failure the rest pose exists to
    prevent, reached by a task file rather than by a run that ended.

    Nothing is asked of the pilot: this is refused before the first call, so a run that could
    never have worked also costs nothing."""
    duck = _arm_duck()
    duck.frontmatter.duck = 1
    duck.frontmatter.requires = ["fly"]
    mock = LeRobotMock(rest_pose=ARM_REST)
    mock.joints["shoulder_pan"] = ARM_REST["shoulder_pan"] + 70.0  # nowhere anybody chose
    provider = FakeProvider.for_duck("hello-world")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(AdapterError, match="fly"):
        await run_duck(
            RunConfig(
                duck=duck,
                provider=provider,
                transport=LeRobotAdapter(mock),
                run_dir=run_dir,
                runs_dir=tmp_path,
            )
        )

    assert mock.sequence == ["stop", "rest", "close"], mock.sequence
    assert mock.torque is False, "the arm reached its pose, so torque could be released"
    assert mock.joints["shoulder_pan"] == ARM_REST["shoulder_pan"]


def _says_no(_why: str) -> bool:
    """The person in the room, refusing. `decide` is asked when the pilot says it is unsure."""
    return False


class ExplodingProvider:
    """A pilot whose call raises whatever it was handed."""

    name, model, supports_vision = "exploding", "test", False

    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def step(self, system: Any, history: Any, tools: Any) -> ProviderTurn:
        raise self.error


class StallingProvider:
    """A pilot that never answers, so the run can be cancelled while it waits."""

    name, model, supports_vision = "stalling", "test", False

    async def step(self, system: Any, history: Any, tools: Any) -> ProviderTurn:
        await asyncio.sleep(10)
        raise AssertionError("never reached")


#: Every way `run()` can leave its own `try`, and the outcome `run_end` records for it.
ENDINGS = {
    "budget": "budget",
    "cancelled": "aborted",
    "failure": "failure",
    "infeasible": "infeasible",
    "keyboard_interrupt": "aborted",
    "provider_error": "error",
    "success": "success",
    "uncertain_no_go": "aborted",
}
#: The two endings the loop re-raises after recording. `cancelled` is driven differently.
RAISES: dict[str, type[BaseException]] = {
    "keyboard_interrupt": KeyboardInterrupt,
    "provider_error": ProviderError,
}


def _ending_setup(ending: str) -> tuple[Any, dict[str, Any]]:
    """The provider, and the `RunConfig` fields, that make a run end this way."""
    if ending == "success":
        script = [ToolCall(name="declare_success", arguments={"reason": "the mug is up"})]
        return FakeProvider(script=script), {}
    if ending == "failure":
        script = [ToolCall(name="declare_failure", arguments={"reason": "it slipped"})]
        return FakeProvider(script=script), {}
    if ending == "infeasible":
        script = [_verdict_call("infeasible", "that is a filing cabinet, not a mug")]
        return FakeProvider(script=script), {}
    if ending == "uncertain_no_go":
        script = [_verdict_call("uncertain", "the mug is out of frame")]
        return FakeProvider(script=script), {"decide": _says_no}
    if ending == "budget":
        return FakeProvider(script=[ToolCall(name="report_state")]), {"max_steps": 1}
    if ending == "provider_error":
        return ExplodingProvider(ProviderError("anthropic: rate limited")), {}
    if ending == "keyboard_interrupt":
        return ExplodingProvider(KeyboardInterrupt()), {}
    return StallingProvider(), {}


@pytest.mark.parametrize("ending", sorted(ENDINGS))
async def test_every_way_a_run_ends_returns_the_arm_to_rest(ending: str, tmp_path: Path) -> None:
    """A LeRobot arm goes limp the moment it is disconnected, so one left anywhere but its rest
    pose falls. On the bench on 2026-09-15 that is what happened at the end of every run,
    whatever the run had been doing. So the rest move sits in the `finally`, between the stop
    that holds the arm where it is and the close that lets it go, and it has to survive every
    way out: a budget, a task the pilot refused, a person saying no, a crash, and the two
    `BaseException` endings that are not `Exception` at all and so miss any branch written for
    one.
    """
    mock = LeRobotMock(rest_pose=ARM_REST)
    provider, extra = _ending_setup(ending)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = RunConfig(
        duck=_arm_duck(),
        provider=provider,
        transport=LeRobotAdapter(mock),
        run_dir=run_dir,
        runs_dir=tmp_path,
        **extra,
    )
    if ending == "cancelled":
        task = asyncio.create_task(run_duck(cfg))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif (error := RAISES.get(ending)) is not None:
        with pytest.raises(error):
            await run_duck(cfg)
    else:
        await run_duck(cfg)

    events = Transcript.read(run_dir / "transcript.jsonl")
    end = next(e for e in events if e["kind"] == "run_end")
    assert end["outcome"] == ENDINGS[ending], end["reason"]
    assert mock.sequence[-3:] == ["stop", "rest", "close"], mock.sequence
    notes = [e["text"] for e in events if e["kind"] == "note"]
    assert any("at the rest pose" in note for note in notes), notes
    assert mock.torque is False, "the arm is down, so torque could be released"
    assert mock.close_note is None, "nothing to warn about: it did not end up in mid-air"


async def test_an_arm_the_run_left_elsewhere_is_driven_home_before_the_torque_drops(
    tmp_path: Path,
) -> None:
    """The order in the teardown is not the whole of it. The arm has to actually travel: a run
    that ends with the elbow out over the desk has to put it back, and only then let go. The
    stop holds the arm where it is and the close releases it, so the move between them is the
    only thing standing between the arm and the desk."""
    mock = LeRobotMock(rest_pose=ARM_REST)
    duck = _arm_duck()
    duck.frontmatter.verbs.allow = [*duck.frontmatter.verbs.allow, "move_joints"]
    script = [
        ToolCall(name="move_joints", arguments={"positions": {"shoulder_pan": -20.0}}),
        ToolCall(name="declare_success", arguments={"reason": "moved and stopped"}),
    ]
    result = await run_duck(
        RunConfig(
            duck=duck,
            provider=FakeProvider(script=script),
            transport=LeRobotAdapter(mock),
            runs_dir=tmp_path,
        )
    )
    assert result.outcome == "success", result.reason
    assert mock.joints["shoulder_pan"] == ARM_REST["shoulder_pan"], "driven back, not left there"
    events = Transcript.read(result.run_dir / "transcript.jsonl")
    notes = [e["text"] for e in events if e["kind"] == "note"]
    assert notes[-1] == "at the rest pose", notes
    assert mock.torque is False, "and only an arm that is down has its torque released"
    assert mock.close_note is None


async def test_a_dry_run_never_moves_the_arm_to_its_rest_pose(tmp_path: Path) -> None:
    """A dry run sends nothing to the robot, and the rest move is the one thing in the teardown
    that is not narration: it is a real motion, so it is the one that has to be checked by
    name.

    The close is still real, because the link has to be let go of either way, and an arm that
    is not at its rest pose is disconnected with its torque still on. So a dry run can still
    end with the warning about torque. That sentence is about the body in the room and not
    about the run, which is exactly why it is not suppressed here."""
    mock = LeRobotMock(rest_pose=ARM_REST)
    script = [ToolCall(name="declare_success", arguments={"reason": "pretended"})]
    result = await run_duck(
        RunConfig(
            duck=_arm_duck(),
            provider=FakeProvider(script=script),
            transport=LeRobotAdapter(mock),
            runs_dir=tmp_path,
            dry_run=True,
        )
    )
    assert result.outcome == "success", result.reason
    assert "rest" not in mock.sequence, mock.sequence
    assert mock.actions == [], "no goal was sent to a joint"
    events = Transcript.read(result.run_dir / "transcript.jsonl")
    notes = [e["text"] for e in events if e["kind"] == "note"]
    assert not any(note.startswith("moving to the rest pose") for note in notes), notes
    assert not any("did not reach its rest pose" in note for note in notes), notes
    assert mock.torque is True, "and the arm it never moved is left holding itself up"


async def test_a_run_whose_first_rest_move_fails_never_asks_the_model_anything(
    tmp_path: Path,
) -> None:
    """A run starts from the pose it will end at, so the pilot improvises from the same arm
    every time. An arm that cannot get there is in an unknown place, and paying a model to
    improvise from that is worse than not starting at all: the abort happens before the first
    request, so the run costs nothing and the record still says why."""
    stalled = "shoulder_lift is at -90 with a goal of -40"
    mock = LeRobotMock(rest_pose=ARM_REST, rest_fails=stalled)
    script = [ToolCall(name="declare_success", arguments={"reason": "never asked"})]
    result = await run_duck(
        RunConfig(
            duck=_arm_duck(),
            provider=FakeProvider(script=script),
            transport=LeRobotAdapter(mock),
            runs_dir=tmp_path,
        )
    )
    assert result.outcome == "aborted", result.reason
    assert "did not reach its rest pose" in result.reason and stalled in result.reason
    assert result.llm_calls == 0, "no model was asked anything"
    assert mock.actions == [], "the move stalled, so nothing was ever sent to a joint"
    events = Transcript.read(result.run_dir / "transcript.jsonl")
    assert events[0]["kind"] == "run_start", "the abort is inside the run, not before it"
    end = next(e for e in events if e["kind"] == "run_end")
    assert end["outcome"] == "aborted" and end["llm_calls"] == 0


async def test_the_note_about_torque_left_on_reaches_the_transcript(tmp_path: Path) -> None:
    """An arm that is not at its rest pose keeps torque when quackd closes it, or it drops on
    the desk. That leaves a robot holding itself up after the run is over, and the person in
    the room has to be told so they can hold it and cut the power by hand. The transport
    records the sentence and prints nothing itself; the run is what says it out loud."""
    mock = LeRobotMock(rest_pose=ARM_REST, rest_fails="the elbow is against the table")
    lines: list[str] = []
    script = [ToolCall(name="declare_success", arguments={"reason": "never asked"})]
    result = await run_duck(
        RunConfig(
            duck=_arm_duck(),
            provider=FakeProvider(script=script),
            transport=LeRobotAdapter(mock),
            runs_dir=tmp_path,
            log=lines.append,
        )
    )
    assert result.outcome == "aborted", result.reason
    assert mock.torque is True, "an arm away from its rest pose keeps torque and does not fall"
    assert mock.close_note is not None and "torque was left on" in mock.close_note
    events = Transcript.read(result.run_dir / "transcript.jsonl")
    notes = [e["text"] for e in events if e["kind"] == "note"]
    assert mock.close_note in notes, "the warning is a line of the record"
    assert mock.close_note in lines, "`log` gets it too; that is what the CLI prints"


async def test_a_body_with_no_rest_pose_says_nothing_about_one(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    """Most bodies quackd drives have no arm to put down, and their runs have to read exactly
    as they did before there was a rest pose at all. Every golden in this suite is one of
    them."""
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=FakeProvider.for_duck("hello-world"),
            transport=MockTransport(),
            runs_dir=tmp_path,
        )
    )
    assert result.outcome == "success", result.reason
    events = Transcript.read(result.run_dir / "transcript.jsonl")
    notes = [e["text"] for e in events if e["kind"] == "note"]
    assert not any("rest pose" in note for note in notes), notes
    calls = [tc["name"] for e in events if e["kind"] == "llm" for tc in e["tool_calls"]]
    assert calls == GOLDEN_HELLO, "and the run itself is unchanged"


# ── several cameras ─────────────────────────────────────────────────────────────────────


class TwoCameraDuck(MockTransport):
    """A body with two cameras. `top` is the primary, the one the detections describe; `side`
    watches the bench from beside it. The two pictures differ, so a test can tell them apart
    on disk as well as by name."""

    async def get_frames(self) -> list[CameraFrame]:
        return [
            CameraFrame("top", Image.new("RGB", (16, 16), (200, 40, 40)), primary=True),
            CameraFrame("side", Image.new("RGB", (16, 16), (40, 40, 200))),
        ]


def test_a_body_with_one_camera_is_never_told_its_view_has_a_name() -> None:
    """The prompt's camera paragraph promises the pilot that every frame is labelled with the
    camera that took it. That promise is kept by `name_cameras`, which reads the same count,
    so the paragraph may only appear where the count is above one.

    Written against the prompt rather than against an adapter because the list is the
    adapter's: the LeRobot arm withholds it for a single camera, and another body's adapter
    may not. A pilot told `This body has 1 cameras: forward` and that its frames are labelled
    would be reading a sentence the wire never honours, and would have no way to know."""
    from quackd.agent.prompts import build_system_prompt
    from quackd_lerobot import lerobot_manifest

    duck, one = _arm_duck(), lerobot_manifest("mock", camera_names=("forward",))
    one.extras["cameras"] = ["forward"]  # an adapter that publishes it for its only camera
    said = build_system_prompt(duck, [], "mock", manifest=one)
    assert "1 cameras" not in said, said[said.find("camera") - 80 :][:240]
    assert "labelled with the name of the camera" not in said

    two = lerobot_manifest("mock", camera_names=("top", "side"))
    both = build_system_prompt(duck, [], "mock", manifest=two)
    assert "This body has 2 cameras: top, side" in both
    assert "top is the primary" in both


class OneCameraLeft(MockTransport):
    """A two-camera body whose primary lens has stalled: `camera_keys` still names both, and
    only `side` answers. This is what `LeRobotReal.get_frames` produces when a webcam stops
    giving frames, and it is the state where getting the naming wrong is worst."""

    camera_keys = ("top", "side")

    async def get_frames(self) -> list[CameraFrame]:
        return [CameraFrame("side", Image.new("RGB", (16, 16), (40, 40, 200)))]


async def test_the_one_lens_left_on_a_two_camera_body_still_says_which_one_it_is(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    """The dangerous case, and the one that is easy to write wrong: count the pictures that
    arrived and a two-camera arm down to one lens looks exactly like a one-camera arm, so the
    survivor goes out bare. It must not. The `camera:` detections line is measured off the
    primary, which is the lens that died, so an unnamed picture from the side camera lands
    directly under a description of a view it is not.

    The body's own camera list is what decides, and that list does not shrink when a lens
    stalls."""
    provider = SeeingProvider(ToolCall(name="declare_success", arguments={"reason": "seen"}))
    result = await run_duck(
        RunConfig(duck=hello_duck, provider=provider, transport=OneCameraLeft(), runs_dir=tmp_path)
    )
    assert result.outcome == "success", result.reason

    # the observation the loop actually built, rendered by a real provider: one picture, and
    # a label in front of it saying which lens it is
    seen = provider.observations[0]
    assert [img.name for img in seen.images] == ["side"], "only the working lens answered"
    assert seen.cameras == ["top", "side"], "the body still has two cameras"
    parts = render_messages("system", [Exchange(observation=seen)])[1]["content"]
    assert [p["type"] for p in parts] == ["text", "text", "image_url"], parts
    assert parts[1]["text"] == "camera side:", "the surviving lens went out unnamed"

    events = Transcript.read(result.run_dir / "transcript.jsonl")
    record = next(e for e in events if e["kind"] == "frame")
    assert record["camera"] == "side" and record["path"].endswith("0000-side.png"), record

    observation = next(e for e in events if e["kind"] == "observation")
    assert "cameras: top (detections above), side" in observation["text"], (
        "the body still has two cameras, and the pilot is told so"
    )


class SeeingProvider:
    """A pilot that can see, and writes down which camera every picture in the request came
    from, exchange by exchange.

    `attachments` is the same reading taken of the pictures that came with the task, which is
    a different list for a different reason: a camera frame is what the body sees now and is
    trimmed away as it ages, and a `--image` is what the task is about and never is. Reading
    both off the same request is the only way to tell the two apart at the point where the
    trim could confuse them.

    `vision` is an INSTANCE attribute, exactly as `FakeProvider(vision=...)` sets one, because
    that is what a pilot that cannot take an image looks like to the loop."""

    name = "seeing"
    model = "test"
    supports_vision = True

    def __init__(self, *script: ToolCall, vision: bool = True) -> None:
        self.script = list(script)
        self.requests: list[list[list[str]]] = []
        self.attachments: list[list[list[str]]] = []
        self.systems: list[str] = []
        self.observations: list[Observation] = []
        self.calls = 0
        self.supports_vision = vision

    async def step(
        self, system: str, history: list[Exchange], tools: list[dict[str, Any]]
    ) -> ProviderTurn:
        self.requests.append([[img.name for img in ex.observation.images] for ex in history])
        self.attachments.append(
            [[img.name for img in ex.observation.attachments] for ex in history]
        )
        self.systems.append(system)
        self.observations.extend(ex.observation for ex in history)
        call = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return ProviderTurn(tool_calls=[call])


async def test_every_camera_frame_reaches_the_provider_and_the_transcript(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    """Two unlabelled pictures in one request are two views of a room with nothing to say which
    is which, and two frames written to one `0000.png` are one view lost. So the camera's name
    travels with its picture the whole way: into the request, into the file name, and into the
    record's own `camera` field, while the step number still says which turn it was."""
    provider = SeeingProvider(
        ToolCall(name="quack", arguments={"text": "hi"}),
        ToolCall(name="declare_success", arguments={"reason": "seen"}),
    )
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=provider,
            transport=TwoCameraDuck(),
            runs_dir=tmp_path,
        )
    )
    assert result.outcome == "success", result.reason
    assert provider.requests[0][-1] == ["top", "side"], "both cameras, the primary first"

    frames = result.run_dir / "frames"
    top, side = frames / "0000-top.png", frames / "0000-side.png"
    assert top.exists() and side.exists(), sorted(p.name for p in frames.iterdir())
    assert top.read_bytes() != side.read_bytes(), "two views, not one picture written twice"

    events = Transcript.read(result.run_dir / "transcript.jsonl")
    records = [e for e in events if e["kind"] == "frame"]
    assert [r["camera"] for r in records[:2]] == ["top", "side"]

    observation = next(e for e in events if e["kind"] == "observation")
    assert "cameras: top (detections above), side" in observation["text"]

    requests = [e for e in events if e["kind"] == "llm_request"]
    assert requests[0]["with_image"] == 1 and requests[0]["images"] == 2
    assert all(r["images"] == 2 * r["with_image"] for r in requests), requests


async def test_only_the_last_n_exchanges_keep_their_images(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    """`keep_images_for_last_n` is the only thing bounding what a run costs in pictures: every
    turn adds one per camera, and a run of twenty that sent them all would send forty. N counts
    exchanges rather than images, so a second camera doubles the bill and the bound still
    holds. The trim is a copy, so the run's own history keeps every picture it was shown."""
    hello_duck.frontmatter.budgets = Budgets()
    keep = 3
    provider = SeeingProvider(
        *[ToolCall(name="quack", arguments={"text": "hi"})] * 4,
        ToolCall(name="declare_success", arguments={"reason": "done"}),
    )
    loop = AgentLoop(
        RunConfig(
            duck=hello_duck,
            provider=provider,
            transport=TwoCameraDuck(),
            runs_dir=tmp_path,
            keep_images_for_last_n=keep,
        )
    )
    result = await loop.run()
    assert result.outcome == "success", result.reason

    last = provider.requests[-1]
    assert len(last) == 5, "five exchanges, the last of them the one being answered"
    assert last[-keep:] == [["top", "side"]] * keep, "the newest keep both views"
    assert last[:-keep] == [[], []], "and the older ones carry no picture at all"
    assert all(ex.observation.images for ex in loop.history), (
        "the run's own history keeps every picture; only the request is trimmed"
    )

    events = Transcript.read(result.run_dir / "transcript.jsonl")
    requests = [e for e in events if e["kind"] == "llm_request"]
    assert requests[-1]["with_image"] == keep and requests[-1]["images"] == 2 * keep


# ── the pictures that came with the task ────────────────────────────────────────────────
#
# `quackd run --image sketch.png` hands a file to the *task* rather than to the robot, and
# everything below turns on that difference. A camera frame is perception: it arrives every
# step, it ages, and the trim above drops it. One of these is none of those things. It is
# fixed for the whole run, it is what the task is about, and no lens on the body can answer
# "draw what is in the picture". So it rides the first observation and no other, nothing
# trims it, and the bytes the model was sent are written beside the transcript rather than
# the file they were made from.


def _picture_file(tmp_path: Path, name: str, colour: tuple[int, int, int]) -> str:
    """One file for `--image` to name, on disk, the way a person's sketch is.

    Every test here goes through `load_task_images` rather than building a `NamedPng` by
    hand, because the re-encode to PNG is what decides both what the run sends and what lands
    in `images/`, and a test that skipped it would be asserting about bytes no run has."""
    path = tmp_path / name
    Image.new("RGB", (32, 24), colour).save(path)
    return str(path)


async def test_the_task_pictures_ride_the_first_turn_and_no_later_one(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    """Sent once, because they never change. Repeating them every step would pay for the same
    picture on every request of a forty turn run, and the first turn is the one that most
    needs them: it is the turn the verdict gate is answered from, and a pilot deciding whether
    this body can do the task has to see what it is being asked about.

    The check that matters is the second half: every later request still carries the exchange
    that holds them, and that exchange must not have grown a copy."""
    pictures = load_task_images(
        [
            _picture_file(tmp_path, "sketch.png", (220, 40, 40)),
            _picture_file(tmp_path, "plan.png", (40, 220, 40)),
        ]
    )
    provider = SeeingProvider(
        _verdict_call("feasible", "a duck can quack about a picture"),
        ToolCall(name="quack", arguments={"text": "a circle"}),
        ToolCall(name="declare_success", arguments={"reason": "said what is in it"}),
    )
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=provider,
            transport=MockTransport(),
            runs_dir=tmp_path / "runs",
            task_images=pictures,
        )
    )
    assert result.outcome == "success", result.reason
    assert provider.attachments[0] == [["sketch.png", "plan.png"]], "on the first turn"
    assert provider.attachments[-1] == [["sketch.png", "plan.png"], [], []], (
        "and on that same first exchange three turns later, and on no other"
    )
    assert all(request[0] == ["sketch.png", "plan.png"] for request in provider.attachments)
    assert all(names == [] for request in provider.attachments for names in request[1:])


async def test_the_trim_that_drops_old_frames_never_drops_a_task_picture(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    """The trim is what bounds a run's picture cost, and it counts exchanges: keep the last N
    and strip the rest. A task picture lives on the oldest exchange there is, so a trim that
    did not know the difference would throw away the one picture the task is about on turn
    three and leave the pilot drawing from memory for the other thirty-seven.

    Both lists are read off the same five requests, because the point is that they diverge:
    the camera frames thin out as they age and the sketch does not move."""
    hello_duck.frontmatter.budgets = Budgets()
    pictures = load_task_images([_picture_file(tmp_path, "sketch.png", (220, 40, 40))])
    keep = 2
    provider = SeeingProvider(
        _verdict_call("feasible", "a duck can quack about a picture"),
        *[ToolCall(name="quack", arguments={"text": "still here"})] * 3,
        ToolCall(name="declare_success", arguments={"reason": "said what is in it"}),
    )
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=provider,
            transport=TwoCameraDuck(),
            runs_dir=tmp_path / "runs",
            keep_images_for_last_n=keep,
            task_images=pictures,
        )
    )
    assert result.outcome == "success", result.reason

    last = provider.requests[-1]
    assert len(last) == 5, "five exchanges, the last of them the one being answered"
    assert last == [[], [], [], ["top", "side"], ["top", "side"]], "the old frames went"
    assert provider.attachments[-1] == [["sketch.png"], [], [], [], []], "the sketch stayed"
    assert all(request[0] == ["sketch.png"] for request in provider.attachments), (
        "every request of the run, not just the ones inside the window"
    )

    events = Transcript.read(result.run_dir / "transcript.jsonl")
    requests = [e for e in events if e["kind"] == "llm_request"]
    assert [r["task_pictures"] for r in requests] == [1] * 5, "and the record counts it too"
    assert requests[-1]["with_image"] == keep


async def test_a_task_picture_is_written_beside_the_transcript_as_the_model_got_it(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    """They go in their own directory rather than among the frames: `frames/` is numbered by
    step and is what the robot saw, and one of these belongs to no step at all. The bytes on
    disk are the re-encoded ones the provider was handed, so somebody arguing about a run
    afterwards is looking at the picture the pilot looked at rather than at the source file it
    was made from, which may since have been edited or deleted.

    The record is written before `run_start`, so a reader of the transcript meets the pictures
    the task is about before the run that was given them."""
    pictures = load_task_images(
        [
            _picture_file(tmp_path, "sketch.png", (220, 40, 40)),
            _picture_file(tmp_path, "plan.png", (40, 220, 40)),
        ]
    )
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=FakeProvider.for_duck("hello-world", vision=True),
            transport=MockTransport(),
            runs_dir=tmp_path / "runs",
            task_images=pictures,
        )
    )
    assert result.outcome == "success", result.reason

    events = Transcript.read(result.run_dir / "transcript.jsonl")
    saved = [e for e in events if e["kind"] == "task_image"]
    assert events[0]["kind"] == "task_image", "before the run that was given them"
    assert [e["name"] for e in saved] == ["sketch.png", "plan.png"]
    # `Path` rather than the string: the record stores a relative path, and this suite runs on
    # a machine whose separator is a backslash
    assert [Path(e["path"]).as_posix() for e in saved] == [
        "images/00-sketch.png",
        "images/01-plan.png",
    ]
    for record, picture in zip(saved, pictures, strict=True):
        written = result.run_dir / Path(record["path"])
        assert written.read_bytes() == picture.png, "the bytes that went on the wire"
        assert record["bytes"] == len(picture.png)


async def test_the_record_says_which_pictures_the_run_was_given_and_how_often(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    """Two numbers, in the two places a reader looks. `run_start` names what the command line
    handed this run, once, at the top where the model and the robot are named. Every
    `llm_request` says how many rode that particular request, beside the count of camera
    frames, because that is the line somebody reads when a bill is larger than they expected
    and the two kinds of picture cost the same."""
    pictures = load_task_images(
        [
            _picture_file(tmp_path, "sketch.png", (220, 40, 40)),
            _picture_file(tmp_path, "plan.png", (40, 220, 40)),
        ]
    )
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=FakeProvider.for_duck("hello-world", vision=True),
            transport=MockTransport(),
            runs_dir=tmp_path / "runs",
            task_images=pictures,
        )
    )
    assert result.outcome == "success", result.reason

    events = Transcript.read(result.run_dir / "transcript.jsonl")
    start = next(e for e in events if e["kind"] == "run_start")
    assert start["images"] == ["sketch.png", "plan.png"]
    requests = [e for e in events if e["kind"] == "llm_request"]
    assert requests, "a run that asked nothing would prove nothing here"
    assert all(r["task_pictures"] == 2 for r in requests), [r["task_pictures"] for r in requests]


async def test_the_prompt_names_every_picture_that_came_with_the_task(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    """A picture arrives in the same message as a photograph of the room, so the pilot is told
    in words which is which and what each one is called. The names are the ones the person
    typed, because the task refers to a picture by what it shows and the model has to be able
    to match the sentence to the file.

    A run with no `--image` must read exactly as it did before there was a flag for it, which
    is every golden in this suite, so the second half of this asserts the section is absent."""
    pictures = load_task_images(
        [
            _picture_file(tmp_path, "sketch.png", (220, 40, 40)),
            _picture_file(tmp_path, "plan.png", (40, 220, 40)),
        ]
    )
    provider = SeeingProvider(ToolCall(name="declare_success", arguments={"reason": "seen"}))
    await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=provider,
            transport=MockTransport(),
            runs_dir=tmp_path / "runs",
            task_images=pictures,
        )
    )
    system = provider.systems[0]
    assert "## The pictures that came with this task" in system
    assert (
        "2 pictures were handed to this task on the command line: `sketch.png`, `plan.png`"
        in system
    )
    assert "task picture NAME:" in system, "the label it will actually see in front of each"

    # and the singular, which is the ordinary `--image sketch.png` and was never asserted: a
    # section that said "1 pictures were handed" would have passed every check above
    one = SeeingProvider(ToolCall(name="declare_success", arguments={"reason": "seen"}))
    await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=one,
            transport=MockTransport(),
            runs_dir=tmp_path / "one",
            task_images=pictures[:1],
        )
    )
    alone = one.systems[0]
    assert "## The picture that came with this task" in alone
    assert "1 picture was handed to this task on the command line: `sketch.png`" in alone
    assert "It is in your first turn" in alone and "it stays in front of you" in alone
    assert "pictures" not in alone.split("## The picture that came")[1].split("## ")[0], (
        "one picture is spoken of in the singular throughout its own section"
    )

    bare = SeeingProvider(ToolCall(name="declare_success", arguments={"reason": "seen"}))
    await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=bare,
            transport=MockTransport(),
            runs_dir=tmp_path / "bare",
        )
    )
    assert "came with this task" not in bare.systems[0], "and a run with none says nothing"


async def test_a_pilot_that_cannot_see_is_sent_no_picture_and_the_run_says_so(
    hello_duck: DuckFile, tmp_path: Path
) -> None:
    """The CLI refuses `--image` for a pilot with no vision before a run directory exists, so
    this is the loop being asked directly: MCP, a flock, a test, anything that builds a
    `RunConfig` itself. Dropping the pictures silently would leave a run whose transcript says
    it was given a sketch and whose requests never carried one, and the argument afterwards
    would be unresolvable.

    Said once and not per turn, because it is a fact about the run rather than about the step,
    and a forty turn run would otherwise repeat it forty times."""
    lines: list[str] = []
    pictures = load_task_images([_picture_file(tmp_path, "sketch.png", (220, 40, 40))])
    provider = SeeingProvider(
        _verdict_call("feasible", "a duck can quack"),
        ToolCall(name="declare_success", arguments={"reason": "worked from the words"}),
        vision=False,
    )
    result = await run_duck(
        RunConfig(
            duck=hello_duck,
            provider=provider,
            transport=MockTransport(),
            runs_dir=tmp_path / "runs",
            task_images=pictures,
            log=lines.append,
        )
    )
    assert result.outcome == "success", result.reason
    assert provider.attachments == [[[]], [[], []]], "nothing was attached to anything"

    blamed = [line for line in lines if "cannot see" in line]
    assert blamed == [
        "1 picture(s) came with this task and seeing test cannot see: they were not sent, "
        "and the task has to stand on its words alone"
    ], lines

    events = Transcript.read(result.run_dir / "transcript.jsonl")
    assert blamed[0] in [e["text"] for e in events if e["kind"] == "note"], "and in the record"
    start = next(e for e in events if e["kind"] == "run_start")
    assert start["images"] == ["sketch.png"], "the run was still given it"
    requests = [e for e in events if e["kind"] == "llm_request"]
    assert all(r["task_pictures"] == 0 for r in requests), "and never sent it"


# ── the arm a person places by hand ─────────────────────────────────────────────────────
#
# `quackd run --by-hand` is the rest pose above, inverted for one run: instead of starting
# from the fold, the arm is released AT the fold, a person lifts it, puts whatever the task
# needs in the gripper and presses Enter, and quackd holds whatever pose they left. Every
# test below is about a seam in that hand-off, because each of them ends with a person's
# fingers on an arm: torque going off somewhere it should not, torque coming back on while a
# hand is still there, a run that walks away from an arm nobody caught, and the hand-back at
# the end where the gripper opens before the arm folds up.
#
# The arm is `LeRobotMock(rest_pose=REST)`, which starts exactly at `REST`, so `let_go` is
# allowed from the first move. The person is `ScriptedPerson`, which is the CLI's terminal
# with the terminal taken out: nothing here calls `input()`, and nothing waits on one.

PLACED = {"shoulder_lift": -20.0, "elbow_flex": 40.0, "wrist_flex": 15.0, "gripper": 35.0}
"""Where the person leaves the arm: lifted off the fold, wrist cocked, gripper half shut on
whatever they put in it. Nothing the arm would ever reach on its own, so the first
observation's joints could only have come from a hand."""


class ScriptedPerson:
    """Somebody standing at the arm, as the loop's `HandOff` protocol sees them.

    The real one is the CLI's `_TerminalHandOff`, which prints and then waits on the key
    thread for Enter. This one answers from a script and, when it answers yes to the first
    ask, moves the mock's joints the way a hand would. `answers` is read in order and the last
    entry repeats, so `[True, False]` is a person who places the arm and then walks off before
    the run is over.

    `takes_s` is their own time, spent on the robot's clock rather than the wall's, which is
    what makes the budget restart testable at all. `on_wait` is the hook for the two endings
    that are not an answer: a Ctrl-C landing in the middle of the wait, and a cancellation
    landing in the hand-back."""

    def __init__(
        self,
        mock: LeRobotMock,
        *,
        places: dict[str, float] | None = None,
        answers: Sequence[bool] = (True, True),
        takes_s: float = 0.0,
        on_wait: Any = None,
    ) -> None:
        self.mock = mock
        self.places = dict(places or {})
        self.answers = list(answers)
        self.takes_s = takes_s
        self.on_wait = on_wait
        self.said: list[str] = []
        self.asked: list[tuple[str, float | None, bool]] = []
        self.waits = 0

    def say(self, text: str) -> None:
        self.said.append(text)

    async def wait(
        self, text: str, *, timeout_s: float | None = None, until_abort: bool = True
    ) -> bool:
        self.asked.append((text, timeout_s, until_abort))
        self.waits += 1
        answer = self.answers[min(self.waits - 1, len(self.answers) - 1)]
        if self.takes_s:
            # the robot's own clock, which is the one the budget reads
            await self.mock.sleep(self.takes_s)
        if self.on_wait is not None:
            self.on_wait(self)
        if answer and self.waits == 1:
            self.mock.joints.update(self.places)
        return answer


def _by_hand(mock: LeRobotMock, person: ScriptedPerson, runs: Path, **extra: Any) -> RunConfig:
    """A by-hand run of the arm task, with a pilot that declares success and counts its
    calls: several of these tests are about a run that must never reach the model at all."""
    declare = ToolCall(name="declare_success", arguments={"reason": "up"})
    return RunConfig(
        duck=_arm_duck(),
        provider=FakeProvider(script=[declare]),
        transport=LeRobotAdapter(mock),
        hand_off=person,
        runs_dir=runs,
        **extra,
    )


def _stages(events: list[dict[str, Any]]) -> list[str]:
    return [e["stage"] for e in events if e["kind"] == "hand_off"]


async def test_the_run_starts_from_the_pose_a_person_put_the_arm_in(tmp_path: Path) -> None:
    """The whole hand-off, in the order it has to happen in. The rest move first, because
    `let_go` is only allowed at the recorded pose: an arm held up by torque alone falls the
    moment torque goes, so the one place it is known to be safe to release is the fold. Then
    the release, then the person, then `take_hold` on whatever they left, and only then the
    first request.

    At the end it runs backwards: the stop holds the arm where the pilot left it, the person
    is asked to take whatever is in the gripper, the gripper opens, and the arm folds up. The
    gripper opens BEFORE the fold, because an arm folding with a pencil in the jaws drives
    that pencil into the bench."""
    mock = LeRobotMock(rest_pose=REST)
    person = ScriptedPerson(mock, places=PLACED)
    result = await run_duck(_by_hand(mock, person, tmp_path))
    assert result.outcome == "success", result.reason
    assert mock.sequence == ["rest", "let_go", "take_hold", "stop", "rest", "close"]

    events = Transcript.read(result.run_dir / "transcript.jsonl")
    observation = next(e for e in events if e["kind"] == "observation")
    joints = observation["features"]["state"]["extras"]["joints"]
    assert {j: joints[j] for j in PLACED} == PLACED, "the pilot improvises from their pose"
    start = next(e for e in events if e["kind"] == "run_start")
    assert "## Where this run starts" in start["system_prompt"], "and is told so in words"

    assert person.asked[0][0] == AgentLoop.PLACE_IT
    assert person.said == [
        "holding the pose you set, you can let go. It is at elbow_flex 40, gripper 35, "
        "shoulder_lift -20, shoulder_pan 0, wrist_flex 15, wrist_roll 0"
    ], person.said
    assert person.asked[1] == (AgentLoop.HAND_IT_BACK, AgentLoop.HAND_BACK_S, False), (
        "the end ask is bounded and does not listen for the kill switch: the run is over"
    )

    opened = [e for e in events if e["kind"] == "intent" and e["intent"] == "gripper"]
    assert [e["params"] for e in opened] == [{"open": True}], "opened once, at the hand-back"
    assert _stages(events) == ["released", "held", "unloaded"]
    assert mock.joints["gripper"] == GRIPPER_OPEN, "and the thing they put in it is theirs"
    assert mock.torque is False and mock.close_note is None, "the arm is down and let go of"


async def test_the_time_a_person_takes_is_not_the_pilots_budget(tmp_path: Path) -> None:
    """`max_minutes` starts before the rest move, and an arm can then sit released while
    somebody goes to find the pencil the task is about. That waiting used to be spent out of
    the budget the model gets for the task, so a five minute task with a ten minute person in
    front of it ended on `max_minutes` without the pilot having been asked anything at all.

    The clock is restarted after `take_hold`, which is the moment the run actually begins. The
    person here takes twice the task's whole budget, on the robot's own clock, and the run
    still gets its full five minutes afterwards."""
    mock = LeRobotMock(rest_pose=REST)
    takes_s = 600.0  # ten minutes, against a task that allows five
    person = ScriptedPerson(mock, places=PLACED, takes_s=takes_s)
    loop = AgentLoop(_by_hand(mock, person, tmp_path))
    result = await loop.run()
    assert result.outcome == "success", result.reason
    assert loop.budget.limits.max_minutes * 60 < takes_s, "or this proves nothing"
    assert loop.budget.started_at == pytest.approx(takes_s), (
        "the clock was restarted at the hold, not at the rest move ten minutes earlier"
    )


async def test_nobody_places_the_arm_so_nothing_is_ever_asked_of_the_model(
    tmp_path: Path,
) -> None:
    """An arm released for a person who is not there is limp at its rest pose with nobody
    coming. There is no run from here: the pilot would improvise from a pose nobody chose, so
    the abort happens before the first request and the run costs nothing.

    The teardown still has to pick the arm back up. `stop` takes hold first when the arm is in
    somebody's hands, because a goal sent to a limp servo stops nothing, so the arm ends
    parked at the fold with torque off like every other run in this file."""
    mock = LeRobotMock(rest_pose=REST)
    person = ScriptedPerson(mock, answers=[False])
    cfg = _by_hand(mock, person, tmp_path)
    result = await run_duck(cfg)
    assert result.outcome == "aborted"
    assert result.reason == AgentLoop.NOBODY_PLACED_IT
    assert cfg.provider.calls == 0, "the model was never asked anything"  # type: ignore[attr-defined]
    assert result.llm_calls == 0

    events = Transcript.read(result.run_dir / "transcript.jsonl")
    assert not [e for e in events if e["kind"] == "llm_request"]
    assert _stages(events) == ["released"], "released, and never held"
    assert mock.sequence == ["rest", "let_go", "take_hold", "stop", "rest", "close"], (
        "the take_hold is the stop picking a limp arm back up, not a pose anybody set"
    )
    assert mock.joints == dict(REST), "nobody moved it"
    assert mock.torque is False and mock.close_note is None, "parked, and let go of"


async def test_a_ctrl_c_during_the_wait_ends_the_run_as_the_kill_switch(tmp_path: Path) -> None:
    """A person who presses Ctrl-C instead of Enter has not failed to place the arm: they have
    stopped the run. Both come back from the wait as False, so the flag is what tells them
    apart, and the record has to say the one that is true.

    `_hand_over` asks the loop for its own reason rather than guessing at one, which is the
    same reason the main loop does: a heartbeat that died and a flock member that broke both
    set this flag too, and neither is a kill switch nobody pressed."""
    mock = LeRobotMock(rest_pose=REST)
    person = ScriptedPerson(mock, answers=[False])
    loop = AgentLoop(_by_hand(mock, person, tmp_path))
    person.on_wait = lambda _p: loop.executor.abort.set()
    result = await loop.run()
    assert result.outcome == "aborted"
    # the kill switch's own words and not the wait's: somebody was there and stopped the run,
    # which is a different ending from nobody having come back to place the arm
    assert result.reason == "kill switch"
    assert result.reason != AgentLoop.NOBODY_PLACED_IT
    assert loop.budget.llm_calls == 0
    assert mock.torque is False and mock.close_note is None, "and the arm is still parked"


async def test_an_arm_with_no_recorded_rest_pose_is_never_handed_over(tmp_path: Path) -> None:
    """There is nowhere it is known to be safe to let go of it. The recorded pose is the one
    place the arm holds itself up without torque, and releasing it anywhere else drops it, so
    an arm that has never had its pose recorded is refused the hand-off rather than released
    somewhere hopeful.

    The CLI refuses this before connecting, with `quackd robot rest-pose NAME` as the fix.
    This is the same refusal reached through the loop, which is where an MCP session or a test
    arrives, and the run ends before the model is asked anything."""
    mock = LeRobotMock()  # no rest pose recorded for this arm
    person = ScriptedPerson(mock, places=PLACED)
    cfg = _by_hand(mock, person, tmp_path)
    result = await run_duck(cfg)
    assert result.outcome == "aborted"
    assert "the arm was not handed over" in result.reason
    assert "no rest pose is recorded for this arm" in result.reason
    assert "quackd robot rest-pose" in result.reason, "and the words say how to fix it"
    assert cfg.provider.calls == 0  # type: ignore[attr-defined]
    assert person.asked == [], "nobody was asked to place an arm that was never released"

    events = Transcript.read(result.run_dir / "transcript.jsonl")
    assert [e["how"] for e in events if e["kind"] == "hand_off"] == ["refused"]
    assert mock.sequence == ["let_go", "stop", "close"], "no rest move: there is no pose"


async def test_an_arm_that_slipped_as_torque_came_on_is_not_run_from(tmp_path: Path) -> None:
    """`take_hold` writes the present position as the goal, enables torque, writes it again
    and re-reads: a joint that moved more than the tolerance while that happened is an arm
    holding a pose nobody chose, and a few degrees at the shoulder is a hand's width at the
    gripper. So it refuses, and the run ends rather than improvising from it.

    Two things to watch. The close note first: `_in_hand` is cleared as soon as torque is
    confirmed on, BEFORE the pose is judged, so this refusal must not end with the arm being
    described as limp in somebody's hands. Torque did come on, the arm is holding itself, and
    it folds up under its own power like any other.

    And the gripper. The person was told to load it before they pressed Enter, so the jaws may
    be holding something whatever the arm then did with the pose, and this ending must still
    ask for it back before the fold. It used to skip that: the hand-back was guarded on the
    hold having succeeded rather than on the person having been asked, so exactly the ending
    most likely to have a payload in the jaws, an arm that sagged under one, folded on it."""
    mock = LeRobotMock(rest_pose=REST)
    mock.hold_slips = {"elbow_flex": TOL_DEG * 2}
    person = ScriptedPerson(mock, places=PLACED)
    result = await run_duck(_by_hand(mock, person, tmp_path))
    assert result.outcome == "aborted"
    assert "the arm is not holding the pose you set" in result.reason
    assert "moved as torque came on" in result.reason

    events = Transcript.read(result.run_dir / "transcript.jsonl")
    notes = [e["text"] for e in events if e["kind"] == "note"]
    assert any(note.startswith("the arm did not take hold:") for note in notes), notes
    assert _stages(events) == ["released", "held", "unloaded"], (
        "asked for, refused, and the gripper still emptied before the arm folded on it"
    )
    opened = [e for e in events if e["kind"] == "intent" and e["intent"] == "gripper"]
    assert [e["params"] for e in opened] == [{"open": True}]
    assert mock.sequence == ["rest", "let_go", "take_hold", "stop", "rest", "close"]
    assert {j: mock.joints[j] for j in rest_goal(REST)} == rest_goal(REST), "folded up"
    assert mock.torque is False, "and only an arm that is down has its torque released"
    assert mock.close_note is None
    assert not any("limp" in note for note in notes), "torque came on; nothing is limp"


async def test_nobody_answers_the_hand_back_so_the_gripper_stays_shut(tmp_path: Path) -> None:
    """The run is over, the arm is holding where the pilot left it, and the person who put
    something in the gripper is not in the room. The ask is bounded for exactly this: a run
    must still end when nobody comes back, and an arm cannot hold its pose for ever.

    So the gripper stays where their fingers left it and the arm folds up with whatever is in
    it. That is the cautious half of the choice: opening the jaws over an empty bench drops
    the thing, and folding with it held does not."""
    mock = LeRobotMock(rest_pose=REST)
    person = ScriptedPerson(mock, places=PLACED, answers=[True, False])
    result = await run_duck(_by_hand(mock, person, tmp_path))
    assert result.outcome == "success", result.reason

    events = Transcript.read(result.run_dir / "transcript.jsonl")
    assert not [e for e in events if e["kind"] == "intent" and e["intent"] == "gripper"]
    assert _stages(events) == ["released", "held", "skipped"]
    notes = [e["text"] for e in events if e["kind"] == "note"]
    assert "nobody unloaded the gripper, so it stays shut and the arm folds up" in notes
    assert mock.joints["gripper"] == PLACED["gripper"], "still where their fingers left it"
    assert mock.sequence == ["rest", "let_go", "take_hold", "stop", "rest", "close"]
    assert {j: mock.joints[j] for j in rest_goal(REST)} == rest_goal(REST), "and folded up"
    assert mock.torque is False and mock.close_note is None


async def test_a_cancellation_in_the_hand_back_still_finishes_the_teardown(
    tmp_path: Path,
) -> None:
    """A second Ctrl-C in this window used to raise straight through the teardown, which cost
    the arm its fold and the run its `run_end` and its summary: the process exited with the
    arm energised wherever the pilot left it, and the record stopped mid-sentence.

    Here it means "skip this and finish". The gripper stays as it is, and everything after it
    still runs: the fold, the close, the last event of the record."""
    mock = LeRobotMock(rest_pose=REST)

    def interrupt(person: ScriptedPerson) -> None:
        if person.waits == 2:  # the hand-back ask, not the one that placed the arm
            raise asyncio.CancelledError

    person = ScriptedPerson(mock, places=PLACED, on_wait=interrupt)
    result = await run_duck(_by_hand(mock, person, tmp_path))
    assert result.outcome == "success", result.reason

    events = Transcript.read(result.run_dir / "transcript.jsonl")
    assert events[-1]["kind"] == "run_end", "the record still ends"
    assert not [e for e in events if e["kind"] == "intent" and e["intent"] == "gripper"]
    assert _stages(events) == ["released", "held", "skipped"]
    notes = [e["text"] for e in events if e["kind"] == "note"]
    assert "the gripper was left as it is, and the arm still folds up" in notes
    assert mock.sequence == ["rest", "let_go", "take_hold", "stop", "rest", "close"]
    assert {j: mock.joints[j] for j in rest_goal(REST)} == rest_goal(REST), "the fold ran"
    assert mock.torque is False and mock.close_note is None
    assert (result.run_dir / "summary.json").exists()


async def test_a_dry_run_never_takes_torque_off_an_arm(tmp_path: Path) -> None:
    """The CLI refuses the two flags together, because they ask for opposite things: one
    takes torque off the arm and the other moves nothing. This is the loop holding the same
    line for every other caller, and it is the safety-critical half of the pair: a rehearsal
    that released a real arm into an empty room would put it on the floor.

    Nothing is asked of anybody either. A dry run that printed "the arm is yours" and waited
    for Enter would be asking a person to act on a robot that is not going to move."""
    mock = LeRobotMock(rest_pose=REST)
    person = ScriptedPerson(mock, places=PLACED)
    result = await run_duck(_by_hand(mock, person, tmp_path, dry_run=True))
    assert result.outcome == "success", result.reason
    assert "let_go" not in mock.sequence and "take_hold" not in mock.sequence, mock.sequence
    assert "rest" not in mock.sequence, "the rest move is a real motion too"
    assert mock.actions == [], "no goal was sent to a joint"
    assert person.asked == [] and person.said == [], "and nobody was asked to do anything"

    events = Transcript.read(result.run_dir / "transcript.jsonl")
    assert not [e for e in events if e["kind"] == "hand_off"]
    start = next(e for e in events if e["kind"] == "run_start")
    assert "## Where this run starts" not in start["system_prompt"], (
        "and the pilot is not told a person placed a body nobody touched"
    )


# ── what an adversarial pass found in the hand-off, once each ───────────────────────────


async def test_a_run_already_ending_never_releases_the_arm_to_nobody(tmp_path: Path) -> None:
    """Ctrl-C between the connect and the first turn, which is a window wide enough to hit on
    purpose: the opening rest move is inside it.

    The abort flag was not read until after the release, so quackd took torque off the arm,
    printed "the arm is yours: torque is off at its rest pose ... hold it where you want the
    run to start", and then ended the run in the same breath. Whoever read that line was being
    invited to place an arm for a run that was already over."""
    mock = LeRobotMock(rest_pose=REST)
    person = ScriptedPerson(mock, places=PLACED)
    loop = AgentLoop(_by_hand(mock, person, tmp_path))
    loop.executor.abort.set()  # the person pressed Ctrl-C while the arm was folding

    result = await loop.run()
    assert result.outcome == "aborted"
    assert "let_go" not in mock.sequence, "the arm was never de-energised"
    assert mock.torque is True or mock.joints == dict(REST), "and never left limp away from rest"
    assert not person.said, "and nobody was told to pick up an arm nobody was going to drive"


async def test_a_gripper_that_will_not_open_is_said_out_loud(tmp_path: Path) -> None:
    """The arm's backend answers a refusal rather than raising it, so the `suppress(Exception)`
    around the hand-back's gripper call was dead for exactly the failure that matters. The run
    ended `success`, the record said the gripper had been opened, and the arm folded up with
    the pencil still clamped in it while somebody stood there with their hand out."""
    mock = LeRobotMock(rest_pose=REST, refuse_kinds={"gripper"})
    person = ScriptedPerson(mock, places=PLACED)
    result = await run_duck(_by_hand(mock, person, tmp_path))
    assert result.outcome == "success", result.reason

    events = Transcript.read(result.run_dir / "transcript.jsonl")
    assert _stages(events) == ["released", "held", "stuck"], "not `unloaded`, which it was not"
    notes = [e["text"] for e in events if e["kind"] == "note"]
    assert any(n.startswith("the gripper did not open:") for n in notes), notes
    assert any("take what is in it by hand" in n for n in notes)
    assert mock.joints["gripper"] == PLACED["gripper"], "and the jaws really are still shut"


async def test_the_gripper_opens_before_the_arm_folds_and_not_after(tmp_path: Path) -> None:
    """The ordering the whole hand-back exists for, asserted as an ordering.

    Every test around this one checked that the gripper opened and that the arm folded, and
    none of them checked which happened first, which is the only part that matters: a gripper
    opened after the fold has already driven whatever was in it into the bench.

    The mock records both in one list, so this reads the sequence rather than two facts."""
    mock = LeRobotMock(rest_pose=REST)
    person = ScriptedPerson(mock, places=PLACED)
    result = await run_duck(_by_hand(mock, person, tmp_path))
    assert result.outcome == "success", result.reason

    events = Transcript.read(result.run_dir / "transcript.jsonl")
    order = [
        "gripper" if e["kind"] == "intent" and e.get("intent") == "gripper" else "fold"
        for e in events
        if (e["kind"] == "intent" and e.get("intent") == "gripper")
        or (e["kind"] == "note" and e.get("text") == "moving to the rest pose")
    ]
    # a run folds twice, once before the pilot and once after it, so what is asserted is that
    # the gripper falls between them rather than merely somewhere
    assert order == ["fold", "gripper", "fold"], (
        f"the gripper must open after the run and before the fold that would jam it: {order}"
    )
    assert mock.joints["gripper"] == GRIPPER_OPEN, "and it really did open"


async def test_a_terminal_that_goes_away_mid_wait_ends_the_run_instead_of_hanging(
    tmp_path: Path,
) -> None:
    """The ending the docs promised and the code could not reach.

    The placement wait has no clock on it on purpose, so somebody can go and find a pencil. It
    also has no keyboard once stdin is finished, which a closed terminal or a Ctrl-D produces,
    and without an answer for that the run waited for ever with the arm limp and nobody coming.

    A `HandOff` that answers no stands in for the real `KillSwitch.wait_for_enter` returning
    False on `keys_ended`; what is asserted is what the loop does with that answer, which is to
    end the run in the words for nobody having placed the arm and still put the arm down."""
    mock = LeRobotMock(rest_pose=REST)
    person = ScriptedPerson(mock, places=PLACED, answers=[False])
    result = await run_duck(_by_hand(mock, person, tmp_path))

    assert result.outcome == "aborted"
    assert "nobody placed the arm" in result.reason
    assert mock.sequence == ["rest", "let_go", "take_hold", "stop", "rest", "close"]
    assert mock.torque is False and mock.close_note is None, "and it is down and let go of"
