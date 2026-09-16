"""The trace: one event stream, and the views that must not lie about it or crash on it."""

from __future__ import annotations

import asyncio
import io
from typing import Any

import pytest
from rich.console import Console

from quackd.trace import (
    ConsoleTrace,
    LineTrace,
    TracedTransport,
    TraceEvent,
    Tracer,
    cap_lines,
    capture_sink,
    capturing,
    counting,
    fan_out,
    flock_caption,
    fmt_value,
    parse_thinking_limit,
    prompt_shown_default,
    render_call,
    render_events,
    render_lines,
    thinking_limit_default,
    trace_enabled_default,
)
from quackd.transport.base import Ack, Intent
from quackd.transport.mock import MockTransport


def events(tracer: Tracer) -> list[TraceEvent]:
    seen: list[TraceEvent] = []
    tracer.add(seen.append)
    return seen


def lines(sink_events: list[TraceEvent], **kwargs: Any) -> list[str]:
    out: list[str] = []
    view = LineTrace(lambda text, _style: out.append(text), **kwargs)
    for event in sink_events:
        view(event)
    view.flush()
    return out


# ── the tracer ──────────────────────────────────────────────────────────────────────────


def test_the_record_sink_gets_every_event_and_its_failure_is_the_runs() -> None:
    """The transcript is the record: a write that fails must not be swallowed the way a
    console that cannot print is."""

    def broken(_event: TraceEvent) -> None:
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        Tracer(record=broken).emit("run_start")


def test_an_observer_that_raises_never_ends_a_run() -> None:
    written: list[str] = []

    def broken(_event: TraceEvent) -> None:
        raise ValueError("a terminal that cannot print")

    tracer = Tracer(record=lambda e: written.append(e.kind), observers=[broken])
    tracer.emit("llm", text="hello")
    assert written == ["llm"] and tracer.dropped == 1


def test_a_payload_may_have_a_field_called_kind() -> None:
    """The intent's own kind used to collide with the event kind, and the transcript wrote
    `{"kind": "move"}` for what was an `intent` event."""
    seen: list[TraceEvent] = []
    Tracer(record=seen.append).emit("intent", intent="move", kind="not the event kind")
    assert seen[0].kind == "intent" and seen[0].data["kind"] == "not the event kind"


def test_events_are_stamped_in_order() -> None:
    tracer = Tracer()
    seen = events(tracer)
    tracer.emit("a")
    tracer.emit("b")
    assert [e.kind for e in seen] == ["a", "b"] and seen[0].t <= seen[1].t


# ── the transport verbs see ─────────────────────────────────────────────────────────────


async def test_every_intent_and_stop_is_an_event() -> None:
    tracer = Tracer()
    seen = events(tracer)
    inner = MockTransport()
    traced = TracedTransport(inner, tracer)
    await traced.send_intent(Intent.move(0.2, 0.0, 0.1))
    await traced.stop()
    assert [e.kind for e in seen] == ["intent", "intent"]
    assert seen[0].data == {
        "intent": "move",
        "params": {"vx": 0.2, "vy": 0.0, "wz": 0.1},
        "accepted": True,
        "reason": None,
        "robot_t": 0.0,  # the mock's clock only moves on `sleep`
    }
    assert seen[1].data["intent"] == "stop"
    # and the real transport really got them
    assert [i.kind for i in inner.intents] == ["move", "stop"]


async def test_a_refused_intent_says_so() -> None:
    tracer = Tracer()
    seen = events(tracer)
    traced = TracedTransport(MockTransport(refuse_kinds={"sound"}), tracer)
    ack = await traced.send_intent(Intent.sound("greet", "hi"))
    assert not ack.accepted
    assert seen[0].data["accepted"] is False and "refuses" in seen[0].data["reason"]


async def test_a_transport_that_raises_is_traced_and_still_raises() -> None:
    class Broken(MockTransport):
        async def send_intent(self, intent: Intent) -> Ack:
            raise ConnectionError("the link is gone")

    tracer = Tracer()
    seen = events(tracer)
    with pytest.raises(ConnectionError):
        await TracedTransport(Broken(), tracer).send_intent(Intent.stop())
    assert seen[0].data["accepted"] is False and "ConnectionError" in seen[0].data["reason"]


async def test_a_stop_that_raises_is_traced_and_still_raises() -> None:
    """A stop that failed is the single most important line in a trace: it is the moment the
    brake did not answer. `TracedTransport.stop` emits before it re-raises, so the caller
    still gets the exception it has to act on and the record still has the line."""

    class Broken(MockTransport):
        async def stop(self) -> None:
            raise ConnectionError("the socket is gone")

    tracer = Tracer()
    seen = events(tracer)
    with pytest.raises(ConnectionError, match="the socket is gone"):
        await TracedTransport(Broken(), tracer).stop()
    assert [e.kind for e in seen] == ["intent"], "the emitted event must not be lost to the raise"
    assert seen[0].data["intent"] == "stop"
    assert seen[0].data["accepted"] is False
    assert seen[0].data["reason"] == "ConnectionError: the socket is gone"


def test_everything_else_is_delegated() -> None:
    """Verbs probe the transport for things only some bodies have. `stop_error` is what the
    `stop` verb reads to tell "stopped" from "could not deliver a stop"."""
    inner = MockTransport()
    inner.stop_error = "the socket is gone"  # type: ignore[attr-defined]
    traced = TracedTransport(inner, Tracer())
    assert getattr(traced, "stop_error", None) == "the socket is gone"
    assert getattr(traced, "camera_error", None) is None
    assert traced.name == "mock" and traced.now() == inner.now()


def test_a_wrapper_without_its_privates_raises_attribute_error_not_recursion() -> None:
    traced = TracedTransport.__new__(TracedTransport)  # never ran __init__
    with pytest.raises(AttributeError):
        traced._inner  # noqa: B018


async def test_a_nested_verbs_intents_count_for_its_parent_too() -> None:
    """`approach_and` sends nothing itself; every intent comes from the `go_to` it runs. A
    parent that reported zero intents would be the misleading kind of true."""
    from collections import Counter

    traced = TracedTransport(MockTransport(), Tracer())
    with counting() as parent:
        await traced.send_intent(Intent.move(0.1))
        with counting() as child:
            await traced.send_intent(Intent.move(0.2))
            await traced.stop()
    assert child == Counter({"move": 1, "stop": 1})
    assert parent == Counter({"move": 2, "stop": 1})


async def test_an_intent_sent_outside_any_verb_is_counted_by_nobody() -> None:
    """The heartbeat's stop belongs to no verb: it must not land on whichever tally happens
    to be open in another task."""
    traced = TracedTransport(MockTransport(), Tracer())
    await traced.stop()  # no `counting()` block: must not raise, must count nowhere
    with counting() as tally:
        pass
    assert tally == {}


# ── rendering ───────────────────────────────────────────────────────────────────────────


def test_a_burst_of_one_intent_kind_becomes_one_line_with_its_ranges() -> None:
    """`go_to` recomputes its twist every 100 ms, so consecutive intents are never identical;
    coalescing by kind is what keeps a 20 s approach from being 200 lines."""
    tracer = Tracer()
    seen = events(tracer)
    for wz in (-0.4, 0.0, 0.35):
        tracer.emit("intent", intent="move", params={"vx": 0.2, "wz": wz}, accepted=True)
    tracer.emit("intent", intent="stop", params={}, accepted=True)
    out = lines(seen)
    assert len(out) == 2
    assert "move x3" in out[0] and "vx 0.2" in out[0] and "wz -0.4..0.35" in out[0]
    assert out[1].split() == ["->", "stop"]  # the label column is padded


def _burst(count: int, *, every: float) -> list[TraceEvent]:
    return [
        TraceEvent("intent", i * every, {"intent": "move", "params": {"vx": 0.2}, "accepted": True})
        for i in range(count)
    ]


def test_a_long_burst_is_shown_as_it_happens() -> None:
    """The only events during a twenty second approach are its own `move` intents, so a view
    that flushed only on a different event showed the verb starting and then nothing at all
    until it ended."""
    out = lines(
        [*_burst(45, every=0.1), TraceEvent("verb_end", 4.5, {"name": "go_to", "ok": True})]
    )
    bursts = [line for line in out if "move x" in line]
    assert len(bursts) == 3, out
    assert sum(int(line.split("move x")[1].split()[0]) for line in bursts) == 45
    for line in bursts:
        assert float(line.split("over ")[1].split(" s")[0]) <= 2.0
    assert "go_to" in out[-1]


def test_a_burst_that_stalls_is_not_split_by_the_count_alone() -> None:
    """A free-running simulator crosses twenty robot seconds in under two wall ones, so the
    count is the bound that matters there."""
    stalled = _burst(150, every=0.0)
    assert len(lines(stalled)) == 1
    assert len(lines(stalled, max_burst=100)) == 2


def test_the_mcp_result_keeps_one_line_per_burst() -> None:
    """That view renders when the call has already ended, so progressive lines would only
    spend the model's line cap saying the same thing ten times."""
    rendered = render_call(_burst(45, every=0.1))
    assert len([line for line in rendered if "move x" in line]) == 1
    assert "move x45" in rendered[0]


def test_a_refused_intent_is_never_folded_into_a_count() -> None:
    tracer = Tracer()
    seen = events(tracer)
    tracer.emit("intent", intent="move", params={"vx": 0.1}, accepted=True)
    tracer.emit("intent", intent="move", params={"vx": 0.1}, accepted=False, reason="too fast")
    out = lines(seen)
    assert len(out) == 2 and "REFUSED: too fast" in out[1]


def test_the_dry_run_gate_shows_a_parameter_the_model_left_null() -> None:
    """`--dry-run` promises every parameter a model would have sent. A parameter it
    explicitly left unset used to render identically to one it never named."""
    event = TraceEvent(
        "gate",
        0.0,
        {
            "name": "go_to",
            "gate": "dry_run",
            "outcome": "skipped",
            "reason": "would run go_to, sent nothing",
            "params": {"target": None, "stop_distance": 0.25},
        },
    )
    ((text, _),) = render_lines(event)
    assert "target=null" in text and "stop_distance=0.25" in text


def test_an_intent_line_still_drops_null_parameters() -> None:
    """A twist's `vy=null` on every one of two hundred burst lines is noise."""
    event = TraceEvent("intent", 0.0, {"intent": "move", "params": {"vx": 0.1, "vy": None}})
    ((text, _),) = render_lines(event)
    assert "vx=0.1" in text and "vy" not in text


def test_a_note_with_several_lines_is_indented_under_its_label() -> None:
    """A planner's multi-line log ran into the label column and was unreadable: every line
    after the first started at column zero, exactly where a reader looks for the next
    event's label, so a three-line note read as three separate events."""
    text = "planner: it can see the ball\nbearing +12 deg, 0.8 m away\nnext: go_to, then kick"
    ((rendered, style),) = render_lines(TraceEvent("note", 0.0, {"text": text}))
    first, *rest = rendered.splitlines()
    assert first.startswith("note") and first.endswith("planner: it can see the ball")
    pad = len(first) - len("planner: it can see the ball")
    assert pad == 8, "the label column"
    assert rest == [" " * pad + "bearing +12 deg, 0.8 m away", " " * pad + "next: go_to, then kick"]
    assert style == "dim"


def test_fmt_value_truncates_long_strings_and_long_reprs() -> None:
    """One `note` carrying a model's whole answer, or a params dict with a frame in it,
    would otherwise be the trace. Both are bounded, and both say where they were cut."""
    assert fmt_value("x" * 500) == repr("x" * 57 + "...")
    assert len(fmt_value("x" * 500)) == 62  # 60 characters, and the quotes repr adds
    assert fmt_value("x" * 60) == repr("x" * 60)  # exactly at the limit, untouched

    big = {f"k{i}": list(range(10)) for i in range(20)}
    assert fmt_value(big) == repr(big)[:117] + "..."
    assert len(fmt_value(big)) == 120


def test_a_burst_with_many_distinct_labels_shows_three_and_an_ellipsis() -> None:
    tracer = Tracer()
    seen = events(tracer)
    for i in range(50):
        tracer.emit("intent", intent="do", params={"skill": f"s{i}"}, accepted=True)
    (out,) = lines(seen)
    assert "s0'/'s1'/'s2'..." in out.replace('"', "'") or "s0" in out
    assert "..." in out


def test_a_write_that_fails_keeps_the_burst_for_the_next_flush() -> None:
    """`flush` used to clear the pending burst before writing it, so a write that raised
    lost the intents entirely."""
    written: list[str] = []
    failed = {"once": True}

    def write(text: str, _style: str) -> None:
        if failed["once"]:
            failed["once"] = False
            raise RuntimeError("the terminal went away")
        written.append(text)

    view = LineTrace(write)
    view(TraceEvent("intent", 0.0, {"intent": "move", "params": {"vx": 0.1}, "accepted": True}))
    with pytest.raises(RuntimeError):
        view.flush()
    view.flush()
    assert len(written) == 1 and "move" in written[0]


def test_a_renderer_bug_never_turns_a_result_into_an_internal_error() -> None:
    """`render_call` runs outside the tracer, so nothing swallows its exceptions: a
    formatting error would have failed the MCP tool call instead of answering it."""
    broken = TraceEvent("verb_end", 0.0, {"elapsed_s": "soon", "intents": {"move": 1}})
    rendered = render_call([broken])
    assert len(rendered) == 1 and "could not be rendered" in rendered[0]


def test_the_llm_line_shows_thinking_the_call_and_the_tokens() -> None:
    event = TraceEvent(
        "llm",
        0.0,
        {
            "step": 2,
            "text": "going for it",
            "thinking": "the ball is 0.4 m away, so walk first",
            "tool_calls": [{"name": "go_to", "arguments": {"target": "ball"}}],
            "usage": {"input_tokens": 1200, "output_tokens": 40},
            "usage_total": {"input_tokens": 5000, "output_tokens": 130},
            "latency_s": 1.25,
            "stop_reason": "tool_use",
        },
    )
    out = [text for text, _ in render_lines(event)]
    assert any("the ball is 0.4 m away" in line for line in out)
    assert any("go_to(target='ball')" in line for line in out)
    assert any("in=1200 out=40" in line and "latency=1.2 s" in line for line in out)


def test_long_thinking_is_cut_with_a_pointer_to_the_transcript() -> None:
    event = TraceEvent("llm", 0.0, {"thinking": "x" * 5000, "tool_calls": [], "usage": {}})
    out = [text for text, _ in render_lines(event, thinking_chars=100)]
    assert "+4900 chars in transcript.jsonl" in out[0] and len(out[0]) < 400
    full = [text for text, _ in render_lines(event, thinking_chars=None)]
    assert "transcript.jsonl" not in full[0] and len(full[0]) > 4000
    none = [text for text, _ in render_lines(event, thinking_chars=0)]
    assert not any("xxx" in line for line in none)


def test_an_llm_call_that_failed_is_a_line_too() -> None:
    event = TraceEvent("llm", 0.0, {"error": "ProviderError: rate limited", "latency_s": 3.0})
    ((text, style),) = render_lines(event)
    assert "rate limited" in text and style == "red"


def test_llm_request_says_when_there_are_more_images_than_messages() -> None:
    """A body with two cameras puts two pictures in one exchange, so the old single count
    could no longer stand for both. It is still one number wherever they agree, which is
    every one-camera run, and the second is only spelled out where they differ.

    The third case is why the fallback exists. A transcript recorded before the loop split
    the count carries `images` alone, and there it meant exchanges: `quackd trace` replays
    those run directories and `tests/golden/trace_lines.json` holds the lines they printed,
    so an old event has to render the string it always did.
    """

    def line(**counts: Any) -> str:
        data = {"step": 2, "messages": 7, "provider": "openai", "model": "gpt-5", **counts}
        ((text, _),) = render_lines(TraceEvent("llm_request", 0.0, data))
        return text

    assert line(images=4, with_image=2) == (
        "llm>    step 2: 7 messages (2 with image, 4 images) to openai gpt-5"
    )
    assert line(images=2, with_image=2) == (
        "llm>    step 2: 7 messages (2 with image) to openai gpt-5"
    )
    assert line(images=1) == "llm>    step 2: 7 messages (1 with image) to openai gpt-5", (
        "an old transcript carries no with_image and must replay line for line"
    )


def test_a_gate_says_which_rule_refused_and_why() -> None:
    event = TraceEvent(
        "gate",
        0.0,
        {"name": "kick", "gate": "allowlist", "outcome": "refused", "reason": "not allowed here"},
    )
    ((text, style),) = render_lines(event)
    assert "allowlist: refused not allowed here" in text and style == "red"


def test_the_dry_run_gate_shows_what_would_have_been_sent() -> None:
    event = TraceEvent(
        "gate",
        0.0,
        {
            "name": "walk",
            "gate": "dry_run",
            "outcome": "skipped",
            "reason": "would run walk, sent nothing",
            "params": {"vx": 0.15, "duration_s": 1.0},
        },
    )
    ((text, _),) = render_lines(event)
    assert "vx=0.15" in text and "duration_s=1" in text


def test_the_verb_end_line_counts_the_intents_and_the_seconds() -> None:
    event = TraceEvent(
        "verb_end",
        0.0,
        {
            "name": "go_to",
            "ok": True,
            "outcome": "ok",
            "summary": "reached the ball",
            "elapsed_s": 12.5,
            "intents": {"move": 120, "stop": 1},
        },
    )
    ((text, style),) = render_lines(event)
    assert "go_to ok: reached the ball (12.5 s, 121 intents)" in text and style == "green"


def test_the_verb_end_line_shows_both_clocks_only_when_they_disagree() -> None:
    """A free-running simulator crosses twenty of the robot's seconds in one of ours. One
    line saying `1.4 s` for that walk is wrong, and two numbers on every sub-second verb is
    noise, so both a half second and a fifth have to separate them."""

    def line(**extra: Any) -> str:
        data = {"name": "go_to", "ok": True, "outcome": "ok", "summary": "there", **extra}
        ((text, _),) = render_lines(TraceEvent("verb_end", 0.0, data))
        return text

    assert "20.0 s sim, 1.4 s wall" in line(elapsed_s=1.4, transport_s=20.0, clock="sim")
    assert "(1.0 s," in line(elapsed_s=1.0, transport_s=1.1, clock="sim")  # too close to say
    assert "(1.4 s," in line(elapsed_s=1.4, transport_s=20.0)  # no clock: hardware, one number


def test_a_burst_spans_the_robots_clock_when_it_has_one() -> None:
    """`move x200 over 1.4 s` implies 140 Hz to a reader when the commanded rate was 10."""
    tracer = Tracer()
    seen = events(tracer)
    for wall, robot in ((0.0, 0.0), (1.4, 20.0)):
        tracer.emit("intent", intent="move", params={"vx": 0.2}, accepted=True, robot_t=robot)
        object.__setattr__(seen[-1], "t", wall)
    (out,) = lines(seen)
    assert "over 20.0 s" in out


def test_the_loops_own_verb_record_renders_nothing() -> None:
    """It is the same call as `verb_end`, kept in the transcript for the readers that pin it."""
    assert render_lines(TraceEvent("verb", 0.0, {"name": "kick", "ok": True})) == []
    assert render_lines(TraceEvent("frame", 0.0, {"path": "frames/0001.png"})) == []
    assert render_lines(TraceEvent("run_end", 0.0, {"outcome": "success"})) == []


def test_run_start_shows_the_prompt_once_and_can_be_asked_not_to() -> None:
    event = TraceEvent(
        "run_start",
        0.0,
        {
            "duck": "find-and-kick",
            "provider": "anthropic",
            "model": "claude-opus-5",
            "transport": "sim2d",
            "adapter": "microduck",
            "tools": ["walk", "kick"],
            "system_prompt": "You are the brain of a duck.\nRules follow.",
            "connect_s": 0.05,
        },
    )
    with_prompt = [text for text, _ in render_lines(event)]
    assert any("You are the brain of a duck." in line for line in with_prompt)
    assert any("robot=microduck:sim2d" in line for line in with_prompt)
    without = [text for text, _ in render_lines(event, prompt=False)]
    assert not any("brain of a duck" in line for line in without)


# ── the console ─────────────────────────────────────────────────────────────────────────


def console_trace(**kwargs: Any) -> tuple[ConsoleTrace, io.StringIO]:
    buffer = io.StringIO()
    console = Console(file=buffer, width=200, force_terminal=False, no_color=True)
    kwargs.setdefault("thinking_chars", None)
    return ConsoleTrace(console, **kwargs), buffer


def test_square_brackets_in_what_the_model_wrote_survive_verbatim() -> None:
    """Rich reads `[dim]` as markup and raises on an unpaired closing tag. Every line here
    carries text a model or a robot wrote, so none of it may be parsed as markup."""
    view, buffer = console_trace()
    view(
        TraceEvent(
            "llm",
            0.0,
            {
                "thinking": "[/think] and [dry-run] and [bold]",
                "text": "the ball is [behind] the sofa",
                "tool_calls": [],
                "usage": {},
            },
        )
    )
    out = buffer.getvalue()
    assert "[/think]" in out and "[dry-run]" in out and "[bold]" in out
    assert "[behind]" in out


def test_the_console_flushes_a_pending_burst_before_the_next_line() -> None:
    view, buffer = console_trace()
    for _ in range(3):
        view(TraceEvent("intent", 0.0, {"intent": "move", "params": {"vx": 0.1}, "accepted": True}))
    view(
        TraceEvent(
            "verb_end", 0.4, {"name": "move", "ok": True, "outcome": "ok", "summary": "walked"}
        )
    )
    out = buffer.getvalue().splitlines()
    assert "move x3" in out[0] and "walked" in out[1]


def test_fan_out_feeds_every_sink_and_one_bad_view_never_starves_the_others() -> None:
    """The loop takes a single observer and a run wants two: the narration and the status
    line that says what it is waiting for. A console that raises must not take the status
    with it, and the Tracer must still count exactly one drop for the event."""
    seen_a: list[str] = []
    seen_b: list[str] = []

    def broken(_event: TraceEvent) -> None:
        raise ValueError("a terminal that cannot print")

    sink = fan_out(lambda e: seen_a.append(e.kind), None, broken, lambda e: seen_b.append(e.kind))
    tracer = Tracer(observers=[sink])
    tracer.emit("llm", text="hello")
    assert seen_a == ["llm"] and seen_b == ["llm"], "a raise stopped a later sink"
    assert tracer.dropped == 1, "the failure has to reach the Tracer, once"


def test_fan_out_of_nothing_is_a_sink_that_does_nothing() -> None:
    fan_out()(TraceEvent("llm", 0.0, {}))  # must not raise


def test_every_kind_of_moment_keeps_the_mark_its_glyph_is_chosen_from() -> None:
    """`mark` is the only thing the terminal reads to decide what a line looks like, and the
    golden cannot see it: it freezes `render_lines`, which throws the mark away. Swapping two
    of these would turn every refusal into a tick and no other test would notice."""
    from tests.golden.trace_cases import events

    marks = {
        name: [line.mark for line in render_events(event) if line.mark] for name, event in events()
    }
    assert marks["verb_start"] == ["start"]
    assert marks["intent_one"] == ["send"] and marks["intent_refused"] == ["fail"]
    assert marks["verb_end_ok"] == ["ok"] and marks["verb_end_fail"] == ["fail"]
    assert marks["verb_end_preempted"] == ["other"], "a handover is not a fault"
    assert marks["gate_refused"] == ["fail"] and marks["gate_allowed"] == ["warn"]
    assert marks["declare_success"] == ["ok"] and marks["declare_failure"] == ["fail"]
    assert marks["llm_error"] == ["fail"] and marks["llm_no_tool_call"] == ["warn"]
    assert marks["observation_error"] == ["fail"] and marks["enforce"] == ["warn"]
    assert marks["flock_claim"] == ["flock"] and marks["member_end"] == ["end"]
    assert marks["note"] == ["note"] and marks["memory"] == ["note"]
    assert marks["assess_feasible"] == ["ok"], "a verdict that clears the way reads like one"
    assert marks["assess_infeasible"] == ["warn"], "nothing broke: this body cannot"
    assert marks["assess_uncertain_human_no"] == ["warn"]
    assert marks["assess_invalid"] == ["fail"]
    assert marks["tool_result"] == ["ok"] and marks["tool_result_sim"] == ["fail"]
    # and nothing invents a mark the glyph table has no field for
    from quackd import ui

    for name, found in marks.items():
        assert all(m in ui.MARKS for m in found), (name, found)


# ── the terminal view ───────────────────────────────────────────────────────────────────


def narrow_trace(**kwargs: Any) -> tuple[ConsoleTrace, io.BytesIO]:
    """A view on a Windows codepage: what `2> trace.log` gives you there."""
    raw = io.BytesIO()
    console = Console(
        file=io.TextIOWrapper(raw, encoding="cp1252", errors="replace"),
        width=200,
        force_terminal=False,
    )
    return ConsoleTrace(console, **kwargs), raw


def shown(raw: io.BytesIO, view: ConsoleTrace) -> str:
    view.console.file.flush()
    return raw.getvalue().decode("cp1252")


def test_the_terminal_puts_the_arrow_in_the_gutter_and_a_word_in_the_column() -> None:
    """The plain renderer's `->` is a contract with a model. A person gets the arrow as a
    glyph and the column says what it stood for."""
    view, buffer = console_trace()
    view(TraceEvent("verb_start", 0.0, {"name": "go_to", "params": {"target": "ball"}}))
    view(TraceEvent("intent", 0.1, {"intent": "move", "params": {"vx": 0.2}, "accepted": True}))
    view(
        TraceEvent(
            "verb_end", 0.2, {"name": "go_to", "ok": True, "outcome": "ok", "summary": "there"}
        )
    )
    out = buffer.getvalue()
    assert "▶  verb    go_to(target='ball')" in out
    assert "→  send    move(vx=0.2)" in out
    assert "✓  result  go_to ok: there" in out
    assert "->" not in out and "<-" not in out, "the arrows are glyphs here, not text"


def test_a_failure_is_a_shape_as_well_as_a_colour() -> None:
    view, buffer = console_trace()
    view(TraceEvent("gate", 0.0, {"gate": "allowlist", "outcome": "refused", "reason": "no"}))
    view(
        TraceEvent(
            "verb_end", 0.1, {"name": "kick", "outcome": "fail", "summary": "missed", "ok": False}
        )
    )
    view(TraceEvent("verb_end", 0.2, {"name": "s", "outcome": "preempted", "summary": "role"}))
    out = buffer.getvalue()
    assert "✗  gate" in out and "✗  result  kick FAIL" in out
    assert "•  result  s PREEMPTED" in out, "a handover is not a fault and must not look like one"


def test_every_glyph_becomes_ascii_on_a_stream_that_cannot_carry_it() -> None:
    """A redirected stderr on Windows is cp1252, and this is exactly the output people
    redirect. ADR-0029 protected that with ASCII everywhere; it is protected here by asking
    the stream."""
    view, raw = narrow_trace()
    view(TraceEvent("verb_start", 0.0, {"name": "go_to", "params": {}}))
    view(TraceEvent("intent", 0.1, {"intent": "move", "params": {"vx": 0.2}, "accepted": True}))
    done = {"name": "go_to", "ok": True, "outcome": "ok", "summary": "x"}
    view(TraceEvent("verb_end", 0.2, done))
    out = shown(raw, view)
    assert out.isascii() and "?" not in out
    assert ">  verb" in out and "-> send" in out and "+  result" in out


def test_what_a_robot_wrote_is_respelled_rather_than_lost() -> None:
    """`bearing 28° left` arrived as `bearing 28? left` on a codepage without a degree sign.
    A stand-in that says the same thing is strictly better than a question mark."""
    view, raw = narrow_trace()
    view(TraceEvent("note", 0.0, {"text": "ball at bearing 28° left ±2°"}))
    out = shown(raw, view)
    assert "28 deg left +/-2 deg" in out and "?" not in out


def test_each_step_is_ruled_off_and_the_budget_is_only_said_once() -> None:
    """The loop writes `[step 3/40 · step 3/40, llm calls ...]` at the top of every
    observation: the one line saying where the run is up to, buried in a paragraph, and
    saying the step twice."""
    view, buffer = console_trace()
    text = "[step 3/40 · step 3/40, llm calls 3/40, 0.1/5 min]\nstate: posture=standing"
    view(TraceEvent("observation", 0.0, {"step": 3, "text": text}))
    out = buffer.getvalue()
    assert "step 3/40, llm calls 3/40, 0.1/5 min" in out
    assert out.count("step 3/40") == 1, "the step was announced twice"
    assert "state: posture=standing" in out
    assert "[step" not in out, "the header became the rule and must not also be a line"


def test_an_observation_that_failed_is_a_line_not_a_rule() -> None:
    view, buffer = console_trace()
    view(TraceEvent("observation", 0.0, {"error": "camera timed out"}))
    assert "ERROR camera timed out" in buffer.getvalue()


def test_a_flock_member_is_never_given_a_rule_of_its_own() -> None:
    """Three members narrate at once; a rule each would be three rules per step and none of
    them would mean the run had moved on."""
    view, buffer = console_trace(prefix="duck-1  ")
    view(TraceEvent("observation", 0.0, {"text": "[step 1/9 · llm calls 1/9]\nstate: up"}))
    view(TraceEvent("verb_start", 0.1, {"name": "kick", "params": {}}))
    out = buffer.getvalue()
    assert "───" not in out and "[step 1/9" in out, "the budget stays in the member's own line"
    assert all(line.startswith("duck-1  ") for line in out.splitlines() if line.strip())


def test_the_system_prompt_is_a_block_between_two_rules() -> None:
    view, buffer = console_trace()
    view(
        TraceEvent(
            "run_start",
            0.0,
            {"duck": "d", "transport": "mock", "system_prompt": "line one\n\nline three"},
        )
    )
    out = buffer.getvalue()
    assert "system prompt, 20 chars, 3 lines" in out
    # the exact lines: a substring check passes on any indent at all, and the point of the
    # block is that seventy lines of somebody else's prose do not eat the terminal's width
    lines = out.splitlines()
    assert "   line one" in lines and "   line three" in lines
    assert "" in lines, "a blank line carries no padding: that is what a redirect diffs on"
    assert out.rstrip().endswith("─" * 10), "the block is closed off so the run visibly starts"


def test_the_system_prompt_is_respelled_for_the_stream_like_every_other_line() -> None:
    view, raw = narrow_trace()
    prompt = "## Rules (enforced by the executor — not optional)"
    view(TraceEvent("run_start", 0.0, {"duck": "d", "transport": "mock", "system_prompt": prompt}))
    out = shown(raw, view)
    assert "executor - not optional" in out and "?" not in out


def test_a_live_run_does_not_repeat_the_header_the_cli_just_printed() -> None:
    view, buffer = console_trace()
    view(
        TraceEvent(
            "run_start",
            0.0,
            {
                "duck": "d",
                "provider": "fake",
                "transport": "mock",
                "connect_s": 0.5,
                "tools": ["a"],
            },
        )
    )
    out = buffer.getvalue()
    assert "connected in 0.50 s" in out
    assert "tools   a" in out
    assert "provider=fake" not in out, "the panel in front of this already said it"


def test_a_replay_introduces_the_run_because_nothing_else_did() -> None:
    view, buffer = console_trace(header=True)
    view(
        TraceEvent(
            "run_start",
            0.0,
            {
                "duck": "find-and-kick",
                "provider": "anthropic",
                "model": "claude-opus-5",
                "adapter": "microduck",
                "transport": "sim2d",
                "tools": ["kick"],
                "connect_s": 0.5,
                "memory": {"notes": 3, "episodes": 5},
            },
        )
    )
    out = buffer.getvalue()
    for needle in (
        "find-and-kick",
        "anthropic (claude-opus-5)",
        "microduck:sim2d",
        "3 notes, 5 earlier runs",
        "connected in 0.50 s",
    ):
        assert needle in out, needle


def test_the_terminal_view_never_reads_what_a_model_wrote_as_markup() -> None:
    """The same promise the plain view makes, made again by the renderer that replaced it."""
    view, buffer = console_trace()
    view(
        TraceEvent(
            "llm",
            0.0,
            {
                "thinking": "[/think] and [dry-run] and [bold]",
                "text": "the ball is [behind] the sofa",
                "tool_calls": [],
                "usage": {},
            },
        )
    )
    out = buffer.getvalue()
    for tag in ("[/think]", "[dry-run]", "[bold]", "[behind]"):
        assert tag in out, tag


# ── capturing one call (the MCP server) ─────────────────────────────────────────────────


async def test_two_concurrent_calls_never_see_each_others_events() -> None:
    """The MCP SDK runs every tool call as its own task. A buffer on the session would put
    one call's intents in the other call's result."""
    tracer = Tracer(observers=[capture_sink])

    async def call(name: str, delay: float) -> list[str]:
        with capturing() as seen:
            tracer.emit("verb_start", name=name)
            await asyncio.sleep(delay)
            tracer.emit("intent", intent=name, params={}, accepted=True)
            await asyncio.sleep(delay)
            tracer.emit("verb_end", name=name, ok=True, outcome="ok", summary="done")
            return [e.data.get("name") or e.data.get("intent") for e in seen]

    slow, fast = await asyncio.gather(call("go_to", 0.02), call("quack", 0.001))
    assert set(slow) == {"go_to"} and set(fast) == {"quack"}


def test_nothing_is_captured_outside_a_call() -> None:
    Tracer(observers=[capture_sink]).emit("note", text="the heartbeat failed")  # must not raise


def test_render_call_is_short_plain_lines() -> None:
    tracer = Tracer(observers=[capture_sink])
    with capturing() as seen:
        tracer.emit("tool_call", tool="robot_run_verb", robot="duck", verb="go_to", params={})
        tracer.emit("verb_start", name="go_to", params={"target": "ball"}, source="mcp")
        for wz in (0.1, 0.2):
            tracer.emit("intent", intent="move", params={"vx": 0.2, "wz": wz}, accepted=True)
        tracer.emit(
            "verb_end",
            name="go_to",
            ok=True,
            outcome="ok",
            summary="reached the ball",
            elapsed_s=1.0,
            intents={"move": 2},
        )
        tracer.emit(
            "tool_result", tool="robot_run_verb", ok=True, elapsed_s=1.1, budget="step 1/40"
        )
    rendered = render_call(seen)
    assert any("go_to(target='ball') from mcp" in line for line in rendered)
    assert any("move x2" in line for line in rendered)
    assert any("reached the ball" in line for line in rendered)
    assert any("step 1/40" in line for line in rendered)
    assert all(isinstance(line, str) for line in rendered)


def test_a_long_trace_is_capped_and_says_what_it_cut() -> None:
    capped = cap_lines([f"line {i}" for i in range(100)], limit=10, head=3)
    assert len(capped) == 10
    assert capped[:3] == ["line 0", "line 1", "line 2"]
    assert "91 more lines" in capped[3] and "stderr" in capped[3]
    assert capped[-1] == "line 99"
    short = ["a", "b"]
    assert cap_lines(short, limit=10) == short


# ── the switch ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "on"),
    [
        (None, True),
        ("", True),
        ("1", True),
        ("yes", True),
        ("0", False),
        ("false", False),
        ("No", False),
        ("off", False),
    ],
)
def test_the_env_switch(monkeypatch: pytest.MonkeyPatch, value: str | None, on: bool) -> None:
    if value is None:
        monkeypatch.delenv("QUACKD_TRACE", raising=False)
    else:
        monkeypatch.setenv("QUACKD_TRACE", value)
    assert trace_enabled_default() is on


@pytest.mark.parametrize(
    ("value", "limit"), [("", 2000), ("all", None), ("0", 0), ("500", 500), ("nonsense", 2000)]
)
def test_how_much_thinking_the_console_shows(
    monkeypatch: pytest.MonkeyPatch, value: str, limit: int | None
) -> None:
    monkeypatch.setenv("QUACKD_TRACE_THINKING", value)
    assert thinking_limit_default() == limit
    assert parse_thinking_limit(value) == limit  # a flag and the environment agree


@pytest.mark.parametrize(
    ("value", "shown"),
    [(None, True), ("", True), ("1", True), ("0", False), ("off", False), ("No", False)],
)
def test_the_prompt_env_switch(
    monkeypatch: pytest.MonkeyPatch, value: str | None, shown: bool
) -> None:
    if value is None:
        monkeypatch.delenv("QUACKD_TRACE_PROMPT", raising=False)
    else:
        monkeypatch.setenv("QUACKD_TRACE_PROMPT", value)
    assert prompt_shown_default() is shown


def test_none_means_unlimited_thinking_on_the_console_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`None` used to mean "read the environment" here and "unlimited" in `render_lines`."""
    monkeypatch.setenv("QUACKD_TRACE_THINKING", "100")
    event = TraceEvent("llm", 0.0, {"thinking": "x" * 5000, "tool_calls": [], "usage": {}})

    buffer = io.StringIO()
    console = Console(file=buffer, width=400, force_terminal=False, no_color=True)
    ConsoleTrace(console, thinking_chars=None)(event)
    assert "transcript.jsonl" not in buffer.getvalue()

    buffer = io.StringIO()
    console = Console(file=buffer, width=400, force_terminal=False, no_color=True)
    ConsoleTrace(console, thinking_chars=thinking_limit_default())(event)
    assert "+4900 chars in transcript.jsonl" in buffer.getvalue()


def test_a_prefixed_view_names_its_robot_on_every_line_including_continuations() -> None:
    out: list[str] = []
    view = LineTrace(lambda text, _style: out.append(text), prefix="duck-1  ")
    view(TraceEvent("note", 0.0, {"text": "first line\nsecond line"}))
    view(TraceEvent("intent", 0.1, {"kind": "move", "params": {}, "ok": True}))
    view.flush()
    assert out
    printed = "\n".join(out).splitlines()
    assert printed, "the view printed nothing at all"
    assert all(line.startswith("duck-1  ") for line in printed), printed
    assert any("second line" in line for line in printed)


def test_the_coordinators_events_render_as_flock_lines_in_the_recorders_words() -> None:
    cases = [
        (
            TraceEvent("auction", 0.0, {"first_bid": "duck-1", "dist": 0.42}),
            "auction first bid duck-1 0.42 m",
        ),
        (
            TraceEvent("claim", 0.0, {"kicker": "duck-1", "dist": 0.62, "spotter": "r-2"}),
            "claim   duck-1 (0.62 m), spotter r-2",
        ),
        (TraceEvent("miss", 0.0, {"duck": "duck-0"}), "miss    duck-0, re-searching"),
        (TraceEvent("kick_done", 0.0, {"kicker": "duck-2"}), "kicked  by duck-2, the spotter"),
        (
            TraceEvent("verdict", 0.0, {"verdict": "moved", "moved_m": 0.51, "spotter": "r-1"}),
            "verdict moved 0.51 m by r-1",
        ),
        (TraceEvent("member_dead", 0.0, {"duck": "duck-2", "last_hb": 3.0}), "dead    duck-2"),
        (
            TraceEvent("member_end", 0.0, {"status": "stopped", "steps": 7}),
            "end     stopped after 7",
        ),
    ]
    for event, needle in cases:
        ((text, _),) = render_lines(event)
        assert text.startswith(needle), (event.kind, text)
    # the GIF's caption and the console line are the same words, one upper and one lower
    word, detail = flock_caption("claim", {"kicker": "duck-1", "dist": 0.62}) or ("", "")
    assert word == "CLAIM" and detail.startswith("duck-1 (0.62 m)")
    assert flock_caption("verb_end", {}) is None


def test_a_verb_ended_by_another_layer_is_yellow_and_keeps_its_own_word() -> None:
    data = {"name": "search_scan", "outcome": "preempted", "summary": "role change to kicker"}
    ((text, style),) = render_lines(TraceEvent("verb_end", 0.0, data))
    assert "PREEMPTED: role change to kicker" in text
    assert style == "yellow", "a routine handover must not read as the red that means a bug"
    ((_, style),) = render_lines(TraceEvent("verb_end", 0.0, {**data, "outcome": "error"}))
    assert style == "red"


def test_a_human_denial_renders_red() -> None:
    """A denial is a person saying no, and the line has to look like the refusal it is."""
    denied = TraceEvent("gate", 0.0, {"gate": "confirm", "outcome": "denied", "verb": "kick"})
    ((text, style),) = render_lines(denied)
    assert style == "red" and "denied" in text
    allowed = TraceEvent("gate", 0.0, {"gate": "confirm", "outcome": "allowed", "verb": "kick"})
    ((_, style),) = render_lines(allowed)
    assert style != "red", "a person saying yes is not a refusal"


def test_the_reprompt_line_quotes_what_the_model_was_told() -> None:
    """A model that answered with no tool call gets told so and asked again. Reading the trace
    afterwards, what it was told is the whole reason the next turn looks the way it does."""
    event = TraceEvent(
        "enforce",
        0.0,
        {
            "issue": "no tool call",
            "action": "re-prompting once",
            "text": "You must call exactly one tool.",
        },
    )
    ((text, _),) = render_lines(event)
    assert "no tool call: re-prompting once" in text
    assert "(You must call exactly one tool.)" in text


def test_the_done_line_says_sim_seconds_on_a_simulator() -> None:
    """An MCP client reading `done ok in 0.2 s` after a twenty second approach would think
    the robot teleported. On a simulator the two clocks are different numbers, and both."""
    sim = TraceEvent(
        "tool_result", 0.0, {"ok": True, "elapsed_s": 0.2, "transport_s": 20.0, "clock": "sim"}
    )
    ((text, _),) = render_lines(sim)
    assert "20.0 s sim, 0.2 s wall" in text
    hardware = TraceEvent("tool_result", 0.0, {"ok": True, "elapsed_s": 0.2, "transport_s": 0.2})
    ((text, _),) = render_lines(hardware)
    assert "in 0.2 s" in text and "sim" not in text
