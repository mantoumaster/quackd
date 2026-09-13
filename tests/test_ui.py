"""The house style: what it draws, what it refuses to parse, and when it stands down.

Three promises are worth a test each. Text a model or a manifest wrote is never read as
markup. A stream that cannot carry a glyph gets an ASCII one rather than a `?`. And the
status line exists only when somebody is watching, because everything this project prints
also has to survive a pipe, a test runner and a CI log.
"""

from __future__ import annotations

import io
from typing import Any

import pytest
from rich.console import Console

from quackd import ui
from quackd.trace import TraceEvent


def narrow() -> tuple[Console, io.BytesIO]:
    """A console on a Windows codepage: what `2> trace.log` gives you there."""
    raw = io.BytesIO()
    return Console(file=io.TextIOWrapper(raw, encoding="cp1252", errors="replace"), width=90), raw


def wide(width: int = 90, **kwargs: Any) -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, width=width, **kwargs), buf


def drawn(console: Console, raw: io.BytesIO) -> str:
    console.file.flush()
    return raw.getvalue().decode("cp1252")


# ── glyphs ──────────────────────────────────────────────────────────────────────────────


def test_the_glyph_half_is_chosen_from_the_stream_not_from_the_platform() -> None:
    console, _ = narrow()
    assert ui.glyphs_for(console) is ui.ASCII
    assert ui.glyphs_for(wide()[0]) is ui.UNICODE


def test_every_gutter_glyph_fits_the_gutter_in_both_halves() -> None:
    """The label column starts at a fixed offset. A glyph wider than the gutter would push
    one line's label out of line with its neighbours' and the trace would read as noise."""
    from rich.cells import cell_len

    for glyphs in (ui.UNICODE, ui.ASCII):
        for mark in ui.MARKS:
            assert 0 < cell_len(glyphs.mark(mark)) <= 2, (glyphs, mark)


def test_a_mark_nobody_defined_draws_nothing() -> None:
    assert ui.UNICODE.mark(None) == ""
    assert ui.UNICODE.mark("duck") == "", "only the marks, never the decorations"
    assert ui.UNICODE.mark("nonsense") == ""


def test_the_registry_status_emoji_becomes_something_a_codepage_can_carry() -> None:
    fancy = "✅ built-in: sim2d · \U0001f9ea jsonrpc — experimental"
    assert ui.degrade(fancy, ui.UNICODE) == fancy, "a terminal that can show it, sees it"
    plain = ui.degrade(fancy, ui.ASCII)
    assert plain == "[ok] built-in: sim2d - [exp] jsonrpc - experimental"
    assert plain.isascii()


# ── text is data ────────────────────────────────────────────────────────────────────────

MARKUP = "the ball is [behind] the sofa [/think] and [bold] quackd[anthropic]"


@pytest.mark.parametrize(
    "renderable",
    [
        pytest.param(ui.fail_line(MARKUP, hint=MARKUP), id="fail_line"),
        pytest.param(ui.run_header(MARKUP, [("robot", MARKUP)], hint=MARKUP), id="run_header"),
        pytest.param(
            ui.verdict("failure", MARKUP, counters=[MARKUP], run_dir="runs/x"), id="verdict"
        ),
        pytest.param(ui.kv_grid([("key", MARKUP)]), id="kv_grid"),
    ],
)
def test_nothing_here_reads_what_somebody_else_wrote_as_markup(renderable: Any) -> None:
    """Rich drops `[bold]` silently and raises on an unpaired `[/think]`. Every string in
    this output came from a model, a manifest, or an error naming an extra."""
    console, buf = wide(400)
    console.print(renderable)
    out = buf.getvalue()
    for tag in ("[behind]", "[/think]", "[bold]", "quackd[anthropic]"):
        assert tag in out, tag


def test_the_verdict_wears_its_outcome() -> None:
    console, buf = wide(60, force_terminal=True)
    for outcome in ("success", "failure", "budget", "aborted", "infeasible"):
        console.print(ui.verdict(outcome, "why", run_dir="runs/x"))
    out = buf.getvalue()
    assert "SUCCESS" in out and "BUDGET" in out and "ABORTED" in out
    assert "INFEASIBLE" in out
    # infeasible is a warning, not a failure: nothing broke and nothing was tried
    assert ui._OUTCOME["infeasible"] == ui._OUTCOME["budget"]
    assert "\x1b[32m" in out, "success is green"
    assert "\x1b[1;31m" in out, "and a failure is not"


def test_a_panel_on_a_codepage_that_has_no_box_still_says_everything() -> None:
    console, raw = narrow()
    console.print(
        ui.run_header("find-and-kick", [("robot", "microduck:sim2d")], hint="Ctrl-C stops it")
    )
    console.print(ui.verdict("success", "kicked it", counters=["steps 7", "llm calls 8"]))
    out = drawn(console, raw)
    assert "?" not in out, "a glyph that arrives as a question mark is a glyph not worth having"
    assert out.isascii()
    for needle in ("find-and-kick", "microduck:sim2d", "Ctrl-C stops it", "SUCCESS", "kicked it"):
        assert needle in out, needle


def test_the_renderables_read_the_console_drawing_them_not_the_one_at_import() -> None:
    """`quackd trace` replays onto stdout and a run narrates onto stderr, and tests hand
    round buffers of their own. Choosing the glyph when the panel is built gets all three
    wrong, which is why they are deferred."""
    panel = ui.verdict("success", "done")
    console, raw = narrow()
    console.print(panel)
    assert drawn(console, raw).isascii()
    console2, buf = wide()
    console2.print(panel)
    assert "✓" in buf.getvalue()


# ── the consoles ────────────────────────────────────────────────────────────────────────


def test_configure_rebuilds_both_consoles_so_no_color_can_reach_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ui.configure()
    assert ui.console.no_color is False
    ui.configure(no_color=True)
    assert ui.console.no_color is True and ui.err_console.no_color is True
    monkeypatch.setenv("NO_COLOR", "1")
    ui.configure()
    assert ui.console.no_color is True, "the variable alone is enough"
    monkeypatch.delenv("NO_COLOR")
    ui.configure()
    assert ui.console.no_color is False


def test_one_console_answers_and_the_other_decorates() -> None:
    ui.configure()
    assert ui.console.stderr is False and ui.err_console.stderr is True


# ── waiting ─────────────────────────────────────────────────────────────────────────────


def ticks() -> Any:
    """A clock that moves a second every time it is read, so the line has to count."""
    state = {"t": 0.0}

    def clock() -> float:
        state["t"] += 1.0
        return state["t"]

    return clock


def test_the_status_line_does_nothing_at_all_when_nobody_is_watching() -> None:
    """Under a pipe, a test runner or `TERM=dumb`: no live region, no refresh thread, and
    nothing added to the output that a script would then have to strip."""
    console, buf = wide()
    with ui.RunStatus(console) as status:
        assert status.active is False
        status.update("waiting on anthropic")
        status.sink(TraceEvent("llm_request", 0.0, {"provider": "anthropic", "model": "opus"}))
        console.print("the answer")
    assert buf.getvalue() == "the answer\n"


def test_the_status_line_draws_and_prints_land_above_it() -> None:
    console, buf = wide(force_terminal=True)
    with ui.RunStatus(console) as status:
        assert status.active is True
        status.update("waiting on anthropic claude-opus-5")
        console.print("a line of trace")
    out = buf.getvalue()
    assert "a line of trace" in out
    assert "waiting on anthropic claude-opus-5" in out


def test_the_status_line_names_the_step_and_counts_a_long_wait() -> None:
    console, _ = wide()
    status = ui.RunStatus(console, clock=ticks())
    text = "[step 3/40 · llm calls 3/40, 0.1/5 min]\nstate: posture=standing"
    status.sink(TraceEvent("observation", 0.0, {"text": text}))
    status.sink(TraceEvent("llm_request", 0.0, {"provider": "anthropic", "model": "claude-opus-5"}))
    rendered = status.__rich__().plain
    assert "step 3/40" in rendered, "the budget's two halves only appear together here"
    assert "waiting on anthropic claude-opus-5" in rendered
    assert rendered.rstrip().endswith("s"), "a wait this long says how long"


def test_a_provider_that_names_no_model_still_reads_as_a_sentence() -> None:
    console, _ = wide()
    status = ui.RunStatus(console)
    status.sink(TraceEvent("llm_request", 0.0, {"provider": "ollama", "model": None}))
    assert "waiting on ollama the first model it serves" in status.__rich__().plain


def test_the_status_line_follows_a_run_through_its_phases() -> None:
    console, _ = wide()
    status = ui.RunStatus(console)
    for event, expected in (
        (TraceEvent("verb_start", 0.0, {"name": "go_to", "params": {"target": "ball"}}), "go_to("),
        (TraceEvent("verb_end", 0.0, {"name": "go_to", "ok": True}), "observing"),
        (TraceEvent("claim", 0.0, {"kicker": "duck-1", "dist": 0.6}), "claim duck-1"),
        (TraceEvent("note", 0.0, {"text": "kill switch: Ctrl-C"}), "stopping the robot"),
        (TraceEvent("run_end", 0.0, {"outcome": "success"}), "finishing"),
    ):
        status.sink(event)
        assert expected in status.__rich__().plain, event.kind


def test_a_status_line_that_cannot_draw_never_counts_as_a_dropped_event() -> None:
    """It is decoration. The count is what the run could not *show*, and what shows things
    is the trace."""
    console, _ = wide()
    status = ui.RunStatus(console)
    status.sink(object())  # not an event at all
    status.sink(TraceEvent("verb_start", 0.0, {"params": "not a mapping"}))


def test_pausing_takes_the_line_down_so_a_prompt_can_be_seen() -> None:
    """A live region redirects stdout and a `y/N` prompt writes without a newline, so under
    a running status the question is invisible until the answer has been typed."""
    console, _ = wide(force_terminal=True)
    with ui.RunStatus(console) as status:
        assert status.active
        with ui.pause_status():
            assert not status._status._live.is_started
        assert status._status._live.is_started


def test_pausing_when_nothing_is_up_is_allowed() -> None:
    with ui.pause_status():
        pass
    console, _ = wide()
    with ui.RunStatus(console) as status, status.paused():
        pass


def test_a_spinner_stands_aside_for_a_status_line_already_on_screen() -> None:
    """Rich 13.7, which is the floor this project declares, refuses a second live region on
    one console."""
    console, buf = wide(force_terminal=True)
    with ui.RunStatus(console), ui.spinner("probing", target=console) as say:
        say("still probing")
    assert "probing" not in buf.getvalue()


def test_a_spinner_says_nothing_into_a_pipe() -> None:
    console, buf = wide()
    with ui.spinner("listening for robots", target=console) as say:
        say("still listening")
    assert buf.getvalue() == ""


def test_the_status_stack_empties_even_when_the_run_raises() -> None:
    console, _ = wide(force_terminal=True)
    with pytest.raises(KeyboardInterrupt), ui.RunStatus(console):
        raise KeyboardInterrupt
    assert not ui._ACTIVE


def test_installing_the_log_handler_leaves_records_reaching_a_root_handler() -> None:
    """Turning off propagation is the obvious way to stop a record being handled twice, and
    it would have been wrong: quackd puts no handler on the root logger, so there is nothing
    to stop, and what it would really have stopped is pytest's own caplog, which is how four
    tests read what the MCP server logged."""
    import logging

    seen: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record.getMessage())

    ui.install_logging()
    ui.install_logging()  # twice: it must replace its handler, not stack a second one
    quackd_log = logging.getLogger("quackd")
    assert quackd_log.propagate is True
    assert sum(1 for h in quackd_log.handlers if type(h).__name__ == "RichHandler") == 1

    root, handler = logging.getLogger(), Capture()
    root.addHandler(handler)
    try:
        logging.getLogger("quackd.mcp").warning("a line an MCP test would assert on")
    finally:
        root.removeHandler(handler)
        for h in list(quackd_log.handlers):
            quackd_log.removeHandler(h)
    assert seen == ["a line an MCP test would assert on"]
