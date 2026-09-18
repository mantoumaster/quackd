"""One place that decides what quackd looks like in a terminal.

The CLI grew its colours inline: a `[green]` here, a `Table` there, an emoji in one command
and a dim line in another, with nothing saying which green meant *this worked* and which
meant *this is installed*. This module is that vocabulary in one file, so a new command
inherits the house style instead of inventing a third one.

Three rules run through it.

**Text is data.** A model's reasoning, a robot's manifest and an error naming
``quackd[anthropic]`` all contain square brackets, and Rich reads those as markup: it drops
`[bold]` silently and raises on an unpaired `[/think]`. Everything here takes plain strings
and wraps them in `rich.text.Text`, which never parses. No renderable in this file is built
from an f-string with a tag in it, and the style names are the concrete Rich ones rather
than theme keys, so these renderables also work on a bare `Console` a test just made.

**The terminal is not the only reader.** The same commands run into a pipe, a file, a CI log
and a Windows console whose codepage has no arrow. Glyphs come from a table with an ASCII
half chosen from the stream's own encoding, and the spinner and the status line stand down
entirely when nobody is watching.

**Chrome goes to stderr.** Spinners and the status line live on `err_console`, answers on
`console`, so `quackd list-adapters > adapters.txt` gets the table and nothing else, and a
live region never has to share a console with the answer it is decorating.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console, RenderableType
from rich.measure import Measurement
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# ── the vocabulary ──────────────────────────────────────────────────────────────────────

STYLES: dict[str, str] = {
    "ok": "green",  # it worked, it is installed, it is there
    "warn": "yellow",  # it wants attention and nothing is broken
    "fail": "bold red",  # it did not do what was asked
    "muted": "dim",  # context: paths, counts, hints
    "key": "bold",  # the name of a thing
    "accent": "cyan",  # a value worth finding again: a robot, a provider
    "rule": "dim",  # the separators between steps
}
"""Concrete Rich styles, not theme keys: `STYLES["ok"]` is the string `green`, so a
renderable built here renders on any console, including the bare ones tests hand around."""

THEME = Theme(STYLES)
"""The same names, installed on quackd's own consoles, so `[ok]` markup works for anyone who
wants it. Nothing in this file relies on it."""

MEMBER_STYLES = (
    "cyan",
    "magenta",
    "green",
    "yellow",
    "blue",
    "bright_cyan",
    "bright_magenta",
    "bright_green",
)
"""One per robot in a flock. Several narrate at once and the eye finds a colour faster than it
reads a name; the coordinator's own lines are bold and wear none of these. Eight of them,
because that is what a pilot flock may hold (`PILOTS_MAX_MEMBERS`); an auction uses four."""

MARKS = ("start", "send", "ok", "fail", "warn", "other", "note", "flock", "end")
"""The kinds of moment a trace line can be, and the glyph fields that answer for them."""


@dataclass(frozen=True)
class Glyphs:
    """What a line wears in its gutter, and the few decorations outside it.

    Every gutter glyph is at most two cells wide, so the label column lines up whichever
    half is in use."""

    start: str
    send: str
    ok: str
    fail: str
    warn: str
    other: str
    note: str
    flock: str
    end: str
    duck: str
    dot: str
    spinner: str

    def mark(self, name: str | None) -> str:
        """The glyph for a `TraceLine.mark`, or nothing for a line that speaks for itself."""
        return getattr(self, name) if name in MARKS else ""


UNICODE = Glyphs(
    start="▶",
    send="→",
    ok="✓",
    fail="✗",
    warn="⚠",
    other="•",
    note="·",
    flock="◆",
    end="■",
    duck="\U0001f986",
    dot="·",
    spinner="dots",
)
ASCII = Glyphs(
    start=">",
    send="->",
    ok="+",
    fail="x",
    warn="!",
    other="*",
    note="-",
    flock="#",
    end="=",
    duck="",
    dot="-",
    spinner="line",
)

_PLAIN = {
    "✅": "[ok]",  # the registry's built-in tick
    "🧪": "[exp]",  # experimental
    "🚧": "[wip]",
    "⏳": "[wip]",
    "⚠️": "[!]",
    "❌": "[no]",
    "🦆": "duck",
    "·": "-",  # the middle dot this project separates fields with
    "—": "-",  # an em dash the codepage has not got
    "–": "-",  # noqa: RUF001 - nor an en dash
    "§": "S",  # and no section sign
    "°": " deg",
    "±": "+/-",
    "…": "...",
    "→": "->",
    "←": "<-",
    "≈": "~",
    "×": "x",  # noqa: RUF001 - the multiplication sign
}


def glyphs_for(target: Console) -> Glyphs:
    """ASCII when the stream cannot carry anything else.

    A redirected stderr on Windows is cp1252, and that is exactly the stream people redirect
    a trace into: without this the arrows and ticks arrive as `?`."""
    return ASCII if target.options.ascii_only else UNICODE


def degrade(text: str, glyphs: Glyphs) -> str:
    """quackd's own decoration, spelled for the stream in hand.

    Only the characters this project puts in its own strings: the adapter registry's status
    emoji, the middle dot it separates fields with, the dashes and the section sign. What a
    model or a robot wrote is left exactly as it wrote it."""
    if glyphs is not ASCII:
        return text
    for fancy, plain in _PLAIN.items():
        text = text.replace(fancy, plain)
    return text


# ── the consoles ────────────────────────────────────────────────────────────────────────


def tolerate_narrow_encodings() -> None:
    """Stop a non-UTF-8 stdout turning quackd's own output into a crash.

    Windows uses the ANSI codepage when Python writes to a pipe, and quackd prints ticks and
    a duck and the status emoji in `doctor`. On cp1252 those raise UnicodeEncodeError and
    take the command with them, which is why `glyphs_for` exists to avoid printing them at
    all. This is the backstop for the text quackd did not choose: replacing what the
    codepage cannot carry costs a glyph, and raising costs the command."""
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if encoding.startswith("utf") or not hasattr(stream, "reconfigure"):
            continue
        with contextlib.suppress(Exception):
            stream.reconfigure(errors="replace")


def make_console(*, stderr: bool = False, no_color: bool = False) -> Console:
    """One console, built the way every quackd console is built.

    `highlight=False` because Rich's guesses at what is a number or a path are wrong as
    often as they are right in this output, and `emoji=False` because a robot that names a
    verb `:kick:` must not have it replaced by a picture."""
    return Console(
        stderr=stderr,
        theme=THEME,
        highlight=False,
        emoji=False,
        no_color=no_color or bool(os.environ.get("NO_COLOR")),
    )


tolerate_narrow_encodings()
# At import, not only in `configure`: Typer renders `--help` from a console of its own while
# it is still parsing, which is before any callback of ours has run, and this project's help
# text has a duck in it.

console = make_console()
"""Answers: tables, verdicts, the things you would redirect into a file."""

err_console = make_console(stderr=True)
"""Everything else: the trace, the status line, warnings, errors."""


def configure(*, no_color: bool = False) -> None:
    """Rebuild both consoles for this invocation.

    Rich reads `NO_COLOR`, `FORCE_COLOR` and the stream's encoding when a console is made,
    and quackd's are made at import, which is before `--no-color` has been parsed and before
    a test has swapped the streams. One call at the top of the root callback is what makes
    those answers current. Both names are module attributes on purpose: reach them as
    `ui.console`, never by importing the object, or you will be holding yesterday's."""
    global console, err_console
    tolerate_narrow_encodings()
    console = make_console(no_color=no_color)
    err_console = make_console(stderr=True, no_color=no_color)


# ── renderables ─────────────────────────────────────────────────────────────────────────

BOX = box.ROUNDED
"""One box everywhere. `│` stays the column separator, which is how the adapter tests read
rows back out, and Rich swaps the whole box for an ASCII one where that cannot be drawn."""


class Deferred:
    """A renderable that waits until it knows which console is drawing it.

    Every panel here has a glyph in it somewhere, and which half of the glyph table is right
    depends on the stream. Deciding that when the object is *built* reads the wrong console
    the moment anything is rendered anywhere but `ui.console`: a buffer a test made, a
    replay on stdout, a cp1252 pipe. Deciding it at render time cannot be got wrong."""

    def __init__(self, build: Callable[[Glyphs], RenderableType]) -> None:
        self._build = build

    def __rich_console__(self, target: Console, options: Any) -> Iterator[RenderableType]:
        yield self._build(glyphs_for(target))

    def __rich_measure__(self, target: Console, options: Any) -> Measurement:
        """How wide this wants to be, which is whatever it turns out to be.

        Without this Rich has to assume the worst and gives the cell the whole terminal, so
        one deferred cell stretched its entire table to the window's width."""
        return Measurement.get(target, options, self._build(glyphs_for(target)))


def _deferred(build: Callable[[Glyphs], RenderableType], glyphs: Glyphs | None) -> RenderableType:
    return build(glyphs) if glyphs is not None else Deferred(build)


def _renderable(value: Any) -> RenderableType:
    """A cell: anything Rich already knows how to draw, or a string that must not be read as
    markup because a manifest wrote it."""
    if isinstance(value, str):
        return Text(value)
    if hasattr(value, "__rich__") or hasattr(value, "__rich_console__"):
        return value  # type: ignore[no-any-return]
    return Text(str(value))


def table(title: str | None = None, **kwargs: Any) -> Table:
    """A table in the house style: a left-aligned bold title over a rounded box."""
    return Table(
        title=Text(title, style=STYLES["key"]) if title else None,
        title_justify="left",
        box=BOX,
        header_style=STYLES["key"],
        **kwargs,
    )


def kv_grid(rows: Iterable[tuple[str, Any]], *, key_style: str | None = None) -> Table:
    """Aligned `key   value` lines with no box: the body of a panel, or a section with two
    columns that does not need ruling off.

    `key_style=""` means the keys carry their own styling and must not be dimmed; None means
    the default. `or` would have collapsed the two, which dimmed a check list's ticks."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=STYLES["muted"] if key_style is None else key_style, no_wrap=True)
    grid.add_column(overflow="fold")
    for key, value in rows:
        grid.add_row(Text(key), _renderable(value))
    return grid


def joined(parts: Sequence[str], glyphs: Glyphs, *, style: str | None = None) -> Text:
    """`steps 7 · llm calls 8 · tokens 6338+128`, with a separator the stream can carry."""
    out = Text(style=style or STYLES["muted"])
    for i, part in enumerate(parts):
        if i:
            out.append(f" {glyphs.dot} ")
        out.append(part)
    return out


def path_text(path: Path | str) -> Text:
    """A path, clickable where the terminal allows it and plain everywhere else."""
    text = Text(str(path), overflow="fold")
    with contextlib.suppress(Exception):
        text.stylize(f"link {Path(path).resolve().as_uri()}")
    return text


def run_header(
    title: str,
    rows: Sequence[tuple[str, Any]],
    *,
    hint: str = "",
    glyphs: Glyphs | None = None,
) -> RenderableType:
    """What is about to happen, in the one place a person looks before looking away.

    Before the trace rather than inside it: with `--no-trace` this is the only thing between
    the command and the verdict, and on a run that never connects it is the only record of
    what was tried."""

    def build(g: Glyphs) -> RenderableType:
        head = Text()
        if g.duck:
            head.append(f"{g.duck} ")
        head.append(title, style=STYLES["key"])
        return Panel(
            kv_grid(rows),
            title=head,
            title_align="left",
            subtitle=Text(hint, style=STYLES["muted"]) if hint else None,
            subtitle_align="left",
            border_style=STYLES["muted"],
            box=BOX,
            padding=(0, 1),
        )

    return _deferred(build, glyphs)


_OUTCOME = {
    "success": ("ok", "ok"),
    "budget": ("warn", "warn"),
    # infeasible is not a failure: nothing was tried, the pilot said the body could not
    "infeasible": ("warn", "warn"),
}
"""outcome -> (style key, glyph mark). Anything else is a failure and looks like one."""


def verdict(
    outcome: str,
    reason: str,
    *,
    counters: Sequence[str] = (),
    run_dir: Path | str | None = None,
    gif_path: Path | str | None = None,
    glyphs: Glyphs | None = None,
) -> RenderableType:
    """How it ended: the word, why, what it cost, and where the evidence is.

    `quackd trace` prints this from the transcript too, so a replay ends exactly the way the
    run did rather than in a second dialect somebody has to keep in step."""
    style_key, mark = _OUTCOME.get(outcome, ("fail", "fail"))
    style = STYLES[style_key]

    def build(g: Glyphs) -> RenderableType:
        head = Text()
        head.append(f"{g.mark(mark)} ", style=style)
        head.append(outcome.upper(), style=style)
        body = Table.grid()
        body.add_column(overflow="fold")
        if reason:
            body.add_row(Text(reason))
        if counters:
            body.add_row(joined(counters, g))
        where = [(k, path_text(v)) for k, v in (("run dir", run_dir), ("gif", gif_path)) if v]
        if where:
            body.add_row(kv_grid(where))
        return Panel(
            body, title=head, title_align="left", border_style=style, box=BOX, padding=(0, 1)
        )

    return _deferred(build, glyphs)


def fail_line(
    message: str, *, hint: str | None = None, glyphs: Glyphs | None = None
) -> RenderableType:
    """`error:` and what went wrong, with one dim line saying where to look next.

    The message routinely names an extra (`quackd[anthropic]`), which is why it is appended
    to a `Text` rather than formatted into markup that would eat the brackets."""

    def build(g: Glyphs) -> RenderableType:
        out = Text()
        out.append(f"{g.fail} error: ", style=STYLES["fail"])
        out.append(message)
        if hint:
            out.append("\n")
            out.append(f"  {hint}", style=STYLES["muted"])
        return out

    return _deferred(build, glyphs)


def plain(text: str, *, style: str | None = None) -> RenderableType:
    """Prose this project wrote, spelled for whichever console draws it.

    Upstream notes carry a section sign and an em dash and quackd's own status strings carry
    emoji, and on a Windows codepage every one of them arrives as a question mark."""
    return _deferred(lambda g: Text(degrade(text, g), style=style or ""), None)


def status_text(word: str, glyphs: Glyphs) -> Text:
    """A registry status string (`built-in ...` behind an emoji), spelled for this stream.

    `glyphs` is positional and required on purpose: this is called from inside a render, and
    a default would quietly read whichever console happened to be the module's."""
    return Text(degrade(word, glyphs))


ADAPTERS_TITLE = "adapters (--robot <adapter>:<backend>)"


def adapters_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str | None = ADAPTERS_TITLE,
    glyphs: Glyphs | None = None,
) -> RenderableType:
    """The adapter roster. `list-adapters` and `doctor` each had a copy of this, and the two
    had already drifted apart in their columns. `title=None` for a caller that has already
    said what this is."""
    return _deferred(lambda g: _adapters_table(rows, g, title), glyphs)


def _adapters_table(rows: Sequence[Mapping[str, Any]], g: Glyphs, title: str | None) -> Table:
    out = table(title)
    out.add_column("adapter", style=STYLES["key"], no_wrap=True)
    out.add_column("backends")
    out.add_column("status")
    out.add_column("extra")
    for row in rows:
        extra = Text(str(row["extra"]))
        if row["extra"] != "built-in":
            extra.append(*_install_state(row))
        out.add_row(
            Text(str(row["name"])),
            Text(f" {g.dot} ".join(row["backends"])),
            status_text(str(row["status"]), g),
            extra,
        )
    return out


def _install_state(row: Mapping[str, Any]) -> tuple[str, str]:
    """Three states, not two, since every adapter became its own package.

    An adapter can be absent, or present with the library its hardware backend needs still
    missing, and those are different things to do about it. Collapsing the second into
    `not installed` told somebody whose `lerobot:mock` was running fine that their adapter
    was gone, and sent them to reinstall a package they already had."""
    if not row.get("adapter_installed", row["installed"]):
        return " not installed", STYLES["muted"]
    if row.get("sdk") is False:
        return " installed, no SDK", STYLES["warn"]
    return " installed", STYLES["ok"]


# ── waiting ─────────────────────────────────────────────────────────────────────────────

_ACTIVE: list[RunStatus] = []
"""The status lines in flight. Rich 13.7 refuses a second live region on one console, so
everything that would open one asks here first."""

_SLOW_S = 1.0
"""How long a wait has to last before the line starts counting it out loud."""

_STEP = re.compile(r"step \d+/\d+")


def _step_of(data: Mapping[str, Any]) -> str | None:
    """`step 3/40`, read out of the observation the model was shown, which is the only place
    the budget's two halves appear together."""
    found = _STEP.search(str(data.get("text") or "").split("\n", 1)[0])
    return found.group(0) if found else None


class RunStatus:
    """The line under the trace that says what the run is waiting for.

    A run spends nearly all its wall clock inside two calls: a model deciding, and a verb
    steering a robot. The trace says what happened once it has happened; until then the
    terminal had nothing, and with `--no-trace` it had nothing at all between the header and
    the verdict. This reads the same event stream and keeps one transient line at the
    bottom, so a slow provider looks like a slow provider rather than like a hang.

    It is a Rich status, so anything printed to the same console appears above it and the
    line is redrawn underneath. It exists only when stderr is a terminal somebody is
    watching: under a pipe, a test runner or `TERM=dumb` nothing starts, no thread runs, and
    `sink` is a few dict lookups. It also stands down while something else owns the
    terminal, which is what `paused` is for: a confirmation prompt writes without a newline,
    and a live region would swallow the question until after it was answered.
    """

    def __init__(
        self,
        target: Console | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.console = target if target is not None else err_console
        self._clock = clock
        self._lock = threading.Lock()
        self._step = ""
        self._text = "starting"
        self._since = clock()
        self._status: Any = None

    # ── the line itself ─────────────────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        """Whether anything is actually being drawn."""
        return self._status is not None

    def __rich__(self) -> Text:
        """Rebuilt on every frame, which is how the seconds tick without a second timer."""
        with self._lock:
            step, text, since = self._step, self._text, self._since
        g = glyphs_for(self.console)
        out = Text(no_wrap=True, overflow="ellipsis")
        if step:
            out.append(step, style=STYLES["key"])
            out.append(f" {g.dot} ", style=STYLES["muted"])
        out.append(text, style=STYLES["muted"])
        waited = self._clock() - since
        if waited >= _SLOW_S:
            out.append(f" {g.dot} {waited:.0f} s", style=STYLES["muted"])
        return out

    def update(self, text: str, *, step: str | None = None, restart: bool = True) -> None:
        """Say something else. `restart` is what makes the seconds count this wait rather
        than the one before it."""
        with self._lock:
            self._text = text
            if step is not None:
                self._step = step
            if restart:
                self._since = self._clock()

    # ── being on screen ─────────────────────────────────────────────────────────────

    def __enter__(self) -> RunStatus:
        watching = self.console.is_terminal and not self.console.is_dumb_terminal
        if watching and not _ACTIVE:
            self._status = self.console.status(
                self, spinner=glyphs_for(self.console).spinner, refresh_per_second=4
            )
            self._status.start()
        _ACTIVE.append(self)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self in _ACTIVE:
            _ACTIVE.remove(self)
        if self._status is not None:
            self._status.stop()
            self._status = None

    @contextlib.contextmanager
    def paused(self) -> Iterator[None]:
        """Take the line down while something else writes to the terminal.

        Stopping is not cosmetic: a live region redirects stdout, and a `y/N` prompt writes
        without a newline, so under a running status the question is invisible until the
        answer has already been typed."""
        if self._status is None:
            yield
            return
        self._status.stop()
        try:
            yield
        finally:
            self._status.start()

    # ── reading the run ─────────────────────────────────────────────────────────────

    def sink(self, event: Any) -> None:
        """A trace observer. Never raises: the status line is decoration, and a decoration
        that failed to draw must not be counted as an event the run could not show."""
        with contextlib.suppress(Exception):
            self._read(event)

    def _read(self, event: Any) -> None:
        kind = str(getattr(event, "kind", ""))
        data: Mapping[str, Any] = getattr(event, "data", None) or {}
        if kind == "run_start":
            self.update("observing", step="")
        elif kind == "observation":
            self.update("observing", step=_step_of(data))
        elif kind == "llm_request":
            model = data.get("model") or "the first model it serves"
            self.update(f"waiting on {data.get('provider')} {model}")
        elif kind == "llm":
            self.update("choosing a verb")
        elif kind == "verb_start":
            from quackd.trace import fmt_params

            self.update(f"{data.get('name')}({fmt_params(data.get('params'))})")
        elif kind == "verb_end":
            self.update("observing")
        elif kind == "hand_off" and data.get("stage") == "released":
            # `how` and not the stage: a release the arm refused is still filed under the
            # stage it was asked for, and the line that invites somebody to pick the arm up
            # must only ever follow a release that actually happened
            if data.get("how") == "released":
                self.update("waiting for you to place the arm")
        elif kind == "note" and str(data.get("text") or "").startswith("kill switch"):
            self.update("stopping the robot")
        elif kind in ("declare", "member_end", "run_end"):
            self.update("finishing")
        else:
            from quackd.trace import flock_caption

            caption = flock_caption(kind, data)
            if caption is not None:
                self.update(f"{caption[0].lower()} {caption[1]}")


@contextlib.contextmanager
def pause_status() -> Iterator[None]:
    """Take down whatever status line is up, for anyone who has no handle on it.

    The confirmation prompts are called from deep inside the executor with nothing but a
    question, which is exactly where a status line has to get out of the way."""
    if not _ACTIVE:
        yield
        return
    with _ACTIVE[-1].paused():
        yield


@contextlib.contextmanager
def spinner(text: str, *, target: Console | None = None) -> Iterator[Callable[[str], None]]:
    """A spinner around one blocking call, yielding a way to say what it is doing now.

    Never around a print: on Rich 13.7, which is the floor this project declares, a second
    live region on one console raises, so this stands aside when a run status is already up,
    and it draws nothing at all when nobody is watching the stream."""
    out = target if target is not None else err_console
    watching = out.is_terminal and not out.is_dumb_terminal
    if not watching or _ACTIVE:
        yield lambda _message: None
        return
    with out.status(
        Text(text, style=STYLES["muted"]), spinner=glyphs_for(out).spinner, refresh_per_second=4
    ) as status:
        yield lambda message: status.update(Text(message, style=STYLES["muted"]))


def install_logging(level: int = 30) -> None:
    """Send quackd's own loggers through the same console as everything else.

    Nothing configured them, so a warning from the camera or the transport arrived through
    Python's last-resort handler: unformatted, and straight down the middle of a live region
    it knew nothing about. `serve-mcp` keeps its own plain handler, because there stdout is
    the wire and stderr is a log somebody greps.

    `propagate` is deliberately left alone. Turning it off would be the obvious way to stop
    a record being handled twice, but quackd never puts a handler on the root logger, so
    there is no second handler to stop, and anything that *does* put one there is entitled
    to see these records: `logging.basicConfig` in a host application, and pytest's own
    `caplog`, which is how four tests read what the MCP server logged.
    """
    import logging

    from rich.logging import RichHandler

    log = logging.getLogger("quackd")
    for handler in [h for h in log.handlers if isinstance(h, RichHandler)]:
        log.removeHandler(handler)
    log.addHandler(
        RichHandler(
            console=err_console,
            level=level,
            show_time=False,
            show_path=False,
            markup=False,
            rich_tracebacks=False,
        )
    )
    log.setLevel(level)
