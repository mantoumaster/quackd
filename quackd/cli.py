"""The command line is the product's front door.

`uvx --from "quackd[microduck]" quackd run find-and-kick --llm anthropic --robot
microduck:sim2d` is the north-star demo; every command here exists to make that line, and the
debugging around it, boring. The `--from` is there because the core ships no robot and the
demo needs one. Commands are thin: they parse, load `.env`, wire objects together, hand off.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import glob
import json
import os
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from dotenv import load_dotenv
from pydantic import ValidationError
from rich.text import Text

from quackd import __version__, ui
from quackd.agent.providers.catalogue import (
    CLOUD_NAMES,
    DEFAULT_LLM,
    LLM_ENV,
    LOCAL_NAMES,
    PROVIDER_NAMES,
    default_model_for,
    models_for,
    vendor_of,
)
from quackd.command import command_text

if TYPE_CHECKING:  # every heavy module is imported inside the command that needs it
    from quackd.safety import KillSwitch

app = typer.Typer(
    name="quackd",
    # the emoji only where the stream can carry it: a cp1252 pipe on Windows renders them as
    # `??`, and the front door is the worst place to look broken
    help="One CLI for all your robots. Connect them, command them, and let them work "
    "together, each with an LLM for a brain."
    + (" 🦆🧠" if ui.glyphs_for(ui.console) is ui.UNICODE else ""),
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
    # A crash must not print this process's local variables: they hold an API key, a robot's
    # address and its bridge token. `main` installs a Rich traceback without them instead.
    pretty_exceptions_enable=False,
    # blank lines rather than spaces: Typer renders the epilog as paragraphs and would
    # otherwise run three examples together into one
    epilog=(
        # `doctor` first: it is the one command that works before a robot is installed, and it
        # says which are. The install is on the line that follows it and covers both run
        # examples below, because the core on its own carries no robot and a first example
        # that refuses is a bad one.
        "[bold]Try[/bold]" + "\n\n"
        "quackd doctor  |  quackd list-adapters  |  quackd log" + "\n\n"
        # the backslash is Rich's escape: an unescaped [microduck] is a style tag, and Typer
        # renders this epilog as markup, so it would print the install line without the extra
        # that makes it work
        r"uv pip install 'quackd\[microduck]' && quackd run find-and-kick --llm fake" + "\n\n"
        "quackd run --goal 'walk in a square' --llm anthropic --robot microduck:mujoco"
    ),
)


_DEPRECATIONS: list[str] = []
"""Every deprecation line this process printed, in order, kept for the saved terminal.

These are printed from the root callback, which Click runs before the subcommand body and so
before there is a capture to forward them into. Rather than move the warning later, where a
reader would meet it after the header panel instead of before it, the text is kept here and
`_terminal_header` puts it back at the top of the file where it was on the screen."""


def _deprecated(msg: str) -> None:
    """One yellow line on stderr, the shape ADR-0017 used to retire a flag over a release.

    `soft_wrap` because the sentence is an instruction a script may grep for and the longest
    of them is 99 characters, which a default 80-column stderr would fold in the middle of
    the new spelling."""
    _DEPRECATIONS.append(msg)
    ui.err_console.print(
        Text(msg, style=ui.STYLES["warn"]), markup=False, highlight=False, soft_wrap=True
    )


def _warn_old_spellings() -> None:
    """Say it once per process, for each name this release stopped reading and finds set.

    Only variables are left here. A flag or a subcommand that is gone fails loudly: Click
    refuses it, names it, and nothing runs. A variable that is gone goes quiet, and the quiet
    is the failure. `QUACKD_MODEL` is the kind of line that sits in a `.env` for a year;
    unread, it does not stop the run, it lets the run bill a model nobody chose.
    `QUACKD_TRACE=0` is the same line with the opposite sign: unread, it switches the log back
    on for the one reader who had deliberately turned it off, which is why 0.11 went on
    reading it for a release. 0.12 stops, as promised, and says so instead.

    Read here rather than where each value used to be, so a `.env` is told about every old
    line in it and not only the one this run would have consulted: 0.11 could warn about a
    name only on a run that read it, and said so.

    The kept lines are cleared first. One process is one command when a person runs quackd,
    but this module is also imported and driven twice in a row by tests, by a wrapper and by
    `quackd.cli.app(...)`, and a second run whose saved terminal opened with the first run's
    deprecations would be a header describing a command nobody typed."""
    _DEPRECATIONS.clear()
    for gone, now in (
        ("QUACKD_MODEL", "QUACKD_LLM=vendor:model"),
        ("QUACKD_JEV", "QUACKD_DECISION_LLM"),
        ("QUACKD_TRACE", "QUACKD_LOG"),
        ("QUACKD_TRACE_THINKING", "QUACKD_LOG_THINKING"),
        ("QUACKD_TRACE_PROMPT", "QUACKD_LOG_PROMPT"),
    ):
        if os.environ.get(gone):
            _deprecated(f"{gone} is not read any more and this run ignores it; set {now} instead")


def _terminal_header() -> list[str]:
    """What the saved terminal opens with: the command, then what ran it and when.

    Two lines and a blank one, followed by any deprecation the root callback already printed,
    which is the one thing said on screen before there was anywhere to write it down."""
    started = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    return [
        f"$ {command_text()}",
        f"quackd {__version__}, started {started}, cwd {Path.cwd()}",
        "",
        *_DEPRECATIONS,
        *([""] if _DEPRECATIONS else []),
    ]


@contextlib.contextmanager
def _terminal_record() -> Iterator[None]:
    """Keep what this run puts on the terminal, and write it into the run directory.

    Around the whole command rather than inside `_run_impl`, so the pre-flight refusals are
    in it too: a run that dies on a bad flag prints one sentence and that sentence is part of
    the story. Nothing is written until there is a run directory to write into, which is what
    keeps a refused run from leaving one behind.

    The three exits are told apart because the file should say which happened. `typer.Exit`
    is a refusal that has already printed its own line. A `KeyboardInterrupt` is somebody
    pressing Ctrl-C twice, which prints nothing at all. Anything else is a crash, and the
    traceback for it is drawn by `sys.excepthook` after this has closed, so the file would
    otherwise end mid-sentence with no sign of why."""
    capture = ui.begin_capture(header=_terminal_header())
    try:
        yield
    except typer.Exit:
        raise
    except KeyboardInterrupt:
        ui.note("^C")
        raise
    except BaseException as e:
        ui.note(f"quackd crashed: {type(e).__name__}: {e}")
        raise
    finally:
        capture.close()


def _version_callback(value: bool) -> None:
    if value:
        ui.console.print(f"quackd {__version__}")
        raise typer.Exit()


def _no_color_callback(value: bool) -> bool:
    """Eager, and it sets the variable rather than only the consoles.

    Typer builds a console of its own for every `--help` it renders and reads `NO_COLOR`
    when it does, so the variable is what makes `quackd --no-color run --help` plain as
    well. Eager means it has run by the time the subcommand is parsed."""
    if value:
        os.environ["NO_COLOR"] = "1"
    return value


@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True, help="Show version."
    ),
    no_color: bool = typer.Option(
        False,
        "--no-color",
        is_eager=True,
        callback=_no_color_callback,
        help="Plain output with no colour. NO_COLOR=1 does the same, and FORCE_COLOR=1 keeps "
        "the colour when the output is a pipe.",
    ),
) -> None:
    """quackd — one CLI for all your robots, real or simulated, piloted by any LLM."""
    # `.env` first, so a NO_COLOR line in it counts, and then the consoles: Rich reads the
    # environment and the stream's encoding when a console is built, and quackd's are built
    # at import, which is before any of this was known.
    #
    # The one beside where you run, then the bare call, which walks up from quackd's own
    # installed directory and so finds the `.env` a `uv venv` user put in their venv root.
    # Neither overrides a variable already in the environment, and the first file to define
    # a name wins, so the file next to the command you typed is the one that counts.
    load_dotenv(Path.cwd() / ".env")
    load_dotenv()
    ui.configure(no_color=no_color)
    _warn_old_spellings()


_JSON = typer.Option(
    False,
    "--json",
    help="One JSON object per line on stdout, and nothing else: for a script rather than "
    "for a person. Exit codes are unchanged.",
    rich_help_panel="Output",
)

_REGISTRY_DIR = typer.Option(
    None,
    "--registry-dir",
    help="Where robots.json and flocks.json live (default: $QUACKD_REGISTRY_DIR or ~/.quackd).",
    rich_help_panel="Robot",
)

NEWLINE = "\n"

_ADAPTER_HINT = "quackd list-adapters shows the seven that ship and their backends"


def _expand(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        out.extend(matches if matches else [pat])
    return out


def _verbose_line(msg: str) -> None:
    """A `--verbose` line, as plain text. A message can carry brackets Rich reads as markup:
    the executor's own `[dry-run] would run ...`, and the flock planner logging a model's raw
    tool arguments. Rich deletes `[bold]` silently and raises on an unpaired `[/think]`."""
    ui.err_console.print(msg, style="dim", markup=False, highlight=False, soft_wrap=True)


EXIT_INFEASIBLE = 3
"""`quackd run` when the pilot judged the task beyond this body and nothing moved."""


def _print_outcome(
    outcome: str,
    reason: str,
    *,
    counters: Sequence[str],
    run_dir: Path | str,
    gif_path: Path | str | None = None,
    log_dropped: int = 0,
) -> None:
    """How the run ended, in the one place a person looks after looking away.

    `quackd log` prints it from the transcript too, so a replay ends exactly the way the
    run itself did rather than in a second dialect somebody has to keep in step. `counters`
    is a list because a flock counts different things than a solo run does."""
    ui.console.print(
        ui.verdict(outcome, reason, counters=counters, run_dir=run_dir, gif_path=gif_path)
    )
    if log_dropped:
        # a console that raised on every event produced a silent log and no sign of it
        ui.err_console.print(
            f"log: {log_dropped} line(s) could not be shown (the console raised); "
            "transcript.jsonl has them",
            style="yellow",
            markup=False,
        )


def _number(value: Any) -> float | None:
    """A figure out of a record, or None where there is not one.

    `quackd log` is pointed at files people hand-edit, truncate and copy between machines,
    and the renderers beside this one already shrug at a field that is not what it should be.
    A counter line is not worth a traceback."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return None if value != value else float(value)  # NaN is not a figure either


def run_counters(end: Mapping[str, Any]) -> list[str]:
    """What a finished run cost, in the line under the verdict.

    Built from the summary dict, which is what `run_end` carries and what `summary.json`
    holds, so the live panel and `quackd log` print one list rather than two that drift.

    Every field is optional on purpose. A SOLO run recorded before any of this existed
    replays with exactly the three counters it always had. A flock is the one deliberate
    exception: it has kept a wall clock of its own since long before a solo run had one, so
    an old flock record does gain a `time` counter from the number it was already writing.
    """
    from quackd.agent.providers.pricing import fmt_usd
    from quackd.log import fmt_duration

    usage = end.get("usage") or {}
    counters = [
        f"steps {int(_number(end.get('steps')) or 0)}",
        f"llm calls {int(_number(end.get('llm_calls')) or 0)}",
        f"tokens {usage.get('input_tokens', 0)}+{usage.get('output_tokens', 0)}",
    ]
    wall = _number(end.get("wall_s"))
    if wall is None:
        wall = _number(end.get("wall_elapsed_s"))
    if wall is not None:
        # The split, not just the total: on the one hardware run this project has, 62.1 of
        # 78.8 seconds were spent waiting on the model, and that ratio is the single most
        # useful number a run produces. The stepper's seconds join it when one ran.
        spent = []
        if (llm := _number(end.get("llm_latency_s"))) is not None:
            spent.append(f"model {fmt_duration(llm)}")
        if (stepper := _number((end.get("decision") or {}).get("latency_s"))) is not None:
            spent.append(f"stepper {fmt_duration(stepper)}")
        where = f" ({', '.join(spent)})" if spent else ""
        counters.append(f"time {fmt_duration(wall)}{where}")
    decision_block = end.get("decision") or {}
    decision_cost = _number(decision_block.get("cost_usd"))
    # On a cost KEY, not on the presence of a stepper block. A stepper block recorded before
    # the costing existed carries no `cost_usd`, and gating on the block gave those records a
    # counter reading `cost unpriced` that they never had and that says nothing true about
    # them: nobody tried to price them.
    if "cost_usd" in end or decision_cost is not None:
        model_cost = _number(end.get("cost_usd")) if end.get("cost_usd") is not None else None
        estimated = bool(decision_block.get("cost_estimated")) and bool(decision_cost)
        if model_cost is None:
            # A model quackd has no rate for must never read as a free one, and the two
            # halves are reported separately rather than summed away: "unpriced" plus a
            # stepper figure is the truth, and a bare "unpriced" would throw away the half
            # that IS known.
            unpriced = "cost unpriced"
            if decision_cost:
                unpriced += f" (stepper {'~' if estimated else ''}{fmt_usd(decision_cost)})"
            counters.append(unpriced)
        else:
            total = model_cost + (decision_cost or 0.0)
            counters.append(f"cost {'~' if estimated else ''}{fmt_usd(total)}")
    return counters


def _detector_row(
    detector: Any, hello: Any, backend: str | None, *, sees: bool = True
) -> str | None:
    """Which detector reads the frames, and where it runs, in one line, or None for a body with
    nothing to look at. The board's is named with the board and what the daemon said it runs,
    because "yolo" alone would not say whether the laptop or the Jetson is doing the work.

    `sees` is whether the body, with the board's camera if it has one, is described with a
    camera. The board's detector is chosen for a body described without one, because it may
    report a camera when it connects, and the row says that it reads nothing until then."""
    if detector is None:
        return None
    from quackd.perception import is_simulated
    from quackd.perception.host import HostDetector

    if isinstance(detector, HostDetector):
        text = f"{detector.name}  {detector.address}"
        if hello is not None and (label := hello.label()):
            text += f"  {label}"
        if is_simulated(backend):
            # never chosen by itself on a simulator, so a reader seeing it there is told it
            # was asked for, and that it is not what this simulator's colours are tuned for
            text += "  (asked for on a simulator)"
        if not sees:
            text += "  (once the body reports a camera)"
        return text
    return f"{getattr(detector, 'name', type(detector).__name__)} on this machine"


def _host_row(board: Any, hello: Any, host_camera: dict[str, Any] | None) -> str:
    """The board: where it is, which daemon answered, and what it has. The camera says whether
    it is the view the run steers by or one it is only shown.

    That is settled at connect, from what the body reports (`HostCameraAdapter.connect`), and
    this is written before. A body described with a camera keeps it, so the board's is an
    extra view. For one described without, the board's is the primary view only if the body
    does not report a camera of its own when it connects (an arm given --camera-url, a
    rosbridge base), so the row says so; `run_start` records which it was."""
    has: list[str] = []
    if hello.has_camera:
        primary = bool((host_camera or {}).get("primary"))
        has.append(
            "camera as the primary view unless the body reports its own"
            if primary
            else "camera as an extra view"
        )
    if hello.can_detect:
        has.append("detect")
    if hello.is_tegra:
        has.append("tegra")
    return f"{board.address}  daemon {hello.daemon_version}  " + (
        ", ".join(has) if has else "health only"
    )


def _host_named(choice: Any) -> str:
    """The board as the reader named it, so a refusal points at the line they wrote: the flag,
    the robot's entry in robots.json, or the environment."""
    if choice.source == "--host":
        return f"--host {choice.host}"
    return f"the host {choice.host} from {choice.source}"


def _host_unreached_hint(choice: Any, robot: str | None) -> str:
    """How to run without the board, from wherever it was named: the flag, the robot that
    stores it, or the environment."""
    from quackd.host import HOST_ENV

    if choice.source == "--host":
        undo = "drop --host"
    elif choice.source == HOST_ENV:
        undo = f"unset {HOST_ENV}"
    else:
        undo = f"quackd robot edit {robot or 'NAME'} --clear host"
    return f"quackd doctor --host {choice.host} shows what the board says; {undo} to run without it"


def _header_rows(
    *,
    provider: Any,
    robot: str,
    seed: int | None,
    dry_run: bool,
    memory: Any,
    detector: Any = None,
    board: Any = None,
    hello: Any = None,
    backend: str | None = None,
    host_camera: dict[str, Any] | None = None,
    sees: bool = True,
) -> list[tuple[str, Any]]:
    """The things worth knowing before a run starts, and nothing else: who pilots, which body,
    what reads its frames, and the board when there is one."""
    rows: list[tuple[str, Any]] = [
        ("provider", f"{provider.name} ({provider.model or 'the first model it serves'})"),
        ("robot", robot + (f"  seed {seed}" if seed is not None else "")),
    ]
    if (seen := _detector_row(detector, hello, backend, sees=sees)) is not None:
        rows.append(("detector", seen))
    if board is not None and hello is not None:
        rows.append(("host", _host_row(board, hello, host_camera)))
    if dry_run:
        rows.append(
            (
                "mode",
                Text(
                    "DRY RUN: every intent is printed and nothing is sent", style=ui.STYLES["warn"]
                ),
            )
        )
    if memory is not None:
        m = memory.summary()
        rows.append(
            (
                "memory",
                Text.assemble(
                    f"{m['notes']} notes, {m['episodes']} earlier runs  ",
                    (f"{m['path']}", ui.STYLES["muted"]),
                    ("  --no-memory to run fresh", ui.STYLES["muted"]),
                ),
            )
        )
    return rows


def _fail(msg: str, code: int = 1, *, hint: str | None = None) -> None:
    """One line saying what went wrong, and one dim line saying where to look next.

    The message routinely names an extra (`quackd[anthropic]`) or a model's own brackets, so
    it travels as text rather than as markup Rich would eat."""
    ui.err_console.print(ui.fail_line(msg, hint=hint))
    raise typer.Exit(code=code)


def _robot_specs(
    robot: str | None, robots: str | None, duck: Any, *, registry_dir: str | None = None
) -> list[Any]:
    """The robots a command talks about, as `Resolved`: --robots, else --robot (a registered
    name or a spec), else the duck's own `robots:` default, else the Microduck simulator."""
    from quackd.adapters.factory import RobotSpec, parse_robot_spec, parse_robots
    from quackd.registry import Registry, Resolved, resolve_robot_ref

    if robots:
        # `--robots name=spec` is ad hoc by design: the names are this run's, not the
        # registry's, and a stored flock is spelled `--flock NAME` instead
        return [Resolved(spec) for spec in parse_robots(robots)]
    registry = Registry(registry_dir)
    default = duck.frontmatter.robots if duck is not None else None
    if isinstance(default, dict):
        if robot:
            return [resolve_robot_ref(robot, registry)]
        # the member names become the robot ids, as `--robots name=spec` would make them
        out = []
        for name, text in default.items():
            parsed = parse_robot_spec(text)
            out.append(Resolved(RobotSpec(parsed.adapter, parsed.backend, name)))
        return out
    if isinstance(default, str) or default is None:
        return [resolve_robot_ref(robot, registry, duck_default=default)]
    return [resolve_robot_ref(robot, registry)]


# ── validate ────────────────────────────────────────────────────────────────────────────


@app.command(rich_help_panel="Inspect")
def validate(
    duckfiles: list[str] = typer.Argument(..., help=".duck files, globs, or bundled names."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only print failures."),
    as_json: bool = _JSON,
    robot: list[str] | None = typer.Option(
        None,
        "--robot",
        "-r",
        help="Check the files against this robot's manifest (a registered name or "
        "<adapter>:<backend>; repeatable).",
    ),
    robots: str | None = typer.Option(
        None, "--robots", help="Check against a flock: name=<adapter>:<backend>,..."
    ),
    registry_dir: str | None = _REGISTRY_DIR,
) -> None:
    """Validate .duck files against the spec and a robot's verbs. Exits 1 on any failure."""
    from quackd.adapters.base import AdapterError
    from quackd.adapters.factory import describe
    from quackd.duckfile.parser import DuckParseError, load_duck
    from quackd.duckfile.validate import validate_duck
    from quackd.registry import Registry, RegistryError, resolve_robot_ref
    from quackd.verbs.registry import default_registry

    registry_ref = Registry(registry_dir)

    registry = default_registry()
    rows: list[dict[str, Any]] = []
    for path in _expand(duckfiles):
        row: dict[str, Any] = {"file": path, "name": None, "verbs": None, "robots": [], "ok": True}
        rows.append(row)
        try:
            duck = load_duck(path)
        except DuckParseError as e:
            row.update(ok=False, problems=[e.reason], summary=[e.reason])
            continue
        row["name"] = duck.name
        row["verbs"] = len(duck.frontmatter.verbs.allow)
        if duck.frontmatter.flock is not None:
            row["flock"] = len(duck.frontmatter.flock.member_names)
        try:
            if robot or robots:
                specs = (
                    [resolve_robot_ref(r, registry_ref).spec for r in robot]
                    if robot
                    else [r.spec for r in _robot_specs(None, robots, duck)]
                )
            elif duck.frontmatter.robots is not None:
                specs = [r.spec for r in _robot_specs(None, None, duck)]
            else:
                specs = []
            manifests = [describe(spec) for spec in specs]
        except (AdapterError, RegistryError) as e:
            # a registered name means reading robots.json, and a broken one refuses
            row.update(ok=False, problems=[str(e)], summary=[str(e)])
            continue
        row["robots"] = [m.id for m in manifests]
        problems = validate_duck(duck, manifests, registry=registry)
        if problems:
            # `str(p)` names the field it came from and is what the plain lines under the
            # table carry; `p.message` is the sentence, which is what fits in a cell
            row.update(
                ok=False,
                problems=[str(p) for p in problems],
                summary=[p.message for p in problems],
            )

    failures = [row for row in rows if not row["ok"]]
    if as_json:
        for row in rows:
            print(json.dumps({**row, "problems": row.get("problems", [])}))
        raise typer.Exit(code=1 if failures else 0)

    shown = failures if quiet else rows
    if shown:
        table = ui.table("quackd validate")
        # nothing here is no_wrap: a path can be any length, and a column that refuses to
        # wrap takes the width out of the one column that carries the answer
        table.add_column("file", overflow="fold")
        table.add_column("name")
        table.add_column("verbs", justify="right")
        table.add_column("result", ratio=2)
        for row in shown:
            table.add_row(*_validate_row(row))
        ui.console.print(table)
    if failures:
        # under the table as plain lines, so a long message survives any terminal width
        for row in failures:
            for problem in row.get("problems", []):
                ui.console.print(Text(f"  {row['file']}: {problem}"), soft_wrap=True)
        _fail(f"{len(failures)} of {len(rows)} {_files(len(rows))} failed", hint=_VALIDATE_HINT)
    ui.console.print(_ok_line(f"{len(rows)} {_files(len(rows))} valid"))


_VALIDATE_HINT = "quackd list-verbs --robot <adapter>:<backend> shows what a body can do"


def _files(n: int) -> str:
    return "file" if n == 1 else "files"


def _ok_line(message: str) -> Any:
    return ui.Deferred(
        lambda g: Text.assemble((f"{g.ok} ", ui.STYLES["ok"]), (message, ui.STYLES["ok"]))
    )


def _warn_line(message: str) -> Any:
    """Something the run carried on without. `_fail` is for what it cannot carry on without."""
    return ui.Deferred(
        lambda g: Text.assemble((f"{g.warn} ", ui.STYLES["warn"]), (message, ui.STYLES["warn"]))
    )


def _validate_row(row: dict[str, Any]) -> list[Any]:
    """One line of the table, with everything a manifest or a parser wrote kept as text."""
    verbs = "-" if row["verbs"] is None else str(row["verbs"])

    def result(g: ui.Glyphs) -> Text:
        if not row["ok"]:
            out = Text(f"{g.fail} ", style=ui.STYLES["fail"])
            out.append("; ".join(row.get("summary", [])) or "invalid", style=ui.STYLES["fail"])
            return out
        out = Text(f"{g.ok} valid", style=ui.STYLES["ok"])
        if row.get("flock"):
            out.append(f" (flock of {row['flock']})", style=ui.STYLES["ok"])
        if row["robots"]:
            out.append(f" for {', '.join(row['robots'])}", style=ui.STYLES["muted"])
        return out

    return [Text(row["file"]), Text(row["name"] or "-"), verbs, ui.Deferred(result)]


# ── list-verbs ──────────────────────────────────────────────────────────────────────────

_SAFETY_STYLE = {"safe": "ok", "confirm": "warn", "dangerous": "fail"}


@app.command("list-verbs", rich_help_panel="Inspect")
def list_verbs(
    robot: str | None = typer.Option(
        None,
        "--robot",
        "-r",
        help="A robot's vocabulary: <adapter>:<backend>, or a registered name. Default Microduck.",
    ),
    registry_dir: str | None = _REGISTRY_DIR,
    as_json: bool = _JSON,
) -> None:
    """List every verb a robot provides, with params and safety class."""
    from quackd.adapters.base import AdapterError
    from quackd.adapters.factory import registry_for
    from quackd.registry import Registry, RegistryError, resolve_robot_ref
    from quackd.verbs.registry import default_registry

    try:
        registry = (
            registry_for(resolve_robot_ref(robot, Registry(registry_dir)).spec)
            if robot
            else default_registry()
        )
    except (AdapterError, RegistryError) as e:
        _fail(str(e), hint=_ADAPTER_HINT)
        return
    aliases: dict[str, list[str]] = {}
    for alias, target in registry.aliases().items():
        aliases.setdefault(target, []).append(alias)
    verbs = registry.verbs()
    if as_json:
        for v in verbs:
            print(
                json.dumps(
                    {
                        "name": v.name,
                        "aliases": aliases.get(v.name, []),
                        "kind": v.kind,
                        "core": v.core,
                        "safety": v.safety_class,
                        "params": v.param_summary(),
                        "description": v.description,
                    }
                )
            )
        return
    table = ui.table(f"verbs ({robot or 'microduck'})")
    # no_wrap on the name: a narrow terminal must never elide the one column you look up
    table.add_column("name", style=ui.STYLES["key"], no_wrap=True)
    table.add_column("aliases")
    table.add_column("kind")
    table.add_column("safety")
    table.add_column("params")
    table.add_column("description")
    for v in verbs:
        kind = Text(v.kind)
        if v.core:
            kind.append(" core", style=ui.STYLES["muted"])
        table.add_row(
            Text(v.name),
            Text(", ".join(aliases.get(v.name, [])), style=ui.STYLES["muted"]),
            kind,
            Text(v.safety_class, style=ui.STYLES[_SAFETY_STYLE.get(v.safety_class, "muted")]),
            Text(v.param_summary(), style=ui.STYLES["muted"]),
            Text(v.description),
        )
    core = sum(1 for v in verbs if v.core)
    table.caption = Text(
        f"{len(verbs)} verbs, {core} core. --robot <adapter>:<backend> for another body",
        style=ui.STYLES["muted"],
    )
    table.caption_justify = "left"
    ui.console.print(table)


@app.command("list-adapters", rich_help_panel="Inspect")
def list_adapters_cmd(as_json: bool = _JSON) -> None:
    """List the robot adapters this build knows, their backends and status."""
    from quackd.adapters.factory import list_adapters

    rows = list_adapters()
    if as_json:
        for row in rows:
            print(json.dumps(row))
        return
    ui.console.print(ui.adapters_table(rows))


def _vendor_hint(vendor: str) -> str:
    """The half-line beside a vendor in shell completion: what taking it bare would mean."""
    if vendor == "fake":
        return "scripted, no key and no network"
    default = default_model_for(vendor)
    return f"the default, {default}" if default else "the first model the server serves"


def _complete_llm(ctx: typer.Context, incomplete: str) -> list[tuple[str, str]]:
    """`--llm` in the shell: vendors before the colon, that vendor's ids after it.

    Reads the catalogue and nothing else. Every press of TAB runs this, and the factory next
    door imports pydantic, so completion that reached for it would make the shell pay for a
    validator in order to spell a model id.

    A vendor is offered twice, bare and with a colon, so one TAB takes its default model and a
    second carries on into its list. After a colon the ids offered are that vendor's own, which
    is why `--llm grok:gpt` offers nothing: it would be refused, and offering it would be
    completion arguing with the parser. A bare id completes too, both because `--llm
    claude-opus-5` is a legal spec on its own and because bash breaks its words at the colon and
    hands this only the half after it.
    """
    head, colon, prefix = incomplete.partition(":")
    vendor = head.strip().lower()
    if colon:
        return [
            (f"{vendor}:{m.id}", m.label) for m in models_for(vendor) if m.id.startswith(prefix)
        ]
    found = [(v, _vendor_hint(v)) for v in PROVIDER_NAMES if v.startswith(vendor)]
    found += [
        (f"{v}:", "then a model id") for v in PROVIDER_NAMES if v != "fake" and v.startswith(vendor)
    ]
    if head:
        # Only once something is typed: a bare TAB should offer the sixteen vendors, not the
        # hundred-odd ids underneath them.
        found += [
            (m.id, f"{v}: {m.label}")
            for v in CLOUD_NAMES
            for m in models_for(v)
            if m.id.startswith(head)
        ]
    return found


# ── list-models ───────────────────────────────────────────────────────────────────────────


@app.command("list-models", rich_help_panel="Inspect")
def list_models_cmd(
    llm: str | None = typer.Option(
        None,
        "--llm",
        "-l",
        help="One vendor only, e.g. --llm openai. A whole spec or a model id is read for its "
        "vendor, so --llm claude-opus-5 lists anthropic. Omitted: every vendor.",
        autocompletion=_complete_llm,
    ),
    as_json: bool = _JSON,
) -> None:
    """List the model ids each cloud vendor accepts after the colon in --llm VENDOR:MODEL."""
    from quackd.agent.providers.base import ProviderError
    from quackd.agent.providers.factory import parse_llm

    provider = None
    if llm is not None:
        # A spec, a bare vendor or a bare id: whichever it is, what this command wants out of
        # it is the vendor, so it is read the same way `--llm` itself is read. Folded once and
        # used for both lookups: folding it for the vendor test and not for the catalogue one
        # refused `--llm CLAUDE-OPUS-5` while quoting back a string that works, which reads as
        # quackd disagreeing with itself about its own shift key.
        head = llm.split(":", 1)[0].strip().lower()
        provider = head if head in PROVIDER_NAMES else vendor_of(head)
        if provider is None:
            _fail(
                f"unknown provider {head!r}",
                hint=f"one of: {', '.join(PROVIDER_NAMES)}",
            )
            return
    vendors = [provider] if provider in CLOUD_NAMES else list(CLOUD_NAMES)
    rows = [
        {
            "provider": str(name),
            "id": m.id,
            "label": m.label,
            "status": m.status,
            "default": i == 0,
            "api": m.api,
            "vision": m.vision,
        }
        for name in vendors
        if provider is None or provider in CLOUD_NAMES
        for i, m in enumerate(models_for(str(name)))
    ]
    if as_json:
        for row in rows:
            print(json.dumps(row))
        return

    if rows:
        table = ui.table("models (--llm VENDOR:MODEL, QUACKD_LLM)")
        table.add_column("provider", style=ui.STYLES["key"], no_wrap=True)
        # An id is meant to be copied in after the colon, so it may wrap but never elide:
        # Rich's default would put an ellipsis through the middle of the one column that has to
        # survive an 80 column pipe intact.
        table.add_column("id", style=ui.STYLES["key"], overflow="fold")
        table.add_column("label", overflow="fold")
        table.add_column("status")
        table.add_column("notes")
        last = ""
        for row in rows:
            # Words, not glyphs: this table is read through a cp1252 pipe on Windows, where a
            # tick mark is the difference between a column and a row of question marks.
            marks: list[str] = []
            if row["default"]:
                marks.append("default")
            if row["api"] == "responses":
                marks.append("Responses API")
            if not row["vision"]:
                marks.append("no frames")
            table.add_row(
                Text(str(row["provider"]) if row["provider"] != last else ""),
                Text(str(row["id"])),
                Text(str(row["label"])),
                Text(str(row["status"])),
                Text(", ".join(marks), style=ui.STYLES["muted"]),
            )
            last = str(row["provider"])
        ui.console.print(table)

    notes: list[str] = []
    if provider is None or provider in LOCAL_NAMES:
        notes.append(
            f"{', '.join(LOCAL_NAMES)}: no catalogue. `--llm PRESET:MODEL` takes any id the "
            "server serves, and without one quackd takes the first entry of /v1/models."
        )
    if provider is None or provider == "fake":
        notes.append("fake: scripted, and a model after the colon is ignored.")
    if pinned := os.environ.get(LLM_ENV):
        try:
            vendor, model_id = parse_llm(pinned, source=LLM_ENV)
            named = f"{vendor}:{model_id}" if model_id else f"{vendor}, its default model"
            notes.append(f"{LLM_ENV}={pinned} pins {named}.")
        except ProviderError as e:
            notes.append(f"{LLM_ENV}={pinned} is refused: {e}")
    for note in notes:
        ui.console.print(Text(note, style=ui.STYLES["muted"]), soft_wrap=True)


# ── run / record ────────────────────────────────────────────────────────────────────────


def _yes_to_go(_why: str) -> bool:
    """`--yes` answers the pilot's doubt the way it answers a confirm gate: go."""
    return True


class _TerminalHandOff:
    """The person at the robot, as a terminal.

    Enter is read through the kill switch rather than with an `input()` of its own. The switch
    already runs the only thread reading stdin, and a second reader would race it for the same
    keystroke: whichever lost would sit on a line the other had taken."""

    asks_a_person = True
    """There is a person at the robot: what this asks goes in the record as a `prompt`.

    Flatly true, and not `_can_prompt` like the three above, because the hand-off is only
    wired in at all where `_can_prompt()` has already said yes.

    A flock member and a `--dry-run` have no hand-off at all, so this mark is never the thing
    that decides it; it is here so the loop does not have to know which of its callables
    reaches a terminal."""

    WAIT_ENDED = {
        "enter": "(Enter)",
        "kill switch": "(the kill switch ended the wait, without an Enter)",
        "no keys": "(no key could be read, so the wait ended without an Enter)",
        "stopped": "(the run was stopped, which ended the wait without an Enter)",
        "timeout": "(nobody answered: the wait ran out without an Enter)",
    }
    """What the saved terminal says about each way a wait ends, since none of them printed."""

    def __init__(self) -> None:
        self.switch: KillSwitch | None = None
        self.ended: str | None = None
        """How the last wait ended: `enter`, `kill switch` (a Ctrl-C, or `q`), `no keys` (no
        key thread, or stdin finished), `stopped` (an abort nobody pressed, on a wait that
        watches the abort flag) or `timeout`. `wait` still answers with a bool, which is
        all the first `--by-hand` wait needs, since it watches the abort flag and reads that
        after a False. The two waits that watch a fresh key press instead, the hand-back at the
        end of a `--by-hand` run and the end-of-run offer, read this as well, because the abort
        flag is set on every run a person ended and says nothing about this wait, and a record
        that files a Ctrl-C under "nobody answered" says the room was empty when somebody in it
        pressed a key."""

    def bind(self, switch: KillSwitch) -> None:
        """The switch is built from the loop's own abort event, which does not exist until the
        loop does, and the loop is built from the config this object is already in."""
        self.switch = switch

    def say(self, text: str) -> None:
        with ui.pause_status():
            ui.err_console.print(Text(text, style=ui.STYLES["warn"]))

    async def wait(
        self, text: str, *, timeout_s: float | None = None, until_abort: bool = True
    ) -> bool:
        if self.switch is None:  # `bind` runs before the loop does, so this is a bug if hit
            raise RuntimeError("the hand-off has no kill switch to read Enter from")
        self.say(text)
        self.ended = None
        # counted rather than read off `pressed`, which the wait itself clears on the way in: a
        # press is counted on the signal's own thread before the loop is told of it, so one
        # that ended this wait has always been counted by the time it returns
        presses = self.switch.presses
        came = await self.switch.wait_for_enter(timeout_s=timeout_s, until_abort=until_abort)
        if came:
            self.ended = "enter"
        elif self.switch.presses > presses:
            self.ended = "kill switch"
        elif self.switch.keys_ended.is_set():
            self.ended = "no keys"
        elif until_abort and self.switch.abort.is_set():
            # a heartbeat that failed, or a flock stopping its members: an abort nobody pressed
            self.ended = "stopped"
        else:
            self.ended = "timeout"
        # Enter is a keystroke nobody printed, and not pressing it is the more interesting
        # half: a hand-off that timed out is why the arm was left where it was.
        ui.note(self.WAIT_ENDED[self.ended])
        return came


_SWITCH: KillSwitch | None = None
"""The kill switch of the run that is going on, or None outside one.

Module level because the prompts below are plain callables handed to `RunConfig` long before
the loop they will be asked from exists, and the switch is built from that loop's own abort
event. `_ask` is the only reader."""


def _ask(question: str) -> bool:
    """A yes or no question, read off the terminal without racing the run for the keystroke.

    `typer.confirm` calls `input()`, and the kill switch's key thread is reading the same
    terminal: whichever of the two took a character first kept it, so a gate asked on a real
    terminal waited for a newline that had already been swallowed, with a robot mid-verb.
    Where a switch is running its own reader answers; where none is, this is `typer.confirm`
    as it always was.

    Both branches tell the saved terminal what was typed, because neither of them printed it:
    the question goes out raw and the answer is echoed by the terminal driver, so a tee on
    quackd's own output sees the asking and not the answering. A run with `--yes` never
    reaches here, which is why a recorded question always means a person was really asked."""
    switch = _SWITCH
    if switch is None:
        agreed = typer.confirm(question, default=False)
        ui.note(f"{question} [y/N]: {'y' if agreed else 'n'}")
        return agreed
    answer = switch.ask(f"{question} [y/N]: ")
    ui.note(f"{question} [y/N]: {answer.strip() or '(Enter)'}")
    return answer.strip().lower() in ("y", "yes")


def _confirm_prompt(name: str, params: dict[str, Any]) -> bool:
    # under a running status line the question is invisible: a live region redirects stdout
    # and a prompt writes without a newline, so it stays buffered until it is too late.
    # `fmt_params` and not the raw dict, because the executor writes the same sentence into the
    # record and the record has to quote the question in the words it was asked in.
    from quackd.log import fmt_params

    with ui.pause_status():
        return _ask(f"run {name}({fmt_params(params)})?")


def _decide_prompt(why: str) -> bool:
    """Asked when the pilot says it is not sure this body can do the task at all."""
    with ui.pause_status():
        ui.err_console.print(Text(why, style=ui.STYLES["warn"]))
        return _ask("Go ahead anyway?")


def _acknowledge_prompt(why: str) -> bool:
    """Asked once, before anything moves, when the human is the only safety left."""
    with ui.pause_status():
        ui.err_console.print(Text(why, style=ui.STYLES["warn"]))
        return _ask("Are you watching the robot right now?")


def _a_person_is_there() -> bool:
    """`_can_prompt` looked up now rather than bound now, because it is the seam the tests
    replace and a reference taken at import would not see the replacement."""
    return _can_prompt()


_confirm_prompt.asks_a_person = _a_person_is_there  # type: ignore[attr-defined]
_decide_prompt.asks_a_person = _a_person_is_there  # type: ignore[attr-defined]
_acknowledge_prompt.asks_a_person = _a_person_is_there  # type: ignore[attr-defined]
"""These three are the ones that reach a terminal, and `allow_all`, `deny_all`, `_yes_to_go`
and the flock's standing answers are not. The mark is `_can_prompt` rather than `True` because
reaching a terminal is a thing to check at the moment of asking and not a property of the
function: these same three run under `yes | quackd run` and under `quackd run < answers.txt`,
where `input()` reads the pipe and returns a yes nobody said. The gate still opens, because
that is what the pipe asked for and it is what quackd has always done; what must not happen is
the record then testifying that a person cleared it."""


def _parse_flock_flag(flock: str | None, registry_dir: str | None) -> tuple[int | None, Any]:
    """`--flock` is either a count of simulated ducks or the name of a stored flock.

    All digits is the count, because that is what it has always meant and `check_name` refuses
    to register anything that could be read as one. Anything else is a name, and its roster is
    read now rather than at the first connection, so a flock with a hole in it refuses before
    a run directory exists."""
    from quackd.registry import Registry, RegistryError

    if flock is None:
        return None, None
    flock = flock.strip()
    if flock.isdigit():
        return int(flock), None
    try:
        return None, Registry(registry_dir).roster(flock)
    except RegistryError as e:
        _fail(str(e), hint="quackd flock list, or a number for N simulated ducks")
        raise


def _run_impl(
    duckfile: str | None,
    goal: str | None,
    llm: str | None,
    seed: int | None,
    dry_run: bool,
    max_steps: int | None,
    runs_dir: str,
    yes: bool,
    live: bool,
    address: str | None,
    camera_url: list[str],
    token: str | None,
    fov_deg: float | None,
    gif: bool,
    gif_size: int,
    verbose: bool,
    base_url: str | None = None,
    api_key: str | None = None,
    vision: bool | None = None,
    extra_body: str | None = None,
    flock: str | None = None,
    *,
    decision_llm: str | None = None,
    decision_url: str | None = None,
    decision_mode: str | None = None,
    images: Sequence[str] = (),
    by_hand: bool = False,
    robot: str | None = None,
    robots: str | None = None,
    memory: bool = True,
    memory_dir: str | None = None,
    registry_dir: str | None = None,
    log_on: bool | None = None,
    log_prompt: bool | None = None,
    run_name: str | None = None,
    price: str | None = None,
    host: str | None = None,
    host_token: str | None = None,
    detector_choice: str | None = None,
) -> None:
    from quackd.adapters.base import AdapterError as _AdapterError
    from quackd.adapters.factory import describe, make_adapter, registry_for
    from quackd.adapters.host_camera import EXTRAS_KEY as HOST_CAMERA_EXTRAS
    from quackd.adapters.host_camera import with_host_camera
    from quackd.agent.decision.base import DecisionError
    from quackd.agent.decision.factory import PRICE_ENV as DECISION_PRICE_ENV
    from quackd.agent.decision.factory import (
        decision_llm_is_available,
        make_decision_llm,
        parse_decision_llm,
        resolve_decision_mode,
        resolve_decision_price,
    )
    from quackd.agent.images import TaskImageError, load_task_images
    from quackd.agent.loop import RunConfig, run_duck
    from quackd.agent.providers.base import ProviderError
    from quackd.agent.providers.factory import make_provider, resolve_llm
    from quackd.agent.providers.pricing import parse_price as _parse_price
    from quackd.agent.transcript import run_label
    from quackd.duckfile.parser import DuckParseError, duck_from_goal, load_duck
    from quackd.duckfile.schema import AUCTION_MAX_MEMBERS, PILOTS_MAX_MEMBERS
    from quackd.duckfile.validate import validate_duck
    from quackd.flock.pilots import ADVISORY_FIELDS, roster_from_specs
    from quackd.flock.runner import member_specs
    from quackd.host import HostChoice, HostClient, HostError, HostHello, reach_host, resolve_host
    from quackd.log import (
        ConsoleLog,
        fan_out,
        log_enabled_default,
        prompt_shown_default,
        thinking_limit_default,
    )
    from quackd.perception import DETECTOR_CHOICES, detector_for, explicit_detector
    from quackd.registry import RegistryError, Resolved
    from quackd.safety import KillSwitch, allow_all
    from quackd.transport.base import TransportError

    if (duckfile is None) == (goal is None):
        _fail('give either a .duck file (or bundled name) or --goal "...", not both')
        return
    # Both before anything is built, connected to or written down: a name that cannot be a
    # directory and a price nobody can parse are typing mistakes, and a typing mistake should
    # cost you one sentence rather than a robot moving and a run directory to clean up after.
    # `QUACKD_DECISION_PRICE` is here rather than beside the stepper for the same reason as
    # the other two: it has no flag of its own, so an unparseable line in a `.env` would
    # otherwise surface as a traceback out of the first turn that needed a rate.
    checks = (
        (run_name, run_label),
        (price, lambda t: _parse_price(t, source="--price")),
        (
            # `or None` because a blank variable is a shell saying unset, which is how the
            # suite clears it and how a `.env` line with nothing after the `=` reads.
            os.environ.get(DECISION_PRICE_ENV) or None,
            lambda t: _parse_price(t, source=DECISION_PRICE_ENV),
        ),
    )
    for text, check in checks:
        if text is None:
            continue
        try:
            check(text)
        except ValueError as e:
            _fail(str(e))
            return
    detector_choice = (detector_choice or "").strip().lower() or None
    if detector_choice is not None and detector_choice not in DETECTOR_CHOICES:
        _fail(f"--detector is one of {', '.join(DETECTOR_CHOICES)}, not {detector_choice!r}")
        return
    flock_n, roster = _parse_flock_flag(flock, registry_dir)
    flock_name = flock if roster is not None else None
    if roster is not None and (robot or robots):
        _fail("--flock NAME brings its own robots: drop --robot and --robots")
        return
    if roster is not None and (address or camera_url or token):
        # they would be silently dropped: both flock paths read each member's own
        _fail(
            "--flock NAME takes every member's address, token and camera from the registry",
            hint="quackd robot edit NAME to change one",
        )
        return
    try:
        duck = load_duck(duckfile) if duckfile is not None else None
        resolved = (
            [Resolved(entry.robot_spec, entry) for entry in roster.values()]
            if roster is not None
            else _robot_specs(robot, robots, duck, registry_dir=registry_dir)
        )
        specs = [r.spec for r in resolved]
        here = resolved[0]
        spec = here.spec
    except (DuckParseError, TransportError, RegistryError) as e:
        _fail(str(e))
        return
    # Before the dispatch below, because a flock takes neither of the two flags and dropping
    # one silently is the failure both of them exist to prevent: a task about a picture that
    # never arrived, and an arm nobody was asked to place. A goal is never a flock.
    several = (
        flock_n is not None
        or roster is not None
        or (duck is not None and duck.frontmatter.flock is not None)
        or len(specs) > 1
    )
    # A host is one machine: one camera, one detector, the health of one board. A fleet is
    # several bodies, and one --host would have to be every member's camera at once, so it is
    # refused here with --image and --by-hand, before anything connects. `--robots` counts even
    # with one member in it, because it is the fleet spelling. A host stored with a member is
    # the same claim made in robots.json, and is refused by that member's name. QUACKD_HOST is
    # not: it is the board you usually use rather than a claim about these bodies, and for a
    # fleet it moves only the local presets' model server, which LocalProvider reads itself.
    fleet = several or bool(robots)
    hosted = [r.entry.name for r in resolved if r.entry is not None and r.entry.host]
    if fleet and (host or "").strip():
        _fail(
            "--host names one machine's camera and detector, and a fleet has several bodies",
            hint="drop --host, or run the task on one body at a time",
        )
        return
    if fleet and hosted:
        _fail(
            f"{hosted[0]} has a host in robots.json, which names one machine's camera and "
            "detector, and a fleet has several bodies",
            hint=f"quackd robot edit {hosted[0]} --clear host, or run it on its own",
        )
        return
    # The one place this run's board is settled, so everything that uses the board reads this
    # value rather than deriving its own. A fleet has none, whatever the environment says.
    stored = here.host_kwargs()
    try:
        host_choice = (
            HostChoice()
            if fleet
            else resolve_host(
                host,
                stored["host"],
                token=host_token,
                stored_token=stored["host_token"],
                robot=here.entry.name if here.entry is not None else None,
            )
        )
    except ValueError as e:
        _fail(str(e))
        return
    # The board is asked what it is before the task file is judged and before anything is
    # built, because both depend on the answer: a camera on the board is a camera the body
    # has, so a camera task on a blind body is not refused for a camera the run will have, and
    # a board that does not answer is a run refused while nothing is powered.
    board: HostClient | None = None
    hello: HostHello | None = None
    try:
        reached = reach_host(host_choice)
    except HostError as e:
        _fail(
            f"{_host_named(host_choice)} did not answer: {e}",
            hint=_host_unreached_hint(
                host_choice, here.entry.name if here.entry is not None else None
            ),
        )
        return
    if reached is not None:
        board, hello = reached
    try:
        if goal is not None:
            # the union across the flock, so a goal run on mixed bodies allows what any of
            # them can do; each member is then trimmed to its own half of that. With a board,
            # the one body's vocabulary includes what the board's camera lets it do.
            safe = sorted(
                {
                    v.name
                    for one in specs
                    for v in registry_for(
                        one,
                        with_host_camera(describe(one), hello) if hello is not None else None,
                    ).verbs()
                    if v.safety_class == "safe"
                }
            )
            duck = duck_from_goal(goal, safe)
        assert duck is not None
        # Refuse before connecting, with the validator's words. `serve-mcp` has always done
        # this; `run` never did, and reached the loop's tool_schemas and died on a raw
        # VerbNotFound with the robot already connected and a run directory already made.
        manifests = [describe(s) for s in specs]
        # the body as its adapter describes it, before the board's camera joins it: all a
        # detector built before connect may know about the lens (below)
        described = manifests[0]
        if hello is not None:
            manifests[0] = with_host_camera(described, hello)
        section = duck.frontmatter.flock
        method = (
            section.allocation.method
            if section is not None
            else ("pilots" if roster is not None else None)
        )
        # a task file with no `flock:` block, run against a stored flock, is still a flock:
        # judged body by body the arm would be refused for not being able to walk. And a
        # pilot flock drops the advisory `verbs.allow` line, because trimming each member to
        # its own vocabulary is its answer to it (`pilots.ADVISORY_FIELDS`).
        problems = [
            p
            for p in validate_duck(duck, manifests, flock=True if roster is not None else None)
            if not (method == "pilots" and p.field in ADVISORY_FIELDS)
        ]
    except (DuckParseError, TransportError, RegistryError) as e:
        _fail(str(e))
        return
    if problems:
        _fail(
            f"{duck.name} cannot run on {', '.join(s.key for s in specs)}: "
            + "; ".join(p.message for p in problems)
        )
        return
    if flock_n is not None and not 2 <= flock_n <= 4:
        _fail("a flock needs 2 to 4 ducks (drop --flock for a single run)")
        return
    if detector_choice is not None and several:
        # every flock path builds its own members' detectors, so a choice made here would be
        # dropped without a word, which is the thing a flag must never do
        _fail(
            "--detector is for one robot, and this run has several",
            hint="drop --detector, or run the task on one body at a time",
        )
        return
    task_images: list[Any] = []
    if images:
        if several:
            _fail(
                "--image is for one robot, and this run has several",
                hint="drop --flock and --robots, or run the task on one body at a time",
            )
            return
        try:
            task_images = load_task_images(list(images))
        except TaskImageError as e:
            _fail(str(e))
            return
    if by_hand:
        if several:
            _fail(
                "--by-hand is one person placing one arm, and this run has several robots",
                hint="drop --flock and --robots",
            )
            return
        if dry_run:
            # a dry run moves nothing at either end, and taking torque off an arm is the one
            # thing here that is not a command to the robot but a change to it
            _fail(
                "--by-hand and --dry-run ask for opposite things: one takes torque off the "
                "arm, the other moves nothing",
                hint="rehearse the task with --dry-run, then run it again with --by-hand",
            )
            return
        if not _can_prompt():
            _fail(
                "--by-hand waits for you to press Enter, and there is no terminal to ask on",
                hint="run it from a terminal, or drop the flag and start from the rest pose",
            )
            return
    # Resolved here rather than beside the solo run below, because both flock branches return
    # before that point: `--decision-mode maybe` on a flock used to run the robots anyway, and
    # a mode that was spelled right used to be accepted and silently do nothing.
    try:
        named_decision = parse_decision_llm(decision_llm)
        decision = resolve_decision_mode(
            decision_mode,
            named=named_decision is not None,
            # Said on the line rather than merely absent. `--decision-llm off` is how one
            # command opts out of a `QUACKD_DECISION_LLM` in a `.env`, and refusing it because
            # the same `.env` also set a mode would answer "I do not want this" with a demand
            # to name one.
            refused=(decision_llm or "").strip().lower() == "off",
        )
    except DecisionError as e:
        _fail(str(e))
        return
    if flock_n is not None or roster is not None or duck.frontmatter.flock is not None:
        if decision != "off" and named_decision is not None:
            # One loop per member, each with its own executor and budget, and the stepper is
            # built per loop. Wiring it through a flock is a thing to do deliberately with a
            # measurement in hand, not a thing to leave half done and unsaid.
            ui.console.print(
                _warn_line(
                    f"--decision-llm {named_decision[0].name} does not apply to a flock: "
                    "every member is piloted by its model, as before"
                ),
                soft_wrap=True,
            )
        if method == "pilots":
            if roster is None and section is not None:
                # `flock.members` plus `robots:` or `--robots` names the bodies without a
                # registry; a stored flock names them with one
                try:
                    roster = roster_from_specs(
                        member_specs(
                            section.member_names,
                            {s.name: s.key for s in specs if s.name} or None,
                            duck.frontmatter.robots,
                            # a pilot flock is N bodies of any kind, so an unnamed member gets
                            # this machine's default rather than the simulated duck a
                            # coordinator flock is made of
                            fallback=None,
                        )
                    )
                except _AdapterError as e:
                    # a member nothing named, on a machine that will not guess: the refusal
                    # names what to install or what to type. It is caught here because this
                    # call sits past the validation block's own handler, and an uncaught
                    # NoRobotNamed reaches the user as a traceback rather than one line.
                    _fail(str(e), hint="name every member's body in the task file's robots:")
                    return
            if roster is None:
                _fail(
                    "allocation.method: pilots needs members: name them in flock.members, "
                    "or run a stored flock with --flock NAME"
                )
                return
            if flock_n is not None:
                _fail(
                    "--flock N is the coordinator, and this task file runs pilots",
                    hint="name the members in flock.members, or run a stored flock: --flock NAME",
                )
                return
            if not 2 <= len(roster) <= PILOTS_MAX_MEMBERS:
                _fail(
                    f"a pilot flock needs 2 to {PILOTS_MAX_MEMBERS} members; "
                    f"{flock_name or 'this task file'} names {len(roster)}"
                )
                return
            _run_pilots_impl(
                duck,
                roster,
                llm=llm,
                seed=seed,
                dry_run=dry_run,
                runs_dir=runs_dir,
                yes=yes,
                live=live,
                verbose=verbose,
                goal=goal,
                base_url=base_url,
                api_key=api_key,
                vision=vision,
                extra_body=extra_body,
                max_steps=max_steps,
                fov_deg=fov_deg,
                memory=memory,
                memory_dir=memory_dir,
                flock_name=flock_name,
                log_on=log_on,
                log_prompt=log_prompt,
                run_name=run_name,
                price=price,
            )
            return
        if roster is not None:
            wrong = [n for n, e in roster.items() if e.robot_spec.key != "microduck:sim2d"]
            if wrong:
                _fail(
                    f"a coordinator flock is sim2d Microducks only (docs/flock.md): "
                    f"{wrong[0]} is {roster[wrong[0]].robot_spec.key}",
                    hint="set flock.allocation.method: pilots to run other bodies",
                )
                return
            if not 2 <= len(roster) <= AUCTION_MAX_MEMBERS:
                # both bounds, because the members are folded in with `model_copy`, which
                # skips the field validator that would otherwise have caught one of them
                _fail(
                    f"a coordinator flock is 2 to {AUCTION_MAX_MEMBERS} ducks "
                    f"(the arena holds {AUCTION_MAX_MEMBERS}): {flock_name} has "
                    f"{_plural(len(roster), 'robot')}"
                )
                return
        _run_flock_impl(
            duck,
            llm=llm,
            specs=specs,
            members=list(roster) if roster is not None else None,
            flock_name=flock_name,
            seed=seed,
            dry_run=dry_run,
            runs_dir=runs_dir,
            yes=yes,
            live=live,
            gif=gif,
            gif_size=gif_size,
            verbose=verbose,
            goal=goal,
            base_url=base_url,
            api_key=api_key,
            vision=vision,
            extra_body=extra_body,
            n_override=flock_n,
            max_steps=max_steps,
            log_on=log_on,
            log_prompt=log_prompt,
            run_name=run_name,
            price=price,
        )
        return
    # The lens a detector built now measures through: --fov-deg, else the body's own. Never
    # the board's yet: a body described without a camera may report one of its own when it
    # connects (an arm given --camera-url, a rosbridge base), and the board's camera is then
    # only an extra view, so its lens would measure the body's camera. The loop sets the lens
    # again from the live manifest at connect, which carries the board's when the board's
    # camera turned out to be the only one (`HostCameraAdapter.connect`).
    lens_fov = fov_deg or described.limits.get("camera_fov_deg")
    try:
        # chosen once, before anything is built: a detector asked for that cannot run here is
        # a sentence now, and the run never changes detector after it starts
        asked_detector = explicit_detector(
            detector_choice,
            client=board,
            hello=hello,
            fov_deg=lens_fov,
            backend=spec.backend,
            has_camera="camera" in described.sensors,
        )
    except (ValueError, ImportError) as e:
        _fail(str(e))
        return
    try:
        # a registered robot may name the pilot that drives it; a flag on the line still wins,
        # and `QUACKD_LLM` sits behind both. One spec carries the vendor and the model
        # together, so there is no longer any way for half an answer to come from each place:
        # `--llm gemini` on a robot registered against OpenAI is Gemini's default, full stop.
        vendor, model_id, llm_source = resolve_llm(
            llm, here.llm, robot=here.entry.name if here.entry is not None else None
        )
        pilot = make_provider(
            vendor,
            model=model_id,
            source=llm_source,
            duck_name=duck.name,
            goal=goal,
            base_url=base_url,
            api_key=api_key,
            vision=vision,
            extra_body=extra_body,
            # only a host a person named for this run or this robot: QUACKD_HOST sits below
            # QUACKD_BASE_URL, and the local provider reads it there for itself
            host=host_choice.explicit,
        )
        duck_transport = make_adapter(
            spec,
            seed=seed,
            live=live,
            # the board's camera joins the body here when it has one; the board has answered
            # its hello already, so building this asks it nothing
            host=board,
            **here.adapter_kwargs(address=address, camera_url=camera_url, token=token),
        )
    except (ProviderError, TransportError, ImportError) as e:
        _fail(str(e))
        return
    if by_hand:
        if not getattr(duck_transport, "supports_hand_off", False):
            _fail(
                f"{spec.key} is not a body a person places by hand: only the LeRobot arm is",
                hint="quackd list-adapters",
            )
            return
        if here.adapter_kwargs()["rest_pose"] is None:
            # the release refuses anywhere but the recorded pose, so an arm without one could
            # never be handed over at all: better said here than after it has connected
            named = here.entry.name if here.entry is not None else None
            _fail(
                "--by-hand releases the arm at its recorded rest pose, and this arm has none "
                "recorded",
                hint=(
                    f"quackd robot rest-pose {named}"
                    if named
                    else "quackd robot add NAME " + spec.key + ", then quackd robot rest-pose NAME"
                ),
            )
            return
    # A name or a mode nobody defined is a typo and stopped the run above. One that is
    # spelled right and cannot run here is a different thing: the stepper is an optimisation,
    # the model is the pilot either way, and a script that always names a decision LLM should
    # still drive the robot on a machine that has not installed it. So it says so once,
    # loudly, and carries on without it. Said before the robot is connected, so nothing is
    # energised while it is read.
    decision_pilot = None
    decision_price = None
    if decision != "off" and named_decision is not None:
        # `chosen` rather than `spec`, which in this function is the robot's.
        chosen, decision_model = named_decision
        available, why = decision_llm_is_available(chosen)
        if not available:
            ui.console.print(
                _warn_line(f"--decision-llm {chosen.name} asked for, running without it: {why}"),
                soft_wrap=True,
            )
            decision = "off"
        else:
            try:
                decision_pilot = make_decision_llm(chosen, model=decision_model, url=decision_url)
            except DecisionError as e:
                _fail(str(e))
                return
            decision_price = resolve_decision_price(chosen)
    if decision == "on":
        # Nobody has run a stepper against a robot, so every speed and cost figure in
        # `docs/decision-llms.md` is arithmetic from published numbers rather than a result.
        # Two of the four confidence floors are the 0.5 and 0.9 TypeSafe publish and two are
        # quackd's own, and all four were shaped around Jev and are inherited unmeasured by
        # every other row.
        # Refusing the flag over that would be the wrong shape of gate, because the executor
        # binds a stepper-authored call exactly as it binds the model's. Saying it once, where
        # the person switching it on is looking, is the right size of one.
        ui.console.print(
            _warn_line(
                "--decision-mode on has not been measured against a real robot: no latency, no "
                "agreement rate, and the figures in docs/decision-llms.md are estimates for "
                "Jev and nothing at all for anything else. Two of the four confidence floors "
                "are published by TypeSafe and two are quackd's own, all four shaped around "
                "Jev. --decision-mode shadow records both and changes nothing "
                "about the run."
            ),
            soft_wrap=True,
        )
    if task_images and not pilot.supports_vision:
        # Refused rather than dropped. A pilot that cannot see would be handed "draw what is
        # in the picture" with no picture, improvise something, and the only sign of why would
        # be a note in a transcript nobody reads twice.
        _fail(
            f"{pilot.name} {pilot.model} does not take images, so it cannot be given "
            f"{_plural(len(task_images), 'picture')}",
            hint="quackd list-models marks the models that take no frames; --vision overrides "
            "it where the vendor does take them, and a local model needs --vision",
        )
        return

    recorder = None
    # Any robot with a camera needs something to look at its frames with, not just the
    # simulator. This is the static manifest, so it is only a head start: the loop asks
    # again with the live one at connect, where a robot may report a camera this does not
    # know about (a rosbridge base) or lack one this promises (a duck built without a head).
    # A detector chosen above is kept as it is, here and at connect. The body's own
    # description, not the one with the board's camera: a body described without a camera
    # gets its colour detector at connect, measured through whichever lens is then primary.
    detector = detector_for(
        described.sensors, asked_detector, fov_deg=lens_fov, backend=spec.backend
    )
    # the recorder is sim2d only: it draws the world, and only the simulator has one
    if spec.backend in ("sim2d", "mujoco") and gif:
        from quackd.sim2d.recorder import FrameRecorder

        recorder = FrameRecorder(duck_transport, size=gif_size)

    # The flag wins; else QUACKD_LOG, read here rather than at import so a `.env` line
    # counts (the root callback loads it after the option defaults exist).
    log_on = log_on if log_on is not None else log_enabled_default()
    console_log = (
        ConsoleLog(
            ui.err_console,
            thinking_chars=thinking_limit_default(),
            prompt=log_prompt if log_prompt is not None else prompt_shown_default(),
        )
        if log_on
        else None
    )

    # on whether or not the log is: with --no-log this is the only thing between the
    # header and the verdict, and a model can think for a minute
    status = ui.RunStatus()
    ui.install_logging()

    def log(msg: str) -> None:
        # the compact view: one line per verb and the executor's notes. The log shows all
        # of that and more, so with it on this prints nothing rather than every verb twice.
        if verbose and console_log is None:
            _verbose_line(msg)

    hand_off = _TerminalHandOff() if by_hand else None
    # Whoever is at this terminal, for the one question a run can put at its very end: the
    # rest move missed, and would they like torque off while they hold the arm. Not `hand_off`,
    # which the loop reads as "this run is handed over by hand", but the same object when
    # there is one, so a run reads Enter through one reader. None without a terminal to ask on
    # and on a dry run, which moves nothing and so has no rest move to miss.
    person = hand_off or (_TerminalHandOff() if _can_prompt() and not dry_run else None)
    robot_memory = None
    if memory:
        from quackd.memory import RobotMemory

        # keyed by adapter:backend, so a simulated duck never inherits a real one's notes,
        # or by the registered name, so two ducks of one kind keep separate notes
        robot_memory = RobotMemory(here.memory_key, memory_dir)
    cfg = RunConfig(
        duck=duck,
        provider=pilot,
        transport=duck_transport,
        detector=detector,
        dry_run=dry_run,
        confirm=allow_all if yes else _confirm_prompt,
        runs_dir=runs_dir,
        run_name=run_name,
        price=price,
        max_steps=max_steps,
        log=log,
        on_frame=recorder.capture if recorder is not None else None,
        memory=robot_memory,
        fov_deg=fov_deg,
        acknowledge=None if yes else _acknowledge_prompt,
        decide=_yes_to_go if yes else _decide_prompt,
        view=fan_out(console_log, status.sink),
        task_images=task_images,
        hand_off=hand_off,
        person=person,
        decision=decision,
        decision_llm=decision_pilot,
        decision_price=decision_price,
        host=hello.record(board.address) if board is not None and hello is not None else None,
    )
    ui.console.print(
        ui.run_header(
            duck.name,
            _header_rows(
                provider=pilot,
                robot=here.label,
                seed=seed,
                dry_run=dry_run,
                memory=robot_memory,
                detector=detector,
                board=board,
                hello=hello,
                backend=spec.backend,
                host_camera=manifests[0].extras.get(HOST_CAMERA_EXTRAS),
                sees="camera" in manifests[0].sensors,
            ),
            hint="Ctrl-C or q stops the duck. Press it twice to quit at once.",
        )
    )

    def killed(msg: str) -> None:
        """Always printed, unlike `log`, which is --verbose only. Someone who has just hit
        Ctrl-C on a walking robot needs to see that it registered."""
        ui.err_console.print(Text(msg, style=ui.STYLES["warn"]))

    async def main() -> Any:
        from quackd.agent.loop import AgentLoop

        global _SWITCH
        loop = AgentLoop(cfg)
        # The first moment there is a directory to write into. Everything printed before
        # now went into the capture's buffer and is carried across by `attach`.
        ui.attach_capture(loop.run_dir)
        ks = KillSwitch(loop.executor.abort, log=killed)
        if hand_off is not None:
            hand_off.bind(ks)
        if person is not None and person is not hand_off:
            person.bind(ks)
        ks.install()
        _SWITCH = ks
        try:
            return await loop.run()
        finally:
            _SWITCH = None
            ks.uninstall()

    _ = run_duck  # imported for symmetry; AgentLoop is used directly so the kill switch can bind
    try:
        with status:
            status.update(f"connecting to {here.label}")
            result = asyncio.run(main())
    except (TransportError, ProviderError) as e:
        # the log has already shown the call that failed; this is the one-line verdict
        _fail(str(e))
        return
    if recorder is not None:
        # after the status line rather than under it: Rich 13.7 refuses a second live region
        # on one console, and a long run can be a thousand frames to quantise
        with ui.spinner(f"encoding {len(recorder.frames)} frames into run.gif"):
            gif_path = recorder.save_gif(result.run_dir / "run.gif")
        result.gif_path = gif_path
    _print_outcome(
        result.outcome,
        result.reason,
        counters=run_counters(result.summary),
        run_dir=result.run_dir,
        gif_path=result.gif_path,
        log_dropped=result.log_dropped,
    )
    if result.summary.get("cost_usd") is None and result.summary.get("provider") not in (
        "fake",
        None,
    ):
        # One line, in the style of the dropped-events warning above and in the same yellow:
        # a run that could not be costed should say why and how to fix it, once, rather than
        # leaving a reader to wonder whether the number is missing or zero.
        ui.err_console.print(
            f"cost: quackd has no published rate for {result.summary.get('provider')} "
            f"{result.summary.get('model')}; pass --price in=N,out=N to compute one",
            style="yellow",
            markup=False,
        )
    if result.outcome == "infeasible":
        # its own code: 1 means the run happened and did not succeed, and a script trying one
        # body after another branches on "this body could not, try the next"
        raise typer.Exit(code=EXIT_INFEASIBLE)
    if result.outcome != "success":
        raise typer.Exit(code=1)


def _member_views(
    member_names: list[str],
    *,
    log_on: bool,
    log_prompt: bool | None,
    status: Any,
) -> tuple[dict[str, Any], Any]:
    """One console view per member, coloured and prefixed by name, plus the flock's own.

    Shared by both kinds of flock, because a person reading either one needs the same thing:
    several robots narrating at once stay several readable columns rather than one
    interleaving. Returns the views (so the caller can flush them) and the `view(name)`
    factory the runner takes."""
    from quackd.flock.runner import FLOCK_LOG
    from quackd.log import (
        ConsoleLog,
        Sink,
        fan_out,
        prompt_shown_default,
        thinking_limit_default,
    )

    views: dict[str, ConsoleLog] = {}
    width = max(len(name) for name in [*member_names, FLOCK_LOG])

    def view_for(name: str) -> Sink | None:
        if name not in views:
            # a colour per member as well as a name, because robots moving at once interleave
            # and the eye finds a colour faster than it reads a prefix
            order = member_names.index(name) if name in member_names else -1
            views[name] = ConsoleLog(
                ui.err_console,
                thinking_chars=thinking_limit_default(),
                prompt=log_prompt if log_prompt is not None else prompt_shown_default(),
                prefix=f"{name:<{width}}  ",
                prefix_style=ui.MEMBER_STYLES[order % len(ui.MEMBER_STYLES)]
                if order >= 0
                else ui.STYLES["key"],
            )
        return fan_out(views[name], status.sink)

    def status_only(_name: str) -> Sink | None:
        """With --no-log nothing narrates, but the status line still has to say which robot
        is doing what, or a flock is a minute of nothing at all."""
        return status.sink

    return views, (view_for if log_on else status_only)


def _run_pilots_impl(
    duck: Any,
    roster: Any,
    *,
    llm: str | None,
    seed: int | None,
    dry_run: bool,
    runs_dir: str,
    yes: bool,
    live: bool,
    verbose: bool,
    goal: str | None,
    base_url: str | None,
    api_key: str | None,
    vision: bool | None,
    extra_body: str | None,
    max_steps: int | None,
    fov_deg: float | None,
    memory: bool,
    memory_dir: str | None,
    flock_name: str | None,
    log_on: bool | None = None,
    log_prompt: bool | None = None,
    run_name: str | None = None,
    price: str | None = None,
) -> None:
    """A pilot per body, all at once. The other flock is `_run_flock_impl`."""
    from quackd.agent.providers.base import ProviderError
    from quackd.agent.providers.factory import make_provider, resolve_llm
    from quackd.agent.providers.pricing import fmt_usd
    from quackd.flock.pilots import run_pilot_flock
    from quackd.log import fmt_duration, log_enabled_default
    from quackd.memory import RobotMemory
    from quackd.safety import KillSwitch
    from quackd.transport.base import TransportError

    members = list(roster)
    if duck.frontmatter.verbs.confirm and not yes:
        _fail("a pilot flock cannot prompt y/N per member: empty verbs.confirm or pass --yes")
        return
    try:
        providers = {}
        for name, entry in roster.items():
            vendor, model_id, llm_source = resolve_llm(llm, entry.llm, robot=name)
            providers[name] = make_provider(
                vendor,
                model=model_id,
                source=llm_source,
                duck_name=duck.name,
                goal=goal,
                base_url=base_url,
                api_key=api_key,
                vision=vision,
                extra_body=extra_body,
            )
    except (ProviderError, ImportError) as e:
        _fail(str(e))
        return
    memories = (
        {name: RobotMemory(entry.memory_key, memory_dir) for name, entry in roster.items()}
        if memory
        else None
    )

    log_on = log_on if log_on is not None else log_enabled_default()

    def log(msg: str) -> None:
        # the log says all of this and more, so two views of one line is noise
        if verbose and not log_on:
            _verbose_line(msg)

    status = ui.RunStatus()
    ui.install_logging()
    views, view_factory = _member_views(
        members, log_on=log_on, log_prompt=log_prompt, status=status
    )
    ui.console.print(
        ui.run_header(
            duck.name,
            _pilot_header_rows(roster, providers, memories, flock_name=flock_name, dry_run=dry_run),
            hint="Ctrl-C or q stops every robot. Press it twice to quit at once.",
        )
    )

    def killed(msg: str) -> None:
        ui.err_console.print(Text(msg, style=ui.STYLES["warn"]))

    async def main() -> Any:
        master = asyncio.Event()
        ks = KillSwitch(master, log=killed)
        ks.install()
        try:
            return await run_pilot_flock(
                duck,
                roster,
                providers=providers,
                seed=seed,
                runs_dir=runs_dir,
                dry_run=dry_run,
                max_steps=max_steps,
                live=live,
                yes=yes,
                memories=memories,
                fov_deg=fov_deg,
                log=log,
                view=view_factory,
                abort=master,
                flock_name=flock_name,
                run_name=run_name,
                price=price,
                on_run_dir=ui.attach_capture,
            )
        finally:
            ks.uninstall()

    try:
        with status:
            status.update(f"connecting {_plural(len(members), 'robot')}")
            result = asyncio.run(main())
    except (ValueError, TransportError, ProviderError, ImportError) as e:
        _fail(str(e))
        return
    finally:
        # a member's last event is the stop it was accepted for, and a pending burst is only
        # written by the next event that is not an intent: without this, never
        for pending in views.values():
            pending.flush()
    ok = sum(1 for row in result.per_member.values() if row["outcome"] == "success")
    _print_outcome(
        result.outcome,
        result.reason,
        counters=[
            f"members {ok}/{len(members)} succeeded",
            f"talk {result.messages}",
            f"steps {result.steps}",
            f"llm calls {result.llm_calls}",
            f"tokens {result.usage.input_tokens}+{result.usage.output_tokens}",
            f"time {fmt_duration(result.wall_elapsed_s)}",
            f"cost {fmt_usd(result.cost_usd)}",
        ],
        run_dir=result.run_dir,
        log_dropped=result.log_dropped,
    )
    if result.outcome == "infeasible":
        raise typer.Exit(code=EXIT_INFEASIBLE)
    if result.outcome != "success":
        raise typer.Exit(code=1)


def _pilot_header_rows(
    roster: Any,
    providers: dict[str, Any],
    memories: Any,
    *,
    flock_name: str | None,
    dry_run: bool,
) -> list[tuple[str, Any]]:
    """What is about to happen: which bodies, which pilots, and what is different about it."""
    pilots = {(p.name, p.model) for p in providers.values()}
    rows: list[tuple[str, Any]] = []
    if len(pilots) == 1:
        name, model = pilots.pop()
        rows.append(("provider", f"{name} ({model or 'the first model it serves'})"))
    else:
        rows.append(
            (
                "pilots",
                Text(
                    NEWLINE.join(
                        f"{n:<{max(len(m) for m in roster)}}  {providers[n].name} "
                        f"({providers[n].model or 'the first model it serves'})"
                        for n in roster
                    )
                ),
            )
        )
    width = max(len(n) for n in roster)
    rows.append(
        (
            "flock",
            Text(
                NEWLINE.join(f"{n:<{width}}  {entry.robot_spec.key}" for n, entry in roster.items())
            ),
        )
    )
    if flock_name:
        rows.append(("stored as", Text(flock_name, style=ui.STYLES["accent"])))
    rows.append(("status", Text("EXPERIMENTAL", style=ui.STYLES["warn"])))
    if dry_run:
        rows.append(("mode", Text("DRY RUN: nothing is sent", style=ui.STYLES["warn"])))
    if memories:
        total = sum(len(m.notes()) for m in memories.values())
        rows.append(
            (
                "memory",
                Text(
                    f"{_plural(total, 'note')} across {_plural(len(memories), 'robot')}"
                    "  --no-memory to run fresh",
                    style=ui.STYLES["muted"],
                ),
            )
        )
    return rows


def _run_flock_impl(
    duck: Any,
    *,
    llm: str | None,
    specs: list[Any],
    members: list[str] | None = None,
    flock_name: str | None = None,
    seed: int | None,
    dry_run: bool,
    runs_dir: str,
    yes: bool,
    live: bool,
    gif: bool,
    gif_size: int,
    verbose: bool,
    goal: str | None,
    base_url: str | None,
    api_key: str | None,
    vision: bool | None,
    extra_body: str | None,
    n_override: int | None,
    max_steps: int | None,
    log_on: bool | None = None,
    log_prompt: bool | None = None,
    run_name: str | None = None,
    price: str | None = None,
) -> None:
    from quackd.agent.providers.base import ProviderError
    from quackd.agent.providers.factory import make_provider, resolve_llm
    from quackd.flock.runner import FLOCK_LOG, run_flock
    from quackd.log import (
        ConsoleLog,
        Sink,
        fan_out,
        flock_caption,
        log_enabled_default,
        prompt_shown_default,
        thinking_limit_default,
    )
    from quackd.safety import KillSwitch
    from quackd.sim2d.recorder import FrameRecorder

    if any(spec.backend != "sim2d" for spec in specs):
        _fail(
            "flock mode is simulator only (docs/flock.md); "
            "every member must be an <adapter>:sim2d robot"
        )
        return
    if duck.frontmatter.verbs.confirm and not yes:
        _fail("a flock cannot prompt y/N per duck: empty verbs.confirm or pass --yes")
        return
    roles = duck.frontmatter.flock.roles if duck.frontmatter.flock is not None else None
    if n_override is not None and roles:
        _fail("--flock N cannot be combined with flock.roles; the task file names its members")
        return
    if members is not None:
        # a stored flock supplies the members the way `--flock N` supplies the count. The
        # size and the bodies were checked by the caller, so this skips the field validator
        # rather than re-deriving a task file that never named them.
        from quackd.duckfile.schema import FlockSection

        section = (duck.frontmatter.flock or FlockSection()).model_copy(update={"members": members})
        duck = duck.model_copy(
            update={"frontmatter": duck.frontmatter.model_copy(update={"flock": section})}
        )
    robots = {spec.name: spec.key for spec in specs if spec.name} or None
    try:
        vendor, model_id, llm_source = resolve_llm(llm)
        pilot = make_provider(
            vendor,
            model=model_id,
            source=llm_source,
            duck_name=duck.name,
            goal=goal,
            base_url=base_url,
            api_key=api_key,
            vision=vision,
            extra_body=extra_body,
        )
    except (ProviderError, ImportError) as e:
        _fail(str(e))
        return

    if n_override is not None:
        count = n_override
    elif duck.frontmatter.flock is not None:
        count = len(duck.frontmatter.flock.member_names)
    else:
        count = 3
    member_names = (
        duck.frontmatter.flock.member_names[:count]
        if duck.frontmatter.flock is not None
        else [f"duck-{i}" for i in range(count)]
    )
    prefix_width = max(len(name) for name in [*member_names, FLOCK_LOG])
    log_on = log_on if log_on is not None else log_enabled_default()

    def log(msg: str) -> None:
        # the log says all of this and more, so two views of one line is noise
        if verbose and not log_on:
            _verbose_line(msg)

    views: dict[str, ConsoleLog] = {}

    def view_for(name: str) -> Sink | None:
        """One view per robot, its name on every line. A shared view would coalesce two
        robots' intents into one line and attribute them to whichever spoke last."""
        if name not in views:
            # a colour per member as well as a name, because three robots moving at once
            # interleave and the eye finds a colour faster than it reads a prefix
            order = member_names.index(name) if name in member_names else -1
            views[name] = ConsoleLog(
                ui.err_console,
                thinking_chars=thinking_limit_default(),
                prompt=log_prompt if log_prompt is not None else prompt_shown_default(),
                prefix=f"{name:<{prefix_width}}  ",
                prefix_style=ui.MEMBER_STYLES[order % len(ui.MEMBER_STYLES)]
                if order >= 0
                else ui.STYLES["key"],
            )
        return fan_out(views[name], status.sink)

    def status_only(_name: str) -> Sink | None:
        """With --no-log nothing narrates, but the status line still has to say which duck
        is doing what, or a flock is a minute of nothing at all."""
        return status.sink

    status = ui.RunStatus()
    ui.install_logging()
    holder: dict[str, Any] = {}

    def on_ready(transport0: Any, coordinator: Any) -> None:
        ks = KillSwitch(coordinator.abort, log=log)
        ks.install()
        holder["ks"] = ks
        if not gif:
            return
        rec = FrameRecorder(transport0, size=gif_size)
        holder["rec"] = rec
        names = sorted(coordinator.members)

        def on_event(kind: str, data: dict[str, Any]) -> None:
            if kind == "claim":
                entity = data.get("entity")
                if entity:
                    rec.set_focus(entity[1])
                else:
                    rec.set_focus(names.index(data["kicker"]))
            # the same function the terminal renders with, so a frame in the GIF and a
            # line on screen say the same thing about the same moment
            if kind in ("auction", "claim", "miss", "kick_done", "verdict"):
                caption = flock_caption(kind, data)
                if caption is not None:
                    rec.set_caption(f"{caption[0]} {caption[1]}")

        coordinator.on_event = on_event

    rows: list[tuple[str, Any]] = [
        ("provider", f"{pilot.name} ({pilot.model or 'the first model it serves'})"),
        (
            "flock",
            (f"{flock_name}: " if flock_name else "")
            + f"{count} ducks in sim2d"
            + (f"  seed {seed}" if seed is not None else ""),
        ),
        ("status", Text("EXPERIMENTAL", style=ui.STYLES["warn"])),
    ]
    if dry_run:
        rows.append(("mode", Text("DRY RUN: nothing is sent", style=ui.STYLES["warn"])))
    ui.console.print(ui.run_header(duck.name, rows, hint="Ctrl-C or q stops every duck."))
    try:
        with status:
            status.update(f"starting {count} ducks")
            result = asyncio.run(
                run_flock(
                    duck,
                    provider=pilot,
                    seed=seed if seed is not None else 0,
                    runs_dir=runs_dir,
                    n_override=n_override,
                    dry_run=dry_run,
                    max_steps=max_steps,
                    live=live,
                    gif_size=gif_size,
                    on_recorder=on_ready,
                    log=log,
                    robots=robots,
                    view=view_for if log_on else status_only,
                    run_name=run_name,
                    price=price,
                    on_run_dir=ui.attach_capture,
                )
            )
    except ValueError as e:
        _fail(str(e))
        return
    finally:
        if "ks" in holder:
            holder["ks"].uninstall()
        # a member's last event is the stop it was accepted for, and a pending burst is
        # only written by the next event that is not an intent: without this, never
        for pending in views.values():
            pending.flush()
    if "rec" in holder:
        rec = holder["rec"]
        with ui.spinner(f"encoding {len(rec.frames)} frames into run.gif"):
            result.gif_path = rec.save_gif(result.run_dir / "run.gif")
    counters = [f"kicker {result.kicker}"]
    if result.spotter:
        counters.insert(0, f"spotter {result.spotter}")
    counters += [
        f"auctions {result.auctions}",
        f"bids {result.bids}",
        f"ball moved {result.ball_displacement_m:.2f} m in {result.sim_elapsed_s:.1f} s sim",
    ]
    _print_outcome(
        result.outcome,
        result.reason,
        counters=counters,
        run_dir=result.run_dir,
        gif_path=result.gif_path,
        log_dropped=result.log_dropped,
    )
    if result.outcome == "infeasible":
        # its own code: 1 means the run happened and did not succeed, and a script trying one
        # body after another branches on "this body could not, try the next"
        raise typer.Exit(code=EXIT_INFEASIBLE)
    if result.outcome != "success":
        raise typer.Exit(code=1)


_DUCK_ARG = typer.Argument(
    None, help="Path to a .duck file, or a bundled name (hello-world, find-and-kick, ...)."
)
_GOAL = typer.Option(
    None,
    "--goal",
    "-g",
    help='A plain-language goal instead of a .duck file, e.g. --goal "find the ball and kick it".',
    rich_help_panel="Task",
)
_IMAGE: list[str] = typer.Option(
    [],
    "--image",
    help='A picture to hand to the task, e.g. --goal "draw what is in the picture" --image '
    "sketch.png. The pilot gets it on its first turn, labelled with the file's name, and keeps "
    "it for the whole run, which is what makes it different from a camera frame. Repeatable. "
    "Needs a pilot that takes images: `quackd list-models` marks the ones that do not, and a "
    "local model needs --vision.",
    rich_help_panel="Task",
)
_GIFSIZE = typer.Option(
    256,
    "--gif-size",
    min=64,
    max=1024,  # `sim3d.scene.OFFSCREEN_PX`; spelled here because cli.py must not import sim3d
    help="Simulators: pixel size of each GIF pane, 64 to 1024.",
    rich_help_panel="Output",
)
_FLOCK = typer.Option(
    None,
    "--flock",
    help="EXPERIMENTAL: a number N runs N cooperating ducks (2-4) in sim2d under the "
    "deterministic coordinator; a name from `quackd flock list` runs that flock's registered "
    "robots, one LLM pilot each. Either overrides the file's flock members.",
    rich_help_panel="Robot",
)
_MEMORY = typer.Option(
    True,
    "--memory/--no-memory",
    help="Carry notes and run outcomes between runs of the same robot (see `quackd memory`).",
    rich_help_panel="Memory",
)
_MEMORY_DIR = typer.Option(
    None,
    "--memory-dir",
    help="Where memory files live (default: $QUACKD_MEMORY_DIR or ~/.quackd/memory).",
    rich_help_panel="Memory",
)


_LLM = typer.Option(
    None,
    "--llm",
    "-l",
    help="Who pilots the robot, as VENDOR[:MODEL]. `anthropic` runs that vendor's default and "
    "`openai:gpt-6-sol` names one; a model id unique to its vendor is enough on its own, so "
    "`claude-opus-5-5` works. `ollama:qwen3:8b` is a local server (the split is at the first "
    "colon, so a tag keeps its own), `local` needs --base-url, and `fake` is a scripted pilot "
    "with no key and no network. Vendors: " + " · ".join(PROVIDER_NAMES) + ". `quackd "
    f"list-models` prints every id. Default: the robot's own, then {LLM_ENV}, then "
    f"{DEFAULT_LLM}.",
    autocompletion=_complete_llm,
    rich_help_panel="Model",
)
_BASEURL = typer.Option(
    None,
    "--base-url",
    help="OpenAI-compatible server, e.g. http://localhost:8000/v1 (local presets).",
    rich_help_panel="Model",
)
# Declared once, like every option two commands share, so `run` and `serve-mcp` cannot come to
# spell the board differently. No `envvar=`: Typer would fold QUACKD_HOST_TOKEN
# into the flag, above a token stored with the robot, and quackd's order everywhere is the
# flag, then the robot, then the environment. `quackd.host.resolve_host` reads both variables.
_HOST = typer.Option(
    None,
    "--host",
    metavar="HOST[:PORT]",
    help="A machine quackd uses and never runs on: its model server, its camera, its detector "
    "and its health. --robot still names the body. The port is the one quackd's daemon on the "
    "board listens on, 9874 unless you changed it, and a local preset keeps its own port on "
    "that machine. Without the flag, a run uses the robot's registered host, then QUACKD_HOST.",
    rich_help_panel="Host",
)
_HOST_TOKEN = typer.Option(
    None,
    "--host-token",
    help="The token the --host daemon was started with. It travels in a header, never in a "
    "URL, and never reaches the run record. Without the flag, a run uses the robot's "
    "registered one, then QUACKD_HOST_TOKEN.",
    rich_help_panel="Host",
)
_DETECTOR = typer.Option(
    None,
    "--detector",
    metavar="color|host|yolo",
    help="What reads the camera's frames. color is the colour detector on this machine; host "
    "is YOLO on the board --host names; yolo is YOLO on this machine and needs quackd[yolo]. "
    "Default: the host's detector on a real body when --host names a daemon that can detect, "
    "else the colour detector on this machine. The run never changes detector once it starts.",
    rich_help_panel="Host",
)
# The same two flags for `quackd robot add` and `edit`, with help of their own for the reason
# `--llm` has its own there: those commands have no --robot, they are what does the
# registering, and the token they take goes into robots.json rather than into a run.
_ROBOT_HOST = typer.Option(
    None,
    "--host",
    metavar="HOST[:PORT]",
    help="The board this robot uses and quackd never runs on: its runs reach their model "
    "server, camera and detector there. The port is quackd's daemon's on the board, 9874 "
    "unless you changed it. --host on a run beats this, and this beats QUACKD_HOST.",
    rich_help_panel="Host",
)
_ROBOT_HOST_TOKEN = typer.Option(
    None,
    "--host-token",
    help="The token that board's daemon was started with. Kept in robots.json, as --token "
    "is. A run of this robot sends it in a header to that board, or to the --host the run "
    "names instead, and --host-token on the run beats it.",
    rich_help_panel="Host",
)
_APIKEY = typer.Option(
    None,
    "--api-key",
    help="API key override (local servers do not need one).",
    rich_help_panel="Model",
)
_EXTRA_BODY = typer.Option(
    None,
    "--extra-body",
    help="A JSON object merged into every request body on the OpenAI-compatible providers, for "
    "a field the server wants and quackd never sends. Qwen3 on vLLM stops thinking with "
    '\'{"chat_template_kwargs": {"enable_thinking": false}}\'. QUACKD_EXTRA_BODY does the '
    "same when the flag is absent, and spares you the shell quoting.",
    rich_help_panel="Model",
)
_VISION = typer.Option(
    None,
    "--vision/--no-vision",
    help="Send camera frames to the model (default: on for cloud, off for local).",
    rich_help_panel="Model",
)


def _complete_decision_llm(ctx: typer.Context, incomplete: str) -> list[tuple[str, str]]:
    """`--decision-llm` in the shell. The same shape as `--llm`: a name, then its own model.

    The names come from the table plus whatever is installed here, so a plugin a reader
    installed this morning completes without quackd having been rebuilt for it."""
    from quackd.agent.decision.catalogue import PRESETS
    from quackd.agent.decision.factory import preset_names

    head, colon, _prefix = incomplete.partition(":")
    name = head.strip().lower()
    if colon:
        spec = PRESETS.get(name)
        return [(f"{name}:{spec.model}", spec.summary)] if spec and spec.model else []
    return [
        (n, PRESETS[n].summary if n in PRESETS else "installed here")
        for n in preset_names()
        if n.startswith(name)
    ]


_DECISION_LLM = typer.Option(
    None,
    "--decision-llm",
    help="EXPERIMENTAL: put a discrete stepper in front of the model, as NAME[:MODEL]. A "
    "decision LLM generates no text at all: it answers the turns whose answer is a choice "
    "among calls this body can make, a read, the brake, a gripper, a gaze, and hands "
    "everything else to the model, including every pose and every sentence. `jev` is "
    "TypeSafe's, hosted; `kev`, `von`, `openjev` and `opendecision` are open ones you run "
    "yourself; `laya` runs inside this process; `local` is any other server, with "
    "--decision-url. Needs "
    r"quackd\[decision] (or quackd\[laya])"
    " and whatever key the one you name asks for, which for every one you run yourself is "
    "none. Off unless you name one. QUACKD_DECISION_LLM does the same.",
    autocompletion=_complete_decision_llm,
    rich_help_panel="Model",
)
_DECISION_URL = typer.Option(
    None,
    "--decision-url",
    help="Where your own System One server listens, e.g. http://localhost:8009 — no path, "
    "because the client adds /v1/systemone itself. Required by --decision-llm local, and the "
    "way to move any other one off the port its row expects. QUACKD_DECISION_URL does the "
    "same.",
    rich_help_panel="Model",
)
_DECISION_MODE = typer.Option(
    None,
    "--decision-mode",
    help="What the decision LLM's answer is allowed to do. `on`, the default once you have "
    "named one, lets it take the turns it is confident enough about. `shadow` asks it every "
    "turn, records what it would have chosen beside what the model did, and changes nothing "
    "about the run: it is how you find out whether to trust one before you do. `off` is quackd "
    "as it has always been. QUACKD_DECISION_MODE does the same.",
    rich_help_panel="Model",
)
_ROBOT = typer.Option(
    None,
    "--robot",
    "-r",
    help="<adapter>:<backend>, e.g. microduck:sim2d · lerobot:real · microduck:mock, or a "
    "name from `quackd robot add`, which brings its own address, token and camera. The core "
    "installs no robot, so the default is whatever is here: the only adapter installed, or "
    "microduck:sim2d where the duck is one of several. With several and no duck, name a "
    "body. See `quackd list-adapters` and `quackd robot list`.",
    rich_help_panel="Robot",
)
_ROBOTS = typer.Option(
    None,
    "--robots",
    help="A flock: name=<adapter>:<backend>,... A coordinator flock needs every "
    "member to be microduck:sim2d, and a pilot flock or serve-mcp takes any of them.",
    rich_help_panel="Robot",
)
_SEED = typer.Option(
    None, "--seed", help="Simulator seed (deterministic runs).", rich_help_panel="Task"
)
_DRY = typer.Option(
    False, "--dry-run", help="Print every intent, send nothing.", rich_help_panel="Task"
)
_MAXSTEPS = typer.Option(
    None, "--max-steps", help="Override the duck's max_steps budget.", rich_help_panel="Task"
)
_RUNS = typer.Option(
    "runs", "--runs-dir", help="Where run directories go.", rich_help_panel="Output"
)
_RUN_NAME = typer.Option(
    None,
    "--run-name",
    help="Name this run on disk: runs/<stamp>-<duck>-<name>/. Lowercased to a slug, so "
    '"Example 1" becomes example-1. Omitted, the directory is named as it always was.',
    rich_help_panel="Output",
)
_PRICE = typer.Option(
    None,
    "--price",
    help="What the model costs, in USD per million tokens: in=3,out=15[,cache_read=0.3,"
    "cache_write=3.75]. Beats QUACKD_PRICE and the built-in rates. Use it for a negotiated "
    "rate, a model quackd has no price for, or a paid server behind a local preset.",
    rich_help_panel="Model",
)
_YES = typer.Option(
    False,
    "--yes",
    "-y",
    help="Auto-confirm gated verbs (careful on hardware).",
    rich_help_panel="Task",
)
_LIVE = typer.Option(
    False,
    "--live",
    help="Simulators: watch the run in real time. sim2d opens a pygame window (needs "
    r"quackd\[live]); mujoco opens MuJoCo's own viewer.",
    rich_help_panel="Output",
)
_ADDR = typer.Option(
    None,
    "--address",
    help="Where the body is, in its own protocol's shape: a LeRobot arm's serial port "
    "(COM5 on Windows, /dev/ttyACM0 elsewhere), a rosbridge websocket "
    "(ws://host:9090), a ZeroMQ or bridge host (tcp://host:5555), or robotd's socket "
    "(unix:///run/robotd.sock, tcp://host:port).",
    rich_help_panel="Robot",
)
_TOKEN = typer.Option(
    None,
    "--token",
    help="The bridge token for a robot that wants one. The Open Duck's installer writes one "
    "on the robot and QUACKD_DUCK_TOKEN carries it when the flag is absent. The ToddlerBot's "
    "daemon has no installer and reads QUACKD_TODDLERBOT_TOKEN instead.",
    rich_help_panel="Robot",
)
_CAMERA_URL: list[str] = typer.Option(
    [],
    "--camera-url",
    help="Where frames come from, overriding whatever the robot advertises. An HTTP snapshot "
    "(http://host:9872/snapshot.jpg), or webrtc://host:8443 to pull mediad's video track off a "
    r"Microduck, which is the only camera upstream offers and needs quackd\[microduck-camera]. "
    "Needed when you reach the robot through a tunnel and its own URL is not routable. On a "
    "LeRobot arm it is a USB webcam by its OpenCV index, opencv://0, with ?width, ?height, "
    "?fps, ?fourcc, ?rotation, ?fov, ?name and ?backend=msmf for a Windows camera that lists "
    "and will not open. Find the index with lerobot-find-cameras opencv. Repeat the flag for "
    "several cameras: every frame reaches the model each step, and the first is the primary, "
    "the one --fov-deg describes and the one the detections and the steering verbs read. "
    "Only the LeRobot arm reads more than one.",
    rich_help_panel="Robot",
)
_BY_HAND = typer.Option(
    False,
    "--by-hand",
    help="Start from a pose you set yourself instead of the recorded rest pose. The arm goes "
    "to its rest pose, quackd takes torque off there, you lift it, load the gripper and press "
    "Enter, and it holds what you left while the model works. At the end it asks before the "
    "gripper opens. Needs a LeRobot arm with a rest pose recorded, and a terminal to ask on.",
    rich_help_panel="Robot",
)
_FOV = typer.Option(
    None,
    "--fov-deg",
    help="Horizontal field of view of the camera actually on your robot, in degrees. The "
    "default is the simulator's 90; a Pi Camera Module 2 is about 62. Getting it wrong "
    "scales every bearing and distance, so detections say so until you set it.",
    rich_help_panel="Robot",
)
_VERBOSE = typer.Option(
    False,
    "--verbose",
    "-v",
    help="The compact view on stderr: one line per verb plus the executor's notes. The log "
    "(on by default) shows all of that and more, so this only adds anything with --no-log.",
    rich_help_panel="Output",
)
_LOG = typer.Option(
    None,
    "--log/--no-log",
    help="Narrate the run on stderr as it happens: the prompt, each observation, what the "
    "model thought and answered, every executor decision, every intent sent to the robot, "
    "every result, tokens and timings. On by default; QUACKD_LOG=0 turns it off too. This "
    "is about what you WATCH: the run directory gets its log either way.",
    rich_help_panel="Output",
)
_LOG_MCP = typer.Option(
    None,
    "--log/--no-log",
    help="Carry a log of what happened on every tool result, and the uncapped version on "
    "stderr: the verb, every gate that fired, every intent sent to the robot, every result "
    "and the budget. Over MCP the pilot is the client, so its own reasoning is not quackd's "
    "to show. On by default; QUACKD_LOG=0 turns it off too.",
    rich_help_panel="Output",
)
_LOG_PROMPT = typer.Option(
    None,
    "--log-prompt/--no-log-prompt",
    help="Print the system prompt once at the start of the log. On by default; "
    "QUACKD_LOG_PROMPT=0 turns it off too. It is in the transcript either way.",
    rich_help_panel="Output",
)


@app.command(rich_help_panel="Run a duck")
def run(
    duckfile: str | None = _DUCK_ARG,
    goal: str | None = _GOAL,
    image: list[str] = _IMAGE,
    by_hand: bool = _BY_HAND,
    llm: str | None = _LLM,
    robot: str | None = _ROBOT,
    robots: str | None = _ROBOTS,
    seed: int | None = _SEED,
    dry_run: bool = _DRY,
    max_steps: int | None = _MAXSTEPS,
    runs_dir: str = _RUNS,
    yes: bool = _YES,
    live: bool = _LIVE,
    address: str | None = _ADDR,
    camera_url: list[str] = _CAMERA_URL,
    token: str | None = _TOKEN,
    fov_deg: float | None = _FOV,
    host: str | None = _HOST,
    host_token: str | None = _HOST_TOKEN,
    detector: str | None = _DETECTOR,
    gif: bool = typer.Option(
        True,
        "--gif/--no-gif",
        help="Simulators: write run.gif into the run dir.",
        rich_help_panel="Output",
    ),
    gif_size: int = _GIFSIZE,
    verbose: bool = _VERBOSE,
    base_url: str | None = _BASEURL,
    api_key: str | None = _APIKEY,
    vision: bool | None = _VISION,
    extra_body: str | None = _EXTRA_BODY,
    decision_llm: str | None = _DECISION_LLM,
    decision_url: str | None = _DECISION_URL,
    decision_mode: str | None = _DECISION_MODE,
    flock: str | None = _FLOCK,
    run_name: str | None = _RUN_NAME,
    price: str | None = _PRICE,
    memory: bool = _MEMORY,
    memory_dir: str | None = _MEMORY_DIR,
    registry_dir: str | None = _REGISTRY_DIR,
    log: bool | None = _LOG,
    log_prompt: bool | None = _LOG_PROMPT,
) -> None:
    """Run a .duck file (or a --goal): the LLM picks verbs, quackd enforces the contract."""
    with _terminal_record():
        _run_impl(
            duckfile,
            goal,
            llm,
            seed,
            dry_run,
            max_steps,
            runs_dir,
            yes,
            live,
            address,
            camera_url,
            token,
            fov_deg,
            gif,
            gif_size,
            verbose,
            base_url=base_url,
            api_key=api_key,
            vision=vision,
            extra_body=extra_body,
            decision_llm=decision_llm,
            decision_url=decision_url,
            decision_mode=decision_mode,
            flock=flock,
            robot=robot,
            robots=robots,
            memory=memory,
            memory_dir=memory_dir,
            registry_dir=registry_dir,
            log_on=log,
            log_prompt=log_prompt,
            images=image,
            by_hand=by_hand,
            run_name=run_name,
            price=price,
            host=host,
            host_token=host_token,
            detector_choice=detector,
        )


@app.command(rich_help_panel="Run a duck")
def record(
    duckfile: str | None = _DUCK_ARG,
    goal: str | None = _GOAL,
    llm: str | None = _LLM,
    seed: int | None = typer.Option(0, "--seed"),
    max_steps: int | None = _MAXSTEPS,
    runs_dir: str = _RUNS,
    run_name: str | None = _RUN_NAME,
    gif_size: int = _GIFSIZE,
    verbose: bool = _VERBOSE,
    base_url: str | None = _BASEURL,
    api_key: str | None = _APIKEY,
    vision: bool | None = _VISION,
    extra_body: str | None = _EXTRA_BODY,
    flock: str | None = typer.Option(
        None,
        "--flock",
        help="EXPERIMENTAL: N cooperating ducks (2-4) in sim2d. A count only: this command "
        "pins the simulator, so a stored flock belongs to `quackd run`.",
        rich_help_panel="Robot",
    ),
    log: bool | None = _LOG,
    log_prompt: bool | None = _LOG_PROMPT,
) -> None:
    """Like `run` on sim2d, but always writes a GIF (for READMEs and launches)."""
    with _terminal_record():
        if flock is not None and not flock.strip().isdigit():
            _fail(
                "record pins the simulator: --flock takes a count here, not a stored flock",
                hint="quackd run <duck> --flock NAME",
            )
            return
        _run_impl(
            duckfile,
            goal,
            llm,
            seed=seed,
            dry_run=False,
            max_steps=max_steps,
            runs_dir=runs_dir,
            yes=True,
            live=False,
            address=None,
            camera_url=[],
            token=None,
            fov_deg=None,
            gif=True,
            gif_size=gif_size,
            verbose=verbose,
            base_url=base_url,
            api_key=api_key,
            vision=vision,
            extra_body=extra_body,
            flock=flock,
            robot="microduck:sim2d",
            # Pinned off, not merely absent. `record` makes the recordings in this repository and
            # has to be reproducible without a network call, and leaving this to default meant
            # `QUACKD_DECISION_LLM` in somebody's environment quietly switched one on.
            decision_mode="off",
            log_on=log,
            log_prompt=log_prompt,
            run_name=run_name,
        )


# ── log: replay a finished run ──────────────────────────────────────────────────────────


_LABELLED = re.compile(r"^\d{8}-\d{6}-")
"""The stamp `new_run_dir` writes, so the rest of a directory name can be read on its own."""


def _ends_with_label(rest: str, label: str) -> bool:
    """Is `label` the name somebody gave this run?

    `-example-1` matches `find-and-kick-example-1` and also `find-and-kick-example-1-1`, which
    is the same run's collision counter and not a different name, but never
    `find-and-kick-example-19`, which is a different run entirely and the reason this pass
    exists at all."""
    return re.search(rf"-{re.escape(label)}(-\d+)?$", rest) is not None


def _resolve_run(run: str | None, runs_dir: str) -> Path:
    """A run directory from what the user typed. In order: a transcript file, a directory, an
    exact name under --runs-dir, the newest carrying that `--run-name`, a timestamp prefix,
    the newest whose name contains the text. Nothing at all means the newest run, which is
    what you want after `quackd run` ends."""
    from quackd.agent.transcript import run_label

    root = Path(runs_dir)
    runs = sorted((d for d in root.glob("*") if d.is_dir()), key=lambda d: d.name)
    if run:
        typed = Path(run)
        if typed.is_file():
            return typed
        if typed.is_dir():
            return typed
        exact = root / run
        if exact.is_dir():
            return exact
        # The `--run-name` pass, ABOVE the timestamp prefix rather than below it. A bench
        # session of a hundred runs has `-example-1` and `-example-19` in it, and a substring
        # match hands you whichever is newest, so an exact label is what somebody typing the
        # name they gave a run meant. It goes first because a label can be all digits: a run
        # named `20260921` was unreachable by its own name while the prefix pass, which every
        # directory's timestamp satisfies, got to answer first.
        with contextlib.suppress(ValueError):
            label = run_label(run)
            for d in reversed(runs):
                # Against what follows the timestamp, and only where something precedes the
                # label there. Otherwise `quackd log hello-world` would match the bare
                # `<stamp>-hello-world` and quietly prefer an older unnamed run of that duck
                # over a newer named one, which is not what typing a duck name asks for.
                rest = _LABELLED.sub("", d.name, count=1)
                if rest != d.name and rest != label and _ends_with_label(rest, label):
                    return d
        for d in reversed(runs):
            if d.name.startswith(run):
                return d
        for d in reversed(runs):
            if run in d.name:
                return d
        newest = ", ".join(d.name for d in runs[-5:]) or "none yet"
        _fail(f"no run matching {run!r} under {root} (newest: {newest})")
    if not runs:
        _fail(f"no runs under {root}: pass a path, or run `quackd run` first")
    return runs[-1]


def _replay(
    records: list[dict[str, Any]],
    view: Any,
    *,
    from_step: int | None,
    frames: bool,
) -> dict[str, Any] | None:
    """Records back through the same renderer that printed them live, and the `run_end`.

    A transcript written before the log existed has `verb` records and no `verb_end`; they
    carry the same fields, so they are shown under the name the renderer knows. `frame` is
    skipped unless asked: one line per camera frame buries everything else."""
    from quackd.log import LogEvent

    end: dict[str, Any] | None = None
    # `verb` and `verb_end` both name the verb that ended; a live run has both and the
    # renderer draws only the second, so promote `verb` only when there is no `verb_end`
    legacy = not any(r.get("kind") == "verb_end" for r in records)
    skipping = from_step is not None
    for rec in records:
        kind = str(rec.get("kind", ""))
        data = {k: v for k, v in rec.items() if k not in ("t", "kind")}
        if kind == "run_end":
            end = data
        if skipping:
            if kind == "observation" and data.get("step") == from_step:
                skipping = False
            elif kind != "run_start":
                continue
        if kind == "frame":
            if frames:
                ui.console.print(f"        frame {data.get('path', '')}", style="dim", markup=False)
            continue
        if kind == "verb" and legacy:
            kind = "verb_end"
        view(LogEvent(kind, float(rec.get("t") or 0.0), data))
    view.flush()
    return end


_LOG_RUN = typer.Argument(
    None, help="A run directory, a transcript file, a name or a prefix. Default: the newest."
)


@app.command("log", rich_help_panel="Run a duck")
def log_cmd(
    run: str | None = _LOG_RUN,
    runs_dir: str = _RUNS,
    prompt: bool | None = typer.Option(
        None, "--prompt/--no-prompt", help="Show the system prompt the run was given."
    ),
    thinking: str | None = typer.Option(
        None, "--thinking", help="Characters of thinking per turn: a number, or `all`."
    ),
    from_step: int | None = typer.Option(
        None, "--from-step", help="Start at this step, skipping the turns before it."
    ),
    frames: bool = typer.Option(False, "--frames", help="Also print one line per camera frame."),
) -> None:
    """Replay a finished run's log as the lines it printed while it ran.

    On stdout, because a replay is what you pipe to a pager or a file, and unaffected by
    QUACKD_LOG: that switch is about narrating live, and asking for a replay is asking."""
    from quackd.agent.transcript import Transcript
    from quackd.log import ConsoleLog, parse_thinking_limit
    from quackd.log import prompt_shown_default as _prompt_default
    from quackd.log import thinking_limit_default as _thinking_default

    target = _resolve_run(run, runs_dir)
    run_dir = target.parent if target.is_file() else target
    transcripts = (
        [target]
        if target.is_file()
        else sorted((run_dir / "ducks").glob("*/transcript.jsonl"))
        or [run_dir / "transcript.jsonl"]
    )
    if not transcripts[0].exists():
        _fail(f"{transcripts[0]} does not exist: that is not a run directory")

    width = max((len(p.parent.name) for p in transcripts), default=0) if len(transcripts) > 1 else 0
    end: dict[str, Any] | None = None
    cut = 0
    for i, path in enumerate(transcripts):
        records = Transcript.read(path, lenient=True)
        cut += int(records[-1].get("_skipped", 0)) if records else 0
        view = ConsoleLog(
            ui.console,
            thinking_chars=(
                parse_thinking_limit(thinking) if thinking is not None else _thinking_default()
            ),
            prompt=prompt if prompt is not None else _prompt_default(),
            progress_s=None,  # a replay is not live: one line per burst, as the record has it
            prefix=f"{path.parent.name:<{width}}  " if width else "",
            prefix_style=ui.MEMBER_STYLES[i % len(ui.MEMBER_STYLES)] if width else "",
            # a replay has no CLI header in front of it, so this is where the run says what
            # it was: which duck, which model, which robot, and how long connecting took
            header=True,
        )
        end = _replay(records, view, from_step=from_step, frames=frames) or end

    if cut:
        ui.err_console.print(
            f"log: {cut} unreadable line(s) skipped, the run was cut while it was writing",
            style="yellow",
            markup=False,
        )
    summary = run_dir / "summary.json"
    if summary.exists() and (end is None or len(transcripts) > 1):
        # A flock replays every member, so `end` is whichever member happened to finish last
        # and its wall clock and its bill are that ONE robot's. The flock's own summary is
        # sitting in the same directory and is the thing the counters are about.
        end = json.loads(summary.read_text(encoding="utf-8"))
    if end is None:
        _fail("no run_end: the run did not finish, or is still running")
        return
    if "kicker" in end:  # a flock counts different things, and its summary is the only source
        counters = [f"spotter {end['spotter']}"] if end.get("spotter") else []
        counters += [
            f"kicker {end.get('kicker')}",
            f"auctions {end.get('auctions')}",
            f"bids {end.get('bids')}",
        ]
    else:
        counters = run_counters(end)
    _print_outcome(
        str(end.get("outcome", "error")),
        str(end.get("reason", "")),
        counters=counters,
        run_dir=run_dir,
        gif_path=gif if (gif := run_dir / "run.gif").exists() else None,
        # `trace_dropped` is still read, so a run directory recorded before the rename
        # replays and its counter still reaches the panel. The value goes through `_number`
        # because this is the path an old run directory takes, and those are hand-edited,
        # truncated and copied between machines, so the field cannot be trusted to hold a
        # number at all.
        log_dropped=int(_number(end.get("log_dropped") or end.get("trace_dropped") or 0) or 0),
    )


# ── doctor / serve-mcp ──────────────────────────────────────────────────────────────────


@app.command(rich_help_panel="Inspect")
def doctor(
    robot: str | None = typer.Option(
        None,
        "--robot",
        "-r",
        help="Also show one robot's manifest (<adapter>:<backend>, or a registered name).",
    ),
    address: str | None = typer.Option(
        None,
        "--address",
        help="With --robot, connect to a real robot and report what it says about itself.",
    ),
    camera_url: list[str] = _CAMERA_URL,
    token: str | None = _TOKEN,
    host: str | None = _HOST,
    host_token: str | None = _HOST_TOKEN,
    registry_dir: str | None = _REGISTRY_DIR,
    as_json: bool = _JSON,
) -> None:
    """Check the environment: keys, optional extras, adapters, upstream assumptions.

    With `--robot X --address Y` it also connects, which is the only way to see what a
    robot actually reports before a run does. With `--host` it asks the daemon on that board
    what it is, reads the board's health over the network, and probes the local model
    presets there."""
    from quackd.doctor import HostReport, collect, refused_host, render
    from quackd.host import HOST_ENV, HostChoice, resolve_host
    from quackd.registry import RobotEntry

    if address and not robot:
        _fail("--address needs --robot, so quackd knows what it is connecting to")
        return
    rest_pose: dict[str, float] | None = None
    entry: RobotEntry | None = None
    name: str | None = None
    if robot:
        # a registered name is a robot too, and it brings the address you registered it with.
        # Only a name that resolves is substituted: anything else stays exactly as typed, so
        # `doctor --robot nope:x --json` still reports the bad spec inside its one JSON
        # document rather than dying with a line of prose before it.
        from quackd.registry import Registry, RegistryError

        with contextlib.suppress(RegistryError):
            entry = Registry(registry_dir).get_robot(robot)
            if entry is not None:
                # The spec for the report, and the name for the body, which is built under it
                # as a run builds it (`entry.robot_spec`): an arm looks its calibration up by
                # that id, and every line it writes names it. The name used to be dropped here,
                # so a probe of `arm-02` read the default id's calibration.
                robot, name = entry.key, entry.name
                where = entry.adapter_kwargs(address=address, camera_url=camera_url, token=token)
                address, camera_url, token = (
                    where["address"],
                    where["camera_url"],
                    where["token"],
                )
                # only a robot you registered has one, because a rest pose is read off the arm
                # and kept under its name rather than typed on a command line
                rest_pose = where["rest_pose"]
    # The board by the ladder `run` climbs, settled by the same function so the two cannot
    # disagree about which board a robot uses: the flag, then the host registered with the
    # robot --robot names, then QUACKD_HOST. A value that is no machine, or a token no header
    # can carry, is reported with the place it came from rather than refused as `run` refuses
    # it: doctor is where a person comes to find a bad setting, one in QUACKD_HOST is there
    # without anybody having typed --host, and --json stays one document, as it does for a bad
    # --robot above. Nothing is asked of it, and it fails the report as a dead daemon does.
    stored = entry.host_kwargs() if entry is not None else {"host": None, "host_token": None}
    unusable: HostReport | None = None
    try:
        board = resolve_host(
            host,
            stored["host"],
            token=host_token,
            stored_token=stored["host_token"],
            robot=entry.name if entry is not None else None,
        )
    except ValueError as e:
        # the ladder stops at the first value that is not blank, so that is the one refused
        named = (host, stored["host"], os.environ.get(HOST_ENV))
        text = next((t.strip() for t in named if t and t.strip()), "")
        board, unusable = HostChoice(), refused_host(text, str(e))

    def warn(adapter: Any) -> None:
        # Before the connect, and only for a body that is handed to people, which is the one
        # whose connect takes torque off: the arm's close note sends a person here when it is
        # holding itself up away from its fold. On stderr under --json, whose stdout is one
        # JSON document and nothing else.
        if getattr(adapter, "supports_hand_off", False):
            (ui.err_console if as_json else ui.console).print(_warn_line(_doctor_warning()))

    if as_json:
        report = collect(
            robot,
            address=address,
            camera_url=camera_url,
            token=token,
            rest_pose=rest_pose,
            robot_name=name,
            before_connect=warn,
            host=board.host,
            host_token=board.token,
        )
        report.host = unusable or report.host
        print(json.dumps(report.to_dict()))
        raise typer.Exit(code=0 if report.ok else 1)
    ui.install_logging()
    # the probes are the slow part: five local servers at 1.5 s each, and a real robot after
    # them. It used to sit silent for ten seconds with no sign it was doing anything.
    with ui.spinner("checking this machine") as say:
        report = collect(
            robot,
            address=address,
            camera_url=camera_url,
            token=token,
            rest_pose=rest_pose,
            host=board.host,
            host_token=board.token,
            progress=say,
            robot_name=name,
            before_connect=warn,
        )
    report.host = unusable or report.host
    render(ui.console, report)
    if not report.ok:
        raise typer.Exit(code=1)


@app.command("serve-mcp", rich_help_panel="Serve")
def serve_mcp(
    robot: str | None = _ROBOT,
    robots: str | None = typer.Option(
        None,
        "--robots",
        help="A flock: name=<adapter>:<backend>,... (nine robot_* tools, one executor each).",
        rich_help_panel="Robot",
    ),
    flock: str | None = typer.Option(
        None,
        "--flock",
        help="A stored flock (`quackd flock list`): every member from the registry, each with "
        "its own address, token and camera. The same flock by another door.",
        rich_help_panel="Robot",
    ),
    registry_dir: str | None = _REGISTRY_DIR,
    duckfile: str | None = typer.Option(
        None, "--duckfile", help="Load a .duck contract at startup (on the default robot)."
    ),
    seed: int | None = _SEED,
    address: str | None = _ADDR,
    camera_url: list[str] = _CAMERA_URL,
    token: str | None = _TOKEN,
    host: str | None = _HOST,
    host_token: str | None = _HOST_TOKEN,
    detector: str | None = _DETECTOR,
    dry_run: bool = _DRY,
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Allow confirm-gated verbs (there is no terminal to ask)."
    ),
    memory: bool = _MEMORY,
    memory_dir: str | None = _MEMORY_DIR,
    log: bool | None = _LOG_MCP,
) -> None:
    """Expose a robot, or a flock of them, as MCP tools over stdio (Claude Code /
    Claude Desktop)."""
    from quackd.adapters.base import AdapterError
    from quackd.mcp_server import serve
    from quackd.registry import RegistryError

    try:
        serve(
            robot=robot,
            robots=robots,
            flock=flock,
            registry_dir=registry_dir,
            duckfile=duckfile,
            seed=seed,
            address=address,
            camera_url=camera_url,
            token=token,
            host=host,
            host_token=host_token,
            detector=detector,
            dry_run=dry_run,
            yes=yes,
            memory=memory,
            memory_dir=memory_dir,
            log=log,
        )
    except (AdapterError, RegistryError) as e:
        _fail(str(e))


# ── robot (the registry) ────────────────────────────────────────────────────────────────

robot_app = typer.Typer(
    name="robot",
    help="The robots you have named: which body, where it is, and who pilots it. Kept in "
    "~/.quackd/robots.json so --robot NAME means the same thing in every command.",
    no_args_is_help=True,
)
app.add_typer(robot_app, name="robot", rich_help_panel="Robots")


def _registry(registry_dir: str | None) -> Any:
    from quackd.registry import Registry

    return Registry(registry_dir)


def _registry_fail(e: Exception) -> None:
    """Every registry refusal is one line. A robot that is not there is not a traceback."""
    from quackd.adapters.base import AdapterError

    _fail(str(e), hint=_ADAPTER_HINT if isinstance(e, AdapterError) else None)


def _can_prompt() -> bool:
    """Whether there is a person at a terminal to ask. The seam tests replace."""
    return bool(sys.stdin is not None and sys.stdin.isatty())


def _rest_pose_text(pose: dict[str, float]) -> Any:
    """A recorded pose on one line per joint, in the arm's own bus order where it has one.

    Alphabetical when the arm's package is not installed, because a pose is worth printing
    whether or not the adapter that recorded it is still here: `quackd robot show` is how you
    read a registry on a machine that cannot drive half of it."""
    joints: tuple[str, ...] = ()
    with contextlib.suppress(ImportError):
        from quackd_lerobot import JOINTS

        joints = JOINTS
    order = {joint: i for i, joint in enumerate(joints)}
    listed = sorted(pose.items(), key=lambda kv: (order.get(kv[0], len(order)), kv[0]))
    return Text("\n".join(f"{joint} {value:.1f}" for joint, value in listed))


def _entry_rows(entry: Any, *, flocks: list[str]) -> list[tuple[str, Any]]:
    from quackd.adapters.factory import describe

    body: Any
    try:
        manifest = describe(entry.robot_spec)
        body = Text(manifest.summary())
    except Exception as e:  # an adapter whose extra is missing still has a name
        body = Text(str(e), style=ui.STYLES["muted"])
    dash = Text("-", style=ui.STYLES["muted"])
    pilot = Text(entry.llm) if entry.llm else dash
    return [
        ("name", Text(entry.name, style=ui.STYLES["key"])),
        ("robot", Text(entry.key, style=ui.STYLES["accent"])),
        ("body", body),
        ("address", Text(entry.address) if entry.address else dash),
        # one per line, because two urls on one line is where a reader stops being able to
        # tell which camera is the primary
        ("camera", Text("\n".join(entry.camera_urls)) if entry.camera_urls else dash),
        ("rest pose", _rest_pose_text(entry.rest_pose) if entry.rest_pose else dash),
        ("token", Text("set") if entry.token else dash),
        ("host", Text(entry.host) if entry.host else dash),
        ("host token", Text("set") if entry.host_token else dash),
        ("pilot", pilot),
        ("note", Text(entry.note) if entry.note else dash),
        ("flocks", Text(", ".join(flocks)) if flocks else dash),
        ("added", Text(entry.added, style=ui.STYLES["muted"])),
        ("updated", Text(entry.updated, style=ui.STYLES["muted"])),
    ]


_ROBOT_NAME = typer.Argument(
    ...,
    help="A slug: lowercase letters, digits and hyphens. Not a number (--flock N already "
    "means N simulated ducks) and not an adapter name.",
)
_PROBE = typer.Option(
    False,
    "--probe",
    help="Connect to each robot and say whether it answered. Costs a connection per robot.",
)
_PROBE_TIMEOUT = typer.Option(
    5.0, "--timeout", min=0.1, max=120.0, help="Seconds to wait per robot when probing."
)


@robot_app.command("add")
def robot_add(
    name: str = _ROBOT_NAME,
    spec: str = typer.Argument(
        ..., help="<adapter>[:<backend>], e.g. microduck:sim2d. See `quackd list-adapters`."
    ),
    address: str | None = _ADDR,
    camera_url: list[str] = _CAMERA_URL,
    token: str | None = _TOKEN,
    host: str | None = _ROBOT_HOST,
    host_token: str | None = _ROBOT_HOST_TOKEN,
    llm: str | None = typer.Option(
        None,
        "--llm",
        "-l",
        help="The pilot a run uses for this robot when --llm is absent, as VENDOR[:MODEL]. "
        "--llm on the run beats this, and this beats QUACKD_LLM.",
        autocompletion=_complete_llm,
        rich_help_panel="Model",
    ),
    note: str | None = typer.Option(None, "--note", help="One line for people: which one is it."),
    registry_dir: str | None = _REGISTRY_DIR,
) -> None:
    """Register a robot under a name, with how to reach it."""
    from quackd.adapters.base import AdapterError
    from quackd.agent.providers.base import ProviderError
    from quackd.agent.providers.factory import parse_llm
    from quackd.host import parse_host
    from quackd.registry import RegistryError, RobotEntry

    try:
        # Checked against the catalogue here and nowhere else. The shelf is deliberately
        # lenient -- an id a later catalogue retires must not make every `quackd robot`
        # command refuse, including the edit that would fix it -- so the door is the one
        # place a typo can still be caught while the person who made it is looking at it.
        if llm is not None:
            parse_llm(llm, source="--llm")
        # The board gets the same gate, and needs it here as well as in `add_robot`: the entry
        # reads a blank host as none, so `--host ""` would otherwise register a robot with no
        # board and report success.
        if host is not None:
            parse_host(host)
        entry = RobotEntry(
            name=name,
            spec=spec,
            address=address,
            camera_url=list(camera_url) or None,
            token=token,
            host=host,
            host_token=host_token,
            llm=llm,
            note=note,
        )
        _registry(registry_dir).add_robot(entry)
    except (RegistryError, AdapterError, ValueError, ProviderError) as e:
        # ValueError is parse_host's, and is also what pydantic's ValidationError is
        _registry_fail(_one_line(e))
        return
    where = f" at {entry.address}" if entry.address else ""
    board = f", host {entry.host}" if entry.host else ""
    ui.console.print(_ok_line(f"added {entry.name}: {entry.key}{where}{board}"))
    ui.console.print(Text(f"  quackd run <duck> --robot {entry.name}", style=ui.STYLES["muted"]))


def _one_line(e: Exception) -> Exception:
    """A pydantic error folded to the one sentence a person needs, keeping its own words."""
    if isinstance(e, ValidationError):
        return ValueError(
            "; ".join(
                str(err["msg"]).removeprefix("Value error, ")
                for err in e.errors()  # type: ignore[attr-defined]
            )
        )
    return e


@robot_app.command("list")
def robot_list(
    probe: bool = _PROBE,
    timeout: float = _PROBE_TIMEOUT,
    registry_dir: str | None = _REGISTRY_DIR,
    as_json: bool = _JSON,
) -> None:
    """Every registered robot. Static by default: --probe connects to each of them."""
    from quackd.registry import RegistryError, probe_all

    try:
        registry = _registry(registry_dir)
        entries = registry.robots()
        holders = {name: registry.flocks_of(name) for name in entries}
    except RegistryError as e:
        _registry_fail(e)
        return
    probes: dict[str, Any] = {}
    if probe and entries:
        with ui.spinner(f"probing {_plural(len(entries), 'robot')} ({timeout:g} s each)"):
            probes = probe_all(entries.values(), timeout_s=timeout)
    if as_json:
        for name, entry in entries.items():
            payload: dict[str, Any] = {**entry.public(), "flocks": holders[name]}
            if probe:
                result = probes.get(name)
                payload["reachable"] = result.reachable if result else None
                payload["probe"] = result.detail if result else ""
            print(json.dumps(payload, ensure_ascii=False))
        _exit_on_unreachable(probes)
        return
    if not entries:
        ui.console.print(
            Text(
                "no robots registered yet: quackd robot add NAME <adapter>:<backend>",
                style=ui.STYLES["muted"],
            )
        )
        return
    # a column nobody has filled is a column that only makes the rest narrower, and on an
    # 80-column terminal a probe's refusal needs every character it can get
    cells: dict[str, list[Any]] = {
        "address": [Text(e.address or "") for e in entries.values()],
        "host": [Text(e.host or "") for e in entries.values()],
        "pilot": [Text(e.llm or "") for e in entries.values()],
        "flocks": [Text(", ".join(holders[n])) for n in entries],
        "note": [Text(e.note or "") for e in entries.values()],
    }
    if probe:
        cells["reachable"] = [_probe_cell(probes.get(n)) for n in entries]
    shown = [
        name for name, column in cells.items() if name == "reachable" or any(str(c) for c in column)
    ]
    table = ui.table("robots (--robot NAME)")
    table.add_column("name", no_wrap=True, style=ui.STYLES["key"])
    table.add_column("robot", no_wrap=True)
    for name in shown:
        table.add_column(name, overflow="fold", ratio=1 if name in ("note", "reachable") else None)
    for i, (name, entry) in enumerate(entries.items()):
        table.add_row(
            Text(name),
            Text(entry.key, style=ui.STYLES["accent"]),
            *(cells[column][i] for column in shown),
        )
    ui.console.print(table)
    _exit_on_unreachable(probes)


PROBE_DETAIL_CHARS = 48
"""A refusal from a socket can be a paragraph (Windows spells one in about 120 characters),
and a column that wide turns the table into a page. `quackd robot show` and `--json` carry
the whole of it; this says which robot to go and look at."""


def _probe_cell(result: Any) -> Any:
    if result is None:
        return Text("-", style=ui.STYLES["muted"])
    detail = " ".join(str(result.detail).split())
    if len(detail) > PROBE_DETAIL_CHARS:
        detail = detail[: PROBE_DETAIL_CHARS - 3].rstrip() + "..."
    if result.reachable is None:
        return Text(detail, style=ui.STYLES["muted"])
    style = ui.STYLES["ok"] if result.reachable else ui.STYLES["fail"]

    def build(g: ui.Glyphs) -> Any:
        mark = g.ok if result.reachable else g.fail
        return Text(f"{mark} {detail}", style=style)

    return ui.Deferred(build)


def _exit_on_unreachable(probes: dict[str, Any]) -> None:
    """A probe that found a robot down is a failing command, so a script can branch on it."""
    down = [name for name, result in probes.items() if result.reachable is False]
    if down:
        raise typer.Exit(code=1)


@robot_app.command("show")
def robot_show(
    name: str = _ROBOT_NAME,
    registry_dir: str | None = _REGISTRY_DIR,
    as_json: bool = _JSON,
) -> None:
    """Everything one registered robot says about itself, and what it remembers."""
    from quackd.memory import RobotMemory
    from quackd.registry import RegistryError

    try:
        registry = _registry(registry_dir)
        entry = registry.robot(name)
        flocks = registry.flocks_of(name)
    except RegistryError as e:
        _fail(str(e), hint="quackd robot list")
        return
    if as_json:
        print(json.dumps({**entry.public(), "flocks": flocks}, ensure_ascii=False))
        return
    rows = _entry_rows(entry, flocks=flocks)
    memory = RobotMemory(entry.memory_key).summary()
    rows.append(
        (
            "memory",
            Text(
                f"{_plural(int(memory['notes']), 'note')}, "
                f"{_plural(int(memory['episodes']), 'run')}  {memory['path']}",
                style=ui.STYLES["muted"],
            ),
        )
    )
    ui.console.print(ui.kv_grid(rows))


_CLEARABLE = ("address", "token", "camera-url", "rest-pose", "host", "host-token", "llm", "note")


@robot_app.command("edit")
def robot_edit(
    name: str = _ROBOT_NAME,
    spec: str | None = typer.Option(None, "--spec", help="Move it to another <adapter>:<backend>."),
    address: str | None = _ADDR,
    camera_url: list[str] = _CAMERA_URL,
    token: str | None = _TOKEN,
    host: str | None = _ROBOT_HOST,
    host_token: str | None = _ROBOT_HOST_TOKEN,
    llm: str | None = typer.Option(
        None,
        "--llm",
        "-l",
        help="The pilot, as VENDOR[:MODEL]. Replaces whatever was stored.",
        autocompletion=_complete_llm,
        rich_help_panel="Model",
    ),
    note: str | None = typer.Option(None, "--note", help="One line for people."),
    clear: list[str] = typer.Option(
        [],
        "--clear",
        help=f"Empty one field: {', '.join(_CLEARABLE)}. Repeatable. Clearing host clears "
        "its token too.",
    ),
    registry_dir: str | None = _REGISTRY_DIR,
) -> None:
    """Change what a registered robot is or where it is."""
    from quackd.adapters.base import AdapterError
    from quackd.agent.providers.base import ProviderError
    from quackd.agent.providers.factory import parse_llm
    from quackd.registry import RegistryError

    given: dict[str, Any] = {
        "spec": spec,
        "address": address,
        # every url given replaces the whole stored set: naming a camera today says where the
        # cameras are today, which is the rule --address and --token already follow
        "camera_url": list(camera_url) or None,
        "token": token,
        "host": host,
        "host_token": host_token,
        "llm": llm,
        "note": note,
    }
    if llm is not None and not llm.strip():
        # An empty value is not a pilot, and silently taking it as "forget the one you had"
        # loses a setting and reports success. `--clear llm` is the way to say that, and it is
        # a word rather than an absence.
        _fail("--llm needs a pilot: quackd robot edit NAME --clear llm forgets the stored one")
        return
    for flag, value, what in (("host", host, "a machine"), ("host-token", host_token, "a token")):
        if value is not None and not value.strip():
            # the same rule as --llm, for the same reason: an empty value would store nothing
            # and report success, and forgetting a board is `--clear`'s job
            _fail(
                f"--{flag} needs {what}: quackd robot edit NAME --clear {flag} forgets the "
                "stored one"
            )
            return
    changes: dict[str, Any] = {k: v for k, v in given.items() if v is not None}
    for field in clear:
        key = field.strip().lower().replace("-", "_")
        if key not in {c.replace("-", "_") for c in _CLEARABLE}:
            _fail(f"--clear {field}: empty one of {', '.join(_CLEARABLE)}")
            return
        if key in changes:
            _fail(f"--{key.replace('_', '-')} and --clear {field} contradict each other")
            return
        changes[key] = None
    if "host" in changes and changes["host"] is None:
        # A host token is its board's, and `resolve_host` sends a robot's token only when the
        # robot stores a board. Left behind, it would sit in robots.json as a secret for a board
        # this robot no longer names, and `robot show` would say `host token set` beside
        # `host -`. `--host-token` given alongside is left to the registry, which refuses it.
        changes.setdefault("host_token", None)
    if not changes:
        _fail("nothing to change: give a field to set, or --clear FIELD")
        return
    try:
        # The same gate `robot add` puts on the door, for the same reason: a spec typed here
        # is typed by a person who is looking at the answer.
        if llm is not None:
            parse_llm(llm, source="--llm")
        _registry(registry_dir).update_robot(name, changes)
    except (RegistryError, AdapterError, ValidationError, ProviderError) as e:
        _registry_fail(_one_line(e))
        return
    touched = ", ".join(sorted(key.replace("_", "-") for key in changes))
    ui.console.print(_ok_line(f"updated {name}: {touched}"))


@robot_app.command("rest-pose")
def robot_rest_pose(
    name: str = _ROBOT_NAME,
    clear: bool = typer.Option(False, "--clear", help="Forget the pose recorded for it."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask."),
    address: str | None = _ADDR,
    registry_dir: str | None = _REGISTRY_DIR,
    as_json: bool = _JSON,
) -> None:
    """Record where this arm rests: connect, read every joint, and keep the pose under NAME.

    A run starts from it and returns to it before torque is released, so the arm stops falling
    when a run ends. Fold the arm by hand first, with nothing connected, so the pose you record
    is one it can hold with torque off.
    """
    from quackd.adapters.base import AdapterError
    from quackd.adapters.factory import describe, make_adapter
    from quackd.registry import RegistryError
    from quackd.transport.base import TransportError

    registry = _registry(registry_dir)
    try:
        entry = registry.robot(name)
    except RegistryError as e:
        _fail(str(e), hint="quackd robot list")
        return

    if clear:
        if entry.rest_pose is None:
            _fail(
                f"{name} has no rest pose recorded",
                hint=f"quackd robot rest-pose {name}",
            )
            return
        try:
            entry = registry.update_robot(name, {"rest_pose": None})
        except (RegistryError, AdapterError, ValidationError) as e:
            _registry_fail(_one_line(e))
            return
        if as_json:
            print(json.dumps(entry.public()))
            return
        ui.console.print(_ok_line(f"cleared {name}'s rest pose"))
        ui.console.print(
            Text(
                "  a run now leaves the arm where it stands, and torque drops there",
                style=ui.STYLES["muted"],
            )
        )
        return

    if as_json and not yes:
        _fail("--json is for a script, and a script cannot answer a prompt: add --yes")
        return

    try:
        static = describe(entry.robot_spec)
    except AdapterError as e:
        _registry_fail(e)
        return
    if "joint" not in static.intents:
        _fail(
            f"{name} ({entry.key}) has no joints, so there is no rest pose to record",
            hint="a rest pose is for an arm: quackd list-adapters",
        )
        return

    kwargs = entry.adapter_kwargs(address=address)
    # no camera: reading joints needs none, and a webcam that will not open refuses the whole
    # connect. No rest pose either: the arm must be let go of at the pose you are choosing now,
    # not driven back to the one it is replacing.
    kwargs.update(camera_url=(), rest_pose=None)
    try:
        adapter = make_adapter(entry.robot_spec, **kwargs)
    except (AdapterError, ImportError) as e:
        _registry_fail(e if isinstance(e, AdapterError) else AdapterError(str(e)))
        return
    if not getattr(adapter, "supports_rest_pose", False):
        _fail(
            f"{name} ({entry.key}) has joints, and quackd does not drive it to a rest pose "
            "yet: only the LeRobot arm does today"
        )
        return

    # the same shape `Resolved.label` prints, which is what every other line about a
    # registered robot says: the name you gave it, and the body under it
    label = f"{entry.name} ({entry.key})"

    async def read() -> dict[str, float]:
        await adapter.connect()
        try:
            state = await adapter.get_state()
            return {str(k): float(v) for k, v in dict(state.extras.get("joints", {})).items()}
        finally:
            with contextlib.suppress(Exception):
                await adapter.close()

    try:
        with ui.spinner(f"reading {label}"):
            joints = asyncio.run(read())
    except (TransportError, OSError) as e:
        where = f" at {kwargs['address']}" if kwargs.get("address") else ""
        _fail(f"{entry.key}{where}: {e}")
        return
    if not joints:
        _fail(f"{label} reported no joint positions")
        return

    ui.console.print(Text(f"{label} is at", style=ui.STYLES["muted"]))
    ui.console.print(ui.kv_grid((j, f"{v:.1f}") for j, v in joints.items()))
    # A fold past the travel this arm's calibration recorded is still recorded: it is where
    # the arm rests, and a run parks at the edge of the travel and lets it settle there. But
    # the person folding it is the one who can fix it, so they hear it now, before answering,
    # in the words a run and `doctor` will use. The adapter owns the rule; this only asks.
    note_for = getattr(adapter, "rest_pose_note", None)
    if callable(note_for) and (warning := note_for(joints)):
        ui.console.print(_warn_line(str(warning)))
    if not yes:
        if not _can_prompt():
            _fail(
                "no terminal to ask on: pass --yes to record it",
                hint=f"quackd robot rest-pose {name} --yes",
            )
            return
        already = ", replacing the one already recorded" if entry.rest_pose else ""
        with ui.pause_status():
            if not typer.confirm(f"record this as {name}'s rest pose{already}?"):
                raise typer.Exit()

    pose = {joint: round(value, 1) for joint, value in joints.items()}
    try:
        entry = registry.update_robot(name, {"rest_pose": pose})
    except (RegistryError, AdapterError, ValidationError) as e:
        _registry_fail(_one_line(e))
        return
    if as_json:
        print(json.dumps(entry.public()))
        return
    ui.console.print(_ok_line(f"recorded {name}'s rest pose ({_plural(len(pose), 'joint')})"))
    ui.console.print(
        Text(
            f"  quackd run <duck> --robot {name} starts from it and returns to it "
            "before letting go",
            style=ui.STYLES["muted"],
        )
    )


def _connect_warning() -> str:
    """What connecting does to a body that is handed to a person, and why, in the one wording
    `quackd robot release`, `quackd doctor` and the arm's own close note share
    (`adapters.base.CONNECTING_TAKES_TORQUE_OFF`). Imported here rather than at the top, like
    every other `quackd.adapters` import in this file, so `quackd --help` stays quick."""
    from quackd.adapters.base import CONNECTING_TAKES_TORQUE_OFF

    return f"{CONNECTING_TAKES_TORQUE_OFF}, because LeRobot configures them with it off"


def _release_warning() -> str:
    """Said before anything connects, and before the question, because both halves happen to
    an arm a person has to be holding already. The first is upstream's (`configure()` runs
    inside `torque_disabled()`), and it is the reason this cannot wait until after the
    connect: by then the arm has already been limp once."""
    return (
        f"{_connect_warning()}, and the release then lets the arm fall from wherever it is: "
        "hold it now, and keep hold of it until it is down"
    )


def _doctor_warning() -> str:
    """`_release_warning`'s first half, for `doctor --robot` on a body handed to people. A
    probe connects, and an arm left holding itself up, which is when its close note sends a
    person to `doctor`, is limp for that moment like any other, so the person is told to
    support it before the connect rather than finding out during it."""
    return f"{_connect_warning()}: support the arm until doctor has finished with it"


@robot_app.command("release")
def robot_release(
    name: str = _ROBOT_NAME,
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Do not ask. Hold the arm before you run it: nothing waits."
    ),
    address: str | None = _ADDR,
    registry_dir: str | None = _REGISTRY_DIR,
) -> None:
    """Take torque off this arm wherever it stands, while you hold it.

    For an arm left holding itself up, which is what a run does when it cannot get the arm back
    to its rest pose: hold the arm, run this, and put it down. It says what connecting and
    releasing do to the arm, asks, and only then connects, prints the joints, releases, and
    reads torque back. No verb and no MCP tool can do this: it is a command a person types.
    """
    from quackd.adapters.base import AdapterError, HandResult, let_go_if_any
    from quackd.adapters.factory import make_adapter
    from quackd.registry import RegistryError
    from quackd.transport.base import TransportError

    registry = _registry(registry_dir)
    try:
        entry = registry.robot(name)
    except RegistryError as e:
        _fail(str(e), hint="quackd robot list")
        return

    kwargs = entry.adapter_kwargs(address=address)
    # No camera: releasing needs none, and a webcam that will not open refuses the whole
    # connect. The rest pose IS kept, unlike `rest-pose`: it is what the close judges the arm
    # against, so an arm this could not release closes under the ordinary torque rule.
    kwargs.update(camera_url=())
    try:
        adapter = make_adapter(entry.robot_spec, **kwargs)
    except (AdapterError, ImportError) as e:
        _registry_fail(e if isinstance(e, AdapterError) else AdapterError(str(e)))
        return
    if not getattr(adapter, "supports_hand_off", False):
        _fail(
            f"{name} ({entry.key}) is not a body quackd takes torque off: only the LeRobot arm is",
            hint="quackd list-adapters",
        )
        return

    label = f"{entry.name} ({entry.key})"
    if not yes and not _can_prompt():
        # refused before the warning, because nothing is going to happen that a person has to
        # get hold of the arm for
        _fail(
            "no terminal to ask on: pass --yes to release it",
            hint=f"hold the arm, then quackd robot release {name} --yes",
        )
        return
    # Before anything connects: connecting is the first thing that takes torque off, so a
    # warning printed after it would arrive after the arm had already been limp once.
    ui.console.print(_warn_line(_release_warning()))
    if not yes:
        with ui.pause_status():
            if not typer.confirm(f"release torque on {name}?"):
                ui.console.print(
                    Text(
                        "  nothing was connected, and the arm is as it was",
                        style=ui.STYLES["muted"],
                    )
                )
                raise typer.Exit()

    async def release() -> tuple[HandResult, str | None]:
        await adapter.connect()
        try:
            joints: dict[str, float] = {}
            with contextlib.suppress(Exception):
                state = await adapter.get_state()
                joints = {str(k): float(v) for k, v in dict(state.extras.get("joints", {})).items()}
            if joints:
                # where it is before it goes, which is where the person is holding it
                ui.console.print(Text(f"{label} is at", style=ui.STYLES["muted"]))
                ui.console.print(ui.kv_grid((j, f"{v:.1f}") for j, v in joints.items()))
            # No stop first. A stop picks an arm in somebody's hands back up (`_hold`), and
            # this arm is about to be in them: the release goes out on its own.
            released = await let_go_if_any(adapter, anywhere=True)
        finally:
            with contextlib.suppress(Exception):
                await adapter.close()
        note = getattr(adapter, "close_note", None)
        return released, str(note) if note else None

    try:
        released, note = asyncio.run(release())
    except (TransportError, OSError) as e:
        where = f" at {kwargs['address']}" if kwargs.get("address") else ""
        _fail(
            f"{entry.key}{where}: {e}",
            hint="nothing was released by quackd, and a connect that failed part way can leave "
            "some motors limp: keep hold of the arm",
        )
        return

    failed = True
    if released.ok and released.torque_on == ():
        failed = False
        ui.console.print(_ok_line(f"torque reads off on every joint of {name}"))
    elif released.torque_on:
        ui.err_console.print(
            ui.fail_line(
                f"torque still reads on for {', '.join(released.torque_on)}: cut the power",
                hint=released.reason,
            )
        )
    elif released.ok:
        # sent, and never read back: the arm is treated as limp, which is what keeps it up, and
        # the motors after one whose write failed may still hold, which only the switch settles
        ui.err_console.print(
            ui.fail_line(
                f"torque was taken off and could not be read back: {released.reason}",
                hint="hold the arm as though nothing holds it, and cut its power to be sure",
            )
        )
    else:
        ui.err_console.print(
            ui.fail_line(
                f"nothing was released: {released.reason}",
                hint="run it again, or hold the arm and cut its power",
            )
        )
    if note:
        # after a release this is the limp-in-your-hands line, which is the one to end on
        ui.console.print(_warn_line(note))
    if failed:
        raise typer.Exit(code=1)


@robot_app.command("remove")
def robot_remove(
    name: str = _ROBOT_NAME,
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask."),
    force: bool = typer.Option(
        False, "--force", help="Also drop it from every flock that lists it."
    ),
    registry_dir: str | None = _REGISTRY_DIR,
) -> None:
    """Forget a registered robot. Its memory file stays, and after this only the path finds it,
    because `quackd memory` addresses a robot by a name that is no longer registered."""
    from quackd.memory import RobotMemory
    from quackd.registry import RegistryError

    try:
        registry = _registry(registry_dir)
        entry = registry.robot(name)
        holding = registry.flocks_of(name)
    except RegistryError as e:
        _fail(str(e), hint="quackd robot list")
        return
    if holding and not force:
        _registry_fail(_in_use(name, holding))
        return
    also = f" and drop it from {', '.join(holding)}" if holding else ""
    if not yes:
        with ui.pause_status():
            if not typer.confirm(f"remove {name} ({entry.key}){also}?"):
                raise typer.Exit()
    try:
        dropped = registry.remove_robot(name, force=force)
    except RegistryError as e:
        _registry_fail(e)
        return
    ui.console.print(_ok_line(f"removed {name}"))
    kept = RobotMemory(entry.memory_key).path
    if kept.exists():
        # `quackd memory clear --robot <name>` cannot reach it any more: the name is gone
        ui.console.print(Text(f"  its notes are still at {kept}", style=ui.STYLES["muted"]))
    if dropped:
        ui.console.print(Text(f"  dropped from {', '.join(dropped)}", style=ui.STYLES["muted"]))


def _in_use(name: str, flocks: list[str]) -> Exception:
    from quackd.registry import RobotInUse

    return RobotInUse(name, flocks)


# ── flock (the stored groups) ───────────────────────────────────────────────────────────

flock_app = typer.Typer(
    name="flock",
    help="Named groups of registered robots, for --flock NAME on run and serve-mcp. Not the "
    "flock: block of a .duck file, which says how a task is shared out: this says which "
    "bodies share it. Kept in ~/.quackd/flocks.json.",
    no_args_is_help=True,
)
app.add_typer(flock_app, name="flock", rich_help_panel="Robots")

_FLOCK_NAME = typer.Argument(..., help="A slug. Not a number: --flock N means N simulated ducks.")
_FLOCK_MEMBER = typer.Option(
    [], "--robot", "-r", help="A registered robot to include. Repeatable, order kept."
)
PROMPT_TRIES = 3
"""How many times the picker re-asks before giving up. Enough for a typo, not a loop."""


def _flock_status(flock: Any, missing: list[str]) -> Any:
    """One cell saying whether this flock could run, and what to do if not."""
    if missing:
        text, style = f"{', '.join(missing)} not registered", ui.STYLES["warn"]
    elif not flock.members:
        text, style = "empty", ui.STYLES["warn"]
    elif len(flock.members) < 2:
        text, style = "1 robot: running needs 2 to 8", ui.STYLES["warn"]
    else:
        return Text("ok", style=ui.STYLES["ok"])

    def build(g: ui.Glyphs) -> Any:
        return Text(f"{g.warn} {text}", style=style)

    return ui.Deferred(build)


def _members_cell(flock: Any, missing: list[str]) -> Any:
    def build(g: ui.Glyphs) -> Any:
        out = Text()
        for i, member in enumerate(flock.members):
            if i:
                out.append(f" {g.dot} ", style=ui.STYLES["muted"])
            if member in missing:
                out.append(f"{g.warn} {member}", style=ui.STYLES["warn"])
            else:
                out.append(member)
        return out or Text("-", style=ui.STYLES["muted"])

    return ui.Deferred(build)


def _pick_members(registry: Any) -> list[str]:
    """The numbered table, and the answer. Only reached when there is a terminal to ask on."""
    entries = registry.robots()
    table = ui.table("robots you have registered")
    table.add_column("#", no_wrap=True, justify="right", style=ui.STYLES["muted"])
    table.add_column("name", no_wrap=True, style=ui.STYLES["key"])
    table.add_column("robot", no_wrap=True)
    table.add_column("note", ratio=1)
    names = list(entries)
    for i, name in enumerate(names, 1):
        entry = entries[name]
        table.add_row(
            Text(str(i)),
            Text(name),
            Text(entry.key, style=ui.STYLES["accent"]),
            Text(entry.note or ""),
        )
    ui.console.print(table)
    for attempt in range(PROMPT_TRIES):
        answer = typer.prompt(
            "which robots? (numbers or names, comma separated, empty to cancel)",
            default="",
            show_default=False,
        )
        if not answer.strip():
            ui.console.print(Text("nothing created", style=ui.STYLES["muted"]))
            raise typer.Exit()
        chosen, problem = _read_picks(answer, names)
        if problem is None:
            return chosen
        ui.err_console.print(Text(problem, style=ui.STYLES["warn"]))
        if attempt == PROMPT_TRIES - 1:
            _fail(f"no valid answer in {PROMPT_TRIES} tries")
    return []


def _read_picks(answer: str, names: list[str]) -> tuple[list[str], str | None]:
    """`1, 3` or `duck-a, arm` or both. Returns the names, or what was wrong with the answer."""
    picked: list[str] = []
    tokens = [t for t in re.split(r"[,\s]+", answer.strip()) if t]
    for token in tokens:
        if token.isdigit():
            index = int(token)
            if not 1 <= index <= len(names):
                return [], f"there is no robot {token}: pick 1 to {len(names)}"
            name = names[index - 1]
        elif token in names:
            name = token
        else:
            return [], (
                f"no robot called {token!r}: pick 1 to {len(names)}, or a name from the table"
            )
        if name in picked:
            return [], f"{name} twice"
        picked.append(name)
    return picked, None


@flock_app.command("create")
def flock_create(
    name: str = _FLOCK_NAME,
    robot: list[str] = _FLOCK_MEMBER,
    description: str | None = typer.Option(None, "--description", help="One line for people."),
    registry_dir: str | None = _REGISTRY_DIR,
) -> None:
    """Name a group of registered robots. With no --robot it lists them and asks."""
    from quackd.registry import MAX_MEMBERS, RegistryError, StoredFlock

    try:
        registry = _registry(registry_dir)
        known = registry.robots()
    except RegistryError as e:
        _registry_fail(e)
        return
    members = list(robot)
    if not members:
        if not known:
            _fail(
                "no robots registered yet: quackd robot add NAME <adapter>:<backend> first",
                hint="quackd robot list",
            )
            return
        if not _can_prompt():
            _fail(
                "no terminal to ask on: name the robots yourself",
                hint="quackd flock create NAME --robot A --robot B",
            )
            return
        members = _pick_members(registry)
    try:
        flock = registry.add_flock(StoredFlock(name=name, members=members, description=description))
    except (RegistryError, ValidationError) as e:
        _registry_fail(_one_line(e))
        return
    ui.console.print(
        _ok_line(
            f"created flock {flock.name}: {', '.join(flock.members) or 'no robots'} "
            f"({_plural(len(flock.members), 'robot')})"
        )
    )
    if len(flock.members) < 2:
        ui.err_console.print(
            Text(
                f"  a flock runs with 2 to {MAX_MEMBERS}: "
                f"quackd flock edit {flock.name} --add NAME",
                style=ui.STYLES["warn"],
            )
        )
        return
    ui.console.print(Text(f"  quackd run <duck> --flock {flock.name}", style=ui.STYLES["muted"]))


@flock_app.command("list")
def flock_list(
    registry_dir: str | None = _REGISTRY_DIR,
    as_json: bool = _JSON,
) -> None:
    """Every flock you have made, and whether it could run."""
    from quackd.registry import RegistryError

    try:
        registry = _registry(registry_dir)
        flocks = registry.flocks()
        missing = {name: registry.missing_members(f) for name, f in flocks.items()}
    except RegistryError as e:
        _registry_fail(e)
        return
    if as_json:
        for name, flock in flocks.items():
            print(json.dumps(flock.public(missing[name]), ensure_ascii=False))
        return
    if not flocks:
        ui.console.print(
            Text(
                "no flocks yet: quackd flock create NAME --robot A --robot B",
                style=ui.STYLES["muted"],
            )
        )
        return
    table = ui.table("flocks (--flock NAME)")
    table.add_column("name", no_wrap=True, style=ui.STYLES["key"])
    table.add_column("robots", overflow="fold")
    table.add_column("status", overflow="fold")
    table.add_column("description", ratio=1)
    for name, flock in flocks.items():
        table.add_row(
            Text(name),
            _members_cell(flock, missing[name]),
            _flock_status(flock, missing[name]),
            Text(flock.description or ""),
        )
    ui.console.print(table)
    if any(missing.values()):
        ui.console.print(
            Text(
                "a robot marked as not registered was removed by hand: quackd robot add it "
                "back, or quackd flock edit NAME --remove it",
                style=ui.STYLES["muted"],
            )
        )


@flock_app.command("show")
def flock_show(
    name: str = _FLOCK_NAME,
    registry_dir: str | None = _REGISTRY_DIR,
    as_json: bool = _JSON,
) -> None:
    """One flock, and the robots in it."""
    from quackd.registry import RegistryError

    try:
        registry = _registry(registry_dir)
        flock = registry.flock(name)
        missing = registry.missing_members(flock)
        known = registry.robots()
    except RegistryError as e:
        _fail(str(e), hint="quackd flock list")
        return
    if as_json:
        payload = {
            **flock.public(missing),
            "robots": [known[m].public() for m in flock.members if m in known],
        }
        print(json.dumps(payload, ensure_ascii=False))
        return
    rows: list[tuple[str, Any]] = [
        ("name", Text(flock.name, style=ui.STYLES["key"])),
        ("robots", _members_cell(flock, missing)),
        ("status", _flock_status(flock, missing)),
    ]
    if flock.description:
        rows.append(("description", Text(flock.description)))
    rows += [
        ("created", Text(flock.created, style=ui.STYLES["muted"])),
        ("updated", Text(flock.updated, style=ui.STYLES["muted"])),
    ]
    ui.console.print(ui.kv_grid(rows))
    if not flock.members:
        return
    table = ui.table("its robots")
    table.add_column("#", no_wrap=True, justify="right", style=ui.STYLES["muted"])
    table.add_column("name", no_wrap=True, style=ui.STYLES["key"])
    table.add_column("robot", no_wrap=True)
    table.add_column("address", overflow="fold")
    table.add_column("note", ratio=1)
    for i, member in enumerate(flock.members, 1):
        entry = known.get(member)
        if entry is None:
            table.add_row(
                Text(str(i)),
                Text(member, style=ui.STYLES["warn"]),
                Text("not registered", style=ui.STYLES["warn"]),
                Text(""),
                Text(""),
            )
            continue
        table.add_row(
            Text(str(i)),
            Text(member),
            Text(entry.key, style=ui.STYLES["accent"]),
            Text(entry.address or "-", style="" if entry.address else ui.STYLES["muted"]),
            Text(entry.note or ""),
        )
    ui.console.print(table)


@flock_app.command("edit")
def flock_edit(
    name: str = _FLOCK_NAME,
    add: list[str] = typer.Option([], "--add", help="A registered robot to add. Repeatable."),
    remove: list[str] = typer.Option([], "--remove", help="A member to drop. Repeatable."),
    description: str | None = typer.Option(
        None, "--description", help="Change it. An empty string clears it."
    ),
    rename: str | None = typer.Option(None, "--rename", help="A new name for the flock."),
    registry_dir: str | None = _REGISTRY_DIR,
) -> None:
    """Add or drop members, change the description, or rename the flock."""
    from quackd.registry import RegistryError

    if not (add or remove or description is not None or rename):
        _fail("nothing to change: --add, --remove, --description or --rename")
        return
    try:
        flock = _registry(registry_dir).update_flock(
            name, add=add, remove=remove, description=description, rename=rename
        )
    except (RegistryError, ValidationError) as e:
        _registry_fail(_one_line(e))
        return
    done: list[str] = []
    if add:
        done.append(f"added {', '.join(add)}")
    if remove:
        done.append(f"removed {', '.join(remove)}")
    if description is not None:
        done.append("cleared the description" if not description else "changed the description")
    if rename:
        done.append(f"renamed to {rename}")
    ui.console.print(_ok_line(f"updated {name}: {'; '.join(done)}"))
    ui.console.print(
        Text(f"  {flock.name}: {', '.join(flock.members) or 'no robots'}", style=ui.STYLES["muted"])
    )


@flock_app.command("delete")
def flock_delete(
    name: str = _FLOCK_NAME,
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask."),
    registry_dir: str | None = _REGISTRY_DIR,
) -> None:
    """Forget a flock. The robots in it stay registered."""
    from quackd.registry import RegistryError

    try:
        registry = _registry(registry_dir)
        flock = registry.flock(name)
    except RegistryError as e:
        _fail(str(e), hint="quackd flock list")
        return
    if not yes:
        with ui.pause_status():
            asked = typer.confirm(
                f"delete flock {name} ({_plural(len(flock.members), 'robot')})? "
                "the robots stay registered"
            )
            if not asked:
                raise typer.Exit()
    registry.delete_flock(name)
    ui.console.print(_ok_line(f"deleted flock {name}"))


# ── memory ──────────────────────────────────────────────────────────────────────────────

memory_app = typer.Typer(
    name="memory",
    help="What a robot remembers between runs: notes the pilot saved, and how runs ended.",
    no_args_is_help=True,
)
app.add_typer(memory_app, name="memory", rich_help_panel="Memory")


def _memory_for(robot: str | None, memory_dir: str | None, registry_dir: str | None = None) -> Any:
    from quackd.adapters.base import AdapterError
    from quackd.memory import RobotMemory
    from quackd.registry import Registry, RegistryError, resolve_robot_ref

    try:
        resolved = resolve_robot_ref(robot, Registry(registry_dir))
    except (AdapterError, RegistryError) as e:
        # every other --robot command answers in one line, not a traceback
        _fail(str(e))
    # a registered robot keys by its name, so two ducks of one kind keep separate notes
    return RobotMemory(resolved.memory_key, memory_dir)


@memory_app.command("show")
def memory_show(
    robot: str | None = _ROBOT,
    memory_dir: str | None = _MEMORY_DIR,
    registry_dir: str | None = _REGISTRY_DIR,
    raw: bool = typer.Option(False, "--raw", help="Print the JSONL file as is."),
) -> None:
    """Print what one robot remembers (default: the Microduck simulator)."""
    mem = _memory_for(robot, memory_dir, registry_dir)
    if raw:
        if mem.path.exists():
            # "as is" means as is: a note saying "the ball is [bold]behind[/bold] the sofa"
            # is markup Rich would silently eat, and an unpaired tag would raise.
            print(mem.path.read_text(encoding="utf-8"), end="")
        return
    info = mem.summary()
    ui.console.print(
        Text.assemble(
            (str(info["robot"]), ui.STYLES["key"]),
            (f"  {_plural(info['notes'], 'note')}, {_plural(info['episodes'], 'run')}  ", ""),
            (str(info["path"]), ui.STYLES["muted"]),
        )
    )
    notes, episodes = mem.notes(), mem.episodes()
    if not notes and not episodes:
        ui.console.print(Text("nothing remembered yet", style=ui.STYLES["muted"]))
        return
    if notes:
        table = ui.table("notes the pilot saved")
        table.add_column("date", no_wrap=True)
        table.add_column("tags", style=ui.STYLES["muted"])
        table.add_column("note", ratio=1)
        for entry in reversed(notes[-50:]):
            table.add_row(Text(entry.date), Text(", ".join(entry.tags)), Text(entry.text))
        ui.console.print(table)
    if episodes:
        table = ui.table("how recent runs ended")
        table.add_column("date", no_wrap=True)
        table.add_column("duck", no_wrap=True)
        table.add_column("outcome", no_wrap=True)
        table.add_column("what happened", ratio=1)
        for entry in reversed(episodes[-10:]):
            outcome = str(entry.outcome or "")
            what = Text(_episode_detail(entry))
            if entry.highlights:
                joined = "; ".join(entry.highlights)
                what.append(NEWLINE + joined, style=ui.STYLES["muted"])
            table.add_row(
                Text(entry.date),
                Text(str(entry.duck or "")),
                Text(outcome, style=ui.STYLES["ok" if outcome == "success" else "warn"]),
                what,
            )
        ui.console.print(table)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _episode_detail(entry: Any) -> str:
    """An episode's text without the duck and outcome it already has columns for."""
    prefix = f"{entry.duck}: {entry.outcome} — "
    if entry.duck and entry.outcome and entry.text.startswith(prefix):
        return str(entry.text[len(prefix) :])
    return str(entry.text)


@memory_app.command("add")
def memory_add(
    text: str = typer.Argument(..., help="One short fact, e.g. 'the ball lives by the sofa'."),
    robot: str | None = _ROBOT,
    memory_dir: str | None = _MEMORY_DIR,
    registry_dir: str | None = _REGISTRY_DIR,
    tag: list[str] = typer.Option([], "--tag", help="Optional label(s)."),
) -> None:
    """Save a note by hand, the same way the pilot's `remember` does."""
    mem = _memory_for(robot, memory_dir, registry_dir)
    entry = mem.remember(text, tags=tag)
    ui.console.print(
        Text.assemble(
            ("remembered for ", ""), (mem.robot_key, ui.STYLES["key"]), (": ", ""), entry.text
        )
    )


@memory_app.command("clear")
def memory_clear(
    robot: str | None = _ROBOT,
    memory_dir: str | None = _MEMORY_DIR,
    registry_dir: str | None = _REGISTRY_DIR,
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask."),
) -> None:
    """Forget everything one robot remembers (deletes its memory file)."""
    mem = _memory_for(robot, memory_dir, registry_dir)
    n = len(mem.entries())
    if n == 0:
        ui.console.print(Text(f"{mem.robot_key}: nothing to forget", style=ui.STYLES["muted"]))
        return
    if not yes and not typer.confirm(f"forget {n} entries for {mem.robot_key}?"):
        raise typer.Exit()
    mem.clear()
    ui.console.print(
        Text.assemble((f"forgot {n} entries for ", ""), (mem.robot_key, ui.STYLES["key"]))
    )


# ── lan (quackd[lan]) ───────────────────────────────────────────────────────────────────


@app.command(rich_help_panel="LAN")
def discover(
    timeout: float = typer.Option(3.0, "--timeout", help="Seconds to listen for answers."),
    as_json: bool = typer.Option(False, "--json", help="One JSON object per robot."),
) -> None:
    r"""List the quackd robots answering on the LAN (zeroconf, needs quackd\[lan])."""
    from quackd.lan import LanNotInstalled
    from quackd.lan import discover as lan_discover

    try:
        # it listens for the whole timeout whether anything answers or not, so say so
        with ui.spinner(f"listening for quackd robots ({timeout:g} s)"):
            robots = lan_discover.discover(timeout)
    except LanNotInstalled as e:
        _fail(str(e), hint="pip install 'quackd[lan]' adds zeroconf")
    if as_json:
        for robot in robots:
            print(json.dumps(robot.row()))
        return
    if not robots:
        ui.console.print(
            Text(f"no quackd robots answered in {timeout:g} s", style=ui.STYLES["muted"])
        )
        return
    t = ui.table(f"quackd robots on the LAN ({len(robots)})")
    t.add_column("manifest id", style=ui.STYLES["key"], no_wrap=True)
    for column in ("adapter", "model", "embodiment", "verbs", "address", "digest"):
        t.add_column(column)
    for robot in robots:
        t.add_row(
            Text(robot.manifest_id),
            Text(robot.adapter),
            Text(robot.model),
            Text(robot.embodiment),
            Text(str(robot.n_verbs)),
            Text(", ".join(robot.addresses) or robot.host),
            Text(robot.digest, style=ui.STYLES["muted"]),
        )
    ui.console.print(t)


@app.command(rich_help_panel="LAN")
def announce(
    robot: str = typer.Option(
        ..., "--robot", "-r", help="<adapter>:<backend> to advertise (static manifest, no robot)."
    ),
    name: str | None = typer.Option(
        None, "--name", help="Manifest id to advertise (default: the adapter's own)."
    ),
    port: int = typer.Option(0, "--port", help="Service port to advertise; 0 = identity only."),
    for_s: float | None = typer.Option(
        None, "--for", help="Seconds to stay announced (default: until Ctrl-C)."
    ),
) -> None:
    r"""Advertise a robot's identity on the LAN (zeroconf, needs quackd\[lan])."""
    import time

    from quackd.adapters.base import AdapterError
    from quackd.adapters.factory import RobotSpec, describe, parse_robot_spec
    from quackd.lan import LanNotInstalled
    from quackd.lan import announce as lan_announce

    try:
        parsed = parse_robot_spec(robot)
        spec = RobotSpec(parsed.adapter, parsed.backend, name)
        manifest = describe(spec)
        ann = lan_announce.announce(manifest, adapter=spec.adapter, port=port)
    except (AdapterError, LanNotInstalled, ValueError) as e:
        _fail(str(e))
    ui.console.print(
        ui.run_header(
            ann.record.name,
            [
                ("robot", manifest.summary()),
                ("at", ", ".join(ann.record.addresses)),
                ("digest", manifest.digest()),
            ],
            hint="Ctrl-C to withdraw" if for_s is None else f"withdrawing in {for_s:g} s",
        )
    )
    try:
        with ui.spinner(f"announcing {ann.record.name}"):
            if for_s is None:
                while True:
                    time.sleep(1.0)
            else:
                time.sleep(for_s)
    except KeyboardInterrupt:
        pass
    finally:
        ann.close()
        ui.console.print(Text("withdrawn", style=ui.STYLES["muted"]))


def _stdout_alive() -> bool:
    """Whether stdout can still be written to. A closed pipe fails the flush."""
    try:
        sys.stdout.flush()
    except OSError:
        return False
    return not sys.stdout.closed


def _leave_quietly() -> None:
    """Stop, with nothing further to say.

    Python flushes stdout as it exits, which raises a second time on a pipe that has already
    gone, so stdout is pointed at the void before leaving."""
    with contextlib.suppress(Exception):
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    raise SystemExit(0)


def _rich_traceback(kind: type[BaseException], exc: BaseException, tb: Any) -> None:
    """A crash, rendered, without this process's locals in it: they hold an API key, a
    robot's address and its bridge token.

    Built here rather than installed once, because the console `--no-color` rebuilt does not
    exist until the root callback has run and an excepthook fires long after that."""
    from rich.traceback import Traceback

    ui.err_console.print(
        Traceback.from_exception(kind, exc, tb, show_locals=False, suppress=[typer])
    )


def main() -> None:
    """The console entry point: `app()`, and the two things a command that prints for a
    living owes its terminal.

    A traceback must not spill this process's locals, because they hold an API key, a
    robot's address and its bridge token. And a reader is allowed to walk away: `quackd
    list-verbs | head` closes the pipe halfway down the table, and Python's answer to that
    is a second wall of text about a broken pipe on top of the output that was asked for."""
    sys.excepthook = _rich_traceback
    try:
        app()
    except BrokenPipeError:
        _leave_quietly()
    except OSError as e:
        # Windows answers a write to a pipe nobody is reading with EINVAL rather than EPIPE,
        # and EINVAL is far too common an errno to swallow on its own word: only when stdout
        # is the stream that has actually stopped accepting writes is this a reader leaving.
        if e.errno not in (errno.EPIPE, errno.EINVAL) or _stdout_alive():
            raise
        _leave_quietly()


if __name__ == "__main__":
    main()
