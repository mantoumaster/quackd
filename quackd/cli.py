"""The command line is the product's front door.

`uvx quackd run find-and-kick --provider anthropic --robot microduck:sim2d` is the
north-star demo; every command here exists to make that line, and the debugging around it,
boring. Commands are thin: they parse, load `.env`, wire objects together, and hand off.
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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from pydantic import ValidationError
from rich.text import Text

from quackd import __version__, ui
from quackd.agent.providers.catalogue import (
    CLOUD_NAMES,
    LOCAL_NAMES,
    PROVIDER_NAMES,
    models_for,
    vendor_of,
)

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
        "[bold]Try[/bold]" + "\n\n"
        "quackd run find-and-kick --provider fake" + "\n\n"
        "quackd run --goal 'walk in a square' --provider anthropic --robot microduck:mujoco"
        + "\n\n"
        "quackd doctor  |  quackd list-verbs  |  quackd trace"
    ),
)


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
    load_dotenv()
    ui.configure(no_color=no_color)


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
    trace_dropped: int = 0,
) -> None:
    """How the run ended, in the one place a person looks after looking away.

    `quackd trace` prints it from the transcript too, so a replay ends exactly the way the
    run itself did rather than in a second dialect somebody has to keep in step. `counters`
    is a list because a flock counts different things than a solo run does."""
    ui.console.print(
        ui.verdict(outcome, reason, counters=counters, run_dir=run_dir, gif_path=gif_path)
    )
    if trace_dropped:
        # a console that raised on every event produced a silent trace and no sign of it
        ui.err_console.print(
            f"trace: {trace_dropped} line(s) could not be shown (the console raised); "
            "transcript.jsonl has them",
            style="yellow",
            markup=False,
        )


def _header_rows(
    *, provider: Any, robot: str, seed: int | None, dry_run: bool, memory: Any
) -> list[tuple[str, Any]]:
    """The four things worth knowing before a run starts, and nothing else."""
    rows: list[tuple[str, Any]] = [
        ("provider", f"{provider.name} ({provider.model or 'the first model it serves'})"),
        ("robot", robot + (f"  seed {seed}" if seed is not None else "")),
    ]
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


# ── list-models ───────────────────────────────────────────────────────────────────────────


@app.command("list-models", rich_help_panel="Inspect")
def list_models_cmd(
    provider: str | None = typer.Option(
        None, "--provider", "-p", help="One vendor only. Omitted: every vendor."
    ),
    as_json: bool = _JSON,
) -> None:
    """List the model ids each cloud provider accepts for --model, and which is the default."""
    if provider is not None:
        provider = provider.lower()
        if provider not in PROVIDER_NAMES:
            _fail(
                f"unknown provider {provider!r}",
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
        table = ui.table("models (--model, QUACKD_MODEL)")
        table.add_column("provider", style=ui.STYLES["key"], no_wrap=True)
        # An id is meant to be copied into `--model`, so it may wrap but must never be elided:
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
            f"{', '.join(LOCAL_NAMES)}: no catalogue. `--model` takes any id the server serves, "
            "and without one quackd takes the first entry of /v1/models."
        )
    if provider is None or provider == "fake":
        notes.append("fake: scripted, and `--model` is ignored.")
    if pinned := os.environ.get("QUACKD_MODEL"):
        whose = vendor_of(pinned)
        where = f"a {whose} model" if whose else "not a model any vendor here lists"
        notes.append(f"QUACKD_MODEL={pinned} is {where}.")
    for note in notes:
        ui.console.print(Text(note, style=ui.STYLES["muted"]), soft_wrap=True)


# ── run / record ────────────────────────────────────────────────────────────────────────


def _yes_to_go(_why: str) -> bool:
    """`--yes` answers the pilot's doubt the way it answers a confirm gate: go."""
    return True


def _confirm_prompt(name: str, params: dict[str, Any]) -> bool:
    # under a running status line the question is invisible: a live region redirects stdout
    # and a prompt writes without a newline, so it stays buffered until it is too late
    with ui.pause_status():
        return typer.confirm(f"run {name}({params})?", default=False)


def _decide_prompt(why: str) -> bool:
    """Asked when the pilot says it is not sure this body can do the task at all."""
    with ui.pause_status():
        ui.err_console.print(Text(why, style=ui.STYLES["warn"]))
        return typer.confirm("Go ahead anyway?", default=False)


def _acknowledge_prompt(why: str) -> bool:
    """Asked once, before anything moves, when the human is the only safety left."""
    with ui.pause_status():
        ui.err_console.print(Text(why, style=ui.STYLES["warn"]))
        return typer.confirm("Are you watching the robot right now?", default=False)


def _entry_model(resolved: Any, provider: str | None) -> str | None:
    """A registered robot's model, but only for the provider it was registered against.

    `--provider openai` on a robot registered `anthropic` + `claude-opus-5` must not carry the
    Claude id into OpenAI's catalogue, where it is refused with a message blaming `--model`."""
    if resolved.model is None:
        return None
    if provider and resolved.provider and provider.lower() != resolved.provider:
        return None
    return str(resolved.model)


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
    provider: str | None,
    model: str | None,
    seed: int | None,
    dry_run: bool,
    max_steps: int | None,
    runs_dir: str,
    yes: bool,
    live: bool,
    address: str | None,
    camera_url: str | None,
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
    robot: str | None = None,
    robots: str | None = None,
    memory: bool = True,
    memory_dir: str | None = None,
    registry_dir: str | None = None,
    trace: bool | None = None,
    trace_prompt: bool | None = None,
) -> None:
    from quackd.adapters.factory import describe, make_adapter, registry_for
    from quackd.agent.loop import RunConfig, run_duck
    from quackd.agent.providers.base import ProviderError
    from quackd.agent.providers.factory import make_provider
    from quackd.duckfile.parser import DuckParseError, duck_from_goal, load_duck
    from quackd.duckfile.schema import AUCTION_MAX_MEMBERS, PILOTS_MAX_MEMBERS
    from quackd.duckfile.validate import validate_duck
    from quackd.flock.pilots import ADVISORY_FIELDS, roster_from_specs
    from quackd.flock.runner import member_specs
    from quackd.perception import detector_for
    from quackd.registry import RegistryError, Resolved
    from quackd.safety import KillSwitch, allow_all
    from quackd.trace import (
        ConsoleTrace,
        fan_out,
        prompt_shown_default,
        thinking_limit_default,
        trace_enabled_default,
    )
    from quackd.transport.base import TransportError

    if (duckfile is None) == (goal is None):
        _fail('give either a .duck file (or bundled name) or --goal "...", not both')
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
        if goal is not None:
            # the union across the flock, so a goal run on mixed bodies allows what any of
            # them can do; each member is then trimmed to its own half of that
            safe = sorted(
                {
                    v.name
                    for one in specs
                    for v in registry_for(one).verbs()
                    if v.safety_class == "safe"
                }
            )
            duck = duck_from_goal(goal, safe)
        assert duck is not None
        # Refuse before connecting, with the validator's words. `serve-mcp` has always done
        # this; `run` never did, and reached the loop's tool_schemas and died on a raw
        # VerbNotFound with the robot already connected and a run directory already made.
        manifests = [describe(s) for s in specs]
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
    if flock_n is not None or roster is not None or duck.frontmatter.flock is not None:
        if method == "pilots":
            if roster is None and section is not None:
                # `flock.members` plus `robots:` or `--robots` names the bodies without a
                # registry; a stored flock names them with one
                roster = roster_from_specs(
                    member_specs(
                        section.member_names,
                        {s.name: s.key for s in specs if s.name} or None,
                        duck.frontmatter.robots,
                    )
                )
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
                provider=provider,
                model=model,
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
                trace=trace,
                trace_prompt=trace_prompt,
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
            provider=provider,
            specs=specs,
            members=list(roster) if roster is not None else None,
            flock_name=flock_name,
            model=model,
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
            trace=trace,
            trace_prompt=trace_prompt,
        )
        return
    try:
        # a registered robot may name the pilot that drives it; a flag on the line still wins
        llm = make_provider(
            provider or here.provider or DEFAULT_PROVIDER,
            model=model or _entry_model(here, provider),
            duck_name=duck.name,
            goal=goal,
            base_url=base_url,
            api_key=api_key,
            vision=vision,
            extra_body=extra_body,
        )
        duck_transport = make_adapter(
            spec,
            seed=seed,
            live=live,
            **here.adapter_kwargs(address=address, camera_url=camera_url, token=token),
        )
    except (ProviderError, TransportError, ImportError) as e:
        _fail(str(e))
        return

    recorder = None
    # Any robot with a camera needs something to look at its frames with, not just the
    # simulator. This is the static manifest, so it is only a head start: the loop asks
    # again with the live one at connect, where a robot may report a camera this does not
    # know about (a rosbridge base) or lack one this promises (a duck built without a head).
    detector = detector_for(
        manifests[0].sensors,
        fov_deg=fov_deg or manifests[0].limits.get("camera_fov_deg"),
        backend=spec.backend,
    )
    # the recorder is sim2d only: it draws the world, and only the simulator has one
    if spec.backend in ("sim2d", "mujoco") and gif:
        from quackd.sim2d.recorder import FrameRecorder

        recorder = FrameRecorder(duck_transport, size=gif_size)

    # The flag wins; else QUACKD_TRACE, read here rather than at import so a `.env` line
    # counts (the root callback loads it after the option defaults exist).
    trace_on = trace if trace is not None else trace_enabled_default()
    console_trace = (
        ConsoleTrace(
            ui.err_console,
            thinking_chars=thinking_limit_default(),
            prompt=trace_prompt if trace_prompt is not None else prompt_shown_default(),
        )
        if trace_on
        else None
    )

    # on whether or not the trace is: with --no-trace this is the only thing between the
    # header and the verdict, and a model can think for a minute
    status = ui.RunStatus()
    ui.install_logging()

    def log(msg: str) -> None:
        # the compact view: one line per verb and the executor's notes. The trace shows all
        # of that and more, so with it on this prints nothing rather than every verb twice.
        if verbose and console_trace is None:
            _verbose_line(msg)

    robot_memory = None
    if memory:
        from quackd.memory import RobotMemory

        # keyed by adapter:backend, so a simulated duck never inherits a real one's notes,
        # or by the registered name, so two ducks of one kind keep separate notes
        robot_memory = RobotMemory(here.memory_key, memory_dir)
    cfg = RunConfig(
        duck=duck,
        provider=llm,
        transport=duck_transport,
        detector=detector,
        dry_run=dry_run,
        confirm=allow_all if yes else _confirm_prompt,
        runs_dir=runs_dir,
        max_steps=max_steps,
        log=log,
        on_frame=recorder.capture if recorder is not None else None,
        memory=robot_memory,
        fov_deg=fov_deg,
        acknowledge=None if yes else _acknowledge_prompt,
        decide=_yes_to_go if yes else _decide_prompt,
        trace=fan_out(console_trace, status.sink),
    )
    ui.console.print(
        ui.run_header(
            duck.name,
            _header_rows(
                provider=llm, robot=here.label, seed=seed, dry_run=dry_run, memory=robot_memory
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

        loop = AgentLoop(cfg)
        ks = KillSwitch(loop.executor.abort, log=killed)
        ks.install()
        try:
            return await loop.run()
        finally:
            ks.uninstall()

    _ = run_duck  # imported for symmetry; AgentLoop is used directly so the kill switch can bind
    try:
        with status:
            status.update(f"connecting to {here.label}")
            result = asyncio.run(main())
    except (TransportError, ProviderError) as e:
        # the trace has already shown the call that failed; this is the one-line verdict
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
        counters=[
            f"steps {result.steps}",
            f"llm calls {result.llm_calls}",
            f"tokens {result.usage.input_tokens}+{result.usage.output_tokens}",
        ],
        run_dir=result.run_dir,
        gif_path=result.gif_path,
        trace_dropped=result.trace_dropped,
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
    trace_on: bool,
    trace_prompt: bool | None,
    status: Any,
) -> tuple[dict[str, Any], Any]:
    """One console view per member, coloured and prefixed by name, plus the flock's own.

    Shared by both kinds of flock, because a person reading either one needs the same thing:
    several robots narrating at once stay several readable columns rather than one
    interleaving. Returns the views (so the caller can flush them) and the `trace(name)`
    factory the runner takes."""
    from quackd.flock.runner import FLOCK_TRACE
    from quackd.trace import (
        ConsoleTrace,
        Sink,
        fan_out,
        prompt_shown_default,
        thinking_limit_default,
    )

    views: dict[str, ConsoleTrace] = {}
    width = max(len(name) for name in [*member_names, FLOCK_TRACE])

    def view_for(name: str) -> Sink | None:
        if name not in views:
            # a colour per member as well as a name, because robots moving at once interleave
            # and the eye finds a colour faster than it reads a prefix
            order = member_names.index(name) if name in member_names else -1
            views[name] = ConsoleTrace(
                ui.err_console,
                thinking_chars=thinking_limit_default(),
                prompt=trace_prompt if trace_prompt is not None else prompt_shown_default(),
                prefix=f"{name:<{width}}  ",
                prefix_style=ui.MEMBER_STYLES[order % len(ui.MEMBER_STYLES)]
                if order >= 0
                else ui.STYLES["key"],
            )
        return fan_out(views[name], status.sink)

    def status_only(_name: str) -> Sink | None:
        """With --no-trace nothing narrates, but the status line still has to say which robot
        is doing what, or a flock is a minute of nothing at all."""
        return status.sink

    return views, (view_for if trace_on else status_only)


def _run_pilots_impl(
    duck: Any,
    roster: Any,
    *,
    provider: str | None,
    model: str | None,
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
    trace: bool | None = None,
    trace_prompt: bool | None = None,
) -> None:
    """A pilot per body, all at once. The other flock is `_run_flock_impl`."""
    from quackd.agent.providers.base import ProviderError
    from quackd.agent.providers.factory import make_provider
    from quackd.flock.pilots import run_pilot_flock
    from quackd.memory import RobotMemory
    from quackd.safety import KillSwitch
    from quackd.trace import trace_enabled_default
    from quackd.transport.base import TransportError

    members = list(roster)
    if duck.frontmatter.verbs.confirm and not yes:
        _fail("a pilot flock cannot prompt y/N per member: empty verbs.confirm or pass --yes")
        return
    try:
        providers = {
            name: make_provider(
                provider or entry.provider or DEFAULT_PROVIDER,
                model=model or _entry_model(entry, provider),
                duck_name=duck.name,
                goal=goal,
                base_url=base_url,
                api_key=api_key,
                vision=vision,
                extra_body=extra_body,
            )
            for name, entry in roster.items()
        }
    except (ProviderError, ImportError) as e:
        _fail(str(e))
        return
    memories = (
        {name: RobotMemory(entry.memory_key, memory_dir) for name, entry in roster.items()}
        if memory
        else None
    )

    trace_on = trace if trace is not None else trace_enabled_default()

    def log(msg: str) -> None:
        # the trace says all of this and more, so two views of one line is noise
        if verbose and not trace_on:
            _verbose_line(msg)

    status = ui.RunStatus()
    ui.install_logging()
    views, view_factory = _member_views(
        members, trace_on=trace_on, trace_prompt=trace_prompt, status=status
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
                trace=view_factory,
                abort=master,
                flock_name=flock_name,
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
        ],
        run_dir=result.run_dir,
        trace_dropped=result.trace_dropped,
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
    provider: str | None,
    specs: list[Any],
    members: list[str] | None = None,
    flock_name: str | None = None,
    model: str | None,
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
    trace: bool | None = None,
    trace_prompt: bool | None = None,
) -> None:
    from quackd.agent.providers.base import ProviderError
    from quackd.agent.providers.factory import make_provider
    from quackd.flock.runner import FLOCK_TRACE, run_flock
    from quackd.safety import KillSwitch
    from quackd.sim2d.recorder import FrameRecorder
    from quackd.trace import (
        ConsoleTrace,
        Sink,
        fan_out,
        flock_caption,
        prompt_shown_default,
        thinking_limit_default,
        trace_enabled_default,
    )

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
        llm = make_provider(
            provider or DEFAULT_PROVIDER,
            model=model,
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
    prefix_width = max(len(name) for name in [*member_names, FLOCK_TRACE])
    trace_on = trace if trace is not None else trace_enabled_default()

    def log(msg: str) -> None:
        # the trace says all of this and more, so two views of one line is noise
        if verbose and not trace_on:
            _verbose_line(msg)

    views: dict[str, ConsoleTrace] = {}

    def view_for(name: str) -> Sink | None:
        """One view per robot, its name on every line. A shared view would coalesce two
        robots' intents into one line and attribute them to whichever spoke last."""
        if name not in views:
            # a colour per member as well as a name, because three robots moving at once
            # interleave and the eye finds a colour faster than it reads a prefix
            order = member_names.index(name) if name in member_names else -1
            views[name] = ConsoleTrace(
                ui.err_console,
                thinking_chars=thinking_limit_default(),
                prompt=trace_prompt if trace_prompt is not None else prompt_shown_default(),
                prefix=f"{name:<{prefix_width}}  ",
                prefix_style=ui.MEMBER_STYLES[order % len(ui.MEMBER_STYLES)]
                if order >= 0
                else ui.STYLES["key"],
            )
        return fan_out(views[name], status.sink)

    def status_only(_name: str) -> Sink | None:
        """With --no-trace nothing narrates, but the status line still has to say which duck
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
        ("provider", f"{llm.name} ({llm.model or 'the first model it serves'})"),
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
                    provider=llm,
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
                    trace=view_for if trace_on else status_only,
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
        trace_dropped=result.trace_dropped,
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


def _complete_model(ctx: typer.Context, incomplete: str) -> list[tuple[str, str]]:
    """Model ids for the `--provider` already on the line, for shell completion.

    Click parses what is left of the cursor before calling this, so the provider is in
    `ctx.params` by the time `--model` is being completed. It has to be left of the cursor to
    count: `--model <TAB> --provider grok` cannot know, and offers the default's ids instead.
    A provider with no catalogue (`fake`, the local presets) offers nothing, which is correct
    rather than empty: only the server it points at knows what it serves."""
    # folded, because `make_provider` and `list-models` both fold: `--provider GROK` runs, and
    # completion that went silent on it would read as a vendor with no models rather than a
    # shift key.
    provider = str(ctx.params.get("provider") or "fake").lower()
    return [(m.id, m.label) for m in models_for(provider) if m.id.startswith(incomplete)]


DEFAULT_PROVIDER = "fake"
"""The pilot when nothing else names one: no key, no network, a rule that plays the starters.
A registered robot may name its own, and `--provider` beats that."""

_PROVIDER = typer.Option(
    None,
    "--provider",
    "-p",
    help=" · ".join(PROVIDER_NAMES) + f"  (default: {DEFAULT_PROVIDER}, or the robot's own)",
    rich_help_panel="Model",
)
_BASEURL = typer.Option(
    None,
    "--base-url",
    help="OpenAI-compatible server, e.g. http://localhost:8000/v1 (local presets).",
    rich_help_panel="Model",
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
_ROBOT = typer.Option(
    None,
    "--robot",
    "-r",
    help="<adapter>:<backend>, e.g. microduck:sim2d (default) · microduck:mock · "
    "microduck:jsonrpc, or a name from `quackd robot add`, which brings its own address, "
    "token and camera. See `quackd list-adapters` and `quackd robot list`.",
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
_YES = typer.Option(
    False,
    "--yes",
    "-y",
    help="Auto-confirm gated verbs (careful on hardware).",
    rich_help_panel="Task",
)
_MODEL = typer.Option(
    None,
    "--model",
    "-m",
    help="A model id from the provider's catalogue (`quackd list-models`). Omitted: that "
    "vendor's default. Local presets take any id the server serves.",
    autocompletion=_complete_model,
    rich_help_panel="Model",
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
_CAMERA_URL = typer.Option(
    None,
    "--camera-url",
    help="Where frames come from, overriding whatever the robot advertises. An HTTP snapshot "
    "(http://host:9872/snapshot.jpg), or webrtc://host:8443 to pull mediad's video track off a "
    r"Microduck, which is the only camera upstream offers and needs quackd\[microduck-camera]. "
    "Needed when you reach the robot through a tunnel and its own URL is not routable. On a "
    "LeRobot arm it is a USB webcam by its OpenCV index, opencv://0, with ?width, ?height, "
    "?fps, ?fourcc, ?rotation, ?fov, ?name and ?backend=msmf for a Windows camera that lists "
    "and will not open. Find the index with lerobot-find-cameras opencv.",
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
    help="The compact view on stderr: one line per verb plus the executor's notes. The trace "
    "(on by default) shows all of that and more, so this only adds anything with --no-trace.",
    rich_help_panel="Output",
)
_TRACE = typer.Option(
    None,
    "--trace/--no-trace",
    help="Show everything behind the scenes on stderr: the prompt, each observation, what the "
    "model thought and answered, every executor decision, every intent sent to the robot, "
    "every result, tokens and timings. On by default; QUACKD_TRACE=0 turns it off too.",
    rich_help_panel="Output",
)
_TRACE_MCP = typer.Option(
    None,
    "--trace/--no-trace",
    help="Carry a trace of what happened on every tool result, and the uncapped version on "
    "stderr: the verb, every gate that fired, every intent sent to the robot, every result "
    "and the budget. Over MCP the pilot is the client, so its own reasoning is not quackd's "
    "to show. On by default; QUACKD_TRACE=0 turns it off too.",
    rich_help_panel="Output",
)
_TRACE_PROMPT = typer.Option(
    None,
    "--trace-prompt/--no-trace-prompt",
    help="Print the system prompt once at the start of the trace. On by default; "
    "QUACKD_TRACE_PROMPT=0 turns it off too. It is in the transcript either way.",
    rich_help_panel="Output",
)


@app.command(rich_help_panel="Run a duck")
def run(
    duckfile: str | None = _DUCK_ARG,
    goal: str | None = _GOAL,
    provider: str | None = _PROVIDER,
    robot: str | None = _ROBOT,
    robots: str | None = _ROBOTS,
    model: str | None = _MODEL,
    seed: int | None = _SEED,
    dry_run: bool = _DRY,
    max_steps: int | None = _MAXSTEPS,
    runs_dir: str = _RUNS,
    yes: bool = _YES,
    live: bool = _LIVE,
    address: str | None = _ADDR,
    camera_url: str | None = _CAMERA_URL,
    token: str | None = _TOKEN,
    fov_deg: float | None = _FOV,
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
    flock: str | None = _FLOCK,
    memory: bool = _MEMORY,
    memory_dir: str | None = _MEMORY_DIR,
    registry_dir: str | None = _REGISTRY_DIR,
    trace: bool | None = _TRACE,
    trace_prompt: bool | None = _TRACE_PROMPT,
) -> None:
    """Run a .duck file (or a --goal): the LLM picks verbs, quackd enforces the contract."""
    _run_impl(
        duckfile,
        goal,
        provider,
        model,
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
        flock=flock,
        robot=robot,
        robots=robots,
        memory=memory,
        memory_dir=memory_dir,
        registry_dir=registry_dir,
        trace=trace,
        trace_prompt=trace_prompt,
    )


@app.command(rich_help_panel="Run a duck")
def record(
    duckfile: str | None = _DUCK_ARG,
    goal: str | None = _GOAL,
    provider: str | None = _PROVIDER,
    model: str | None = _MODEL,
    seed: int | None = typer.Option(0, "--seed"),
    max_steps: int | None = _MAXSTEPS,
    runs_dir: str = _RUNS,
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
    trace: bool | None = _TRACE,
    trace_prompt: bool | None = _TRACE_PROMPT,
) -> None:
    """Like `run` on sim2d, but always writes a GIF (for READMEs and launches)."""
    if flock is not None and not flock.strip().isdigit():
        _fail(
            "record pins the simulator: --flock takes a count here, not a stored flock",
            hint="quackd run <duck> --flock NAME",
        )
        return
    _run_impl(
        duckfile,
        goal,
        provider,
        model=model,
        seed=seed,
        dry_run=False,
        max_steps=max_steps,
        runs_dir=runs_dir,
        yes=True,
        live=False,
        address=None,
        camera_url=None,
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
        trace=trace,
        trace_prompt=trace_prompt,
    )


# ── trace: replay a finished run ────────────────────────────────────────────────────────


def _resolve_run(run: str | None, runs_dir: str) -> Path:
    """A run directory from what the user typed. In order: a transcript file, a directory, an
    exact name under --runs-dir, a timestamp prefix, the newest whose name contains the text.
    Nothing at all means the newest run, which is what you want after `quackd run` ends."""
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

    A transcript written before the trace existed has `verb` records and no `verb_end`; they
    carry the same fields, so they are shown under the name the renderer knows. `frame` is
    skipped unless asked: one line per camera frame buries everything else."""
    from quackd.trace import TraceEvent

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
        view(TraceEvent(kind, float(rec.get("t") or 0.0), data))
    view.flush()
    return end


_TRACE_RUN = typer.Argument(
    None, help="A run directory, a transcript file, a name or a prefix. Default: the newest."
)


@app.command("trace", rich_help_panel="Run a duck")
def trace_cmd(
    run: str | None = _TRACE_RUN,
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
    """Replay a finished run's transcript as the trace it printed while it ran.

    On stdout, because a replay is what you pipe to a pager or a file, and unaffected by
    QUACKD_TRACE: that switch is about narrating live, and asking for a replay is asking."""
    from quackd.agent.transcript import Transcript
    from quackd.trace import ConsoleTrace, parse_thinking_limit
    from quackd.trace import prompt_shown_default as _prompt_default
    from quackd.trace import thinking_limit_default as _thinking_default

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
        view = ConsoleTrace(
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
            f"trace: {cut} unreadable line(s) skipped, the run was cut while it was writing",
            style="yellow",
            markup=False,
        )
    summary = run_dir / "summary.json"
    if end is None and summary.exists():
        end = json.loads(summary.read_text(encoding="utf-8"))
    if end is None:
        _fail("no run_end: the run did not finish, or is still running")
        return
    usage = end.get("usage") or {}
    if "kicker" in end:  # a flock counts different things, and its summary is the only source
        counters = [f"spotter {end['spotter']}"] if end.get("spotter") else []
        counters += [
            f"kicker {end.get('kicker')}",
            f"auctions {end.get('auctions')}",
            f"bids {end.get('bids')}",
        ]
    else:
        counters = [
            f"steps {int(end.get('steps') or 0)}",
            f"llm calls {int(end.get('llm_calls') or 0)}",
            f"tokens {usage.get('input_tokens', 0)}+{usage.get('output_tokens', 0)}",
        ]
    _print_outcome(
        str(end.get("outcome", "error")),
        str(end.get("reason", "")),
        counters=counters,
        run_dir=run_dir,
        gif_path=gif if (gif := run_dir / "run.gif").exists() else None,
        trace_dropped=int(end.get("trace_dropped") or 0),
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
    camera_url: str | None = _CAMERA_URL,
    token: str | None = _TOKEN,
    registry_dir: str | None = _REGISTRY_DIR,
    as_json: bool = _JSON,
) -> None:
    """Check the environment: keys, optional extras, adapters, upstream assumptions.

    With `--robot X --address Y` it also connects, which is the only way to see what a
    robot actually reports before a run does."""
    from quackd.doctor import collect, render

    if address and not robot:
        _fail("--address needs --robot, so quackd knows what it is connecting to")
        return
    if robot:
        # a registered name is a robot too, and it brings the address you registered it with.
        # Only a name that resolves is substituted: anything else stays exactly as typed, so
        # `doctor --robot nope:x --json` still reports the bad spec inside its one JSON
        # document rather than dying with a line of prose before it.
        from quackd.registry import Registry, RegistryError

        with contextlib.suppress(RegistryError):
            entry = Registry(registry_dir).get_robot(robot)
            if entry is not None:
                robot = entry.key
                where = entry.adapter_kwargs(address=address, camera_url=camera_url, token=token)
                address, camera_url, token = (
                    where["address"],
                    where["camera_url"],
                    where["token"],
                )
    if as_json:
        report = collect(robot, address=address, camera_url=camera_url, token=token)
        print(json.dumps(report.to_dict()))
        raise typer.Exit(code=0 if report.ok else 1)
    ui.install_logging()
    # the probes are the slow part: five local servers at 1.5 s each, and a real robot after
    # them. It used to sit silent for ten seconds with no sign it was doing anything.
    with ui.spinner("checking this machine") as say:
        report = collect(robot, address=address, camera_url=camera_url, token=token, progress=say)
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
    camera_url: str | None = _CAMERA_URL,
    token: str | None = _TOKEN,
    dry_run: bool = _DRY,
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Allow confirm-gated verbs (there is no terminal to ask)."
    ),
    memory: bool = _MEMORY,
    memory_dir: str | None = _MEMORY_DIR,
    trace: bool | None = _TRACE_MCP,
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
            dry_run=dry_run,
            yes=yes,
            memory=memory,
            memory_dir=memory_dir,
            trace=trace,
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


def _entry_rows(entry: Any, *, flocks: list[str]) -> list[tuple[str, Any]]:
    from quackd.adapters.factory import describe

    body: Any
    try:
        manifest = describe(entry.robot_spec)
        body = Text(manifest.summary())
    except Exception as e:  # an adapter whose extra is missing still has a name
        body = Text(str(e), style=ui.STYLES["muted"])
    dash = Text("-", style=ui.STYLES["muted"])
    pilot = (
        Text(" ".join(p for p in (entry.provider, entry.model) if p))
        if entry.provider or entry.model
        else dash
    )
    return [
        ("name", Text(entry.name, style=ui.STYLES["key"])),
        ("robot", Text(entry.key, style=ui.STYLES["accent"])),
        ("body", body),
        ("address", Text(entry.address) if entry.address else dash),
        ("camera", Text(entry.camera_url) if entry.camera_url else dash),
        ("token", Text("set") if entry.token else dash),
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
    camera_url: str | None = _CAMERA_URL,
    token: str | None = _TOKEN,
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="The provider a run uses for this robot when --provider is absent.",
        rich_help_panel="Model",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Its model id. --model on the run beats this, and this beats QUACKD_MODEL.",
        rich_help_panel="Model",
    ),
    note: str | None = typer.Option(None, "--note", help="One line for people: which one is it."),
    registry_dir: str | None = _REGISTRY_DIR,
) -> None:
    """Register a robot under a name, with how to reach it."""
    from quackd.adapters.base import AdapterError
    from quackd.registry import RegistryError, RobotEntry

    try:
        entry = RobotEntry(
            name=name,
            spec=spec,
            address=address,
            camera_url=camera_url,
            token=token,
            provider=provider,
            model=model,
            note=note,
        )
        _registry(registry_dir).add_robot(entry)
    except (RegistryError, AdapterError, ValidationError) as e:
        _registry_fail(_one_line(e))
        return
    where = f" at {entry.address}" if entry.address else ""
    ui.console.print(_ok_line(f"added {entry.name}: {entry.key}{where}"))
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
        "pilot": [Text(" ".join(p for p in (e.provider, e.model) if p)) for e in entries.values()],
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


_CLEARABLE = ("address", "token", "camera-url", "provider", "model", "note")


@robot_app.command("edit")
def robot_edit(
    name: str = _ROBOT_NAME,
    spec: str | None = typer.Option(None, "--spec", help="Move it to another <adapter>:<backend>."),
    address: str | None = _ADDR,
    camera_url: str | None = _CAMERA_URL,
    token: str | None = _TOKEN,
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="The provider a run uses for this robot when --provider is absent.",
        rich_help_panel="Model",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Its model id, used with the provider above.",
        rich_help_panel="Model",
    ),
    note: str | None = typer.Option(None, "--note", help="One line for people."),
    clear: list[str] = typer.Option(
        [],
        "--clear",
        help=f"Empty one field: {', '.join(_CLEARABLE)}. Repeatable.",
    ),
    registry_dir: str | None = _REGISTRY_DIR,
) -> None:
    """Change what a registered robot is or where it is."""
    from quackd.adapters.base import AdapterError
    from quackd.registry import RegistryError

    given = {
        "spec": spec,
        "address": address,
        "camera_url": camera_url,
        "token": token,
        "provider": provider,
        "model": model,
        "note": note,
    }
    changes: dict[str, str | None] = {k: v for k, v in given.items() if v is not None}
    for field in clear:
        key = field.strip().lower().replace("-", "_")
        if key not in {c.replace("-", "_") for c in _CLEARABLE}:
            _fail(f"--clear {field}: empty one of {', '.join(_CLEARABLE)}")
            return
        if key in changes:
            _fail(f"--{key.replace('_', '-')} and --clear {field} contradict each other")
            return
        changes[key] = None
    if not changes:
        _fail("nothing to change: give a field to set, or --clear FIELD")
        return
    try:
        _registry(registry_dir).update_robot(name, changes)
    except (RegistryError, AdapterError, ValidationError) as e:
        _registry_fail(_one_line(e))
        return
    touched = ", ".join(sorted(key.replace("_", "-") for key in changes))
    ui.console.print(_ok_line(f"updated {name}: {touched}"))


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
