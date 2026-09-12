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
import sys
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from rich.markup import escape
from rich.text import Text

from quackd import __version__, ui

app = typer.Typer(
    name="quackd",
    # the emoji only where the stream can carry it: a cp1252 pipe on Windows renders them as
    # `??`, and the front door is the worst place to look broken
    help="Give your small robot a brain. Any LLM, one .duck file."
    + (" 🦆🧠" if ui.glyphs_for(ui.console) is ui.UNICODE else ""),
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
    # A crash must not print this process's local variables: they hold an API key, a robot's
    # address and its bridge token. `main` installs a Rich traceback without them instead.
    pretty_exceptions_enable=False,
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
    """quackd — pilot a small robot (real or simulated) with any LLM."""
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
)

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


def _print_outcome(
    outcome: str,
    reason: str,
    *,
    detail: str,
    run_dir: Path | str,
    gif_path: Path | str | None = None,
    trace_dropped: int = 0,
) -> None:
    """The closing lines of a run: the verdict, one line of counters, where it all went.

    `quackd trace` prints them from the transcript, so a replay ends exactly the way the run
    itself did rather than in a second dialect somebody has to keep in step. `detail` is the
    counter line because a flock counts different things than a solo run does."""
    colour = {"success": "green", "failure": "red", "budget": "yellow", "aborted": "red"}.get(
        outcome, "red"
    )
    ui.console.print(f"[{colour}]{outcome.upper()}[/{colour}] — {escape(reason)}")
    if detail:
        ui.console.print(detail)
    ui.console.print(f"run dir: {run_dir}" + (f" · gif: {gif_path}" if gif_path else ""))
    if trace_dropped:
        # a console that raised on every event produced a silent trace and no sign of it
        ui.err_console.print(
            f"trace: {trace_dropped} line(s) could not be shown (the console raised); "
            "transcript.jsonl has them",
            style="yellow",
            markup=False,
        )


def _fail(msg: str, code: int = 1, *, hint: str | None = None) -> None:
    """One line saying what went wrong, and one dim line saying where to look next.

    The message routinely names an extra (`quackd[anthropic]`) or a model's own brackets, so
    it travels as text rather than as markup Rich would eat."""
    ui.err_console.print(ui.fail_line(msg, hint=hint))
    raise typer.Exit(code=code)


def _robot_specs(robot: str | None, robots: str | None, duck: Any) -> list:
    """The robots a command talks about: --robots, else --robot, else the duck's own
    `robots:` default, else the Microduck simulator."""
    from quackd.adapters.factory import RobotSpec, parse_robot_spec, parse_robots, resolve_robot

    if robots:
        return parse_robots(robots)
    default = duck.frontmatter.robots if duck is not None else None
    if isinstance(default, dict):
        if robot:
            return [resolve_robot(robot)]
        # the member names become the robot ids, as `--robots name=spec` would make them
        specs = []
        for name, text in default.items():
            parsed = parse_robot_spec(text)
            specs.append(RobotSpec(parsed.adapter, parsed.backend, name))
        return specs
    return [resolve_robot(robot, duck_default=default)]


# ── validate ────────────────────────────────────────────────────────────────────────────


@app.command()
def validate(
    duckfiles: list[str] = typer.Argument(..., help=".duck files, globs, or bundled names."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only print failures."),
    as_json: bool = _JSON,
    robot: list[str] | None = typer.Option(
        None,
        "--robot",
        "-r",
        help="Check the files against this robot's manifest (<adapter>:<backend>; repeatable).",
    ),
    robots: str | None = typer.Option(
        None, "--robots", help="Check against a fleet: name=<adapter>:<backend>,..."
    ),
) -> None:
    """Validate .duck files against the spec and a robot's verbs. Exits 1 on any failure."""
    from quackd.adapters.base import AdapterError
    from quackd.adapters.factory import describe, parse_robot_spec
    from quackd.duckfile.parser import DuckParseError, load_duck
    from quackd.duckfile.validate import validate_duck
    from quackd.verbs.registry import default_registry

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
                    [parse_robot_spec(r) for r in robot]
                    if robot
                    else _robot_specs(None, robots, duck)
                )
            elif duck.frontmatter.robots is not None:
                specs = _robot_specs(None, None, duck)
            else:
                specs = []
            manifests = [describe(spec) for spec in specs]
        except AdapterError as e:
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
    verbs = "—" if row["verbs"] is None else str(row["verbs"])

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

    return [Text(row["file"]), Text(row["name"] or "—"), verbs, ui.Deferred(result)]


# ── list-verbs ──────────────────────────────────────────────────────────────────────────

_SAFETY_STYLE = {"safe": "ok", "confirm": "warn", "dangerous": "fail"}


@app.command("list-verbs")
def list_verbs(
    robot: str | None = typer.Option(
        None, "--robot", "-r", help="A robot's vocabulary (<adapter>:<backend>); default Microduck."
    ),
    as_json: bool = _JSON,
) -> None:
    """List every verb a robot provides, with params and safety class."""
    from quackd.adapters.base import AdapterError
    from quackd.adapters.factory import parse_robot_spec, registry_for
    from quackd.verbs.registry import default_registry

    try:
        registry = registry_for(parse_robot_spec(robot)) if robot else default_registry()
    except AdapterError as e:
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


@app.command("list-adapters")
def list_adapters_cmd(as_json: bool = _JSON) -> None:
    """List the robot adapters this build knows, their backends and status."""
    from quackd.adapters.factory import list_adapters

    rows = list_adapters()
    if as_json:
        for row in rows:
            print(json.dumps(row))
        return
    ui.console.print(ui.adapters_table(rows))


# ── run / record ────────────────────────────────────────────────────────────────────────


def _confirm_prompt(name: str, params: dict[str, Any]) -> bool:
    return typer.confirm(f"⚠️  run {name}({params})?", default=False)


def _acknowledge_prompt(why: str) -> bool:
    """Asked once, before anything moves, when the human is the only safety left."""
    ui.err_console.print(f"[yellow]⚠️  {why}[/yellow]")
    return typer.confirm("Are you watching the robot right now?", default=False)


def _run_impl(
    duckfile: str | None,
    goal: str | None,
    provider: str,
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
    flock: int | None = None,
    *,
    robot: str | None = None,
    robots: str | None = None,
    memory: bool = True,
    memory_dir: str | None = None,
    trace: bool | None = None,
    trace_prompt: bool | None = None,
) -> None:
    from quackd.adapters.factory import describe, make_adapter, registry_for
    from quackd.agent.loop import RunConfig, run_duck
    from quackd.agent.providers.base import ProviderError
    from quackd.agent.providers.factory import make_provider
    from quackd.duckfile.parser import DuckParseError, duck_from_goal, load_duck
    from quackd.duckfile.validate import validate_duck
    from quackd.perception import detector_for
    from quackd.safety import KillSwitch, allow_all
    from quackd.trace import (
        ConsoleTrace,
        prompt_shown_default,
        thinking_limit_default,
        trace_enabled_default,
    )
    from quackd.transport.base import TransportError

    if (duckfile is None) == (goal is None):
        _fail('give either a .duck file (or bundled name) or --goal "...", not both')
        return
    try:
        duck = load_duck(duckfile) if duckfile is not None else None
        specs = _robot_specs(robot, robots, duck)
        spec = specs[0]
        if goal is not None:
            safe = [v.name for v in registry_for(spec).verbs() if v.safety_class == "safe"]
            duck = duck_from_goal(goal, safe)
        assert duck is not None
        # Refuse before connecting, with the validator's words. `serve-mcp` has always done
        # this; `run` never did, and reached the loop's tool_schemas and died on a raw
        # VerbNotFound with the robot already connected and a run directory already made.
        manifests = [describe(s) for s in specs]
        problems = validate_duck(duck, manifests)
    except (DuckParseError, TransportError) as e:
        _fail(str(e))
        return
    if problems:
        _fail(
            f"{duck.name} cannot run on {', '.join(s.key for s in specs)}: "
            + "; ".join(p.message for p in problems)
        )
        return
    if flock is not None and not 2 <= flock <= 4:
        _fail("a flock needs 2 to 4 ducks (drop --flock for a single run)")
        return
    if flock is not None or duck.frontmatter.flock is not None:
        _run_flock_impl(
            duck,
            provider=provider,
            specs=specs,
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
            n_override=flock,
            max_steps=max_steps,
            trace=trace,
            trace_prompt=trace_prompt,
        )
        return
    try:
        llm = make_provider(
            provider,
            model=model,
            duck_name=duck.name,
            goal=goal,
            base_url=base_url,
            api_key=api_key,
            vision=vision,
        )
        duck_transport = make_adapter(
            spec,
            seed=seed,
            address=address,
            live=live,
            camera_url=camera_url,
            token=token,
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

    def log(msg: str) -> None:
        # the compact view: one line per verb and the executor's notes. The trace shows all
        # of that and more, so with it on this prints nothing rather than every verb twice.
        if verbose and console_trace is None:
            _verbose_line(msg)

    robot_memory = None
    if memory:
        from quackd.memory import RobotMemory

        # keyed by adapter:backend, so a simulated duck never inherits a real one's notes
        robot_memory = RobotMemory(spec.key, memory_dir)
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
        trace=console_trace,
    )
    ui.console.print(
        f"🦆 [bold]{duck.name}[/bold] · provider=[cyan]{llm.name}[/cyan] "
        f"({llm.model or 'model: first served'}) · "
        f"robot=[cyan]{spec.key}[/cyan]"
        + (f" · seed={seed}" if seed is not None else "")
        + (" · [yellow]DRY RUN[/yellow]" if dry_run else "")
    )
    if robot_memory is not None:
        m = robot_memory.summary()
        ui.console.print(
            f"[dim]memory: {m['notes']} notes, {m['episodes']} earlier runs "
            f"({m['path']}) · --no-memory to run fresh[/dim]"
        )
    ui.console.print("[dim]Ctrl-C or q stops the duck. Press it twice to quit at once.[/dim]")

    def killed(msg: str) -> None:
        """Always printed, unlike `log`, which is --verbose only. Someone who has just hit
        Ctrl-C on a walking robot needs to see that it registered."""
        ui.err_console.print(f"[yellow]{msg}[/yellow]")

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
        result = asyncio.run(main())
    except (TransportError, ProviderError) as e:
        # the trace has already shown the call that failed; this is the one-line verdict
        _fail(str(e))
        return
    if recorder is not None:
        gif_path = recorder.save_gif(result.run_dir / "run.gif")
        result.gif_path = gif_path
    _print_outcome(
        result.outcome,
        result.reason,
        detail=(
            f"steps={result.steps} llm_calls={result.llm_calls} "
            f"tokens={result.usage.input_tokens}+{result.usage.output_tokens}"
        ),
        run_dir=result.run_dir,
        gif_path=result.gif_path,
        trace_dropped=result.trace_dropped,
    )
    if result.outcome != "success":
        raise typer.Exit(code=1)


def _run_flock_impl(
    duck: Any,
    *,
    provider: str,
    specs: list[Any],
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
    robots = {spec.name: spec.key for spec in specs if spec.name} or None
    try:
        llm = make_provider(
            provider,
            model=model,
            duck_name=duck.name,
            goal=goal,
            base_url=base_url,
            api_key=api_key,
            vision=vision,
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
            views[name] = ConsoleTrace(
                ui.err_console,
                thinking_chars=thinking_limit_default(),
                prompt=trace_prompt if trace_prompt is not None else prompt_shown_default(),
                prefix=f"{name:<{prefix_width}}  ",
            )
        return views[name]

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

    ui.console.print(
        f"🦆x{count} [bold]{duck.name}[/bold] · provider=[cyan]{llm.name}[/cyan] "
        f"({llm.model or 'model: first served'}) · flock (sim2d, EXPERIMENTAL)"
        + (f" · seed={seed}" if seed is not None else "")
        + (" · [yellow]DRY RUN[/yellow]" if dry_run else "")
    )
    ui.console.print("[dim]Ctrl-C or q stops every duck.[/dim]")
    try:
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
                trace=view_for if trace_on else None,
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
        result.gif_path = holder["rec"].save_gif(result.run_dir / "run.gif")
    spotter = f"spotter={result.spotter} " if result.spotter else ""
    _print_outcome(
        result.outcome,
        result.reason,
        detail=(
            f"{spotter}kicker={result.kicker} auctions={result.auctions} bids={result.bids} "
            f"ball moved {result.ball_displacement_m:.2f} m in {result.sim_elapsed_s:.1f}s sim"
        ),
        run_dir=result.run_dir,
        gif_path=result.gif_path,
        trace_dropped=result.trace_dropped,
    )
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
)
_GIFSIZE = typer.Option(
    256,
    "--gif-size",
    min=64,
    max=1024,  # `sim3d.scene.OFFSCREEN_PX`; spelled here because cli.py must not import sim3d
    help="Simulators: pixel size of each GIF pane, 64 to 1024.",
)
_FLOCK = typer.Option(
    None,
    "--flock",
    help="EXPERIMENTAL: run N cooperating ducks (2-4) in sim2d. Overrides the file's flock block.",
)
_MEMORY = typer.Option(
    True,
    "--memory/--no-memory",
    help="Carry notes and run outcomes between runs of the same robot (see `quackd memory`).",
)
_MEMORY_DIR = typer.Option(
    None,
    "--memory-dir",
    help="Where memory files live (default: $QUACKD_MEMORY_DIR or ~/.quackd/memory).",
)
_PROVIDER = typer.Option(
    "fake",
    "--provider",
    "-p",
    help="fake · anthropic · openai · gemini · grok · local · ollama · vllm · llamacpp · lmstudio",
)
_BASEURL = typer.Option(
    None,
    "--base-url",
    help="OpenAI-compatible server, e.g. http://localhost:8000/v1 (local presets).",
)
_APIKEY = typer.Option(None, "--api-key", help="API key override (local servers do not need one).")
_VISION = typer.Option(
    None,
    "--vision/--no-vision",
    help="Send camera frames to the model (default: on for cloud, off for local).",
)
_ROBOT = typer.Option(
    None,
    "--robot",
    "-r",
    help="<adapter>:<backend>, e.g. microduck:sim2d (default) · microduck:mock · "
    "microduck:jsonrpc. See `quackd list-adapters`.",
)
_ROBOTS = typer.Option(
    None,
    "--robots",
    help="A flock or fleet: name=<adapter>:<backend>,... (simulator only for flocks).",
)
_MODEL = typer.Option(None, "--model", "-m", help="Override the provider's model.")
_SEED = typer.Option(None, "--seed", help="Simulator seed (deterministic runs).")
_DRY = typer.Option(False, "--dry-run", help="Print every intent, send nothing.")
_MAXSTEPS = typer.Option(None, "--max-steps", help="Override the duck's max_steps budget.")
_RUNS = typer.Option("runs", "--runs-dir", help="Where run directories go.")
_YES = typer.Option(False, "--yes", "-y", help="Auto-confirm gated verbs (careful on hardware).")
_LIVE = typer.Option(
    False,
    "--live",
    help="Simulators: watch the run in real time. sim2d opens a pygame window (needs "
    r"quackd\[live]); mujoco opens MuJoCo's own viewer.",
)
_ADDR = typer.Option(None, "--address", help="jsonrpc: unix:///run/robotd.sock or tcp://host:port")
_TOKEN = typer.Option(
    None,
    "--token",
    help="The bridge token for a robot that wants one. The Open Duck's installer writes one "
    "on the robot and QUACKD_DUCK_TOKEN carries it when the flag is absent. The ToddlerBot's "
    "daemon has no installer and reads QUACKD_TODDLERBOT_TOKEN instead.",
)
_CAMERA_URL = typer.Option(
    None,
    "--camera-url",
    help="Where frames come from, overriding whatever the robot advertises. An HTTP snapshot "
    "(http://host:9872/snapshot.jpg), or webrtc://host:8443 to pull mediad's video track off a "
    r"Microduck, which is the only camera upstream offers and needs quackd\[microduck-camera]. "
    "Needed when you reach the robot through a tunnel and its own URL is not routable.",
)
_FOV = typer.Option(
    None,
    "--fov-deg",
    help="Horizontal field of view of the camera actually on your robot, in degrees. The "
    "default is the simulator's 90; a Pi Camera Module 2 is about 62. Getting it wrong "
    "scales every bearing and distance, so detections say so until you set it.",
)
_VERBOSE = typer.Option(
    False,
    "--verbose",
    "-v",
    help="The compact view on stderr: one line per verb plus the executor's notes. The trace "
    "(on by default) shows all of that and more, so this only adds anything with --no-trace.",
)
_TRACE = typer.Option(
    None,
    "--trace/--no-trace",
    help="Show everything behind the scenes on stderr: the prompt, each observation, what the "
    "model thought and answered, every executor decision, every intent sent to the robot, "
    "every result, tokens and timings. On by default; QUACKD_TRACE=0 turns it off too.",
)
_TRACE_MCP = typer.Option(
    None,
    "--trace/--no-trace",
    help="Carry a trace of what happened on every tool result, and the uncapped version on "
    "stderr: the verb, every gate that fired, every intent sent to the robot, every result "
    "and the budget. Over MCP the pilot is the client, so its own reasoning is not quackd's "
    "to show. On by default; QUACKD_TRACE=0 turns it off too.",
)
_TRACE_PROMPT = typer.Option(
    None,
    "--trace-prompt/--no-trace-prompt",
    help="Print the system prompt once at the start of the trace. On by default; "
    "QUACKD_TRACE_PROMPT=0 turns it off too. It is in the transcript either way.",
)


@app.command()
def run(
    duckfile: str | None = _DUCK_ARG,
    goal: str | None = _GOAL,
    provider: str = _PROVIDER,
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
        True, "--gif/--no-gif", help="Simulators: write run.gif into the run dir."
    ),
    gif_size: int = _GIFSIZE,
    verbose: bool = _VERBOSE,
    base_url: str | None = _BASEURL,
    api_key: str | None = _APIKEY,
    vision: bool | None = _VISION,
    flock: int | None = _FLOCK,
    memory: bool = _MEMORY,
    memory_dir: str | None = _MEMORY_DIR,
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
        flock=flock,
        robot=robot,
        robots=robots,
        memory=memory,
        memory_dir=memory_dir,
        trace=trace,
        trace_prompt=trace_prompt,
    )


@app.command()
def record(
    duckfile: str | None = _DUCK_ARG,
    goal: str | None = _GOAL,
    provider: str = _PROVIDER,
    model: str | None = _MODEL,
    seed: int | None = typer.Option(0, "--seed"),
    max_steps: int | None = _MAXSTEPS,
    runs_dir: str = _RUNS,
    gif_size: int = _GIFSIZE,
    verbose: bool = _VERBOSE,
    base_url: str | None = _BASEURL,
    api_key: str | None = _APIKEY,
    vision: bool | None = _VISION,
    flock: int | None = _FLOCK,
    trace: bool | None = _TRACE,
    trace_prompt: bool | None = _TRACE_PROMPT,
) -> None:
    """Like `run` on sim2d, but always writes a GIF (for READMEs and launches)."""
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


@app.command("trace")
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
    for path in transcripts:
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
        spotter = f"spotter={end['spotter']} " if end.get("spotter") else ""
        detail = (
            f"{spotter}kicker={end.get('kicker')} auctions={end.get('auctions')} "
            f"bids={end.get('bids')}"
        )
    else:
        detail = (
            f"steps={int(end.get('steps') or 0)} llm_calls={int(end.get('llm_calls') or 0)} "
            f"tokens={usage.get('input_tokens', 0)}+{usage.get('output_tokens', 0)}"
        )
    _print_outcome(
        str(end.get("outcome", "error")),
        str(end.get("reason", "")),
        detail=detail,
        run_dir=run_dir,
        gif_path=gif if (gif := run_dir / "run.gif").exists() else None,
        trace_dropped=int(end.get("trace_dropped") or 0),
    )


# ── doctor / serve-mcp ──────────────────────────────────────────────────────────────────


@app.command()
def doctor(
    robot: str | None = typer.Option(
        None, "--robot", "-r", help="Also show one robot's manifest (<adapter>:<backend>)."
    ),
    address: str | None = typer.Option(
        None,
        "--address",
        help="With --robot, connect to a real robot and report what it says about itself.",
    ),
    camera_url: str | None = _CAMERA_URL,
    token: str | None = _TOKEN,
) -> None:
    """Check the environment: keys, optional extras, adapters, upstream assumptions.

    With `--robot X --address Y` it also connects, which is the only way to see what a
    robot actually reports before a run does."""
    from quackd.doctor import run_doctor

    if address and not robot:
        _fail("--address needs --robot, so quackd knows what it is connecting to")
        return
    ok = run_doctor(ui.console, robot=robot, address=address, camera_url=camera_url, token=token)
    if not ok:
        raise typer.Exit(code=1)


@app.command("serve-mcp")
def serve_mcp(
    robot: str | None = _ROBOT,
    robots: str | None = typer.Option(
        None,
        "--robots",
        help="A fleet: name=<adapter>:<backend>,... (eight robot_* tools, one executor each).",
    ),
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
    """Expose the robot as MCP tools over stdio (Claude Code / Claude Desktop)."""
    from quackd.adapters.base import AdapterError
    from quackd.mcp_server import serve

    try:
        serve(
            robot=robot,
            robots=robots,
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
    except AdapterError as e:
        _fail(str(e))


# ── memory ──────────────────────────────────────────────────────────────────────────────

memory_app = typer.Typer(
    name="memory",
    help="What a robot remembers between runs: notes the pilot saved, and how runs ended.",
    no_args_is_help=True,
)
app.add_typer(memory_app, name="memory")


def _memory_for(robot: str | None, memory_dir: str | None) -> Any:
    from quackd.adapters.base import AdapterError
    from quackd.adapters.factory import resolve_robot
    from quackd.memory import RobotMemory

    try:
        spec = resolve_robot(robot)
    except AdapterError as e:  # every other --robot command answers in one line, not a traceback
        _fail(str(e))
    return RobotMemory(spec.key, memory_dir)


@memory_app.command("show")
def memory_show(
    robot: str | None = _ROBOT,
    memory_dir: str | None = _MEMORY_DIR,
    raw: bool = typer.Option(False, "--raw", help="Print the JSONL file as is."),
) -> None:
    """Print what one robot remembers (default: the Microduck simulator)."""
    mem = _memory_for(robot, memory_dir)
    if raw:
        if mem.path.exists():
            # "as is" means as is: a note saying "the ball is [bold]behind[/bold] the sofa"
            # is markup Rich would silently eat, and an unpaired tag would raise.
            print(mem.path.read_text(encoding="utf-8"), end="")
        return
    info = mem.summary()
    ui.console.print(
        f"[bold]{info['robot']}[/bold] · {info['notes']} notes · {info['episodes']} runs · "
        f"[dim]{info['path']}[/dim]"
    )
    text = mem.recall(max_notes=50, max_episodes=10)
    ui.console.print(escape(text) if text else "[dim](nothing remembered yet)[/dim]")


@memory_app.command("add")
def memory_add(
    text: str = typer.Argument(..., help="One short fact, e.g. 'the ball lives by the sofa'."),
    robot: str | None = _ROBOT,
    memory_dir: str | None = _MEMORY_DIR,
    tag: list[str] = typer.Option([], "--tag", help="Optional label(s)."),
) -> None:
    """Save a note by hand, the same way the pilot's `remember` does."""
    mem = _memory_for(robot, memory_dir)
    entry = mem.remember(text, tags=tag)
    ui.console.print(f"remembered for [bold]{mem.robot_key}[/bold]: {escape(entry.text)}")


@memory_app.command("clear")
def memory_clear(
    robot: str | None = _ROBOT,
    memory_dir: str | None = _MEMORY_DIR,
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask."),
) -> None:
    """Forget everything one robot remembers (deletes its memory file)."""
    mem = _memory_for(robot, memory_dir)
    n = len(mem.entries())
    if n == 0:
        ui.console.print(f"[dim]{mem.robot_key}: nothing to forget[/dim]")
        return
    if not yes and not typer.confirm(f"forget {n} entries for {mem.robot_key}?"):
        raise typer.Exit()
    mem.clear()
    ui.console.print(f"forgot {n} entries for [bold]{mem.robot_key}[/bold]")


# ── lan (quackd[lan]) ───────────────────────────────────────────────────────────────────


@app.command()
def discover(
    timeout: float = typer.Option(3.0, "--timeout", help="Seconds to listen for answers."),
    as_json: bool = typer.Option(False, "--json", help="One JSON object per robot."),
) -> None:
    r"""List the quackd robots answering on the LAN (zeroconf, needs quackd\[lan])."""
    from rich.table import Table

    from quackd.lan import LanNotInstalled
    from quackd.lan import discover as lan_discover

    try:
        robots = lan_discover.discover(timeout)
    except LanNotInstalled as e:
        _fail(str(e))
    if as_json:
        for robot in robots:
            print(json.dumps(robot.row()))
        return
    if not robots:
        ui.console.print(f"[dim]no quackd robots answered in {timeout:g} s[/dim]")
        return
    t = Table(title=f"quackd robots on the LAN ({len(robots)})")
    for column in ("manifest id", "adapter", "model", "embodiment", "verbs", "address", "digest"):
        t.add_column(column)
    for robot in robots:
        t.add_row(
            robot.manifest_id,
            robot.adapter,
            robot.model,
            robot.embodiment,
            str(robot.n_verbs),
            ", ".join(robot.addresses) or robot.host,
            robot.digest,
        )
    ui.console.print(t)


@app.command()
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
        f"announcing {ann.record.name} ({manifest.summary()}) at "
        f"{', '.join(ann.record.addresses)} · digest {manifest.digest()}"
    )
    try:
        if for_s is None:
            ui.console.print("[dim]Ctrl-C to withdraw[/dim]")
            while True:
                time.sleep(1.0)
        else:
            time.sleep(for_s)
    except KeyboardInterrupt:
        pass
    finally:
        ann.close()
        ui.console.print("withdrawn")


def _leave_quietly() -> None:
    """Stop, with nothing further to say.

    Python flushes stdout as it exits, which raises a second time on a pipe that has already
    gone, so stdout is pointed at the void before leaving."""
    with contextlib.suppress(Exception):
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    raise SystemExit(0)


def main() -> None:
    """The console entry point: `app()`, and the two things a command that prints for a
    living owes its terminal.

    A traceback must not spill this process's locals, because they hold an API key, a
    robot's address and its bridge token. And a reader is allowed to walk away: `quackd
    list-verbs | head` closes the pipe halfway down the table, and Python's answer to that
    is a second wall of text about a broken pipe on top of the output that was asked for."""
    from rich.traceback import install

    install(console=ui.err_console, show_locals=False, suppress=[typer])
    try:
        app()
    except BrokenPipeError:
        _leave_quietly()
    except OSError as e:  # Windows raises EINVAL rather than EPIPE on a pipe that has gone
        if e.errno not in (errno.EPIPE, errno.EINVAL):
            raise
        _leave_quietly()


if __name__ == "__main__":
    main()
