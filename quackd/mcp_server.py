"""`quackd serve-mcp`: a robot, or a fleet of them, as MCP tools over stdio.

This is the second wow-demo — "I asked Claude to make the duck patrol my desk" — and it
goes through the *same* `Executor` as `.duck` runs, so allowlists, confirm gates, budgets
and the heartbeat apply to an interactive session too. Since 0.4 one server can front
several robots (`--robots duck=microduck:sim2d,arm=lerobot:mock`): eight `robot_*`
tools take a robot name and every robot has its own executor, budget, heartbeat and
contract. stdout is the wire, and every log line goes to stderr.

The model here is the client, so what it thinks never reaches this process. What quackd can
see it narrates (`quackd.trace`): each call that reaches an executor comes back with a
`trace` list saying which gates fired, which intents went to the robot, what came back and
how long it took, and the same lines go to stderr. `--no-trace` or `QUACKD_TRACE=0` turns
both off.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver import Image

from quackd import __version__
from quackd.adapters.base import adapter_name, backend_name
from quackd.adapters.manifest import RobotManifest
from quackd.agent.transcript import png_bytes
from quackd.duckfile.parser import DuckParseError, load_duck
from quackd.duckfile.schema import Budgets, DuckFile
from quackd.duckfile.validate import validate_duck
from quackd.memory import RobotMemory
from quackd.perception import detector_for
from quackd.perception.base import Detector
from quackd.safety import (
    Aborted,
    Budget,
    BudgetExceeded,
    ConfirmDenied,
    Executor,
    Heartbeat,
    VerbNotAllowed,
    allow_all,
    deny_all,
)
from quackd.trace import (
    Sink,
    TraceEvent,
    Tracer,
    call_lines,
    cap_lines,
    capture_sink,
    capturing,
    render_lines,
    trace_enabled_default,
    unless_capturing,
)
from quackd.transport.base import DuckTransport
from quackd.verbs.registry import (
    Verb,
    VerbRegistry,
    VerbResult,
    default_registry,
    registry_from_manifest,
)

log = logging.getLogger("quackd.mcp")

TOOL_NAMES = (
    "robot_list",
    "robot_list_verbs",
    "robot_run_verb",
    "robot_observe",
    "robot_say",
    "robot_load_duckfile",
    "robot_recall",
    "robot_remember",
)
"""Every tool the server registers, in this order; `docs/mcp.md` must list each one."""

INSTRUCTIONS = """You are piloting one robot through quackd: {names}, which is {blurb}.
Call robot_list_verbs first: the verbs come from that robot's own manifest, so what it can
do is what it lists and nothing else. Every action is a *verb*; the executor enforces an
allowlist, budgets and confirmation gates, so a refused call is a rule, not a bug. Prefer
composite verbs (search_scan, go_to) over micro-managing velocities. Load a .duck file with
robot_load_duckfile(path) to adopt a task contract; then follow its body as your
instructions.{memory} Call robot_run_verb(verb="stop") if anything looks wrong."""

SOLO_MEMORY = """ Call robot_recall early: it is what this robot learned in earlier
sessions, and robot_remember(text) keeps one short fact for the next one."""

FLEET_INSTRUCTIONS = """You are piloting {n} robot(s) through quackd: {names}.
Call robot_list first, then robot_list_verbs(robot) for each body you will use: verbs come
from each robot's own manifest, so they differ per robot. Every action is a *verb*; each
robot's executor enforces its own allowlist, budgets and confirmation gates, so a refused
call is a rule, not a bug. Prefer composite verbs (search_scan, go_to) over micro-managing
velocities. Load a .duck file with robot_load_duckfile(path, robot) to adopt a task
contract on one robot; then follow its body as your instructions.{memory} Without a robot
argument a tool acts on the default, {default}.
Call robot_run_verb(verb="stop", robot=...) if anything looks wrong."""

FLEET_MEMORY = """ robot_recall(robot) is what that robot learned in earlier sessions;
robot_remember(text, robot) keeps one short fact for the next one."""


def _prefixed(emit: Callable[..., None], name: str) -> Callable[[str], None]:
    """Log lines from a fleet say which robot they are about."""

    def log_line(message: str) -> None:
        emit("%s: %s", name, message)

    return log_line


def _stderr_view(name: str) -> Sink:
    """Events that belong to no tool call, rendered to stderr the moment they happen.

    A call's own lines are logged as one block when it ends (`RobotSession._call`), because
    one coalescing view shared by concurrent calls interleaved their bursts and attributed
    one call's intents to another. What is left is the heartbeat's: its note and the stop it
    sends when the link dies. Those must render immediately and one at a time, because a
    buffered burst is only flushed by the next event, and after the heartbeat fails there is
    no next event: the emergency stop was the one line that never reached the log."""

    def write(event: TraceEvent) -> None:
        for text, _style in render_lines(event, prompt=False):
            log.info("%s: %s", name, text)

    return unless_capturing(write)


def _stash_frames(session: RobotSession) -> Callable[[Any, str], None]:
    """The executor's `on_frame` hook: keep the frame `observe` captured for `robot_observe`."""

    def on_frame(img: Any, _cause: str) -> None:
        session.last_frame = img

    return on_frame


@dataclass
class RobotSession:
    """One robot behind the server: its own executor, budget, heartbeat and contract."""

    name: str
    transport: DuckTransport
    registry: VerbRegistry
    executor: Executor
    heartbeat: Heartbeat
    detector: Detector | None = None
    duck: DuckFile | None = None
    manifest: RobotManifest | None = None
    """Set on connect when the transport is an adapter that describes itself."""
    frames: int = 0
    calls: int = 0
    log_lines: list[str] = field(default_factory=list)
    last_frame: Any = None
    """The most recent frame an `observe` captured, so `robot_observe` can return it."""
    explicit_registry: bool = False
    """A caller-supplied registry is kept as is; otherwise the manifest builds one."""
    memory: RobotMemory | None = None
    """What this robot keeps between sessions (`quackd memory`). None = off."""
    tracer: Tracer | None = None
    """Narrates this robot's calls to stderr and into each result's `trace`. None = off."""

    def _gate(self, name: str, gate: str, reason: str) -> None:
        """A refusal the session makes before the executor sees the call, told the same way."""
        if self.tracer is not None:
            self.tracer.emit("gate", name=name, gate=gate, outcome="refused", reason=reason)

    async def _call(
        self, tool: str, args: dict[str, Any], fn: Callable[[], Awaitable[dict[str, Any]]]
    ) -> dict[str, Any]:
        """One tool call, narrated. The SDK runs every call as its own task, and `capturing`
        is a context variable, so two calls on one robot never see each other's events.

        stderr gets the call as one block when it ends, rather than line by line as they
        happen: two concurrent calls sharing one coalescing view merged their bursts, and a
        `verb_end` from one split the other's at an arbitrary point. The result's `trace` is
        the same lines, capped."""
        if self.tracer is None:
            return await fn()
        with capturing() as events:
            started = time.perf_counter()
            robot_started = self.executor._robot_now()
            self.tracer.emit("tool_call", tool=tool, robot=self.name, **args)
            payload = await fn()
            budget = self.executor.budget
            clocks: dict[str, Any] = {}
            robot_now = self.executor._robot_now()
            if robot_started is not None and robot_now is not None:
                clocks["transport_s"] = round(robot_now - robot_started, 3)
                if (label := self.executor._clock()) is not None:
                    clocks["clock"] = label
            self.tracer.emit(
                "tool_result",
                tool=tool,
                ok=bool(payload.get("ok")),
                summary=payload.get("summary"),
                elapsed_s=round(time.perf_counter() - started, 3),
                budget=budget.status() if budget is not None else None,
                **clocks,
            )
        lines = call_lines(events)
        for line in lines:
            log.info("%s: %s", self.name, line)
        payload["trace"] = cap_lines(lines)
        return payload

    def shown_name(self, verb: Verb) -> str:
        """The name a client sees: the loaded contract's own spelling when it used an alias."""
        if self.duck is not None:
            for spelled in self.executor.allowed:
                if spelled != verb.name and self.registry.canonical(spelled) == verb.name:
                    return spelled
        return verb.name

    def adopt(self, duck: DuckFile) -> None:
        """Take on a task contract. Loading a second one never refunds the first.

        `robot_load_duckfile` is a tool the *model* holds, so a fresh `Budget` here was the
        way out of one: a pilot that had spent its steps, or been refused a verb, could load
        a wider duck and start counting from zero. The limits become the new contract's. The
        steps, the llm calls, the clock and the failure tallies stay the session's."""
        # "First contract" used to be spelled `budget is None`, which stopped being true when
        # a contractless session gained a default budget of its own. Ask the question
        # directly: it is the first if no duck has been adopted yet.
        first = self.duck is None
        spent = self.executor.budget
        self.duck = duck
        self.executor.contract = duck.frontmatter
        self.executor.budget = Budget(duck.frontmatter.budgets, now=self.transport.now)
        if first or spent is None:
            # The contract's budget is the task's, counted from when the task starts. The
            # carry-over below exists to stop a *second* load refunding a spent budget, and
            # applying it to the first would make "hello-world allows 5 steps" depend on
            # whatever happened before the duck was loaded.
            self.executor.budget.start()
            self.executor.consecutive_failures.clear()
            return
        self.executor.budget.steps = spent.steps
        self.executor.budget.llm_calls = spent.llm_calls
        self.executor.budget.started_at = spent.started_at

    async def connect(self) -> None:
        connected = await self.transport.connect()
        if isinstance(connected, RobotManifest):
            # an adapter: the vocabulary is the manifest's, not the Microduck default
            self.manifest = connected
            self.executor.manifest = connected
            if not self.explicit_registry:
                self.registry = registry_from_manifest(connected, self.transport)
                self.executor.registry = self.registry
        self.heartbeat.start()

    async def close(self) -> None:
        await self.heartbeat.stop()
        with contextlib.suppress(Exception):
            await self.transport.stop()
        with contextlib.suppress(Exception):
            await self.transport.close()

    async def run(self, name: str, params: dict[str, Any] | None) -> dict[str, Any]:
        """`robot_run_verb`: the verb through the executor, with its trace."""
        return await self._call(
            "robot_run_verb",
            {"verb": name, "params": params or {}},
            lambda: self._run(name, params),
        )

    async def _run(self, name: str, params: dict[str, Any] | None) -> dict[str, Any]:
        self.calls += 1
        # `stop` is exempt on purpose, the same way the Executor exempts it. An aborted
        # session is exactly the situation the pilot reaches for the brake in — the heartbeat
        # has just fired, and a verb that was already walking may still be finishing — and
        # refusing `stop` here closed the only control the tool surface offers.
        if self.executor.abort.is_set() and self.registry.canonical(name) != "stop":
            # say *what* went wrong. The heartbeat's own note goes to stderr and reaches no
            # call's trace (its task predates every `capturing` block), so without this the
            # pilot was told only that the session had aborted, never that the link had died.
            why = self.heartbeat.failure
            reason = (
                f"session aborted: the heartbeat failed ({why}); restart quackd. "
                if why is not None
                else "session aborted (kill switch or abort_when); restart quackd. "
            ) + "`stop` still works and is worth sending."
            self._gate(name, "session_aborted", reason)
            return _result(VerbResult.fail(reason))
        try:
            result = await self.executor.run_verb(name, params or {}, source="mcp")
        except VerbNotAllowed as e:
            result = VerbResult.fail(str(e))
        except ConfirmDenied as e:
            result = VerbResult.fail(
                f"{e}: this verb needs human confirmation; "
                "start `quackd serve-mcp --yes` to allow it"
            )
        except BudgetExceeded as e:
            result = VerbResult.fail(f"budget exhausted: {e}")
        except Aborted as e:
            why = self.heartbeat.failure
            result = VerbResult.fail(
                f"aborted: {e}" + (f" (the heartbeat failed: {why})" if why is not None else "")
            )
        return _result(result)

    async def info(self, *, default: bool) -> dict[str, Any]:
        m = self.manifest
        healthy: bool | None = None
        reason: str | None = None
        health = getattr(self.transport, "health", None)
        if health is not None:
            try:
                h = await health()
                healthy, reason = bool(h.ok), h.reason
            except Exception as e:  # informational: a sick robot is a row, not a crash
                healthy, reason = False, str(e)
        return {
            "name": self.name,
            "adapter": adapter_name(self.transport),
            "backend": backend_name(self.transport),
            "vendor": m.vendor if m else None,
            "model": m.model if m else None,
            "embodiment": m.embodiment if m else None,
            "mobility": m.mobility if m else None,
            "manifest_id": m.id if m else None,
            "digest": m.digest() if m else None,
            "contract": self.duck.name if self.duck else None,
            "healthy": healthy,
            "health_reason": reason,
            "aborted": self.executor.abort.is_set(),
            "default": default,
        }

    def verbs_payload(self) -> dict[str, Any]:
        reg = self.registry
        aliases = reg.aliases()
        return {
            "robot": self.name,
            "contract": self.duck.name if self.duck else None,
            "manifest_id": self.manifest.id if self.manifest else None,
            "verbs": [
                {
                    "name": self.shown_name(v),
                    "canonical": v.name,
                    "aliases": [a for a, c in aliases.items() if c == v.name],
                    "core": v.core,
                    "kind": v.kind,
                    "safety_class": v.safety_class,
                    "allowed": self.executor.is_allowed(v.name),
                    "description": v.description,
                    "params": v.tool_schema()["input_schema"],
                }
                for v in reg.verbs()
            ],
        }

    async def observe(self) -> list[str | Image]:
        """The `observe` verb through the executor, then the frame it captured, then the
        trace as one text block (this tool returns content, not a dict)."""
        self.last_frame = None
        result = await self._call("robot_observe", {}, lambda: self._run("observe", {}))
        content: list[str | Image]
        if not result["ok"] or self.last_frame is None:
            # refused, no camera, or a dry run (nothing was captured): words only
            content = [f"{self.name}: {result['summary']}"]
        else:
            self.frames += 1
            summary = str(result["summary"]).removeprefix("frame captured; ")
            content = [
                f"{self.name} camera: {summary}",
                Image(data=png_bytes(self.last_frame), format="png"),
            ]
        if result.get("trace"):
            content.append("trace:\n" + "\n".join(result["trace"]))
        return content

    async def say(self, text: str) -> dict[str, Any]:
        async def inner() -> dict[str, Any]:
            if self.manifest is not None and "sound" not in self.manifest.intents:
                reason = f"{self.name} ({self.manifest.model}) has no sound intent"
                self._gate("say", "no_sound_intent", reason)
                return {"ok": False, "summary": reason, "data": {}}
            return await self._run("say", {"text": text})

        return await self._call("robot_say", {"text": text}, inner)

    def recall(self) -> dict[str, Any]:
        if self.memory is None:
            return {"ok": False, "robot": self.name, "summary": "memory is off for this server"}
        text = self.memory.recall()
        return {
            "ok": True,
            "robot": self.name,
            "summary": text or "nothing remembered yet: this is the first session on this robot",
            "notes": [e.text for e in self.memory.notes()[-20:]],
            "episodes": [e.text for e in self.memory.episodes()[-5:]],
            "path": str(self.memory.path),
        }

    def remember(self, text: str, tags: list[str] | None = None) -> dict[str, Any]:
        if self.memory is None:
            return {"ok": False, "robot": self.name, "summary": "memory is off for this server"}
        try:
            entry = self.memory.remember(
                text, tags=tags, duck=self.duck.name if self.duck else None
            )
        except (ValueError, OSError) as e:
            return {"ok": False, "robot": self.name, "summary": f"could not remember: {e}"}
        return {
            "ok": True,
            "robot": self.name,
            "summary": f"remembered for future sessions: {entry.text}",
            "notes": len(self.memory.notes()),
        }

    def load(self, path: str) -> dict[str, Any]:
        try:
            duck = load_duck(path)
        except DuckParseError as e:
            return {"ok": False, "error": str(e)}
        if duck.frontmatter.flock is not None:
            # same guard as serve(): one MCP pilot must not adopt a many-robot contract
            return {
                "ok": False,
                "error": (
                    "flock ducks are not available over MCP (this session is one pilot, "
                    f"a flock needs a coordinator). Run it with: quackd run {path}"
                ),
            }
        if self.manifest is not None:
            problems = validate_duck(duck, [self.manifest], registry=self.registry)
            if problems:
                return {
                    "ok": False,
                    "error": "; ".join(p.message for p in problems),
                    "problems": [p.message for p in problems],
                }
        reloaded = self.executor.budget is not None
        self.adopt(duck)
        note = f"The executor now enforces this contract for every call to {self.name}."
        budget = self.executor.budget
        if reloaded and budget is not None:
            # say it, so a human reading the session sees the carry-over rather than
            # wondering why the new contract's budget is already part spent
            note += f" What this session already spent still counts: {budget.status()}."
        return {
            "ok": True,
            "robot": self.name,
            "name": duck.name,
            "contract": duck.frontmatter.model_dump(),
            "instructions": duck.body,
            "note": note,
        }


DuckSession = RobotSession
"""The 0.3 name."""


def _result(r: VerbResult) -> dict[str, Any]:
    return {"ok": r.ok, "summary": r.summary, "data": r.data}


@dataclass
class Fleet:
    sessions: dict[str, RobotSession]
    default: str

    def get(self, name: str | None) -> RobotSession | None:
        return self.sessions.get(name or self.default)

    def unknown(self, name: str | None) -> dict[str, Any]:
        return {
            "ok": False,
            "error": f"unknown robot {name!r}; robots: {', '.join(self.sessions)}",
        }

    async def connect_all(self) -> None:
        """Sequential and fail-fast: a fleet with a hole in it is not served."""
        connected: list[RobotSession] = []
        for session in self.sessions.values():
            try:
                await session.connect()
            except BaseException:
                for done in connected:
                    with contextlib.suppress(Exception):
                        await done.close()
                raise
            connected.append(session)

    async def close_all(self) -> None:
        for session in self.sessions.values():
            with contextlib.suppress(Exception):
                await session.close()


def _pick_default(robots: Mapping[str, Any]) -> str:
    """The only robot; else the first Microduck, because a caller that names no robot on
    a mixed fleet most likely means the duck; else the first declared."""
    names = list(robots)
    if len(names) == 1:
        return names[0]
    for name, transport in robots.items():
        if adapter_name(transport) in (None, "microduck"):
            return name
    return names[0]


def _instructions(fleet: Fleet) -> str:
    """One robot gets a prompt about that robot, whichever body it is.

    Until 0.5 the solo prompt was hardcoded to a 25 cm Microduck, which was wrong for every
    other body. The description now comes from the manifest's own blurb."""
    # with --no-memory both tools answer "memory is off", so telling the pilot to call
    # them early is an instruction to waste a turn
    on = any(s.memory is not None for s in fleet.sessions.values())
    if len(fleet.sessions) == 1:
        solo = fleet.sessions[fleet.default]
        manifest = solo.manifest
        blurb = manifest.blurb if manifest and manifest.blurb else "a small robot"
        return INSTRUCTIONS.format(
            names=fleet.default, blurb=blurb, memory=SOLO_MEMORY if on else ""
        )
    return FLEET_INSTRUCTIONS.format(
        n=len(fleet.sessions),
        names=", ".join(fleet.sessions),
        default=fleet.default,
        memory=FLEET_MEMORY if on else "",
    )


def build_fleet_server(
    robots: Mapping[str, DuckTransport],
    *,
    duckfile: str | None = None,
    dry_run: bool = False,
    yes: bool = False,
    registry: VerbRegistry | None = None,
    detector: Detector | None = None,
    heartbeat_period_s: float = 0.5,
    default: str | None = None,
    memory: bool = True,
    memory_dir: str | Path | None = None,
    trace: bool = True,
) -> tuple[MCPServer, Fleet]:
    """One MCP server over several robots, each behind its own executor.

    `--yes` and `--dry-run` are global; contracts, budgets and abort flags are per robot.
    A `.duck` given at startup is adopted by the default robot. With `memory` on, each
    robot gets its `RobotMemory` (keyed adapter:backend, so a simulated body never
    inherits a real one's notes) behind `robot_recall` / `robot_remember`. With `trace` on,
    each robot narrates its calls: a `trace` list in every result that reached its executor,
    and the same lines on stderr in place of the executor's own log lines."""
    if not robots:
        raise ValueError("a fleet needs at least one robot")
    sessions: dict[str, RobotSession] = {}
    for name, transport in robots.items():
        tracer: Tracer | None = None
        if trace:
            tracer = Tracer(observers=[_stderr_view(name), capture_sink])
        reg = registry or default_registry()
        det = detector
        if det is None and backend_name(transport) in ("sim2d", "mujoco"):
            # a bare transport has no manifest to ask; an adapter is upgraded after connect
            from quackd.perception.color_blob import ColorBlobDetector

            det = ColorBlobDetector()
        # A session with no `.duck` has no contract, and used to have no Budget either — so
        # `quackd serve-mcp --robot open_duck:bridge`, which is the setup this module's own
        # docstring advertises, handed an MCP client unlimited, uncounted control of a
        # physical biped. The default budget is generous; what matters is that it is finite
        # and that the step count is visible. Loading a duck replaces it with the contract's.
        budget = Budget(Budgets(), now=transport.now)
        budget.start()
        executor = Executor(
            registry=reg,
            transport=transport,
            contract=None,
            budget=budget,
            detector=det,
            dry_run=dry_run,
            confirm=allow_all if yes else deny_all,
            # with the trace on, its lines replace the executor's own (which would say the
            # same verb twice on stderr); those drop to DEBUG rather than vanish
            log=_prefixed(log.debug if trace else log.info, name),
            trace=tracer,
        )
        heartbeat = Heartbeat(
            transport,
            executor.abort,
            period_s=heartbeat_period_s,
            log=_prefixed(log.warning, name),
            trace=tracer,
        )
        session = RobotSession(
            name=name,
            transport=transport,
            registry=reg,
            executor=executor,
            heartbeat=heartbeat,
            detector=det,
            explicit_registry=registry is not None,
            memory=(
                RobotMemory(
                    f"{adapter_name(transport) or name}:{backend_name(transport)}", memory_dir
                )
                if memory
                else None
            ),
            tracer=tracer,
        )
        executor.on_frame = _stash_frames(session)
        sessions[name] = session
    fleet = Fleet(sessions, default or _pick_default(robots))
    if fleet.default not in sessions:
        raise ValueError(f"default robot {fleet.default!r} is not one of {list(sessions)}")
    if duckfile:
        sessions[fleet.default].adopt(load_duck(duckfile))

    @contextlib.asynccontextmanager
    async def lifespan(_server: MCPServer) -> AsyncIterator[Fleet]:
        await fleet.connect_all()
        for session in fleet.sessions.values():
            # now that the robot has said what it actually has, not what its description
            # claims. `build_fleet_server` can only see a bare transport's backend.
            live = getattr(session.transport, "manifest", None)
            if live is not None:
                session.detector = detector_for(
                    live.sensors,
                    session.detector,
                    fov_deg=live.limits.get("camera_fov_deg"),
                    backend=live.backend,
                )
                session.executor.detector = session.detector
            log.info(
                "quackd MCP server up: robot=%s transport=%s dry_run=%s",
                session.name,
                backend_name(session.transport),
                dry_run,
            )
        try:
            yield fleet
        finally:
            await fleet.close_all()

    mcp = MCPServer(
        "quackd", instructions=_instructions(fleet), version=__version__, lifespan=lifespan
    )

    def me() -> RobotSession:
        return fleet.sessions[fleet.default]

    # ── the fleet tools ─────────────────────────────────────────────────────────────

    @mcp.tool(
        description="Every robot this server fronts (adapter, body, manifest, contract, "
        "health) and which one is the default. Call this first."
    )
    async def robot_list() -> dict[str, Any]:
        return {
            "robots": [
                await s.info(default=(name == fleet.default)) for name, s in fleet.sessions.items()
            ],
            "default": fleet.default,
        }

    @mcp.tool(
        description="One robot's verbs from its own manifest: params, safety class, "
        "canonical name, aliases, and whether its contract allows each now."
    )
    async def robot_list_verbs(robot: str | None = None) -> dict[str, Any]:
        session = fleet.get(robot)
        return session.verbs_payload() if session else fleet.unknown(robot)

    @mcp.tool(
        description="Run a verb on one robot through its executor, with JSON params. "
        "Refusals come back as ok=false; a verb its manifest lacks is a refusal too. "
        "`trace` lists what happened behind the scenes: gates, intents sent, timing."
    )
    async def robot_run_verb(
        verb: str, params: dict[str, Any] | None = None, robot: str | None = None
    ) -> dict[str, Any]:
        session = fleet.get(robot)
        return await session.run(verb, params) if session else fleet.unknown(robot)

    @mcp.tool(
        description="The observe verb on one robot, through its executor: the camera frame "
        "as a PNG plus a detection summary, then a trace block of what happened.",
        structured_output=False,
    )
    async def robot_observe(robot: str | None = None) -> list[str | Image]:
        session = fleet.get(robot)
        return await session.observe() if session else [str(fleet.unknown(robot)["error"])]

    @mcp.tool(
        description="Say something on one robot: tones on a Microduck. A robot without a "
        "sound intent refuses."
    )
    async def robot_say(text: str, robot: str | None = None) -> dict[str, Any]:
        session = fleet.get(robot)
        return await session.say(text) if session else fleet.unknown(robot)

    @mcp.tool(
        description="Load a .duck contract on one robot: its requires are checked against "
        "that robot's manifest, then its allowlist and budgets apply there; the body comes "
        "back as instructions."
    )
    async def robot_load_duckfile(path: str, robot: str | None = None) -> dict[str, Any]:
        session = fleet.get(robot)
        return session.load(path) if session else fleet.unknown(robot)

    @mcp.tool(
        description="What one robot remembers from earlier sessions and runs: the notes a "
        "pilot saved with robot_remember, and how its recent runs ended. Call it before "
        "planning; it costs no step."
    )
    async def robot_recall(robot: str | None = None) -> dict[str, Any]:
        session = fleet.get(robot)
        return session.recall() if session else fleet.unknown(robot)

    @mcp.tool(
        description="Keep one short fact for future sessions on one robot (where things "
        "usually are, what worked, what to avoid). Moves nothing, costs no step. The same "
        "sentence twice updates the old note instead of duplicating it."
    )
    async def robot_remember(
        text: str, tags: list[str] | None = None, robot: str | None = None
    ) -> dict[str, Any]:
        session = fleet.get(robot)
        return session.remember(text, tags) if session else fleet.unknown(robot)

    return mcp, fleet


def build_server(
    transport: DuckTransport,
    *,
    duckfile: str | None = None,
    dry_run: bool = False,
    yes: bool = False,
    registry: VerbRegistry | None = None,
    detector: Detector | None = None,
    heartbeat_period_s: float = 0.5,
    memory: bool = True,
    memory_dir: str | Path | None = None,
    trace: bool = True,
) -> tuple[MCPServer, RobotSession]:
    """One robot, the 0.3 entry point: a fleet of one named after its adapter."""
    name = adapter_name(transport) or "duck"
    mcp, fleet = build_fleet_server(
        {name: transport},
        duckfile=duckfile,
        dry_run=dry_run,
        yes=yes,
        registry=registry,
        detector=detector,
        heartbeat_period_s=heartbeat_period_s,
        memory=memory,
        memory_dir=memory_dir,
        trace=trace,
    )
    return mcp, fleet.sessions[name]


def serve(
    duckfile: str | None = None,
    seed: int | None = None,
    address: str | None = None,
    camera_url: str | None = None,
    token: str | None = None,
    dry_run: bool = False,
    yes: bool = False,
    *,
    robot: str | None = None,
    robots: str | None = None,
    warn: Any = None,
    memory: bool = True,
    memory_dir: str | None = None,
    trace: bool | None = None,
) -> None:
    from quackd.adapters.factory import (
        RobotSpec,
        describe,
        make_adapter,
        parse_robots,
        resolve_robot,
    )

    if robots and robot:
        raise SystemExit("choose one: --robots name=<adapter>:<backend>,... or --robot")
    probe: DuckFile | None = None
    default = None
    if duckfile:
        probe = load_duck(duckfile)
        if probe.frontmatter.flock is not None:
            raise SystemExit(
                "flock ducks are not available over MCP yet (the MCP client is one pilot, "
                "a flock needs a coordinator). Run it with: quackd run " + duckfile
            )
        if isinstance(probe.frontmatter.robots, str):
            default = probe.frontmatter.robots
    specs: list[RobotSpec] = (
        parse_robots(robots) if robots else [resolve_robot(robot, duck_default=default)]
    )
    manifests = {spec.name or describe(spec).id: describe(spec) for spec in specs}
    if probe is not None:
        # the contract lands on the default robot: refuse now, with the validator's words
        target = _pick_default(
            {name: _Probe(spec) for name, spec in zip(manifests, specs, strict=True)}
        )
        problems = validate_duck(probe, [manifests[target]])
        if problems:
            raise SystemExit(
                f"{duckfile} cannot run on {target} ({manifests[target].model}): "
                + "; ".join(p.message for p in problems)
            )
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format="quackd-mcp %(levelname)s %(message)s"
    )
    adapters = {
        name: make_adapter(
            spec,
            seed=seed if seed is not None else 0,
            address=address,
            camera_url=camera_url,
            token=token,
        )
        for name, spec in zip(manifests, specs, strict=True)
    }
    mcp, _fleet = build_fleet_server(
        adapters,
        duckfile=duckfile,
        dry_run=dry_run,
        yes=yes,
        memory=memory,
        memory_dir=memory_dir,
        # the env is the switch a desktop-spawned server has (no shell, no cwd `.env`)
        trace=trace if trace is not None else trace_enabled_default(),
    )
    mcp.run(transport="stdio")


class _Probe:
    """Enough of an adapter for `_pick_default` to choose before anything is built."""

    def __init__(self, spec: Any) -> None:
        self.name = spec.adapter
        self.backend = spec.backend


if __name__ == "__main__":  # pragma: no cover
    serve()


__all__ = [
    "TOOL_NAMES",
    "DuckSession",
    "Fleet",
    "RobotSession",
    "build_fleet_server",
    "build_server",
    "serve",
]
