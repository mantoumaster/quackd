"""Wires a flock run together: world, clock, bus, members, coordinator, recorder, summary.

The flock analogue of `agent.loop.run_duck`. Success comes from sim ground truth
(`ball_displacement_m`), so the summary cannot claim what the world did not see. Since 0.4
members are adapters sharing one arena: Microducks on the same lockstep clock, each with
its own manifest-built registry (ADR-0020).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quackd.adapters.factory import RobotSpec, describe, parse_robot_spec
from quackd.agent.providers.base import LLMProvider, Usage
from quackd.agent.transcript import new_run_dir
from quackd.duckfile.schema import DuckFile, FlockSection
from quackd.flock.auction import AuctionPolicy
from quackd.flock.bus import Bus, InProcessBus
from quackd.flock.coordinator import FlockCoordinator, FlockOutcome
from quackd.flock.member import FlockMember
from quackd.flock.messages import FlockMessage
from quackd.flock.planner import plan_flock_task
from quackd.flock.transcript import FlockTranscript
from quackd.perception.color_blob import ColorBlobDetector
from quackd.sim2d.clock import FlockClock
from quackd.sim2d.world import World
from quackd.trace import Sink, Tracer
from quackd.transport.sim2d import make_flock

DEFAULT_MEMBER = "microduck:sim2d"

BusFactory = Callable[[Callable[[FlockMessage], None]], Bus]
"""`bus_factory(tap) -> Bus`: the seam for `MqttBus`; the in-process bus is the default."""

TraceFactory = Callable[[str], "Sink | None"]
"""`trace(name) -> Sink | None`: a view per member, by member name, and one more for
`FLOCK_TRACE`: the flock's own events (the planner's call, the coordinator's story). Return a
distinct sink per name, or two members' intent bursts fold into one line."""

FLOCK_TRACE = "flock"


@dataclass
class FlockResult:
    outcome: FlockOutcome
    reason: str
    kicker: str | None
    auctions: int
    bids: int
    ball_displacement_m: float
    sim_elapsed_s: float
    run_dir: Path
    per_duck: dict[str, dict[str, Any]] = field(default_factory=dict)
    gif_path: Path | None = None
    usage: Usage = field(default_factory=Usage)
    spotter: str | None = None
    assignments: dict[str, str] = field(default_factory=dict)
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    trace_dropped: int = 0
    """Events a view raised on and never showed. The transcripts have them all."""

    @property
    def ok(self) -> bool:
        return self.outcome == "success"


def member_specs(
    members: list[str], robots: dict[str, str] | None, duck_default: str | dict[str, str] | None
) -> dict[str, RobotSpec]:
    """Each member's robot: `--robots`, else the duck's `robots:`, else a simulated duck."""
    given: dict[str, str] = {}
    if robots:
        given = dict(robots)
    elif isinstance(duck_default, dict):
        given = dict(duck_default)
    elif isinstance(duck_default, str):
        given = dict.fromkeys(members, duck_default)
    specs: dict[str, RobotSpec] = {}
    for name in members:
        parsed = parse_robot_spec(given.get(name, DEFAULT_MEMBER))
        specs[name] = RobotSpec(parsed.adapter, parsed.backend, name)
    return specs


def make_sim_flock(
    specs: dict[str, RobotSpec], *, seed: int, live: bool, person: bool = True
) -> tuple[World, FlockClock, dict[str, Any]]:
    """One arena, one clock, one adapter per member in sorted member order, via the 0.3
    path (`make_flock`), so its worlds stay byte-identical."""
    from quackd.adapters.microduck import MicroduckAdapter

    ordered = sorted(specs)
    ducks = [n for n in ordered if specs[n].adapter == "microduck"]
    unknown = [n for n in ordered if n not in ducks]
    if unknown:
        raise ValueError(f"flock mode knows microduck, not {unknown}")
    adapters: dict[str, Any] = {}
    transports = make_flock(len(ducks), seed=seed, live=live, person=person)
    for i, name in enumerate(ducks):
        adapters[name] = MicroduckAdapter(transports[i], robot_id=name)
    return transports[0].world, transports[0].clock, adapters


async def run_flock(
    duck: DuckFile,
    *,
    provider: LLMProvider,
    seed: int = 0,
    runs_dir: str | Path = "runs",
    n_override: int | None = None,
    dry_run: bool = False,
    max_steps: int | None = None,
    live: bool = False,
    gif_size: int = 256,
    on_recorder: Any = None,
    log: Any = lambda *_: None,
    robots: dict[str, str] | None = None,
    bus_factory: BusFactory | None = None,
    trace: TraceFactory | None = None,
) -> FlockResult:
    flock: FlockSection = duck.frontmatter.flock or FlockSection()
    if n_override is not None:
        if flock.roles:
            raise ValueError("--flock N cannot be combined with flock.roles; name the members")
        flock = flock.model_copy(update={"members": n_override})
    members = flock.member_names
    specs = member_specs(members, robots, duck.frontmatter.robots)
    for name, spec in specs.items():
        if spec.backend != "sim2d":
            raise ValueError(f"flock mode is simulator only (docs/flock.md): {name} is {spec.key}")
    manifests = {name: describe(spec) for name, spec in specs.items()}
    mobile = [name for name in members if manifests[name].mobility != "none"]
    policy = AuctionPolicy.from_flock(flock)

    run_name = duck.name if duck.name.startswith("flock") else f"flock-{duck.name}"
    run_dir = new_run_dir(runs_dir, run_name)
    world, clock, adapters = make_sim_flock(specs, seed=seed, live=live)
    transcript = FlockTranscript(run_dir, now=clock.now)
    bus: Bus = (
        bus_factory(transcript.on_bus) if bus_factory else InProcessBus(tap=transcript.on_bus)
    )
    start_bus = getattr(bus, "start", None)
    if callable(start_bus):
        start_bus()  # inside the event loop, so remote deliveries are marshalled onto it

    def view(name: str) -> Sink | None:
        return trace(name) if trace is not None else None

    flock_views = [v] if (v := view(FLOCK_TRACE)) is not None else []
    # the planner's call is recorded here because nothing else records it: there is no
    # per-member transcript it belongs to, so flock.jsonl is its paper trail
    planner_trace = Tracer(record=transcript.sink, observers=flock_views)

    task_id = uuid.uuid4().hex[:8]
    task, wedges, usage, llm_calls, fallback = await plan_flock_task(
        duck,
        members,
        provider,
        task_id,
        log=log,
        wedge_members=mobile or members,
        trace=planner_trace,
    )
    frame_hints = flock.frame_hints == "on" or (
        flock.frame_hints == "auto" and all(s.backend == "sim2d" for s in specs.values())
    )
    task = task.model_copy(
        update={
            "success_moved_m": max(task.success_moved_m, 0.3),
            "roles": dict(flock.roles or {}),
            "frame_hints": bool(flock.roles) and frame_hints,
        }
    )
    transcript.write(
        "plan",
        task=task.model_dump(),
        wedges={k: w.model_dump() for k, w in wedges.items()},
        robots={name: spec.key for name, spec in specs.items()},
        provider=provider.name,
        model=provider.model,
        llm_calls=llm_calls,
        fallback=fallback,
        usage=usage.model_dump(),
    )

    contract = duck.frontmatter
    if max_steps is not None:
        # --max-steps overrides for one run, per duck, exactly as in a solo run
        contract = contract.model_copy(
            update={"budgets": contract.budgets.model_copy(update={"max_steps": max_steps})}
        )
    flock_members: dict[str, FlockMember] = {}
    ordered = sorted(members)
    for name in ordered:
        flock_members[name] = FlockMember(
            name,
            contract,
            adapters[name],
            ColorBlobDetector(),
            bus,
            transcript,
            task,
            hb_period_s=flock.safety.per_duck_heartbeat_s,
            dry_run=dry_run,
            trace=view(name),
        )

    # no record: every one of the coordinator's kinds is already in flock.jsonl under its
    # own name, written the line before each `_event`
    story = Tracer(observers=flock_views)
    coordinator = FlockCoordinator(
        task=task,
        members=flock_members,
        wedges=wedges,
        bus=bus,
        clock=clock,
        transcript=transcript,
        policy=policy,
        success_moved_m=task.success_moved_m,
        log=log,
        trace=story,
    )
    if on_recorder is not None:
        on_recorder(adapters[ordered[0]], coordinator)

    transcript.write(
        "flock_start",
        duck=duck.name,
        members=ordered,
        robots={name: spec.key for name, spec in specs.items()},
        seed=seed,
        dry_run=dry_run,
        contract=duck.frontmatter.model_dump(),
    )
    wall0 = time.perf_counter()
    outcome, reason = await coordinator.run()

    truth = world.ball_displacement_m
    if outcome == "success" and truth < task.success_moved_m and not dry_run:
        # a member claimed a kick the world did not record: the world wins
        outcome, reason = (
            "failure",
            (f"claimed kick not confirmed by telemetry (ball moved {truth:.2f} m)"),
        )
    kicker = coordinator.kicker if coordinator.kicker is not None else coordinator.prev_kicker

    def kicks_connected(name: str) -> int:
        index = getattr(adapters[name], "duck_index", None)
        return int(world.ducks[index].kicks_connected) if index is not None else 0

    per_duck = {
        name: {
            "final_status": m.final_status,
            "steps": m.steps,
            "verbs_failed": m.verbs_failed,
            "kicks_connected": kicks_connected(name),
            "robot": specs[name].key,
        }
        for name, m in flock_members.items()
    }
    trace_dropped = (
        planner_trace.dropped
        + story.dropped
        + sum(m.tracer.dropped for m in flock_members.values())
    )
    summary = {
        "duck": duck.name,
        "outcome": outcome,
        "reason": reason,
        "flock": {"members": ordered, "method": flock.allocation.method},
        "robots": {name: spec.key for name, spec in specs.items()},
        "roles": {name: role.model_dump() for name, role in (flock.roles or {}).items()},
        "assignments": coordinator.assignments,
        "kicker": kicker,
        "spotter": coordinator.spotter,
        "verdicts": coordinator.verdicts,
        "frame_hints": task.frame_hints,
        "auctions": coordinator.auctions,
        "bids": coordinator.bids,
        "bus_messages": getattr(bus, "published", 0),
        "ball_displacement_m": round(truth, 3),
        "planner": {
            "provider": provider.name,
            "model": provider.model,
            "llm_calls": llm_calls,
            "fallback": fallback,
            "usage": usage.model_dump(),
        },
        "policy": policy.__dict__,
        "per_duck": per_duck,
        "sim_elapsed_s": round(clock.now(), 2),
        "wall_elapsed_s": round(time.perf_counter() - wall0, 2),
        "seed": seed,
        "transport": "sim2d",
        "dry_run": dry_run,
        "trace_dropped": trace_dropped,
    }
    transcript.write("flock_end", **{k: v for k, v in summary.items() if k != "per_duck"})
    transcript.write_summary(summary)
    transcript.close()
    close_bus = getattr(bus, "close", None)
    if callable(close_bus):
        close_bus()
    return FlockResult(
        outcome=outcome,
        reason=reason,
        kicker=kicker,
        auctions=coordinator.auctions,
        bids=coordinator.bids,
        ball_displacement_m=truth,
        sim_elapsed_s=clock.now(),
        run_dir=run_dir,
        per_duck=per_duck,
        gif_path=None,
        usage=usage,
        spotter=coordinator.spotter,
        assignments=dict(coordinator.assignments),
        verdicts=list(coordinator.verdicts),
        trace_dropped=trace_dropped,
    )
