"""The one place an LLM appears in a flock run, and the proof it stayed there.

Wedges are ALWAYS computed deterministically (an equal partition of the circle over the
sorted member names — the model does not get to draw geometry). A real provider gets one
forced `plan_flock_task` call to tune the task knobs; anything invalid falls back to the
deterministic defaults with a logged `planner_fallback`. The fake provider makes zero
calls. `summary.json` records `llm_calls` (0 or 1).
"""

from __future__ import annotations

import time
from typing import Any

from quackd.agent.providers.base import Exchange, LLMProvider, Observation, ProviderTurn, Usage
from quackd.duckfile.schema import DuckFile
from quackd.flock.messages import FlockTask, Wedge
from quackd.trace import Tracer

PLAN_TOOL = {
    "name": "plan_flock_task",
    "description": (
        "Plan the flock task. Choose the detector target and the approach parameters. "
        "Call this tool exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "target": {"type": "string", "description": "Detector label to hunt (e.g. ball)."},
            "kick_leg": {"type": "string", "enum": ["left", "right"]},
            "stop_distance": {"type": "number", "minimum": 0.1, "maximum": 1.0},
            "step_deg": {"type": "number", "minimum": 15, "maximum": 120},
            "timeout_s": {"type": "number", "minimum": 10, "maximum": 600},
        },
        "required": [],
        "additionalProperties": False,
    },
}

TUNABLE = ("target", "kick_leg", "stop_distance", "step_deg", "timeout_s")
CLAMPS = {"stop_distance": (0.1, 1.0), "step_deg": (15.0, 120.0), "timeout_s": (10.0, 600.0)}


def equal_wedges(members: list[str]) -> dict[str, Wedge]:
    """Deterministic search partition: the circle split equally over sorted names."""
    ordered = sorted(members)
    width = 360.0 / len(ordered)
    return {
        name: Wedge(start_deg=i * width, end_deg=(i + 1) * width) for i, name in enumerate(ordered)
    }


def default_task(duck: DuckFile, task_id: str) -> FlockTask:
    goal = duck.body.strip()
    target = "person" if "person" in duck.name else "ball"
    flock = duck.frontmatter.flock
    restart_s = flock.search.restart_s if flock is not None else 8.0
    roles = dict(flock.roles or {}) if flock is not None else {}
    return FlockTask(
        task_id=task_id, name=duck.name, goal=goal, target=target, restart_s=restart_s, roles=roles
    )


async def plan_flock_task(
    duck: DuckFile,
    members: list[str],
    provider: LLMProvider,
    task_id: str,
    log: Any = lambda *_: None,
    wedge_members: list[str] | None = None,
    *,
    trace: Tracer | None = None,
) -> tuple[FlockTask, dict[str, Wedge], Usage, int, bool]:
    """Returns (task, wedges, usage, llm_calls, fallback_used). Wedges are split over the
    members that can move (`wedge_members`); a member with no wedge sweeps its whole range.

    `trace` narrates the one model call this run makes, as the same `llm_request`/`llm` pair
    a solo run emits, so a flock's transcript reads like any other run's."""
    wedges = equal_wedges(wedge_members or members)
    task = default_task(duck, task_id)
    if provider.name == "fake":
        return task, wedges, Usage(), 0, False

    def emit(kind: str, **data: Any) -> None:
        if trace is not None:
            trace.emit(kind, **data)

    def note(text: str) -> None:
        """One line for both audiences: `log` is `--verbose`, the trace shows the same words."""
        log(text)
        emit("note", text=text)

    roles = ""
    if task.roles:
        parts = [f"{name} requires {', '.join(r.requires)}" for name, r in task.roles.items()]
        roles = f" Roles are fixed by the task file ({'; '.join(parts)}); you tune parameters only."
    prompt = (
        "You are planning a task for a flock of small robots in a 2 m square arena. "
        f"Members: {', '.join(sorted(members))}. Each will search its own heading sector, "
        "the closest sighting wins an auction, and the winner approaches and kicks."
        f"{roles}\n\n"
        f"Task file '{duck.name}':\n{duck.body}\n\n"
        "Call plan_flock_task once with your chosen parameters (omit any you would keep "
        "at the defaults)."
    )
    fallback = False
    usage = Usage()
    # `step=0`: the planner runs before the flock has taken a step, and a reader of the
    # transcript should be able to read it exactly as a solo run's first turn
    emit(
        "llm_request",
        step=0,
        provider=provider.name,
        model=provider.model,
        messages=1,
        images=0,
        reprompt=False,
        purpose="plan_flock_task",
    )
    started = time.perf_counter()
    turn: ProviderTurn | None = None
    try:
        turn = await provider.step(
            "You plan tasks for cooperating duck robots. Answer with one tool call.",
            [Exchange(observation=Observation(text=prompt))],
            [PLAN_TOOL],
        )
        emit(
            "llm",
            step=0,
            provider=provider.name,
            model=provider.model,
            text=turn.text,
            thinking=turn.thinking,
            tool_calls=[tc.model_dump() for tc in turn.tool_calls],
            usage=turn.usage.model_dump(),
            stop_reason=turn.stop_reason,
            latency_s=round(time.perf_counter() - started, 3),
        )
        usage = turn.usage
        call = next((c for c in turn.tool_calls if c.name == "plan_flock_task"), None)
        if call is None:
            raise ValueError("no plan_flock_task call in the reply")
        # per-field: numeric arguments clamp into range, the rest validate individually,
        # so one bad argument never discards the model's valid choices
        dropped: dict[str, Any] = {}
        for k, v in call.arguments.items():
            if k not in TUNABLE:
                continue
            if k in CLAMPS and isinstance(v, int | float) and not isinstance(v, bool):
                lo, hi = CLAMPS[k]
                v = min(max(float(v), lo), hi)
            try:
                task = FlockTask(**{**task.model_dump(), k: v})
            except Exception:
                dropped[k] = v
        if dropped:
            note(f"planner dropped invalid arguments: {dropped}")
    except Exception as e:  # planner trouble (refusal, no call, network) -> defaults
        fallback = True
        if turn is None:
            # only when the CALL failed. A reply that arrived and then disappointed us was
            # already narrated above, and one `llm` per `llm_request` is what a reader counts on
            emit(
                "llm",
                step=0,
                provider=provider.name,
                model=provider.model,
                error=f"{type(e).__name__}: {e}",
                latency_s=round(time.perf_counter() - started, 3),
            )
        note(f"planner fallback: {type(e).__name__}: {e}")
        task = default_task(duck, task_id)
    return task, wedges, usage, 1, fallback
