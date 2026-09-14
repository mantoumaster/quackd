"""An arm's own verbs: joints, a gripper, place, and pick as a LeRobot policy.

The thesis holds here too: `pick` is one skill intent, and the robot's own controller (a
LeRobot policy) moves the arm; quackd never writes a grasp control law. `place` is the
one thing that needs no policy: open the gripper where it is. Joint names are the SO-101
follower's six motors, verified upstream (`upstream_api.SO_MOTORS`).

Every verb that moves a joint watches the measurement rather than timing the travel. Two
upstream facts make that necessary rather than tidy. One `send_action` moves a joint at most
the configured step (`upstream_api.SO_ACTION_CLAMP`), so a goal takes as many sends as it
takes, and the same cap applies to the gripper in its own 0..100 units. And nothing reports
whether a goal was reached, so an arm that stalled against an obstacle and an arm that
arrived look identical unless somebody compares the goal with the position.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from quackd.transport.base import DuckState, Intent
from quackd.verbs.core import send_or_fail
from quackd.verbs.registry import NoParams, Precondition, Verb, VerbContext, VerbResult

JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
"""The SO-101 follower's motors, in bus order (upstream_api.SO_MOTORS)."""
GRIPPER_OPEN = 100.0
GRIPPER_CLOSED = 0.0

TICK_S = 0.1
"""How often a goal is re-sent while a verb runs, matching the core `move` verb's cadence."""
TOL_DEG = 5.0
"""Close enough, for a body joint. The arm's controller is P-only by default (I is 0), so a
loaded joint settles a little short of its goal and never closes the last degree."""
GRIPPER_TOL = 5.0
STALL_TICKS = 5
STALL_DEG = 0.5
"""No joint moved this far for this many ticks, and it is not going to."""
GRIPPER_S = 6.0
"""How long to give the gripper. The step cap applies to its 0..100 range too, so a full
open takes 100 divided by the step, times the tick."""

PICK_POLL_S = 0.5


class MoveJointsParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    positions: dict[str, float] = Field(
        ...,
        description="Joint -> goal in degrees (the gripper in 0..100). Only the joints given move.",
    )
    duration_s: float = Field(
        default=5.0,
        ge=0.2,
        le=12,
        description=(
            "How long to give the motion before giving up. The arm moves at its own capped "
            "speed whatever this says, and the verb ends as soon as every joint has arrived."
        ),
    )

    @field_validator("positions")
    @classmethod
    def _known_joints(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("give at least one joint")
        unknown = sorted(set(value) - set(JOINTS))
        if unknown:
            raise ValueError(f"unknown joints {unknown}; this arm has {', '.join(JOINTS)}")
        for joint, goal in value.items():
            lo, hi = (0.0, 100.0) if joint == "gripper" else (-180.0, 180.0)
            if not lo <= goal <= hi:
                raise ValueError(f"{joint}={goal} is outside {lo}..{hi}")
        return value


class GripperParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    open: bool = Field(default=True, description="True opens the gripper, False closes it.")


class PickParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str = Field(default="object", description="What to pick, as the policy's task text.")
    max_s: float = Field(default=20.0, ge=1, le=60, description="Give up after this long.")


# ── preconditions the manifest references by name ───────────────────────────────────────


def _torque_on(state: DuckState) -> str | None:
    if state.extras.get("torque", True):
        return None
    return (
        "the arm's torque is off; enable it from LeRobot first (quackd never toggles torque). "
        "A servo that has tripped its own overload protection reads this way too"
    )


def _holding(state: DuckState) -> str | None:
    return None if state.holding else "nothing is held: pick something first"


def _not_hot(state: DuckState) -> str | None:
    """Only on the verbs that move the five body joints. LeRobot caps the gripper's torque
    and current and caps nothing else (`upstream_api.SO_BODY_HAS_NO_TORQUE_CAP`), so this is
    the only thing standing between a stalled elbow and a servo cooking itself. A backend
    that cannot read a temperature says nothing and is believed."""
    hot = [str(j) for j in state.extras.get("hot", [])]
    if not hot:
        return None
    temperatures: dict[str, Any] = state.extras.get("temperature_c", {})
    worst = max(hot, key=lambda joint: temperatures.get(joint, 0))
    reading = temperatures.get(worst, "?")
    return (
        f"{worst} reads {reading}°C: let the arm cool before moving it. The servo's own "
        "cut-off is 70°C and a joint that trips it goes slack without announcing it"
    )


def lerobot_conditions() -> dict[str, Precondition]:
    return {"torque_on": _torque_on, "holding": _holding, "not_hot": _not_hot}


# ── watching a goal arrive ──────────────────────────────────────────────────────────────


def _joints_of(state: DuckState) -> dict[str, float]:
    return {str(k): float(v) for k, v in dict(state.extras.get("joints", {})).items()}


def _shortfall(goal: dict[str, float], joints: dict[str, float]) -> str:
    """The joint furthest from where it was asked to be, in words."""
    behind = {k: abs(joints[k] - v) for k, v in goal.items() if k in joints}
    if not behind:
        return "the arm reported no joint positions"
    worst = max(behind, key=lambda joint: behind[joint])
    return f"{worst} is at {joints[worst]:.0f} with a goal of {goal[worst]:.0f}"


def _stall_threshold(ctx: VerbContext) -> float:
    """How little a joint may move per tick before it counts as stopped. Scaled to the step
    cap: an arm told to move half a degree a tick is moving at full speed at half a degree
    a tick, and calling that a stall would fail every legitimate move."""
    step = (ctx.manifest.limits.get("step_deg") if ctx.manifest is not None else None) or 0.0
    return min(STALL_DEG, step / 2) if step > 0 else STALL_DEG


async def _drive(
    ctx: VerbContext,
    intent: Intent,
    goal: dict[str, float],
    *,
    budget_s: float,
    tolerance: Callable[[str], float],
) -> tuple[dict[str, float], DuckState | None, str, str | None]:
    """Re-send a goal until the arm is there, stops moving, or the budget runs out.

    Returns the last joint reading, the last state, how it ended (`arrived`, `stalled`,
    `timeout` or `refused`) and a reason when it did not arrive. A failure stops the arm
    first: `stop` holds the present position and deliberately leaves the gripper's goal
    alone, so stopping mid-move never drops what is held."""
    started = ctx.transport.now()
    stall = _stall_threshold(ctx)
    joints: dict[str, float] = {}
    previous: dict[str, float] = {}
    state: DuckState | None = None
    still = 0
    while ctx.transport.now() - started < budget_s:
        if (fail := await send_or_fail(ctx, intent)) is not None:
            await ctx.transport.stop()
            return joints, state, "refused", fail.summary
        await ctx.transport.sleep(TICK_S)
        state = await ctx.transport.get_state()
        joints = _joints_of(state)
        error = {k: abs(joints[k] - v) for k, v in goal.items() if k in joints}
        if error and all(gap <= tolerance(joint) for joint, gap in error.items()):
            return joints, state, "arrived", None
        moved = [abs(joints[k] - previous[k]) for k in previous if k in joints]
        still = still + 1 if moved and max(moved) <= stall else 0
        previous = {k: joints[k] for k in goal if k in joints}
        if still >= STALL_TICKS:
            await ctx.transport.stop()
            return (
                joints,
                state,
                "stalled",
                f"{_shortfall(goal, joints)}, and it has stopped moving",
            )
    await ctx.transport.stop()
    return joints, state, "timeout", f"{_shortfall(goal, joints)} when the time ran out"


# ── the verbs ───────────────────────────────────────────────────────────────────────────


async def report_state(ctx: VerbContext, _: NoParams) -> VerbResult:
    """What this arm knows, in the sentence rather than in the data.

    A pilot reads a verb's summary text and never its data: the dump goes to the transcript
    and to an MCP client, and the observation the model is handed carries the summary. The
    core verb's summary is a posture and a policy name, which on a bolted-down arm is two
    facts it has not got and none of the four it has. So this one says them."""
    state = await ctx.transport.get_state()
    extras = state.extras
    joints = _joints_of(state)
    where = ", ".join(f"{name} {value:.0f}" for name, value in joints.items())
    torque = "torque on" if extras.get("torque", True) else "TORQUE OFF"
    temperatures = {str(k): float(v) for k, v in (extras.get("temperature_c") or {}).items()}
    if hot := [str(joint) for joint in extras.get("hot", [])]:
        heat = f"TOO HOT TO MOVE: {', '.join(hot)}"
    elif temperatures:
        hottest = max(temperatures, key=lambda joint: temperatures[joint])
        heat = f"hottest {hottest} {temperatures[hottest]:.0f}°C"
    else:
        heat = "no temperature reported"
    held = "holding something" if state.holding else "holding nothing"
    parts = [where or "no joints reported", torque, heat, held]
    # a camera earns a clause only when a read has actually failed. Not when it is merely
    # unread: `ok` is false until the first frame, and a camera that opened and has not been
    # asked yet is not news. A working one is already in every observation as detections,
    # and a dead one is otherwise silent on a run that cannot call `observe`.
    camera = extras.get("camera")
    if isinstance(camera, dict) and camera.get("error"):
        parts.append(f"CAMERA DOWN: {camera['error']}")
    return VerbResult.success("; ".join(parts), state=state.model_dump())


async def move_joints(ctx: VerbContext, p: MoveJointsParams) -> VerbResult:
    goal = dict(p.positions)
    joints, _state, _how, why = await _drive(
        ctx,
        Intent.joint(goal, p.duration_s),
        goal,
        budget_s=p.duration_s,
        tolerance=lambda joint: GRIPPER_TOL if joint == "gripper" else TOL_DEG,
    )
    if why is not None:
        return VerbResult.fail(f"move_joints: {why}", goal=goal, joints=joints)
    return VerbResult.success(
        "moved " + ", ".join(f"{k}={joints.get(k, v):.0f}" for k, v in goal.items()),
        goal=goal,
        joints=joints,
    )


async def _drive_gripper(
    ctx: VerbContext, *, open_: bool
) -> tuple[float | None, DuckState | None, str, str | None]:
    """The gripper is a joint like the others: the step cap moves it a few units per send,
    and where it stops is the whole of what quackd knows about holding something."""
    goal = {"gripper": GRIPPER_OPEN if open_ else GRIPPER_CLOSED}
    joints, state, how, why = await _drive(
        ctx, Intent.gripper(open_), goal, budget_s=GRIPPER_S, tolerance=lambda _: GRIPPER_TOL
    )
    return joints.get("gripper"), state, how, why


async def gripper(ctx: VerbContext, p: GripperParams) -> VerbResult:
    position, state, how, why = await _drive_gripper(ctx, open_=p.open)
    where = "" if position is None else f" (stopped at {position:.0f}/100)"
    if p.open:
        if how != "arrived":
            return VerbResult.fail(f"gripper did not open: {why}", open=True, position=position)
        return VerbResult.success(f"gripper open{where}", open=True, position=position)
    # closing on something is the one move that is supposed to stop short
    holding = bool(state is not None and state.holding)
    if how == "arrived" or (how == "stalled" and holding):
        grasped = "on something" if holding else "on nothing"
        return VerbResult.success(
            f"gripper closed {grasped}{where}", open=False, position=position, holding=holding
        )
    return VerbResult.fail(
        f"gripper did not close: {why}", open=False, position=position, holding=holding
    )


async def pick(ctx: VerbContext, p: PickParams) -> VerbResult:
    """One skill intent; the policy drives. quackd only watches the clock and `holding`."""
    if (fail := await send_or_fail(ctx, Intent.do(f"policy:pick:{p.target}"))) is not None:
        return fail
    t0 = ctx.transport.now()
    state = await ctx.transport.get_state()
    while ctx.transport.now() - t0 < p.max_s:
        await ctx.transport.sleep(PICK_POLL_S)
        state = await ctx.transport.get_state()
        if state.holding:
            return VerbResult.success(
                f"picked {p.target}", target=p.target, seconds=round(ctx.transport.now() - t0, 1)
            )
        if not str(state.policy).startswith("policy:"):
            break  # the policy finished without a grasp
    await ctx.transport.stop()
    broke = state.extras.get("policy_error")
    why = f" (the policy raised {broke})" if broke else ""
    return VerbResult.fail(
        f"pick {p.target!r} did not end with something held{why}", target=p.target, error=broke
    )


async def place(ctx: VerbContext, _: NoParams) -> VerbResult:
    position, _state, how, why = await _drive_gripper(ctx, open_=True)
    if how != "arrived":
        return VerbResult.fail(f"place: the gripper did not open: {why}", position=position)
    return VerbResult.success("placed: gripper opened where the arm is", position=position)


def lerobot_verbs(*, policy: bool) -> dict[str, Verb]:
    verbs = [
        Verb(
            "report_state",
            "Report the arm: every joint in degrees, whether torque is on, how warm the "
            "servos are, and whether anything is held.",
            report_state,
            NoParams,
            timeout_s=5,
            read_only=True,
            core=True,
        ),
        Verb(
            "move_joints",
            "Move one or more joints to goal angles in degrees (gripper 0..100). The arm's "
            "own controller does the motion, at a capped speed, and this waits for it.",
            move_joints,
            MoveJointsParams,
            timeout_s=20,
            done_condition="every joint given is within a few degrees of its goal",
        ),
        Verb("gripper", "Open or close the gripper.", gripper, GripperParams, timeout_s=10),
        Verb(
            "place",
            "Release what is held by opening the gripper where the arm is.",
            place,
            NoParams,
            timeout_s=10,
        ),
    ]
    if policy:
        verbs.append(
            Verb(
                "pick",
                "Pick the target with the arm's own learned policy. Ends when something is "
                "held or the time is up.",
                pick,
                PickParams,
                timeout_s=70,
                safety_class="confirm",
            )
        )
    return {v.name: v for v in verbs}


__all__ = [
    "GRIPPER_CLOSED",
    "GRIPPER_OPEN",
    "JOINTS",
    "GripperParams",
    "MoveJointsParams",
    "PickParams",
    "lerobot_conditions",
    "lerobot_verbs",
]
