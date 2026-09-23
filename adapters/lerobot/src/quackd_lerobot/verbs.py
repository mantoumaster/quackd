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
PICK_SETTLE_S = STALL_TICKS * TICK_S
"""After the policy stops, how long to let the gripper come to rest before reading `holding`
one last time. `_holding` needs two gripper samples at least `SETTLE_GAP_S` apart that agree,
so the read that catches the policy going idle can be too early to see a grasp that is still
closing. This is the same window `_drive` calls a stall, which is this file's own definition
of a joint that has stopped moving, and it is comfortably wider than that gap."""


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
        "the arm's torque is off, so a goal would reach a limp servo. A servo that has tripped "
        "its own overload protection reads this way too, and so does an arm quackd released "
        "into somebody's hands for `--by-hand`, which it takes hold of again before the first "
        "turn. No verb can toggle torque either way"
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


def shortfall(
    goal: dict[str, float],
    joints: dict[str, float],
    recorded: dict[str, float] | None = None,
) -> str:
    """The joint furthest from where it was asked to be, in words.

    Public because the rest move says it too, and it reads from the transport rather
    than through a verb. With the recorded rest pose beside a reachable goal, a joint that is
    at rest by the half-line rule (`joint_at_rest`) is never the one named, however far it
    reads from its goal: an arm folded past its travel is not short of anything, and naming
    it would send somebody to look at the one joint that is fine."""
    behind = {k: abs(joints[k] - v) for k, v in goal.items() if k in joints}
    if not behind:
        return "the arm reported no joint positions"
    rec = recorded or {}
    short = {
        k: gap for k, gap in behind.items() if not joint_at_rest(goal[k], joints[k], rec.get(k))
    } or behind
    worst = max(short, key=lambda joint: short[joint])
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
                f"{shortfall(goal, joints)}, and it has stopped moving",
            )
    await ctx.transport.stop()
    return joints, state, "timeout", f"{shortfall(goal, joints)} when the time ran out"


# ── the rest pose, shared by both backends ──────────────────────────────────────────────

REST_MARGIN_S = 2.0
"""Slack on top of the time the step cap says the move needs, for the reads between sends."""
REST_MIN_S = 2.0
REST_MAX_S = 30.0
"""A teardown is bounded. `QUACKD_LEROBOT_MAX_STEP_DEG` can be lowered until a long move
would take minutes, and an arm nobody is watching must not hold a run open that long."""

TORQUE_LEFT_ON = (
    "the arm is not at its rest pose ({why}), so torque was left on and it will not fall: "
    "hold the arm and cut its power, or run again"
)
"""Said once, by whichever caller closed the arm. The transport records it and prints
nothing itself: a library that writes to a terminal has picked one, and quackd has four."""


def rest_goal(rest_pose: dict[str, float]) -> dict[str, float]:
    """The part of a recorded pose that is ever driven: the five body joints.

    The gripper is left out for the reason `_hold()` leaves it out. LeRobot writes only the
    keys it is given, so omitting it keeps whatever squeeze is already commanded, and a rest
    move that re-sent the gripper would open a hand that is holding something."""
    return {j: float(v) for j, v in rest_pose.items() if j in JOINTS and j != "gripper"}


Clip = tuple[str, float, float]
"""`(joint, recorded, reachable)`: a joint of the rest pose that lies past its travel."""


def reachable_rest_goal(
    rest_pose: dict[str, float], ranges: dict[str, tuple[float, float]]
) -> tuple[dict[str, float], tuple[Clip, ...]]:
    """The rest goal the servos can actually be driven to, and the joints it had to move.

    A pose is recorded off an arm that was folded by hand with torque off, and nothing stops a
    hand folding a joint past the travel its calibration recorded. A goal is a different
    matter. LeRobot's calibration writes each joint's travel into the servo itself as its two
    position limits, and the servo clamps every goal it is written to them
    (`upstream_api.POSITION_LIMITS_CLAMP_GOALS`). So a goal past the travel is never refused
    and never reached: the servo drives to the limit and stops there, and a rest move watching
    for the recorded angle watches a joint that will not come, stalls, and calls the arm lost.

    So each body joint of `rest_goal` is clipped into its travel from `ranges`, which is what
    `joint_ranges()` read off this arm's calibration, whichever end it is past and however
    many joints are. A joint with no known range passes through unchanged, because there is
    nothing to clip it to and inventing a range would be worse. The gripper is never in it,
    for `rest_goal`'s reason.

    The second value names every joint that was moved at all, as `(joint, recorded,
    reachable)` in the pose's own order. Callers decide which of them are worth saying
    (`worth_saying`); this one only records what happened."""
    goal: dict[str, float] = {}
    clipped: list[Clip] = []
    for joint, recorded in rest_goal(rest_pose).items():
        span = ranges.get(joint)
        if span is None:
            goal[joint] = recorded
            continue
        lo, hi = span
        reachable = min(hi, max(lo, recorded))
        goal[joint] = reachable
        if reachable != recorded:
            clipped.append((joint, recorded, reachable))
    return goal, tuple(clipped)


def worth_saying(clipped: tuple[Clip, ...]) -> tuple[Clip, ...]:
    """The clipped joints a person needs to hear about: those further past their travel than
    `TOL_DEG`. A joint clipped by less is parked inside the tolerance any reached pose is
    allowed to miss by, so its fold and its parked angle are the same pose as far as the rest
    move is concerned, and a sentence about it would be a sentence about nothing. The
    half-line rule in `joint_at_rest` still applies to it."""
    return tuple(c for c in clipped if abs(c[1] - c[2]) > TOL_DEG)


def _listed(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def rest_clip_note(clipped: tuple[Clip, ...]) -> str | None:
    """What a clipped rest pose means, in the numbers of this arm, or None when nothing was.

    One sentence that the rest move's narrator, `doctor` and `quackd robot rest-pose` all say,
    so a person hears the same thing wherever they first meet it. It says what happens rather
    than what went wrong, because nothing did: the arm parks at the edge of its travel, torque
    is released there, and the joint is free to settle toward the fold on its own. And it says
    how to make the fold itself reachable, which is a calibration that saw the arm folded."""
    if not clipped:
        return None
    if len(clipped) == 1:
        joint, recorded, reachable = clipped[0]
        said = (
            f"{joint} is recorded at {recorded:.0f} in the rest pose and this calibration lets "
            f"its servo be driven to {reachable:.0f} and no further, so it parks there"
        )
    else:
        where = _listed([f"{joint} at {recorded:.0f}" for joint, recorded, _ in clipped])
        limits = _listed([f"{reachable:.0f}" for _, _, reachable in clipped])
        said = (
            f"the rest pose records {where}, and this calibration lets their servos be driven "
            f"to {limits} and no further, so each parks there"
        )
    return (
        f"{said} and is let go of there, free to settle the rest of the way on its own. "
        "Calibrate again with the arm folded (lerobot-calibrate) and record the pose again "
        "(quackd robot rest-pose NAME) to make the fold reachable"
    )


TORQUE_COULD_NOT_BE_KEPT = (
    "the arm is not at its rest pose ({why}), and quackd could not keep torque on, so the "
    "arm was released where it stood: check whether it is still where you left it"
)
"""When the one seam that holds torque did not take.

`close()` keeps an arm up by writing a flag on LeRobot's config object just before the
disconnect that reads it. If that write raises, the disconnect releases torque anyway, and
the note promising the opposite would be the worst line quackd could print: somebody reads
that the arm is being held and walks away from an arm that is not."""

LIMP_IN_HAND = (
    "the arm is limp and in your hands ({why}): put it down before you let go of it, because "
    "nothing is holding it up"
)
"""When the run ends with the arm still released into a person's hands.

`--by-hand` takes torque off at the rest pose and takes hold again before the first turn, so
the only way to reach a close in this state is a run that ended in the gap between: a Ctrl-C
during the wait, a heartbeat that died, a `take_hold` the arm refused. Whoever is holding the
arm is the one reading this, and the opposite note, the one about torque being left on, would
tell them the arm is holding itself up while it hangs off their hand."""

NO_DRIVABLE_JOINT = (
    "the recorded rest pose names no joint this arm drives ({named}). A pose is only kept "
    "for the five body joints ({drivable}), because the gripper is never re-sent: "
    "quackd robot rest-pose NAME re-reads it off the arm, or --clear forgets it"
)
"""Why a pose that survived the registry is still refused here.

The registry checks that a pose names a joint and that its angles are numbers; it does not
know this arm's motors, and nothing should teach it. So a hand-edited `robots.json` can name
`elbow` where the arm says `elbow_flex`, and that pose drives nothing. Refusing it is the
registry's own rule about a file that says something untrue: the alternative is an arm that
reports a rest pose, ignores it, and lets go where it stands."""


def drivable_rest_joints() -> tuple[str, ...]:
    """The joints a rest pose may name, for the refusal that lists them."""
    return tuple(j for j in JOINTS if j != "gripper")


def joint_at_rest(goal: float, reading: float, recorded: float | None = None) -> bool:
    """One joint of a rest pose, judged against its reachable goal.

    A joint whose recorded angle is its goal, which is every joint whose fold is inside its
    travel, is at rest within `TOL_DEG` of it, as it always was.

    A joint whose recorded angle lies past its goal was clipped (`reachable_rest_goal`), and
    for that one the rule is a half-line rather than a point: it is at rest anywhere from
    `TOL_DEG` short of its goal out past it, on the side the recorded angle lies. Below the
    floor that is `reading <= goal + TOL_DEG`, above the ceiling `reading >= goal - TOL_DEG`.
    The reason is what the servo does. It clamps every goal to its travel, so past the limit
    nothing quackd can send moves it there: the only ways a joint gets past it are settling
    with torque off and being placed there by hand, and that is what a fold is. A joint
    reading past its limit is therefore folded, never lost, and driving it "to rest" would
    haul it up to the limit and hold it there against its own weight. Which side is read off
    the sign of `recorded - goal`, so a fold past the ceiling works exactly like one past the
    floor."""
    if recorded is not None and recorded < goal:
        return reading <= goal + TOL_DEG
    if recorded is not None and recorded > goal:
        return reading >= goal - TOL_DEG
    return abs(reading - goal) <= TOL_DEG


def past_reach(goal: float, reading: float, recorded: float | None) -> bool:
    """The joint reads beyond its reachable goal, on the side its recorded angle lies.

    That goal is the servo's limit, so writing it to this joint is writing a goal the servo
    moves *away* from the fold to reach: the rest move leaves such a joint out of what it
    sends rather than haul a folded arm up. A joint whose recorded angle is its goal is never
    past it, and a joint exactly at its goal is not past it either: that goal moves nothing."""
    if recorded is None:
        return False
    return (recorded < goal and reading < goal) or (recorded > goal and reading > goal)


def at_rest(
    goal: dict[str, float],
    joints: dict[str, float],
    recorded: dict[str, float] | None = None,
) -> bool:
    """Every joint of the goal is reported, and every one of them is at rest.

    `goal` is the reachable goal and `recorded` the pose as it was recorded; without it every
    joint is judged by the point rule, which is also what a pose inside its travel gets with
    it. `joint_at_rest` is the rule. A joint the arm did not report is never at rest."""
    if not goal or any(j not in joints for j in goal):
        return False
    rec = recorded or {}
    return all(joint_at_rest(v, joints[j], rec.get(j)) for j, v in goal.items())


def rest_budget_s(distance_deg: float, step_deg: float) -> float:
    """How long to give the rest move: the travel at the step cap, plus slack, bounded.

    One `send_action` moves a joint at most the step cap and they go out every `TICK_S`, so
    the fastest the arm can cross a gap is that distance divided by that rate."""
    rate = max(step_deg, 0.01) / TICK_S
    return min(REST_MAX_S, max(REST_MIN_S, distance_deg / rate + REST_MARGIN_S))


# ── the verbs ───────────────────────────────────────────────────────────────────────────


def _past_travel(ctx: VerbContext, extras: dict[str, Any], joints: dict[str, float]) -> list[str]:
    """One clause per joint the arm reports outside its calibrated travel.

    A pilot handed a joint reading beyond the end of the travel line in its prompt has been
    handed a contradiction, and a careful one refuses to move an arm whose state it cannot
    explain: on the bench one did exactly that. The explanation is the servo's: its goals are
    clamped to the travel, and its readings are not, so an arm folded or placed with torque
    off can read past the end. The clause says that much and no more. It does not say how the
    joint got there, because `out_of_range` carries a margin, and a joint the servo parked at
    its limit that then sagged under its own weight qualifies too.

    The travel is the connected arm's, from the manifest; where it is not known the clause
    says only that the joint reads outside it."""
    ranges = (
        ctx.manifest.extras.get("joint_range_deg") if ctx.manifest is not None else None
    ) or {}
    said = []
    for name in extras.get("out_of_range") or []:
        joint = str(name)
        reading = joints.get(joint)
        if reading is None:
            continue
        span = ranges.get(joint)
        if not span:
            said.append(f"{joint} reads {reading:.0f}, outside its calibrated travel")
            continue
        lo, hi = float(span[0]), float(span[1])
        limit = lo if reading < lo else hi
        said.append(
            f"{joint} reads {reading:.0f}, past the {limit:g} its servo can be driven to; "
            "goals are still limited to its travel"
        )
    return said


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
    parts += _past_travel(ctx, extras, joints)
    # a camera earns a clause only when a read has actually failed. Not when it is merely
    # unread: `ok` is false until the first frame, and a camera that opened and has not been
    # asked yet is not news. A working one is already in every observation as detections,
    # and a dead one is otherwise silent on a run that cannot call `observe`.
    camera = extras.get("camera")
    if isinstance(camera, dict):
        # with several cameras each is named, because "CAMERA DOWN" over two views does not
        # say which eye closed, and the arm keeps working with the other one
        rows = camera.get("cameras")
        if isinstance(rows, list):
            dead = [r for r in rows if isinstance(r, dict) and r.get("error")]
            if dead:
                named = "; ".join(f"{r.get('name')}: {r['error']}" for r in dead)
                parts.append(f"CAMERA DOWN: {named}")
        elif camera.get("error"):
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

    def picked() -> VerbResult:
        return VerbResult.success(
            f"picked {p.target}", target=p.target, seconds=round(ctx.transport.now() - t0, 1)
        )

    state = await ctx.transport.get_state()
    while ctx.transport.now() - t0 < p.max_s:
        await ctx.transport.sleep(PICK_POLL_S)
        state = await ctx.transport.get_state()
        if state.holding:
            return picked()
        if not str(state.policy).startswith("policy:"):
            # The policy has stopped and its last grasp may still be closing. `holding` is
            # inferred from the gripper coming to rest short of shut, which is only knowable
            # once two readings a real interval apart agree, so the read that caught the
            # policy going idle can be one sample too early. Give it one settle and look
            # again: without this a policy that grasps and finishes inside a single poll is
            # reported as a failed pick while the object is in the jaws.
            await ctx.transport.sleep(PICK_SETTLE_S)
            state = await ctx.transport.get_state()
            if state.holding:
                return picked()
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
