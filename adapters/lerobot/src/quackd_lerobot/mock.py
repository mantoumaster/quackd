"""An arm that does exactly what the test tells it to.

`LeRobotMock` is a six-joint arm in memory: joint goals land instantly, the gripper closes
on an object when the arm is near it, a scripted "policy" answers `pick`, and the camera
is a synthetic frame with an orange disc that slides as the shoulder pans. Enough to run
every arm verb, the executor's gates and the detector offline.

Two things here exist to mirror the real backend rather than to be realistic. A goal
outside a joint's range is refused in the same words, because on an arm LeRobot writes an
unclamped degrees goal straight to the servo. And a gripper that closes on the object stops
short of shut, because that is the whole of what a real arm knows about holding something:
there is no force sensor anywhere on this body.
"""

from __future__ import annotations

import math
from typing import Any

from PIL import Image, ImageDraw

from quackd.adapters.base import HandResult, RestResult
from quackd.sim2d.render import BALL, FLOOR, HORIZON, SKY, focal_px
from quackd.transport.base import Ack, DuckState, Intent
from quackd.transport.mock import MockTransport
from quackd_lerobot.verbs import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    IN_HAND_NOT_MOVED,
    JOINTS,
    LET_GO_TO_PLACE,
    LET_GO_WHERE_IT_STOOD,
    LIMP_AT_REST,
    LIMP_IN_HAND,
    TOL_DEG,
    TORQUE_KEPT_AFTER_REFUSAL,
    UNCONFIRMED_IN_HAND,
    UNREAD_IN_HAND,
    Clip,
    at_rest,
    past_reach,
    placed_past_travel,
    range_refusal,
    reachable_rest_goal,
    released_by_the_close,
    rest_clip_note,
    rest_goal,
    shortfall,
    still_holding_in_hand,
    torque_left_on,
    worth_saying,
)

REST = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -90.0,
    "elbow_flex": 90.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": GRIPPER_OPEN,
}
OBJECT_AT = {"shoulder_pan": 30.0, "shoulder_lift": -20.0, "elbow_flex": 40.0}
"""Where the object is, in joint space: the arm is "there" when these three are close."""
NEAR_DEG = 10.0
MOCK_CAM_HEIGHT_M = 0.3
MOCK_OBJECT_R = 0.03

RANGE_DEG = 100.0
"""A plausible travel for a calibrated body joint. A real one is read off the arm's own
calibration file at connect and is different on every arm."""
FULL_TURN_DEG = 180.0
"""`wrist_roll` is the joint LeRobot's calibration leaves as a full turn."""
MOCK_RANGES: dict[str, tuple[float, float]] = {
    joint: (
        (GRIPPER_CLOSED, GRIPPER_OPEN)
        if joint == "gripper"
        else (-FULL_TURN_DEG, FULL_TURN_DEG)
        if joint == "wrist_roll"
        else (-RANGE_DEG, RANGE_DEG)
    )
    for joint in JOINTS
}

GRIP_ON_OBJECT = 30.0
"""Where the gripper stops when something is between the jaws."""
MOCK_TEMPERATURE_C = 30.0
"""A fixed reading, so the heat refusal has something to refuse offline. A mock has no
servos and this number measures nothing."""


class LeRobotMock(MockTransport):
    name = "mock"
    mobility = "none"
    camera_available = True
    policy_available = True

    def __init__(
        self,
        *,
        object_bearing_deg: float | None = 30.0,
        object_distance_m: float = 0.35,
        frame_size: int = 128,
        fail_heartbeat_after: int | None = None,
        refuse_kinds: set[str] | None = None,
        hot_joints: tuple[str, ...] = (),
        rest_pose: dict[str, float] | None = None,
        rest_fails: str | None = None,
        registered_name: str | None = None,
    ) -> None:
        super().__init__(
            states=[DuckState(policy="idle", posture="unknown", battery_percent=None)],
            fail_heartbeat_after=fail_heartbeat_after,
            refuse_kinds=refuse_kinds,
            frame_size=(frame_size, frame_size),
        )
        self.joints: dict[str, float] = dict(REST)
        self.joint_range_deg = dict(MOCK_RANGES)
        self.holding = False
        self.torque = True
        self.policy = "idle"
        self.object_bearing_deg = object_bearing_deg
        self.object_distance_m = object_distance_m
        self.actions: list[dict[str, float]] = []
        self.policy_runs: list[str] = []
        self.rest_pose = dict(rest_pose) if rest_pose else None
        self.rest_fails = rest_fails
        """Set to a reason and the rest move stalls without moving, which is the one thing
        an offline arm cannot do to itself and every caller of the rest move has to handle."""
        self.registered_name = registered_name
        """The name the close note gives the commands it names, as on the real backend."""
        self.close_note: str | None = None
        self.in_hand = False
        """The arm is limp because `let_go()` put it there, as on the real backend."""
        self.let_go_why: str | None = None
        """What the close says the arm was let go of for: the real backend's `_let_go_why`."""
        self.hold_slips: dict[str, float] | None = None
        """Set to joint offsets and `take_hold` finds the arm somewhere else than where it
        was read, which is how an offline arm stands in for one that moved as torque came on."""
        self.release_holdouts: tuple[str, ...] = ()
        """Motors that take the release and keep their torque anyway, as the real backend's
        read-back can find them: some of them is an arm limp in part and still in a hand, all
        of them is a release that was refused. A rehearsal of either ending says what the arm
        would, which an in-memory release that always takes cannot."""
        self.holding_in_hand: tuple[str, ...] = ()
        """The motors the last release left on, in the bus's order, for the close to name."""
        self.release_unread: str | None = None
        """Set to a reason and the release goes out with nothing to read it back, as on the real
        backend when its read-back fails, and that reason is why the read did not answer. The
        arm is in a hand, its torque is unknown, and the close says so in the real close's words
        (`UNREAD_IN_HAND`), which an in-memory release that always reads back cannot.

        That is the real close's ending only where no read of its own answers either. On an arm
        with a rest pose the real close reads the arm before it decides, and when that read
        answers it says what it found, `LIMP_IN_HAND` or the joints still on by name, rather
        than that nothing read the release back. This close takes no read of its own, so what
        it rehearses is the arm with no rest pose, or one whose every read after the release
        went unanswered."""
        self.release_read_back = False
        """A read of the torque register answered for the last torque write, the release's or a
        take-hold's, as on the real backend: the close names what it found only then."""
        self.hold_unread: str | None = None
        """Set to a reason and `take_hold` sends its torque write with nothing to read it back,
        as on the real backend when the torque register does not answer after it, and that
        reason is why. The arm is still in a hand and may be energised, so the take-hold is
        refused with `energised=None`, and the close says quackd cannot tell which
        (`UNCONFIRMED_IN_HAND`). The in-memory arm does take the write, since a servo that
        answers nothing may well have: the rehearsal of this ending is of an arm that is
        energised while quackd says it does not know."""
        self.hold_refusal: HandResult | None = None
        """The take-hold refused since the last release that left the arm in a hand: the real
        backend's `_refused_hold`. The stop takes hold of an arm in a hand only while this is
        None, and the close reads its `energised`."""
        self.release_refused = False
        """The last release a person asked for through the second door was refused, as on the
        real backend: the close then does not send them back to it."""
        self.sequence: list[str] = []
        """`stop`, `rest` and `close` in the order they were called. A run's teardown is an
        order as much as a set, and this is what a test reads to check it."""
        self.temperature_c = {joint: MOCK_TEMPERATURE_C for joint in JOINTS}
        for joint in hot_joints:
            self.temperature_c[joint] = 65.0

    # ── state and camera ────────────────────────────────────────────────────────────

    def _near_object(self) -> bool:
        return all(abs(self.joints[k] - v) <= NEAR_DEG for k, v in OBJECT_AT.items())

    @property
    def hot_joints(self) -> list[str]:
        return sorted(k for k, v in self.temperature_c.items() if v >= 60.0)

    async def get_state(self) -> DuckState:
        state = await super().get_state()
        return state.model_copy(
            update={
                "policy": self.policy,
                "holding": self.holding,
                "extras": {
                    "joints": {k: round(v, 1) for k, v in self.joints.items()},
                    "torque": self.torque,
                    "temperature_c": {k: round(v) for k, v in self.temperature_c.items()},
                    "hot": self.hot_joints,
                    "out_of_range": [],
                    "near_object": self._near_object(),
                },
            }
        )

    async def get_frame(self) -> Image.Image | None:
        size = self._frame_size[0]
        img = Image.new("RGB", (size, size), SKY)
        draw = ImageDraw.Draw(img)
        horizon = int(size * HORIZON)
        draw.rectangle([0, horizon, size, size], fill=FLOOR)
        if self.object_bearing_deg is None or self.holding:
            return img  # nothing on the table, or it is in the gripper
        rel = math.radians(self.object_bearing_deg - self.joints["shoulder_pan"])
        if abs(rel) > math.radians(45):
            return img
        f = focal_px(size)
        cx = size / 2 - math.tan(rel) * f
        ground_y = horizon + f * MOCK_CAM_HEIGHT_M / self.object_distance_m
        r = f * MOCK_OBJECT_R / self.object_distance_m
        draw.ellipse([cx - r, ground_y - 2 * r, cx + r, ground_y], fill=BALL)
        return img

    # ── intents ─────────────────────────────────────────────────────────────────────

    def _refuse_out_of_range(self, goals: dict[str, float]) -> str | None:
        """The real backend's refusal, in the same one sentence (`range_refusal`)."""
        return range_refusal(goals, self.joint_range_deg)

    @property
    def rest_reachable(self) -> dict[str, float]:
        """The rest goal clipped into `MOCK_RANGES`, the real backend's rule on the mock's own
        travel. `_goto` clamps a goal the way the servo does, so without this a pose recorded
        past the travel would stall offline exactly as it did on the bench."""
        return reachable_rest_goal(self.rest_pose or {}, self.joint_range_deg)[0]

    @property
    def rest_clipped(self) -> tuple[Clip, ...]:
        """Every joint the recorded pose puts past `MOCK_RANGES`, as the real backend names
        them, so the manifest of a rehearsal carries the same key as the arm's."""
        return reachable_rest_goal(self.rest_pose or {}, self.joint_range_deg)[1]

    def _outside_travel(self, joint: str, reading: float) -> bool:
        span = self.joint_range_deg.get(joint)
        return span is not None and not span[0] <= reading <= span[1]

    def _goto(self, goals: dict[str, float]) -> None:
        for joint, goal in goals.items():
            if joint in JOINTS:
                lo, hi = self.joint_range_deg[joint]
                self.joints[joint] = min(hi, max(lo, float(goal)))
        self.actions.append(dict(goals))

    def _set_gripper(self, open_: bool) -> None:
        if open_:
            self._goto({"gripper": GRIPPER_OPEN})
            self.holding = False
            return
        # a gripper that closes on something stops where the something is
        self.holding = self._near_object()
        self._goto({"gripper": GRIP_ON_OBJECT if self.holding else GRIPPER_CLOSED})

    async def send_intent(self, intent: Intent) -> Ack:
        ack = await super().send_intent(intent)
        if not ack.accepted:
            return ack
        p = intent.params
        match intent.kind:
            case "joint":
                if not self.torque:
                    return Ack(accepted=False, reason="torque is off")
                goals = {str(k): float(v) for k, v in dict(p.get("positions", {})).items()}
                if (refusal := self._refuse_out_of_range(goals)) is not None:
                    return Ack(accepted=False, reason=refusal)
                self._goto(goals)
            case "gripper":
                self._set_gripper(bool(p.get("open", True)))
            case "do":
                kind, _, rest = str(p.get("skill")).partition(":")
                if kind != "policy":
                    return Ack(accepted=False, reason=f"unknown skill {p.get('skill')!r}")
                name, _, task = rest.partition(":")
                if name != "pick":
                    return Ack(accepted=False, reason=f"no policy named {name!r}")
                # the scripted policy: go to the object and close on it
                self.policy = f"policy:pick:{task}"
                self.policy_runs.append(task)
                self._goto(dict(OBJECT_AT))
                self._set_gripper(False)
                self.policy = "idle"
            case "stop":
                self.policy = "idle"
            case "move":
                return Ack(accepted=False, reason="an arm cannot drive")
            case "enable":
                if not p.get("on", True):
                    return Ack(accepted=False, reason="quackd never limps a robot")
            case _:
                return Ack(accepted=False, reason=f"an arm cannot {intent.kind}")
        return ack

    async def stop(self) -> None:
        if self.in_hand and self.hold_refusal is None:
            # the real `_hold()`'s order: an arm somebody is holding is picked up before it is
            # told to stay where it is, because a goal to a limp servo stops nothing. Only
            # while no take-hold has been refused since the release: after one, the real stop
            # sends the arm nothing and takes no second hold, and neither does this
            await self.take_hold()
        await super().stop()
        self.sequence.append("stop")
        self.policy = "idle"

    async def go_to_rest(self) -> RestResult:
        """The real arm's rest move, in memory: goals land at once, so it either is there
        already, gets there in one action, or was told to fail.

        The same reachable goal and the same half-line rule as the real backend, and arrival
        is judged rather than assumed: `_goto` clamps like the servo, so a goal it could not
        land is one this reads back and reports, the way the arm's own rest move would.

        An arm in somebody's hands is moved by nothing here, as on the arm: `already` where it
        is at its rest pose, refused in the real backend's words otherwise. It used to stall
        over a limp arm and say "it has stopped moving" of an arm that never moved, to a
        person just told to keep hold of it, and the real rest move wrote goals into its limp
        servos meanwhile."""
        self.sequence.append("rest")
        if self.rest_pose is None:
            return RestResult.none("no rest pose is recorded for this arm")
        recorded = rest_goal(self.rest_pose)
        goal = self.rest_reachable
        if not goal:
            # the real arm's answer, for the same reason: a pose that drives nothing is a
            # reason to keep holding, not a reason to behave as though none was recorded
            return RestResult("refused", "the recorded pose names no joint this arm drives")
        if self.in_hand:
            result = (
                RestResult("already", "already at the rest pose")
                if at_rest(goal, self.joints, recorded)
                else RestResult("refused", IN_HAND_NOT_MOVED)
            )
        elif self.rest_fails is not None:
            result = RestResult("stalled", self.rest_fails)
        elif at_rest(goal, self.joints, recorded):
            result = RestResult("already", "already at the rest pose")
        else:
            # a joint folded past its limit is left out, as on the arm: the limit is the one
            # goal it would take, and that goal hauls it up out of its fold
            self._goto(
                {
                    j: v
                    for j, v in goal.items()
                    if not past_reach(v, self.joints.get(j, v), recorded.get(j))
                }
            )
            if at_rest(goal, self.joints, recorded):
                result = RestResult("arrived", "moved to the rest pose")
            else:
                why = shortfall(goal, self.joints, recorded)
                result = RestResult("stalled", f"{why}, and it has stopped moving")
        if clipped := worth_saying(self.rest_clipped):
            note = rest_clip_note(clipped, self.registered_name) if result.reached else None
            result = RestResult(result.how, result.reason, clipped, note)
        return result

    async def let_go(self, *, anywhere: bool = False) -> HandResult:
        """Torque off for a person to place the arm, refused wherever the real one refuses.

        `anywhere` is the real backend's second door, opened the same way: the two refusals
        about the pose are skipped and the arm is released where it stands, in the same words,
        so a rehearsal of `quackd robot release` or of the end-of-run offer says what the arm
        would. An in-memory release takes wherever `release_holdouts` does not say otherwise,
        and those motors read on afterwards in the real backend's words, and is read back
        unless `release_unread` says the read did not answer.

        `release_refused` is set where a refusal is returned and nowhere else, as on the real
        backend, whose release an interrupt can land on before anything is sent."""
        self.sequence.append("let_go")

        def refused(reason: str, **kw: Any) -> HandResult:
            self.release_refused = anywhere
            return HandResult("refused", reason, **kw)

        if not anywhere and self.rest_pose is None:
            return refused(
                "no rest pose is recorded for this arm, so there is nowhere it is known to be "
                "safe to let go of it: quackd robot rest-pose NAME",
            )
        recorded = rest_goal(self.rest_pose or {})
        goal = self.rest_reachable
        if not anywhere and not goal:
            return refused("the recorded pose names no joint this arm drives")
        resting = bool(goal) and at_rest(goal, self.joints, recorded)
        if not anywhere and not resting:
            return refused(
                f"the arm is not at its rest pose ({shortfall(goal, self.joints, recorded)}), "
                "and an arm held up by torque alone falls when torque goes",
            )
        where = "at the rest pose" if resting else "where the arm stands"
        holding = tuple(j for j in JOINTS if j in self.release_holdouts)
        if holding == JOINTS and self.release_unread is None:
            # every motor kept its torque, so nothing was released and nobody holds anything
            return refused(
                "the arm still reports torque on, so it was not released",
                joints=dict(self.joints),
                torque_on=holding,
            )
        self.release_refused = False
        self.torque = holding == JOINTS
        self.in_hand = True
        self.hold_refusal = None
        self.let_go_why = LET_GO_WHERE_IT_STOOD if anywhere else None
        if self.release_unread is not None:
            # sent, and nothing read it back: the real backend's words for a read-back whose
            # torque register did not answer, and what the motors did is not known
            self.release_read_back = False
            self.holding_in_hand = ()
            return HandResult(
                "released",
                f"torque is off {where}, and the torque register did not answer to confirm "
                f"it ({self.release_unread})",
                joints=dict(self.joints),
            )
        self.release_read_back = True
        self.holding_in_hand = holding
        if holding:
            return HandResult(
                "released",
                f"torque is off {where} except on {', '.join(holding)}, which still read on",
                joints=dict(self.joints),
                torque_on=holding,
            )
        return HandResult(
            "released", f"torque is off {where}", joints=dict(self.joints), torque_on=()
        )

    async def take_hold(self) -> HandResult:
        """Hold wherever a test left the joints, and record the goal that pins them there.

        Refused, in the real backend's words and before anything is written, when a body joint
        was placed outside `MOCK_RANGES`: torque stays off and the arm stays in a hand, because
        on the arm neither a goal written there (pulled to the limit) nor none (the last goal
        the servo had) keeps that joint where it was put. A rehearsal that took hold of it
        would teach the one ending the arm refuses.

        Every result says whether it may have left torque on (`HandResult.energised`), as the
        real one does, and a refusal that leaves the arm in a hand is kept (`hold_refusal`)."""
        self.sequence.append("take_hold")
        held = self._take_hold()
        self.hold_refusal = held if not held.ok and self.in_hand else None
        return held

    def _take_hold(self) -> HandResult:
        placed = dict(self.joints)
        outside = {
            j: v
            for j, v in placed.items()
            if j in JOINTS and j != "gripper" and self._outside_travel(j, v)
        }
        if outside:
            return HandResult(
                "refused",
                placed_past_travel(outside, self.joint_range_deg),
                joints=placed,
                energised=False,
            )
        # the real backend writes the present position as the goal before torque comes on and
        # again after, and an in-memory arm is already exactly where it is told to be
        self._goto({j: v for j, v in placed.items() if j in JOINTS})
        self.torque = True
        if self.hold_unread is not None:
            # the torque write went out and no read came back: the real backend's words, the
            # arm still taken to be in a hand, and no read from before the write speaking for it
            self.release_read_back = False
            return HandResult(
                "refused",
                f"the arm did not say whether torque came back on ({self.hold_unread}), and a "
                "hold nothing confirmed is not a hold",
                energised=None,
            )
        # torque is on, so the arm holds itself whatever else went wrong: the real backend
        # clears this here and for the same reason, before it judges the pose
        self.in_hand = False
        if self.hold_slips:
            for joint, gap in self.hold_slips.items():
                self.joints[joint] = self.joints.get(joint, 0.0) + gap
            worst = max(self.hold_slips, key=lambda j: abs(self.hold_slips[j]))  # type: ignore[index]
            if abs(self.hold_slips[worst]) > TOL_DEG:
                return HandResult(
                    "refused",
                    f"the arm moved as torque came on ({worst} by "
                    f"{abs(self.hold_slips[worst]):.0f} degrees), so it is not holding the "
                    "pose you set; it is holding where it is now",
                    joints=dict(self.joints),
                    energised=True,
                )
        return HandResult(
            "held", "holding the pose you set", joints=dict(self.joints), energised=True
        )

    async def close(self) -> None:
        """Torque drops only where the arm can be let go of, as it does on a real one: at the
        reachable rest pose or past it on a clipped joint's side, with nothing to say about it.

        And the real close's words for the endings a person drove: an arm in a hand that a
        release left partly energised names what still holds, one whose release nothing read
        back says so and to cut its power to be sure, one whose take-hold sent its torque write
        with nothing to read it back says quackd cannot tell whether it has torque, one the
        placing release left at its rest pose is limp there rather than in anybody's hands, and
        an arm whose release was just refused is neither sent back to that release nor let go
        of without a word."""
        self.sequence.append("close")
        self.close_note = None
        recorded = rest_goal(self.rest_pose or {})
        goal = self.rest_reachable
        if self.in_hand:
            limp = self.let_go_why or LET_GO_TO_PLACE
            wrote = self.hold_refusal is not None and self.hold_refusal.energised is not False
            if not self.release_read_back:
                self.close_note = UNCONFIRMED_IN_HAND if wrote else UNREAD_IN_HAND.format(why=limp)
            elif self.holding_in_hand:
                self.close_note = still_holding_in_hand(self.holding_in_hand, limp)
            elif goal and at_rest(goal, self.joints, recorded) and not self.let_go_why:
                self.close_note = LIMP_AT_REST
            else:
                self.close_note = LIMP_IN_HAND.format(why=limp)
        elif self.rest_pose and not goal:
            self.close_note = torque_left_on(
                "the recorded pose names no joint this arm drives", self.registered_name
            )
        elif goal and not at_rest(goal, self.joints, recorded):
            why = shortfall(goal, self.joints, recorded)
            self.close_note = (
                TORQUE_KEPT_AFTER_REFUSAL.format(why=why)
                if self.release_refused
                else torque_left_on(why, self.registered_name)
            )
        else:
            self.torque = False
            if self.release_refused:
                self.close_note = released_by_the_close(at_rest=self.rest_pose is not None)
        await super().close()
