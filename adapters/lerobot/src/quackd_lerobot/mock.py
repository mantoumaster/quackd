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

from PIL import Image, ImageDraw

from quackd.adapters.base import HandResult, RestResult
from quackd.sim2d.render import BALL, FLOOR, HORIZON, SKY, focal_px
from quackd.transport.base import Ack, DuckState, Intent
from quackd.transport.mock import MockTransport
from quackd_lerobot.verbs import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    JOINTS,
    LIMP_IN_HAND,
    TOL_DEG,
    TORQUE_LEFT_ON,
    at_rest,
    rest_goal,
    shortfall,
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
        self.close_note: str | None = None
        self.in_hand = False
        """The arm is limp because `let_go()` put it there, as on the real backend."""
        self.hold_slips: dict[str, float] | None = None
        """Set to joint offsets and `take_hold` finds the arm somewhere else than where it
        was read, which is how an offline arm stands in for one that moved as torque came on."""
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
        for joint, goal in sorted(goals.items()):
            span = self.joint_range_deg.get(joint)
            if span is None:
                continue
            if not span[0] <= float(goal) <= span[1]:
                return (
                    f"{joint}={float(goal):.0f} is outside this arm's calibrated range "
                    f"{span[0]:.0f}..{span[1]:.0f}; LeRobot does not clamp a degrees goal, "
                    "so quackd refuses it"
                )
        return None

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
        if self.in_hand:
            # the real `_hold()`'s order: an arm somebody is holding is picked up before it is
            # told to stay where it is, because a goal to a limp servo stops nothing
            await self.take_hold()
        await super().stop()
        self.sequence.append("stop")
        self.policy = "idle"

    async def go_to_rest(self) -> RestResult:
        """The real arm's rest move, in memory: goals land at once, so it either is there
        already, gets there in one action, or was told to fail."""
        self.sequence.append("rest")
        if self.rest_pose is None:
            return RestResult.none("no rest pose is recorded for this arm")
        goal = rest_goal(self.rest_pose)
        if not goal:
            # the real arm's answer, for the same reason: a pose that drives nothing is a
            # reason to keep holding, not a reason to behave as though none was recorded
            return RestResult("refused", "the recorded pose names no joint this arm drives")
        if self.rest_fails is not None:
            return RestResult("stalled", self.rest_fails)
        if at_rest(goal, self.joints):
            return RestResult("already", "already at the rest pose")
        self._goto(goal)
        return RestResult("arrived", "moved to the rest pose")

    async def let_go(self) -> HandResult:
        """Torque off for a person to place the arm, refused wherever the real one refuses."""
        self.sequence.append("let_go")
        if self.rest_pose is None:
            return HandResult(
                "refused",
                "no rest pose is recorded for this arm, so there is nowhere it is known to be "
                "safe to let go of it: quackd robot rest-pose NAME",
            )
        goal = rest_goal(self.rest_pose)
        if not goal:
            return HandResult("refused", "the recorded pose names no joint this arm drives")
        if not at_rest(goal, self.joints):
            return HandResult(
                "refused",
                f"the arm is not at its rest pose ({shortfall(goal, self.joints)}), and an "
                "arm held up by torque alone falls when torque goes",
            )
        self.torque = False
        self.in_hand = True
        return HandResult("released", "torque is off at the rest pose", joints=dict(self.joints))

    async def take_hold(self) -> HandResult:
        """Hold wherever a test left the joints, and record the goal that pins them there."""
        self.sequence.append("take_hold")
        placed = dict(self.joints)
        # the real backend writes the present position as the goal before torque comes on and
        # again after, and an in-memory arm is already exactly where it is told to be
        self._goto({j: v for j, v in placed.items() if j in JOINTS})
        self.torque = True
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
                )
        return HandResult("held", "holding the pose you set", joints=dict(self.joints))

    async def close(self) -> None:
        """Torque drops only where the arm can be let go of, as it does on a real one."""
        self.sequence.append("close")
        self.close_note = None
        goal = rest_goal(self.rest_pose or {})
        if self.in_hand:
            self.close_note = LIMP_IN_HAND.format(
                why="it was let go of for you to place and never taken hold of again"
            )
        elif self.rest_pose and not goal:
            self.close_note = TORQUE_LEFT_ON.format(
                why="the recorded pose names no joint this arm drives"
            )
        elif goal and not at_rest(goal, self.joints):
            self.close_note = TORQUE_LEFT_ON.format(why=shortfall(goal, self.joints))
        else:
            self.torque = False
        await super().close()
