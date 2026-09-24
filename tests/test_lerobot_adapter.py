"""The LeRobot adapter: an arm with joints and a gripper, and nothing a duck has."""

from __future__ import annotations

import asyncio
import functools
import importlib.util
import itertools
import logging
import math
import re
import threading
import time
from collections.abc import Mapping, Sequence
from pprint import pformat
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from quackd.adapters.base import AdapterError, AdapterNotInstalled, RobotAdapter
from quackd.adapters.factory import RobotSpec, describe, make_adapter, parse_robot_spec
from quackd.duckfile.parser import load_duck, parse_duck_text
from quackd.duckfile.validate import validate_duck
from quackd.perception.color_blob import ColorBlobDetector
from quackd.safety import ConfirmDenied, Executor, VerbNotAllowed, allow_all, deny_all
from quackd.transport.base import HeartbeatError, Intent, TransportError, primary_of
from quackd.verbs.registry import registry_from_manifest
from quackd_lerobot import JOINTS, LeRobotAdapter, lerobot_manifest, make, published_travel
from quackd_lerobot.mock import GRIP_ON_OBJECT, MOCK_RANGES, REST, LeRobotMock
from quackd_lerobot.real import (
    CONNECT_ATTEMPTS,
    ENCODER_TICKS,
    MAX_STEP_DEG,
    SPLIT_TORQUE,
    STEP_ENV,
    TORQUE_RETRIES,
    LeRobotReal,
    check_port,
    joint_ranges,
    load_policy,
    motor_in_error,
    parse_camera_url,
    parse_camera_urls,
    step_from_env,
)
from quackd_lerobot.verbs import (
    HOLD_NOT_CONFIRMED,
    IN_HAND_NOT_MOVED,
    LET_GO_TO_PLACE,
    LET_GO_WHERE_IT_STOOD,
    LIMP_AT_REST,
    LIMP_IN_HAND,
    MOVE_HEADROOM_S,
    MOVE_JOINTS_TIMEOUT_S,
    MOVE_MAX_S,
    MOVE_MIN_S,
    MOVE_SETTLE_S,
    RAMP_DECIMALS,
    RAMP_RESOLUTION,
    REST_MAX_S,
    REST_MIN_S,
    STALL_TICKS,
    TICK_S,
    TOL_DEG,
    TORQUE_LEFT_ON,
    UNCONFIRMED_IN_HAND,
    UNREAD_IN_HAND,
    MoveJointsParams,
    at_rest,
    lerobot_verbs,
    move_budget_s,
    placed_past_travel,
    ramp_target,
    range_refusal,
    reachable_rest_goal,
    rest_budget_s,
    rest_goal,
    still_holding_in_hand,
    worth_saying,
)

ARM_VERBS = {"observe", "report_state", "stop", "move_joints", "gripper", "place", "pick"}
DUCK_ONLY = {"move", "walk", "go_to", "walk_to", "search_scan", "say", "gaze", "kick", "sit"}
ARM_DUCK = parse_duck_text(
    "---\nduck: 1\nname: arm\ndescription: d\nrequires: [move_joints, gripper]\nverbs:\n"
    "  allow: [observe, report_state, stop, move_joints, gripper, place, pick]\n"
    "  confirm: [pick]\nsuccess: [x]\n---\n# Task\nx\n"
)
NO_LEROBOT = importlib.util.find_spec("lerobot") is None


def test_manifest_is_an_arm_with_no_duck_verbs() -> None:
    m = lerobot_manifest("mock", camera=True, policy=True)
    assert m.id == "arm-01" and m.model == "lerobot-so101" and m.vendor == "huggingface"
    assert m.embodiment == "arm" and m.mobility == "none"
    assert set(m.intents) == {"joint", "gripper", "skill"} and "twist" not in m.intents
    assert set(m.verb_names()) == ARM_VERBS
    assert not any(m.provides(v) for v in DUCK_ONLY)
    assert m.verb("pick") is not None and m.verb("pick").safety_class == "confirm"
    # not_hot guards the five joints LeRobot writes no torque cap for; the gripper has its
    # own caps, and refusing to open a hot one would strand whatever it is holding
    assert m.preconditions == {
        "move_joints": ["torque_on", "not_hot"],
        "place": ["holding"],
        "pick": ["torque_on", "not_hot"],
    }
    assert m.safety_authority.native == "torque_limit" and not m.safety_authority.deadman
    assert m.extras["joints"] == list(JOINTS)
    assert m.extras["torque_limit_scope"] == "gripper_only"
    assert describe(parse_robot_spec("lerobot:mock")) == m
    # the static manifest of a real arm claims neither a camera nor a policy, and no joint
    # ranges either: they are read off the arm's own calibration file at connect
    real = describe(parse_robot_spec("lerobot:real"))
    assert set(real.verb_names()) == {"report_state", "stop", "move_joints", "gripper", "place"}
    assert set(real.intents) == {"joint", "gripper"} and real.sensors == ["joint_state"]
    assert "joint_range_deg" not in real.extras and "step_deg" not in real.limits
    assert real.digest() != m.digest()


def test_the_datasheet_declines_to_guess_a_mass() -> None:
    """Vendor listings put this arm anywhere from 0.8 to 2.5 kg and nobody official says.
    A figure nobody published is listed as not published, which is the whole rule."""
    sheet = lerobot_manifest("mock").datasheet
    assert sheet is not None and sheet.mass_kg is None
    assert "mass" in sheet.unknown()
    assert sheet.payload_kg is not None and sheet.payload_kg.value == 0.5
    assert any("0.8 to 2.5 kg" in note for note in sheet.notes)


def test_the_reach_is_read_off_the_makers_urdf_and_says_so() -> None:
    """Nobody publishes a reach for the SO-101, and the sheet said so, which told the pilot to
    decline whatever turned on reaching: every task an arm has. The maker's URDF gives every
    link, so the reach is quackd's arithmetic on the maker's file, and it is labelled as that:
    an estimate, the file named, what was summed in the note. The cannot line that used to say
    "the reach is not published" states the figure itself, so the two cannot disagree.

    The payload line used to forbid "nothing whose weight is not known", which is nearly every
    object a task names. It keeps the limit and gives the pilot something to judge by."""
    sheet = lerobot_manifest("mock").datasheet
    assert sheet is not None and "reach" not in sheet.unknown()
    reach = sheet.reach_m
    assert reach is not None and reach.confidence == "estimate"
    assert "TheRobotStudio/SO-ARM100" in reach.source and "so101_new_calib.urdf" in reach.source
    assert "gripper frame" in reach.note and "rounded down" in reach.note
    lines = [line for line in sheet.cannot if line.startswith("reach ")]
    assert len(lines) == 1 and f"{reach.value:g} m" in lines[0]
    assert not any("not published" in line or "not known" in line for line in sheet.cannot)
    payload = next(line for line in sheet.cannot if line.startswith("lift or hold"))
    assert "half a kilogram" in payload and "a pen" in payload


def test_registry_from_the_manifest_has_joints_not_legs() -> None:
    adapter = LeRobotAdapter(LeRobotMock())
    registry = registry_from_manifest(lerobot_manifest("mock", camera=True, policy=True), adapter)
    assert set(registry.names()) == ARM_VERBS
    assert "move" not in registry and "walk" not in registry and "get_frame" in registry
    schema = registry.get("move_joints").tool_schema()["input_schema"]
    assert "positions" in schema["properties"] and "duration_s" in schema["properties"]


def _executor(adapter: LeRobotAdapter, manifest: Any, **kwargs: Any) -> Executor:
    return Executor(registry_from_manifest(manifest, adapter), adapter, manifest=manifest, **kwargs)


async def test_mock_arm_runs_every_verb_through_the_executor() -> None:
    adapter = LeRobotAdapter(LeRobotMock())
    assert isinstance(adapter, RobotAdapter)
    manifest = await adapter.connect()
    ex = Executor(
        registry_from_manifest(manifest, adapter),
        adapter,
        contract=ARM_DUCK.frontmatter,
        detector=ColorBlobDetector(),
        confirm=allow_all,
        manifest=manifest,
    )
    mock = adapter.transport
    assert isinstance(mock, LeRobotMock)
    assert "ball" in (await ex.run_verb("observe")).summary  # the object is in view at rest
    moved = await ex.run_verb("move_joints", {"positions": {"shoulder_pan": 30}, "duration_s": 2})
    assert moved.ok, moved.summary
    assert mock.joints["shoulder_pan"] == 30.0
    assert moved.data["joints"]["shoulder_pan"] == 30.0
    bad = await ex.run_verb("move_joints", {"positions": {"tail": 5}})
    assert not bad.ok and "unknown joints" in bad.summary
    far = await ex.run_verb("move_joints", {"positions": {"elbow_flex": 400}})
    assert not far.ok
    # nothing is held yet: place is refused by its precondition, closing far away holds nothing
    nothing = await ex.run_verb("place")
    assert not nothing.ok and "nothing is held" in nothing.summary
    closed = await ex.run_verb("gripper", {"open": False})
    assert closed.ok and closed.data["holding"] is False and "on nothing" in closed.summary
    # pick is one skill intent: the scripted policy goes there and grasps
    picked = await ex.run_verb("pick", {"target": "cup", "max_s": 5})
    assert picked.ok, picked.summary
    assert mock.policy_runs == ["cup"] and (await adapter.get_state()).holding
    assert "ball" not in (await ex.run_verb("observe")).summary  # it is in the gripper now
    placed = await ex.run_verb("place")
    assert placed.ok and not (await adapter.get_state()).holding
    with pytest.raises(VerbNotAllowed):
        await ex.run_verb("move", {"vx": 0.1})
    assert not (await adapter.send_intent(Intent.move(0.1, 0.0, 0.0))).accepted
    assert not (await adapter.send_intent(Intent.enable(False))).accepted
    health = await adapter.health()
    assert health.ok and health.battery_percent is None and health.extras["holding"] is False


async def test_the_mock_refuses_a_goal_outside_the_calibrated_range() -> None:
    """The schema bound is plus or minus 180; the arm's own travel is narrower, and on a
    real arm LeRobot writes an unclamped degrees goal straight to the servo."""
    mock = LeRobotMock()
    adapter = LeRobotAdapter(mock)
    manifest = await adapter.connect()
    ex = _executor(adapter, manifest)
    out = await ex.run_verb("move_joints", {"positions": {"shoulder_pan": 150}})
    assert not out.ok and "calibrated range" in out.summary and "-100.0..100.0" in out.summary
    assert out.summary == f"move_joints: {mock._refuse_out_of_range({'shoulder_pan': 150.0})}"
    turn = await ex.run_verb("move_joints", {"positions": {"wrist_roll": 150}, "duration_s": 2})
    assert turn.ok, turn.summary  # wrist_roll is the full-turn joint


async def test_the_mock_gripper_stops_on_the_object_and_says_where() -> None:
    mock = LeRobotMock()
    adapter = LeRobotAdapter(mock)
    manifest = await adapter.connect()
    ex = _executor(adapter, manifest)
    assert (await ex.run_verb("move_joints", {"positions": dict(mock.joints) | {}})).ok is not None
    await ex.run_verb("move_joints", {"positions": {"shoulder_pan": 30}, "duration_s": 2})
    await ex.run_verb("move_joints", {"positions": {"shoulder_lift": -20}, "duration_s": 2})
    await ex.run_verb("move_joints", {"positions": {"elbow_flex": 40}, "duration_s": 2})
    closed = await ex.run_verb("gripper", {"open": False})
    assert closed.ok and closed.data["holding"] is True
    assert "on something" in closed.summary and f"{GRIP_ON_OBJECT:.0f}/100" in closed.summary


async def test_pick_is_confirm_gated_and_a_sick_arm_reports_it() -> None:
    adapter = LeRobotAdapter(LeRobotMock(fail_heartbeat_after=0))
    manifest = await adapter.connect()
    ex = Executor(registry_from_manifest(manifest, adapter), adapter, confirm=deny_all)
    with pytest.raises(ConfirmDenied):
        await ex.run_verb("pick", {})
    assert not (await adapter.health()).ok


async def test_a_hot_joint_refuses_the_verbs_that_move_the_uncapped_ones() -> None:
    adapter = LeRobotAdapter(LeRobotMock(hot_joints=("shoulder_lift",)))
    manifest = await adapter.connect()
    ex = _executor(adapter, manifest, confirm=allow_all)
    hot = await ex.run_verb("move_joints", {"positions": {"shoulder_pan": 10}})
    assert not hot.ok and "shoulder_lift reads 65" in hot.summary and "70" in hot.summary
    # the gripper has LeRobot's own torque and current caps, and opening a hot one is how
    # you put down what it is holding, so it is not gated on heat
    assert (await ex.run_verb("gripper", {"open": True})).ok


# ── the real backend, against a fake arm ────────────────────────────────────────────────


DEFAULT_TRAVEL_DEG = 200.0
"""Every joint of a plain `FakeArm` travels this far, centred on zero."""


class FakeCalibration:
    """The two fields quackd reads off a `MotorCalibration` (`up.MOTOR_CALIBRATION`)."""

    def __init__(self, travel_deg: float) -> None:
        span = travel_deg * (ENCODER_TICKS - 1) / 360.0
        middle = (ENCODER_TICKS - 1) / 2
        self.range_min = int(middle - span / 2)
        self.range_max = int(middle + span / 2)


class FakeBus:
    """The registers `get_observation()` does not read (`up.STS3215_REGISTERS`), the port a
    retried connect closes (`up.BUS_DISCONNECT`), and the motor table a bus error's id is looked
    up in (`up.BUS_MOTORS`).

    The port's open flag lives on the arm and not here, the way `SOFollower.is_connected` is
    the bus's flag (`up.SO_IS_CONNECTED`): a test that swaps in another bus mid-session keeps
    the arm it had connected."""

    def __init__(self, arm: FakeArm, motors: dict[str, Any] | None = None) -> None:
        self.arm = arm
        self.motors: dict[str, Any] = (
            dict(motors)
            if motors is not None
            else {joint: SimpleNamespace(id=n) for n, joint in enumerate(JOINTS, start=1)}
        )
        """name -> something with an `.id`, the one field of a `Motor` quackd reads. A test
        about naming the joint in an error builds its own table, in no particular order."""
        self.port_handler = SimpleNamespace(is_using=False)
        """The servo SDK's port handler, for its busy flag (`up.BUS_DISCONNECT`): raised before
        every packet and lowered once the reply is in, so a serial error in between leaves it
        raised, and while it is raised every packet is answered "port in use" unsent."""

    @property
    def is_connected(self) -> bool:
        """`up.BUS_IS_CONNECTED`: the port's own flag."""
        return self.arm.connected

    def disconnect(self, disable_torque: bool = True) -> None:
        """`MotorsBus.disconnect`: the busy flag cleared and torque off first, only when asked,
        then the port shut.

        It refuses a port that is not open, as upstream's `check_if_not_connected` does, which
        is the call a retry makes after an attempt that never got the port open."""
        self.arm.calls.append(("bus.disconnect", disable_torque))
        if not self.arm.connected:
            raise ConnectionError("FeetechMotorsBus is not connected. Run `.connect()` first.")
        if disable_torque:
            self.port_handler.is_using = False  # motors_bus.py line 558, under the same flag
            self.disable_torque(num_retry=5)  # upstream's own count (motors_bus.py line 559)
        self.arm.connected = False

    def _handshake(self) -> None:
        """`up.BUS_HANDSHAKE`, under upstream's own name, because the name is how quackd tells
        a failure in here from one in `configure()`: pings and reads, and no write.

        A busy flag left raised answers every ping "port in use", so every motor is missing,
        as upstream reports it. Otherwise the next of `arm.handshake_errors` is raised."""
        if self.port_handler.is_using:
            raise _motor_check_failed(missing=[motor.id for motor in self.motors.values()])
        if self.arm.handshake_errors and (refused := self.arm.handshake_errors.pop(0)):
            raise refused

    def sync_read(
        self, data_name: str, motors: Any = None, *, normalize: bool = True, num_retry: int = 0
    ) -> dict[str, int]:
        self.arm.reads.append((data_name, normalize, num_retry))
        if self.arm.bus_error:
            raise RuntimeError("Incorrect status packet!")
        if data_name == "Torque_Enable":
            # per motor, as upstream reads it: a motor that kept its torque through a release
            # reads 1 while the rest read 0
            return {
                joint: 1 if self.arm.torque or joint in self.arm.torque_holdouts else 0
                for joint in JOINTS
            }
        if data_name == "Present_Temperature":
            return {joint: int(self.arm.temperature.get(joint, 30)) for joint in JOINTS}
        raise KeyError(data_name)

    def enable_torque(self, motors: Any = None, num_retry: int = 0) -> None:
        """`up.BUS_ENABLE_TORQUE`, which is how `take_hold` picks the arm back up.

        A joint named in `slips` sags as torque arrives and stays sagged: the goal goes out
        again immediately afterwards, and a servo that could not hold the pose the first time
        does not reach it on the second ask either. The signature is upstream's, retries
        included, and the retries asked for are kept for a test to read."""
        self.arm.torque_retries.append(("enable_torque", num_retry))
        if self.arm.bus_error:
            raise RuntimeError("Incorrect status packet!")
        self.arm.timeline.append("enable_torque")
        if self.arm.torque_refuses:
            return
        self.arm.torque = True
        for joint, gap in self.arm.slips.items():
            self.arm.positions[joint] += gap
            self.arm.stuck.add(joint)

    def disable_torque(self, motors: Any = None, num_retry: int = 0) -> None:
        """`up.BUS_DISABLE_TORQUE`: the one call in quackd that de-energises a robot.

        It goes through the bus and never through the `Robot`, which is why this fake has no
        `disable_torque` of its own for a test to be accidentally green against."""
        self.arm.torque_retries.append(("disable_torque", num_retry))
        if self.arm.bus_error:
            raise RuntimeError("Incorrect status packet!")
        self.arm.timeline.append("disable_torque")
        self.arm.torque = False
        if self.arm.bus_error_after_release:
            self.arm.bus_error = True


class FakeArm:
    """The slice of a LeRobot `Robot` the real backend touches, verified names only.

    It caps every step the way `max_relative_target` does, so a goal takes as many sends as
    a real one would, and its gripper stops on an object instead of closing.

    And it clamps like the servo (`up.POSITION_LIMITS_CLAMP_GOALS`): after the step cap, a
    body joint's goal is clamped into the travel its own `FakeCalibration` gives it, computed
    from the ticks with LeRobot's degrees formula rather than borrowed from quackd, while a
    test is free to set a position anywhere, a fold past the limit included. Before this the
    fake followed any goal, which is how a rest pose past the travel was "driven to" in a test
    that passed for a week while the arm on the bench could never arrive."""

    def __init__(
        self,
        *,
        calibrated: bool = True,
        camera: bool = True,
        step: float = 25.0,
        object_in_jaws: bool = False,
        stuck: tuple[str, ...] = (),
    ) -> None:
        self.calibrated = calibrated
        self.camera = camera
        self.step = step
        self.object_in_jaws = object_in_jaws
        self.stuck = set(stuck)
        self.obstacles: dict[str, tuple[float, float]] = {}
        """The lowest and highest angle something in the way lets a joint reach: an obstacle
        partway along a move, which the servo pushes against and stops at."""
        self.trail: list[dict[str, float]] = []
        """Where every joint was after each `send_action`, in order: how far the arm actually
        went per send, which `actions`, what was asked for, cannot say."""
        self.connected = False
        self.dead = False
        self.send_fails = False
        self.bus_error = False
        self.torque = True
        self.torque_refuses = False
        """The servo takes `enable_torque` and stays limp anyway, which is the one failure
        `take_hold` cannot tell from a success without reading the register back."""
        self.slips: dict[str, float] = {}
        """Degrees each joint sags by as torque comes on, which is what an arm a person is
        still holding does when it is handed back to the servos."""
        self.bus_error_after_release = False
        """The status packets start coming back corrupt the moment torque drops, so the read
        that would confirm the release is the one that fails."""
        self.torque_holdouts: set[str] = set()
        """Motors that answer the release and keep their torque anyway, so the read-back finds
        them on while the rest are off: an arm limp in part and holding in part."""
        self.connect_errors: list[BaseException | None] = []
        """What the next `connect()` calls raise, one entry each, once the port is open: a
        write in `configure()` that lost its status packet, which leaves the port open behind
        it (`up.CONFIGURE_TORQUE_WRITES_ONCE`). None, or a list run out, is a connect that goes
        through."""
        self.port_errors: list[BaseException | None] = []
        """The same, raised before the port opens: a port that is wrong, or owned by something
        else, so the attempt never reached a motor. None is a port that opens."""
        self.handshake_errors: list[BaseException | None] = []
        """The same, raised by the bus's handshake once the port is open and before anything is
        written: a servo that did not answer its ping, or answered as another model. An
        `OSError` among them comes out as LeRobot's port error, raised from it, as upstream's
        `_connect` does. None is a handshake that passes."""
        self.torque_retries: list[tuple[str, int]] = []
        """Every torque call on the bus and the `num_retry` it was given, in order."""
        self.temperature: dict[str, float] = dict.fromkeys(JOINTS, 30.0)
        self.calls: list[tuple[Any, ...]] = []
        self.actions: list[dict[str, float]] = []
        self.timeline: list[str] = []
        """`send`, `enable_torque` and `disable_torque` in the order they happened. The whole
        of `take_hold` is the order it does those in, and this is what a test reads to check
        it: `actions` alone cannot say which side of the torque call a goal went out on."""
        self.reads: list[tuple[str, bool, int]] = []
        self.positions = dict.fromkeys(JOINTS, 0.0)
        self.positions["gripper"] = 100.0
        self.torque_disabled = 0
        """How many times `disconnect()` dropped torque by LeRobot's own default. A release
        asked for by `let_go` goes through the bus and is not counted here, which is how a
        test tells an arm that was let go of from one that was merely disconnected."""
        self.calibration = {joint: FakeCalibration(DEFAULT_TRAVEL_DEG) for joint in JOINTS}
        self.calibration_fpath = "/tmp/lerobot/calibration/robots/so_follower/arm-01.json"
        self.bus = FakeBus(self)
        self.config = SimpleNamespace(disable_torque_on_disconnect=True)
        """`disconnect()` reads this flag off the config instance when it runs rather than
        copying it at construction (`up.SO_DISCONNECT_READS_ITS_CONFIG_LATE`), which is the
        seam `close()` uses to leave an arm holding a pose it could not reach."""

    @property
    def observation_features(self) -> dict[str, Any]:
        feats: dict[str, Any] = {f"{j}.pos": float for j in JOINTS}
        if self.camera:
            feats["front"] = (48, 64, 3)
        return feats

    @property
    def is_connected(self) -> bool:
        return self.connected

    @property
    def is_calibrated(self) -> bool:
        return self.calibrated

    def connect(self, calibrate: bool = True) -> None:
        """`SOFollower.connect` in upstream's order: refused while the port is open
        (`up.SO_CONNECT_REFUSES_WHILE_OPEN`), then the port, then the bus's handshake, then
        `configure()`, and nothing closes the port again when either of the last two raises."""
        self.calls.append(("connect", calibrate))
        if self.connected:
            # DeviceAlreadyConnectedError is a ConnectionError upstream
            raise ConnectionError("SOFollower is already connected.")
        if self.port_errors and (refused := self.port_errors.pop(0)) is not None:
            raise refused
        self.connected = True
        try:
            self.bus._handshake()
        except OSError as e:  # MotorsBus._connect, motors_bus.py lines 536 to 540
            raise ConnectionError(
                "\nCould not connect on port 'COM7'. Make sure you are using the correct port."
                "\nTry running `lerobot-find-port`\n"
            ) from e
        if self.connect_errors and (lost := self.connect_errors.pop(0)) is not None:
            raise lost

    def disconnect(self) -> None:
        self.calls.append(("disconnect",))
        self.connected = False
        if self.config.disable_torque_on_disconnect:
            self.torque_disabled += 1
            self.torque = False

    def get_observation(self) -> dict[str, Any]:
        if self.dead:
            raise ConnectionError("Failed to sync read 'Present_Position'")
        obs: dict[str, Any] = {f"{j}.pos": v for j, v in self.positions.items()}
        if self.camera:
            obs["front"] = np.zeros((48, 64, 3), dtype=np.uint8)
        return obs

    def travel(self, joint: str) -> tuple[float, float]:
        """The limits the servo clamps a goal to, in degrees: the calibrated ticks through
        LeRobot's formula, `(tick - mid) * 360 / 4095` with `mid` halfway between them."""
        cal = self.calibration[joint]
        mid = (cal.range_min + cal.range_max) / 2
        scale = 360.0 / (ENCODER_TICKS - 1)
        return (cal.range_min - mid) * scale, (cal.range_max - mid) * scale

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if self.send_fails:
            raise ConnectionError("Failed to sync write 'Goal_Position'")
        self.actions.append(dict(action))
        self.timeline.append("send")
        sent = {}
        for key, value in action.items():
            joint = key.removesuffix(".pos")
            present = self.positions[joint]
            capped = present + max(-self.step, min(self.step, float(value) - present))
            # what LeRobot reports sending is the step-capped goal; the clamp to the limits is
            # the servo's own, below that, and nothing reports it
            sent[key] = capped
            # a limp servo takes the goal into its register and does not move to it. That is
            # why `take_hold` writes the pose again once torque is back, and why a stop over
            # an arm somebody is holding has to pick it up before it sends anything.
            if joint in self.stuck or not self.torque:
                continue
            if joint != "gripper" and joint in self.calibration:
                lo, hi = self.travel(joint)
                capped = min(hi, max(lo, capped))
            if (room := self.obstacles.get(joint)) is not None:
                capped = min(room[1], max(room[0], capped))
            if joint == "gripper" and self.object_in_jaws:
                capped = max(capped, GRIP_ON_OBJECT)
            self.positions[joint] = capped
        self.trail.append(dict(self.positions))
        return sent


class FakePolicy:
    """Moves the arm and then closes the gripper, which is the only way anything on this
    body can end up holding something."""

    def __init__(self) -> None:
        self.n = 0
        self.tasks: list[str] = []

    def act(self, observation: dict[str, Any], *, task: str) -> dict[str, float] | None:
        assert "shoulder_pan.pos" in observation
        self.tasks.append(task)
        self.n += 1
        if self.n < 3:
            return {"shoulder_pan": 5.0 * self.n}
        if self.n < 8:
            return {"gripper": 0.0}
        return None


async def test_real_backend_maps_intents_to_verified_names_and_never_limps() -> None:
    arm = FakeArm(object_in_jaws=True)
    adapter = LeRobotAdapter(LeRobotReal("COM5", robot=arm))
    manifest = await adapter.connect()
    assert arm.calls[0] == ("connect", False)  # calibration is interactive: never triggered
    # no camera without --camera-url, whatever the arm's own features say: quackd passes
    # cameras={} to the follower and owns any webcam itself
    assert not manifest.provides("observe") and not manifest.provides("pick")
    assert "camera" not in manifest.sensors and manifest.backend == "real"
    # the joint ranges come off the arm's own calibration: 200 degrees of travel, centred
    assert manifest.extras["joint_range_deg"]["shoulder_pan"] == [-100.0, 100.0]
    assert manifest.extras["joint_range_deg"]["gripper"] == [0.0, 100.0]
    assert manifest.extras["calibration_file"].endswith("arm-01.json")
    assert manifest.limits["step_deg"] == MAX_STEP_DEG
    ex = _executor(adapter, manifest)
    assert (
        await ex.run_verb(
            "move_joints", {"positions": {"shoulder_pan": 10}, "duration_s": MOVE_MIN_S}
        )
    ).ok
    assert arm.actions[-1] == {"shoulder_pan.pos": 10.0}
    closed = await ex.run_verb("gripper", {"open": False})
    assert closed.ok and closed.data["holding"] is True, closed.summary
    # closing on something stops short, which is a stall, and a stall ends in a hold
    assert {"gripper.pos": 0.0} in arm.actions and (await adapter.get_state()).holding
    stopped = await ex.run_verb("stop")
    assert stopped.ok
    # a hold is the five body joints and deliberately not the gripper: re-sending its
    # measured position would open a hand that is squeezing something
    assert set(arm.actions[-1]) == {f"{j}.pos" for j in JOINTS if j != "gripper"}
    assert arm.actions[-1]["shoulder_pan.pos"] == 10.0 and arm.torque_disabled == 0
    assert (await adapter.get_state()).holding  # the stop did not drop it
    assert await adapter.get_frame() is None
    state = await adapter.get_state()
    assert state.extras["joints"]["shoulder_pan"] == 10.0 and state.battery_percent is None
    assert state.extras["torque"] is True and state.extras["temperature_c"]["elbow_flex"] == 30
    assert "GRIPPER_OPEN_VALUE" in state.extras["assumptions"]
    await adapter.heartbeat()
    await adapter.close()
    assert ("disconnect",) in arm.calls
    with pytest.raises(HeartbeatError):
        await adapter.heartbeat()


def test_the_config_spells_out_every_field_that_is_a_safety_choice() -> None:
    """Inheriting an upstream default is fine until upstream changes one. The step cap is
    the field upstream leaves at None, and it has to be a float, not an int."""
    kwargs = LeRobotReal("COM5", robot_id="arm-09")._config_kwargs()
    assert kwargs == {
        "port": "COM5",
        "id": "arm-09",
        "use_degrees": True,
        "disable_torque_on_disconnect": True,
        "cameras": {},
        "max_relative_target": 5.0,
    }
    assert isinstance(kwargs["max_relative_target"], float)


def test_the_port_has_to_look_like_a_port() -> None:
    check_port("COM5")
    check_port("/dev/ttyACM0")
    with pytest.raises(TransportError, match="must be the arm's serial port"):
        check_port("")
    with pytest.raises(TransportError, match="not a serial port"):
        check_port("tcp://192.168.1.42:5555")


def test_the_step_cap_comes_from_the_environment_or_refuses(monkeypatch: Any) -> None:
    assert step_from_env() == MAX_STEP_DEG
    monkeypatch.setenv(STEP_ENV, "2.5")
    assert step_from_env() == 2.5
    monkeypatch.setenv(STEP_ENV, "0")
    with pytest.raises(TransportError, match="above 0"):
        step_from_env()
    monkeypatch.setenv(STEP_ENV, "quickly")
    with pytest.raises(TransportError, match="not a number"):
        step_from_env()


async def test_a_joint_that_stops_moving_is_a_failure_and_not_a_success() -> None:
    """Upstream reports nothing about whether a goal was reached: an arm against an
    obstacle and an arm that arrived look identical unless somebody compares them."""
    arm = FakeArm(stuck=("shoulder_lift",))
    _adapter, ex, _clock = await _paced(arm)
    stalled = await ex.run_verb(
        "move_joints", {"positions": {"shoulder_lift": 40}, "duration_s": 5}
    )
    assert not stalled.ok
    assert "shoulder_lift is at 0 with a goal of 40" in stalled.summary
    assert "stopped moving" in stalled.summary


# ── move_joints paces its motion across duration_s ──────────────────────────────────────


class SteppedClock:
    """Time that passes only when it is slept (`real.Clock`), so a ramp of many seconds costs
    no wall time and every tick lands exactly one `TICK_S` after the one before it."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.t += seconds
        await asyncio.sleep(0)


async def _paced(
    arm: FakeArm, *, step: float = MAX_STEP_DEG
) -> tuple[LeRobotAdapter, Executor, SteppedClock]:
    """The real backend over `arm`, on a stepped clock, with `step` as quackd's own cap."""
    clock = SteppedClock()
    adapter = LeRobotAdapter(LeRobotReal("COM5", robot=arm, max_step_deg=step, clock=clock))
    manifest = await adapter.connect()
    return adapter, _executor(adapter, manifest), clock


def _sent(actions: list[dict[str, float]], joint: str) -> list[float]:
    """Every goal one joint was sent, in order, from the arm's own `<joint>.pos` actions or the
    mock's plain ones."""
    return [a[k] for a in actions for k in (joint, f"{joint}.pos") if k in a]


def _walked(sent: list[float], start: float, goal: float, duration_s: float) -> None:
    """`sent` is a ramp from `start` to `goal` across `duration_s`, one target a tick: each the
    straight-line share of the way the time gone says (to the tenth a target is rounded to),
    never back, never a jump, the goal first sent once the time is up and exactly the goal from
    then on."""
    rounding = 0.5 * 10**-RAMP_DECIMALS + 1e-9
    assert sent[0] == pytest.approx(start), "the ramp starts where the joint is"
    for k, target in enumerate(sent):
        share = min(1.0, k * TICK_S / duration_s)
        assert target == pytest.approx(start + (goal - start) * share, abs=rounding), (k, sent)
    steps = [b - a for a, b in itertools.pairwise(sent)]
    assert all(s * (goal - start) >= 0 for s in steps), f"a target went backwards: {sent}"
    per_tick = abs(goal - start) * TICK_S / duration_s
    assert all(abs(s) <= per_tick + 2 * rounding for s in steps), f"a jump, not a ramp: {sent}"
    first_goal = sent.index(goal)
    assert first_goal * TICK_S == pytest.approx(duration_s, abs=TICK_S + 1e-9), (
        f"the goal went out at {first_goal * TICK_S:.1f} s of a {duration_s} s move"
    )


LONG_MOVES = [
    pytest.param({"shoulder_pan": 0.6}, {}, 4.0, id="one joint up"),
    pytest.param({"elbow_flex": -0.7}, {"elbow_flex": 0.5}, 9.5, id="one joint down"),
    pytest.param(
        {"wrist_flex": -0.8, "shoulder_lift": 0.3},
        {"shoulder_lift": -0.42},
        6.3,
        id="two joints, opposite ways",
    ),
    pytest.param({"gripper": 0.15}, {}, 3.3, id="the gripper as a joint"),
]
"""Goals and starting points as shares of each joint's travel from its middle (`_share_of`), so
nothing here is an angle anybody measured, and durations of several lengths."""


def _share_of(travel: tuple[float, float], share: float) -> float:
    lo, hi = travel
    middle = (lo + hi) / 2
    return middle + share * ((hi - middle) if share > 0 else (middle - lo))


@pytest.mark.parametrize(("goals", "starts", "duration_s"), LONG_MOVES)
async def test_a_long_move_walks_its_goal_there_across_the_time_asked_for_on_the_arm(
    goals: dict[str, float], starts: dict[str, float], duration_s: float
) -> None:
    """On the bench a model was asked to raise the arm slowly and declined, correctly: the verb
    said `duration_s` was a budget and the arm moved at its capped speed whatever it said,
    because every tick re-sent the final goal and LeRobot's step cap was the only pace there
    was. Now the goal is walked from where the joint is to where it was asked to be, so the
    joint arrives when the time is up and not a moment before."""
    arm = _spanned()
    for joint, share in starts.items():
        # to the tenth the arm reports a reading in, so the ramp's start is the placed angle
        arm.positions[joint] = round(_share_of(arm.travel(joint), share), 1)
    _adapter, ex, clock = await _paced(arm)
    travel = {j: (0.0, 100.0) if j == "gripper" else arm.travel(j) for j in goals}
    goal = {j: round(_share_of(travel[j], share), 1) for j, share in goals.items()}
    began = dict(arm.positions)
    t0 = clock.t
    moved = await ex.run_verb("move_joints", {"positions": goal, "duration_s": duration_s})
    assert moved.ok, moved.summary
    for joint, value in goal.items():
        _walked(_sent(arm.actions, joint), began[joint], value, duration_s)
        assert arm.positions[joint] == pytest.approx(value)
    took = clock.t - t0
    assert duration_s <= took <= duration_s + 3 * TICK_S, f"a {duration_s} s move took {took}"


@pytest.mark.parametrize(("goals", "starts", "duration_s"), LONG_MOVES)
async def test_a_long_move_walks_its_goal_there_across_the_time_asked_for_on_the_mock(
    goals: dict[str, float], starts: dict[str, float], duration_s: float
) -> None:
    """The same on the mock, whose goals land at once: a rehearsal of a slow move shows the
    ramp in its record, and every intent in that record still carries the time asked for."""
    mock = LeRobotMock()
    for joint, share in starts.items():
        mock.joints[joint] = round(_share_of(MOCK_RANGES[joint], share), 1)
    adapter = LeRobotAdapter(mock)
    ex = _executor(adapter, await adapter.connect())
    goal = {j: round(_share_of(MOCK_RANGES[j], share), 1) for j, share in goals.items()}
    began = dict(mock.joints)
    t0, sent = mock.now(), len(mock.intents)
    moved = await ex.run_verb("move_joints", {"positions": goal, "duration_s": duration_s})
    assert moved.ok, moved.summary
    for joint, value in goal.items():
        _walked(_sent(mock.actions, joint), began[joint], value, duration_s)
    took = mock.now() - t0
    assert duration_s <= took <= duration_s + 3 * TICK_S, f"a {duration_s} s move took {took}"
    joint_intents = [i for i in mock.intents[sent:] if i.kind == "joint"]
    assert joint_intents and all(i.params["duration_s"] == duration_s for i in joint_intents)


@pytest.mark.parametrize(
    ("step", "joint", "share", "duration_s"),
    [
        pytest.param(MAX_STEP_DEG, "shoulder_pan", 0.86, MOVE_MIN_S, id="the default cap"),
        pytest.param(2.0, "elbow_flex", -0.8, 1.0, id="a lowered cap, downward"),
        pytest.param(0.7, "wrist_flex", 0.5, 2.5, id="a small cap, a longer time"),
    ],
)
async def test_a_move_asked_to_be_quicker_than_the_cap_allows_runs_at_the_cap(
    step: float, joint: str, share: float, duration_s: float
) -> None:
    """The step cap stays the ceiling. A time too short for the distance is not refused and not
    obeyed: the ramp runs ahead of the arm, LeRobot clips every send to one step from where the
    joint is, and the joint travels one step a tick until it is there, however soon it was
    asked to be."""
    arm = _spanned(step=step)
    _adapter, ex, clock = await _paced(arm, step=step)
    goal = round(_share_of(arm.travel(joint), share), 1)
    distance = abs(goal)
    assert distance / duration_s * TICK_S > step, "the ramp must be quicker than the cap"
    moved = await ex.run_verb("move_joints", {"positions": {joint: goal}, "duration_s": duration_s})
    assert moved.ok, moved.summary
    trail = [0.0] + [where[joint] for where in arm.trail]
    per_send = [abs(b - a) for a, b in itertools.pairwise(trail)]
    assert per_send[0] == 0.0, "the ramp's first target is where the joint already is"
    assert all(moved_by <= step + 1e-9 for moved_by in per_send), per_send
    assert all(moved_by == pytest.approx(step) for moved_by in per_send[1:-1]), per_send
    took = clock.t
    assert took > duration_s, "a move the cap cannot make in time ends later than asked"
    assert took >= (distance - TOL_DEG) / step * TICK_S
    assert took < move_budget_s(distance, duration_s, step), "it arrived, it did not run out"


@pytest.mark.parametrize(
    ("step", "joint", "share", "duration_s"),
    [
        pytest.param(0.5, "shoulder_lift", 0.25, MOVE_MAX_S, id="a small cap"),
        pytest.param(MAX_STEP_DEG, "elbow_flex", -0.35, 8.5, id="the default cap"),
        pytest.param(MAX_STEP_DEG, "wrist_roll", 0.12, 11.0, id="the full-turn joint"),
    ],
)
async def test_a_slow_ramp_is_never_mistaken_for_a_stall(
    step: float, joint: str, share: float, duration_s: float
) -> None:
    """A slow move is a fraction of a degree a tick, under the threshold the stall rule calls
    a joint stopped. Counted during the ramp, that rule failed every slow move a few ticks in,
    with the arm doing exactly what it was told. It counts once the ramp is over."""
    arm = _spanned(step=step)
    _adapter, ex, clock = await _paced(arm, step=step)
    goal = round(_share_of(arm.travel(joint), share), 1)
    threshold = min(0.5, step / 2)
    per_tick = abs(goal) / duration_s * TICK_S
    assert per_tick < threshold, "the ramp must crawl under the stall threshold"
    moved = await ex.run_verb("move_joints", {"positions": {joint: goal}, "duration_s": duration_s})
    assert moved.ok, moved.summary
    assert clock.t >= duration_s


async def test_a_slow_ramp_on_the_mock_is_never_mistaken_for_a_stall() -> None:
    """The mock has no step cap, so its stall threshold is the full one: a slow move crawls
    under it on every tick of the ramp, and a rehearsal must not fail what the arm would do."""
    mock = LeRobotMock()
    adapter = LeRobotAdapter(mock)
    ex = _executor(adapter, await adapter.connect())
    goal = round(_share_of(MOCK_RANGES["wrist_flex"], 0.13), 1)
    moved = await ex.run_verb("move_joints", {"positions": {"wrist_flex": goal}, "duration_s": 9.0})
    assert moved.ok, moved.summary
    assert mock.joints["wrist_flex"] == goal


@pytest.mark.parametrize(
    ("joint", "start", "wall", "share", "duration_s"),
    [
        pytest.param("shoulder_lift", 0.0, 0.3, 0.7, 3.5, id="blocked on the way up"),
        pytest.param("wrist_flex", 0.55, -0.15, -0.75, 5.5, id="blocked on the way down"),
        pytest.param("elbow_flex", -0.2, -0.2, 0.5, 2.0, id="blocked where it starts"),
    ],
)
async def test_a_joint_blocked_partway_is_a_stall_once_the_ramp_is_done_and_says_where(
    joint: str, start: float, wall: float, share: float, duration_s: float
) -> None:
    """Something in the way still ends the move as a failure that names the joint, where it
    stopped and where it was going. It is found when the ramp has handed the servo the final
    goal and the joint does not follow for the stall rule's ticks: not before, because a ramp
    that slow is indistinguishable from a stop until then, and not much later."""
    arm = _spanned()
    travel = arm.travel(joint)
    arm.positions[joint] = round(_share_of(travel, start), 1)
    at = round(_share_of(travel, wall), 1)
    goal = round(_share_of(travel, share), 1)
    arm.obstacles[joint] = (-math.inf, at) if goal > at else (at, math.inf)
    _adapter, ex, clock = await _paced(arm)
    stalled = await ex.run_verb(
        "move_joints", {"positions": {joint: goal}, "duration_s": duration_s}
    )
    assert not stalled.ok
    stopped = arm.positions[joint]
    assert stopped == pytest.approx(at)
    assert (
        f"{joint} is at {round(stopped, 1):.0f} with a goal of {goal:.0f}, and it has stopped "
        "moving"
    ) in stalled.summary, stalled.summary
    assert duration_s <= clock.t <= duration_s + (STALL_TICKS + 3) * TICK_S, clock.t
    # and the arm was stopped where it stood, as every failed verb ends
    assert arm.actions[-1][f"{joint}.pos"] == pytest.approx(stopped)


@pytest.mark.parametrize(
    ("past", "goals"),
    [
        pytest.param({"shoulder_lift": -17.0}, {"shoulder_lift": 0.4}, id="below the floor"),
        pytest.param({"elbow_flex": 9.0}, {"elbow_flex": -0.2}, id="above the ceiling"),
        pytest.param(
            {"shoulder_lift": -23.0, "wrist_flex": 6.0},
            {"shoulder_lift": -0.5, "wrist_flex": 0.35},
            id="one each way at once",
        ),
    ],
)
async def test_a_joint_read_past_its_travel_starts_its_ramp_at_its_limit(
    past: dict[str, float], goals: dict[str, float]
) -> None:
    """A joint folded past its travel reads past it, and the servo clamps any goal to the
    travel, so a ramp from the reading would be a ramp whose first stretch the servo turns into
    "go to the limit", and whose first targets the backend refuses outright. The ramp starts at
    the limit the manifest publishes: the joint gets there at the servo's own speed, as it
    would whatever quackd sent, and is paced from there."""
    arm = _spanned()
    for joint, by in past.items():
        arm.positions[joint] = _past(arm, joint, by)
    adapter, ex, _clock = await _paced(arm)
    published = adapter.manifest.extras["joint_range_deg"]  # type: ignore[union-attr]
    goal = {j: round(_share_of(arm.travel(j), share), 1) for j, share in goals.items()}
    moved = await ex.run_verb("move_joints", {"positions": goal, "duration_s": 2.5})
    assert moved.ok, moved.summary
    assert all(_within_travel(arm, action) for action in arm.actions), arm.actions
    for joint, by in past.items():
        limit = published[joint][0 if by < 0 else 1]
        _walked(_sent(arm.actions, joint), limit, goal[joint], 2.5)


async def test_a_joint_on_the_mock_read_past_its_travel_starts_its_ramp_at_its_limit() -> None:
    """The mock refuses a goal outside its travel in the arm's words, so a ramp begun at a
    reading past the ceiling would be refused on its first tick in a rehearsal too."""
    mock = LeRobotMock()
    lo, hi = MOCK_RANGES["elbow_flex"]
    mock.joints["elbow_flex"] = hi + 7.0
    adapter = LeRobotAdapter(mock)
    ex = _executor(adapter, await adapter.connect())
    goal = round(_share_of((lo, hi), -0.3), 1)
    moved = await ex.run_verb("move_joints", {"positions": {"elbow_flex": goal}, "duration_s": 1.7})
    assert moved.ok, moved.summary
    _walked(_sent(mock.actions, "elbow_flex"), hi, goal, 1.7)


def test_a_ramp_ends_on_the_goal_to_the_bit_and_never_ramps_a_joint_with_no_start() -> None:
    """`start + (goal - start) * 1.0` is not always `goal` in floating point, and a goal asked
    for at the very edge of the travel, sent back a few ulps past it, is a goal the backend
    refuses as outside the travel at the last tick of a move that was going fine; nor is a goal
    that is not on the tenths the targets on the way are rounded to. So the end of a ramp is the
    goal itself. A joint the arm did not report has no start and is sent its goal at every
    share, rather than a ramp from nowhere."""
    goals = [g / 100 for g in range(-9971, 9972, 373)]
    starts = [s / 10 for s in range(-1003, 1004, 29)]
    noisy = off_the_tenths = 0
    for begin in starts:
        for end in goals:
            assert ramp_target({"j": begin}, {"j": end}, 1.0) == {"j": end}, (begin, end)
            noisy += begin + (end - begin) * 1.0 != end
            off_the_tenths += round(end, RAMP_DECIMALS) != end
    assert noisy and off_the_tenths, "no pair here misses its goal, so this proves nothing"
    goal = {"shoulder_pan": 12.5, "wrist_flex": -31.5}
    for share in (0.0, 0.37, 0.999):
        target = ramp_target({"shoulder_pan": -40.0}, goal, share)
        assert target["wrist_flex"] == goal["wrist_flex"], "a joint with no start was ramped"
        along = target["shoulder_pan"]
        assert along == pytest.approx(-40.0 + 52.5 * share, abs=0.5 * 10**-RAMP_DECIMALS)
        assert along == round(along, RAMP_DECIMALS), "a target on the way is sent to a tenth"
    # and rounding never carries a target past a goal that is not on the tenths
    near = ramp_target({"elbow_flex": 0.0}, {"elbow_flex": 84.97}, 0.9999)["elbow_flex"]
    assert near <= 84.97


def test_the_published_travel_never_promises_a_degree_the_backend_refuses() -> None:
    """The manifest publishes each travel to a tenth of a degree, and rounding to nearest could
    move an end out by up to a twentieth: a pilot asking for the edge it was shown could be
    refused, and a ramp from a joint folded past its travel would begin a hair outside it and be
    refused before it moved. Rounded inward, every published angle is one the backend accepts."""
    spans = [*SPANS.values(), 143.3, 171.7, 199.9, 77.7]
    outward = 0
    for travel_deg in spans:
        exact_lo, exact_hi = joint_ranges({"elbow_flex": FakeCalibration(travel_deg)})["elbow_flex"]
        lo, hi = published_travel(exact_lo, exact_hi)
        assert exact_lo <= lo <= exact_lo + 0.1 and exact_hi - 0.1 <= hi <= exact_hi, travel_deg
        outward += round(exact_hi, 1) > exact_hi
    assert outward, "no span here would have rounded outward, so this proves nothing"
    assert published_travel(-100.0, 100.0) == [-100.0, 100.0]
    assert published_travel(0.0, 100.0) == [0.0, 100.0]
    # an end a hair short of a tenth: ten times it can round to the whole tenth, and the tenth
    # it gives would then be a hair outside the float the backend compares a goal with
    hairs = 0
    for tenths in range(1, 1800, 7):
        end = math.nextafter(tenths / 10, 0.0)
        lo, hi = published_travel(-end, end)
        assert -end <= lo and hi <= end and end - hi < 0.2, end
        hairs += math.floor(end * 10) / 10 > end
    assert hairs, "no end here rounds up to its tenth, so this proves nothing"


async def test_the_edge_the_pilot_is_shown_is_an_edge_it_can_ask_for() -> None:
    """The pilot's prompt prints the published travel, and a careful pilot asks for exactly its
    end. Rounded to nearest, a travel could be published a twentieth of a degree wider than the
    one the backend checks against, and then that goal was refused as outside it. A goal at the
    edge is inside the travel, so it is walked there like any other."""
    arm = _spanned()
    adapter, ex, _clock = await _paced(arm)
    published = adapter.manifest.extras["joint_range_deg"]  # type: ignore[union-attr]
    for joint, end in (("shoulder_lift", 0), ("elbow_flex", 1)):
        edge = published[joint][end]
        began, sent = arm.positions[joint], len(arm.actions)
        moved = await ex.run_verb("move_joints", {"positions": {joint: edge}, "duration_s": 1.2})
        assert moved.ok, moved.summary
        _walked(_sent(arm.actions[sent:], joint), began, edge, 1.2)


async def test_a_goal_in_the_sliver_past_the_published_edge_is_refused_before_it_moves() -> None:
    """The manifest publishes each travel rounded inward and the backend refuses against the
    exact one, and the verb sent any goal past the published edge whole so that the backend's
    refusal would answer before anything moved. A goal in the sliver between the two edges was
    not refused by anything: it went out whole, at the step cap, however long `duration_s` was.
    The verb now refuses a goal outside the travel the pilot was shown itself, before the arm
    is read or anything is sent, in the backend's own words."""
    arm = _spanned(step=MAX_STEP_DEG)
    adapter, ex, clock = await _paced(arm)
    transport = adapter.transport
    assert isinstance(transport, LeRobotReal)
    published = adapter.manifest.extras["joint_range_deg"]  # type: ignore[union-attr]
    slivers = []
    for joint, (lo, hi) in published.items():
        exact_lo, exact_hi = transport.joint_range_deg[joint]
        slivers += [(joint, (lo + exact_lo) / 2)] if exact_lo < lo else []
        slivers += [(joint, (hi + exact_hi) / 2)] if exact_hi > hi else []
    assert slivers, "no joint here has a sliver between its edges, so this proves nothing"
    for joint, goal in slivers:
        assert transport._refuse_out_of_range({joint: goal}) is None, "the backend takes it"
        sent, t0 = len(arm.actions), clock.t
        moved = await ex.run_verb("move_joints", {"positions": {joint: goal}, "duration_s": 8.0})
        assert not moved.ok, f"{joint}={goal} went out: {moved.summary}"
        assert moved.summary == f"move_joints: {range_refusal({joint: goal}, published)}"
        assert "is outside this arm's calibrated range" in moved.summary
        assert len(arm.actions) == sent and clock.t == t0, "something moved before the refusal"


REFUSAL = re.compile(
    r"^(\w+)=(-?\d+(?:\.\d+)?) is outside this arm's calibrated range (-?\d+\.\d)\.\.(-?\d+\.\d);"
)
"""The numbers out of `range_refusal`'s sentence: the goal as said, and the range as said."""


def test_a_range_refusal_never_names_a_goal_inside_the_range_it_gives() -> None:
    """The refusal printed whole degrees. A goal a hair past an edge of 85 was refused as "85 is
    outside -85..85", a goal in the sliver past a published edge was named inside the range the
    same sentence gave, and Python's half-to-even rounding printed a published edge of n.5 as n,
    below what the arm took. Now the range is printed to a tenth, rounded inward, so on the
    verb's side it is the published travel exactly and on a backend's it is never wider than the
    travel that backend checks, and the goal to a tenth unless a tenth would put it inside that
    range, when it is printed as given. Synthetic spans, each end both ways."""
    spans = [*SPANS.values(), 143.3, 171.7, 199.9, 77.7, 209.1]
    checked = as_given = 0
    for travel_deg in spans:
        exact = joint_ranges({"elbow_flex": FakeCalibration(travel_deg)})["elbow_flex"]
        published = tuple(published_travel(*exact))
        sliver = (published[1] + exact[1]) / 2
        for travel, verbs_side in ((exact, False), (published, True)):
            lo, hi = travel
            for goal in (hi + 1e-3, hi + 0.04, hi + 0.06, hi + 0.5, hi + 12.3, lo - 0.04, sliver):
                if lo <= goal <= hi:
                    continue
                refusal = range_refusal({"elbow_flex": goal}, {"elbow_flex": travel}) or ""
                said = REFUSAL.match(refusal)
                assert said is not None, refusal
                shown, low, high = float(said[2]), float(said[3]), float(said[4])
                assert not low <= shown <= high, f"a refused goal named inside: {refusal}"
                assert lo <= low and high <= hi, f"a range wider than the travel: {refusal}"
                if verbs_side:
                    assert (low, high) == published, refusal
                checked += 1
                as_given += said[2] == repr(goal)
    assert checked and as_given, "no goal needed more than a tenth, so the fallback is untested"
    assert range_refusal({"elbow_flex": 0.0}, {"elbow_flex": published}) is None


@pytest.mark.parametrize(
    ("joint", "off", "duration_s", "step"),
    [
        pytest.param("shoulder_pan", TOL_DEG - 1.0, 6.0, MAX_STEP_DEG, id="just inside"),
        pytest.param("wrist_flex", -(TOL_DEG - 2.5), 4.4, MAX_STEP_DEG, id="the other way"),
        pytest.param("gripper", -(TOL_DEG - 0.5), MOVE_MIN_S, MAX_STEP_DEG, id="the gripper"),
        pytest.param("elbow_flex", TOL_DEG - 1.0, 6.0, (TOL_DEG - 1.0) / 8, id="a lowered cap"),
    ],
)
async def test_a_move_inside_the_arrival_tolerance_is_walked_across_its_time_too(
    joint: str, off: float, duration_s: float, step: float
) -> None:
    """A move whose every joint already read within the tolerance the verb calls arrived went
    out whole and was judged after one tick, whatever `duration_s` said, although the pilot is
    told every joint travels across that time: a slow nudge of a few degrees near something
    arrived in a tenth of a second. With a lowered step cap it was worse: the one send moved the
    joint a step, the goal register stayed there, and the verb still said it had moved. Now it
    is walked like any other move and judged once the ramp is done, where it has arrived."""
    arm = _spanned(step=step)
    _adapter, ex, clock = await _paced(arm, step=step)
    began = arm.positions[joint]
    goal = began + off
    moved = await ex.run_verb("move_joints", {"positions": {joint: goal}, "duration_s": duration_s})
    assert moved.ok, moved.summary
    _walked(_sent(arm.actions, joint), began, goal, duration_s)
    assert arm.positions[joint] == pytest.approx(goal), "it stopped short of its goal"
    assert duration_s <= clock.t <= duration_s + 3 * TICK_S, clock.t


async def test_a_small_move_on_the_mock_is_walked_across_its_time_too() -> None:
    """The same on the mock, whose goals land at once: a rehearsal of a nudge takes the time
    asked for and its record shows the walk, as the arm's would."""
    mock = LeRobotMock()
    adapter = LeRobotAdapter(mock)
    ex = _executor(adapter, await adapter.connect())
    began = mock.joints["wrist_flex"]
    goal, duration_s = began - (TOL_DEG - 1.5), 3.7
    t0 = mock.now()
    moved = await ex.run_verb(
        "move_joints", {"positions": {"wrist_flex": goal}, "duration_s": duration_s}
    )
    assert moved.ok, moved.summary
    _walked(_sent(mock.actions, "wrist_flex"), began, goal, duration_s)
    assert duration_s <= mock.now() - t0 <= duration_s + 3 * TICK_S


@pytest.mark.parametrize(
    "off",
    [
        pytest.param(0.0, id="the goal is the reading"),
        pytest.param(RAMP_RESOLUTION / 2, id="under a tenth above it"),
        pytest.param(-RAMP_RESOLUTION / 3, id="under a tenth below it"),
    ],
)
async def test_a_move_with_nothing_to_walk_goes_out_at_once(off: float) -> None:
    """The one move still sent whole: every joint already nearer its goal than a ramp's
    resolution, where the ramp has no target between the two and could only hold the start
    for the whole time and then send the goal. It goes out once and is judged after a tick."""
    arm = _spanned()
    _adapter, ex, clock = await _paced(arm)
    goal = {"shoulder_pan": arm.positions["shoulder_pan"] + off}
    goal["elbow_flex"] = arm.positions["elbow_flex"] - off
    moved = await ex.run_verb("move_joints", {"positions": goal, "duration_s": MOVE_MAX_S})
    assert moved.ok, moved.summary
    for joint, value in goal.items():
        assert _sent(arm.actions, joint) == [value], arm.actions
    assert clock.t == pytest.approx(TICK_S)


def test_the_move_budget_always_ends_inside_the_executor_s_timeout() -> None:
    """The budget is the time asked for, or the time the cap needs if longer, plus a settle,
    and it has to end before the executor's own timeout does: past that the executor cancels
    the verb and says only "timed out", where the verb would have said which joint fell short
    and where it stopped. One constant is both numbers, so they cannot drift apart."""
    registered = lerobot_verbs(policy=False)["move_joints"]
    assert registered.timeout_s == MOVE_JOINTS_TIMEOUT_S
    from_manifest = registry_from_manifest(
        lerobot_manifest("real"), LeRobotAdapter(LeRobotMock())
    ).get("move_joints")
    assert from_manifest.timeout_s == MOVE_JOINTS_TIMEOUT_S
    ceiling = MOVE_JOINTS_TIMEOUT_S - MOVE_HEADROOM_S
    assert ceiling >= MOVE_MAX_S + MOVE_SETTLE_S, "the longest ramp must keep its settle"
    assert MOVE_SETTLE_S >= STALL_TICKS * TICK_S, "a stall must be callable inside the settle"
    for distance in (0.0, 3.0, 47.0, 133.0, 359.0):
        for duration_s in (MOVE_MIN_S, 1.3, 6.8, MOVE_MAX_S):
            for step in (None, 0.05, 0.6, MAX_STEP_DEG, 30.0):
                budget = move_budget_s(distance, duration_s, step)
                assert budget <= ceiling < registered.timeout_s, (distance, duration_s, step)
                need = max(duration_s, distance / (step / TICK_S) if step else 0.0)
                assert budget == pytest.approx(min(need + MOVE_SETTLE_S, ceiling))
    schema = MoveJointsParams.model_json_schema()["properties"]["duration_s"]
    assert schema["minimum"] == MOVE_MIN_S and schema["maximum"] == MOVE_MAX_S


async def test_a_move_the_cap_cannot_finish_runs_out_of_time_inside_the_executor_s() -> None:
    """With the step lowered far enough, the longest time and a long reach need more than the
    executor allows. The verb stops at its own budget and says how far it got."""
    step = 0.4
    arm = _spanned(step=step)
    _adapter, ex, clock = await _paced(arm, step=step)
    goal = round(_share_of(arm.travel("shoulder_pan"), -0.86), 1)
    late = await ex.run_verb(
        "move_joints", {"positions": {"shoulder_pan": goal}, "duration_s": MOVE_MAX_S}
    )
    ceiling = MOVE_JOINTS_TIMEOUT_S - MOVE_HEADROOM_S
    assert move_budget_s(abs(goal), MOVE_MAX_S, step) == ceiling, "the cap must need longer"
    assert not late.ok and "when the time ran out" in late.summary, late.summary
    assert f"with a goal of {goal:.0f}" in late.summary
    assert ceiling <= clock.t <= ceiling + TICK_S + 1e-9, clock.t


async def test_the_gripper_verb_sends_its_yes_or_no_and_is_never_ramped() -> None:
    """`gripper` sends `Intent.gripper(open)`, which each backend turns into fully open or
    fully shut and which the mock reads to decide whether it closed on something. A ramp would
    have to make up the numbers in between, so it is left as it was."""
    mock = LeRobotMock()
    adapter = LeRobotAdapter(mock)
    ex = _executor(adapter, await adapter.connect())
    sent = len(mock.intents)
    closed = await ex.run_verb("gripper", {"open": False})
    assert closed.ok, closed.summary
    kinds = {i.kind for i in mock.intents[sent:]}
    assert kinds == {"gripper"}, kinds
    assert all(i.params == {"open": False} for i in mock.intents[sent:])


async def test_the_rest_move_and_the_hold_wait_on_the_backend_s_own_clock() -> None:
    """The clock seam is honest only if every wait this backend measures against `now()` goes
    through it. A rest move that read a test's clock and slept on the wall's would never see its
    budget run out; so each of its ticks, and the settle `take_hold` gives a hold before reading
    it back, is one tick on whatever clock the backend was given."""
    clock = SteppedClock()
    arm = FakeArm(step=MAX_STEP_DEG)
    where_it_sits = {joint: arm.positions[joint] for joint in JOINTS if joint != "gripper"}
    pose = where_it_sits | {"elbow_flex": where_it_sits["elbow_flex"] + 3 * MAX_STEP_DEG}
    transport = LeRobotReal("COM5", robot=arm, rest_pose=pose, clock=clock)
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    rested = await adapter.go_to_rest()
    assert rested.how == "arrived", rested.reason
    assert arm.actions and clock.t == pytest.approx(len(arm.actions) * TICK_S)
    assert (await adapter.let_go()).how == "released"
    before = clock.t
    assert (await adapter.take_hold()).how == "held"
    assert clock.t - before == pytest.approx(TICK_S)


async def test_a_goal_outside_the_calibrated_range_is_refused_with_the_range() -> None:
    """Refused by the verb before anything is sent, in the words the backend refuses with, so
    a pilot hears one sentence for the one rule wherever it is enforced."""
    arm = FakeArm()
    transport = LeRobotReal("COM5", robot=arm)
    adapter = LeRobotAdapter(transport)
    manifest = await adapter.connect()
    ex = _executor(adapter, manifest)
    sent = len(arm.actions)
    out = await ex.run_verb("move_joints", {"positions": {"elbow_flex": 170}})
    assert not out.ok and "-100.0..100.0" in out.summary and "does not clamp" in out.summary
    backend = transport._refuse_out_of_range({"elbow_flex": 170.0})
    assert backend is not None and out.summary == f"move_joints: {backend}"
    assert len(arm.actions) == sent, "the verb sent something before refusing"
    assert arm.positions["elbow_flex"] == 0.0


async def test_an_unplugged_arm_fails_the_heartbeat_even_though_is_connected_is_true() -> None:
    """`is_connected` is the serial port's own open flag, so it stays True until a read
    fails. The heartbeat reads the arm rather than the flag."""
    arm = FakeArm()
    transport = LeRobotReal("COM5", robot=arm)
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    await adapter.heartbeat()
    arm.dead = True
    assert arm.is_connected is True
    with pytest.raises(HeartbeatError, match="did not answer"):
        await adapter.heartbeat()


async def test_a_wedged_call_refuses_every_later_call_instead_of_sharing_the_bus() -> None:
    """A call that blows its deadline is not over: its thread is still sitting on a
    half-duplex bus. Starting another would put two talkers on it."""
    import threading

    release = threading.Event()
    arm = FakeArm()
    transport = LeRobotReal("COM5", robot=arm, timeout_s=0.2)
    adapter = LeRobotAdapter(transport)
    await adapter.connect()

    def block() -> None:
        release.wait(5.0)

    try:
        with pytest.raises(TimeoutError):
            await transport._call(block, deadline_s=0.2)
        assert transport.stop_error is not None and "one owner" in transport.stop_error
        with pytest.raises(HeartbeatError):
            await adapter.heartbeat()
        assert not (await adapter.send_intent(Intent.gripper(True))).accepted
    finally:
        release.set()


async def test_torque_and_temperature_are_measured_rather_than_assumed() -> None:
    arm = FakeArm()
    adapter = LeRobotAdapter(LeRobotReal("COM5", robot=arm))
    manifest = await adapter.connect()
    ex = _executor(adapter, manifest)
    assert ("Torque_Enable", False, 2) in arm.reads
    assert ("Present_Temperature", False, 2) in arm.reads
    arm.torque = False
    refused = await ex.run_verb("move_joints", {"positions": {"shoulder_pan": 10}})
    assert not refused.ok and "torque is off" in refused.summary
    arm.torque = True
    arm.temperature["elbow_flex"] = 65.0
    hot = await ex.run_verb("move_joints", {"positions": {"shoulder_pan": 10}})
    assert not hot.ok and "elbow_flex reads 65" in hot.summary
    health = await adapter.health()
    assert health.ok and health.extras["hottest_c"] == 65


async def test_a_corrupt_register_read_costs_a_reading_and_not_the_run() -> None:
    """A Feetech bus returns the odd corrupt status packet. Losing the run over one would
    be worse than the disease, so the positions are the liveness check and the registers
    are not."""
    arm = FakeArm()
    adapter = LeRobotAdapter(LeRobotReal("COM5", robot=arm))
    await adapter.connect()
    arm.bus_error = True
    await adapter.heartbeat()
    state = await adapter.get_state()
    assert "Incorrect status packet" in state.extras["register_error"]
    assert state.extras["torque"] is True  # the last known reading stands


async def test_real_backend_refuses_an_uncalibrated_arm() -> None:
    arm = FakeArm(calibrated=False)
    adapter = LeRobotAdapter(LeRobotReal("COM5", robot=arm))
    with pytest.raises(TransportError, match="not calibrated"):
        await adapter.connect()
    assert ("disconnect",) in arm.calls


async def test_real_backend_refuses_an_arm_with_no_calibration_file() -> None:
    """`is_calibrated` compares the motors with a file. Without the file there is nothing
    that knows how far each joint travels, and the range refusal has nothing to stand on."""
    arm = FakeArm()
    arm.calibration = {}
    adapter = LeRobotAdapter(LeRobotReal("COM5", robot=arm))
    with pytest.raises(TransportError, match="no calibration file"):
        await adapter.connect()


async def test_real_backend_runs_an_injected_policy_for_pick() -> None:
    arm = FakeArm(camera=False, object_in_jaws=True)
    policy = FakePolicy()
    adapter = LeRobotAdapter(LeRobotReal("COM5", robot=arm, policy=policy))
    manifest = await adapter.connect()
    assert (
        manifest.provides("pick")
        and "skill" in manifest.intents
        and not manifest.provides("observe")
    )
    ex = Executor(registry_from_manifest(manifest, adapter), adapter, confirm=allow_all)
    picked = await ex.run_verb("pick", {"target": "cup", "max_s": 10})
    assert picked.ok, picked.summary
    assert policy.tasks[0] == "cup"
    assert {"shoulder_pan.pos": 5.0} in arm.actions and {"shoulder_pan.pos": 10.0} in arm.actions
    assert (await adapter.get_state()).holding
    assert (await adapter.get_state()).policy == "idle"
    await adapter.close()


async def test_a_verb_is_refused_while_a_policy_has_the_arm() -> None:
    class Forever:
        def act(self, observation: dict[str, Any], *, task: str) -> dict[str, float] | None:
            return {"shoulder_pan": 5.0}

    arm = FakeArm(camera=False)
    transport = LeRobotReal("COM5", robot=arm, policy=Forever())
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    assert (await adapter.send_intent(Intent.do("policy:pick:cup"))).accepted
    try:
        refused = await adapter.send_intent(Intent.joint({"shoulder_pan": 0.0}, 1.0))
        assert not refused.accepted and "pick is running" in (refused.reason or "")
    finally:
        await adapter.close()


async def test_pick_looks_once_more_before_calling_a_finished_policy_a_failed_grasp() -> None:
    """A policy can grasp and finish inside a single poll, and `holding` is not knowable the
    instant it does: it is inferred from two gripper readings a real interval apart agreeing,
    so the read that catches the policy going idle can be one sample too early.

    Without the second look that is a `pick` reporting nothing held with the object in the
    jaws. It is also what a macOS runner saw on 2026-09-21, where the same race failed
    `test_real_backend_runs_an_injected_policy_for_pick` once and has not since."""

    class SettlesAfterTheLastPoll(LeRobotMock):
        """Idle by the time `pick` first looks, and holding only on the read after that."""

        def __init__(self) -> None:
            super().__init__()
            self.looks = 0
            self.watching = False

        async def send_intent(self, intent: Intent) -> Any:
            ack = await super().send_intent(intent)
            if intent.kind == "do":
                self.watching = True  # the policy ran and finished between polls
                self.policy = "idle"
            return ack

        async def get_state(self) -> Any:
            if self.watching:
                self.looks += 1
                # 1 is the read before the loop, 2 is the poll that finds the policy idle
                # with the grasp still settling, and 3 is the look after the settle
                self.holding = self.looks >= 3
            return await super().get_state()

    transport = SettlesAfterTheLastPoll()
    adapter = LeRobotAdapter(transport)
    manifest = await adapter.connect()
    ex = Executor(registry_from_manifest(manifest, adapter), adapter, confirm=allow_all)
    picked = await ex.run_verb("pick", {"target": "cup", "max_s": 10})
    assert picked.ok, picked.summary
    assert transport.looks >= 3, "pick decided without giving the grasp a settle"


async def test_a_policy_that_raises_is_a_failed_pick_and_says_so() -> None:
    class Broken:
        def act(self, observation: dict[str, Any], *, task: str) -> dict[str, float] | None:
            raise RuntimeError("no accelerated backend")

    arm = FakeArm(camera=False)
    adapter = LeRobotAdapter(LeRobotReal("COM5", robot=arm, policy=Broken()))
    manifest = await adapter.connect()
    ex = Executor(registry_from_manifest(manifest, adapter), adapter, confirm=allow_all)
    failed = await ex.run_verb("pick", {"target": "cup", "max_s": 2})
    assert not failed.ok
    assert "no accelerated backend" in failed.summary and "RuntimeError" in failed.summary


async def test_the_lookout_duck_validates_against_the_arm_and_moves_nothing() -> None:
    """`lerobot:real` has no camera configured, so it has no `observe`: the bring-up task
    has to ask for something the arm it is pointed at actually provides."""
    duck = load_duck("lerobot-lookout")
    for backend in ("mock", "real"):
        manifest = describe(parse_robot_spec(f"lerobot:{backend}"))
        assert validate_duck(duck, [manifest]) == [], backend
    mock = LeRobotMock()
    adapter = LeRobotAdapter(mock)
    manifest = await adapter.connect()
    ex = Executor(
        registry_from_manifest(manifest, adapter),
        adapter,
        contract=duck.frontmatter,
        manifest=manifest,
    )
    assert (await ex.run_verb("report_state")).ok
    assert mock.actions == []


@pytest.mark.skipif(not NO_LEROBOT, reason="lerobot is installed here")
async def test_real_backend_without_the_extra_names_it() -> None:
    adapter = make_adapter(RobotSpec("lerobot", "real", "arm-01"), address="COM5")
    with pytest.raises(AdapterNotInstalled, match=r"quackd\[lerobot\]"):
        await adapter.connect()
    with pytest.raises(AdapterNotInstalled, match=r"quackd\[lerobot\]"):
        load_policy("some/checkpoint")


# ── what an adversarial review of the branch found, and what now pins it ────────────────


async def test_a_caller_already_queued_on_the_lock_is_refused_when_the_call_ahead_wedges() -> None:
    """The wedge is checked inside the lock as well as before it: the lock's release is what
    wakes the next caller, and that is exactly the moment a second thread must not start."""
    import threading

    release = threading.Event()
    arm = FakeArm()
    transport = LeRobotReal("COM5", robot=arm, timeout_s=0.2)
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    entered: list[str] = []

    def block() -> None:
        entered.append("blocker")
        release.wait(5.0)

    def second() -> None:
        entered.append("second")

    async def first() -> None:
        with pytest.raises(TimeoutError):
            await transport._call(block, deadline_s=0.2)

    async def queued() -> None:
        await asyncio.sleep(0.05)  # parked on the lock while `block` holds it
        with pytest.raises(TransportError, match="one owner"):
            await transport._call(second, deadline_s=1.0)

    try:
        await asyncio.gather(first(), queued())
        assert entered == ["blocker"]
    finally:
        release.set()


async def test_a_cancelled_call_wedges_like_a_timed_out_one() -> None:
    import threading

    release = threading.Event()
    arm = FakeArm()
    transport = LeRobotReal("COM5", robot=arm, timeout_s=5.0)
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    task = asyncio.create_task(transport._call(lambda: release.wait(5.0), deadline_s=5.0))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    try:
        assert transport.stop_error is not None
        assert adapter.stop_error is not None  # the core `stop` verb reads this one
        assert not (await adapter.send_intent(Intent.gripper(True))).accepted
    finally:
        release.set()


async def test_two_picks_at_once_leave_exactly_one_policy_loop_that_stop_cancels() -> None:
    class Forever:
        def act(self, observation: dict[str, Any], *, task: str) -> dict[str, float] | None:
            return {"shoulder_pan": 5.0}

    arm = FakeArm(camera=False)
    transport = LeRobotReal("COM5", robot=arm, policy=Forever())
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    await asyncio.gather(
        adapter.send_intent(Intent.do("policy:pick:a")),
        adapter.send_intent(Intent.do("policy:pick:b")),
        adapter.send_intent(Intent.do("policy:pick:c")),
    )
    await asyncio.sleep(0.15)
    await adapter.stop()
    assert not transport.policy_running
    sent = len(arm.actions)
    await asyncio.sleep(0.35)
    assert len(arm.actions) == sent, "an orphaned policy loop is still driving the arm"
    await adapter.close()


async def test_holding_is_never_true_from_two_samples_a_millisecond_apart() -> None:
    """A heartbeat probe and a verb's poll can land on the same reading. Two samples that
    close agree whatever the gripper is doing, so they say nothing about settling."""
    arm = FakeArm(step=5.0, object_in_jaws=True)
    transport = LeRobotReal("COM5", robot=arm)
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    assert (await adapter.send_intent(Intent.gripper(False))).accepted  # 100 -> 95
    await adapter.get_state()
    await adapter.get_state()  # the same reading, microseconds later
    assert not (await adapter.get_state()).holding


async def test_holding_is_false_on_an_empty_gripper_that_shuts() -> None:
    arm = FakeArm(object_in_jaws=False)
    adapter = LeRobotAdapter(LeRobotReal("COM5", robot=arm))
    manifest = await adapter.connect()
    ex = _executor(adapter, manifest)
    closed = await ex.run_verb("gripper", {"open": False})
    assert closed.ok and closed.data["holding"] is False and "on nothing" in closed.summary
    assert not (await adapter.get_state()).holding


async def test_a_gripper_that_does_not_move_is_a_failure_not_a_report() -> None:
    arm = FakeArm(stuck=("gripper",))
    arm.positions["gripper"] = 100.0
    adapter = LeRobotAdapter(LeRobotReal("COM5", robot=arm))
    manifest = await adapter.connect()
    ex = _executor(adapter, manifest)
    closed = await ex.run_verb("gripper", {"open": False})
    assert not closed.ok and "did not close" in closed.summary


async def test_a_small_step_cap_is_not_mistaken_for_a_stall() -> None:
    arm = FakeArm(step=0.5)
    transport = LeRobotReal("COM5", robot=arm, max_step_deg=0.5)
    adapter = LeRobotAdapter(transport)
    manifest = await adapter.connect()
    assert manifest.limits["step_deg"] == 0.5
    ex = _executor(adapter, manifest)
    moved = await ex.run_verb("move_joints", {"positions": {"shoulder_pan": 4}, "duration_s": 6})
    assert moved.ok, moved.summary


async def test_a_hot_gripper_does_not_gate_the_body_joints() -> None:
    arm = FakeArm()
    adapter = LeRobotAdapter(LeRobotReal("COM5", robot=arm))
    manifest = await adapter.connect()
    ex = _executor(adapter, manifest)
    arm.temperature["gripper"] = 66.0
    state = await adapter.get_state()
    assert state.extras["temperature_c"]["gripper"] == 66 and state.extras["hot"] == []
    assert (await ex.run_verb("move_joints", {"positions": {"shoulder_pan": 5}})).ok


async def test_the_pilot_is_told_the_calibrated_travel_and_not_the_schema_bound() -> None:
    from quackd.agent.prompts import body_lines

    arm = FakeArm()
    adapter = LeRobotAdapter(LeRobotReal("COM5", robot=arm))
    manifest = await adapter.connect()
    text = "\n".join(body_lines(manifest))
    assert "shoulder_pan -100 to 100" in text and "within 180 degrees" not in text
    health = await adapter.health()
    assert health.extras["calibration_file"].endswith("arm-01.json")
    assert health.extras["joint_range_deg"]["elbow_flex"] == [-100, 100]


async def test_a_reading_past_the_travel_is_explained_to_the_pilot_in_this_arm_s_numbers() -> None:
    """On the bench a model was handed a joint reading well past the travel line in its prompt,
    with nothing to explain it, and refused to move "on this inconsistent state". It was right
    to be suspicious and wrong about the cause: goals are clamped to the travel and readings
    are not. So `report_state` names each joint reading past it, what it reads and the limit
    it is past, computed off this arm, and the prompt's travel line states the rule. Neither
    claims how the joint got there: a joint parked at its limit that sagged past it qualifies
    too."""
    from quackd.agent.prompts import body_lines

    arm = _spanned()
    adapter = LeRobotAdapter(LeRobotReal("COM5", robot=arm))
    manifest = await adapter.connect()
    arm.positions["shoulder_lift"] = _past(arm, "shoulder_lift", -23.0)
    arm.positions["elbow_flex"] = _past(arm, "elbow_flex", 8.0)
    said = await _executor(adapter, manifest).run_verb("report_state")
    assert said.ok, said.summary
    ranges = manifest.extras["joint_range_deg"]
    for joint, end in (("shoulder_lift", 0), ("elbow_flex", 1)):
        reading = arm.positions[joint]
        assert (
            f"{joint} reads {reading:.0f}, past the {ranges[joint][end]:g} its servo can be "
            "driven to; goals are still limited to its travel"
        ) in said.summary, said.summary
    assert "shoulder_pan reads" not in said.summary, "a joint inside its travel is not news"
    assert "folded" not in said.summary and "torque off" not in said.summary
    # where the travel is not known to whoever asks, only what is known is said
    blind = await _executor(adapter, describe(parse_robot_spec("lerobot:real"))).run_verb(
        "report_state"
    )
    reading = arm.positions["shoulder_lift"]
    assert f"shoulder_lift reads {reading:.0f}, outside its calibrated travel" in blind.summary
    assert "its servo can be driven to" not in blind.summary, blind.summary

    travel = next(line for line in body_lines(manifest) if "Each joint's travel" in line)
    assert (
        "A joint can read past its travel when it was folded or placed there with torque off, "
        "which is where a rest pose usually is; goals are still limited to the travel."
    ) in travel, travel


async def test_report_state_puts_the_arm_s_own_facts_where_a_pilot_can_read_them() -> None:
    """A pilot reads a verb's summary text and never its data. The core verb's summary is a
    posture and a policy name, which an arm has not got, so `lerobot-lookout` asked for three
    things no real model could have seen. The arm supplies its own `report_state`."""
    from quackd.agent.prompts import build_observation_text

    adapter = LeRobotAdapter(LeRobotMock(hot_joints=("elbow_flex",)))
    manifest = await adapter.connect()
    ex = _executor(adapter, manifest)
    result = await ex.run_verb("report_state")
    assert result.ok
    for expected in ("shoulder_pan", "torque on", "TOO HOT TO MOVE: elbow_flex", "holding nothing"):
        assert expected in result.summary, result.summary
    # and it survives into the observation the model is actually handed
    text = build_observation_text(
        step=1,
        max_steps=12,
        state=await adapter.get_state(),
        detections=[],
        last_verb="report_state",
        last_result=result,
        budget_status="0/12",
    )
    assert "torque on" in text and "elbow_flex" in text


# ── the camera: one USB webcam, quackd's own, beside the follower ───────────────────────


class FakeCamera:
    """The slice of a LeRobot `Camera` quackd touches, verified names only."""

    def __init__(
        self,
        *,
        fail_open: bool = False,
        stalled: bool = False,
        slow_open_s: float = 0.0,
        slow_close_s: float = 0.0,
    ) -> None:
        self.fail_open = fail_open
        self.stalled = stalled
        self.slow_open_s = slow_open_s
        self.slow_close_s = slow_close_s
        self.connected = False
        self.calls: list[str] = []

    @property
    def is_connected(self) -> bool:
        return self.connected

    def connect(self, warmup: bool = True) -> None:
        self.calls.append("connect")
        if self.slow_open_s:
            time.sleep(self.slow_open_s)
        if self.fail_open:
            raise ConnectionError(
                "Failed to open OpenCVCamera(7).Run `lerobot-find-cameras opencv` to find "
                "available cameras."
            )
        self.connected = True

    def read_latest(self, max_age_ms: int = 500) -> Any:
        if self.stalled:
            raise TimeoutError("OpenCVCamera(0) latest frame is too old: 1200.0 ms")
        return np.zeros((48, 64, 3), dtype=np.uint8)

    def disconnect(self) -> None:
        self.calls.append("disconnect")
        if self.slow_close_s:
            time.sleep(self.slow_close_s)
        self.connected = False


def _camera_url(**query: Any) -> str:
    tail = "&".join(f"{k}={v}" for k, v in query.items())
    return f"opencv://0?{tail}" if tail else "opencv://0"


def test_a_camera_url_is_read_strictly_or_refused() -> None:
    spec = parse_camera_url("opencv://0?width=640&height=480&fps=30&backend=msmf&name=wrist")
    assert spec.index_or_path == 0 and spec.name == "wrist" and spec.backend == "msmf"
    assert (spec.width, spec.height, spec.fps) == (640, 480, 30)
    bare = parse_camera_url("opencv://0")
    # nothing is asked of the camera by default, so it keeps the mode it already has
    assert (bare.width, bare.height, bare.fps, bare.fourcc) == (None, None, None, None)
    assert bare.name == "front" and bare.backend == "any" and bare.rotation == 0
    assert parse_camera_url("opencv:///dev/video2").index_or_path == "/dev/video2"
    assert parse_camera_url("opencv://0?fov=70").fov_deg == 70.0
    for bad in (
        "http://host/snapshot.jpg",
        "0",
        "opencv://",
        "opencv://0?width=abc",
        "opencv://0?height=0",
        "opencv://0?backend=cuda",
        "opencv://0?rotation=45",
        "opencv://0?fourcc=MJP",
        "opencv://0?zoom=2",
    ):
        with pytest.raises(AdapterError, match="opencv://0"):
            parse_camera_url(bad)


def test_the_camera_is_not_the_followers_and_the_config_says_so() -> None:
    """A follower's cameras are part of its connected state, so a webcam that came unplugged
    would make every move and every hold raise. quackd passes cameras={} and owns its own."""
    transport = LeRobotReal("COM5", camera=parse_camera_url("opencv://0"))
    assert transport._config_kwargs()["cameras"] == {}


async def test_a_camera_gives_the_arm_observe_and_a_frame() -> None:
    camera = FakeCamera()
    transport = LeRobotReal(
        "COM5",
        robot=FakeArm(),
        camera=parse_camera_url(_camera_url(fov=70)),
        camera_object=camera,
    )
    adapter = LeRobotAdapter(transport)
    manifest = await adapter.connect()
    assert camera.calls == ["connect"]
    assert manifest.provides("observe") and "camera" in manifest.sensors
    assert manifest.extras["camera"] == "opencv://0?fov=70"
    assert manifest.limits["camera_fov_deg"] == 70.0
    frame = await adapter.get_frame()
    assert frame is not None and frame.size == (64, 48)
    health = transport.camera_health()  # the method `doctor` reaches, past the adapter
    assert health["ok"] and health["size"] == "64x48"
    ex = _executor(adapter, manifest, detector=ColorBlobDetector())
    assert (await ex.run_verb("observe")).ok
    await adapter.close()
    assert camera.calls == ["connect", "disconnect"]


async def test_a_camera_that_will_not_open_refuses_and_leaves_the_arm_clean() -> None:
    arm = FakeArm()
    adapter = LeRobotAdapter(
        LeRobotReal(
            "COM5",
            robot=arm,
            camera=parse_camera_url("opencv://7"),
            camera_object=FakeCamera(fail_open=True),
        )
    )
    with pytest.raises(TransportError, match="opencv://7"):
        await adapter.connect()
    # the camera opens before the arm is touched, so a bad index energises nothing and
    # lets nothing go slack on the way out
    assert arm.calls == []


async def test_a_stalled_camera_costs_the_picture_and_not_the_run() -> None:
    """`observe` moves nothing, so a camera that stopped delivering must not end a session:
    the arm keeps answering and the frame says why it is missing."""
    arm = FakeArm()
    camera = FakeCamera()
    transport = LeRobotReal(
        "COM5", robot=arm, camera=parse_camera_url("opencv://0"), camera_object=camera
    )
    adapter = LeRobotAdapter(transport)
    manifest = await adapter.connect()
    ex = _executor(adapter, manifest)
    camera.stalled = True
    assert await adapter.get_frame() is None
    assert "TimeoutError" in (adapter.camera_error or "")
    health = transport.camera_health()  # the method `doctor` reaches, past the adapter
    assert not health["ok"] and "too old" in health["error"]
    observed = await ex.run_verb("observe")
    assert not observed.ok and "too old" in observed.summary
    # the arm is untouched by any of it
    await adapter.heartbeat()
    moved = await ex.run_verb(
        "move_joints", {"positions": {"shoulder_pan": 10}, "duration_s": MOVE_MIN_S}
    )
    assert moved.ok, moved.summary


async def test_the_policy_is_handed_the_frame_under_the_cameras_own_name() -> None:
    seen: list[str] = []

    class Peeking:
        def act(self, observation: dict[str, Any], *, task: str) -> dict[str, float] | None:
            seen.extend(k for k in observation if not k.endswith(".pos"))
            return None

    adapter = LeRobotAdapter(
        LeRobotReal(
            "COM5",
            robot=FakeArm(object_in_jaws=True),
            policy=Peeking(),
            camera=parse_camera_url(_camera_url(name="wrist")),
            camera_object=FakeCamera(),
        )
    )
    manifest = await adapter.connect()
    ex = Executor(registry_from_manifest(manifest, adapter), adapter, confirm=allow_all)
    await ex.run_verb("pick", {"target": "cup", "max_s": 3})
    assert "wrist" in seen
    await adapter.close()


async def test_a_hold_that_never_reached_the_arm_is_not_reported_as_stopped() -> None:
    arm = FakeArm()
    transport = LeRobotReal("COM5", robot=arm)
    adapter = LeRobotAdapter(transport)
    manifest = await adapter.connect()
    ex = _executor(adapter, manifest)
    # reads still work, the write does not: the arm answers every question and obeys none
    arm.send_fails = True
    stopped = await ex.run_verb("stop")
    assert not stopped.ok and "could not be delivered" in stopped.summary
    assert "Goal_Position" in (adapter.stop_error or "")
    arm.send_fails = False
    assert (await ex.run_verb("stop")).ok and adapter.stop_error is None


async def test_a_stop_over_a_folded_arm_leaves_the_fold_alone() -> None:
    """What `stop` did to the bench arm, which is why this exists. A hold writes each joint's
    present position as its goal, and for a joint folded past its travel the servo clamps that
    goal to the limit and drives there at full speed: the stop at the end of a run hauled a
    folded shoulder up out of its fold, and the record said only that it had stopped.

    A joint reading past its travel is left out of the hold, either way past it. The joints
    inside their travel are held as they always were, and nothing moves.

    And the stop says which it left alone. It used to answer "stopped (velocity zeroed)" over
    a hold of three joints out of five in the same words as a hold of all five, so neither
    the pilot nor the record could tell that two joints had been written no goal. A stop over
    an arm back inside its travel says nothing more, because the list is the last hold's."""
    arm = _spanned()
    transport = LeRobotReal("COM5", robot=arm)
    adapter = LeRobotAdapter(transport)
    manifest = await adapter.connect()
    ex = _executor(adapter, manifest)
    arm.positions["shoulder_lift"] = _past(arm, "shoulder_lift", -21.0)
    arm.positions["elbow_flex"] = _past(arm, "elbow_flex", 11.0)
    arm.positions["wrist_flex"] = _inside(arm, "wrist_flex", 0.6)
    before = dict(arm.positions)

    stopped = await ex.run_verb("stop")
    assert stopped.ok, stopped.summary
    assert adapter.stop_error is None
    assert set(arm.actions[-1]) == {"shoulder_pan.pos", "wrist_flex.pos", "wrist_roll.pos"}
    assert arm.actions[-1]["wrist_flex.pos"] == before["wrist_flex"]
    assert arm.positions == before, "the stop moved the arm"
    assert adapter.stop_skipped == ("shoulder_lift", "elbow_flex")
    assert stopped.summary == (
        "stopped (velocity zeroed); shoulder_lift and elbow_flex read past their travel, so no "
        "goal was written for them"
    ), stopped.summary
    assert stopped.data["not_held"] == ["shoulder_lift", "elbow_flex"]

    # a hold that never read the arm cannot say what it left alone, and must not repeat the
    # list of one that did
    arm.dead = True
    await adapter.stop()
    assert adapter.stop_error is not None, "the hold never read the arm"
    assert adapter.stop_skipped == (), "a failed hold kept the last hold's list"
    arm.dead = False

    arm.positions["elbow_flex"] = _inside(arm, "elbow_flex", 0.2)
    one = await ex.run_verb("stop")
    assert one.summary.endswith(
        "shoulder_lift reads past its travel, so no goal was written for it"
    ), one.summary

    arm.positions["shoulder_lift"] = _inside(arm, "shoulder_lift", -0.2)
    again = await ex.run_verb("stop")
    assert again.ok and again.summary == "stopped (velocity zeroed)", again.summary
    assert adapter.stop_skipped == (), "the last hold's list outlived it"


async def test_a_stop_with_every_body_joint_past_its_travel_sends_nothing_and_is_a_stop() -> None:
    """The branch where the skip leaves nothing to send. Nothing is written, and the stop is
    not reported as undelivered, because it started nothing. That is all it can say: any goal
    quackd writes to a joint while it reads past its travel is the limit to the servo, so a
    joint a move had begun lifting out of its fold goes on rising to that limit whatever a stop
    does, and only the power switch stops that stretch. What the stop owes the pilot is the list of
    joints it wrote no goal for, which here is all five. A stop reported as undelivered over
    an arm lying still in its fold would send somebody for the switch for nothing."""
    arm = _spanned()
    transport = LeRobotReal("COM5", robot=arm)
    adapter = LeRobotAdapter(transport)
    manifest = await adapter.connect()
    ex = _executor(adapter, manifest)
    for joint, by in zip(SPANS, (-4.0, -19.0, 13.0, 6.5, -8.0), strict=True):
        arm.positions[joint] = _past(arm, joint, by)
    before = dict(arm.positions)

    stopped = await ex.run_verb("stop")
    assert stopped.ok, stopped.summary
    assert adapter.stop_error is None
    assert arm.actions == [], "a goal was written to a joint past its travel"
    assert arm.positions == before
    assert stopped.data["not_held"] == list(SPANS), stopped.data
    assert "shoulder_pan, shoulder_lift, elbow_flex, wrist_flex and wrist_roll read past" in (
        stopped.summary
    ), stopped.summary


async def test_a_slow_camera_release_never_costs_the_arm_its_disconnect() -> None:
    """The camera touches no bus, so it is never under the serial lock: a webcam whose
    release takes seconds (routine on Windows) must not be filed as a wedged serial call,
    which would refuse the arm's own disconnect and leave torque on."""
    arm = FakeArm()
    camera = FakeCamera(slow_close_s=0.4)
    transport = LeRobotReal(
        "COM5", robot=arm, camera=parse_camera_url("opencv://0"), camera_object=camera
    )
    transport.camera_close_s = 0.1
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    await adapter.close()
    assert ("disconnect",) in arm.calls, "the arm's disconnect was skipped"
    assert transport._wedged is None and transport.stop_error is None


async def test_a_camera_that_hangs_on_open_says_so_and_leaves_the_arm_alone() -> None:
    arm = FakeArm()
    transport = LeRobotReal(
        "COM5",
        robot=arm,
        camera=parse_camera_url("opencv://3"),
        camera_object=FakeCamera(slow_open_s=0.4),
    )
    transport.camera_connect_s = 0.1
    with pytest.raises(TransportError, match="did not return within"):
        await LeRobotAdapter(transport).connect()
    assert arm.calls == [] and transport._wedged is None


async def test_an_arm_that_refuses_after_the_camera_opened_lets_the_camera_go() -> None:
    camera = FakeCamera()
    adapter = LeRobotAdapter(
        LeRobotReal(
            "COM5",
            robot=FakeArm(calibrated=False),
            camera=parse_camera_url("opencv://0"),
            camera_object=camera,
        )
    )
    with pytest.raises(TransportError, match="not calibrated"):
        await adapter.connect()
    assert camera.calls == ["connect", "disconnect"]


@pytest.mark.skipif(not NO_LEROBOT, reason="lerobot is installed here")
async def test_a_camera_that_cannot_be_built_never_touches_the_arm() -> None:
    arm = FakeArm()
    adapter = LeRobotAdapter(LeRobotReal("COM5", robot=arm, camera=parse_camera_url("opencv://0")))
    with pytest.raises(AdapterNotInstalled):
        await adapter.connect()
    assert arm.calls == []


def test_the_documented_defaults_and_a_real_lens_are_accepted() -> None:
    assert parse_camera_url("opencv://0?rotation=0").rotation == 0
    assert parse_camera_url("opencv://0?rotation=270").rotation == 270
    assert parse_camera_url("opencv://0?fov=62.5").fov_deg == 62.5
    for bad in ("opencv://0?width=1280", "opencv://0?height=720", "opencv://0?fov=0"):
        with pytest.raises(AdapterError, match="opencv://0"):
            parse_camera_url(bad)


async def test_a_camera_that_died_is_in_the_arms_own_report() -> None:
    """A run that cannot call `observe` would otherwise lose the camera in silence: the
    frames stop, the observation loses a line, and nothing in the transcript says why. So the
    health goes into the state every heartbeat reads, and report_state says it out loud."""
    arm = FakeArm()
    camera = FakeCamera()
    adapter = LeRobotAdapter(
        LeRobotReal("COM5", robot=arm, camera=parse_camera_url("opencv://0"), camera_object=camera)
    )
    manifest = await adapter.connect()
    ex = _executor(adapter, manifest)

    healthy = await ex.run_verb("report_state")
    assert healthy.ok and "CAMERA DOWN" not in healthy.summary, "a working camera is not news"

    camera.stalled = True
    assert await adapter.get_frame() is None
    said = await ex.run_verb("report_state")
    assert said.ok, "a dead camera is not a failed read of the arm"
    assert "CAMERA DOWN" in said.summary and "too old" in said.summary
    health = said.data["state"]["extras"]["camera"]
    assert health["configured"] and not health["ok"] and "too old" in health["error"]


async def test_an_arm_without_a_camera_says_nothing_about_one() -> None:
    adapter = LeRobotAdapter(LeRobotReal("COM5", robot=FakeArm()))
    manifest = await adapter.connect()
    said = await _executor(adapter, manifest).run_verb("report_state")
    assert said.ok and "CAMERA" not in said.summary
    assert "camera" not in said.data["state"]["extras"]


# ── the rest pose: where the arm is put down before torque is let go ────────────────────

FOLDED = {
    "shoulder_pan": -20.0,
    "shoulder_lift": -90.0,
    "elbow_flex": 90.0,
    "wrist_flex": 45.0,
    "wrist_roll": 30.0,
    "gripper": 100.0,
}
BODY_JOINTS = {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"}
"""Named here rather than derived from `rest_goal`, so a `rest_goal` that quietly stopped
parking one of them has something to fail against. Every value in `FOLDED` is away from the
zero a `FakeArm` starts at, for the same reason: a joint that never had to move proves nothing."""
"""A pose recorded off an arm, gripper and all. Only the five body joints are ever driven."""


async def test_the_rest_move_drives_the_five_body_joints_and_never_the_gripper() -> None:
    """The gripper is left out of the goal for the reason a hold leaves it out: LeRobot
    writes only the keys it is given, so a rest move that re-sent the gripper would open a
    hand that is holding something on its way to being put down. The move also has to happen
    while the arm is still connected, which is the failure the whole thing exists for."""
    arm = FakeArm(step=40.0)
    transport = LeRobotReal("COM5", robot=arm, rest_pose=dict(FOLDED))
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    result = await adapter.go_to_rest()
    assert result.how == "arrived", result.reason
    assert result.reached and result.recorded
    assert arm.actions, "the rest goal never reached the arm"
    assert all("gripper.pos" not in action for action in arm.actions), arm.actions
    sent = {key.removesuffix(".pos") for action in arm.actions for key in action}
    assert sent == BODY_JOINTS, "every body joint is driven, and only those"
    for joint in BODY_JOINTS:
        assert abs(arm.positions[joint] - FOLDED[joint]) <= TOL_DEG, (
            joint,
            arm.positions[joint],
        )
    assert arm.positions["gripper"] == 100.0, "the rest move squeezed the gripper"
    assert ("disconnect",) not in arm.calls, "the arm was let go of before it was parked"
    await adapter.close()
    assert ("disconnect",) in arm.calls
    assert arm.torque_disabled == 1 and arm.torque is False
    assert transport.close_note is None and adapter.close_note is None


async def test_a_transport_that_missed_its_pose_once_does_not_keep_torque_on_for_ever() -> None:
    """The flag lives on the robot rather than on the call, so writing it only when torque had
    to stay on left it off afterwards. A later session that did reach the pose would then let
    go of an arm it had quietly kept energised, on the strength of a session that had already
    ended, and say nothing about it."""
    arm = FakeArm(step=40.0, stuck=("shoulder_lift",))
    transport = LeRobotReal("COM5", robot=arm, rest_pose=dict(FOLDED))
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    assert (await adapter.go_to_rest()).how == "stalled"
    await adapter.close()
    assert arm.torque is True, "the arm never reached its pose, so it keeps holding"
    assert transport.close_note is not None

    # the same transport again, this time already where it was asked to be
    arm.stuck = ()
    arm.positions.update(rest_goal(FOLDED))
    await adapter.connect()
    assert (await adapter.go_to_rest()).how == "already"
    await adapter.close()
    assert arm.torque is False, "the arm is at its pose, so torque may drop"
    assert transport.close_note is None, "nothing to warn about the second time"


def test_a_pose_the_arm_did_not_report_is_not_a_pose_it_is_resting_in() -> None:
    """`at_rest` is read inside `close()`, where its answer decides whether torque drops. An
    arm that answered but said nothing about a joint the pose names is an arm nobody can
    place, and reading a missing joint as a match would let go of it on a guess."""
    goal = rest_goal(FOLDED)
    assert at_rest(goal, dict(goal))
    silent = {joint: value for joint, value in goal.items() if joint != "wrist_flex"}
    assert not at_rest(goal, silent), "a joint that did not report is not a joint at rest"
    assert not at_rest(goal, {}), "an arm that reported nothing is not at rest"


async def test_the_primary_camera_is_marked_rather_than_read_off_the_order() -> None:
    """A camera that gave nothing is absent from the list, so on a two-camera arm whose primary
    lens died the first entry is the other camera. Position is not identity: the detector reads
    the frame that says it is the primary, and a bearing taken off the wrong lens points
    somewhere `--fov-deg` never measured."""
    top, side = FakeCamera(), FakeCamera()
    transport = LeRobotReal(
        "COM5",
        robot=FakeArm(),
        cameras=parse_camera_urls(("opencv://1?name=top", "opencv://2?name=side")),
        camera_objects={"top": top, "side": side},
    )
    await transport.connect()
    frames = await transport.get_frames()
    assert [(f.name, f.primary) for f in frames] == [("top", True), ("side", False)]
    assert primary_of(frames) is not None

    top.stalled = True
    frames = await transport.get_frames()
    assert [f.name for f in frames] == ["side"], "the live camera still answers"
    assert not any(f.primary for f in frames), "the primary is dead, so no frame claims to be it"
    assert primary_of(frames) is None, "nothing to run the detector over"


SPANS = {
    "shoulder_pan": 190.0,
    "shoulder_lift": 150.0,
    "elbow_flex": 170.0,
    "wrist_flex": 130.0,
    "wrist_roll": 250.0,
}
"""A synthetic calibration, a different travel on every body joint. Nothing here is any real
arm's: the rules under test read the travel off whatever calibration the arm answers with, so
the tests build their poses from that travel rather than from numbers anybody measured."""


def _spanned(**kwargs: Any) -> FakeArm:
    """A `FakeArm` calibrated with `SPANS`, so each body joint has its own travel."""
    arm = FakeArm(**kwargs)
    for joint, travel_deg in SPANS.items():
        arm.calibration[joint] = FakeCalibration(travel_deg)
    return arm


def _past(arm: FakeArm, joint: str, by: float) -> float:
    """An angle `by` degrees past `joint`'s travel: below its floor when `by` is negative,
    above its ceiling when it is positive. Where a fold past the travel would be recorded."""
    lo, hi = arm.travel(joint)
    return lo + by if by < 0 else hi + by


def _inside(arm: FakeArm, joint: str, share: float) -> float:
    """An angle inside `joint`'s travel, `share` of the way from its middle to an end."""
    lo, hi = arm.travel(joint)
    return share * (hi if share > 0 else -lo)


def _within_travel(arm: FakeArm, action: dict[str, float]) -> bool:
    return all(
        arm.travel(key.removesuffix(".pos"))[0] - 1e-9
        <= value
        <= arm.travel(key.removesuffix(".pos"))[1] + 1e-9
        for key, value in action.items()
        if key != "gripper.pos"
    )


async def test_a_rest_pose_past_the_travel_parks_at_the_limit_and_is_let_go_of_there() -> None:
    """The rest pose the bench arm was recorded in had its shoulder folded past the travel its
    calibration recorded, and a test here claimed that pose "is still driven to". The fake
    arm followed any goal, so it was; the real one never could, because the servo clamps
    every goal to the limits calibration wrote into it. The rest move stalled at the limit
    and aborted the run before the first model call, and the close then kept torque on and
    told a person to cut the power.

    So the pose is clipped into the travel, one joint below its floor and another above its
    ceiling here, on a calibration with a different span on every joint. The arm parks at the
    two limits and has arrived; nothing ever sends the angle it cannot reach; the manifest and
    the result both say which joints were clipped, with this arm's own numbers; and the close
    lets go of it with nothing to warn about."""
    arm = _spanned(step=40.0)
    pose = {
        "shoulder_pan": _inside(arm, "shoulder_pan", 0.2),
        "shoulder_lift": _past(arm, "shoulder_lift", -22.0),
        "elbow_flex": _past(arm, "elbow_flex", 16.0),
        "wrist_flex": _inside(arm, "wrist_flex", -0.3),
        "wrist_roll": _inside(arm, "wrist_roll", 0.1),
        "gripper": 100.0,
    }
    lift_floor = arm.travel("shoulder_lift")[0]
    elbow_ceiling = arm.travel("elbow_flex")[1]
    transport = LeRobotReal("COM5", robot=arm, rest_pose=pose)
    adapter = LeRobotAdapter(transport)
    manifest = await adapter.connect()
    assert manifest.extras["rest_pose_clipped"] == {
        "shoulder_lift": {
            "recorded": round(pose["shoulder_lift"], 1),
            "reachable": round(lift_floor, 1),
        },
        "elbow_flex": {
            "recorded": round(pose["elbow_flex"], 1),
            "reachable": round(elbow_ceiling, 1),
        },
    }

    result = await adapter.go_to_rest()
    assert result.how == "arrived" and result.reached, result.reason
    assert result.reason == "moved to the rest pose"
    assert [(j, r) for j, r, _ in result.clipped] == [
        ("shoulder_lift", pose["shoulder_lift"]),
        ("elbow_flex", pose["elbow_flex"]),
    ]
    assert [v for _, _, v in result.clipped] == pytest.approx([lift_floor, elbow_ceiling])
    assert abs(arm.positions["shoulder_lift"] - lift_floor) <= TOL_DEG
    assert abs(arm.positions["elbow_flex"] - elbow_ceiling) <= TOL_DEG
    assert all(_within_travel(arm, action) for action in arm.actions), arm.actions
    assert transport._range_clips == 0, "the reachable goal is inside the travel already"
    note = result.note or ""
    for joint, recorded, reachable in result.clipped:
        assert f"{joint} at {recorded:.0f}" in note, note
        assert f"{reachable:.0f}" in note, note
    assert "lerobot-calibrate" in note and "quackd robot rest-pose NAME" in note, note

    await adapter.close()
    assert arm.torque_disabled == 1 and arm.torque is False, "parked, so torque may drop"
    assert adapter.close_note is None, "a release at the reachable pose has nothing to warn"


async def test_a_pose_with_every_body_joint_past_its_travel_parks_every_one_at_its_limit() -> None:
    """Nothing about the rule is special to one joint or one end. Every body joint here is
    recorded past its own travel, three below their floors and two above their ceilings, and
    every one parks at its own limit, is named, and is let go of there."""
    arm = _spanned(step=60.0)
    beyond = {
        "shoulder_pan": -9.0,
        "shoulder_lift": -31.0,
        "elbow_flex": 12.0,
        "wrist_flex": -7.5,
        "wrist_roll": 18.0,
    }
    pose = {joint: _past(arm, joint, by) for joint, by in beyond.items()}
    transport = LeRobotReal("COM5", robot=arm, rest_pose=pose)
    adapter = LeRobotAdapter(transport)
    manifest = await adapter.connect()
    assert set(manifest.extras["rest_pose_clipped"]) == set(beyond)

    result = await adapter.go_to_rest()
    assert result.how == "arrived", result.reason
    assert [joint for joint, _, _ in result.clipped] == list(beyond)
    for joint, by in beyond.items():
        lo, hi = arm.travel(joint)
        limit = lo if by < 0 else hi
        assert abs(arm.positions[joint] - limit) <= TOL_DEG, (joint, arm.positions[joint])
    assert all(_within_travel(arm, action) for action in arm.actions), arm.actions
    assert "each parks there" in (result.note or ""), result.note
    await adapter.close()
    assert arm.torque is False and adapter.close_note is None


async def test_a_pose_inside_the_travel_is_driven_exactly_as_it_always_was() -> None:
    """The clip is for a pose past the travel and for nothing else. A pose inside it, on the
    same several-span calibration, gets the result it always got, byte for byte, and the
    manifest gains no key: every arm whose fold is inside its travel is unchanged."""
    from quackd.adapters.base import RestResult

    arm = _spanned(step=40.0)
    pose = {
        joint: _inside(arm, joint, share)
        for joint, share in zip(SPANS, (0.3, -0.8, 0.9, -0.5, 0.2), strict=True)
    }
    transport = LeRobotReal("COM5", robot=arm, rest_pose=pose)
    adapter = LeRobotAdapter(transport)
    manifest = await adapter.connect()
    assert "rest_pose_clipped" not in manifest.extras
    assert transport.rest_reachable == pose and transport.rest_clipped == ()
    result = await adapter.go_to_rest()
    assert result == RestResult("arrived", "moved to the rest pose")
    assert result.clipped == () and result.note is None
    for joint, value in pose.items():
        assert abs(arm.positions[joint] - value) <= TOL_DEG
    await adapter.close()
    assert arm.torque is False and adapter.close_note is None


async def test_an_arm_folded_past_its_travel_is_at_rest_and_is_never_hauled_up_to_the_limit() -> (
    None
):
    """The run starts where the last one left the arm, folded, and the old rest move sent that
    fold's joint its goal. The goal was the limit, the servo drove there at full speed, and
    the arm was hauled up out of its fold before the run had asked the model anything.

    A joint reading past its limit on the side its pose was recorded is at rest, and is sent
    nothing. Here one joint is folded past its floor and another past its ceiling, and a
    third is away from its pose: the rest move drives the third and only the third."""
    arm = _spanned(step=40.0)
    pose = {
        "shoulder_pan": _inside(arm, "shoulder_pan", -0.2),
        "shoulder_lift": _past(arm, "shoulder_lift", -26.0),
        "elbow_flex": _past(arm, "elbow_flex", 14.0),
        "wrist_flex": _inside(arm, "wrist_flex", 0.4),
        "wrist_roll": 0.0,
    }
    transport = LeRobotReal("COM5", robot=arm, rest_pose=pose)
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    # the arm is exactly where the pose was recorded: folded past both limits
    arm.positions.update(pose)
    already = await adapter.go_to_rest()
    assert already.how == "already" and already.reached, already.reason
    assert arm.actions == [], "a folded arm was sent a goal"
    assert already.note is not None, "the note is about the pose, and a folded arm has it"

    # the folds sag a little further and the pan is knocked away: only the pan is driven
    arm.positions["shoulder_lift"] -= 3.0
    arm.positions["elbow_flex"] += 2.0
    arm.positions["shoulder_pan"] = _inside(arm, "shoulder_pan", 0.9)
    folded = {j: arm.positions[j] for j in ("shoulder_lift", "elbow_flex")}
    moved = await adapter.go_to_rest()
    assert moved.how == "arrived", moved.reason
    assert arm.actions, "the pan was away from its pose and was not driven"
    for action in arm.actions:
        assert "shoulder_lift.pos" not in action and "elbow_flex.pos" not in action, action
    assert {j: arm.positions[j] for j in folded} == folded, "a fold was moved"

    # `--by-hand` releases it there, since the fold is its rest pose, and does not take hold
    # again while the folds read past their travel: a goal for either is its limit, and no
    # goal leaves it the last one its servo had (the take-hold tests below say why). Nobody
    # lifted it, and the close reads it lying in its fold with every motor off, so it says the
    # arm is limp at its rest pose rather than telling somebody to put down an arm that is down
    written = len(arm.actions)
    released = await adapter.let_go()
    assert released.how == "released", released.reason
    refused = await adapter.take_hold()
    assert refused.how == "refused", refused.reason
    assert "shoulder_lift" in refused.reason and "elbow_flex" in refused.reason, refused.reason
    assert refused.energised is False, "refused before its torque write, so nothing came on"
    assert len(arm.actions) == written and arm.torque is False, arm.timeline
    assert {j: arm.positions[j] for j in folded} == folded, "a fold was moved"
    await adapter.close()
    assert arm.torque is False
    assert adapter.close_note == LIMP_AT_REST, adapter.close_note


async def test_a_joint_stopped_short_inside_its_travel_is_still_a_miss_and_keeps_torque() -> None:
    """The half-line is for a fold past the travel, never for a joint stopped short of its
    pose. A clipped joint that the desk or a hand holds up inside its travel has not reached
    anything, and letting go of it drops the arm, so torque stays on exactly as before, with
    the reachable goal named. No sentence about settling is said over an arm that did not
    park."""
    arm = _spanned(step=40.0, stuck=("shoulder_lift",))
    pose = {"shoulder_lift": _past(arm, "shoulder_lift", -18.0), "wrist_flex": 0.0}
    floor = arm.travel("shoulder_lift")[0]
    transport = LeRobotReal("COM5", robot=arm, rest_pose=pose)
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    result = await adapter.go_to_rest()
    assert result.how == "stalled" and not result.reached, result.reason
    assert result.note is None, "a note about settling over an arm that is not parked"
    assert [j for j, _, _ in result.clipped] == ["shoulder_lift"]
    await adapter.close()
    assert arm.torque is True and arm.torque_disabled == 0, "a missed pose keeps torque"
    note = adapter.close_note or ""
    assert note.startswith(TORQUE_LEFT_ON.split("(")[0]), note
    assert f"shoulder_lift is at 0 with a goal of {floor:.0f}" in note, note


async def test_a_miss_names_the_joint_that_is_short_and_never_the_one_that_is_folded() -> None:
    """A folded joint reads far from its reachable goal and is at rest; a stuck one reads
    nearer its goal and is not. Naming the furthest joint by distance alone sends somebody to
    look at the fold, which is the one joint that is fine."""
    arm = _spanned(step=40.0, stuck=("elbow_flex",))
    pose = {"shoulder_lift": _past(arm, "shoulder_lift", -30.0), "elbow_flex": 20.0}
    transport = LeRobotReal("COM5", robot=arm, rest_pose=pose)
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    arm.positions["shoulder_lift"] = pose["shoulder_lift"]  # folded, far past its floor
    result = await adapter.go_to_rest()
    assert result.how == "stalled", result.reason
    assert "elbow_flex is at 0 with a goal of 20" in result.reason, result.reason
    assert "shoulder_lift" not in result.reason, result.reason
    assert all("shoulder_lift.pos" not in action for action in arm.actions), arm.actions


def test_at_rest_is_a_half_line_on_the_side_a_clipped_joint_was_recorded() -> None:
    """The rule on its own, both ways round, on a goal and pose of no particular arm. Past the
    limit nothing quackd sends moves a servo there, so a reading anywhere out there is a fold;
    short of it by more than the tolerance is a joint that has not arrived. A joint recorded
    inside its travel keeps the point rule, and the two-argument call is that rule for all."""
    floor, ceiling, inside = -61.0, 47.0, 12.0
    goal = {"a": floor, "b": ceiling, "c": inside}
    recorded = {"a": floor - 25.0, "b": ceiling + 9.0, "c": inside}
    folded = {"a": floor - 40.0, "b": ceiling + 30.0, "c": inside + TOL_DEG}
    assert at_rest(goal, folded, recorded)
    assert at_rest(goal, {"a": floor + TOL_DEG, "b": ceiling - TOL_DEG, "c": inside}, recorded)
    for joint, short in (("a", floor + TOL_DEG + 1), ("b", ceiling - TOL_DEG - 1)):
        assert not at_rest(goal, folded | {joint: short}, recorded), joint
    # the point rule for the joint recorded inside its travel, in both directions
    assert not at_rest(goal, folded | {"c": inside + TOL_DEG + 1}, recorded)
    assert not at_rest(goal, folded | {"c": inside - TOL_DEG - 1}, recorded)
    # and without the recorded pose every joint is a point, as it always was
    assert not at_rest(goal, folded)
    assert at_rest(goal, dict(goal))
    assert not at_rest(goal, {"a": floor, "b": ceiling}, recorded), "c did not report"


def test_the_reachable_goal_clips_each_joint_to_its_own_travel_and_nothing_else() -> None:
    """Any joint, either end, any number of them. A joint with no known range passes through,
    because there is nothing to clip it to, and the gripper is never in the goal at all."""
    ranges = {"shoulder_pan": (-80.0, 95.0), "elbow_flex": (-70.0, 66.0), "gripper": (0.0, 100.0)}
    pose = {"shoulder_pan": -93.0, "elbow_flex": 71.5, "wrist_flex": -140.0, "gripper": 120.0}
    goal, clipped = reachable_rest_goal(pose, ranges)
    assert goal == {"shoulder_pan": -80.0, "elbow_flex": 66.0, "wrist_flex": -140.0}
    assert clipped == (("shoulder_pan", -93.0, -80.0), ("elbow_flex", 71.5, 66.0))
    # clipped by more than the tolerance is worth a sentence; by less it is the same pose
    assert worth_saying(clipped) == (("shoulder_pan", -93.0, -80.0), ("elbow_flex", 71.5, 66.0))
    assert worth_saying((("elbow_flex", 66.0 + TOL_DEG / 2, 66.0),)) == ()
    assert reachable_rest_goal({"elbow_flex": 10.0}, ranges) == ({"elbow_flex": 10.0}, ())


async def test_an_arm_that_cannot_reach_its_rest_pose_keeps_its_torque_and_says_so() -> None:
    """LeRobot's disconnect drops torque by its own default, which is right for an arm that
    is folded down and wrong for one stopped halfway there: on the bench on 2026-09-15 the
    arm fell at the end of every run. The flag is read off the config instance inside
    disconnect() rather than copied at construction, so close() turns it off for this case
    and this case only, and the arm is still let go of either way."""
    arm = FakeArm(step=40.0, stuck=("shoulder_lift",))
    transport = LeRobotReal("COM5", robot=arm, rest_pose=dict(FOLDED))
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    result = await adapter.go_to_rest()
    assert result.how == "stalled" and not result.reached and result.recorded
    assert "shoulder_lift" in result.reason and "stopped moving" in result.reason
    # a hold followed, so the servo stops pushing at a goal it has been told it cannot reach
    assert set(arm.actions[-1]) == {f"{j}.pos" for j in JOINTS if j != "gripper"}
    await adapter.close()
    assert ("disconnect",) in arm.calls, "the arm was never let go of"
    assert arm.config.disable_torque_on_disconnect is False
    assert arm.torque_disabled == 0 and arm.torque is True
    note = adapter.close_note
    assert note is not None
    assert "torque was left on" in note and "it will not fall" in note
    assert "shoulder_lift" in note, note


async def test_a_pose_that_names_no_joint_this_arm_drives_is_refused_rather_than_ignored() -> None:
    """`quackd robot rest-pose` reads the pose off the arm and cannot write one of these. A
    hand-edited `robots.json` can, by typing `elbow` where the arm says `elbow_flex`, and the
    registry does not know this arm's motors and should not.

    Left alone it is the worst possible outcome: the pose is stored, `robot show` prints it,
    nothing is driven anywhere, and torque drops where the arm happens to stand. So the arm
    refuses it by name, the run aborts before the pilot is asked anything, and an arm that
    somehow reached a close anyway keeps holding itself up."""
    with pytest.raises(AdapterError, match="names no joint this arm drives"):
        make("real", address="COM5", rest_pose={"elbow": 0.0, "gripper": 50.0})

    # and the transport underneath refuses too, since the adapter is not the only way in
    arm = FakeArm()
    transport = LeRobotReal("COM5", robot=arm, rest_pose={"elbow": 0.0})
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    result = await adapter.go_to_rest()
    assert result.how == "refused", result
    assert result.recorded and not result.reached, "a run over this must abort, not proceed"
    assert arm.actions == []
    await adapter.close()
    assert arm.config.disable_torque_on_disconnect is False, "the arm was let go anyway"
    assert arm.torque is True
    assert "torque was left on" in (adapter.close_note or ""), adapter.close_note


async def test_an_arm_quackd_cannot_keep_powered_is_told_so_instead_of_the_opposite() -> None:
    """The one seam that holds torque is a flag written on LeRobot's config just before the
    disconnect that reads it. If that write does not take, the disconnect releases torque
    anyway, and the usual note would tell somebody the arm is being held while it goes limp.

    A config object that refuses writes is not a case anybody has seen; it is a case where
    being wrong sends a person away from a falling arm, which is the whole subject."""

    class NoWrites:
        """A config that will not take the flag, however it is asked."""

        disable_torque_on_disconnect = True

        def __setattr__(self, name: str, value: object) -> None:
            raise AttributeError(name)

    arm = FakeArm(step=40.0, stuck=("shoulder_lift",))
    arm.config = NoWrites()  # type: ignore[assignment]
    transport = LeRobotReal("COM5", robot=arm, rest_pose=dict(FOLDED))
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    await adapter.go_to_rest()
    await adapter.close()
    note = adapter.close_note or ""
    assert "could not keep torque on" in note, note
    assert "released where it stood" in note, note
    assert "it will not fall" not in note, "the note promised the arm was being held"


async def test_an_arm_with_no_rest_pose_recorded_moves_nothing_and_goes_limp() -> None:
    """The whole thing is opt-in. Without a recorded pose there is nothing to check the
    joints against, so the rest move is a no-op and LeRobot's own default stands."""
    arm = FakeArm()
    transport = LeRobotReal("COM5", robot=arm)
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    assert adapter.rest_pose is None
    result = await adapter.go_to_rest()
    assert result.how == "none" and not result.recorded and not result.reached
    assert arm.actions == [], "an arm with no rest pose was driven somewhere"
    await adapter.close()
    assert arm.config.disable_torque_on_disconnect is True
    assert arm.torque_disabled == 1 and arm.torque is False
    assert adapter.close_note is None


async def test_an_arm_already_at_its_rest_pose_sends_nothing() -> None:
    arm = FakeArm()
    where_it_sits = {j: arm.positions[j] for j in JOINTS if j != "gripper"}
    transport = LeRobotReal("COM5", robot=arm, rest_pose=where_it_sits)
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    result = await adapter.go_to_rest()
    assert result.how == "already" and result.reached and result.recorded
    assert arm.actions == [], "an arm that was already there was driven anyway"
    await adapter.close()
    assert arm.torque_disabled == 1 and arm.torque is False and adapter.close_note is None


async def test_close_reads_the_joints_itself_even_when_no_rest_move_ran() -> None:
    """close() is the last thing to touch the arm and the only one that knows whether
    letting go would drop it, so it reads the pose rather than trusting that somebody called
    the rest move first. A run that died between the two is exactly that case, and it is the
    case where an unchecked disconnect costs you the arm."""
    away = FakeArm()
    strayed = LeRobotAdapter(LeRobotReal("COM5", robot=away, rest_pose={"shoulder_lift": -90.0}))
    await strayed.connect()
    await strayed.close()
    assert away.torque_disabled == 0 and away.torque is True
    note = strayed.close_note
    assert note is not None and "nothing moved it there" in note, note
    assert "shoulder_lift is at 0 with a goal of -90" in note

    parked = FakeArm()
    rested = LeRobotAdapter(LeRobotReal("COM5", robot=parked, rest_pose={"shoulder_lift": 0.0}))
    await rested.connect()
    await rested.close()
    assert parked.torque_disabled == 1 and parked.torque is False
    assert rested.close_note is None


async def test_the_rest_move_on_a_wedged_bus_is_an_answer_and_not_an_exception() -> None:
    """Every caller of the rest move is a teardown or the first moment of a run, and a
    teardown that raised would cost the arm the disconnect it was in the middle of. A wedged
    bus is the one state where nothing at all can be asked of the arm."""
    import threading

    release = threading.Event()
    arm = FakeArm()
    transport = LeRobotReal("COM5", robot=arm, timeout_s=0.2, rest_pose=dict(FOLDED))
    adapter = LeRobotAdapter(transport)
    await adapter.connect()

    def block() -> None:
        release.wait(5.0)

    try:
        with pytest.raises(TimeoutError):
            await transport._call(block, deadline_s=0.2)
        result = await adapter.go_to_rest()
        assert result.how == "refused" and not result.reached
        assert "one owner" in result.reason, result.reason
        assert arm.actions == [], "a goal was written onto a bus with a thread still on it"
    finally:
        release.set()


async def test_a_rest_goal_that_never_reaches_the_arm_leaves_it_holding() -> None:
    """The arm answers every question and obeys none: the reads that decide whether it is
    resting still work, so close() can tell that it is not, and keeps the torque."""
    arm = FakeArm()
    transport = LeRobotReal("COM5", robot=arm, rest_pose=dict(FOLDED))
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    arm.send_fails = True
    result = await adapter.go_to_rest()
    assert result.how == "refused" and not result.reached
    assert "Goal_Position" in result.reason, result.reason
    await adapter.close()
    assert arm.torque_disabled == 0 and arm.torque is True
    note = adapter.close_note
    assert note is not None and "torque was left on" in note
    assert "Goal_Position" in note, "the note does not say why the arm is not where it should be"


def test_the_rest_budget_is_the_travel_at_the_step_cap_plus_slack_and_is_bounded() -> None:
    """One send_action moves a joint at most the step cap and they go out every tick, so the
    fastest the arm can cross a gap is that distance divided by that rate. The upper bound is
    there because the cap can be lowered by the environment until a long move would take
    minutes, and an arm nobody is watching must not hold a run open that long."""
    assert rest_budget_s(0.0, 5.0) == REST_MIN_S
    assert rest_budget_s(90.0, 5.0) == pytest.approx(3.8)  # 90 degrees at 50 a second, plus 2
    assert rest_budget_s(90.0, 1.0) == pytest.approx(11.0)  # a fifth of the step, far longer
    assert rest_budget_s(180.0, 10.0) == pytest.approx(3.8)  # twice as far at twice the step
    assert rest_budget_s(10_000.0, 5.0) == REST_MAX_S
    assert rest_budget_s(1.0, 0.0) == pytest.approx(12.0)  # a zero step is not a division


async def test_the_mock_arm_goes_to_its_rest_pose_and_records_the_order_it_happened_in() -> None:
    """A teardown is an order as much as a set: the rest move has to land before the close,
    because after the close there is no arm to move."""
    mock = LeRobotMock(rest_pose=dict(REST))
    adapter = LeRobotAdapter(mock)
    await adapter.connect()
    assert (await adapter.send_intent(Intent.joint({"shoulder_pan": 30.0}, 1.0))).accepted
    assert mock.sequence == []
    result = await adapter.go_to_rest()
    assert result.how == "arrived" and result.reached
    assert mock.actions[-1] == rest_goal(REST), mock.actions
    assert "gripper" not in mock.actions[-1]
    assert mock.joints["shoulder_pan"] == 0.0
    await adapter.close()
    assert mock.sequence == ["rest", "close"]
    assert mock.torque is False and mock.close_note is None


async def test_a_mock_arm_told_to_fail_its_rest_move_stalls_without_moving() -> None:
    """Offline, goals land the instant they are sent, so the one thing a mock cannot do to
    itself is fail to arrive. Every caller of the rest move has to handle that, so the mock
    can be told to."""
    mock = LeRobotMock(rest_pose=dict(REST), rest_fails="elbow_flex is stuck against the desk")
    adapter = LeRobotAdapter(mock)
    await adapter.connect()
    assert (await adapter.send_intent(Intent.joint({"shoulder_pan": 30.0}, 1.0))).accepted
    sent = len(mock.actions)
    result = await adapter.go_to_rest()
    assert result.how == "stalled" and not result.reached and result.recorded
    assert result.reason == "elbow_flex is stuck against the desk"
    assert len(mock.actions) == sent, "a rest move that was told to fail moved the arm anyway"
    await adapter.close()
    assert mock.sequence == ["rest", "close"]
    assert mock.torque is True, "the mock let go of an arm that is not at its rest pose"
    note = adapter.close_note
    assert note is not None and "torque was left on" in note


def _mock_past(joint: str, by: float) -> float:
    """`_past` for the mock, whose travel is `MOCK_RANGES` rather than a calibration."""
    lo, hi = MOCK_RANGES[joint]
    return lo + by if by < 0 else hi + by


async def test_the_mock_parks_a_pose_past_its_travel_and_lets_go_of_it_the_same_way() -> None:
    """Offline is where a person rehearses a run, and the mock's `_goto` already clamps like
    the servo, so before this a pose past its travel stalled offline the way it did on the
    bench, and went unnoticed because the mock called any move `arrived` without looking.
    Now it clips, judges, names and releases exactly as the arm does, in the same words, and
    carries the same manifest key."""
    pose = dict(REST) | {
        "shoulder_lift": _mock_past("shoulder_lift", -14.0),
        "elbow_flex": _mock_past("elbow_flex", 9.0),
    }
    mock = LeRobotMock(rest_pose=pose)
    adapter = LeRobotAdapter(mock)
    manifest = await adapter.connect()
    assert set(manifest.extras["rest_pose_clipped"]) == {"shoulder_lift", "elbow_flex"}
    result = await adapter.go_to_rest()
    assert result.how == "arrived", result.reason
    floor, ceiling = MOCK_RANGES["shoulder_lift"][0], MOCK_RANGES["elbow_flex"][1]
    assert mock.joints["shoulder_lift"] == floor and mock.joints["elbow_flex"] == ceiling
    assert [j for j, _, _ in result.clipped] == ["shoulder_lift", "elbow_flex"]
    assert result.note is not None and "lerobot-calibrate" in result.note
    await adapter.close()
    assert mock.torque is False and mock.close_note is None

    # folded past both limits, as the pose was recorded: already at rest, nothing sent, and
    # neither the stop nor the hand-off drags it anywhere. The hand-off releases it at the
    # fold and will not take hold again while the folds read past their travel, as on the arm
    folded = LeRobotMock(rest_pose=pose)
    folded.joints.update(pose)
    held = LeRobotAdapter(folded)
    await held.connect()
    assert (await held.go_to_rest()).how == "already"
    await folded.stop()
    released = await held.let_go()
    assert released.how == "released", released.reason
    written = list(folded.actions)
    refused = await held.take_hold()
    assert refused.how == "refused", refused.reason
    assert "shoulder_lift" in refused.reason and "elbow_flex" in refused.reason, refused.reason
    assert refused.energised is False, "refused before its torque write, as on the arm"
    assert folded.actions == written and folded.torque is False, folded.actions
    assert {j: folded.joints[j] for j in ("shoulder_lift", "elbow_flex")} == {
        j: pose[j] for j in ("shoulder_lift", "elbow_flex")
    }
    await held.close()
    assert folded.torque is False
    assert folded.close_note == LIMP_AT_REST, "a fold nobody lifted was called in a hand"


async def test_the_mock_judges_its_own_rest_move_rather_than_assuming_it_arrived() -> None:
    """A mock whose goals do not land is not a mock that arrived. Its `rest_fails` hook is how
    a test asks for a stall on purpose; this is the other way to get one, a goal that went out
    and did nothing, and it has to be read back like the arm's."""

    class Seized(LeRobotMock):
        def _goto(self, goals: dict[str, float]) -> None:
            self.actions.append(dict(goals))

    mock = Seized(rest_pose=dict(REST) | {"elbow_flex": 30.0})
    adapter = LeRobotAdapter(mock)
    await adapter.connect()
    result = await adapter.go_to_rest()
    assert result.how == "stalled" and not result.reached, result.reason
    assert "elbow_flex is at 90 with a goal of 30" in result.reason, result.reason


def test_the_lerobot_factory_hands_the_rest_pose_to_both_backends() -> None:
    pose = dict(FOLDED)
    mock = make("mock", rest_pose=pose)
    assert mock.supports_rest_pose and mock.rest_pose == pose
    real = make("real", address="COM5", rest_pose=pose)
    assert real.rest_pose == pose
    assert make("mock").rest_pose is None
    # and through the general factory, which is what the CLI and the registry call
    assert getattr(make_adapter("lerobot:mock", rest_pose=pose), "rest_pose", None) == pose


# ── several cameras: the same arm with two views of the table ───────────────────────────


def test_several_camera_urls_each_name_their_own_camera_and_no_name_repeats() -> None:
    """With one camera the name is quackd's own default and nothing depends on it. With
    several it is the only thing telling two views apart, in the model's prompt, in a
    policy's observation dict and in frames/NNNN-<name>.png. Two handles on one webcam is
    not two views either: it is a camera that will not open twice."""
    specs = parse_camera_urls(("opencv://1?name=top", "opencv://2?name=side"))
    assert [s.name for s in specs] == ["top", "side"] and all(s.name_given for s in specs)
    assert parse_camera_urls(("opencv://0",))[0].name == "front"  # one camera needs no name
    assert parse_camera_urls(()) == ()
    with pytest.raises(AdapterError, match=r"has no \?name= and 2 cameras were given"):
        parse_camera_urls(("opencv://1?name=top", "opencv://2"))
    with pytest.raises(AdapterError, match="already the name of"):
        parse_camera_urls(("opencv://1?name=top", "opencv://2?name=top"))
    with pytest.raises(AdapterError, match="One url per camera"):
        parse_camera_urls(("opencv://1?name=top", "opencv://1?name=side"))


async def test_two_cameras_open_in_order_and_every_frame_carries_its_own_name() -> None:
    arm = FakeArm(camera=False)
    top, side = FakeCamera(), FakeCamera()
    transport = LeRobotReal(
        "COM5",
        robot=arm,
        cameras=parse_camera_urls(("opencv://1?name=top", "opencv://2?name=side")),
        camera_objects={"top": top, "side": side},
    )
    adapter = LeRobotAdapter(transport)
    manifest = await adapter.connect()
    assert transport.camera_keys == ("top", "side")
    assert top.calls == ["connect"] and side.calls == ["connect"]
    frames = await adapter.get_frames()
    assert [f.name for f in frames] == ["top", "side"]
    assert all(f.image.size == (64, 48) for f in frames)
    primary = await adapter.get_frame()  # the first url's camera, and only it
    assert primary is not None and primary.size == frames[0].image.size
    assert manifest.extras["cameras"] == ["top", "side"]
    assert manifest.extras["camera"] == "opencv://1?name=top"
    await adapter.close()
    assert top.calls == ["connect", "disconnect"] and side.calls == ["connect", "disconnect"]


async def test_a_stalled_second_camera_costs_its_picture_and_nothing_else() -> None:
    """A camera that stopped delivering is absent from the frames rather than an empty slot:
    the model is shown the views that exist, and the health is where the missing one says
    what happened to it."""
    arm = FakeArm(camera=False)
    top, side = FakeCamera(), FakeCamera()
    transport = LeRobotReal(
        "COM5",
        robot=arm,
        cameras=parse_camera_urls(("opencv://1?name=top", "opencv://2?name=side")),
        camera_objects={"top": top, "side": side},
    )
    adapter = LeRobotAdapter(transport)
    manifest = await adapter.connect()
    ex = _executor(adapter, manifest)
    side.stalled = True
    frames = await adapter.get_frames()
    assert [f.name for f in frames] == ["top"], "a dead camera took the live one with it"
    health = transport.camera_health()
    assert health["ok"], "the one-camera keys still describe the primary"
    rows = health["cameras"]
    assert [row["name"] for row in rows] == ["top", "side"]
    assert rows[0]["ok"] and rows[0]["error"] is None and rows[0]["size"] == "64x48"
    assert not rows[1]["ok"] and "too old" in rows[1]["error"]
    moved = await ex.run_verb(
        "move_joints", {"positions": {"shoulder_pan": 10}, "duration_s": MOVE_MIN_S}
    )
    assert moved.ok, moved.summary


async def test_a_second_camera_that_will_not_open_refuses_before_the_arm_is_energised() -> None:
    """Half a set of eyes nobody asked for is worse than the refusal, because the frames
    would still arrive: everything reading them would believe that was all there was to see.
    So the one that opened is let go of and the arm is never touched."""
    arm = FakeArm()
    top = FakeCamera()
    adapter = LeRobotAdapter(
        LeRobotReal(
            "COM5",
            robot=arm,
            cameras=parse_camera_urls(("opencv://1?name=top", "opencv://7?name=side")),
            camera_objects={"top": top, "side": FakeCamera(fail_open=True)},
        )
    )
    with pytest.raises(TransportError, match="opencv://7"):
        await adapter.connect()
    assert arm.calls == [], "the arm was energised before the cameras were known good"
    assert top.calls == ["connect", "disconnect"], "the camera that opened was not let go of"


async def test_the_policy_is_handed_every_camera_under_its_own_name() -> None:
    seen: list[list[str]] = []

    class Peeking:
        def act(self, observation: dict[str, Any], *, task: str) -> dict[str, float] | None:
            seen.append([k for k in observation if not k.endswith(".pos")])
            return None

    adapter = LeRobotAdapter(
        LeRobotReal(
            "COM5",
            robot=FakeArm(camera=False),
            policy=Peeking(),
            cameras=parse_camera_urls(("opencv://1?name=top", "opencv://2?name=side")),
            camera_objects={"top": FakeCamera(), "side": FakeCamera()},
        )
    )
    manifest = await adapter.connect()
    ex = Executor(registry_from_manifest(manifest, adapter), adapter, confirm=allow_all)
    await ex.run_verb("pick", {"target": "cup", "max_s": 3})
    assert seen and seen[0] == ["top", "side"], seen
    await adapter.close()


async def test_one_camera_keeps_its_default_name_and_the_health_shape_it_always_had() -> None:
    """The compatibility guarantee. Everything written when an arm had at most one camera
    reads the same dict and the same manifest: `doctor` gates its verdict on these keys, and
    a pilot told its only camera is called `front` would start naming it in sentences that
    nobody needs."""
    arm = FakeArm(camera=False)
    camera = FakeCamera()
    transport = LeRobotReal(
        "COM5", robot=arm, camera=parse_camera_url("opencv://0"), camera_object=camera
    )
    adapter = LeRobotAdapter(transport)
    manifest = await adapter.connect()
    assert transport.camera_keys == ("front",)
    assert "cameras" not in manifest.extras, "a single camera named itself to the pilot"
    assert manifest.extras["camera"] == "opencv://0"
    frames = await adapter.get_frames()
    assert [f.name for f in frames] == ["front"]
    health = transport.camera_health()
    assert set(health) == {"configured", "url", "ok", "age_s", "size", "error"}
    assert health["configured"] and health["ok"] and health["size"] == "64x48"


# ── handing the arm to a person: let go, placed by hand, taken hold of again ────────────

PLACED_PAST_BY = 9.0
"""How far past a joint's travel a test places it by hand. Any amount would do; what matters is
that it is past the travel, not how far, and that its tenth is not the edge."""

HAND_PLACED = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -DEFAULT_TRAVEL_DEG / 2 + PLACED_PAST_BY,
    "elbow_flex": 40.0,
    "wrist_flex": 15.0,
    "wrist_roll": 0.0,
    "gripper": 35.0,
}
"""Where a person left the arm, with the gripper closed on something, and every body joint
inside the travel of a plain `FakeArm`, the shoulder close to its floor.

That is the pose `take_hold` holds, written where it stands, unclipped. A pose with a joint
placed past its travel is the other case, refused before torque comes on, and the tests of it
build their poses from the calibration (`_past`)."""

BY_HAND_POSE = {"shoulder_lift": -20.0, "elbow_flex": 40.0, "wrist_flex": 15.0, "gripper": 35.0}
"""The pose the operator set in the captured `--by-hand` run, for the mock's narrower ranges."""


def _handover_arm() -> tuple[FakeArm, LeRobotReal]:
    """A fake arm sitting in the pose it was recorded resting in.

    That is the only place `let_go` releases it, and the condition is the same one `close()`
    uses: a pose the arm demonstrably holds with no torque on it. Anywhere else, dropping
    torque drops the arm, and the person who asked for this has their hands nowhere near it."""
    arm = FakeArm()
    where_it_sits = {joint: arm.positions[joint] for joint in JOINTS if joint != "gripper"}
    return arm, LeRobotReal("COM5", robot=arm, rest_pose=where_it_sits)


async def test_let_go_without_a_rest_pose_recorded_says_where_to_record_one() -> None:
    """Without a recorded pose there is nowhere this arm is known to be safe to release, so
    the refusal is the whole of the answer and it names the command that fixes it. The run
    refuses the flag long before this, and this is the backstop under that refusal."""
    arm = FakeArm()
    transport = LeRobotReal("COM5", robot=arm)
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    refused = await adapter.let_go()
    assert refused.how == "refused" and not refused.ok
    assert refused.joints == {}, "a refusal that read nothing should report nothing"
    assert "no rest pose is recorded for this arm" in refused.reason
    assert "quackd robot rest-pose NAME" in refused.reason, refused.reason
    assert arm.timeline == [], "torque was touched on an arm with nowhere safe to put it down"
    assert arm.torque is True and transport._in_hand is False


async def test_let_go_away_from_the_rest_pose_names_the_shortfall_and_keeps_torque() -> None:
    """An arm standing up is held up by torque and by nothing else, so releasing it there is
    the fall the rest pose exists to prevent. The refusal names the joint furthest from where
    it should be, because "not at its rest pose" on its own does not tell anybody what to do."""
    arm = FakeArm(step=40.0)
    transport = LeRobotReal("COM5", robot=arm, rest_pose=dict(FOLDED))
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    refused = await adapter.let_go()
    assert refused.how == "refused" and not refused.ok
    assert "the arm is not at its rest pose" in refused.reason
    assert "shoulder_lift is at 0 with a goal of -90" in refused.reason, refused.reason
    assert "falls when torque goes" in refused.reason
    assert arm.timeline == [] and arm.torque is True
    assert transport._in_hand is False


async def test_let_go_at_the_rest_pose_releases_the_arm_into_somebodys_hands() -> None:
    arm, transport = _handover_arm()
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    released = await adapter.let_go()
    assert released.how == "released" and released.ok
    assert released.reason == "torque is off at the rest pose"
    assert arm.timeline == ["disable_torque"], arm.timeline
    assert arm.torque is False
    assert arm.torque_disabled == 0, "the arm was disconnected rather than released"
    assert arm.actions == [], "a goal was written to an arm on its way to being let go of"
    assert released.joints["shoulder_pan"] == 0.0 and released.joints["gripper"] == 100.0
    assert transport._in_hand is True, "the arm is limp, so it is in somebody's hands"


async def test_a_register_read_that_failed_after_the_release_is_a_release_that_took() -> None:
    """Deliberate, and the reason is which of the two wrong answers gets an arm dropped.

    Torque was taken off and the read that would confirm it came back corrupt, so the last
    reading stands and it still says torque is on. Believing it ends with `close()` telling
    the person holding a limp arm that it is holding itself up, and they let go of it. The
    close says it is in their hands, and, since nothing read the release back, not that
    nothing holds it up either (`UNREAD_IN_HAND`)."""
    arm, transport = _handover_arm()
    arm.bus_error_after_release = True
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    released = await adapter.let_go()
    assert released.how == "released", released.reason
    assert transport._register_error is not None, "the read after the release was meant to fail"
    assert transport._torque is True, "the stale register still claims torque, which is the case"
    assert arm.torque is False and transport._in_hand is True
    await adapter.close()
    note = adapter.close_note or ""
    assert note.startswith("the arm is in your hands"), note
    assert "hold it as though nothing holds it" in note and "holding itself up" not in note, note


async def test_take_hold_writes_the_pose_before_torque_comes_on_and_again_after() -> None:
    """The order is the whole method. Nothing upstream documents what a servo does with the
    goal it was last told when it is re-energised (`up.TORQUE_ENABLE_HOLDS_PRESENT`), and the
    goal this one was last told is a rest pose it has since been lifted out of by hand:
    enabling torque first could snap the arm back to the fold with a hand still in it.

    It goes out again afterwards and is read back, so the answer says whether the arm stayed
    where it was put rather than assuming it did."""
    arm, transport = _handover_arm()
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    assert (await adapter.let_go()).how == "released"
    arm.positions.update(HAND_PLACED)  # a person lifts it and puts it where the run starts

    assert all(
        arm.travel(joint)[0] <= value <= arm.travel(joint)[1]
        for joint, value in HAND_PLACED.items()
        if joint != "gripper"
    ), "every body joint is placed inside its travel, or this is the refusal's test"

    held = await adapter.take_hold()
    assert held.how == "held" and held.ok and held.energised is True
    assert held.reason == "holding the pose you set"
    assert arm.timeline == ["disable_torque", "send", "enable_torque", "send"], arm.timeline
    # every joint where it was placed, the gripper included, before torque and after
    goal = {f"{joint}.pos": value for joint, value in HAND_PLACED.items()}
    assert arm.actions == [goal, goal], arm.actions
    assert transport._range_clips == 0, "a hand-placed pose was walked back inside the travel"
    assert held.joints == HAND_PLACED and arm.positions == HAND_PLACED
    assert arm.torque is True and transport._in_hand is False


async def test_take_hold_refuses_when_torque_never_comes_on_and_the_arm_is_still_in_hand() -> None:
    """The arm took the call and stayed limp, so nothing is holding it and a hand still is.
    This is the one refusal that leaves the arm where `let_go` left it, and `close()` has to
    go on saying so."""
    arm, transport = _handover_arm()
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    assert (await adapter.let_go()).how == "released"
    arm.positions.update(HAND_PLACED)
    arm.torque_refuses = True

    refused = await adapter.take_hold()
    assert refused.how == "refused" and not refused.ok
    assert refused.reason == "the arm still reports torque off, so nothing holds it"
    assert refused.energised is False, "every motor read off after the torque write"
    assert arm.torque is False
    assert transport._in_hand is True, "nothing holds the arm up, so somebody's hand does"
    await adapter.close()
    note = adapter.close_note or ""
    assert "the arm is limp and in your hands" in note and "nothing is holding it up" in note
    assert "torque was left on" not in note, note


async def test_a_joint_that_slipped_as_torque_arrived_is_refused_but_not_called_limp() -> None:
    """The bug this found, and the distinction the whole refusal turns on.

    Torque coming on is what takes the arm out of somebody's hands; the pose it settled in is
    a separate question. An arm reported limp in a hand while it is energised sends the person
    to cut the power on a robot that is holding perfectly well, so `_in_hand` is cleared the
    moment torque is confirmed and the refusal below is about the pose alone."""
    arm, transport = _handover_arm()
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    assert (await adapter.let_go()).how == "released"
    arm.positions.update(HAND_PLACED)
    arm.slips = {"elbow_flex": -12.0}  # it sags as the servos take the weight back

    refused = await adapter.take_hold()
    assert refused.how == "refused" and not refused.ok
    assert "the arm moved as torque came on (elbow_flex by 12 degrees)" in refused.reason
    assert "it is holding where it is now" in refused.reason, refused.reason
    assert refused.joints["elbow_flex"] == 28.0, "the refusal says where the arm ended up"
    assert refused.energised is True
    assert arm.torque is True
    assert transport._in_hand is False, "torque is on, so the arm is not hanging off a hand"
    await adapter.close()
    note = adapter.close_note or ""
    assert "torque was left on" in note and "it will not fall" in note
    assert "limp and in your hands" not in note, note


def _handed_over_far_from(arm: FakeArm, joint: str, by: float) -> LeRobotReal:
    """The real backend over `arm`, resting at a pose whose `joint` lies near the end of its
    travel *opposite* the one it is about to be placed past (`by`'s sign says which), which is
    where a hand-off leaves its servo's last goal: the rest move's, written before the arm was
    lifted. Every other joint rests in the middle of its travel."""
    rest = {j: 0.0 for j in SPANS} | {joint: _inside(arm, joint, -0.8 if by > 0 else 0.8)}
    arm.positions.update(rest)
    return LeRobotReal("COM5", robot=arm, rest_pose=rest)


@pytest.mark.parametrize("by", [-PLACED_PAST_BY, PLACED_PAST_BY], ids=["floor", "ceiling"])
async def test_take_hold_leaves_torque_off_under_a_joint_placed_past_its_travel(by: float) -> None:
    """What went wrong, twice. `take_hold` first wrote a hand-placed joint's present position
    as its goal before torque came on, and past the travel the servo clamps that goal to the
    limit, so torque hauled the joint to the near end with a hand on it. The fix skipped the
    joint instead, and that was worse: the servo then keeps the last goal it was written, which
    after a hand-off is the rest move's, written before the arm was lifted and here near the far
    end of the travel, so a servo that drives to its stored goal when torque comes on
    (`up.TORQUE_ENABLE_HOLDS_PRESENT`, unverified) swings the joint across the whole travel.

    Neither keeps the joint where the person put it, so `take_hold` refuses before it writes
    anything or switches torque on, the arm stays limp in their hands, and they are told which
    joint, where it reads and where its travel is, in this arm's numbers, and to move it
    inside. The `FakeArm` does not model a servo driving to its stored goal, so what this reads
    is that no goal went out and `enable_torque` was never called. Past either end."""
    arm = _spanned()
    joint = "wrist_flex"
    transport = _handed_over_far_from(arm, joint, by)
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    assert (await adapter.let_go()).how == "released"
    placed = _past(arm, joint, by)
    arm.positions[joint] = placed

    refused = await adapter.take_hold()
    assert refused.how == "refused" and not refused.ok
    assert refused.energised is False, "refused before its torque write, so nothing came on"
    assert arm.timeline == ["disable_torque"], "a goal or a torque write reached a placed joint"
    assert arm.torque is False and transport.in_hand is True, "the arm is still in the hand"
    assert arm.positions[joint] == placed and refused.joints[joint] == placed
    lo, hi = published_travel(*transport.joint_range_deg[joint])
    assert refused.reason.startswith(
        f"{joint} reads {placed:.1f}, outside its calibrated travel of {lo:.1f}..{hi:.1f}, so "
        "quackd left torque off"
    ), refused.reason
    assert refused.reason.endswith(f"takes hold of the arm only with {joint} inside its travel")

    # why the arm is limp, said the way the mock says it: it was let go of for the person to
    # place and never taken hold of again, and not the distance from its rest pose
    await adapter.close()
    assert adapter.close_note == LIMP_IN_HAND.format(why=LET_GO_TO_PLACE), adapter.close_note


async def test_take_hold_names_every_joint_placed_past_its_travel_at_either_end() -> None:
    """Two joints placed past opposite ends of their travel in one hand-off: both are named, in
    the order the arm reports them, each with its own reading and its own travel, and the
    sentence asks for both to be moved. A refusal naming only the first would send the person
    back to the arm to fix one joint and be refused again over the other."""
    arm = _spanned()
    transport = _handed_over_far_from(arm, "elbow_flex", PLACED_PAST_BY)
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    assert (await adapter.let_go()).how == "released"
    low = _past(arm, "shoulder_lift", -PLACED_PAST_BY)
    high = _past(arm, "elbow_flex", PLACED_PAST_BY)
    arm.positions |= {"shoulder_lift": low, "elbow_flex": high}

    refused = await adapter.take_hold()
    assert refused.how == "refused" and arm.timeline == ["disable_torque"], arm.timeline
    shoulder = published_travel(*transport.joint_range_deg["shoulder_lift"])
    elbow = published_travel(*transport.joint_range_deg["elbow_flex"])
    assert refused.reason.startswith(
        f"shoulder_lift reads {low:.1f} and elbow_flex reads {high:.1f}, outside their "
        f"calibrated travel of {shoulder[0]:.1f}..{shoulder[1]:.1f} and "
        f"{elbow[0]:.1f}..{elbow[1]:.1f}, so quackd left torque off"
    ), refused.reason
    assert refused.reason.endswith("only with shoulder_lift and elbow_flex inside their travel")
    assert "goals written where those joints are lie past their travel" in refused.reason


def test_the_refusal_names_the_goal_written_where_the_joint_is_and_no_other() -> None:
    """The refusal said the servo pulls "any goal written for that joint" to the end of its
    travel. The servo clamps a goal past the travel and leaves one inside it alone, and the
    clause after it in the same sentence is about exactly such a goal, the last one the servo
    was given. The goal it means is the one written where the joint reads, which lies past
    the travel, and that is the one it names now. Synthetic travel."""
    lo, hi = -40.0, 40.0
    said = placed_past_travel({"wrist_flex": hi + PLACED_PAST_BY}, {"wrist_flex": (lo, hi)})
    assert "any goal written" not in said, said
    assert (
        "a goal written where that joint is lies past its travel and the servo would pull it to "
        "the end of its travel, and with none written the servo may drive it to the last goal "
        "it was given"
    ) in said, said


def test_a_joint_a_hair_past_its_travel_is_never_named_inside_it() -> None:
    """The reading is printed the way the range refusal prints a goal (`said_past`): to a tenth
    unless a tenth would put it on the travel the same sentence gives, and then as it is. A
    joint a few hundredths past its ceiling read to a tenth is the ceiling itself, and "reads
    65.0, outside its calibrated travel of -65.0..65.0" is a sentence that contradicts itself."""
    hi = 50.0 + RAMP_RESOLUTION / 4  # a travel whose end is no tenth, so it is printed inward
    reading = hi + RAMP_RESOLUTION / 4
    said = placed_past_travel({"wrist_flex": reading}, {"wrist_flex": (-hi, hi)})
    lo_shown, hi_shown = published_travel(-hi, hi)
    printed = float(said.split(" reads ", 1)[1].split(",", 1)[0])
    assert not lo_shown <= printed <= hi_shown, said
    assert f"{lo_shown:.1f}..{hi_shown:.1f}" in said, said


async def test_a_teardown_over_a_joint_placed_past_its_travel_writes_the_arm_nothing() -> None:
    """The teardown a refused take-hold leads into, on the real backend: a stop, a rest move
    and a close, over an arm in somebody's hands. It used to write goals into the limp servos
    all the way through, the stop's hold, six rest ticks and the stall's hold, each one the
    stale goal the refusal was about for the next torque write to drive to, and it narrated a
    fold of an arm that never moved.

    Now the stop takes no second hold and sends nothing, and says why; the rest move reads the
    arm and refuses to move it, sending nothing either; and the close ends on the note for an
    arm in somebody's hands, with why it is limp in the words the mock uses rather than the
    rest move's shortfall said twice."""
    clock = SteppedClock()
    arm = _spanned()
    joint = "elbow_flex"
    transport = _handed_over_far_from(arm, joint, -PLACED_PAST_BY)
    transport.clock = clock
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    assert (await adapter.let_go()).how == "released"
    placed = _past(arm, joint, -PLACED_PAST_BY)
    arm.positions[joint] = placed
    assert (await adapter.take_hold()).how == "refused"  # the take-hold that ends the wait

    await transport.stop()
    assert arm.timeline == ["disable_torque"], "the stop wrote to an arm in a hand"
    assert arm.torque is False and transport.in_hand is True
    assert (transport.stop_error or "").startswith(
        f"the arm is in somebody's hands, so the stop sent it nothing: {joint} reads {placed:.1f}"
    ), transport.stop_error

    rested = await adapter.go_to_rest()
    assert (rested.how, rested.reason) == ("refused", IN_HAND_NOT_MOVED), rested
    assert arm.timeline == ["disable_torque"], "the rest move wrote to an arm in a hand"
    assert arm.positions[joint] == placed, "a limp joint was moved"

    await adapter.close()
    assert adapter.close_note == LIMP_IN_HAND.format(why=LET_GO_TO_PLACE), adapter.close_note


async def test_a_stop_while_the_arm_is_in_a_hand_picks_it_up_before_it_sends_anything() -> None:
    """Every teardown begins with a stop, so this is what a Ctrl-C during the wait reaches. A
    stop is "stay where you are", and a goal written to a limp servo stops nothing: the arm is
    energised where the hand has it, and the rest move that follows can then put it down. Every
    body joint here is inside its travel, which is the only pose `take_hold` energises."""
    arm, transport = _handover_arm()
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    assert (await adapter.let_go()).how == "released"
    arm.positions.update(HAND_PLACED)

    await transport.stop()
    assert arm.timeline == ["disable_torque", "send", "enable_torque", "send", "send"]
    assert arm.torque is True and transport._in_hand is False
    # the hold is the body joints and not the gripper, which would drop what it is holding
    body = {f"{joint}.pos" for joint in JOINTS if joint != "gripper"}
    assert set(arm.actions[-1]) == body, "the hold re-sent the gripper and dropped the object"
    assert arm.actions[-1]["shoulder_lift.pos"] == HAND_PLACED["shoulder_lift"]
    assert arm.positions == HAND_PLACED, "the stop moved the arm out of the pose it was given"
    assert transport.stop_error is None


async def test_a_close_with_the_arm_still_in_a_hand_says_it_is_limp_not_the_opposite() -> None:
    """`--by-hand` takes hold again before the first turn, so the only way to reach a close in
    this state is a run that ended in the gap between: a Ctrl-C during the wait, a heartbeat
    that died, a `take_hold` the arm refused. Whoever is holding the arm is the one reading
    the note, and the torque note would tell them it is holding itself up."""
    arm, transport = _handover_arm()
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    assert (await adapter.let_go()).how == "released"
    arm.positions.update(HAND_PLACED)  # lifted out of the fold, and held there

    await adapter.close()
    note = adapter.close_note
    assert note is not None
    assert "the arm is limp and in your hands" in note
    assert "put it down before you let go of it" in note
    assert "never taken hold of again" in note, note
    assert "torque was left on" not in note and "could not keep torque on" not in note
    assert ("disconnect",) in arm.calls, "the arm was never let go of"


async def test_an_arm_let_go_of_at_its_rest_pose_and_still_there_is_not_said_to_be_in_a_hand() -> (
    None
):
    """The placing release lets go only at the rest pose, and an arm nobody lifted is still
    lying there when the close reads it. The close used to end on "the arm is limp and in your
    hands ... put it down before you let go of it" whatever its own read found, which on the
    bench arm, whose fold lies past its travel, was the ending of every placement wait that
    ended unanswered: telling somebody who may never have touched it to put down an arm that is
    down. Its read finds the arm at its rest pose and every motor off, so it says that."""
    arm, transport = _handover_arm()
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    assert (await adapter.let_go()).how == "released"

    await adapter.close()
    assert adapter.close_note == LIMP_AT_REST, adapter.close_note
    assert arm.torque is False


async def test_the_mock_refuses_the_hand_off_wherever_the_real_backend_refuses_it() -> None:
    """Offline is where a person rehearses this, so the mock has to refuse for the same
    reasons and in the same words: a rehearsal that releases an arm the real one would not is
    a rehearsal that teaches the wrong thing."""
    bare = LeRobotMock()
    nowhere = await bare.let_go()
    assert nowhere.how == "refused" and not nowhere.ok
    assert "no rest pose is recorded for this arm" in nowhere.reason
    assert "quackd robot rest-pose NAME" in nowhere.reason, nowhere.reason
    assert bare.torque is True and bare.in_hand is False
    assert bare.sequence == ["let_go"], "a refusal is still an attempt, and it is recorded"

    mock = LeRobotMock(rest_pose=dict(REST))
    adapter = LeRobotAdapter(mock)
    await adapter.connect()
    assert (await adapter.send_intent(Intent.joint({"shoulder_pan": 30.0}, 1.0))).accepted
    away = await adapter.let_go()
    assert away.how == "refused" and "the arm is not at its rest pose" in away.reason
    assert "shoulder_pan is at 30 with a goal of 0" in away.reason, away.reason
    assert mock.torque is True and mock.in_hand is False


async def test_a_mock_stop_over_an_arm_in_a_hand_takes_hold_before_it_stops() -> None:
    """The real `_hold()`'s order, in the sequence a test can read: a stop that sent a goal to
    a limp servo would stop nothing and the arm would still be limp afterwards."""
    mock = LeRobotMock(rest_pose=dict(REST))
    adapter = LeRobotAdapter(mock)
    await adapter.connect()
    assert (await adapter.let_go()).how == "released"
    assert mock.torque is False and mock.in_hand is True

    await mock.stop()
    assert mock.sequence == ["let_go", "take_hold", "stop"], mock.sequence
    assert mock.torque is True and mock.in_hand is False


async def test_the_mock_leaves_torque_off_under_a_joint_placed_past_its_travel() -> None:
    """The real `take_hold`'s refusal offline, in the same words and with nothing written: the
    mock used to skip such a joint and take hold of the rest, which rehearsed an ending the arm
    no longer has. Its teardown then meets the arm the way the real one does: the stop takes
    no second hold and sends nothing, the rest move refuses to move an arm in a hand instead of
    stalling over it and saying it stopped moving, and the close says the arm is in somebody's
    hands, let go of for them to place."""
    mock = LeRobotMock(rest_pose=dict(REST))
    adapter = LeRobotAdapter(mock)
    await adapter.connect()
    assert (await adapter.let_go()).how == "released"
    placed = MOCK_RANGES["wrist_flex"][1] + PLACED_PAST_BY
    mock.joints["wrist_flex"] = placed
    written = list(mock.actions)

    refused = await adapter.take_hold()
    assert refused.how == "refused" and not refused.ok and refused.energised is False
    assert refused.reason == placed_past_travel({"wrist_flex": placed}, MOCK_RANGES)
    assert mock.actions == written, "a goal was written for an arm that was refused"
    assert mock.torque is False and mock.in_hand is True

    await mock.stop()
    assert mock.torque is False and mock.in_hand is True
    rested = await adapter.go_to_rest()
    assert (rested.how, rested.reason) == ("refused", IN_HAND_NOT_MOVED), rested
    assert mock.joints["wrist_flex"] == placed, "the rest move folded a limp arm"
    assert mock.actions == written, "the teardown wrote a goal to an arm in a hand"
    await adapter.close()
    assert mock.close_note == LIMP_IN_HAND.format(why=LET_GO_TO_PLACE), mock.close_note
    assert mock.sequence == ["let_go", "take_hold", "stop", "rest", "close"], mock.sequence


async def test_a_mock_arm_that_slipped_as_torque_came_on_refuses_and_is_not_in_a_hand() -> None:
    mock = LeRobotMock(rest_pose=dict(REST))
    adapter = LeRobotAdapter(mock)
    await adapter.connect()
    assert (await adapter.let_go()).how == "released"
    mock.joints.update(BY_HAND_POSE)
    mock.hold_slips = {"elbow_flex": -9.0}

    refused = await adapter.take_hold()
    assert refused.how == "refused" and not refused.ok
    assert "the arm moved as torque came on (elbow_flex by 9 degrees)" in refused.reason
    assert mock.joints["elbow_flex"] == 31.0
    assert mock.torque is True and mock.in_hand is False
    await adapter.close()
    note = mock.close_note or ""
    assert "torque was left on" in note and "limp and in your hands" not in note, note


async def test_a_mock_run_handed_over_and_taken_back_closes_with_nothing_to_say() -> None:
    """The whole `--by-hand` teardown offline: released at the rest pose, placed, taken hold
    of, stopped, folded up, closed. Nothing about this run is unusual by the end of it, and a
    close that still had something to say would be saying it about a run that went right."""
    mock = LeRobotMock(rest_pose=dict(REST))
    adapter = LeRobotAdapter(mock)
    await adapter.connect()
    assert (await adapter.let_go()).how == "released"
    mock.joints.update(BY_HAND_POSE)  # the person places it and presses Enter

    held = await adapter.take_hold()
    assert held.how == "held" and held.ok and held.reason == "holding the pose you set"
    assert mock.actions[-1] == held.joints, "the pose it was left in is the goal that pins it"
    assert held.joints["elbow_flex"] == 40.0 and held.joints["gripper"] == 35.0
    await mock.stop()
    assert (await adapter.go_to_rest()).how == "arrived"
    await adapter.close()
    assert mock.sequence == ["let_go", "take_hold", "stop", "rest", "close"], mock.sequence
    assert mock.close_note is None and adapter.close_note is None
    assert mock.torque is False, "an arm back at its rest pose may be let go of"
    assert mock.in_hand is False


async def test_a_take_hold_that_found_some_motors_on_names_them_and_does_not_call_it_limp() -> None:
    """A torque write some motors took and some did not: the read after it found those on and
    the rest off. The refusal said "the arm still reports torque off, so nothing holds it",
    over motors the same read had found holding. It names them now, and says the take-hold may
    have left the arm energised, since in part it did. The motor that holds is synthetic."""
    arm, transport = _handover_arm()
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    holds = sorted(BODY_JOINTS)[0]
    arm.torque_holdouts = {holds}
    assert (await adapter.let_go()).how == "released"
    arm.positions.update(HAND_PLACED)
    arm.torque_refuses = True  # the write goes out and only the holdout reads on after it

    refused = await adapter.take_hold()
    assert refused.how == "refused", refused.reason
    assert refused.reason == (
        f"torque came back on only on {holds}, and the rest of the arm still reads off"
    )
    assert refused.energised is True and transport.in_hand is True
    await adapter.close()
    assert adapter.close_note == still_holding_in_hand((holds,), HOLD_NOT_CONFIRMED)


async def test_a_refused_hold_is_kept_only_until_a_hold_takes_or_the_arm_is_released_again() -> (
    None
):
    """After a refused take-hold the stop leaves an arm in a hand alone. That lasts until a
    take-hold is confirmed or the arm is let go of again, and no longer: after either it is a
    different hand-off, and a stop over an arm released again takes hold of it where the hand
    has it, as after any release (ADR-0039). A refusal that outlived its hand-off would leave
    the next one's arm limp in the teardown for a reason that no longer holds. On the real
    backend and on the mock, with a joint placed past its synthetic travel."""
    arm, transport = _handover_arm()
    await transport.connect()
    joint = "wrist_flex"
    assert (await transport.let_go()).how == "released"
    arm.positions[joint] = _past(arm, joint, PLACED_PAST_BY)
    assert (await transport.take_hold()).how == "refused"
    arm.positions[joint] = HAND_PLACED[joint]
    assert (await transport.take_hold()).how == "held"  # asked for directly, and it took
    assert transport._refused_hold is None, "a hold that took left the refusal standing"

    arm.positions.update({j: 0.0 for j in BODY_JOINTS})  # back at its rest pose
    assert (await transport.let_go()).how == "released"
    arm.positions[joint] = _past(arm, joint, PLACED_PAST_BY)
    assert (await transport.take_hold()).how == "refused"
    arm.positions.update({j: 0.0 for j in BODY_JOINTS})
    assert (await transport.let_go()).how == "released"  # a new hand-off
    arm.positions.update(HAND_PLACED)
    await transport.stop()
    assert arm.torque is True and transport.in_hand is False, "the new hand-off was not held"

    mock = LeRobotMock(rest_pose=dict(REST))
    assert (await mock.let_go()).how == "released"
    mock.joints[joint] = MOCK_RANGES[joint][1] + PLACED_PAST_BY
    assert (await mock.take_hold()).how == "refused" and mock.hold_refusal is not None
    mock.joints.update(REST)
    assert (await mock.let_go()).how == "released"
    await mock.stop()
    assert mock.torque is True and mock.in_hand is False, "the new hand-off was not held"


# ── a take-hold refused, and the teardown after it, on the real backend through a run ───────
#
# The loop's teardown over `LeRobotReal` on a `FakeArm`: the stop, the rest move and the close
# a `--by-hand` run makes, with the person stood in for by `_Hands`. The rule every test here
# holds: while the arm is, or may be, in a person's hands, nothing writes it a goal, moves it
# or switches its torque on again, and nothing tells them torque is off unless nothing was sent
# or a read said so. Every pose is built from the arm's own calibration, and every fault is
# made up for these tests.


class _Hands:
    """Somebody at the arm, as the loop's `HandOff` sees them: `ScriptedPerson` in
    tests/test_loop.py, over a `FakeArm`. The first wait answered yes leaves the arm at
    `places`; `on_wait` runs in every wait and `on_say` on every line said to them."""

    asks_a_person = False

    def __init__(
        self,
        arm: FakeArm,
        places: Mapping[str, float] | None = None,
        *,
        answers: Sequence[bool] = (True,),
        on_wait: Any = None,
        on_say: Any = None,
    ) -> None:
        self.arm = arm
        self.places = dict(places or {})
        self.answers = list(answers)
        self.on_wait = on_wait
        self.on_say = on_say
        self.said: list[str] = []
        self.asked: list[str] = []

    def say(self, text: str) -> None:
        self.said.append(text)
        if self.on_say is not None:
            self.on_say(text)

    async def wait(
        self, text: str, *, timeout_s: float | None = None, until_abort: bool = True
    ) -> bool:
        self.asked.append(text)
        answer = self.answers[min(len(self.asked), len(self.answers)) - 1]
        if self.on_wait is not None:
            self.on_wait(self)
        if answer and len(self.asked) == 1:
            self.arm.positions.update(self.places)
        return answer


def _by_hand_run(transport: LeRobotReal, hands: _Hands, runs: Any) -> Any:
    """A `--by-hand` run on the real backend with one terminal for both questions, as the CLI
    wires it, and a pilot that would declare success if a run ever got as far as asking it."""
    from quackd.agent.loop import RunConfig
    from quackd.agent.providers.base import ToolCall
    from quackd.agent.providers.fake import FakeProvider

    declare = ToolCall(name="declare_success", arguments={"reason": "up"})
    return RunConfig(
        duck=ARM_DUCK,
        provider=FakeProvider(script=[declare]),
        transport=LeRobotAdapter(transport),
        hand_off=hands,
        person=hands,
        runs_dir=runs,
    )


def _notes(run_dir: Any) -> list[str]:
    from quackd.agent.transcript import Transcript

    events = Transcript.read(run_dir / "transcript.jsonl")
    return [str(e["text"]) for e in events if e["kind"] == "note"]


class EnergisesThenRaises(FakeBus):
    """A bus whose torque write reaches every motor and then raises, as a lost status packet
    after the last write does: the arm is energised and the call says it failed."""

    def enable_torque(self, motors: Any = None, num_retry: int = 0) -> None:
        super().enable_torque(motors, num_retry=num_retry)
        raise ConnectionError(f"Failed to write 'Torque_Enable' after {num_retry + 1} tries.")


@pytest.mark.parametrize("fault", ["the torque register goes quiet", "enable_torque raises"])
async def test_a_take_hold_refused_after_its_torque_write_is_never_called_limp_or_folded(
    tmp_path: Any, fault: str
) -> None:
    """Every body joint placed inside its travel, and the take-hold at Enter refused after its
    torque write went out: the torque register stops answering from Enter on, or the torque
    call energises the arm and then raises. Either way the arm may be holding itself up.

    What went wrong: the person was told quackd never took hold and the arm was still in their
    hands, the teardown's stop sent the torque write again, they were told to reach into the
    gripper and keep hold of the arm, and the rest move then folded the energised arm under
    those hands with nothing asked. The close then said the arm was limp and nothing held it,
    from the read that confirmed the release, before the torque write.

    Now they are told quackd cannot confirm whether the arm has torque, to hold it as though it
    may move or drop and cut its power; the one torque write is the take-hold's own, nothing is
    written after it, nothing folds, and the close says what a read found, or that none did."""
    from quackd.agent.loop import AgentLoop, run_duck

    arm = FakeArm()
    quiet = fault == "the torque register goes quiet"
    arm.bus = OneRegisterDown(arm, dead=()) if quiet else EnergisesThenRaises(arm)
    rest = {j: arm.positions[j] for j in JOINTS if j != "gripper"}
    transport = LeRobotReal("COM5", robot=arm, rest_pose=rest, clock=SteppedClock())

    def enter(_hands: _Hands) -> None:
        if isinstance(arm.bus, OneRegisterDown):
            arm.bus.dead = ("Torque_Enable",)  # from the moment Enter is pressed, for good

    hands = _Hands(arm, HAND_PLACED, on_wait=enter)
    result = await run_duck(_by_hand_run(transport, hands, tmp_path))

    assert result.outcome == "aborted", result.reason
    assert result.reason.startswith("quackd could not confirm whether the arm has torque ("), (
        result.reason
    )
    assert hands.said == [result.reason, AgentLoop.STILL_UNCONFIRMED], hands.said
    assert hands.asked == [AgentLoop.PLACE_IT], "a question was put over an arm in a hand"
    taken = ["send", "enable_torque", "send"] if quiet else ["send", "enable_torque"]
    assert arm.timeline == ["disable_torque", *taken], "something was written after the refusal"
    assert arm.positions == HAND_PLACED, "the arm was folded in somebody's hands"
    assert arm.torque is True and arm.config.disable_torque_on_disconnect is False
    expected = (
        UNCONFIRMED_IN_HAND if quiet else still_holding_in_hand(tuple(JOINTS), HOLD_NOT_CONFIRMED)
    )
    assert transport.close_note == expected, transport.close_note
    notes = _notes(result.run_dir)
    assert notes.count("moving to the rest pose") == 1, "a fold was announced over the arm"
    assert AgentLoop.NOT_FOLDED in notes, notes


@pytest.mark.parametrize(
    "when", ["as the refusal is said", "as the hand-back line is said"], ids=["stop", "rest"]
)
async def test_a_joint_moved_back_inside_after_a_refused_hold_gets_no_torque_and_no_goal(
    tmp_path: Any, when: str
) -> None:
    """The refusal names a joint placed past its travel, and the person moves it back inside.
    The teardown then took hold of the arm the moment it could: the stop's own take-hold when
    the joint was already inside by then, or the take-hold the rest move's stall makes when it
    came inside while the rest move was writing goals into the limp servos. Torque came on under
    their hand with nothing said, and the run closed on "torque was left on".

    Nothing takes hold after a refusal now, whenever the joint comes inside, and nothing is
    written: the arm's whole record after the release is empty, and the close says it is limp
    in their hands, let go of for them to place."""
    from quackd.agent.loop import AgentLoop, run_duck

    arm = _spanned()
    joint = "elbow_flex"
    transport = _handed_over_far_from(arm, joint, PLACED_PAST_BY)
    transport.clock = SteppedClock()
    line = AgentLoop.NOT_TAKEN_HOLD if when == "as the refusal is said" else "quackd never took"

    def moves_it_inside(said: str) -> None:
        if said.startswith(line):
            arm.positions[joint] = _inside(arm, joint, 0.5)

    placed = {joint: _past(arm, joint, PLACED_PAST_BY), "gripper": HAND_PLACED["gripper"]}
    hands = _Hands(arm, placed, on_say=moves_it_inside)
    result = await run_duck(_by_hand_run(transport, hands, tmp_path))

    assert result.reason.startswith(f"{AgentLoop.NOT_TAKEN_HOLD}: {joint} reads"), result.reason
    assert hands.said == [result.reason, AgentLoop.STILL_IN_YOUR_HANDS], hands.said
    assert arm.positions[joint] == _inside(arm, joint, 0.5), "the joint the person moved moved"
    assert arm.timeline == ["disable_torque"], "a goal or a torque write reached the arm"
    assert arm.torque is False and transport.in_hand is True
    assert transport.close_note == LIMP_IN_HAND.format(why=LET_GO_TO_PLACE)
    assert "torque was left on" not in (transport.close_note or "")


async def test_a_ctrl_c_in_the_placement_wait_on_the_arm_still_takes_hold_and_folds_it(
    tmp_path: Any,
) -> None:
    """ADR-0039's ending, left as it was: a Ctrl-C while the person holds the arm up, with no
    take-hold refused yet, reaches the teardown's stop, which is the run's first take-hold. It
    energises the arm where their hand has it, the rest move folds it, narrated as a fold, and
    the close lets go at the fold with nothing to say."""
    from quackd.agent.loop import AgentLoop

    arm, transport = _handover_arm()
    transport.clock = SteppedClock()
    hands = _Hands(arm, answers=[False])
    loop = AgentLoop(_by_hand_run(transport, hands, tmp_path))

    def lifts_it_and_presses_ctrl_c(_hands: _Hands) -> None:
        arm.positions.update(HAND_PLACED)
        loop.executor.abort.set()

    hands.on_wait = lifts_it_and_presses_ctrl_c
    result = await loop.run()

    assert result.reason == "kill switch"
    assert arm.timeline.count("enable_torque") == 1, arm.timeline
    assert arm.timeline[:4] == ["disable_torque", "send", "enable_torque", "send"], arm.timeline
    folded = rest_goal(transport.rest_pose or {})
    assert all(arm.positions[j] == v for j, v in folded.items()), "it was not folded"
    assert transport.close_note is None and arm.torque is False
    notes = _notes(result.run_dir)
    assert notes.count("moving to the rest pose") == 2 and "at the rest pose" in notes, notes


async def test_a_ctrl_c_in_the_placement_wait_over_a_fold_past_the_travel_writes_it_nothing(
    tmp_path: Any,
) -> None:
    """The bench arm, whose fold lies past its travel, released at its fold and never lifted,
    and the placement wait ended on a Ctrl-C. The stop's take-hold is refused over the folded
    joint, so nothing is written; the rest move reads the arm already at rest and sends
    nothing; and the close reads it lying in its fold with every motor off. It used to end the
    run telling somebody who never touched the arm that it was limp in their hands, and to put
    it down."""
    from quackd.agent.loop import AgentLoop

    arm = _spanned()
    rest = {j: 0.0 for j in SPANS} | {"shoulder_lift": _past(arm, "shoulder_lift", -TOL_DEG * 2)}
    arm.positions.update(rest)
    transport = LeRobotReal("COM5", robot=arm, rest_pose=rest, clock=SteppedClock())
    hands = _Hands(arm, answers=[False])
    loop = AgentLoop(_by_hand_run(transport, hands, tmp_path))
    hands.on_wait = lambda _hands: loop.executor.abort.set()
    result = await loop.run()

    assert result.reason == "kill switch"
    assert arm.timeline == ["disable_torque"], "a goal or a torque write reached the fold"
    assert all(arm.positions[j] == v for j, v in rest.items()), "the fold was moved"
    assert transport.close_note == LIMP_AT_REST, transport.close_note
    notes = _notes(result.run_dir)
    assert notes[-1] == LIMP_AT_REST and AgentLoop.NOT_FOLDED not in notes, notes


# ── a person at the arm asks for torque off, wherever it stands ─────────────────────────
#
# `let_go(anywhere=True)` is the second door, and only a person at a terminal opens it: `quackd
# robot release` and the offer at the end of a run whose rest move missed. Both tell the person
# to hold the arm first, so the two refusals about the pose are skipped and nothing else is:
# the joints are read before, the release is read back after, and the arm is in somebody's
# hands from then on. On 2026-09-23 the power switch was the only way to take torque off an
# arm a run had left holding itself up, and every run that got to its end finished there.
#
# Every pose below is built from the synthetic `SPANS` calibration, on either side of a
# joint's travel, and none is any real arm's.


def _stopped_short(arm: FakeArm) -> tuple[dict[str, float], dict[str, float]]:
    """Recorded low in every joint's travel, standing high in it: a rest move that stalled
    against something inside the travel, which is a genuine miss."""
    return {j: _inside(arm, j, -0.6) for j in SPANS}, {j: _inside(arm, j, 0.3) for j in SPANS}


def _held_out_of_a_fold(arm: FakeArm) -> tuple[dict[str, float], dict[str, float]]:
    """`shoulder_lift` recorded folded past its floor and standing well up out of it, the
    rest of the arm where it was recorded."""
    recorded = dict.fromkeys(SPANS, 0.0) | {"shoulder_lift": _past(arm, "shoulder_lift", -12.0)}
    return recorded, dict.fromkeys(SPANS, 0.0) | {
        "shoulder_lift": _inside(arm, "shoulder_lift", 0.4)
    }


def _past_the_ceiling(arm: FakeArm) -> tuple[dict[str, float], dict[str, float]]:
    """`wrist_flex` reading past its ceiling and `elbow_flex` low in its travel, against a
    pose recorded at the middle of both."""
    recorded = dict.fromkeys(SPANS, 0.0)
    return recorded, recorded | {
        "wrist_flex": _past(arm, "wrist_flex", 9.0),
        "elbow_flex": _inside(arm, "elbow_flex", -0.7),
    }


@pytest.mark.parametrize(
    "placed",
    [
        pytest.param(_stopped_short, id="stopped short inside the travel"),
        pytest.param(_held_out_of_a_fold, id="held up out of a fold past the floor"),
        pytest.param(_past_the_ceiling, id="one joint past its ceiling and one low"),
    ],
)
async def test_a_person_can_have_torque_taken_off_an_arm_that_is_away_from_its_rest_pose(
    placed: Any,
) -> None:
    """The arm on the bench that could not get back to its fold kept torque on and held
    itself up, which is right for an empty room and a dead end for the person standing next
    to it: `let_go` refused it, and the power switch was the only other way.

    The second door releases it where it stands, with nothing sent first, and the arm is then
    in somebody's hands, so the close says so instead of promising it holds itself up. The
    first door still refuses the same arm, because `--by-hand` must never do this."""
    arm = _spanned()
    recorded, reading = placed(arm)
    arm.positions.update(reading)
    transport = LeRobotReal("COM5", robot=arm, rest_pose=recorded)
    adapter = LeRobotAdapter(transport)
    await adapter.connect()

    refused = await adapter.let_go()
    assert refused.how == "refused" and "the arm is not at its rest pose" in refused.reason
    assert arm.timeline == [] and arm.torque is True, "the default released an arm off its pose"

    released = await adapter.let_go(anywhere=True)
    assert released.how == "released", released.reason
    assert released.reason == "torque is off where the arm stands"
    assert released.torque_on == (), "every motor read off, and the result says so"
    assert arm.timeline == ["disable_torque"] and arm.torque is False
    assert arm.actions == [], "a goal went out to an arm on its way to being let go of"
    assert {j: released.joints[j] for j in reading} == reading, "where the person is holding it"
    assert transport._in_hand is True

    await adapter.close()
    note = adapter.close_note or ""
    assert note == LIMP_IN_HAND.format(why=LET_GO_WHERE_IT_STOOD), note
    assert "torque was left on" not in note, "a limp arm was said to be holding itself up"
    assert "nothing moved it there" not in note, "an arm a person is holding was moved there"
    assert arm.config.disable_torque_on_disconnect is False and arm.torque_disabled == 0


async def test_an_arm_with_no_rest_pose_recorded_is_released_only_through_the_second_door() -> None:
    """No pose recorded is the first door's other refusal, and it is about where the arm may
    be let go of on quackd's own judgement. A person holding it is judging for themselves."""
    arm = _spanned()
    arm.positions.update({j: _inside(arm, j, 0.5) for j in SPANS})
    transport = LeRobotReal("COM5", robot=arm)
    await transport.connect()

    refused = await transport.let_go()
    assert refused.how == "refused" and "no rest pose is recorded" in refused.reason
    assert arm.timeline == []

    released = await transport.let_go(anywhere=True)
    assert released.how == "released" and released.torque_on == (), released.reason
    assert released.reason == "torque is off where the arm stands"
    assert arm.torque is False and transport._in_hand is True
    await transport.close()
    assert (transport.close_note or "").startswith(LIMP_IN_HAND.split("(")[0])


async def test_the_second_door_at_the_rest_pose_says_the_arm_is_at_it() -> None:
    """The words follow the arm and not the door: released at its pose, it says so."""
    _arm, transport = _handover_arm()
    await transport.connect()
    released = await transport.let_go(anywhere=True)
    assert released.how == "released" and released.reason == "torque is off at the rest pose"
    assert released.torque_on == ()


@pytest.mark.parametrize(
    "holdouts", [("elbow_flex",), ("shoulder_pan", "wrist_roll", "gripper")], ids=["one", "three"]
)
async def test_a_motor_that_kept_its_torque_through_the_release_is_named_and_never_called_off(
    holdouts: tuple[str, ...],
) -> None:
    """`_torque` is whether every motor reads on, which is the question a hold asks. A release
    asks the other one, and read through `_torque` a motor that kept its torque was simply part
    of an arm reported released: "torque reads off" over a joint still holding in the hands of
    somebody told it is limp. The result names the motors that read on, in the bus's order, and
    the arm is still in a hand, because the rest of it is limp.

    And the close's last line names them too. It used to end every release on `LIMP_IN_HAND`,
    "nothing is holding it up", which `quackd robot release` printed right after "torque still
    reads on for elbow_flex: cut the power", as the last thing a person holding the arm read,
    over a joint still energised and kept so past the close."""
    arm = _spanned()
    arm.positions.update({j: _inside(arm, j, -0.4) for j in SPANS})
    arm.torque_holdouts = set(holdouts)
    transport = LeRobotReal("COM5", robot=arm, rest_pose=dict.fromkeys(SPANS, 0.0))
    await transport.connect()

    released = await transport.let_go(anywhere=True)
    in_order = tuple(j for j in JOINTS if j in holdouts)
    assert released.how == "released", released.reason
    assert released.torque_on == in_order
    assert f"except on {', '.join(in_order)}, which still read on" in released.reason
    assert transport._in_hand is True

    await transport.close()
    note = transport.close_note or ""
    named = in_order[0] if len(in_order) == 1 else f"{', '.join(in_order[:-1])} and {in_order[-1]}"
    assert note.startswith(f"the arm is in your hands ({LET_GO_WHERE_IT_STOOD}), but {named} "), (
        note
    )
    assert "torque on" in note and "cut its power" in note, note
    assert "nothing is holding it up" not in note, "a joint that holds was said to hold nothing"
    assert arm.config.disable_torque_on_disconnect is False, "the close dropped a partly limp arm"


async def test_the_mock_names_a_joint_its_release_left_holding() -> None:
    """The mock's twin of the partial release, so a rehearsal of `quackd robot release` and of
    the end-of-run offer ends on the line the arm would: its in-memory release used to take on
    every motor, so its close could only ever say nothing held the arm."""
    mock = LeRobotMock(rest_pose=dict(REST))
    mock.release_holdouts = ("wrist_roll",)
    released = await mock.let_go(anywhere=True)
    assert released.how == "released" and released.torque_on == ("wrist_roll",), released
    assert "except on wrist_roll, which still read on" in released.reason
    await mock.close()
    note = mock.close_note or ""
    assert "but wrist_roll still reads torque on and holds" in note, note
    assert "cut its power to let go of it" in note and "nothing is holding it up" not in note


async def test_an_arm_that_kept_torque_on_every_motor_was_not_released_and_is_in_no_hand() -> None:
    """Nothing let go, so nothing is in anybody's hands, and the close treats it as the arm
    holding itself up that it is: away from its pose, torque is kept and said to be kept.

    In words that do not send the person back to the release that just failed. The close
    used to end on `TORQUE_LEFT_ON`, which names `quackd robot release`, printed by that very
    command one line below its own failure; the one way out left is the switch."""
    arm = _spanned()
    recorded, reading = _stopped_short(arm)
    arm.positions.update(reading)
    arm.torque_holdouts = set(JOINTS)
    transport = LeRobotReal("COM5", robot=arm, rest_pose=recorded)
    await transport.connect()

    refused = await transport.let_go(anywhere=True)
    assert refused.how == "refused" and "still reports torque on" in refused.reason
    assert refused.torque_on == JOINTS
    assert transport._in_hand is False
    await transport.close()
    note = transport.close_note or ""
    assert note.startswith(TORQUE_LEFT_ON.split("(")[0]), note
    assert "the release did not take" in note and note.endswith("hold it and cut its power")
    assert "quackd robot release" not in note, "sent back to the command that just failed"
    assert arm.config.disable_torque_on_disconnect is False and arm.torque_disabled == 0


@pytest.mark.parametrize("pose", ["recorded", "none"], ids=["at the rest pose", "no rest pose"])
async def test_a_refused_release_the_close_then_let_go_of_says_the_close_did(pose: str) -> None:
    """The other half. An arm at its rest pose, or with none recorded, is let go of by the
    close's disconnect, as every such session ends, and that used to happen without a word
    after `quackd robot release` had just printed "torque still reads on: cut the power".
    What the person was told then no longer matched what quackd did: the same `Torque_Enable`
    0 went out again, and nothing read it back. It is said now, with what to do if it did not
    take this time either. A close with no refused release before it still says nothing."""
    arm = _spanned()
    arm.torque_holdouts = set(JOINTS)
    rest = {j: arm.positions[j] for j in SPANS} if pose == "recorded" else None
    transport = LeRobotReal("COM5", robot=arm, rest_pose=rest)
    await transport.connect()
    assert (await transport.let_go(anywhere=True)).how == "refused"
    await transport.close()
    note = transport.close_note or ""
    where = "at the rest pose" if pose == "recorded" else "where the arm stands"
    assert note.startswith(f"the release did not take, and the close then took torque off {where}")
    assert "cut its power if it still holds itself up" in note, note
    assert arm.torque_disabled == 1, "the close did not let go"

    quiet = _spanned()
    plain = LeRobotReal("COM5", robot=quiet, rest_pose=rest)
    await plain.connect()
    await plain.close()
    assert plain.close_note is None, "an ordinary close at rest said something"


@pytest.mark.parametrize("fault", ["the read-back", "the release call"])
async def test_a_release_that_went_out_and_was_not_confirmed_is_a_release_in_a_hand(
    fault: str,
) -> None:
    """ADR-0039's rule, now held from the moment the release is sent and not only once it has
    returned. A release call that raised has written some motors and not others, one at a
    time, so part of the arm may be limp: it used to come back `released` with the arm still
    reported in nobody's hands, and the close would then have told the person holding it that
    it holds itself up. `torque_on` is None, so nobody is told torque reads off."""
    arm = _spanned()
    recorded, reading = _stopped_short(arm)
    arm.positions.update(reading)
    transport = LeRobotReal("COM5", robot=arm, rest_pose=recorded)
    await transport.connect()
    if fault == "the read-back":
        arm.bus_error_after_release = True
    else:
        arm.bus_error = True  # positions still answer; the registers and the release do not

    released = await transport.let_go(anywhere=True)
    assert released.how == "released", released.reason
    assert released.torque_on is None, "a read that never came back is not 'every motor off'"
    if fault == "the read-back":
        assert "did not answer to confirm it" in released.reason, released.reason
    else:
        assert "the call did not come back" in released.reason, released.reason
        assert "hold the arm as though nothing holds it" in released.reason
    assert transport._in_hand is True
    await transport.close()
    assert transport.close_note == UNREAD_IN_HAND.format(why=LET_GO_WHERE_IT_STOOD)
    assert arm.config.disable_torque_on_disconnect is False


@pytest.mark.parametrize("anywhere", [False, True], ids=["first door", "second door"])
async def test_an_arm_that_did_not_answer_before_the_release_was_never_released(
    anywhere: bool,
) -> None:
    """The read before the release failed, so nothing was sent and the arm still holds itself.
    It used to come back `released` from either door, which on `--by-hand` told a person that
    an energised arm was theirs to lift, and from `quackd robot release` would have said torque
    was off on an arm that had not been touched."""
    arm, transport = _handover_arm()
    await transport.connect()
    arm.dead = True
    refused = await transport.let_go(anywhere=anywhere)
    assert refused.how == "refused", refused.reason
    assert "nothing was released" in refused.reason, refused.reason
    assert arm.timeline == [] and arm.torque is True
    assert transport._in_hand is False


@pytest.mark.parametrize(
    "away",
    [{"shoulder_pan": 30.0}, {"elbow_flex": -40.0, "wrist_flex": 25.0}],
    ids=["the pan", "the elbow and the wrist"],
)
async def test_the_mock_releases_away_from_its_rest_pose_only_when_a_person_asks(
    away: dict[str, float],
) -> None:
    """The rehearsal of `quackd robot release` and of the end-of-run offer: the mock refuses
    the first door where the arm does and opens the second in the same words, and its close
    then says the arm is limp in a hand."""
    mock = LeRobotMock(rest_pose=dict(REST))
    adapter = LeRobotAdapter(mock)
    await adapter.connect()
    assert (await adapter.send_intent(Intent.joint(away, 1.0))).accepted

    refused = await adapter.let_go()
    assert refused.how == "refused" and "the arm is not at its rest pose" in refused.reason
    assert mock.torque is True and mock.in_hand is False

    released = await adapter.let_go(anywhere=True)
    assert released.how == "released", released.reason
    assert released.reason == "torque is off where the arm stands"
    assert released.torque_on == () and mock.torque is False and mock.in_hand is True
    assert {j: released.joints[j] for j in away} == away
    await adapter.close()
    assert mock.close_note == LIMP_IN_HAND.format(why=LET_GO_WHERE_IT_STOOD), mock.close_note

    bare = LeRobotMock()
    assert (await bare.let_go()).how == "refused", "no pose, first door"
    assert (await bare.let_go(anywhere=True)).how == "released", "no pose, second door"


# ── the torque note names the ways out ──────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["lab-arm", "bench-2"])
async def test_the_torque_note_names_the_ways_out_under_the_name_the_arm_was_registered_by(
    name: str,
) -> None:
    """The note used to end at "hold the arm and cut its power, or run again", and on the
    bench every run that got that far ended at the switch. It now names the command that
    releases the arm and the one that parks it, spelled with this arm's own name: the registry
    builds a robot with its name as the id (`make(robot_id=...)`), and a command with the wrong
    name in it fails, or reaches another arm.

    And the hold comes before every one of them, with the reason. Both commands connect, and
    connecting takes torque off every motor for a moment, so an arm held up by torque alone
    is limp for that moment whichever of them reaches it. The note once tied the hold to the
    release alone and offered `doctor` beside it as though the arm could be left to hold
    itself while `doctor` connected, and said "it will not fall" of an arm about to be let
    go of by the connect: it is said now of the arm as it stands, and no further."""
    mock = make("mock", robot_id=name, rest_pose=dict(REST))
    await mock.connect()
    assert (await mock.send_intent(Intent.joint({"shoulder_pan": 30.0}, 1.0))).accepted
    await mock.close()
    real = make("real", address="COM5", robot_id=name, rest_pose=dict(FOLDED))
    assert getattr(real.transport, "registered_name", None) == name

    arm = FakeArm(step=40.0, stuck=("shoulder_lift",))
    transport = LeRobotReal("COM5", robot=arm, rest_pose=dict(FOLDED), registered_name=name)
    await transport.connect()
    assert (await transport.go_to_rest()).how == "stalled"
    await transport.close()

    for note in (mock.close_note or "", transport.close_note or ""):
        assert note.startswith(TORQUE_LEFT_ON.split("(")[0]), note
        assert "it will not fall as it stands" in note, note
        hold = note.index("hold it first, because connecting takes torque off every motor")
        release = note.index(f"quackd robot release {name}")
        doctor = note.index(f"quackd doctor --robot {name} to park it")
        power = note.index("or cut its power")
        assert hold < release < doctor < power, "the hold does not come before every way out"
        assert "hold" not in note[release:], "a way out stands after the hold as its own route"


async def test_an_arm_built_without_a_name_says_name_rather_than_its_default_id() -> None:
    """A bare spec (`doctor --robot lerobot:real`) or a backend called directly builds the arm
    with no name, and the real backend's id then defaults to the id a calibration is looked up
    under. That is not necessarily the name anybody registered, so the note says NAME, which
    a person can see is a placeholder."""
    mock = make("mock", rest_pose=dict(REST))
    await mock.connect()
    assert (await mock.send_intent(Intent.joint({"elbow_flex": 40.0}, 1.0))).accepted
    await mock.close()

    arm = FakeArm(step=40.0, stuck=("shoulder_lift",))
    transport = LeRobotReal("COM5", robot=arm, rest_pose=dict(FOLDED))
    await transport.connect()
    await transport.go_to_rest()
    await transport.close()

    for note in (mock.close_note or "", transport.close_note or ""):
        assert "quackd robot release NAME" in note and "doctor --robot NAME" in note, note
        assert transport.robot_id not in note, "the default id was offered as the name"


@pytest.mark.parametrize("name", ["lab-arm", None], ids=["registered", "unnamed"])
async def test_the_clip_note_names_the_arm_as_the_torque_note_does(name: str | None) -> None:
    """The two sentences a clipped pose and a missed one end on both give a person a command
    to type, and only one of them used to spell it with the arm's name: the clip note said
    `quackd robot rest-pose NAME` in a run on a registered arm, and in `quackd robot
    rest-pose <name>` itself, one line above a hint that used the name. Both now take the name
    the arm was built with, and both fall back to the same visible placeholder without one.
    Asked of the real backend's rest move, the mock's, and the adapter's own `rest_pose_note`,
    which is what `quackd robot rest-pose` prints."""
    shown = name or "NAME"
    arm = _spanned(step=60.0)
    pose = dict.fromkeys(SPANS, 0.0) | {"elbow_flex": _past(arm, "elbow_flex", 17.0)}
    transport = LeRobotReal("COM5", robot=arm, rest_pose=pose, registered_name=name)
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    parked = await adapter.go_to_rest()
    assert parked.reached and parked.note, parked

    ceiling = MOCK_RANGES["elbow_flex"][1]
    mock = LeRobotMock(rest_pose=dict(REST) | {"elbow_flex": ceiling + 11.0}, registered_name=name)
    rehearsed = await mock.go_to_rest()
    assert rehearsed.reached and rehearsed.note, rehearsed

    for note in (parked.note, rehearsed.note, adapter.rest_pose_note(pose) or ""):
        assert f"(quackd robot rest-pose {shown}) to make the fold reachable" in note, note
    await adapter.close()


# ── an arm that did not answer, and a release a Ctrl-C landed on ────────────────────────


async def test_a_rest_move_the_arm_stopped_answering_says_so_and_a_lost_write_does_not() -> None:
    """The end-of-run offer tells a person "it is holding itself up. Hold it and press Enter",
    and it used to say so after any missed rest move, including one refused because the arm
    stopped answering: which is what cutting the servo supply looks like, the one e-stop the
    safety page names. Nothing had been read that says the arm holds anything. The result now
    carries whether the arm answered the move's last read, which a write lost after a good
    read does not change: that arm answered, and is holding whatever it last read."""
    arm, transport = _handover_arm()
    transport.rest_pose = dict.fromkeys(SPANS, 0.0) | {"elbow_flex": 30.0}
    await transport.connect()
    arm.send_fails = True
    lost = await transport.go_to_rest()
    assert lost.how == "refused" and lost.answered, lost

    arm.send_fails = False
    arm.dead = True
    silent = await transport.go_to_rest()
    assert silent.how == "refused" and not silent.answered, silent
    assert "Present_Position" in silent.reason, silent.reason


async def test_the_close_of_an_arm_that_did_not_answer_never_says_it_holds_itself_up() -> None:
    """The close keeps what torque there may be when its own read of the joints fails, which
    is right, and it used to say "torque was left on and it will not fall" over that arm. An
    arm that stopped answering is as often one whose supply was cut at the switch, limp in
    somebody's hands, as one whose cable came out in front of live servos, and quackd cannot
    tell them apart. So it says it cannot, and what to do either way: hold it, cut its power.
    `robot list --probe` keeps which way round the note is when it shortens it."""
    from quackd.registry import _torque_phrase

    arm, transport = _handover_arm()
    await transport.connect()
    arm.dead = True
    await transport.close()
    note = transport.close_note or ""
    assert note.startswith("quackd cannot tell whether the arm is holding itself up"), note
    assert "Present_Position" in note and "cut its power" in note, note
    assert "will not fall" not in note and "torque was left on" not in note, note
    assert arm.config.disable_torque_on_disconnect is False, "whatever torque there is, is kept"
    assert _torque_phrase(note) == "torque unknown: the arm did not answer the close"


class SlowRelease(FakeBus):
    """A bus whose release write sits on the wire until the test lets it go, which is where a
    Ctrl-C after Enter lands: the call has been issued and its thread is still writing."""

    def __init__(self, arm: FakeArm) -> None:
        super().__init__(arm)
        self.writing = threading.Event()
        self.done = threading.Event()

    def disable_torque(self, motors: Any = None, num_retry: int = 0) -> None:
        self.writing.set()
        self.done.wait(timeout=5.0)
        super().disable_torque(motors, num_retry=num_retry)


async def test_a_release_a_ctrl_c_landed_on_is_an_arm_in_a_hand() -> None:
    """`let_go` takes the arm to be in somebody's hands from the moment the release is sent,
    because the other reading ends with the close telling a person holding a limp arm that it
    holds itself up. It did so only for a release that returned or raised an `Exception`: a
    cancellation landing while the write was on the wire, which is a Ctrl-C after Enter at
    the end-of-run offer, left the flag off while the thread went on and made the arm limp.
    The interrupt still goes on up, since it is not `let_go`'s to swallow."""
    arm = _spanned()
    recorded, reading = _stopped_short(arm)
    arm.positions.update(reading)
    arm.bus = SlowRelease(arm)
    transport = LeRobotReal("COM5", robot=arm, rest_pose=recorded)
    await transport.connect()

    release = asyncio.ensure_future(transport.let_go(anywhere=True))
    assert await asyncio.to_thread(arm.bus.writing.wait, 5.0), "the release never went out"
    release.cancel()
    with pytest.raises(asyncio.CancelledError):
        await release
    assert transport._in_hand is True, "a release on the wire left the arm in nobody's hands"

    arm.bus.done.set()
    for _ in range(50):
        if arm.torque is False and transport._wedged is not None and transport._wedged.done():
            break
        await asyncio.sleep(0.02)
    await transport.close()
    note = transport.close_note or ""
    assert note == LIMP_IN_HAND.format(why=LET_GO_WHERE_IT_STOOD), note
    assert arm.config.disable_torque_on_disconnect is False


@pytest.mark.parametrize("pose", ["recorded", "none"], ids=["a rest pose", "no rest pose"])
async def test_a_close_while_an_interrupted_release_is_still_out_says_nothing_read_it_back(
    pose: str,
) -> None:
    """The close picked its line with "the motors a read found on, if a read has answered since
    the release, else none", and none meant `LIMP_IN_HAND`: "nothing is holding it up". So a
    Ctrl-C on the release at the end-of-run offer, with the loop going straight on to the close
    while the release's thread was still writing, closed on that line over a bus that refused
    every read, and over a joint the release had not reached yet, kept energised past the
    close. Nothing read the release back, and the close now says exactly that, and to cut the
    power to be sure. The joint left holding is synthetic, and so is the pose."""
    arm = _spanned()
    recorded, reading = _stopped_short(arm)
    arm.positions.update(reading)
    arm.bus = SlowRelease(arm)
    arm.torque_holdouts = {sorted(SPANS)[1]}
    transport = LeRobotReal("COM5", robot=arm, rest_pose=recorded if pose == "recorded" else None)
    await transport.connect()

    release = asyncio.ensure_future(transport.let_go(anywhere=True))
    assert await asyncio.to_thread(arm.bus.writing.wait, 5.0), "the release never went out"
    release.cancel()
    with pytest.raises(asyncio.CancelledError):
        await release
    try:
        await transport.close()  # the release's thread is still on the wire
        note = transport.close_note or ""
        assert note == UNREAD_IN_HAND.format(why=LET_GO_WHERE_IT_STOOD), note
        assert "nothing is holding it up" not in note, note
        assert arm.config.disable_torque_on_disconnect is False, "a partly limp arm was dropped"
    finally:
        arm.bus.done.set()


class GivesUpPartWay(FakeBus):
    """A bus whose release writes the motors before one of them and then raises on it, as
    upstream's `disable_torque` does when one motor's write fails: the motors after it keep
    their torque (`arm.torque_holdouts`)."""

    def disable_torque(self, motors: Any = None, num_retry: int = 0) -> None:
        super().disable_torque(motors, num_retry=num_retry)
        (stuck,) = self.arm.torque_holdouts
        raise ConnectionError(
            f"Failed to write 'Torque_Enable' on id_={self.motors[stuck].id} with '0' after "
            f"{num_retry + 1} tries. {NO_STATUS}"
        )


async def test_a_release_that_raised_part_way_on_an_arm_with_no_pose_is_not_called_limp() -> None:
    """`quackd robot release` on an arm registered with no rest pose: the release raised part
    way, which leaves the motors after the failed one energised, and the close, which makes no
    read of its own without a pose to judge against, ended on "nothing is holding it up" right
    after telling the person that some joints may still hold. It ends on nothing having read
    the release back now, and on the power switch."""
    arm = _spanned()
    arm.positions.update({j: _inside(arm, j, 0.3) for j in SPANS})
    arm.bus = GivesUpPartWay(arm)
    arm.torque_holdouts = {sorted(SPANS)[-1]}
    transport = LeRobotReal("COM5", robot=arm)
    await transport.connect()

    released = await transport.let_go(anywhere=True)
    assert released.how == "released" and released.torque_on is None, released
    assert "some joints may be limp and some may still hold" in released.reason
    reads = len(arm.reads)
    await transport.close()
    assert len(arm.reads) == reads, "the close made a read of its own after all"
    note = transport.close_note or ""
    assert note == UNREAD_IN_HAND.format(why=LET_GO_WHERE_IT_STOOD), note
    assert "nothing is holding it up" not in note and "cut its power to be sure" in note
    assert arm.config.disable_torque_on_disconnect is False


async def test_the_mock_closes_a_release_nothing_read_back_in_the_real_close_s_words() -> None:
    """The mock's twin, so a rehearsal of `quackd robot release` or of the end-of-run offer can
    end the way an arm whose read-back failed ends: its in-memory release always read back, so
    its close could only ever say the arm was limp or name a joint still on."""
    mock = LeRobotMock()
    mock.release_unread = "a reason made up for this test"
    released = await mock.let_go(anywhere=True)
    assert released.how == "released" and released.torque_on is None, released
    assert "did not answer to confirm it (a reason made up for this test)" in released.reason
    await mock.close()
    assert mock.close_note == UNREAD_IN_HAND.format(why=LET_GO_WHERE_IT_STOOD), mock.close_note

    read = LeRobotMock()
    assert (await read.let_go(anywhere=True)).torque_on == ()
    await read.close()
    assert read.close_note == LIMP_IN_HAND.format(why=LET_GO_WHERE_IT_STOOD), read.close_note


class HeldRead(FakeBus):
    """A bus whose torque register read sits on the wire until the test lets it go, once told
    to: the read `let_go` makes before it sends anything."""

    def __init__(self, arm: FakeArm) -> None:
        super().__init__(arm)
        self.hold = False
        self.reading = threading.Event()
        self.done = threading.Event()

    def sync_read(
        self, data_name: str, motors: Any = None, *, normalize: bool = True, num_retry: int = 0
    ) -> dict[str, int]:
        if self.hold and data_name == "Torque_Enable":
            self.reading.set()
            self.done.wait(timeout=5.0)
        return super().sync_read(data_name, motors, normalize=normalize, num_retry=num_retry)


async def test_an_interrupt_on_the_read_before_a_release_refuses_nothing_and_sends_nothing() -> (
    None
):
    """`let_go` marked a release through the second door refused on the way in and cleared the
    mark only once its first read had come back. A Ctrl-C on that read went on up with nothing
    sent and the mark still set, so the close said "the release did not take" of a release that
    never went out. The mark is set only where a refusal is returned now, and `in_hand` stays
    False, which is how the caller tells an interrupt before the send from one after it."""
    arm = _spanned()
    recorded, reading = _stopped_short(arm)
    arm.positions.update(reading)
    arm.bus = HeldRead(arm)
    transport = LeRobotReal("COM5", robot=arm, rest_pose=recorded)
    await transport.connect()

    arm.bus.hold = True
    release = asyncio.ensure_future(transport.let_go(anywhere=True))
    try:
        assert await asyncio.to_thread(arm.bus.reading.wait, 5.0), "the read never went out"
        release.cancel()
        with pytest.raises(asyncio.CancelledError):
            await release
        assert "disable_torque" not in arm.timeline and arm.torque is True, "a release went out"
        assert transport.in_hand is False and LeRobotAdapter(transport).in_hand is False
        assert transport._release_refused is False, "a release nobody refused was marked so"
    finally:
        arm.bus.hold = False
        arm.bus.done.set()
    for _ in range(50):
        if transport._wedged is None or transport._wedged.done():
            break
        await asyncio.sleep(0.02)
    await transport.close()
    note = transport.close_note or ""
    assert note.startswith("the arm is not at its rest pose"), note
    assert "the release did not take" not in note, note


@pytest.mark.parametrize("hangs", ["the goal write", "the hold after a stall"])
async def test_a_rest_move_whose_write_never_came_back_did_not_answer(hangs: str) -> None:
    """`RestResult.answered` was taken from the last read alone, which only a read clears, so a
    rest move whose goal write never came back, and left the bus wedged behind it, said the arm
    had answered, and so did one that stalled and whose hold, sent quietly as it gave up, never
    came back. The loop then told a person the arm was holding itself up and to hold it and
    press Enter, and the release that followed was refused at its first read on the same wedge.
    A call that never came back is an arm that did not answer, read or write. A write the arm
    refused after a read that came back is still an arm that answered (the lost-write test
    above). Synthetic travel, goal and obstacle."""
    release = threading.Event()
    arm_travel = FakeArm().travel("elbow_flex")
    goal = arm_travel[1] / 2

    class Hangs(FakeArm):
        def send_action(self, action: dict[str, float]) -> dict[str, float]:
            # the rest move's goal, or any other write: the hold, which writes where it stopped
            if (hangs == "the goal write") == (action.get("elbow_flex.pos") == goal):
                release.wait(5.0)
            return super().send_action(action)

    arm = Hangs()
    recorded = {j: arm.positions[j] for j in JOINTS if j != "gripper"} | {"elbow_flex": goal}
    if hangs == "the hold after a stall":
        arm.obstacles["elbow_flex"] = (arm_travel[0], goal / 3)  # something in the way
    transport = LeRobotReal("COM5", robot=arm, rest_pose=recorded, timeout_s=0.2)
    try:
        await transport.connect()
        missed = await transport.go_to_rest()
        assert not missed.reached, missed
        if hangs == "the goal write":
            assert missed.how == "refused" and "has not come back" in missed.reason, missed
        else:
            assert missed.how == "stalled" and transport._wedged is not None, missed
        assert transport._answered is True, "the reads did come back"
        assert missed.answered is False, "a write that never came back was said to be answered"
    finally:
        release.set()


class OneRegisterDown(FakeBus):
    """A bus where one named register is corrupt and the rest answer honestly.

    The whole shape of the blocker below: the two status registers used to be read inside one
    `try`, so a bad packet on either was reported as a failure of both."""

    def __init__(self, arm: FakeArm, dead: tuple[str, ...]) -> None:
        super().__init__(arm)
        self.dead = dead

    def sync_read(
        self, data_name: str, motors: Any = None, *, normalize: bool = True, num_retry: int = 0
    ) -> dict[str, int]:
        if data_name in self.dead:
            self.arm.reads.append((data_name, normalize, num_retry))
            raise RuntimeError("Incorrect status packet!")
        return super().sync_read(data_name, motors, normalize=normalize, num_retry=num_retry)


async def test_a_corrupt_temperature_packet_cannot_answer_for_the_torque_register() -> None:
    """The blocker this was all written to prevent.

    A servo that takes `enable_torque()` and stays limp is the case `take_hold` reads the
    register back for. The two registers used to share one `try`, so a corrupt temperature
    packet set the same error flag as a corrupt torque one, and the guard was written to skip
    the torque check whenever that flag was set. quackd therefore had an honest reading saying
    the arm was limp, discarded it, answered `held`, and the run told the person holding an
    unpowered arm 70 degrees out of the fold that they could let go."""
    arm = FakeArm()
    arm.positions.update(FOLDED)
    transport = LeRobotReal("COM5", robot=arm, rest_pose=dict(FOLDED))
    arm.bus = OneRegisterDown(arm, dead=("Present_Temperature",))
    arm.torque_refuses = True  # the servo ignores enable_torque, as one in lockout does
    await transport.connect()

    assert (await transport.let_go()).how == "released"
    arm.positions["shoulder_lift"] = -20.0  # the person lifts it out of the fold
    held = await transport.take_hold()

    assert held.how == "refused", "an arm that is still limp is not an arm that is holding"
    assert "torque off" in held.reason
    assert transport._in_hand is True, "so it is still in somebody's hands"
    assert arm.torque is False, "and it really is limp, which is what was read and believed"


async def test_a_torque_register_that_says_nothing_is_not_a_hold() -> None:
    """The other half: the bus goes silent at the moment torque is asked for, which is what a
    marginal connector or a browning-out rail looks like.

    `let_go` reads the same silence the other way round on purpose, because there a release
    that did not happen costs a refusal and here a hold that did not happen costs the arm."""
    arm = FakeArm()
    arm.positions.update(FOLDED)
    transport = LeRobotReal("COM5", robot=arm, rest_pose=dict(FOLDED))
    await transport.connect()
    assert (await transport.let_go()).how == "released"

    arm.bus = OneRegisterDown(arm, dead=("Torque_Enable", "Present_Temperature"))
    arm.torque_refuses = True
    held = await transport.take_hold()

    assert held.how == "refused"
    assert "did not say whether torque came back on" in held.reason
    assert held.energised is None, "the torque write went out and nothing read it back"
    assert transport._in_hand is True


async def test_an_arm_closed_on_while_still_in_a_hand_keeps_whatever_torque_it_has() -> None:
    """`close()` on an arm in somebody's hand used to return before the seam that keeps
    torque, so it always disconnected on LeRobot's drop-torque default.

    That is a no-op on an arm that is genuinely limp, which is the state the branch was
    written for. It is not a no-op on the state the blocker above now produces: an arm whose
    torque could not be read back, which may be energised and is certainly not at its fold.
    Dropping it there drops the arm, and keeping it costs nothing either way.

    And its line is not the limp one. It said "the arm is limp and in your hands ... nothing is
    holding it up" from the read that confirmed the release, taken before the take-hold's
    torque write went out, over an arm that write may have energised. It says quackd cannot
    tell, to hold the arm against both, and the switch."""
    arm = FakeArm()
    arm.positions.update(FOLDED)
    transport = LeRobotReal("COM5", robot=arm, rest_pose=dict(FOLDED))
    await transport.connect()
    await transport.let_go()
    arm.positions["shoulder_lift"] = -20.0
    arm.bus = OneRegisterDown(arm, dead=("Torque_Enable",))
    arm.torque_refuses = True
    assert (await transport.take_hold()).how == "refused"

    await transport.close()
    assert arm.config.disable_torque_on_disconnect is False, "quackd kept what torque there is"
    assert arm.torque_disabled == 0, "and the disconnect dropped none"
    assert transport.close_note == UNCONFIRMED_IN_HAND, transport.close_note


# ── a connect the bus loses a packet in ─────────────────────────────────────────────────

REWIRED = {
    "wrist_flex": 17,
    "shoulder_pan": 72,
    "gripper": 7,
    "elbow_flex": 31,
    "wrist_roll": 9,
    "shoulder_lift": 26,
}
"""A motor table in no order a follower lists its motors in, with ids nowhere near one to six,
and one id (7) that is a prefix of another (72). A test that names the right joint off this
cannot be doing it by position or by a prefix match."""

NO_STATUS = "[TxRxResult] There is no status packet!"
CORRUPT = "[TxRxResult] Incorrect status packet!"
NOT_SENT = "[TxRxResult] Failed transmit instruction packet!"


def _lost_write(register: str, motor_id: int, value: int, result: str) -> ConnectionError:
    """`MotorsBus.write`'s own sentence for a write whose status packet did not come back
    (`up.BUS_WRITE_ERROR_NAMES_THE_ID`), for whatever register, id, value and transaction
    result a test picks. One try, which is what `configure()` gives each torque write."""
    return ConnectionError(
        f"Failed to write '{register}' on id_={motor_id} with '{value}' after 1 tries. {result}"
    )


SOME_MODEL = 4321
"""A servo model number for a handshake's message. Made up: nothing quackd does reads it."""


def _motor_check_failed(
    missing: Sequence[int] = (),
    wrong: Mapping[int, tuple[str, int]] | None = None,
    port: str = "COM7",
) -> RuntimeError:
    """`_assert_motors_exist`'s own refusal, line for line (`up.HANDSHAKE_NAMES_THE_ID`), for
    whatever ids a test picks: a line per motor that did not answer its ping, and one per motor
    that answered as another model, given as `id -> (joint, the model it answered as)`."""
    wrong = dict(wrong or {})
    lines = [f"FeetechMotorsBus motor check failed on port '{port}':"]
    if missing:
        lines.append("\nMissing motor IDs:")
        lines.extend(f"  - {n} (expected model: {SOME_MODEL})" for n in missing)
    if wrong:
        lines.append("\nMotors with incorrect model numbers:")
        lines.extend(
            f"  - {n} ({joint}): expected {SOME_MODEL}, found {found}"
            for n, (joint, found) in wrong.items()
        )
    lines.append("\nFull expected motor list (id: model_number):")
    lines.append(pformat(dict.fromkeys([*missing, *wrong], SOME_MODEL), indent=4))
    lines.append("\nFull found motor list (id: model_number):")
    lines.append(pformat({n: found for n, (_joint, found) in wrong.items()}, indent=4))
    return RuntimeError("\n".join(lines))


def _rewired(port: str = "COM7") -> tuple[FakeArm, FakeCamera, LeRobotReal]:
    """An arm on the `REWIRED` table with a camera open beside it, and no pause between
    attempts, because a test's clock is not the bench's."""
    arm = FakeArm()
    arm.bus = FakeBus(arm, {joint: SimpleNamespace(id=n) for joint, n in REWIRED.items()})
    camera = FakeCamera()
    transport = LeRobotReal(
        port, robot=arm, camera=parse_camera_url("opencv://0"), camera_object=camera
    )
    transport.connect_pause_s = 0.0
    return arm, camera, transport


@pytest.mark.parametrize(
    ("joint", "register", "value", "result", "failures"),
    [
        ("elbow_flex", "Lock", 1, NO_STATUS, 1),
        ("gripper", "Torque_Enable", 0, CORRUPT, CONNECT_ATTEMPTS - 1),
        ("shoulder_pan", "P_Coefficient", 16, NOT_SENT, 1),
    ],
)
async def test_a_packet_lost_at_connect_closes_the_port_writing_nothing_and_connects_again(
    joint: str,
    register: str,
    value: int,
    result: str,
    failures: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Bench, 2026-09-23: runs ended before they began, at connect, on LeRobot's `Failed to
    write ... after 1 tries`, each on the first connect after a power cut and on a different
    motor each time, and the next connect by hand went through. quackd gave up on the first
    lost packet and left the port open behind it.

    Now the port is closed through the bus with no torque write (the follower's own disconnect
    would have switched torque off on every motor, on the bus that had just lost a packet), the
    camera stays open, connect runs again, and every retry is said while it happens and kept
    for the record, naming the joint through the bus's own motor table."""
    arm, camera, transport = _rewired()
    motor_id = REWIRED[joint]
    arm.connect_errors = [_lost_write(register, motor_id, value, result)] * failures
    adapter = LeRobotAdapter(transport)
    with caplog.at_level(logging.WARNING, logger="quackd.lerobot"):
        manifest = await adapter.connect()

    attempt = [("connect", False)]
    assert arm.calls == [*[*attempt, ("bus.disconnect", False)] * failures, *attempt], arm.calls
    assert arm.timeline == [] and arm.torque_retries == [], "torque was written between attempts"
    assert arm.torque_disabled == 0 and arm.is_connected
    assert camera.calls == ["connect"], "the camera was closed or reopened across the attempts"
    assert manifest.provides("observe") and manifest.extras["joint_range_deg"]
    notes = adapter.connect_notes
    assert len(notes) == failures, notes
    for k, note in enumerate(notes, start=1):
        assert note.startswith(
            f"connect attempt {k} of {CONNECT_ATTEMPTS} failed on {joint} (id {motor_id}): "
            f"Failed to write '{register}' on id_={motor_id}"
        ), note
        assert result in note and "without a write to any motor" in note, note
    warned = [r.getMessage() for r in caplog.records if r.name == "quackd.lerobot"]
    assert warned == list(notes), "said in the log as it happened, and the same words kept"
    assert all(r.levelno == logging.WARNING for r in caplog.records if r.name == "quackd.lerobot")


@pytest.mark.parametrize(
    ("joint", "register", "result", "port"),
    [
        ("wrist_flex", "Lock", CORRUPT, "COM7"),
        ("shoulder_lift", "Torque_Enable", NO_STATUS, "/dev/ttyACM1"),
        ("gripper", "Max_Torque_Limit", NOT_SENT, "COM12"),
    ],
)
async def test_a_connect_that_fails_every_attempt_names_the_joint_and_says_hold_the_arm(
    joint: str, register: str, result: str, port: str
) -> None:
    """The arm that keeps losing packets. Each attempt may have stopped anywhere in the
    torque writes, which go off on every motor and back on one at a time, so the motors before
    the failed write can be holding and the ones after it limp. The refusal carries LeRobot's
    words, names the joint it named, says the arm may be half energised, and names the cable
    and the port to check. The port is left shut and the camera closed, and not one torque write
    went out on the way, on the bus that would not answer."""
    arm, camera, transport = _rewired(port)
    motor_id = REWIRED[joint]
    arm.connect_errors = [_lost_write(register, motor_id, 1, result)] * CONNECT_ATTEMPTS
    with pytest.raises(TransportError) as raised:
        await LeRobotAdapter(transport).connect()

    why = str(raised.value)
    assert why.startswith(
        f"lerobot real: connect failed {CONNECT_ATTEMPTS} times, the last on {joint} "
        f"(id {motor_id}): Failed to write '{register}' on id_={motor_id}"
    ), why
    assert result in why
    assert "some motors may be left with torque on and others off" in why, why
    assert "keep a hand under the arm" in why, why
    assert (
        f"Check {joint}'s cable and connectors, that the servo supply is on, and that nothing "
        f"else has {port} open"
    ) in why, why
    assert arm.calls.count(("connect", False)) == CONNECT_ATTEMPTS
    assert arm.calls.count(("bus.disconnect", False)) == CONNECT_ATTEMPTS
    assert ("bus.disconnect", True) not in arm.calls and ("disconnect",) not in arm.calls
    assert arm.timeline == [] and arm.torque_retries == [] and arm.torque_disabled == 0
    assert not arm.is_connected, "the port was left open behind the refusal"
    assert camera.calls == ["connect", "disconnect"], "the camera was left open"
    assert len(transport.connect_notes) == CONNECT_ATTEMPTS - 1


def test_the_joint_is_named_through_the_bus_s_own_table_and_never_by_position() -> None:
    """The id in LeRobot's message is a bus address nobody at the arm can see. It is turned
    into a joint through the motor table that gave each servo its address, whatever register,
    value, try count or transaction result surrounds it, and read or write alike. An id the
    table does not know is still said, as a motor; a message that names several motors or none
    names no joint, because a guess sends somebody to the wrong cable.

    The handshake names its motors in lists instead, and only `id_=` was read, so a servo that
    stopped answering its ping, the failure the connect retry says a further try will not fix,
    was refused with "check the arm's cables and power" although LeRobot had said which one."""
    table = {joint: SimpleNamespace(id=n) for joint, n in REWIRED.items()}
    for joint, n in REWIRED.items():
        for message in (
            f"Failed to write 'Lock' on id_={n} with '1' after 1 tries. {CORRUPT}",
            f"Failed to write 'Goal_Position' on id_={n} with '2311' after 4 tries. "
            "[RxPacketError] Overload error!",
            f"Failed to read 'Present_Temperature' on id_={n} after 3 tries. {NO_STATUS}",
        ):
            assert motor_in_error(message, table) == (f"{joint} (id {n})", joint), message
    stranger = max(REWIRED.values()) + 1
    said = f"Failed to write 'Lock' on id_={stranger} with '0' after 1 tries. {NOT_SENT}"
    assert motor_in_error(said, table) == (f"motor {stranger}", f"motor {stranger}")
    assert motor_in_error(said, None) == (f"motor {stranger}", f"motor {stranger}")
    # the handshake lists the servos it could not find, or found as another model, one a line;
    # the first listed is named, the missing one when there are both, raw or on one line
    ids = list(REWIRED.values())
    for n, joint in ((n, j) for j, n in REWIRED.items()):
        others = [m for m in ids if m != n]
        for check in (
            _motor_check_failed(missing=[n, *others[:2]]),
            _motor_check_failed(wrong={n: (joint, SOME_MODEL + 1), others[0]: ("x", 0)}),
            _motor_check_failed(missing=[n], wrong={others[0]: ("x", SOME_MODEL + 2)}),
        ):
            for message in (str(check), " ".join(str(check).split())):
                assert motor_in_error(message, table) == (f"{joint} (id {n})", joint), message
    lost = str(_motor_check_failed(missing=[stranger]))
    assert motor_in_error(lost, table) == (f"motor {stranger}", f"motor {stranger}")
    for nameless in (
        f"Failed to sync read 'Present_Position' on ids=[7, 9, 17] after 3 tries. {NO_STATUS}",
        f"Failed to sync write 'Goal_Position' with ids_values={{72: 2048}} after 1 tries. "
        f"{CORRUPT}",
        "\nCould not connect on port 'COM7'. Make sure you are using the correct port."
        "\nTry running `lerobot-find-port`\n",
    ):
        assert motor_in_error(nameless, table) is None, nameless


@pytest.mark.parametrize(
    ("script", "torque_said"),
    [
        pytest.param("port", False, id="the port never opened"),
        pytest.param("configure", True, id="the port opened and the message names no motor"),
        pytest.param("opened_once", True, id="the port opened once and then never again"),
    ],
)
async def test_a_failure_that_names_no_motor_is_retried_and_names_none(
    script: str, torque_said: bool
) -> None:
    """Not every connect failure is a lost torque write. A port that will not open (a wrong
    one, or one something else is holding) is as likely to be a passing moment as a lost
    packet, so it is tried again too, and its message names no motor, so the refusal names no
    joint and sends the person to the cables and the port. Whether it warns about torque turns
    on whether any attempt can have written it: one that never opened the port wrote to no
    motor, and one that failed after the handshake, even once, may have left the motors in two
    states. The nameless failure here is raised after the handshake, where `configure()` is."""
    arm = FakeArm()
    port_says = ConnectionError(
        "\nCould not connect on port 'COM3'. Make sure you are using the correct port."
        "\nTry running `lerobot-find-port`\n"
    )
    nameless = RuntimeError("the servo answered something configure() did not expect")
    if script == "port":
        arm.port_errors = [port_says] * CONNECT_ATTEMPTS
    elif script == "configure":
        arm.connect_errors = [nameless] * CONNECT_ATTEMPTS
    else:
        arm.port_errors = [None, *[port_says] * (CONNECT_ATTEMPTS - 1)]
        arm.connect_errors = [nameless]
    transport = LeRobotReal("COM3", robot=arm)
    transport.connect_pause_s = 0.0
    with pytest.raises(TransportError) as raised:
        await transport.connect()

    why = str(raised.value)
    assert why.startswith(f"lerobot real: connect failed {CONNECT_ATTEMPTS} times: "), why
    assert "\n" not in why and ", the last on" not in why, why
    assert ("torque on and others off" in why) is torque_said, why
    assert "Check the arm's cables and power, and that nothing else has COM3 open" in why
    assert arm.calls.count(("connect", False)) == CONNECT_ATTEMPTS
    assert arm.timeline == [] and not arm.is_connected
    # each note says what that attempt did to the port, which is not always what the last did
    opened_on = {
        "port": set(),
        "configure": set(range(1, CONNECT_ATTEMPTS + 1)),
        "opened_once": {1},
    }[script]
    assert len(transport.connect_notes) == CONNECT_ATTEMPTS - 1
    for k, note in enumerate(transport.connect_notes, start=1):
        assert note.startswith(f"connect attempt {k} of {CONNECT_ATTEMPTS} failed: "), note
        closed = "The port was closed without a write to any motor" in note
        never = "The port never opened, so nothing reached a motor" in note
        assert (closed, never) == (k in opened_on, k not in opened_on), note


@pytest.mark.parametrize("how", ["blows its deadline", "raises TimeoutError"])
async def test_a_connect_that_timed_out_is_never_tried_again(how: str) -> None:
    """A connect that has not come back is a worker thread still sitting on the serial bus,
    and `_call` has wedged the transport for it. Trying again would put a second talker on a
    half-duplex bus, which is how packets get lost in the first place, so a timeout, whether
    the deadline ran out or LeRobot raised one itself, ends the connect at the first attempt.
    The camera is still let go of."""
    release = threading.Event()

    class Hangs(FakeArm):
        def connect(self, calibrate: bool = True) -> None:
            super().connect(calibrate)
            release.wait(5.0)

    arm = Hangs() if how == "blows its deadline" else FakeArm()
    if how == "raises TimeoutError":
        arm.connect_errors = [TimeoutError("the port read timed out")] * CONNECT_ATTEMPTS
    camera = FakeCamera()
    transport = LeRobotReal(
        "COM7", robot=arm, camera=parse_camera_url("opencv://0"), camera_object=camera
    )
    transport.connect_deadline_s = 0.2
    transport.connect_pause_s = 0.0
    try:
        with pytest.raises(TransportError, match="lerobot real: connect failed: ") as raised:
            await LeRobotAdapter(transport).connect()
        assert arm.calls.count(("connect", False)) == 1, "a timed-out connect was tried again"
        assert transport.connect_notes == []
        assert camera.calls == ["connect", "disconnect"]
        if how == "blows its deadline":
            assert "has not come back" in str(raised.value), raised.value
            assert transport._wedged is not None
            assert ("bus.disconnect", False) not in arm.calls, "the port was touched while wedged"
        else:
            assert "the port read timed out" in str(raised.value)
            assert not arm.is_connected, "a port nothing owns any more was left open"
    finally:
        release.set()


@pytest.mark.parametrize(
    ("script", "named", "torque_said"),
    [
        pytest.param("missing", True, False, id="a servo that never answers its ping"),
        pytest.param("wrong model", True, False, id="a servo that answers as another model"),
        pytest.param("serial", False, False, id="a serial error LeRobot calls a port error"),
        pytest.param("wrote first", True, True, id="a lost write, then a servo that is gone"),
        pytest.param("a write", True, True, id="a failed write, wherever it was raised"),
        pytest.param("a lost write", False, True, id="a lost write LeRobot calls a port error"),
    ],
)
async def test_a_connect_refused_in_the_handshake_names_the_servo_and_wrote_no_torque(
    script: str, named: bool, torque_said: bool
) -> None:
    """A servo that stopped answering, or the cable to it that came out, is the failure a
    further attempt will not fix, and LeRobot reports it from the handshake that opens every
    connect: a list of the ids that did not answer their ping. quackd read only `id_=`, so the
    refusal sent the person to "the arm's cables and power" although LeRobot had named the
    servo, and told them to keep a hand under the arm because the port had opened, when the
    handshake pings and reads and not one torque write had gone out.

    Now the joint is named through the bus's own table, and the torque warning is given only
    when an attempt can have written torque: a lost write earlier in the same connect still
    earns it, because a later attempt that writes nothing leaves the motors where it found
    them. A serial error in the handshake comes out as LeRobot's port error, raised from it,
    and is placed by the error it was raised from. A failed write is the other sign torque can
    have been written, read off the words and those of the error it was raised from, and it
    warns wherever it was raised: the handshake at 0.6.1 writes nothing, and a LeRobot whose
    handshake did would be one to warn about."""
    arm, camera, transport = _rewired()
    joint, n = next(iter(REWIRED.items()))
    refused: BaseException = {
        "missing": _motor_check_failed(missing=[n]),
        "wrong model": _motor_check_failed(wrong={n: (joint, SOME_MODEL + 1)}),
        "serial": OSError("ClearCommError failed (a synthetic glitch)"),
        "wrote first": _motor_check_failed(missing=[n]),
        "a write": RuntimeError(
            f"Failed to write 'Lock' on id_={n} with '0' after 1 tries. [RxPacketError] "
            "Overload error!"
        ),
        "a lost write": _lost_write("Lock", n, 0, NOT_SENT),
    }[script]
    arm.handshake_errors = [refused] * CONNECT_ATTEMPTS
    if script == "wrote first":
        arm.handshake_errors[0] = None
        arm.connect_errors = [_lost_write("Torque_Enable", REWIRED["gripper"], 0, NO_STATUS)]
    with pytest.raises(TransportError) as raised:
        await LeRobotAdapter(transport).connect()

    why = str(raised.value)
    head = f"lerobot real: connect failed {CONNECT_ATTEMPTS} times"
    assert why.startswith(f"{head}, the last on {joint} (id {n}): " if named else f"{head}: ")
    assert (SPLIT_TORQUE in why) is torque_said, why
    look = (
        f"{joint}'s cable and connectors, that the servo supply is on,"
        if named
        else "the arm's cables and power,"
    )
    assert f"Check {look} and that nothing else has COM7 open" in why, why
    assert arm.calls.count(("connect", False)) == CONNECT_ATTEMPTS
    assert arm.timeline == [] and not arm.is_connected
    assert camera.calls == ["connect", "disconnect"]
    for note in transport.connect_notes:
        assert "The port was closed without a write to any motor" in note, note


@pytest.mark.parametrize(
    ("script", "torque_said"),
    [
        pytest.param("hangs", True, id="the first attempt blows its deadline"),
        pytest.param("times out", True, id="LeRobot raises TimeoutError"),
        pytest.param("wrote, then hangs", True, id="a lost write, then the deadline"),
        pytest.param("missing, then hangs", True, id="a servo missing, then the deadline"),
        pytest.param("wrote, then the close hangs", True, id="a lost write, then a wedged bus"),
        pytest.param("wedged", False, id="a wedged bus refused before anything went out"),
    ],
)
async def test_a_connect_that_ends_on_a_timeout_says_hold_the_arm_when_torque_may_be_split(
    script: str, torque_said: bool
) -> None:
    """The timeout exit raised a bare "connect failed" and never looked at what the attempts
    before it had done. So a connect whose first attempt lost a torque write, leaving some
    motors holding and others limp, and whose next attempt then blew its deadline, was refused
    without the warning to keep a hand under the arm. A timeout earns the warning on its own
    as well: an attempt still on the wire when quackd stopped waiting is somewhere in the
    handshake or in `configure()` and nothing says which, and one LeRobot timed out itself
    could have been in either. quackd's own refusal of a wedged bus says it when an earlier
    attempt can have written torque (a close between attempts that did not come back wedges the
    bus as any call does), and only then: with nothing written before it, it is the one exit
    through here that sent nothing and says nothing about torque."""
    release = threading.Event()
    hang_on: set[int] = {1} if script == "hangs" else {2}

    class Hangs(FakeArm):
        def connect(self, calibrate: bool = True) -> None:
            super().connect(calibrate)
            if self.calls.count(("connect", False)) in hang_on:
                release.wait(5.0)

    class ClosesSlowly(FakeBus):
        def disconnect(self, disable_torque: bool = True) -> None:
            release.wait(5.0)
            super().disconnect(disable_torque)

    arm = Hangs()
    if script == "times out":
        arm.connect_errors = [TimeoutError("the port read timed out")]
    elif script == "wrote, then hangs":
        arm.connect_errors = [_lost_write("Lock", arm.bus.motors["wrist_roll"].id, 1, CORRUPT)]
    elif script == "missing, then hangs":
        arm.handshake_errors = [_motor_check_failed(missing=[arm.bus.motors["gripper"].id])]
    elif script == "wrote, then the close hangs":
        lost = _lost_write("Torque_Enable", arm.bus.motors["shoulder_pan"].id, 0, NO_STATUS)
        arm.connect_errors = [lost]
        arm.bus = ClosesSlowly(arm)
    transport = LeRobotReal("COM7", robot=arm)
    transport.connect_deadline_s = 0.2
    transport.port_close_deadline_s = 0.2
    transport.connect_pause_s = 0.0
    if script == "wedged":
        transport._wedged = asyncio.get_running_loop().create_future()
        transport.stop_error = "a LeRobot call (send_action) has not come back"
    try:
        with pytest.raises(TransportError, match="lerobot real: connect failed: ") as raised:
            await transport.connect()
        why = str(raised.value)
        assert (SPLIT_TORQUE in why) is torque_said, why
        assert why.count(SPLIT_TORQUE) <= 1 and ".." not in why, why
        if script == "wedged":
            assert arm.calls == [], "a wedged transport put something on the bus"
        if script == "wrote, then the close hangs":
            assert "a LeRobot call (disconnect) has not come back" in why, why
            assert arm.calls.count(("connect", False)) == 1, "a wedged bus was connected again"
    finally:
        release.set()


async def test_a_serial_error_that_left_the_port_busy_does_not_fail_the_next_attempt() -> None:
    """The servo SDK raises its port handler's busy flag before every packet and lowers it when
    the reply is in, so a serial error in between (a USB glitch in a write) leaves it raised,
    and reopening the port does not lower it. quackd closed the port between attempts without
    the torque-off, and upstream clears the flag only inside that same branch, so every packet
    of the next attempts was answered "port in use" unsent: the handshake found no motor, and
    the retry built for a passing fault failed in the one it should have absorbed, blaming
    every motor. The close now lowers the flag itself, and writes nothing doing so."""

    class Glitch(FakeArm):
        def connect(self, calibrate: bool = True) -> None:
            super().connect(calibrate)
            if self.calls.count(("connect", False)) == 1:
                self.bus.port_handler.is_using = True  # raised for a packet, never lowered
                raise OSError("WriteFile failed (a synthetic glitch)")

    arm = Glitch()
    transport = LeRobotReal("COM7", robot=arm)
    transport.connect_pause_s = 0.0
    await transport.connect()

    assert arm.is_connected and arm.bus.port_handler.is_using is False
    assert arm.calls.count(("connect", False)) == 2, "the attempt after the glitch failed"
    assert ("bus.disconnect", False) in arm.calls and ("bus.disconnect", True) not in arm.calls
    assert arm.timeline == [] and arm.torque_retries == [], "torque was written between them"
    (note,) = transport.connect_notes
    assert "WriteFile failed" in note, note


async def test_a_port_close_that_ran_out_of_time_waiting_for_the_bus_leaves_its_flag_alone() -> (
    None
):
    """The busy flag was cleared after the close's own call whenever the transport was not
    wedged, and a close that ran out of time still waiting for the bus's lock wedges nothing,
    because its call never started. So the flag of whichever call held the lock was lowered in
    the middle of that call's packet, by a close that had not closed anything. It is cleared in
    the same call as the close now, straight after it, so only under the lock and only once the
    port has shut."""
    arm = FakeArm()
    transport = LeRobotReal("COM7", robot=arm, timeout_s=2.0)
    await transport.connect()
    transport.port_close_deadline_s = 0.1
    seen: list[bool] = []

    def a_packet() -> None:
        arm.bus.port_handler.is_using = True  # raised for its packet, as the servo SDK does
        time.sleep(0.4)
        seen.append(arm.bus.port_handler.is_using)
        arm.bus.port_handler.is_using = False

    other = asyncio.ensure_future(transport._call(a_packet))
    await asyncio.sleep(0.05)  # the packet holds the lock
    await transport._close_port()
    assert ("bus.disconnect", False) not in arm.calls, "the close ran after all"
    await other
    assert seen == [True], "a close that never ran lowered another call's flag mid-packet"
    assert arm.is_connected

    await transport._close_port()  # with the bus free, it closes and lowers the flag
    assert ("bus.disconnect", False) in arm.calls and not arm.is_connected
    assert arm.bus.port_handler.is_using is False


async def test_a_handshake_that_found_no_motor_names_no_joint_and_says_cables_and_power() -> None:
    """With the servo supply off, which is how an arm is after the power cut a session ends
    on, the handshake lists every motor as missing, and the first id listed was named: every
    retry note and the refusal sent the person to one joint's cable, and no longer said a word
    about power. A handshake that found none of the arm's motors names no joint now, and the
    refusal says to check the arm's cables and power. Where the bus's motor table is not known,
    the message's own list of every motor expected stands in for it. One motor that answered
    is not the whole arm gone, and the first one missing is still named."""
    arm, _camera, transport = _rewired()
    everyone = sorted(REWIRED.values())
    arm.handshake_errors = [_motor_check_failed(missing=everyone)] * CONNECT_ATTEMPTS
    with pytest.raises(TransportError) as raised:
        await LeRobotAdapter(transport).connect()
    why = str(raised.value)
    assert why.startswith(f"lerobot real: connect failed {CONNECT_ATTEMPTS} times: "), why
    assert ", the last on" not in why and "cable and connectors" not in why, why
    assert "Check the arm's cables and power, and that nothing else has COM7 open" in why, why
    assert SPLIT_TORQUE not in why, "the handshake writes nothing"
    assert len(transport.connect_notes) == CONNECT_ATTEMPTS - 1
    for k, note in enumerate(transport.connect_notes, start=1):
        assert note.startswith(f"connect attempt {k} of {CONNECT_ATTEMPTS} failed: "), note

    table = {joint: SimpleNamespace(id=n) for joint, n in REWIRED.items()}
    said = str(_motor_check_failed(missing=everyone))
    for motors in (table, None):
        assert motor_in_error(said, motors) is None, motors
        assert motor_in_error(" ".join(said.split()), motors) is None, motors
    joint, first = min(REWIRED.items(), key=lambda kv: kv[1])
    all_but_one = str(_motor_check_failed(missing=everyone[:-1]))
    assert motor_in_error(all_but_one, table) == (f"{joint} (id {first})", joint)


class ChecksItsCalibration(FakeArm):
    """`SOFollower.connect` in upstream's order with the calibration check in its place: the
    port, the handshake, then `not self.is_calibrated and calibrate` (so_follower.py line 99,
    evaluated whatever `calibrate` says), then `configure()`. `calibration_errors` are raised
    from inside the check's reads, and `configure_errors` from inside `configure()`, from a
    calibration read it makes itself when `configure_reads` says so."""

    def __init__(self) -> None:
        super().__init__()
        self.calibration_errors: list[BaseException] = []
        self.configure_errors: list[BaseException] = []
        self.configure_reads = False

    def read_calibration(self) -> dict[str, Any]:
        if self.calibration_errors:
            raise self.calibration_errors.pop(0)
        return dict(self.calibration)

    def connect(self, calibrate: bool = True) -> None:
        self.calls.append(("connect", calibrate))
        self.connected = True
        self.bus._handshake()

        def is_calibrated() -> bool:
            return bool(self.read_calibration())

        if not is_calibrated() and calibrate:
            raise AssertionError("quackd never asks for a calibration")

        def configure() -> None:
            if not self.configure_errors:
                return
            if self.configure_reads:
                self.calibration_errors.append(self.configure_errors.pop(0))
                self.read_calibration()
            raise self.configure_errors.pop(0)

        configure()


@pytest.mark.parametrize(
    ("where", "torque_said"),
    [
        pytest.param("calibration check", False, id="a read in the calibration check"),
        pytest.param("configure", True, id="a read inside configure()"),
        pytest.param("both", True, id="a calibration read made inside configure()"),
    ],
)
async def test_a_read_lost_in_the_calibration_check_before_configure_wrote_no_torque(
    where: str, torque_said: bool
) -> None:
    """LeRobot's connect checks the calibration between the handshake and `configure()`,
    reading each motor's limits and offset with no retry, and writes nothing there. quackd
    placed every failure after the handshake in `configure()`, so a read lost in that check
    told the person to keep a hand under an arm nothing had written to. A failure placed in the
    check is a read now, like one in the handshake; one placed under `configure()`, or one that
    cannot be placed, still earns the warning, and so does a calibration read made from inside
    `configure()`, which would come after its torque-off. Synthetic ids and registers."""
    arm = ChecksItsCalibration()
    n = arm.bus.motors[sorted(JOINTS)[2]].id
    lost = ConnectionError(f"Failed to read 'Homing_Offset' on id_={n} after 1 tries. {NO_STATUS}")
    if where == "calibration check":
        arm.calibration_errors = [lost] * CONNECT_ATTEMPTS
    else:
        arm.configure_errors = [lost] * CONNECT_ATTEMPTS
        arm.configure_reads = where == "both"
    transport = LeRobotReal("COM7", robot=arm)
    transport.connect_pause_s = 0.0
    with pytest.raises(TransportError) as raised:
        await transport.connect()
    why = str(raised.value)
    assert "Failed to read 'Homing_Offset'" in why, why
    assert (SPLIT_TORQUE in why) is torque_said, why
    assert arm.calls.count(("connect", False)) == CONNECT_ATTEMPTS
    assert arm.timeline == [] and not arm.is_connected


class StoppedDuring(FakeArm):
    """An arm whose connect fails on a lost write, and during whose first attempt `stop` runs:
    a person asking for a stop, which is all the kill switch's first press does
    (`KillSwitch._fire`), then or a moment later."""

    def __init__(self, stop: Any) -> None:
        super().__init__()
        self.stop = stop

    def connect(self, calibrate: bool = True) -> None:
        if self.calls.count(("connect", False)) == 0:
            self.stop()
        super().connect(calibrate)


@pytest.mark.parametrize(
    ("when", "pause"),
    [("attempt", 0.0), ("attempt", 0.3), ("pause", 0.6)],
    ids=["during the attempt, no pause", "during the attempt", "during the pause"],
)
async def test_a_stop_asked_for_while_a_connect_fails_ends_the_attempts(
    when: str, pause: float
) -> None:
    """A Ctrl-C while the first attempt was failing set the run's abort flag and cancelled
    nothing, so the second and third attempts still went out, each a `configure()` that
    switches torque off every motor and on again, and one that connected went on into the
    start of the run. The connect is told how to hear a stop now, looks for one once an
    attempt has failed, before it says it will try again, and throughout the pause, and refuses
    on the spot: one attempt, the port closed without a write, the camera let go of, and the
    stop named as why."""
    asked = threading.Event()
    arm = StoppedDuring(asked.set if when == "attempt" else threading.Timer(0.15, asked.set).start)
    n = arm.bus.motors[sorted(JOINTS)[-1]].id
    arm.connect_errors = [_lost_write("Lock", n, 1, NO_STATUS)] * CONNECT_ATTEMPTS
    camera = FakeCamera()
    transport = LeRobotReal(
        "COM7", robot=arm, camera=parse_camera_url("opencv://0"), camera_object=camera
    )
    transport.connect_pause_s = pause
    adapter = LeRobotAdapter(transport)
    adapter.set_stop_check(asked.is_set)
    with pytest.raises(TransportError) as raised:
        await adapter.connect()
    why = str(raised.value)
    assert why.startswith(
        f"lerobot real: connect stopped after attempt 1 of {CONNECT_ATTEMPTS}, because a stop "
        "was asked for."
    ), why
    assert f"Failed to write 'Lock' on id_={n}" in why and SPLIT_TORQUE in why, why
    assert "The port was closed without a write to any motor, and connect was not tried" in why
    assert arm.calls == [("connect", False), ("bus.disconnect", False)], arm.calls
    assert arm.timeline == [] and not arm.is_connected
    assert camera.calls == ["connect", "disconnect"]
    # a stop already asked for is not told "connect runs again" first
    assert len(transport.connect_notes) == (0 if when == "attempt" else 1), transport.connect_notes

    # and taken away, a connect nobody stopped tries again as it always did
    again = StoppedDuring(lambda: None)
    again.connect_errors = [_lost_write("Lock", n, 1, NO_STATUS)]
    plain = LeRobotReal("COM7", robot=again)
    plain.connect_pause_s = 0.0
    plain.set_stop_check(None)
    await plain.connect()
    assert again.calls.count(("connect", False)) == 2 and again.is_connected


async def test_a_run_whose_kill_switch_was_pressed_during_a_failing_connect_connects_no_more(
    tmp_path: Any,
) -> None:
    """The same, through the agent loop, which is where the kill switch is: the loop hands the
    arm its abort flag before it connects, so a press during the first failing attempt is one
    connect call, no rest move and no second attempt, and the run is refused as stopped."""
    from quackd.agent.loop import AgentLoop, RunConfig
    from quackd.agent.providers.fake import FakeProvider

    holder: dict[str, Any] = {}
    running = asyncio.get_running_loop()
    # from the connect's own thread, the way the kill switch's handler reaches the loop
    arm = StoppedDuring(lambda: running.call_soon_threadsafe(holder["loop"].executor.abort.set))
    n = arm.bus.motors[sorted(JOINTS)[0]].id
    arm.connect_errors = [_lost_write("Torque_Enable", n, 0, CORRUPT)] * CONNECT_ATTEMPTS
    rest = {j: arm.positions[j] for j in JOINTS if j != "gripper"}
    transport = LeRobotReal("COM7", robot=arm, rest_pose=rest | {"elbow_flex": 30.0})
    transport.connect_pause_s = 0.0
    loop = AgentLoop(
        RunConfig(
            duck=ARM_DUCK,
            provider=FakeProvider(script=[]),
            transport=LeRobotAdapter(transport),
            runs_dir=tmp_path,
        )
    )
    holder["loop"] = loop
    try:
        with pytest.raises(TransportError, match="connect stopped after attempt 1"):
            await loop.run()
    finally:
        loop.transcript.close()  # the run that would have closed it never started
    assert arm.calls.count(("connect", False)) == 1, arm.calls
    assert arm.actions == [], "the arm was driven after the stop"
    assert transport._stop_check is None, "the loop left its abort flag on the arm"


async def test_the_release_and_the_hold_give_each_torque_write_upstream_s_own_retries() -> None:
    """LeRobot tries each torque write once unless told otherwise, and on the bench the bus
    lost packets in exactly those writes, at connect. The same writes are the hand-off: one
    lost there releases the motors before it and not the ones after, or holds half an arm in
    somebody's hand. Upstream's own disconnect gives its torque-off retries
    (`up.BUS_DISCONNECT`), and the release and the hold now ask for the same count, through a
    partial, because `_call` forwards positional arguments only."""
    arm, transport = _handover_arm()
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    assert (await adapter.let_go()).how == "released"
    arm.positions.update({"elbow_flex": 12.0})
    assert (await adapter.take_hold()).how == "held"
    assert TORQUE_RETRIES > 0
    assert arm.torque_retries == [
        ("disable_torque", TORQUE_RETRIES),
        ("enable_torque", TORQUE_RETRIES),
    ], arm.torque_retries


async def test_a_wedged_call_made_through_a_partial_is_named_by_its_function() -> None:
    """The torque calls now go through `functools.partial`, and a wedge names the call that
    has not come back. A partial has no `__name__`, so without looking through it the reason a
    person reads would be a repr of a bound method instead of the call's name."""
    release = threading.Event()

    def enable_torque(num_retry: int = 0) -> None:
        release.wait(5.0)

    transport = LeRobotReal("COM7", robot=FakeArm(), timeout_s=0.2)
    try:
        with pytest.raises(TimeoutError):
            await transport._call(functools.partial(enable_torque, num_retry=TORQUE_RETRIES))
        assert transport.stop_error is not None
        assert "(enable_torque) has not come back" in transport.stop_error, transport.stop_error
        assert "partial" not in transport.stop_error
    finally:
        release.set()
