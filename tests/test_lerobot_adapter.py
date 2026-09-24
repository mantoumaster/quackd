"""The LeRobot adapter: an arm with joints and a gripper, and nothing a duck has."""

from __future__ import annotations

import asyncio
import functools
import importlib.util
import itertools
import logging
import math
import threading
import time
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
    LET_GO_WHERE_IT_STOOD,
    LIMP_IN_HAND,
    MOVE_HEADROOM_S,
    MOVE_JOINTS_TIMEOUT_S,
    MOVE_MAX_S,
    MOVE_MIN_S,
    MOVE_SETTLE_S,
    RAMP_DECIMALS,
    REST_MAX_S,
    REST_MIN_S,
    STALL_TICKS,
    TICK_S,
    TOL_DEG,
    TORQUE_LEFT_ON,
    MoveJointsParams,
    at_rest,
    lerobot_verbs,
    move_budget_s,
    ramp_target,
    reachable_rest_goal,
    rest_budget_s,
    rest_goal,
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
    adapter = LeRobotAdapter(LeRobotMock())
    manifest = await adapter.connect()
    ex = _executor(adapter, manifest)
    out = await ex.run_verb("move_joints", {"positions": {"shoulder_pan": 150}})
    assert not out.ok and "calibrated range" in out.summary and "-100..100" in out.summary
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

    @property
    def is_connected(self) -> bool:
        """`up.BUS_IS_CONNECTED`: the port's own flag."""
        return self.arm.connected

    def disconnect(self, disable_torque: bool = True) -> None:
        """`MotorsBus.disconnect`: torque off first only when asked, then the port shut.

        It refuses a port that is not open, as upstream's `check_if_not_connected` does, which
        is the call a retry makes after an attempt that never got the port open."""
        self.arm.calls.append(("bus.disconnect", disable_torque))
        if not self.arm.connected:
            raise ConnectionError("FeetechMotorsBus is not connected. Run `.connect()` first.")
        if disable_torque:
            self.disable_torque(num_retry=5)  # upstream's own count (motors_bus.py line 559)
        self.arm.connected = False

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
        (`up.SO_CONNECT_REFUSES_WHILE_OPEN`), then the port, then `configure()`, and nothing
        closes the port again when that last part raises."""
        self.calls.append(("connect", calibrate))
        if self.connected:
            # DeviceAlreadyConnectedError is a ConnectionError upstream
            raise ConnectionError("SOFollower is already connected.")
        if self.port_errors and (refused := self.port_errors.pop(0)) is not None:
            raise refused
        self.connected = True
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
    one the backend checks against, and then that goal was refused as outside it."""
    arm = _spanned()
    adapter, ex, _clock = await _paced(arm)
    published = adapter.manifest.extras["joint_range_deg"]  # type: ignore[union-attr]
    for joint, end in (("shoulder_lift", 0), ("elbow_flex", 1)):
        edge = published[joint][end]
        moved = await ex.run_verb("move_joints", {"positions": {joint: edge}, "duration_s": 1.2})
        assert moved.ok, moved.summary


@pytest.mark.parametrize(
    ("joint", "off", "duration_s"),
    [
        pytest.param("shoulder_pan", TOL_DEG - 1.0, MOVE_MAX_S, id="just inside, the longest"),
        pytest.param("wrist_flex", -(TOL_DEG - 2.5), 4.4, id="the other way"),
        pytest.param("gripper", -(TOL_DEG - 0.5), MOVE_MIN_S, id="the gripper"),
    ],
)
async def test_a_joint_already_where_it_was_asked_to_be_is_arrived_at_once(
    joint: str, off: float, duration_s: float
) -> None:
    """A goal within the tolerance the verb calls arrived is a move already over, whatever
    `duration_s` says: it goes out whole and the verb returns after one tick, as it always did,
    rather than spend the whole time walking a few degrees."""
    arm = _spanned()
    _adapter, ex, clock = await _paced(arm)
    goal = arm.positions[joint] + off
    moved = await ex.run_verb("move_joints", {"positions": {joint: goal}, "duration_s": duration_s})
    assert moved.ok, moved.summary
    assert _sent(arm.actions, joint) == [goal] and clock.t == pytest.approx(TICK_S)


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
    arm = FakeArm()
    adapter = LeRobotAdapter(LeRobotReal("COM5", robot=arm))
    manifest = await adapter.connect()
    ex = _executor(adapter, manifest)
    out = await ex.run_verb("move_joints", {"positions": {"elbow_flex": 170}})
    assert not out.ok and "-100..100" in out.summary and "does not clamp" in out.summary
    # the goal never reached the arm: what did is the hold that every failed verb ends with
    assert all(action.get("elbow_flex.pos") != 170.0 for action in arm.actions)
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
    not reported as undelivered, because it started nothing. That is all it can say: for a
    joint past its travel any goal quackd has written is the limit to the servo, so a joint a
    move had begun lifting out of its fold goes on rising to that limit whatever a stop does,
    and only the power switch stops that stretch. What the stop owes the pilot is the list of
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

    # `--by-hand` releases it there, since the fold is its rest pose, and takes hold again
    # without writing either folded joint a goal: both are past their travel
    released = await adapter.let_go()
    assert released.how == "released", released.reason
    held = await adapter.take_hold()
    assert held.how == "held", held.reason
    assert arm.actions, "the joints inside their travel were written where they were placed"
    for action in arm.actions:
        assert "shoulder_lift.pos" not in action and "elbow_flex.pos" not in action, action

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
    await adapter.close()
    assert arm.torque is False and adapter.close_note is None


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
    # neither the stop nor the hand-off drags it anywhere
    folded = LeRobotMock(rest_pose=pose)
    folded.joints.update(pose)
    held = LeRobotAdapter(folded)
    await held.connect()
    assert (await held.go_to_rest()).how == "already"
    await folded.stop()
    released = await held.let_go()
    assert released.how == "released", released.reason
    assert (await held.take_hold()).how == "held"
    assert folded.actions[-1].keys().isdisjoint({"shoulder_lift", "elbow_flex"}), folded.actions
    assert {j: folded.joints[j] for j in ("shoulder_lift", "elbow_flex")} == {
        j: pose[j] for j in ("shoulder_lift", "elbow_flex")
    }
    await held.close()
    assert folded.torque is False and folded.close_note is None, "a fold is let go of"


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

PLACED_PAST_BY = 17.0
"""How far past the default `FakeArm`'s floor the hand-placed shoulder sits. Any amount would
do; what matters is that it is past the travel, not how far."""

HAND_PLACED = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -DEFAULT_TRAVEL_DEG / 2 - PLACED_PAST_BY,
    "elbow_flex": 40.0,
    "wrist_flex": 15.0,
    "wrist_roll": 0.0,
    "gripper": 35.0,
}
"""Where a person left the arm, with the gripper closed on something.

`shoulder_lift` is outside the travel the calibration recorded on purpose. A hand-placed arm
easily is, the bench arm's own fold was, and the servo clamps any goal to its travel: a goal
for that joint, clipped or not, is a goal at the limit, and writing it drags the arm out of
the pose the person spent the wait setting. So that joint is written no goal at all."""

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
    the person holding a limp arm that it is holding itself up, and they let go of it."""
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
    assert "limp and in your hands" in (adapter.close_note or ""), adapter.close_note


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

    held = await adapter.take_hold()
    assert held.how == "held" and held.ok
    assert held.reason == "holding the pose you set"
    assert arm.timeline == ["disable_torque", "send", "enable_torque", "send"], arm.timeline
    # every joint where it was placed, the gripper included, except the one placed past its
    # travel: the servo would clamp that goal to the limit, so it is written none, clipped or
    # otherwise, before torque or after
    goal = {
        f"{joint}.pos": value for joint, value in HAND_PLACED.items() if joint != "shoulder_lift"
    }
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
    assert arm.torque is True
    assert transport._in_hand is False, "torque is on, so the arm is not hanging off a hand"
    await adapter.close()
    note = adapter.close_note or ""
    assert "torque was left on" in note and "it will not fall" in note
    assert "limp and in your hands" not in note, note


async def test_a_joint_placed_past_its_travel_that_moved_is_named_with_the_travel() -> None:
    """The skip avoids writing a goal the servo would clamp, and that is all it does. What a
    servo does with the goal it already has when torque comes back is not documented anywhere
    quackd can read, so a joint placed past its travel can still move, and the read-back is
    what finds out. When it is that joint, the person is told which, that it was past its
    calibrated travel and where that ends, in this arm's own numbers, and by how much it
    moved. Here it is a joint placed past its ceiling, on a calibration of several spans."""
    arm = _spanned()
    arm.positions.update({joint: 0.0 for joint in SPANS})
    transport = LeRobotReal("COM5", robot=arm, rest_pose={joint: 0.0 for joint in SPANS})
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    assert (await adapter.let_go()).how == "released"
    placed = _past(arm, "wrist_flex", 9.0)
    arm.positions["wrist_flex"] = placed
    arm.slips = {"wrist_flex": -(TOL_DEG + 6.0)}  # it drops as the servos take the weight

    refused = await adapter.take_hold()
    assert refused.how == "refused" and arm.torque is True
    assert all("wrist_flex.pos" not in action for action in arm.actions), arm.actions
    ceiling = transport.joint_range_deg["wrist_flex"][1]
    reason = refused.reason
    assert f"the arm moved as torque came on (wrist_flex by {TOL_DEG + 6.0:.0f} degrees)" in reason
    assert f"wrist_flex was placed at {placed:.0f}, past the {round(ceiling, 1):g}" in reason
    assert "calibrated travel" in reason, reason


async def test_a_stop_while_the_arm_is_in_a_hand_picks_it_up_before_it_sends_anything() -> None:
    """Every teardown begins with a stop, so this is what a Ctrl-C during the wait reaches. A
    stop is "stay where you are", and a goal written to a limp servo stops nothing: the arm is
    energised where the hand has it, and the rest move that follows can then put it down."""
    arm, transport = _handover_arm()
    adapter = LeRobotAdapter(transport)
    await adapter.connect()
    assert (await adapter.let_go()).how == "released"
    arm.positions.update(HAND_PLACED)

    await transport.stop()
    assert arm.timeline == ["disable_torque", "send", "enable_torque", "send", "send"]
    assert arm.torque is True and transport._in_hand is False
    # the hold is the body joints and not the gripper, and not the joint placed past its
    # travel either: a goal for that one is the limit, and the servo would haul it there
    body = {f"{joint}.pos" for joint in JOINTS if joint not in ("gripper", "shoulder_lift")}
    assert set(arm.actions[-1]) == body, "the hold re-sent the gripper or the placed shoulder"
    assert all("shoulder_lift.pos" not in action for action in arm.actions), arm.actions
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

    await adapter.close()
    note = adapter.close_note
    assert note is not None
    assert "the arm is limp and in your hands" in note
    assert "put it down before you let go of it" in note
    assert "never taken hold of again" in note, note
    assert "torque was left on" not in note and "could not keep torque on" not in note
    assert ("disconnect",) in arm.calls, "the arm was never let go of"


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
    assert (transport.close_note or "").startswith(LIMP_IN_HAND.split("(")[0])
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
    assert "keep hold of the arm" in held.reason
    assert transport._in_hand is True


async def test_an_arm_closed_on_while_still_in_a_hand_keeps_whatever_torque_it_has() -> None:
    """`close()` on an arm in somebody's hand used to return before the seam that keeps
    torque, so it always disconnected on LeRobot's drop-torque default.

    That is a no-op on an arm that is genuinely limp, which is the state the branch was
    written for. It is not a no-op on the state the blocker above now produces: an arm whose
    torque could not be read back, which may be energised and is certainly not at its fold.
    Dropping it there drops the arm, and keeping it costs nothing either way."""
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
    assert LIMP_IN_HAND.split("(")[0] in (transport.close_note or ""), transport.close_note


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
    assert f"Check {joint}'s cable and connectors, and that nothing else has {port} open" in why
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
    names no joint, because a guess sends somebody to the wrong cable."""
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
    on whether any attempt opened the port: one that never did wrote to no motor, and one that
    did, even once, may have left the motors in two states."""
    arm = FakeArm()
    port_says = ConnectionError(
        "\nCould not connect on port 'COM3'. Make sure you are using the correct port."
        "\nTry running `lerobot-find-port`\n"
    )
    nameless = RuntimeError("the servo table did not match what the bus found")
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
