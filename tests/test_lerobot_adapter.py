"""The LeRobot adapter: an arm with joints and a gripper, and nothing a duck has."""

from __future__ import annotations

import asyncio
import importlib.util
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
from quackd_lerobot import JOINTS, LeRobotAdapter, lerobot_manifest, make
from quackd_lerobot.mock import GRIP_ON_OBJECT, REST, LeRobotMock
from quackd_lerobot.real import (
    ENCODER_TICKS,
    MAX_STEP_DEG,
    STEP_ENV,
    LeRobotReal,
    check_port,
    load_policy,
    parse_camera_url,
    parse_camera_urls,
    step_from_env,
)
from quackd_lerobot.verbs import (
    REST_MAX_S,
    REST_MIN_S,
    TOL_DEG,
    at_rest,
    rest_budget_s,
    rest_goal,
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
    assert "mass" in sheet.unknown() and "reach" in sheet.unknown()
    assert sheet.payload_kg is not None and sheet.payload_kg.value == 0.5
    assert any("0.8 to 2.5 kg" in note for note in sheet.notes)


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


class FakeCalibration:
    """The two fields quackd reads off a `MotorCalibration` (`up.MOTOR_CALIBRATION`)."""

    def __init__(self, travel_deg: float) -> None:
        span = travel_deg * (ENCODER_TICKS - 1) / 360.0
        middle = (ENCODER_TICKS - 1) / 2
        self.range_min = int(middle - span / 2)
        self.range_max = int(middle + span / 2)


class FakeBus:
    """The registers `get_observation()` does not read (`up.STS3215_REGISTERS`)."""

    def __init__(self, arm: FakeArm) -> None:
        self.arm = arm

    def sync_read(
        self, data_name: str, motors: Any = None, *, normalize: bool = True, num_retry: int = 0
    ) -> dict[str, int]:
        self.arm.reads.append((data_name, normalize, num_retry))
        if self.arm.bus_error:
            raise RuntimeError("Incorrect status packet!")
        if data_name == "Torque_Enable":
            return dict.fromkeys(JOINTS, 1 if self.arm.torque else 0)
        if data_name == "Present_Temperature":
            return {joint: int(self.arm.temperature.get(joint, 30)) for joint in JOINTS}
        raise KeyError(data_name)

    def enable_torque(self) -> None:
        """`up.BUS_ENABLE_TORQUE`, which is how `take_hold` picks the arm back up.

        A joint named in `slips` sags as torque arrives and stays sagged: the goal goes out
        again immediately afterwards, and a servo that could not hold the pose the first time
        does not reach it on the second ask either."""
        if self.arm.bus_error:
            raise RuntimeError("Incorrect status packet!")
        self.arm.timeline.append("enable_torque")
        if self.arm.torque_refuses:
            return
        self.arm.torque = True
        for joint, gap in self.arm.slips.items():
            self.arm.positions[joint] += gap
            self.arm.stuck.add(joint)

    def disable_torque(self) -> None:
        """`up.BUS_DISABLE_TORQUE`: the one call in quackd that de-energises a robot.

        It goes through the bus and never through the `Robot`, which is why this fake has no
        `disable_torque` of its own for a test to be accidentally green against."""
        if self.arm.bus_error:
            raise RuntimeError("Incorrect status packet!")
        self.arm.timeline.append("disable_torque")
        self.arm.torque = False
        if self.arm.bus_error_after_release:
            self.arm.bus_error = True


class FakeArm:
    """The slice of a LeRobot `Robot` the real backend touches, verified names only.

    It caps every step the way `max_relative_target` does, so a goal takes as many sends as
    a real one would, and its gripper stops on an object instead of closing."""

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
        self.calibration = {joint: FakeCalibration(200.0) for joint in JOINTS}
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
        self.calls.append(("connect", calibrate))
        self.connected = True

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
            sent[key] = capped
            # a limp servo takes the goal into its register and does not move to it. That is
            # why `take_hold` writes the pose again once torque is back, and why a stop over
            # an arm somebody is holding has to pick it up before it sends anything.
            if joint in self.stuck or not self.torque:
                continue
            if joint == "gripper" and self.object_in_jaws:
                capped = max(capped, GRIP_ON_OBJECT)
            self.positions[joint] = capped
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
        await ex.run_verb("move_joints", {"positions": {"shoulder_pan": 10}, "duration_s": 2})
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


async def test_a_move_takes_as_many_sends_as_the_step_cap_needs() -> None:
    """One send_action moves a joint at most the cap, so the verb re-sends and watches."""
    arm = FakeArm(step=5.0)
    adapter = LeRobotAdapter(LeRobotReal("COM5", robot=arm))
    manifest = await adapter.connect()
    ex = _executor(adapter, manifest)
    moved = await ex.run_verb("move_joints", {"positions": {"shoulder_pan": 30}, "duration_s": 8})
    assert moved.ok, moved.summary
    assert len(arm.actions) >= 5, arm.actions
    assert arm.positions["shoulder_pan"] == pytest.approx(30.0, abs=5.0)
    assert moved.data["goal"] == {"shoulder_pan": 30.0}


async def test_a_joint_that_stops_moving_is_a_failure_and_not_a_success() -> None:
    """Upstream reports nothing about whether a goal was reached: an arm against an
    obstacle and an arm that arrived look identical unless somebody compares them."""
    arm = FakeArm(stuck=("shoulder_lift",))
    adapter = LeRobotAdapter(LeRobotReal("COM5", robot=arm))
    manifest = await adapter.connect()
    ex = _executor(adapter, manifest)
    stalled = await ex.run_verb(
        "move_joints", {"positions": {"shoulder_lift": 40}, "duration_s": 5}
    )
    assert not stalled.ok
    assert "shoulder_lift is at 0 with a goal of 40" in stalled.summary
    assert "stopped moving" in stalled.summary


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
    assert (await ex.run_verb("move_joints", {"positions": {"shoulder_pan": 10}})).ok


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


async def test_a_rest_pose_outside_the_calibrated_range_is_still_driven_to() -> None:
    """The bench arm's folded pose read shoulder_lift -113.5 against a calibrated travel of
    plus or minus 84.2 degrees. A pose recorded off the arm is where the arm physically was,
    so the range refusal that guards a pilot's goal would refuse this arm its own resting
    place and leave it standing up with the torque about to drop."""
    arm = FakeArm(step=200.0)
    arm.calibration["shoulder_lift"] = FakeCalibration(168.4)
    transport = LeRobotReal("COM5", robot=arm, rest_pose={"shoulder_lift": -113.5})
    adapter = LeRobotAdapter(transport)
    manifest = await adapter.connect()
    assert manifest.extras["joint_range_deg"]["shoulder_lift"] == [-84.2, 84.2]
    result = await adapter.go_to_rest()
    assert result.how == "arrived", result.reason
    assert {"shoulder_lift.pos": -113.5} in arm.actions, arm.actions
    assert arm.positions["shoulder_lift"] == -113.5
    assert transport._range_clips == 0, "the rest goal was walked back inside the range"


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
    assert (await ex.run_verb("move_joints", {"positions": {"shoulder_pan": 10}})).ok


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

HAND_PLACED = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -113.5,
    "elbow_flex": 40.0,
    "wrist_flex": 15.0,
    "wrist_roll": 0.0,
    "gripper": 35.0,
}
"""Where a person left the arm, with the gripper closed on something.

`shoulder_lift` is outside the travel the calibration recorded on purpose. A hand-placed arm
easily is, the bench arm's own folded pose read -113.5 against a range of plus or minus 84.2,
and a goal walked back inside that range is a goal somewhere the arm is not: writing it would
drag the arm out of the pose the person spent the wait setting."""

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
    goal = {f"{joint}.pos": value for joint, value in HAND_PLACED.items()}
    assert arm.actions == [goal, goal], arm.actions
    assert arm.actions[0]["shoulder_lift.pos"] == -113.5, "the goal was clipped to the range"
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
    body = {f"{joint}.pos" for joint in JOINTS if joint != "gripper"}
    assert set(arm.actions[-1]) == body, "the hold re-sent the gripper and dropped the object"
    assert arm.actions[-1]["shoulder_lift.pos"] == -113.5
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
