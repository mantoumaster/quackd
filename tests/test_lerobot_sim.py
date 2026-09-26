"""The arm simulator's model and world, on the stand-in arm: no network and nothing fetched.

`quackd_lerobot/sim/model.py` sets an arm in quackd's scene and maps LeRobot's units onto it;
`sim/world.py` steps it, holds or drops its joints and keeps the truth about the table. Every
test here runs on the primitives-only stand-in, which is what CI's physics job has, except the
one marked `so101_model`, which needs the maker's model already fetched and skips without it.

No number here comes off an arm. Ranges come from the loaded model, travel from calibrations
built in the test out of the model's own ranges, and motor ids from upstream's bus table.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quackd_lerobot import REACH, lerobot_manifest
from quackd_lerobot import upstream_api as lr
from quackd_lerobot.real import ENCODER_TICKS, joint_ranges
from quackd_lerobot.sim import standin
from quackd_lerobot.sim import upstream_api as up
from quackd_lerobot.sim.model import (
    CUBE,
    FIXED_PAD,
    MOUNTS,
    MOVING_PAD,
    OBJECT_CONDIM,
    PAD_CONDIM,
    PAD_SOLREF,
    PALM_PAD,
    PLACE_FAR,
    PLACE_NEAR,
    PLACE_SPREAD_DEG,
    VIEW_ASPECT,
    VIEW_HFOV_DEG,
    ArmModel,
    CalibrationError,
    GripperMap,
    JointMap,
    ModelError,
    calibration_path,
    generic_calibration,
    load,
    read_calibration,
)
from quackd_lerobot.sim.world import LIFT_MIN_M, ROOM_TEMPERATURE_C, ArmWorld, WorldError
from quackd_lerobot.verbs import GRIPPER_CLOSED, GRIPPER_OPEN, JOINTS, TOL_DEG

mujoco = pytest.importorskip("mujoco")

BODY = JOINTS[:-1]
TICK_DEG = 360.0 / (ENCODER_TICKS - 1)
"""One encoder tick in LeRobot's degrees (`upstream_api.DEGREES_FORMULA`)."""


@pytest.fixture(scope="module")
def mjcf() -> str:
    return standin.mjcf()


def _arm(mjcf: str, seed: int = 0) -> ArmModel:
    return load(mjcf, seed=seed)


def _deg(world: ArmWorld, joint: str) -> float:
    return world.arm.joints[joint].to_lerobot(world.position(joint))


# ── the stand-in and the scene ──────────────────────────────────────────────────────────


def test_the_stand_in_has_lerobots_joints_and_the_manifests_ranges(mjcf: str) -> None:
    arm = _arm(mjcf)
    model = arm.model
    limit = lerobot_manifest("mujoco").limits["joint_deg"]
    assert list(arm.joints) == list(JOINTS)
    for name in JOINTS:
        assert model.joint(name).id >= 0 and model.actuator(name).id >= 0
    for name in BODY:
        joint = arm.joints[name]
        assert isinstance(joint, JointMap)
        assert joint.stops == pytest.approx((-limit, limit))
    assert isinstance(arm.joints[JOINTS[-1]], GripperMap)
    assert model.body(up.WRIST_CAMERA_BODY).id >= 0
    assert {model.camera(i).name for i in range(model.ncam)} == set(MOUNTS)
    assert int(model.cam_bodyid[model.camera("wrist").id]) == model.body(up.WRIST_CAMERA_BODY).id
    assert model.npair == 1  # the fingers are a body and its parent, and collide only by it


def test_the_stand_in_reaches_as_far_as_the_datasheet_says(mjcf: str) -> None:
    """From the shoulder_lift axis up to the fingertips, with the arm standing straight."""
    arm = _arm(mjcf)
    model, data = arm.model, mujoco.MjData(arm.model)
    mujoco.mj_kinematics(model, data)
    shoulder = data.xanchor[model.joint(JOINTS[1]).id][2]
    fixed = model.geom(FIXED_PAD).id
    tip = data.geom_xpos[fixed][2] + model.geom_size[fixed][2]
    assert tip - shoulder == pytest.approx(float(REACH.value))


def test_the_scene_is_laid_out_from_the_model(mjcf: str) -> None:
    arm = _arm(mjcf)
    model = arm.model
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC and model.opt.impratio > 1
    assert model.opt.noslip_iterations > 0
    table = arm.table
    top = model.geom_pos[table][2] + model.geom_size[table][2]
    assert top == pytest.approx(arm.workspace.table_top)
    # the views are laid out for the lens quackd's detector assumes by default
    from quackd.perception.color_blob import DEFAULT_FOV_DEG

    assert VIEW_HFOV_DEG == DEFAULT_FOV_DEG


def _in_view(model: Any, data: Any, camera: str, point: Any) -> tuple[float, float] | None:
    """Where a point in the world falls across the camera's frame, from -1 to 1 either way, at
    the shape the views are laid out for, or None if it is behind the camera or out of frame.
    MuJoCo's cameras look down their own -z, with x to the right of the frame and y up it."""
    cam = model.camera(camera).id
    x, y, z = data.cam_xmat[cam].reshape(3, 3).T @ (np.asarray(point) - data.cam_xpos[cam])
    if z >= 0:
        return None
    half_high = math.tan(math.radians(model.cam_fovy[cam]) / 2) * -z
    across, up_ = x / (half_high * VIEW_ASPECT), y / half_high
    return (across, up_) if abs(across) <= 1 and abs(up_) <= 1 else None


def test_every_view_sees_what_it_is_for(mjcf: str) -> None:
    """The table views take in every place an object can be laid, and the wrist view looks
    between the fingers, one to either side, with the gripper half open."""
    arm = _arm(mjcf)
    model = arm.model
    data = mujoco.MjData(model)
    lo, hi = model.jnt_range[model.joint(JOINTS[-1]).id]
    data.qpos[arm.gripper.qpos] = (lo + hi) / 2
    mujoco.mj_forward(model, data)
    workspace = arm.workspace
    cx, cy = workspace.center
    for view in ("front", "top"):
        for share in (PLACE_NEAR, PLACE_FAR):
            for bearing in (-PLACE_SPREAD_DEG, 0.0, PLACE_SPREAD_DEG):
                r, a = share * workspace.reach, math.radians(bearing)
                spot = (cx + r * math.cos(a), cy + r * math.sin(a), workspace.table_top)
                assert _in_view(model, data, view, spot) is not None, (view, share, bearing)
    fixed = _in_view(model, data, "wrist", data.geom_xpos[arm.fixed_pad])
    moving = _in_view(model, data, "wrist", data.geom_xpos[arm.moving_pad])
    assert fixed is not None and moving is not None
    assert fixed[0] * moving[0] < 0  # one finger either side of the frame


def test_a_pad_meets_everything_with_its_own_contact_settings(mjcf: str) -> None:
    """MuJoCo mixes two touching geoms' settings unless one outranks the other, so the pads
    outrank everything, and the pair between the fingers carries the same settings."""
    arm = _arm(mjcf)
    model = arm.model
    pads = [model.geom(name).id for name in (FIXED_PAD, MOVING_PAD, PALM_PAD)]
    others = [g for g in range(model.ngeom) if g not in pads]
    for g in pads:
        assert model.geom_condim[g] == PAD_CONDIM
        assert tuple(model.geom_solref[g]) == pytest.approx(PAD_SOLREF)
        assert model.geom_priority[g] > max(model.geom_priority[others])
    assert {int(model.geom_condim[g]) for geoms in arm.object_geoms for g in geoms} == {
        OBJECT_CONDIM
    }
    assert model.pair_dim[0] == PAD_CONDIM
    assert tuple(model.pair_solref[0]) == pytest.approx(PAD_SOLREF)
    # and each setting does what its reason says: rolling friction takes all six dimensions and
    # torsional four, and a pad is stiffer than MuJoCo's default yet above refsafe's floor
    assert OBJECT_CONDIM == 6 and PAD_CONDIM >= 4
    bare = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><geom size='1'/></worldbody></mujoco>"
    )
    assert 2 * model.opt.timestep < PAD_SOLREF[0] < bare.geom_solref[0][0]


def test_the_gripper_gets_the_share_of_its_force_lerobot_writes(mjcf: str) -> None:
    raw = mujoco.MjSpec.from_string(mjcf).compile()
    arm = _arm(mjcf)
    share = lr.GRIPPER_MAX_TORQUE_LIMIT / lr.MAX_TORQUE_LIMIT_FULL
    gripper = arm.joints[JOINTS[-1]].actuator
    assert arm.model.actuator_forcerange[gripper] == pytest.approx(
        raw.actuator_forcerange[gripper] * share
    )
    for name in BODY:
        a = arm.joints[name].actuator
        assert arm.model.actuator_forcerange[a] == pytest.approx(raw.actuator_forcerange[a])


def test_a_model_without_lerobots_joints_or_fingers_is_refused(mjcf: str) -> None:
    spec = mujoco.MjSpec.from_string(mjcf)
    spec.delete(spec.actuator("wrist_flex"))
    with pytest.raises(ModelError, match="actuator wrist_flex"):
        load(spec.to_xml(), seed=0)
    spec = mujoco.MjSpec.from_string(mjcf)
    spec.delete(spec.geom(PALM_PAD))
    with pytest.raises(ModelError, match="all three"):
        load(spec.to_xml(), seed=0)


# ── the maps ────────────────────────────────────────────────────────────────────────────


def test_the_maps_round_trip_with_lerobots_sign_and_zero(mjcf: str) -> None:
    arm = _arm(mjcf)
    for name in BODY:
        joint = arm.joints[name]
        lo, hi = joint.stops
        for deg in (lo, lo / 3, 0.0, hi / 2, hi):
            q = joint.to_model(deg)
            assert q == pytest.approx(math.radians(deg))  # JOINT_SIGN +1, JOINT_ZERO 0
            assert joint.to_lerobot(q) == pytest.approx(deg)
    gripper = arm.gripper
    for value in (GRIPPER_CLOSED, 12.5, 50.0, 99.0, GRIPPER_OPEN):
        assert gripper.to_lerobot(gripper.to_model(value)) == pytest.approx(value)
    assert gripper.to_model(GRIPPER_CLOSED) == gripper.closed
    assert gripper.to_model(GRIPPER_OPEN) == gripper.open
    # both ways bounded to 0..100, as LeRobot bounds a RANGE_0_100 motor
    assert gripper.to_model(GRIPPER_OPEN + 20) == gripper.open
    assert gripper.to_model(GRIPPER_CLOSED - 20) == gripper.closed
    beyond = gripper.open + (gripper.open - gripper.closed)
    assert gripper.to_lerobot(beyond) == GRIPPER_OPEN


def test_the_grippers_closed_end_is_found_from_the_fingers(mjcf: str) -> None:
    arm = _arm(mjcf)
    gripper = arm.gripper
    assert gripper.closed == gripper.lo and gripper.open == gripper.hi
    # the same hand with its hinge turned the other way round closes at the top of its range
    spec = mujoco.MjSpec.from_string(mjcf)
    joint, actuator = spec.joint(JOINTS[-1]), spec.actuator(JOINTS[-1])
    joint.axis = [-x for x in joint.axis]
    joint.range = [-joint.range[1], -joint.range[0]]
    actuator.ctrlrange = [-actuator.ctrlrange[1], -actuator.ctrlrange[0]]
    mirrored = load(spec.to_xml(), seed=0).gripper
    assert mirrored.closed == mirrored.hi and mirrored.open == mirrored.lo
    assert mirrored.closed == pytest.approx(-gripper.closed)


# ── calibration ─────────────────────────────────────────────────────────────────────────


def test_the_generic_arm_gives_the_real_backend_the_models_travel(mjcf: str) -> None:
    arm = _arm(mjcf)
    calibration = generic_calibration(arm)
    ranges = joint_ranges(calibration)
    for name in BODY:
        lo, hi = arm.joints[name].stops
        widest = min(-lo, hi)
        got_lo, got_hi = ranges[name]
        assert got_lo == -got_hi
        assert widest - TICK_DEG < got_hi <= widest  # within a tick, never past a stop
    assert ranges[JOINTS[-1]] == (GRIPPER_CLOSED, GRIPPER_OPEN)
    for name, cal in calibration.items():
        assert cal.id == lr.SO_MOTOR_IDS[name]
        assert 0 <= cal.range_min < cal.range_max <= ENCODER_TICKS - 1
        assert cal.drive_mode == 0 and cal.homing_offset == 0


def _synthetic(arm: ArmModel, share: float = 0.5) -> dict[str, dict[str, int]]:
    """A calibration file's contents: each joint's travel a share of the model's, off centre
    the way a recorded one is, with ids that are not in the bus table's order."""
    ids = list(reversed(list(lr.SO_MOTOR_IDS.values())))
    per_deg = (ENCODER_TICKS - 1) / 360.0
    out = {}
    for offset, (name, joint) in enumerate(arm.joints.items(), start=1):
        lo, hi = joint.stops
        half = math.floor(share * min(-lo, hi) * per_deg) if name != JOINTS[-1] else offset * 10
        middle = ENCODER_TICKS // 2 + offset
        out[name] = {
            "id": ids[offset - 1],
            "drive_mode": 0,
            "homing_offset": -offset,
            "range_min": middle - half,
            "range_max": middle + half,
        }
    return out


def test_a_calibration_file_is_read_as_lerobot_reads_it(mjcf: str, tmp_path: Path) -> None:
    arm = _arm(mjcf)
    raw = _synthetic(arm)
    path = tmp_path / "arm.json"
    path.write_text(json.dumps(raw, indent=4), encoding="utf-8")
    calibration = read_calibration(path)
    assert list(calibration) == list(JOINTS)
    for name, cal in calibration.items():
        assert cal.__dict__ == raw[name]
    assert [c.id for c in calibration.values()] != sorted(c.id for c in calibration.values())
    ranges = joint_ranges(calibration)
    for name in BODY:
        span = raw[name]["range_max"] - raw[name]["range_min"]
        assert ranges[name] == pytest.approx((-span / 2 * TICK_DEG, span / 2 * TICK_DEG))


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda raw: raw.pop("wrist_roll"), "not the SO-101's six motors"),
        (lambda raw: raw.update(elbow=raw["elbow_flex"]), "not the SO-101's six motors"),
        (lambda raw: raw["gripper"].pop("range_max"), "not LeRobot's"),
        (lambda raw: raw["gripper"].update(offset=0), "not LeRobot's"),
        (lambda raw: raw["shoulder_lift"].update(range_min=1.5), "whole number"),
        (lambda raw: raw["shoulder_lift"].update(range_max=4000.0), "whole number"),
        (lambda raw: raw["elbow_flex"].update(homing_offset="three"), "whole number"),
        (
            lambda raw: raw["wrist_flex"].update(range_min=raw["wrist_flex"]["range_max"]),
            "equal to its range_max",
        ),
    ],
)
def test_a_calibration_file_lerobot_would_refuse_is_refused(
    mjcf: str, tmp_path: Path, change: Any, match: str
) -> None:
    raw = _synthetic(_arm(mjcf))
    change(raw)
    path = tmp_path / "arm.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(CalibrationError, match=match):
        read_calibration(path)


def test_each_field_is_decoded_as_draccus_decodes_it_for_lerobot(mjcf: str, tmp_path: Path) -> None:
    """`int()` of anything but a float (`upstream_api.CALIBRATION_INTS`): a twin loads the file
    its arm loads, however the file spells a whole number. A null draccus lets through, and the
    simulator, which builds the travel from every field, refuses it."""
    raw = _synthetic(_arm(mjcf))
    wanted = {k: raw["wrist_flex"][k] for k in ("id", "range_min", "homing_offset")}
    raw["gripper"]["drive_mode"] = True
    raw["wrist_flex"].update(
        id=str(wanted["id"]),
        range_min=f" {wanted['range_min']} ",
        homing_offset=str(wanted["homing_offset"]),
    )
    path = tmp_path / "arm.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    calibration = read_calibration(path)
    assert calibration["gripper"].drive_mode == 1
    assert {k: getattr(calibration["wrist_flex"], k) for k in wanted} == wanted
    raw["gripper"]["id"] = None
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(CalibrationError, match="gives gripper a value for id that is not"):
        read_calibration(path)


def test_a_missing_or_unreadable_calibration_file_says_how_to_go_on(tmp_path: Path) -> None:
    with pytest.raises(CalibrationError, match=r"no calibration file .*generic arm"):
        read_calibration(tmp_path / "nope.json")
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(CalibrationError, match="not a calibration file LeRobot could read"):
        read_calibration(tmp_path / "bad.json")
    (tmp_path / "list.json").write_text("[]", encoding="utf-8")
    with pytest.raises(CalibrationError, match="names no motors"):
        read_calibration(tmp_path / "list.json")


def test_the_calibration_search_is_lerobots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each variable wins over every one after it, and one set to nothing still counts."""
    tail = Path(lr.ROBOTS_SUBDIR) / lr.SO_FOLLOWER_NAME / "arm-7.json"
    for name in (lr.CALIBRATION_ENV, lr.LEROBOT_HOME_ENV, lr.HF_HOME_ENV, lr.XDG_CACHE_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    cache = tmp_path / "home" / ".cache"
    hub = Path(lr.HF_SUBDIR) / lr.LEROBOT_SUBDIR / lr.CALIBRATION_SUBDIR
    assert calibration_path("arm-7") == cache / hub / tail

    monkeypatch.setenv(lr.XDG_CACHE_ENV, str(tmp_path / "xdg"))
    assert calibration_path("arm-7") == tmp_path / "xdg" / hub / tail

    monkeypatch.setenv(lr.HF_HOME_ENV, str(tmp_path / "hf"))
    below_hf = Path(lr.LEROBOT_SUBDIR) / lr.CALIBRATION_SUBDIR
    assert calibration_path("arm-7") == tmp_path / "hf" / below_hf / tail

    monkeypatch.setenv(lr.LEROBOT_HOME_ENV, str(tmp_path / "lerobot"))
    assert calibration_path("arm-7") == tmp_path / "lerobot" / lr.CALIBRATION_SUBDIR / tail

    monkeypatch.setenv(lr.CALIBRATION_ENV, str(tmp_path / "calibration"))
    assert calibration_path("arm-7") == tmp_path / "calibration" / tail

    monkeypatch.setenv(lr.CALIBRATION_ENV, "")
    assert calibration_path("arm-7") == tail  # os.getenv hands back the empty string


# ── the world ───────────────────────────────────────────────────────────────────────────


def test_a_rest_pose_past_a_stop_starts_at_the_stop_and_says_so(mjcf: str) -> None:
    arm = _arm(mjcf)
    _, pan_hi = arm.joints["shoulder_pan"].stops
    elbow_lo, _ = arm.joints["elbow_flex"].stops
    _, wrist_hi = arm.joints["wrist_flex"].stops
    rest = {
        "shoulder_pan": pan_hi + 10.0,  # past the ceiling
        "elbow_flex": elbow_lo - 10.0,  # past the floor
        "wrist_flex": wrist_hi / 2,
        "gripper": GRIPPER_OPEN + 5.0,
    }
    world = ArmWorld(arm, rest_pose=rest)
    assert _deg(world, "shoulder_pan") == pytest.approx(pan_hi)
    assert _deg(world, "elbow_flex") == pytest.approx(elbow_lo)
    assert _deg(world, "wrist_flex") == pytest.approx(wrist_hi / 2)
    assert _deg(world, "gripper") == pytest.approx(GRIPPER_OPEN)
    assert _deg(world, "shoulder_lift") == 0.0  # not in the pose: the model's zero
    notes = " ".join(world.notes)
    assert len(world.notes) == 3
    assert "shoulder_pan" in notes and "elbow_flex" in notes and "gripper" in notes
    assert "wrist_flex" not in notes
    assert all(world.goal(j) == pytest.approx(world.position(j)) for j in JOINTS)


def test_every_joint_follows_its_goal(mjcf: str) -> None:
    world = ArmWorld(_arm(mjcf))
    goals = {}
    for name in BODY:
        goals[name] = (
            world.arm.joints[name].stops[1] / 9
        )  # a few tens of degrees, whatever the manifest's limit
        world.set_goal(name, world.arm.joints[name].to_model(goals[name]))
    world.set_goal(JOINTS[-1], world.arm.gripper.to_model(GRIPPER_OPEN))
    world.step(2.0)
    for name, goal in goals.items():
        assert _deg(world, name) == pytest.approx(goal, abs=TOL_DEG), name
    assert _deg(world, JOINTS[-1]) == pytest.approx(GRIPPER_OPEN, abs=TOL_DEG)
    assert world.t == pytest.approx(2.0)


def test_a_limp_joint_falls_and_torque_drives_it_back_to_its_stored_goal(mjcf: str) -> None:
    """The worst case of TORQUE_ENABLE_HOLDS_PRESENT: the goal register outlives the torque."""
    world = ArmWorld(_arm(mjcf))
    joint = world.arm.joints["shoulder_lift"]
    tilt = joint.stops[1] / 6  # tipped forward, so gravity pulls it further over
    world.set_goal("shoulder_lift", joint.to_model(tilt))
    world.step(2.0)
    held = _deg(world, "shoulder_lift")
    assert held == pytest.approx(tilt, abs=TOL_DEG)

    world.set_torque("shoulder_lift", False)
    assert not world.torque("shoulder_lift") and world.torque("elbow_flex")
    world.step(1.0)
    fallen = _deg(world, "shoulder_lift")
    assert fallen > held + TOL_DEG  # it fell the way gravity pulls it
    assert world.goal("shoulder_lift") == pytest.approx(joint.to_model(tilt))
    assert _deg(world, "elbow_flex") == pytest.approx(0.0, abs=TOL_DEG)  # the rest still hold

    world.set_torque("shoulder_lift", True)
    world.step(2.0)
    assert _deg(world, "shoulder_lift") == pytest.approx(tilt, abs=TOL_DEG)


def test_a_joint_let_go_in_one_world_holds_in_the_next(mjcf: str) -> None:
    """LeRobot lets every joint go at a disconnect by default, and a reconnect builds a new
    world from the same loaded model, whose joints must hold and not only say they do."""
    arm = _arm(mjcf)
    joint = arm.joints["shoulder_lift"]
    tilt = joint.stops[1] / 6
    gains = arm.model.actuator_gainprm.copy()
    first = ArmWorld(arm, rest_pose={"shoulder_lift": tilt})
    first.set_torque("shoulder_lift", False)
    first.close()
    assert (arm.model.actuator_gainprm == gains).all()
    again = ArmWorld(arm, rest_pose={"shoulder_lift": tilt})
    assert again.torque("shoulder_lift")
    again.step(2.0)
    assert _deg(again, "shoulder_lift") == pytest.approx(tilt, abs=TOL_DEG)


def test_every_motor_reads_the_room(mjcf: str) -> None:
    world = ArmWorld(_arm(mjcf))
    assert {world.temperature(j) for j in JOINTS} == {ROOM_TEMPERATURE_C}
    with pytest.raises(ValueError, match="no joint 'elbow'"):
        world.temperature("elbow")


def test_objects_are_laid_out_by_the_seed_within_reach(mjcf: str) -> None:
    def places(world: ArmWorld) -> list[tuple[float, float, float]]:
        return [o.position for o in world.truth().objects.values()]

    first, again, other = (ArmWorld(_arm(mjcf, seed)) for seed in (5, 5, 6))
    assert places(first) == places(again)
    assert places(first) != places(other)
    workspace = first.arm.workspace
    for obj, (x, y, z) in zip(first.arm.objects, places(first), strict=True):
        r = math.dist((x, y), workspace.center)
        assert PLACE_NEAR * workspace.reach <= r <= PLACE_FAR * workspace.reach
        assert z == pytest.approx(workspace.table_top + obj.rest_height, abs=1e-3)
    other.place_objects(5)
    assert places(other) == pytest.approx(places(first), abs=1e-9)


def _pointing_down(arm: ArmModel, reach_share: float, tip_height: float) -> dict[str, float]:
    """The stand-in's joints, in LeRobot's units, with its hand pointing straight down and its
    fingertips `tip_height` above the table, `reach_share` of the reach out: two-link inverse
    kinematics on the stand-in's own proportions."""
    reach = float(REACH.value)
    shares = (standin.UPPER_ARM, standin.FOREARM, standin.WRIST, standin.HAND)
    upper, fore, wrist, hand = (share * reach for share in shares)
    x = reach_share * arm.workspace.reach
    z = tip_height + hand + wrist - standin.PEDESTAL * reach  # the wrist above the shoulder
    elbow = math.acos((x * x + z * z - upper**2 - fore**2) / (2 * upper * fore))
    shoulder = math.atan2(x, z) - math.atan2(fore * math.sin(elbow), upper + fore * math.cos(elbow))
    return {
        "shoulder_lift": math.degrees(shoulder),
        "elbow_flex": math.degrees(elbow),
        "wrist_flex": math.degrees(math.pi - shoulder - elbow),
        "gripper": GRIPPER_OPEN,
    }


def test_the_truth_follows_a_grasp_and_a_latch_keeps_what_it_saw(mjcf: str) -> None:
    arm = _arm(mjcf)
    half = CUBE.size[0]
    world = ArmWorld(arm, rest_pose=_pointing_down(arm, (PLACE_NEAR + PLACE_FAR) / 2, half / 2))
    with world.locked() as (model, data):
        fixed = arm.fixed_pad
        face = data.geom_xpos[fixed] + data.geom_xmat[fixed].reshape(3, 3) @ [
            model.geom_size[fixed][0],
            0.0,
            0.0,
        ]
        inward = data.geom_xpos[arm.moving_pad] - face
    inward[2] = 0.0
    inward /= float(math.hypot(*inward))
    cube = face + inward * (half + half / 5)
    world.set_object_pose(
        CUBE.name, (float(cube[0]), float(cube[1]), arm.workspace.table_top + half)
    )
    world.step(0.5)
    assert world.truth().objects[CUBE.name].on_table

    world.set_goal(JOINTS[-1], arm.gripper.to_model(GRIPPER_CLOSED))
    world.step(1.0)
    held = world.truth().objects[CUBE.name]
    assert held.pinched and held.touching and not held.lifted

    lift = arm.joints["shoulder_lift"]
    world.set_goal("shoulder_lift", world.goal("shoulder_lift") - lift.to_model(lift.stops[1] / 9))
    world.step(1.5)
    up_ = world.truth().objects[CUBE.name]
    assert up_.lifted and not up_.on_table and up_.pinched

    latched = world.latch("stop")
    world.set_goal(JOINTS[-1], arm.gripper.to_model(GRIPPER_OPEN))
    world.step(1.0)
    now = world.truth()
    assert not now.objects[CUBE.name].lifted and now.objects[CUBE.name].on_table
    peaks = now.peaks[CUBE.name]
    assert peaks.lifted and peaks.pinched and peaks.touched
    assert peaks.lift_m >= up_.lift_m - 1e-9
    assert peaks.moved_m >= up_.moved_m - 1e-9 and peaks.moved_m >= LIFT_MIN_M
    assert world.latched("stop") is latched and latched.objects[CUBE.name].lifted
    assert world.latched("rest") is None


def _pressed_into(world: ArmWorld, pad: int, other: int) -> Any:
    """Where the cube's centre goes to sink a millimetre into the face of `pad` that looks
    toward `other`: the inside of one finger."""
    with world.locked() as (model, data):
        centre = data.geom_xpos[pad].copy()
        rot = data.geom_xmat[pad].reshape(3, 3)
        toward = rot.T @ (data.geom_xpos[other] - centre)
        axis = int(np.argmax(np.abs(toward)))
        normal = rot[:, axis] * np.sign(toward[axis])
        return centre + normal * (model.geom_size[pad][axis] + CUBE.size[0] - 0.001)


def test_one_finger_and_then_the_other_is_not_a_pinch(mjcf: str) -> None:
    """A pinch is both fingers on an object in the same instant. The cube touches the inside
    of one finger, then of the other, and at its peak it was touched but never pinched. Each
    touch is at the pad's own contact settings."""
    arm = _arm(mjcf)
    world = ArmWorld(arm, rest_pose={JOINTS[-1]: GRIPPER_OPEN})
    cube = [obj.name for obj in arm.objects].index(CUBE.name)
    for pad, other in ((arm.fixed_pad, arm.moving_pad), (arm.moving_pad, arm.fixed_pad)):
        where = _pressed_into(world, pad, other)
        with world.locked() as (model, data):
            # not set_object_pose, which is a scene's setup and starts the peaks over
            q, v = arm.object_qpos[cube], arm.object_dofs[cube]
            data.qpos[q : q + 7] = [*where, 1.0, 0.0, 0.0, 0.0]
            data.qvel[v : v + 6] = 0.0
            mujoco.mj_forward(model, data)
        world.step(world.timestep)
        now = world.truth().objects[CUBE.name]
        assert now.touching and not now.pinched
        with world.locked() as (model, data):
            touches = [
                c
                for c in data.contact[: data.ncon]
                if pad in (c.geom1, c.geom2) and {c.geom1, c.geom2} & arm.object_geoms[cube]
            ]
            assert touches
            for c in touches:
                assert c.dim == PAD_CONDIM and tuple(c.solref) == pytest.approx(PAD_SOLREF)
    peaks = world.truth().peaks[CUBE.name]
    assert peaks.touched and not peaks.pinched


def test_a_closed_world_refuses_everything_but_its_latches(mjcf: str) -> None:
    world = ArmWorld(_arm(mjcf))
    world.latch("stop")
    world.close()
    assert world.closed
    for call in (
        lambda: world.step(0.1),
        lambda: world.positions(),
        lambda: world.set_goal("gripper", 0.0),
        lambda: world.truth(),
        lambda: world.t,
    ):
        with pytest.raises(WorldError, match="closed"):
            call()
    assert world.latched("stop") is not None


def test_physics_that_diverge_stop_the_world_rather_than_teleport_the_arm(mjcf: str) -> None:
    """MuJoCo answers a state gone bad by resetting it to the model's zero and carrying on,
    which would move the arm in one step to a pose it never drove to."""
    world = ArmWorld(_arm(mjcf))
    world.step(0.1)
    with world.locked() as (_, data):
        data.qvel[world.arm.joints["elbow_flex"].dof] = math.nan
    with pytest.raises(WorldError, match="diverged"):
        world.step(0.1)


def test_a_step_shorter_than_the_physics_is_refused(mjcf: str) -> None:
    world = ArmWorld(_arm(mjcf))
    with pytest.raises(ValueError, match="advance nothing"):
        world.step(world.timestep / 4)
    assert world.step(world.timestep * 2.4) == pytest.approx(world.timestep * 2)


# ── the real model ──────────────────────────────────────────────────────────────────────


@pytest.mark.so101_model
def test_the_so101_model_loads_with_its_fingers_cut_to_pads() -> None:
    from quackd_lerobot.sim.assets import AssetError, ensure_so101

    try:
        so101 = ensure_so101(offline=True)
    except AssetError as e:
        pytest.skip(f"the SO-101's model is not fetched: {e}")
    raw = mujoco.MjSpec.from_file(str(so101.model_path)).compile()
    arm = load(so101.model_path, seed=0)
    model = arm.model
    for pad in (FIXED_PAD, MOVING_PAD, PALM_PAD):
        g = model.geom(pad).id
        assert model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and (model.geom_size[g] > 0).all()
    for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.mesh(model.geom_dataid[g]).name
            if mesh in (up.FIXED_FINGER_MESH, up.MOVING_JAW_MESH):
                assert not model.geom_contype[g] and not model.geom_conaffinity[g], mesh
    gripper = arm.gripper
    share = lr.GRIPPER_MAX_TORQUE_LIMIT / lr.MAX_TORQUE_LIMIT_FULL
    assert model.actuator_forcerange[gripper.actuator] == pytest.approx(
        raw.actuator_forcerange[gripper.actuator] * share
    )
    data = mujoco.MjData(model)
    apart = {}
    for end in (gripper.closed, gripper.open):
        data.qpos[:] = model.qpos0
        data.qpos[gripper.qpos] = end
        mujoco.mj_forward(model, data)
        apart[end] = mujoco.mj_geomDistance(model, data, arm.fixed_pad, arm.moving_pad, 1.0, None)
    assert 0 < apart[gripper.closed] < apart[gripper.open]
    world = ArmWorld(arm)
    start = world.positions()
    world.step(1.0)
    for name in BODY:
        assert _deg(world, name) == pytest.approx(
            arm.joints[name].to_lerobot(start[name]), abs=TOL_DEG
        )
    assert all(not o.touching for o in world.truth().objects.values())
