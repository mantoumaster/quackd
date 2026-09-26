"""The arm's physics world: one model and its state, behind one lock.

What the simulated follower reads and writes lives here, in the model's own units: joint
positions and goals in radians, torque on or off per joint, and a temperature per motor. The
maps between those and what LeRobot reads are `model.py`'s. So is the scene; this module steps
it, starts the arm in its pose, and keeps the truth about the objects on the table.

Every access takes the lock. `mj_step` releases the GIL, and the follower's reads and writes
run on the transport's worker thread while the clock steps the world on the event loop's, so
without it a reader could see a state halfway through a step, or write a goal into one. A step
runs whole physics substeps under it and never a part of one.

Torque off makes that joint's actuator limp and leaves its goal register as it was, so turning
torque back on drives the joint to the goal it held before, wherever the joint has fallen to
since. Whether a real servo does that is the unverified TORQUE_ENABLE_HOLDS_PRESENT, and the
simulator gives the worst case, which is the one quackd's take-hold is written against. Limp
means the actuator's gains are zeroed in the model, so each world steps its own copy of the
loaded model: a joint one world let go of is never limp in the next world built from it.

Ground truth is what a rehearsal is checked against and never what the pilot sees: where each
object is, whether the gripper touches it, holds it between both fingers, or has it up off the
table, now and at its peak over the run. `latch()` snapshots it, so a check reads the scene as
it was when the run ended, before any teardown moved the arm through it.

Nothing here renders, and nothing here imports `mujoco` when the module is imported.
"""

from __future__ import annotations

import contextlib
import copy
import math
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from quackd.transport.base import TransportError
from quackd_lerobot.sim.model import LABEL, ArmModel, GripperMap, object_poses
from quackd_lerobot.verbs import JOINTS

ROOM_TEMPERATURE_C = 25
"""What every simulated motor's temperature register reads. The model has no heat in it, so
each servo reads the room it would stand in when idle, well under the heat quackd refuses to
move a joint at, and a fault that needs a hot servo has to say so itself. One byte, as the
register is."""
LIFT_MIN_M = 0.01
"""How far an object's centre must rise above where it was laid, off the table and touching
the gripper, before it counts as lifted. Closing on an object can nudge it up a few
millimetres, and that is not a lift."""

_TABLE, _GRIPPER, _FIXED, _MOVING = 1, 2, 4, 8
"""What a geom is to an object touching it, as bits: the table, anything on the gripper, and
the fixed and the moving finger's pads."""
_PINCH = _FIXED | _MOVING


def _pinched(flags: int) -> bool:
    """Both fingers on it in the same physics step. One finger now and the other a moment
    later is two pushes, not a pinch."""
    return flags & _PINCH == _PINCH


def _lifted(flags: int, lift: float) -> bool:
    return not flags & _TABLE and bool(flags & _GRIPPER) and lift >= LIFT_MIN_M


class WorldError(TransportError):
    """The simulated world cannot go on: its physics diverged, or it has been closed."""


@dataclass(frozen=True)
class ObjectTruth:
    """One object, now."""

    position: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    moved_m: float
    """How far its centre is from where it was laid."""
    lift_m: float
    """How far its centre is above where it was laid; negative if it has fallen off the
    table."""
    on_table: bool
    touching: bool
    """Anything on the gripper touches it."""
    pinched: bool
    """Both fingers touch it."""
    lifted: bool
    """Off the table, touching the gripper, and up at least `LIFT_MIN_M`."""


@dataclass(frozen=True)
class ObjectPeaks:
    """One object, at its most over the run so far."""

    moved_m: float
    lift_m: float
    """The highest it has been while it counted as lifted, and 0 if it never has."""
    touched: bool
    pinched: bool
    """Both fingers touched it at once, at some instant: a finger on each side in turn does
    not count."""
    lifted: bool


@dataclass(frozen=True)
class Truth:
    """What was so at sim time `t`: every object now, and at its peak so far."""

    t: float
    objects: Mapping[str, ObjectTruth]
    peaks: Mapping[str, ObjectPeaks]


class ArmWorld:
    """An arm in its scene, stepping.

    Built from a loaded model, which it copies and never changes, and a rest pose in LeRobot's
    units: the arm starts at that pose held under torque, limited only by the model's hard
    stops, and at the model's zero where no pose is given. A stop that truncates the pose leaves
    a sentence in `notes`. Objects start where the model's seed laid them."""

    def __init__(self, arm: ArmModel, *, rest_pose: Mapping[str, float] | None = None) -> None:
        import mujoco

        self.arm = arm
        self.notes: list[str] = []
        self._mj = mujoco
        self._lock = threading.Lock()
        # Torque off edits the gains in place, so this world's model is its own: a disconnect
        # that let a joint go, as LeRobot's does by default, must not leave the next world
        # built from the same loaded model with a joint that says it holds and falls.
        self._model: Any = copy.copy(arm.model)
        self._data: Any = mujoco.MjData(self._model)
        self._closed = False
        self._gain = np.array(self._model.actuator_gainprm, copy=True)
        self._bias = np.array(self._model.actuator_biasprm, copy=True)
        self._torque = dict.fromkeys(JOINTS, True)
        self._goal: dict[str, float] = {}
        self._diverged = self._warnings()

        for name, q in self._start(rest_pose).items():
            joint = arm.joints[name]
            self._data.qpos[joint.qpos] = q
            self._data.ctrl[joint.actuator] = q
            self._goal[name] = q
        # The truth is folded in after every physics step, so it is kept in plain lists: for a
        # handful of objects and contacts, numpy's cost per call would be most of the work.
        n = len(arm.objects)
        self._role = [0] * arm.model.ngeom
        self._role[arm.table] |= _TABLE
        for g in arm.gripper_geoms:
            self._role[g] |= _GRIPPER
        self._role[arm.fixed_pad] |= _FIXED
        self._role[arm.moving_pad] |= _MOVING
        self._owner = [-1] * arm.model.ngeom
        for i, geoms in enumerate(arm.object_geoms):
            for g in geoms:
                self._owner[g] = i
        self._bodies = list(arm.object_bodies)
        self._flags = [0] * n
        self._laid = [(0.0, 0.0, 0.0)] * n
        self._peak_moved = [0.0] * n
        self._peak_lift = [0.0] * n
        self._ever_touched = [False] * n
        self._ever_pinched = [False] * n
        self._ever_lifted = [False] * n
        self._latched: dict[str, Truth] = {}
        mujoco.mj_forward(self._model, self._data)
        self._lay()

    # ── the starting pose ───────────────────────────────────────────────────────────────

    def _start(self, rest_pose: Mapping[str, float] | None) -> dict[str, float]:
        """Each joint's starting angle in radians: the rest pose where it names the joint,
        limited only by the model's stops, and the model's zero, within its stops, elsewhere."""
        pose = dict(rest_pose or {})
        start: dict[str, float] = {}
        for name, joint in self.arm.joints.items():
            if name not in pose:
                start[name] = min(max(0.0, joint.lo), joint.hi)
                continue
            value = float(pose[name])
            if not math.isfinite(value):
                raise ValueError(f"{LABEL} the rest pose's {name} is {value}, not a number.")
            lo, hi = joint.stops
            if not lo <= value <= hi:
                reachable = min(max(value, lo), hi)
                unit = "" if isinstance(joint, GripperMap) else " degrees"
                self.notes.append(
                    f"the rest pose puts {name} at {value:.1f}{unit}, past the model's stop at "
                    f"{reachable:.1f}{unit}, so the simulated arm starts at the stop instead"
                )
            start[name] = min(max(joint.to_model(value), joint.lo), joint.hi)
        return start

    # ── time ────────────────────────────────────────────────────────────────────────────

    @property
    def t(self) -> float:
        """Sim time in seconds."""
        with self._lock:
            self._open()
            return float(self._data.time)

    @property
    def timestep(self) -> float:
        """One physics step, in seconds: the finest a step can advance by."""
        return float(self.arm.model.opt.timestep)

    def step(self, dt: float) -> float:
        """Advance by `dt` rounded to whole physics steps, at least one, and say by how much.

        Raises `WorldError` if MuJoCo had to reset a state gone bad, rather than carry on in the
        state it reset to, which is the model's zero with the arm teleported there. A reset
        counts its warning and puts the clock back to zero, and either one gives it away."""
        timestep = self.timestep
        n = round(dt / timestep)
        if n < 1:
            raise ValueError(
                f"{LABEL} a step of {dt} s is shorter than half the model's timestep of "
                f"{timestep} s, so it would advance nothing."
            )
        with self._lock:
            self._open()
            due = float(self._data.time) + (n - 0.5) * timestep
            for _ in range(n):
                self._mj.mj_step(self._model, self._data)
                self._track()
            seen = self._warnings()
            if seen != self._diverged or self._data.time < due:
                self._diverged = seen
                raise WorldError(
                    f"{LABEL} the physics diverged at t={self._data.time:.3f} s and MuJoCo reset "
                    "the arm, so the run cannot go on. Something drove the model past what it "
                    "can take; report it with the task file."
                )
        return n * timestep

    def _warnings(self) -> tuple[int, ...]:
        w = self._mj.mjtWarning
        kinds = (w.mjWARN_BADQACC, w.mjWARN_BADQPOS, w.mjWARN_BADQVEL)
        return tuple(int(self._data.warning[k].number) for k in kinds)

    # ── the joints ──────────────────────────────────────────────────────────────────────

    def _joint(self, name: str) -> Any:
        try:
            return self.arm.joints[name]
        except KeyError:
            raise ValueError(
                f"{LABEL} there is no joint {name!r}; the arm has {', '.join(JOINTS)}."
            ) from None

    def positions(self) -> dict[str, float]:
        """Every joint's angle in radians, in LeRobot's order."""
        with self._lock:
            self._open()
            return {n: float(self._data.qpos[j.qpos]) for n, j in self.arm.joints.items()}

    def position(self, name: str) -> float:
        joint = self._joint(name)
        with self._lock:
            self._open()
            return float(self._data.qpos[joint.qpos])

    def goal(self, name: str) -> float:
        """The joint's goal register in radians, kept whether its torque is on or off."""
        self._joint(name)
        with self._lock:
            self._open()
            return self._goal[name]

    def set_goal(self, name: str, q: float) -> None:
        """Write the joint's goal register. A limp joint keeps it and goes there when its
        torque comes back on."""
        joint = self._joint(name)
        q = float(q)
        if not math.isfinite(q):
            raise ValueError(f"{LABEL} a goal for {name} must be a number, not {q}.")
        with self._lock:
            self._open()
            self._goal[name] = q
            self._data.ctrl[joint.actuator] = q

    def torque(self, name: str) -> bool:
        self._joint(name)
        with self._lock:
            self._open()
            return self._torque[name]

    def set_torque(self, name: str, on: bool) -> None:
        """Torque on or off for one joint. Off zeroes its actuator's gains, so it pushes on
        nothing and the joint moves only as gravity and whatever touches it move it; on puts
        them back, and the joint drives to its goal register."""
        joint = self._joint(name)
        a = joint.actuator
        with self._lock:
            self._open()
            self._torque[name] = bool(on)
            if on:
                self._model.actuator_gainprm[a] = self._gain[a]
                self._model.actuator_biasprm[a] = self._bias[a]
            else:
                self._model.actuator_gainprm[a] = 0.0
                self._model.actuator_biasprm[a] = 0.0

    def temperature(self, name: str) -> int:
        self._joint(name)
        with self._lock:
            self._open()
            return ROOM_TEMPERATURE_C

    # ── the objects ─────────────────────────────────────────────────────────────────────

    def place_objects(self, seed: int) -> None:
        """Lay every object out again as `seed` lays it, at rest, and start its truth over."""
        poses = object_poses(self.arm.workspace, self.arm.objects, seed)
        with self._lock:
            self._open()
            for i, (x, y, z, yaw) in enumerate(poses):
                self._put(i, (x, y, z), (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))
            self._mj.mj_forward(self._model, self._data)
            self._lay()

    def set_object_pose(
        self,
        name: str,
        position: tuple[float, float, float],
        quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        """Put one object somewhere, at rest, and make that where it was laid: a scene's setup,
        such as a cube between the jaws, rather than something the arm did."""
        names = [obj.name for obj in self.arm.objects]
        if name not in names:
            raise ValueError(
                f"{LABEL} there is no object {name!r} on the table; there is "
                f"{', '.join(names) or 'nothing'}."
            )
        with self._lock:
            self._open()
            i = names.index(name)
            self._put(i, position, quat)
            self._mj.mj_forward(self._model, self._data)
            self._lay(only=i)

    def _put(
        self,
        i: int,
        position: tuple[float, float, float],
        quat: tuple[float, float, float, float],
    ) -> None:
        q, v = self.arm.object_qpos[i], self.arm.object_dofs[i]
        self._data.qpos[q : q + 3] = position
        self._data.qpos[q + 3 : q + 7] = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
        self._data.qvel[v : v + 6] = 0.0

    def _lay(self, only: int | None = None) -> None:
        """Take the objects' present places as where they were laid, and start their peaks
        over. Under the lock, after a forward pass."""
        for i in range(len(self._bodies)) if only is None else (only,):
            x, y, z = (float(v) for v in self._data.xpos[self._bodies[i]])
            self._laid[i] = (x, y, z)
            self._peak_moved[i] = 0.0
            self._peak_lift[i] = 0.0
            self._ever_touched[i] = False
            self._ever_pinched[i] = False
            self._ever_lifted[i] = False
        self._track()

    # ── the truth ───────────────────────────────────────────────────────────────────────

    def _track(self) -> None:
        """Read the objects' state off the data and fold it into the peaks. Under the lock,
        after every physics step."""
        data = self._data
        flags = [0] * len(self._bodies)
        if ncon := data.ncon:
            owner, role = self._owner, self._role
            for a, b in data.contact.geom[:ncon].tolist():
                if (i := owner[a]) >= 0:
                    flags[i] |= role[b]
                if (j := owner[b]) >= 0:
                    flags[j] |= role[a]
        self._flags = flags
        for i, (moved, lift) in enumerate(self._offsets(data.xpos[self._bodies].tolist())):
            if moved > self._peak_moved[i]:
                self._peak_moved[i] = moved
            if flags[i] & _GRIPPER:
                self._ever_touched[i] = True
            if _pinched(flags[i]):
                self._ever_pinched[i] = True
            if _lifted(flags[i], lift):
                self._ever_lifted[i] = True
                if lift > self._peak_lift[i]:
                    self._peak_lift[i] = lift

    def _offsets(self, positions: list[list[float]]) -> list[tuple[float, float]]:
        """How far each object is from where it was laid, and how far above it."""
        return [
            (math.dist(p, laid), p[2] - laid[2])
            for p, laid in zip(positions, self._laid, strict=True)
        ]

    def truth(self) -> Truth:
        """Every object now, and at its peak over the run so far."""
        with self._lock:
            self._open()
            return self._truth()

    def _truth(self) -> Truth:
        data = self._data
        positions = data.xpos[self._bodies].tolist()
        quats = data.xquat[self._bodies].tolist()
        objects: dict[str, ObjectTruth] = {}
        peaks: dict[str, ObjectPeaks] = {}
        for i, (obj, (moved, lift)) in enumerate(
            zip(self.arm.objects, self._offsets(positions), strict=True)
        ):
            f = self._flags[i]
            x, y, z = positions[i]
            qw, qx, qy, qz = quats[i]
            objects[obj.name] = ObjectTruth(
                position=(x, y, z),
                quat=(qw, qx, qy, qz),
                moved_m=moved,
                lift_m=lift,
                on_table=bool(f & _TABLE),
                touching=bool(f & _GRIPPER),
                pinched=_pinched(f),
                lifted=_lifted(f, lift),
            )
            peaks[obj.name] = ObjectPeaks(
                moved_m=self._peak_moved[i],
                lift_m=self._peak_lift[i],
                touched=self._ever_touched[i],
                pinched=self._ever_pinched[i],
                lifted=self._ever_lifted[i],
            )
        return Truth(
            t=float(data.time), objects=MappingProxyType(objects), peaks=MappingProxyType(peaks)
        )

    def latch(self, label: str) -> Truth:
        """Snapshot the truth under `label`, replacing any earlier snapshot of that name.

        Called on the way into a stop or a rest move, before either moves anything, so that a
        check reads the scene as the run left it and not as the teardown did."""
        with self._lock:
            self._open()
            snapshot = self._truth()
            self._latched[label] = snapshot
            return snapshot

    def latched(self, label: str) -> Truth | None:
        with self._lock:
            return self._latched.get(label)

    # ── the rest ────────────────────────────────────────────────────────────────────────

    @contextlib.contextmanager
    def locked(self) -> Iterator[tuple[Any, Any]]:
        """The model and its data, held still, for whatever reads them wholesale: a renderer."""
        with self._lock:
            self._open()
            yield self._model, self._data

    def _open(self) -> None:
        if self._closed:
            raise WorldError(f"{LABEL} the simulator is closed.")

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Let go of the model's state. Anything asked of the world afterwards is refused; its
        latched truth stays readable, because that is what a check reads after the close."""
        with self._lock:
            self._closed = True
            self._data = None
            self._model = None
