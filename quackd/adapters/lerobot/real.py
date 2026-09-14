"""EXPERIMENTAL: a real arm through LeRobot. Verified names, never run on an arm.

Every LeRobot name comes from `upstream_api.py` (ADR-0022). LeRobot is synchronous, so
every call runs in a worker thread under one lock with a deadline, and a call that blows its
deadline wedges the transport rather than letting a second thread onto a half-duplex bus.
`stop` re-sends the present position as the goal (hold); torque is never disabled by quackd
(LeRobot's own `disconnect()` does, by its default, and that is documented).

What this backend refuses to take on faith, because upstream cannot tell it:

- **the arm is still there.** `is_connected` is the serial port's open flag, so the
  heartbeat reads the arm instead (`up.BUS_IS_CONNECTED`).
- **torque is on.** `get_observation()` reads positions only, so torque state and joint
  temperature come off the bus by register (`up.STS3215_REGISTERS`).
- **the goal is reachable.** A degrees goal outside the calibrated range is written as-is
  (`up.DEGREES_NO_CLAMP`), so quackd computes each joint's travel from the calibration file
  and refuses the goal.
- **the arm can be told to jump.** `max_relative_target` is `None` upstream; quackd sets it,
  so one `send_action` moves a joint at most `max_step_deg`.

`pick` runs an injected policy object; building one from a Hub checkpoint (`load_policy`)
uses verified names but has never been exercised (`upstream_api.POLICY_PIPELINE`). LeRobot
is imported inside `connect()` and `load_policy()` only: `quackd[lerobot]` is an extra.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import os
import re
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image

from quackd.adapters.base import AdapterError, AdapterNotInstalled
from quackd.adapters.lerobot import upstream_api as up
from quackd.adapters.lerobot.verbs import GRIPPER_CLOSED, GRIPPER_OPEN, JOINTS
from quackd.transport.base import Ack, DuckState, HeartbeatError, Intent, TransportError

STATUS = "EXPERIMENTAL: LeRobot names verified at a pinned commit, never run against an arm"
POLICY_HZ = 10.0

MAX_STEP_DEG = 5.0
"""How far one `send_action` may move a joint, in degrees. At the 10 Hz a verb re-sends a
goal that is also the top joint speed: 5 degrees a step is 50 degrees a second. The figure
is quackd's own choice for a first run and nothing upstream recommends one for this arm;
upstream's default is no cap at all."""
STEP_ENV = "QUACKD_LEROBOT_MAX_STEP_DEG"

ENCODER_TICKS = 4096
"""An sts3215 turn in encoder counts (`up.STS3215_RESOLUTION`); degrees use it less one."""

HOT_C = 60.0
"""Refuse to move a joint at or above this. The servo's own cut-off is 70 (`up.TEMPERATURE_C`)."""

HOLD_MIN = 8.0
HOLD_MAX = 90.0
"""A gripper told to close and settled strictly inside this band is holding something."""
GRIPPER_SETTLE = 1.0
"""Two readings this close together mean the gripper has stopped moving."""
SETTLE_GAP_S = 0.08
"""How far apart in time those two readings have to be to say anything."""

OUT_OF_RANGE_DEG = 2.0
"""How far past its calibrated travel a joint must read before the state says so."""

PORT_SHAPE = re.compile(r"^(COM\d+|/dev/[\w./-]+)$", re.IGNORECASE)
"""A serial port looks like COM5 or /dev/ttyACM0. Which one is the arm is not our business
(`up.SERIAL_PORT`); a goal pasted into --address by mistake is."""

CAMERA_SCHEME = "opencv"
CAMERA_NAME = "front"
CAMERA_BACKENDS = ("any", "v4l2", "dshow", "avfoundation", "msmf")
CAMERA_ROTATIONS = (0, 90, 180, 270)
CAMERA_KEYS = ("name", "width", "height", "fps", "fourcc", "backend", "rotation", "fov")
CAMERA_CONNECT_S = 15.0
"""Long enough for the open plus upstream's warmup, which reads frames before returning."""


@dataclass(frozen=True)
class CameraSpec:
    """One USB webcam, as `--camera-url` described it.

    quackd owns this camera rather than handing it to the follower. A follower's cameras are
    part of its connected state (`up.SO_CAMERAS_ARE_THE_FOLLOWERS`): `send_action` and
    `disconnect()` are gated on `is_connected`, which is the bus AND every camera, so a
    webcam that came unplugged mid-session would make every move and every hold raise while
    the arm itself was fine. Beside the follower, a dead camera costs you `observe`."""

    url: str
    name: str
    index_or_path: int | str
    width: int | None
    height: int | None
    fps: int | None
    fourcc: str | None
    backend: str
    rotation: int
    fov_deg: float | None


def _camera_int(key: str, raw: str, url: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        raise _camera_refusal(f"{key}={raw!r} is not a whole number", url) from None
    if value <= 0:
        raise _camera_refusal(f"{key}={raw!r} must be above 0", url)
    return value


def _camera_refusal(why: str, url: str) -> AdapterError:
    return AdapterError(
        f"lerobot real: --camera-url {url!r}: {why}. A camera is one USB webcam named by "
        "its OpenCV index, opencv://0, or by a device path, opencv:///dev/video2, with any "
        f"of {', '.join(CAMERA_KEYS)} after a ?. Run `lerobot-find-cameras opencv` to see "
        "which index is which: it saves a frame per camera"
    )


def parse_camera_url(url: str) -> CameraSpec:
    """`opencv://0?width=640&height=480&fps=30&backend=msmf` into a `CameraSpec`.

    Strict, in the shape of the rosbridge adapter's address parser: an unknown scheme, key
    or value is refused with the shape rather than quietly ignored, because the alternative
    is an owner who believes they configured a camera and did not. Nothing here imports
    lerobot, so a bad url is refused before the extra is even looked for.

    Size and rate are left unset by default, which keeps whatever mode the camera already
    has (`up.OPENCV_MODE_DEFAULTS_TO_THE_CAMERA`). Asking for one it cannot do is a refusal
    at connect (`up.OPENCV_MODE_IS_A_DEMAND`), and the webcam in a lab drawer is unknown."""
    parts = urlsplit(url)
    if parts.scheme != CAMERA_SCHEME:
        seen = f"{parts.scheme!r} is not a scheme quackd knows" if parts.scheme else "no scheme"
        raise _camera_refusal(seen, url)
    target = (parts.netloc + parts.path).rstrip("/")
    if not target:
        raise _camera_refusal("no camera index or device path", url)
    index_or_path: int | str = int(target) if target.isdigit() else target
    query = parse_qs(parts.query, keep_blank_values=True)
    if unknown := sorted(set(query) - set(CAMERA_KEYS)):
        raise _camera_refusal(f"unknown {'keys' if len(unknown) > 1 else 'key'} {unknown}", url)

    def one(key: str) -> str | None:
        values = query.get(key)
        return values[-1].strip() if values else None

    name = one("name") or CAMERA_NAME
    if not name.replace("_", "").replace("-", "").isalnum():
        raise _camera_refusal(f"name={name!r} is not a plain name", url)
    fourcc = one("fourcc")
    if fourcc is not None and len(fourcc) != 4:
        raise _camera_refusal(f"fourcc={fourcc!r} must be four characters", url)
    backend = (one("backend") or "any").lower()
    if backend not in CAMERA_BACKENDS:
        raise _camera_refusal(f"backend={backend!r} is not one of {CAMERA_BACKENDS}", url)
    rotation = _camera_int("rotation", one("rotation") or "0", url) if one("rotation") else 0
    if rotation not in CAMERA_ROTATIONS:
        raise _camera_refusal(f"rotation={rotation} is not one of {CAMERA_ROTATIONS}", url)
    fov = one("fov")
    return CameraSpec(
        url=url,
        name=name,
        index_or_path=index_or_path,
        width=_camera_int("width", w, url) if (w := one("width")) else None,
        height=_camera_int("height", h, url) if (h := one("height")) else None,
        fps=_camera_int("fps", f, url) if (f := one("fps")) else None,
        fourcc=fourcc,
        backend=backend,
        rotation=rotation,
        fov_deg=float(_camera_int("fov", fov, url)) if fov else None,
    )


def step_from_env(default: float = MAX_STEP_DEG) -> float:
    """`QUACKD_LEROBOT_MAX_STEP_DEG`, or the default. A bad value is an error, not a shrug:
    silently falling back would give somebody a faster arm than they asked for."""
    raw = os.environ.get(STEP_ENV)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise TransportError(f"{STEP_ENV}={raw!r} is not a number of degrees") from None
    if not 0.0 < value <= 180.0:
        raise TransportError(f"{STEP_ENV}={raw!r} must be above 0 and at most 180")
    return value


def check_port(port: str) -> None:
    """A serial port, or a clear refusal. The shape is all quackd checks: which port is the
    arm is the owner's business (`up.SERIAL_PORT`), but an empty --address, or a robot name
    that never resolved, is worth catching before LeRobot opens something."""
    if not port:
        raise TransportError("lerobot real: --address must be the arm's serial port")
    if not PORT_SHAPE.match(port):
        raise TransportError(
            f"lerobot real: --address {port!r} is not a serial port; it looks like COM5 on "
            "Windows or /dev/ttyACM0 elsewhere"
        )


def joint_ranges(calibration: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Each joint's travel in degrees, from the arm's own calibration file.

    LeRobot records raw encoder ticks and converts with `up.DEGREES_FORMULA`, which centres
    a joint on the middle of its recorded travel. So the reachable range is half the
    recorded span either side of zero. The gripper is a 0..100 range whatever else is."""
    ranges: dict[str, tuple[float, float]] = {}
    for joint in JOINTS:
        cal = calibration.get(joint)
        if cal is None:
            continue
        if joint == "gripper":
            ranges[joint] = (GRIPPER_CLOSED, GRIPPER_OPEN)
            continue
        span = abs(float(cal.range_max) - float(cal.range_min))
        half = span / 2.0 * 360.0 / (ENCODER_TICKS - 1)
        ranges[joint] = (-half, half)
    return ranges


class PolicyLike(Protocol):
    """What `pick` needs from a policy: one observation in, one joint goal out (or None
    when it considers the task done). The `real` backend never builds one on its own."""

    def act(self, observation: dict[str, Any], *, task: str) -> dict[str, float] | None: ...


class LeRobotReal:
    name = "real"
    mobility = "none"

    def __init__(
        self,
        address: str | None = None,
        *,
        robot: Any = None,
        policy: PolicyLike | None = None,
        robot_type: str = up.ROBOT_TYPE_SO101.name,
        robot_id: str = "arm-01",
        timeout_s: float = 1.0,
        max_step_deg: float = MAX_STEP_DEG,
        camera: CameraSpec | None = None,
        camera_object: Any = None,
    ) -> None:
        self.port = address or ""
        self.robot_type = robot_type
        self.robot_id = robot_id
        self.timeout_s = timeout_s
        self.max_step_deg = max_step_deg
        self._robot: Any = robot  # injected in tests; built in connect() otherwise
        self.camera_spec = camera
        self._camera: Any = camera_object  # injected in tests; built in connect() otherwise
        self._policy = policy
        self._lock = asyncio.Lock()
        self._closed = False
        self._wedged: asyncio.Future[Any] | None = None
        self._policy_task: asyncio.Task[None] | None = None
        self._policy_name = "idle"
        self._policy_error: str | None = None
        self._joints: dict[str, float] = {}
        self._torque = True
        self._temperature_c: dict[str, float] = {}
        self._register_error: str | None = None
        self._gripper_goal: float | None = None
        self._gripper_trace: deque[tuple[float, float]] = deque(maxlen=16)
        self._policy_lock = asyncio.Lock()
        self._range_clips = 0
        self.joint_range_deg: dict[str, tuple[float, float]] = {}
        self.calibration_file: str | None = None
        self.camera_keys: tuple[str, ...] = ()
        self.camera_error: str | None = None
        self._frame_size: tuple[int, int] | None = None
        self._frame_at: float | None = None
        self.lerobot_version: str | None = None
        self.stop_error: str | None = None
        self.post_sleep: Callable[[], None] | None = None

    @property
    def camera_available(self) -> bool:
        return bool(self.camera_keys)

    @property
    def policy_available(self) -> bool:
        return self._policy is not None

    @property
    def policy_running(self) -> bool:
        return self._policy_task is not None and not self._policy_task.done()

    # ── plumbing ────────────────────────────────────────────────────────────────────

    async def _call(
        self, fn: Callable[..., Any], *args: Any, deadline_s: float | None = None
    ) -> Any:
        """One LeRobot call at a time (thread safety is UNVERIFIED), each with a deadline.

        A call that blows its deadline is not over: the worker thread is still sitting on a
        half-duplex serial bus waiting for a reply. Releasing the lock and starting another
        would put two talkers on that bus, so the transport stays wedged until the thread
        comes back, and says so instead. The arm holds its last goal meanwhile, which is the
        one thing that needs no rescuing (`up.NO_CLIENT_DEADMAN`)."""
        self._refuse_if_wedged()
        loop = asyncio.get_running_loop()
        pending: asyncio.Future[Any] | None = None
        try:
            async with asyncio.timeout(deadline_s or self.timeout_s):
                async with self._lock:
                    # a caller parked on the lock passed the check above before the call
                    # ahead of it wedged; the lock's release is what woke it, so ask again
                    self._refuse_if_wedged()
                    pending = loop.run_in_executor(None, functools.partial(fn, *args))
                    return await asyncio.shield(pending)
        except (TimeoutError, asyncio.CancelledError):
            # a cancelled verb (Ctrl-C mid-move) leaves its thread on the wire exactly as
            # a timed-out one does, and the stop that follows must not join it there
            if pending is not None and not pending.done():
                self._wedged = pending
                self.stop_error = (
                    f"a LeRobot call ({getattr(fn, '__name__', fn)}) has not come back; the "
                    "serial bus has one owner, so quackd refuses every call until it does"
                )
            raise

    def _refuse_if_wedged(self) -> None:
        if self._wedged is None:
            return
        if not self._wedged.done():
            raise TransportError(self.stop_error or "a LeRobot call has not come back")
        self._wedged = None
        self.stop_error = None

    def _config_kwargs(self) -> dict[str, Any]:
        """Every safety-shaped field of `up.SO_CONFIG`, spelled out.

        Inheriting a default is fine until upstream changes one. `max_relative_target` is the
        field upstream leaves at None, and it has to be a float, not an int
        (`up.SO_ACTION_CLAMP_IS_FLOAT`)."""
        return {
            "port": self.port,
            "id": self.robot_id,
            "use_degrees": True,
            "disable_torque_on_disconnect": True,
            # quackd owns its camera instead: up.SO_CAMERAS_ARE_THE_FOLLOWERS
            "cameras": {},
            "max_relative_target": float(self.max_step_deg),
        }

    def _build_robot(self) -> Any:
        try:
            import lerobot
            from lerobot.robots import make_robot_from_config
            from lerobot.robots.so_follower import SO101FollowerConfig
        except ImportError as e:
            raise AdapterNotInstalled("lerobot", "quackd[lerobot]") from e
        self.lerobot_version = getattr(lerobot, "__version__", None)
        check_port(self.port)
        if self.robot_type != up.ROBOT_TYPE_SO101.name:
            raise TransportError(f"lerobot real: only {up.ROBOT_TYPE_SO101.name} is wired")
        config = SO101FollowerConfig(**self._config_kwargs())
        return make_robot_from_config(config)

    def _build_camera(self) -> Any:
        """The webcam, built by quackd and not by the follower (`up.OPENCV_CAMERA`)."""
        spec = self.camera_spec
        assert spec is not None
        try:
            from lerobot.cameras import ColorMode, Cv2Backends, Cv2Rotation
            from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
        except ImportError as e:  # up.CAMERA_EXPORTS: the config is not in lerobot.cameras
            raise AdapterNotInstalled("lerobot", "quackd[lerobot]") from e
        return OpenCVCamera(
            OpenCVCameraConfig(
                index_or_path=spec.index_or_path,
                fps=spec.fps,
                width=spec.width,
                height=spec.height,
                color_mode=ColorMode.RGB,
                rotation=Cv2Rotation(spec.rotation if spec.rotation != 270 else -90),
                fourcc=spec.fourcc,
                backend=Cv2Backends[spec.backend.upper()],
            )
        )

    # ── protocol ────────────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        self._closed = False
        if self._robot is None:
            self._robot = await asyncio.to_thread(self._build_robot)
        try:
            await self._call(self._robot.connect, False, deadline_s=30.0)  # never calibrate
        except Exception as e:
            raise TransportError(f"lerobot real: connect failed: {e}") from e
        if not bool(self._robot.is_calibrated):
            await self._give_up(
                "lerobot real: the arm is not calibrated; run LeRobot's calibration first "
                "(it is interactive, quackd never triggers it)"
            )
        calibration = dict(getattr(self._robot, "calibration", None) or {})
        if not calibration:
            await self._give_up(
                "lerobot real: the arm reports no calibration file, so nothing knows how far "
                "each joint travels; run LeRobot's calibration first"
            )
        if getattr(self._robot, "bus", None) is None:
            await self._give_up(
                "lerobot real: this robot has no motors bus, so torque and temperature "
                "cannot be read; quackd drives an SO-101 follower and nothing else"
            )
        self.joint_range_deg = joint_ranges(calibration)
        path = getattr(self._robot, "calibration_fpath", None)
        self.calibration_file = str(path) if path else None
        # the follower is built with cameras={}, so its observation_features never name
        # one; the only camera here is the one quackd opened (up.SO_CAMERAS_ARE_THE_FOLLOWERS)
        await self._probe()
        await self._connect_camera()

    async def _connect_camera(self) -> None:
        """A camera the owner asked for and did not get is a refusal, not a warning.

        They named an index on the command line and `doctor` gates its verdict on a frame,
        so failing quietly would leave somebody believing they had eyes. The arm is
        disconnected cleanly first: it has no camera of its own to be missing
        (`up.SO_CAMERAS_ARE_THE_FOLLOWERS`), so it connects again without --camera-url."""
        spec = self.camera_spec
        if spec is None:
            return
        if self._camera is None:
            self._camera = await asyncio.to_thread(self._build_camera)
        try:
            await self._call(self._camera.connect, deadline_s=CAMERA_CONNECT_S)
        except Exception as e:
            await self._give_up(
                f"lerobot real: --camera-url {spec.url!r} did not open: {e}. The arm itself "
                "is fine and connects without --camera-url"
            )
        self.camera_keys = (spec.name,)

    async def _give_up(self, why: str) -> None:
        with contextlib.suppress(Exception):
            await self._call(self._robot.disconnect)
        raise TransportError(why)

    async def close(self) -> None:
        self._closed = True
        await self._cancel_policy()
        if self._camera is not None:
            with contextlib.suppress(Exception):
                await self._call(self._camera.disconnect)  # up.CAMERA_DISCONNECT
        if self._robot is not None:
            with contextlib.suppress(Exception):
                await self._call(self._robot.disconnect, deadline_s=5.0)  # up.SO_DISCONNECT_TORQUE

    # ── reading ─────────────────────────────────────────────────────────────────────

    def _read_all(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None]:
        """Three bus transactions in one worker thread, so nothing interleaves on the wire.

        The positions are the liveness check and are allowed to raise. The two registers are
        not: a corrupt status packet should cost a reading, not the run, so a failure there
        comes back as a note and the last known values stand."""
        obs: dict[str, Any] = self._robot.get_observation()
        torque: dict[str, Any] = {}
        temperature: dict[str, Any] = {}
        error: str | None = None
        try:
            bus = self._robot.bus
            torque = bus.sync_read("Torque_Enable", normalize=False, num_retry=2)
            temperature = bus.sync_read("Present_Temperature", normalize=False, num_retry=2)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        return obs, torque, temperature, error

    async def _probe(self) -> dict[str, Any]:
        obs, torque, temperature, error = await self._call(self._read_all)
        self._joints = self._joints_of(obs)
        gripper = self._joints.get("gripper")
        if gripper is not None:
            self._gripper_trace.append((self.now(), gripper))
        self._register_error = error
        if torque:
            self._torque = all(int(v) == 1 for v in torque.values())
        if temperature:
            self._temperature_c = {str(k): float(v) for k, v in temperature.items()}
        return obs

    async def _observe(self) -> dict[str, Any]:
        """The plain observation for the policy loop. `_probe` is the one that also reads
        registers; a policy ticking at 10 Hz does not need them.

        The camera is added under its own name, which is the dict the follower would have
        built had it owned the camera (`up.SO_CAMERA_KEYS`), so a policy trained against
        that key sees what it expects. A camera failure here ends the pick and says why,
        rather than being swallowed the way `observe`'s is."""
        obs: dict[str, Any] = await self._call(self._robot.get_observation)
        if self._camera is not None and self.camera_spec is not None:
            obs[self.camera_spec.name] = await asyncio.to_thread(self._camera.read_latest)
        return obs

    @staticmethod
    def _joints_of(obs: dict[str, Any]) -> dict[str, float]:
        return {k.removesuffix(".pos"): float(v) for k, v in obs.items() if k.endswith(".pos")}

    async def get_frame(self) -> Image.Image | None:
        """The newest frame, or None and a reason. Never raises.

        `observe` moves nothing, so a camera that has stopped delivering should cost the
        picture and not the run: the agent loop asks for a frame every step and `doctor`
        polls for one, and neither expects an exception. The read does not go through
        `_call`, because it touches no serial bus and a wedge must never be filed as a
        camera fault (`up.CAMERA_READ_LATEST`)."""
        if self._camera is None:
            return None
        try:
            frame = await asyncio.to_thread(self._camera.read_latest)
            image = Image.fromarray(np.asarray(frame))  # up.CAMERA_COLOR_MODE_DEFAULT: RGB
        except Exception as e:
            self.camera_error = f"{type(e).__name__}: {e}"
            return None
        self.camera_error = None
        self._frame_size = image.size
        self._frame_at = self.now()
        return image

    def camera_health(self) -> dict[str, Any]:
        """What `doctor` prints and gates its verdict on, shaped like every other backend's."""
        spec = self.camera_spec
        size = f"{self._frame_size[0]}x{self._frame_size[1]}" if self._frame_size else None
        return {
            "configured": spec is not None,
            "url": spec.url if spec else None,
            "ok": spec is not None and self.camera_error is None and self._frame_at is not None,
            "age_s": None if self._frame_at is None else round(self.now() - self._frame_at, 2),
            "size": size,
            "error": self.camera_error,
        }

    @property
    def hot_joints(self) -> list[str]:
        """The body joints at or above `HOT_C`. The gripper is left out on purpose: it has
        LeRobot's own torque and current caps, and its temperature is still reported."""
        return sorted(k for k, v in self._temperature_c.items() if v >= HOT_C and k != "gripper")

    def _out_of_range(self) -> list[str]:
        """Joints reading outside the travel their own calibration recorded. Not a refusal:
        it means the file and the arm disagree, which is worth seeing before a goal is."""
        out = []
        for joint, value in self._joints.items():
            span = self.joint_range_deg.get(joint)
            if span and not (span[0] - OUT_OF_RANGE_DEG <= value <= span[1] + OUT_OF_RANGE_DEG):
                out.append(joint)
        return sorted(out)

    def _holding(self) -> bool:
        """Inferred, never sensed (`up.HOLDING_INFERRED`): the gripper was told to close, it
        has stopped moving, and it stopped short of shut."""
        if self._gripper_goal is None or self._gripper_goal > HOLD_MIN:
            return False
        if not self._gripper_trace:
            return False
        at, latest = self._gripper_trace[-1]
        # settled means the reading has not moved over a real interval. The heartbeat and a
        # verb's poll can land a millisecond apart, and two samples that close agree whatever
        # the gripper is doing, so the comparison reaches back at least one tick
        earlier = [pos for t, pos in self._gripper_trace if at - t >= SETTLE_GAP_S]
        if not earlier or abs(latest - earlier[-1]) > GRIPPER_SETTLE:
            return False
        return HOLD_MIN < latest < HOLD_MAX

    async def get_state(self) -> DuckState:
        await self._probe()
        extras: dict[str, Any] = {
            "joints": {k: round(v, 1) for k, v in self._joints.items()},
            "torque": self._torque,  # measured, not assumed: up.STS3215_REGISTERS
            "temperature_c": {k: round(v) for k, v in self._temperature_c.items()},
            "hot": self.hot_joints,
            "out_of_range": self._out_of_range(),
            "step_deg": self.max_step_deg,
            "range_clips": self._range_clips,
            "assumptions": [
                up.GRIPPER_OPEN_VALUE.name,
                up.HOLDING_INFERRED.name,
                up.TEMPERATURE_C.name,
            ],
        }
        if self._policy_error is not None:
            extras["policy_error"] = self._policy_error
        if self._register_error is not None:
            extras["register_error"] = self._register_error
        return DuckState(
            t=self.now(),
            policy=self._policy_name,
            posture="unknown",
            fallen=False,
            battery_percent=None,
            holding=self._holding(),
            extras=extras,
        )

    # ── writing ─────────────────────────────────────────────────────────────────────

    def _clip(self, joint: str, goal: float) -> float:
        span = self.joint_range_deg.get(joint)
        if span is None:
            return goal
        clipped = min(span[1], max(span[0], goal))
        if abs(clipped - goal) > 1e-6:
            self._range_clips += 1
        return clipped

    async def _send(self, goals: dict[str, float], *, clip: bool = True) -> dict[str, float]:
        """One `send_action`, and what it says it actually sent (`up.SO_SEND_ACTION_RETURN`).

        `clip` is off for a hold, where the goal is the position the arm is already in: an
        arm sitting outside its recorded travel should stay there when told to stop, not be
        walked back inside it."""
        action = {
            f"{joint}.pos": (self._clip(joint, float(goal)) if clip else float(goal))
            for joint, goal in goals.items()
            if joint in JOINTS
        }
        if not action:
            return {}
        if "gripper.pos" in action:
            self._gripper_goal = action["gripper.pos"]
        sent = await self._call(self._robot.send_action, action)
        written = dict(sent) if sent else action
        return {str(k).removesuffix(".pos"): float(v) for k, v in written.items()}

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

    async def send_intent(self, intent: Intent) -> Ack:
        p = intent.params
        if self._closed and intent.kind != "stop":
            return Ack(accepted=False, reason="the arm's transport is closed")
        if intent.kind in ("joint", "gripper") and self.policy_running:
            return Ack(accepted=False, reason="pick is running: stop first")
        try:
            match intent.kind:
                case "joint":
                    goals = {str(k): float(v) for k, v in dict(p.get("positions", {})).items()}
                    if (refusal := self._refuse_out_of_range(goals)) is not None:
                        return Ack(accepted=False, reason=refusal)
                    await self._send(goals)
                case "gripper":
                    open_ = bool(p.get("open", True))
                    await self._send({"gripper": GRIPPER_OPEN if open_ else GRIPPER_CLOSED})
                case "do":
                    return await self._do(str(p.get("skill")))
                case "stop":
                    await self._hold()
                case "enable":
                    if not p.get("on", True):
                        return Ack(accepted=False, reason="quackd never limps a robot")
                case _:
                    return Ack(accepted=False, reason=f"an arm cannot {intent.kind}")
        except TimeoutError:
            return Ack(accepted=False, reason=f"{intent.kind} timed out on LeRobot")
        except Exception as e:  # LeRobot's own errors are feedback, not a crash
            return Ack(accepted=False, reason=f"{intent.kind} failed: {type(e).__name__}: {e}")
        return Ack()

    async def _do(self, skill: str) -> Ack:
        kind, _, rest = skill.partition(":")
        name, _, task = rest.partition(":")
        if kind != "policy" or name != "pick":
            return Ack(accepted=False, reason=f"unknown skill {skill!r}")
        if self._policy is None:
            return Ack(accepted=False, reason="no policy was given to this backend")
        async with self._policy_lock:  # two picks at once must not each start a loop
            await self._cancel_policy()
            self._policy_error = None
            self._policy_name = f"policy:pick:{task}"
            self._policy_task = asyncio.create_task(self._run_policy(task))
        return Ack()

    async def _run_policy(self, task: str) -> None:
        """The policy's own observe/act loop at its own rate; quackd only says 'pick'.

        Its actions are clipped and capped exactly like a verb's: a learned policy is still
        a stranger, and `pick` is confirm-gated because it moves the whole arm."""
        assert self._policy is not None
        try:
            while not self._closed:
                obs = await self._observe()
                action = await asyncio.to_thread(self._policy.act, obs, task=task)
                if action is None:
                    break  # the policy considers the task done; the gripper says if it is
                await self._send(action)
                await asyncio.sleep(1.0 / POLICY_HZ)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._policy_error = f"{type(e).__name__}: {e}"
        finally:
            self._policy_name = "idle"

    async def _cancel_policy(self) -> None:
        task, self._policy_task = self._policy_task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._policy_name = "idle"

    async def _hold(self) -> None:
        """Stop is 'stay where you are': the present position becomes the goal.

        The gripper is left out of that goal on purpose. LeRobot writes only the keys it is
        given, so omitting it keeps whatever squeeze is already commanded: a stop that
        re-sent the gripper's measured position would open a hand that is holding something
        against its own goal, and every failed verb ends in a stop."""
        await self._cancel_policy()
        try:
            await self._probe()
            body = {k: v for k, v in self._joints.items() if k in JOINTS and k != "gripper"}
            if body:
                await self._send(body, clip=False)
        except Exception as e:
            # the core `stop` verb reads this and refuses to say "stopped" over a hold that
            # never reached the arm; a wedged call has already set its own, better, reason
            if self.stop_error is None:
                self.stop_error = f"the hold did not reach the arm: {type(e).__name__}: {e}"
            raise
        self.stop_error = None

    async def subscribe(self, topic: str) -> AsyncIterator[dict[str, Any]]:  # type: ignore[override]
        while not self._closed:
            await self.sleep(0.5)
            yield {"topic": topic, **(await self.get_state()).model_dump()}

    async def heartbeat(self) -> None:
        """A round trip to the arm, not a flag. `is_connected` is the serial port's own open
        flag (`up.BUS_IS_CONNECTED`): pull the cable and it stays True until a read fails."""
        if self._closed:
            raise HeartbeatError("lerobot real transport is closed")
        if self._robot is None or not bool(self._robot.is_connected):
            raise HeartbeatError("the arm is not connected")
        if self._wedged is not None and not self._wedged.done():
            raise HeartbeatError(self.stop_error or "a LeRobot call has not come back")
        try:
            await self._probe()
        except HeartbeatError:
            raise
        except Exception as e:
            raise HeartbeatError(f"the arm did not answer: {type(e).__name__}: {e}") from e

    async def stop(self) -> None:
        with contextlib.suppress(Exception):
            await self._hold()

    def now(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)
        if self.post_sleep is not None:
            self.post_sleep()


def load_policy(path: str, *, device: str = "cpu") -> PolicyLike:
    """A `PolicyLike` from a LeRobot checkpoint, from verified names. UNTESTED end to end
    (`upstream_api.POLICY_PIPELINE`); inject your own `policy=` to bypass this."""
    try:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    except ImportError as e:
        raise AdapterNotInstalled("lerobot", "quackd[lerobot]") from e

    config = PreTrainedConfig.from_pretrained(path)
    config.device = device
    policy = get_policy_class(config.type).from_pretrained(path, config=config)
    preprocess, postprocess = make_pre_post_processors(config, pretrained_path=path)

    class _HubPolicy:
        def act(self, observation: dict[str, Any], *, task: str) -> dict[str, float] | None:
            batch = preprocess({**observation, "task": task})
            with torch.no_grad():
                action = policy.select_action(batch)
            out = postprocess(action)
            return {str(k): float(v) for k, v in dict(out).items()}

    return _HubPolicy()
