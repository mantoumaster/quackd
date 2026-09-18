"""A real arm through LeRobot. Every name verified at a pin, and driven on one arm.

One SO-101 ran this backend on 2026-09-15 (lerobot 0.6.1, Windows 11, Python 3.12.12):
connect, `get_observation`, `send_action`, the two register reads and `disconnect` all
behaved as the rows in `upstream_api.py` say. The arm fell at the end of every one of those
runs, which is why `close()` now returns it to a recorded rest pose first and keeps torque
on when it could not get there.

Every LeRobot name comes from `upstream_api.py` (ADR-0022). LeRobot is synchronous, so
every call runs in a worker thread under one lock with a deadline, and a call that blows its
deadline wedges the transport rather than letting a second thread onto a half-duplex bus.
`stop` re-sends the present position as the goal (hold). quackd disables torque in exactly
one place, `let_go()`, which a person asks for with `--by-hand` and which refuses anywhere but
the recorded rest pose; `take_hold()` is how the arm is picked back up. LeRobot's own
`disconnect()` disables it too, by its default, which quackd keeps and documents.

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
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image

from quackd.adapters.base import AdapterError, AdapterNotInstalled, HandResult, RestResult
from quackd.transport.base import (
    Ack,
    CameraFrame,
    DuckState,
    HeartbeatError,
    Intent,
    TransportError,
)
from quackd_lerobot import upstream_api as up
from quackd_lerobot.verbs import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    JOINTS,
    LIMP_IN_HAND,
    STALL_DEG,
    STALL_TICKS,
    TICK_S,
    TOL_DEG,
    TORQUE_COULD_NOT_BE_KEPT,
    TORQUE_LEFT_ON,
    at_rest,
    rest_budget_s,
    rest_goal,
    shortfall,
)

STATUS = "LeRobot names verified at a pinned commit; one SO-101 driven on 2026-09-15"
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
CAMERA_CLOSE_S = 10.0
"""A release stops a read thread and joins it; a few seconds is routine on Windows."""


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
    name_given: bool = False
    """The url said `?name=`. With several cameras it must, because the name is the only
    thing telling two views apart; with one it may, and `front` is the default."""


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

    given = one("name")
    name = given or CAMERA_NAME
    if not name.replace("_", "").replace("-", "").isalnum():
        raise _camera_refusal(f"name={name!r} is not a plain name", url)
    fourcc = one("fourcc")
    if fourcc is not None and len(fourcc) != 4:
        raise _camera_refusal(f"fourcc={fourcc!r} must be four characters", url)
    backend = (one("backend") or "any").lower()
    if backend not in CAMERA_BACKENDS:
        raise _camera_refusal(f"backend={backend!r} is not one of {CAMERA_BACKENDS}", url)
    raw_rotation = one("rotation")
    rotation = 0
    if raw_rotation:
        try:
            rotation = int(raw_rotation)
        except ValueError:
            raise _camera_refusal(f"rotation={raw_rotation!r} is not a whole number", url) from None
    if rotation not in CAMERA_ROTATIONS:
        raise _camera_refusal(f"rotation={rotation} is not one of {CAMERA_ROTATIONS}", url)
    fov_deg: float | None = None
    if raw_fov := one("fov"):
        try:
            fov_deg = float(raw_fov)
        except ValueError:
            raise _camera_refusal(f"fov={raw_fov!r} is not a number of degrees", url) from None
        if not 0.0 < fov_deg < 180.0:
            raise _camera_refusal(f"fov={raw_fov!r} must be between 0 and 180", url)
    width = _camera_int("width", w, url) if (w := one("width")) else None
    height = _camera_int("height", h, url) if (h := one("height")) else None
    if (width is None) != (height is None):
        # upstream keeps the camera's own mode unless BOTH are set, so one alone would be
        # accepted here and quietly dropped there
        raise _camera_refusal("width and height come together or not at all", url)
    return CameraSpec(
        url=url,
        name=name,
        index_or_path=index_or_path,
        width=width,
        height=height,
        fps=_camera_int("fps", f, url) if (f := one("fps")) else None,
        fourcc=fourcc,
        backend=backend,
        rotation=rotation,
        fov_deg=fov_deg,
        name_given=given is not None,
    )


def parse_camera_urls(urls: Sequence[str]) -> tuple[CameraSpec, ...]:
    """Every `--camera-url` this arm was given, in order. The first is the primary.

    With one camera this is `parse_camera_url` and nothing more. With several, each url has
    to name its own camera and the names have to differ, because the name is what the model
    reading two pictures, a policy's observation dict and `frames/NNNN-<name>.png` all tell
    them apart by. An index may only appear once: two handles on one webcam is not two
    views, it is a camera that will not open twice."""
    specs = tuple(parse_camera_url(url) for url in urls)
    if len(specs) < 2:
        return specs
    by_name: dict[str, CameraSpec] = {}
    by_index: dict[int | str, CameraSpec] = {}
    for spec in specs:
        if not spec.name_given:
            raise _camera_refusal(
                f"it has no ?name= and {len(specs)} cameras were given. With several, every "
                "url names its own camera, opencv://1?name=top --camera-url "
                "opencv://2?name=side, because the name is what the model, a pick policy "
                "and frames/NNNN-<name>.png tell them apart by",
                spec.url,
            )
        if (clash := by_name.get(spec.name)) is not None:
            raise _camera_refusal(
                f"name={spec.name!r} is already the name of {clash.url!r}. With several "
                "cameras every name is its own",
                spec.url,
            )
        if (same := by_index.get(spec.index_or_path)) is not None:
            raise _camera_refusal(
                f"{spec.index_or_path} is already {same.url!r}. One url per camera",
                spec.url,
            )
        by_name[spec.name] = spec
        by_index[spec.index_or_path] = spec
    return specs


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
        cameras: Sequence[CameraSpec] = (),
        camera_objects: Mapping[str, Any] | None = None,
        rest_pose: dict[str, float] | None = None,
    ) -> None:
        self.port = address or ""
        self.robot_type = robot_type
        self.robot_id = robot_id
        self.timeout_s = timeout_s
        self.max_step_deg = max_step_deg
        self._robot: Any = robot  # injected in tests; built in connect() otherwise
        self.camera_specs: tuple[CameraSpec, ...] = (
            tuple(cameras) if cameras else ((camera,) if camera is not None else ())
        )
        self.camera_connect_s = CAMERA_CONNECT_S
        self.camera_close_s = CAMERA_CLOSE_S
        # name -> camera object, in the order the urls were given. Injected in tests; built
        # in connect() otherwise.
        self._cameras: dict[str, Any] = dict(camera_objects or {})
        if camera_object is not None:
            self._cameras.setdefault(self._primary_name(), camera_object)
        self.rest_pose = dict(rest_pose) if rest_pose else None
        """Where this arm rests, recorded off the arm by `quackd robot rest-pose`. The five
        body joints are what is ever driven; the gripper is kept for the record only."""
        self._rest_result: RestResult | None = None
        self.close_note: str | None = None
        """Set by `close()` when it left torque on. Said by whichever caller closed the arm,
        because a library that prints has picked one terminal and quackd has four callers."""
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
        self.camera_errors: dict[str, str | None] = {}
        self._frame_sizes: dict[str, tuple[int, int]] = {}
        self._frame_ats: dict[str, float] = {}
        self.lerobot_version: str | None = None
        self.stop_error: str | None = None
        self.post_sleep: Callable[[], None] | None = None
        self._in_hand = False
        """This arm is limp in somebody's hands, because `let_go()` put it there.

        Set only by a release that was read back off the arm, cleared only by a `take_hold()`
        that confirmed torque came on. Every teardown begins with `stop`, which is what picks
        the arm back up, so the window this is true in is the wait itself."""

    def _primary_name(self) -> str:
        """The first `--camera-url`'s camera: the one the detections describe, the one
        `--fov-deg` measures, and the only one a verb that steers by sight reads."""
        return self.camera_specs[0].name if self.camera_specs else CAMERA_NAME

    @property
    def camera_spec(self) -> CameraSpec | None:
        """The primary camera, for everything written when an arm had at most one."""
        return self.camera_specs[0] if self.camera_specs else None

    @property
    def camera_error(self) -> str | None:
        """Why the primary camera gave no frame. `camera_health()` has all of them."""
        return self.camera_errors.get(self._primary_name())

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

    def _build_camera(self, spec: CameraSpec) -> Any:
        """One webcam, built by quackd and not by the follower (`up.OPENCV_CAMERA`)."""
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
        # the camera first, before the arm is touched: a bad index then refuses with the
        # arm never energised, never de-torqued on the way back out, and nothing to undo
        await self._connect_cameras()
        try:
            await self._call(self._robot.connect, False, deadline_s=30.0)  # never calibrate
        except Exception as e:
            await self._close_cameras()
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

    async def _camera_call(self, fn: Callable[..., Any], *args: Any, timeout_s: float) -> Any:
        """A camera call: its own thread and its own deadline, and never the serial lock.

        The camera touches no bus, so it has no business under `_lock`. Worse than needless:
        `_call` files a call that blows its deadline as a wedged serial thread and refuses
        every later call until it returns, so a webcam whose open or release takes a few
        seconds (routine on Windows) would have refused the arm's own disconnect and left
        torque on. A camera that hangs here is simply abandoned, and says so."""
        try:
            return await asyncio.wait_for(asyncio.to_thread(fn, *args), timeout_s)
        except TimeoutError:
            raise TimeoutError(
                f"{getattr(fn, '__name__', 'the call')}() did not return within {timeout_s:g} s"
            ) from None

    async def _connect_cameras(self) -> None:
        """A camera the owner asked for and did not get is a refusal, not a warning.

        They named an index on the command line and `doctor` gates its verdict on a frame,
        so failing quietly would leave somebody believing they had eyes. It runs before the
        arm is touched, so a refusal here has energised nothing and let nothing go slack; the
        arm connects without --camera-url. With several, they open in the order the urls were
        given and the second one failing lets go of the first: half a set of eyes nobody
        asked for is worse than the refusal, because the frames would still arrive."""
        if not self.camera_specs:
            return
        self.camera_keys = ()
        self.camera_errors = {}
        self._frame_sizes = {}
        self._frame_ats = {}
        opened: list[str] = []
        for spec in self.camera_specs:
            try:
                if spec.name not in self._cameras:
                    self._cameras[spec.name] = await asyncio.to_thread(self._build_camera, spec)
                # warmup reads frames before it returns (up.CAMERA_CONNECT); an index that
                # will not open raises here, with upstream's own instructions in it
                # (up.OPENCV_OPEN_FAILS)
                await self._camera_call(
                    self._cameras[spec.name].connect, timeout_s=self.camera_connect_s
                )
            except AdapterNotInstalled:
                raise
            except Exception as e:
                await self._close_cameras()
                raise TransportError(
                    f"lerobot real: --camera-url {spec.url!r} did not open: {e}. The arm was "
                    "not touched, and it connects without --camera-url"
                ) from e
            opened.append(spec.name)
        self.camera_keys = tuple(opened)

    async def _close_cameras(self) -> None:
        for camera in list(self._cameras.values()):
            with contextlib.suppress(Exception):
                # up.CAMERA_DISCONNECT
                await self._camera_call(camera.disconnect, timeout_s=self.camera_close_s)
        self.camera_keys = ()

    async def _give_up(self, why: str) -> None:
        """Let go of everything opened so far, then say why. The arm's disconnect is the one
        LeRobot ships, and it drops torque (`up.SO_DISCONNECT_TORQUE`)."""
        await self._close_cameras()
        with contextlib.suppress(Exception):
            await self._call(self._robot.disconnect, deadline_s=5.0)
        raise TransportError(why)

    async def close(self) -> None:
        """Let go of the arm, and let go of its torque only where it can be let go of.

        LeRobot's `disconnect()` disables torque by its own default, which quackd keeps: an
        arm at rest should be limp, because that is what "at rest" means. An arm that is not
        at rest is an arm that would fall, so this reads the joints one last time and, where
        they are not the recorded pose, turns that default off and says so. Without a rest
        pose recorded there is nothing to check against and nothing changes.

        An arm still limp in somebody's hands is the one case where neither of those notes is
        true, and it says so in its own words: there is no torque to keep and nothing to keep
        it from."""
        self._closed = True
        await self._cancel_policy()
        # the cameras on their own deadline and never the serial lock, so however long a
        # release takes, the arm's disconnect below still runs
        await self._close_cameras()
        if self._robot is None:
            return
        self.close_note = None
        why = await self._not_resting() if self.rest_pose is not None else None
        if self._in_hand:
            # Whoever is reading this has the arm in their hand. The torque note below would
            # tell them it is holding itself up, which is the one thing it is not.
            self.close_note = LIMP_IN_HAND.format(
                why=why or "it was let go of for you to place and never taken hold of again"
            )
            with contextlib.suppress(Exception):
                await self._call(self._robot.disconnect, deadline_s=5.0)
            return
        if why is not None:
            self.close_note = TORQUE_LEFT_ON.format(why=why)
        wrote = False
        with contextlib.suppress(Exception):
            # up.SO_DISCONNECT_READS_ITS_CONFIG_LATE: the flag is read off the config instance
            # inside disconnect() rather than copied at construction, so this is the seam.
            # _config_kwargs() still asks for True.
            #
            # Written every time rather than only when torque has to stay on. The flag lives
            # on the robot, not on this call, so a transport that missed its pose once and
            # reached it the next time would have kept the arm energised on the strength of
            # the earlier session, with nothing said about it.
            self._robot.config.disable_torque_on_disconnect = why is None
            wrote = True
        if why is not None and not wrote:
            # the seam did not take, so the disconnect below releases torque after all. Saying
            # the arm is being held when it is about to be let go is worse than saying nothing.
            self.close_note = TORQUE_COULD_NOT_BE_KEPT.format(why=why)
        with contextlib.suppress(Exception):
            await self._call(self._robot.disconnect, deadline_s=5.0)  # up.SO_DISCONNECT_TORQUE

    async def _not_resting(self) -> str | None:
        """Why this arm must keep its torque, or None if it may let go. Reads, never moves."""
        goal = rest_goal(self.rest_pose or {})
        if not goal:
            # a pose was recorded and none of it can be driven: the arm is somewhere nobody
            # chose, so it keeps holding rather than being let go there
            return "the recorded pose names no joint this arm drives"
        try:
            await self._probe()
        except Exception as e:
            return f"the arm did not answer: {type(e).__name__}: {e}"
        joints = dict(self._joints)
        if at_rest(goal, joints):
            return None
        why = shortfall(goal, joints)
        if self._rest_result is not None and not self._rest_result.reached:
            return f"{why}; {self._rest_result.reason}"
        return f"{why}; nothing moved it there"

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
        for name in self.camera_keys:
            camera = self._cameras.get(name)
            if camera is not None:
                obs[name] = await asyncio.to_thread(camera.read_latest)
        return obs

    @staticmethod
    def _joints_of(obs: dict[str, Any]) -> dict[str, float]:
        return {k.removesuffix(".pos"): float(v) for k, v in obs.items() if k.endswith(".pos")}

    async def _read_frame(self, name: str) -> Image.Image | None:
        """One camera's newest frame, or None and a reason. Never raises.

        `observe` moves nothing, so a camera that has stopped delivering should cost the
        picture and not the run: the agent loop asks for a frame every step and `doctor`
        polls for one, and neither expects an exception. The read does not go through
        `_call`, because it touches no serial bus and a wedge must never be filed as a
        camera fault (`up.CAMERA_READ_LATEST`)."""
        camera = self._cameras.get(name)
        if camera is None:
            return None
        try:
            frame = await asyncio.to_thread(camera.read_latest)
            image = Image.fromarray(np.asarray(frame))  # up.CAMERA_COLOR_MODE_DEFAULT: RGB
        except Exception as e:
            self.camera_errors[name] = f"{type(e).__name__}: {e}"
            return None
        self.camera_errors[name] = None
        self._frame_sizes[name] = image.size
        self._frame_ats[name] = self.now()
        return image

    async def get_frame(self) -> Image.Image | None:
        """The primary camera's newest frame. What steers, and what the detector reads."""
        return await self._read_frame(self._primary_name())

    async def get_frames(self) -> list[CameraFrame]:
        """Every camera's newest frame, primary first, each under its own name.

        A camera that gave nothing this time is simply absent from the list rather than an
        empty slot: the model is shown the views that exist, and `camera_health()` is where
        the one that stopped says so."""
        primary = self._primary_name()
        frames = []
        for name in self.camera_keys or (primary,):
            image = await self._read_frame(name)
            if image is not None:
                frames.append(CameraFrame(name, image, primary=name == primary))
        return frames

    def _camera_row(self, spec: CameraSpec) -> dict[str, Any]:
        size = self._frame_sizes.get(spec.name)
        at = self._frame_ats.get(spec.name)
        return {
            "name": spec.name,
            "url": spec.url,
            "ok": self.camera_errors.get(spec.name) is None and at is not None,
            "age_s": None if at is None else round(self.now() - at, 2),
            "size": f"{size[0]}x{size[1]}" if size else None,
            "error": self.camera_errors.get(spec.name),
        }

    def camera_health(self) -> dict[str, Any]:
        """What `doctor` prints and gates its verdict on, shaped like every other backend's.

        One camera answers exactly what it always answered. Several answer that plus a
        `cameras` list, one row each, so a reader written for one camera still reads the
        primary and a reader that knows about several gets all of them."""
        spec = self.camera_spec
        at = self._frame_ats.get(self._primary_name())
        size = self._frame_sizes.get(self._primary_name())
        health: dict[str, Any] = {
            "configured": spec is not None,
            "url": spec.url if spec else None,
            "ok": spec is not None and self.camera_error is None and at is not None,
            "age_s": None if at is None else round(self.now() - at, 2),
            "size": f"{size[0]}x{size[1]}" if size else None,
            "error": self.camera_error,
        }
        if len(self.camera_specs) > 1:
            health["cameras"] = [self._camera_row(s) for s in self.camera_specs]
        return health

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
        if self.camera_spec is not None:
            # A camera that dies mid-run is otherwise invisible on a `quackd run`: the only
            # reader of camera_error is the `observe` verb, and `observe` cannot be in a
            # .duck's allowlist on this backend because the static manifest has no camera.
            # So the frames stop, the observation loses a line, and nothing says why. This
            # is pure field reads, no bus and no camera call, so the heartbeat pays nothing.
            extras["camera"] = self.camera_health()
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
        against its own goal, and every failed verb ends in a stop.

        An arm somebody is holding is taken hold of first. Every teardown begins with a stop,
        so this is what a Ctrl-C during the hand-off wait reaches: the arm is energised where
        the person's hand has it, and the rest move that follows can then put it down. Sending
        a goal to a limp servo instead would be a stop that stopped nothing."""
        await self._cancel_policy()
        retaken: HandResult | None = None
        if self._in_hand:
            retaken = await self.take_hold()
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
        if retaken is not None and not retaken.ok and self._in_hand:
            # the goal above went to a limp servo and moved nothing, so this is not a stop.
            # Only where the arm is still in a hand: a `take_hold` that refused because the
            # arm slipped did energise it, and that arm is holding itself perfectly well.
            self.stop_error = f"the arm is limp in somebody's hands: {retaken.reason}"
            return
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

    # ── handing the arm to a person ─────────────────────────────────────────────────

    async def let_go(self) -> HandResult:
        """Take torque off, so somebody can pick the arm up and place it. Never raises.

        This is the only call in quackd that de-energises a robot, and it is deliberately the
        narrowest one that could do the job. It refuses anywhere but the recorded rest pose,
        which is the same condition `close()` uses to decide whether letting go is safe: a
        pose the arm demonstrably holds with no torque on it. Releasing an arm held up by
        torque alone would drop it, and the person asking for this has their hands nowhere
        near it yet.

        A register read that failed after the release is treated as a release that took. The
        alternative reading, that torque is still on, ends with `close()` printing that the
        arm is holding itself up over an arm hanging limp in somebody's hand, and of the two
        wrong answers that is the one that gets an arm dropped."""
        if self._closed:
            return HandResult("refused", "the arm's transport is closed")
        if self.rest_pose is None:
            return HandResult(
                "refused",
                "no rest pose is recorded for this arm, so there is nowhere it is known to be "
                "safe to let go of it: quackd robot rest-pose NAME",
            )
        goal = rest_goal(self.rest_pose)
        if not goal:
            return HandResult("refused", "the recorded pose names no joint this arm drives")
        try:
            await self._cancel_policy()
            await self._probe()
            if not at_rest(goal, self._joints):
                return HandResult(
                    "refused",
                    f"the arm is not at its rest pose ({shortfall(goal, self._joints)}), and "
                    "an arm held up by torque alone falls when torque goes",
                )
            await self._call(self._robot.bus.disable_torque)  # up.BUS_DISABLE_TORQUE
            await self._probe()
        except Exception as e:
            return HandResult("refused", self.stop_error or f"{type(e).__name__}: {e}")
        if self._torque and self._register_error is None:
            return HandResult("refused", "the arm still reports torque on, so it was not released")
        self._in_hand = True
        return HandResult("released", "torque is off at the rest pose", joints=dict(self._joints))

    async def take_hold(self) -> HandResult:
        """Hold the pose the arm is in right now, so the person can let go of it.

        The order matters and is the whole of this method. The present position is written as
        the goal *before* torque comes on, because nothing upstream documents what a servo
        does with the goal it was last told when it is re-energised
        (`up.TORQUE_ENABLE_HOLDS_PRESENT`), and the goal it was last told here is a rest pose
        the arm has since been lifted out of by hand. Enabling torque first could therefore
        snap the arm back to the fold with a person's hand in it.

        It is written again afterwards, and read back, so the answer says whether the arm
        actually stayed where it was put rather than assuming it. A refusal here leaves torque
        on: the arm is holding *something*, and the caller is told what moved."""
        if self._closed:
            return HandResult("refused", "the arm's transport is closed")
        try:
            await self._cancel_policy()
            await self._probe()
            placed = dict(self._joints)
            body = {j: v for j, v in placed.items() if j in JOINTS}
            if not body:
                return HandResult("refused", "the arm reported no joint to hold")
            # unclipped, for `_drive_to_rest`'s reason: this is where the arm physically is,
            # and a hand-placed arm can easily sit outside the travel its calibration recorded
            await self._send(body, clip=False)
            await self._call(self._robot.bus.enable_torque)  # up.BUS_ENABLE_TORQUE
            await self._send(body, clip=False)
            await asyncio.sleep(TICK_S)
            await self._probe()
        except Exception as e:
            return HandResult("refused", self.stop_error or f"{type(e).__name__}: {e}")
        if not self._torque and self._register_error is None:
            # still limp, so still in somebody's hands, and `close()` should still say so
            return HandResult("refused", "the arm still reports torque off, so nothing holds it")
        # Torque is on, so the arm is holding itself up whatever else went wrong, and it is no
        # longer hanging off a hand. The refusal below is about the *pose* and not about that:
        # an arm reported limp in somebody's hands while it is energised sends them to cut the
        # power on a robot that is holding perfectly well.
        self._in_hand = False
        held = dict(self._joints)
        moved = {j: abs(held[j] - v) for j, v in placed.items() if j in held}
        slipped = sorted(j for j, gap in moved.items() if gap > TOL_DEG)
        if slipped:
            worst = max(slipped, key=lambda j: moved[j])
            return HandResult(
                "refused",
                f"the arm moved as torque came on ({worst} by {moved[worst]:.0f} degrees), so "
                "it is not holding the pose you set; it is holding where it is now",
                joints=held,
            )
        return HandResult("held", "holding the pose you set", joints=held)

    async def go_to_rest(self) -> RestResult:
        """Drive the arm to the pose it was recorded resting in. Never raises.

        Every caller is a teardown or the first moment of a run, so a wedged bus, an arm
        that stopped answering or a send that never landed are answers here rather than
        exceptions: the caller still has to disconnect, and `close()` reads the joints
        itself before deciding whether torque may drop."""
        if self.rest_pose is None:
            return RestResult.none("no rest pose is recorded for this arm")
        if self._closed:
            return RestResult("refused", "the arm's transport is closed")
        goal = rest_goal(self.rest_pose)
        if not goal:
            # "refused", never "none": `none` means there is nothing to go to, and the run
            # would start anyway and the arm be released at the end. There is a pose here,
            # it names nothing this arm drives, and that is a reason to keep holding.
            return RestResult("refused", "the recorded pose names no joint this arm drives")
        try:
            await self._cancel_policy()
            await self._probe()
            if at_rest(goal, self._joints):
                result = RestResult("already", "already at the rest pose")
            else:
                result = await self._drive_to_rest(goal)
        except Exception as e:
            result = RestResult("refused", self.stop_error or f"{type(e).__name__}: {e}")
        self._rest_result = result
        return result

    async def _drive_to_rest(self, goal: dict[str, float]) -> RestResult:
        """Re-send the rest goal until the arm is there, stops moving, or the time is up.

        This looks like `verbs._drive` and cannot be it. That one goes through the executor,
        whose abort is already set by the time a person's Ctrl-C reaches a teardown, and this
        move has to run on exactly that path. It also sends with `clip=False`: a pose read
        off the arm is where the arm physically was, and an arm folded to rest often sits
        outside the travel its calibration recorded, which the range refusal would refuse."""
        joints = dict(self._joints)
        gap = max((abs(joints[j] - v) for j, v in goal.items() if j in joints), default=0.0)
        budget_s = rest_budget_s(gap, self.max_step_deg)
        stall = min(STALL_DEG, self.max_step_deg / 2) if self.max_step_deg > 0 else STALL_DEG
        started = self.now()
        previous: dict[str, float] = {}
        still = 0
        while self.now() - started < budget_s:
            await self._send(goal, clip=False)
            await asyncio.sleep(TICK_S)
            await self._probe()
            joints = dict(self._joints)
            if at_rest(goal, joints):
                return RestResult("arrived", "moved to the rest pose")
            moved = [abs(joints[j] - previous[j]) for j in previous if j in joints]
            still = still + 1 if moved and max(moved) <= stall else 0
            previous = {j: joints[j] for j in goal if j in joints}
            if still >= STALL_TICKS:
                # hold, so the servo stops pushing at a goal it has been told it cannot reach
                with contextlib.suppress(Exception):
                    await self._hold()
                return RestResult(
                    "stalled", f"{shortfall(goal, joints)}, and it has stopped moving"
                )
        with contextlib.suppress(Exception):
            await self._hold()
        return RestResult(
            "timeout", f"{shortfall(goal, joints)} when the time ran out ({budget_s:.0f} s)"
        )

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
