"""The LeRobot adapter: a desktop arm (an SO-101 class follower) as a quackd robot.

An arm has no legs, no head and no voice, so its manifest lists none of that: `move`,
`go_to`, `search_scan`, `say` and `gaze` do not exist here. What it has is joints, a
gripper, `place`, and, when a policy is available, `pick` as one skill intent that the
arm's own learned controller executes (the thesis, unchanged). Two backends: `mock`
(offline, scripted) and `real` (LeRobot behind `quackd[lerobot]`, Python 3.12 or newer,
first driven on an arm on 2026-09-15).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from PIL import Image

from quackd.adapters.base import (
    AdapterError,
    HandResult,
    RestResult,
    camera_urls,
    go_to_rest_if_any,
    let_go_if_any,
    one_camera_url,
    take_hold_if_any,
)
from quackd.adapters.manifest import (
    Datasheet,
    Figure,
    Frame,
    Health,
    RobotManifest,
    SafetyAuthority,
    verb_spec,
)
from quackd.transport.base import (
    Ack,
    CameraFrame,
    DuckState,
    DuckTransport,
    HeartbeatError,
    Intent,
    frames_of,
)
from quackd.verbs.core import CORE
from quackd.verbs.registry import Precondition, Verb
from quackd_lerobot.verbs import JOINTS, lerobot_conditions, lerobot_verbs

__version__ = "0.11.0"
"""Kept in step with quackd's own version by scripts/set_version.py. It lives here rather
than being read from the core, because this file is all an adapter's sdist contains."""

BACKENDS = ("mock", "real")
DEFAULT_ID = "arm-01"
ROBOT_TYPE = "so101_follower"
BLURB = (
    "a six-joint desktop robot arm with a parallel gripper (an SO-101 class arm driven "
    "by LeRobot), bolted to a table"
)

DATASHEET = Datasheet(
    height_m=Figure(
        value=0.53,
        confidence="estimate",
        source="one vendor's listing",
        note="reaching straight up",
    ),
    dof=Figure(
        value=6,
        confidence="official",
        source="the LeRobot SO-101 docs",
        note="five joints and a gripper",
    ),
    payload_kg=Figure(value=0.5, confidence="estimate", source="one vendor's listing"),
    manipulator="gripper",
    arms=1,
    tethered=True,
    cannot=[
        "go anywhere: it is bolted to a table and has no base",
        "lift or hold more than about half a kilogram, and nothing whose weight is not known",
        "reach anything that is not already within arm's length of its base: the reach is not "
        "published",
        "feel what it holds: nothing reports grip force, so holding is inferred from the "
        "gripper stopping short of shut, which an empty hand that binds also does",
        "know its own mass: vendor listings disagree by a factor of three",
    ],
    notes=[
        "How wide the gripper opens and how hard it grips are not published",
        "No sensors beyond the servo positions unless a camera is configured, and the servo "
        "temperature, which quackd reads off the bus because LeRobot does not",
        "Vendor listings give the mass as anything from 0.8 to 2.5 kg and nobody official "
        "publishes one, so it is listed as not published rather than picked from a hat",
        "There are 7.4 V and 12 V builds of the same arm, with different stall torque and "
        "different supplies. Which one is on the desk is not something quackd can ask",
    ],
)


def lerobot_manifest(
    backend: str,
    robot_id: str | None = None,
    *,
    camera: bool = False,
    policy: bool = False,
    robot_type: str = ROBOT_TYPE,
    lerobot_version: str | None = None,
    joint_range_deg: dict[str, tuple[float, float]] | None = None,
    step_deg: float | None = None,
    calibration_file: str | None = None,
    camera_url: str | None = None,
    camera_fov_deg: float | None = None,
    camera_names: Sequence[str] = (),
) -> RobotManifest:
    """The arm as data. `camera` and `policy` are what the backend found at connect: the
    static manifest of `real` claims neither, the mock has both. So are the joint ranges,
    which come off the arm's own calibration file and are unknown until it has answered."""
    own = lerobot_verbs(policy=policy)
    verbs = [
        verb_spec(own["report_state"], core=True),
        verb_spec(CORE["stop"], core=True),
        verb_spec(own["move_joints"], core=False),
        verb_spec(own["gripper"], core=False),
        verb_spec(own["place"], core=False),
    ]
    if camera:
        verbs.insert(0, verb_spec(CORE["observe"], core=True))
    # not_hot guards the five body joints, which are the ones LeRobot caps nothing on; the
    # gripper has its own torque and current caps, and refusing to open a hot one would
    # strand whatever it is holding
    preconditions = {"move_joints": ["torque_on", "not_hot"], "place": ["holding"]}
    if policy:
        verbs.append(verb_spec(own["pick"], core=False, safety_class="confirm"))
        preconditions["pick"] = ["torque_on", "not_hot"]
    intents: list[Any] = ["joint", "gripper"] + (["skill"] if policy else [])
    sensors: list[Any] = ["joint_state"] + (["camera"] if camera else [])
    limits = {"joint_deg": 180.0, "gripper": 100.0}
    if step_deg is not None:
        limits["step_deg"] = float(step_deg)
    if camera_fov_deg is not None:
        # what the detector calibrates bearings with; over MCP there is no --fov-deg, so
        # the lens travels with the camera instead
        limits["camera_fov_deg"] = float(camera_fov_deg)
    extras: dict[str, Any] = {
        "robot_type": robot_type,
        "joints": list(JOINTS),
        "policy": policy,
        "lerobot_version": lerobot_version,
        # the gripper is the only joint LeRobot writes a torque or current cap for
        "torque_limit_scope": "gripper_only",
    }
    if joint_range_deg:
        extras["joint_range_deg"] = {
            joint: [round(lo, 1), round(hi, 1)] for joint, (lo, hi) in joint_range_deg.items()
        }
    if calibration_file:
        extras["calibration_file"] = calibration_file
    if camera_url:
        extras["camera"] = camera_url
    if len(camera_names) > 1:
        # only when there is more than one: the name of a single camera is quackd's own
        # default rather than anything the owner chose, and a pilot told its one camera is
        # called `front` would start naming it in sentences nobody needs
        extras["cameras"] = list(camera_names)
    return RobotManifest(
        id=robot_id or DEFAULT_ID,
        vendor="huggingface",
        model="lerobot-so101",
        embodiment="arm",
        mobility="none",
        intents=intents,
        sensors=sensors,
        verbs=verbs,
        preconditions=preconditions,
        # the gripper's torque and current caps written by LeRobot at configure() are the
        # only native limit; no deadman: an arm holds its goal when the client goes quiet
        safety_authority=SafetyAuthority(native="torque_limit", deadman=False, heartbeat_hz=2.0),
        frame=Frame(
            reference="base",
            note="joint space in degrees (gripper 0..100); no camera-to-base calibration",
        ),
        limits=limits,
        backend=backend,
        blurb=BLURB,
        datasheet=DATASHEET,
        extras=extras,
    )


class LeRobotAdapter:
    """A `RobotAdapter` over the mock or the real backend."""

    name = "lerobot"
    supports_hand_off = True
    """A person can be handed this body: quackd takes torque off at its recorded rest pose,
    waits while they place it, and holds whatever pose they left it in (`quackd run
    --by-hand`). Declared rather than inferred, because the run refuses the flag outright on a
    body that does not offer it rather than connecting and finding out."""
    supports_rest_pose = True
    """This body is driven to a recorded pose before torque is released. The registry's
    `rest-pose` command asks for exactly this, because a body with joints that quackd does
    not park would take the recording and never use it."""

    def __init__(self, transport: DuckTransport, *, robot_id: str | None = None) -> None:
        self.transport = transport
        self.backend = transport.name
        self.robot_id = robot_id or DEFAULT_ID
        self.manifest: RobotManifest | None = None
        # known before connect for the mock (a class attribute) and for a real backend
        # with an injected policy; refreshed at connect
        self._policy = bool(getattr(transport, "policy_available", False))

    async def connect(self) -> RobotManifest:
        await self.transport.connect()
        self._policy = bool(getattr(self.transport, "policy_available", False))
        spec = getattr(self.transport, "camera_spec", None)
        self.manifest = lerobot_manifest(
            self.backend,
            self.robot_id,
            camera=bool(getattr(self.transport, "camera_available", False)),
            policy=self._policy,
            lerobot_version=getattr(self.transport, "lerobot_version", None),
            joint_range_deg=getattr(self.transport, "joint_range_deg", None) or None,
            step_deg=getattr(self.transport, "max_step_deg", None),
            calibration_file=getattr(self.transport, "calibration_file", None),
            camera_url=getattr(spec, "url", None),
            camera_fov_deg=getattr(spec, "fov_deg", None),
            camera_names=getattr(self.transport, "camera_keys", ()),
        )
        return self.manifest

    async def disconnect(self) -> None:
        await self.transport.close()

    async def close(self) -> None:
        await self.disconnect()

    async def get_state(self) -> DuckState:
        return await self.transport.get_state()

    async def get_frame(self) -> Image.Image | None:
        return await self.transport.get_frame()

    async def get_frames(self) -> list[CameraFrame]:
        """Every camera this arm has, primary first. A mock has one and says so."""
        return await frames_of(self.transport)

    @property
    def camera_keys(self) -> tuple[str, ...]:
        """The cameras this arm was opened with, whether or not one is answering now.

        Forwarded for the same reason `camera_error` is, and read for a sharper one: it is how
        a caller knows a lone picture came off a two-camera arm and still needs its name."""
        return tuple(getattr(self.transport, "camera_keys", ()))

    async def go_to_rest(self) -> RestResult:
        return await go_to_rest_if_any(self.transport)

    async def let_go(self) -> HandResult:
        return await let_go_if_any(self.transport)

    async def take_hold(self) -> HandResult:
        return await take_hold_if_any(self.transport)

    @property
    def rest_pose(self) -> dict[str, float] | None:
        """Where this arm rests, if it was recorded. Read by the loop and the MCP session to
        decide whether there is a move to narrate at all."""
        pose = getattr(self.transport, "rest_pose", None)
        return dict(pose) if pose else None

    async def send_intent(self, intent: Intent) -> Ack:
        return await self.transport.send_intent(intent)

    async def health(self) -> Health:
        try:
            await self.transport.heartbeat()
        except HeartbeatError as e:
            return Health(ok=False, reason=str(e))
        state = await self.transport.get_state()
        extras: dict[str, Any] = {
            "holding": state.holding,
            "policy": state.policy,
            "torque": state.extras.get("torque"),
        }
        temperatures = [float(v) for v in state.extras.get("temperature_c", {}).values()]
        if temperatures:
            extras["hottest_c"] = round(max(temperatures))
        # what doctor is the first place to show: which calibration file the arm answered
        # with, and the travel that file gives each joint
        if path := getattr(self.transport, "calibration_file", None):
            extras["calibration_file"] = path
        if ranges := getattr(self.transport, "joint_range_deg", None):
            extras["joint_range_deg"] = {
                j: [round(lo), round(hi)] for j, (lo, hi) in ranges.items()
            }
        return Health(ok=True, battery_percent=None, extras=extras)

    @property
    def camera_error(self) -> str | None:
        """Why the last frame did not arrive, when the backend knows. A verb's `ctx.transport`
        is this adapter, not the transport underneath, so the hint has to be proxied here to
        reach `observe`."""
        error = getattr(self.transport, "camera_error", None)
        return str(error) if error else None

    @property
    def close_note(self) -> str | None:
        """What `close()` has to say about torque, when it left it on. Proxied for the same
        reason `camera_error` is: the callers hold this adapter, not the transport."""
        note = getattr(self.transport, "close_note", None)
        return str(note) if note else None

    @property
    def stop_error(self) -> str | None:
        """Why the last stop did not reach the arm, when the backend knows: the core `stop`
        verb reads this and refuses to say "stopped" over a hold that never got there."""
        error = getattr(self.transport, "stop_error", None)
        return str(error) if error else None

    async def heartbeat(self) -> None:
        await self.transport.heartbeat()

    async def stop(self) -> None:
        await self.transport.stop()

    def subscribe(self, topic: str) -> AsyncIterator[dict[str, Any]]:
        return self.transport.subscribe(topic)

    def now(self) -> float:
        return self.transport.now()

    async def sleep(self, seconds: float) -> None:
        await self.transport.sleep(seconds)

    def preconditions(self) -> dict[str, Precondition]:
        return lerobot_conditions()

    def implementations(self) -> dict[str, Verb]:
        return lerobot_verbs(policy=self._policy)

    @property
    def mobility(self) -> str:
        return "none"

    @property
    def post_sleep(self) -> Callable[[], None] | None:
        return getattr(self.transport, "post_sleep", None)

    @post_sleep.setter
    def post_sleep(self, hook: Callable[[], None] | None) -> None:
        self.transport.post_sleep = hook  # type: ignore[attr-defined]


# ── what the factory calls ──────────────────────────────────────────────────────────────


def describe(backend: str, robot_id: str | None = None) -> RobotManifest:
    """Static: the mock always has its camera and its scripted policy; the real backend
    claims neither until connect() finds them, and claims no joint ranges either, because
    they are read off the arm's calibration file."""
    offline = backend == "mock"
    return lerobot_manifest(backend, robot_id, camera=offline, policy=offline)


def implementations() -> dict[str, Verb]:
    return lerobot_verbs(policy=True)


def conditions() -> dict[str, Precondition]:
    return lerobot_conditions()


def _check_rest_pose(rest_pose: dict[str, float] | None) -> None:
    """Refuse a pose that names nothing this arm would drive, rather than ignoring it.

    `quackd robot rest-pose` reads the pose off the arm and cannot produce one of these; a
    hand-edited `robots.json` can, and did in review. Without this the pose is accepted,
    printed by `robot show`, driven nowhere, and the arm is released where it stands: the one
    failure this whole feature exists to prevent, arrived at by a typed joint name."""
    from quackd_lerobot.verbs import NO_DRIVABLE_JOINT, drivable_rest_joints, rest_goal

    if rest_pose and not rest_goal(rest_pose):
        raise AdapterError(
            NO_DRIVABLE_JOINT.format(
                named=", ".join(sorted(rest_pose)), drivable=", ".join(drivable_rest_joints())
            )
        )


def make(
    backend: str,
    *,
    robot_id: str | None = None,
    seed: int | None = None,
    address: str | None = None,
    live: bool = False,
    camera_url: str | Sequence[str] | None = None,
    token: str | None = None,
    rest_pose: dict[str, float] | None = None,
) -> LeRobotAdapter:
    _check_rest_pose(rest_pose)
    if backend == "mock":
        from quackd_lerobot.mock import LeRobotMock

        # the mock draws its own view and reads no url, but a second camera is still refused
        # rather than dropped: only the real arm opens more than one, and a task rehearsed
        # against the mock should fail here rather than at the bench
        one_camera_url(camera_url, spec="lerobot:mock")
        return LeRobotAdapter(LeRobotMock(rest_pose=rest_pose), robot_id=robot_id)
    if backend == "real":
        from quackd_lerobot.real import LeRobotReal, parse_camera_urls, step_from_env

        return LeRobotAdapter(
            LeRobotReal(
                address=address,
                robot_id=robot_id or DEFAULT_ID,
                max_step_deg=step_from_env(),
                cameras=parse_camera_urls(camera_urls(camera_url)),
                rest_pose=rest_pose,
            ),
            robot_id=robot_id,
        )
    raise ValueError(f"unknown lerobot backend {backend!r}; choose one of {BACKENDS}")


__all__ = [
    "BACKENDS",
    "DEFAULT_ID",
    "JOINTS",
    "LeRobotAdapter",
    "conditions",
    "describe",
    "implementations",
    "lerobot_manifest",
    "make",
]


# What this adapter reads from upstream, for `quackd doctor`. Declared here rather than in a
# table in the core, because the list belongs to whoever wrote the adapter (ADR-0022). The
# import is deferred so that naming the upstream costs nothing until doctor asks.
def _upstream_rows() -> tuple[tuple[str, object, str, str], ...]:
    from quackd_lerobot import upstream_api

    # The one row in this table that is not a list of what nobody has tried. An SO-101 ran the
    # real backend on 2026-09-15, so the column says what that run did and did not cover.
    return (
        (
            "lerobot",
            upstream_api,
            "docs/adapters/lerobot.md",
            "run on an SO-101 on 2026-09-15; the pick policy was not exercised",
        ),
    )


UPSTREAMS = _upstream_rows()
