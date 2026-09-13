"""The rosbridge adapter: any wheeled base that takes a Twist over rosbridge.

The first robot in quackd that is not a specific product: a ROS 2 base reachable through
`rosbridge_server`. Its manifest is small and honest: one intent (`twist`), odometry,
optionally a compressed image topic, and therefore `move`, `stop`, `report_state`, plus
`observe`, `go_to`, `search_scan` and `approach_and` only when a camera topic is given.
No `say`, no `gaze`. Two backends: `mock` (offline kinematics with deadman semantics) and
`ws` (roslibpy behind `quackd[rosbridge]`, never run against a bridge by us).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from PIL import Image

from quackd.adapters.manifest import (
    Datasheet,
    Frame,
    Health,
    RobotManifest,
    SafetyAuthority,
    verb_spec,
)
from quackd.transport.base import Ack, DuckState, DuckTransport, HeartbeatError, Intent
from quackd.verbs.core import CORE
from quackd.verbs.registry import Precondition, Verb

BACKENDS = ("mock", "ws")
DEFAULT_ID = "base-01"
MAX_VX = 0.3
MAX_WZ = 1.0
BLURB = (
    "a small wheeled base driven over rosbridge (a ROS 2 robot that takes velocity "
    "commands and reports odometry)"
)

DATASHEET = Datasheet(
    manipulator="none",
    cannot=[
        "carry, push or hold anything through quackd: this adapter commands a velocity and "
        "nothing else",
    ],
    notes=[
        "rosbridge is a software bridge: the name says nothing about the body under it. What is "
        "unknown here is unknown, not zero",
    ],
)
_MOVE_DESCRIPTION = (
    "Drive with a velocity for a duration: vx forward m/s, wz rad/s (+ = left). The base's "
    "own driver moves; quackd re-sends the command while the verb runs."
)


def rosbridge_manifest(
    backend: str,
    robot_id: str | None = None,
    *,
    camera: bool = False,
    max_vx: float = MAX_VX,
    max_wz: float = MAX_WZ,
    cmd_vel: str = "/cmd_vel",
    odom: str = "/odom",
    image: str | None = None,
    roslibpy_version: str | None = None,
) -> RobotManifest:
    """The base as data. `camera` is whether an image topic is configured."""
    verbs = [
        verb_spec(CORE["report_state"], core=True),
        verb_spec(CORE["stop"], core=True),
        verb_spec(CORE["move"], core=True, description=_MOVE_DESCRIPTION),
    ]
    if camera:
        verbs = [
            verb_spec(CORE["observe"], core=True),
            *verbs,
            verb_spec(CORE["go_to"], core=True),
            verb_spec(CORE["search_scan"], core=True),
            verb_spec(CORE["approach_and"], core=True),
        ]
    sensors: list[Any] = ["odometry"] + (["camera"] if camera else [])
    return RobotManifest(
        id=robot_id or DEFAULT_ID,
        vendor="ros",
        model="rosbridge-base",
        embodiment="wheeled",
        mobility="wheeled",
        intents=["twist"],
        sensors=sensors,
        verbs=verbs,
        preconditions={},
        # no deadman was verified anywhere: quackd re-sends and zeroes, and says so
        safety_authority=SafetyAuthority(native="none", deadman=False, heartbeat_hz=2.0),
        frame=Frame(reference="base", note="Twist in the base frame; odometry in its odom frame"),
        limits={"max_vx": max_vx, "max_vy": 0.0, "max_wz": max_wz},
        backend=backend,
        blurb=BLURB,
        datasheet=DATASHEET,
        extras={
            "ros": "2",
            "cmd_vel": cmd_vel,
            "odom": odom,
            "image": image,
            "roslibpy_version": roslibpy_version,
        },
    )


class RosbridgeAdapter:
    """A `RobotAdapter` over the mock or the ws backend."""

    name = "rosbridge"

    def __init__(self, transport: DuckTransport, *, robot_id: str | None = None) -> None:
        self.transport = transport
        self.backend = transport.name
        self.robot_id = robot_id or DEFAULT_ID
        self.manifest: RobotManifest | None = None

    async def connect(self) -> RobotManifest:
        await self.transport.connect()
        endpoint = getattr(self.transport, "endpoint", None)
        self.manifest = rosbridge_manifest(
            self.backend,
            self.robot_id,
            camera=bool(getattr(self.transport, "camera_available", False)),
            cmd_vel=getattr(endpoint, "cmd_vel", "/cmd_vel"),
            odom=getattr(endpoint, "odom", "/odom"),
            image=getattr(endpoint, "image", None),
            roslibpy_version=getattr(self.transport, "roslibpy_version", None),
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

    async def send_intent(self, intent: Intent) -> Ack:
        return await self.transport.send_intent(intent)

    async def health(self) -> Health:
        try:
            await self.transport.heartbeat()
        except HeartbeatError as e:
            return Health(ok=False, reason=str(e))
        state = await self.transport.get_state()
        return Health(ok=True, battery_percent=None, extras={"odom": state.extras.get("odom")})

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
        return {}

    def implementations(self) -> dict[str, Verb]:
        return {}  # every verb is a core verb here

    @property
    def mobility(self) -> str:
        return "wheeled"

    @property
    def post_sleep(self) -> Callable[[], None] | None:
        return getattr(self.transport, "post_sleep", None)

    @post_sleep.setter
    def post_sleep(self, hook: Callable[[], None] | None) -> None:
        self.transport.post_sleep = hook  # type: ignore[attr-defined]


# ── what the factory calls ──────────────────────────────────────────────────────────────


def describe(backend: str, robot_id: str | None = None) -> RobotManifest:
    """Static: the mock serves a frame; the ws backend has a camera only when the address
    names an image topic, which connect() finds out."""
    return rosbridge_manifest(backend, robot_id, camera=backend == "mock")


def implementations() -> dict[str, Verb]:
    return {}


def conditions() -> dict[str, Precondition]:
    return {}


def make(
    backend: str,
    *,
    robot_id: str | None = None,
    seed: int | None = None,
    address: str | None = None,
    live: bool = False,
    camera_url: str | None = None,
    token: str | None = None,
) -> RosbridgeAdapter:
    if backend == "mock":
        from quackd.adapters.rosbridge.mock import RosbridgeMock

        return RosbridgeAdapter(RosbridgeMock(), robot_id=robot_id)
    if backend == "ws":
        from quackd.adapters.rosbridge.ws import RosbridgeWs

        return RosbridgeAdapter(RosbridgeWs(address=address), robot_id=robot_id)
    raise ValueError(f"unknown rosbridge backend {backend!r}; choose one of {BACKENDS}")


__all__ = [
    "BACKENDS",
    "DEFAULT_ID",
    "RosbridgeAdapter",
    "conditions",
    "describe",
    "implementations",
    "make",
    "rosbridge_manifest",
]
