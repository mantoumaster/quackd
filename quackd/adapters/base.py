"""One interface for "a robot", whatever its body.

`RobotAdapter` is a superset of `DuckTransport`: everything 0.3 took a transport for
(the executor, the loop, the heartbeat, the MCP session, a flock member) accepts an
adapter unchanged. What an adapter adds is self-description: `connect()` returns a
`RobotManifest`, `preconditions()` names the checks its verbs need, `implementations()`
supplies the verbs only this robot has. `heartbeat()` stays the watchdog contract and
`health()` is the informational call (ADR-0017).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from PIL import Image

from quackd.adapters.manifest import Health, RobotManifest
from quackd.transport.base import Ack, DuckState, Intent, TransportError
from quackd.verbs.registry import Precondition, Verb


class AdapterError(TransportError):
    """Subclass of `TransportError` so every existing `except TransportError` still fires."""


class AdapterNotInstalled(AdapterError):
    """The adapter package, or the SDK one of its backends needs, is absent here.

    Both causes read the same on purpose, because the fix is the same command: the extra
    buys the adapter distribution and the SDK its real backend wants. The one case where
    that command can succeed and leave the SDK missing anyway is a python_version marker,
    which is why both hardware pages answer it in their troubleshooting tables."""

    def __init__(self, adapter: str, extra: str) -> None:
        super().__init__(f"adapter {adapter!r} needs an extra: uv pip install '{extra}'")
        self.adapter = adapter
        self.extra = extra


# ── the rest pose ───────────────────────────────────────────────────────────────────────

RestHow = Literal["none", "already", "arrived", "stalled", "timeout", "refused"]


@dataclass(frozen=True)
class RestResult:
    """What `go_to_rest()` did, as a value rather than an exception.

    Every caller is a teardown or the first moment of a run, and a teardown that raised
    would cost the body the disconnect it was in the middle of."""

    how: RestHow
    reason: str

    @property
    def reached(self) -> bool:
        """The body is at its rest pose now, whether or not it had to move."""
        return self.how in ("already", "arrived")

    @property
    def recorded(self) -> bool:
        """There was a rest pose to go to at all."""
        return self.how != "none"

    @classmethod
    def none(cls, reason: str = "this body has no rest pose") -> RestResult:
        return cls("none", reason)


HandHow = Literal["released", "held", "refused"]


@dataclass(frozen=True)
class HandResult:
    """What `let_go()` or `take_hold()` did, as a value rather than an exception.

    Both are asked for by a person standing at the robot, and both can fail for reasons that
    are not bugs: an arm away from the pose it may be released at, a register read that came
    back corrupt, a joint that moved while the hand was still on it. The caller has to be able
    to say which of those happened and then put the body down safely either way."""

    how: HandHow
    reason: str
    joints: dict[str, float] = field(default_factory=dict)
    """Where the arm was when this finished. Empty for a refusal that never read it."""

    @property
    def ok(self) -> bool:
        return self.how in ("released", "held")


async def let_go_if_any(transport: Any) -> HandResult:
    """Release the body into a person's hands, on anything that can be handed over.

    Duck-typed like `go_to_rest_if_any`, and for the same reason: one body out of seven does
    this, and the other six should not have to carry a method to say so."""
    hand = getattr(transport, "let_go", None)
    if not callable(hand):
        return HandResult("refused", "this body is not handed to a person")
    try:
        return await hand()
    except Exception as e:
        return HandResult("refused", f"{type(e).__name__}: {e}")


async def take_hold_if_any(transport: Any) -> HandResult:
    """Hold whatever pose the body is in now, so the person can let go of it.

    Never raises, because the caller of this is either starting a run or tearing one down,
    and in both cases what it does next depends on the answer rather than on an exception."""
    hold = getattr(transport, "take_hold", None)
    if not callable(hold):
        return HandResult("refused", "this body is not handed to a person")
    try:
        return await hold()
    except Exception as e:
        return HandResult("refused", f"{type(e).__name__}: {e}")


async def go_to_rest_if_any(transport: Any) -> RestResult:
    """The rest move on anything: an adapter, a bare transport, a traced wrapper.

    A transport with no `go_to_rest` is a body quackd does not park, which is most of
    them, so this costs a `getattr` rather than a method on every mock in the suite."""
    move = getattr(transport, "go_to_rest", None)
    if not callable(move):
        return RestResult.none()
    try:
        return await move()
    except Exception as e:  # a teardown never raises: the caller still has to disconnect
        return RestResult("refused", f"{type(e).__name__}: {e}")


def refuse_rest_pose(adapter: str, rest_pose: Any) -> None:
    """Refuse a rest pose a body cannot hold, rather than accepting and ignoring it.

    The only way one reaches an adapter that does not park is a hand-edited
    `robots.json`, and the registry's rule is that a file which says something untrue
    names itself rather than being quietly dropped."""
    if rest_pose is not None:
        raise AdapterError(
            f"{adapter} does not return to a rest pose, so one cannot be kept for it: "
            "quackd robot edit NAME --clear rest-pose"
        )


# ── cameras ─────────────────────────────────────────────────────────────────────────────

MULTI_CAMERA_SPECS = ("lerobot:real",)
"""The bodies that read more than one `--camera-url`. Every other `make()` refuses a
second one rather than opening the first and silently dropping the rest."""


def camera_urls(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """`--camera-url` however it arrived: nothing, one url, or several.

    Order is kept, because the first is the primary: the camera the detections describe,
    the one `--fov-deg` measures, and the only one a verb that steers by sight reads."""
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    return tuple(url.strip() for url in value if url and url.strip())


def one_camera_url(value: str | Sequence[str] | None, *, spec: str) -> str | None:
    """The single url a one-camera body takes, or a refusal that says who takes several."""
    urls = camera_urls(value)
    if len(urls) > 1:
        raise AdapterError(
            f"{spec} takes one --camera-url and {len(urls)} were given; "
            f"only {', '.join(MULTI_CAMERA_SPECS)} takes several"
        )
    return urls[0] if urls else None


@runtime_checkable
class RobotAdapter(Protocol):
    name: str
    """Adapter name: microduck, lerobot, ..."""
    backend: str
    """Backend name: sim2d, mock, jsonrpc, real, ..."""
    manifest: RobotManifest | None
    """None until connect()."""

    async def connect(self) -> RobotManifest: ...

    async def disconnect(self) -> None: ...

    async def close(self) -> None:
        """Same as `disconnect()`; kept so an adapter satisfies `DuckTransport`."""
        ...

    async def get_state(self) -> DuckState: ...

    async def get_frame(self) -> Image.Image | None:
        """The primary camera's newest picture, or None. A body with several cameras also
        offers `get_frames()`; `transport.frames_of()` is how callers ask for all of them."""
        ...

    async def send_intent(self, intent: Intent) -> Ack: ...

    async def health(self) -> Health:
        """Informational: doctor, robot_list, discovery. Never raises for a sick robot."""
        ...

    async def heartbeat(self) -> None:
        """The watchdog contract, unchanged: raise `HeartbeatError` to stop and abort."""
        ...

    async def stop(self) -> None: ...

    async def go_to_rest(self) -> RestResult:
        """Drive the body to the rest pose it was built with, if it has one.

        Never raises: this runs in teardowns, where the caller still has to disconnect.
        A body quackd does not park answers `RestResult.none()`."""
        ...

    # `let_go` and `take_hold` are deliberately NOT here. This protocol is runtime-checkable
    # and structural, so a method on it is a method every one of the seven bodies must carry,
    # and six of them are never handed to a person. A body that can be declares
    # `supports_hand_off = True`; the callers reach it through `let_go_if_any` and
    # `take_hold_if_any`, which is the same way a rest move reaches a bare transport.

    def subscribe(self, topic: str) -> AsyncIterator[dict[str, Any]]: ...

    def now(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...

    def preconditions(self) -> dict[str, Precondition]:
        """Condition name -> predicate over `DuckState` (a reason string, or None)."""
        ...

    def implementations(self) -> dict[str, Verb]:
        """Extension verbs and core overrides, keyed by canonical name."""
        ...


def backend_name(transport: Any) -> str:
    """The backend name: `sim2d` for a bare `Sim2DTransport` and for an adapter over one.

    The prompt's simulator note, the CLI's detector and recorder gating and the pinned
    `transport` keys in transcripts all key on this string."""
    backend = getattr(transport, "backend", None)
    return str(backend) if backend else str(getattr(transport, "name", "unknown"))


def adapter_name(transport: Any) -> str | None:
    """The adapter name, or None for a bare transport."""
    if getattr(transport, "backend", None):
        return str(getattr(transport, "name", None))
    return None
