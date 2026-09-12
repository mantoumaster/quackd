"""One interface for "a robot", whatever its body.

`RobotAdapter` is a superset of `DuckTransport`: everything 0.3 took a transport for
(the executor, the loop, the heartbeat, the MCP session, a flock member) accepts an
adapter unchanged. What an adapter adds is self-description: `connect()` returns a
`RobotManifest`, `preconditions()` names the checks its verbs need, `implementations()`
supplies the verbs only this robot has. `heartbeat()` stays the watchdog contract and
`health()` is the informational call (ADR-0017).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from PIL import Image

from quackd.adapters.manifest import Health, RobotManifest
from quackd.transport.base import Ack, DuckState, Intent, TransportError
from quackd.verbs.registry import Precondition, Verb


class AdapterError(TransportError):
    """Subclass of `TransportError` so every existing `except TransportError` still fires."""


class AdapterNotInstalled(AdapterError):
    """The adapter's SDK is an extra that is not installed here."""

    def __init__(self, adapter: str, extra: str) -> None:
        super().__init__(f"adapter {adapter!r} needs an extra: uv pip install '{extra}'")
        self.adapter = adapter
        self.extra = extra


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

    async def get_frame(self) -> Image.Image | None: ...

    async def send_intent(self, intent: Intent) -> Ack: ...

    async def health(self) -> Health:
        """Informational: doctor, robot_list, discovery. Never raises for a sick robot."""
        ...

    async def heartbeat(self) -> None:
        """The watchdog contract, unchanged: raise `HeartbeatError` to stop and abort."""
        ...

    async def stop(self) -> None: ...

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
