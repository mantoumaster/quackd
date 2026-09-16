"""`--robot <adapter>:<backend>` -> a `RobotAdapter`, plus everything the CLI needs to talk
about adapters without connecting to one (static manifests, the status table).

Adapter packages are imported lazily, so listing adapters never imports an SDK.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from quackd.adapters.base import AdapterError, RobotAdapter, camera_urls
from quackd.adapters.manifest import RobotManifest
from quackd.verbs.registry import VerbRegistry, registry_from_manifest

DEFAULT_ROBOT = "microduck:sim2d"

# name -> (backends, status line, pip extra for the SDK backends, SDK import to probe)
_ADAPTERS: dict[str, tuple[tuple[str, ...], str, str | None, str | None]] = {
    # No extra and no probe, even though `mujoco` needs one: this column asks whether the
    # *adapter* is usable here, and the Microduck's is built in. Probing for mujoco made
    # `list-adapters` report the whole robot as missing on a machine that can still run
    # sim2d, mock and a real duck. The extra is named in the status line, in the transports
    # table and in `doctor`'s optional extras, which is where a per-backend answer belongs.
    "microduck": (
        ("sim2d", "mujoco", "mock", "jsonrpc", "websocket"),
        "✅ built-in: sim2d (default), mock · ✅ mujoco (physics, needs quackd[mujoco]) · "
        "🧪 jsonrpc · ⏳ websocket",
        None,
        None,
    ),
    "lerobot": (
        ("mock", "real"),
        "✅ built-in: mock · ✅ real (one SO-101 driven on 2026-09-15: the lookout, waves, "
        "the gripper and a webcam; Python 3.12+)",
        "quackd[lerobot]",
        "lerobot",
    ),
    "rosbridge": (
        ("mock", "ws"),
        "✅ built-in: mock · 🧪 ws via roslibpy (verified names, never run against a bridge)",
        "quackd[rosbridge]",
        "roslibpy",
    ),
    # Appended, never inserted: the doctor and list-adapters tables are order-sensitive, and
    # tests/test_adapters.py pins the order. No extra: the client is stdlib, and the robot's
    # own runtime is not installable here.
    "open_duck": (
        ("sim2d", "mock", "bridge"),
        "✅ built-in: sim2d, mock · 🧪 bridge (quackd's own daemon on the duck's Pi, "
        "never run on a robot)",
        None,
        None,
    ),
    # XLeRobot is not an installable package, so quackd speaks its ZeroMQ host protocol
    # rather than importing it: the extra is pyzmq and nothing else (ADR-0026).
    "xlerobot": (
        ("mock", "zmq"),
        "✅ built-in: mock · 🧪 zmq (wire format VERIFIED at a pinned commit, exercised "
        "against a fake host over loopback, never run on a cart)",
        "quackd[xlerobot]",
        "zmq",
    ),
    # Also not an installable package: a fork of LeRobot that calls itself lerobot and is not
    # on PyPI, so quackd speaks its ZeroMQ host protocol too (ADR-0027).
    "alohamini": (
        ("mock", "sim2d", "zmq"),
        "✅ built-in: mock, sim2d · 🧪 zmq (wire format VERIFIED at a pinned commit, "
        "exercised against a fake host over loopback, never run on a robot)",
        "quackd[alohamini]",
        "zmq",
    ),
    # No network API of any kind upstream: no socket, no daemon, no IPC. So quackd ships
    # the daemon, as it does for the Open Duck Mini, and the client is stdlib (ADR-0028).
    "toddlerbot": (
        ("mock", "sim2d", "bridge"),
        "✅ built-in: mock, sim2d · 🧪 bridge (quackd's own daemon on the robot, "
        "never run on a robot)",
        None,
        None,
    ),
}
ADAPTER_NAMES = tuple(_ADAPTERS)
BACKENDS = {name: info[0] for name, info in _ADAPTERS.items()}
ADAPTER_STATUS = {name: info[1] for name, info in _ADAPTERS.items()}
ADAPTER_EXTRAS = {name: info[2] for name, info in _ADAPTERS.items() if info[2]}

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


@dataclass(frozen=True)
class RobotSpec:
    adapter: str
    backend: str
    name: str | None = None
    """The member or fleet name (`duck=microduck:sim2d`), which becomes the manifest id."""

    @property
    def key(self) -> str:
        return f"{self.adapter}:{self.backend}"

    @property
    def robot_id(self) -> str | None:
        """The manifest id to ask for: the fleet name, or the adapter's own default."""
        return self.name


def parse_robot_spec(text: str) -> RobotSpec:
    """`microduck:sim2d`, or `microduck` (its first backend). Unknown names list the choices."""
    text = text.strip().lower()
    adapter, _, backend = text.partition(":")
    if adapter not in _ADAPTERS:
        raise AdapterError(f"unknown adapter {adapter!r}; choose one of {', '.join(ADAPTER_NAMES)}")
    backends = BACKENDS[adapter]
    backend = backend or backends[0]
    if backend not in backends:
        raise AdapterError(
            f"unknown backend {backend!r} for {adapter}; choose one of {', '.join(backends)}"
        )
    return RobotSpec(adapter, backend)


def parse_robots(text: str) -> list[RobotSpec]:
    """`duck=microduck:sim2d,arm=lerobot:mock` -> named specs, order preserved."""
    specs: list[RobotSpec] = []
    for item in [part.strip() for part in text.split(",") if part.strip()]:
        name, sep, spec_text = item.partition("=")
        if not sep or not _NAME_RE.match(name.strip()):
            raise AdapterError(f"{item!r} is not name=<adapter>:<backend> (name is a slug)")
        if any(s.name == name.strip() for s in specs):
            raise AdapterError(f"duplicate robot name {name.strip()!r}")
        spec = parse_robot_spec(spec_text)
        specs.append(RobotSpec(spec.adapter, spec.backend, name.strip()))
    if not specs:
        raise AdapterError("--robots needs at least one name=<adapter>:<backend>")
    return specs


def resolve_robot(robot: str | None, *, duck_default: str | None = None) -> RobotSpec:
    """`--robot` wins; without it, the duck's own `robots:` default, then `microduck:sim2d`."""
    if robot:
        return parse_robot_spec(robot)
    if duck_default:
        return parse_robot_spec(duck_default)
    return parse_robot_spec(DEFAULT_ROBOT)


def _module(adapter: str) -> Any:
    return importlib.import_module(f"quackd.adapters.{adapter}")


def describe(spec: RobotSpec) -> RobotManifest:
    """The static manifest: no SDK import, no socket. What `validate` and `announce` use."""
    return _module(spec.adapter).describe(spec.backend, spec.robot_id)


def registry_for(spec: RobotSpec) -> VerbRegistry:
    """The vocabulary of a robot that is not connected (`list-verbs --robot`, `--goal`)."""
    module = _module(spec.adapter)
    return registry_from_manifest(
        describe(spec), implementations=module.implementations(), conditions=module.conditions()
    )


def make_adapter(
    spec: RobotSpec | str,
    *,
    seed: int | None = None,
    address: str | None = None,
    live: bool = False,
    camera_url: str | Sequence[str] | None = None,
    token: str | None = None,
    rest_pose: Mapping[str, float] | None = None,
) -> RobotAdapter:
    """Build a robot. `camera_url` may name several cameras; every `make()` is handed the
    tuple and decides whether this body reads more than one (`MULTI_CAMERA_SPECS`), and a
    `rest_pose` reaches a body that parks or is refused by one that does not."""
    if isinstance(spec, str):
        spec = parse_robot_spec(spec)
    adapter: RobotAdapter = _module(spec.adapter).make(
        spec.backend,
        robot_id=spec.robot_id,
        seed=seed,
        address=address,
        live=live,
        camera_url=camera_urls(camera_url),
        token=token,
        rest_pose=dict(rest_pose) if rest_pose else None,
    )
    return adapter


def list_adapters() -> list[dict[str, Any]]:
    """Rows for `quackd list-adapters` and `doctor`, without importing any SDK."""
    rows = []
    for name, (backends, status, extra, probe) in _ADAPTERS.items():
        installed = True if probe is None else importlib.util.find_spec(probe) is not None
        rows.append(
            {
                "name": name,
                "backends": list(backends),
                "status": status,
                "extra": extra or "built-in",
                "installed": installed,
            }
        )
    return rows


def shipped_manifests() -> list[tuple[str, RobotManifest]]:
    """(adapter name, its static manifest) for every body quackd ships, in table order.

    The first backend of each, because a body is the same body on all of them. Static, so
    this costs no SDK import and no connection."""
    return [(name, describe(RobotSpec(name, BACKENDS[name][0]))) for name in ADAPTER_NAMES]


def bodies_that_could(needs: Mapping[str, Any]) -> list[tuple[str, RobotManifest, list[str]]]:
    """The shipped bodies whose own datasheets meet these needs, and what each other one lacks.

    Rows are (name, manifest, missing): an empty `missing` is a body that could be asked. Only
    those are returned, so a caller naming them is naming bodies, not hopes; the lacking ones
    come back in the same shape for a caller that wants to say why not."""
    from quackd.verdict import missing_needs

    rows = [(name, m, missing_needs(needs, m)) for name, m in shipped_manifests()]
    return [row for row in rows if not row[2]]
