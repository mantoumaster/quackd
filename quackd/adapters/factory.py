"""`--robot <adapter>:<backend>` -> a `RobotAdapter`, plus everything the CLI needs to talk
about adapters without connecting to one (static manifests, the status table).

Adapter packages are imported lazily, so listing adapters never imports an SDK.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from quackd.adapters.base import AdapterError, AdapterNotInstalled, RobotAdapter, camera_urls
from quackd.adapters.catalogue import BY_NAME, ENTRY_POINT_GROUP, OFFICIAL, AdapterInfo
from quackd.adapters.manifest import RobotManifest
from quackd.verbs.registry import VerbRegistry, registry_from_manifest

DEFAULT_ROBOT = "microduck:sim2d"
"""The body a command falls back to when nothing named one and several are installed.

It survives because `find-and-kick` and the other v0 starters carry no `robots:` line and a
reader who types `quackd run find-and-kick` means the cartoon. Where exactly one adapter is
installed, that one is the default instead, and where none is, there is no default at all."""


@lru_cache(maxsize=1)
def _installed() -> dict[str, str]:
    """Every adapter that can be built here, name to the module that builds it.

    Two ways in, and they are the same question asked before and after the adapters became
    their own distributions. An installed adapter announces itself through the entry point
    group, which is how a third party's is found and the only way one can be. An adapter
    that still lives inside the core wheel is found by looking, because it is installed by
    virtue of being here at all, and asking the metadata about it would answer only whether
    somebody had reinstalled since the entry points were declared.

    Cached: `entry_points()` walks the whole environment, and this is on the path of every
    `doctor`. Nothing installs an adapter mid-process."""
    found: dict[str, str] = {
        ep.name: ep.value for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    }
    for name in BY_NAME:
        if name in found:
            continue
        module = f"quackd.adapters.{name}"
        if importlib.util.find_spec(module) is not None:
            found[name] = module
    return found


def adapter_names() -> tuple[str, ...]:
    """Every adapter quackd can talk about: the seven it publishes, in their fixed order,
    then anything else installed here, alphabetically. A name quackd publishes keeps its
    place whether or not it is installed, because the tables that list them are a catalogue
    rather than an inventory."""
    third_party = sorted(name for name in _installed() if name not in BY_NAME)
    return tuple(BY_NAME) + tuple(third_party)


def info(name: str) -> AdapterInfo:
    """What quackd knows about this adapter without importing it.

    A third party's adapter has no catalogue row, so its own module is asked instead, which
    means importing it. That is fine: it is installed, or this raises anyway."""
    if (known := BY_NAME.get(name)) is not None:
        return known
    module = _module(name)
    return AdapterInfo(
        name=name,
        backends=tuple(getattr(module, "BACKENDS", ())),
        status=str(getattr(module, "STATUS", "installed here, not published by quackd")),
        summary=str(getattr(module, "SUMMARY", "a robot quackd does not publish")),
        extra=getattr(module, "EXTRA", None),
        sdk=getattr(module, "SDK", None),
    )


def is_installed(name: str) -> bool:
    return name in _installed()


def is_official(name: str) -> bool:
    return name in BY_NAME


ADAPTER_NAMES = tuple(BY_NAME)
BACKENDS = {i.name: i.backends for i in OFFICIAL}
ADAPTER_STATUS = {i.name: i.status for i in OFFICIAL}
ADAPTER_EXTRAS = {i.name: i.extra for i in OFFICIAL if i.extra}

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
    """`microduck:sim2d`, or `microduck` (its first backend). Unknown names list the choices.

    A name quackd publishes parses whether or not it is installed here, so a robot can be
    registered, listed and printed on a machine that cannot build it. Asking for the body
    itself is what refuses."""
    text = text.strip().lower()
    adapter, _, backend = text.partition(":")
    known = adapter_names()
    if adapter not in known:
        raise AdapterError(f"unknown adapter {adapter!r}; choose one of {', '.join(known)}")
    backends = info(adapter).backends
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
    """The module that builds this robot, or a refusal naming what to install.

    Imported here and nowhere earlier, so listing adapters, parsing a spec and printing a
    registry all work on a machine where the adapter is not installed at all."""
    where = _installed().get(adapter)
    if where is None:
        if (row := BY_NAME.get(adapter)) is not None:
            raise AdapterNotInstalled(adapter, row.extra or f"quackd[{adapter}]")
        known = ", ".join(adapter_names())
        raise AdapterError(f"unknown adapter {adapter!r}; installed here: {known}")
    return importlib.import_module(where)


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
    """Rows for `quackd list-adapters` and `doctor`, without importing any SDK.

    `installed` is whether the adapter itself is here, and `sdk` whether the library its
    real backend needs is. They are separate questions: an adapter with a mock is useful
    with no SDK at all, and reporting the whole robot missing because a wheel it only needs
    for hardware is absent told a simulator user their duck was gone."""
    rows = []
    for name in adapter_names():
        row = info(name)
        sdk = None if row.sdk is None else importlib.util.find_spec(row.sdk) is not None
        rows.append(
            {
                "name": name,
                "backends": list(row.backends),
                "status": row.status,
                "extra": row.extra or "built-in",
                "official": is_official(name),
                # an adapter with no SDK to probe is usable as soon as it is here at all
                "installed": is_installed(name) and sdk is not False,
                "adapter_installed": is_installed(name),
                "sdk": sdk,
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
