"""The robots you have named, and the flocks you made of them.

Not the *verb* registry (`quackd/verbs/registry.py`), which is a robot's vocabulary. This is
the other half of the word: a name for a body plus how to reach it, so `--robot duck-a` means
the same thing in every command instead of five flags retyped on every line, one of them a
secret that then lives in shell history.

Two files under one directory, `robots.json` and `flocks.json`. A flock is a list of robot
names, which is why they share a directory and a version: a flock that names a robot nobody
registered is the one broken state either file can be in, and it is reported rather than
repaired.

Reads are strict, unlike `RobotMemory`'s. Memory is notes a model wrote and a bad line there
is worth skipping; this is configuration a person wrote, and a silently-dropped field would
send a robot to the wrong address. A typo names its own field and stops the command.

Writes are the same trick memory uses: a temporary file renamed over the old one, so a reader
never sees half a file. Also the same accepted cost: nothing is serialised, so two commands
writing at the same instant are a read-modify-write race and the later rename wins. That is
the price of a file you can open in an editor, and it is the right price for something a
person edits by hand once a week (ADR-0034).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from quackd.adapters.base import AdapterError
from quackd.adapters.factory import (
    ADAPTER_NAMES,
    BACKENDS,
    DEFAULT_ROBOT,
    RobotSpec,
    parse_robot_spec,
)
from quackd.memory import robot_slug

DEFAULT_DIR = "~/.quackd"
ENV_DIR = "QUACKD_REGISTRY_DIR"
ROBOTS_FILE = "robots.json"
FLOCKS_FILE = "flocks.json"
VERSION = 1

MAX_MEMBERS = 8
"""What a stored flock may hold. The run decides what it may *run*: an auction takes four."""

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
"""The same slug every member name and fleet name in quackd already is (`factory._NAME_RE`)."""


def registry_dir(override: str | Path | None = None) -> Path:
    """`--registry-dir`, else `$QUACKD_REGISTRY_DIR`, else `~/.quackd`."""
    raw = override or os.environ.get(ENV_DIR) or DEFAULT_DIR
    return Path(raw).expanduser()


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _reserved_slugs() -> set[str]:
    """Every memory file name an ad-hoc `adapter:backend` run would open.

    A robot registered as `microduck-sim2d` would key its memory to the same file a bare
    `--robot microduck:sim2d` run does, which is exactly the confusion registering it was
    meant to end. So the name is refused (ADR-0034 amends ADR-0025)."""
    return {robot_slug(f"{a}:{b}") for a, backends in BACKENDS.items() for b in backends}


class RegistryError(Exception):
    """Anything the registry refuses. The CLI prints it as one line, never a traceback."""


class UnknownRobot(RegistryError):
    pass


class UnknownFlock(RegistryError):
    pass


class RobotInUse(RegistryError):
    def __init__(self, name: str, flocks: Sequence[str]) -> None:
        joined = ", ".join(flocks)
        super().__init__(
            f"{name} is in {'flock' if len(flocks) == 1 else 'flocks'} {joined}: "
            f"quackd flock edit {flocks[0]} --remove {name}, or --force to drop it from "
            f"{'it' if len(flocks) == 1 else 'them'} too"
        )
        self.name = name
        self.flocks = list(flocks)


class FlockNotRunnable(RegistryError):
    pass


def check_name(name: str, *, kind: str = "robot") -> str:
    """A name a person types on a command line, and never anything else it could be mistaken
    for: a number (`--flock 3` means three simulated ducks), an adapter (`--robot microduck`
    already means one), or a memory file some ad-hoc run already owns."""
    if not _NAME_RE.match(name):
        raise RegistryError(
            f"{name!r} is not a valid {kind} name: lowercase letters, digits and hyphens, "
            "starting with a letter or digit, 64 characters at most"
        )
    if name.isdigit():
        raise RegistryError(
            f"{name!r} cannot be a {kind} name: a number after --flock means that many "
            "simulated ducks"
        )
    if kind == "robot":
        if name in ADAPTER_NAMES:
            raise RegistryError(
                f"{name!r} is an adapter, and --robot {name} already means "
                f"{name}:{BACKENDS[name][0]}: pick another name"
            )
        if name in _reserved_slugs():
            raise RegistryError(
                f"{name!r} is the memory file name of an unregistered robot: pick another name"
            )
    return name


class RobotEntry(BaseModel):
    """One robot you have named: which body, where it is, and who pilots it."""

    model_config = ConfigDict(extra="forbid")

    name: str
    spec: str
    """`<adapter>:<backend>`, normalised: `microduck` is stored as `microduck:sim2d`."""
    address: str | None = None
    token: str | None = None
    camera_url: str | None = None
    provider: str | None = None
    """The provider a run uses for this robot when `--provider` is absent."""
    model: str | None = None
    """Its model id. `--model` beats this, this beats `QUACKD_MODEL`."""
    note: str | None = None
    added: str = Field(default_factory=_now)
    updated: str = Field(default_factory=_now)

    @field_validator("spec")
    @classmethod
    def _spec(cls, value: str) -> str:
        try:
            return parse_robot_spec(value).key
        except AdapterError as e:
            # re-raised as a ValueError so pydantic folds it into the one-line message that
            # names which robot in the file is wrong; the wording stays the adapter's own
            raise ValueError(str(e)) from e

    @field_validator("provider")
    @classmethod
    def _provider(cls, value: str | None) -> str | None:
        if value is None:
            return None
        from quackd.agent.providers.catalogue import PROVIDER_NAMES

        folded = value.strip().lower()
        if folded not in PROVIDER_NAMES:
            raise ValueError(f"unknown provider {value!r}; one of {', '.join(PROVIDER_NAMES)}")
        return folded

    @property
    def adapter(self) -> str:
        return self.spec.split(":", 1)[0]

    @property
    def backend(self) -> str:
        return self.spec.split(":", 1)[1]

    @property
    def key(self) -> str:
        return self.spec

    @property
    def robot_spec(self) -> RobotSpec:
        """`RobotSpec` carries the name, so the manifest id is the name you registered, which
        is what `--robots name=spec` has always done for a fleet."""
        return RobotSpec(self.adapter, self.backend, self.name)

    @property
    def memory_key(self) -> str:
        return self.name

    def adapter_kwargs(
        self,
        *,
        address: str | None = None,
        camera_url: str | None = None,
        token: str | None = None,
    ) -> dict[str, str | None]:
        """The `make_adapter` keywords for this robot. A flag on the command line wins: you
        are reaching the same robot through a tunnel today, not renaming it."""
        return {
            "address": address or self.address,
            "camera_url": camera_url or self.camera_url,
            "token": token or self.token,
        }

    def public(self) -> dict[str, Any]:
        """The `--json` shape. The token is never printed, only whether there is one."""
        return {
            "name": self.name,
            "spec": self.spec,
            "adapter": self.adapter,
            "backend": self.backend,
            "address": self.address,
            "camera_url": self.camera_url,
            "token_set": self.token is not None,
            "provider": self.provider,
            "model": self.model,
            "note": self.note,
            "added": self.added,
            "updated": self.updated,
        }


class StoredFlock(BaseModel):
    """A named group of registered robots. Running one needs 2 to 8 of them; storing fewer is
    allowed, because that is what a flock looks like while you are still building it."""

    model_config = ConfigDict(extra="forbid")

    name: str
    members: list[str] = Field(default_factory=list, max_length=MAX_MEMBERS)
    description: str | None = None
    created: str = Field(default_factory=_now)
    updated: str = Field(default_factory=_now)

    @field_validator("members")
    @classmethod
    def _members(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        for name in value:
            if not _NAME_RE.match(name):
                raise ValueError(f"{name!r} is not a valid robot name")
            if name in seen:
                raise ValueError(f"{name!r} is listed twice")
            seen.add(name)
        return value

    def public(self, missing: Sequence[str] = ()) -> dict[str, Any]:
        return {
            "name": self.name,
            "members": list(self.members),
            "missing": list(missing),
            "runnable": not missing and 2 <= len(self.members) <= MAX_MEMBERS,
            "description": self.description,
            "created": self.created,
            "updated": self.updated,
        }


@dataclass(frozen=True)
class Resolved:
    """What `--robot <something>` turned out to mean: a registered robot, or a bare spec."""

    spec: RobotSpec
    entry: RobotEntry | None = None

    @property
    def registered(self) -> bool:
        return self.entry is not None

    @property
    def memory_key(self) -> str:
        return self.entry.memory_key if self.entry is not None else self.spec.key

    @property
    def label(self) -> str:
        """What a header and a status line call it: the name, and the body under it."""
        return f"{self.entry.name} ({self.spec.key})" if self.entry is not None else self.spec.key

    @property
    def provider(self) -> str | None:
        return self.entry.provider if self.entry is not None else None

    @property
    def model(self) -> str | None:
        return self.entry.model if self.entry is not None else None

    def adapter_kwargs(
        self,
        *,
        address: str | None = None,
        camera_url: str | None = None,
        token: str | None = None,
    ) -> dict[str, str | None]:
        if self.entry is not None:
            return self.entry.adapter_kwargs(address=address, camera_url=camera_url, token=token)
        return {"address": address, "camera_url": camera_url, "token": token}


class Registry:
    """Both files, under one directory. Every method re-reads the file it needs, so a `run`
    and a `quackd robot edit` in another terminal always see each other's latest writes."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base = registry_dir(base_dir)
        self.robots_path = self.base / ROBOTS_FILE
        self.flocks_path = self.base / FLOCKS_FILE

    # ── storage ─────────────────────────────────────────────────────────────────────

    def _read(self, path: Path, key: str) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise RegistryError(
                f"{path}: not valid JSON ({e.msg} at line {e.lineno}): fix it or move it aside"
            ) from e
        if not isinstance(loaded, dict):
            raise RegistryError(f"{path}: expected an object, found {type(loaded).__name__}")
        version = loaded.get("version", VERSION)
        if not isinstance(version, int) or version > VERSION:
            raise RegistryError(
                f"{path} is version {version}, and this quackd reads version {VERSION}: "
                "upgrade quackd"
            )
        section = loaded.get(key) or {}
        if not isinstance(section, dict):
            raise RegistryError(f"{path}: {key} must be an object of name -> entry")
        return section

    def _write(self, path: Path, key: str, section: Mapping[str, dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": VERSION, key: {k: section[k] for k in sorted(section)}}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        try:
            tmp.replace(path)
        except PermissionError as e:  # Windows: another program is holding the file open
            raise RegistryError(f"could not replace {path}: another program has it open") from e
        if key == "robots":
            # it holds robot tokens. A no-op on Windows, which is why SECURITY.md says this
            # is a file in a home directory rather than a secret store.
            with contextlib.suppress(OSError):
                path.chmod(0o600)

    @staticmethod
    def _one_line(path: Path, name: str, e: ValidationError) -> RegistryError:
        why = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
        )
        return RegistryError(f"{path.name}: {name}: {why}")

    # ── robots ──────────────────────────────────────────────────────────────────────

    def robots(self) -> dict[str, RobotEntry]:
        """Every registered robot, by name, sorted."""
        out: dict[str, RobotEntry] = {}
        for name, raw in sorted(self._read(self.robots_path, "robots").items()):
            try:
                out[name] = RobotEntry.model_validate({**raw, "name": name})
            except ValidationError as e:
                raise self._one_line(self.robots_path, name, e) from e
        return out

    def get_robot(self, name: str) -> RobotEntry | None:
        return self.robots().get(name)

    def robot(self, name: str) -> RobotEntry:
        entry = self.get_robot(name)
        if entry is None:
            raise UnknownRobot(f"no robot called {name!r} is registered")
        return entry

    def _save_robots(self, entries: Mapping[str, RobotEntry]) -> None:
        self._write(
            self.robots_path,
            "robots",
            {n: e.model_dump(exclude={"name"}) for n, e in entries.items()},
        )

    def add_robot(self, entry: RobotEntry) -> RobotEntry:
        check_name(entry.name, kind="robot")
        entries = self.robots()
        if entry.name in entries:
            raise RegistryError(
                f"{entry.name!r} is already registered ({entries[entry.name].key}): "
                f"quackd robot edit {entry.name}, or remove it first"
            )
        entries[entry.name] = entry
        self._save_robots(entries)
        return entry

    def update_robot(self, name: str, changes: Mapping[str, str | None]) -> RobotEntry:
        """Set or clear fields. A key mapped to None clears it; a key absent is left alone."""
        entries = self.robots()
        current = entries.get(name)
        if current is None:
            raise UnknownRobot(f"no robot called {name!r} is registered")
        data = current.model_dump()
        data.update(changes)
        data["updated"] = _now()
        try:
            entry = RobotEntry.model_validate(data)
        except ValidationError as e:
            raise self._one_line(self.robots_path, name, e) from e
        entries[name] = entry
        self._save_robots(entries)
        return entry

    def remove_robot(self, name: str, *, force: bool = False) -> list[str]:
        """Forget a robot. Returns the flocks it was dropped from (empty unless `force`)."""
        entries = self.robots()
        if name not in entries:
            raise UnknownRobot(f"no robot called {name!r} is registered")
        holding = self.flocks_of(name)
        if holding and not force:
            raise RobotInUse(name, holding)
        del entries[name]
        self._save_robots(entries)
        if holding:
            flocks = self.flocks()
            for flock_name in holding:
                flock = flocks[flock_name]
                flocks[flock_name] = flock.model_copy(
                    update={
                        "members": [m for m in flock.members if m != name],
                        "updated": _now(),
                    }
                )
            self._save_flocks(flocks)
        return holding

    def flocks_of(self, name: str) -> list[str]:
        return [f.name for f in self.flocks().values() if name in f.members]

    # ── flocks ──────────────────────────────────────────────────────────────────────

    def flocks(self) -> dict[str, StoredFlock]:
        out: dict[str, StoredFlock] = {}
        for name, raw in sorted(self._read(self.flocks_path, "flocks").items()):
            try:
                out[name] = StoredFlock.model_validate({**raw, "name": name})
            except ValidationError as e:
                raise self._one_line(self.flocks_path, name, e) from e
        return out

    def get_flock(self, name: str) -> StoredFlock | None:
        return self.flocks().get(name)

    def flock(self, name: str) -> StoredFlock:
        found = self.get_flock(name)
        if found is None:
            raise UnknownFlock(f"no flock called {name!r}")
        return found

    def _save_flocks(self, flocks: Mapping[str, StoredFlock]) -> None:
        self._write(
            self.flocks_path,
            "flocks",
            {n: f.model_dump(exclude={"name"}) for n, f in flocks.items()},
        )

    def add_flock(self, flock: StoredFlock) -> StoredFlock:
        check_name(flock.name, kind="flock")
        flocks = self.flocks()
        if flock.name in flocks:
            raise RegistryError(f"a flock called {flock.name!r} already exists")
        known = self.robots()
        for member in flock.members:
            if member not in known:
                raise UnknownRobot(f"no robot called {member!r} is registered: quackd robot list")
        flocks[flock.name] = flock
        self._save_flocks(flocks)
        return flock

    def update_flock(
        self,
        name: str,
        *,
        add: Sequence[str] = (),
        remove: Sequence[str] = (),
        description: str | None = None,
        rename: str | None = None,
    ) -> StoredFlock:
        """`description=""` clears it, None leaves it. Removing a member that is no longer
        registered is allowed: that is how a dangling flock is repaired."""
        flocks = self.flocks()
        flock = flocks.get(name)
        if flock is None:
            raise UnknownFlock(f"no flock called {name!r}")
        known = self.robots()
        members = list(flock.members)
        for member in remove:
            if member not in members:
                raise RegistryError(f"{member!r} is not in {name}")
            members.remove(member)
        for member in add:
            if member not in known:
                raise UnknownRobot(f"no robot called {member!r} is registered: quackd robot list")
            if member in members:
                raise RegistryError(f"{member!r} is already in {name}")
            members.append(member)
        if len(members) > MAX_MEMBERS:
            raise RegistryError(
                f"a flock holds at most {MAX_MEMBERS} robots; that would be {len(members)}"
            )
        data = flock.model_dump()
        data["members"] = members
        if description is not None:
            data["description"] = description or None
        data["updated"] = _now()
        if rename is not None and rename != name:
            check_name(rename, kind="flock")
            if rename in flocks:
                raise RegistryError(f"a flock called {rename!r} already exists")
            data["name"] = rename
            del flocks[name]
            name = rename
        try:
            updated = StoredFlock.model_validate(data)
        except ValidationError as e:
            raise self._one_line(self.flocks_path, name, e) from e
        flocks[name] = updated
        self._save_flocks(flocks)
        return updated

    def delete_flock(self, name: str) -> StoredFlock:
        flocks = self.flocks()
        flock = flocks.pop(name, None)
        if flock is None:
            raise UnknownFlock(f"no flock called {name!r}")
        self._save_flocks(flocks)
        return flock

    def missing_members(self, flock: StoredFlock) -> list[str]:
        known = self.robots()
        return [m for m in flock.members if m not in known]

    def roster(self, name: str) -> dict[str, RobotEntry]:
        """A flock's robots, in the order it lists them, ready to run.

        Refuses rather than quietly running a smaller flock: a member you registered and then
        removed is a robot you meant to be there."""
        flock = self.flock(name)
        if not flock.members:
            raise FlockNotRunnable(
                f"flock {name!r} has no robots: quackd flock edit {name} --add NAME"
            )
        missing = self.missing_members(flock)
        if missing:
            joined = ", ".join(missing)
            raise FlockNotRunnable(
                f"flock {name!r} names {joined}, which "
                f"{'is' if len(missing) == 1 else 'are'} no longer registered: "
                f"quackd robot add {missing[0]} <adapter>:<backend>, or "
                f"quackd flock edit {name} --remove {missing[0]}"
            )
        known = self.robots()
        return {m: known[m] for m in flock.members}


def resolve_robot_ref(
    text: str | None,
    registry: Registry | None = None,
    *,
    duck_default: str | None = None,
) -> Resolved:
    """What `--robot <something>` means here: a registered name, or `<adapter>[:<backend>]`.

    A slug cannot contain a colon and a spec always reads as one when it has a backend, so the
    two vocabularies never collide; a bare `microduck` is an adapter because that is what it
    has always been, and `check_name` refuses registering a robot under an adapter's name."""
    reg = registry if registry is not None else Registry()
    if text is None:
        return Resolved(parse_robot_spec(duck_default or DEFAULT_ROBOT))
    text = text.strip()
    if ":" not in text:
        entry = reg.get_robot(text)
        if entry is not None:
            return Resolved(entry.robot_spec, entry)
    try:
        return Resolved(parse_robot_spec(text))
    except AdapterError as e:
        if ":" in text:
            raise
        raise AdapterError(
            f"{text!r} is neither a registered robot (quackd robot list) nor "
            f"<adapter>[:<backend>]: {e}"
        ) from e


# ── probing ─────────────────────────────────────────────────────────────────────────────

PROBE_TIMEOUT_S = 5.0
_UNPROBEABLE = {"mujoco": "loads the physics model"}
"""Backends `--probe` declines to reach. Connecting to one fetches a model from upstream,
which is a download, not a liveness check."""


@dataclass(frozen=True)
class ProbeResult:
    reachable: bool | None
    """True, False, or None for a backend this deliberately does not reach."""
    detail: str
    elapsed_s: float


async def probe_entry(entry: RobotEntry, *, timeout_s: float = PROBE_TIMEOUT_S) -> ProbeResult:
    """Connect, ask how it is, and close. Never raises: an unreachable robot is an answer."""
    import asyncio
    import time

    from quackd.adapters.factory import make_adapter

    if (why := _UNPROBEABLE.get(entry.backend)) is not None:
        return ProbeResult(None, f"skipped: {why}", 0.0)
    started = time.perf_counter()

    def elapsed() -> float:
        return round(time.perf_counter() - started, 3)

    adapter: Any = None
    try:
        adapter = make_adapter(
            entry.robot_spec,
            seed=0,
            address=entry.address,
            camera_url=entry.camera_url,
            token=entry.token,
        )
        await asyncio.wait_for(adapter.connect(), timeout=timeout_s)
        health = await asyncio.wait_for(adapter.health(), timeout=timeout_s)
        detail = health.reason or "ok"
        if health.battery_percent is not None:
            detail = f"{detail}, battery {health.battery_percent:.0f}%"
        return ProbeResult(bool(health.ok), detail, elapsed())
    except TimeoutError:
        return ProbeResult(False, f"timed out after {timeout_s:g} s", elapsed())
    except Exception as e:  # an unreachable robot must not end the command
        first = str(e).splitlines()[0] if str(e) else type(e).__name__
        return ProbeResult(False, first, elapsed())
    finally:
        if adapter is not None:
            with contextlib.suppress(Exception):
                await adapter.close()


def probe_all(
    entries: Iterable[RobotEntry], *, timeout_s: float = PROBE_TIMEOUT_S
) -> dict[str, ProbeResult]:
    """Every robot at once, because they are separate machines and waiting serially is rude."""
    import asyncio

    rows = list(entries)

    async def run() -> list[ProbeResult]:
        return list(await asyncio.gather(*(probe_entry(e, timeout_s=timeout_s) for e in rows)))

    return dict(zip((e.name for e in rows), asyncio.run(run()), strict=True))


__all__ = [
    "DEFAULT_DIR",
    "ENV_DIR",
    "FLOCKS_FILE",
    "MAX_MEMBERS",
    "PROBE_TIMEOUT_S",
    "ROBOTS_FILE",
    "VERSION",
    "FlockNotRunnable",
    "ProbeResult",
    "Registry",
    "RegistryError",
    "Resolved",
    "RobotEntry",
    "RobotInUse",
    "StoredFlock",
    "UnknownFlock",
    "UnknownRobot",
    "check_name",
    "probe_all",
    "probe_entry",
    "registry_dir",
    "resolve_robot_ref",
]
