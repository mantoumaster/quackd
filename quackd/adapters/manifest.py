"""What a connected robot is and can do, as data.

A `RobotManifest` is what every adapter returns from `connect()`: pydantic in code, JSON on
the wire (MCP, mDNS, the flock bus), never YAML on disk. It decides *which* verbs exist on
a robot and how they are gated; the adapter and `verbs/core.py` decide *how* they run. A
verb that is not in the manifest does not exist anywhere in quackd (ADR-0017).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quackd.verbs.aliases import ALIASES
from quackd.verbs.registry import SafetyClass, Verb

MANIFEST_VERSION = 1

Embodiment = Literal["biped", "quadruped", "wheeled", "arm", "humanoid"]
Mobility = Literal["none", "legged", "wheeled"]
IntentName = Literal["twist", "skill", "gaze", "sound", "joint", "pose", "gripper"]
Sensor = Literal["camera", "battery", "odometry", "imu", "tof", "microphone", "joint_state"]
NativeSafety = Literal["robotd_deadman", "lease", "torque_limit", "estop", "none"]

# The manifest speaks the vocabulary other systems read; the transports speak the intent
# kinds quackd has used since 0.1. One table maps between them.
INTENT_KIND_FOR: dict[str, str] = {
    "twist": "move",
    "skill": "do",
    "gaze": "look",
    "sound": "sound",
    "joint": "joint",
    "pose": "pose",
    "gripper": "gripper",
}

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class VerbSpec(BaseModel):
    """One verb a robot provides. `name` is canonical (never an alias)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    core: bool = Field(default=False, description="A core verb: the same on every robot.")
    description: str = Field(
        default="", description="LLM-facing text. Empty means the implementation's default."
    )
    params_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON schema of the parameters (informational)."
    )
    safety_class: SafetyClass = "safe"
    timeout_s: float | None = Field(default=None, gt=0, le=600)

    @field_validator("name")
    @classmethod
    def _slug(cls, value: str) -> str:
        if not _SLUG_RE.match(value):
            raise ValueError(f"{value!r} is not a valid verb name")
        return value


class SafetyAuthority(BaseModel):
    """Who stops the body when quackd goes quiet. Honesty matters more than the enum."""

    model_config = ConfigDict(extra="forbid")

    native: NativeSafety = "none"
    deadman: bool = Field(default=False, description="The robot zeroes motion on silence.")
    heartbeat_hz: float = Field(default=2.0, gt=0, le=50)


class Frame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference: Literal["body", "head", "base", "world"] = "body"
    note: str = ""


class Health(BaseModel):
    """The informational liveness call. The watchdog contract stays `heartbeat()`."""

    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    reason: str | None = None
    battery_percent: float | None = None
    extras: dict[str, Any] = Field(default_factory=dict)


class RobotManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", title="quackd robot manifest v1")

    manifest: Literal[1] = Field(default=1, description="Manifest schema version.")
    id: str = Field(..., description="Slug, unique within a run or flock (e.g. arm-01).")
    vendor: str
    model: str
    embodiment: Embodiment
    mobility: Mobility
    intents: list[IntentName]
    sensors: list[Sensor] = Field(default_factory=list)
    verbs: list[VerbSpec]
    preconditions: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Verb -> condition names. The adapter supplies the predicates by name.",
    )
    safety_authority: SafetyAuthority = Field(default_factory=SafetyAuthority)
    frame: Frame = Field(default_factory=Frame)
    limits: dict[str, float] = Field(default_factory=dict)
    backend: str = Field(default="", description="Which backend produced this (informational).")
    blurb: str = Field(default="", description="Prompt intro: 'a small biped duck robot ...'.")
    extras: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _slug(cls, value: str) -> str:
        if not _SLUG_RE.match(value):
            raise ValueError("id must match ^[a-z0-9][a-z0-9_-]{0,63}$")
        return value

    @field_validator("intents", "sensors")
    @classmethod
    def _unique(cls, values: list[Any]) -> list[Any]:
        if len(set(values)) != len(values):
            raise ValueError(f"duplicates in {values}")
        return values

    @model_validator(mode="after")
    def _invariants(self) -> RobotManifest:
        from quackd.verbs.core import core_requirements_unmet  # lazy: core imports this module

        names = [v.name for v in self.verbs]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate verbs in manifest {self.id!r}")
        for v in self.verbs:
            if v.name in ALIASES:
                raise ValueError(f"declare {ALIASES[v.name]!r}, not its alias {v.name!r}")
        if "stop" not in names:
            # stop is universal: always present, always allowed, never gated
            self.verbs.append(VerbSpec(name="stop", core=True))
            names.append("stop")
        for v in self.verbs:
            if v.name == "stop" and v.safety_class != "safe":
                raise ValueError("stop can never be gated")
            if v.core:
                unmet = core_requirements_unmet(v.name, self)
                if unmet:
                    raise ValueError(f"{self.id}: core verb {v.name!r} {unmet}")
        for verb in self.preconditions:
            if verb not in names:
                raise ValueError(f"preconditions reference an undeclared verb {verb!r}")
        return self

    # ── queries ─────────────────────────────────────────────────────────────────────

    def verb_names(self) -> list[str]:
        return [v.name for v in self.verbs]

    def verb(self, name: str) -> VerbSpec | None:
        """Alias-aware lookup: `verb("walk")` is the `move` spec when `move` is declared."""
        wanted = {name, ALIASES.get(name, name)}
        return next((v for v in self.verbs if v.name in wanted), None)

    def provides(self, name: str) -> bool:
        return self.verb(name) is not None

    def digest(self) -> str:
        """A capability fingerprint: the same robot over sim2d and mock hashes the same.

        `id` and `backend` are excluded on purpose; discovery carries the id separately."""
        payload = self.model_dump(mode="json", exclude={"id", "backend"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def summary(self) -> str:
        return (
            f"{self.model} ({self.embodiment}, mobility {self.mobility}) "
            f"{len(self.verbs)} verbs: {', '.join(self.verb_names())}"
        )


def verb_spec(
    verb: Verb,
    *,
    core: bool,
    description: str | None = None,
    safety_class: SafetyClass | None = None,
    timeout_s: float | None = None,
) -> VerbSpec:
    """A manifest entry for an implementation template (schema derived from its params)."""
    schema = verb.params.model_json_schema()
    schema.pop("title", None)
    return VerbSpec(
        name=verb.name,
        core=core,
        description=description if description is not None else "",
        params_schema=schema,
        safety_class=safety_class or verb.safety_class,
        timeout_s=timeout_s,
    )


def manifest_json_schema() -> dict[str, Any]:
    """The JSON Schema for a manifest, as exported to `manifest.schema.json`."""
    schema = RobotManifest.model_json_schema()
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/rokbenko/quackd/blob/main/quackd/adapters/manifest.schema.json",
        **schema,
    }
