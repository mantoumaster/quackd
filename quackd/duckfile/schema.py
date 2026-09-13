"""The machine-enforced half of a `.duck` file.

Everything in the YAML frontmatter is validated here, strictly (unknown keys are errors),
because the executor trusts this model and nothing else. The Markdown body is free text
for the LLM and is deliberately not modelled beyond "it is a string".
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quackd.adapters.manifest import (
    Confidence,
    Manipulator,
    Terrain,
    datasheet_sentences,
)
from quackd.verbs.aliases import canonical
from quackd.verdict import check_needs

DUCK_SPEC_VERSION = 2

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_ROBOT_SPEC_RE = re.compile(r"^[a-z][a-z0-9_]*(:[a-z][a-z0-9_]*)?$")

# The flock roles 0.4 has behaviour for. A requires-only role with no behaviour would be a
# fabricated capability, so the vocabulary is closed (ADR-0019, ADR-0020).
KNOWN_ROLES = ("spotter", "kicker")

# The two `abort_when` phrasings the executor enforces itself. Anything else in the list is
# passed to the LLM as an instruction, which is honest about what is and is not policed.
BATTERY_ABORT_RE = re.compile(r"battery\s+(?:below|under|<)\s*(\d+(?:\.\d+)?)\s*%", re.I)
REPEAT_FAIL_ABORT_RE = re.compile(r"same\s+verb\s+fails\s+(\d+)\s+times?\s+in\s+a\s+row", re.I)


def _verb_list(names: list[str]) -> list[str]:
    """Valid, unique verb names, where a verb and its alias count as one verb."""
    seen: set[str] = set()
    by_canonical: dict[str, str] = {}
    for name in names:
        if not _NAME_RE.match(name.replace("_", "-")):
            raise ValueError(f"{name!r} is not a valid verb name")
        if name in seen:
            raise ValueError(f"duplicate verb {name!r}")
        seen.add(name)
        other = by_canonical.setdefault(canonical(name), name)
        if other != name:
            raise ValueError(f"{other!r} and {name!r} are the same verb; list one of them")
    return names


class VerbsSection(BaseModel):
    """Which verbs the LLM may call, and which need a human to say yes first."""

    model_config = ConfigDict(extra="forbid")

    allow: list[str] = Field(
        default=..., min_length=1, description="Verbs the LLM may call. Anything else is refused."
    )
    confirm: list[str] = Field(
        default_factory=list,
        description="Verbs (subset of `allow`) that prompt a human y/N before executing.",
    )

    @field_validator("allow", "confirm")
    @classmethod
    def _unique_names(cls, names: list[str]) -> list[str]:
        return _verb_list(names)

    @model_validator(mode="after")
    def _confirm_subset_of_allow(self) -> VerbsSection:
        allowed = {canonical(v) for v in self.allow}
        extra = [v for v in self.confirm if canonical(v) not in allowed]
        if extra:
            raise ValueError(f"confirm lists verbs that are not allowed: {extra}")
        if "stop" in self.confirm:
            raise ValueError("stop can never be confirm-gated; it is the kill switch's verb")
        return self


class Budgets(BaseModel):
    """Hard stops. The loop ends when any of these is hit, whatever the LLM thinks."""

    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(
        default=40, ge=1, le=1000, description="Maximum number of LLM decisions."
    )
    max_minutes: float = Field(
        default=5.0, gt=0, le=180, description="Wall-clock (or sim-clock) cap."
    )
    max_llm_calls: int = Field(default=40, ge=1, le=2000, description="Maximum provider calls.")


class FigureOverride(BaseModel):
    """A number the task file asserts about the build in front of it.

    The source defaults to the file itself and the confidence to `estimate`; an author who
    weighed the thing says `measured`."""

    model_config = ConfigDict(extra="forbid")

    value: float = Field(..., ge=0)
    confidence: Confidence = "estimate"
    source: str = Field(default="", description="Optional: 'weighed with the printed gripper'.")
    note: str = ""


class SpanOverride(BaseModel):
    """A band the task file asserts, the same way."""

    model_config = ConfigDict(extra="forbid")

    low: float = Field(..., ge=0)
    high: float = Field(..., ge=0)
    confidence: Confidence = "estimate"
    source: str = ""
    note: str = ""


def _figure_shorthand(value: Any) -> Any:
    """`payload_kg: 0.3` is `payload_kg: {value: 0.3}`."""
    if isinstance(value, bool):
        return value
    return {"value": value} if isinstance(value, int | float) else value


class DatasheetOverride(BaseModel):
    """v2: what a `.duck` says about the body it was written for (`docs/duck-spec.md`).

    Merged field-wise into the robot's own datasheet by `apply_datasheet_override`: a figure
    given here replaces the adapter's and is rendered as coming from the task file; `cannot`,
    `notes` and `not_rated` extend the adapter's; whatever is omitted is left alone. A task
    file can add a `cannot`; it can never delete one.
    """

    model_config = ConfigDict(extra="forbid")

    mass_kg: FigureOverride | None = None
    height_m: FigureOverride | None = None
    dof: FigureOverride | None = None
    payload_kg: FigureOverride | None = None
    reach_m: FigureOverride | None = None
    workspace_height_m: SpanOverride | None = None
    endurance_min: FigureOverride | None = None
    manipulator: Manipulator | None = None
    arms: int | None = Field(default=None, ge=0, le=4)
    tethered: bool | None = None
    terrain: Terrain | None = None
    not_rated: list[str] = Field(default_factory=list)
    cannot: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @field_validator(
        "mass_kg", "height_m", "dof", "payload_kg", "reach_m", "endurance_min", mode="before"
    )
    @classmethod
    def _numbers(cls, value: Any) -> Any:
        return _figure_shorthand(value)

    @field_validator("cannot", "notes", "not_rated")
    @classmethod
    def _prose(cls, values: list[str]) -> list[str]:
        return datasheet_sentences(values)


class LearnedVerbRef(BaseModel):
    """Reserved for v2. The shape a `.duck` will use to pull in a learned (ONNX) verb.

    See `docs/learned-verbs.md`. Parsed and validated today; nothing executes it.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Verb name it registers as.")
    policy: str = Field(..., description="Path or URL of the ONNX policy.")
    description: str = Field(default="", description="LLM-facing description.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class FlockAllocation(BaseModel):
    """How a flock decides who kicks. Deterministic; the LLM never runs the auction."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["auction"] = "auction"  # Contract Net; the only method in v0.3
    bid: Literal["ball_distance"] = "ball_distance"  # lower bid wins
    tie_break: Literal["duck_id"] = "duck_id"  # lexicographic member name
    hysteresis_pct: float = Field(
        default=20.0,
        ge=0,
        le=100,
        description="A challenger must bid this much lower to unseat the current claimant.",
    )
    claim_lease_s: float = Field(
        default=6.0,
        gt=0,
        le=60,
        description="Longest a claim may be held before re-auction (sim clock). A fixed "
        "fuse from the moment the claim is granted, not a progress timer.",
    )


class FlockSafety(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_separation_m: float = Field(default=0.4, ge=0.1, le=2.0)
    one_claimant: bool = Field(
        default=True,
        description="At most one duck approaches the ball at a time. Always enforced in "
        "v0.3; false is rejected rather than silently ignored.",
    )
    per_duck_heartbeat_s: float = Field(default=1.0, gt=0, le=10)

    @field_validator("one_claimant")
    @classmethod
    def _one_claimant(cls, value: bool) -> bool:
        if not value:
            raise ValueError(
                "one_claimant: false is not supported in v0.3 "
                "(the coordinator always enforces a single claimant)"
            )
        return value


class FlockSearch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    partition: Literal["heading"] = "heading"  # each duck owns a heading sector
    restart_s: float = Field(
        default=8.0,
        gt=0,
        le=120,
        description="Re-scan the sector when nothing was found for this long.",
    )


class FlockRole(BaseModel):
    """A role in a heterogeneous flock (v1): who may take it is decided by capability.

    v2 adds a second half to that: a role may also state what the body has to be able to do
    physically, in the datasheet's own words. A robot whose datasheet does not say is not
    offered the role, because a robot that cannot say what it carries is not the one to ask
    to carry something."""

    model_config = ConfigDict(extra="forbid")

    requires: list[str] = Field(
        default=..., min_length=1, description="Verbs a robot must provide to bid for this role."
    )
    needs: dict[str, float | str] = Field(
        default_factory=dict,
        description="v2: what the body must be, in the datasheet vocabulary (payload_kg, "
        "reach_m, manipulator, mobility, ...). Numbers are minimums; a figure nobody "
        "published counts as not met.",
    )
    count: Literal[1] = Field(default=1, description="Robots per role. Only 1 so far.")

    @field_validator("requires")
    @classmethod
    def _names(cls, names: list[str]) -> list[str]:
        return _verb_list(names)

    @field_validator("needs", mode="before")
    @classmethod
    def _needs(cls, value: Any) -> Any:
        return check_needs(value) if isinstance(value, Mapping) else value


class FlockSection(BaseModel):
    """Cooperating robots (simulator only). The coordinator enforces this block."""

    model_config = ConfigDict(extra="forbid")

    members: int | list[str] = Field(
        default=3,
        description="Count (2-4, named duck-0..) or a list of 2-4 unique slugs.",
    )
    allocation: FlockAllocation = Field(default_factory=FlockAllocation)
    safety: FlockSafety = Field(default_factory=FlockSafety)
    search: FlockSearch = Field(default_factory=FlockSearch)
    roles: dict[str, FlockRole] | None = Field(
        default=None,
        description="v1: named roles (spotter, kicker) with the verbs each requires.",
    )
    frame_hints: Literal["auto", "on", "off"] = Field(
        default="auto",
        description="v1: share arena-frame target hints between robots. auto = only when "
        "every member runs in sim2d (there is no shared frame on hardware).",
    )

    @model_validator(mode="after")
    def _roles_are_known_and_complete(self) -> FlockSection:
        if self.roles is None:
            return self
        unknown = sorted(set(self.roles) - set(KNOWN_ROLES))
        if unknown:
            raise ValueError(
                f"unknown flock role {unknown[0]!r}; 0.4 knows {', '.join(KNOWN_ROLES)}"
            )
        if any(role not in self.roles for role in KNOWN_ROLES):
            raise ValueError("flock.roles needs both spotter and kicker")
        if isinstance(self.members, int):
            raise ValueError("name the members (a list) when flock.roles is given")
        return self

    @field_validator("members")
    @classmethod
    def _members(cls, value: int | list[str]) -> int | list[str]:
        if isinstance(value, int):
            if not 2 <= value <= 4:
                raise ValueError("a flock needs 2 to 4 ducks")
            return value
        if not 2 <= len(value) <= 4:
            raise ValueError("a flock needs 2 to 4 named ducks")
        seen: set[str] = set()
        for name in value:
            if not _NAME_RE.match(name):
                raise ValueError(f"{name!r} is not a valid member name (slug)")
            if name in seen:
                raise ValueError(f"duplicate member {name!r}")
            seen.add(name)
        return value

    @property
    def member_names(self) -> list[str]:
        if isinstance(self.members, int):
            return [f"duck-{i}" for i in range(self.members)]
        return list(self.members)


class DuckFrontmatter(BaseModel):
    """The contract. This is what `schema.json` describes and what the executor enforces."""

    model_config = ConfigDict(extra="forbid", title="quackd .duck frontmatter (v0, v1, v2)")

    duck: Literal[0, 1, 2] = Field(
        ...,
        description="Spec version: 0 (quackd 0.1 to 0.3), 1 (0.4: requires, robots, "
        "flock.roles, flock.frame_hints) or 2 (0.9: datasheet, flock.roles.needs). Older "
        "files parse unchanged.",
    )
    name: str = Field(..., description="Slug: lowercase letters, digits, hyphens.")
    description: str = Field(..., min_length=1, description="One line, human-facing.")
    author: str | None = None
    verbs: VerbsSection
    budgets: Budgets = Budgets()
    success: list[str] = Field(
        default=..., min_length=1, description="Success criteria the LLM must judge itself against."
    )
    abort_when: list[str] = Field(
        default_factory=list,
        description=(
            "Abort conditions. 'Battery below N%' and 'Same verb fails N times in a row' are "
            "enforced by the executor; other entries are passed to the LLM as instructions."
        ),
    )
    persona: str | None = Field(default=None, description="Tone for the LLM. Optional.")
    providers: list[str] = Field(
        default_factory=list, description="Providers this duck was tested with (not enforced)."
    )
    learned_verbs: list[LearnedVerbRef] = Field(
        default_factory=list, description="Reserved for v2 learned verbs. Must be empty in v0.1."
    )
    flock: FlockSection | None = Field(
        default=None,
        description="Cooperating robots (simulator only). Absent means a single robot.",
    )
    requires: list[str] = Field(
        default_factory=list,
        description="v1: verbs the task needs. `quackd validate --robot` checks them against "
        "the robot's manifest. For v0 files every allowed verb is required.",
    )
    datasheet: DatasheetOverride | None = Field(
        default=None,
        description="v2: corrections and additions to the robot's datasheet for this build, "
        "rendered in the prompt as coming from the task file.",
    )
    robots: str | dict[str, str] | None = Field(
        default=None,
        description="v1: default robot as <adapter>[:<backend>], or a mapping from flock "
        "member name to such a spec, so `quackd run <duck>` needs no --robot flag.",
    )

    @field_validator("name")
    @classmethod
    def _slug(cls, value: str) -> str:
        if not _NAME_RE.match(value):
            raise ValueError("name must match ^[a-z0-9][a-z0-9-]{0,63}$")
        return value

    @field_validator("requires")
    @classmethod
    def _requires_names(cls, names: list[str]) -> list[str]:
        return _verb_list(names)

    @field_validator("robots")
    @classmethod
    def _robot_specs(cls, value: str | dict[str, str] | None) -> str | dict[str, str] | None:
        specs = [value] if isinstance(value, str) else list((value or {}).values())
        for spec in specs:
            if not _ROBOT_SPEC_RE.match(spec):
                raise ValueError(f"{spec!r} is not <adapter>[:<backend>]")
        if isinstance(value, dict):
            for member in value:
                if not _NAME_RE.match(member):
                    raise ValueError(f"{member!r} is not a valid member name (slug)")
        return value

    @model_validator(mode="after")
    def _version_and_cross_field_rules(self) -> DuckFrontmatter:
        if self.duck < 2 and self.datasheet is not None:
            raise ValueError("datasheet needs duck: 2")
        if self.duck < 2 and self.flock is not None and self.flock.roles is not None:
            for role, spec in self.flock.roles.items():
                if spec.needs:
                    raise ValueError(f"flock.roles.{role}.needs needs duck: 2")
        if self.datasheet is not None and self.flock is not None:
            raise ValueError("datasheet describes one body; a flock duck cannot carry one")
        if self.duck == 0:
            v1_keys = {
                "requires": bool(self.requires),
                "robots": self.robots is not None,
                "flock.roles": self.flock is not None and self.flock.roles is not None,
                "flock.frame_hints": self.flock is not None
                and "frame_hints" in self.flock.model_fields_set,
            }
            for key, used in v1_keys.items():
                if used:
                    raise ValueError(f"{key} needs duck: 1")
        allowed = {canonical(v) for v in self.verbs.allow}
        extra = [v for v in self.requires if canonical(v) not in allowed]
        if extra:
            raise ValueError(f"requires lists verbs that are not allowed: {extra}")
        if self.flock is not None and self.flock.roles is not None:
            for role, spec in self.flock.roles.items():
                extra = [v for v in spec.requires if canonical(v) not in allowed]
                if extra:
                    raise ValueError(
                        f"flock.roles.{role} requires verbs that are not allowed: {extra}"
                    )
            if isinstance(self.robots, dict):
                unknown = sorted(set(self.robots) - set(self.flock.member_names))
                if unknown:
                    raise ValueError(f"robots names members the flock does not have: {unknown}")
        return self

    @property
    def effective_requires(self) -> list[str]:
        """What `validate --robot` checks: v1 says it; a v0 task needs everything it allows."""
        return list(self.requires) if self.duck >= 1 else list(self.verbs.allow)

    # ── derived, machine-enforced abort thresholds ──────────────────────────────────

    @property
    def battery_abort_percent(self) -> float | None:
        for line in self.abort_when:
            m = BATTERY_ABORT_RE.search(line)
            if m:
                return float(m.group(1))
        return None

    @property
    def repeat_failure_abort(self) -> int | None:
        for line in self.abort_when:
            m = REPEAT_FAIL_ABORT_RE.search(line)
            if m:
                return int(m.group(1))
        return None

    @property
    def advisory_abort_conditions(self) -> list[str]:
        """The `abort_when` entries we cannot enforce and therefore hand to the LLM."""
        return [
            line
            for line in self.abort_when
            if not BATTERY_ABORT_RE.search(line) and not REPEAT_FAIL_ABORT_RE.search(line)
        ]


class DuckFile(BaseModel):
    """A parsed `.duck`: the enforced contract plus the LLM-facing body."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    frontmatter: DuckFrontmatter
    body: str
    path: str | None = None

    @property
    def name(self) -> str:
        return self.frontmatter.name


def json_schema() -> dict[str, Any]:
    """The JSON Schema for the frontmatter, as exported to `schema.json`."""
    schema = DuckFrontmatter.model_json_schema()
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/rokbenko/quackd/blob/main/quackd/duckfile/schema.json",
        **schema,
    }
    return schema
