"""The pilot's word on whether the body can do the task, and the vocabulary it says it in.

Every other gate in quackd refuses one verb. This one refuses *all motion* until the model
has said, against the datasheet in its prompt, whether the task fits the body it is driving.
The verdict is the model's own judgement and is recorded as such: what it estimated, what it
read, and what the task would need, so a transcript shows the reasoning rather than only the
refusal.

`needs` is the one vocabulary three readers share: the tool schema the model fills in, the
matcher that says which other body could, and a flock role that asks for a body that can
carry. A number is a minimum, a word must match, and a figure the maker never published is
not met: a robot that cannot say what it carries is not offered a task that carries something.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

from quackd.adapters.manifest import Manipulator, Mobility, Terrain

if TYPE_CHECKING:
    from quackd.adapters.manifest import RobotManifest

VerdictWord = Literal["feasible", "infeasible", "uncertain"]
Human = Literal["go", "no_go"]

BEFORE_VERDICT = frozenset(
    {"stop", "observe", "report_state", "say", "quack", "express", "gaze", "look", "introspect"}
)
"""What runs before a verdict: the verbs that speak, look, or read, and the brake.

A pilot has to be able to look at the thing before judging whether it can lift it. Everything
else waits, including a verb quackd has never heard of: the gate reads this set and one
other thing, the verb's own `read_only` flag, so a body quackd has never shipped can still
say "this one only looks" about its own sensing verbs. A learned verb never carries that
flag (`learned.py` registers it as `confirm`, unproven), so it is refused until somebody
classifies it on purpose."""

MOVES_THE_BODY = frozenset(
    {
        "move",
        "go_to",
        "search_scan",
        "approach_and",
        "sit",
        "stand",
        "stand_up",
        "kick",
        "grab",
        "move_joints",
        "gripper",
        "grip",
        "place",
        "pick",
        "lift",
        "home_arms",
        "perform",
    }
)
"""The other half. The gate never reads this; it exists so a test can say every verb every
shipped adapter offers was classified deliberately, in both directions."""

TERRAIN_ORDER: tuple[Terrain, ...] = ("indoor_flat", "indoor", "outdoor")
"""Rated for more than the task asks is fine; rated for less is not."""

MANIPULATOR_WORDS = (*(w for w in get_args(Manipulator) if w != "none"), "any")
MOBILITY_WORDS = (*(w for w in get_args(Mobility) if w != "none"), "any")

NEEDS_NUMBERS: tuple[str, ...] = (
    "payload_kg",
    "reach_m",
    "endurance_min",
    "work_height_m",
    "arms",
)
"""Minimums, in the datasheet's own field names. `work_height_m` is the odd one: it asks for a
height the hands must be able to reach, which is a point inside `workspace_height_m`."""

NEEDS_WORDS: dict[str, tuple[str, ...]] = {
    "manipulator": MANIPULATOR_WORDS,
    "mobility": MOBILITY_WORDS,
    "terrain": TERRAIN_ORDER,
}


def check_needs(value: Mapping[str, Any]) -> dict[str, float | str]:
    """What the task demands of a body, validated. Shared by `assess_task` and a flock role."""
    out: dict[str, float | str] = {}
    for key, want in value.items():
        if key in NEEDS_NUMBERS:
            if isinstance(want, bool) or not isinstance(want, int | float):
                raise ValueError(f"needs.{key} must be a number (a minimum)")
            if want < 0:
                raise ValueError(f"needs.{key} must be 0 or more")
            out[key] = float(want)
        elif key in NEEDS_WORDS:
            if want not in NEEDS_WORDS[key]:
                raise ValueError(f"needs.{key} must be one of {', '.join(NEEDS_WORDS[key])}")
            out[key] = str(want)
        else:
            vocabulary = ", ".join((*NEEDS_NUMBERS, *NEEDS_WORDS))
            raise ValueError(f"needs.{key} is not in the datasheet vocabulary ({vocabulary})")
    return out


def needs_properties() -> dict[str, dict[str, Any]]:
    """The `needs` object's JSON schema, so the model is shown the vocabulary it must use."""
    properties: dict[str, dict[str, Any]] = {
        key: {"type": "number", "description": f"At least this much {key}."}
        for key in NEEDS_NUMBERS
    }
    properties["arms"] = {"type": "integer", "description": "At least this many arms."}
    for key, words in NEEDS_WORDS.items():
        properties[key] = {"type": "string", "enum": list(words)}
    return properties


def needs_text(needs: Mapping[str, Any]) -> str:
    """`payload_kg=3, manipulator=gripper`, in a stable order."""
    return ", ".join(f"{key}={_number(needs[key])}" for key in sorted(needs))


def _number(value: Any) -> str:
    return f"{value:g}" if isinstance(value, int | float) and not isinstance(value, bool) else value


def _figure_value(facts: Mapping[str, Any], key: str) -> float | None:
    figure = facts.get(key)
    if isinstance(figure, Mapping):
        raw = figure.get("value")
        return float(raw) if isinstance(raw, int | float) else None
    return (
        float(figure) if isinstance(figure, int | float) and not isinstance(figure, bool) else None
    )


def missing_needs_in(
    needs: Mapping[str, Any], facts: Mapping[str, Any], mobility: str | None
) -> list[str]:
    """One line per unmet need, sorted by key; empty means this body can be asked.

    `facts` is a datasheet dumped to a dict, which is what a flock bid carries, so a
    coordinator can judge a bid from a robot it does not run. A figure nobody published is
    unmet, never assumed."""
    out: list[str] = []
    for key in sorted(needs):
        want = needs[key]
        if key == "mobility":
            has = mobility or "unknown"
            if not (want == "any" and has not in ("none", "unknown")) and has != want:
                out.append(f"mobility = {want} (has {has})")
            continue
        if key == "manipulator":
            has = str(facts.get("manipulator") or "unknown")
            if not (want == "any" and has not in ("none", "unknown")) and has != want:
                out.append(f"manipulator = {want} (has {has})")
            continue
        if key == "terrain":
            rated = facts.get("terrain")
            if rated is None:
                out.append(f"terrain = {want} (not published)")
            elif TERRAIN_ORDER.index(str(rated)) < TERRAIN_ORDER.index(str(want)):  # type: ignore[arg-type]
                out.append(f"terrain = {want} (rated {rated})")
            continue
        if key == "endurance_min" and facts.get("tethered") is True:
            continue  # mains powered: nothing to run down
        if key == "arms":
            arms = facts.get("arms")
            if not isinstance(arms, int) or arms < float(want):
                out.append(f"arms >= {_number(want)} (has {arms if arms is not None else 0})")
            continue
        if key == "work_height_m":
            band = facts.get("workspace_height_m")
            if not isinstance(band, Mapping):
                out.append(f"work_height_m = {_number(want)} (not published)")
            elif not float(band["low"]) <= float(want) <= float(band["high"]):
                out.append(
                    f"work_height_m = {_number(want)} "
                    f"(reaches {float(band['low']):g} to {float(band['high']):g} m)"
                )
            continue
        have = _figure_value(facts, key)
        if have is None:
            out.append(f"{key} >= {_number(want)} (not published)")
        elif have < float(want):
            out.append(f"{key} >= {_number(want)} (has {have:g})")
    return out


def missing_needs(needs: Mapping[str, Any], manifest: RobotManifest) -> list[str]:
    """The same, for a robot whose manifest is in hand."""
    sheet = manifest.datasheet
    return missing_needs_in(
        needs, sheet.model_dump() if sheet is not None else {}, manifest.mobility
    )


def datasheet_value(manifest: RobotManifest, field: str) -> float | str | None:
    """What this body reports for one need's field, or None when nobody published it."""
    if field == "mobility":
        return manifest.mobility
    sheet = manifest.datasheet
    if sheet is None:
        return None
    if field in ("manipulator", "terrain"):
        return getattr(sheet, field)
    if field == "arms":
        return float(sheet.arms)
    if field == "work_height_m":
        band = sheet.workspace_height_m
        return band.high if band is not None else None
    figure = getattr(sheet, field, None)
    return figure.value if figure is not None else None


class Estimate(BaseModel):
    """One thing the pilot guessed about the world on its way to a verdict, and how."""

    model_config = ConfigDict(extra="forbid")

    object: str
    quantity: Literal["mass_kg", "size_m", "distance_m", "height_m", "count", "other"]
    value: float
    basis: Literal["image", "detections", "task_text", "prior_knowledge"]
    confidence: Literal["low", "medium", "high"]

    def text(self) -> str:
        return f"{self.object} {self.quantity}={self.value:g} ({self.basis}, {self.confidence})"


class Verdict(BaseModel):
    """The pilot's judgement of this task against this body."""

    model_config = ConfigDict(extra="forbid")

    verdict: VerdictWord
    reason: str = Field(..., min_length=1)
    limits_consulted: list[str] = Field(default_factory=list)
    estimates: list[Estimate] = Field(default_factory=list)
    needs: dict[str, float | str] = Field(default_factory=dict)
    human: Human | None = Field(
        default=None,
        description="What a person answered about an `uncertain` verdict. Never the model's: "
        "`assess_task` has no such field, and both surfaces refuse one that arrives with it.",
    )

    @field_validator("needs", mode="before")
    @classmethod
    def _vocabulary(cls, value: Any) -> Any:
        return check_needs(value) if isinstance(value, Mapping) else value

    @property
    def go(self) -> bool:
        """Whether verbs that move the body may run on this verdict."""
        return self.verdict == "feasible" or (self.verdict == "uncertain" and self.human == "go")

    def summary(self) -> str:
        return f"{self.verdict}: {self.reason}"

    def question(self) -> str:
        """What the person in the room is asked."""
        parts = [f"The pilot is not sure this robot can do the task: {self.reason}"]
        if self.estimates:
            parts.append("It estimated " + "; ".join(e.text() for e in self.estimates) + ".")
        if self.needs:
            parts.append(f"The task would need {needs_text(self.needs)}.")
        return " ".join(parts)

    def blocking_reason(self) -> str:
        """Why the gate is still shut, in the words the pilot gets back."""
        if self.verdict == "infeasible":
            return f"the task was judged infeasible ({self.reason})"
        if self.human == "no_go":
            return "a human was asked about an uncertain verdict and said no"
        return f"the verdict is uncertain and nobody has cleared it ({self.reason})"


def solo_hint(needs: Mapping[str, Any], here: RobotManifest | None) -> str:
    """Which bodies here could, by their own datasheets. Empty when the task named no need.

    Read from the static descriptions, so it costs no connection. It names only robots whose
    adapter is installed, because a manifest is built by the adapter and quackd cannot
    describe a body whose package is absent. So the answer is about this machine rather than
    about the seven quackd publishes, and a machine with one robot can only speak for that
    one. A body whose maker never published the figure is not named: unknown is not a yes."""
    if not needs:
        return ""
    from quackd.adapters.factory import bodies_that_could

    able = bodies_that_could(needs)
    listed = needs_text(needs)
    if able:
        names = [name for name, _manifest, _missing in able]
        joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"
        sentence = f"By their datasheets, {joined} could (needs {listed})."
    else:
        sentence = f"No robot installed here meets needs {listed}."
        if best := _best_numeric(needs):
            sentence = sentence[:-1] + f": {best}."
    if here is not None and not missing_needs(needs, here):
        sentence += (
            " This body's own datasheet meets those needs, so the verdict is the pilot's "
            "judgement rather than a limit."
        )
    return sentence


def _best_numeric(needs: Mapping[str, Any]) -> str:
    """`the most is toddlerbot at 1.48 kg`, for the first numeric need nobody meets."""
    from quackd.adapters.factory import installed_manifests

    for key in NEEDS_NUMBERS:
        if key not in needs:
            continue
        ranked = [
            (value, name)
            for name, manifest in installed_manifests()
            if isinstance(value := datasheet_value(manifest, key), float)
        ]
        if not ranked:
            continue
        value, name = max(ranked)
        unit = key.rsplit("_", 1)[-1] if "_" in key else key
        return f"the most is {name} at {value:g} {unit}"
    return ""
