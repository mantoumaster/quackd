"""A discrete stepper in front of the model: the turns that are a choice, answered as one.

quackd's loop asks one question a turn — which single tool call now — and pays a frontier
model's full latency for it whether the answer is `report_state` or a six-joint pose. On the
SO-101 run at the top of `README.md` that is 62.1 seconds of a 78.8 second run. Some of those
turns are not writing, they are choosing, and TypeSafe's Jev answers a choice without
generating anything: typed questions against a named state, back as a value and a probability
distribution (https://docs.typesafe.ai/introduction).

Which turns those are is decided here, from each tool's own JSON Schema and nothing else, so a
body quackd has never shipped is classified by the same rule as the seven that do. A verb whose
meaning is a number — every `move_joints`, every `move` — is not a choice and never becomes
one. On the arm that is not even a judgement call: `move_joints` takes a free-form object of
joint names the schema never lists, because they live in a `field_validator` rather than in an
enum, so there is nothing for a classifier to enumerate even in principle.

Off by default, off when the SDK is absent, off when the key is missing. Nothing in this file
imports `typesafe_sdk` at module scope: `quackd.agent.loop` imports this module on every run
and must not pay for a vendor that is not in the run.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from quackd.verbs.registry import Verb
from quackd.verdict import MOVES_THE_BODY

JevMode = Literal["off", "shadow", "on"]

DEFAULT_MODEL = "jev-1.13.0"
"""Pinned, not `jev-latest`. The aliases move, and a run whose stepper changed under it is a
run whose transcript describes a model that is no longer the one that answered."""

KEY_ENV = "TYPESAFE_API_KEY"
EXTRA = "jev"

ESCALATE = "escalate"
"""The way out, offered on every turn. Without it a Choice always returns *something*, and the
confidence floor is then the only thing standing between "none of these is right" and a servo."""

MAX_CALLS_PER_VERB = 12
"""Past this a verb stops being a choice. TypeSafe's own guidance is that a question should be
a gut-check a knowledgeable person could make in a few seconds, and picking one of thirteen
shapes of the same verb is not that. The widest verb quackd ships is six (`gripper` on a
two-armed body: three sides times open or shut), so this is headroom rather than a limit."""

# The confidence a Choice must clear before the stepper acts on it, by what the verb does.
# Every number here is one TypeSafe publishes, and the citation is the point: quackd is not in
# a position to invent thresholds for somebody else's model, and their own confidence page says
# the right values are domain-specific and have to be tuned on your own data. `--jev shadow`
# records what would have happened at each of these, which is how they get moved.
FLOORS: dict[str, float] = {
    # `stop`, and deliberately the lowest floor in the system. Below 0.5 is "genuinely unsure"
    # in TypeSafe's own words, and 0.5 is exactly where an unsure stepper should still be
    # allowed to reach for the brake: a wrong `stop` costs one step, and a wrong anything-else
    # costs a move nobody chose.
    "brake": 0.50,
    # Sends no intent: reads state or a camera. TypeSafe's universal floor for a cheap action.
    "read": 0.60,
    # Everything that sends an intent. Their high-stakes number. Not 0.9: their 0.9 is paired
    # with "proceed with confirmation", and quackd expresses confirmation separately, below.
    "motion": 0.85,
    # A verb the manifest or the `.duck` gated on a human. Literally their ">0.9, high stakes,
    # proceed with confirmation" — and quackd's own confirm gate still runs on top of it, so a
    # person is still asked.
    "confirm": 0.90,
    # Not a floor, a refusal: the label is never offered, so no confidence can reach it.
    "never": 1.01,
}


@dataclass(frozen=True)
class Call:
    """One concrete tool call the stepper may author, and the words it is offered in.

    `label` is what Jev chooses between and what the trace prints, so it has to read like
    something a person would say out loud: `gripper(open=false)`, not a schema fragment."""

    name: str
    arguments: dict[str, Any]
    label: str


# ── what counts as a choice ─────────────────────────────────────────────────────────────


def _closed_values(spec: Mapping[str, Any]) -> list[Any] | None:
    """The values this property can take, when they are a closed set, else None."""
    if "const" in spec:
        return [spec["const"]]
    if isinstance(spec.get("enum"), list):
        return list(spec["enum"])
    if spec.get("type") == "boolean":
        return [True, False]
    return None


def _allows_null(spec: Mapping[str, Any]) -> bool:
    if spec.get("type") == "null":
        return True
    return any(
        isinstance(member, Mapping) and member.get("type") == "null"
        for member in spec.get("anyOf") or ()
    )


def _inert(spec: Mapping[str, Any], *, required: bool) -> bool:
    """Whether leaving this property out chooses nothing by leaving it out.

    This is the hinge of the whole rule, and the two cases it separates look alike until you
    read the default. `gaze` has a nullable `bearing_deg` defaulting to null: omitted, it has
    no value at all, and the verb does exactly what its enum says. `move` has `vx` defaulting
    to 0.15, so `move()` walks the robot at 0.15 m/s — a speed chosen by not choosing. A
    default of null means nothing; a default of 0.15 means 0.15."""
    if required:
        return False
    return "default" in spec and spec["default"] is None and _allows_null(spec)


def _render(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return json.dumps(value)


def discrete_calls(schema: Mapping[str, Any]) -> list[Call] | None:
    """Every concrete call this tool allows, or None when it is not a choice.

    Takes a whole tool schema, as `Verb.tool_schema()` builds it. A tool is a choice when every
    property it has is either a closed set — an enum, a const, or a boolean — or inert. One
    property that is a number, a free string, an object or an array and the answer is a value
    rather than a choice: that turn belongs to the model, whatever else is true about it.
    """
    name = str(schema.get("name") or "")
    inner = schema.get("input_schema") or {}
    properties: Mapping[str, Any] = inner.get("properties") or {}
    required = set(inner.get("required") or ())
    closed: list[tuple[str, list[Any]]] = []
    for prop in sorted(properties):
        spec = properties[prop]
        if not isinstance(spec, Mapping):
            return None
        # An object or an array is continuous whatever else it says about itself. The arm's
        # `positions` is the case this exists for: a free-form map of joint name to degrees
        # whose keys the schema never lists.
        if spec.get("type") in ("object", "array"):
            return None
        values = _closed_values(spec)
        if values is not None:
            closed.append((prop, values))
        elif not _inert(spec, required=prop in required):
            return None
    total = 1
    for _prop, values in closed:
        total *= max(len(values), 1)
    if total > MAX_CALLS_PER_VERB:
        return None
    calls: list[Call] = []
    for combination in itertools.product(*(values for _prop, values in closed)):
        arguments = {
            prop: value for (prop, _values), value in zip(closed, combination, strict=True)
        }
        shown = ", ".join(f"{prop}={_render(value)}" for prop, value in arguments.items())
        calls.append(
            Call(name=name, arguments=arguments, label=f"{name}({shown})" if shown else name)
        )
    return calls


def verb_class(verb: Verb, canonical: str | None = None) -> str:
    """Which confidence floor this verb answers to. Always a key of `FLOORS`.

    Read off what the verb says about itself rather than off its name, with the one exception
    the verdict gate already makes (`safety.py`): `MOVES_THE_BODY` wins over `read_only`,
    because a verb arriving under a name quackd has recorded as motion while claiming to only
    read is saying two contradictory things, and quackd believes its own record.
    """
    name = canonical or verb.name
    if verb.safety_class == "dangerous":
        return "never"
    if verb.safety_class == "confirm":
        return "confirm"
    if name == "stop":
        return "brake"
    if verb.read_only and name not in MOVES_THE_BODY:
        return "read"
    return "motion"


def labels(calls: Sequence[Call]) -> list[str]:
    """The label set for one turn's Choice: every call on offer, then the way out."""
    return [call.label for call in calls] + [ESCALATE]
