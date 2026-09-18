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

import importlib
import itertools
import json
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, get_args

from quackd.agent.providers.base import ToolCall
from quackd.verbs.registry import Verb
from quackd.verdict import BEFORE_VERDICT, MOVES_THE_BODY

if TYPE_CHECKING:
    from quackd.agent.providers.base import Observation
    from quackd.verbs.registry import VerbRegistry

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


# ── is it here at all ───────────────────────────────────────────────────────────────────


def resolve_jev_mode(flag: str | None) -> JevMode:
    """`--jev`, then `QUACKD_JEV`, then off.

    Off is not a fallback, it is the answer: a key sitting in a `.env` file is somebody's
    other project, and quackd never switches a paid dependency on because it found one."""
    raw = (flag if flag is not None else os.environ.get("QUACKD_JEV") or "off").strip().lower()
    if raw not in get_args(JevMode):
        allowed = ", ".join(get_args(JevMode))
        raise ValueError(f"unknown --jev mode {raw!r}; choose one of {allowed}")
    return raw  # type: ignore[return-value]


def jev_is_available() -> tuple[bool, str]:
    """Whether a stepper could run here, and in plain words why not.

    Phrased like `ProviderNotInstalled` and `ProviderMissingKey` next door, because a reader
    who has met one of those should recognise this one."""
    try:
        importlib.import_module("typesafe_sdk")
    except Exception:
        return False, (
            f"the stepper needs the optional extra quackd[{EXTRA}] — "
            f'run: uv pip install "quackd[{EXTRA}]"'
        )
    if not os.environ.get(KEY_ENV):
        return False, f"the stepper needs {KEY_ENV} (set it in .env or the environment)"
    return True, ""


# ── what one turn looks like ────────────────────────────────────────────────────────────

STATE_SOFT_CHARS = 6_000
STATE_HARD_CHARS = 24_000
"""The API allows 32k tokens of state; neither of these is near it, because the limit that
binds is accuracy rather than the API. TypeSafe say plainly that a Jev answer gets worse as
the state grows with content unrelated to the decision, so the cap is an accuracy budget and
the trim order below is which parts of a turn are least likely to decide it."""

NEVER_TRIMMED = ("goal", "success_when", "body", "where", "now", "last")
"""A turn without these is not a turn worth answering, so if they alone blow the hard cap the
call is not made at all and the model takes it."""

TRIM_ORDER = ("flock", "notes", "recent", "tried", "camera")
"""Dropped in this order until the state fits: the flock's chatter first, then what earlier
runs remembered, then the tail of this run, then the counters, and the camera last because on
a body that can see it is often the whole question."""

TIMEOUT_S = 1.0
"""A stepper that has not answered in a second is not worth waiting for: the point of it is
that it is quicker than the model, and past this the turn is cheaper spent on the model
directly. The turn escalates and the record says the call timed out."""

MAX_RETRIES = 1

MAX_IN_A_ROW = 8
"""Turns the stepper may answer before the model is consulted whatever it says.

Only the model can record a verdict, declare an outcome or write a note, so a run that never
reaches it is a run that can only end on a budget. This is the backstop that stops a confident
stepper eating a whole run: a `lerobot-lookout` whose stepper answered `report_state` at 0.99
every turn spent all twelve steps on it and ended `budget`, having asked the model nothing.

A starting value, like the floors, to be moved once `--jev shadow` has said what real runs
look like. Eight because the longest wholly discrete sequence quackd can presently describe is
the arm's six-turn grip loop, and this has to clear it with room."""


@dataclass(frozen=True)
class Advice:
    """One turn's answer, and the record of how it was reached.

    `call` is None whenever the model should take this turn, for any reason at all: the
    stepper declined, it was not confident enough, it thinks the job is done, it errored, or
    it was never asked. The caller does not need to know which; the record says."""

    call: ToolCall | None
    record: dict[str, Any]

    @property
    def gate(self) -> str:
        return str(self.record.get("gate", ""))

    def event(self) -> dict[str, Any]:
        return dict(self.record)


# ── the state, and the questions asked against it ───────────────────────────────────────


def _trimmed_state(raw: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """The state as it will be sent, and the names of whatever had to go.

    Trimming is recorded rather than silent: a stepper answering badly on a long run and a
    stepper answering badly on a short one are different problems, and the only way to tell
    them apart afterwards is to know what it was actually looking at."""
    state = {k: v for k, v in raw.items() if v}
    dropped: list[str] = []
    for name in TRIM_ORDER:
        if len(json.dumps(state)) <= STATE_SOFT_CHARS:
            break
        if state.pop(name, None) is not None:
            dropped.append(name)
    return state, dropped


def _fits(state: Mapping[str, str]) -> bool:
    return len(json.dumps(state)) <= STATE_HARD_CHARS


def _one_line(text: str, limit: int = 240) -> str:
    """Somebody else's prose as one field of a named state. Jev reads text, not layout."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


DONE_THRESHOLD = 0.5
HUMAN_THRESHOLD = 0.5
"""A Noul carries no confidence, so this is the raw probability and 0.5 is "more likely than
not". Escalating when the job is not in fact done costs one model call; not escalating when it
is costs a robot that carries on working after the task is finished."""


def _escalate_criterion() -> str:
    return (
        "None of the other options is the right action now, or the right action needs a "
        "number, an angle, a distance, a target name or a sentence. Choose this whenever the "
        "answer is a value rather than one of the listed actions."
    )


def _questions(sdk: Any, offered: Sequence[Call], what: Mapping[str, str]) -> dict[str, Any]:
    """One fan-out per turn.

    All four go every time. TypeSafe run questions in parallel and in isolation on the same
    state, so adding one costs almost nothing, and `feasible` is asked on every turn even
    though v1 never acts on it: recording it beside the model's own verdict is the cheapest
    possible way to earn the right to act on it later."""
    criteria = {call.label: what[call.label] for call in offered}
    criteria[ESCALATE] = _escalate_criterion()
    return {
        "next_verb": sdk.Choice(
            instructions=(
                "Which single action should the robot take right now to make progress on "
                "`goal`? Read `now` for what the robot reports about itself, `last` for what "
                "the previous action returned, and `tried` for how often each action has "
                "already been used on this task."
            ),
            criteria=criteria,
        ),
        "done": sdk.Noul(
            instructions=(
                "Everything listed under `success_when` has already happened, according to "
                "`now`, `last` and `recent`. Something merely planned or in progress is not "
                "done."
            )
        ),
        "need_human": sdk.Noul(
            instructions=(
                "A person has to decide before this robot does anything else: the readings "
                "contradict each other, something is stuck or jammed, or the obvious next "
                "step could damage the body or what it is holding."
            )
        ),
        "feasible": sdk.Choice(
            instructions=(
                "Can this body, as `body` describes it, do `goal` at all? Judge the body "
                "against the task, not how far along it is."
            ),
            criteria={
                "feasible": "This body can do it with the actions it has.",
                "infeasible": "This body cannot do it however well it is driven.",
                "uncertain": "It depends on something the body does not report.",
            },
        ),
    }


# ── the stepper ─────────────────────────────────────────────────────────────────────────


def _answer(result: Any, key: str) -> Any:
    """One question's answer, however this SDK version hands them back.

    Tolerant on purpose. Jev is early access, the answer container has already been spelled
    two ways in its own docs, and an attribute rename upstream must cost a run one escalation
    rather than ending it with a traceback while an arm is energised."""
    for holder in (getattr(result, "answers", None), result):
        if holder is None:
            continue
        with_key = None
        if isinstance(holder, Mapping):
            with_key = holder.get(key)
        elif hasattr(holder, key):
            with_key = getattr(holder, key)
        if with_key is not None:
            return with_key
    return None


def _field(answer: Any, name: str, default: Any = None) -> Any:
    if answer is None:
        return default
    if isinstance(answer, Mapping):
        return answer.get(name, default)
    return getattr(answer, name, default)


@dataclass
class Stepper:
    """The turns that are a choice, on this body, under this contract.

    Built once per run, after the allowlist is final, because its whole vocabulary is the
    allowlist's discrete calls and nothing else. It is never offered a verb the executor would
    refuse this turn, so a stepper-authored call cannot meet the verdict gate or the allowlist:
    those refusals are unreachable rather than caught.
    """

    mode: JevMode
    goal: str
    success: Sequence[str] = ()
    body: str = ""
    model: str = ""
    client: Any = None
    """A test double, or None to build the real one lazily on the first turn."""

    calls: dict[str, Call] = field(default_factory=dict)
    classes: dict[str, str] = field(default_factory=dict)
    what: dict[str, str] = field(default_factory=dict)
    """Label to the verb's own one-line description: what Jev is told each option means."""
    early: set[str] = field(default_factory=set)
    """Labels whose verb may run before the pilot has judged the task."""
    tried: Counter[str] = field(default_factory=Counter)
    last_call: str | None = None
    """The label it authored on the previous turn, or None when the model took that turn."""
    in_a_row: int = 0
    asked: int = 0
    taken: int = 0
    errors: int = 0
    latency_s: float = 0.0

    @classmethod
    def build(
        cls,
        *,
        mode: JevMode,
        registry: VerbRegistry,
        allow: Sequence[str],
        goal: str,
        success: Sequence[str] = (),
        body: str = "",
        model: str | None = None,
        client: Any = None,
    ) -> Stepper:
        stepper = cls(
            mode=mode,
            goal=goal,
            success=list(success),
            body=body,
            model=model or os.environ.get("TYPESAFE_DEFAULT_MODEL") or DEFAULT_MODEL,
            client=client,
        )
        for name in allow:
            verb = registry.view(name)
            canonical = registry.canonical(name)
            kind = verb_class(verb, canonical)
            if kind == "never":
                continue  # a dangerous verb is never offered, so no confidence can reach it
            found = discrete_calls(verb.tool_schema())
            if found is None:
                continue
            # the same two conditions the verdict gate itself uses, so what the stepper may
            # reach for before a verdict is exactly what the executor would let through
            runs_early = (canonical in BEFORE_VERDICT and verb.kind != "learned") or (
                verb.read_only and canonical not in MOVES_THE_BODY
            )
            described = _one_line(verb.description, 200)
            for call in found:
                stepper.calls[call.label] = call
                stepper.classes[call.label] = kind
                stepper.what[call.label] = described
                if runs_early:
                    stepper.early.add(call.label)
        return stepper

    # ── what is on offer this turn ──

    def labels_for(self, *, cleared: bool) -> list[Call]:
        """The calls the executor would actually run right now.

        Before the pilot has recorded a feasibility verdict that is the reading verbs and the
        brake, which is exactly what the hero run reached for unprompted on its first turn."""
        return [call for label, call in self.calls.items() if cleared or label in self.early]

    def summary(self) -> dict[str, Any]:
        """The `jev` block of `summary.json`, written only when the stepper actually ran."""
        return {
            "mode": self.mode,
            "model": self.model,
            "asked": self.asked,
            "taken": self.taken,
            "errors": self.errors,
            "latency_s": round(self.latency_s, 3),
        }

    # ── one turn ──

    def _client(self) -> Any:
        """The SDK, imported the first time a turn actually needs it and not before."""
        if self.client is None:
            sdk = importlib.import_module("typesafe_sdk")
            self.client = sdk.AsyncTypeSafeClient(
                model=self.model,
                retry=sdk.RetryPolicy(max_retries=MAX_RETRIES, backoff_max=0.2, timeout=TIMEOUT_S),
            )
        return self.client

    def _build_state(
        self,
        obs: Observation,
        *,
        budget: str,
        stepped: Sequence[str],
        notes: str | None,
    ) -> tuple[dict[str, str], list[str]]:
        """A named JSON object, in English, carrying only what the questions need.

        Never a picture: Jev is documented as text only, and quackd would rather escalate a
        turn that needs eyes than pretend otherwise. Never the system prompt either, because
        the instructions belong in the questions, and never the history, which is the
        distractor TypeSafe warn about by name."""
        from quackd.perception.base import Detection, summarize_detections
        from quackd.transport.base import DuckState

        features = obs.features or {}
        dump = features.get("state") or {}
        now = DuckState(**dump).summary() if dump else "not reported"
        detections = [Detection(**d) for d in features.get("detections") or []]
        last = features.get("last_result")
        raw = {
            # Whole, never shortened. Every other field here is a summary of something the
            # robot can be asked again for, but a goal cut off at a comma is a different
            # goal: "wave, but do not raise the shoulder" truncates into its own opposite.
            # If that alone will not fit, the turn goes to the model rather than going out
            # half-said.
            "goal": " ".join(self.goal.split()),
            "success_when": " ".join("; ".join(self.success).split()),
            "body": _one_line(self.body, 300),
            "where": _one_line(budget),
            "now": _one_line(now, 400),
            "camera": _one_line(summarize_detections(detections)),
            "last": (
                _one_line(
                    f"{last.get('verb')}: {'ok' if last.get('ok') else 'FAILED'} — "
                    f"{last.get('summary') or ''}"
                )
                if last
                else "nothing has run yet"
            ),
            "recent": _one_line(" | ".join(list(stepped)[-4:]), 600),
            "tried": ", ".join(f"{name} {n}" for name, n in sorted(self.tried.items())),
            "notes": _one_line(notes or "", 400),
            "flock": _one_line(json.dumps(features["flock"])) if features.get("flock") else "",
        }
        return _trimmed_state(raw)

    async def advise(
        self,
        obs: Observation,
        *,
        cleared: bool,
        budget: str,
        stepped: Sequence[str] = (),
        notes: str | None = None,
    ) -> Advice:
        """Ask this turn's questions, and say whether the answer is good enough to act on."""
        offered = self.labels_for(cleared=cleared)
        record: dict[str, Any] = {
            "mode": self.mode,
            "model": self.model,
            "labels": labels(offered),
        }
        if not offered:
            # Nothing on this body is a choice right now, so there is no question to ask and
            # nothing to pay for. A duck whose whole allowlist is continuous never reaches the
            # network at all.
            return Advice(None, {**record, "gate": "not_offered", "latency_s": 0.0})

        state, dropped = self._build_state(obs, budget=budget, stepped=stepped, notes=notes)
        record["state_chars"] = len(json.dumps(state))
        record["state_tokens_est"] = record["state_chars"] // 4
        record["trimmed"] = dropped
        if not _fits(state):
            return Advice(None, {**record, "gate": "state_too_large", "latency_s": 0.0})

        started = time.perf_counter()
        try:
            sdk = importlib.import_module("typesafe_sdk")
            result = await self._client().system_one(state, _questions(sdk, offered, self.what))
        except Exception as e:
            self.errors += 1
            self.asked += 1
            took = round(time.perf_counter() - started, 3)
            self.latency_s += took
            return Advice(
                None,
                {
                    **record,
                    "gate": "error",
                    "error": f"{type(e).__name__}: {e}",
                    "latency_s": took,
                },
            )
        self.asked += 1
        took = round(time.perf_counter() - started, 3)
        self.latency_s += took
        record["latency_s"] = took
        record["questions"] = 4
        return self._route(result, record)

    def _route(self, result: Any, record: dict[str, Any]) -> Advice:
        """What one fan-out means, in the order the gates have to be read.

        The Nouls come first, so a stepper that thinks the job is finished never moves
        anything else. Either of them hands the turn back: one ends a run and the other asks a
        person, and the stepper is allowed to do neither."""
        verb = _answer(result, "next_verb")
        choice = str(_field(verb, "choice", ESCALATE))
        confidence = float(_field(verb, "confidence", 0.0) or 0.0)
        spread = {str(k): float(v) for k, v in (_field(verb, "probabilities") or {}).items()}
        done = float(_field(_answer(result, "done"), "noul", 0.0) or 0.0)
        human = float(_field(_answer(result, "need_human"), "noul", 0.0) or 0.0)
        feasible = _answer(result, "feasible")
        record.update(
            choice=choice,
            confidence=confidence,
            probabilities=spread,
            done=done,
            need_human=human,
            feasible={
                "choice": _field(feasible, "choice"),
                "confidence": _field(feasible, "confidence"),
            },
        )
        if done >= DONE_THRESHOLD:
            return self._hand_back({**record, "gate": "done"})
        if human >= HUMAN_THRESHOLD:
            return self._hand_back({**record, "gate": "need_human"})
        # `not in self.calls` is not defensive padding: a Choice answers with one of the
        # labels it was given, and a label that is not one of ours means the SDK and this
        # build disagree about what was asked. That is the model's turn.
        if choice == ESCALATE or choice not in self.calls:
            return self._hand_back({**record, "gate": "escalate"})
        # A reflex that fires twice identically is not deciding, it is looping. quackd already
        # reads repetition as the signature of a stuck pilot (`abort_when: Same verb fails 3
        # times in a row`), and these calls succeed, so nothing else here would catch it.
        if choice == self.last_call:
            return self._hand_back({**record, "gate": "repeat"})
        if self.in_a_row >= MAX_IN_A_ROW:
            return self._hand_back({**record, "gate": "handover"})
        kind = self.classes[choice]
        floor = FLOORS[kind]
        record.update({"class": kind, "floor": floor})
        if confidence < floor:
            return self._hand_back({**record, "gate": "below_floor"})
        call = self.calls[choice]
        self.taken += 1
        self.tried[call.name] += 1
        self.last_call = choice
        self.in_a_row += 1
        return Advice(
            ToolCall(id=f"jev-{self.asked}", name=call.name, arguments=dict(call.arguments)),
            {
                **record,
                "gate": "taken",
                "call": {"name": call.name, "arguments": dict(call.arguments)},
            },
        )

    def _hand_back(self, record: dict[str, Any]) -> Advice:
        """The model takes this turn, so the run of stepper turns is over and both counters
        start again from the next one."""
        self.last_call = None
        self.in_a_row = 0
        return Advice(None, record)

    def shadow_event(
        self, advice: Advice, call: ToolCall, llm: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """What the stepper would have done, beside what the model did, on the same reading.

        The only record shadow mode leaves, and what turns the arithmetic in `docs/jev.md`
        into a measurement."""
        chosen = advice.record.get("choice")
        return {
            "jev_choice": chosen,
            "jev_confidence": advice.record.get("confidence"),
            "jev_gate": advice.gate,
            "jev_latency_s": advice.record.get("latency_s"),
            "model_verb": call.name,
            "model_arguments": dict(call.arguments),
            "agree": bool(chosen) and str(chosen).split("(")[0] == call.name,
            "would_have_acted": advice.gate == "taken",
            "llm_latency_s": (llm or {}).get("latency_s"),
            "llm_usage": (llm or {}).get("usage"),
        }
