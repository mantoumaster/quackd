"""The feasibility verdict: the vocabulary it is said in, and the gate that waits for it."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from quackd.adapters.factory import ADAPTER_NAMES, BACKENDS, RobotSpec, describe
from quackd.adapters.manifest import Datasheet, Figure, RobotManifest, Span, VerbSpec
from quackd.adapters.microduck import microduck_manifest
from quackd.agent.prompts import ASSESS_TASK, DECLARE_NAMES, META_TOOL_NAMES
from quackd.safety import Executor, VerdictRequired
from quackd.transport.mock import MockTransport
from quackd.verbs.aliases import ALIASES, canonical
from quackd.verbs.registry import default_registry
from quackd.verdict import (
    BEFORE_VERDICT,
    MOVES_THE_BODY,
    NEEDS_NUMBERS,
    NEEDS_WORDS,
    Verdict,
    check_needs,
    datasheet_value,
    missing_needs,
    missing_needs_in,
    solo_hint,
)


def _body(**over: object) -> RobotManifest:
    base: dict[str, object] = {
        "id": "bot-01",
        "vendor": "acme",
        "model": "bot",
        "embodiment": "wheeled",
        "mobility": "wheeled",
        "intents": ["twist"],
        "verbs": [VerbSpec(name="move", core=True)],
    }
    base.update(over)
    return RobotManifest(**base)  # type: ignore[arg-type]


# ── the vocabulary ──────────────────────────────────────────────────────────────────────


def test_the_tool_and_the_model_describe_the_same_verdict() -> None:
    schema = ASSESS_TASK["input_schema"]
    assert set(schema["properties"]) == set(Verdict.model_fields) - {"human"}
    assert schema["required"] == ["verdict", "reason"]
    assert ASSESS_TASK["name"] in META_TOOL_NAMES
    assert ASSESS_TASK["name"] not in DECLARE_NAMES, "a verdict is a gate, not an ending"
    assert set(schema["properties"]["needs"]["properties"]) == {*NEEDS_NUMBERS, *NEEDS_WORDS}


def test_the_needs_vocabulary_is_the_datasheets_own() -> None:
    fields = set(Datasheet.model_fields)
    for key in NEEDS_NUMBERS:
        assert key in fields or key == "work_height_m", key
    assert set(NEEDS_WORDS) <= fields | {"mobility"}


def test_a_need_outside_the_vocabulary_is_refused() -> None:
    assert check_needs({"payload_kg": 3}) == {"payload_kg": 3.0}
    assert check_needs({"manipulator": "gripper"}) == {"manipulator": "gripper"}
    with pytest.raises(ValueError, match="not in the datasheet vocabulary"):
        check_needs({"strength": 3})
    with pytest.raises(ValueError, match="must be a number"):
        check_needs({"payload_kg": "heavy"})
    with pytest.raises(ValueError, match="must be 0 or more"):
        check_needs({"payload_kg": -1})
    with pytest.raises(ValueError, match="must be one of"):
        check_needs({"manipulator": "claw"})
    with pytest.raises(ValueError, match="must be a number"):
        check_needs({"arms": True})


def test_a_verdict_validates_its_own_needs() -> None:
    with pytest.raises(ValidationError, match="not in the datasheet vocabulary"):
        Verdict(verdict="infeasible", reason="too heavy", needs={"oomph": 2})
    with pytest.raises(ValidationError, match="at least 1 character"):
        Verdict(verdict="feasible", reason="")
    verdict = Verdict(verdict="uncertain", reason="cannot see it")
    assert not verdict.go
    verdict.human = "go"
    assert verdict.go
    verdict.human = "no_go"
    assert not verdict.go
    assert "said no" in verdict.blocking_reason()


# ── which verbs wait ────────────────────────────────────────────────────────────────────


def test_every_shipped_verb_was_classified_on_purpose() -> None:
    """A verb in neither set is a verb nobody decided about, and the gate would refuse it
    for ever without anybody noticing."""
    from quackd.adapters.factory import _module

    shipped = {canonical(v.name) for v in default_registry().verbs()}
    for adapter in ADAPTER_NAMES:
        shipped |= {canonical(name) for name in _module(adapter).implementations()}
    shipped |= {canonical(alias) for alias in ALIASES}

    assert not BEFORE_VERDICT & MOVES_THE_BODY, "a verb cannot be both"
    unclassified = shipped - BEFORE_VERDICT - MOVES_THE_BODY
    assert not unclassified, f"classify these, one way or the other: {sorted(unclassified)}"
    stale = (BEFORE_VERDICT | MOVES_THE_BODY) - shipped
    assert not stale, f"no shipped robot has these any more: {sorted(stale)}"
    assert "stop" in BEFORE_VERDICT, "the brake can never wait for a verdict"


# ── the gate ────────────────────────────────────────────────────────────────────────────


def _executor(**over: object) -> tuple[Executor, MockTransport, list[dict[str, object]]]:
    transport = MockTransport()
    events: list[dict[str, object]] = []

    class _Tracer:
        def emit(self, kind: str, **data: object) -> None:
            events.append({"kind": kind, **data})

    executor = Executor(
        registry=default_registry(),
        transport=transport,
        manifest=microduck_manifest("mock"),
        trace=_Tracer(),  # type: ignore[arg-type]
        **over,  # type: ignore[arg-type]
    )
    return executor, transport, events


async def test_nothing_moves_until_a_verdict_clears_it() -> None:
    executor, transport, events = _executor(require_verdict=True)
    with pytest.raises(VerdictRequired, match="moves the body"):
        await executor.run_verb("walk", {"vx": 0.1, "duration_s": 0.1})
    gate = next(e for e in events if e["kind"] == "gate")
    assert gate["gate"] == "verdict" and gate["outcome"] == "refused"
    assert transport.intents_of("move") == []

    # looking and speaking are how a pilot works out what it has been asked to do
    for verb in ("quack", "get_frame", "report_state", "stop"):
        assert (await executor.run_verb(verb)).ok, verb

    executor.verdict = Verdict(verdict="feasible", reason="light enough")
    assert (await executor.run_verb("walk", {"vx": 0.1, "duration_s": 0.1})).ok
    assert transport.intents_of("move")


async def test_an_unanswered_doubt_does_not_clear_the_gate() -> None:
    executor, _transport, _events = _executor(require_verdict=True)
    executor.verdict = Verdict(verdict="uncertain", reason="the basket is out of frame")
    with pytest.raises(VerdictRequired, match="nobody has cleared it"):
        await executor.run_verb("walk", {"vx": 0.1, "duration_s": 0.1})

    executor.verdict = Verdict(verdict="uncertain", reason="out of frame", human="no_go")
    with pytest.raises(VerdictRequired, match="said no"):
        await executor.run_verb("walk", {"vx": 0.1, "duration_s": 0.1})

    executor.verdict = Verdict(verdict="infeasible", reason="3 kg of clothes")
    with pytest.raises(VerdictRequired, match="judged infeasible"):
        await executor.run_verb("walk", {"vx": 0.1, "duration_s": 0.1})

    executor.verdict = Verdict(verdict="uncertain", reason="out of frame", human="go")
    assert (await executor.run_verb("walk", {"vx": 0.1, "duration_s": 0.1})).ok


async def test_a_flock_member_is_never_asked_for_a_verdict() -> None:
    """A member is a state machine, not a pilot: there is nobody there to ask, so the gate is
    off by default and the flock never turns it on."""
    import inspect

    from quackd.flock import member as flock_member

    executor, _transport, _events = _executor()
    assert executor.require_verdict is False
    assert (await executor.run_verb("walk", {"vx": 0.1, "duration_s": 0.1})).ok
    assert "require_verdict" not in inspect.getsource(flock_member)


# ── the matcher ─────────────────────────────────────────────────────────────────────────


def test_a_figure_nobody_published_is_not_a_yes() -> None:
    duck = describe(RobotSpec("microduck", "sim2d"))
    assert missing_needs({"payload_kg": 0.1}, duck) == ["payload_kg >= 0.1 (not published)"]
    assert missing_needs({"manipulator": "gripper"}, duck) == ["manipulator = gripper (has beak)"]

    cart = describe(RobotSpec("alohamini", "mock"))
    assert missing_needs({"payload_kg": 1.0, "reach_m": 0.5, "mobility": "wheeled"}, cart) == []
    assert missing_needs({"payload_kg": 1.1}, cart) == ["payload_kg >= 1.1 (has 1)"]

    arm = describe(RobotSpec("lerobot", "mock"))
    assert missing_needs({"mobility": "any"}, arm) == ["mobility = any (has none)"]
    assert missing_needs({"endurance_min": 90}, arm) == [], "mains powered: nothing runs down"

    toddler = describe(RobotSpec("toddlerbot", "mock"))
    assert missing_needs({"payload_kg": 1.4}, toddler) == []
    assert missing_needs({"payload_kg": 1.5}, toddler) == ["payload_kg >= 1.5 (has 1.484)"]


def test_a_working_height_is_a_band_not_a_maximum() -> None:
    cart = describe(RobotSpec("xlerobot", "mock"))
    assert missing_needs({"work_height_m": 0.9}, cart) == []
    assert missing_needs({"work_height_m": 0.2}, cart) == [
        "work_height_m = 0.2 (reaches 0.5 to 1.25 m)"
    ]
    arm = describe(RobotSpec("lerobot", "mock"))
    assert missing_needs({"work_height_m": 0.9}, arm) == ["work_height_m = 0.9 (not published)"]


def test_terrain_is_met_by_a_body_rated_for_more() -> None:
    rugged = _body(datasheet=Datasheet(manipulator="none", terrain="outdoor"))
    assert missing_needs({"terrain": "indoor_flat"}, rugged) == []
    indoor = _body(datasheet=Datasheet(manipulator="none", terrain="indoor_flat"))
    assert missing_needs({"terrain": "outdoor"}, indoor) == [
        "terrain = outdoor (rated indoor_flat)"
    ]
    silent = _body(datasheet=Datasheet(manipulator="none"))
    assert missing_needs({"terrain": "indoor"}, silent) == ["terrain = indoor (not published)"]


def test_a_bid_carries_its_facts_so_a_stranger_can_judge_them() -> None:
    """`missing_needs_in` reads a dumped datasheet, which is what arrives over a bus."""
    sheet = Datasheet(
        manipulator="gripper",
        arms=2,
        payload_kg=Figure(value=1.0, confidence="official", source="the docs"),
        workspace_height_m=Span(low=0.4, high=0.9, confidence="official", source="the docs"),
    )
    facts = sheet.model_dump()
    assert missing_needs_in({"payload_kg": 1.0, "arms": 2}, facts, "wheeled") == []
    assert missing_needs_in({"payload_kg": 2.0}, facts, "wheeled") == ["payload_kg >= 2 (has 1)"]
    assert missing_needs_in({"arms": 3}, facts, "wheeled") == ["arms >= 3 (has 2)"]
    assert missing_needs_in({"payload_kg": 1.0}, {}, None) == ["payload_kg >= 1 (not published)"]
    assert missing_needs_in({"mobility": "wheeled"}, {}, None) == [
        "mobility = wheeled (has unknown)"
    ]


def test_datasheet_value_reads_one_field_or_says_nothing() -> None:
    duck = describe(RobotSpec("microduck", "sim2d"))
    assert datasheet_value(duck, "mass_kg") == 0.8
    assert datasheet_value(duck, "payload_kg") is None
    assert datasheet_value(duck, "manipulator") == "beak"
    assert datasheet_value(duck, "mobility") == "legged"


# ── the hint ────────────────────────────────────────────────────────────────────────────


def test_the_hint_names_a_body_that_could_or_says_none_does() -> None:
    duck = describe(RobotSpec("microduck", "sim2d"))
    able = solo_hint({"payload_kg": 1.0, "manipulator": "gripper"}, duck)
    assert able.startswith("By their datasheets, xlerobot and alohamini could")
    assert "needs manipulator=gripper, payload_kg=1" in able

    nobody = solo_hint({"payload_kg": 3.0}, duck)
    assert nobody.startswith("No shipped body meets needs payload_kg=3")
    assert "the most is toddlerbot at 1.484 kg" in nobody

    assert solo_hint({}, duck) == "", "a task that named no need has nothing to match"


def test_the_hint_says_so_when_this_body_already_meets_the_need() -> None:
    cart = describe(RobotSpec("alohamini", "mock"))
    text = solo_hint({"payload_kg": 0.5, "manipulator": "gripper"}, cart)
    assert "This body's own datasheet meets those needs" in text
    assert "the pilot's judgement rather than a limit" in text


def test_bodies_that_could_reads_only_the_static_descriptions() -> None:
    from quackd.adapters.factory import bodies_that_could, shipped_manifests

    assert [name for name, _m in shipped_manifests()] == list(ADAPTER_NAMES)
    assert [name for name, _m, _missing in bodies_that_could({"payload_kg": 3.0})] == []
    named = [name for name, _m, _missing in bodies_that_could({"manipulator": "gripper"})]
    assert named == ["lerobot", "xlerobot", "alohamini"]
    # every backend of a named body agrees, because the sheet does not vary with one
    for backend in BACKENDS["alohamini"]:
        sheet = describe(RobotSpec("alohamini", backend)).datasheet
        assert sheet is not None and sheet.manipulator == "gripper"
