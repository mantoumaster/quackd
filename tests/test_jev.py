"""The discrete stepper: which turns are a choice, and which belong to the model.

Nothing here imports `typesafe_sdk`, and nothing here reaches the network. The classification
is pure: it reads a tool's JSON Schema and answers, so it is testable against every body quackd
ships without connecting to any of them.

The table below is frozen on purpose, the way `MOVES_THE_BODY` is: a verb added to an adapter
with a parameter nobody thought about is a verb the stepper would either offer or refuse
silently, and this is the test that makes somebody say which.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from quackd.adapters.factory import ADAPTER_NAMES, _module
from quackd.agent.jev import (
    ESCALATE,
    FLOORS,
    MAX_CALLS_PER_VERB,
    discrete_calls,
    labels,
    verb_class,
)
from quackd.agent.prompts import META_TOOLS, REMEMBER, TELL
from quackd.verbs.aliases import canonical
from quackd.verbs.registry import Verb, default_registry

# (how many concrete calls, which confidence floor) per body, or None for "not a choice".
#
# Keyed by body and not by name, because the same word is a different question on a different
# robot: `gripper` is two calls on a one-armed SO-101 and six on a cart with two hands and a
# `side`, and `stand` is a safe verb on a duck and a confirm-gated one on a humanoid that
# cannot get up if it falls.
TABLE: dict[str, dict[str, tuple[int | None, str]]] = {
    "core": {
        "approach_and": (None, "motion"),
        "gaze": (5, "motion"),
        "go_to": (None, "motion"),
        "grab": (1, "motion"),
        "kick": (2, "motion"),
        "move": (None, "motion"),
        "observe": (1, "read"),
        "quack": (1, "motion"),
        "report_state": (1, "read"),
        "say": (None, "motion"),
        "search_scan": (None, "motion"),
        "sit": (1, "motion"),
        "stand": (1, "motion"),
        "stand_up": (1, "motion"),
        "stop": (1, "brake"),
    },
    "microduck": {
        "gaze": (5, "motion"),
        "grab": (1, "motion"),
        "kick": (2, "motion"),
        "quack": (1, "motion"),
        "say": (None, "motion"),
        "sit": (1, "motion"),
        "stand": (1, "motion"),
        "stand_up": (1, "motion"),
    },
    "lerobot": {
        "gripper": (2, "motion"),
        "move_joints": (None, "motion"),
        "pick": (None, "confirm"),
        "place": (1, "motion"),
        "report_state": (1, "read"),
    },
    "rosbridge": {
        "introspect": (1, "read"),
    },
    "open_duck": {
        "express": (3, "motion"),
        "gaze": (5, "motion"),
        "quack": (1, "motion"),
        "say": (None, "motion"),
    },
    "xlerobot": {
        "gripper": (6, "motion"),
        "move_joints": (None, "confirm"),
    },
    "alohamini": {
        "gripper": (6, "motion"),
        "home_arms": (1, "confirm"),
        "lift": (None, "confirm"),
        "move_joints": (None, "confirm"),
    },
    "toddlerbot": {
        "grip": (6, "motion"),
        "look": (None, "motion"),
        "perform": (5, "confirm"),
        "search_scan": (None, "motion"),
        "stand": (1, "confirm"),
    },
}


def _verbs(body: str) -> dict[str, Verb]:
    """The verb templates one body registers, named as that body spells them."""
    if body == "core":
        return {v.name: v for v in default_registry().verbs()}
    return dict(_module(body).implementations())


def _classify(verb: Verb, name: str) -> tuple[int | None, str]:
    calls = discrete_calls(verb.tool_schema())
    return (len(calls) if calls is not None else None, verb_class(verb, canonical(name)))


# ── the table ───────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("body", sorted(TABLE))
def test_every_verb_of_every_shipped_body_is_classified_and_this_table_says_how(body: str) -> None:
    """A new verb, or a new parameter on an old one, has to be classified on purpose.

    The stepper decides from the schema, so a parameter added without a thought about it
    silently changes what a cheap model is allowed to author. That is the failure this test
    exists to make loud, and the message names the verb rather than the diff."""
    verbs = _verbs(body)
    frozen = TABLE[body]
    assert set(verbs) == set(frozen), (
        f"{body}'s verbs changed: classify "
        f"{sorted(set(verbs) - set(frozen))} and drop {sorted(set(frozen) - set(verbs))}"
    )
    for name, verb in sorted(verbs.items()):
        assert _classify(verb, name) == frozen[name], f"{body}.{name} classifies differently now"


def test_the_table_covers_every_adapter_quackd_ships() -> None:
    """Otherwise a whole new body could arrive unclassified and the parametrised test above
    would simply not run for it."""
    assert set(TABLE) - {"core"} == set(ADAPTER_NAMES)


def test_every_class_in_the_table_has_a_floor_to_answer_to() -> None:
    used = {cls for body in TABLE.values() for _count, cls in body.values()}
    assert used <= set(FLOORS), f"no floor for {sorted(used - set(FLOORS))}"


# ── the rule, case by case ──────────────────────────────────────────────────────────────


def test_move_joints_is_never_a_choice_on_any_body_that_has_one() -> None:
    """The requirement the whole rule was written to satisfy.

    Three bodies offer `move_joints` and all three are refused, for the same reason twice
    over: `positions` is a required object, and its keys — the joint names — are not in the
    schema at all. They live in a `field_validator`, so there is nothing to enumerate even in
    principle, and a classifier that tried would be inventing angles."""
    bodies = [b for b in ADAPTER_NAMES if "move_joints" in _verbs(b)]
    assert len(bodies) == 3, f"expected three arms, found {bodies}"
    for body in bodies:
        verb = _verbs(body)["move_joints"]
        assert discrete_calls(verb.tool_schema()) is None, f"{body} would author joint angles"


def test_a_free_string_with_a_default_is_not_a_choice_even_though_omitting_it_is_legal() -> None:
    """`go_to()` is a legal call, and the string it defaults to is not the vocabulary.

    The detector's label set is what `target` can usefully be, and that set is nowhere in the
    schema. A stepper offered `go_to` would be choosing the word "ball" because somebody typed
    it as a default, which is not the same as choosing where to go."""
    core = _verbs("core")
    for name in ("go_to", "search_scan"):
        assert discrete_calls(core[name].tool_schema()) is None, name
    assert discrete_calls(_verbs("lerobot")["pick"].tool_schema()) is None


def test_a_number_with_a_number_for_a_default_is_a_speed_chosen_by_not_choosing() -> None:
    """`move()` walks at 0.15 m/s because that is the default, so omitting `vx` picks it."""
    assert discrete_calls(_verbs("core")["move"].tool_schema()) is None
    assert discrete_calls(_verbs("toddlerbot")["look"].tool_schema()) is None


def test_a_nullable_number_with_a_null_default_leaves_the_verb_a_choice() -> None:
    """The exemption, in both directions, because it is the one clause anybody will widen.

    `gaze` has an optional exact `bearing_deg` that defaults to null: left out it has no value
    at all and the verb does what its enum says, so the five directions are still five
    choices. `quack` is the same with its text. `move` and `look` are the other side."""
    for body in ("microduck", "open_duck"):
        gaze = discrete_calls(_verbs(body)["gaze"].tool_schema())
        assert gaze is not None and len(gaze) == 5, body
        assert all("bearing_deg" not in call.arguments for call in gaze), (
            "an inert parameter is left out of the call, never sent as null"
        )
        quack = discrete_calls(_verbs(body)["quack"].tool_schema())
        assert quack is not None and len(quack) == 1 and quack[0].arguments == {}


def test_a_verbs_labels_are_the_product_of_its_closed_sets_and_nothing_else() -> None:
    """And they read like something a person would say out loud, because Jev chooses between
    these strings and the trace prints them."""
    gripper = discrete_calls(_verbs("lerobot")["gripper"].tool_schema())
    assert gripper is not None
    assert [c.label for c in gripper] == ["gripper(open=true)", "gripper(open=false)"]
    assert [c.arguments for c in gripper] == [{"open": True}, {"open": False}]

    two_handed = discrete_calls(_verbs("xlerobot")["gripper"].tool_schema())
    assert two_handed is not None and len(two_handed) == 6
    assert "gripper(open=false, side=both)" in [c.label for c in two_handed]

    no_params = discrete_calls(_verbs("lerobot")["place"].tool_schema())
    assert no_params is not None
    assert [c.label for c in no_params] == ["place"] and no_params[0].arguments == {}


def test_a_verb_with_more_shapes_than_a_person_could_weigh_stops_being_a_choice() -> None:
    """A Choice is meant to be a gut-check. Past a dozen shapes of one verb it is not one,
    and the answer is to escalate rather than to offer a menu nobody can read."""
    wide = {
        "name": "wide",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"enum": list("abcde")},
                "b": {"enum": list("fghij")},
            },
        },
    }
    assert discrete_calls(wide) is None
    narrow = dict(wide)
    narrow["input_schema"] = {"type": "object", "properties": {"a": {"enum": list("abcde")}}}
    calls = discrete_calls(narrow)
    assert calls is not None and len(calls) == 5 <= MAX_CALLS_PER_VERB


def test_perform_is_never_offered_a_motion_this_build_did_not_load() -> None:
    """A ToddlerBot reports which keyframes it managed to load, and `perform`'s schema is
    built from that list rather than from everything upstream ships."""
    from quackd_toddlerbot.verbs import toddlerbot_verbs

    verb = toddlerbot_verbs(motions=("hold", "kneel"))["perform"]
    calls = discrete_calls(verb.tool_schema())
    assert calls is not None
    assert [c.label for c in calls] == ["perform(motion=hold)", "perform(motion=kneel)"]


# ── what the stepper may never touch ────────────────────────────────────────────────────


def test_no_meta_tool_is_ever_a_choice() -> None:
    """A safety property rather than an accident.

    Every meta tool needs a sentence, and the stepper writes none, so it can never end a run,
    never record a feasibility verdict, never write to memory and never speak to a flock.
    Every ending goes through the model or through a budget."""
    for tool in [*META_TOOLS, REMEMBER, TELL]:
        assert discrete_calls(tool) is None, f"{tool['name']} must stay the model's"


def test_the_arm_splits_the_way_the_docs_say_it_does() -> None:
    """The LeRobot claim `docs/jev.md` is built on, as an assertion.

    Six concrete calls the stepper may author, and every angle still the model's."""
    verbs = _verbs("lerobot")
    offered = sorted(
        call.label
        for name, verb in verbs.items()
        for call in (discrete_calls(verb.tool_schema()) or ())
    )
    core = _verbs("core")
    offered += [call.label for call in (discrete_calls(core["stop"].tool_schema()) or ())]
    offered += [call.label for call in (discrete_calls(core["observe"].tool_schema()) or ())]
    assert sorted(offered) == [
        "gripper(open=false)",
        "gripper(open=true)",
        "observe",
        "place",
        "report_state",
        "stop",
    ]


def test_the_way_out_is_on_every_label_set() -> None:
    """Without it a Choice always returns something, and the floor becomes the only thing
    between "none of these is right" and a servo."""
    calls = discrete_calls(_verbs("lerobot")["gripper"].tool_schema())
    assert calls is not None
    assert labels(calls) == ["gripper(open=true)", "gripper(open=false)", ESCALATE]
    assert labels([]) == [ESCALATE]


def test_the_brake_answers_to_the_lowest_floor_and_a_gated_verb_to_the_highest() -> None:
    assert FLOORS[verb_class(_verbs("core")["stop"])] == min(FLOORS.values())
    assert verb_class(_verbs("alohamini")["home_arms"], "home_arms") == "confirm"
    assert FLOORS["confirm"] > FLOORS["motion"] > FLOORS["read"] > FLOORS["brake"]


# ── the stepper ─────────────────────────────────────────────────────────────────────────


async def _arm(mode: str = "on", allow: Sequence[str] | None = None) -> tuple[Any, Any, Any]:
    """A stepper on the mock arm, with the adapter it was built from."""
    from quackd.agent.jev import Stepper
    from quackd.verbs.registry import registry_from_manifest
    from quackd_lerobot import LeRobotAdapter
    from quackd_lerobot.mock import LeRobotMock

    adapter = LeRobotAdapter(LeRobotMock())
    manifest = await adapter.connect()
    registry = registry_from_manifest(manifest, adapter)
    names = list(allow or ["report_state", "stop", "gripper", "place", "move_joints"])
    stepper = Stepper.build(
        mode=mode,
        registry=registry,
        allow=names,
        goal="Say whether you are holding anything, then let it go",
        success=["you have said whether anything is held"],
        body=manifest.summary(),
    )
    return stepper, adapter, manifest


async def _observation(adapter: Any, last: dict[str, Any] | None = None) -> Any:
    from quackd.agent.providers.base import Observation

    return Observation(
        text="x",
        features={
            "state": (await adapter.get_state()).model_dump(),
            "detections": [],
            "last_result": last,
            "allowed": [],
        },
    )


BUDGET = "step 0/12, llm calls 0/12, 0.0/3 min"


async def _advise(fake: Any, monkeypatch: pytest.MonkeyPatch, **over: Any) -> Any:
    """Build a stepper on the mock arm, install `fake`, and take one turn."""
    from tests import fake_typesafe

    fake_typesafe.install(monkeypatch, fake)
    stepper, adapter, _manifest = await _arm(
        mode=over.pop("mode", "on"), allow=over.pop("allow", None)
    )
    if "goal" in over:
        stepper.goal = over.pop("goal")
    obs = await _observation(adapter, over.pop("last", None))
    advice = await stepper.advise(obs, budget=BUDGET, **over)
    await adapter.disconnect()
    return advice, stepper, fake


async def test_the_arm_never_offers_the_stepper_a_pose() -> None:
    """The headline claim, on the body that has run on real hardware."""
    stepper, adapter, _m = await _arm()
    offered = [call.label for call in stepper.labels_for(cleared=True)]
    assert offered == [
        "report_state",
        "stop",
        "gripper(open=true)",
        "gripper(open=false)",
        "place",
    ]
    assert not any("move_joints" in label for label in offered)
    await adapter.disconnect()


async def test_before_a_verdict_the_stepper_is_offered_only_what_the_gate_would_pass() -> None:
    """So `VerdictRequired` is unreachable from a stepper-authored call rather than caught.

    This is what the hero run did unprompted: its first call was `report_state` and its second
    was the verdict."""
    stepper, adapter, _m = await _arm()
    assert [c.label for c in stepper.labels_for(cleared=False)] == ["report_state", "stop"]
    assert len(stepper.labels_for(cleared=True)) == 5
    await adapter.disconnect()


async def test_a_confident_choice_is_taken_and_becomes_a_real_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests import fake_typesafe

    advice, stepper, _fake = await _advise(
        fake_typesafe.FakeJev(answers=fake_typesafe.turn("report_state", 0.91)),
        monkeypatch,
        cleared=False,
    )
    assert advice.gate == "taken"
    assert advice.call is not None
    assert (advice.call.name, advice.call.arguments) == ("report_state", {})
    assert advice.record["class"] == "read" and advice.record["floor"] == FLOORS["read"]
    assert stepper.taken == 1 and stepper.asked == 1


async def test_a_choice_below_its_floor_goes_to_the_model_and_says_which_floor_it_missed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Motion answers to a higher floor than a read, so the same 0.70 is taken for one and
    refused for the other. The record has to say which floor applied, or a calibration pass
    afterwards cannot tell a near miss from a wild guess."""
    from tests import fake_typesafe

    advice, _s, _f = await _advise(
        fake_typesafe.FakeJev(answers=fake_typesafe.turn("gripper(open=false)", 0.70)),
        monkeypatch,
        cleared=True,
    )
    assert advice.gate == "below_floor" and advice.call is None
    assert advice.record["floor"] == FLOORS["motion"] == 0.85
    assert advice.record["confidence"] == 0.70


async def test_the_same_confidence_clears_a_read_and_misses_a_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests import fake_typesafe

    read, _s, _f = await _advise(
        fake_typesafe.FakeJev(answers=fake_typesafe.turn("report_state", 0.70)),
        monkeypatch,
        cleared=True,
    )
    move, _s2, _f2 = await _advise(
        fake_typesafe.FakeJev(answers=fake_typesafe.turn("place", 0.70)),
        monkeypatch,
        cleared=True,
    )
    assert read.gate == "taken" and move.gate == "below_floor"


async def test_the_way_out_hands_the_turn_back_however_confident_it_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests import fake_typesafe

    advice, _s, _f = await _advise(
        fake_typesafe.FakeJev(answers=fake_typesafe.turn(ESCALATE, 1.0)),
        monkeypatch,
        cleared=True,
    )
    assert advice.gate == "escalate" and advice.call is None


async def test_a_stepper_that_thinks_the_job_is_done_moves_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Nouls are read before the Choice, so a finished task never gets one more move.

    A Noul carries no confidence, so 0.5 here is a raw probability and means "more likely
    than not"."""
    from tests import fake_typesafe

    advice, _s, _f = await _advise(
        fake_typesafe.FakeJev(answers=fake_typesafe.turn("place", 0.99, done=0.62)),
        monkeypatch,
        cleared=True,
    )
    assert advice.gate == "done" and advice.call is None


async def test_a_stepper_that_wants_a_person_hands_the_turn_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests import fake_typesafe

    advice, _s, _f = await _advise(
        fake_typesafe.FakeJev(answers=fake_typesafe.turn("place", 0.99, need_human=0.8)),
        monkeypatch,
        cleared=True,
    )
    assert advice.gate == "need_human" and advice.call is None


async def test_a_typesafe_error_costs_the_turn_and_not_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An arm is energised while this runs. A vendor outage escalates the turn and is written
    down; it never reaches the loop as an exception."""
    from tests import fake_typesafe

    advice, stepper, _f = await _advise(
        fake_typesafe.FakeJev(raises=fake_typesafe.APITimeoutError("took too long")),
        monkeypatch,
        cleared=True,
    )
    assert advice.gate == "error" and advice.call is None
    assert "APITimeoutError" in advice.record["error"]
    assert stepper.errors == 1 and stepper.asked == 1


async def test_a_body_with_nothing_discrete_never_reaches_the_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A duck whose whole allowlist is a number is not a duck the stepper can help with, and
    finding that out must not cost a request."""
    from tests import fake_typesafe

    fake = fake_typesafe.FakeJev(answers=fake_typesafe.turn("move_joints", 0.99))
    advice, _s, used = await _advise(fake, monkeypatch, cleared=True, allow=["move_joints"])
    assert advice.gate == "not_offered" and advice.call is None
    assert used.calls == [], "a question was asked when there was nothing to ask about"


# ── the state ───────────────────────────────────────────────────────────────────────────


async def test_the_state_is_named_english_fields_and_carries_no_picture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Jev is documented as text only, so a frame must never reach it, and the instructions
    belong in the questions rather than in the state."""
    from tests import fake_typesafe

    _a, _s, fake = await _advise(
        fake_typesafe.FakeJev(answers=fake_typesafe.turn("report_state", 0.91)),
        monkeypatch,
        cleared=False,
    )
    state, questions = fake.calls[0]
    assert set(questions) == {"next_verb", "done", "need_human", "feasible"}
    assert "goal" in state and "now" in state and "body" in state
    assert all(isinstance(v, str) for v in state.values()), "every field is a sentence"
    blob = " ".join(state.values()).lower()
    for forbidden in ("png", "base64", "data:image", "jpeg"):
        assert forbidden not in blob, f"{forbidden} reached a text-only model"
    assert chr(10) not in blob, "a field carried layout rather than a sentence"


async def test_the_goal_survives_a_state_that_has_to_be_trimmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Notes go first and the goal never goes, and the record says what went, because a
    stepper answering badly on a long run is a different problem from one answering badly on
    a short one and afterwards the trim is the only way to tell."""
    from tests import fake_typesafe

    _a, _s, fake = await _advise(
        fake_typesafe.FakeJev(answers=fake_typesafe.turn("report_state", 0.91)),
        monkeypatch,
        cleared=False,
        goal="Tidy the bench. " * 700,
        notes="a remembered fact worth keeping",
    )
    state, _questions = fake.calls[0]
    assert "goal" in state and "success_when" in state and "now" in state
    assert "notes" not in state and "camera" not in state, "the trim ran in its stated order"


async def test_a_state_nobody_could_answer_against_is_not_sent_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the fields that can never be dropped are themselves over the hard cap, the turn goes
    to the model without a request being made."""
    from tests import fake_typesafe

    fake = fake_typesafe.FakeJev(answers=fake_typesafe.turn("report_state", 0.99))
    fake_typesafe.install(monkeypatch, fake)
    stepper, adapter, _m = await _arm()
    stepper.goal = "x" * 40_000
    advice = await stepper.advise(await _observation(adapter), cleared=False, budget=BUDGET)
    await adapter.disconnect()
    assert advice.gate == "state_too_large" and advice.call is None
    assert fake.calls == []


# ── mode, availability, shadow ──────────────────────────────────────────────────────────


def test_the_stepper_is_off_unless_somebody_asks_for_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key in a `.env` file is somebody's other project. quackd never switches a paid
    dependency on because it found one lying about."""
    from quackd.agent.jev import resolve_jev_mode

    monkeypatch.delenv("QUACKD_JEV", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-live-nobody-asked")
    assert resolve_jev_mode(None) == "off"
    monkeypatch.setenv("QUACKD_JEV", "shadow")
    assert resolve_jev_mode(None) == "shadow"
    assert resolve_jev_mode("on") == "on", "the flag beats the environment"
    assert resolve_jev_mode("OFF") == "off"


def test_an_unknown_mode_is_refused_by_name() -> None:
    from quackd.agent.jev import resolve_jev_mode

    with pytest.raises(ValueError, match="unknown --jev mode 'maybe'"):
        resolve_jev_mode("maybe")


def test_without_the_extra_the_stepper_says_which_install_it_wants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phrased like `ProviderNotInstalled` next door, because a reader who has met that one
    should recognise this."""
    import sys

    from quackd.agent.jev import jev_is_available

    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    ok, why = jev_is_available()
    assert not ok and "quackd[jev]" in why


def test_with_the_extra_and_no_key_the_stepper_says_which_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quackd.agent.jev import jev_is_available
    from tests import fake_typesafe

    fake_typesafe.install(monkeypatch, fake_typesafe.FakeJev())
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    ok, why = jev_is_available()
    assert not ok and "TYPESAFE_API_KEY" in why
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test")
    assert jev_is_available() == (True, "")


async def test_shadow_records_what_the_stepper_would_have_done_beside_what_the_model_did(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The record that turns the arithmetic in docs/jev.md into a measurement."""
    from quackd.agent.providers.base import ToolCall
    from tests import fake_typesafe

    advice, stepper, _f = await _advise(
        fake_typesafe.FakeJev(answers=fake_typesafe.turn("report_state", 0.91)),
        monkeypatch,
        mode="shadow",
        cleared=False,
    )
    agreed = stepper.shadow_event(
        advice, ToolCall(name="report_state"), {"latency_s": 8.2, "usage": {"input_tokens": 4465}}
    )
    assert agreed["agree"] is True and agreed["would_have_acted"] is True
    assert agreed["llm_latency_s"] == 8.2 and agreed["jev_choice"] == "report_state"

    differed = stepper.shadow_event(advice, ToolCall(name="move_joints"), None)
    assert differed["agree"] is False and differed["model_verb"] == "move_joints"


async def test_the_summary_block_counts_the_turns_it_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests import fake_typesafe

    _a, stepper, _f = await _advise(
        fake_typesafe.FakeJev(answers=fake_typesafe.turn("report_state", 0.91)),
        monkeypatch,
        cleared=False,
    )
    block = stepper.summary()
    assert block["asked"] == 1 and block["taken"] == 1 and block["errors"] == 0
    assert block["mode"] == "on" and block["model"].startswith("jev-")
