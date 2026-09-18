"""The discrete stepper: which turns are a choice, and which belong to the model.

Nothing here imports `typesafe_sdk`, and nothing here reaches the network. The classification
is pure: it reads a tool's JSON Schema and answers, so it is testable against every body quackd
ships without connecting to any of them.

The table below is frozen on purpose, the way `MOVES_THE_BODY` is: a verb added to an adapter
with a parameter nobody thought about is a verb the stepper would either offer or refuse
silently, and this is the test that makes somebody say which.
"""

from __future__ import annotations

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
