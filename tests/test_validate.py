"""`validate_duck`: one wording for "this robot cannot do this task", used everywhere."""

from __future__ import annotations

from quackd.adapters.manifest import RobotManifest, VerbSpec
from quackd.adapters.microduck import microduck_manifest
from quackd.duckfile.parser import load_duck, parse_duck_text
from quackd.duckfile.validate import validate_duck

# A fixed camera with a voice, no legs: a hypothetical bolted-down body, used only to
# exercise validate_duck against a robot that cannot move or kick.
HEAD = RobotManifest(
    id="head-01",
    vendor="acme",
    model="fixed-cam",
    embodiment="arm",
    mobility="none",
    intents=["gaze", "sound"],
    sensors=["camera"],
    verbs=[
        VerbSpec(name="observe", core=True),
        VerbSpec(name="report_state", core=True),
        VerbSpec(name="say", core=True),
        VerbSpec(name="search_scan", core=True),
        VerbSpec(name="gaze"),
        VerbSpec(name="express"),
    ],
)
DUCK = microduck_manifest("mock", "duck-01")


def test_find_and_kick_against_a_head_names_the_missing_verbs() -> None:
    problems = validate_duck(load_duck("find-and-kick"), [HEAD])
    messages = [p.message for p in problems]
    assert "requires kick, but head-01 (fixed-cam) does not provide it" in messages
    assert "requires walk_to, but head-01 (fixed-cam) does not provide it" in messages
    assert all(p.field == "requires" for p in problems)
    assert {p.verb for p in problems} == {
        "walk_to",
        "kick",
        "quack",
    }  # get_frame -> observe is fine


def test_bundled_ducks_validate_against_the_microduck() -> None:
    for name in ("hello-world", "find-and-kick", "patrol-and-quack", "follow-me", "fetch"):
        assert validate_duck(load_duck(name), [DUCK]) == [], name
    assert validate_duck(load_duck("find-and-kick")) == []  # registry vocabulary, no manifests


def test_v1_requires_is_what_is_checked_and_allow_is_advisory() -> None:
    duck = parse_duck_text(
        "---\nduck: 1\nname: t\ndescription: d\nrequires: [observe, say]\n"
        "verbs:\n  allow: [observe, say, gaze, kick, stop]\nsuccess: [x]\n---\n# T\nx\n"
    )
    problems = validate_duck(duck, [HEAD])
    assert [(p.field, p.verb) for p in problems] == [("verbs.allow", "kick")]
    assert "kick is not provided by head-01" in problems[0].message


def test_flock_requires_is_a_union_and_roles_must_be_fillable() -> None:
    duck = parse_duck_text(
        "---\nduck: 1\nname: t\ndescription: d\nrequires: [observe, kick]\n"
        "verbs:\n  allow: [observe, gaze, go_to, kick, stop, express]\nsuccess: [x]\n"
        "flock:\n  members: [head-01, duck-01]\n  roles:\n"
        "    spotter: {requires: [observe, gaze]}\n    kicker: {requires: [go_to, kick]}\n"
        "---\n# T\nx\n"
    )
    assert validate_duck(duck, [HEAD, DUCK]) == []
    only_heads = validate_duck(duck, [HEAD])
    assert any(p.field == "flock.roles.kicker" for p in only_heads)
    assert any(p.field == "requires" and p.verb == "kick" for p in only_heads)
    two_ducks = validate_duck(duck, [DUCK, microduck_manifest("mock", "duck-02")])
    assert [p.field for p in two_ducks] == ["verbs.allow"]  # two ducks can spot AND kick


def test_the_registry_vocabulary_rules_stay_worded_as_before() -> None:
    duck = parse_duck_text(
        "---\nduck: 0\nname: t\ndescription: d\nverbs:\n  allow: [fly, kick]\n  confirm: [kick]\n"
        "success: [x]\nflock:\n  members: 2\n---\n# T\nx\n"
    )
    messages = [str(p) for p in validate_duck(duck)]
    assert messages == [
        "verbs.allow: unknown verbs: fly",
        "verbs.confirm: a flock cannot prompt y/N per duck: empty verbs.confirm",
    ]
