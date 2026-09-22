"""`quackd robot add|list|show|edit|remove`, and what a registered name means elsewhere.

The group is modelled on `quackd memory`, so this file is modelled on `test_cli_memory.py`:
every refusal is one line and never a traceback, `--json` is one object per line and never
prints a token, and a destructive command asks unless it is told not to.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quackd.cli import app
from quackd.registry import Registry, RobotEntry, StoredFlock
from quackd_lerobot.mock import REST

runner = CliRunner()

FOLDED_LIFT = -113.5
"""The bench arm's folded `shoulder_lift`, kept here so a stored pose in a test is a pose a
real arm was actually left in rather than a round number."""


def _reg(tmp_path: Path) -> list[str]:
    return ["--registry-dir", str(tmp_path)]


def _seed(tmp_path: Path, **kw: object) -> Registry:
    registry = Registry(tmp_path)
    registry.add_robot(RobotEntry(name="duck-a", spec="microduck:mock", **kw))  # type: ignore[arg-type]
    registry.add_robot(RobotEntry(name="arm", spec="lerobot:mock"))
    return registry


def _seed_arm(tmp_path: Path, **kw: object) -> Registry:
    """An arm on its own: `lerobot` is the one body quackd drives to a rest pose."""
    registry = Registry(tmp_path)
    registry.add_robot(RobotEntry(name="arm-01", spec="lerobot:mock", **kw))  # type: ignore[arg-type]
    return registry


# ── the round trip ──────────────────────────────────────────────────────────────────────


def test_add_list_show_edit_remove(tmp_path: Path) -> None:
    added = runner.invoke(
        app,
        ["robot", "add", "duck-a", "microduck:mock", "--note", "the cream one", *_reg(tmp_path)],
    )
    assert added.exit_code == 0, added.output
    assert "added duck-a: microduck:mock" in added.output

    listed = runner.invoke(app, ["robot", "list", *_reg(tmp_path)])
    assert listed.exit_code == 0, listed.output
    assert "duck-a" in listed.output and "microduck:mock" in listed.output

    shown = runner.invoke(app, ["robot", "show", "duck-a", *_reg(tmp_path)])
    assert shown.exit_code == 0, shown.output
    assert "the cream one" in shown.output
    assert "biped" in shown.output, "show carries the body's own description"

    edited = runner.invoke(
        app, ["robot", "edit", "duck-a", "--address", "tcp://10.0.0.5:9871", *_reg(tmp_path)]
    )
    assert edited.exit_code == 0, edited.output
    assert "updated duck-a: address" in edited.output
    assert Registry(tmp_path).robot("duck-a").address == "tcp://10.0.0.5:9871"

    removed = runner.invoke(app, ["robot", "remove", "duck-a", "--yes", *_reg(tmp_path)])
    assert removed.exit_code == 0, removed.output
    assert Registry(tmp_path).get_robot("duck-a") is None


def test_an_empty_registry_says_how_to_fill_it(tmp_path: Path) -> None:
    result = runner.invoke(app, ["robot", "list", *_reg(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "quackd robot add" in result.output


# ── refusals are one line ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("argv", "needle"),
    [
        (["robot", "add", "3", "microduck:mock"], "simulated ducks"),
        (["robot", "add", "microduck", "microduck:mock"], "is an adapter"),
        (["robot", "add", "Duck-A", "microduck:mock"], "not a valid robot name"),
        (["robot", "add", "duck-a", "bogus:nope"], "unknown adapter 'bogus'"),
        (["robot", "add", "duck-a", "microduck:bogus"], "unknown backend 'bogus'"),
        (["robot", "add", "duck-a", "microduck:mock", "--llm", "hal"], "unknown provider"),
        (["robot", "show", "ghost"], "no robot called 'ghost'"),
        (["robot", "edit", "ghost", "--note", "x"], "no robot called 'ghost'"),
        (["robot", "remove", "ghost", "--yes"], "no robot called 'ghost'"),
        (["robot", "rest-pose", "ghost", "--yes"], "no robot called 'ghost'"),
    ],
)
def test_a_refusal_is_one_line_and_never_a_traceback(
    tmp_path: Path, argv: list[str], needle: str
) -> None:
    result = runner.invoke(app, [*argv, *_reg(tmp_path)])
    assert result.exit_code == 1, result.output
    assert needle in result.output
    assert "Traceback" not in result.output


def test_the_same_name_twice_is_refused(tmp_path: Path) -> None:
    _seed(tmp_path)
    result = runner.invoke(app, ["robot", "add", "duck-a", "microduck:sim2d", *_reg(tmp_path)])
    assert result.exit_code == 1
    assert "already registered" in result.output


def test_edit_needs_something_to_change_and_refuses_a_contradiction(tmp_path: Path) -> None:
    _seed(tmp_path)
    nothing = runner.invoke(app, ["robot", "edit", "duck-a", *_reg(tmp_path)])
    assert nothing.exit_code == 1 and "nothing to change" in nothing.output
    both = runner.invoke(
        app, ["robot", "edit", "duck-a", "--note", "x", "--clear", "note", *_reg(tmp_path)]
    )
    assert both.exit_code == 1 and "contradict" in both.output
    unknown = runner.invoke(app, ["robot", "edit", "duck-a", "--clear", "wings", *_reg(tmp_path)])
    assert unknown.exit_code == 1 and "--clear wings" in unknown.output


def test_clear_empties_a_field(tmp_path: Path) -> None:
    _seed(tmp_path, note="a note", address="tcp://x:1")
    result = runner.invoke(app, ["robot", "edit", "duck-a", "--clear", "note", *_reg(tmp_path)])
    assert result.exit_code == 0, result.output
    entry = Registry(tmp_path).robot("duck-a")
    assert entry.note is None and entry.address == "tcp://x:1"


# ── the rest pose ───────────────────────────────────────────────────────────────────────


def test_rest_pose_records_the_arms_own_joints_and_show_prints_them(tmp_path: Path) -> None:
    """A LeRobot arm goes limp the moment it is disconnected, so an arm still standing when a
    run ends falls. The pose is read off the arm rather than typed, because the only pose worth
    returning to is one somebody folded the arm into by hand and watched it hold with torque
    off, and `show` prints every joint because that is how you check you recorded that one."""
    _seed_arm(tmp_path)
    recorded = runner.invoke(app, ["robot", "rest-pose", "arm-01", "--yes", *_reg(tmp_path)])
    assert recorded.exit_code == 0, recorded.output
    assert "recorded arm-01's rest pose" in recorded.output
    assert Registry(tmp_path).robot("arm-01").rest_pose == {
        joint: round(value, 1) for joint, value in REST.items()
    }, "the stored pose is where the arm was, rounded to a tenth of a degree"

    shown = runner.invoke(app, ["robot", "show", "arm-01", *_reg(tmp_path)])
    assert shown.exit_code == 0, shown.output
    flat = " ".join(shown.output.split())
    assert "rest pose shoulder_pan 0.0" in flat, flat
    assert "shoulder_lift -90.0" in flat, flat

    as_json = runner.invoke(app, ["robot", "show", "arm-01", "--json", *_reg(tmp_path)])
    assert as_json.exit_code == 0, as_json.output
    assert json.loads(as_json.output.strip())["rest_pose"]["shoulder_lift"] == -90.0


def test_rest_pose_asks_unless_yes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_arm(tmp_path)
    monkeypatch.setattr("quackd.cli._can_prompt", lambda: True)
    said_no = runner.invoke(app, ["robot", "rest-pose", "arm-01", *_reg(tmp_path)], input="n\n")
    assert said_no.exit_code == 0, said_no.output
    assert Registry(tmp_path).robot("arm-01").rest_pose is None, "answering no records nothing"
    said_yes = runner.invoke(app, ["robot", "rest-pose", "arm-01", *_reg(tmp_path)], input="y\n")
    assert said_yes.exit_code == 0, said_yes.output
    assert Registry(tmp_path).robot("arm-01").rest_pose is not None


def test_rest_pose_refuses_a_body_with_no_joints(tmp_path: Path) -> None:
    _seed(tmp_path)
    result = runner.invoke(app, ["robot", "rest-pose", "duck-a", *_reg(tmp_path)])
    assert result.exit_code == 1, result.output
    assert "has no joints" in " ".join(result.output.split())
    assert "Traceback" not in result.output


def test_rest_pose_refuses_a_body_with_joints_quackd_does_not_park(tmp_path: Path) -> None:
    """The XLeRobot lists a `joint` intent, so the joints gate lets it straight through. The
    second gate, on the adapter itself, is what stops a pose being recorded for a body no code
    would ever drive back to it, which would be a promise the arm keeps and this one does not."""
    registry = Registry(tmp_path)
    registry.add_robot(RobotEntry(name="xarm", spec="xlerobot:mock"))
    result = runner.invoke(app, ["robot", "rest-pose", "xarm", *_reg(tmp_path)])
    assert result.exit_code == 1, result.output
    flat = " ".join(result.output.split())
    assert "does not drive it to a rest pose yet" in flat, flat
    assert "has no joints" not in flat, "the joints gate is not the one that refused this body"
    assert Registry(tmp_path).robot("xarm").rest_pose is None


def test_rest_pose_clear_forgets_it_and_says_so_when_there_is_none(tmp_path: Path) -> None:
    _seed_arm(tmp_path, rest_pose={"shoulder_lift": FOLDED_LIFT})
    cleared = runner.invoke(app, ["robot", "rest-pose", "arm-01", "--clear", *_reg(tmp_path)])
    assert cleared.exit_code == 0, cleared.output
    assert "cleared arm-01's rest pose" in cleared.output
    assert Registry(tmp_path).robot("arm-01").rest_pose is None

    again = runner.invoke(app, ["robot", "rest-pose", "arm-01", "--clear", *_reg(tmp_path)])
    assert again.exit_code == 1, again.output
    assert "has no rest pose recorded" in " ".join(again.output.split())


def test_rest_pose_json_without_yes_is_refused(tmp_path: Path) -> None:
    """`--json` is for a script, and a script has no answer for a prompt. The pair is refused
    before anything connects, so the arm is not read and then abandoned at the prompt."""
    _seed_arm(tmp_path)
    result = runner.invoke(app, ["robot", "rest-pose", "arm-01", "--json", *_reg(tmp_path)])
    assert result.exit_code == 1, result.output
    assert "--json is for a script" in " ".join(result.output.split())
    assert Registry(tmp_path).robot("arm-01").rest_pose is None


def test_rest_pose_with_no_terminal_to_ask_on_refuses_rather_than_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing on stdin is an answer when there is nobody at the terminal: a `y` in a pipe is
    whatever the pipe happened to hold, and recording the wrong pose is how an arm is driven
    into the table on the next run. The refusal names `--yes`, which is the real answer."""
    _seed_arm(tmp_path)
    monkeypatch.setattr("quackd.cli._can_prompt", lambda: False)
    result = runner.invoke(app, ["robot", "rest-pose", "arm-01", *_reg(tmp_path)], input="y\n")
    assert result.exit_code == 1, result.output
    flat = " ".join(result.output.split())
    assert "no terminal to ask on" in flat, flat
    assert "--yes" in flat, flat
    assert Registry(tmp_path).robot("arm-01").rest_pose is None


def test_edit_clear_rest_pose_empties_it(tmp_path: Path) -> None:
    _seed_arm(tmp_path, rest_pose={"shoulder_lift": FOLDED_LIFT}, note="the bench arm")
    result = runner.invoke(
        app, ["robot", "edit", "arm-01", "--clear", "rest-pose", *_reg(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "updated arm-01: rest-pose" in result.output
    entry = Registry(tmp_path).robot("arm-01")
    assert entry.rest_pose is None
    assert entry.note == "the bench arm", "clearing one field leaves the others alone"


# ── several cameras ─────────────────────────────────────────────────────────────────────


def test_camera_url_repeats_and_show_prints_every_camera(tmp_path: Path) -> None:
    """The first url is the primary, the camera the detections describe and the steering verbs
    read, so the pair is an ordered list and not a set. One camera is still stored as a bare
    string, which is why a robots.json written before an arm could have two still loads."""
    added = runner.invoke(
        app,
        [
            "robot",
            "add",
            "arm-real",
            "lerobot:real",
            "--camera-url",
            "opencv://0?name=top",
            "--camera-url",
            "opencv://1?name=side",
            *_reg(tmp_path),
        ],
    )
    assert added.exit_code == 0, added.output
    entry = Registry(tmp_path).robot("arm-real")
    assert entry.camera_url == ["opencv://0?name=top", "opencv://1?name=side"]
    assert entry.camera_urls == ("opencv://0?name=top", "opencv://1?name=side")

    shown = runner.invoke(app, ["robot", "show", "arm-real", *_reg(tmp_path)])
    assert shown.exit_code == 0, shown.output
    flat = " ".join(shown.output.split())
    assert "camera opencv://0?name=top opencv://1?name=side" in flat, flat

    as_json = runner.invoke(app, ["robot", "show", "arm-real", "--json", *_reg(tmp_path)])
    assert as_json.exit_code == 0, as_json.output
    assert json.loads(as_json.output.strip())["camera_url"] == [
        "opencv://0?name=top",
        "opencv://1?name=side",
    ], "--json carries the list, in the order the cameras were given"


def test_one_camera_url_on_edit_replaces_the_pair_and_clear_empties_it(tmp_path: Path) -> None:
    """`--camera-url` on an edit says where the cameras are today, which is the rule `--address`
    already follows: it replaces the whole stored set rather than adding a third camera to it."""
    registry = Registry(tmp_path)
    registry.add_robot(
        RobotEntry(
            name="arm-real",
            spec="lerobot:real",
            camera_url=["opencv://0?name=top", "opencv://1?name=side"],
        )
    )
    edited = runner.invoke(
        app, ["robot", "edit", "arm-real", "--camera-url", "opencv://2", *_reg(tmp_path)]
    )
    assert edited.exit_code == 0, edited.output
    replaced = Registry(tmp_path).robot("arm-real")
    assert replaced.camera_urls == ("opencv://2",)
    assert replaced.camera_url == "opencv://2", "one camera goes back to being a plain string"

    cleared = runner.invoke(
        app, ["robot", "edit", "arm-real", "--clear", "camera-url", *_reg(tmp_path)]
    )
    assert cleared.exit_code == 0, cleared.output
    emptied = Registry(tmp_path).robot("arm-real")
    assert emptied.camera_url is None and emptied.camera_urls == ()


def test_a_body_that_reads_one_camera_refuses_a_second(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "robot",
            "add",
            "duck-a",
            "microduck:mock",
            "--camera-url",
            "http://one:9872/snapshot.jpg",
            "--camera-url",
            "http://two:9872/snapshot.jpg",
            *_reg(tmp_path),
        ],
    )
    assert result.exit_code == 1, result.output
    flat = " ".join(result.output.split())
    assert "only lerobot:real takes several" in flat, flat
    assert Registry(tmp_path).get_robot("duck-a") is None


# ── flocks hold on to their robots ──────────────────────────────────────────────────────


def test_remove_refuses_a_robot_a_flock_names_and_force_drops_it(tmp_path: Path) -> None:
    registry = _seed(tmp_path)
    registry.add_flock(StoredFlock(name="kitchen", members=["duck-a", "arm"]))
    refused = runner.invoke(app, ["robot", "remove", "duck-a", "--yes", *_reg(tmp_path)])
    assert refused.exit_code == 1
    assert "kitchen" in refused.output and "--force" in refused.output
    assert Registry(tmp_path).get_robot("duck-a") is not None

    forced = runner.invoke(app, ["robot", "remove", "duck-a", "--yes", "--force", *_reg(tmp_path)])
    assert forced.exit_code == 0, forced.output
    assert "dropped from kitchen" in forced.output
    assert Registry(tmp_path).flock("kitchen").members == ["arm"]


def test_remove_asks_unless_yes(tmp_path: Path) -> None:
    _seed(tmp_path)
    said_no = runner.invoke(app, ["robot", "remove", "duck-a", *_reg(tmp_path)], input="n\n")
    assert said_no.exit_code == 0, said_no.output
    assert Registry(tmp_path).get_robot("duck-a") is not None
    said_yes = runner.invoke(app, ["robot", "remove", "duck-a", *_reg(tmp_path)], input="y\n")
    assert said_yes.exit_code == 0
    assert Registry(tmp_path).get_robot("duck-a") is None


# ── --json and --probe ──────────────────────────────────────────────────────────────────


def test_json_is_one_object_per_line_and_never_prints_the_token(tmp_path: Path) -> None:
    _seed(tmp_path, token="s3cret", address="tcp://x:1")
    result = runner.invoke(app, ["robot", "list", "--json", *_reg(tmp_path)])
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in result.output.strip().splitlines()]
    assert [r["name"] for r in rows] == ["arm", "duck-a"]
    assert "s3cret" not in result.output
    assert next(r for r in rows if r["name"] == "duck-a")["token_set"] is True
    one = runner.invoke(app, ["robot", "show", "duck-a", "--json", *_reg(tmp_path)])
    assert json.loads(one.output.strip())["address"] == "tcp://x:1"
    assert "s3cret" not in one.output


def test_list_is_static_unless_asked_to_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plain `list` must touch nothing: it is what you run to remember a name."""
    _seed(tmp_path)

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("robot list must not connect")

    monkeypatch.setattr("quackd.registry.probe_all", boom)
    assert runner.invoke(app, ["robot", "list", *_reg(tmp_path)]).exit_code == 0


def test_probe_marks_a_mock_reachable_and_a_dead_address_not(tmp_path: Path) -> None:
    registry = _seed(tmp_path)
    registry.add_robot(RobotEntry(name="far", spec="open_duck:bridge", address="tcp://127.0.0.1:1"))
    result = runner.invoke(
        app, ["robot", "list", "--probe", "--timeout", "3", "--json", *_reg(tmp_path)]
    )
    rows = {r["name"]: r for r in (json.loads(x) for x in result.output.strip().splitlines())}
    assert rows["duck-a"]["reachable"] is True
    assert rows["far"]["reachable"] is False and rows["far"]["probe"]
    assert result.exit_code == 1, "a robot that did not answer is a failing command"


# ── a registered name everywhere --robot is taken ───────────────────────────────────────


def test_a_run_by_name_says_the_name_and_keys_its_memory_by_it(tmp_path: Path) -> None:
    _seed(tmp_path)
    memory = tmp_path / "mem"
    result = runner.invoke(
        app,
        [
            "run",
            "hello-world",
            "--robot",
            "duck-a",
            "--llm",
            "fake",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--no-gif",
            "--memory-dir",
            str(memory),
            *_reg(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "duck-a (microduck:mock)" in result.output
    assert (memory / "duck-a.jsonl").exists()
    assert not (memory / "microduck-mock.jsonl").exists(), "the name is the key, not the body"


def test_a_run_by_name_takes_the_robots_own_pilot_unless_a_flag_says_otherwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three places can name a pilot, and one spec now carries both halves of the answer.

    That last part is what changed the behaviour here rather than only the spelling. A vendor
    and a model used to be two independent flags, so `--model gemini-9` against a robot stored
    as OpenAI meant OpenAI serving a Gemini id -- a combination nobody typed on purpose and
    the parser could not see. One spec cannot be half overridden: `--llm gemini` is Gemini's
    default, full stop, and the stored `gpt-4o` goes with the vendor it was stored against.
    """
    _seed(tmp_path, llm="openai:gpt-4o")
    seen: list[tuple[str, str | None]] = []
    real = __import__("quackd.agent.providers.factory", fromlist=["make_provider"]).make_provider

    def recorder(name: str, **kw: object) -> object:
        seen.append((name, kw.get("model")))  # type: ignore[arg-type]
        return real("fake", duck_name=kw.get("duck_name"))  # type: ignore[arg-type]

    monkeypatch.setattr("quackd.agent.providers.factory.make_provider", recorder)
    common = [
        "run",
        "hello-world",
        "--robot",
        "duck-a",
        "--runs-dir",
        str(tmp_path / "r"),
        "--no-gif",
        "--memory-dir",
        str(tmp_path / "m"),
        *_reg(tmp_path),
    ]
    monkeypatch.delenv("QUACKD_LLM", raising=False)
    assert runner.invoke(app, common).exit_code == 0
    assert seen[-1] == ("openai", "gpt-4o")
    assert runner.invoke(app, [*common, "--llm", "gemini"]).exit_code == 0
    assert seen[-1] == ("gemini", None), "a bare vendor on the line is that vendor's default"
    assert runner.invoke(app, [*common, "--llm", "claude-opus-5"]).exit_code == 0
    assert seen[-1] == ("anthropic", "claude-opus-5"), "a bare id brings its own vendor"
    # the robot was registered by a person who meant it; a variable in their shell was not
    monkeypatch.setenv("QUACKD_LLM", "gemini")
    assert runner.invoke(app, common).exit_code == 0
    assert seen[-1] == ("openai", "gpt-4o"), "the environment does not beat the robot"


def test_a_run_by_name_is_refused_when_the_robots_own_model_is_no_longer_listed(
    tmp_path: Path,
) -> None:
    """The other half of "strict at the door, lenient on the shelf".

    `robots.json` still loads with a retired id, because refusing the read would take
    `quackd robot edit` down with it. The run is where it stops, and the refusal has to say
    which robot and which file: nothing on the command line is wrong, so a reader told only
    "unknown model" would search the line they just typed and find nothing to fix.
    """
    _seed(tmp_path, llm="openai:gpt-5")
    result = runner.invoke(
        app,
        [
            "run",
            "hello-world",
            "--robot",
            "duck-a",
            "--runs-dir",
            str(tmp_path / "r"),
            "--no-gif",
            "--memory-dir",
            str(tmp_path / "m"),
            *_reg(tmp_path),
        ],
    )
    assert result.exit_code == 1
    flat = " ".join(result.output.split())
    assert "unknown model 'gpt-5'" in flat
    assert "robot duck-a (robots.json)" in flat
    assert "Traceback" not in result.output


def test_a_pilot_the_catalogue_would_refuse_is_not_registered_at_all(tmp_path: Path) -> None:
    """The shelf is lenient about model ids and `robot add` is not, because the door is where
    the typo is actually made. A robot registered against `openai:grok-4.6` would look fine in
    `quackd robot list` and fail only on the day somebody ran it."""
    result = runner.invoke(
        app,
        ["robot", "add", "duck-a", "microduck:mock", "--llm", "openai:grok-4.6", *_reg(tmp_path)],
    )
    assert result.exit_code == 1
    flat = " ".join(result.output.split())
    assert "grok-4.6" in flat and "--llm grok:grok-4.6" in flat
    assert "Traceback" not in result.output
    assert not (tmp_path / "robots.json").exists(), "a refused add wrote the robot anyway"


def test_a_run_by_name_reaches_the_address_it_was_registered_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(tmp_path)
    Registry(tmp_path).update_robot("duck-a", {"address": "tcp://10.0.0.5:9871", "token": "s3cret"})
    seen: dict[str, object] = {}
    real = __import__("quackd.adapters.factory", fromlist=["make_adapter"]).make_adapter

    def recorder(spec: object, **kw: object) -> object:
        seen.update(kw)
        return real(spec, seed=kw.get("seed"))  # type: ignore[arg-type]

    monkeypatch.setattr("quackd.adapters.factory.make_adapter", recorder)
    result = runner.invoke(
        app,
        [
            "run",
            "hello-world",
            "--robot",
            "duck-a",
            "--llm",
            "fake",
            "--no-gif",
            "--runs-dir",
            str(tmp_path / "r"),
            "--memory-dir",
            str(tmp_path / "m"),
            *_reg(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["address"] == "tcp://10.0.0.5:9871"
    assert seen["token"] == "s3cret"


def test_a_flag_on_the_line_beats_the_stored_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(tmp_path)
    Registry(tmp_path).update_robot("duck-a", {"address": "tcp://stored:1", "token": "stored"})
    seen: dict[str, object] = {}
    real = __import__("quackd.adapters.factory", fromlist=["make_adapter"]).make_adapter

    def recorder(spec: object, **kw: object) -> object:
        seen.update(kw)
        return real(spec, seed=kw.get("seed"))  # type: ignore[arg-type]

    monkeypatch.setattr("quackd.adapters.factory.make_adapter", recorder)
    result = runner.invoke(
        app,
        [
            "run",
            "hello-world",
            "--robot",
            "duck-a",
            "--llm",
            "fake",
            "--no-gif",
            "--address",
            "tcp://tunnel:9999",
            "--runs-dir",
            str(tmp_path / "r"),
            "--memory-dir",
            str(tmp_path / "m"),
            *_reg(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["address"] == "tcp://tunnel:9999"
    assert seen["token"] == "stored", "one flag overrides one field"


def test_memory_commands_key_a_registered_robot_by_its_name(tmp_path: Path) -> None:
    _seed(tmp_path)
    memory = tmp_path / "mem"
    added = runner.invoke(
        app,
        [
            "memory",
            "add",
            "the charger is under the desk",
            "--robot",
            "duck-a",
            "--memory-dir",
            str(memory),
            *_reg(tmp_path),
        ],
    )
    assert added.exit_code == 0, added.output
    assert "remembered for duck-a" in added.output
    assert (memory / "duck-a.jsonl").exists()
    shown = runner.invoke(
        app, ["memory", "show", "--robot", "duck-a", "--memory-dir", str(memory), *_reg(tmp_path)]
    )
    assert "the charger is under the desk" in shown.output


def test_validate_takes_a_registered_name(tmp_path: Path) -> None:
    _seed(tmp_path)
    result = runner.invoke(app, ["validate", "hello-world", "--robot", "duck-a", *_reg(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "duck-a" in result.output, "the manifest id is the name you registered"


def test_an_unknown_bare_name_names_both_things_it_could_have_been(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "hello-world", "--robot", "ghost", *_reg(tmp_path)])
    assert result.exit_code == 1
    assert "neither a registered robot" in result.output
    assert "unknown adapter 'ghost'" in result.output, "the adapter's own words still show"


@pytest.mark.parametrize("command", ["add", "list", "show", "edit", "remove", "rest-pose"])
def test_every_robot_command_answers_help(command: str) -> None:
    result = runner.invoke(app, ["robot", command, "--help"])
    assert result.exit_code == 0, result.output
