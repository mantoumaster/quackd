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

runner = CliRunner()


def _reg(tmp_path: Path) -> list[str]:
    return ["--registry-dir", str(tmp_path)]


def _seed(tmp_path: Path, **kw: object) -> Registry:
    registry = Registry(tmp_path)
    registry.add_robot(RobotEntry(name="duck-a", spec="microduck:mock", **kw))  # type: ignore[arg-type]
    registry.add_robot(RobotEntry(name="arm", spec="lerobot:mock"))
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
        (["robot", "add", "duck-a", "microduck:mock", "--provider", "hal"], "unknown provider"),
        (["robot", "show", "ghost"], "no robot called 'ghost'"),
        (["robot", "edit", "ghost", "--note", "x"], "no robot called 'ghost'"),
        (["robot", "remove", "ghost", "--yes"], "no robot called 'ghost'"),
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
            "--provider",
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
    _seed(tmp_path, provider="openai", model="gpt-5")
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
    assert runner.invoke(app, common).exit_code == 0
    assert seen[-1] == ("openai", "gpt-5")
    assert runner.invoke(app, [*common, "--provider", "gemini"]).exit_code == 0
    assert seen[-1] == ("gemini", "gpt-5"), "the flag names the vendor, the entry still the model"


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
            "--provider",
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
            "--provider",
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


@pytest.mark.parametrize("command", ["add", "list", "show", "edit", "remove"])
def test_every_robot_command_answers_help(command: str) -> None:
    result = runner.invoke(app, ["robot", command, "--help"])
    assert result.exit_code == 0, result.output
