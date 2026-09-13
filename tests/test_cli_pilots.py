"""`quackd run --flock`: a number is still the coordinator, a name is a flock of pilots.

The counterpart to `tests/test_pilots.py`, which drives the runner directly. This drives the
command a person types.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from quackd.cli import app
from quackd.registry import Registry, RobotEntry, StoredFlock

runner = CliRunner()


def _reg(tmp_path: Path) -> list[str]:
    return ["--registry-dir", str(tmp_path)]


def _run_args(tmp_path: Path) -> list[str]:
    return [
        "--provider",
        "fake",
        "--no-gif",
        "--no-trace",
        "--runs-dir",
        str(tmp_path / "runs"),
        "--memory-dir",
        str(tmp_path / "mem"),
        *_reg(tmp_path),
    ]


def _seed(tmp_path: Path, robots: dict[str, str], flocks: dict[str, list[str]]) -> Registry:
    registry = Registry(tmp_path)
    for name, spec in robots.items():
        registry.add_robot(RobotEntry(name=name, spec=spec))
    for name, members in flocks.items():
        registry.add_flock(StoredFlock(name=name, members=members))
    return registry


def _newest(tmp_path: Path) -> Path:
    return sorted((tmp_path / "runs").iterdir())[-1]


def _summary(tmp_path: Path) -> dict[str, Any]:
    return json.loads((_newest(tmp_path) / "summary.json").read_text(encoding="utf-8"))


# ── the number is still the coordinator ─────────────────────────────────────────────────


def test_flock_n_is_still_the_lockstep_simulator(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["run", "flock-kick", "--flock", "2", "--seed", "3", *_run_args(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    summary = _summary(tmp_path)
    assert summary["flock"] == {"members": ["duck-0", "duck-1"], "method": "auction"}
    assert "ball_displacement_m" in summary, "the coordinator still judges from the world"


def test_flock_one_still_says_two_to_four(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "flock-kick", "--flock", "1", *_run_args(tmp_path)])
    assert result.exit_code == 1
    assert "a flock needs 2 to 4 ducks" in result.output


def test_a_stored_flock_of_simulated_ducks_runs_the_coordinator(tmp_path: Path) -> None:
    _seed(
        tmp_path,
        {"left": "microduck:sim2d", "right": "microduck:sim2d"},
        {"pair": ["left", "right"]},
    )
    result = runner.invoke(
        app, ["run", "flock-kick", "--flock", "pair", "--seed", "3", *_run_args(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "pair: 2 ducks in sim2d" in result.output
    assert _summary(tmp_path)["flock"]["members"] == ["left", "right"]


# ── a name is a flock of pilots ─────────────────────────────────────────────────────────


def test_the_demo_duck_runs_with_no_registry_and_no_key(tmp_path: Path) -> None:
    """`quackd run flock-hello --provider fake` has to work in a fresh checkout."""
    result = runner.invoke(app, ["run", "flock-hello", *_run_args(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "members 2/2 succeeded" in result.output
    assert "talk 2" in result.output
    summary = _summary(tmp_path)
    assert summary["flock"]["method"] == "pilots"
    assert summary["robots"] == {"duck": "microduck:sim2d", "arm": "lerobot:mock"}
    for name in ("duck", "arm"):
        assert (_newest(tmp_path) / "ducks" / name / "transcript.jsonl").exists()


def test_three_different_bodies_run_as_one_stored_flock(tmp_path: Path) -> None:
    _seed(
        tmp_path,
        {"duck": "microduck:mock", "arm": "lerobot:mock", "cart": "rosbridge:mock"},
        {"trio": ["duck", "arm", "cart"]},
    )
    result = runner.invoke(app, ["run", "flock-hello", "--flock", "trio", *_run_args(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "members 3/3 succeeded" in result.output
    assert "stored as" in result.output and "trio" in result.output
    assert _summary(tmp_path)["flock"]["name"] == "trio"


def test_two_bodies_of_one_kind_run_and_keep_separate_notes(tmp_path: Path) -> None:
    _seed(
        tmp_path,
        {"duck-a": "microduck:mock", "duck-b": "microduck:mock"},
        {"twins": ["duck-a", "duck-b"]},
    )
    result = runner.invoke(app, ["run", "flock-hello", "--flock", "twins", *_run_args(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "mem" / "duck-a.jsonl").exists()
    assert (tmp_path / "mem" / "duck-b.jsonl").exists()
    assert not (tmp_path / "mem" / "microduck-mock.jsonl").exists()


def test_a_goal_can_be_given_to_a_stored_flock(tmp_path: Path) -> None:
    _seed(
        tmp_path,
        {"duck": "microduck:mock", "arm": "lerobot:mock"},
        {"pair": ["duck", "arm"]},
    )
    result = runner.invoke(
        app,
        [
            "run",
            "--goal",
            "say hello to your flock and stop",
            "--flock",
            "pair",
            *_run_args(tmp_path),
        ],
    )
    assert result.exit_code in (0, 1), result.output
    assert "members" in result.output, "it ran rather than refusing"


def test_a_member_that_fails_fails_the_flock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quackd.agent.providers.base import ToolCall
    from quackd.agent.providers.fake import FakeProvider

    def give_up(obs: Any, step: int, history: list[Any]) -> ToolCall:
        return ToolCall(name="declare_failure", arguments={"reason": "I do not want to"})

    monkeypatch.setattr(
        "quackd.agent.providers.factory.make_provider",
        lambda *a, **k: FakeProvider(strategy=give_up),
    )
    result = runner.invoke(app, ["run", "flock-hello", *_run_args(tmp_path)])
    assert result.exit_code == 1, result.output
    assert "members 0/2 succeeded" in result.output
    assert "I do not want to" in result.output


def test_dry_run_reaches_a_pilot_flock(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "flock-hello", "--dry-run", *_run_args(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "DRY RUN" in result.output
    assert _summary(tmp_path)["dry_run"] is True


def test_no_memory_leaves_nothing_behind(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "flock-hello", "--no-memory", *_run_args(tmp_path)])
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "mem").exists()


# ── refusals ────────────────────────────────────────────────────────────────────────────


def test_a_pilots_duck_under_flock_n_is_refused(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "flock-hello", "--flock", "3", *_run_args(tmp_path)])
    assert result.exit_code == 1
    assert "this task file runs pilots" in " ".join(result.output.split())


def test_a_coordinator_flock_of_real_bodies_points_at_pilots(tmp_path: Path) -> None:
    _seed(
        tmp_path,
        {"duck": "microduck:mock", "arm": "lerobot:mock"},
        {"pair": ["duck", "arm"]},
    )
    result = runner.invoke(app, ["run", "flock-kick", "--flock", "pair", *_run_args(tmp_path)])
    assert result.exit_code == 1, result.output
    assert "sim2d Microducks only" in result.output
    assert "method: pilots" in result.output


def test_a_coordinator_flock_of_five_is_refused(tmp_path: Path) -> None:
    robots = {f"duck-{i}": "microduck:sim2d" for i in range(5)}
    _seed(tmp_path, robots, {"five": list(robots)})
    result = runner.invoke(app, ["run", "flock-kick", "--flock", "five", *_run_args(tmp_path)])
    assert result.exit_code == 1, result.output
    assert "2 to 4 ducks" in result.output


def test_a_pilot_flock_of_one_is_refused(tmp_path: Path) -> None:
    _seed(tmp_path, {"duck": "microduck:mock"}, {"solo": ["duck"]})
    result = runner.invoke(app, ["run", "flock-hello", "--flock", "solo", *_run_args(tmp_path)])
    assert result.exit_code == 1, result.output
    assert "2 to 8 members" in result.output


def test_an_unknown_flock_names_both_things_it_could_have_been(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "flock-hello", "--flock", "ghost", *_run_args(tmp_path)])
    assert result.exit_code == 1
    assert "no flock called 'ghost'" in result.output
    assert "a number for N simulated ducks" in result.output


def test_a_dangling_flock_refuses_before_it_runs(tmp_path: Path) -> None:
    _seed(
        tmp_path,
        {"duck": "microduck:mock", "arm": "lerobot:mock"},
        {"pair": ["duck", "arm"]},
    )
    # by hand, which is the only way to make a flock dangle: `robot remove` refuses
    (tmp_path / "robots.json").write_text(
        json.dumps({"version": 1, "robots": {"duck": {"spec": "microduck:mock"}}}),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["run", "flock-hello", "--flock", "pair", *_run_args(tmp_path)])
    assert result.exit_code == 1, result.output
    assert "no longer registered" in result.output
    assert not (tmp_path / "runs").exists(), "nothing was written"


def test_a_stored_flock_brings_its_own_robots(tmp_path: Path) -> None:
    _seed(
        tmp_path,
        {"duck": "microduck:mock", "arm": "lerobot:mock"},
        {"pair": ["duck", "arm"]},
    )
    for extra in (["--robot", "duck"], ["--robots", "a=microduck:mock,b=lerobot:mock"]):
        result = runner.invoke(
            app, ["run", "flock-hello", "--flock", "pair", *extra, *_run_args(tmp_path)]
        )
        assert result.exit_code == 1, result.output
        assert "brings its own robots" in result.output


def test_a_gated_verb_needs_yes(tmp_path: Path) -> None:
    duck = tmp_path / "gated.duck"
    duck.write_text(
        "---\nduck: 1\nname: gated\ndescription: d\nverbs:\n  allow: [report_state, stop, pick]\n"
        "  confirm: [pick]\nsuccess: [x]\nrequires: [report_state]\n"
        "flock:\n  members: [duck, arm]\n  allocation:\n    method: pilots\n"
        "---\n# Task\nx\n",
        encoding="utf-8",
    )
    _seed(
        tmp_path,
        {"duck": "microduck:mock", "arm": "lerobot:mock"},
        {"pair": ["duck", "arm"]},
    )
    refused = runner.invoke(app, ["run", str(duck), "--flock", "pair", *_run_args(tmp_path)])
    assert refused.exit_code == 1, refused.output
    assert "cannot prompt y/N per member" in refused.output


def test_record_takes_a_count_and_not_a_stored_flock(tmp_path: Path) -> None:
    _seed(tmp_path, {"duck": "microduck:sim2d"}, {})
    result = runner.invoke(
        app,
        [
            "record",
            "flock-kick",
            "--flock",
            "pair",
            "--provider",
            "fake",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == 1, result.output
    assert "record pins the simulator" in result.output
