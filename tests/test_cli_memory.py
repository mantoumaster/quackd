"""`quackd memory show|add|clear` and the memory flags on `run`.

The command group shipped with no test of its own, and two of its three subcommands were
wrong in ways only a person running them would see: a bad `--robot` printed a traceback
instead of the one-line error every sibling command prints, and `--raw`, whose whole job is
to print the file *as is*, ran it through Rich and silently ate anything in square brackets.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from quackd.cli import app
from quackd.memory import RobotMemory

runner = CliRunner()


def _mem(tmp_path: Path) -> list[str]:
    return ["--memory-dir", str(tmp_path)]


def test_add_then_show(tmp_path: Path) -> None:
    added = runner.invoke(
        app, ["memory", "add", "the charger is under the desk", "--tag", "place", *_mem(tmp_path)]
    )
    assert added.exit_code == 0, added.output
    assert "microduck:sim2d" in added.output
    shown = runner.invoke(app, ["memory", "show", *_mem(tmp_path)])
    assert shown.exit_code == 0, shown.output
    assert "the charger is under the desk" in shown.output
    assert "1 note," in shown.output


def test_show_raw_prints_the_file_as_is(tmp_path: Path) -> None:
    """A note is text a model wrote, so it can contain anything, including Rich markup."""
    note = "the ball is [bold]behind[/bold] the sofa"
    RobotMemory("microduck:sim2d", tmp_path).remember(note, now=1.0)
    raw = runner.invoke(app, ["memory", "show", "--raw", *_mem(tmp_path)])
    assert raw.exit_code == 0, raw.output
    assert note in raw.output, "--raw must not let Rich eat the markup it promises to print"
    assert json.loads(raw.output.strip())["text"] == note  # still one parseable JSON line


def test_show_and_add_and_clear_on_an_unknown_robot_are_one_line(tmp_path: Path) -> None:
    for argv in (
        ["memory", "show", "--robot", "bogus:nope"],
        ["memory", "add", "x", "--robot", "bogus:nope"],
        ["memory", "clear", "--yes", "--robot", "bogus:nope"],
    ):
        result = runner.invoke(app, [*argv, *_mem(tmp_path)])
        assert result.exit_code == 1, result.output
        assert "unknown adapter 'bogus'" in result.output
        assert "Traceback" not in result.output


def test_clear_on_an_empty_memory_says_so_and_succeeds(tmp_path: Path) -> None:
    result = runner.invoke(app, ["memory", "clear", "--yes", *_mem(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "nothing to forget" in result.output


def test_clear_forgets_and_show_survives_it(tmp_path: Path) -> None:
    mem = RobotMemory("microduck:sim2d", tmp_path)
    mem.remember("a", now=1.0)
    mem.record_episode(duck="d", outcome="success", reason="r", steps=1, now=2.0)
    cleared = runner.invoke(app, ["memory", "clear", "--yes", *_mem(tmp_path)])
    assert cleared.exit_code == 0 and "forgot 2 entries" in cleared.output
    assert not mem.path.exists()
    shown = runner.invoke(app, ["memory", "show", *_mem(tmp_path)])
    assert shown.exit_code == 0 and "nothing remembered yet" in shown.output


def test_a_run_writes_an_episode_and_the_next_run_is_told(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    common = ["run", "hello-world", "--llm", "fake", "--runs-dir", str(runs)]
    first = runner.invoke(app, [*common, *_mem(tmp_path)])
    assert first.exit_code == 0, first.output
    assert "0 notes, 0 earlier runs" in first.output
    second = runner.invoke(app, [*common, *_mem(tmp_path)])
    assert second.exit_code == 0, second.output
    assert "0 notes, 1 earlier runs" in second.output


def test_no_memory_writes_nothing_at_all(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    memory_dir = tmp_path / "mem"
    result = runner.invoke(
        app,
        [
            "run",
            "hello-world",
            "--llm",
            "fake",
            "--runs-dir",
            str(runs),
            "--no-memory",
            "--memory-dir",
            str(memory_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "earlier runs" not in result.output
    assert not memory_dir.exists(), "--no-memory must not even create the directory"
