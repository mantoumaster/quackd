"""`quackd log` replays a finished run. What it prints has to be what the run printed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quackd.cli import app

from .conftest import help_text

runner = CliRunner()


WIDE = {"COLUMNS": "200"}
"""Rich folds a panel at the terminal width, and at the default eighty columns the line of
counters breaks in the middle of `cost $0.0327`, with the panel's border sitting between the
two halves. Several tests here read that line, so everything here is run wide enough to leave
it whole."""


def _run_output(tmp_path: Path, *flags: str) -> str:
    """One real run under `tmp_path`, and what it printed.

    Says nothing about how many run directories are there afterwards: the label test below
    makes two on purpose."""
    result = runner.invoke(
        app,
        [
            "run",
            "find-and-kick",
            "--provider",
            "fake",
            "--seed",
            "3",
            "--robot",
            "microduck:sim2d",
            "--runs-dir",
            str(tmp_path),
            "--no-gif",
            *flags,
        ],
        env=WIDE,
    )
    assert result.exit_code == 0, result.output
    return result.output


def _run(tmp_path: Path, *flags: str) -> Path:
    """One real run under `tmp_path`, and the directory it wrote."""
    _run_output(tmp_path, *flags)
    (run_dir,) = [d for d in tmp_path.iterdir() if d.is_dir()]
    return run_dir


def _log_output(tmp_path: Path, *args: str) -> tuple[int, str]:
    """A replay's exit code and its output as it was printed, line breaks and all."""
    result = runner.invoke(app, ["log", *args, "--runs-dir", str(tmp_path)], env=WIDE)
    return result.exit_code, result.output


def _replay(tmp_path: Path, *args: str) -> tuple[int, str]:
    code, out = _log_output(tmp_path, *args)
    return code, " ".join(out.split())


def _counters(output: str) -> str:
    """The one line of counters under the verdict, without the panel's border and padding.

    Read as a line rather than as a substring of the whole output, so that "no cost counter"
    is a claim about the counters and not about whether the word appears somewhere on
    screen. Matched on `steps ` and not on `llm calls`, which the rule above every turn of a
    replay also says, as `step 3/40, llm calls 3/40`."""
    lines = [
        line.strip("│| ").strip()
        for line in output.splitlines()
        if "steps " in line and "llm calls" in line
    ]
    assert len(lines) == 1, output
    return lines[0]


def test_log_replays_a_finished_run(tmp_path: Path) -> None:
    """The point of the command: the run is over, and you want to read what happened."""
    run_dir = _run(tmp_path)
    code, out = _replay(tmp_path, run_dir.name)
    assert code == 0, out
    assert "system prompt" in out
    assert "send move" in out and "result walk_to ok" in out
    assert "[scripted]" in out  # the model's reasoning, as the run showed it
    assert "SUCCESS" in out and "llm calls" in out


def test_a_summary_from_before_the_stepper_was_priced_gains_no_cost_counter(
    tmp_path: Path,
) -> None:
    """Every `--jev` run recorded before this change already wrote a `jev` block, so gating the
    cost counter on that block gave those records a fourth counter reading `cost unpriced`
    that they never had and that says nothing true about them: nobody tried to price them.
    The gate is a cost key, and a stepper block without one is a run from before there were
    any."""
    run_dir = tmp_path / "20260101-120000-arm-grip-check"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "duck": "arm-grip-check",
                "outcome": "success",
                "reason": "the gripper reported closed",
                "steps": 6,
                "llm_calls": 4,
                "elapsed_s": 31.2,
                "usage": {"input_tokens": 9000, "output_tokens": 120},
                "jev": {
                    "mode": "on",
                    "model": "jev-1.13.0",
                    "asked": 6,
                    "taken": 4,
                    "errors": 0,
                    "latency_s": 0.7,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "transcript.jsonl").write_text("", encoding="utf-8")

    code, out = _replay(tmp_path, run_dir.name)
    assert code == 0, out
    assert "steps 6" in out and "llm calls 4" in out and "tokens 9000+120" in out
    assert "cost" not in out, out
    assert "time" not in out, "and no clock either: it recorded none"


def test_log_survives_a_summary_somebody_edited_by_hand(tmp_path: Path) -> None:
    """`quackd log` is pointed at files people edit, truncate and copy between machines. The
    renderers beside the counter line already shrug at a field that is not what it should be,
    and a counter is not worth a traceback."""
    run_dir = tmp_path / "20260101-120000-hand-edited"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "duck": "hand-edited",
                "outcome": "success",
                "reason": "somebody was in here with an editor",
                "steps": "six",
                "llm_calls": None,
                "usage": {},
                "wall_s": "a while",
                "cost_usd": "free",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "transcript.jsonl").write_text("", encoding="utf-8")

    code, out = _replay(tmp_path, run_dir.name)
    assert code == 0, out
    assert "cost unpriced" in out, "a cost nobody can read is not a cost of nothing"


def test_log_with_no_argument_picks_the_newest_run(tmp_path: Path) -> None:
    """What you want right after a run ends is the run that just ended."""
    _run(tmp_path)
    first = sorted(d.name for d in tmp_path.iterdir())[0]
    newest = tmp_path / "20991231-235959-later"
    newest.mkdir()
    (newest / "transcript.jsonl").write_text(
        json.dumps({"t": 0.0, "kind": "run_end", "outcome": "success", "reason": "the newer one"})
        + "\n",
        encoding="utf-8",
    )
    code, out = _replay(tmp_path)
    assert code == 0 and "the newer one" in out
    assert first not in out


def test_log_resolves_a_timestamp_prefix_and_a_duck_name(tmp_path: Path) -> None:
    run_dir = _run(tmp_path)
    stamp = run_dir.name.split("-")[0]
    for typed in (stamp, "find-and-kick"):
        code, out = _replay(tmp_path, typed)
        assert code == 0, (typed, out)
        assert "SUCCESS" in out


def test_log_accepts_the_transcript_file_itself(tmp_path: Path) -> None:
    """A transcript someone copied out of a run directory is still a run to read."""
    run_dir = _run(tmp_path)
    code, out = _replay(tmp_path, str(run_dir / "transcript.jsonl"))
    assert code == 0 and "result walk_to ok" in out


def test_log_from_step_skips_earlier_turns(tmp_path: Path) -> None:
    run_dir = _run(tmp_path)
    code, out = _replay(tmp_path, run_dir.name, "--from-step", "2")
    assert code == 0
    assert "step 2/40" in out
    assert "step 0/40" not in out and "step 1/40" not in out
    # the panel, not the run directory that also contains the name: a replay has no other
    # header, so this is the only thing saying what is being replayed
    assert "provider fake (scripted:find-and-kick)" in out


def test_log_no_prompt_and_thinking_flags(tmp_path: Path) -> None:
    run_dir = _run(tmp_path)
    _, full = _replay(tmp_path, run_dir.name)
    _, terse = _replay(tmp_path, run_dir.name, "--no-prompt", "--thinking", "0")
    assert "system prompt" in full and "[scripted]" in full
    assert "system prompt" not in terse and "[scripted]" not in terse
    assert "result walk_to ok" in terse  # everything else survives


def test_log_frames_are_off_unless_asked(tmp_path: Path) -> None:
    """One line per camera frame buries the four lines a reader came for."""
    run_dir = _run(tmp_path, "--vision")
    frames = [
        r
        for r in (json.loads(x) for x in (run_dir / "transcript.jsonl").read_text().splitlines())
        if r["kind"] == "frame"
    ]
    if not frames:  # a run without a camera has nothing to hide
        return
    needle = Path(frames[0]["path"]).name  # the prompt itself says "camera frame"
    _, without = _replay(tmp_path, run_dir.name, "--no-prompt")
    _, with_them = _replay(tmp_path, run_dir.name, "--no-prompt", "--frames")
    assert needle not in without
    assert needle in with_them


def test_log_of_a_transcript_without_verb_end_still_shows_results(tmp_path: Path) -> None:
    """The bundled example predates the log: it has `verb` records and no `verb_end`, and
    a replay that showed no results at all would be a replay of nothing."""
    code, out = _replay(tmp_path, "docs/assets/transcript-example.jsonl")
    assert code == 0, out
    assert "result search_scan" in out or "result quack" in out
    assert out.count("result quack ok") <= 1, "a live transcript has both kinds and must not double"


def test_log_of_a_cut_transcript_says_so(tmp_path: Path) -> None:
    """A run killed mid-write leaves half a line. Show the run, and say a line was lost."""
    run_dir = _run(tmp_path)
    path = run_dir / "transcript.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + '{"t": 9.9, "kind": "not', encoding="utf-8")
    code, out = _replay(tmp_path, run_dir.name)
    assert code == 0, out
    assert "1 unreadable line(s) skipped" in out
    assert "SUCCESS" in out


def test_log_on_a_missing_run_is_one_line(tmp_path: Path) -> None:
    _run(tmp_path)
    code, out = _replay(tmp_path, "no-such-run")
    assert code == 1
    assert "no run matching" in out and "newest:" in out


def test_log_on_an_empty_runs_dir_says_what_to_do(tmp_path: Path) -> None:
    code, out = _replay(tmp_path)
    assert code == 1 and "quackd run" in out


def test_log_on_a_flock_run_names_every_member(tmp_path: Path) -> None:
    """A flock writes no top-level transcript: the record is one file per robot, and a
    replay that read only the first of them would quietly show a third of the run."""
    result = runner.invoke(
        app,
        [
            "run",
            "flock-kick",
            "--provider",
            "fake",
            "--seed",
            "3",
            "--no-gif",
            "--runs-dir",
            str(tmp_path),
            "--no-log",
        ],
    )
    assert result.exit_code == 0, result.output
    code, out = _replay(tmp_path, "--no-prompt", "--thinking", "0")
    assert code == 0, out
    for name in ("duck-0", "duck-1", "duck-2"):
        # the glyph gutter sits between the member's name and the label it prefixes
        assert f"{name} ▶ verb" in out, (name, out[:400])
        assert f"{name} ■ end stopped after" in out
    # the outcome comes from summary.json, since no member wrote a run_end
    assert "SUCCESS" in out and "kicker duck-" in out
    assert "llm calls" not in out, "a flock counts auctions and bids, not steps and tokens"


# ── the clocks, the price, and the name the run was given ───────────────────────────────


def test_a_replay_prints_the_time_and_the_cost_the_run_itself_printed(tmp_path: Path) -> None:
    """A replay is the run again, so its counters have to be the same line rather than a
    second dialect that drifts from the first: the live verdict, `run_end` and `summary.json`
    are one dict through one function, and this asserts the two lines character for character.

    The header is what a replay has to add for itself. Until the record carried a wall clock
    the directory name was the only thing that said when a run happened, and a directory gets
    copied and renamed; the price is there because a cost figure with no rate beside it is a
    number nobody can argue with."""
    live = _run_output(tmp_path, "--run-name", "Example 1", "--price", "in=3,out=15")
    counters = _counters(live)
    assert "time " in counters and "cost $" in counters, counters

    code, out = _log_output(tmp_path, "example-1")
    assert code == 0, out
    assert _counters(out) == counters

    (run_dir,) = [d for d in tmp_path.iterdir() if d.is_dir()]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    flat = " ".join(out.split())
    started = str(summary["started_at"]).replace("T", " ").replace("Z", " UTC")
    assert f"started {started}" in flat, flat[:400]
    assert "run name Example 1" in flat, "the header shows the name as typed, not the slug"
    assert "price $3/M in, $15/M out (--price)" in flat


def test_log_prefers_the_run_whose_name_ends_in_the_label_over_a_newer_one(
    tmp_path: Path,
) -> None:
    """The whole reason the `--run-name` pass exists, and the ten minutes it saves.

    A bench session has `-example-1` and `-example-19` in it. The loose substring match that
    used to be the last resort walks the runs newest first and returns the first name the text
    appears in, which is `-example-19` every time: you type the name you gave a run and get a
    different run, with nothing on screen to say so."""
    _run_output(tmp_path, "--run-name", "example 1")
    _run_output(tmp_path, "--run-name", "example 19")
    names = sorted(d.name for d in tmp_path.iterdir() if d.is_dir())
    (one,) = [n for n in names if n.endswith("-example-1")]
    (nineteen,) = [n for n in names if n.endswith("-example-19")]
    assert nineteen > one, "the trap is that the decoy sorts newest; these two do not"

    code, out = _replay(tmp_path, "example-1")
    assert code == 0, out
    assert nineteen not in out, "the substring pass answered before the label pass"
    assert one in out


def test_a_summary_written_before_the_clocks_replays_with_the_counters_it_had(
    tmp_path: Path,
) -> None:
    """The compatibility guarantee, written out as the thing it protects: a summary from
    before this release carries steps, llm_calls, usage, outcome and reason and nothing else,
    and `quackd log` has to replay it with exactly the three counters it printed then.

    Not `time None`, and not a cost of zero dollars for a run nobody priced. Every field
    `run_counters` reads is optional for this reason, and the time and the cost appear only
    when the summary really carries them.

    The transcript here has no `run_end` in it, which is what sends the replay to
    `summary.json` for the outcome: a run from back then, read through the file that is meant
    to outlive its transcript."""
    run_dir = tmp_path / "20260101-120000-hello-world"
    run_dir.mkdir()
    (run_dir / "transcript.jsonl").write_text(
        json.dumps(
            {"t": 0.4, "kind": "verb_end", "name": "quack", "outcome": "ok", "summary": "quacked"}
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "steps": 3,
                "llm_calls": 5,
                "usage": {"input_tokens": 120, "output_tokens": 40},
                "outcome": "success",
                "reason": "quacked and walked one step",
            }
        ),
        encoding="utf-8",
    )
    code, out = _log_output(tmp_path, run_dir.name)
    assert code == 0, out
    counters = _counters(out)
    assert counters.startswith("steps 3"), counters
    assert "llm calls 5" in counters and "tokens 120+40" in counters, counters
    assert "time" not in counters and "cost" not in counters, counters
    assert "SUCCESS" in out and "quacked and walked one step" in out


# ── the spellings the trace left behind, kept until 0.12 ────────────────────────────────


@pytest.fixture(autouse=True)
def _forget_earlier_deprecations(monkeypatch: pytest.MonkeyPatch) -> None:
    """`quackd.cli._DEPRECATIONS` is filled once per process and never emptied.

    That is right for a command that runs once and wrong for a suite that invokes the CLI a
    hundred times in one interpreter: the saved terminal replays that list into its header,
    so without this a run here opens with the old spellings an earlier test typed."""
    monkeypatch.setattr("quackd.cli._DEPRECATIONS", [])


def _typed(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    """What the deprecation warner reads.

    `_warn_old_spellings` goes to `sys.argv` rather than to the parsed value, because Click
    hands both spellings of an option to the same parameter and by then they cannot be told
    apart. CliRunner passes its arguments to the command directly and leaves `sys.argv`
    alone, so without this the warner is reading pytest's own command line."""
    monkeypatch.setattr(sys, "argv", ["quackd", *argv])


def _deprecations(stderr: str) -> list[str]:
    """Every deprecation line on stderr and nothing else, so "exactly one" is a claim about
    how many were printed rather than about whether the text appears somewhere."""
    return [line.strip() for line in stderr.splitlines() if "goes in 0.12" in line]


def _flag_warning(old: str, new: str) -> str:
    """The line an old flag prints, spelled out here so a change to it has to be deliberate."""
    return f"the flag `{old}` is now `{new}`; the old spelling still works and goes in 0.12"


def _name_warning(old: str, new: str) -> str:
    """The same, for an environment variable, which says name where a flag says spelling."""
    return f"{old} is now {new}; the old name still works and goes in 0.12"


def _narration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *flags: str
) -> tuple[str, list[str]]:
    """One run on the mock, what it narrated, and the deprecation lines it printed.

    The mock sends no frames and writes no GIF, so stderr carries the live view and the
    warnings and nothing else. The suite runs with the live view off (conftest), which is
    what makes a flag that turns it on visible here as a change."""
    _typed(monkeypatch, "run", *flags)
    result = runner.invoke(
        app,
        [
            "run",
            "hello-world",
            "--provider",
            "fake",
            "--robot",
            "microduck:mock",
            "--no-gif",
            "--runs-dir",
            str(tmp_path),
            *flags,
        ],
        env=WIDE,
    )
    assert result.exit_code == 0, result.output
    return " ".join(result.stderr.split()), _deprecations(result.stderr)


def _narrow_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *flags: str) -> list[str]:
    """The same run at the eighty columns a terminal has when nobody widened it, line by line.

    Everything else in this file runs at two hundred so that the panel's counters stay whole,
    and that width is exactly what would hide a warning folded at eighty. Not flattened
    either, for the same reason: a fold is a line break and nothing else."""
    _typed(monkeypatch, "run", *flags)
    result = runner.invoke(
        app,
        [
            "run",
            "hello-world",
            "--provider",
            "fake",
            "--robot",
            "microduck:mock",
            "--no-gif",
            "--runs-dir",
            str(tmp_path),
            *flags,
        ],
        env={"COLUMNS": "80"},
    )
    assert result.exit_code == 0, result.output
    return [line.strip() for line in result.stderr.splitlines()]


def _terminal(tmp_path: Path) -> list[str]:
    """The saved terminal of the one run under `tmp_path`, line by line.

    Read off the disk rather than out of the runner's streams, because what this file has to
    keep is what a reader opens afterwards and not what the run happened to print."""
    (path,) = list(tmp_path.rglob("terminal.txt"))
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def test_the_help_offers_log_and_keeps_trace_hidden() -> None:
    """A spelling kept alive for one release must not also still be taught. `trace` answers
    for anyone who types it and appears nowhere, so nobody learns the name that goes next."""
    flat = help_text(["--help"])
    assert "log Replay a finished run" in flat
    assert "trace" not in flat, flat


def test_the_trace_command_replays_what_log_replays_and_says_so_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The alias is one command under two names, not a second renderer to keep in step: what
    it prints is the same bytes, plus one line saying what to type next time.

    The line goes to stderr, so a replay piped into a pager or a file is byte for byte the
    replay it was before, warning and all."""
    run_dir = _run(tmp_path)
    _typed(monkeypatch, "log", run_dir.name)
    new = runner.invoke(app, ["log", run_dir.name, "--runs-dir", str(tmp_path)], env=WIDE)
    _typed(monkeypatch, "trace", run_dir.name)
    old = runner.invoke(app, ["trace", run_dir.name, "--runs-dir", str(tmp_path)], env=WIDE)

    assert new.exit_code == 0 and old.exit_code == 0, old.output
    flat = " ".join(new.stdout.split())
    assert "SUCCESS" in flat and "result walk_to ok" in flat, "the new spelling replayed a run"
    assert old.stdout == new.stdout
    assert _deprecations(new.stderr) == []
    assert _deprecations(old.stderr) == [
        "the command `trace` is now `log`; the old spelling still works and goes in 0.12"
    ]


@pytest.mark.parametrize("on", [True, False])
def test_the_old_trace_flags_still_switch_the_live_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, on: bool
) -> None:
    """`--trace/--no-trace` is a second spelling on the same option rather than a second
    option, so what has to hold is that it lands on the same switch: the run narrates, or it
    does not, exactly as the new spelling leaves it, and saying it the old way costs one
    line."""
    new, old = ("--log", "--trace") if on else ("--no-log", "--no-trace")
    with_new, quiet = _narration(tmp_path, monkeypatch, new)
    with_old, warned = _narration(tmp_path, monkeypatch, old)
    assert ("system prompt" in with_new) is on
    assert ("result quack ok" in with_new) is on
    assert ("system prompt" in with_old) is on
    assert ("result quack ok" in with_old) is on
    assert quiet == []
    assert warned == [_flag_warning(old, new)]


@pytest.mark.parametrize("shown", [True, False])
def test_the_old_trace_prompt_flags_still_switch_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shown: bool
) -> None:
    """The same arrangement on the prompt switch. Both runs keep `--log`, because what this
    flag decides is the sixty lines at the top of a log that is running either way, and the
    rest of the log has to still be there to show that only the prompt went."""
    new, old = (
        ("--log-prompt", "--trace-prompt") if shown else ("--no-log-prompt", "--no-trace-prompt")
    )
    with_new, quiet = _narration(tmp_path, monkeypatch, "--log", new)
    with_old, warned = _narration(tmp_path, monkeypatch, "--log", old)
    assert ("system prompt" in with_new) is shown
    assert ("system prompt" in with_old) is shown
    assert "result quack ok" in with_old
    assert quiet == []
    assert warned == [_flag_warning(old, new)]


def test_the_old_env_names_are_read_only_where_the_new_one_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`QUACKD_TRACE` sits in `.env` files people wrote months ago, and switching their log
    back on would be the worse half of a rename. It is read where `QUACKD_LOG` is not there,
    and only there: a reader who has already moved over hears nothing about either name.

    `quackd.log` remembers which old names it has warned about so that a run prints one line
    and not one per lookup, which makes the set a decision an earlier test would otherwise
    get to make. Cleared here, twice, so each half of this asserts about its own run."""
    monkeypatch.setattr("quackd.log._warned", set())
    monkeypatch.delenv("QUACKD_LOG", raising=False)
    monkeypatch.setenv("QUACKD_TRACE", "0")
    fallback, warned = _narration(tmp_path, monkeypatch)
    assert "system prompt" not in fallback and "result quack ok" not in fallback
    assert warned == [_name_warning("QUACKD_TRACE", "QUACKD_LOG")]

    monkeypatch.setattr("quackd.log._warned", set())
    monkeypatch.setenv("QUACKD_LOG", "")  # an empty value is on
    live, quiet = _narration(tmp_path, monkeypatch)
    assert "system prompt" in live and "result quack ok" in live
    assert quiet == [], "nothing fell back, so there was nothing to say"


def test_a_run_named_trace_is_not_somebody_typing_the_old_spelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The warner reads `sys.argv`, where a flag's value sits beside the flag and is another
    word on the line. `--run-name trace` is what the afternoon of this rename looks like on
    disk, and a migration notice for it is a notice about nothing: what was named `trace` is
    a run, and the reader is already typing the new spelling of everything else.

    The second half is the same word as a positional. Somebody who named a run `trace`
    replays it by that name, and a name is not a subcommand."""
    _, deprecations = _narration(tmp_path, monkeypatch, "--run-name", "trace")
    assert deprecations == []
    (run_dir,) = [d for d in tmp_path.iterdir() if d.is_dir()]
    assert run_dir.name.endswith("-trace"), f"the value never reached the run: {run_dir.name}"

    _typed(monkeypatch, "log", "trace")
    result = runner.invoke(app, ["log", "trace", "--runs-dir", str(tmp_path)], env=WIDE)
    assert result.exit_code == 0, result.output
    assert _deprecations(result.stderr) == []
    assert "SUCCESS" in result.stdout, "and the run it names is the one that came back"


def test_a_root_option_before_the_old_subcommand_does_not_hide_the_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The subcommand is the first argument that is not an option, not `argv[1]`.

    A script that says `quackd --no-color trace <run>` is exactly the script this notice is
    written for: it is automated, nobody is reading its output for fun, and it breaks in 0.12.
    Reading only `argv[1]` told that script nothing at all, which is the one audience a
    deprecation cannot afford to be silent for."""
    run_dir = _run(tmp_path)
    where = ["--runs-dir", str(tmp_path)]

    _typed(monkeypatch, "--no-color", "trace", run_dir.name)
    old = runner.invoke(app, ["--no-color", "trace", run_dir.name, *where], env=WIDE)
    assert old.exit_code == 0, old.output
    assert _deprecations(old.stderr) == [
        "the command `trace` is now `log`; the old spelling still works and goes in 0.12"
    ]

    # and the guard in the other direction still holds: a root option in front of the NEW
    # spelling says nothing at all
    _typed(monkeypatch, "--no-color", "log", run_dir.name)
    new = runner.invoke(app, ["--no-color", "log", run_dir.name, *where], env=WIDE)
    assert new.exit_code == 0, new.output
    assert _deprecations(new.stderr) == []


def test_an_old_flag_beside_that_value_still_says_its_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A narrower reading is only right if it kept the case it was narrowed for. One command
    line carries both the decoy and a real old flag, and is told about the flag, once."""
    _, deprecations = _narration(tmp_path, monkeypatch, "--no-trace", "--run-name", "trace")
    assert deprecations == [_flag_warning("--no-trace", "--no-log")]


def test_a_deprecation_line_is_not_folded_at_eighty_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The longest of these sentences is a hundred characters, and it is an instruction a
    script greps for. Folded at the default width it breaks inside the new spelling, so the
    one part of the line worth reading is the part a grep no longer finds."""
    sentence = _flag_warning("--no-trace-prompt", "--no-log-prompt")
    assert len(sentence) > 80, "this stopped being a test about folding"
    lines = _narrow_run(tmp_path, monkeypatch, "--no-trace-prompt")
    assert sentence in lines, lines[:4]


def test_a_summary_written_before_the_rename_still_replays(tmp_path: Path) -> None:
    """A run directory is a file on disk that outlives the release that wrote it, and until
    0.11 the count of lines the console could not print was `trace_dropped`. Every reader
    takes both spellings, so the number a run really lost still reaches the panel.

    No `run_end` in the transcript, which is what sends the replay to `summary.json`: the
    file meant to outlive its transcript is the one that has to stay readable."""
    run_dir = tmp_path / "20260101-120000-before-the-rename"
    run_dir.mkdir()
    (run_dir / "transcript.jsonl").write_text(
        json.dumps(
            {"t": 0.4, "kind": "verb_end", "name": "quack", "outcome": "ok", "summary": "quacked"}
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "steps": 3,
                "llm_calls": 5,
                "usage": {"input_tokens": 120, "output_tokens": 40},
                "outcome": "success",
                "reason": "quacked and walked one step",
                "trace_dropped": 2,
            }
        ),
        encoding="utf-8",
    )
    code, out = _log_output(tmp_path, run_dir.name)
    assert code == 0, out
    assert _counters(out).startswith("steps 3"), out
    flat = " ".join(out.split())
    assert "SUCCESS" in flat and "quacked and walked one step" in flat
    assert "log: 2 line(s) could not be shown" in flat, "the old key never reached the panel"


@pytest.mark.parametrize(
    ("dropped", "says"),
    [
        ("2", ""),
        (2.7, "log: 2 line(s) could not be shown"),
        (None, ""),
        ("twenty", ""),
    ],
    ids=["a number in quotes", "a fraction", "null", "a word"],
)
def test_a_dropped_counter_nobody_can_read_replays_as_no_counter(
    tmp_path: Path, dropped: object, says: str
) -> None:
    """The one counter that reaches the panel as an `int()` rather than through `_number`
    used to be this one, and a summary somebody had opened in an editor ended the replay in a
    traceback instead of showing the run.

    A figure that is not a figure is not a count of lost lines, so the panel says nothing
    about dropped lines at all rather than guessing at one. A real number still counts, which
    is what the fraction is here for: it is the only one of these four that says anything."""
    run_dir = tmp_path / "20260101-120000-hand-edited-counter"
    run_dir.mkdir()
    (run_dir / "transcript.jsonl").write_text("", encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "steps": 3,
                "llm_calls": 5,
                "usage": {"input_tokens": 120, "output_tokens": 40},
                "outcome": "success",
                "reason": "quacked and walked one step",
                "log_dropped": dropped,
            }
        ),
        encoding="utf-8",
    )
    code, out = _log_output(tmp_path, run_dir.name)
    assert code == 0, out
    flat = " ".join(out.split())
    assert "SUCCESS" in flat and _counters(out).startswith("steps 3"), flat
    if says:
        assert says in flat, flat
    else:
        assert "could not be shown" not in flat, flat


# ── and what the saved terminal keeps of all this ───────────────────────────────────────


def test_an_old_flag_is_written_down_above_the_header_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The root callback says this before there is a run directory to write it into, which is
    the one thing a run says with nowhere to keep it. The header replays it rather than the
    warning being moved later, because on the screen it came before the panel and a saved
    terminal that reorders what you saw is a record of a different afternoon."""
    _, deprecations = _narration(tmp_path, monkeypatch, "--trace")
    assert deprecations == [_flag_warning("--trace", "--log")]

    lines = _terminal(tmp_path)
    assert lines[0].startswith("$ quackd run"), lines[:3]
    (at,) = [i for i, line in enumerate(lines) if line == _flag_warning("--trace", "--log")]
    panel = next(i for i, line in enumerate(lines) if "provider" in line)
    assert at < panel, lines[: panel + 1]


def test_the_old_environment_name_is_written_down_there_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This one is said from inside `quackd.log`, which is imported long before the consoles
    exist and knows nothing about the capture. It goes out through `ui.err_console` for the
    sake of this file: the reason a reader is looking at a log they did not ask for is a name
    in a `.env` file they wrote months ago, and the answer belongs where they will find it."""
    monkeypatch.setattr("quackd.log._warned", set())
    monkeypatch.delenv("QUACKD_LOG", raising=False)
    monkeypatch.setenv("QUACKD_TRACE", "0")
    _, deprecations = _narration(tmp_path, monkeypatch)
    assert deprecations == [_name_warning("QUACKD_TRACE", "QUACKD_LOG")]
    assert _name_warning("QUACKD_TRACE", "QUACKD_LOG") in _terminal(tmp_path)
