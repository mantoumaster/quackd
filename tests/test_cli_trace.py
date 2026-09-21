"""`quackd trace` replays a finished run. What it prints has to be what the run printed."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from quackd.cli import app

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


def _trace_output(tmp_path: Path, *args: str) -> tuple[int, str]:
    """A replay's exit code and its output as it was printed, line breaks and all."""
    result = runner.invoke(app, ["trace", *args, "--runs-dir", str(tmp_path)], env=WIDE)
    return result.exit_code, result.output


def _trace(tmp_path: Path, *args: str) -> tuple[int, str]:
    code, out = _trace_output(tmp_path, *args)
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


def test_trace_replays_a_finished_run(tmp_path: Path) -> None:
    """The point of the command: the run is over, and you want to read what happened."""
    run_dir = _run(tmp_path)
    code, out = _trace(tmp_path, run_dir.name)
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

    code, out = _trace(tmp_path, run_dir.name)
    assert code == 0, out
    assert "steps 6" in out and "llm calls 4" in out and "tokens 9000+120" in out
    assert "cost" not in out, out
    assert "time" not in out, "and no clock either: it recorded none"


def test_trace_survives_a_summary_somebody_edited_by_hand(tmp_path: Path) -> None:
    """`quackd trace` is pointed at files people edit, truncate and copy between machines. The
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

    code, out = _trace(tmp_path, run_dir.name)
    assert code == 0, out
    assert "cost unpriced" in out, "a cost nobody can read is not a cost of nothing"


def test_trace_with_no_argument_picks_the_newest_run(tmp_path: Path) -> None:
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
    code, out = _trace(tmp_path)
    assert code == 0 and "the newer one" in out
    assert first not in out


def test_trace_resolves_a_timestamp_prefix_and_a_duck_name(tmp_path: Path) -> None:
    run_dir = _run(tmp_path)
    stamp = run_dir.name.split("-")[0]
    for typed in (stamp, "find-and-kick"):
        code, out = _trace(tmp_path, typed)
        assert code == 0, (typed, out)
        assert "SUCCESS" in out


def test_trace_accepts_the_transcript_file_itself(tmp_path: Path) -> None:
    """A transcript someone copied out of a run directory is still a run to read."""
    run_dir = _run(tmp_path)
    code, out = _trace(tmp_path, str(run_dir / "transcript.jsonl"))
    assert code == 0 and "result walk_to ok" in out


def test_trace_from_step_skips_earlier_turns(tmp_path: Path) -> None:
    run_dir = _run(tmp_path)
    code, out = _trace(tmp_path, run_dir.name, "--from-step", "2")
    assert code == 0
    assert "step 2/40" in out
    assert "step 0/40" not in out and "step 1/40" not in out
    # the panel, not the run directory that also contains the name: a replay has no other
    # header, so this is the only thing saying what is being replayed
    assert "provider fake (scripted:find-and-kick)" in out


def test_trace_no_prompt_and_thinking_flags(tmp_path: Path) -> None:
    run_dir = _run(tmp_path)
    _, full = _trace(tmp_path, run_dir.name)
    _, terse = _trace(tmp_path, run_dir.name, "--no-prompt", "--thinking", "0")
    assert "system prompt" in full and "[scripted]" in full
    assert "system prompt" not in terse and "[scripted]" not in terse
    assert "result walk_to ok" in terse  # everything else survives


def test_trace_frames_are_off_unless_asked(tmp_path: Path) -> None:
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
    _, without = _trace(tmp_path, run_dir.name, "--no-prompt")
    _, with_them = _trace(tmp_path, run_dir.name, "--no-prompt", "--frames")
    assert needle not in without
    assert needle in with_them


def test_trace_of_a_transcript_without_verb_end_still_shows_results(tmp_path: Path) -> None:
    """The bundled example predates the trace: it has `verb` records and no `verb_end`, and
    a replay that showed no results at all would be a replay of nothing."""
    code, out = _trace(tmp_path, "docs/assets/transcript-example.jsonl")
    assert code == 0, out
    assert "result search_scan" in out or "result quack" in out
    assert out.count("result quack ok") <= 1, "a live transcript has both kinds and must not double"


def test_trace_of_a_cut_transcript_says_so(tmp_path: Path) -> None:
    """A run killed mid-write leaves half a line. Show the run, and say a line was lost."""
    run_dir = _run(tmp_path)
    path = run_dir / "transcript.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + '{"t": 9.9, "kind": "not', encoding="utf-8")
    code, out = _trace(tmp_path, run_dir.name)
    assert code == 0, out
    assert "1 unreadable line(s) skipped" in out
    assert "SUCCESS" in out


def test_trace_on_a_missing_run_is_one_line(tmp_path: Path) -> None:
    _run(tmp_path)
    code, out = _trace(tmp_path, "no-such-run")
    assert code == 1
    assert "no run matching" in out and "newest:" in out


def test_trace_on_an_empty_runs_dir_says_what_to_do(tmp_path: Path) -> None:
    code, out = _trace(tmp_path)
    assert code == 1 and "quackd run" in out


def test_trace_on_a_flock_run_names_every_member(tmp_path: Path) -> None:
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
            "--no-trace",
        ],
    )
    assert result.exit_code == 0, result.output
    code, out = _trace(tmp_path, "--no-prompt", "--thinking", "0")
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

    code, out = _trace_output(tmp_path, "example-1")
    assert code == 0, out
    assert _counters(out) == counters

    (run_dir,) = [d for d in tmp_path.iterdir() if d.is_dir()]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    flat = " ".join(out.split())
    started = str(summary["started_at"]).replace("T", " ").replace("Z", " UTC")
    assert f"started {started}" in flat, flat[:400]
    assert "run name Example 1" in flat, "the header shows the name as typed, not the slug"
    assert "price $3/M in, $15/M out (--price)" in flat


def test_trace_prefers_the_run_whose_name_ends_in_the_label_over_a_newer_one(
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

    code, out = _trace(tmp_path, "example-1")
    assert code == 0, out
    assert nineteen not in out, "the substring pass answered before the label pass"
    assert one in out


def test_a_summary_written_before_the_clocks_replays_with_the_counters_it_had(
    tmp_path: Path,
) -> None:
    """The compatibility guarantee, written out as the thing it protects: a summary from
    before this release carries steps, llm_calls, usage, outcome and reason and nothing else,
    and `quackd trace` has to replay it with exactly the three counters it printed then.

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
    code, out = _trace_output(tmp_path, run_dir.name)
    assert code == 0, out
    counters = _counters(out)
    assert counters.startswith("steps 3"), counters
    assert "llm calls 5" in counters and "tokens 120+40" in counters, counters
    assert "time" not in counters and "cost" not in counters, counters
    assert "SUCCESS" in out and "quacked and walked one step" in out
