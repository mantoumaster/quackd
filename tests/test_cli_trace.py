"""`quackd trace` replays a finished run. What it prints has to be what the run printed."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from quackd.cli import app

runner = CliRunner()


def _run(tmp_path: Path, *flags: str) -> Path:
    """One real run under `tmp_path`, and the directory it wrote."""
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
    )
    assert result.exit_code == 0, result.output
    (run_dir,) = [d for d in tmp_path.iterdir() if d.is_dir()]
    return run_dir


def _trace(tmp_path: Path, *args: str) -> tuple[int, str]:
    result = runner.invoke(app, ["trace", *args, "--runs-dir", str(tmp_path)])
    return result.exit_code, " ".join(result.output.split())


def test_trace_replays_a_finished_run(tmp_path: Path) -> None:
    """The point of the command: the run is over, and you want to read what happened."""
    run_dir = _run(tmp_path)
    code, out = _trace(tmp_path, run_dir.name)
    assert code == 0, out
    assert "system prompt" in out
    assert "send move" in out and "result walk_to ok" in out
    assert "[scripted]" in out  # the model's reasoning, as the run showed it
    assert "SUCCESS" in out and "llm calls" in out


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
