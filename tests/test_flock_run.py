"""Whole flock runs: the seed sweep, the one-claimant invariant, and the failure paths."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from quackd.agent.providers.fake import FakeProvider
from quackd.cli import app
from quackd.duckfile.parser import load_duck
from quackd.flock.runner import FLOCK_TRACE, run_flock
from quackd.trace import TraceEvent

runner = CliRunner()
DUCK = load_duck("flock-kick")
# CI sets QUACKD_STRICT_SEEDS=1 (see tests/test_acceptance_sim2d.py)
MIN_SUCCESSES = 10 if os.environ.get("QUACKD_STRICT_SEEDS") == "1" else 8


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def assert_one_claimant(events: list[dict[str, Any]]) -> None:
    """Replay flock.jsonl: at most one live claim, and only the claimant approaches."""
    claimant: str | None = None
    for ev in events:
        if ev["kind"] == "auction_decision":
            assert claimant is None, "a claim was granted while another was live"
            claimant = ev["kicker"]
        elif ev["kind"] == "miss":
            if ev["duck"] == claimant:
                claimant = None
        elif ev["kind"] == "verb" and ev.get("canonical", ev["name"]) in ("go_to", "kick"):
            assert ev["duck"] == claimant, f"{ev['duck']} moved on the ball without the claim"


async def test_flock_acceptance_seeds(tmp_path: Path) -> None:
    successes = 0
    report = []
    for seed in range(10):
        t0 = time.perf_counter()
        result = await asyncio.wait_for(
            run_flock(
                DUCK, provider=FakeProvider.for_duck("flock-kick"), seed=seed, runs_dir=tmp_path
            ),
            timeout=120,  # a clock wedge must FAIL the test, not hang CI
        )
        wall = time.perf_counter() - t0
        ok = result.outcome == "success" and result.ball_displacement_m >= 0.3
        successes += ok
        report.append(f"seed {seed}: {result.outcome} truth={result.ball_displacement_m:.2f}")
        assert wall < 60, report[-1]
        events = read_jsonl(result.run_dir / "flock.jsonl")
        assert_one_claimant(events)
        kinds = {e["kind"] for e in events}
        assert {"plan", "bus", "flock_start", "flock_end"} <= kinds
        summary = json.loads((result.run_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["planner"]["llm_calls"] == 0  # fake provider: zero model calls
        assert summary["kicker"] in summary["flock"]["members"] or summary["kicker"] is None
        for name in summary["flock"]["members"]:
            assert (result.run_dir / "ducks" / name / "transcript.jsonl").exists()
            assert not (result.run_dir / "ducks" / name / "summary.json").exists()
    assert successes >= MIN_SUCCESSES, "\n".join(report)


async def test_flock_dry_run_moves_nothing(tmp_path: Path) -> None:
    result = await asyncio.wait_for(
        run_flock(
            DUCK,
            provider=FakeProvider.for_duck("flock-kick"),
            seed=1,
            runs_dir=tmp_path,
            dry_run=True,
        ),
        timeout=120,
    )
    assert result.outcome != "success"
    assert result.ball_displacement_m == 0.0
    assert all(d["kicks_connected"] == 0 for d in result.per_duck.values())


async def test_a_flock_is_traced_per_member_and_every_view_sees_only_its_own_duck(
    tmp_path: Path,
) -> None:
    seen: dict[str, list[TraceEvent]] = {}

    result = await asyncio.wait_for(
        run_flock(
            DUCK,
            provider=FakeProvider.for_duck("flock-kick"),
            seed=3,
            runs_dir=tmp_path,
            trace=lambda name: seen.setdefault(name, []).append,
        ),
        timeout=120,
    )
    assert result.trace_dropped == 0  # nothing raised, so nothing was lost
    assert set(seen) == set(result.per_duck) | {FLOCK_TRACE}

    for name in result.per_duck:
        records = read_jsonl(result.run_dir / "ducks" / name / "transcript.jsonl")
        kinds = {r["kind"] for r in records}
        assert {"verb_start", "intent", "verb_end", "member_end"} <= kinds, f"{name}: {kinds}"
        # the view saw this duck's verbs, all of them and only them: a factory that handed
        # two members one sink would show each of these twice
        mine = [(r["name"], r.get("params")) for r in records if r["kind"] == "verb_start"]
        theirs = [
            (e.data["name"], e.data.get("params")) for e in seen[name] if e.kind == "verb_start"
        ]
        assert theirs == mine

    story = {e.kind for e in seen[FLOCK_TRACE]}
    assert "claim" in story and "verb_start" not in story  # decisions here, verbs per duck

    flock_kinds = {e["kind"] for e in read_jsonl(result.run_dir / "flock.jsonl")}
    assert "member_log" in flock_kinds  # the members' log lines still land here
    assert "verb_start" not in flock_kinds  # and their narration still does not


async def test_flock_kill_switch_stops_every_duck(tmp_path: Path) -> None:
    # fire the kill switch on the FIRST auction event: deterministic in sim time, mid-run
    def on_ready(_transport: Any, coordinator: Any) -> None:
        def on_event(kind: str, _data: dict[str, Any]) -> None:
            if kind == "auction":
                coordinator.abort.set()

        coordinator.on_event = on_event

    result = await asyncio.wait_for(
        run_flock(
            DUCK,
            provider=FakeProvider.for_duck("flock-kick"),
            seed=4,
            runs_dir=tmp_path,
            on_recorder=on_ready,
        ),
        timeout=120,
    )
    assert result.outcome == "aborted"
    assert "kill switch" in result.reason
    # every duck actually stopped, and nobody reached the ball after the abort
    assert all(d["final_status"] in ("aborted", "stopped") for d in result.per_duck.values())
    assert result.ball_displacement_m < 0.3


async def test_flock_survives_a_dead_member(tmp_path: Path) -> None:
    # duck-2 goes SILENT from the start: aborted and muted, so no HB and no RESULT ever
    # reach the bus. Only the sim-time watchdog can notice it. Deterministic, no races.
    def on_ready(_transport: Any, coordinator: Any) -> None:
        victim = coordinator.members["duck-2"]
        victim._publish = lambda _msg: None
        victim.executor.abort.set()

    result = await asyncio.wait_for(
        run_flock(
            DUCK,
            provider=FakeProvider.for_duck("flock-kick"),
            seed=0,
            runs_dir=tmp_path,
            on_recorder=on_ready,
        ),
        timeout=120,  # the run must END (no clock wedge)
    )
    assert result.outcome in ("success", "failure")
    assert result.per_duck["duck-2"]["final_status"] == "aborted"
    events = read_jsonl(result.run_dir / "flock.jsonl")
    assert any(e["kind"] == "member_dead" and e["duck"] == "duck-2" for e in events), (
        "the watchdog never declared the silent duck dead"
    )
    assert any(e["kind"] == "verb" and e["duck"] != "duck-2" for e in events), (
        "the survivors did nothing"
    )


async def test_flock_enforces_max_minutes(tmp_path: Path) -> None:
    # regression: budget.start() was never called, so max_minutes was silently dead
    from quackd.duckfile.parser import parse_duck_text

    duck = parse_duck_text(
        "---\nduck: 0\nname: flock-blink\ndescription: d\n"
        "verbs:\n  allow: [search_scan, walk_to, walk, kick, quack, get_frame, stop]\n"
        "budgets:\n  max_steps: 60\n  max_minutes: 0.02\n"
        "success: [x]\nflock:\n  members: 2\n---\n# T\nx\n",
        "flock-blink.duck",
    )
    result = await asyncio.wait_for(
        run_flock(duck, provider=FakeProvider.for_duck("flock-kick"), seed=1, runs_dir=tmp_path),
        timeout=120,
    )
    assert result.outcome == "failure"
    assert all(d["final_status"] == "budget" for d in result.per_duck.values())


async def test_flock_honours_max_steps_override(tmp_path: Path) -> None:
    # regression: run --flock silently dropped --max-steps
    result = await asyncio.wait_for(
        run_flock(
            DUCK,
            provider=FakeProvider.for_duck("flock-kick"),
            seed=1,
            runs_dir=tmp_path,
            max_steps=1,
        ),
        timeout=120,
    )
    assert result.outcome == "failure"
    assert all(d["final_status"] == "budget" for d in result.per_duck.values())
    assert all(d["steps"] <= 2 for d in result.per_duck.values())


def test_cli_flock_run_and_guards(tmp_path: Path) -> None:
    ok = runner.invoke(
        app,
        ["run", "flock-kick", "--provider", "fake", "--seed", "3", "--runs-dir", str(tmp_path)],
    )
    assert ok.exit_code == 0, ok.output
    assert "flock" in ok.output and "kicker " in ok.output
    run_dir = next(tmp_path.iterdir())
    assert (run_dir / "flock.jsonl").exists() and (run_dir / "run.gif").exists()

    one = runner.invoke(app, ["run", "hello-world", "--flock", "1", "--runs-dir", str(tmp_path)])
    assert one.exit_code == 1 and "2 to 4" in one.output

    wrong_backend = runner.invoke(
        app,
        ["run", "flock-kick", "--robot", "microduck:mock", "--runs-dir", str(tmp_path)],
    )
    assert wrong_backend.exit_code == 1 and "simulator only" in wrong_backend.output


def test_cli_flock_flag_on_a_solo_duck(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "find-and-kick",
            "--flock",
            "2",
            "--provider",
            "fake",
            "--seed",
            "1",
            "--no-gif",
            "--runs-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads((next(tmp_path.iterdir()) / "summary.json").read_text(encoding="utf-8"))
    assert summary["flock"]["members"] == ["duck-0", "duck-1"]


def test_validate_rejects_flock_with_confirm(tmp_path: Path) -> None:
    bad = tmp_path / "bad-flock.duck"
    bad.write_text(
        "---\nduck: 0\nname: bad-flock\ndescription: d\nverbs:\n  allow: [kick, stop]\n"
        "  confirm: [kick]\nsuccess: [x]\nflock:\n  members: 2\n---\n# T\nx\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["validate", str(bad)])
    assert result.exit_code == 1 and "y/N" in result.output


def test_serve_mcp_refuses_flock_ducks() -> None:
    from quackd.mcp_server import serve

    try:
        serve(duckfile="flock-kick")
        raise AssertionError("expected SystemExit")
    except SystemExit as e:
        assert "flock" in str(e)


def test_cli_flock_trace_prefixes_every_line_with_its_duck_and_tells_the_flocks_story(
    tmp_path: Path,
) -> None:
    """Three robots moving at once used to print nothing at all. Interleaved without a name
    on each line they would be worse than nothing, so the prefix is the whole feature."""
    args = [
        "run",
        "flock-kick",
        "--provider",
        "fake",
        "--seed",
        "3",
        "--no-gif",
        "--runs-dir",
        str(tmp_path),
    ]
    on = runner.invoke(app, [*args, "--trace"])
    assert on.exit_code == 0, on.output
    shown = " ".join(on.output.split())
    for needle in (
        "duck-0 verb search_scan(",  # a member's own verb, under its own name
        "-> move x",  # what went to that robot
        "flock auction first bid duck-",  # the coordinator's story
        "flock claim duck-",
        "end stopped after",  # and each member's ending
    ):
        assert needle in shown, (needle, shown[:400])

    off = runner.invoke(app, [*args, "--no-trace"])
    assert off.exit_code == 0, off.output
    quiet = " ".join(off.output.split())
    for needle in ("duck-0 verb", "-> move x", "flock auction", "end stopped after"):
        assert needle not in quiet, needle
    assert "kicker " in quiet, "the outcome is not part of the trace and must survive"
