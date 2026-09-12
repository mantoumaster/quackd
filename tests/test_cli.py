"""The CLI wires things together; these tests prove the wiring, not the parts."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from quackd.cli import app

from .conftest import DUCKS

runner = CliRunner()


# ── the trace ───────────────────────────────────────────────────────────────────────────


def _traced(tmp_path: Path, monkeypatch, *args: str, env: str | None = "") -> str:
    """Any command whose stderr the CliRunner folds into `output`, with the label column's
    padding squeezed out so an assertion can name a line as a reader would say it. `env` is
    what QUACKD_TRACE says: "" is on (an empty value must never read as off), None removes
    it — the suite turns the trace off for everyone else (conftest)."""
    if env is None:
        monkeypatch.delenv("QUACKD_TRACE", raising=False)
    else:
        monkeypatch.setenv("QUACKD_TRACE", env)
    result = runner.invoke(app, [*args, "--runs-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    return " ".join(result.output.split())


def _trace_run(tmp_path: Path, monkeypatch, *flags: str, env: str | None = "") -> str:
    """One `run` on the mock, which sends no frames and writes no GIF: the trace itself is
    what these tests read."""
    return _traced(
        tmp_path,
        monkeypatch,
        "run",
        "hello-world",
        "--provider",
        "fake",
        "--robot",
        "microduck:mock",
        "--no-gif",
        *flags,
        env=env,
    )


def _trace_record(tmp_path: Path, monkeypatch, *flags: str, env: str | None = "") -> str:
    """The same run through `record`, which pins the simulator and always writes a GIF."""
    return _traced(
        tmp_path,
        monkeypatch,
        "record",
        "hello-world",
        "--provider",
        "fake",
        *flags,
        env=env,
    )


def _kinds(tmp_path: Path) -> list[str]:
    """Every `kind` the run's transcript recorded, in order."""
    from quackd.agent.transcript import Transcript

    return [e["kind"] for e in Transcript.read(next(tmp_path.rglob("transcript.jsonl")))]


def test_the_trace_is_on_by_default(tmp_path: Path, monkeypatch) -> None:
    """Someone who types `quackd run` and watches a robot move should see why it moved."""
    out = _trace_run(tmp_path, monkeypatch)
    assert "system prompt" in out  # what the model was told
    assert "quack(text='hello!')" in out  # what it chose
    assert "-> sound" in out  # what went to the robot
    assert "<- quack ok" in out  # what came back
    assert "SUCCESS" in out  # and the outcome still reaches stdout


def test_no_trace_prompt_hides_only_the_prompt(tmp_path: Path, monkeypatch) -> None:
    """The prompt is forty to seventy lines, worth reading once and tiresome on the fiftieth
    run of an afternoon. Hiding it must not cost the verbs and the intents."""
    out = _trace_run(tmp_path, monkeypatch, "--no-trace-prompt")
    assert "system prompt" not in out and "You are the brain" not in out
    assert "-> sound" in out and "<- quack ok" in out


def test_the_env_hides_the_prompt_and_the_flag_wins(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QUACKD_TRACE_PROMPT", "0")
    assert "system prompt" not in _trace_run(tmp_path, monkeypatch)
    assert "system prompt" in _trace_run(tmp_path, monkeypatch, "--trace-prompt")


def test_no_trace_leaves_the_header_and_the_outcome(tmp_path: Path, monkeypatch) -> None:
    out = _trace_run(tmp_path, monkeypatch, "--no-trace")
    assert "-> sound" not in out and "system prompt" not in out
    assert "SUCCESS" in out and "hello-world" in out


def test_the_env_can_turn_the_trace_off_too(tmp_path: Path, monkeypatch) -> None:
    """A `.env` line has to work, so the default is read when the command runs, not when the
    module is imported."""
    assert "-> sound" not in _trace_run(tmp_path, monkeypatch, env="0")
    assert "-> sound" in _trace_run(tmp_path, monkeypatch, env=None)


def test_the_flag_beats_the_env(tmp_path: Path, monkeypatch) -> None:
    assert "-> sound" in _trace_run(tmp_path, monkeypatch, "--trace", env="0")


def test_verbose_is_the_compact_view_and_does_not_double_the_trace(
    tmp_path: Path, monkeypatch
) -> None:
    """With the trace on, the executor's own log lines would say every verb a second time."""
    traced = _trace_run(tmp_path, monkeypatch, "--verbose")
    assert "-> sound" in traced
    assert "→ quack" not in traced, "the old compact line must not double the trace"
    compact = _trace_run(tmp_path, monkeypatch, "--verbose", "--no-trace")
    assert "→ quack" in compact and "-> sound" not in compact


def test_record_writes_a_gif_and_a_transcript(tmp_path: Path, monkeypatch) -> None:
    """Every GIF in the README and every launch post comes out of `record`, and no test
    ever ran the command: it pins its own robot, always renders, and could have been broken
    for a whole release without a single failure to say so."""
    out = _trace_record(tmp_path, monkeypatch)
    assert "SUCCESS" in out
    gif = next(tmp_path.rglob("run.gif"))
    assert gif.stat().st_size > 0, "an empty GIF is a README with a broken image in it"
    assert {"run_start", "llm", "verb_end", "run_end"} <= set(_kinds(tmp_path))


def test_record_no_trace_still_writes_every_event(tmp_path: Path, monkeypatch) -> None:
    """The switch is about the console and nothing else (ADR-0029). The transcript is the
    record, and a run recorded quietly must be as complete as a noisy one."""
    out = _trace_record(tmp_path, monkeypatch, "--no-trace")
    assert "-> sound" not in out and "SUCCESS" in out
    kinds = _kinds(tmp_path)
    assert "intent" in kinds and "verb_end" in kinds


def test_no_trace_verbose_prints_the_dry_run_line_intact(tmp_path: Path, monkeypatch) -> None:
    """`--verbose` predates the trace and people's scripts still pass it, so with the trace
    off it is still the only thing that says what a dry run would have done — and the line
    opens with `[dry-run]`, which Rich would eat as a style tag if it were printed as
    markup."""
    out = _trace_run(tmp_path, monkeypatch, "--no-trace", "--dry-run", "--verbose")
    assert "[dry-run] would run quack(" in out
    assert "Traceback" not in out


def test_serve_mcp_forwards_no_trace(monkeypatch) -> None:
    """`--no-trace` is what silences an MCP server that logs to the same stderr its client
    reads. The flag is parsed by one module and honoured by another, so nothing but a call
    recorder proves it survives the hand-off."""
    from quackd import mcp_server
    from quackd.trace import trace_enabled_default

    captured: dict[str, object] = {}

    def recorder(**kwargs: object) -> None:  # `serve` blocks on mcp.run(); this returns
        captured.update(kwargs)

    monkeypatch.setattr(mcp_server, "serve", recorder)
    monkeypatch.setenv("QUACKD_TRACE", "")  # an empty value is on

    off = runner.invoke(app, ["serve-mcp", "--no-trace", "--robot", "microduck:mock"])
    assert off.exit_code == 0, off.output
    assert captured["trace"] is False

    captured.clear()
    on = runner.invoke(app, ["serve-mcp", "--robot", "microduck:mock"])
    assert on.exit_code == 0, on.output
    # without the flag the CLI forwards None — "ask the environment" — because a server a
    # desktop spawned has no shell to read QUACKD_TRACE in. `serve` resolves it, and on.
    assert captured["trace"] is None and trace_enabled_default() is True


def test_the_outcome_line_prints_the_models_reason_verbatim(tmp_path: Path, monkeypatch) -> None:
    """The reason is the model's own `declare_failure` text. A local model that leaks
    `[/think]` into it used to crash the CLI with a Rich MarkupError after a completed run,
    replacing the verdict with a traceback."""
    from quackd.agent.providers import factory
    from quackd.agent.providers.base import ToolCall
    from quackd.agent.providers.fake import FakeProvider

    reason = "the ball is [behind] the sofa [/think]"
    monkeypatch.setattr(
        factory,
        "make_provider",
        lambda *a, **k: FakeProvider(
            script=[ToolCall(name="declare_failure", arguments={"reason": reason})]
        ),
    )
    result = runner.invoke(
        app,
        [
            "run",
            "hello-world",
            "--provider",
            "fake",
            "--robot",
            "microduck:mock",
            "--runs-dir",
            str(tmp_path),
            "--no-gif",
        ],
    )
    # not the exit code: it is 1 both for the failure outcome and for the old traceback
    assert "[behind]" in result.output and "[/think]" in result.output
    assert "Traceback" not in result.output


def test_list_verbs_prints_a_description_with_brackets_verbatim(monkeypatch) -> None:
    """A verb description is manifest text, which Rich would silently eat as a style tag."""
    from quackd.verbs import registry as registry_mod
    from quackd.verbs.registry import NoParams, Verb, VerbRegistry

    reg = VerbRegistry()
    reg.register(Verb("kick", "kick [left] or [right]", lambda c, p: None, params=NoParams))  # type: ignore[arg-type]
    monkeypatch.setattr(registry_mod, "default_registry", lambda: reg)
    result = runner.invoke(app, ["list-verbs"])
    assert result.exit_code == 0, result.output
    assert "[left]" in result.output


def test_a_verbose_line_survives_a_bracket_a_planner_logged(monkeypatch) -> None:
    """The flock's planner logs a model's raw tool arguments through this line."""
    import io

    from rich.console import Console

    from quackd import cli as cli_mod

    buf = io.StringIO()
    monkeypatch.setattr(cli_mod, "err_console", Console(file=buf, force_terminal=False, width=200))
    cli_mod._verbose_line("planner: [/think] chose [bold]walk")
    out = buf.getvalue()
    assert "[/think]" in out and "[bold]walk" in out


def test_validate_starter_ducks() -> None:
    result = runner.invoke(app, ["validate", *[str(p) for p in sorted(DUCKS.glob("*.duck"))]])
    assert result.exit_code == 0, result.output
    assert "12 file(s) valid" in result.output


def test_validate_expands_globs_itself() -> None:
    result = runner.invoke(app, ["validate", str(DUCKS / "*.duck")])
    assert result.exit_code == 0, result.output


def test_validate_fails_fast(tmp_path: Path) -> None:
    bad = tmp_path / "bad.duck"
    bad.write_text("---\nduck: 0\nname: bad\n---\nbody\n", encoding="utf-8")
    unknown = tmp_path / "unknown.duck"
    unknown.write_text(
        "---\nduck: 0\nname: unknown\ndescription: d\nverbs:\n  allow: [fly]\n"
        "success: [x]\n---\n# Task\nx\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app, ["validate", str(bad), str(unknown), str(DUCKS / "hello-world.duck")]
    )
    assert result.exit_code == 1
    assert "unknown verbs: fly" in result.output
    assert "✗" in result.output and "✓" in result.output


def test_list_verbs() -> None:
    result = runner.invoke(app, ["list-verbs"])
    assert result.exit_code == 0
    for name in ("walk", "kick", "walk_to", "quack"):
        assert name in result.output


def test_run_hello_world_on_mock(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "hello-world",
            "--provider",
            "fake",
            "--robot",
            "microduck:mock",
            "--runs-dir",
            str(tmp_path),
            "--no-gif",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "SUCCESS" in result.output
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1 and (run_dirs[0] / "transcript.jsonl").exists()


def test_missing_extra_hint_survives_rich_markup(tmp_path: Path, monkeypatch) -> None:
    from quackd.agent.providers import factory
    from quackd.agent.providers.base import ProviderNotInstalled

    def missing(name: str, **_: object) -> None:
        raise ProviderNotInstalled(name, "anthropic")

    monkeypatch.setattr(factory, "make_provider", missing)
    result = runner.invoke(
        app,
        [
            "run",
            "hello-world",
            "--provider",
            "anthropic",
            "--robot",
            "microduck:mock",
            "--runs-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "quackd[anthropic]" in result.output  # Rich must not eat the [anthropic] "tag"


def test_run_goal_builds_an_ad_hoc_duck(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--goal",
            "say hello and stop",
            "--provider",
            "fake",
            "--robot",
            "microduck:mock",
            "--runs-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "goal" in result.output
    transcript = next(tmp_path.rglob("transcript.jsonl")).read_text(encoding="utf-8")
    assert "say hello and stop" in transcript  # the goal is the task body
    assert '"kick"' in transcript  # safe verbs are allowed
    assert "SUCCESS" in result.output


def test_goal_picks_a_matching_scripted_strategy(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--goal",
            "find the ball and kick it",
            "--provider",
            "fake",
            "--seed",
            "4",
            "--runs-dir",
            str(tmp_path),
            "--no-gif",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "scripted:goal:find-and-kick" in result.output
    transcript = next(tmp_path.rglob("transcript.jsonl")).read_text(encoding="utf-8")
    assert '"name": "kick"' in transcript  # it really kicked, not just "nothing more to do"


def test_run_needs_exactly_one_of_duck_or_goal(tmp_path: Path) -> None:
    both = runner.invoke(
        app,
        [
            "run",
            "hello-world",
            "--goal",
            "x",
            "--robot",
            "microduck:mock",
            "--runs-dir",
            str(tmp_path),
        ],
    )
    neither = runner.invoke(app, ["run", "--robot", "microduck:mock", "--runs-dir", str(tmp_path)])
    assert both.exit_code == 1 and neither.exit_code == 1
    assert "either" in both.output and "either" in neither.output


def _run_hello(tmp_path: Path, *flags: str) -> object:
    return runner.invoke(
        app,
        [
            "run",
            "hello-world",
            "--provider",
            "fake",
            "--runs-dir",
            str(tmp_path),
            "--no-gif",
            *flags,
        ],
    )


def test_transport_flag_is_gone(tmp_path: Path) -> None:
    """0.4 deprecated `--transport X` in favour of `--robot microduck:X` and said it would
    be removed in 0.5. It is."""
    old = _run_hello(tmp_path, "--transport", "mock")
    assert old.exit_code != 0  # type: ignore[attr-defined]
    assert "No such option" in old.output  # type: ignore[attr-defined]
    new = _run_hello(tmp_path, "--robot", "microduck:mock")
    assert new.exit_code == 0, new.output  # type: ignore[attr-defined]
    assert "robot=microduck:mock" in new.output  # type: ignore[attr-defined]


def test_robot_flag_errors_are_clean(tmp_path: Path) -> None:
    unknown = _run_hello(tmp_path, "--robot", "hal9000:mock")
    assert unknown.exit_code == 1 and "unknown adapter" in unknown.output  # type: ignore[attr-defined]
    bad_backend = _run_hello(tmp_path, "--robot", "microduck:hovercraft")
    assert bad_backend.exit_code == 1 and "unknown backend" in bad_backend.output  # type: ignore[attr-defined]


def test_list_adapters() -> None:
    result = runner.invoke(app, ["list-adapters"])
    assert result.exit_code == 0, result.output
    for needle in ("microduck", "sim2d", "mock", "jsonrpc", "open_duck", "bridge"):
        assert needle in result.output


def test_list_verbs_for_a_robot() -> None:
    result = runner.invoke(app, ["list-verbs", "--robot", "microduck:mock"])
    assert result.exit_code == 0 and "move" in result.output and "walk" in result.output
    bad = runner.invoke(app, ["list-verbs", "--robot", "nope"])
    assert bad.exit_code == 1 and "unknown adapter" in bad.output


def test_validate_against_a_robot(tmp_path: Path) -> None:
    ok = runner.invoke(app, ["validate", "hello-world", "--robot", "microduck:mock"])
    assert ok.exit_code == 0 and "for microduck" in ok.output
    duck = tmp_path / "needs-express.duck"
    duck.write_text(
        "---\nduck: 1\nname: needs-express\ndescription: d\nrequires: [express]\n"
        "robots: microduck:mock\nverbs:\n  allow: [quack, express, stop]\nsuccess: [x]\n"
        "---\n# Task\nx\n",
        encoding="utf-8",
    )
    bad = runner.invoke(app, ["validate", str(duck)])  # the duck's own robots: default applies
    assert bad.exit_code == 1
    # printed as a plain line under the table, so it survives any terminal width
    assert "requires express, but microduck (microduck) does not provide it" in bad.output
    with_robot = runner.invoke(app, ["validate", str(duck), "--robot", "bogus:x"])
    assert with_robot.exit_code == 1 and "unknown adapter" in with_robot.output


def test_run_unknown_provider_is_a_clean_error(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "hello-world",
            "--provider",
            "hal9000",
            "--robot",
            "microduck:mock",
            "--runs-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "unknown provider" in result.output


def test_run_refuses_a_duck_the_robot_cannot_do(tmp_path: Path) -> None:
    """`serve-mcp` always validated the contract against the robot; `run` never did, and
    died halfway in with a raw VerbNotFound once the robot was already connected."""
    result = runner.invoke(
        app,
        [
            "run",
            "find-and-kick",
            "--provider",
            "fake",
            "--robot",
            "open_duck:mock",
            "--runs-dir",
            str(tmp_path),
            "--no-gif",
        ],
    )
    assert result.exit_code == 1
    flat = " ".join(result.output.split())  # rich wraps the line
    assert "find-and-kick cannot run on open_duck:mock" in flat
    assert "requires kick, but open-duck-01 (open-duck-mini-v2) does not provide it" in flat
    assert "Traceback" not in result.output
    assert list(tmp_path.iterdir()) == []  # refused before a run directory was made


def test_run_still_starts_when_the_duck_fits(tmp_path: Path) -> None:
    """The guard must refuse the impossible without over-refusing the possible."""
    result = runner.invoke(
        app,
        [
            "run",
            "hello-world",
            "--provider",
            "fake",
            "--robot",
            "microduck:mock",
            "--runs-dir",
            str(tmp_path),
            "--no-gif",
        ],
    )
    assert result.exit_code == 0, result.output


def test_a_camera_robot_that_is_not_the_simulator_still_gets_a_detector(tmp_path: Path) -> None:
    """The detector used to be attached only for sim2d, so every hardware backend with a
    camera ran blind: it fetched frames, detected nothing because nothing was detecting,
    and reported that it could not see the ball."""
    result = runner.invoke(
        app,
        [
            "run",
            "open-duck-scout",
            "--provider",
            "fake",
            "--robot",
            "open_duck:mock",
            "--runs-dir",
            str(tmp_path),
            "--no-gif",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "SUCCESS" in result.output


def test_a_console_that_raises_is_reported_once_at_the_end(tmp_path: Path, monkeypatch) -> None:
    """An observer that raises never ends a run, which is right. It also meant a console that
    raised on every event produced a silent trace and no sign at all that it had."""
    import quackd.trace as trace_module

    class Broken(trace_module.ConsoleTrace):  # type: ignore[misc]
        def __call__(self, event: object) -> None:
            raise RuntimeError("the terminal went away")

    monkeypatch.setattr(trace_module, "ConsoleTrace", Broken)
    out = _trace_run(tmp_path, monkeypatch)
    assert "could not be shown" in out
    assert "transcript.jsonl has them" in out
    assert "SUCCESS" in out, "a broken console must not change the outcome"


def test_a_gif_pane_larger_than_the_offscreen_buffer_is_refused_before_anything_runs() -> None:
    """The physics model compiles a 1024 px offscreen buffer, so a larger pane fails inside
    MuJoCo halfway through a run. Typer refuses it at the boundary instead, and the number is
    spelled in `cli.py` because `cli.py` must not import `sim3d`. `tests/test_sim3d.py` pins
    the two to each other."""
    result = runner.invoke(app, ["run", "hello-world", "--gif-size", "4096"])
    assert result.exit_code == 2
    assert "1024" in result.output
