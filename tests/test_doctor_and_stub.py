"""`quackd doctor`: it runs everywhere, it answers in data, and the WebSocket stub refuses
honestly.

The split between `collect` and `render` is the thing worth testing. The collector must not
know what green means, or `--json` has nothing underneath it to print, and the renderer must
not know anything the report does not hold.
"""

from __future__ import annotations

import io
import json

import pytest
from rich.console import Console
from typer.testing import CliRunner

from quackd import doctor
from quackd.cli import app
from quackd.transport.base import TransportError
from quackd.transport.websocket_stub import WebSocketTransport


def test_doctor_runs() -> None:
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    for needle in (
        "providers",
        "adapters",
        "transports",
        "upstream assumptions",
        "sim2d",
        "jsonrpc",
    ):
        assert needle in result.output


def test_doctor_shows_a_robot_manifest() -> None:
    result = CliRunner().invoke(app, ["doctor", "--robot", "microduck:mock"])
    assert result.exit_code == 0, result.output
    assert "microduck (biped" in result.output and "standing" in result.output
    bad = CliRunner().invoke(app, ["doctor", "--robot", "nope:x"])
    assert bad.exit_code == 1 and "unknown adapter" in bad.output


async def test_websocket_stub_points_at_upstream() -> None:
    t = WebSocketTransport()
    with pytest.raises(TransportError, match=r"architecture\.md"):
        await t.connect()
    await t.stop()  # never raises: a stop must always be safe


# ── the report, and the renderer that is not allowed to know anything it does not ───────


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """doctor probes five local servers at 1.5 s each. Nothing here is about whether one is
    running, and a suite that waits eight seconds to find out is a suite nobody runs."""
    monkeypatch.setattr(doctor, "_probe_models", lambda url, timeout_s=1.5: ("down", "not running"))


def test_collect_answers_in_data_with_no_styling_in_it() -> None:
    """The collector must not know what green means. It did: every cell used to be a markup
    string, which is why there was nothing underneath for --json to print."""
    report = doctor.collect()
    blob = json.dumps(report.to_dict())
    for tag in ("[green]", "[red]", "[yellow]", "[dim]", "[/"):
        assert tag not in blob, tag
    assert report.ok is True
    assert [c.name for c in report.core] == ["pydantic", "mcp", "opencv", "numpy", "Pillow"]
    assert len(report.adapters) == 7
    assert report.bundled_ducks > 0
    assert {p.upstream for p in report.pins} >= {"microduck", "lerobot", "microduck_rl"}
    assert report.assumptions, "there are unverified assumptions and doctor says so"


def test_a_missing_core_package_is_a_failure_and_a_missing_extra_is_not() -> None:
    report = doctor.collect()
    assert report.ok and not report.missing_core
    report.core[0].ok = False
    assert not report.ok and report.missing_core == [report.core[0].name]
    report = doctor.collect()
    report.extras[0].ok = False
    assert report.ok, "an extra nobody installed is a choice, not a fault"


def test_an_unknown_robot_fails_the_report_rather_than_raising() -> None:
    report = doctor.collect("nope:x")
    assert report.robot is not None and report.robot.error is not None
    assert "unknown adapter" in report.robot.error
    assert report.ok is False


def test_the_report_reads_a_robot_manifest() -> None:
    report = doctor.collect("microduck:mock")
    assert report.robot is not None and report.robot.error is None
    names = {v.name for v in report.robot.verbs}
    assert {"move", "stop", "observe"} <= names
    assert report.ok is True


def test_render_says_everything_the_report_holds() -> None:
    buf = io.StringIO()
    doctor.render(Console(file=buf, width=200), doctor.collect("microduck:mock"))
    out = buf.getvalue()
    for needle in (
        "core",
        "providers",
        "local LLM servers",
        "adapters",
        "transports",
        "optional extras",
        "upstream assumptions",
        "SUCCESS",
        "microduck (biped",
    ):
        assert needle in out, needle


def test_render_puts_nothing_on_a_codepage_that_cannot_carry_it() -> None:
    """A Windows pipe is cp1252 and this is a command people paste into an issue."""
    raw = io.BytesIO()
    console = Console(file=io.TextIOWrapper(raw, encoding="cp1252", errors="replace"), width=120)
    doctor.render(console, doctor.collect())
    console.file.flush()
    out = raw.getvalue().decode("cp1252")
    assert "?" not in out.replace("?cmd_vel", ""), "a lost glyph arrives as a question mark"
    assert "[ok] built-in" in out, "the registry's tick becomes something readable"


def test_doctor_json_is_one_document_and_carries_the_exit_code() -> None:
    result = CliRunner().invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["version"] and payload["python"]
    bad = CliRunner().invoke(app, ["doctor", "--robot", "nope:x", "--json"])
    assert bad.exit_code == 1
    assert json.loads(bad.output)["robot"]["error"].startswith("unknown adapter")


def test_the_progress_callback_names_the_slow_questions() -> None:
    """It feeds the spinner, which is the only reason anybody knows doctor is still alive
    while it waits on five local servers."""
    said: list[str] = []
    doctor.collect(progress=said.append)
    assert any("probing ollama" in line for line in said)
    assert any("extras" in line for line in said)
