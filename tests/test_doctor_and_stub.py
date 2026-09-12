"""`quackd doctor` runs without crashing anywhere; the WebSocket stub refuses honestly."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from quackd.agent.providers.factory import CLOUD_NAMES, KEY_ENV, PROVIDER_NAMES
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


def test_doctor_lists_every_provider_and_its_key() -> None:
    """The providers table is built from the same tables `make_provider` dispatches on, so a
    vendor that is wired up but missing a row here would be a vendor nobody could discover."""
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    for name in PROVIDER_NAMES:
        assert name in flat, f"{name} has no row in doctor's providers table"
    for cloud in CLOUD_NAMES:
        assert KEY_ENV[cloud] in flat, f"{cloud} does not say which key it wants"


def test_doctor_says_when_a_pinned_model_belongs_to_another_vendor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`QUACKD_MODEL` pins one id for every provider, so on all but one of them it is wrong.
    Doctor is where that should be found, and it must still exit 0: this is a report, not a run."""
    monkeypatch.setenv("QUACKD_MODEL", "claude-opus-5")
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert "does not list" in flat


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
