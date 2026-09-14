"""Shared fixtures. Nothing here touches the network or needs an API key."""

from __future__ import annotations

import faulthandler
import sys
from pathlib import Path

import pytest

from quackd.duckfile.parser import load_duck
from quackd.duckfile.schema import DuckFile
from quackd.transport.mock import MockTransport
from quackd.verbs.registry import VerbRegistry, default_registry

REPO = Path(__file__).resolve().parents[1]
DUCKS = REPO / "ducks"

EXIT_GRACE_S = 120


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    """If the interpreter has not exited two minutes after pytest is done, dump every thread
    and force the exit. `faulthandler_timeout` watches a test; nothing watches the shutdown
    after the last one, and that is where a `zmq.Context` left unclosed by a failing test
    was garbage collected into a `term()` that waits forever, which held three macOS jobs
    for six hours with no trace of what they were doing. This names the frame.

    Unconfigure rather than sessionfinish, and a flush first: the failure report is printed
    inside sessionfinish, and `_exit` flushes nothing, so arming the timer any earlier
    turned the one line that mattered into a lost buffer."""
    sys.stdout.flush()
    sys.stderr.flush()
    faulthandler.dump_traceback_later(EXIT_GRACE_S, exit=True)


@pytest.fixture(autouse=True)
def _memory_in_tmp(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A test run must never write the developer's real `~/.quackd/memory`: every test that
    runs the CLI with memory on (the default) gets a throwaway directory instead. Not inside
    `tmp_path`: tests count the run directories they make there."""
    monkeypatch.setenv("QUACKD_MEMORY_DIR", str(tmp_path_factory.mktemp("quackd-memory")))


@pytest.fixture(autouse=True)
def _registry_in_tmp(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same rule as memory, for the same reason: a test that registers a robot must never
    write the developer's real `~/.quackd/robots.json`, and a developer who has one must never
    be running a different suite from CI because a flock they made is lying there."""
    monkeypatch.setenv("QUACKD_REGISTRY_DIR", str(tmp_path_factory.mktemp("quackd-registry")))


@pytest.fixture(autouse=True)
def _asset_cache_in_tmp(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The physics backend's downloaded model lives in `~/.quackd/cache`, and a developer who
    has one was running a different suite from CI: `ensure_microduck(offline=True)` found it
    and the tests that skip everywhere else ran here. Everything gets a throwaway cache and no
    checkout override, so a skip is a skip on both machines.

    Except the `real_duck` tests, whose whole purpose is the developer's real cache. They
    still never fetch — an empty one skips them — so this decides which machine they run on,
    not whether they download."""
    if request.node.get_closest_marker("real_duck") is not None:
        return
    monkeypatch.setenv("QUACKD_CACHE_DIR", str(tmp_path_factory.mktemp("quackd-cache")))
    monkeypatch.delenv("QUACKD_MICRODUCK_ASSETS", raising=False)


@pytest.fixture(autouse=True)
def _trace_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """The trace is on by default and goes to stderr, which CliRunner folds into `output`,
    so every CLI and acceptance test would carry pages of it in its failure message and its
    substring assertions would match by accident. Off for the suite; the tests that prove
    the default is on set `QUACKD_TRACE` to an empty string themselves (an empty value is
    on, and unlike `delenv` it also shields them from a developer's own `.env`)."""
    monkeypatch.setenv("QUACKD_TRACE", "0")


@pytest.fixture(autouse=True)
def _no_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """`QUACKD_MODEL` pins the model for every provider, and the CLI loads a developer's `.env`
    in its root callback, so one line in an untracked file could make half this suite assert
    against a model nobody chose. Empty reads as unset everywhere it is consumed, and unlike
    `delenv` it survives `load_dotenv`, which does not overwrite a name already in the
    environment. The tests that exercise the variable set it themselves."""
    monkeypatch.setenv("QUACKD_MODEL", "")


@pytest.fixture(autouse=True)
def _no_extra_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """`QUACKD_EXTRA_BODY` adds fields to every request an OpenAI-compatible provider sends, so
    one line in a developer's `.env` would reach every test in this suite that reads a request
    body, and the tests asserting a field is *absent* would fail on their machine and nowhere
    else. Empty reads as unset, and unlike `delenv` it survives `load_dotenv`. The tests that
    exercise the variable set it themselves."""
    monkeypatch.setenv("QUACKD_EXTRA_BODY", "")


@pytest.fixture
def registry() -> VerbRegistry:
    return default_registry()


@pytest.fixture
def mock_transport() -> MockTransport:
    return MockTransport()


@pytest.fixture
def hello_duck() -> DuckFile:
    return load_duck(str(DUCKS / "hello-world.duck"))


@pytest.fixture
def kick_duck() -> DuckFile:
    return load_duck(str(DUCKS / "find-and-kick.duck"))


@pytest.fixture(autouse=True)
def _colour_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--no-color` works by setting NO_COLOR, because Typer builds a console of its own for
    every `--help` it renders and reads the variable when it does. Setting a variable is a
    side effect on the process, so the next test must not inherit it, and a developer's own
    FORCE_COLOR must not reach the suite either."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
