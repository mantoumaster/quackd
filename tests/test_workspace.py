"""Eight packages are released together, so nothing about them may drift apart.

quackd is one repository and eight distributions now: the core and one per robot. That buys
an install with no robot in it, and it costs a set of facts that have to agree and that no
single file owns. A version in eight places, a catalogue in the core that names seven
adapters none of which it imports, an entry point in each of those seven, and a dependency
window in both directions. Every one of those is a thing somebody can half-update.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
import tomllib
from pathlib import Path

import pytest

import quackd
from quackd.adapters.catalogue import BY_NAME, ENTRY_POINT_GROUP, OFFICIAL
from quackd.adapters.factory import adapter_names, info, is_installed

REPO = Path(__file__).resolve().parents[1]
MEMBERS = sorted(p for p in (REPO / "adapters").glob("*/pyproject.toml"))
NAMES = [p.parent.name for p in MEMBERS]


def _toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_every_adapter_directory_is_a_workspace_member() -> None:
    """`uv` resolves a member from the glob in the root, so a package that is not matched by
    it installs from PyPI instead of from the checkout, and a contributor's change to it is
    silently not the one under test."""
    root = _toml(REPO / "pyproject.toml")
    assert root["tool"]["uv"]["workspace"]["members"] == ["adapters/*"]
    sources = root["tool"]["uv"]["sources"]
    for name in NAMES:
        dist = f"quackd-{name.replace('_', '-')}"
        assert sources.get(dist) == {"workspace": True}, dist


@pytest.mark.parametrize("member", MEMBERS, ids=NAMES)
def test_an_adapter_carries_the_same_version_as_the_core(member: Path) -> None:
    """Each package carries its own `__version__`, because an adapter's sdist holds only its
    own source and cannot read the core's. They ship together, so they must not differ.
    `scripts/set_version.py` writes all eight at once for exactly this reason."""
    src = next((member.parent / "src").glob("quackd_*/__init__.py"))
    found = re.search(r'^__version__ = "([^"]+)"$', src.read_text(encoding="utf-8"), re.M)
    assert found, f"{src} has no __version__"
    assert found.group(1) == quackd.__version__


@pytest.mark.parametrize("member", MEMBERS, ids=NAMES)
def test_an_adapter_allows_the_core_it_is_released_with(member: Path) -> None:
    """The window an adapter allows has to admit the core it ships beside, or the release
    resolves to nothing on the day it is published."""
    data = _toml(member)
    pin = next(d for d in data["project"]["dependencies"] if d.startswith("quackd>="))
    low, high = re.findall(r"\d+\.\d+", pin)
    version = tuple(int(n) for n in quackd.__version__.split(".")[:2])
    assert tuple(int(n) for n in low.split(".")) <= version, pin
    assert version < tuple(int(n) for n in high.split(".")), pin


@pytest.mark.parametrize("member", MEMBERS, ids=NAMES)
def test_an_adapter_declares_the_entry_point_that_makes_it_findable(member: Path) -> None:
    """The entry point is the whole mechanism: an adapter that does not declare one is
    installed and invisible, which looks exactly like an adapter that is not installed."""
    name = member.parent.name
    declared = _toml(member)["project"]["entry-points"][ENTRY_POINT_GROUP]
    assert declared == {name: f"quackd_{name}"}, declared


def test_the_core_publishes_an_extra_for_every_robot_it_names() -> None:
    """The catalogue is what `list-adapters` prints on a machine with nothing installed, so
    the extra it names has to be one that exists and that pulls in that adapter's package."""
    extras = _toml(REPO / "pyproject.toml")["project"]["optional-dependencies"]
    for row in OFFICIAL:
        assert row.extra, f"{row.name} names no extra to install it with"
        key = row.extra.removeprefix("quackd[").removesuffix("]")
        assert key in extras, f"{row.extra} is not an extra this package has"
        dist = f"quackd-{row.name.replace('_', '-')}"
        assert any(d.startswith(dist) for d in extras[key]), (key, extras[key])
    assert set(extras["robots"]) >= {
        f"quackd-{r.name.replace('_', '-')}"
        for r in OFFICIAL
        if not any(d.startswith(f"quackd-{r.name.replace('_', '-')}[") for d in extras["robots"])
    }


def test_the_catalogue_and_the_installed_adapter_agree_about_the_robot() -> None:
    """The catalogue describes a robot without importing it, which is the only way to list one
    that is not installed. Nothing checks that description against the adapter itself, so this
    does: a backend added to one and not the other is a `--robot` that parses and cannot run."""
    for name in adapter_names():
        if not is_installed(name):
            continue
        module = importlib.import_module(f"quackd_{name}")
        assert tuple(module.BACKENDS) == info(name).backends, name
        assert name in BY_NAME, f"{name} is installed but quackd does not publish it"


def test_every_published_adapter_is_installed_for_the_suite() -> None:
    """The dev extra carries all seven without their SDKs, so the suite drives every body
    against a mock. One missing would not fail loudly; its tests would quietly skip."""
    installed = {ep.name for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)}
    assert set(BY_NAME) <= installed, set(BY_NAME) - installed
