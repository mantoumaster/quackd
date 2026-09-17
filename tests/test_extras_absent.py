"""The default install never imports an SDK: every adapter lists, describes and constructs
with torch, lerobot, roslibpy, zmq, zeroconf and paho all absent."""

from __future__ import annotations

import subprocess
import sys

import pytest

from quackd.adapters.catalogue import BY_NAME

ADAPTERS = tuple(BY_NAME)
"""The seven quackd publishes, in the order every table that prints them uses."""

HEAVY = (
    "torch",
    "lerobot",
    "roslibpy",
    "zmq",
    "zeroconf",
    "paho",
    "mujoco",
    "onnxruntime",
)

SCRIPT = f"""
import sys
for name in {HEAVY!r}:
    sys.modules[name] = None  # any import of it now raises ImportError
import quackd
import quackd.cli
import quackd.doctor
import quackd.lan
import quackd.lan.announce
import quackd.lan.discover
import quackd.flock.mqtt_bus
import quackd.registry
from quackd.adapters.factory import BACKENDS, RobotSpec, describe, list_adapters, make_adapter
rows = list_adapters()
assert [r["name"] for r in rows] == [
    "microduck",
    "lerobot",
    "rosbridge",
    "open_duck",
    "xlerobot",
    "alohamini",
    "toddlerbot",
], rows
# every adapter that needs a library reports itself unusable when that library is gone. Asked
# through `sdk` rather than through `extra`, because an extra now also buys the adapter package
# itself, and a body with no SDK at all (the Open Duck, the ToddlerBot) is unaffected by torch
# being absent and should not claim to be.
assert not any(r["installed"] for r in rows if r["sdk"] is not None), rows
assert not any(r["sdk"] for r in rows), rows
for adapter, backends in BACKENDS.items():
    for backend in backends:
        m = describe(RobotSpec(adapter, backend))
        assert "stop" in m.verb_names(), (adapter, backend)
make_adapter("microduck:mujoco")
make_adapter("lerobot:real", address="COM5")
make_adapter("rosbridge:ws", address="ws://robot.local:9090")
for name in {HEAVY!r}:
    assert sys.modules.get(name) is None, name
print("OK")
"""


def test_everything_imports_without_any_extra() -> None:
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_the_default_path_did_not_import_a_heavy_module() -> None:
    import quackd.flock.mqtt_bus  # noqa: F401

    for name in ("torch", "lerobot", "roslibpy"):
        assert name not in sys.modules, f"{name} was imported on the default path"


def test_a_machine_with_no_adapter_installed_still_works_and_says_what_to_install() -> None:
    """The claim the split exists to make. quackd on its own is the loop, the executor and the
    contract, so it has to import, list every robot it publishes and refuse to guess at a body,
    all without a single adapter present.

    Driven through the discovery seam rather than by uninstalling anything, because the suite
    runs in an environment that has all seven and a test may not take them away from it."""
    import quackd.adapters.factory as factory
    from quackd.adapters.base import AdapterNotInstalled
    from quackd.adapters.factory import NoRobotNamed

    factory._installed.cache_clear()
    real = factory._installed
    factory._installed = dict  # type: ignore[assignment]  # no entry points, nothing importable
    try:
        rows = factory.list_adapters()
        assert [r["name"] for r in rows] == list(ADAPTERS), rows
        assert not any(r["adapter_installed"] for r in rows), rows
        assert all(r["extra"].startswith("quackd[") for r in rows), rows

        # a spec still parses, so a robot can be registered, printed and reasoned about on a
        # machine that cannot build it. Asking for the body itself is what refuses.
        assert factory.parse_robot_spec("lerobot:real").key == "lerobot:real"
        with pytest.raises(AdapterNotInstalled, match=r"quackd\[lerobot\]"):
            factory.describe(factory.parse_robot_spec("lerobot:real"))
        with pytest.raises(AdapterNotInstalled, match=r"quackd\[lerobot\]"):
            factory.make_adapter("lerobot:mock")

        with pytest.raises(NoRobotNamed, match="no robot adapter is installed"):
            factory.resolve_robot(None)
    finally:
        factory._installed = real
        factory._installed.cache_clear()


def test_the_only_adapter_installed_is_the_one_a_command_means() -> None:
    """A machine with one robot has no ambiguity to resolve, so making its owner type the name
    would be ceremony. With several installed quackd will not invent an answer, except that the
    Microduck stays the default it has always been, because the `duck: 0` starters mean it."""
    import quackd.adapters.factory as factory

    factory._installed.cache_clear()
    real = factory._installed
    try:
        factory._installed = lambda: {"lerobot": "quackd_lerobot"}  # type: ignore[assignment]
        assert factory.resolve_robot(None).key == "lerobot:mock"

        factory._installed = lambda: {  # type: ignore[assignment]
            "lerobot": "quackd_lerobot",
            "rosbridge": "quackd_rosbridge",
        }
        with pytest.raises(factory.NoRobotNamed, match="no robot named"):
            factory.resolve_robot(None)

        factory._installed = lambda: {  # type: ignore[assignment]
            "microduck": "quackd_microduck",
            "lerobot": "quackd_lerobot",
        }
        assert factory.resolve_robot(None).key == "microduck:sim2d"
    finally:
        factory._installed = real
        factory._installed.cache_clear()


def test_a_verdict_speaks_only_for_the_robots_this_machine_has() -> None:
    """An `infeasible` verdict ends a run by naming which other body could have done the task,
    and it builds each answer from that adapter's own manifest. Once the adapters became
    separate packages that became a question quackd cannot always answer: describing a robot
    whose package is absent raises, and it would raise here, inside the one outcome the whole
    verdict gate exists to produce. So the hint speaks for what is installed and nothing else,
    and a machine with one robot can only speak for that one."""
    import quackd.adapters.factory as factory
    from quackd.verdict import solo_hint

    factory._installed.cache_clear()
    real = factory._installed
    try:
        factory._installed = lambda: {"lerobot": "quackd_lerobot"}  # type: ignore[assignment]
        assert [name for name, _ in factory.installed_manifests()] == ["lerobot"]
        # the run-ending path itself: this used to raise AdapterNotInstalled for the six
        # bodies that are not here, turning a verdict into a crash
        hint = solo_hint({"payload_kg": 3}, None)
        assert "lerobot" in hint, hint
        assert not any(other in hint for other in ("microduck", "toddlerbot", "xlerobot")), hint
    finally:
        factory._installed = real
        factory._installed.cache_clear()
