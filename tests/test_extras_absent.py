"""The default install never imports an SDK: every adapter lists, describes and constructs
with torch, lerobot, roslibpy, zmq, zeroconf and paho all absent."""

from __future__ import annotations

import subprocess
import sys

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
assert not any(r["installed"] for r in rows if r["extra"] != "built-in"), rows
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
    import quackd.adapters.lerobot
    import quackd.adapters.rosbridge
    import quackd.flock.mqtt_bus  # noqa: F401

    for name in ("torch", "lerobot", "roslibpy"):
        assert name not in sys.modules, f"{name} was imported on the default path"
