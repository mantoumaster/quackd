"""`quackd serve-mcp --flock NAME` and `--robot NAME`: the fleet, built from the registry.

No new tools. A stored flock and `--robots name=spec` spell the same fleet; what the registry
adds is that each member brings its own address, token and camera, and its own memory file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from quackd.mcp_server import fleet_from_flags
from quackd.registry import Registry, RobotEntry, StoredFlock
from tests.test_mcp_fleet import _data, connected


def _seed(tmp_path: Path, robots: dict[str, dict[str, Any]], members: list[str]) -> Registry:
    registry = Registry(tmp_path)
    for name, fields in robots.items():
        registry.add_robot(RobotEntry(name=name, **fields))
    if members:
        registry.add_flock(StoredFlock(name="kitchen", members=members))
    return registry


# ── building the fleet ──────────────────────────────────────────────────────────────────


def test_a_stored_flock_gives_every_member_its_own_endpoint(tmp_path: Path) -> None:
    _seed(
        tmp_path,
        {
            "duck-a": {
                "spec": "open_duck:bridge",
                "address": "tcp://10.0.0.5:9871",
                "token": "s3cret",
            },
            "duck-b": {"spec": "open_duck:mock"},
        },
        ["duck-a", "duck-b"],
    )
    plan = fleet_from_flags(flock="kitchen", registry_dir=str(tmp_path))
    assert list(plan.adapters) == ["duck-a", "duck-b"]
    assert plan.default == "duck-a", "the flock's own order decides, not the first Microduck"
    bridge = plan.adapters["duck-a"].transport
    assert bridge.address == "tcp://10.0.0.5:9871"
    assert bridge.token == "s3cret"
    assert plan.manifests["duck-a"].id == "duck-a"
    assert plan.memory_keys == {"duck-a": "duck-a", "duck-b": "duck-b"}


def test_an_ad_hoc_fleet_keys_its_memory_by_the_body_as_it_always_did(tmp_path: Path) -> None:
    plan = fleet_from_flags(robots="a=microduck:mock,b=lerobot:mock", registry_dir=str(tmp_path))
    assert list(plan.adapters) == ["a", "b"]
    assert plan.memory_keys == {}, "nothing wrote a name down, so nothing is keyed by one"
    assert plan.default is None


def test_a_registered_robot_brings_its_endpoint_and_a_flag_still_wins(tmp_path: Path) -> None:
    _seed(
        tmp_path,
        {"scout": {"spec": "open_duck:bridge", "address": "tcp://stored:1", "token": "stored"}},
        [],
    )
    plan = fleet_from_flags(robot="scout", registry_dir=str(tmp_path))
    assert plan.adapters["scout"].transport.address == "tcp://stored:1"
    assert plan.memory_keys == {"scout": "scout"}
    through = fleet_from_flags(
        robot="scout", registry_dir=str(tmp_path), address="tcp://tunnel:9999"
    )
    assert through.adapters["scout"].transport.address == "tcp://tunnel:9999"
    assert through.adapters["scout"].transport.token == "stored", "one flag, one field"


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"flock": "kitchen", "robot": "duck-a"}, "choose one"),
        ({"flock": "kitchen", "robots": "a=microduck:mock"}, "choose one"),
        ({"flock": "kitchen", "address": "tcp://x:1"}, "from the registry"),
        ({"flock": "kitchen", "token": "x"}, "from the registry"),
        ({"flock": "ghost"}, "no flock called 'ghost'"),
    ],
)
def test_the_flags_that_contradict_each_other_are_refused(
    tmp_path: Path, kwargs: dict[str, Any], needle: str
) -> None:
    _seed(tmp_path, {"duck-a": {"spec": "microduck:mock"}}, ["duck-a"])
    with pytest.raises(SystemExit, match=needle):
        fleet_from_flags(registry_dir=str(tmp_path), **kwargs)


def test_a_dangling_flock_is_refused_in_one_line(tmp_path: Path) -> None:
    _seed(tmp_path, {"duck-a": {"spec": "microduck:mock"}}, ["duck-a"])
    (tmp_path / "robots.json").write_text(
        json.dumps({"version": 1, "robots": {}}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="no longer registered"):
        fleet_from_flags(flock="kitchen", registry_dir=str(tmp_path))


def test_a_flock_duck_is_still_refused_over_mcp(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="quackd run"):
        fleet_from_flags(duckfile="flock-hello", registry_dir=str(tmp_path))


# ── serving it ──────────────────────────────────────────────────────────────────────────


async def test_a_stored_flock_serves_and_names_its_members(tmp_path: Path) -> None:
    _seed(
        tmp_path,
        {"duck-a": {"spec": "microduck:mock"}, "arm": {"spec": "lerobot:mock"}},
        ["duck-a", "arm"],
    )
    plan = fleet_from_flags(flock="kitchen", registry_dir=str(tmp_path))
    async with connected(
        plan.adapters,
        manifests=plan.manifests,
        memory_keys=plan.memory_keys,
        default=plan.default,
        memory_dir=str(tmp_path / "mem"),
    ) as (client, fleet):
        listed = await client.call_tool("robot_list", {})
        names = [row["name"] for row in _data(listed)["robots"]]
        assert names == ["duck-a", "arm"]
        assert fleet.default == "duck-a"
        await client.call_tool(
            "robot_remember", {"text": "the charger is under the desk", "robot": "duck-a"}
        )
    assert (tmp_path / "mem" / "duck-a.jsonl").exists(), "keyed by the name it was registered as"
    assert not (tmp_path / "mem" / "microduck-mock.jsonl").exists()


async def test_an_unregistered_fleet_still_keys_memory_by_the_body(tmp_path: Path) -> None:
    """The regression guard for the keying change: nothing about `--robots` moved."""
    plan = fleet_from_flags(robots="a=microduck:mock,b=microduck:mock", registry_dir=str(tmp_path))
    async with connected(
        plan.adapters,
        manifests=plan.manifests,
        memory_keys=plan.memory_keys,
        memory_dir=str(tmp_path / "mem"),
    ) as (client, _fleet):
        await client.call_tool("robot_remember", {"text": "a note", "robot": "a"})
    assert (tmp_path / "mem" / "microduck-mock.jsonl").exists()
    assert not (tmp_path / "mem" / "a.jsonl").exists()


def test_serve_mcp_forwards_the_flock_and_the_registry_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from quackd.cli import app

    seen: dict[str, Any] = {}
    monkeypatch.setattr("quackd.mcp_server.serve", lambda **kw: seen.update(kw))
    result = CliRunner().invoke(
        app,
        ["serve-mcp", "--flock", "kitchen", "--registry-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert seen["flock"] == "kitchen"
    assert seen["registry_dir"] == str(tmp_path)
