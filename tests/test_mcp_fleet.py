"""A fleet over MCP: eight robot_* tools, one executor, budget and heartbeat per robot.

Driven in-process by the SDK's own client over memory streams, exactly like the one-robot
tests.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

import pytest
from mcp.client.session import ClientSession
from mcp.shared.memory import create_client_server_memory_streams

from quackd.adapters.factory import RobotSpec, make_adapter
from quackd.mcp_server import Fleet, build_fleet_server
from quackd.transport.base import TransportError
from quackd.transport.mock import MockTransport


def one_duck() -> dict[str, Any]:
    return {"duck": make_adapter(RobotSpec("microduck", "sim2d", "duck"), seed=1)}


@contextlib.asynccontextmanager
async def connected(
    robots: dict[str, Any] | None = None, **kwargs: Any
) -> AsyncIterator[tuple[ClientSession, Fleet]]:
    server, fleet = build_fleet_server(robots or one_duck(), heartbeat_period_s=0.05, **kwargs)
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        low = server._lowlevel_server
        task = asyncio.create_task(
            low.run(server_streams[0], server_streams[1], low.create_initialization_options())
        )
        try:
            async with ClientSession(client_streams[0], client_streams[1]) as client:
                await client.initialize()
                yield client, fleet
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


def _data(result: Any) -> dict[str, Any]:
    assert not result.is_error, result
    assert result.structured_content is not None
    return result.structured_content


async def test_dry_run_still_observes_but_moves_nothing() -> None:
    async with connected(dry_run=True) as (client, fleet):
        seen = await client.call_tool("robot_observe", {"robot": "duck"})
        assert {c.type for c in seen.content} == {"text", "image"}  # read-only verbs run
        moved = _data(
            await client.call_tool("robot_run_verb", {"verb": "move", "params": {"vx": 0.2}})
        )
        assert moved["ok"] and moved["data"].get("dry_run") is True
        assert fleet.sessions["duck"].transport.world.steps == 0


class _Dead:
    name = "mock"

    def __init__(self) -> None:
        self.closed = False

    async def connect(self) -> None:
        raise TransportError("no robot answered")

    async def stop(self) -> None:
        pass

    async def close(self) -> None:
        self.closed = True

    def now(self) -> float:
        return 0.0

    async def heartbeat(self) -> None:
        pass


async def test_connect_is_fail_fast_and_closes_what_did_connect() -> None:
    first = MockTransport()
    _server, fleet = build_fleet_server({"a": first, "b": _Dead()}, heartbeat_period_s=0.05)
    with pytest.raises(TransportError, match="no robot answered"):
        await fleet.connect_all()
    assert first.connected is False  # connected, then closed again
    assert fleet.default == "a"


def test_build_needs_a_robot_and_a_known_default() -> None:
    with pytest.raises(ValueError, match="at least one robot"):
        build_fleet_server({})
    with pytest.raises(ValueError, match="default robot"):
        build_fleet_server({"a": MockTransport()}, default="zz")


@pytest.mark.parametrize("spec", ["open_duck:mock", "microduck:mock"])
async def test_a_camera_robot_that_is_not_the_simulator_gets_a_detector(spec: str) -> None:
    """The 0.5 fix landed in `quackd run` and missed this entry point.

    `build_fleet_server` still keyed the detector on the backend being `sim2d`, so every
    hardware body over MCP fetched frames, detected nothing because nothing was detecting,
    and reported that it could not see. The decision now happens after connect, against
    what the robot said it has."""
    robot = make_adapter(spec)
    async with connected({"r": robot}) as (client, fleet):
        session = fleet.sessions["r"]
        assert robot.manifest is not None and "camera" in robot.manifest.sensors
        assert session.detector is not None, f"{spec} runs blind over MCP"
        assert session.executor.detector is session.detector
        # and the verbs that need one no longer refuse
        res = _data(await client.call_tool("robot_run_verb", {"verb": "search_scan"}))
        assert "needs a detector" not in res["summary"]


def test_the_detector_policy_is_the_camera_and_nothing_else() -> None:
    """Both entry points call one function so they cannot drift again. Every body that can
    connect offline has a camera, so the two negative cases are pinned here: a body with no
    camera gets nothing, and an explicit `--detector` is never overruled."""
    from quackd.perception import detector_for
    from quackd.perception.color_blob import ColorBlobDetector

    assert detector_for(["camera", "imu"]) is not None
    assert detector_for(["joint_state"]) is None  # lerobot:real
    assert detector_for(["odometry"]) is None  # rosbridge:ws before it connects
    mine = ColorBlobDetector()
    assert detector_for(["camera"], mine) is mine
    assert detector_for(["odometry"], mine) is mine
