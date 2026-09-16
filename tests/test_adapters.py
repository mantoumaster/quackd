"""Adapters wrap robots; the Microduck one must be a no-op over the 0.3 transports."""

from __future__ import annotations

import inspect

import pytest

import quackd.safety
from quackd.adapters.base import RobotAdapter, adapter_name, backend_name
from quackd.duckfile.parser import parse_duck_text
from quackd.safety import Executor
from quackd.transport.base import DuckState, Intent
from quackd.transport.mock import MockTransport
from quackd.verbs.registry import default_registry, registry_from_manifest
from quackd_microduck import MicroduckAdapter

DUCK = parse_duck_text(
    "---\nduck: 0\nname: t\ndescription: d\nverbs:\n  allow: [walk, kick, quack, get_frame]\n"
    "success: [x]\n---\n# T\nx\n"
)
SCRIPT: list[tuple[str, dict[str, float | str]]] = [
    ("walk", {"vx": 0.1, "duration_s": 0.3}),
    ("kick", {"leg": "left"}),
    ("quack", {"text": "hello there"}),
    ("get_frame", {}),
]


async def test_a_microduck_with_no_camera_does_not_advertise_the_verbs_that_need_one() -> None:
    """Upstream serves no frames over robotd's socket, so `--camera-url` is the whole camera.

    The manifest used to say `camera` either way, which made `observe` a core verb on a run
    that could only ever answer "this transport has no camera".
    """
    from quackd_microduck import microduck_manifest

    seeing = microduck_manifest("jsonrpc", camera=True)
    blind = microduck_manifest("jsonrpc", camera=False)
    needs_eyes = {"observe", "go_to", "search_scan", "approach_and"}
    assert needs_eyes <= set(seeing.verb_names())
    assert not (needs_eyes & set(blind.verb_names()))
    assert "camera" in seeing.sensors and "camera" not in blind.sensors
    # the verbs that do not look are unaffected
    assert {"move", "stop", "say", "gaze", "quack"} <= set(blind.verb_names())


async def test_connect_returns_the_manifest_and_the_registry_matches_the_default() -> None:
    adapter = MicroduckAdapter(MockTransport())
    assert isinstance(adapter, RobotAdapter)
    assert adapter.manifest is None
    manifest = await adapter.connect()
    assert manifest.model == "microduck" and manifest.backend == "mock"
    assert manifest.safety_authority.native == "robotd_deadman"
    assert adapter.transport.connected
    registry = registry_from_manifest(manifest, adapter)
    assert registry.names() == default_registry().names()
    assert backend_name(adapter) == "mock" and adapter_name(adapter) == "microduck"
    assert backend_name(MockTransport()) == "mock" and adapter_name(MockTransport()) is None
    await adapter.disconnect()
    assert not adapter.transport.connected


async def _run_script(executor: Executor) -> list[dict[str, object]]:
    for name, params in SCRIPT:
        result = await executor.run_verb(name, dict(params))
        assert result.ok, result.summary
    mock = getattr(executor.transport, "transport", executor.transport)  # unwrap the adapter
    return [i.model_dump() for i in mock.intents]


async def test_adapter_emits_exactly_the_intents_of_the_bare_transport() -> None:
    bare = MockTransport()
    await bare.connect()
    plain = await _run_script(Executor(default_registry(), bare, contract=DUCK.frontmatter))

    adapter = MicroduckAdapter(MockTransport())
    manifest = await adapter.connect()
    wrapped = await _run_script(
        Executor(registry_from_manifest(manifest, adapter), adapter, contract=DUCK.frontmatter)
    )
    assert wrapped == plain
    assert plain[0] == Intent.move(0.1, 0.0, 0.0).model_dump()


async def test_preconditions_come_from_the_adapter_by_name() -> None:
    fallen = MicroduckAdapter(MockTransport(states=[DuckState(fallen=True, posture="fallen")]))
    manifest = await fallen.connect()
    ex = Executor(registry_from_manifest(manifest, fallen), fallen, contract=DUCK.frontmatter)
    result = await ex.run_verb("walk")
    assert not result.ok and "fallen" in result.summary
    assert fallen.transport.intents_of("move") == []  # type: ignore[attr-defined]

    # a different robot can attach a different meaning to the same condition name
    custom = registry_from_manifest(
        manifest,
        implementations=fallen.implementations(),
        conditions={"standing": lambda _s: "nope, custom", "not_fallen": lambda _s: None},
    )
    ex = Executor(custom, MicroduckAdapter(MockTransport()), contract=DUCK.frontmatter)
    result = await ex.run_verb("move")
    assert not result.ok and "nope, custom" in result.summary


def test_the_executor_hardcodes_no_posture() -> None:
    source = inspect.getsource(quackd.safety)
    assert "sitting" not in source and "fallen" not in source


def test_robot_spec_parsing() -> None:
    from quackd.adapters.base import AdapterError
    from quackd.adapters.factory import RobotSpec, parse_robot_spec, parse_robots

    assert parse_robot_spec("microduck:mock") == RobotSpec("microduck", "mock")
    assert parse_robot_spec("microduck") == RobotSpec("microduck", "sim2d")  # first backend
    assert parse_robot_spec(" Microduck:SIM2D ").key == "microduck:sim2d"
    with pytest.raises(AdapterError, match="unknown adapter 'bogus'; choose one of"):
        parse_robot_spec("bogus:mock")
    with pytest.raises(AdapterError, match="unknown backend 'usb' for microduck"):
        parse_robot_spec("microduck:usb")
    specs = parse_robots("duck=microduck:sim2d, other=microduck:mock")
    assert [s.robot_id for s in specs] == ["duck", "other"]
    assert specs[1].key == "microduck:mock"
    with pytest.raises(AdapterError, match="name=<adapter>"):
        parse_robots("microduck:sim2d")
    with pytest.raises(AdapterError, match="duplicate robot name"):
        parse_robots("a=microduck:sim2d,a=microduck:mock")


def test_resolve_robot_is_the_flag_then_the_duck_then_the_simulator() -> None:
    """0.4's `--transport X` alias was promised for removal in 0.5, and is gone."""
    import inspect

    from quackd.adapters import factory
    from quackd.adapters.factory import resolve_robot

    assert resolve_robot("microduck:mock").key == "microduck:mock"
    assert resolve_robot(None, duck_default="microduck:mock").key == "microduck:mock"
    assert resolve_robot(None).key == "microduck:sim2d"
    assert "transport" not in inspect.signature(resolve_robot).parameters
    assert not hasattr(factory, "warn_once") and not hasattr(factory, "reset_warnings")


async def test_factory_describes_and_makes_adapters() -> None:
    from quackd.adapters.factory import (
        describe,
        list_adapters,
        make_adapter,
        parse_robot_spec,
        registry_for,
    )

    spec = parse_robot_spec("microduck:mock")
    static = describe(spec)
    assert static.model == "microduck" and static.backend == "mock" and static.id == "microduck"
    assert registry_for(spec).names() == default_registry().names()
    adapter = make_adapter("microduck:mock")
    live = await adapter.connect()
    assert live.digest() == static.digest()  # the static description is the real one
    rows = list_adapters()
    assert [r["name"] for r in rows] == [
        "microduck",
        "lerobot",
        "rosbridge",
        "open_duck",
        "xlerobot",
        "alohamini",
        "toddlerbot",
    ]
    assert rows[0]["installed"] and "sim2d" in rows[0]["backends"]
    assert rows[1]["extra"] == "quackd[lerobot]" and "real" in rows[1]["backends"]


async def test_health_wraps_the_heartbeat() -> None:
    healthy = MicroduckAdapter(MockTransport())
    await healthy.connect()
    report = await healthy.health()
    assert report.ok and report.battery_percent == 88 and report.extras["posture"] == "standing"

    sick = MicroduckAdapter(MockTransport(fail_heartbeat_after=0))
    await sick.connect()
    report = await sick.health()
    assert not report.ok and report.reason and "heartbeat" in report.reason


# ── the bodies that are not an arm: rest poses and cameras they do not have ─────────────

PARKS_NOTHING = (
    "microduck:mock",
    "rosbridge:mock",
    "open_duck:mock",
    "xlerobot:mock",
    "alohamini:mock",
    "toddlerbot:mock",
)
"""Every adapter but lerobot: the bodies that hold their own posture when the power goes."""

ONE_CAMERA = (
    "microduck:sim2d",
    "microduck:mock",
    "open_duck:mock",
    "rosbridge:mock",
    "xlerobot:mock",
    "alohamini:mock",
    "toddlerbot:mock",
    "lerobot:mock",
)
"""Every spec outside `MULTI_CAMERA_SPECS`, which is `lerobot:real` and nothing else."""


@pytest.mark.parametrize("spec", PARKS_NOTHING)
async def test_a_body_that_cannot_park_refuses_a_rest_pose_instead_of_ignoring_it(
    spec: str,
) -> None:
    """A rest pose is a joint-angle map for an arm that goes limp when it is disconnected,
    and none of these bodies is one. The only way a pose reaches one is a hand-edited
    `robots.json`, and taking it without ever driving to it would read as a promise quackd
    is keeping, so the file names itself instead."""
    from quackd.adapters.base import AdapterError, RestResult
    from quackd.adapters.factory import make_adapter

    with pytest.raises(AdapterError, match="does not return to a rest pose"):
        make_adapter(spec, rest_pose={"shoulder_pan": 1.0})

    parked = await make_adapter(spec).go_to_rest()
    assert parked == RestResult.none()
    assert not parked.recorded and not parked.reached


@pytest.mark.parametrize("spec", ONE_CAMERA)
def test_a_one_camera_body_refuses_a_second_camera_url_instead_of_dropping_it(spec: str) -> None:
    """`--camera-url` is repeatable because the LeRobot arm reads several cameras, and a flag
    that repeats on the command line repeats for every body. Opening the first url and
    quietly dropping the rest would put the reason the second camera is missing nowhere on
    the screen, so a body that reads one refuses and names the body that reads several."""
    from quackd.adapters.base import AdapterError
    from quackd.adapters.factory import make_adapter

    with pytest.raises(AdapterError) as refusal:
        make_adapter(spec, camera_url=["opencv://0?name=top", "opencv://1?name=side"])
    assert str(refusal.value) == (
        f"{spec} takes one --camera-url and 2 were given; only lerobot:real takes several"
    )
