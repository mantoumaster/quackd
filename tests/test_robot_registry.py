"""`quackd/registry.py`: the robots you have named and the flocks you made of them.

Not `tests/test_registry.py`, which is the verb registry. The two share a word and nothing
else.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from quackd.adapters.base import AdapterError
from quackd.registry import (
    FlockNotRunnable,
    Registry,
    RegistryError,
    RobotEntry,
    RobotInUse,
    StoredFlock,
    UnknownFlock,
    UnknownRobot,
    check_name,
    probe_all,
    probe_entry,
    registry_dir,
    resolve_robot_ref,
)


def _entry(name: str = "duck-a", spec: str = "microduck:mock", **kw: object) -> RobotEntry:
    return RobotEntry(name=name, spec=spec, **kw)  # type: ignore[arg-type]


# ── storage ─────────────────────────────────────────────────────────────────────────────


def test_a_robot_survives_the_process_with_every_field(tmp_path: Path) -> None:
    reg = Registry(tmp_path)
    reg.add_robot(
        _entry(
            address="tcp://10.0.0.5:9871",
            token="s3cret",
            camera_url="http://10.0.0.5:9872/snapshot.jpg",
            provider="anthropic",
            model="claude-opus-5",
            note="the cream one",
        )
    )
    again = Registry(tmp_path).robot("duck-a")
    assert again.spec == "microduck:mock"
    assert again.address == "tcp://10.0.0.5:9871"
    assert again.token == "s3cret"
    assert again.provider == "anthropic"
    assert again.note == "the cream one"
    stored = json.loads((tmp_path / "robots.json").read_text(encoding="utf-8"))
    assert stored["version"] == 1
    assert "name" not in stored["robots"]["duck-a"], "the key is the name; storing it twice drifts"


def test_a_write_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "nested" / "deeper")
    reg.add_robot(_entry())
    assert reg.robots_path.exists()
    assert not list(reg.robots_path.parent.glob("*.tmp"))


def test_the_directory_is_the_flag_then_the_env_then_the_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QUACKD_REGISTRY_DIR", str(tmp_path / "from-env"))
    assert registry_dir(tmp_path / "from-flag") == tmp_path / "from-flag"
    assert registry_dir() == tmp_path / "from-env"
    monkeypatch.delenv("QUACKD_REGISTRY_DIR")
    assert registry_dir() == Path("~/.quackd").expanduser()


def test_broken_json_is_one_line_and_the_file_is_never_overwritten(tmp_path: Path) -> None:
    (tmp_path / "robots.json").write_text("{ oh no", encoding="utf-8")
    reg = Registry(tmp_path)
    with pytest.raises(RegistryError, match="not valid JSON"):
        reg.robots()
    with pytest.raises(RegistryError):
        reg.add_robot(_entry())
    assert (tmp_path / "robots.json").read_text(encoding="utf-8") == "{ oh no"


def test_a_hand_edited_typo_names_its_own_field(tmp_path: Path) -> None:
    """Strict, unlike memory: a dropped `address` would send a robot to the wrong place."""
    (tmp_path / "robots.json").write_text(
        json.dumps({"version": 1, "robots": {"duck-a": {"spec": "microduck:mock", "adress": "x"}}}),
        encoding="utf-8",
    )
    with pytest.raises(RegistryError, match="adress"):
        Registry(tmp_path).robots()


def test_a_newer_file_is_refused_rather_than_half_read(tmp_path: Path) -> None:
    (tmp_path / "flocks.json").write_text(
        json.dumps({"version": 99, "flocks": {}}), encoding="utf-8"
    )
    with pytest.raises(RegistryError, match="version 99"):
        Registry(tmp_path).flocks()


# ── names ───────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "why"),
    [
        ("Duck-A", "not a valid"),
        ("-duck", "not a valid"),
        ("x" * 65, "not a valid"),
        ("3", "simulated ducks"),
        ("microduck", "is an adapter"),
        ("microduck-sim2d", "memory file name"),
    ],
)
def test_a_name_may_not_be_something_else_it_could_be_mistaken_for(name: str, why: str) -> None:
    with pytest.raises(RegistryError, match=why):
        check_name(name)


def test_a_flock_may_be_called_after_an_adapter_but_not_after_a_number() -> None:
    assert check_name("microduck", kind="flock") == "microduck"
    with pytest.raises(RegistryError, match="simulated ducks"):
        check_name("4", kind="flock")


def test_the_spec_is_normalised_and_a_bad_one_keeps_the_adapters_own_words() -> None:
    assert _entry(spec="MicroDuck").spec == "microduck:sim2d"
    with pytest.raises(ValidationError, match="unknown adapter 'bogus'"):
        _entry(spec="bogus:nope")


def test_a_hand_edited_bad_spec_names_the_robot_it_belongs_to(tmp_path: Path) -> None:
    (tmp_path / "robots.json").write_text(
        json.dumps({"version": 1, "robots": {"duck-a": {"spec": "bogus:nope"}}}), encoding="utf-8"
    )
    with pytest.raises(RegistryError, match=r"duck-a: spec: .*unknown adapter"):
        Registry(tmp_path).robots()


def test_an_unknown_provider_is_refused_by_name() -> None:
    with pytest.raises(ValidationError, match="unknown provider"):
        _entry(provider="hal9000")


# ── robots ──────────────────────────────────────────────────────────────────────────────


def test_registering_the_same_name_twice_is_refused(tmp_path: Path) -> None:
    reg = Registry(tmp_path)
    reg.add_robot(_entry())
    with pytest.raises(RegistryError, match="already registered"):
        reg.add_robot(_entry(spec="microduck:sim2d"))


def test_an_unknown_robot_is_one_line(tmp_path: Path) -> None:
    with pytest.raises(UnknownRobot, match="no robot called 'nope'"):
        Registry(tmp_path).robot("nope")


def test_update_sets_clears_and_moves_the_clock_on(tmp_path: Path) -> None:
    reg = Registry(tmp_path)
    added = reg.add_robot(_entry(address="tcp://old:1", note="keep me"))
    changed = reg.update_robot("duck-a", {"address": "tcp://new:2"})
    assert changed.address == "tcp://new:2"
    assert changed.note == "keep me", "a field nobody named is left alone"
    assert changed.added == added.added
    cleared = reg.update_robot("duck-a", {"address": None})
    assert cleared.address is None


def test_update_rejects_a_bad_value_without_writing_it(tmp_path: Path) -> None:
    reg = Registry(tmp_path)
    reg.add_robot(_entry())
    with pytest.raises(RegistryError, match="unknown provider"):
        reg.update_robot("duck-a", {"provider": "hal9000"})
    assert reg.robot("duck-a").provider is None


def test_the_memory_key_is_the_name_registered_and_the_spec_otherwise(tmp_path: Path) -> None:
    reg = Registry(tmp_path)
    reg.add_robot(_entry())
    assert resolve_robot_ref("duck-a", reg).memory_key == "duck-a"
    assert resolve_robot_ref("microduck:mock", reg).memory_key == "microduck:mock"


# ── the rest pose ───────────────────────────────────────────────────────────────────────

_FOLDED = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -113.5,
    "elbow_flex": 95.0,
    "wrist_flex": 12.5,
}
"""The pose the bench arm was folded into. `shoulder_lift` is past the travel its own
calibration file gives it, which is normal for a pose a person pushed the arm into by hand,
and the reason nothing here treats a recorded pose as a goal it may range-check."""


def test_a_rest_pose_survives_the_file_and_a_file_written_without_one_still_loads(
    tmp_path: Path,
) -> None:
    """A robots.json written before arms were parked has no `rest_pose` key at all, and
    upgrading quackd must not mean editing it: the field is absent rather than null, and
    every robot in that file has to keep loading, with no pose recorded for it."""
    reg = Registry(tmp_path)
    reg.add_robot(_entry("arm", "lerobot:real", rest_pose=_FOLDED))
    again = Registry(tmp_path).robot("arm")
    assert again.rest_pose == _FOLDED
    assert again.public()["rest_pose"] == _FOLDED
    stored = json.loads((tmp_path / "robots.json").read_text(encoding="utf-8"))
    assert stored["robots"]["arm"]["rest_pose"] == _FOLDED, "degrees by joint, as they were read"

    (tmp_path / "robots.json").write_text(
        json.dumps({"version": 1, "robots": {"arm": {"spec": "lerobot:real"}}}), encoding="utf-8"
    )
    older = Registry(tmp_path).robot("arm")
    assert older.rest_pose is None
    assert older.public()["rest_pose"] is None
    assert older.adapter_kwargs()["rest_pose"] is None


@pytest.mark.parametrize(
    ("pose", "why"),
    [
        ({}, "rest_pose must name at least one joint"),
        ({"elbow_flex": float("inf")}, r"rest_pose\.elbow_flex must be a finite number of degrees"),
        (
            {"elbow_flex": float("-inf")},
            r"rest_pose\.elbow_flex must be a finite number of degrees",
        ),
        ({"elbow_flex": float("nan")}, r"rest_pose\.elbow_flex must be a finite number of degrees"),
    ],
)
def test_a_rest_pose_that_is_not_finite_degrees_by_joint_is_refused(
    pose: dict[str, float], why: str
) -> None:
    """`inf` and `nan` are floats as far as pydantic is concerned, so they reach the field
    itself. An arm told to park at infinity would push until it stalled, and one told to park
    at `nan` compares false against every reading and so is never at its pose."""
    with pytest.raises(ValidationError, match=why):
        _entry("arm", "lerobot:real", rest_pose=pose)


def test_a_rest_pose_angle_that_is_not_a_number_names_the_joint_it_sits_on() -> None:
    """Here the joint is in the locator rather than in the sentence: pydantic refuses the
    value before the field validator ever runs, so the wording is its own. What matters is
    that a person editing robots.json is told which joint to look at."""
    for bad in ("sideways", None, [1]):
        with pytest.raises(
            ValidationError, match=r"rest_pose\.elbow_flex\n\s+Input should be a valid number"
        ):
            _entry("arm", "lerobot:real", rest_pose={"elbow_flex": bad})


def test_adapter_kwargs_carries_the_rest_pose_and_an_unregistered_spec_has_none(
    tmp_path: Path,
) -> None:
    reg = Registry(tmp_path)
    reg.add_robot(_entry("arm", "lerobot:real", rest_pose=_FOLDED))
    kwargs = resolve_robot_ref("arm", reg).adapter_kwargs()
    assert kwargs["rest_pose"] == _FOLDED
    kwargs["rest_pose"]["elbow_flex"] = 0.0
    assert reg.robot("arm").rest_pose == _FOLDED, "the kwargs hold a copy, not the stored dict"
    bare = resolve_robot_ref("lerobot:real", reg)
    assert not bare.registered
    assert bare.adapter_kwargs()["rest_pose"] is None, "a pose is something you get by naming it"


# ── cameras ─────────────────────────────────────────────────────────────────────────────

_TOP = "http://arm:9872/top.jpg"
_SIDE = "http://arm:9873/side.jpg"


def test_one_camera_is_stored_as_a_string_and_several_as_a_list(tmp_path: Path) -> None:
    """The string form is not a leftover: a robot with one camera has to write a file an
    older quackd can still read, so the list shape appears only when there is a second."""
    reg = Registry(tmp_path)
    reg.add_robot(_entry("arm", "lerobot:real", camera_url=_TOP))
    one = Registry(tmp_path).robot("arm")
    assert one.camera_url == _TOP
    assert one.camera_urls == (_TOP,), "everything reads camera_urls, which is always a tuple"
    stored = json.loads((tmp_path / "robots.json").read_text(encoding="utf-8"))
    assert stored["robots"]["arm"]["camera_url"] == _TOP, "an older quackd reads this file too"

    reg.update_robot("arm", {"camera_url": [_TOP, _SIDE]})
    both = Registry(tmp_path).robot("arm")
    assert both.camera_url == [_TOP, _SIDE]
    assert both.camera_urls == (_TOP, _SIDE), "the order given is kept: the first is the primary"
    stored = json.loads((tmp_path / "robots.json").read_text(encoding="utf-8"))
    assert stored["robots"]["arm"]["camera_url"] == [_TOP, _SIDE]


def test_a_robots_json_written_before_the_second_camera_still_loads(tmp_path: Path) -> None:
    """The back-compat guarantee for anyone upgrading: 0.9 wrote `camera_url` as a plain
    string, and that file has to load without an edit and mean one camera."""
    (tmp_path / "robots.json").write_text(
        json.dumps(
            {
                "version": 1,
                "robots": {"duck-a": {"spec": "microduck:mock", "camera_url": "http://old:9872"}},
            }
        ),
        encoding="utf-8",
    )
    entry = Registry(tmp_path).robot("duck-a")
    assert entry.camera_url == "http://old:9872"
    assert entry.camera_urls == ("http://old:9872",)
    assert entry.adapter_kwargs()["camera_url"] == ("http://old:9872",)


def test_a_blank_or_repeated_camera_url_is_refused_in_its_own_words() -> None:
    with pytest.raises(ValidationError, match="camera_url: an empty url is not a camera"):
        _entry("arm", "lerobot:real", camera_url="")
    with pytest.raises(ValidationError, match="camera_url: an empty url is not a camera"):
        _entry("arm", "lerobot:real", camera_url=[_TOP, "   "])
    with pytest.raises(ValidationError, match=f"camera_url: '{_TOP}' is listed twice"):
        _entry("arm", "lerobot:real", camera_url=[_TOP, _TOP])


def test_a_second_camera_is_refused_on_a_body_that_reads_one(tmp_path: Path) -> None:
    """A hand-edited file is the only way a duck gets two urls, and the rule is that a file
    saying something untrue names itself rather than having the extra camera quietly
    dropped: the duck would open the first one and nobody would learn why."""
    with pytest.raises(
        ValidationError,
        match="microduck:mock takes one camera url; only lerobot:real takes several",
    ):
        _entry("duck-a", "microduck:mock", camera_url=["http://duck:1", "http://duck:2"])
    (tmp_path / "robots.json").write_text(
        json.dumps(
            {
                "version": 1,
                "robots": {
                    "duck-a": {
                        "spec": "microduck:mock",
                        "camera_url": ["http://duck:1", "http://duck:2"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RegistryError, match=r"duck-a: .*takes one camera url"):
        Registry(tmp_path).robots()
    arm = _entry("arm", "lerobot:real", camera_url=[_TOP, _SIDE])
    assert arm.camera_urls == (_TOP, _SIDE), "the one body that reads several takes them"


def test_adapter_kwargs_carries_every_camera_and_a_flag_replaces_the_whole_set(
    tmp_path: Path,
) -> None:
    """`--camera-url` on the line says where the cameras are today, so it replaces the stored
    set rather than adding to it. An arm whose side camera is unplugged is run by naming the
    one that is plugged in, not by editing the registry and putting it back afterwards."""
    reg = Registry(tmp_path)
    reg.add_robot(_entry("arm", "lerobot:real", camera_url=[_TOP, _SIDE]))
    resolved = resolve_robot_ref("arm", reg)
    assert resolved.adapter_kwargs()["camera_url"] == (_TOP, _SIDE)
    today = resolved.adapter_kwargs(camera_url=["http://laptop:8080/usb0"])
    assert today["camera_url"] == ("http://laptop:8080/usb0",), "a flag replaces, never appends"
    assert reg.robot("arm").camera_urls == (_TOP, _SIDE), "and it changes nothing on the file"
    assert resolve_robot_ref("lerobot:real", reg).adapter_kwargs()["camera_url"] == ()


# ── flocks ──────────────────────────────────────────────────────────────────────────────


def _pair(tmp_path: Path) -> Registry:
    reg = Registry(tmp_path)
    reg.add_robot(_entry("duck-a"))
    reg.add_robot(_entry("duck-b"))
    reg.add_robot(_entry("arm", "lerobot:mock"))
    return reg


def test_a_flock_keeps_the_order_you_listed_its_robots_in(tmp_path: Path) -> None:
    reg = _pair(tmp_path)
    reg.add_flock(StoredFlock(name="kitchen", members=["arm", "duck-a"]))
    assert list(Registry(tmp_path).roster("kitchen")) == ["arm", "duck-a"]


def test_a_flock_may_only_name_robots_that_are_registered(tmp_path: Path) -> None:
    reg = _pair(tmp_path)
    with pytest.raises(UnknownRobot, match="ghost"):
        reg.add_flock(StoredFlock(name="kitchen", members=["duck-a", "ghost"]))


def test_a_flock_may_not_list_one_robot_twice_or_hold_nine(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="listed twice"):
        StoredFlock(name="k", members=["duck-a", "duck-a"])
    with pytest.raises(ValidationError, match="at most 8"):
        StoredFlock(name="k", members=[f"duck-{i}" for i in range(9)])


def test_two_flocks_may_not_share_a_name(tmp_path: Path) -> None:
    reg = _pair(tmp_path)
    reg.add_flock(StoredFlock(name="kitchen", members=["duck-a"]))
    with pytest.raises(RegistryError, match="already exists"):
        reg.add_flock(StoredFlock(name="kitchen", members=["duck-b"]))


def test_an_unknown_flock_is_one_line(tmp_path: Path) -> None:
    with pytest.raises(UnknownFlock, match="no flock called 'nope'"):
        Registry(tmp_path).flock("nope")


def test_edit_adds_removes_describes_and_renames(tmp_path: Path) -> None:
    reg = _pair(tmp_path)
    reg.add_flock(StoredFlock(name="kitchen", members=["duck-a"]))
    reg.update_flock("kitchen", add=["arm"], description="the two by the sink")
    assert reg.flock("kitchen").members == ["duck-a", "arm"]
    reg.update_flock("kitchen", remove=["duck-a"])
    assert reg.flock("kitchen").members == ["arm"]
    reg.update_flock("kitchen", rename="sink")
    assert reg.get_flock("kitchen") is None
    assert reg.flock("sink").members == ["arm"]
    assert reg.update_flock("sink", description="").description is None


def test_edit_refuses_a_member_it_already_has_or_never_had(tmp_path: Path) -> None:
    reg = _pair(tmp_path)
    reg.add_flock(StoredFlock(name="kitchen", members=["duck-a"]))
    with pytest.raises(RegistryError, match="already in kitchen"):
        reg.update_flock("kitchen", add=["duck-a"])
    with pytest.raises(RegistryError, match="not in kitchen"):
        reg.update_flock("kitchen", remove=["arm"])


def test_deleting_a_flock_leaves_its_robots_registered(tmp_path: Path) -> None:
    reg = _pair(tmp_path)
    reg.add_flock(StoredFlock(name="kitchen", members=["duck-a", "arm"]))
    reg.delete_flock("kitchen")
    assert reg.get_flock("kitchen") is None
    assert set(reg.robots()) == {"duck-a", "duck-b", "arm"}


# ── the one broken state ────────────────────────────────────────────────────────────────


def test_removing_a_robot_a_flock_names_is_refused_and_force_drops_it(tmp_path: Path) -> None:
    reg = _pair(tmp_path)
    reg.add_flock(StoredFlock(name="kitchen", members=["duck-a", "arm"]))
    reg.add_flock(StoredFlock(name="garage", members=["duck-a"]))
    with pytest.raises(RobotInUse, match="garage, kitchen"):
        reg.remove_robot("duck-a")
    assert reg.get_robot("duck-a") is not None, "a refused remove removes nothing"
    assert reg.remove_robot("duck-a", force=True) == ["garage", "kitchen"]
    assert reg.get_robot("duck-a") is None
    assert reg.flock("kitchen").members == ["arm"]
    assert reg.flock("garage").members == []


def test_a_dangling_member_is_reported_and_refuses_to_run(tmp_path: Path) -> None:
    """A hand-edited file is the only way in, and it must not turn into a smaller flock."""
    reg = _pair(tmp_path)
    reg.add_flock(StoredFlock(name="kitchen", members=["duck-a", "arm"]))
    (tmp_path / "robots.json").write_text(
        json.dumps({"version": 1, "robots": {"arm": {"spec": "lerobot:mock"}}}), encoding="utf-8"
    )
    flock = reg.flock("kitchen")
    assert reg.missing_members(flock) == ["duck-a"]
    assert flock.public(["duck-a"])["runnable"] is False
    with pytest.raises(FlockNotRunnable, match="no longer registered"):
        reg.roster("kitchen")
    reg.update_flock("kitchen", remove=["duck-a"])
    assert list(reg.roster("kitchen")) == ["arm"], "removing a dangling member is the repair"


def test_an_empty_flock_says_how_to_fill_it(tmp_path: Path) -> None:
    reg = _pair(tmp_path)
    reg.add_flock(StoredFlock(name="kitchen"))
    with pytest.raises(FlockNotRunnable, match="has no robots"):
        reg.roster("kitchen")


# ── resolving a reference ───────────────────────────────────────────────────────────────


def test_a_reference_is_the_registry_then_a_spec_then_one_line(tmp_path: Path) -> None:
    reg = _pair(tmp_path)
    assert resolve_robot_ref(None, reg).spec.key == "microduck:sim2d"
    assert resolve_robot_ref(None, reg, duck_default="lerobot:mock").spec.key == "lerobot:mock"
    registered = resolve_robot_ref("duck-a", reg)
    assert registered.registered and registered.spec.name == "duck-a"
    assert registered.label == "duck-a (microduck:mock)"
    bare = resolve_robot_ref("microduck", reg)
    assert not bare.registered and bare.spec.key == "microduck:sim2d"
    with pytest.raises(AdapterError, match="neither a registered robot"):
        resolve_robot_ref("ghost", reg)
    with pytest.raises(AdapterError, match="unknown adapter"):
        resolve_robot_ref("ghost:nope", reg)


def test_a_flag_beats_the_stored_endpoint(tmp_path: Path) -> None:
    reg = Registry(tmp_path)
    reg.add_robot(_entry(address="tcp://stored:1", token="stored", camera_url="http://stored"))
    resolved = resolve_robot_ref("duck-a", reg)
    assert resolved.adapter_kwargs()["address"] == "tcp://stored:1"
    through_a_tunnel = resolved.adapter_kwargs(address="tcp://localhost:9999")
    assert through_a_tunnel["address"] == "tcp://localhost:9999"
    assert through_a_tunnel["token"] == "stored", "one flag overrides one field"


# ── probing ─────────────────────────────────────────────────────────────────────────────


async def test_a_probe_reaches_a_mock_and_reports_a_dead_address() -> None:
    ok = await probe_entry(_entry("duck-a", "microduck:mock"))
    assert ok.reachable is True and "ok" in ok.detail
    dead = await probe_entry(
        _entry("far", "open_duck:bridge", address="tcp://127.0.0.1:1"), timeout_s=3.0
    )
    assert dead.reachable is False and dead.detail


async def test_a_probe_declines_the_backend_that_would_download_a_model() -> None:
    skipped = await probe_entry(_entry("phys", "microduck:mujoco"))
    assert skipped.reachable is None and "skipped" in skipped.detail


async def test_a_probe_times_out_rather_than_hanging(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    class Slow:
        async def connect(self) -> None:
            await asyncio.sleep(30)

        async def close(self) -> None:
            return None

    monkeypatch.setattr("quackd.adapters.factory.make_adapter", lambda *a, **k: Slow())
    result = await probe_entry(_entry(), timeout_s=0.05)
    assert result.reachable is False and "timed out" in result.detail


def test_probe_all_answers_for_every_robot_at_once() -> None:
    rows = probe_all([_entry("a", "microduck:mock"), _entry("b", "lerobot:mock")])
    assert set(rows) == {"a", "b"}
    assert all(r.reachable for r in rows.values())


def _hand_back(monkeypatch: pytest.MonkeyPatch, arm: Any, seen: dict[str, Any]) -> None:
    """Every probe builds this one in-memory arm, and `seen` keeps what it was built with.

    `probe_entry` imports `make_adapter` inside the call, so the factory's module attribute
    is the seam: there is no other way to hand a probe a body it did not make itself."""
    from quackd.adapters.lerobot import LeRobotAdapter

    def fake_make_adapter(spec: Any, **kwargs: Any) -> Any:
        seen.clear()
        seen["spec"] = spec.key
        seen.update(kwargs)
        return LeRobotAdapter(arm)

    monkeypatch.setattr("quackd.adapters.factory.make_adapter", fake_make_adapter)


async def test_a_probe_hands_the_rest_pose_over_and_drives_the_arm_nowhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A probe reads and lets go, so it must never move the arm: `quackd robot list --probe`
    is something you type while the arm is holding something, and a probe that parked it
    would put whatever it held on the floor.

    The pose still has to reach the adapter, because it is what decides whether torque may
    drop when the probe disconnects. An arm away from it keeps torque, and the row says so on
    the same line that says the arm answered: somebody has to hold it before the power goes."""
    from quackd.adapters.lerobot.mock import LeRobotMock

    seen: dict[str, Any] = {}
    arm = LeRobotMock(rest_pose=_FOLDED)
    _hand_back(monkeypatch, arm, seen)

    result = await probe_entry(_entry("arm", "lerobot:real", rest_pose=_FOLDED))

    assert seen["spec"] == "lerobot:real"
    assert seen["rest_pose"] == _FOLDED, "the probe builds the arm with the pose it registered"
    assert "rest" not in arm.sequence, f"the probe drove the arm: {arm.sequence}"
    assert arm.sequence == ["close"]
    assert arm.actions == [], "and sent it no goal of any kind"
    assert result.reachable is True
    assert result.detail.startswith("ok")
    assert result.detail.endswith("torque left on: not at its rest pose"), result.detail
    assert arm.torque is True, "torque is what the note is about"


async def test_a_probe_on_an_arm_already_at_its_pose_says_nothing_about_torque(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of the same rule: the torque line is news, and an arm that is where it
    rests is let go of the way it always was, with nothing added to the row."""
    from quackd.adapters.lerobot.mock import REST, LeRobotMock

    seen: dict[str, Any] = {}
    arm = LeRobotMock(rest_pose=dict(REST))
    _hand_back(monkeypatch, arm, seen)

    result = await probe_entry(_entry("arm", "lerobot:real", rest_pose=dict(REST)))

    assert "rest" not in arm.sequence
    assert result.detail == "ok"
    assert arm.torque is False, "an arm at its rest pose may be let go of"
