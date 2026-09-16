"""`quackd doctor`: it runs everywhere, it answers in data, and the WebSocket stub refuses
honestly.

The split between `collect` and `render` is the thing worth testing. The collector must not
know what green means, or `--json` has nothing underneath it to print, and the renderer must
not know anything the report does not hold.
"""

from __future__ import annotations

import io
import json
import sys
from typing import Any

import pytest
from rich.console import Console
from typer.testing import CliRunner

from quackd import doctor
from quackd.agent.providers.factory import CLOUD_NAMES, KEY_ENV, PROVIDER_NAMES
from quackd.cli import app
from quackd.transport.base import CameraFrame, TransportError
from quackd_lerobot import LeRobotAdapter
from quackd_lerobot.mock import REST, LeRobotMock
from quackd_microduck.transports.websocket_stub import WebSocketTransport


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


def test_doctor_lists_every_provider_and_its_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The providers table is built from the same tables `make_provider` dispatches on, so a
    vendor that is wired up but missing a row here would be a vendor nobody could discover.

    The keys are blanked, and blanked rather than deleted: doctor names the variable it wants
    only when it is empty, and the CLI loads a developer's `.env` in its root callback, so on
    a machine with one real key this asserted against a masked value instead. Empty reads as
    unset and, unlike `delenv`, survives `load_dotenv`, which does not overwrite a name that
    is already in the environment."""
    for env in set(KEY_ENV.values()):
        monkeypatch.setenv(env, "")
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    for name in PROVIDER_NAMES:
        assert name in flat, f"{name} has no row in doctor's providers table"
    for cloud in CLOUD_NAMES:
        assert KEY_ENV[cloud] in flat, f"{cloud} does not say which key it wants"


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
        # eight table titles used to carry these paths; losing them was losing where to read
        "read more:",
        "docs/adapters/lerobot.md",
        "docs/adr/0030-mujoco-physics-backend.md",
    ):
        assert needle in out, needle


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_render_loses_nothing_on_a_codepage_that_cannot_carry_it(
    monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    """A Windows pipe is cp1252 and this is the command people paste into an issue.

    `errors="strict"` is the assertion: the stream raises rather than substituting, so this
    passes only if every character doctor wrote can be carried. Counting question marks
    cannot do the same job, because the transports note says "not on a robot?" on a machine
    with no socket and that question mark is prose, not a casualty. Both platform branches of
    that note are checked, since only one of them runs on any given machine."""
    monkeypatch.setattr(sys, "platform", platform)
    raw = io.BytesIO()
    console = Console(file=io.TextIOWrapper(raw, encoding="cp1252", errors="strict"), width=120)
    doctor.render(console, doctor.collect())
    console.file.flush()
    out = raw.getvalue().decode("cp1252")
    assert out.isascii()
    assert "[ok] built-in" in out, "the registry's tick becomes something readable"
    assert "duck-ipc-proto API" in out


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


# ── what the probe does to a real arm: park it, and look through every camera ───────────

AWAY_FROM_REST = {"shoulder_pan": 40.0, "shoulder_lift": -80.0}
"""A pose the mock arm does not start in, so reaching it takes an actual move."""


def _probe_of(report: doctor.DoctorReport) -> doctor.ProbeReport:
    assert report.robot is not None and report.robot.error is None, report.robot
    assert report.robot.probe is not None and report.robot.probe.error is None, report.robot.probe
    return report.robot.probe


def _row(report: doctor.DoctorReport, what: str) -> doctor.ProbeRow:
    rows = _probe_of(report).rows
    found = [r for r in rows if r.what == what]
    assert len(found) == 1, f"wanted one {what!r} row, the probe reported {[r.what for r in rows]}"
    return found[0]


def _probed(monkeypatch: pytest.MonkeyPatch, transport: Any, **kwargs: Any) -> doctor.DoctorReport:
    """Probe one transport the test built and can read afterwards.

    `probe` makes its own adapter through the factory, so that call is the only seam a fake
    fits through. What these tests need on the other side of it reaches no command line: an
    arm that refuses to move, and a second camera."""
    monkeypatch.setattr(
        "quackd.adapters.factory.make_adapter", lambda *_a, **_k: LeRobotAdapter(transport)
    )
    return doctor.collect("lerobot:mock", address="mock://arm", **kwargs)


def test_doctor_returns_a_probed_arm_to_its_rest_pose_and_reports_that_it_did() -> None:
    """The probe is the one command that moves an arm without being given a task, and it has
    to be. It used to connect, ask its questions and disconnect, and a LeRobot arm goes limp
    the moment it is disconnected, which is how the bench arm fell at the end of a check.

    The factory is left alone here on purpose: a pose handed to `collect` reaching the arm at
    all is half of what this pins, and "returned to it" is only possible if it arrived."""
    report = doctor.collect("lerobot:mock", address="mock://arm", rest_pose=AWAY_FROM_REST)
    row = _row(report, "rest pose")
    assert (row.value, row.state) == ("returned to it", "ok")
    assert report.ok is True, "parking the arm is how a probe ends, not a fault to report"
    advisories = _probe_of(report).advisories
    assert not any("torque was left on" in a for a in advisories), advisories


def test_an_arm_already_at_its_rest_pose_says_so_rather_than_driving_it_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing on this arm reports whether a goal was reached, so the rest move compares the
    goal with the measured position. An arm that is already parked is left alone: re-sending
    the pose would push servos at a goal they are already holding, for a row that would read
    the same either way."""
    arm = LeRobotMock(rest_pose=dict(REST))
    report = _probed(monkeypatch, arm, rest_pose=dict(REST))
    row = _row(report, "rest pose")
    assert (row.value, row.state) == ("at it already", "ok")
    assert arm.actions == [], f"an arm already at rest was sent {arm.actions}"
    assert arm.sequence == ["rest", "close"], arm.sequence
    assert arm.torque is False, "an arm at its rest pose can be let go of"
    assert report.ok is True


def test_a_robot_with_no_rest_pose_recorded_says_how_to_record_one_and_still_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not recording a pose is a choice, the way an extra nobody installed is a choice, and a
    robot that has always been let go of where it stood is not suddenly broken. Torque still
    drops, which is the behaviour of every body here except an arm with a pose to hold."""
    arm = LeRobotMock()
    report = _probed(monkeypatch, arm)
    row = _row(report, "rest pose")
    assert row.value == "none recorded (quackd robot rest-pose <name>)"
    assert row.state == "plain", "a pose nobody recorded is not a failure"
    assert report.ok is True
    assert arm.torque is False, "with no pose to hold, the arm is released as it always was"
    assert arm.close_note is None


def test_an_arm_that_cannot_reach_its_rest_pose_fails_the_verdict_and_says_torque_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An arm that stalled on the way to its rest pose is the one case where the disconnect
    must not drop torque, because dropping it is how the arm falls. That leaves the arm
    holding itself up with nothing on the machine saying so, so doctor fails the verdict and
    repeats the note the close left behind."""
    arm = LeRobotMock(rest_pose=AWAY_FROM_REST, rest_fails="elbow_flex is at 12 with a goal of 90")
    report = _probed(monkeypatch, arm, rest_pose=AWAY_FROM_REST)
    row = _row(report, "rest pose")
    assert row.value == "not reached: elbow_flex is at 12 with a goal of 90"
    assert row.state == "fail"
    assert _probe_of(report).ok is False
    assert report.ok is False, "an arm left holding itself up is not a machine in a good state"
    advisories = _probe_of(report).advisories
    assert any("torque was left on" in a for a in advisories), advisories
    assert arm.torque is True and arm.close_note is not None
    buf = io.StringIO()
    Console(file=buf, width=200).print(doctor.verdict(report))
    said = " ".join(buf.getvalue().split())
    assert "FAILURE" in said and "rest pose: not reached" in said, said


class _TwoEyes(LeRobotMock):
    """An arm with two cameras, answering `camera_health()` the shape the real backend does.

    Each row leaves its size out: `probe` fills that in from the frames it actually read, so
    a camera can only claim a size by having handed a picture over."""

    camera_keys = ("top", "side")

    def __init__(self, *, blind: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.blind = blind

    async def get_frames(self) -> list[CameraFrame]:
        image = await self.get_frame()
        if image is None:
            return []
        live = [name for name in self.camera_keys if name not in self.blind]
        # the first camera is the primary, and it is marked rather than assumed: a body whose
        # primary lens is the blind one must not promote the other into its place
        return [CameraFrame(n, image, primary=n == self.camera_keys[0]) for n in live]

    def camera_health(self) -> dict[str, Any]:
        return {
            "configured": True,
            "url": "opencv://0?name=top",
            "ok": "top" not in self.blind,
            "age_s": 0.0,
            "size": None,
            "error": None,
            "cameras": [
                {
                    "name": name,
                    "url": f"opencv://{index}?name={name}",
                    "ok": name not in self.blind,
                    "age_s": None if name in self.blind else 0.0,
                    "size": None,
                    "error": "TimeoutError: no frame in 2.0 s" if name in self.blind else None,
                }
                for index, name in enumerate(self.camera_keys)
            ],
        }


class _OneEye(LeRobotMock):
    """One camera, answering exactly the dict every backend answered before there were two."""

    def camera_health(self) -> dict[str, Any]:
        return {
            "configured": True,
            "url": "opencv://0",
            "ok": True,
            "age_s": 0.0,
            "size": "128x128",
            "error": None,
        }


def test_the_probe_gives_each_camera_a_row_and_names_the_one_that_gave_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A camera URL nothing checks is a camera URL that fails at the first observe, which is
    why the probe asks for a picture rather than accepting the url. With two of them, "no
    frame came back" does not say which eye closed, and the arm goes on working with the
    other one, so both the row and the advisory have to name it."""
    arm = _TwoEyes(blind=("side",))
    report = _probed(monkeypatch, arm, camera_url="opencv://0?name=top")
    top, side = _row(report, "camera top"), _row(report, "camera side")
    assert (top.value, top.state) == ("128x128", "ok")
    assert (side.value, side.state) == ("no frame", "fail")
    assert _probe_of(report).ok is False
    assert report.ok is False, "a camera that sent nothing fails the machine's verdict"
    advisories = _probe_of(report).advisories
    assert any("from side" in a and "observe" in a for a in advisories), advisories


def test_a_single_camera_still_reads_under_the_row_label_it_always_had(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A camera's name is only spoken by a body that has more than one. Labelling the row
    "camera camera" on every one-camera robot would rename the row everybody reading this
    output already knows, in exchange for a word that is quackd's own default and not
    anything its owner chose."""
    report = _probed(monkeypatch, _OneEye(), camera_url="opencv://0")
    row = _row(report, "camera")
    assert (row.value, row.state) == ("128x128", "ok")
    labels = [r.what for r in _probe_of(report).rows if r.what.startswith("camera")]
    assert labels == ["camera"], labels
    assert report.ok is True
