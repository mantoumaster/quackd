"""`quackd doctor`: it runs everywhere, it answers in data, and the WebSocket stub refuses
honestly.

The split between `collect` and `render` is the thing worth testing. The collector must not
know what green means, or `--json` has nothing underneath it to print, and the renderer must
not know anything the report does not hold.
"""

from __future__ import annotations

import io
import json
import platform
import sys
import time
from pathlib import Path
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
    assert "[ok] in the package" in out, "the registry's tick becomes something readable"
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


# ── the board underneath, when somebody runs this on a Jetson ───────────────────────────
#
# Every one of these builds a board out of files in a tmp_path, because the thing being tested
# is the reading and not the hardware. The one fact no fixture can supply is whether a real
# Orin's files look like these; `docs/jetson.md` says so and says what to send back.

NUL = chr(0)
TAB = chr(9)

_REAL_RUN_QUIET = doctor._run_quiet
"""Captured before `_no_board` replaces it. Two tests below are about that function rather
than about a board, and without this they would assert against the stub and pass whatever it
did."""

ORIN_NANO = "NVIDIA Jetson Orin Nano Developer Kit"
COMPATIBLE = NUL.join(("nvidia,p3768-0000+p3767-0005", "nvidia,p3767-0005", "nvidia,tegra234"))
RELEASE_36_4_3 = (
    "# R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, EABI: aarch64, "
    "DATE: Wed Jan  8 01:51:37 UTC 2025"
)
MEMINFO = "MemTotal:        7650336 kB\nMemAvailable:    5123456 kB\nSwapTotal:       1017852 kB\n"
NL = chr(10)


def _swap_line(*fields: str) -> str:
    return TAB.join(fields) + NL


SWAPS_HEADER = _swap_line("Filename", "", "", "", "Type", "", "Size", "", "Used", "", "Priority")
ZRAM_SWAPS = SWAPS_HEADER + _swap_line("/dev/zram0", "partition", "1017852", "0", "5")
NVME_SWAPS = SWAPS_HEADER + _swap_line("/ssd/16GB.swap", "file", "16777212", "0", "-2")
NO_SWAPS = SWAPS_HEADER


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _tegra_tree(
    root: Path,
    *,
    release: str | None = RELEASE_36_4_3,
    swaps: str = ZRAM_SWAPS,
    gpu_node: bool = True,
    device_tree: bool = True,
) -> Path:
    """A board, as files. The NUL terminators are real: `/proc/device-tree/*` are the device
    tree's own bytes, and the first version of this reader put one in a Rich cell."""
    if device_tree:
        _write(root, "proc/device-tree/model", ORIN_NANO + NUL)
        _write(root, "proc/device-tree/compatible", COMPATIBLE + NUL)
    if release is not None:
        _write(root, "etc/nv_tegra_release", release)
    _write(root, "proc/meminfo", MEMINFO)
    _write(root, "proc/swaps", swaps)
    if gpu_node:
        (root / "dev/nvgpu/igpu0").mkdir(parents=True)
    return root


@pytest.fixture(autouse=True)
def _no_board(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No test in this file may depend on what the machine running it is.

    Without this the suite reads a different report on a Jetson than on a laptop, and the one
    place that would show up is somebody else's machine."""
    monkeypatch.setattr(doctor, "_HOST_ROOT", tmp_path / "not-a-board")
    monkeypatch.setattr(doctor, "_run_quiet", lambda argv, timeout_s=3.0: None)


@pytest.fixture
def fake_tegra(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = _tegra_tree(tmp_path / "board")
    monkeypatch.setattr(doctor, "_HOST_ROOT", root)
    return root


def test_a_machine_that_is_not_a_tegra_says_nothing_about_one() -> None:
    report = doctor.collect()
    assert report.jetson is None
    assert report.to_dict()["jetson"] is None
    buf = io.StringIO()
    doctor.render(Console(file=buf, width=200), report)
    assert "Jetson" not in buf.getvalue()


def test_a_tegra_tree_is_read_field_by_field(tmp_path: Path) -> None:
    got = doctor._jetson(_tegra_tree(tmp_path))
    assert got is not None
    assert got.board == ORIN_NANO, "the NUL terminator is stripped, not carried into a cell"
    assert (got.l4t, got.jetpack) == ("36.4.3", "6.2")
    assert got.mem_total_bytes == 7650336 * 1024
    assert got.mem_available_bytes == 5123456 * 1024
    assert got.swap_total_bytes == 1017852 * 1024
    assert got.swap_devices == ["/dev/zram0"] and got.swap_only_zram is True
    assert got.gpu_device == "/dev/nvgpu/igpu0"
    assert got.power_mode is None and got.docker_default_runtime is None


def test_a_swapfile_on_the_ssd_is_not_zram(tmp_path: Path) -> None:
    """The warning is about zram specifically, and a board that was set up properly must not
    read as one that was not."""
    got = doctor._jetson(_tegra_tree(tmp_path, swaps=NVME_SWAPS))
    assert got is not None and got.swap_only_zram is False
    assert got.swap_devices == ["/ssd/16GB.swap"]
    none = doctor._jetson(_tegra_tree(tmp_path / "b", swaps=NO_SWAPS))
    assert none is not None and none.swap_total_bytes == 0 and none.swap_only_zram is False


def test_a_release_file_it_cannot_parse_is_still_a_tegra(tmp_path: Path) -> None:
    got = doctor._jetson(_tegra_tree(tmp_path, release="something nobody has seen"))
    assert got is not None, "the device tree still said Tegra"
    assert got.l4t is None and got.jetpack is None


def test_the_two_ways_l4t_can_be_unknown_read_differently(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A file that is not there and a file it cannot read are different facts about the

    board, and the renderer used to assert the first for both. One of them means a
    container; the other means a board this build has not met."""
    missing = doctor._jetson(_tegra_tree(tmp_path / "a", release=None))
    unreadable = doctor._jetson(_tegra_tree(tmp_path / "b", release="not a release line"))
    assert missing is not None and unreadable is not None
    assert missing.release_seen is False and unreadable.release_seen is True
    assert missing.l4t is None and unreadable.l4t is None

    monkeypatch.setattr(doctor, "_HOST_ROOT", tmp_path / "b")
    buf = io.StringIO()
    doctor.render(Console(file=buf, width=200), doctor.collect())
    assert "is here and its first line is not one this build knows" in buf.getvalue()


def test_a_device_tree_with_no_release_file_is_a_board_with_no_l4t(tmp_path: Path) -> None:
    """A privileged container on a Jetson, or one started with `--security-opt
    systempaths=unconfined`: the device tree is readable, because nothing masks `/sys/firmware`,
    and `/etc/nv_tegra_release` is a file in the host's root filesystem that a plain Python image
    has none of. Reporting the image's userspace as the board's would be the one wrong answer
    available here. An ordinary container sees neither and gets no section."""
    got = doctor._jetson(_tegra_tree(tmp_path, release=None))
    assert got is not None and got.board == ORIN_NANO
    assert got.l4t is None


def test_only_a_tegra_answers_at_all(tmp_path: Path) -> None:
    (tmp_path / "proc").mkdir()
    (tmp_path / "proc" / "device-tree").mkdir()
    _write(tmp_path, "proc/device-tree/compatible", "raspberrypi,4-model-b" + NUL + "brcm,bcm2711")
    assert doctor._jetson(tmp_path) is None


@pytest.mark.parametrize(
    ("l4t", "jetpack"),
    [
        ("36.4.3", "6.2"),
        ("36.4.4", "6.2.1"),
        ("36.5.0", "6.2.2"),
        ("36.4", "6.1"),
        ("36.3.0", "6.0"),
        ("39.2.1", "7.2.1"),
        ("35.4.1", "5.x"),
        ("36.9.9", "6.x"),
        ("99.1", None),
    ],
)
def test_the_jetpack_table_names_exact_releases_and_the_major_for_the_rest(
    l4t: str, jetpack: str | None
) -> None:
    """`36.4` and `36.4.0` are the same release written two ways, and NVIDIA writes both.

    The major-only fallback is deliberate: L4T 35.1 was JetPack 5.0.2 and 35.2.1 was 5.1, so a
    guessed minor would be wrong about a board somebody owns."""
    assert doctor._jetpack_for(l4t) == jetpack


def test_the_quiet_runner_answers_none_for_anything_that_did_not_work() -> None:
    """It is the first subprocess in this file, and doctor is what people run when something is
    already wrong: a probe that raises there is worse than a probe that says nothing."""
    assert _REAL_RUN_QUIET(["quackd-no-such-binary-anywhere", "-q"]) is None
    assert _REAL_RUN_QUIET([sys.executable, "-c", "raise SystemExit(3)"]) is None
    assert _REAL_RUN_QUIET([sys.executable, "-c", "print('hello')"]).strip() == "hello"


def test_the_quiet_runner_gives_up_rather_than_hanging() -> None:
    """Against a 30 second sleep, so returning at all is the assertion.

    The bound is generous rather than tight because this machine may be building a
    container next to the suite, and a flaky timing test is worse than a loose one. The
    second case is the only thing pinning the 3.0 second default the code actually ships."""
    started = time.monotonic()
    assert _REAL_RUN_QUIET([sys.executable, "-c", "import time; time.sleep(30)"], 1.0) is None
    assert time.monotonic() - started < 10, "the timeout did not fire"

    started = time.monotonic()
    assert _REAL_RUN_QUIET([sys.executable, "-c", "import time; time.sleep(30)"]) is None
    assert time.monotonic() - started < 15, "the default timeout is not the 3.0 s claimed"


def test_a_power_mode_line_with_no_colon_is_none_rather_than_a_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`split(":", 1)[1]` raised IndexError here, on a machine already having a bad day."""
    monkeypatch.setattr(doctor, "_run_quiet", lambda argv, timeout_s=3.0: "NV Power Mode" + NL)
    got = doctor._jetson(_tegra_tree(tmp_path))
    assert got is not None and got.power_mode is None


def test_the_subprocess_answers_are_parsed_and_an_absent_binary_is_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    answers = {
        "nvpmodel": "NV Fan Mode:quiet\nNV Power Mode: MAXN_SUPER\n2\n",
        "docker": "nvidia\n",
    }
    monkeypatch.setattr(doctor, "_run_quiet", lambda argv, timeout_s=3.0: answers.get(argv[0]))
    got = doctor._jetson(_tegra_tree(tmp_path))
    assert got is not None
    assert got.power_mode == "MAXN_SUPER"
    assert got.docker_default_runtime == "nvidia"


def test_the_jetson_section_warns_about_zram_and_a_runtime_that_is_not_nvidia(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(doctor, "_HOST_ROOT", _tegra_tree(tmp_path, gpu_node=False))
    monkeypatch.setattr(doctor, "_run_quiet", lambda argv, timeout_s=3.0: "runc\n")
    buf = io.StringIO()
    doctor.render(Console(file=buf, width=200), doctor.collect())
    out = buf.getvalue()
    for needle in (
        "Jetson",
        ORIN_NANO,
        "36.4.3 (JetPack 6.2)",
        "shared with the GPU",
        "all zram",
        "quackd never asks for one",
        "--runtime nvidia",
    ):
        assert needle in out, needle


def test_a_board_where_docker_does_not_answer_says_so_rather_than_dropping_the_row(
    fake_tegra: Path,
) -> None:
    """The row people are sent here to read, on a board where `docker info` gives no answer.

    That is a board on the page's native route, which installs no Docker, or a user outside the
    docker group, and the renderer used to drop the row entirely. Three documents tell a reader
    to check this setting with this command, so silence there is the one unacceptable answer."""
    buf = io.StringIO()
    doctor.render(Console(file=buf, width=200), doctor.collect())
    out = buf.getvalue()
    assert "docker default runtime" in out
    assert "docker did not answer here, because it is not installed" in out


def test_the_jetson_section_survives_a_codepage_that_cannot_carry_it(
    monkeypatch: pytest.MonkeyPatch, fake_tegra: Path
) -> None:
    """The same bar the rest of this command clears, on the section most likely to be pasted
    into an issue by somebody whose terminal is not UTF-8."""
    monkeypatch.setattr(sys, "platform", "linux")
    raw = io.BytesIO()
    console = Console(file=io.TextIOWrapper(raw, encoding="cp1252", errors="strict"), width=120)
    doctor.render(console, doctor.collect())
    console.file.flush()
    out = raw.getvalue().decode("cp1252")
    assert out.isascii()
    assert "Jetson" in out and ORIN_NANO in out


def test_doctor_json_carries_the_board(fake_tegra: Path) -> None:
    result = CliRunner().invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0, result.output
    jetson = json.loads(result.output)["jetson"]
    assert jetson["board"] == ORIN_NANO
    assert jetson["l4t"] == "36.4.3" and jetson["jetpack"] == "6.2"
    assert jetson["swap_only_zram"] is True


def test_nothing_a_jetson_reports_can_change_the_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The whole section is informational. quackd is CPU Python here exactly as it is on a
    laptop, so a board with no swap, no GPU node and the wrong docker runtime is a board this
    command has advice for, not a machine that cannot run anything."""
    monkeypatch.setattr(doctor, "_HOST_ROOT", _tegra_tree(tmp_path, swaps=NO_SWAPS, gpu_node=False))
    monkeypatch.setattr(doctor, "_run_quiet", lambda argv, timeout_s=3.0: "runc\n")
    report = doctor.collect()
    assert report.jetson is not None
    assert report.ok is True
    buf = io.StringIO()
    Console(file=buf, width=200).print(doctor.verdict(report))
    assert "SUCCESS" in buf.getvalue()


def test_the_header_names_the_architecture() -> None:
    """`Linux 5.15.148-tegra` and `Linux 5.15.148-generic` are two different machines, and the
    difference that matters for a wheel is the one the header did not carry."""
    report = doctor.collect()
    assert report.platform.split()[-1] == platform.machine()
    buf = io.StringIO()
    doctor.render(Console(file=buf, width=200), report)
    assert platform.machine() in buf.getvalue()


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


def test_an_arm_that_arrived_and_was_still_held_at_the_close_fails_the_verdict_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gap between the two reports. The rest move says it arrived, and the disconnect's
    own re-read is a separate reading that can disagree with it: the arm drifted, or it
    stopped answering. When it does, torque is kept and a note is written, and reading only
    the rest move would print that note under a green tick and exit 0.

    A person runs `doctor` to be told whether they can walk away. Saying yes over a note that
    says the arm is still powered is the one answer this command must never give."""
    arm = LeRobotMock(rest_pose=dict(REST))

    async def arrive_then_drift() -> None:
        arm.sequence.append("close")
        arm.close_note = "the arm is not at its rest pose (it stopped answering), so torque was left on and it will not fall: hold the arm and cut its power, or run again"  # noqa: E501

    monkeypatch.setattr(arm, "close", arrive_then_drift)
    report = _probed(monkeypatch, arm, rest_pose=dict(REST))

    assert _row(report, "rest pose").state == "ok", "the move itself did report arriving"
    advisories = _probe_of(report).advisories
    assert any("torque was left on" in a for a in advisories), advisories
    assert report.ok is False, "a green verdict over a torque note walks somebody away"


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
