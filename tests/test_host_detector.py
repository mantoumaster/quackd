"""The board's YOLO as a `Detector`: `quackd.perception.host.HostDetector`.

Driven against the fake daemon in `tests/fake_jetson_hostd.py` over real HTTP on loopback, and
beside `YoloDetector` with a fake `ultralytics` standing in for the model, so the claim that the
same boxes are the same detections on either machine is a comparison and not a recollection.
None of it says anything about a Jetson, which no test here has seen.
"""

from __future__ import annotations

import asyncio
import io
import logging
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from quackd import host as host_module
from quackd.adapters.factory import make_adapter
from quackd.agent.loop import RunConfig, run_duck
from quackd.agent.providers.base import ToolCall
from quackd.agent.providers.fake import FakeProvider
from quackd.agent.transcript import Transcript
from quackd.duckfile.parser import parse_duck_text
from quackd.host import HostClient
from quackd.perception import detector_for
from quackd.perception.base import Detection, detect_off_loop
from quackd.perception.host import JPEG_QUALITY, HostDetector
from quackd.perception.yolo import YoloDetector, detections_from_boxes
from quackd.transport.base import Ack, Intent
from quackd.verbs.core import HOLD_TTL_S, _see_holding
from quackd.verbs.registry import VerbContext
from tests.fake_jetson_hostd import BALL_BOX, FRAME_SIZE, FakeHostd, dead_address, jpeg_bytes

Box = tuple[str, float, float, float, float, float]


@pytest.fixture
def hostd() -> Iterator[FakeHostd]:
    with FakeHostd() as fake:
        yield fake


def _frame() -> Image.Image:
    """The fake daemon's own default frame, the grey floor with the orange ball."""
    with Image.open(io.BytesIO(jpeg_bytes())) as picture:
        return picture.convert("RGB")


def _boxes() -> list[Box]:
    """Every name quackd maps and some it does not, at fractional corners across the frame, so a
    reordered sum or a lost float would show in the last bit of a bearing."""
    names = ["sports ball", "person", "cat", "dog", "potted plant", "car"]
    boxes: list[Box] = []
    for i in range(24):
        x1 = (i * 37.13) % 560.0
        y1 = (i * 11.71) % 400.0
        boxes.append(
            (
                names[i % len(names)],
                round(0.3 + (i * 0.137) % 0.65, 4),
                x1,
                y1,
                x1 + 5.3 + (i * 7.07) % 70.0,
                y1 + 2.1 + (i * 5.31) % 70.0,
            )
        )
    return boxes


def _reply(boxes: list[Box], size: tuple[int, int] = FRAME_SIZE) -> dict[str, Any]:
    return {
        "ok": True,
        "w": size[0],
        "h": size[1],
        "boxes": [
            {"name": n, "conf": c, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
            for n, c, x1, y1, x2, y2 in boxes
        ],
        "model": "yolov8n.pt",
        "device": "cuda",
        "ms": 21.0,
    }


class _FakeYOLO:
    """`ultralytics.YOLO`, reduced to what `YoloDetector` reads: the model's own class names and
    boxes with `cls`, `conf` and `xyxy`."""

    boxes: list[Box] = []

    def __init__(self, model: str) -> None:
        self.model = model

    def predict(self, image: Image.Image, *, conf: float, verbose: bool) -> list[Any]:
        names = sorted({name for name, *_ in _FakeYOLO.boxes})
        cls_of = {name: i for i, name in enumerate(names)}
        return [
            SimpleNamespace(
                names=dict(enumerate(names)),
                boxes=[
                    SimpleNamespace(cls=cls_of[n], conf=c, xyxy=[(x1, y1, x2, y2)])
                    for n, c, x1, y1, x2, y2 in _FakeYOLO.boxes
                ],
            )
        ]


@pytest.fixture
def fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> type[_FakeYOLO]:
    module = ModuleType("ultralytics")
    module.YOLO = _FakeYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    monkeypatch.setattr(_FakeYOLO, "boxes", [])
    return _FakeYOLO


# ── the same boxes are the same detections ──────────────────────────────────────────────


def test_the_host_and_yolo_detectors_agree_on_the_same_boxes(
    hostd: FakeHostd, fake_ultralytics: type[_FakeYOLO]
) -> None:
    """One model on the laptop, one on the board, the same boxes out of each: the detections
    must be equal to the last bit, because both go through `detections_from_boxes` and nothing
    else. Calibrated and not, since `calibrated` rides on every detection."""
    boxes = _boxes()
    fake_ultralytics.boxes = boxes
    hostd.detect_reply = _reply(boxes)
    frame = _frame()
    for calibrated in (True, False):
        local = YoloDetector(fov_deg=62.2, calibrated=calibrated).detect(frame)
        board = HostDetector(HostClient(hostd.address), fov_deg=62.2, calibrated=calibrated)
        remote = board.detect(frame)
        assert len(remote) == len(local) > 10
        assert [d.model_dump() for d in remote] == [d.model_dump() for d in local]
        assert remote == detections_from_boxes(
            boxes, *FRAME_SIZE, fov_deg=62.2, calibrated=calibrated
        )
        assert board.error is None


def test_the_frame_goes_to_the_board_as_one_jpeg_of_the_frame_itself(hostd: FakeHostd) -> None:
    """A JPEG, at the quality this module names, the size the camera took it at, and the
    daemon's own confidence floor unless one was asked for."""
    HostDetector(HostClient(hostd.address), fov_deg=62.2).detect(_frame())
    HostDetector(HostClient(hostd.address), fov_deg=62.2, conf=0.25).detect(_frame())
    first, second = hostd.requests_to("/detect")
    assert first.method == "POST" and first.headers["content-type"] == "image/jpeg"
    assert first.body.startswith(b"\xff\xd8"), "a JPEG's first two bytes"
    with Image.open(io.BytesIO(first.body)) as sent:
        assert sent.format == "JPEG" and sent.size == FRAME_SIZE
    assert first.query == "" and second.query == "conf=0.25"
    assert JPEG_QUALITY == 85


def test_names_the_board_has_no_word_for_are_dropped(hostd: FakeHostd) -> None:
    """The daemon maps nothing and sends the model's own names; quackd keeps the ones its verbs
    have a word for. A dog is a pet, a potted plant and a car are nothing the verbs can steer
    at, and the one ball is where the fake frame drew it."""
    x1, y1, x2, y2 = BALL_BOX
    hostd.detect_reply = _reply(
        [
            ("potted plant", 0.9, 10.0, 10.0, 60.0, 90.0),
            ("sports ball", 0.87, x1, y1, x2, y2),
            ("car", 0.8, 400.0, 100.0, 600.0, 300.0),
            ("dog", 0.7, 100.0, 200.0, 220.0, 320.0),
        ]
    )
    found = HostDetector(HostClient(hostd.address), fov_deg=62.2).detect(_frame())
    assert [d.label for d in found] == ["ball", "pet"]
    ball = found[0]
    assert ball.bearing_deg == 0.0, "the ball's box is centred in the frame"
    assert ball.calibrated is True


# ── a board that stops answering ────────────────────────────────────────────────────────


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "quackd.perception" and r.levelno >= logging.WARNING
    ]


def test_a_dead_daemon_yields_nothing_and_warns_once_then_recovers(
    hostd: FakeHostd, caplog: pytest.LogCaptureFixture
) -> None:
    """Three frames into an outage are three empty answers and one warning, because `go_to`
    asks ten times a second and a warning per frame buries the sentence. The next answer ends
    the outage and says so, and a second outage is warned about again."""
    caplog.set_level(logging.INFO, logger="quackd.perception")
    detector = HostDetector(HostClient(hostd.address), fov_deg=62.2)
    hostd.detect_status = 503
    hostd.detect_reply = {"ok": False, "reason": "detection is not available: CUDA out of memory"}
    for _ in range(3):
        assert detector.detect(_frame()) == []
    assert detector.error is not None and "CUDA out of memory" in detector.error
    down = _warnings(caplog)
    assert len(down) == 1, down
    assert hostd.address in down[0] and "keeps this detector" in down[0]

    hostd.detect_status = 200
    hostd.detect_reply = _reply([("sports ball", 0.9, *BALL_BOX)])
    (ball,) = detector.detect(_frame())
    assert ball.label == "ball" and detector.error is None
    assert _warnings(caplog)[-1].endswith("answers again")

    hostd.detect_status = 500
    hostd.detect_reply = {"ok": False, "reason": "the model raised: RuntimeError"}
    assert detector.detect(_frame()) == [] and detector.detect(_frame()) == []
    assert len(_warnings(caplog)) == 3, "one for the second outage, and only one"


def test_a_daemon_that_is_not_there_at_all_is_the_same_empty_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing listening is a connection refused rather than a 503, and costs one bounded
    wait. The wait is shortened here, because Windows retries a refused loopback connect for
    about the whole timeout."""
    monkeypatch.setattr(host_module, "DETECT_TIMEOUT_S", 0.3)
    detector = HostDetector(HostClient(dead_address()), fov_deg=62.2)
    assert detector.detect(_frame()) == []
    assert detector.error is not None


def test_an_answer_for_an_image_of_another_size_is_no_detections(
    hostd: FakeHostd, caplog: pytest.LogCaptureFixture
) -> None:
    """The boxes are pixels in the image the daemon decoded, which is the frame it was sent. A
    reply naming another size describes some other image. Taken at its word, this one's ball,
    at pixels a 320x240 image does not have, became a bearing of 58 degrees, outside a 62
    degree lens altogether, and `go_to` would have steered at it. It is a failed call instead,
    with the reason, the same one warning as an outage, and the next honest answer ends it."""
    caplog.set_level(logging.INFO, logger="quackd.perception")
    hostd.detect_reply = _reply([("sports ball", 0.9, 560.0, 400.0, 600.0, 440.0)], (320, 240))
    detector = HostDetector(HostClient(hostd.address), fov_deg=62.2)

    assert detector.detect(_frame()) == []
    assert detector.error == (
        f"{hostd.address} answered for a 320x240 image, not the 640x480 frame it was sent"
    )
    assert detector.detect(_frame()) == []
    (told,) = _warnings(caplog)
    assert "keeps this detector" in told and "320x240" in told

    hostd.detect_reply = _reply([("sports ball", 0.9, *BALL_BOX)])
    (ball,) = detector.detect(_frame())
    assert ball.bearing_deg == 0.0 and detector.error is None


# ── off the event loop ──────────────────────────────────────────────────────────────────


class _Probe:
    """A detector that records which threads ran it."""

    name = "probe"

    def __init__(self, *, blocking: bool) -> None:
        self.blocking = blocking
        self.threads: list[int] = []

    def detect(self, image: Image.Image) -> list[Detection]:
        self.threads.append(threading.get_ident())
        return []


async def test_only_a_detector_that_blocks_is_run_off_the_event_loop() -> None:
    """The board's detector says it blocks, and runs in a worker thread. Every other detector
    runs on the loop as it always has, so a run without a board is scheduled as it was."""
    loop_thread = threading.get_ident()
    waits, local = _Probe(blocking=True), _Probe(blocking=False)
    await detect_off_loop(waits, _frame())
    await detect_off_loop(local, _frame())
    assert waits.threads and loop_thread not in waits.threads
    assert local.threads == [loop_thread]
    assert HostDetector.blocking is True


async def test_the_loop_and_observe_run_a_blocking_detector_off_the_event_loop(
    tmp_path: Path,
) -> None:
    """Every place a run hands a frame to its detector: the observation the loop makes each
    step, and the `observe` verb. `go_to` and `search_scan` go through `_see`, above."""
    loop_thread = threading.get_ident()
    probe = _Probe(blocking=True)
    duck = parse_duck_text(
        "---\nduck: 0\nname: look\ndescription: d\nverbs:\n"
        "  allow: [observe, stop]\nsuccess: [x]\n---\n# Task\nLook.\n"
    )
    result = await run_duck(
        RunConfig(
            duck=duck,
            provider=FakeProvider(
                script=[
                    ToolCall(name="observe", arguments={}),
                    ToolCall(name="declare_success", arguments={"reason": "looked"}),
                ]
            ),
            transport=make_adapter("microduck:mock", seed=0),
            detector=probe,  # type: ignore[arg-type]
            runs_dir=tmp_path,
        )
    )
    assert result.outcome == "success", result.reason
    assert len(probe.threads) >= 3, "the loop's observations and the verb's"
    assert loop_thread not in probe.threads


class _Walking:
    """A body that is walking: one frame to look at, and every intent sent to it timed on the
    event loop's clock, which is the clock `_see_holding` holds its twist by."""

    def __init__(self) -> None:
        self.moves: list[float] = []

    async def get_frame(self) -> Image.Image:
        return _frame()

    async def send_intent(self, intent: Intent) -> Ack:
        if intent.kind == "move":
            self.moves.append(self.now())
        return Ack()

    def now(self) -> float:
        return asyncio.get_running_loop().time()


async def test_a_slow_board_does_not_starve_the_twist_go_to_holds_while_it_looks(
    hostd: FakeHostd,
) -> None:
    """`go_to` looks while it walks, and `_see_holding` re-sends the last twist every
    `MOVE_RESEND_S` for one deadman window while the frame is looked at, so a slow frame does
    not stop the body. The board's detector used to run on the event loop, and a board that
    took 0.6 s held the loop for all of it: no twist went out, and a body with a 300 ms
    deadman stopped on every frame. In a worker thread the resends go out while the board is
    still answering, and the detections still arrive."""
    hostd.delays["/detect"] = 0.6
    body = _Walking()
    ctx = VerbContext(
        transport=body, detector=HostDetector(HostClient(hostd.address), fov_deg=62.2)
    )
    started = time.monotonic()
    _img, hits = await _see_holding(ctx, "ball", "go_to", (0.1, 0.0, 0.0))
    assert time.monotonic() - started >= 0.5, "the board really was slow"
    assert [hit.label for hit in hits] == ["ball"]
    assert len(body.moves) >= 2, body.moves
    assert body.moves[-1] - body.moves[0] < HOLD_TTL_S, "held for one deadman window, no more"


async def test_it_never_swaps_itself_for_the_colour_detector(
    hostd: FakeHostd, tmp_path: Path
) -> None:
    """A board that fails mid-run leaves the run with no detections and the same detector:
    `detector_for` keeps it, the loop keeps it, `observe` says why nothing was seen, and the
    record has one note for the outage rather than one per frame.

    The body is the Microduck's mock, which has a camera of its own; the detector is the
    board's, asked for by name, which is the one way it runs on a simulator."""
    hostd.detect_status = 503
    hostd.detect_reply = {"ok": False, "reason": "detection is not available: CUDA out of memory"}
    detector = HostDetector(HostClient(hostd.address), fov_deg=62.2)
    assert detector_for(["camera"], detector, fov_deg=90.0, backend="mock") is detector

    duck = parse_duck_text(
        "---\nduck: 0\nname: look-twice\ndescription: d\nverbs:\n"
        "  allow: [observe, report_state, stop]\nsuccess: [x]\n---\n# Task\nLook.\n"
    )
    cfg = RunConfig(
        duck=duck,
        provider=FakeProvider(
            script=[
                ToolCall(name="observe", arguments={}),
                ToolCall(name="observe", arguments={}),
                ToolCall(name="declare_success", arguments={"reason": "looked"}),
            ]
        ),
        transport=make_adapter("microduck:mock", seed=0),
        detector=detector,
        runs_dir=tmp_path,
    )
    result = await run_duck(cfg)
    assert result.outcome == "success", result.reason
    assert cfg.detector is detector, "the run ends with the detector it started with"

    events = Transcript.read(result.run_dir / "transcript.jsonl")
    start = next(e for e in events if e["kind"] == "run_start")
    assert start["detector"] == "yolo@host"
    notes = [e["text"] for e in events if e["kind"] == "note" and "yolo@host" in e["text"]]
    assert len(notes) == 1, notes
    assert "keeps it rather than switching" in notes[0] and "CUDA out of memory" in notes[0]
    looked = [e for e in events if e["kind"] == "verb_end" and e.get("name") == "observe"]
    assert looked, [e["kind"] for e in events]
    for verb in looked:
        assert "nothing detected (host detector:" in str(verb.get("summary")), verb


class _Flaky:
    """The board's detector on a flaky link, one reason per frame from `reasons` and then
    answering, with the per-request figures a daemon's reasons carry."""

    name = "yolo@host"
    blocking = False

    def __init__(self, reasons: list[str | None]) -> None:
        self.reasons = list(reasons)
        self.error: str | None = None

    def detect(self, image: Image.Image) -> list[Detection]:
        self.error = self.reasons.pop(0) if self.reasons else None
        return []


async def test_a_flaky_link_is_a_note_per_outage_and_per_kind_of_reason_not_per_frame(
    tmp_path: Path,
) -> None:
    """A daemon's reasons carry numbers that change from one request to the next, such as a
    truncated upload's byte count. Compared as raw text, each frame's reason was a new one and
    each got its own note, which is the per-frame noise one note per outage exists to prevent.
    Ten frames here are two outages, the first with two kinds of reason: five notes, where
    comparing the text wrote ten."""
    truncated = "the body ended after {} of 30416 bytes"
    slow = "did not answer /detect within {}s"
    reasons: list[str | None] = [
        truncated.format(812),
        truncated.format(1024),
        truncated.format(2048),
        slow.format(2),
        truncated.format(4096),
        slow.format(3),
        None,
        truncated.format(17),
        truncated.format(90),
        None,
    ]
    duck = parse_duck_text(
        "---\nduck: 0\nname: wait\ndescription: d\nverbs:\n"
        "  allow: [report_state, stop]\nsuccess: [x]\n---\n# Task\nWait.\n"
    )
    script = [ToolCall(name="report_state", arguments={}) for _ in range(len(reasons) + 1)]
    result = await run_duck(
        RunConfig(
            duck=duck,
            provider=FakeProvider(
                script=[*script, ToolCall(name="declare_success", arguments={"reason": "done"})]
            ),
            transport=make_adapter("microduck:mock", seed=0),
            detector=_Flaky(reasons),
            runs_dir=tmp_path,
        )
    )
    assert result.outcome == "success", result.reason
    events = Transcript.read(result.run_dir / "transcript.jsonl")
    notes = [e["text"] for e in events if e["kind"] == "note" and "yolo@host" in e["text"]]
    assert len(notes) == 5, notes
    assert "keeps it rather than switching: " + truncated.format(812) in notes[0]
    assert notes[1] == "yolo@host still fails, now: " + slow.format(2)
    assert notes[2] == "yolo@host answers again"
    assert notes[3].endswith(truncated.format(17)) and "keeps it rather than switching" in notes[3]
    assert notes[4] == "yolo@host answers again"


# ── the lens ────────────────────────────────────────────────────────────────────────────


def test_uncalibrated_when_nobody_said_the_fov(
    hostd: FakeHostd, caplog: pytest.LogCaptureFixture
) -> None:
    """On a real body with no field of view from anyone, the lens is a guess: 62 degrees,
    every detection marked uncalibrated, and one warning however often it is recalibrated. A
    field of view from anyone makes it measured; a simulator's own camera is known."""
    caplog.set_level(logging.WARNING, logger="quackd.perception")
    hostd.detect_reply = _reply([("sports ball", 0.9, *BALL_BOX)])
    detector = HostDetector(HostClient(hostd.address), fov_deg=62.2)

    detector.calibrate(None, backend="bridge")
    detector.calibrate(None, backend="bridge")
    assert (detector.fov_deg, detector.calibrated) == (62.0, False)
    (guessed,) = detector.detect(_frame())
    assert guessed.calibrated is False
    assert "(uncalibrated" in guessed.summary()
    told = _warnings(caplog)
    assert len(told) == 1 and "no camera field of view given for a bridge camera" in told[0]

    detector.calibrate(62.2, backend="bridge")
    assert (detector.fov_deg, detector.calibrated) == (62.2, True)
    (measured,) = detector.detect(_frame())
    assert measured.calibrated is True

    detector.calibrate(None, backend="sim2d")
    assert (detector.fov_deg, detector.calibrated) == (90.0, True)


def test_a_confidence_floor_outside_nought_to_one_is_refused_when_it_is_built(
    hostd: FakeHostd,
) -> None:
    with pytest.raises(ValueError, match="from 0 to 1"):
        HostDetector(HostClient(hostd.address), fov_deg=62.2, conf=1.5)
    assert hostd.requests_to("/detect") == []
