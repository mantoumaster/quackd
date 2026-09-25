"""A detector whose model runs on the board `--host` names: YOLO on a Jetson's GPU, reached
from the laptop.

The frame is the one the run already has, the primary view that every other detector reads.
It goes to the board as one JPEG, the board's daemon runs its YOLO on it and sends back boxes in
that JPEG's pixels, and the geometry is done here by `detections_from_boxes`, the same function
the in-process `YoloDetector` uses. So the same boxes are the same detections whichever machine
ran the model, and the daemon never has to know quackd's labels or its lens.

Its `detect` is synchronous, like every `Detector`, and says it blocks (`blocking`), so the
verbs and the loop run it in a worker thread (`perception.base.detect_off_loop`): a round trip
to a board must not hold up the event loop that re-sends a walking body's twist. That keeps the
keepalives flowing, and no more: `go_to` holds its last twist for one deadman window
(`verbs.core.HOLD_TTL_S`), so a board slower than that still lets the body stop between frames,
which is the right thing for a steering loop that has stalled.

Each wait on the board's socket is bounded by `quackd.host.DETECT_TIMEOUT_S`. Two waits are not,
as `HostClient._exchange` says: a name that has stopped resolving is looked up on the system
resolver's own clock, before there is a socket, and is paid again on every call, and a reply
trickled a byte at a time can outlast the timeout. Both happen in the worker thread, and a call
that fails gives its frame no detections.

A board that fails is not a reason to pick another detector. A failed call returns no
detections and keeps the reason in `error`, and the run goes on with this detector: the colour
detector does not label the same things on a real camera, so switching to it quietly would
change what `go_to` steers at with nothing in the record saying so, and the header would name a
detector the run had stopped using. No detections is already a shape every verb handles: `go_to`
counts the target as lost and stops the body, and `observe` reports nothing seen, with the
reason beside it.

Nothing here has run against a Jetson. The tests drive it against the fake daemon in
`tests/fake_jetson_hostd.py`, which proves the protocol is read as written and says nothing
about a board.
"""

from __future__ import annotations

import io
import logging

from PIL import Image

from quackd.host import HostClient, HostError
from quackd.perception.base import Detection
from quackd.perception.yolo import detections_from_boxes

logger = logging.getLogger("quackd.perception")

JPEG_QUALITY = 85
"""What a frame is sent to the board at. Near the camera daemon's own 80, so a frame the host
camera took is not made worse on the way back, and small enough that a 640x480 frame is tens of
kilobytes on a robot's Wi-Fi rather than hundreds."""

GUESSED_FOV_DEG = 62.0
"""The lens assumed when nobody said which one this is: a Pi Camera Module 2's, the same guess
`YoloDetector` makes. Detections made with it carry `calibrated=False`."""


class HostDetector:
    """YOLO on the board `--host` names, as a `Detector`.

    `fov_deg` and `calibrated` are the lens the detections are measured through, as
    `YoloDetector` carries them. `calibrate()` sets both from what is known once the body has
    connected. `conf` overrides the daemon's own confidence floor, and None keeps the
    daemon's.

    `error` is None while the board answers, and the reason it did not the last time it did not.
    It is what `observe` and the loop read to say why nothing was seen."""

    name = "yolo@host"
    blocking = True
    """`detect` waits on the network, so `detect_off_loop` runs it in a worker thread."""

    def __init__(
        self,
        client: HostClient,
        *,
        fov_deg: float,
        calibrated: bool = True,
        conf: float | None = None,
    ) -> None:
        # checked here, once, because `HostClient.detect` raises ValueError for a bad floor on
        # every call and a caller mistake should surface when the detector is built, not as the
        # first frame of a run failing
        if conf is not None and (isinstance(conf, bool) or not 0.0 <= float(conf) <= 1.0):
            raise ValueError(f"conf is a confidence floor from 0 to 1, not {conf!r}")
        self.client = client
        self.fov_deg = fov_deg
        self.calibrated = calibrated
        self.conf = conf
        self.error: str | None = None
        self._told_lens = False

    def __repr__(self) -> str:
        return f"HostDetector({self.client.address!r}, fov_deg={self.fov_deg:g})"

    @property
    def address(self) -> str:
        """The board this detector asks, as `host:port`."""
        return self.client.address

    def calibrate(self, fov_deg: float | None, *, backend: str | None) -> None:
        """Set the lens from what is known about the camera in front of the run.

        `fov_deg` is the field of view somebody gave (`--fov-deg`, the body's own
        `camera_fov_deg`, or the one the board's daemon was started with when its camera is the
        primary view). With none on a simulator, the simulator's own camera is known and the
        detections are calibrated. With none on a real body, the lens is a guess: the
        detections still point the right way, carry `calibrated=False`, and this says so once,
        in the words `detector_for` uses for the colour detector."""
        from quackd.perception import lens

        self.fov_deg, self.calibrated = lens(fov_deg, backend, guess=GUESSED_FOV_DEG)
        if not self.calibrated and not self._told_lens:
            self._told_lens = True
            logger.warning(
                "no camera field of view given for a %s camera, so the host detector at %s "
                "uses %.0f degrees. Distances will be out by tens of percent: pass --fov-deg "
                "(a Pi Camera Module 2 is about 62) once you know yours.",
                backend,
                self.client.address,
                self.fov_deg,
            )

    def detect(self, image: Image.Image) -> list[Detection]:
        """The board's YOLO on this frame, as quackd's detections, or none when the board did
        not answer.

        A failure is warned about once per outage rather than once per frame, because `go_to`
        asks ten times a second and a log of the same sentence ten times a second hides the
        sentence. The next answer ends the outage, so a board that fails again is warned
        about again.

        An answer for an image of another size is a failure too. The boxes are pixels in the
        image the daemon decoded, which is the frame sent, so a reply that names another size
        describes some other image, and read against this frame its boxes would be bearings
        `go_to` steers at with nothing true about them."""
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY)
        try:
            found = self.client.detect(buf.getvalue(), conf=self.conf)
        except HostError as e:
            return self._failed(str(e))
        sent = image.size
        if (found.w, found.h) != sent:
            return self._failed(
                f"{self.client.address} answered for a {found.w}x{found.h} image, not the "
                f"{sent[0]}x{sent[1]} frame it was sent"
            )
        if self.error is not None:
            logger.warning("the host detector at %s answers again", self.client.address)
            self.error = None
        return detections_from_boxes(
            (box.as_tuple() for box in found.boxes),
            found.w,
            found.h,
            fov_deg=self.fov_deg,
            calibrated=self.calibrated,
        )

    def _failed(self, reason: str) -> list[Detection]:
        """No detections for this frame, the reason kept in `error`, and one warning for the
        outage it starts."""
        if self.error is None:
            logger.warning(
                "the host detector at %s failed, so this frame has no detections and the "
                "run keeps this detector: %s",
                self.client.address,
                reason,
            )
        self.error = reason
        return []
