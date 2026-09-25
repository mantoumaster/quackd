"""Features, not frames.

Mirroring upstream's principle, the LLM sees "ball at bearing 12° left, ~0.8 m", not pixels,
and the steering loop closes on detections at ~10 Hz without an LLM in the way. This package
turns images into that.
"""

from __future__ import annotations

import logging
from collections.abc import Container
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from quackd.host import HostClient, HostHello
    from quackd.perception.base import Detector

__all__ = ["DETECTOR_CHOICES", "detector_for", "explicit_detector", "is_simulated", "lens"]

logger = logging.getLogger("quackd.perception")

#: Backends whose camera really is the one the default geometry assumes.
_SIMULATED = ("sim2d", "mujoco", "mock")


def is_simulated(backend: str | None) -> bool:
    """Whether this backend's camera is a simulator's, whose lens the default geometry already
    knows. No backend at all is a bare transport, which is the simulator's."""
    return backend is None or backend in _SIMULATED


DETECTOR_CHOICES = ("color", "host", "yolo")
"""What `--detector` takes. Absent is its own choice: the board's detector on a real body when
`--host` names a daemon that can detect, else the colour detector on this machine."""


def lens(fov_deg: float | None, backend: str | None, *, guess: float) -> tuple[float, bool]:
    """(field of view, calibrated) for a detector that measures through a lens.

    A field of view somebody gave is the lens, and calibrated. With none, a simulator's camera
    is still known, because it is the one `color_blob.DEFAULT_FOV_DEG` describes. A real
    body with none gets `guess`, uncalibrated, so every detection says its size is a guess
    rather than looking measured."""
    if fov_deg is not None:
        return float(fov_deg), True
    if is_simulated(backend):
        from quackd.perception.color_blob import DEFAULT_FOV_DEG

        return DEFAULT_FOV_DEG, True
    return guess, False


class _Calibrates(Protocol):
    def calibrate(self, fov_deg: float | None, *, backend: str | None) -> None: ...


def _lens_if_known(
    detector: _Calibrates, fov_deg: float | None, backend: str | None, *, has_camera: bool
) -> None:
    """Set a detector's lens before the body connects, when it is already known: a field of
    view somebody gave, a simulator's own camera, or a body described with a camera of its own.

    A body described without one is left alone, at the guess it was built with, and nothing is
    said about it yet. Such a body may report a camera of its own when it connects (an arm given
    --camera-url, a rosbridge base, a ToddlerBot whose daemon owns its cameras), and until then
    nobody knows whether the primary view will be that camera or a `--host` board's. A lens
    taken from the board now would measure the body's own camera through the board's; the loop
    sets the lens at connect, from the manifest the body reported, and warns then if there is
    none."""
    if fov_deg is not None or has_camera or is_simulated(backend):
        detector.calibrate(fov_deg, backend=backend)


def explicit_detector(
    choice: str | None,
    *,
    client: HostClient | None,
    hello: HostHello | None,
    fov_deg: float | None,
    backend: str | None,
    has_camera: bool = True,
) -> Detector | None:
    """The detector a run was asked for, or that `--host` brings, before `detector_for` fills in
    the default. None means the default: `detector_for` then builds the colour detector for a
    body with a camera.

    - `color` is None, which is how a run opts out of a board's detector.
    - `host` is the board's detector, honoured even on a simulator because somebody asked for
      it by name. Without a board, or with one whose daemon cannot detect, it is a ValueError
      whose sentence says what the daemon said, raised before anything connects.
    - `yolo` is YOLO in this process, which raises its own ImportError naming the extra when
      `ultralytics` is not installed.
    - None, the default, is the board's detector when the daemon can detect and the body is
      real. Never on a simulator: YOLO does not see a cartoon ball, and the colour detector is
      tuned for exactly what a simulator draws.

    `client` and `hello` are the board `--host` named, already asked what it is, or both None.
    `fov_deg` is `--fov-deg` or the body's own `camera_fov_deg`, never the board's: whether the
    board's camera is the primary view is only settled at connect. `has_camera` is whether the
    body's own description has a camera. The lens is set here when it is already known
    (`_lens_if_known`), and the detector is marked `lens_at_connect`, so the loop and the MCP
    server call `calibrate()` again once the body has said what it really has. A detector
    handed to a run from Python has no such mark and keeps the lens it was built with.

    A body that describes no camera still gets the board's detector by itself. It may report a
    camera when it connects, as a ToddlerBot does when its daemon owns the cameras and the
    board's daemon was started with `--camera none`, and the board's detector is then the one a
    real body is meant to have."""
    if choice is not None and choice not in DETECTOR_CHOICES:
        raise ValueError(f"--detector is one of {', '.join(DETECTOR_CHOICES)}, not {choice!r}")
    if choice == "color":
        return None
    if choice == "yolo":
        from quackd.perception.yolo import GUESSED_FOV_DEG, YoloDetector

        yolo = YoloDetector(fov_deg=GUESSED_FOV_DEG, calibrated=False)
        yolo.lens_at_connect = True
        _lens_if_known(yolo, fov_deg, backend, has_camera=has_camera)
        return yolo
    can_detect = client is not None and hello is not None and hello.can_detect
    if choice == "host":
        if client is None or hello is None:
            raise ValueError(
                "--detector host runs the detector on the board --host names, and no board is "
                "named: add --host, or drop --detector host"
            )
        if not can_detect:
            said = hello.detect_error or "it did not say why"
            raise ValueError(
                f"--detector host: the daemon at {client.address} cannot detect (detect=false: "
                f"{said}). Start it without --no-detect, with ultralytics on the board, or drop "
                "--detector host"
            )
    elif not can_detect or is_simulated(backend):
        return None
    from quackd.perception.host import GUESSED_FOV_DEG, HostDetector

    assert client is not None  # can_detect, or refused above
    detector = HostDetector(client, fov_deg=GUESSED_FOV_DEG, calibrated=False)
    detector.lens_at_connect = True
    _lens_if_known(detector, fov_deg, backend, has_camera=has_camera)
    return detector


def detector_for(
    sensors: Container[str],
    current: Detector | None = None,
    *,
    fov_deg: float | None = None,
    backend: str | None = None,
) -> Detector | None:
    """A robot with a camera needs something to look at its frames with. Any robot.

    Keying this on the backend being `sim2d` is the bug that made every hardware body run
    blind: it fetched a frame, detected nothing because nothing was detecting, and reported
    that it could not see. 0.5 fixed that in `quackd run` and missed `serve-mcp`, which is
    why the decision now lives in one function that both entry points call.

    Call it with what the robot said when it *connected*, not with its description. A
    rosbridge base has no camera in its static manifest and may well have one in its live
    one, and the description of a fully built duck promises a camera the duck in front of
    you may not have been built with.

    `fov_deg` is the horizontal field of view of the lens actually in front of you. The
    default is the simulator's 90 degrees, and a real Pi camera module is nearer 62, which
    makes the focal length half what it should be: bearings come out inflated by about 1.7x
    and distances short by about 40 percent. `go_to` then announces it has arrived while the
    duck is still half a metre out — outside `open-duck-scout`'s own success criterion, so
    the run reports success and the ground truth says it failed. There was no flag, env var
    or manifest key to correct it; `docs/faq.md` said to edit quackd's source.
    """
    if current is not None or "camera" not in sensors:
        return current
    from quackd.perception.color_blob import DEFAULT_FOV_DEG, ColorBlobDetector

    simulated = backend is None or backend in _SIMULATED
    if fov_deg is not None:
        return ColorBlobDetector(fov_deg=fov_deg)
    if simulated:
        return ColorBlobDetector()
    # A real lens, and nobody said which. The geometry still works — the target is in the
    # right direction — but the numbers are the wrong size, and saying so beats a confident
    # measurement nobody can act on.
    logger.warning(
        "no camera field of view given for a %s camera, so detections use the simulator's "
        "%.0f degrees. Distances will be out by tens of percent: pass --fov-deg (a Pi Camera "
        "Module 2 is about 62) once you know yours.",
        backend,
        DEFAULT_FOV_DEG,
    )
    return ColorBlobDetector(calibrated=False)
