"""An optional YOLO detector (`quackd[yolo]`) for real cameras and cluttered scenes.

Lazily imports `ultralytics` so the default install never pays for it. Maps COCO classes
onto the same labels the colour-blob detector emits — `ball`, `person`, `pet` — so verbs
and `.duck` files do not care which detector produced a detection.

The box maths lives in `detections_from_boxes`, apart from the model, because the model does not
have to run on this machine: a Jetson reached with `--host` runs the same YOLO on its own GPU and
sends back boxes, and those must become exactly the detections a local model's boxes would.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from PIL import Image

from quackd.perception.base import Detection

COCO_TO_LABEL = {"sports ball": "ball", "person": "person", "cat": "pet", "dog": "pet"}
HEIGHT_M = {"ball": 0.10, "person": 1.6, "pet": 0.35}


def detections_from_boxes(
    boxes: Iterable[tuple[str, float, float, float, float, float]],
    w: int,
    h: int,
    *,
    fov_deg: float,
    calibrated: bool = True,
) -> list[Detection]:
    """Boxes in pixels, as a YOLO model reports them, turned into quackd's detections.

    Each box is (the model's own class name, confidence, x1, y1, x2, y2) in the pixels of a
    `w` by `h` image. Names `COCO_TO_LABEL` does not map are dropped, because a detection the
    verbs have no word for is one they cannot act on. Bearing comes from the box centre's
    horizontal offset through a pinhole lens of `fov_deg`, and distance from the box height
    against `HEIGHT_M`, a typical real height for the label. Both are only as good as `fov_deg`,
    which is why `calibrated` is carried onto every detection: False says the lens was a guess.

    One function for every detector that sees boxes, so a local YOLO and one on a board reached
    with `--host` cannot drift apart: the same boxes are the same detections, to the last bit.
    """
    f = (w / 2) / math.tan(math.radians(fov_deg) / 2)
    out: list[Detection] = []
    for name, conf, *corners in boxes:
        label = COCO_TO_LABEL.get(name)
        if label is None:
            continue
        # float() as the inline version did on the model's tensors, so integer pixel boxes and
        # float ones take the same arithmetic
        x1, y1, x2, y2 = (float(v) for v in corners)
        cx, bh = (x1 + x2) / 2, max(1.0, y2 - y1)
        out.append(
            Detection(
                label=label,
                cx=cx / w,
                cy=(y1 + y2) / 2 / h,
                area=(x2 - x1) * bh / (w * h),
                confidence=float(conf),
                bearing_deg=round(-math.degrees(math.atan((cx - w / 2) / f)), 1),
                est_distance_m=round(f * HEIGHT_M[label] / bh, 3),
                calibrated=calibrated,
            )
        )
    return out


class YoloDetector:
    name = "yolo"

    def __init__(
        self,
        model: str = "yolov8n.pt",
        conf: float = 0.4,
        fov_deg: float = 62.0,
        calibrated: bool = True,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "YoloDetector needs ultralytics: uv pip install 'quackd[yolo]'"
            ) from e
        self._model: Any = YOLO(model)
        self.conf = conf
        self.fov_deg = fov_deg
        # False when `fov_deg` is a default rather than this lens's, as `ColorBlobDetector`
        # carries it: the detections still point the right way and say their size is a guess
        self.calibrated = calibrated

    def detect(self, image: Image.Image) -> list[Detection]:
        w, h = image.size
        results = self._model.predict(image, conf=self.conf, verbose=False)
        boxes: list[tuple[str, float, float, float, float, float]] = []
        for res in results:
            names = res.names
            for box in res.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                boxes.append((names[int(box.cls)], float(box.conf), x1, y1, x2, y2))
        return detections_from_boxes(boxes, w, h, fov_deg=self.fov_deg, calibrated=self.calibrated)
