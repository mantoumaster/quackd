# ADR-0008: Features, not frames — a colour-blob detector by default

**Status:** accepted · **Date:** 2026-08-28

## Context

Upstream's principle: put perception next to the sensor and publish *features* ("ball at
(x, y)", "person detected"), tens of bytes at 10–30 Hz. The steering loop needs detections
at ~10 Hz on a laptop; the LLM needs a sentence, not a pixel grid, to decide the next verb.

## Decision

- `Detector` protocol → `[Detection(label, cx, cy, area, confidence, bearing_deg,
  est_distance_m)]`. Verbs and prompts consume only this.
- Default `ColorBlobDetector` (OpenCV HSV threshold, ~1 ms/frame, zero downloads).
  The sim draws the ball in a known orange (H≈16 in OpenCV's 0–180 scale), the person in
  blue, a pet in green. Geometry: `bearing = -atan((cx - w/2) / f)` (positive = left,
  upstream's +yaw convention); `distance = f · r / radius_px` for round targets and
  `f · half_width / (w_px/2)` for upright ones, with `f = (w/2) / tan(FOV/2)`.
- `YoloDetector` (`quackd[yolo]`, lazy `ultralytics` import) maps COCO classes to the same
  labels for real cameras. Not needed for the demo.
- The LLM sees a summary line (`ball at bearing 12° left, ~0.80 m`) plus, for vision
  providers, the last frame or two as images; older images are dropped from history.

## Consequences

- Tuning for a real orange ball = one `HSVRange` (documented in `docs/faq.md`); a real
  IMX219 = `fov_deg=62`. Nothing else changes.
- Distance from apparent size is crude (±20 %) but monotonic, which is all `walk_to`
  needs; the stop condition is verified by the kick result or a fresh frame, not trusted.
- When upstream ships `mediad`'s feature stream, it becomes one more `Detector` that reads
  a socket instead of an image.

## Note (2026-09-16): several views, still one detector

A LeRobot arm can be registered with more than one camera (`--camera-url` repeats), so "the last
frame or two as images" in *Decision* is now the last frame or two from every camera it has. They
go out in one message, each picture preceded by a text part naming its camera, because two
unlabelled views of a desk say nothing about which lens took which. The saved frames carry the
name too (`frames/0007-top.png` where a single camera writes `frames/0007.png`).

This does not change the decision, and the three things it rests on are untouched:

- The pilot still reads features. The summary line is still the designed path, and a text-only
  model still gets the whole task through it.
- The `Detector` still runs on one camera, the first url, the one `--fov-deg` describes. A
  bearing is only meaningful from the lens it was measured on, so when that camera is the one
  that stalls the line reports nothing seen even though the other views still reach the model.
- The verbs that steer by sight still read that one camera at ~10 Hz, for the reason this ADR
  was written: fetching every camera inside that loop would spend the deadman window on pixels.

The cost it does add is pictures per request. The last two exchanges keep their images and that
limit counts exchanges rather than images, so two cameras is four pictures in a request rather
than two.
