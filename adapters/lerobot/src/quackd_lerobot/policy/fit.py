"""Whether the policy a server serves fits the arm about to be handed to it, asked at connect.

The arm's side asks `GET /v1/policy` once its cameras are open and before its motors are
energised (`real.LeRobotReal._fit_policy`), and refuses to connect over a policy that could not
drive this arm, in a sentence that says what to do. Nothing here is typed in: the motors are
the bus's, in its order, each camera's size is a frame it gave, the travel is read off this
arm's calibration file, and the slack past it is the backend's own (`real.OUT_OF_RANGE_DEG`).

- **The state and the action** are as long as the bus has motors, and where the checkpoint
  names its action's dimensions (`upstream_api.ACTION_FEATURE_NAMES`), they are the bus's
  motors in the bus's order. Where it does not, the order is trusted rather than checked: the
  policy's state and action are taken to be in the order of the bus that recorded its data.
- **The cameras.** Every image a policy looks at needs a camera of this arm's mapped to it,
  unless the policy pads a missing one (`upstream_api.MISSING_IMAGES_PADDED`), which is said
  instead. A frame whose height and width are not the image's is refused, and the refusal says
  to give the camera that size, which every camera URL takes. A caller from Python may accept
  it instead (`accept_frame_size`), knowing an ACT runs on another size and sees what it never
  saw.
- **The frame.** The 1st and 99th percentiles of the state the policy learned from lie inside
  this arm's calibrated travel. A policy trained on an arm calibrated another way asks for
  goals that pin this one at its limits, so it is refused, unless the person running it knows
  better (`accept_other_frame`, which `--accept-other-frame` sets). Its goals are clipped to
  this arm's travel either way, so the override lets the policy connect and never moves the
  arm anywhere it could not go without it.

This module needs nothing but the protocol's messages, so it runs in the arm's process, which
never imports torch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from quackd_lerobot.policy import protocol as wire

POS = ".pos"
"""What a motor's name is followed by in LeRobot's observation and action keys."""


class PolicyMisfit(ValueError):
    """A policy that does not fit this arm, in one sentence that says what to do."""


@dataclass(frozen=True)
class Fit:
    """What the check found: why the arm must not connect over this policy, or None, and what
    the record should say either way."""

    refusal: str | None
    notes: tuple[str, ...] = ()


def _joined(names: Sequence[str]) -> str:
    return ", ".join(names) if names else "none"


def fit(
    info: wire.PolicyInfo,
    *,
    motors: Sequence[str],
    cameras: Sequence[wire.CameraInfo],
    travel: Mapping[str, tuple[float, float]],
    slack_deg: float,
    where: str,
    accept_frame_size: bool = False,
    accept_other_frame: bool = False,
) -> Fit:
    """Whether the policy `info` describes fits an arm whose bus lists `motors`, whose cameras
    gave `cameras`, and whose calibration gives `travel` (degrees, the gripper 0..100), with
    `slack_deg` of reading past the travel forgiven. `where` is the server, as it is named in a
    sentence. The first thing that does not fit is the refusal; everything else goes in the
    notes, the overrides taken included."""
    features = info.features
    notes = [
        f"the policy at {where} is {info.policy}, at {info.rate_hz:g} Hz from {info.rate_source}"
    ]
    if info.loaded:
        notes.append(f"the policy server loaded {'; '.join(info.loaded)}")

    def refused(why: str) -> Fit:
        return Fit(f"the policy at {where}, {info.policy}, {why}", tuple(notes))

    # the state and the action, by count and, where the checkpoint says, by name
    for what, size in (("state", features.state), ("action", features.action)):
        if size is not None and size != len(motors):
            return refused(
                f"takes {'a' if what == 'state' else 'an'} {what} of {size} numbers, and this "
                f"arm's bus has {len(motors)} motors ({_joined(motors)}): it learned from another "
                "arm, so serve a checkpoint trained on this one"
            )
    if features.action_names is not None:
        named = [name.removesuffix(POS) for name in features.action_names]
        if named != list(motors):
            return refused(
                f"names its actions {_joined(named)}, and this arm's bus lists "
                f"{_joined(motors)}: its goals would go to the wrong joints"
            )

    # the cameras: every image needs one, unless the policy pads it, and one of its size
    given = {camera.name: camera for camera in cameras}
    seen = {key: given[name] for name, key in info.cameras.items() if name in given}
    if features.images and not any(image.key in seen for image in features.images):
        return refused(
            f"looks at {_joined([i.key for i in features.images])}, and no camera of this arm "
            f"({_joined(sorted(given))}) is mapped to any of them: give the arm the cameras it "
            "was trained with, or start the server with --cameras NAME=KEY for the ones it has"
        )
    for image in features.images:
        camera = seen.get(image.key)
        if camera is None:
            if not features.pads_images:
                return refused(
                    f"looks at {image.key}, and no camera of this arm "
                    f"({_joined(sorted(given))}) is mapped to it: give the arm the camera it "
                    f"was trained with, or start the server with --cameras NAME={image.key}"
                )
            notes.append(f"{image.key} has no camera on this arm, so the policy sees it padded")
            continue
        if image.height is None or image.width is None:
            continue
        if (camera.height, camera.width) != (image.height, image.width):
            said = (
                f"the {camera.name} camera gives {camera.width}x{camera.height} frames, and the "
                f"policy learned {image.key} at {image.width}x{image.height}"
            )
            if not accept_frame_size:
                return refused(
                    f"cannot use them as they come: {said}. Give the camera that size with "
                    f"--camera-url's width={image.width} and height={image.height}"
                )
            notes.append(f"{said}, which was accepted (accept_frame_size)")

    # the frame: what the policy learned from lies inside this arm's travel
    quantiles = info.state_quantiles
    unchecked = "so whether it learned from an arm calibrated like this one was not checked"
    if quantiles is None:
        if features.state is not None:
            notes.append(f"the policy reports no state quantiles, {unchecked}")
    elif len(quantiles.q01) != len(motors):
        notes.append(
            f"the policy reports {len(quantiles.q01)} state quantiles for {len(motors)} "
            f"motors, {unchecked}"
        )
    elif not travel:
        notes.append(f"this arm has no calibrated travel to hold them against, {unchecked}")
    else:
        outside = []
        for motor, low, high in zip(motors, quantiles.q01, quantiles.q99, strict=True):
            span = travel.get(motor)
            if span is None:
                continue
            if low < span[0] - slack_deg or high > span[1] + slack_deg:
                outside.append(
                    f"{motor} {low:.1f}..{high:.1f} against {span[0]:.1f}..{span[1]:.1f}"
                )
        if outside:
            said = (
                f"learned from an arm whose readings (q01..q99) lie outside this arm's "
                f"calibrated travel: {'; '.join(outside)}"
            )
            if not accept_other_frame:
                return refused(
                    f"{said}. It was trained on an arm calibrated another way, and its goals "
                    "would pin this one at its limits: serve a checkpoint trained on this arm, "
                    "or give --accept-other-frame if you know the two frames match, which lets "
                    "it connect with every goal still clipped to this arm's travel"
                )
            notes.append(f"the policy {said}, which was accepted (--accept-other-frame)")
        missing = [m for m in motors if m not in travel]
        if missing:
            notes.append(
                f"{_joined(missing)} {'has' if len(missing) == 1 else 'have'} no travel in the "
                "calibration, so the policy's readings there were not checked"
            )
    return Fit(None, tuple(notes))


__all__ = ["Fit", "PolicyMisfit", "fit"]
