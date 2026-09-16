"""What the walking policy can actually do, and the arithmetic around it.

The gait envelope here is the part of the physics backend that decides whether a duck moves
or only says it did, so it lives in the one module of `sim3d` that imports no `mujoco` and
needs no downloaded model. Everything here is pure arithmetic over floats, and it is tested
on every runner rather than only where the physics extra and a filled asset cache happen to
meet. The posture geometry keeps it company for the same reason: it is read on every single
observation, so it is the last place that should be able to raise.

The numbers are measurements, not upstream's: see `upstream_api.GAIT_THRESHOLD`, which is
tagged UNVERIFIED and records the machine and the day they were taken on.
"""

from __future__ import annotations

import math

#: The command envelope the walking policy actually uses. Below the floor it does not step at
#: all: it stands and shifts its weight. The ceilings are the ranges upstream trains on.
GAIT_FLOOR_VX = 0.22
GAIT_FLOOR_VY = 0.30
GAIT_FLOOR_WZ = 1.00
CMD_MAX_VX = 0.40
CMD_MAX_VY = 0.30
CMD_MAX_WZ = 1.50

#: A twist below a third of the floor is dropped rather than raised: a steering loop that
#: asks for a two-degree correction should get nothing, not a full-rate lurch.
DEAD_FRACTION = 1 / 3

#: Roughly what fraction of a command the body achieves, for the honest line in the state.
ACHIEVED_FRACTION = 0.42

FLOORS = (GAIT_FLOOR_VX, GAIT_FLOOR_VY, GAIT_FLOOR_WZ)
CEILINGS = (CMD_MAX_VX, CMD_MAX_VY, CMD_MAX_WZ)


def usable_twist(
    cmd: tuple[float, float, float], *, standing: bool = True
) -> tuple[float, float, float]:
    """The commanded twist, mapped onto what the gait can actually do.

    The floor belongs to the twist as a whole, not to each axis: a duck already walking
    forward turns happily at a rate that would not start a turn on its own. So the measure is
    how close the twist is to stepping at all, and a twist that is not there yet is scaled up
    bodily. Scaling keeps the ratio between the axes, which is what makes an arc an arc:
    raising `wz` alone would turn "walk in a circle" into a spin.

    `standing` is passed in rather than read off a body, so the half that matters most — a
    duck that is down is sent nothing at all — is testable without a model to knock over.

    A non-finite command raises. Nothing here can do anything honest with one: every
    comparison against NaN is False, so it would slip past the dead zone with a gain of 1.0
    and reach the servos unchanged. The world refuses one before it gets this far; this is
    the backstop that makes the refusal a rule rather than a habit of one caller.
    """
    if not all(math.isfinite(v) for v in cmd):
        raise ValueError(f"a twist must be finite, got {cmd}")
    if not standing:
        return (0.0, 0.0, 0.0)
    activity = max(abs(v) / floor for v, floor in zip(cmd, FLOORS, strict=True))
    if activity < DEAD_FRACTION:
        return (0.0, 0.0, 0.0)  # too small to step: standing still is the honest answer
    gain = 1.0 / activity if activity < 1.0 else 1.0
    scaled = [
        math.copysign(min(abs(v) * gain, ceiling), v)
        for v, ceiling in zip(cmd, CEILINGS, strict=True)
    ]
    return (scaled[0], scaled[1], scaled[2])


def tilt_deg(gravity_z: float) -> float:
    """How far off vertical the trunk is, from projected gravity's z. 0 upright, 180 inverted.

    The clamp is both-sided on purpose. `gravity_z` is a matrix element, so floating point
    leaves it outside [-1, 1] by about 1e-10 at either pole, and `acos` raises a `ValueError`
    rather than saturating. Only the upper bound used to be clamped, which meant that reading
    the state of a duck lying on its back raised out of `extras()`, out of `snapshot()`, and
    out of every `get_state()` the pilot made while it was down: the observation a fall is
    exactly when you need it.
    """
    return math.degrees(math.acos(max(-1.0, min(1.0, -gravity_z))))
