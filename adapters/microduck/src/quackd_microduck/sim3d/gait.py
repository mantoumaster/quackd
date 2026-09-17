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
#:
#: The forward floor is a property of the physics as much as of the policy, so it moves when
#: MuJoCo does. It was 0.22 on 3.12 and is 0.23 on 3.13: at 0.225 the duck now stands and
#: shifts its weight for ten seconds and covers thirteen millimetres, and at 0.230 it walks
#: 0.81 to 0.89 m in the same ten on every one of the ten sweep seeds. That difference is the
#: whole of `go_to` on this body, which asks for 0.2 and is raised to exactly the floor: a
#: floor one hundredth too low is a duck that reports walking to a ball dead ahead and never
#: arrives. `upstream_api.GAIT_THRESHOLD` records both measurements and the versions.
GAIT_FLOOR_VX = 0.23
GAIT_FLOOR_VY = 0.30
GAIT_FLOOR_WZ = 1.00
CMD_MAX_VX = 0.40
CMD_MAX_VY = 0.30
CMD_MAX_WZ = 1.50

#: A twist below a third of the floor is dropped rather than raised: a steering loop that
#: asks for a two-degree correction should get nothing, not a full-rate lurch.
DEAD_FRACTION = 1 / 3

#: Roughly what fraction of a command the body achieves, for the honest line in the state.
#: Re-measured with the floor on 2026-09-17: 0.88 m in ten seconds at a commanded 0.23 is
#: 0.38, where MuJoCo 3.12 gave 0.42.
ACHIEVED_FRACTION = 0.38

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
