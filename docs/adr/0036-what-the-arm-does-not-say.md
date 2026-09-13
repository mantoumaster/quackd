# ADR-0036: The arm adapter stops taking the arm's word for it

**Status:** accepted · **Date:** 2026-09-13 · Extends [ADR-0022](0022-per-adapter-upstream-refs.md) (every SDK name is a verified ref) and [ADR-0017](0017-robot-adapters-and-manifest.md) (a robot is a manifest) · Implemented in `quackd/adapters/lerobot/` ([page](../adapters/lerobot.md), [checklist](../lerobot-hardware-checklist.md))

## Context

The LeRobot adapter drives an SO-101 follower through the `Robot` interface of
`huggingface/lerobot`, pinned at `fbb811f`. It has never been run against an arm. Before this
decision it was written the way an adapter is written when the SDK is trusted: every fact it
needed, it took from the SDK's own answer. Re-reading the pinned source, line by line, showed
how little of what the adapter needed the SDK actually says:

- `Robot.is_connected` is the serial port's open flag. Unplug the arm and it stays `True`
  until a read happens to fail. The old heartbeat checked the flag.
- `get_observation()` is one read of `Present_Position` and nothing else. No torque state, no
  temperature, no fault. The old state reported `torque: True` as a constant, and a servo that
  had tripped its own overload protection looked healthy.
- The DEGREES branch of the motors bus's un-normalise does not clamp, though its two 0..100
  modes do. A goal past a joint's calibrated travel goes to the servo as-is, and what the
  firmware does with it is Feetech's business.
- `max_relative_target` defaults to `None`: one action may slew a joint across its whole
  travel. `send_action()` returns the goal it actually wrote, and the old adapter discarded it.
- `configure()` writes a torque cap, a current cap and an overload cap on the **gripper**,
  inside a check for that one motor's name. The five body joints get nothing from LeRobot.
- There is no watchdog, timer or timeout anywhere in the class. A position-controlled arm holds
  its last goal under torque until the next write or `disconnect()`, and `disconnect()` drops
  torque by default.

None of this is a defect in LeRobot. It is a teleoperation library whose human is the safety
system. quackd's human is an LLM three seconds away, so each gap had to be closed on quackd's
side, and each closure is a decision worth recording, because every one of them puts a number
or a rule between the pilot and the arm that upstream never had.

## Decision

**The adapter reads two registers below the `Robot` interface.** Torque state and per-joint
temperature come off the Feetech bus by name (`Torque_Enable`, `Present_Temperature`), through
`SOFollower.bus` and `MotorsBus.sync_read`, in the same worker thread as the position read so
nothing interleaves on a half-duplex wire. This is the one place quackd steps below an SDK's
public interface, and it stays inside [ADR-0022](0022-per-adapter-upstream-refs.md): the
attribute, the method, the two register names and their addresses are verified refs at the pin.
The position read is the liveness check and may raise; a failed register read costs one
reading, is recorded in `extras.register_error`, and the last known values stand, because a
Feetech bus returns the odd corrupt packet and losing a run over one would be worse than the
disease.

**The heat gate covers exactly the joints LeRobot does not cap.** A new precondition,
`not_hot`, refuses `move_joints` and `pick` when any joint reads at or above `HOT_C`
(`real.py`). It is not on `gripper` or `place`: the gripper has LeRobot's own caps, and opening
a hot gripper is how you put down what it is holding. The threshold and the servo's own
cut-off above it are Feetech's documentation, not anything measured here, and both are listed
in `extras.assumptions` under `TEMPERATURE_C`.

**A goal outside the calibrated travel is refused for a verb and clipped for a policy.** At
connect the adapter reads `Robot.calibration` and turns each joint's recorded tick range into
degrees with LeRobot's own formula, publishing it as `extras.joint_range_deg`. A verb's goal
outside it is refused with the range in the reason, before anything is written. A policy's
action outside it is clipped and counted (`extras.range_clips`), because a learned policy
that is one tick over should not abort a grasp it is in the middle of, and because it is
confirm-gated already.

**One action moves a joint at most one step, so verbs re-send and watch.** quackd sets
`max_relative_target` to `MAX_STEP_DEG` (`real.py`), re-sent at the core verbs' tick, which
makes the step also the top joint speed; `QUACKD_LEROBOT_MAX_STEP_DEG` changes it. It is a
float, because upstream's clamp raises on an int, and a scalar rather than a per-motor dict,
because the dict form must name exactly the joints in the action and quackd sends partial
actions. The same cap applies to the gripper in its own 0..100 units. Because one action no
longer reaches the goal, and because nothing upstream reports arrival, `move_joints` and
`gripper` compare the goal with the measurement every tick: `duration_s` became a budget, a
joint that stops short is a failure that says where it stopped, and the tolerance
(`TOL_DEG`, `verbs.py`) is wide because the controller is P-only and a loaded joint settles a
little short.

**Holding is inferred, and says so.** Nothing on this body senses grip force. `holding` is
true when the gripper was told to close, its reading has settled, and it settled inside a band
(`HOLD_MIN`..`HOLD_MAX`, `real.py`) short of shut. That is a guess with a name,
`HOLDING_INFERRED`, in `extras.assumptions`; the checklist asks a real arm to say what a
grasp actually reads.

**`stop` holds the five body joints and leaves the gripper's goal alone.** LeRobot writes only
the keys it is given. A stop that re-sent the gripper's *measured* position would open a hand
that is squeezing something against its own goal, and every failed verb ends in a stop.

**A blown deadline wedges the transport.** Every LeRobot call runs alone under one lock in a
worker thread with a deadline. A call that blows it has not finished: its thread is still on
the wire. Releasing the lock and starting another would put two talkers on a half-duplex bus,
so the transport refuses every later call, names the wedged call in `stop_error`, and the
heartbeat fails until the thread comes back. The arm holds its goal meanwhile, which is the
one thing that needs no rescue.

**Nothing here changes what quackd never does.** It never disables torque, never calibrates
(upstream's calibration is interactive, and an arm without a calibration file is refused
because that file is where the joint ranges come from), and keeps LeRobot's default of
dropping torque on `disconnect()`, documented rather than overridden.

**The datasheet stops claiming a mass.** Vendor listings put the arm anywhere from 0.8 to
2.5 kg and nobody official publishes one. Under [ADR-0032](0032-datasheets-and-the-verdict.md)
that is not published, not an estimate.

## Consequences

- A pilot sees `torque` measured, `temperature_c` and `hot` per joint, `joint_range_deg`,
  `calibration_file` and `limits.step_deg` on a connected arm, a `not_hot` gate on the two
  verbs that move the body joints, and two verbs that can now fail with where the arm stopped
  instead of reporting the move they were asked for.
- An operator has an order to try it in
  ([lerobot-hardware-checklist.md](../lerobot-hardware-checklist.md)), a calibration id that
  must match quackd's robot id, and one environment variable for the step. What only a real arm
  can settle is that checklist's *What to report*, and this ADR does not repeat it.
- What was deliberately not done: the real backend still configures no camera, so `observe`
  does not exist on it and the lookout task asks for `report_state` instead; and quackd's
  executor still does not serialise concurrent verbs, so two MCP calls can still fight over one
  joint, which is a property of every adapter and not of this one.
- The numbers named above live with their constants in `real.py` and `verbs.py`, and are
  expected to move once an arm has been watched. This file records why each exists, not what
  it is today; the adapter page records what the adapter does; `upstream_api.py` records what
  upstream says. Each fact is in one of the three.
