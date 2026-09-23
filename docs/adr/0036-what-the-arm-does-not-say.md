# ADR-0036: The arm adapter stops taking the arm's word for it

**Status:** accepted, amended · **Date:** 2026-09-13 · Extends [ADR-0022](0022-per-adapter-upstream-refs.md) (every SDK name is a verified ref) and [ADR-0017](0017-robot-adapters-and-manifest.md) (a robot is a manifest) · Implemented in `quackd/adapters/lerobot/` ([page](../adapters/lerobot.md), [checklist](../lerobot-hardware-checklist.md))

**Amended 2026-09-18 by [ADR-0039](0039-an-arm-placed-by-hand.md):** the decision below that
opens "Nothing here changes what quackd never does" loses one clause of one sentence. quackd now
disables torque in one place, `let_go()`, which `quackd run --by-hand` calls once and nothing
else calls at all. It refuses anywhere but the recorded rest pose, which is the same read this
ADR's `close()` already makes before it lets torque drop on `disconnect()`, and it happens only
because a person standing at the arm asked for it: there is no verb for it, no MCP tool, and no
method on the `RobotAdapter` protocol, so a pilot cannot reach it. `take_hold()` is how the arm
is picked back up, and it writes the present position as the goal before torque returns because
what a servo does with its last goal on re-enable is a new UNVERIFIED ref. The rest of that
sentence is untouched and still holds: the adapter never calibrates, still refuses an arm with
no calibration file because that file is where the joint ranges come from, and still keeps
LeRobot's default of dropping torque on `disconnect()` rather than overriding it.

**Amended 2026-09-23 by [ADR-0045](0045-a-rest-pose-the-calibration-cannot-reach.md):** the
Context below leaves open what the firmware does with a goal past a joint's calibrated travel,
and an SO-101 has answered it. LeRobot's calibration writes that travel into each servo as its
two position limits, and the servo clamps every goal to them. A reading is not clamped, so a
joint folded with torque off can sit past either end. The range refusal below stands, and has a
second reason now: a goal it let through would be one the arm silently stops short of. The
decision that `stop` holds the five body joints now holds each one that reads inside its travel
and writes nothing for one that reads past it, because the only goal the servo would take for
that joint is its limit, and it drives there at full speed. The rest pose the amendment above
names is now driven clipped into the travel, and a joint recorded past the travel is at rest at
its edge or anywhere beyond it.

**Amended 2026-09-23, the same afternoon:** the decision below that made `duration_s` a budget
is reversed, because a budget left the step cap as the only pace the arm had, and a model on
the bench asked to raise the arm slowly read the verb's text correctly and declined.
`duration_s` is now how long the motion should take: `move_joints` reads the arm once and walks
its goal from there to the goal across that time, one target a tick, then sends the goal itself
until the joints arrive. The step cap is unchanged and is now a ceiling rather than the speed:
a time too short for the distance runs at the cap and ends later than asked. Arrival and stalls
are judged once the ramp is over, since a slow ramp moves a joint less per tick than the stall
threshold. The verb's budget is the time asked for or the time the cap needs, whichever is
longer, plus a settle, and ends inside the executor's timeout for the verb, which is the same
constant. A joint reading past its travel ramps from the edge of it, because the servo takes it
there at its own speed whatever is sent. `gripper` is not ramped. The comparison of goal and
measurement every tick, the failure that says where a joint stopped, and the tolerance all
stand.

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
`not_hot`, refuses `move_joints` and `pick` when any body joint reads at or above `HOT_C`
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

- The arm supplies its own `report_state`, because a pilot reads a verb's summary text and
  never its data, and the core verb's summary is a posture and a policy name: two facts a
  bolted-down arm has not got and none of the four it has. Its own says where every joint is,
  whether torque is on, how warm the servos are and whether anything is held, which is what
  `lerobot-lookout` asks a pilot to report.
- A pilot sees `torque` measured, `temperature_c` and `hot` per joint, `joint_range_deg`,
  `calibration_file` and `limits.step_deg` on a connected arm, a `not_hot` gate on the two
  verbs that move the body joints, and two verbs that can now fail with where the arm stopped
  instead of reporting the move they were asked for.
- An operator has an order to try it in
  ([lerobot-hardware-checklist.md](../lerobot-hardware-checklist.md)), a calibration id that
  must match quackd's robot id, and one environment variable for the step. What only a real arm
  can settle is that checklist's *What to report*, and this ADR does not repeat it.
- A camera is one USB webcam named by `--camera-url opencv://N`, and quackd builds it
  itself rather than handing it to the follower, because a follower's `is_connected`,
  `send_action` and `disconnect()` all include its cameras and one unplugged webcam would
  make every move and every hold raise. A camera asked for and not opened refuses at
  connect, before the arm is touched at all; one that dies later costs `observe` and a `pick`
  in flight and nothing else. The lookout task still asks
  for `report_state`, because a `.duck` is checked against the static manifest, which cannot
  know whether this arm has a camera; and quackd's
  executor still does not serialise concurrent verbs, so two MCP calls can still fight over one
  joint, which is a property of every adapter and not of this one.
- The numbers named above live with their constants in `real.py` and `verbs.py`, and are
  expected to move once an arm has been watched. This file records why each exists, not what
  it is today; the adapter page records what the adapter does; `upstream_api.py` records what
  upstream says. Each fact is in one of the three.
