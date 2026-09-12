# Safety

A biped falls in 0.3 s; an LLM answers in 3 s. Everything here follows from that.

## Layers

| Layer | Owner | What it guarantees |
|---|---|---|
| Body | the robot's own controller | **Whatever that particular body actually offers, which is not the same everywhere.** The Microduck's `robotd` gives joint and thermal clamps, fall detection and a **deadman**: velocity goes to zero when `robot.move` notifications stop. An Open Duck Mini v2 gives *none of those*: its deadman is quackd's own daemon on the Pi and the human watching is its only fall detector (details under "On hardware"). The body is still the sole safety authority: clients send intents, never motor writes. What each body offers is declared in its manifest's `safety_authority`, and `quackd doctor` prints what the robot itself reported (see "On other bodies"). |
| Conversation | quackd `Executor` | The LLM and MCP clients can only do what the `.duck` allows, as often as the budget allows, with a human in the loop where the contract says so. |
| Session | quackd `Heartbeat` + `KillSwitch` | A dead transport or a worried human ends in a `stop` intent. |

## The executor (mirrors upstream's own rules)

Every verb call — from the agent loop or an MCP session — passes `Executor.run_verb`, in
this order: abort flag (`stop` is exempt, so the brake still works) → **allowlist**
(`verbs.allow`; `stop` always allowed) → param validation (errors are feedback to the
model, not crashes) → **confirm gate** (`verbs.confirm` or `safety_class` ∈ {confirm,
dangerous}; y/N in the terminal, `--yes` to auto-accept, MCP refuses unless `--yes`) →
**budgets** (`max_steps` and `max_minutes` here, which is what caps an MCP session since
there is no loop there; `max_llm_calls` is the loop's own) →
machine-enforced **`abort_when`** (the battery threshold here, consecutive failures once
the result is in) →
**preconditions** (not fallen, not sitting) → `--dry-run` → execute, racing the **timeout**
against the abort, so a kill switch cancels the verb. A verb that times out or raises stops
the duck and reports a failure.

So does a call whose *caller* goes away: an MCP client dropping the request, or a second
Ctrl-C. The verb is cancelled and a `stop` goes out, recorded as `gate cancelled`. That path
used to return at once and leave the legs moving with nothing to halt them, which is the
failure this page exists to rule out.

## Heartbeat

A task pings `transport.heartbeat()` every 500 ms (`robot.health` on a Microduck, each
backend's own health call elsewhere, a liveness check in sim). One failure → `stop` intent → abort flag → the loop ends with
`outcome: aborted`. Upstream's own rationale: "LLMs stall mid-inference".

## Kill switch

Ctrl-C and `q` (when stdin is a terminal) set the same abort flag; the loop's `finally`
always sends `stop` and closes the transport. Works on Windows (signal handler, not
`loop.add_signal_handler`).

## Dry run

`--dry-run` sends nothing, and the trace names every verb a model *would* have run, with the
parameters it chose:

```
gate    dry_run: skipped would run search_scan, sent nothing (target='ball', step_deg=45, max_steps=8)
```

A parameter the model left unset shows as `null` rather than being dropped, because on a dry
run the omission is the thing you are checking. Read-only verbs (`observe`, alias
`get_frame`, and `report_state`) still run. Use it the first time you point a new `.duck` at
hardware.

## On hardware

Nothing here has run on hardware yet, on any body. If the body is a Microduck, run the contract
in the physics simulator first (`--robot microduck:mujoco`): it is the only place quackd can
show you a body that undershoots, refuses and falls over, and nothing there can be hurt. When
you do reach the robot, start with `--dry-run`
every time, then a `.duck` whose `allow` list is the smallest thing that could work, then
widen it. **You are responsible for your robot.**

**An XLeRobot (a 12 kg dual-arm cart):**

- **The watchdog stops the wheels and nothing else.** Upstream's 500 ms deadman calls
  `stop_base()`, so the fourteen arm and head servos keep holding their last goal under
  torque. `deadman_scope` says `base_only`, and that is the robot's entire safety authority.
- **Nothing reports a battery**, so a battery abort can never fire. The power station's
  switch is the only e-stop and it is not on the network.
- **The host exits by itself after an hour** with no supervisor anywhere upstream, so a long
  session ends as a heartbeat failure rather than an error.
- Blocks under the wheels until you have checked the turn direction: quackd converts rad/s to
  the deg/s the wire wants, and a wrong conversion is a 57x error.
- Work through [xlerobot-hardware-checklist.md](xlerobot-hardware-checklist.md).

**An AlohaMini (two arms on a 600 mm motorised lift):**

- **As upstream ships it the arms are limp**, so the safest bring-up is on the stock host,
  where the base and the lift can be exercised with no arm risk. quackd's own host wrapper is
  what turns torque on, and upstream's `disconnect()` turns it off again, so a loaded arm
  falls when that host exits.
- **The watchdog covers the base and the lift, never the arms** (`base_and_lift_only`).
- **`home()` leaves the lift travelling** at full speed, because the write that would zero
  that register is commented out upstream. quackd sends `stop` as its first command after
  connecting for exactly this reason.
- Clear the lift's whole travel before powering it. How fast it moves in mm/s is not stated
  anywhere upstream, so quackd's duration estimate for `lift` is an assumption.
- Work through [alohamini-hardware-checklist.md](alohamini-hardware-checklist.md).

**A ToddlerBot (a 56 cm, 3 kg humanoid):**

- **It cannot get up.** There is no get-up policy for this body at the pinned commit, so
  a fall ends the run and needs a human. Every moving verb refuses once it is down.
- **Work through [toddlerbot-hardware-checklist.md](toddlerbot-hardware-checklist.md) in
  order.** It keeps the feet off the ground until step 13, and steps 11 and 12 (pull the
  network cable mid-move, then send `SIGTERM`) are the two that matter most.
- **The deadman is a slew, not a stop.** There is no velocity at this hardware boundary:
  the command is an absolute pose. On silence the daemon quackd ships slews to the safe
  pose at upstream's own rate, waist first, and holds. It never goes limp, because on
  this body torque off is a fall.
- **quackd owns the control loop here**, which is true of no other body. Upstream's own
  `step()` is a no-op, so nothing times out and nothing re-arms without the daemon.
- A good first contract is the shipped `toddlerbot-lookout`: it moves no leg, no arm and
  no waist.

**A Microduck (a 25 cm biped):**

- **Run on the floor, not a table.** A 25 cm biped and a table edge do not mix.
- **Keep pets and kids clear of `kick`** (and `grab`, and `roulade`).
- **The gamepad preempts remote control.** Upstream arbitrates authority; there is no
  stop button because releasing the sticks stops the robot via the deadman. quackd does not
  try to out-rank the pad.
- quackd never sends `robot.relax` (torque off — the robot collapses) or `robot.init`
  (moves every joint). Use `robotctl` for those, with the robot on its stand.
- A good first contract: `allow: [quack, gaze, stop]`, then add walking.

**A LeRobot arm (an SO-101 class arm on a desk):**

- An arm sweeps a volume. Clear it before `move_joints`, and keep hands out of the path.
  A gripper is a pinch hazard even at the 50 % torque cap LeRobot writes at `configure()`.
- `pick` hands the whole arm to a learned policy for up to a minute. It is confirm-gated
  for that reason. Watch it, and keep `stop` within reach.
- `stop` holds position, it does not release. LeRobot's own `disconnect()` releases torque
  at the end of a session by its default, so the arm can sag when the run ends: do not
  leave it holding something fragile.
- Calibration is interactive and quackd never triggers it. An uncalibrated arm is refused.

**A wheeled base over rosbridge:**

- No deadman was verified anywhere in that stack, so if quackd dies mid-verb the base
  keeps its last Twist until its own driver times out, if it does at all. Test on blocks
  with the wheels off the ground before testing on the floor.
- quackd re-sends the Twist at 10 Hz while a verb runs and publishes a zero Twist on
  `stop`, on close, and when the heartbeat fails. That is the entire stop authority.
- The speed limits are quackd's caution (`limits.max_vx`, `max_wz` in the manifest), not
  the base's capability. Lower them before the first real drive.

**An Open Duck Mini v2:**

- **If it falls, quackd cannot pick it up, and on hardware it cannot tell that it has.**
  There is no get-up policy, so `stand_up` does not exist for it, and nothing on the bridge
  backend detects a fall: no verb refuses because the duck is down, and every observation
  says `fall-blind`. You are the fall detector: keep it on a stand until you trust the link.
- The deadman is quackd's own, and it runs on the robot. quackd's bridge daemon zeroes the
  velocity after 300 ms of silence, inside the call the control loop makes every tick, so a
  server thread that is starved, wedged or dead still stops the duck. Test it by pulling
  your laptop's Wi-Fi mid-walk before you rely on it.
- Going limp is unreachable rather than forbidden: the only channel from the network to the
  body is seven floats and a few buttons, so no message reaches a torque register.
- Head control is off unless you start the daemon with it on, and then it is clamped to 80
  percent of the runtime's own range and rate limited. Upstream warns that head control can
  break the head, and the four head values are offsets added to wherever the walk policy is
  holding the head, not absolute angles, so that clamp bounds an offset rather than a joint.
- The Feetech serial bus has exactly one owner. The bridge *is* the walk loop, so do not run
  it and upstream's script at the same time.
- The bridge binds loopback and wants a token, because a port that walks a robot on a shared
  network is a hazard. Prefer `ssh -L 9871:127.0.0.1:9871 your-pi`.
- The camera is a second process (`quackd_duck_camd.py`) serving one JPEG over HTTP with
  **no authentication at all**. It binds loopback and warns if you bind it wider, because
  it shows whatever the robot can see. Tunnel it rather than exposing it.
- The only e-stop is the power switch.
- The order to bring one up in, feet off the ground until step 10, with an abort condition
  at every step: [open-duck-hardware-checklist.md](open-duck-hardware-checklist.md).

## On other bodies

quackd drives more than the duck, and the honest answer to "what stops it when
quackd goes quiet" differs per body. Each manifest says so
(`safety_authority: {native, deadman}`), and `stop` always means stop, never collapse:

| Body | Native authority | What `stop` does | Never sent |
|---|---|---|---|
| Microduck (`microduck:*`) | `robotd_deadman`: velocity zeroes when intents stop | `robot.stop` | `robot.relax`, `robot.init` |
| LeRobot arm (`lerobot:*`) | `torque_limit`: the gripper's torque and current caps, plus `max_relative_target` when configured; no deadman, a position-controlled arm holds its goal | re-sends the present position as the goal (hold) | `disable_torque` (LeRobot's own `disconnect()` does, by its default, at the end of a session) |
| rosbridge base (`rosbridge:*`) | `none`: neither rosbridge nor the driver has a deadman we verified | publishes a zero Twist; quackd also re-sends the Twist at 10 Hz while a verb runs | silence |
| Open Duck Mini v2 (`open_duck:*`) | `none` in the robot, but quackd's own bridge daemon runs on it and zeroes the velocity after 300 ms of silence, inside the 50 Hz loop | zero velocity, head held, torque still on | anything that reaches torque, the head-control mode button, any direct servo or IMU read |
| XLeRobot (`xlerobot:*`) | `none`: the host's own 500 ms watchdog is real but calls `stop_base()`, which zeroes the three wheels and **nothing else**, so the 14 arm and head servos keep holding under torque. `deadman_scope` says `base_only` | zeroes the three velocity keys and leaves every arm goal exactly where it was, deliberately not rebuilding a hold from an unstamped reading that may be cycles old | `disconnect()`, which is upstream's torque-off, and any `enable(on=False)` |
| AlohaMini (`alohamini:*`) | `none`: the host's 1 s watchdog calls `stop_motion()`, which is the base and the lift and never the arms. `deadman_scope` says `base_and_lift_only` | one payload carrying all three velocity zeros **and** a lift velocity zero, because omitting either leaves the robot travelling | anything that disables arm torque. As shipped the arms are already limp, which is why the arm verbs need quackd's own host wrapper and refuse without it |
| ToddlerBot (`toddlerbot:*`) | `none` in the robot, and nothing upstream has a watchdog, timeout or e-stop at all. quackd's own daemon runs on it and after 500 ms of silence slews to the safe pose at upstream's own 0.3 rad/s, waist first, and **holds** | holds the last verified-good measured pose. There is no velocity at this hardware boundary, so `stop` cannot mean zero velocity | torque off, ever. Silence on this body means hold forever and torque off means fall, so the deadman is a trajectory rather than a message |

The verbs a body lacks are not gated, they do not exist: an arm cannot `move`, a base
cannot `say`, and `validate --robot` says so before a run starts.
`pick` on the arm is confirm-gated in its manifest because it hands the whole arm to a
controller quackd does not write.

## What quackd does not protect against

A model that is *allowed* to `walk` can walk into a wall; the sim has walls, your living
room has stairs. The allowlist is your tool: a `.duck` for a new space should start small.

There is one more thing to know about. A robot's memory
([memory.md](memory.md)) is text a model wrote, kept on disk, and handed to the *next*
model as part of its system prompt. The executor never reads it, so a note cannot widen an
allowlist, lift a budget or open a confirm gate: none of the guarantees above depend on it
being true. What a note can do is persuade a later run, including a later run of a different
task on the same body. A model that concludes something wrong ("the sofa is safe to walk
under") will keep telling itself so until somebody deletes the line. That is the whole point
of the feature and also its whole risk, which is why the file is plain text you can read,
`quackd memory show` prints exactly what the pilot was told, `quackd memory clear` forgets
it, and `--no-memory` runs as if it were never there.

There is a second thing that is not about the body. `--robot microduck:mujoco` downloads an
MJCF, 38 meshes and two ONNX policies from GitHub and the Hugging Face Hub the first time it
runs, into `~/.quackd/cache`, and then runs those policies in quackd's own process. What guards
that: the commit and the revision are pinned in `quackd/sim3d/upstream_api.py`, every file is
checked against a sha256 recorded when it was read and a mismatch is a refusal, and the tarball
is unpacked by name against a fixed list rather than by whatever it contains. What does not:
`QUACKD_MICRODUCK_ASSETS` points quackd at a checkout of your own, and there a file that differs
from the pin is a warning and the run continues, which is deliberate, because a newer export is
what somebody with a checkout usually wants. The state says which you got
(`extras.model_pinned`), so the transcript records it. Nothing in this path reaches a robot: the
physics backend has no address and drives nothing outside the process.

Report anything that lets a model bypass the executor — see [`SECURITY.md`](../SECURITY.md).
