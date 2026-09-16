# A LeRobot SO-101 arm: the order to try it in

quackd has run on an SO-101 once, on 2026-09-15: one arm, one bench, one afternoon, and
[adapter-status.md](adapter-status.md) lists exactly what it did and what fell over. That
makes this a robot quackd has worked on, not a robot quackd is tested on, and four of the
questions at the foot of this page came back from that day still unanswered. This is the
order to find out in, written so that each step can only fail in a way you can recover from.
**Nothing moves until step 10, and from there a hand stays on the power switch.**

This robot is unusual in a quiet way, and the quiet thing is what makes the order matter:
**LeRobot writes a torque and current cap on the gripper and on nothing else.** The five
body joints run with whatever their firmware defaults to, so the elbow has no cap to save
your finger or its own gears. An arm also sweeps a volume rather than occupying a spot, and
a gripper is a pinch hazard at any torque. Read
[adapters/lerobot.md](adapters/lerobot.md) first, or
[lerobot-first-run.md](lerobot-first-run.md) if you have never run quackd or LeRobot at all:
it is the same ground at walking pace, and it hands back to this file the moment anything is
about to move.

## Before you power anything

1. **Know which build you have, and give it the supply its own parts list names.** An
   SO-101 is sourced from a bill of materials rather than bought as one thing, and the
   motors come in more than one variant, with different stall torque and different supplies.
   Upstream's assembly page sends you to
   [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) for that list, and
   it is the authority on which supply yours takes. Neither LeRobot nor quackd reads the
   voltage, so neither can warn you: the wrong supply is either an arm that cannot hold
   itself up or an arm with more torque than you were expecting.
2. **Clear the whole sweep**, not the footprint. Take anything fragile out of the gripper and
   off the desk within arm's length, and keep hands out of the volume from here on.
3. **Fit a switch you can reach.** There is no e-stop on an SO-101 and quackd cannot give it
   one: cutting the servo supply is the only thing that stops this arm in every case,
   including the one where the controlling process has died with a goal still standing. Put
   an inline switch on that supply, on the side of the desk you will be standing on.
4. **Install the extra and check both halves are there.** Nothing is energised by this
   step, and it comes before the calibration below because `lerobot-calibrate` is one of the
   commands it installs.

   ```bash
   uv pip install 'quackd[lerobot]'      # Python 3.12 or newer, and it pulls torch
   uv run quackd doctor
   ```

   That extra is two packages rather than one: `quackd-lerobot`, the arm adapter, which
   installs on any Python quackd supports, and through its own `[sdk]` extra, LeRobot. A
   plain `uv pip install quackd` brings neither, and no other robot's extra does either.

   Two rows matter: `lerobot`, and `lerobot (feetech bus)`. The Feetech SDK lives in
   lerobot's own `[feetech]` extra rather than its base dependencies, so a lerobot installed
   without it imports perfectly and then cannot open a serial port. If `lerobot` itself stays
   `not installed` after a successful install, check `python --version`: the SDK half carries
   a `python_version >= '3.12'` marker and resolves to nothing below that, which leaves you
   the adapter, `lerobot:mock` and no arm.

## First power: the port, then the calibration

This is where the arm is energised for the first time, so the sweep from step 2 has to be
clear and the switch from step 3 has to be fitted and within reach before you start.

5. **Find the port, then calibrate with upstream's own tool**, which is interactive and which
   quackd never triggers:

   ```bash
   lerobot-find-port
   ```

   It lists the ports, asks you to unplug the arm, and names the one that disappeared. That is
   worth doing even when you are sure, because it is the only answer that is not a guess. On
   Windows the port is `COMx` and shows up by itself under Ports (COM & LPT); if nothing
   appears at all, suspect the cable or the power before you go looking for a driver. On Linux
   it is `/dev/ttyACM0`, and upstream's own fix for permissions is `sudo chmod 666
   /dev/ttyACM0`, with adding your user to that port's group being the version that survives a
   reboot.

   ```bash
   lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=arm-01
   ```

   It writes `<calibration dir>/robots/so_follower/<id>.json`, where the calibration
   directory is `$HF_LEROBOT_CALIBRATION`, else `$HF_LEROBOT_HOME/calibration`, else
   `$HF_HOME/lerobot/calibration`. **The id has to be the one quackd will use**, which is
   `arm-01` unless you named the robot with `--robots <name>=lerobot:real` or registered it
   with `quackd robot add <name> lerobot:real`. quackd reads every joint's travel out of that
   file and refuses to drive an arm without one, and two arms sharing an id share a file with
   nothing in it to say which arm it came from.

   Upstream will ask you to move every joint through its range **except `wrist_roll`**, and it
   records a full encoder turn for that one rather than anything you swept. That is not a
   mistake you can correct, and it is why the out-of-range refusal you will test in step 12
   works on the other joints and cannot work on that one.

## The host, with the arm still

6. **Connect and read the arm back.**

   ```bash
   uv run quackd doctor --robot lerobot:real --address /dev/ttyACM0
   ```

   This one connects, unlike `list-verbs`, which does not. Read four things off it: that the
   calibration file it found is the one you just wrote, that each joint's range looks like
   the travel you swept during calibration, that torque is on, and what the servos say their
   temperature is with the arm cold. That last number is the baseline for every later step.

   **Support the arm before this command finishes, and while it starts.** `configure()` runs
   with torque off, so connecting drops it for a moment, and LeRobot's `disconnect()` disables
   it again by default at the end of every clean session, a `doctor` probe included. An arm
   folded somewhere awkward will fall at either end. The second half of that is what the rest
   pose below is for, and until one is recorded this probe still lets go.

   **Then give the arm a name.** A registered name carries the port, the camera and the rest
   pose, so from step 7 on this page writes `--robot arm-01` where it used to write the pair:

   ```bash
   uv run quackd robot add arm-01 lerobot:real --address /dev/ttyACM0
   ```

   The name has to be the calibration id from step 5. `--robot lerobot:real --address
   /dev/ttyACM0` still works everywhere below and reaches the same arm, with one difference
   that starts mattering here: a rest pose is recorded against a name, so the spec form has
   none and goes on letting go of the arm wherever it stops.

   **Then fold it by hand and record where it rests.** Nothing is connected now, so the arm is
   limp. Fold it into the shape you want it to end every run in: low, resting on its own stops
   or on the desk, a shape it holds with torque off and cannot topple out of. Then:

   ```bash
   uv run quackd robot rest-pose arm-01
   ```

   It connects, reads every joint, prints them, and asks before it writes anything. **It drives
   nothing.** That is why it belongs at step 6 rather than after step 10: recording a pose is a
   read. The block below was captured against `lerobot:mock`, with `--yes`, which is the flag
   that skips the question. Those are the mock arm's joints, and yours are whatever you folded
   it to:

   ```
   $ quackd robot rest-pose arm-01 --yes
   arm-01 (lerobot:mock) is at
   shoulder_pan   0.0
   shoulder_lift  -90.0
   elbow_flex     90.0
   wrist_flex     0.0
   wrist_roll     0.0
   gripper        100.0
   ✓ recorded arm-01's rest pose (6 joints)
     quackd run <duck> --robot arm-01 starts from it and returns to it before letting go
   ```

   A folded arm usually sits **outside** the travel your calibration recorded: the arm this was
   written for folded to `shoulder_lift` -113.5 against a calibrated range of about plus or
   minus 84.2. So the rest pose is sent unclipped, and the out-of-range refusal you will test in
   step 12 does not apply to it. That is deliberate, and it is the one place the refusal is
   deliberately out of the way: a refusal there would be a refusal to put the arm down.

   The gripper is recorded and never driven, for the same reason `stop` leaves it alone.
   Re-sending it would open a hand that is holding something. Only the five body joints move.

   What each command does with the pose from here on:

   | What you run | What it does about the pose |
   |---|---|
   | `quackd run ... --robot arm-01` | drives the arm there before the pilot gets its first turn, and a run that cannot get there aborts before the first LLM call. Returns it there between the final `stop` and the disconnect, on every exit: success, failure, infeasible, budget, abort, an error, Ctrl-C |
   | `quackd doctor --robot arm-01` | probes the arm, returns it there afterwards, and says which in a `rest pose` row |
   | `quackd robot list --probe` | reads the arm and never moves it, so it says `torque left on: not at its rest pose` when it had to keep the arm up |
   | `quackd serve-mcp --robot arm-01` | the same at both ends, and it refuses to start if it cannot get there |
   | `--dry-run` | nothing at all: a dry run never moves the arm |

   Torque is released only where the arm is known to be at that pose. Anywhere else quackd turns
   LeRobot's `disconnect()` flag off, leaves the arm holding itself up, and says so once:

   ```
   the arm is not at its rest pose (...), so torque was left on and it will not fall: hold the
   arm and cut its power, or run again
   ```

   > [!WARNING]
   > That is a change in behaviour and it is the one to read twice. A probe or a dry run on an
   > arm away from its recorded rest pose now leaves torque **on** where it used to drop it.
   > The arm will not fall, and it will also not let go until you hold it and cut its power, or
   > until a run puts it back.

   The `rest pose` row in `doctor` says which of four things happened:

   | Row | What it means |
   |---|---|
   | `none recorded (quackd robot rest-pose <name>)` | nothing is recorded, so this probe let go wherever the arm stood |
   | `at it already` | the arm was there, and torque was released there |
   | `returned to it` | the probe drove it back and let go there |
   | `not reached: <reason>` | it could not get there, so torque is still on |

   `quackd robot rest-pose arm-01 --clear` forgets the pose again and says what that costs: a
   run then leaves the arm where it stands, and torque drops there.

   > [!IMPORTANT]
   > Step 6 keeps the promise at the top of this page, because recording reads the arm and
   > drives nothing. What you do next can break it. The arm is now **at** its rest pose, so
   > steps 7 to 9 find it there and drive nothing either. Move it by hand in between and the
   > first thing step 7 does is drive it back, which is motion before step 10. Leave it folded,
   > or record the pose again wherever it now stands.

7. **`lerobot-lookout`.** No verb in it moves a joint.

   ```bash
   uv run quackd run lerobot-lookout --robot arm-01
   ```

   It reads the arm and reports where the joints are, whether torque is on and whether
   anything is hot. This is the first thing to point at a real arm, and it is what the arm on
   the bench ran first on 2026-09-15.

   The rest pose is the only thing in this run that can move the arm, and the run prints a note
   at each end saying which it did, `moving to the rest pose` or `already at the rest pose`. On
   an arm still folded where you left it in step 6, both notes say it was already there and
   nothing moves.

8. **Add a camera, if you brought one.** No SO-101 has one built in: it is a USB webcam into
   the laptop, and the arm's own cable carries no video. Find which index it is, which is the
   part nobody can guess for you:

   ```bash
   lerobot-find-cameras opencv
   ```

   It lists every camera it can open and saves a frame from each under
   `outputs/captured_images/`. Open the pictures: on a laptop index 0 is usually the built-in
   webcam, so the one you plugged in is often 1 or 2. On the bench on 2026-09-15 it was
   `opencv://1`, and `opencv://2` after a replug, at 640x480, with no `?backend=` key needed.
   That is one laptop's answer and not a prediction about yours. Then ask quackd for a frame:

   ```bash
   uv run quackd doctor --robot arm-01 --camera-url "opencv://1"
   ```

   Quote the url: a bare `&` is a parse error in PowerShell and backgrounds the command in
   bash. A
   camera you asked for and did not get is a refusal, and it happens before the arm is
   touched, so a wrong index costs you nothing but the message: try another index, add
   `?backend=msmf` if the camera listed and would not open, or drop a size or a rate you
   pinned and let it keep its own mode, which is the default. When it does open, `doctor`
   grows a `camera` row with the frame size in it, or `no frame` if it opened and then gave
   nothing. Then see what the pilot will see, which means an MCP session carrying the same
   url, because a daemon started without it has no `observe` verb at all:

   ```bash
   uv run quackd serve-mcp --robot arm-01 --camera-url "opencv://1"
   ```

   Then `robot_observe` from the client. What comes back is the frame and a line of
   detections, and on a real desk that line is usually nothing: the detector's colour ranges
   are the simulator's, which [the adapter page](adapters/lerobot.md#camera) explains.

   Aim it at what you want watched, and check what it crops. On the bench it framed the
   gripper and cut off the raised arm, so the model ended up verifying its own waves from the
   joint readings rather than from the picture, which is a thing it can do and not a thing you
   should count on.

   A camera is optional: the arm works without one, and `lerobot-lookout` never asks for it.

   **More than one camera.** `--camera-url` repeats, and this arm is the only body that reads
   more than one. Every other body refuses a second one and names who takes several.

   ```bash
   uv run quackd robot edit arm-01 \
       --camera-url "opencv://1?name=top" --camera-url "opencv://2?name=side"
   ```

   | Rule | Why |
   |---|---|
   | with several, every url carries `?name=` and the names are unique | the name is what the model, a pick policy's observation and `frames/NNNN-<name>.png` tell the views apart by |
   | an index may not repeat | two entries for one camera is a typo, not a setup |
   | the **first** url is the primary | it is the camera `--fov-deg` describes, the one the `camera:` detections line reports, and the only one the verbs that steer by sight read: they run at 10 Hz, and fetching every camera there would blow the deadman window |
   | a camera that stalls later costs its own picture and nothing else | `report_state` and `doctor` then name which one, in a `camera <name>` row each |
   | if the **primary** is the one that died, the others still reach the model and the detections line reports nothing seen | a bearing read off a different lens would point somewhere else |
   | a second camera that will not open refuses before the arm is energised | and it lets go of the first, so a wrong index still costs you nothing but the message |

   Get the `?name=` wrong and you get the rule back, before anything is touched. The first part
   of it, the rest being the list of keys a camera url takes:

   ```
   ✗ error: lerobot:real at COM5: lerobot real: --camera-url 'opencv://1': it has no ?name= and 2
   cameras were given. With several, every url names its own camera, opencv://1?name=top
   --camera-url opencv://2?name=side, because the name is what the model, a pick policy and
   frames/NNNN-<name>.png tell them apart by.
   ```

   Every frame reaches the model each step, labelled with its camera name, on Claude, both
   OpenAI APIs, Gemini, and any OpenAI-compatible local server with `--vision` on. **It costs
   what it sounds like.** The last two exchanges keep their images, so two cameras is four
   pictures in every request rather than two, on every step, for the length of the run. A local
   server or a model that takes one image per message will refuse outright, and the answer
   there is a single `--camera-url`.

9. **Rehearse the whole thing with `--dry-run`, which sends the arm nothing.** This is the
   last step before anything moves, and it is the one that tells you whether the parts you
   cannot see are working.

   ```bash
   uv run quackd run --goal "roll the wrist ten degrees, then stop" \
       --robot arm-01 --provider anthropic --max-steps 3 --dry-run
   ```

   A dry run connects to the arm for real and holds the connection open. Read-only verbs
   actually run, so `report_state` reads the servos and the heartbeat keeps its round trip
   going the whole time; everything that would move a joint is printed and skipped:

   ```
   [dry-run] would run move_joints({'positions': {'wrist_roll': 10.0}, 'duration_s': 2.0})
   [dry-run] move_joints not sent
   ```

   Read two things off it. That the model reached for the verb you expected, with arguments
   that look sane, rather than for something you had not thought about. And that the arm
   answered every heartbeat for the length of the run, because an arm that drops out here
   would have dropped out mid-move in the next section. This costs an API call or three and
   is the cheapest rehearsal you will get.

   Two dry runs on the bench on 2026-09-15 ended early, and both endings were the machinery
   working rather than failing. One stopped with `the arm did not answer: TimeoutError` after a
   single heartbeat round trip failed, and nothing like it happened again all afternoon: that
   is the abort you want, at the cheapest moment to get it. The other stopped before it began,
   because the pilot answered `uncertain` to the feasibility question and the human at the
   keyboard said no.

   A dry run never moves the arm, and that includes the rest pose: it neither drives the arm
   there at the start nor puts it back at the end. So an arm that was away from its rest pose
   when you started is still away from it when the dry run finishes, and quackd keeps torque on
   rather than dropping the arm there.

## Moving, one joint at a time

There is no command that runs one verb. Either drive the daemon from an MCP client
(`quackd serve-mcp --robot arm-01`, then `robot_run_verb`, which is what these steps assume.
The camera comes with the name if you stored it in step 8, and a session with no camera has no
`observe` verb at all) or give a model a goal narrow enough to reach one verb
(`quackd run --goal "..." --robot arm-01 --provider anthropic --max-steps 3`).
`--provider fake` will not do: it answers a free-form goal with a fixed script that ignores it.

Both of those drive the arm to its rest pose before you get a turn, and back to it before they
let go. An MCP session refuses to start at all if it cannot get there, which is the same rule
as the run's, moved to the moment the daemon comes up.

10. **`gripper` open, then closed on nothing, and watch which way it goes.** quackd assumes
    100 is open and 0 is closed, and that is an assumption about how your arm was assembled
    and calibrated, not a fact about the model. If yours runs the other way, stop here and
    say so in an issue: everything quackd believes about holding something rests on this, and
    it would be believing the opposite. The bench arm on 2026-09-15 ran the way quackd assumes:
    commanded 100 it reported 98 and stood open, commanded closed it reported 3 with the jaws
    nearly touching. That is one arm, built and calibrated by one person, which is why this
    step is still here.
11. **One joint, small, in the middle of its range.** `move_joints` with
    `{"wrist_roll": 10}`. It should take about a fifth of a second and stop. The arm moves at
    5 degrees per action re-sent ten times a second, so 50 degrees a second, and
    `QUACKD_LEROBOT_MAX_STEP_DEG` lowers that if it looks fast in the room. For scale, the
    free-form goals on the bench on 2026-09-15 came out as wrist-roll waves of about plus or
    minus 27 degrees, and as wider poses with `shoulder_lift` at -39 and `elbow_flex` between
    24 and 30. Nobody wrote down whether 50 degrees a second looked right standing next to it,
    which is why that question is still at the foot of this page.
12. **Ask for something out of range.** A goal of 170 on a joint whose travel is about 100
    either way. It is refused with the range in the reason and nothing reaches the arm. This
    is worth doing deliberately, because LeRobot does not clamp a degrees goal and the servo
    is the only thing downstream of it. Use any body joint **except `wrist_roll`**: that one
    is recorded as a full turn at calibration, so its range is -180..180 and this refusal
    cannot fire on it.
13. **Pull the USB cable mid-move.** The run should end within about a second, saying the arm
    did not answer. The arm holds its last goal under torque: it must not sag and it must not
    carry on. There is no rest pose in this one, because quackd cannot drive an arm it cannot
    reach: the arm stays where it stopped, holding itself up, which is the safe half of the two
    ways this could end. Plug it back in and reconnect before the next step.
14. **Ctrl-C mid-move.** quackd's kill switch sends `stop`, which re-sends the present
    position as the goal. The arm should freeze where it is rather than sag, and rather than
    finish the motion it was in the middle of. Then, with a rest pose recorded in step 6, the
    freeze is not the end of it: quackd folds the arm back to that pose and lets go there,
    because that is the one place letting go is safe. Expect that motion, and keep the hand on
    the switch through it.

## The gripper, and only then a policy

15. **Close the gripper on something soft and forgiving.** A foam block, not a cup. It should
    stop short of shut, `report_state` should say it is holding, and `stop` should not drop
    it: a hold deliberately leaves the gripper's goal alone. Then `place` to let go.
16. **`pick` cannot be reached from the CLI or from MCP today, and this step is here to say
    so rather than to be done.** The verb exists on `lerobot:real` only when a policy object
    was handed to the transport in Python, and nothing in quackd hands it one: there is no
    flag, `make()` has no policy parameter, and `load_policy()` has no caller outside a test.
    So a trained checkpoint reaches this arm only through code you write around the adapter.
    If you do write it, `pick` is confirm-gated because it hands the whole arm to a controller
    quackd did not write for up to a minute, and the policy's own actions are step-capped and
    range-refused exactly like a verb's, which is quackd's rule rather than LeRobot's. Keep a
    hand on the switch, and please report what happened.

## What to report

[Open a LeRobot hardware report](https://github.com/rokbenko/quackd/issues/new?template=lerobot-hardware-report.yml),
which asks for exactly the list below, or a plain issue with the transcript and `quackd
doctor` output. A report that says it did not work is worth as much as one that says it did.

**Four things one afternoon on one bench did not answer**, and which still need a real arm:

- **Whether the holding band is anywhere near right.** quackd calls it holding when the
  gripper is told to close, settles, and settles between 8 and 90 of 100. Nobody has yet seen
  what a real grasp reads. The gripper on 2026-09-15 closed on air, not on an object.
- **What a joint reads in degrees Celsius**, cold and after ten minutes of work. quackd
  refuses to move at 60, below the servo's own 70 cut-off, and both numbers are Feetech's
  documentation rather than anything measured here. Nothing on the bench ran long enough to
  find out what the second number is.
- **Whether 5 degrees an action felt right.** The figure is quackd's own choice for a first
  run, not anything upstream recommends for this arm, and nobody has said whether it looked
  right standing next to the arm.
- **Whether a stall is caught.** Hold a joint gently against its goal and see whether the verb
  fails with where it stopped. Nobody has done this on purpose yet.

**And two with one answer each, from one arm on one laptop.** A second answer is what turns
either of them from an anecdote into a fact:

- **Which end of the gripper's 0..100 range is open.** quackd assumes 100. That arm agreed
  (step 10), and which end yours opens at is still a question about how it was built and
  calibrated rather than about the model.
- **Which OpenCV index the camera turned out to be**, if you brought one, and whether it
  needed `?backend=msmf`. That laptop's webcam was `opencv://1`, then `opencv://2` after a
  replug, at 640x480, and it needed no `?backend=` key. Nobody can guess yours.

The `real` row in [adapter-status.md](adapter-status.md) was flipped on 2026-09-15, and it says
what that arm did and what fell over afterwards. The next report either widens that row or
contradicts it, and the one that contradicts it is worth more.
