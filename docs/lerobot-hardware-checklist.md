# A LeRobot SO-101 arm: the order to try it in

Nothing in quackd has run on an SO-101. This is the order to find out in, written so that
each step can only fail in a way you can recover from. **Nothing moves until step 10, and
from there a hand stays on the power switch.**

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

   Two rows matter: `lerobot`, and `lerobot (feetech bus)`. The Feetech SDK lives in
   lerobot's own `[feetech]` extra rather than its base dependencies, so a lerobot installed
   without it imports perfectly and then cannot open a serial port. If `lerobot` itself stays
   `not installed` after a successful install, check `python --version`: the extra carries a
   `python_version >= '3.12'` marker and resolves to nothing below that.

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
   folded somewhere awkward will fall at either end.
7. **`lerobot-lookout`.** It moves no joint.

   ```bash
   uv run quackd run lerobot-lookout --robot lerobot:real --address /dev/ttyACM0
   ```

   It reads the arm and reports where the joints are, whether torque is on and whether
   anything is hot. This is the first thing to point at a real arm.

8. **Add a camera, if you brought one.** No SO-101 has one built in: it is a USB webcam into
   the laptop, and the arm's own cable carries no video. Find which index it is, which is the
   part nobody can guess for you:

   ```bash
   lerobot-find-cameras opencv
   ```

   It lists every camera it can open and saves a frame from each under
   `outputs/captured_images/`. Open the pictures: on a laptop index 0 is usually the built-in
   webcam, so the one you plugged in is often 1 or 2. Then ask quackd for a frame:

   ```bash
   uv run quackd doctor --robot lerobot:real --address /dev/ttyACM0 --camera-url "opencv://1"
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
   uv run quackd serve-mcp --robot lerobot:real --address /dev/ttyACM0 --camera-url "opencv://1"
   ```

   Then `robot_observe` from the client. What comes back is the frame and a line of
   detections, and on a real desk that line is usually nothing: the detector's colour ranges
   are the simulator's, which [the adapter page](adapters/lerobot.md#camera) explains.

   A camera is optional: the arm works without one, and `lerobot-lookout` never asks for it.

9. **Rehearse the whole thing with `--dry-run`, which sends the arm nothing.** This is the
   last step before anything moves, and it is the one that tells you whether the parts you
   cannot see are working.

   ```bash
   uv run quackd run --goal "roll the wrist ten degrees, then stop"      --robot lerobot:real --address /dev/ttyACM0 --provider anthropic --max-steps 3 --dry-run
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

## Moving, one joint at a time

There is no command that runs one verb. Either drive the daemon from an MCP client
(`quackd serve-mcp --robot lerobot:real --address /dev/ttyACM0`, then `robot_run_verb`, which is
what these steps assume, and add the same `--camera-url` if you did step 8, because a session
without it has no `observe`) or give a model a goal narrow enough to reach one verb
(`quackd run --goal "..." --robot lerobot:real --address /dev/ttyACM0 --provider anthropic
--max-steps 3`). `--provider fake` will not do: it answers a free-form goal with a fixed
script that ignores it.

10. **`gripper` open, then closed on nothing, and watch which way it goes.** quackd assumes
    100 is open and 0 is closed, and that is an assumption about how your arm was assembled
    and calibrated, not a fact about the model. If yours runs the other way, stop here and
    say so in an issue: everything quackd believes about holding something rests on this, and
    it would be believing the opposite.
11. **One joint, small, in the middle of its range.** `move_joints` with
    `{"wrist_roll": 10}`. It should take about a fifth of a second and stop. The arm moves at
    5 degrees per action re-sent ten times a second, so 50 degrees a second, and
    `QUACKD_LEROBOT_MAX_STEP_DEG` lowers that if it looks fast in the room.
12. **Ask for something out of range.** A goal of 170 on a joint whose travel is about 100
    either way. It is refused with the range in the reason and nothing reaches the arm. This
    is worth doing deliberately, because LeRobot does not clamp a degrees goal and the servo
    is the only thing downstream of it. Use any body joint **except `wrist_roll`**: that one
    is recorded as a full turn at calibration, so its range is -180..180 and this refusal
    cannot fire on it.
13. **Pull the USB cable mid-move.** The run should end within about a second, saying the arm
    did not answer. The arm holds its last goal under torque: it must not sag and it must not
    carry on. Plug it back in and reconnect before the next step.
14. **Ctrl-C mid-move.** quackd's kill switch sends `stop`, which re-sends the present
    position as the goal. The arm should freeze where it is rather than sag, and rather than
    finish the motion it was in the middle of.

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
The six things that most need a real arm:

- **Which end of the gripper's 0..100 range is open.** Assumed, and everything about holding
  depends on it.
- **Whether the holding band is anywhere near right.** quackd calls it holding when the
  gripper is told to close, settles, and settles between 8 and 90 of 100. Nobody has ever
  seen what a real grasp reads.
- **What a joint reads in degrees Celsius**, cold and after ten minutes of work. quackd
  refuses to move at 60, below the servo's own 70 cut-off, and both numbers are Feetech's
  documentation rather than anything measured here.
- **Whether 5 degrees an action felt right.** The figure is quackd's own choice for a first
  run, not anything upstream recommends for this arm, and it has never been watched.
- **Whether a stall is caught.** Hold a joint gently against its goal and see whether the verb
  fails with where it stopped.
- **Which OpenCV index the camera turned out to be**, if you brought one, and whether it
  needed `?backend=msmf`. Nobody has pointed quackd at a webcam either.

Only flip the `real` row in [adapter-status.md](adapter-status.md) once a real arm has done
it, and say in the same commit what it did.
