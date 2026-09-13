# A LeRobot SO-101 arm: the order to try it in

Nothing in quackd has run on an SO-101. This is the order to find out in, written so that
each step can only fail in a way you can recover from. **Nothing moves until step 8, and
from there a hand stays on the power switch.**

This robot is unusual in a quiet way, and the quiet thing is what makes the order matter:
**LeRobot writes a torque and current cap on the gripper and on nothing else.** The five
body joints run with whatever their firmware defaults to, so the elbow has no cap to save
your finger or its own gears. An arm also sweeps a volume rather than occupying a spot, and
a gripper is a pinch hazard at any torque. Read
[adapters/lerobot.md](adapters/lerobot.md) first.

## Before you power anything

1. **Know which build you have.** The same arm ships with 7.4 V servos and with 12 V ones,
   with different stall torque and different supplies. Match the supply to the build before
   anything is energised: the wrong one is either an arm that cannot hold itself up or an
   arm with half again the torque you were expecting.
2. **Clear the whole sweep**, not the footprint. Take anything fragile out of the gripper and
   off the desk within arm's length, and keep hands out of the volume from here on.
3. **Fit a switch you can reach.** There is no e-stop on an SO-101 and quackd cannot give it
   one: cutting the servo supply is the only thing that stops this arm in every case,
   including the one where the controlling process has died with a goal still standing. Put
   an inline switch on that supply, on the side of the desk you will be standing on.
4. **Calibrate with upstream's own tool**, because it is interactive and quackd never
   triggers it:

   ```bash
   lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=arm-01
   ```

   It writes `<calibration dir>/robots/so_follower/<id>.json`, where the calibration
   directory is `$HF_LEROBOT_CALIBRATION`, else `$HF_LEROBOT_HOME/calibration`, else
   `$HF_HOME/lerobot/calibration`. **The id has to be the one quackd will use**, which is
   `arm-01` unless you named the robot with `--robots <name>=lerobot:real`. quackd reads
   every joint's travel out of that file and refuses to drive an arm without one, and two
   arms sharing an id share a file with nothing in it to say which arm it came from.
5. **Install the extra and check both halves are there.**

   ```bash
   uv pip install 'quackd[lerobot]'      # Python 3.12 or newer, and it pulls torch
   uv run quackd doctor
   ```

   Two rows matter: `lerobot`, and `lerobot (feetech bus)`. The Feetech SDK lives in
   lerobot's own `[feetech]` extra rather than its base dependencies, so a lerobot installed
   without it imports perfectly and then cannot open a serial port. On Windows the port is
   `COMx` and needs the board's USB-to-UART driver, usually CH340 or CP210x; on Linux it is
   `/dev/ttyACM0` and you need to be in the right group to open it.

## The host, with the arm still

6. **Connect and read the arm back.**

   ```bash
   uv run quackd doctor --robot lerobot:real --address /dev/ttyACM0
   ```

   This one connects, unlike `list-verbs`, which does not. Read four things off it: that the
   calibration file it found is the one you just wrote, that each joint's range looks like
   the travel you swept during calibration, that torque is on, and what the servos say their
   temperature is with the arm cold. That last number is the baseline for every later step.

   **Support the arm before this command finishes.** LeRobot's `disconnect()` disables
   torque by default, so the arm goes limp at the end of every clean session, a `doctor`
   probe included. An arm folded somewhere awkward will fall when the probe returns.
7. **`lerobot-lookout`.** It moves no joint.

   ```bash
   uv run quackd run lerobot-lookout --robot lerobot:real --address /dev/ttyACM0
   ```

   It reads the arm and reports where the joints are, whether torque is on and whether
   anything is hot. This is the first thing to point at a real arm.

## Moving, one joint at a time

There is no command that runs one verb. Either drive the daemon from an MCP client
(`quackd serve-mcp --robot lerobot:real --address /dev/ttyACM0`, then `robot_run_verb`, which is
what these steps assume) or give a model a goal narrow enough to reach one verb
(`quackd run --goal "..." --robot lerobot:real --address /dev/ttyACM0 --provider anthropic
--max-steps 3`). `--provider fake` will not do: it answers a free-form goal with a fixed
script that ignores it.

8. **`gripper` open, then closed on nothing, and watch which way it goes.** quackd assumes
   100 is open and 0 is closed, and that is an assumption about how your arm was assembled
   and calibrated, not a fact about the model. If yours runs the other way, stop here and say
   so in an issue: everything quackd believes about holding something rests on this, and it
   would be believing the opposite.
9. **One joint, small, in the middle of its range.** `move_joints` with
   `{"wrist_roll": 10}`. It should take about a fifth of a second and stop. The arm moves at
   5 degrees per action re-sent ten times a second, so 50 degrees a second, and
   `QUACKD_LEROBOT_MAX_STEP_DEG` lowers that if it looks fast in the room.
10. **Ask for something out of range.** A goal of 170 on a joint whose travel is about 100
    either way. It is refused with the range in the reason and nothing reaches the arm. This
    is worth doing deliberately, because LeRobot does not clamp a degrees goal and the servo
    is the only thing downstream of it.
11. **Pull the USB cable mid-move.** The run should end within about a second, saying the arm
    did not answer. The arm holds its last goal under torque: it must not sag and it must not
    carry on. Plug it back in and reconnect before the next step.
12. **Ctrl-C mid-move.** quackd's kill switch sends `stop`, which re-sends the present
    position as the goal. The arm should freeze where it is rather than sag, and rather than
    finish the motion it was in the middle of.

## The gripper, and only then a policy

13. **Close the gripper on something soft and forgiving.** A foam block, not a cup. It should
    stop short of shut, `report_state` should say it is holding, and `stop` should not drop
    it: a hold deliberately leaves the gripper's goal alone. Then `place` to let go.
14. **`pick`, only if you have a policy for this arm**, and only with somebody watching. It
    is confirm-gated because it hands the whole arm to a controller quackd did not write, for
    up to a minute. Keep the hand on the switch, and remember that the policy's actions are
    capped and range-refused exactly like a verb's, which is quackd's rule rather than
    LeRobot's.

## What to report

Open an issue with the transcript and `quackd doctor` output. The five things that most need
a real arm:

- **Which end of the gripper's 0..100 range is open.** Assumed, and everything about holding
  depends on it.
- **Whether the holding band is anywhere near right.** quackd calls it holding when the
  gripper is told to close, settles, and settles between 8 and 90 of 100. Nobody has ever
  seen what a real grasp reads.
- **What a joint reads in degrees Celsius**, cold and after ten minutes of work. quackd
  refuses to move at 60, below the servo's own 70 cut-off, and both numbers are Feetech's
  documentation rather than anything measured here.
- **Whether 5 degrees an action felt right.** It is upstream's own suggested figure and it
  has never been watched.
- **Whether a stall is caught.** Hold a joint gently against its goal and see whether the verb
  fails with where it stopped.

Only flip the `real` row in [adapter-status.md](adapter-status.md) once a real arm has done
it, and say in the same commit what it did.
