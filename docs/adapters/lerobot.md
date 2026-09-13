# LeRobot (an SO-101 class arm)

A six-joint desktop arm with a parallel gripper, driven through
[LeRobot](https://github.com/huggingface/lerobot). No legs, no head, no voice, so its
manifest lists none of that: `move`, `go_to`, `search_scan`, `say` and `gaze` do not exist
on this robot. What it has is joints, a gripper, `place`, and, when a policy is available,
`pick` as one skill intent that the arm's own learned policy executes. The thesis holds:
the LLM picks the verb, LeRobot moves the arm, quackd enforces the contract.

Upstream pinned at
[`fbb811f`](https://github.com/huggingface/lerobot/tree/fbb811fca92504439792b97d216f0d00c2268382)
(`main`, 2026-09-01), first read 2026-09-02 and read again on 2026-09-13. Every name quackd
spells lives in
[`quackd/adapters/lerobot/upstream_api.py`](../../quackd/adapters/lerobot/upstream_api.py).

**The `real` backend has never been run against an arm by us.**

```bash
# offline, the default
uv run quackd run lerobot-lookout --robot lerobot:mock --provider fake

uvx quackd list-verbs --robot lerobot:mock
uvx quackd validate ducks/find-and-kick.duck --robot lerobot:mock     # exit 1: requires ... does not provide it
uvx quackd serve-mcp --robots arm=lerobot:mock,duck=microduck:sim2d   # an arm and a duck behind one MCP server

# a real arm, after LeRobot's own calibration (see the checklist below)
uv pip install "quackd[lerobot]" && quackd doctor --robot lerobot:real --address /dev/ttyACM0   # Python 3.12+
```

## Backends

| `--robot` | Status | What it is |
|---|---|---|
| `lerobot:mock` | ✅ | an arm in memory: joint goals land instantly, a gripper that stops at 30 of 100 when something is between the jaws, a scripted policy for `pick`, a synthetic camera frame, a fixed 30 °C on every joint so the heat refusal has something to refuse, and the same out-of-range refusal the real one gives |
| `lerobot:real` | 🧪 | an SO-101 follower through LeRobot (extra `quackd[lerobot]`, Python 3.12 or newer, torch); every LeRobot name VERIFIED against a pinned commit, unverified end to end, never run on an arm |

`--address` is the arm's serial port (`/dev/ttyACM0`, `COM5`), and quackd checks that it
looks like one before LeRobot opens anything. The `real` backend calls
`connect(calibrate=False)` and refuses an uncalibrated arm, in these words:

```
lerobot real: the arm is not calibrated; run LeRobot's calibration first
(it is interactive, quackd never triggers it)
```

LeRobot's calibration calls `input()`, so it is a human's step and never quackd's. Run
upstream's own `lerobot-calibrate --robot.type=so101_follower --robot.port=<port>` first, or
whatever your LeRobot version spells it, and then connect. It is also what writes the
calibration file quackd reads each joint's travel out of, so there is no driving an arm
without it.

## The manifest

```json
{
  "manifest": 1, "id": "arm-01", "vendor": "huggingface", "model": "lerobot-so101",
  "embodiment": "arm", "mobility": "none",
  "intents": ["joint", "gripper", "skill"], "sensors": ["joint_state", "camera"],
  "verbs": ["observe", "report_state", "stop", "move_joints", "gripper", "place", "pick"],
  "preconditions": {"move_joints": ["torque_on", "not_hot"], "place": ["holding"], "pick": ["torque_on", "not_hot"]},
  "safety_authority": {"native": "torque_limit", "deadman": false, "heartbeat_hz": 2.0},
  "frame": {"reference": "base", "note": "joint space in degrees (gripper 0..100); no camera-to-base calibration"},
  "limits": {"joint_deg": 180.0, "gripper": 100.0},
  "extras": {"robot_type": "so101_follower", "joints": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"], "policy": true, "torque_limit_scope": "gripper_only"}
}
```

That is the mock's static manifest. The static manifest of `lerobot:real` claims neither a
camera nor a policy; `connect()` adds `observe` when the arm's `observation_features` name a
camera and `pick` when a policy object was injected. It also adds three things that cannot
be known until the arm has answered: `extras.joint_range_deg`, every joint's travel in
degrees read out of the calibration file, `extras.calibration_file`, the path that came
from, and `limits.step_deg`, how far one action may move a joint.

| Verb | Kind | What it does here |
|---|---|---|
| `observe` (alias `get_frame`) | core | the arm's camera frame plus detections, only when a camera is configured |
| `report_state` | core | joint positions in degrees, whether torque is on, each servo's temperature, whether something is held |
| `stop` | core | hold: the present position becomes the goal, for the five body joints and not the gripper. Never limp (see Safety) |
| `move_joints(positions, duration_s)` | extension | goal angles for one or more of the six joints. Re-sent until the arm is there, and a failure if it stops short |
| `gripper(open)` | extension | open or close the gripper, and report where it stopped |
| `place` | extension | open the gripper where the arm is; needs `holding` |
| `pick(target, max_s)` | extension, **confirm** | one skill intent; the arm's learned policy runs its own observe/act loop at its own rate until something is held or the time is up |

Its datasheet, which the pilot is shown and told to judge a task against before anything moves ([manifest-spec.md](../manifest-spec.md)):

| | |
|---|---|
| Height | 0.53 m (estimate: one vendor's listing; reaching straight up) |
| Actuated joints | 6 (official: the LeRobot SO-101 docs; five joints and a gripper) |
| Payload | 0.5 kg (estimate: one vendor's listing) |
| Not published | mass, reach |

Mass is on that last row rather than in the table because vendor listings put this arm
anywhere from 0.8 to 2.5 kg and nobody official publishes one. A figure that disagreed with
itself by a factor of three used to sit there as an estimate; it is now listed as not
published, which is what the rule has always said to do.

And what it cannot do whatever the task says, which is the half a refusal usually turns on, in the words the pilot is shown:

- go anywhere: it is bolted to a table and has no base
- lift or hold more than about half a kilogram, and nothing whose weight is not known
- reach anything that is not already within arm's length of its base: the reach is not
  published
- feel what it holds: nothing reports grip force, so holding is inferred from the gripper
  stopping short of shut, which an empty hand that binds also does
- know its own mass: vendor listings disagree by a factor of three

A figure nobody published is listed as not published, and the pilot is told to decline whatever hinges on it rather than guess. A `.duck` file can correct any of it for the build in front of you ([duck-spec.md](../duck-spec.md)).

## What the arm does not tell you, and what quackd does instead

Four things in this adapter exist because upstream cannot answer a question quackd has to
ask. Each was read out of the pinned source rather than guessed at, and each is a row in the
VERIFIED table below.

- **Is the arm still there?** `is_connected` is the serial port's own open flag
  (`port_handler.is_open`). Pull the USB cable and it stays `True` until something tries to
  read and fails. So the heartbeat is a real round trip to the motors every 500 ms, not a
  flag, and a dead arm ends the run rather than being driven at.
- **Is torque on, and how warm is it?** `get_observation()` reads `Present_Position` and
  nothing else. quackd reads `Torque_Enable` and `Present_Temperature` off the bus in the
  same worker thread, so `extras.torque` is measured rather than asserted and
  `extras.temperature_c` exists at all. A servo that has tripped its own overload protection
  drops torque silently, and that is the only way to see it.
- **Is the goal reachable?** The `DEGREES` branch of LeRobot's un-normalise does not clamp
  to the calibrated range, though the two 0..100 modes do. A degrees goal past the travel is
  written to `Goal_Position` as-is, and what the firmware does with it is Feetech's business.
  quackd computes each joint's range from the calibration file at connect and refuses the
  goal instead.
- **How fast can it be told to move?** `max_relative_target` is `None` upstream, so one
  action could slew a joint across its whole travel. quackd sets it: 5 degrees per action,
  re-sent at 10 Hz, so 50 degrees a second. `QUACKD_LEROBOT_MAX_STEP_DEG` changes it.

That last one is why `move_joints` and `gripper` are loops rather than a send and a sleep.
An action moves a joint at most one step, so a goal takes as many actions as it takes, and
since nothing reports arrival, the verb compares the goal with the measurement each tick. An
arm that stalls against an obstacle fails with where it stopped instead of reporting the
move it was asked for.

## Safety

- **No deadman.** Nothing in LeRobot's `Robot` stops an arm when the client goes quiet, and
  that is now read rather than assumed: the class has no thread, no timer and no timeout. A
  position-controlled arm holds its last goal. quackd's `stop` re-sends the present position
  as the goal and never calls `disable_torque()`, the same principle as never sending
  `robot.relax` to a Microduck.
- **`stop` leaves the gripper's goal alone.** It sends the five body joints and omits the
  gripper key, because LeRobot writes only the keys it is given. A stop that re-sent the
  gripper's *measured* position would open a hand that is squeezing something, and every
  failed verb ends in a stop.
- **The native limit is the gripper and only the gripper.** `configure()` writes
  `Max_Torque_Limit 500`, `Protection_Current 250` and `Overload_Torque 25` on the gripper
  inside a check for that motor's name. The five body joints get nothing, so
  `extras.torque_limit_scope` says `gripper_only` and the heat refusal covers exactly the
  joints LeRobot does not. The gripper itself is not heat-gated, because opening it is how
  you put down what it is holding.
- **A wedged call is not a finished call.** LeRobot is synchronous over a half-duplex bus
  where two talkers is a corrupt packet, so every call runs alone under one lock with a
  deadline. A call that blows its deadline leaves a thread still waiting on the wire, so the
  transport refuses every later call until that thread comes back rather than starting a
  second one. The arm holds its goal meanwhile, which is the one thing that needs no rescue.
- **LeRobot's `disconnect()` disables torque by default** (`disable_torque_on_disconnect`).
  quackd keeps that default: at the end of a session the arm goes limp, and that is
  LeRobot's choice, documented here rather than overridden. It fires on every clean exit, a
  `doctor` probe included, and not at all if the process is killed.
- **`pick` is confirm-gated** in the manifest: a learned policy moves the whole arm. Its
  actions go through the same step cap and the same range refusal as a verb's.

## Upstream API

Pinned at `fbb811f` (main, 2026-09-01; read 2026-09-02, read again 2026-09-13). PyPI had
0.6.1 on both days.

### VERIFIED (read from source at the pin)

| Name | Note |
|---|---|
| `lerobot` | PyPI and import name |
| `>=3.12` | requires-python; quackd's floor is 3.11, so the extra carries a python_version marker |
| `0.6.2` | PyPI had 0.6.1 |
| `lerobot[feetech]` | feetech-servo-sdk (imported as scservo_sdk) and pyserial live in this extra and not in lerobot's base dependencies, so a plain lerobot imports cleanly and then cannot open the arm's port. quackd[lerobot] asks for lerobot[feetech] and doctor has a row for the SDK |
| `lerobot.robots.Robot` | the abstract base every robot implements |
| `Robot.connect(calibrate=True)` | quackd passes calibrate=False: calibration is interactive (see ROBOT_CALIBRATE). The SO follower's connect() runs bus.connect() and configure() and writes no calibration into the motors at all, so the file on disk and the arm must already agree |
| `Robot.disconnect()` |  |
| `Robot.get_observation() -> dict` | a flat dict: '<motor>.pos' floats plus one array per camera, keyed by camera name |
| `Robot.send_action(action: dict) -> dict` | '<motor>.pos' -> goal; returns what was actually sent, possibly clipped |
| `Robot.observation_features` | key -> float, or a (h, w, c) shape tuple for a camera; usable before connect() |
| `Robot.action_features` |  |
| `Robot.is_connected` |  |
| `Robot.is_calibrated` |  |
| `Robot.calibrate() is interactive` | the SO follower's calibrate() calls input() (also line 131); quackd never triggers it and refuses to drive an uncalibrated arm |
| `Robot.configure()` |  |
| `Robot.__enter__/__exit__` | connect on enter, disconnect on exit |
| `RobotAction = dict[str, Any]; RobotObservation = dict[str, Any]` |  |
| `Robot.calibration` | motor name -> MotorCalibration, loaded from the file in __init__ when one exists. It is where quackd reads each joint's travel, so it is populated before connect() and empty when this arm has never been calibrated on this machine |
| `Robot.calibration_fpath` | calibration_dir / '<id>.json'; quackd reports the path, so a wrong id is visible |
| `HF_LEROBOT_CALIBRATION/robots/so_follower/` | the default calibration directory: $HF_LEROBOT_CALIBRATION, else $HF_LEROBOT_HOME/calibration, else $HF_HOME/lerobot/calibration, then 'robots' and the robot class's own name. Two arms sharing an id share a file, and nothing in it names a serial number |
| `lerobot.robots.make_robot_from_config(config)` |  |
| `so101_follower` | the registered config type; make_robot_from_config dispatches on it (utils.py line 41) |
| `lerobot.robots.so_follower.SO101Follower` | an alias of SOFollower (SO100Follower too); exported by the package __init__ |
| `SOFollower.name is so_follower` | the class name, and therefore the calibration subdirectory: an SO-100 and an SO-101 share one, because at this commit they are the same class |
| `SO101FollowerConfig(port, disable_torque_on_disconnect=True, max_relative_target=None, cameras={}, use_degrees=True, position_p_coefficient=16, position_i_coefficient=0, position_d_coefficient=32, num_read_retries=2)` | an alias of SOFollowerRobotConfig; id and calibration_dir come from RobotConfig. quackd passes every safety-shaped field explicitly rather than inheriting a default it has not read, and sets max_relative_target, which upstream leaves at None |
| `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` | six Feetech sts3215 motors, ids 1..6 |
| `'<motor>.pos'` | joint positions; the same keys are the action |
| `get_observation() reads Present_Position and nothing else` | one sync_read of the positions, with num_read_retries extra attempts, plus a frame per camera. No torque state, no current, no temperature and no error flag, so a joint that has tripped its own protection looks exactly like one that has arrived |
| `camera name -> array` | get_observation() adds cam.read_latest() under each configured camera's name |
| `max_relative_target caps each step` | ensure_safe_goal_position (utils.py line 93) clips a goal to present +/- the cap and logs a warning when it does; None, which is upstream's default, means no cap. Setting it costs one extra sync_read of the present position per send_action |
| `max_relative_target must be a float or a dict per motor` | ensure_safe_goal_position tests isinstance(float), then isinstance(dict), and raises TypeError on anything else, so an int cap raises rather than capping. quackd casts |
| `send_action() returns the goal actually sent` | the clipped goal in '<motor>.pos' keys, which is not the measured position. quackd keeps it rather than assuming the goal it asked for was the one written |
| `use_degrees=True -> body joints in degrees` | MotorNormMode.DEGREES; False means a -100..100 range |
| `gripper is 0..100` | MotorNormMode.RANGE_0_100 whatever use_degrees |
| `disconnect() disables torque by default` | disable_torque_on_disconnect defaults to True (config line 31): LeRobot lets the arm go limp when the session ends, and quackd keeps that default and says so. It fires on every clean exit, a doctor probe included, and not at all when the process is killed |
| `Max_Torque_Limit 500 on the gripper` | configure() caps the gripper at 50 % torque, 50 % current (Protection_Current 250) and 25 % torque once overloaded (Overload_Torque 25): the native safety authority |
| `the five body joints get no torque or current cap` | the three caps above are inside a check for the gripper. Every other joint keeps whatever its firmware defaults to, so the manifest's torque_limit authority covers the gripper only and an elbow against an obstacle is the servo's own protection and nothing else |
| `configure_motors() writes Return_Delay_Time 0 and Acceleration 254` | plus Maximum_Acceleration 254 on protocol 0 and an sts3215 Phase fix. configure() calls it inside torque_disabled(), so connecting always drops torque briefly |
| `SOFollower.is_connected is the serial port plus the cameras` | bus.is_connected and every camera's; BUS_IS_CONNECTED is what the bus half means |
| `SOFollower.bus is a FeetechMotorsBus` | the attribute quackd reads registers through when the Robot interface has no answer |
| `no deadman: nothing stops the arm when the client goes quiet` | read end to end at the pin: SOFollower has no thread, timer, timeout or watchdog, and send_action writes Goal_Position and returns. A position-controlled arm holds its last goal under torque until the next write or disconnect(). quackd's stop re-sends the present position as the goal (hold) and never disables torque |
| `MotorsBus.disable_torque()` | NEVER called by quackd (limp) |
| `MotorsBus.enable_torque()` |  |
| `MotorsBus.disconnect(disable_torque=True)` |  |
| `MotorsBus.is_connected is port_handler.is_open` | a serial port's open flag, not a reply from a motor. Unplug the arm and it stays True until a read fails, which is why quackd's heartbeat reads the arm rather than the flag |
| `FeetechMotorsBus.is_calibrated reads the motors back` | it reads Min_Position_Limit, Max_Position_Limit and Homing_Offset off every motor and compares them with the cached file. A missing file, a stale file, and the file of a different arm all come back False, which is the check quackd refuses on |
| `write_calibration() is reached only through calibrate()` | it writes the limits and the homing offset into the motors; connect(calibrate=False) never calls it, so quackd cannot move an arm's zero even by accident |
| `MotorsBus.sync_read(data_name, motors=None, normalize=True, num_retry=0)` | one framed transaction for every motor named; quackd uses it for the registers get_observation() does not read |
| `NORMALIZED_DATA is Goal_Position and Present_Position` | the only two names sync_read normalises (motors_bus.py line 1167), so every other register comes back raw whatever normalize says. quackd passes normalize=False anyway, because a raw register is what it means to read |
| `MotorCalibration(id, drive_mode, homing_offset, range_min, range_max)` | range_min and range_max are raw encoder ticks recorded by calibration, not degrees |
| `degrees = (raw - mid) * 360 / 4095` | mid is (range_min + range_max) / 2 and 4095 is the resolution less one, so a joint's travel in degrees is (range_max - range_min) * 360 / 4095, centred on zero. That is how quackd turns a calibration file into the range it will accept a goal inside |
| `a degrees goal is not clamped to the calibrated range` | _unnormalize bounds the RANGE_0_100 and RANGE_M100_100 modes and does not bound DEGREES: the tick it computes is written to Goal_Position as-is. So the gripper is clamped by LeRobot and the five body joints are not, and what the firmware does with a tick outside Min_Position_Limit is Feetech's. quackd refuses the goal instead |
| `sts3215 resolution 4096` | 12 bits over a full turn, so one tick is about 0.088 degrees |
| `Torque_Enable (40, 1) and Present_Temperature (63, 1)` | address and length in the STS/SMS control table, temperature at line 87. Present_Load, Present_Current, Present_Voltage, Status and Max_Temperature_Limit are in the same table and quackd reads none of them yet. Nothing upstream reads these two either: the arm's own protection can drop torque and get_observation() will not mention it |
| `Camera.async_read(timeout_ms)` | the most recent new frame |
| `Camera.read()` |  |
| `OpenCVCamera converts BGR to RGB when color_mode is RGB` | so a camera array's channel order is a config choice, not a constant |
| `OpenCVCameraConfig.color_mode defaults to ColorMode.RGB` | which settles the channel order quackd used to assume: a stock OpenCV camera hands over RGB. The shipped real backend configures no camera at all, so this matters only to somebody who passes one in |
| `lerobot.policies.pretrained.PreTrainedPolicy` |  |
| `PreTrainedPolicy.from_pretrained(path, *, config=None, local_files_only=False, revision=None, strict=False)` | a local directory or a Hub repo id; sets eval mode |
| `PreTrainedPolicy.select_action(batch: dict[str, Tensor]) -> Tensor` | one action per call, the policy handles its own action-chunk cache |
| `PreTrainedPolicy.reset()` |  |
| `lerobot.policies.factory.get_policy_class(name)` |  |
| `lerobot.policies.factory.make_pre_post_processors(policy_cfg, pretrained_path)` | a raw observation goes through the pre-processor and the action tensor through the post-processor before it is a RobotAction |
| `lerobot.policies.factory.make_policy(cfg)` |  |

### UNVERIFIED (our assumptions, and what quackd does about each)

| Name | What quackd does |
|---|---|
| `POLICY_PIPELINE` | wiring a PreTrainedPolicy end to end (pre-processor, select_action, post-processor, device) has never been run by us. The real backend takes an injected policy with act(observation, task=...) -> action; load_policy() builds one from the verified names and is untested. A policy's actions go through the same step cap and the same range refusal as a verb's, which is quackd's rule and not upstream's |
| `GRIPPER_OPEN_VALUE` | which end of the gripper's 0..100 range is open. 0 is the range_min tick of that motor's calibration and 100 the range_max, so which one is the open jaw is how the arm was assembled and calibrated. quackd assumes 100 is open, and the checklist asks for it to be confirmed by hand before anything is believed about holding |
| `HOLDING_INFERRED` | nothing reports grip force, so holding is inferred: the gripper was told to close, its reading has settled, and it settled short of shut. The band is quackd's own guess, an empty hand that binds reads as holding, and a thin enough object may not |
| `TEMPERATURE_C` | Present_Temperature is one byte, and Feetech's documentation calls it degrees Celsius, which nothing in LeRobot reads or converts. quackd reports it raw and refuses to move a joint at or above 60, which is the datasheet's operating maximum and below the servo's own 70 cut-off, so the number and the threshold are both ours to be wrong about |
| `JOINT_RANGES` | the reachable range of each joint is whatever calibration recorded, and no vendor publishes what it ought to be. quackd computes each joint's travel from the calibration file and refuses a goal outside it rather than writing a tick LeRobot will not clamp (DEGREES_NO_CLAMP); whether that travel is the real mechanical limit is unverified |
| `SERIAL_PORT` | the arm's serial port (/dev/ttyACM0, COM5) comes from --address. quackd checks its shape and nothing more: which port is the arm, and whether a CH340 or CP210x driver is installed, is between the owner and their machine |
| `THREAD_SAFETY` | Robot is synchronous and not documented as thread-safe, over a half-duplex serial bus where two talkers is a corrupt packet. quackd serialises every call under one lock in a worker thread with a deadline, and when a call blows its deadline it refuses every later call rather than starting a second thread on the same bus |

## Status

`lerobot:mock` runs every arm verb through the executor in the test suite, including the
confirm gate on `pick`, the `holding` precondition on `place` and the heat refusal on
`move_joints`. `lerobot:real` is exercised with an injected fake arm and an injected fake
policy (verified method names, no serial port): the fake caps each action the way
`max_relative_target` does, stalls a joint on request, stops its gripper on an object, and
can be unplugged mid-run. Nobody has run it on an arm, and this page will say so until
someone has.

## How to help

If you have an SO-101 on a desk, the useful thing is a first run. Work through
[lerobot-hardware-checklist.md](../lerobot-hardware-checklist.md) in order: nothing moves
until step 8, and the first thing it asks for is `quackd doctor --address`, which connects,
rather than `list-verbs`, which does not.

Note that `lerobot-lookout` asks for `report_state` and not `observe`. The real backend
configures no camera, so the arm has no `observe` to allow, and a bring-up task that refuses
before it runs is no use to anybody.

What most needs a real arm: which end of the gripper's 0..100 range is open, whether the
holding band is anywhere near right, what a joint actually reads in degrees Celsius under
load, and whether 5 degrees an action felt right or slow. Open an issue with the transcript.
