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
[`quackd/adapters/lerobot/upstream_api.py`](../../quackd/adapters/lerobot/upstream_api.py),
and why the adapter is shaped the way it is is
[ADR-0036](../adr/0036-what-the-arm-does-not-say.md).

**The `real` backend has never been run against an arm by us.**

```bash
# offline, the default
uv run quackd run lerobot-lookout --robot lerobot:mock --provider fake

uvx quackd list-verbs --robot lerobot:mock
uvx quackd validate ducks/find-and-kick.duck --robot lerobot:mock     # exit 1: requires ... does not provide it
uvx quackd serve-mcp --robots arm=lerobot:mock,duck=microduck:sim2d   # an arm and a duck behind one MCP server

# a real arm, after LeRobot's own calibration (see the checklist)
uv pip install "quackd[lerobot]" && quackd doctor --robot lerobot:real --address /dev/ttyACM0   # Python 3.12+
```

## Backends

| `--robot` | Status | What it is |
|---|---|---|
| `lerobot:mock` | ✅ | an arm in memory: goals land instantly, the gripper stops on the object, a scripted policy answers `pick`, and it refuses an out-of-range goal in the same words the real one does |
| `lerobot:real` | 🧪 | an SO-101 follower through LeRobot (extra `quackd[lerobot]`, Python 3.12 or newer, torch); every name VERIFIED at the pin, exercised against a fake arm, never on hardware |

`--address` is the arm's serial port (`/dev/ttyACM0`, `COM5`), and quackd checks that it
looks like one before LeRobot opens anything. The `real` backend calls
`connect(calibrate=False)` and refuses an uncalibrated arm, in these words:

```
lerobot real: the arm is not calibrated; run LeRobot's calibration first
(it is interactive, quackd never triggers it)
```

Calibration is upstream's own interactive step, under the id quackd will use, and it writes
the file every joint's range is read from: step 4 of
[lerobot-hardware-checklist.md](../lerobot-hardware-checklist.md).

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
camera and `pick` when a policy object was injected. It also adds what cannot be known until
the arm has answered: `extras.joint_range_deg`, every joint's travel in degrees read out of
the calibration file, `extras.calibration_file`, the path that came from, and
`limits.step_deg`, how far one action may move a joint (`QUACKD_LEROBOT_MAX_STEP_DEG` sets
it).

| Verb | Kind | What it does here |
|---|---|---|
| `observe` (alias `get_frame`) | core | the arm's camera frame plus detections, only when a camera is configured |
| `report_state` | core | joint positions in degrees, whether torque is on, each servo's temperature, whether something is held |
| `stop` | core | hold: the present position becomes the goal. Never limp (see Safety) |
| `move_joints(positions, duration_s)` | extension | goal angles for one or more of the six joints, re-sent at 10 Hz until the measurement arrives; a joint that stops short is a failure, and `duration_s` is the budget |
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

And what it cannot do whatever the task says, which is the half a refusal usually turns on, in the words the pilot is shown:

- go anywhere: it is bolted to a table and has no base
- lift or hold more than about half a kilogram, and nothing whose weight is not known
- reach anything that is not already within arm's length of its base: the reach is not
  published
- feel what it holds: nothing reports grip force, so holding is inferred from the gripper
  stopping short of shut, which an empty hand that binds also does
- know its own mass: vendor listings disagree by a factor of three

A figure nobody published is listed as not published, and the pilot is told to decline whatever hinges on it rather than guess. A `.duck` file can correct any of it for the build in front of you ([duck-spec.md](../duck-spec.md)).

## Safety

Each of these exists because upstream could not answer a question quackd has to ask; the
reasoning is in [ADR-0036](../adr/0036-what-the-arm-does-not-say.md).

- **The heartbeat reads the arm.** `is_connected` is the serial port's open flag and stays
  `True` with the cable pulled, so the heartbeat is a round trip to the motors, and a dead arm
  ends the run.
- **Torque and temperature are measured.** `get_observation()` reads positions only, so
  `Torque_Enable` and `Present_Temperature` are read off the bus. A body joint at or above
  60 °C refuses `move_joints` and `pick`; the servo's own cut-off is 70 °C.
- **A goal outside the calibrated range is refused.** LeRobot does not clamp a degrees goal,
  so quackd computes each joint's travel from the calibration file and refuses instead.
- **One action moves a joint at most one step.** `max_relative_target` is unset upstream;
  quackd sets it to 5 degrees, re-sent at 10 Hz, so 50 degrees a second.
- **No deadman.** Nothing in LeRobot's `Robot` stops an arm when the client goes quiet: read
  from the class, not assumed. quackd's `stop` re-sends the present position as the goal and
  never calls `disable_torque()`, the same principle as never sending `robot.relax` to a
  Microduck.
- **`stop` leaves the gripper's goal alone.** It sends the five body joints and omits the
  gripper key, so a stop never opens a hand that is squeezing something, and every failed
  verb ends in a stop.
- **The native limit is the gripper and only the gripper.** `configure()` caps the gripper's
  torque and current inside a check for that motor's name; the five body joints get nothing,
  so `extras.torque_limit_scope` says `gripper_only`. The gripper itself is not heat-gated,
  because opening it is how you put down what it is holding.
- **A wedged call is not a finished call.** A call that blows its deadline leaves a thread on
  a half-duplex bus, so the transport refuses every later call until that thread comes back
  rather than starting a second one. The arm holds its goal meanwhile.
- **Torque drops at both ends of a session.** `disconnect()` disables it by LeRobot's default,
  which quackd keeps and documents; and `configure()` runs inside `torque_disabled()`, so
  connecting drops it briefly too.
- **`pick` is confirm-gated**: a learned policy moves the whole arm. Its actions go through
  the same step cap and range check as a verb's.

## Upstream API

### VERIFIED (read from source at the pin)

| Name | Why quackd relies on it |
|---|---|
| `lerobot` | PyPI and import name |
| `>=3.12` | requires-python; quackd's floor is 3.11, so the extra carries a marker |
| `0.6.2` | the version at the pin; PyPI had 0.6.1 |
| `lerobot[feetech]` | the only home of the serial SDK and pyserial; `quackd[lerobot]` asks for it |
| `lerobot.robots.Robot` | the abstract base every robot implements |
| `Robot.connect(calibrate=True)` | quackd passes False; connect() then writes no calibration into the motors |
| `Robot.disconnect()` | |
| `Robot.get_observation() -> dict` | flat: `'<motor>.pos'` floats plus one array per camera |
| `Robot.send_action(action: dict) -> dict` | `'<motor>.pos'` goals; returns what was actually sent |
| `Robot.observation_features` | camera keys carry shape tuples; usable before connect() |
| `Robot.action_features` | |
| `Robot.is_connected` | |
| `Robot.is_calibrated` | |
| `Robot.calibrate() is interactive` | it calls `input()`; quackd never triggers it |
| `Robot.configure()` | |
| `Robot.__enter__/__exit__` | connect on enter, disconnect on exit |
| `RobotAction = dict[str, Any]; RobotObservation = dict[str, Any]` | |
| `Robot.calibration` | motor name -> MotorCalibration, loaded from the file; where joint ranges come from |
| `Robot.calibration_fpath` | `calibration_dir / '<id>.json'`; reported so a wrong id is visible |
| `HF_LEROBOT_CALIBRATION/robots/so_follower/` | the default calibration directory; two arms sharing an id share a file |
| `lerobot.robots.make_robot_from_config(config)` | |
| `so101_follower` | the registered config type |
| `lerobot.robots.so_follower.SO101Follower` | an alias of SOFollower |
| `SOFollower.name is so_follower` | the calibration subdirectory, shared by SO-100 and SO-101 |
| `SO101FollowerConfig(port, disable_torque_on_disconnect=True, max_relative_target=None, cameras={}, use_degrees=True, position_p_coefficient=16, position_i_coefficient=0, position_d_coefficient=32, num_read_retries=2)` | every safety-shaped field is passed explicitly rather than inherited |
| `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` | six Feetech sts3215 motors, ids 1..6 |
| `'<motor>.pos'` | the observation and action keys |
| `get_observation() reads Present_Position and nothing else` | no torque, current, temperature or fault: why quackd reads registers |
| `camera name -> array` | `cam.read_latest()` under each configured camera's name |
| `max_relative_target caps each step` | clips a goal to present +/- the cap per send_action |
| `max_relative_target must be a float or a dict per motor` | an int raises; a dict must name exactly the action's joints |
| `send_action() returns the goal actually sent` | the clipped goal, not the measured position |
| `use_degrees=True -> body joints in degrees` | |
| `gripper is 0..100` | whatever use_degrees says |
| `disconnect() disables torque by default` | the arm goes limp at every clean exit, a doctor probe included |
| `Max_Torque_Limit 500 on the gripper` | with Protection_Current 250 and Overload_Torque 25: the native authority |
| `the five body joints get no torque or current cap` | the caps sit inside a check for the gripper's name |
| `configure_motors() writes Return_Delay_Time 0 and Acceleration 254` | called inside torque_disabled(), so connecting drops torque briefly |
| `SOFollower.is_connected is the serial port plus the cameras` | |
| `SOFollower.bus is a FeetechMotorsBus` | the attribute registers are read through |
| `no deadman: nothing stops the arm when the client goes quiet` | the class has no thread, timer or timeout; a goal stands until the next write |
| `MotorsBus.disable_torque()` | NEVER called by quackd (limp) |
| `MotorsBus.enable_torque()` | |
| `MotorsBus.disconnect(disable_torque=True)` | |
| `MotorsBus.is_connected is port_handler.is_open` | a port flag, not a reply: why the heartbeat reads the arm |
| `FeetechMotorsBus.is_calibrated reads the motors back` | a missing, stale or foreign file all read as not calibrated |
| `write_calibration() is reached only through calibrate()` | quackd cannot move an arm's zero by accident |
| `MotorsBus.sync_read(data_name, motors=None, normalize=True, num_retry=0)` | one transaction for every motor named |
| `NORMALIZED_DATA is Goal_Position and Present_Position` | every other register comes back raw |
| `MotorCalibration(id, drive_mode, homing_offset, range_min, range_max)` | raw encoder ticks, not degrees |
| `degrees = (raw - mid) * 360 / 4095` | how a calibration file becomes a range in degrees, centred on zero |
| `a degrees goal is not clamped to the calibrated range` | the two 0..100 modes are clamped and DEGREES is not: why quackd refuses |
| `sts3215 resolution 4096` | one tick is about 0.088 degrees |
| `Torque_Enable (40, 1) and Present_Temperature (63, 1)` | the two registers quackd reads; upstream reads neither |
| `Camera.async_read(timeout_ms)` | the most recent new frame |
| `Camera.read()` | |
| `OpenCVCamera converts BGR to RGB when color_mode is RGB` | channel order is a config choice |
| `OpenCVCameraConfig.color_mode defaults to ColorMode.RGB` | the shipped real backend configures no camera at all |
| `lerobot.policies.pretrained.PreTrainedPolicy` | |
| `PreTrainedPolicy.from_pretrained(path, *, config=None, local_files_only=False, revision=None, strict=False)` | a local directory or a Hub repo id |
| `PreTrainedPolicy.select_action(batch: dict[str, Tensor]) -> Tensor` | one action per call |
| `PreTrainedPolicy.reset()` | |
| `lerobot.policies.factory.get_policy_class(name)` | |
| `lerobot.policies.factory.make_pre_post_processors(policy_cfg, pretrained_path)` | the observation and the action tensor each go through one |
| `lerobot.policies.factory.make_policy(cfg)` | |

### UNVERIFIED (our assumptions, and what quackd does about each)

| Name | What quackd does |
|---|---|
| `POLICY_PIPELINE` | `pick` runs an injected policy object; `load_policy()` builds one from verified names and is untested. A policy's actions get the same step cap and range check as a verb's |
| `GRIPPER_OPEN_VALUE` | 100 is assumed open; which end is open is how the arm was calibrated, and the checklist asks for it by hand |
| `HOLDING_INFERRED` | holding is the gripper told to close, settled, and short of shut; listed in `extras.assumptions` |
| `TEMPERATURE_C` | the register is read raw and treated as Celsius; the 60 °C refusal and the 70 °C cut-off are Feetech's numbers, not measured |
| `JOINT_RANGES` | each joint's travel is computed from the calibration file and a goal outside it is refused; whether that is the mechanical limit is unverified |
| `SERIAL_PORT` | `--address` is checked for shape and nothing more |
| `THREAD_SAFETY` | every call is serialised under one lock in a worker thread with a deadline; a blown deadline wedges the transport |

## Status

`lerobot:mock` runs every arm verb through the executor in the test suite, including the
confirm gate on `pick`, the `holding` precondition on `place` and the heat refusal.
`lerobot:real` is exercised against a fake arm and a fake policy (verified method names, no
serial port). Nobody has run it on an arm, and this page will say so until someone has.

## How to help

If you have an SO-101 on a desk, work through
[lerobot-hardware-checklist.md](../lerobot-hardware-checklist.md) in order: nothing moves
until step 8. `lerobot-lookout` is the first task to point at it; it asks for `report_state`
rather than `observe`, because the real backend configures no camera. What most needs a real
arm is that checklist's *What to report*. Open an issue with the transcript.
