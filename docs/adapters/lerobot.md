# LeRobot (an SO-101 class arm)

A six-joint desktop arm with a parallel gripper, driven through
[LeRobot](https://github.com/huggingface/lerobot). No legs, no head, no voice, so its
manifest lists none of that: `move`, `go_to`, `search_scan`, `say` and `gaze` do not exist
on this robot. What it has is joints, a gripper, `place`, and, when a policy is available,
`pick` as one skill intent that the arm's own learned policy executes. The thesis holds:
the LLM picks the verb, LeRobot moves the arm, quackd enforces the contract. The `real`
backend has **never been run against an arm by us**.

```bash
uvx quackd list-verbs --robot lerobot:mock
uvx quackd validate ducks/find-and-kick.duck --robot lerobot:mock     # exit 1: requires ... does not provide it
uvx quackd serve-mcp --robots arm=lerobot:mock,duck=microduck:sim2d   # an arm and a duck behind one MCP server
uv pip install "quackd[lerobot]" && quackd doctor --robot lerobot:real --address /dev/ttyACM0   # Python 3.12+
```

## Backends

| `--robot` | Status | What it is |
|---|---|---|
| `lerobot:mock` | ✅ | an arm in memory: joint goals land instantly, the gripper closes on an object when the arm is near it, a scripted policy answers `pick`, a synthetic camera frame shows the object |
| `lerobot:real` | 🧪 | an SO-101 follower through LeRobot (extra `quackd[lerobot]`, Python 3.12 or newer, torch); every LeRobot name VERIFIED against a pinned commit, unverified end to end, never run on an arm |

`--address` is the arm's serial port (`/dev/ttyACM0`, `COM5`). The `real` backend calls
`connect(calibrate=False)` and refuses an uncalibrated arm: LeRobot's calibration is
interactive (it calls `input()`), so it is a human's step, never quackd's.

## The manifest

```json
{
  "manifest": 1, "id": "arm-01", "vendor": "huggingface", "model": "lerobot-so101",
  "embodiment": "arm", "mobility": "none",
  "intents": ["joint", "gripper", "skill"], "sensors": ["joint_state", "camera"],
  "verbs": ["observe", "report_state", "stop", "move_joints", "gripper", "place", "pick"],
  "preconditions": {"move_joints": ["torque_on"], "place": ["holding"], "pick": ["torque_on"]},
  "safety_authority": {"native": "torque_limit", "deadman": false, "heartbeat_hz": 2.0},
  "frame": {"reference": "base", "note": "joint space in degrees (gripper 0..100); no camera-to-base calibration"},
  "limits": {"joint_deg": 180.0, "gripper": 100.0},
  "extras": {"robot_type": "so101_follower", "joints": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"], "policy": true}
}
```

That is the mock's manifest. The static manifest of `lerobot:real` claims neither a
camera nor a policy; `connect()` adds `observe` when the arm's `observation_features`
name a camera and `pick` when a policy object was injected.

| Verb | Kind | What it does here |
|---|---|---|
| `observe` (alias `get_frame`) | core | the arm's camera frame plus detections, only when a camera is configured |
| `report_state` | core | joint positions in degrees, whether something is held, torque |
| `stop` | core | hold: the present position becomes the goal. Never limp (see Safety) |
| `move_joints(positions, duration_s)` | extension | goal angles for one or more of the six joints; the arm's own position controller moves |
| `gripper(open)` | extension | open or close the gripper |
| `place` | extension | open the gripper where the arm is; needs `holding` |
| `pick(target, max_s)` | extension, **confirm** | one skill intent; the arm's learned policy runs its own observe/act loop at its own rate until something is held or the time is up |

## Safety

- **No deadman.** Nothing in LeRobot's `Robot` stops an arm when the client goes quiet; a
  position-controlled arm holds its last goal. quackd's `stop` re-sends the present
  position as the goal and never calls `disable_torque()`, the same principle as never
  sending `robot.relax` to a Microduck.
- **The native limit** is what LeRobot writes at `configure()`: the gripper's torque and
  current caps (50 %), plus `max_relative_target` when a config sets it. That is why the
  manifest says `torque_limit`.
- **LeRobot's `disconnect()` disables torque by default** (`disable_torque_on_disconnect`).
  quackd keeps that default: at the end of a session the arm goes limp, and that is
  LeRobot's choice, documented here rather than overridden.
- **`pick` is confirm-gated** in the manifest: a learned policy moves the whole arm.

## Upstream API

Pinned at `fbb811f` (main, 2026-09-01; read 2026-09-02). PyPI had 0.6.1 that day.

### VERIFIED (read from source at the pin)

| Name | Note |
|---|---|
| `lerobot` | PyPI and import name |
| `>=3.12` | requires-python; quackd's floor is 3.11, hence the marker on the extra |
| `0.6.2` | the version at the pin |
| `lerobot.robots.Robot` | the abstract base every robot implements |
| `Robot.connect(calibrate=True)` | quackd passes `calibrate=False` |
| `Robot.disconnect()` | |
| `Robot.get_observation() -> dict` | flat: `'<motor>.pos'` floats plus one array per camera |
| `Robot.send_action(action: dict) -> dict` | returns what was sent, possibly clipped |
| `Robot.observation_features` | usable before connect; camera keys carry shape tuples |
| `Robot.action_features` | |
| `Robot.is_connected` | |
| `Robot.is_calibrated` | |
| `Robot.calibrate() is interactive` | the SO follower's `calibrate()` calls `input()` |
| `Robot.configure()` | |
| `Robot.__enter__/__exit__` | connect on enter, disconnect on exit |
| `RobotAction = dict[str, Any]; RobotObservation = dict[str, Any]` | |
| `lerobot.robots.make_robot_from_config(config)` | |
| `so101_follower` | the registered config type |
| `lerobot.robots.so_follower.SO101Follower` | an alias of `SOFollower` |
| `SO101FollowerConfig(port, disable_torque_on_disconnect=True, max_relative_target=None, cameras={}, use_degrees=True)` | |
| `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` | six Feetech sts3215 motors |
| `'<motor>.pos'` | the observation and action keys |
| `camera name -> array` | `cam.read_latest()` under each configured camera's name |
| `max_relative_target caps each step` | `ensure_safe_goal_position` |
| `use_degrees=True -> body joints in degrees` | |
| `gripper is 0..100` | |
| `disconnect() disables torque by default` | |
| `Max_Torque_Limit 500 on the gripper` | the native safety authority |
| `MotorsBus.disable_torque()` | NEVER called by quackd |
| `MotorsBus.enable_torque()` | |
| `MotorsBus.disconnect(disable_torque=True)` | |
| `lerobot.policies.pretrained.PreTrainedPolicy` | |
| `PreTrainedPolicy.from_pretrained(path, *, config=None, local_files_only=False, revision=None, strict=False)` | |
| `PreTrainedPolicy.select_action(batch: dict[str, Tensor]) -> Tensor` | |
| `PreTrainedPolicy.reset()` | |
| `lerobot.policies.factory.get_policy_class(name)` | |
| `lerobot.policies.factory.make_pre_post_processors(policy_cfg, pretrained_path)` | |
| `lerobot.policies.factory.make_policy(cfg)` | |
| `Camera.async_read(timeout_ms)` | |
| `Camera.read()` | |
| `OpenCVCamera converts BGR to RGB when color_mode is RGB` | |

### UNVERIFIED (our assumptions, and what quackd does about each)

| Name | What quackd does |
|---|---|
| `NO_CLIENT_DEADMAN` | stop is hold; torque is never disabled by quackd |
| `POLICY_PIPELINE` | `pick` runs an injected policy object; `load_policy()` builds one from a checkpoint and is untested |
| `CAMERA_COLOR_ORDER` | RGB assumed; the assumption is listed in `report_state` |
| `GRIPPER_OPEN_VALUE` | 100 is open, 0 is closed; `holding` is what was commanded, not sensed |
| `JOINT_RANGES` | `move_joints` takes -180..180 degrees as a schema bound; the motors clip the rest |
| `SERIAL_PORT` | `--address` is passed through unvalidated |
| `THREAD_SAFETY` | one call at a time, in a worker thread, with a deadline |

## Status

`lerobot:mock` runs every arm verb through the executor in the test suite, including the
confirm gate on `pick` and the `holding` precondition on `place`. `lerobot:real` is
exercised with an injected fake arm and an injected fake policy (verified method names,
no serial port). Nobody has run it on an arm, and this page will say so until someone
has.
