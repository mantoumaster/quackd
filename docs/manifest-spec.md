# The robot manifest (v1)

What a connected robot is and can do, as data. Pydantic in code
(`quackd/adapters/manifest.py`), JSON on the wire (MCP `robot_list`, mDNS by digest, the
flock bus), never YAML on disk. The JSON Schema is exported to
`quackd/adapters/manifest.schema.json` (`python -m quackd.adapters.export`) and
drift-tested. Every adapter's `connect()` returns one; `describe()` returns the static
version without touching the robot.

## Fields

| Field | Type | Meaning |
|---|---|---|
| `manifest` | `1` | schema version |
| `id` | slug | unique within a run or a flock: `microduck`, `open-duck-01`, `duck-01`, or the fleet name from `--robots name=...` |
| `vendor`, `model` | string | who made it and what it is (`huggingface`, `lerobot-so101`) |
| `embodiment` | `biped`, `quadruped`, `wheeled`, `arm`, `humanoid` | the body |
| `mobility` | `none`, `legged`, `wheeled` | whether it can go somewhere |
| `intents` | list of `twist`, `skill`, `gaze`, `sound`, `joint`, `pose`, `gripper` | the command channels the backend accepts |
| `sensors` | list of `camera`, `battery`, `odometry`, `imu`, `tof`, `microphone`, `joint_state` | what it can report |
| `verbs` | list of `VerbSpec` | the vocabulary (see below) |
| `preconditions` | verb → list of condition names | checks the executor runs before a verb; the adapter supplies the predicates by name |
| `safety_authority` | `{native, deadman, heartbeat_hz}` | who stops the body when quackd goes quiet: `native` is `robotd_deadman`, `lease`, `torque_limit`, `estop` or `none`; `deadman` is whether motion zeroes on silence, wherever that code lives. An Open Duck Mini declares `native: none` with `deadman: true`, because the thing doing the zeroing is quackd's own daemon running on the robot — on the `bridge` backend, only when that daemon reported a `deadman_ms` window at connect |
| `frame` | `{reference, note}` | `body`, `head`, `base` or `world`; what bearings are relative to |
| `limits` | name → number | `max_vx`, `max_vy`, `max_wz`, `gaze_yaw_deg`, `gaze_pitch_deg`, `joint_deg`, ...; the core verbs clamp to them. `camera_fov_deg` is the exception: not a clamp but the lens the detector assumes, and without it (or `--fov-deg`) a real camera falls back to the simulator's 90° and every detection is labelled uncalibrated |
| `backend` | string | informational: which backend produced this |
| `blurb` | string | the prompt's one-line intro: "a small biped duck robot (25 cm, 800 g)" |
| `extras` | object | adapter-specific facts (`speech: tones`, `camera_calibrated: false`, `policy: true`) |

A `VerbSpec`: `name` (canonical, never an alias), `core` (the same verb on every robot
that has it), `description` (LLM-facing; empty means the implementation's default),
`params_schema` (informational JSON schema), `safety_class` (`safe`, `confirm`,
`dangerous`), `timeout_s`.

## Invariants the model enforces

- No duplicate verbs; no alias as a verb name (`walk` is declared as `move`).
- `stop` is inserted when missing, is `core`, and can never be anything but `safe`.
- A core verb must have what it requires (`quackd/verbs/core.py`, `REQUIREMENTS`):

| Core verb | Requires |
|---|---|
| `observe` (alias `get_frame`) | a `camera` sensor |
| `report_state` | nothing |
| `stop` | nothing |
| `say` | the `sound` intent |
| `move` (alias `walk`) | the `twist` intent and mobility |
| `go_to` (alias `walk_to`) | `twist`, mobility and a camera |
| `search_scan` | a camera and either `twist` or `gaze` (it turns in place or sweeps the head) |
| `approach_and` | `twist`, mobility and a camera |

- Every verb in `preconditions` is declared; every condition name has a predicate in the
  adapter's `conditions()` (checked when the registry is built).
- `intents` and `sensors` carry no duplicates.

## Intents on the wire

The manifest speaks the vocabulary other systems read; the backends speak the intent kinds
quackd has used since 0.1. One table maps them (`INTENT_KIND_FOR`): `twist → move`,
`skill → do`, `gaze → look`, `sound → sound`, `joint → joint`, `pose → pose`,
`gripper → gripper`.

## The digest

`digest()` is sha256 of the canonical sorted-key JSON **excluding `id` and `backend`**,
first 16 hex characters: a capability fingerprint. The same robot over `sim2d`, `mujoco` and `mock`
hashes the same; a robot with one more verb hashes differently. Discovery advertises it in
TXT (`sha`) so a manifest fetched out of band can be checked against what was announced,
and MCP `robot_list` returns it.

## Examples

The seven shipped manifests, from `quackd list-verbs --robot ...` or `describe()`:

| Robot | embodiment / mobility | intents | verbs |
|---|---|---|---|
| `microduck:sim2d` | biped / legged | twist, skill, gaze, sound, pose | observe, report_state, stop, say, move, go_to, search_scan, approach_and, sit, stand, stand_up, kick, grab, gaze, quack |
| `lerobot:mock` | arm / none | joint, gripper, skill | observe, report_state, stop, move_joints, gripper, place, pick |
| `rosbridge:mock` | wheeled / wheeled | twist | observe, report_state, stop, move, go_to, search_scan, approach_and |
| `open_duck:sim2d` | biped / legged | twist, gaze, sound, skill | report_state, stop, move, observe, go_to, search_scan, approach_and, say, quack, gaze, express |
| `xlerobot:mock` | wheeled / wheeled | twist, joint, gripper | observe, report_state, stop, move, move_joints, gripper, go_to, search_scan, approach_and |
| `alohamini:mock` | wheeled / wheeled | twist, pose, joint, gripper | observe, report_state, stop, move, lift, move_joints, gripper, home_arms, go_to, search_scan, approach_and |
| `toddlerbot:mock` | humanoid / legged | skill, gaze, twist | observe, report_state, stop, stand, perform, look, move, go_to, approach_and, search_scan |

What each body lacks is as important as what it has: the arm cannot `move`, neither the
base nor the cart can `say`, and a `.duck` that `requires` one of those fails
validation against that robot with a field-level message
([duck-spec.md](duck-spec.md)).
