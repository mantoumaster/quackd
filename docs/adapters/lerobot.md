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

## If you already own one

You have LeRobot, and LeRobot already does what this arm is known for: teleoperation from a
leader, recording episodes, training a policy, evaluating one. quackd does none of that and
is not trying to. It does the one thing LeRobot leaves to you, which is deciding what the arm
should do next, and it does it by putting a language model in that seat under a contract the
model cannot exceed.

| What you want | What does it |
|---|---|
| drive the follower from a leader arm, record a dataset, train or run a policy | LeRobot's own tools. quackd never calls them and never writes to your datasets |
| have a model choose the next verb, inside limits you wrote down, with every refusal recorded | quackd |
| run one verb by hand, right now | quackd over MCP (`robot_run_verb`), or LeRobot's own Python API |

The contract is a `.duck` file: the verbs the model may use, the budget in steps and minutes,
and what counts as success ([duck-spec.md](../duck-spec.md)). The model never emits a motor
command. It picks a verb, and quackd checks the allowlist, the preconditions, this arm's
calibrated range and the step cap before anything reaches the bus.

Three things are true of this body that are not true of a simulator, and they shape
everything below:

- **The five body joints have no torque cap.** LeRobot writes a torque and current cap on the
  gripper and on nothing else, so a stalled elbow has nothing to save it or your finger. quackd
  reads each servo's temperature off the bus and refuses to move a joint at or above 60 °C.
- **A goal in degrees is not clamped by LeRobot.** quackd computes each joint's travel from
  your calibration file and refuses a goal outside it, rather than passing the number down.
- **There is no e-stop and no deadman.** Nothing in LeRobot stops the arm when the controlling
  process goes quiet, and a position-controlled servo holds the last goal it was given. Cutting
  the servo supply is the only thing that stops this arm in every case.

## Start here

1. **Run it with no arm attached.** `lerobot:mock` is the same verbs, the same executor and
   the same refusals, in memory, so you can see what a run looks like before you risk
   anything:

   ```bash
   uvx quackd run lerobot-lookout --robot lerobot:mock --provider fake
   ```

   The scripted pilot needs no API key. It answers `assess_task`, calls `report_state`, and
   declares:

   ```
   +  declare success: elbow_flex 90, gripper 100, shoulder_lift -90, shoulder_pan 0, wrist_flex 0, wrist_roll 0; torque on; nothing hot
   ```

2. **Bring the real arm up in the order that can only fail safely.**
   [lerobot-hardware-checklist.md](../lerobot-hardware-checklist.md) is sixteen steps and
   nothing moves until step 10. Do not skip the calibration step: quackd refuses an arm that
   has not been calibrated, because the calibration file is where every joint's travel
   comes from.
3. **Then drive it**, which is [three commands](#driving-it) depending on who is choosing the
   verbs: a task file, a one-line goal, or you from an MCP client.

## Backends

| `--robot` | Status | What it is |
|---|---|---|
| `lerobot:mock` | ✅ | an arm in memory: goals land instantly, the gripper stops on the object, a scripted policy answers `pick`, and it refuses an out-of-range goal in the same words the real one does |
| `lerobot:real` | 🧪 | an SO-101 follower through LeRobot (extra `quackd[lerobot]`, Python 3.12 or newer, torch), and one USB webcam when `--camera-url` names one; every name VERIFIED at the pin, exercised against a fake arm and a fake camera, never on hardware |

`--address` is the arm's serial port (`/dev/ttyACM0`, `COM5`), and quackd checks that it
looks like one before LeRobot opens anything. The `real` backend calls
`connect(calibrate=False)` and refuses an uncalibrated arm, in these words:

```
lerobot real: the arm is not calibrated; run LeRobot's calibration first
(it is interactive, quackd never triggers it)
```

Calibration is upstream's own interactive step, under the id quackd will use, and it writes
the file every joint's range is read from: step 5 of
[lerobot-hardware-checklist.md](../lerobot-hardware-checklist.md).

## Installing it

```bash
uv pip install "quackd[lerobot]"
uv run quackd doctor
```

Two rows in `doctor` decide whether a serial port can be opened at all:

```
- lerobot                    not installed (quackd[lerobot])
- lerobot (feetech bus)      not installed (quackd[lerobot])
```

The second one is the trap. The Feetech SDK lives in LeRobot's own `[feetech]` extra rather
than in its base dependencies, so a `pip install lerobot` gives you a package that imports
perfectly and then cannot talk to a motor. `quackd[lerobot]` asks for `lerobot[feetech]` for
that reason. Both rows have to be green before `lerobot:real` can do anything.

The extra carries a `python_version >= '3.12'` marker, because that is LeRobot's floor while
quackd's own is 3.11. On 3.11 the extra resolves to nothing at all and `doctor` keeps saying
`not installed` however many times you install it: check `python --version` first.

Without the extra, every real-arm command ends the same way, and this is what it looks like:

```
+- x FAILURE ------------------------------------------------------------------+
| lerobot:real at COM5: adapter 'lerobot' needs an extra: uv pip install       |
| 'quackd[lerobot]'                                                            |
+------------------------------------------------------------------------------+
```

## The name you give the arm is its calibration id

This catches people once, and the symptom is an arm that refuses to connect after a
calibration you watched succeed.

LeRobot stores a calibration under an **id** you choose, at
`<calibration dir>/robots/so_follower/<id>.json`. quackd asks LeRobot for the arm under the
id the manifest carries, and that id comes from the name you used:

| How you name it | The manifest id, and the calibration file LeRobot must have |
|---|---|
| `--robot lerobot:real` | `arm-01`, the default |
| `--robots arm=lerobot:real` | `arm` |
| `quackd robot add lab-arm lerobot:real --address COM5`, then `--robot lab-arm` | `lab-arm` |

So calibrate under the name you intend to use:

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=lab-arm
```

Two arms sharing an id share one file, and nothing in it says which arm it came from. `doctor`
prints the path it actually loaded, which is the fastest way to see that you calibrated `arm`
and are connecting as `arm-01`.

Registering the arm is worth it beyond the id: `quackd robot add` stores the address, the
camera url and the token, so `--robot lab-arm` carries all three and you stop retyping a COM
port ([registry.md](../registry.md)).

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
camera nor a policy; `connect()` adds `observe` when `--camera-url` named a camera and it
opened, and `pick` when a policy object was injected. It also adds what cannot be known until
the arm has answered: `extras.joint_range_deg`, every joint's travel in degrees read out of
the calibration file (`wrist_roll` included, where that travel is the whole turn upstream
records rather than anything swept, see Safety), `extras.calibration_file`, the path that came
from, `limits.step_deg`,
how far one action may move a joint (`QUACKD_LEROBOT_MAX_STEP_DEG` sets it), and
`extras.camera` with `limits.camera_fov_deg` when there is a camera.

| Verb | Kind | What it does here |
|---|---|---|
| `observe` (alias `get_frame`) | core | the arm's camera frame plus detections, only when a camera is configured |
| `report_state` | core | joint positions in degrees, whether torque is on, each servo's temperature, whether something is held |
| `stop` | core | hold: the present position becomes the goal. Never limp (see Safety) |
| `move_joints(positions, duration_s)` | extension | goal angles for one or more of the six joints, re-sent at 10 Hz until the measurement arrives; a joint that stops short is a failure, and `duration_s` is the budget |
| `gripper(open)` | extension | open or close the gripper, and report where it stopped |
| `place` | extension | open the gripper where the arm is; needs `holding` |
| `pick(target, max_s)` | extension, **confirm** | one skill intent; the arm's learned policy runs its own observe/act loop at its own rate until something is held or the time is up |

Joints are named, not numbered, and a `move_joints` call may name any subset of them:

```json
{"positions": {"wrist_roll": 10, "elbow_flex": 45}, "duration_s": 2.0}
```

The five body joints are degrees, centred on zero, and the gripper is `0..100` whatever the
others are. `duration_s` is a budget rather than a speed: the arm moves at the capped step
(5 degrees per action, re-sent ten times a second) and the verb fails if the measurement has
not arrived when the budget runs out, with where the joint actually stopped in the reason.

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

## Camera

No SO-101 has a camera in it. Whatever the kit's listing said, the arm is six servos and a
serial board, and every camera on one is a USB webcam that plugs into the *computer*: the
arm's own USB cable carries motor traffic and no video. So quackd's camera is a separate
thing you point it at.

```bash
uv run quackd doctor --robot lerobot:real --address COM5 --camera-url "opencv://0"
```

The index is OpenCV's, and `lerobot-find-cameras opencv` is what tells you which is which:
it lists every camera it can open and saves a frame from each under
`outputs/captured_images/`, so you can look at the pictures rather than guess. On a laptop
index 0 is usually the built-in webcam, so a plugged-in one is often 1 or 2. An index is a
scan position and not an identity: it can move when you replug or reboot.

| Key | Default | What it is |
|---|---|---|
| (the index) | required | `opencv://0`, or a device path, `opencv:///dev/video2` |
| `name` | `front` | what the frame is called in a policy's observation |
| `width`, `height`, `fps` | the camera's own | a mode the camera cannot do is a refusal at connect, so these are worth setting only when you know it can. Width and height come together or not at all |
| `fourcc` | the camera's own | `MJPG` is the one worth asking for if you add a second camera later: raw YUYV eats USB bandwidth |
| `rotation` | `0` | 90, 180 or 270, for a camera mounted sideways |
| `backend` | `any` | `msmf` or `dshow` on Windows, when a camera lists and then will not open |
| `fov` | unset | the lens's horizontal field of view in degrees, which is what bearings are computed from |

Anything else in the query is refused, with the shape, before LeRobot is even imported.

**The camera is quackd's, not the arm's.** LeRobot lets you give a follower its cameras, and
quackd deliberately does not: a follower's `is_connected` is the bus *and* every camera, and
`send_action` and `disconnect()` are gated on it, so one unplugged webcam would make every
move and every hold raise while the arm itself was perfectly fine. Beside the follower, a
camera that dies costs you `observe`, and a `pick` already running, and nothing else: the
heartbeat still reads the arm, the joints still move, `stop` still holds.

What that means at the bench:

- A camera you asked for and did not get is a **refusal at connect**, naming the url. You
  asked for it, and `doctor` gates its verdict on a real frame, so failing quietly would
  leave you believing you had eyes. The camera opens *before* the arm is touched, so a wrong
  index energises nothing and leaves nothing to undo; the arm connects without
  `--camera-url`.
- A camera that stops delivering later is **not** a refusal. `observe` fails with what the
  camera said (`the camera gave no frame: TimeoutError: ... too old`), and a `pick` running
  at that moment ends as a policy error, because the policy is handed the frame. The moving
  verbs carry on. `report_state` keeps working and gains a `CAMERA DOWN:` clause with the
  reason, and the health goes into the arm's state, which is what puts a dead webcam in the
  transcript on a run that has no `observe` to call. `quackd doctor` shows the same thing
  under `camera`.
- `observe` gives you bearings in the camera's own frame, and the detector behind it is an
  HSV threshold whose colour ranges are the *simulator's*. On a real desk it labels whatever
  happens to fall in one of those bands, `ball` for an orange thing and `person` for a blue
  one, and reports nothing when nothing does, so the label is a colour range's name rather
  than recognition. Distances assume the simulated ball's size, and without `?fov=` the
  bearing is uncalibrated and says so. Tuning the ranges to your own ball is in
  [the FAQ](../faq.md). What is honest whatever the detector makes of it is the frame
  itself, which a cloud model is shown every step unless you pass `--no-vision`.

**One camera.** `observe` returns one frame, so the url names one. A second view is a later
feature, not a flag that exists and is ignored.

**What a `.duck` may ask of this arm, and where.** `quackd run` and `quackd validate` check
a task against the *static* manifest, before anything is connected, and that manifest claims
no camera for `lerobot:real` because nothing knows whether you plugged one in until the arm
has answered. Both `requires:` and the allowlist are checked against it, the allowlist as a
weaker line, so a task that so much as **allows** `observe` is refused before the arm is
touched:

```
x error: lerobot-lookout cannot run on lerobot:real: observe is not provided by arm-01
```

That is why `lerobot-lookout` asks for `report_state` instead, and on an arm nobody has driven
yet that is the better question anyway: whether it answers at all, whether torque is on, and
how warm it is.

Over MCP it is the other way, and better. `robot_load_duckfile` validates against the manifest
of the robot **already connected**, so the same task loads cleanly on a session started with
`--camera-url` and is refused on one without. If you want a contract that allows `observe` on
this arm, that is where it works today.

**The pilot sees the camera whether or not `observe` is allowed.** Every step, the run loop
takes a frame and runs the detector over it, and the detections go into the observation the
model reads. On a provider that accepts pictures the frame itself is attached as well, which
is `--vision`, on by default for cloud models and off for local ones. What `observe` adds is
the ability to *ask* for a look as a deliberate act and get the frame back as a verb result,
which is what `robot_observe` does over MCP.

## Driving it

Three ways, and they differ only in who chooses the verbs.

**A task file**, which is the one with guard rails. The `.duck` names the allowlist, the
budgets and the success test, and quackd enforces all three:

```bash
quackd run lerobot-lookout --robot lerobot:real --address /dev/ttyACM0 --provider anthropic
```

`lerobot-lookout` ships with quackd and moves no joint: it reads the arm back and says what
it found. It is the first thing to point at a real arm. Writing your own is a file and a
`requires:` line, and `quackd validate <file> --robot lerobot:real` refuses it before a run
if this arm does not provide a verb it asks for.

**A one-line goal**, for when you want a single verb and there is no file for it. The model
still has to pass `assess_task`, and everything else still applies:

```bash
quackd run --goal "roll the wrist ten degrees and stop" --robot lerobot:real \
  --address /dev/ttyACM0 --provider anthropic --max-steps 3
```

Keep `--max-steps` small. `--provider fake` will not do here: the scripted pilot answers a
free-form goal with a fixed script that ignores it.

**From an MCP client**, which is you choosing each verb with the model doing the talking.
This is the only way to call `observe` on a real arm today, and the only way to run exactly
one verb and stop:

```json
{
  "mcpServers": {
    "arm": {
      "command": "uvx",
      "args": ["quackd", "serve-mcp", "--robot", "lerobot:real",
               "--address", "COM5", "--camera-url", "opencv://1"]
    }
  }
}
```

Nine `robot_*` tools appear. `robot_list_verbs` first, then `robot_assess_task` with a
verdict, which `robot_run_verb` requires before anything that moves the body, then
`robot_run_verb(verb="move_joints", params={...})`. Both clients, the full tool list and a
two-minute script: [mcp.md](../mcp.md).

### The rehearsal: `--dry-run`

`--dry-run` connects to the arm for real and sends it nothing. Read-only verbs actually run,
so `report_state` reads the servos and `observe` takes a frame; every verb that would move
something is printed and skipped:

```
[dry-run] would run move_joints({'positions': {'wrist_roll': 10.0}, 'duration_s': 2.0})
[dry-run] move_joints not sent
```

This is worth doing on a real arm before the first real run. It exercises the port, the
calibration, the temperature read, the model, the allowlist and the budgets, and proves
which verbs the model is going to reach for, with the arm standing still. The heartbeat is
live throughout, so a dry run also tells you whether the arm answers reliably.

### What `pick` needs, and what it does not have

`pick` hands the whole arm to a learned policy, and it is confirm-gated for that reason. On
`lerobot:real` it is **absent from the manifest unless a policy object was injected in
Python**, and there is no CLI flag that loads one today. `real.py` has `load_policy(path)`,
built entirely from verified upstream names, but nothing has run it end to end: it is the
`POLICY_PIPELINE` row in the UNVERIFIED table below. So a trained ACT checkpoint reaches this
arm only through code you write around the adapter, and `quackd run --robot lerobot:real`
will not offer `pick` at all. Every other verb is fully reachable from the CLI. If you get a
policy running this way, the policy's own actions still pass the step cap and the range
refusal, which is quackd's rule and not LeRobot's.

## Safety

Each of these exists because upstream could not answer a question quackd has to ask; the
reasoning is in [ADR-0036](../adr/0036-what-the-arm-does-not-say.md). What stops each body in
quackd, side by side, is [safety.md](../safety.md).

- **The heartbeat reads the arm.** `is_connected` is the serial port's open flag and stays
  `True` with the cable pulled, so the heartbeat is a round trip to the motors, and a dead arm
  ends the run.
- **Torque and temperature are measured.** `get_observation()` reads positions only, so
  `Torque_Enable` and `Present_Temperature` are read off the bus. A body joint at or above
  60 °C refuses `move_joints` and `pick`; the servo's own cut-off is 70 °C.
- **A goal outside the calibrated range is refused, on four joints of the five.** LeRobot
  does not clamp a degrees goal, so quackd computes each joint's travel from the calibration
  file and refuses instead. `wrist_roll` is the exception, and it is upstream's: its
  calibration deliberately does not sweep that joint, printing *move all joints except
  'wrist_roll'* and recording a full encoder turn for it instead
  (`up.WRIST_ROLL_IS_A_FULL_TURN`). Its travel therefore comes out as -180..180, and a refusal
  that cannot be narrower than the whole turn cannot catch anything. Treat `wrist_roll` as
  unguarded and give it small goals.
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
  connecting drops it briefly too. Support the arm at both ends, including at the end of a
  `doctor` probe.
- **`pick` is confirm-gated**: a learned policy moves the whole arm. Its actions go through
  the same step cap and range check as a verb's.

## When it will not work

Every message below is quackd's own, quoted from the code that raises it. The arm is not
touched by anything in the first block: these all happen before or during connect.

| What you see | What it means | What to do |
|---|---|---|
| `adapter 'lerobot' needs an extra: uv pip install 'quackd[lerobot]'` | the extra is not installed in the environment you are running from | install it, and check `python --version` is 3.12 or newer, because the extra's marker silently resolves to nothing below that |
| `doctor` shows `lerobot` green and `lerobot (feetech bus)` missing | LeRobot is installed without its `[feetech]` extra, so it imports and cannot open a serial port | `uv pip install "quackd[lerobot]"`, which asks for `lerobot[feetech]` |
| `lerobot real: --address must be the arm's serial port` | no `--address` at all | pass the port. `--address needs --robot, so quackd knows what it is connecting to` means the opposite mistake, an address with no robot to apply it to |
| `lerobot real: --address 'x' is not a serial port; it looks like COM5 on Windows or /dev/ttyACM0 elsewhere` | the address is not port-shaped | on Windows find it in Device Manager under Ports; on Linux it is usually `/dev/ttyACM0` |
| `lerobot real: connect failed: ...` naming a port that cannot be opened | the port is wrong, or something else already owns it | LeRobot's own words come through, and they name `lerobot-find-port`, which is the way to be sure. The Feetech bus has one owner at a time, so close any teleoperation, recording or serial monitor still holding it, and on Linux check that your user can open the port (upstream's own line is `sudo chmod 666 /dev/ttyACM0`; the port's group, usually `dialout`, is the version that survives a reboot) |
| `lerobot real: the arm is not calibrated; run LeRobot's calibration first` | LeRobot read the motors back and they do not match a calibration | run `lerobot-calibrate` under the id quackd will use, and see [the id section](#the-name-you-give-the-arm-is-its-calibration-id) |
| `lerobot real: the arm reports no calibration file, so nothing knows how far each joint travels` | there is no file for this id | the same fix, and check the path `doctor` prints |
| `lerobot real: only so101_follower is wired` / `this robot has no motors bus` | the config is not an SO-101 follower | quackd drives this one body; an SO-100 shares the calibration directory but is not wired here |
| `lerobot real: --camera-url 'opencv://7' did not open: ...` | the index is wrong, or the camera will not open under this backend | try the index `lerobot-find-cameras opencv` printed, add `?backend=msmf` on Windows, or drop a `width`/`height`/`fps` you pinned. The arm was not touched |
| `lerobot real: --camera-url '...': fps='abc' is not a whole number` | a query key or value quackd does not accept | the message lists every key; this is refused before LeRobot is imported |

And once it is running:

| What you see | What it means | What to do |
|---|---|---|
| `move_joints: shoulder_pan=170 is outside this arm's calibrated range -100..100` | the goal is outside the travel in your calibration file | aim inside it, or recalibrate if the file does not match the arm's real travel. Nothing was sent. On `wrist_roll` this refusal will never fire, because upstream records a full turn for that joint rather than a sweep |
| `cannot move_joints: elbow_flex reads 61°C: let the arm cool before moving it ...` | the heat gate, below the servo's own 70 °C cut-off | let it cool. A joint that trips its own protection goes slack without announcing it |
| `move_joints: elbow_flex is at 12 with a goal of 45, and it has stopped moving` | a stall: five ticks of 0.1 s in which no watched joint moved more than half a step | something is in the way, a mechanical limit the calibration does not know about, or a tripped servo. The arm is held first. The same sentence ending `when the time ran out` means `duration_s` was too short for the capped 50 degrees a second |
| `the camera gave no frame: TimeoutError: ... too old` | the webcam stalled or was unplugged | only `observe` is affected, and a `pick` in flight. The arm carries on, and `report_state` starts saying `CAMERA DOWN:` with the reason, so a run that cannot call `observe` still records it |
| `cannot move_joints: the arm's torque is off; enable it from LeRobot first (quackd never toggles torque)` | torque reads off | quackd never switches torque on or off, by design. A fresh connect re-enables it, so torque still off after one points at a tripped servo or the supply |
| `cannot place: nothing is held: pick something first` | the `holding` precondition | holding is inferred from the gripper stopping short of shut, so an empty hand reads as nothing held |
| the run ends saying the arm did not answer | the heartbeat's round trip to the motors failed | the cable, the power, or a servo that has tripped. The arm holds its last goal under torque |
| the arm sags when the run ends | LeRobot's `disconnect()` disables torque by its own default, at the end of every clean session | support it, or fold it somewhere it can rest before you exit |

### Reported by owners, not by us

Nobody here has had an SO-101 on a desk, so the list above is what quackd's own code does.
These are things SO-101 owners report, collected while writing this page and **not verified
against hardware by anyone in this repo**. They are here because they are the failures that
cost an afternoon, not because we can vouch for them.

- **No serial port appears at all.** A charge-only USB cable, or the arm's barrel jack out:
  USB does not power the controller board.
- **A fresh kit has every motor on id 1**, so the bus sees duplicates. Upstream's
  `lerobot-setup-motors` walks them one at a time, each motor connected alone and not yet
  daisy-chained.
- **A Waveshare board has two jumpers**, and they belong on the `B` (USB) channel. Upstream's
  own tip is that the power cable can work loose while you handle the board.
- **Two identical webcams swap indices** between boots. On Linux a stable path avoids it
  entirely: `opencv:///dev/v4l/by-id/<device-id>`.
- **Two uncompressed camera streams on one USB controller exceed its bandwidth**, and the
  camera opens and then delivers nothing. `?fourcc=MJPG` is the usual answer.
- **On Linux a pinned size can fail under the default backend** rather than in the camera:
  `?backend=v4l2` is worth trying before you conclude the webcam cannot do the mode.

If you hit one of these, or fail to, that is exactly what the
[checklist](../lerobot-hardware-checklist.md)'s *What to report* is asking for.

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
| `calibrate() records wrist_roll as a full turn` | upstream sweeps every joint except that one and writes 0..4095 for it, so quackd's range refusal is real on four body joints and inert on the fifth |
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
| `OpenCVCameraConfig.color_mode defaults to ColorMode.RGB` | quackd passes RGB explicitly anyway |
| `lerobot.cameras.opencv.OpenCVCamera(config)` | the camera quackd builds and owns, beside the follower |
| `OpenCVCameraConfig(index_or_path, fps=None, width=None, height=None, color_mode=ColorMode.RGB, rotation=Cv2Rotation.NO_ROTATION, warmup_s=1, fourcc=None, backend=Cv2Backends.ANY)` | what `--camera-url`'s query keys fill in |
| `lerobot.cameras exports Camera, CameraConfig, ColorMode, Cv2Backends, Cv2Rotation` | the config is deliberately not among them, so quackd imports from both |
| `Cv2Backends: ANY, V4L2, DSHOW, PVAPI, ANDROID, AVFOUNDATION, MSMF` | the backend is a config field, so `?backend=msmf` needs no patched source. `--camera-url` takes the five that name a platform you could be on; the other two would only ever be a refusal |
| `Camera.connect(warmup=True)` | it reads frames before returning, so a camera that opens and never delivers fails here |
| `connect() raises ConnectionError on an index that will not open` | quackd passes its words through, and they name `lerobot-find-cameras opencv` |
| `a requested fps or size that the camera refuses raises RuntimeError` | why quackd asks for no mode unless you name one |
| `an unset fps, width or height keeps the camera's own mode` | what makes an unknown webcam in a lab drawer work |
| `Camera.read_latest(max_age_ms=500)` | the newest buffered frame; it raises when the camera has stalled, and `get_frame` turns that into a reason |
| `Camera.disconnect()` | |
| `a follower's cameras are part of its connected state` | `is_connected`, `send_action` and `disconnect()` all include them, which is why quackd's camera is not the follower's |
| `lerobot-find-cameras opencv` | how an owner learns which index is which: it saves a frame per camera |
| `lerobot-find-port` | upstream's own port finder: it names the port that disappears when you unplug the arm, which is the only way to be sure which one it is |
| `lerobot.policies.pretrained.PreTrainedPolicy` | |
| `lerobot.configs.policies.PreTrainedConfig` | a checkpoint's own config, read before the policy class is built; the one policy name that is not in the factory |
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
| `CAMERA_INDEX_MOVES` | an index is a scan position, not an identity: it can move on a replug or a reboot, and a laptop's own webcam usually holds 0. quackd records the index it opened and cannot tell you it is the camera you meant |
| `WINDOWS_CAMERA_BACKEND` | which backend a Windows machine needs for a given webcam is not knowable in advance, so quackd keeps upstream's ANY and gives the owner `?backend=msmf` |

## Status

`lerobot:mock` runs every arm verb through the executor in the test suite, including the
confirm gate on `pick`, the `holding` precondition on `place` and the heat refusal.
`lerobot:real` is exercised against a fake arm and a fake policy (verified method names, no
serial port). Nobody has run it on an arm, and this page will say so until someone has.

## How to help

If you have an SO-101 on a desk, work through
[lerobot-hardware-checklist.md](../lerobot-hardware-checklist.md) in order: nothing moves
until step 10. `lerobot-lookout` is the first task to point at it; it asks for `report_state`
rather than `observe`, because a `.duck` is checked against the static manifest, which
cannot know whether you brought a webcam. What most needs a real arm is that checklist's
*What to report*, and a camera is the last line of it. Open an issue with the transcript.
