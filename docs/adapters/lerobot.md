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
[`adapters/lerobot/src/quackd_lerobot/upstream_api.py`](../../adapters/lerobot/src/quackd_lerobot/upstream_api.py),
and why the adapter is shaped the way it is is
[ADR-0036](../adr/0036-what-the-arm-does-not-say.md).

> [!NOTE]
> **The `real` backend has run on an arm. Once.** On 2026-09-15 an SO-101 follower, calibrated
> as `arm-01` and reached with no registered name, ran `lerobot-lookout` and then free-form `--goal` runs
> on Windows 11 with Python 3.12.12, lerobot 0.6.1 and quackd 0.9.0, piloted by OpenAI's
> `gpt-6-astra`. It also went limp and fell at the end of every one of those runs, which is what
> [the rest pose](#the-rest-pose) now exists to fix. What ran, what went wrong and what that
> afternoon did not measure is in [Status](#status); the account from the other end, an empty
> laptop to a waving arm, is [lerobot-first-run.md](../lerobot-first-run.md).

```bash
# offline, the default
uv run quackd run lerobot-lookout --robot lerobot:mock --provider fake

uvx --from "quackd[lerobot]" quackd list-verbs --robot lerobot:mock
uvx --from "quackd[lerobot]" quackd validate ducks/find-and-kick.duck --robot lerobot:mock     # exit 1: requires ... does not provide it
uvx --from "quackd[lerobot,microduck]" quackd serve-mcp --robots arm=lerobot:mock,duck=microduck:sim2d   # an arm and a duck behind one MCP server

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

If you have never run quackd or LeRobot before, start at
[lerobot-first-run.md](../lerobot-first-run.md) instead: it is this arm from an empty laptop,
including which model to bring and what it can actually see. This page assumes you already
drive the arm.

1. **Run it with no arm attached.** `lerobot:mock` is the same verbs, the same executor and
   the same refusals, in memory, so you can see what a run looks like before you risk
   anything:

   ```bash
   uvx --from "quackd[lerobot]" quackd run lerobot-lookout --robot lerobot:mock --provider fake
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
| `lerobot:real` | 🧪 | an SO-101 follower through LeRobot (extra `quackd[lerobot]`, Python 3.12 or newer, torch), and as many USB webcams as `--camera-url` names; every name VERIFIED at the pin, exercised against a fake arm and a fake camera, and run for one afternoon on one real arm, 2026-09-15, with lerobot 0.6.1, one webcam and no policy |

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

A bare `uv pip install quackd` brings no robot at all, this arm included. The adapter is its
own distribution, `quackd-lerobot`, and the extra is what pulls it in:

```bash
uv pip install "quackd[lerobot]"
uv run quackd doctor
```

That is two packages: quackd's own adapter, which is where `lerobot:mock` and `lerobot:real`
both live, and `lerobot[feetech]`, the SDK `real` drives the arm with. The adapter announces
itself through the `quackd.adapters` entry point group, so there is nothing to register by
hand. `uv pip install quackd-lerobot` is that adapter without the SDK, which is enough for
`lerobot:mock` and for reading the manifest, and enough for nothing else.

Two rows in `doctor` decide whether a serial port can be opened at all:

```
· lerobot                    not installed (quackd[lerobot])
· lerobot (feetech bus)      not installed (quackd[lerobot])
```

The second one is the trap. The Feetech SDK lives in LeRobot's own `[feetech]` extra rather
than in its base dependencies, so a `pip install lerobot` gives you a package that imports
perfectly and then cannot talk to a motor. `quackd[lerobot]` asks for `lerobot[feetech]` for
that reason. Both rows have to be green before `lerobot:real` can do anything.

The SDK carries a `python_version >= '3.12'` marker, because that is LeRobot's floor while
quackd's own is 3.11. On 3.11 the extra installs the adapter and the SDK resolves to nothing,
so `lerobot:mock` works and `doctor` keeps saying `not installed` however many times you
install it: check `python --version` first.

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

Registering the arm is worth it beyond the id: `quackd robot add` stores the address, every
camera url and the token, so `--robot lab-arm` carries all of them and you stop retyping a COM
port ([registry.md](../registry.md)). It is also the only place a
[rest pose](#the-rest-pose) can live, which is what stops the arm falling when a run ends.

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
`extras.camera` with `limits.camera_fov_deg` when there is a camera. With more than one camera
it also carries `extras.cameras`, their names in the order the urls were given, primary first.
That key is written only when there are two or more: a single camera's name is quackd's own
default rather than anything you chose, and a pilot told its one view is called `front` starts
naming it in sentences nobody needs.

| Verb | Kind | What it does here |
|---|---|---|
| `observe` (alias `get_frame`) | core | one frame plus detections, only when a camera is configured. With several cameras it is the primary's frame, because a detection is a bearing and a bearing belongs to one lens |
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
arm's own USB cable carries motor traffic and no video. So a camera here is a separate thing
you point quackd at, one of them or [several](#several-cameras).

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
| `name` | `front` | what the frame is called in a policy's observation, in the label the model reads and in `frames/NNNN-<name>.png`. **Required on every url once you give more than one** |
| `width`, `height`, `fps` | the camera's own | a mode the camera cannot do is a refusal at connect, so these are worth setting only when you know it can. Width and height come together or not at all |
| `fourcc` | the camera's own | `MJPG` is the one worth asking for the moment there are two cameras: raw YUYV eats USB bandwidth |
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
  under `camera`. With several cameras a stall costs that camera's picture and nothing else,
  and both places name it: `CAMERA DOWN: side: ...` rather than a clause that does not say
  which eye closed, and one `camera <name>` row per camera in `doctor`.
- `observe` gives you bearings in the camera's own frame, and the detector behind it is an
  HSV threshold whose colour ranges are the *simulator's*. On a real desk it labels whatever
  happens to fall in one of those bands, `ball` for an orange thing and `person` for a blue
  one, and reports nothing when nothing does, so the label is a colour range's name rather
  than recognition. Distances assume the simulated ball's size, and without `?fov=` the
  bearing is uncalibrated and says so. Tuning the ranges to your own ball is in
  [the FAQ](../faq.md). What is honest whatever the detector makes of it is the frame
  itself, which a cloud model is shown every step unless you pass `--no-vision`.

### Several cameras

`--camera-url` repeats. This arm is the only body that reads more than one: every other one
refuses a second rather than opening the first and dropping the rest, and says who takes
several.

```bash
quackd robot add arm-01 lerobot:real --address COM5 \
    --camera-url "opencv://1?name=top" --camera-url "opencv://2?name=side"
```

```
✓ added arm-01: lerobot:real at COM5
  quackd run <duck> --robot arm-01
```

`quackd robot show arm-01` lists them back in the order they were given (the top of it):

```
name       arm-01
robot      lerobot:real
body       lerobot-so101 (arm, mobility none) 5 verbs: report_state, stop, move_joints, gripper,
           place
address    COM5
camera     opencv://1?name=top
           opencv://2?name=side
rest pose  -
```

The rules, all of them refusals rather than surprises:

| Rule | Why |
|---|---|
| with two or more, **every** url carries `?name=` | the name is the only thing telling the views apart, in the label on each picture the model reads, in a `pick` policy's observation dict, where every camera arrives under its own name, and in `frames/NNNN-<name>.png`. With one camera the name is optional, `front` is the default, and the frames are `frames/NNNN.png` |
| the names are unique | two views called `top` are two pictures a model cannot tell apart |
| an index appears once | two handles on one webcam is not two views, it is a camera that will not open twice |
| the **first** url is the primary | `--fov-deg` describes it, the `camera:` detections line reports it and nothing else, `observe` returns its frame, and the verbs that steer by sight read it alone. Those run at 10 Hz, and fetching every camera inside that loop would blow the deadman window. This arm has no such verb today, so here the rule is about the detections line and `observe` |

Everything else arrives at the model. Every frame reaches it each step, labelled with the
name of the camera that took it, on Claude, both OpenAI APIs, Gemini, and any
OpenAI-compatible local server with `--vision` on. The pilot is also told, in as many words,
which one is primary and that the `camera:` line describes that view and no other.

What happens when one of them dies mid-run depends on which one:

| Which camera stopped | What the model still gets |
|---|---|
| a secondary | every other view, and the `camera:` detections line unchanged. The failure is named in `report_state` and in `doctor`'s `camera <name>` row |
| the primary | the other views still arrive as pictures, and the detections line reports nothing seen. A bearing read off a different lens would point somewhere else, so quackd reports nothing rather than something from the wrong camera |

A second camera that will not open is a refusal **before the arm is energised**, and it lets
go of the first on the way out. Half a set of eyes nobody asked for is worse than the
refusal, because the frames would still arrive and look right.

> [!WARNING]
> Two uncompressed 640x480 streams on one USB controller can exceed its bandwidth. Both
> cameras open, and then one or both deliver nothing. `?fourcc=MJPG` on each is the answer,
> and a different physical USB controller for the second camera is the other one. This is an
> owner report rather than something measured here: the 2026-09-15 bench ran one webcam.

**The cost is pictures.** Each request carries the images from the last two exchanges, so two
cameras is four pictures per request where one camera is two, and that is what you pay in
tokens on every step of every run. It is worth it for a wrist view plus an overhead view. It
is not worth it for two views of the same thing.

> [!NOTE]
> A local server, or a particular model behind one, may accept only one image per message.
> If it refuses a request with two, pass a single `--camera-url`.

`robots.json` stores a string for one camera and a list for several, so a registry file
written by 0.9 loads unchanged:

```json
"camera_url": ["opencv://1?name=top", "opencv://2?name=side"]
```

**What a `.duck` may ask of this arm, and where.** `quackd run` and `quackd validate` check
a task against the *static* manifest, before anything is connected, and that manifest claims
no camera for `lerobot:real` because nothing knows whether you plugged one in until the arm
has answered. Both `requires:` and the allowlist are checked against it, the allowlist as a
weaker line, so a task that so much as **allows** `observe` is refused before the arm is
touched:

```
x error: lerobot-lookout cannot run on lerobot:real: observe is not provided by arm-01
```

That is why `lerobot-lookout` asks for `report_state` instead, and on an arm you have not
driven before that is the better question anyway: whether it answers at all, whether torque is
on, and how warm it is. It is what ran first on the bench on 2026-09-15, and it is what ran
again with `--provider fake` to separate the arm from the model.

Over MCP it is the other way, and better. `robot_load_duckfile` validates against the manifest
of the robot **already connected**, so the same task loads cleanly on a session started with
`--camera-url` and is refused on one without. If you want a contract that allows `observe` on
this arm, that is where it works today.

**The pilot sees the camera whether or not `observe` is allowed.** Every step, the run loop
takes a frame from every camera, runs the detector over the primary's, and the detections go
into the observation the model reads. On a provider that accepts pictures the frames
themselves are attached as well, each labelled with its camera's name, which is `--vision`, on
by default for cloud models and off for local ones. What `observe` adds is the ability to *ask*
for a look as a deliberate act and get the frame back as a verb result, which is what
`robot_observe` does over MCP.

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

A session parks the arm at both ends, the same as a run does, and **refuses to start** if it
cannot reach the recorded rest pose. Repeat `--camera-url` here too, and the session reads
every camera you name.

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
live throughout, so a dry run also tells you whether the arm answers reliably. Two of the
bench's dry runs on 2026-09-15 ended early, and both endings were the rehearsal doing its job:
one on a heartbeat round trip that failed once and never again, one on a pilot that answered
`uncertain` and a human who said no.

A dry run **never moves the arm**, and that includes the rest move at either end. It also means
a dry run on an arm that is not at its recorded rest pose ends with torque left on, because
nothing drove it there. That is [the rest pose](#the-rest-pose)'s rule and not an exception to
it.

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
- **Torque is released only where the arm is known to be at its recorded rest pose.**
  `disconnect()` disables it by LeRobot's default, which quackd keeps, because an arm at rest
  should be limp: that is what "at rest" means. So before the disconnect quackd reads the
  joints one last time, and where they are not the pose you recorded it turns that default off
  and leaves the arm holding itself up, with one line saying so:

  ```
  the arm is not at its rest pose (...), so torque was left on and it will not fall:
  hold the arm and cut its power, or run again
  ```

  With no rest pose recorded there is nothing to check against, nothing changes, and the arm
  goes limp at the end of every clean session exactly as it did in 0.9. See
  [The rest pose](#the-rest-pose).
- **Connecting still drops torque briefly, and that has not changed.** `configure()` runs
  inside `torque_disabled()`, so the arm is limp for the moment between the port opening and
  the configuration landing, whatever any rest pose says. Support the arm when a session
  starts, including at the start of a `doctor` probe.
- **`pick` is confirm-gated**: a learned policy moves the whole arm. Its actions go through
  the same step cap and range check as a verb's.

## The rest pose

A LeRobot arm goes limp when it is disconnected, because `disconnect()` disables torque by its
own default and quackd keeps that default. On the bench on 2026-09-15 that meant the arm fell
at the end of every single run. Runs also started from wherever the previous one had left the
arm, so the pose a model was improvising from was different every time.

A rest pose fixes both. You fold the arm by hand, tell quackd where that is, and quackd drives
it there at both ends of every run.

> [!IMPORTANT]
> None of this has been run on an arm. It was written after the day the arm fell, and it is
> exercised against a fake arm and the mock, the same standing as every other `lerobot:real`
> behaviour on this page. The first person to record a pose on real hardware is finding out
> whether it holds, so read [section 07 of the first run](../lerobot-first-run.md#07-record-the-rest-pose)
> with a hand near the power switch and say what happened.

### Recording it

```
quackd robot rest-pose NAME [--clear] [--yes] [--address ADDR] [--registry-dir DIR] [--json]
```

Fold the arm first, with nothing connected, so the pose you record is one it can hold with
torque off. Then run the command: it connects, reads every joint, prints them, asks, and keeps
the pose in `~/.quackd/robots.json` beside the address and the cameras. It opens no camera,
because reading joints needs none, and it drives the arm nowhere, because the point is to let
go of it at the pose you are choosing now rather than the one you are replacing.

```
$ quackd robot rest-pose arm-01 --yes     # without --yes it prints the joints and asks
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

Those numbers are the **mock** arm's, which is where that transcript was captured. A real
SO-101 folded on a desk reports its own, and they will not look like these.

`quackd robot show NAME` prints the pose back in a `rest pose` row, and `--json` puts it under
`rest_pose`. `--clear` forgets it:

```
$ quackd robot rest-pose arm-01 --clear
✓ cleared arm-01's rest pose
  a run now leaves the arm where it stands, and torque drops there
```

A rest pose belongs to a **registered** robot, because it is read off the arm and kept under
its name. There is no `--rest-pose` flag on `quackd run`.

### What a run does with it

| When | What happens |
|---|---|
| the start of every run | the arm is driven to the pose before the pilot gets control, so what the model improvises from is the same arm every time. A run that cannot get there aborts **before the first LLM call** |
| the end of every run | between the `stop` and the disconnect, which is the only window in which putting the arm down changes whether it falls. On every exit path there is: success, failure, infeasible, a spent budget, an abort, an error and Ctrl-C |
| a dry run | nothing. `--dry-run` never moves the arm, and that includes the rest move |
| an MCP session | the same at both ends, and the session refuses to start if it cannot get there |

Both ends are narrated, so the transcript says what happened rather than leaving you to infer
it from a joint reading:

```
·  note    moving to the rest pose
·  note    already at the rest pose
```

### The torque rule

Torque is released **only** where the arm is known to be at the pose you recorded. A joint is
at its recorded angle when it reads within **5 degrees** of it, the same tolerance a
`move_joints` goal is judged by, and every joint of the pose has to be reported and within it.
Where that does not hold, quackd turns LeRobot's `disable_torque_on_disconnect` off on the
config instance before the call, closes the port with every motor still holding its goal, and
prints one line:

```
the arm is not at its rest pose (...), so torque was left on and it will not fall:
hold the arm and cut its power, or run again
```

The parenthesis names the joints and how far short they are, and, where the rest move itself
failed, why it failed.

> [!CAUTION]
> This is a behaviour change. A probe or a dry run on an arm away from its recorded rest pose
> now leaves torque **on** where it used to drop it. The arm is holding itself up and the
> servos are drawing current until something stops them: hold the arm and cut its power, or run
> again and let the arm park itself. An arm with no rest pose recorded behaves as it always
> did.

### What is driven, and what is not

Only the **five body joints** are ever driven. The gripper is recorded, printed and stored, and
never commanded, for the same reason `stop` omits it: LeRobot writes only the keys it is given,
so leaving the gripper out keeps whatever squeeze is already commanded, and a rest move that
re-sent the gripper would open a hand that is holding something.

The rest move is also the one move quackd sends **unclipped**. A pose read off the arm is where
the arm physically was, and an arm folded to rest often sits outside the travel its own
calibration recorded: the bench arm folded to `shoulder_lift` -113.5 against a calibrated
±84.2. The out-of-range refusal that guards `move_joints` would refuse to put that arm down, so
the rest move does not go through it. Record a pose you are willing to have the arm driven to
without that check.

The move itself is re-sent at 10 Hz under the same 5 degree step cap as any other, on a budget
computed from how far the arm has to travel and bounded so a teardown cannot hold a run open
for minutes. It stalls the same way a `move_joints` does, five ticks of 0.1 s in which nothing
moved, and on a stall or a spent budget it holds where it is and reports how far short it
stopped. Either of those is an arm that keeps its torque.

### doctor, and `robot list --probe`

`quackd doctor` returns a probed arm to its rest pose too, and says so in a `rest pose` row.
That matters because a `doctor` probe disconnects like anything else, and a disconnect is what
dropped torque and put the bench arm on the desk:

```
│ rest pose         │ at it already                                                               │
```

| What the row says | What it means |
|---|---|
| `at it already` | the arm was within tolerance when `doctor` looked, and nothing moved |
| `returned to it` | `doctor` drove it back, and torque dropped at the end |
| `none recorded (quackd robot rest-pose <name>)` | nothing to park to, so the probe ended the way it always did: limp |
| `not reached: ...` | the probe **fails**, the arm kept its torque, and the line about holding it and cutting the power comes out as an advisory under the table |

`quackd robot list --probe` does **not** move the arm. It reads, lets go, and where the arm was
not already at its pose it keeps torque and says so on the same line that says the arm
answered:

```
┌────────┬──────────────┬────────────────────────────────────────────┐
│ name   │ robot        │ reachable                                  │
├────────┼──────────────┼────────────────────────────────────────────┤
│ arm-01 │ lerobot:mock │ ✓ ok, torque left on: not at its rest pose │
└────────┴──────────────┴────────────────────────────────────────────┘
```

### Only this arm parks

Every other body refuses a rest pose rather than accepting one and ignoring it, whether it has
joints or not:

```
$ quackd robot rest-pose duck --yes
✗ error: duck (microduck:mock) has no joints, so there is no rest pose to record
  a rest pose is for an arm: quackd list-adapters

$ quackd robot rest-pose cart --yes     # xlerobot:mock, which HAS joints
✗ error: cart (xlerobot:mock) has joints, and quackd does not drive it to a rest pose yet:
only the LeRobot arm does today
```

## When it will not work

Every message below is quackd's own, quoted from the code that raises it. The arm is not
touched by anything in the first block: these all happen before or during connect.

| What you see | What it means | What to do |
|---|---|---|
| `adapter 'lerobot' needs an extra: uv pip install 'quackd[lerobot]'` | either the adapter package or LeRobot itself is missing from the environment you are running from | install the extra, and check `python --version` is 3.12 or newer, because the SDK's marker silently resolves to nothing below that |
| `doctor` shows `lerobot` green and `lerobot (feetech bus)` missing | LeRobot is installed without its `[feetech]` extra, so it imports and cannot open a serial port | `uv pip install "quackd[lerobot]"`, which asks for `lerobot[feetech]` |
| `lerobot real: --address must be the arm's serial port` | no `--address` at all | pass the port. `--address needs --robot, so quackd knows what it is connecting to` means the opposite mistake, an address with no robot to apply it to |
| `lerobot real: --address 'x' is not a serial port; it looks like COM5 on Windows or /dev/ttyACM0 elsewhere` | the address is not port-shaped | on Windows find it in Device Manager under Ports; on Linux it is usually `/dev/ttyACM0` |
| `lerobot real: connect failed: ...` naming a port that cannot be opened | the port is wrong, or something else already owns it | LeRobot's own words come through, and they name `lerobot-find-port`, which is the way to be sure. The Feetech bus has one owner at a time, so close any teleoperation, recording or serial monitor still holding it, and on Linux check that your user can open the port (upstream's own line is `sudo chmod 666 /dev/ttyACM0`; the port's group, usually `dialout`, is the version that survives a reboot) |
| `lerobot real: the arm is not calibrated; run LeRobot's calibration first` | LeRobot read the motors back and they do not match a calibration | run `lerobot-calibrate` under the id quackd will use, and see [the id section](#the-name-you-give-the-arm-is-its-calibration-id) |
| `lerobot real: the arm reports no calibration file, so nothing knows how far each joint travels` | there is no file for this id | the same fix, and check the path `doctor` prints |
| `lerobot real: only so101_follower is wired` / `this robot has no motors bus` | the config is not an SO-101 follower | quackd drives this one body; an SO-100 shares the calibration directory but is not wired here |
| `lerobot real: --camera-url 'opencv://7' did not open: ...` | the index is wrong, or the camera will not open under this backend | try the index `lerobot-find-cameras opencv` printed, add `?backend=msmf` on Windows, or drop a `width`/`height`/`fps` you pinned. The arm was not touched |
| `lerobot real: --camera-url '...': fps='abc' is not a whole number` | a query key or value quackd does not accept | the message lists every key; this is refused before LeRobot is imported |
| `lerobot real: --camera-url 'opencv://2': it has no ?name= and 2 cameras were given` | several cameras, and one of them is unnamed | add `?name=` to every url. The message shows the shape, `opencv://1?name=top --camera-url opencv://2?name=side`, and says what the name is for |
| `lerobot real: --camera-url 'opencv://2?name=top': name='top' is already the name of 'opencv://1?name=top'` | two cameras with one name | rename one. Two views the model cannot tell apart are worse than one view |
| `lerobot real: --camera-url 'opencv://1?name=side': 1 is already 'opencv://1?name=top'` | the same index given twice | drop the duplicate, or find the other camera's index with `lerobot-find-cameras opencv`. Two handles on one webcam is not two views |
| `microduck:mock takes one --camera-url and 2 were given; only lerobot:real takes several` | a body that reads one camera was handed more | pass one url to that body. Only this arm reads more than one, and the message names it rather than opening the first and dropping the rest |

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
| the arm sags when the run ends | no rest pose is recorded, so LeRobot's `disconnect()` disables torque by its own default, at the end of every clean session | record one: `quackd robot rest-pose <name>`. Until you do, support it or fold it somewhere it can rest before you exit |
| `the arm is not at its rest pose (...), so torque was left on and it will not fall: hold the arm and cut its power, or run again` | the arm did not reach the pose you recorded, so quackd kept torque rather than dropping it | hold the arm and cut the servo supply, or run again and let the rest move try from where it now is. The parenthesis names the joints that fell short |
| a run aborts with `the arm did not reach its rest pose: ...` before any model call | the run could not start from the recorded pose | something is in the way, or the pose no longer matches the arm. Clear it with `--clear` and record it again, or move whatever is blocking the fold |

### Reported by owners, not by us

One SO-101 has been on a desk here, for one afternoon on 2026-09-15, and the list above is
still what quackd's own code does. These are things SO-101 owners report, collected while
writing this page and **not verified against hardware by anyone in this repo**. They are here
because they are the failures that cost an afternoon, not because we can vouch for them.

None of the first three came up that afternoon: the arm was already assembled, calibrated and
on a working cable, which is the state this page assumes rather than the state a kit arrives
in. The two-camera and the Linux ones could not come up either, because the bench ran one
webcam on Windows, at 640x480 and with no `?backend=` key needed. What did come up is an index
moving, from the other direction than the bullet describes: a single webcam, `opencv://1` on
one session and `opencv://2` on a later one.

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
| `disconnect() disables torque by default` | `disable_torque_on_disconnect` defaults to True, so the arm goes limp at every clean exit, a `doctor` probe included, and not at all when the process is killed. quackd keeps that default, and turns it off for the one case where letting go would drop the arm: one that did not reach the pose it was recorded resting in |
| `disconnect() reads config.disable_torque_on_disconnect when it runs` | the flag is read off the config instance inside `disconnect()` rather than copied at construction, and the config is a plain dataclass, so setting it False on the instance immediately before the call is what leaves an arm holding. quackd uses that seam for an arm away from its rest pose and nothing else; the config is still built asking for True. Read against lerobot 0.6.1, the version the first real arm ran |
| `Max_Torque_Limit 500 on the gripper` | with Protection_Current 250 and Overload_Torque 25: the native authority |
| `the five body joints get no torque or current cap` | the caps sit inside a check for the gripper's name |
| `configure_motors() writes Return_Delay_Time 0 and Acceleration 254` | called inside torque_disabled(), so connecting drops torque briefly |
| `SOFollower.is_connected is the serial port plus the cameras` | |
| `SOFollower.bus is a FeetechMotorsBus` | the attribute registers are read through |
| `no deadman: nothing stops the arm when the client goes quiet` | the class has no thread, timer or timeout; a goal stands until the next write |
| `MotorsBus.disable_torque()` | NEVER called by quackd (limp) |
| `MotorsBus.enable_torque()` | |
| `MotorsBus.disconnect(disable_torque=True)` | the `disable_torque()` call is inside `if disable_torque`, so False closes the port and leaves every motor holding the goal it was last written: what an arm that missed its rest pose gets instead of falling |
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
serial port), and it has now run on one real arm, once.

### What one afternoon proved, and what it did not

On 2026-09-15 an SO-101 follower, calibrated as `arm-01` and reached as `--robot lerobot:real
--address COM5` with no registered name, ran on Windows 11 with
Python 3.12.12, lerobot 0.6.1 and quackd 0.9.0, piloted by OpenAI's `gpt-6-astra`. This is the
only one of quackd's seven bodies that has been on hardware at all.

What ran:

- `lerobot-lookout`, with a real pilot and once with `--provider fake`, which is how you tell
  an arm that will not answer from a model that will not decide.
- Free-form `--goal` runs. It waved by rolling the wrist about 27 degrees either way, waved
  again from an extended pose with `shoulder_lift` at -39 and `elbow_flex` between 24 and 30,
  and opened and closed the gripper: commanded 100, reported 98 open and 3 closed with the jaws
  nearly touching. One run mimed a duck quacking with the gripper.
- A USB webcam at `opencv://1`, and at `opencv://2` on a later session, 640x480, with no
  `?backend=` key needed on that machine.

What went wrong, which belongs in the same breath as the above:

- **The arm fell at the end of every run**, because `disconnect()` drops torque. That is the
  whole reason [the rest pose](#the-rest-pose) exists, and the rest pose is the one thing on
  this page written after hardware rather than before it.
- One dry run aborted with `the arm did not answer: TimeoutError` when a single heartbeat round
  trip failed. It never recurred, and nothing since has explained it.
- One dry run aborted because the pilot answered `uncertain` at `assess_task` and the human
  said no. That is the gate working, not a fault.
- The camera framed the gripper and cropped the raised arm, so the model verified its own waves
  from joint readings rather than from the picture. A view that shows the whole arm is the
  first thing the second bench should fix, and it is now two `--camera-url` flags rather than a
  choice between views.

What nobody measured, and what this page therefore still cannot tell you:

- whether the `holding` band is right, or whether an empty hand that binds reads the same as a
  grip
- what a joint actually reads after ten minutes of work, so whether the 60 °C refusal fires
  before anything is hot enough to matter
- whether a stall is caught on purpose. Nothing was deliberately obstructed
- whether 5 degrees an action felt right in the room, which is the one number on this page that
  only a person standing next to the arm can judge

`pick` and `load_policy()` are still untried on hardware: no policy was loaded that afternoon.

## How to help

If you have an SO-101 on a desk, work through
[lerobot-hardware-checklist.md](../lerobot-hardware-checklist.md) in order: nothing moves
until step 10. `lerobot-lookout` is the first task to point at it; it asks for `report_state`
rather than `observe`, because a `.duck` is checked against the static manifest, which
cannot know whether you brought a webcam. What most needs a real arm is that checklist's
*What to report*, and the four unmeasured things above are the top of it. One afternoon on one
arm is a sample of one: what differs on yours is the part worth writing down. Open an issue
with the transcript.
