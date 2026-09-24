# LeRobot (an SO-101 class arm)

A six-joint desktop arm with a parallel gripper, driven through
[LeRobot](https://github.com/huggingface/lerobot). No legs, no head, no voice, so its
manifest lists none of that: `move`, `go_to`, `search_scan`, `say` and `gaze` do not exist
on this robot. What it has is joints, a gripper, `place`, and, when a policy is available,
`pick` as one skill intent that the arm's own learned policy executes. The thesis holds:
the LLM picks the verb, LeRobot moves the arm, quackd enforces the contract. (With the optional `--decision-llm`, some of the verbs that are a choice rather than a number can be picked by a decision LLM instead; every angle is still the model's, and it is off unless you name one: [decision-llms.md](../decision-llms.md).)

Upstream pinned at
[`fbb811f`](https://github.com/huggingface/lerobot/tree/fbb811fca92504439792b97d216f0d00c2268382)
(`main`, 2026-09-01), first read 2026-09-02 and read again on 2026-09-13. Every name quackd
spells lives in
[`adapters/lerobot/src/quackd_lerobot/upstream_api.py`](../../adapters/lerobot/src/quackd_lerobot/upstream_api.py),
and why the adapter is shaped the way it is is
[ADR-0036](../adr/0036-what-the-arm-does-not-say.md).

> [!NOTE]
> **The `real` backend has run on an arm.** On 2026-09-15 an SO-101 follower, calibrated
> as `arm-01` and reached with no registered name, ran `lerobot-lookout` and then free-form `--goal` runs
> on Windows 11 with Python 3.12.12, lerobot 0.6.1 and quackd 0.9.0, piloted by OpenAI's
> `gpt-6-astra`. It also went limp and fell at the end of every one of those runs, which is what
> [the rest pose](#the-rest-pose) exists to fix. On 2026-09-23 the same arm ran again with a rest
> pose recorded, and the pose lay past the travel its calibration recorded, where the servo will
> not be driven: [The rest pose](#the-rest-pose) says what that did and what quackd does about
> it now. What ran, what went wrong and what that first afternoon did not measure is in
> [Status](#status); the account from the other end, an empty laptop to a waving arm, is
> [lerobot-first-run.md](../lerobot-first-run.md).

```bash
# offline, the default
uv run quackd run lerobot-lookout --robot lerobot:mock --llm fake

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
   uvx --from "quackd[lerobot]" quackd run lerobot-lookout --robot lerobot:mock --llm fake
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
| `lerobot:real` | ✅ | an SO-101 follower through LeRobot (extra `quackd[lerobot]`, Python 3.12 or newer, torch), and as many USB webcams as `--camera-url` names; every name VERIFIED at the pin, exercised against a fake arm and a fake camera, and run for one afternoon on one real arm, 2026-09-15, with lerobot 0.6.1, one webcam and no policy |

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
the calibration file, to a tenth of a degree rounded inward so that every angle it names is one
the arm accepts (`wrist_roll` included, where that travel is the whole turn upstream
records rather than anything swept, see Safety), `extras.calibration_file`, the path that came
from, `extras.rest_pose_clipped` when the recorded rest pose lies past that travel
([A pose past the travel](#a-pose-past-the-travel)), `limits.step_deg`,
how far one action may move a joint (`QUACKD_LEROBOT_MAX_STEP_DEG` sets it), and
`extras.camera` with `limits.camera_fov_deg` when there is a camera. With more than one camera
it also carries `extras.cameras`, their names in the order the urls were given, primary first.
That key is written only when there are two or more: a single camera's name is quackd's own
default rather than anything you chose, and a pilot told its one view is called `front` starts
naming it in sentences nobody needs.

| Verb | Kind | What it does here |
|---|---|---|
| `observe` (alias `get_frame`) | core | one frame plus detections, only when a camera is configured. With several cameras it is the primary's frame, because a detection is a bearing and a bearing belongs to one lens |
| `report_state` | core | joint positions in degrees, whether torque is on, each servo's temperature, whether something is held, and a clause for any joint that reads more than 2 degrees past its travel |
| `stop` | core | hold: the present position becomes the goal of every body joint inside its travel. Never limp (see Safety) |
| `move_joints(positions, duration_s)` | extension | goal angles for one or more of the six joints, walked there across `duration_s` seconds one goal a tick at 10 Hz, then re-sent until the measurement arrives; the step cap is the ceiling on speed, and a joint that stops short is a failure |
| `gripper(open)` | extension | open or close the gripper, and report where it stopped |
| `place` | extension | open the gripper where the arm is; needs `holding` |
| `pick(target, max_s)` | extension, **confirm** | one skill intent; the arm's learned policy runs its own observe/act loop at its own rate until something is held or the time is up |

Joints are named, not numbered, and a `move_joints` call may name any subset of them:

```json
{"positions": {"wrist_roll": 10, "elbow_flex": 45}, "duration_s": 2.0}
```

The five body joints are degrees, centred on zero, and the gripper is `0..100` whatever the
others are.

`duration_s` is how long the motion should take, from 0.2 to 12 seconds, and 5 when the pilot
names none. The arm is read once, and every tick sends a goal as far along the straight line
from where each joint is to where it was asked to be as the time is through: halfway at half the
time, every joint starting together and arriving together. Once the time is up the goal
itself goes out, tick after tick, until every joint is within 5 degrees of it (5 units on the
gripper). So "slowly" is something a pilot can ask for, and a longer time is a slower move.
Each tick's intent still carries the `duration_s` the pilot asked for, so the record shows the
ramp and the time it was meant to take.

The step cap stays the ceiling. One action moves a joint at most 5 degrees and ten go out a
second, so nothing moves faster than 50 degrees a second, whatever `duration_s` says. A time too
short for the distance is neither refused nor obeyed: the goal runs ahead of the arm, LeRobot
clips every send to one step from where the joint is, and the joint travels at the cap until it
is there, later than asked. `QUACKD_LEROBOT_MAX_STEP_DEG` lowers the ceiling.

Four exceptions, each on purpose:

- **A move whose every joint is already within 5 degrees of its goal is sent whole** and
  judged after one tick, as it always was, however long `duration_s` is. There is nothing to
  pace.
- **A goal outside the travel is sent whole**, so the range refusal answers before anything has
  moved. Ramped, the arm would travel to the edge of its travel and be refused there.
- **A joint that reads past its travel starts its ramp at the edge of it.** The servo clamps
  every goal to the travel its calibration wrote into it, so such a joint first rises to that
  edge at the servo's own speed, whatever quackd sends, and is paced from there. That first
  stretch is the one part of a move `duration_s` cannot slow down: support the arm or place it
  inside its travel if that matters ([A pose past the travel](#a-pose-past-the-travel)).
- **`gripper` is not ramped.** It sends open or shut, which each backend maps to 100 or 0, and
  closes at the cap. A gripper named in `move_joints` is a joint with a goal and ramps like one.

Arrival and stalls are judged only once the goal itself is going out. A slow ramp moves a joint
less per tick than the stall rule's threshold, so a stall counted during the ramp would fail
every slow move. The price is that a joint blocked partway is found when the ramp ends rather
than when it stopped, pushing meanwhile against a goal one step ahead of it, as it always did.
The verb then fails naming the joint, where it stopped and its goal, and holds the arm. It
gives the move the time asked for, or the time the cap needs if that is longer, plus 2.5
seconds to settle, and never more than 18: the executor's own timeout for `move_joints` is 20,
and the verb ends first so that the reason names the joint that fell short.

Its datasheet, which the pilot is shown and told to judge a task against before anything moves ([manifest-spec.md](../manifest-spec.md)):

| | |
|---|---|
| Height | 0.53 m (estimate: one vendor's listing; reaching straight up) |
| Actuated joints | 6 (official: the LeRobot SO-101 docs; five joints and a gripper) |
| Payload | 0.5 kg (estimate: one vendor's listing) |
| Reach | 0.4 m (estimate: the maker's URDF, TheRobotStudio/SO-ARM100 Simulation/SO101/so101_new_calib.urdf; link lengths from the shoulder to the gripper frame, summed with the arm straight and rounded down) |
| Not published | mass |

And what it cannot do whatever the task says, which is the half a refusal usually turns on, in the words the pilot is shown:

- go anywhere: it is bolted to a table and has no base
- lift or hold more than about half a kilogram: a pen, an empty cup or a wooden block weighs
  far less than that, and a full bottle or a tool may weigh more
- reach anything more than about 0.4 m from its shoulder: that is the arm held straight out,
  and any bent pose reaches less
- feel what it holds: nothing reports grip force, so holding is inferred from the gripper
  stopping short of shut, which an empty hand that binds also does
- know its own mass: vendor listings disagree by a factor of three

A figure nobody published is listed as not published, and the pilot is told to answer `uncertain` and name it, rather than guess, where a task turns on it. A `.duck` file can correct any of it for the build in front of you ([duck-spec.md](../duck-spec.md)).

**Where the reach comes from.** Nobody publishes a reach for the SO-101, and until 2026-09-23 the sheet said so, which told the pilot to decline whatever turned on reaching: every task an arm has. The maker's URDF gives every link, so the figure is quackd's arithmetic on the maker's file. From the `shoulder_lift` joint outwards the joint origins are 0.116 m to `elbow_flex`, 0.135 m to `wrist_flex`, 0.064 m to `wrist_roll` and 0.098 m to the gripper frame, 0.413 m in all, which is an upper bound because links only add up in full when they are in a line. A grid sweep of the elbow and both wrist joints through their URDF limits puts the farthest the gripper frame gets from the shoulder axis at about 0.41 m. The sheet says 0.4, as an estimate, and the adapter's source (`REACH` in `quackd_lerobot/__init__.py`) keeps the four vectors so anyone can check them. It is measured from the shoulder joint rather than the base, and to the gripper frame rather than the fingertips.

**What the pilot is told about a pen.** The payload line used to end "and nothing whose weight is not known", which is nearly every object a task names: nobody tells the pilot what a pen weighs. It keeps the half kilogram, itself an estimate from one vendor's listing, and gives the pilot objects to judge by instead.

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

Either way the pictures that did arrive keep their names, on the wire and in
`frames/NNNN-<name>.png`. Whether a picture is named is decided by how many cameras the arm
has and never by how many answered this step, which matters most in exactly the case above: a
lone unnamed picture sitting under a detections line measured off the lens that died is the
one thing the naming exists to prevent, and it is also the case a count of the frames that
arrived cannot tell apart from a one-camera arm.

A second camera that will not open is a refusal **before the arm is energised**, and it lets
go of the first on the way out. Half a set of eyes nobody asked for is worse than the
refusal, because the frames would still arrive and look right.

> [!WARNING]
> Two uncompressed 640x480 streams on one USB controller can exceed its bandwidth. Both
> cameras open, and then one or both deliver nothing. `?fourcc=MJPG` on each is the answer,
> and a different physical USB controller for the second camera is the other one. This is an
> owner report rather than something measured here: the 2026-09-15 bench ran one webcam.

**The cost is pictures.** Each request carries the images from the last two exchanges, so two
cameras is four pictures per request where one camera is two, and that is what you pay in tokens
on every step of every run. On Claude Opus 5.5 and Fable 5.1, whose old frames are trimmed every
eight exchanges rather than on every one, it is up to eighteen where one camera is nine. It is
worth it for a wrist view plus an overhead view. It is not worth it for two views of the same
thing.

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
again with `--llm fake` to separate the arm from the model.

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
quackd run lerobot-lookout --robot lerobot:real --address /dev/ttyACM0 --llm anthropic
```

`lerobot-lookout` ships with quackd and moves no joint: it reads the arm back and says what
it found. It is the first thing to point at a real arm. Writing your own is a file and a
`requires:` line, and `quackd validate <file> --robot lerobot:real` refuses it before a run
if this arm does not provide a verb it asks for.

**A one-line goal**, for when you want a single verb and there is no file for it. The model
still has to pass `assess_task`, and everything else still applies:

```bash
quackd run --goal "roll the wrist ten degrees and stop" --robot lerobot:real \
  --address /dev/ttyACM0 --llm anthropic --max-steps 3
```

Keep `--max-steps` small. `--llm fake` will not do here: the scripted pilot answers a
free-form goal with a fixed script that ignores it.

**From an MCP client**, which is you choosing each verb with the model doing the talking.
This is the only way to call `observe` on a real arm today, and the only way to run exactly
one verb and stop:

```json
{
  "mcpServers": {
    "arm": {
      "command": "uvx",
      "args": ["--from", "quackd[lerobot]", "quackd", "serve-mcp", "--robot", "lerobot:real",
               "--address", "COM5", "--camera-url", "opencv://1"]
    }
  }
}
```

Nine `robot_*` tools appear. `robot_list_verbs` first, then `robot_assess_task` with a
verdict, which `robot_run_verb` requires before anything that moves the body, then
`robot_run_verb(verb="move_joints", params={...})`. Both clients, the full tool list and a
two-minute script: [mcp.md](../mcp.md). The same path at walking pace, from an empty laptop to
a waving arm in fifteen steps, is
[Part 2 of the first run](../lerobot-first-run.md#part-2-from-claude-over-mcp).

A session parks the arm at both ends, the same as a run does, and **refuses to start** if it
cannot reach the recorded rest pose. Repeat `--camera-url` here too, and the session reads
every camera you name.

**A picture that comes with the task** is `--image PATH`, repeatable, and it is the flag that
makes "draw what is in the picture" a sentence this body can be given. This arm is the body a
task like that runs on: it is the one that holds a pen, and a drawing is a thing you show
somebody rather than describe. The picture is not a camera frame and is not treated like one:
it rides on the pilot's **first** turn labelled `task picture <name>:`, it is never trimmed out
of the history the way old frames are, and it stays in front of the model for the whole run, so
a task about a sketch is still about that sketch twenty turns later. Where this arm also has a
camera, both go out together, task pictures first and then the frames, each named, so the model
can tell the drawing it is copying from the desk it is copying onto:

```bash
quackd run --goal "draw what is in the picture" --robot arm-01 --image sketch.png --llm anthropic
```

Every request line says what actually went out, so a picture that never arrived is something
you read in the transcript rather than infer from a bad drawing (captured with `--llm fake
--vision`, the one pilot here that takes a picture and needs no key):

```
   llm>    step 0: 1 messages (1 with image, 1 task picture) to fake scripted:goal
```

Six formats are accepted, PNG, JPEG, WebP, GIF, BMP and TIFF, and every one of them is
re-encoded to PNG on the way in, brought down to a longest edge of 1568 pixels and then, where
the encoded picture is still over 1.5 MB, shrunk again. The copy kept at
`runs/<id>/images/00-sketch.png` is therefore byte for byte what the model was sent rather
than the file it was derived from, and a multi-frame GIF or TIFF goes out as its first frame.
Two `--image` flags whose files share a basename are both numbered by their place in the
list, because two directories with a `sketch.png` in each would otherwise arrive under one
label and a task naming one of them would be ambiguous in exactly the way a label exists to
prevent. Both refusals are worth knowing before you write a task around the flag. A
pilot that does not take images is refused rather than handed the words without the picture,
because a model told to draw what is in a picture it never received will improvise something
and the only sign of why would be a line in a transcript nobody reads twice:

```
✗ error: fake scripted:goal does not take images, so it cannot be given 1 picture
  quackd list-models marks the models that take no frames; --vision overrides it where
the vendor does take them, and a local model needs --vision
```

And `--image` is refused with `--flock` or `--robots`, because one picture handed to several
bodies is a task to write as several runs rather than one, and dropping the flag quietly on
the way into a flock would be a task about a picture that never arrived.

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

## Which of this arm's verbs are a choice

Only relevant with the optional `--decision-llm` ([decision-llms.md](../decision-llms.md)), and
off unless you name one.
The stepper decides what it may answer from each tool's own JSON schema, and on this arm the
split falls like this:

| | Tools | The calls it can author |
|---|---|---|
| **A choice** | `report_state`, `stop`, `place`, `gripper`, `observe` (when a camera is configured) | `report_state`, `stop`, `place`, `gripper(open=true)`, `gripper(open=false)`, `observe` |
| **A number** | `move_joints`, `pick` | none, ever |
| **A sentence** | `assess_task`, `declare_success`, `declare_failure`, `remember` | none, ever |

`gripper` is a choice because its only parameter is a boolean. `move_joints` is not, for two
reasons that hold independently. Its `positions` is a required object, which is enough on its
own. And the joint names are nowhere in the schema: they are enforced by a `field_validator`
against `JOINTS`, so there is nothing for a decision LLM to enumerate even in principle, and no
version of this could be talked into offering one.

`pick` is out on both counts, being a free string and confirm gated. The step cap, the range
refusal and the hot-servo precondition apply to a stepper-authored call exactly as they apply
to a model's, because both go through the same executor.

[`ducks/arm-grip-check.duck`](../../ducks/arm-grip-check.duck) is the task built out of the
first row alone, and it is the worked example on that page.

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
  does not clamp a degrees goal and the servo does, silently, to the travel calibration wrote
  into it (`up.POSITION_LIMITS_CLAMP_GOALS`), so a goal let through would be one the arm quietly
  stops short of. quackd computes each joint's travel from the calibration file and refuses
  instead. `wrist_roll` is the exception, and it is upstream's: its
  calibration deliberately does not sweep that joint, printing *move all joints except
  'wrist_roll'* and recording a full encoder turn for it instead
  (`up.WRIST_ROLL_IS_A_FULL_TURN`). Its travel therefore comes out as -180..180, and a refusal
  that cannot be narrower than the whole turn cannot catch anything. Treat `wrist_roll` as
  unguarded and give it small goals.
- **One action moves a joint at most one step.** `max_relative_target` is unset upstream;
  quackd sets it to 5 degrees, re-sent at 10 Hz, so 50 degrees a second at most. That is a
  ceiling and not a pace: `move_joints` walks its goal across the `duration_s` it is given, and
  only a time too short for the distance runs at the cap.
- **No deadman.** Nothing in LeRobot's `Robot` stops an arm when the client goes quiet: read
  from the class, not assumed. quackd's `stop` re-sends the present position as the goal and
  never calls `disable_torque()`, the same principle as never sending `robot.relax` to a
  Microduck. A joint that reads past its travel gets no goal from a stop at all, because the
  servo would clamp "stay here" to its limit and drive there
  ([A pose past the travel](#a-pose-past-the-travel)).
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
- **Torque is released only where the arm is known to be at its recorded rest pose, unless a
  person holding it asks.** `disconnect()` disables it by LeRobot's default, which quackd
  keeps, because an arm at rest should be limp: that is what "at rest" means. So before the
  disconnect quackd reads the joints one last time, and where they are not the pose you
  recorded, or as near it as the calibrated travel lets the servo go, it turns that default off
  and leaves the arm holding itself up, with one line saying so and naming the ways out:

  ```
  the arm is not at its rest pose (...), so torque was left on and it will not fall as it
  stands: hold it first, because connecting takes torque off every motor for a moment, then
  run quackd robot release NAME, or quackd doctor --robot NAME to park it, or cut its power
  ```

  The hold comes before every way out, because both commands begin by connecting, and the
  connect is the next bullet. With no rest pose recorded there is nothing to check against,
  nothing changes, and the arm goes limp at the end of every clean session exactly as it did
  in 0.9. See [The rest pose](#the-rest-pose), and for the person holding an arm left up this
  way, [Releasing it where it stands](#releasing-it-where-it-stands).
- **Connecting still drops torque briefly, and that has not changed.** `configure()` runs
  inside `torque_disabled()`, so the arm is limp for the moment between the port opening and
  the configuration landing, whatever any rest pose says. Support the arm when a session
  starts, including at the start of a `doctor` probe, which says so before it connects to a
  body that is handed to people:

  ```
  ⚠ connecting takes torque off every motor for a moment, because LeRobot configures them with it
  off: support the arm until doctor has finished with it
  ```
- **A packet lost while connecting is tried again, and said.** Those torque writes are a
  `Torque_Enable` and a `Lock` per motor, off on every motor and back on one at a time, each
  tried once (`up.CONFIGURE_TORQUE_WRITES_ONCE`), so one status packet the bus drops fails the
  whole connect and leaves the port open behind it. On 2026-09-23 that ended three of 26 runs
  on an SO-101 before they began, on a different motor each time, and the next connect went
  through every time. quackd now closes the port through the bus without writing to any motor
  (`up.BUS_DISCONNECT` with `disable_torque` False: the follower's own `disconnect()` would
  first switch torque off on every motor again, on the bus that has just lost a packet), waits
  half a second and connects again, up to three attempts in all. Each retry is a WARNING line
  while it happens, a note in the run's transcript and an advice line in `doctor`, and it names
  the joint through the bus's own motor table (`up.BUS_MOTORS`) rather than by an assumed order:

  ```
  connect attempt 1 of 3 failed on <joint> (id <N>): Failed to write 'Lock' on id_=<N> with
  '1' after 1 tries. [TxRxResult] There is no status packet! The port was closed without a
  write to any motor, and connect runs again
  ```

  A camera opened before the arm stays open across the attempts. A connect that blew its 30
  second deadline is never tried again, because its thread is still on the bus and a second
  talker there is how packets get lost. Every attempt runs `configure()` again, so the limp
  moment above happens once per attempt. When the last attempt fails too, the port is closed
  the same way and the refusal says the arm may be left half energised
  ([When it will not work](#when-it-will-not-work)).
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
> **The first arm to run a rest pose could not reach it.** On 2026-09-23 the SO-101 of the 15th
> ran with a pose recorded, and that pose had `shoulder_lift` folded about 20 degrees past the
> floor of the travel its calibration recorded, because the calibration never saw the shoulder
> folded all the way back. LeRobot's calibration writes that travel into each servo as its
> position limits, and the servo clamps every goal to them. So the rest move drove the shoulder
> to the limit, stalled there and called the arm lost: runs aborted before their first model
> call, every run that got to its end kept torque on and finished at the power switch, and the
> `stop` at the end of a run hauled a folded shoulder up out of its fold. What quackd does with
> a pose past the travel now is [A pose past the travel](#a-pose-past-the-travel), below, and
> the whole account is [ADR-0045](../adr/0045-a-rest-pose-the-calibration-cannot-reach.md).
> That answer has run against a fake arm that clamps the way the servo does and against
> `lerobot:mock`, and not yet on the arm, so read
> [section 07 of the first run](../lerobot-first-run.md#07-record-the-rest-pose) with a hand
> near the power switch and say what happened.

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

**Calibrate with the arm folded, before you record.** A servo will not be driven past the
travel its calibration recorded, so the fold you record should be inside that travel. When
`lerobot-calibrate` asks you to move every joint through its whole range, take each one all
the way into the fold you mean to rest the arm in. Where the fold still lies past the travel,
`rest-pose` says so as a warning before it asks, in the same sentence a run will use, and
records the pose anyway: it is where the arm rests, and a run parks as near it as the servo
goes ([A pose past the travel](#a-pose-past-the-travel)). A new calibration also moves the zero
of any joint whose travel it records differently, because a joint's zero in degrees is the
middle of its recorded travel. So a pose recorded before it names a different shape after it:
record the pose again, and read any task or remembered note that names an angle as meaning a
different pose too.

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
| a `--by-hand` run | the rest move still happens first, and then the arm is released **at** that pose for somebody to place. The pose is what makes the release safe rather than something the flag skips, and the run ends back at it ([Placing it by hand](#placing-it-by-hand)) |
| a dry run | nothing. `--dry-run` never moves the arm, and that includes the rest move |
| an MCP session | the same at both ends, and the session refuses to start if it cannot get there |
| a pose past the travel | the arm is driven as near the pose as its servos go, which is the edge of the travel, and that counts as getting there. Torque is released at the edge, and the run says once which joint is free to settle the rest of the way ([A pose past the travel](#a-pose-past-the-travel)) |

Both ends are narrated, so the transcript says what happened rather than leaving you to infer
it from a joint reading:

```
·  note    moving to the rest pose
·  note    already at the rest pose
```

### A pose past the travel

A servo on this arm will not be driven outside the travel its calibration recorded. LeRobot's
calibration writes each joint's travel into its servo as two position limits, and the servo
clamps every goal it is written to them (`up.POSITION_LIMITS_CLAMP_GOALS`). A reading is not
clamped: with torque off a joint goes wherever a hand or its own weight puts it, past either
end. So a pose recorded off a hand-folded arm can lie past the travel, and a goal of that pose
is one the arm drives to the limit, stops at, and never reaches. That is the bench of
2026-09-23 in the box above.

quackd now plans for it, on any joint, at either end of its travel, from the numbers this
arm's own calibration gives it at connect:

- **The rest move drives to the reachable pose.** Each body joint of the recorded pose is
  clipped into its travel. The pose in `robots.json` stays exactly as you recorded it, because
  the clip is a fact about this calibration and not about your fold.
- **A clipped joint is at rest anywhere from 5 degrees short of the edge of its travel out
  past it**, on the side its fold lies. Nothing quackd sends can drive a joint past its limit,
  so a joint that reads well past it was put there with torque off: it is folded, not lost.
  Every other joint keeps the ordinary rule of within 5 degrees of its recorded angle.
- **So the arm parks at the edge, which counts as arriving, and torque is released there.** The
  joint is then free to settle the rest of the way toward its fold. An arm already folded past
  the edge is `already` at rest, and the rest move sends that joint nothing, because the only
  goal it could send is the limit, and that would haul the joint up out of its fold.
- **A stop writes no goal for a joint that reads past its travel**, and neither does the
  take-hold of a [hand-placed start](#placing-it-by-hand). "Stay where you are" written to that
  joint arrives as "go to the limit", and the servo does it at full speed: on the bench, the
  stop at the end of a run hauled a folded shoulder up out of its fold this way. The joint keeps
  whatever goal its servo already holds instead, and the stop's summary names each joint it
  wrote no goal for and says why, so the pilot and the record can tell a stop that held all
  five from one that held some. If every body joint reads past its travel, nothing is sent at
  all. What the skip does is avoid *starting* a rise, and that is all: for a joint past its
  travel, any goal quackd has written is the limit to the servo, so a joint a move had already
  begun lifting out of its fold keeps rising to that limit whatever a stop writes or leaves
  out. Only the power switch stops that stretch ([safety.md](../safety.md)).
- **A move out of the fold starts at the edge.** `move_joints` on a joint that reads past its
  travel paces its ramp from the edge of the travel, not from the reading: any goal between the
  two is, to the servo, the edge, so the joint rises to it at the servo's own speed first,
  whatever quackd sends. Only the rest of the move is paced by `duration_s`
  ([The manifest](#the-manifest)).
- **A joint that stops short *inside* its travel is still a miss.** A hand, the desk or a
  tripped servo in the way keeps torque on and prints the line in
  [The torque rule](#the-torque-rule), exactly as before.

The run says what happened, once, right after its first `at the rest pose`, in this arm's own
numbers. Captured on `lerobot:mock`, whose `shoulder_lift` travels -100 to 100, registered as
`arm-01` with a rest pose that has `shoulder_lift` at -118:

```
·  note    moving to the rest pose
·  note    at the rest pose
·  note    shoulder_lift is recorded at -118 in the rest pose and this calibration lets its servo be driven to -100 and no further, so it parks there and is let go of there, free to settle the rest of the way on its own. Calibrate again with the arm folded (lerobot-calibrate) and record the pose again (quackd robot rest-pose arm-01) to make the fold reachable
```

The end of that run's transcript reads `already at the rest pose`, and says nothing more,
because the note is about the pose rather than about either move. The same sentence is what
`doctor` prints as advice under its table, with the `rest pose` row still green
([doctor](#doctor-and-robot-list---probe)), what an MCP session logs when it parks at connect,
and what `quackd robot rest-pose` warns before it asks. It names only a joint clipped by more
than 5 degrees, since a smaller clip is inside the tolerance any reached pose may miss by. The
run's first record names every clipped joint, whatever the amount, in the manifest:

```
"rest_pose_clipped": {"shoulder_lift": {"recorded": -118.0, "reachable": -100.0}}
```

**The pilot is told why a joint can read past its travel.** The travel line of its system prompt
now ends *A joint can read past its travel when it was folded or placed there with torque off,
which is where a rest pose usually is; goals are still limited to the travel.* And
`report_state` adds a clause for each joint that reads more than 2 degrees past its travel, in
the form `<joint> reads <angle>, past the <limit> its servo can be driven to; goals are still
limited to its travel`. On the bench, a pilot handed a shoulder reading past the end of its travel line, with
nothing to explain it, refused to move the arm at all.

**The fix is a calibration that saw the fold.** Calibrate again with every joint taken all the
way into the fold during the sweep, then record the pose again, for the reasons in
[Recording it](#recording-it). The note goes away once no joint of the pose lies more than 5
degrees past its travel.

> [!NOTE]
> Parking at the edge and letting go there has run against a fake arm that clamps goals the way
> the servo does and against `lerobot:mock`, and not yet on the arm. Two things only a bench
> can say: whether a joint let go at the edge settles onto its fold, and gently, and whether one
> folded past its *ceiling* settles at all, since its weight need not pull it toward the fold.
> The note says the joint is free to settle, not that it will. And a joint the take-hold writes
> no goal for is left to whatever its servo does when torque comes on, which is
> `up.TORQUE_ENABLE_HOLDS_PRESENT` below: the read-back still catches it, and when it is that
> joint that moved, the refusal says it was placed past its travel.

### Placing it by hand

A rest pose is a fold, and a fold is the wrong place to begin some tasks from. A drawing run
starts with a pencil in the gripper and its tip near the paper, and nothing the arm can be
driven to from a folded start puts it there: the pencil has to be handed to it. `quackd run
--by-hand` gives you the arm for exactly that moment and takes it back when you are done:

```bash
quackd run --goal "draw the circle in the picture" --robot arm-01 --by-hand \
  --image circle.png --llm anthropic --max-steps 12
```

The order below is the whole of the feature, and none of it is a step you can skip:

1. **The arm goes to its recorded rest pose first**, the same move that starts every other run.
   A run that cannot get there aborts here, before any torque is touched, because an arm that
   did not reach the pose is precisely an arm that must not be released at it.
2. **Torque comes off, at that pose and nowhere else.** `let_go()` re-reads the joints and
   refuses anywhere but the recorded pose, judged exactly as the close judges it, so a joint
   recorded past its travel counts at the edge of it or beyond
   ([A pose past the travel](#a-pose-past-the-travel)): an arm held up by torque alone falls the
   moment torque goes, and the person who asked for this still has their hands nowhere near it. It
   also refuses a pose that names no joint this arm drives, and an arm that still reports
   torque on after the call, which is a release that did not take rather than one to walk
   away from. LeRobot tries each torque write once unless told otherwise, and one lost packet
   there would release the motors before it and not the ones after, so the release asks for
   the five tries LeRobot's own `disconnect()` gives the same writes, and so does the
   take-hold in step 4.
3. **You are told the arm is yours, and quackd waits for Enter.** There is no timeout on this
   wait. The arm is limp at a pose it holds by its own shape, so nothing is being spent by
   waiting, and somebody who has gone to find a pencil should come back to a run that is still
   there.
4. **quackd takes hold of whatever you left.** The present position is written as the goal
   *before* torque comes on, written again after, and then the arm is read back. A joint that
   moved more than the same **5 degrees** every other goal on this page is judged by is a
   refusal and the run ends, because a run that started from a pose nobody chose is a run whose
   first observation is a lie. A joint you placed past its calibrated travel is left out of both
   writes, because the servo would clamp its goal to the limit and drive it there under your
   hand. That joint is left with whatever goal its servo already had, and if it is the one that
   moved, the refusal says it was placed past its travel, gives the edge in this arm's numbers,
   and asks for it to be placed inside.
5. **The pilot runs from those angles.** The step cap, the range refusal, the heat gate and the
   budgets are all the ones any other run gets. The minutes clock restarts the moment the arm
   is holding your pose, so the time you spent looking for a pencil is not taken out of the
   model's: a person's time and a pilot's budget are not the same clock.
6. **At the end you are asked before the gripper opens.** The run's own `stop` is holding the
   arm where it finished; the question comes next and the opening after it, because an arm
   folding to its rest pose with a pencil still in the jaws drives that pencil into the bench,
   and the person who put it there is the one who should take it out. That wait is bounded at
   **120 seconds**, since a run has to end even when the room is empty.
7. **Then the fold.** Enter sends `gripper(open=true)` through the logged transport, so it is
   in the record like every other intent, and then the rest move folds the arm and torque drops
   at the fold exactly as it would have without the flag.

What the person at the arm is told, in order, captured from a `--by-hand` run on `lerobot:mock`
in which the arm was placed at `shoulder_lift` -20, `elbow_flex` 40, `wrist_flex` 15 with the
gripper squeezed to 35:

```
the arm is yours: torque is off at its rest pose, so lift it, put whatever it needs in
the gripper, close the gripper on that, hold it where you want the run to start, and
press Enter
holding the pose you set, you can let go. It is at elbow_flex 40, gripper 35,
shoulder_lift -20, shoulder_pan 0, wrist_flex 15, wrist_roll 0
the run is over and the arm is holding where it ended. Take hold of whatever is in the
gripper and press Enter, and the gripper opens before the arm folds up. Leave it and
the arm folds up with the gripper shut
```

The middle line reads the arm back rather than repeating what it was told to hold, which is how
you find out that the wrist sagged two degrees as it took the weight. The same run in the
transcript, where every stage is a `hand` event:

```
·  note    moving to the rest pose
·  note    already at the rest pose
·  hand    released: torque is off at the rest pose
·  hand    held: holding the pose you set (elbow_flex 40, gripper 35, shoulder_lift -20, shoulder_pan 0, wrist_flex 15, wrist_roll 0)
▶  verb    report_state()
✓  result  report_state ok: shoulder_pan 0, shoulder_lift -20, elbow_flex 40, wrist_flex 15, wrist_roll 0, gripper 35; torque on; hottest shoulder_pan 30°C; holding nothing (0.0 s, 0 intents)
→  send    stop
·  hand    unloaded: opening the gripper
→  send    gripper(open=true)
·  note    moving to the rest pose
·  note    at the rest pose
```

**The pilot is told where it is starting from**, in a `## Where this run starts` section of the
system prompt that only a hand-placed run has. A model that assumed the fold would improvise
from a shape the arm is not in, so it is told to read `report_state` and work from the angles
rather than from any remembered geometry. That section also says what `holding nothing` above
means, in whichever of the two ways applies: a task whose allowlist has the gripper verb is
told to close on the object before leaning on it, and a task like `lerobot-lookout`, which
cannot work the gripper at all, is told that whatever is between the jaws is held at the
squeeze the person left and cannot be tightened.

**Ctrl-C in the first wait** ends the run with the arm limp in your hands, and the teardown it
goes into is built for exactly that arm. Every teardown starts with a `stop`, and a `stop` on
an arm somebody is holding takes hold of it first: the arm is re-energised where your hand has
it, and only then does the rest move fold it. A `stop` that sent a goal to a limp servo would
stop nothing, and the fold after it would be a fold of an arm that is not listening. A wait
that ends any other way lands in the same teardown and says so rather than blaming a key
nobody touched. There is no clock on the first wait, so that ending means the keyboard itself
went away: the terminal was closed, or the input it was reading finished. quackd notices,
because a wait for a keystroke that nothing can deliver has to end rather than hold an arm
limp for ever (captured on `lerobot:mock`, where nobody had moved the arm out of the fold
either, so the rest move had nothing to do):

```
·  hand    released: torque is off at the rest pose
→  send    stop
·  note    moving to the rest pose
·  note    already at the rest pose
┌─ ✗ ABORTED ─────────────────────────────────────────────────────────────────────────┐
│ nobody placed the arm: it was released at its rest pose for somebody to put it      │
│ somewhere, and nothing was pressed                                                  │
│ steps 0 · llm calls 0 · tokens 0+0                                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**Ctrl-C in the second wait means "skip this and finish"**, not "abandon the arm". The gripper
stays shut, the stage is recorded as skipped, and the rest of the teardown runs anyway, which
is the difference between a run that ends with the arm folded and a record written and one that
leaves an energised arm and no transcript. A third press lands somewhere without that guard and
quits at once:

```
·  hand    skipped: interrupted while waiting
·  note    the gripper was left as it is, and the arm still folds up
```

Walking away and pressing nothing at all is the same ending by a different route, once the
120 seconds are up:

```
·  hand    skipped: nobody answered
·  note    nobody unloaded the gripper, so it stays shut and the arm folds up
```

> [!WARNING]
> There is one window in which you are holding an arm that nothing is holding up, and it runs
> from the release to the moment quackd takes hold again. A run that ends inside it goes into
> a teardown that tries to pick the arm back up first, and where that does not work, the close
> says so in its own words rather than printing the line about torque being left on, which
> would tell somebody with a limp arm in their hand that it is holding itself up.
>
> ```
> the arm is limp and in your hands (...): put it down before you let go of it, because
> nothing is holding it up
> ```
>
> Put it back in the fold before you let go, then run again.

**The five refusals**, all five of them before anything is released, before anything
connects and before a run directory exists:

| What you see | Why |
|---|---|
| `--by-hand is one person placing one arm, and this run has several robots` | one pair of hands and one terminal. Dropping `--flock` and `--robots` is the fix, and the flag is refused rather than quietly applied to the first body |
| `--by-hand and --dry-run ask for opposite things: one takes torque off the arm, the other moves nothing` | a dry run moves nothing at either end, and taking torque off an arm is the one thing here that is not a command to the robot but a change to it. Rehearse with `--dry-run`, then run it again with `--by-hand` |
| `--by-hand waits for you to press Enter, and there is no terminal to ask on` | the wait reads a real keystroke. With nobody to ask, the release would happen and nothing would ever pick the arm back up |
| `microduck:mock is not a body a person places by hand: only the LeRobot arm is` | a body declares `supports_hand_off`, and six of the seven do not. `quackd list-adapters` |
| `--by-hand releases the arm at its recorded rest pose, and this arm has none recorded` | the release refuses anywhere but the recorded pose, so an arm without one could never be handed over at all. Said here rather than after it has connected: `quackd robot rest-pose NAME` |

> [!NOTE]
> The hand-off is exercised against `lerobot:mock` and in the test suite, and not yet on a
> real arm. The step that most wants one is the take-hold: whether a servo re-energised under
> the weight of an outstretched arm actually stays within 5 degrees of where a hand left it is
> `up.TORQUE_ENABLE_HOLDS_PRESENT` in the table below, and it stays an assumption until
> somebody stands there and watches it happen. Say what it did.

### The torque rule

Torque is released **only** where the arm is known to be at the pose you recorded, or as near
it as its calibration lets the servos go. A joint is at its recorded angle when it reads within
**5 degrees** of it, the same tolerance a `move_joints` goal is judged by, and every joint of
the pose has to be reported and within it. A joint recorded past its travel is at rest within
the same 5 degrees of the edge of that travel, or anywhere beyond the edge on the side its fold
lies, for the reasons in [A pose past the travel](#a-pose-past-the-travel). Where that does not
hold, quackd turns LeRobot's `disable_torque_on_disconnect` off on the
config instance before the call, closes the port with every motor still holding its goal, and
prints one line:

```
the arm is not at its rest pose (...), so torque was left on and it will not fall as it
stands: hold it first, because connecting takes torque off every motor for a moment, then
run quackd robot release NAME, or quackd doctor --robot NAME to park it, or cut its power
```

The parenthesis names the joints and how far short they are, and, where the rest move itself
failed, why it failed. It never names a joint that is folded past its travel, because that
joint is at rest. An arm parked at the edge of its travel and let go there prints nothing at
the close: the sentence about its fold was said once, by the rest move, and a close line is
read everywhere as torque left on.

The hold comes first because both commands connect, and connecting takes every motor's torque
off for a moment, so "it will not fall" is true of the arm as it stands and of nothing that
connects to it. Two closes say something else, because this line would be wrong in them. An arm
that did not answer the close's own read may be one whose supply you have just cut, limp in your
hands, as easily as one whose cable came out in front of live servos, so its line says quackd
cannot tell whether the arm is holding itself up, and to hold it and cut its power. And an arm
whose release you have just asked for and been refused is not sent back to that release
([Releasing it where it stands](#releasing-it-where-it-stands)).

`NAME` is the name the arm was registered under wherever quackd built it from the registry,
which is a run, an MCP session, a flock, `doctor --robot NAME` and the `robot` commands. A bare
spec such as `doctor --robot lerobot:real` builds the arm with no name, and leaves `NAME` as it
is rather than offering the calibration id as a name nobody may have registered. Captured from a
`--dry-run` on `lerobot:mock` registered as `arm-01` with its pose's `shoulder_pan` at 30, which
a dry run never drives it to:

```
·  note    the arm is not at its rest pose (shoulder_pan is at 0 with a goal of 30), so torque was left on and it will not fall as it stands: hold it first, because connecting takes torque off every motor for a moment, then run quackd robot release arm-01, or quackd doctor --robot arm-01 to park it, or cut its power
```

> [!CAUTION]
> This is a behaviour change. A probe or a dry run on an arm away from its recorded rest pose
> now leaves torque **on** where it used to drop it. The arm is holding itself up and the
> servos are drawing current until something stops them: hold it first, and then run
> `quackd robot release NAME`, or run `quackd doctor --robot NAME` and let the arm park itself,
> or cut its power. Both commands connect, and connecting drops torque for a moment. An arm with
> no rest pose recorded behaves as it always did.

**The exceptions are a person asking for it, out loud.** Everything above is about quackd's own
initiative, and on its own initiative quackd still de-energises nothing: no verb disables
torque, no model can reach it, and `stop` is a hold rather than a release. `let_go()` is the
single call in the project that takes torque off a robot, and it has two doors, both opened by
a person at a terminal and neither by anything else.

The first is [`--by-hand`](#placing-it-by-hand), and it is guarded by the rule above read from
the other side. The close keeps torque on where the arm is not at its recorded rest pose; the
release refuses where the arm is not at it. Both are the same question, *is this arm somewhere
it can be let go of*, asked of the same joint reading with the same 5 degrees of slack, and the
answer that leaves an arm holding itself up is also the answer that will not hand it to you,
because the person who asked for it has their hands nowhere near it yet.

The second is for the arm the close has just left holding itself up, and it releases that arm
wherever it stands, because the person asking is holding it:
[Releasing it where it stands](#releasing-it-where-it-stands), below, which is
`quackd robot release` and the offer a run makes at a terminal when its rest move missed.

### Releasing it where it stands

On 2026-09-23 every run that got to its end kept torque on, because its rest pose could not be
reached, and every one of them finished at the power switch: `let_go()` refused anywhere but the
rest pose, and nothing else in quackd would take torque off. An arm holding itself up against a
pose it could not reach is the right thing to leave in an empty room, and a dead end for a
person standing next to it. So there is a second door, for exactly that person:

```
quackd robot release NAME [--yes] [--address ADDR] [--registry-dir DIR]
```

It says two things before it touches anything, because both happen to an arm you should already
be holding. Connecting takes torque off every motor for a moment, since LeRobot's `configure()`
runs inside `torque_disabled()`, so a warning printed after the connect would come after the arm
had already been limp once. And the release lets the arm fall from wherever it is. Then it asks,
and only then connects, with no camera and with the registered rest pose, so a release that does
not happen closes under the rule above. It prints the joints, sends the release with no `stop`
before it (a stop picks an arm in somebody's hands back up), reads `Torque_Enable` back off every
motor, and says what it read. Captured on `lerobot:mock` registered as `arm-01`, with `y` typed
at the question:

```
⚠ connecting takes torque off every motor for a moment, because LeRobot configures them with it
off, and the release then lets the arm fall from wherever it is: hold it now, and keep hold of it
until it is down
release torque on arm-01? [y/N]: y
arm-01 (lerobot:mock) is at
shoulder_pan   0.0
shoulder_lift  -90.0
elbow_flex     90.0
wrist_flex     0.0
wrist_roll     0.0
gripper        100.0
✓ torque reads off on every joint of arm-01
⚠ the arm is limp and in your hands (torque was taken off where it stood, because you asked for
it): put it down before you let go of it, because nothing is holding it up
```

The last line is the close's own, and it is the right one to end on: nothing is holding the arm
up, and you are. Unless a motor kept its torque, and then the close's line is not that one (the
table below). `--yes` skips the question and nothing else, so hold the arm before you run it.
With no terminal and no `--yes` it refuses before anything connects (`no terminal to ask on: pass
--yes to release it`), and answering `n` connects nothing.

`torque reads off on every joint` is printed only when the register was read and every motor
said 0, and the command exits 0 only then. Everything else exits 1 and says which it was:

| What you see | What it means |
|---|---|
| `torque still reads on for <joints>: cut the power` | those motors kept their torque through the release, or every motor did and nothing was released. The ones not named are limp, and the close's line after it says which case it was: below |
| `torque was taken off and could not be read back: ...` | the release went out and the read that would confirm it failed, or the release call itself did not come back part way through its motors. quackd reads that silence as a release, so treat the arm as limp, and cut the power if it still holds itself up |
| `nothing was released: ...` | the arm did not answer before the release, so nothing was sent. Whether it is holding itself up is not something quackd could read, so hold it |
| `lerobot:real at COM5: ...` and `keep hold of the arm` | the connect failed, and a connect that fails part way can leave some motors limp ([When it will not work](#when-it-will-not-work)) |
| `... is not a body quackd takes torque off: only the LeRobot arm is` | the name is a body that does not declare `supports_hand_off` |

After `torque still reads on`, the close's last line says what quackd then did, and never sends
you back to the command that has just failed. Some motors held and the rest let go, so the arm
is in your hands with those joints still energised, and the close keeps whatever torque there
is. Captured on `lerobot:mock` registered as `arm-01`, its release made to keep `elbow_flex` on,
since an in-memory release otherwise always takes:

```
✗ error: torque still reads on for elbow_flex: cut the power
  torque is off where the arm stands except on elbow_flex, which still read on
⚠ the arm is in your hands (torque was taken off where it stood, because you asked for it), but
elbow_flex still reads torque on and holds: keep hold of the arm, put it down, and cut its power to
let go of it
```

Every motor held, away from the rest pose, so nothing was released and torque is kept. The same
arm with its pose's `shoulder_pan` at 30, every motor kept on:

```
✗ error: torque still reads on for shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll,
gripper: cut the power
  the arm still reports torque on, so it was not released
⚠ the arm is not at its rest pose (shoulder_pan is at 0 with a goal of 30) and the release did not
take, so torque was left on and it will not fall as it stands: hold it and cut its power
```

Every motor held at the rest pose, or on an arm with no pose recorded: the close lets go of it
there, as every such close does, which sends the same release a second time with nothing to read
it back. The line then begins `the release did not take, and the close then took torque off at
the rest pose`, and ends on cutting the power if the arm still holds itself up.

**The offer at the end of a run.** A run with you at its terminal makes the same offer itself,
between its last rest move and the close, when that rest move missed: never on a dry run, never
over MCP and never in a flock. And never over an arm that stopped answering: a rest move that
failed because the arm went quiet is what cutting the servo supply looks like, nothing read says
that arm is holding itself up, and the release would refuse at its first read anyway. Enter
releases the arm through the same door. Sixty seconds with no Enter
(`AgentLoop.RELEASE_OFFER_S`), no keyboard to read, or a Ctrl-C leaves it exactly as a run
without the offer would, holding itself up with the torque line said. Captured on
`lerobot:mock` registered as `arm-01` with `shoulder_pan` at 30 in its pose and its rest move
scripted to miss, since a mock cannot miss on its own (`shoulder_pan stopped 30 deg short` is the
capture script's wording), with Enter pressed:

```
·  note    moving to the rest pose
·  note    the arm did not reach its rest pose: shoulder_pan stopped 30 deg short
the arm did not reach its rest pose (shoulder_pan stopped 30 deg short), so it
is holding itself up. Hold it and press Enter to release torque now. Leave it,
and after 60 s it stays that way
·  asked   release: the arm did not reach its rest pose (shoulder_pan stopped 30 deg short), so it is holding itself up. Hold it and press Enter to release torque now. Leave it, and after 60 s it stays that way -> yes
·  release released: torque is off where the arm stands
torque is off where the arm stands: the arm is in your hands, so put it down
before you let go of it
·  note    the arm is limp and in your hands (torque was taken off where it stood, because you asked for it): put it down before you let go of it, because nothing is holding it up
```

And left alone:

```
·  release kept: nobody pressed Enter
·  note    nobody pressed Enter, so torque stays on and the arm holds itself up
·  note    the arm is not at its rest pose (shoulder_pan is at 0 with a goal of 30), so torque was left on and it will not fall as it stands: hold it first, because connecting takes torque off every motor for a moment, then run quackd robot release arm-01, or quackd doctor --robot arm-01 to park it, or cut its power
```

The record says which of those ended the wait. A Ctrl-C at the offer, the first of the run or a
later one, reads `release kept: interrupted while waiting` rather than `nobody pressed Enter`,
and a terminal with no keys to read reads `release kept: no key could be read`. A Ctrl-C that
lands on the release itself, after Enter, is caught too: the release may have reached some motors
and not others, so the arm is taken to be limp in your hands, you are told to hold it as though
nothing holds it, the record says `release interrupted: interrupted during the release`, and
the close, the summary and the rest of the teardown still happen. When the release is refused, what you are told follows what was
read: every motor reading torque on says the arm is still holding itself up and to cut its
power, and a release refused before anything was read says quackd cannot tell whether torque is
on, to keep holding the arm, and to cut its power.

The offer and what came of it are said to you directly whether or not the log is on, and the
record keeps them as a `release` event and a `prompt` row ([architecture.md](../architecture.md)).

**Why a command and a prompt, and not a verb.** Every guard on this arm is there because a model
is three seconds away from it and nobody's hands are on it. Releasing torque away from the rest
pose is the one thing that drops the arm, so it goes through the only two doors a model cannot
reach: a command a person types, and a question put to a person at the run's own terminal. It is
not a verb and is in no `allow` list, it is not an MCP tool, and `let_go` is still not on the
`RobotAdapter` protocol ([ADR-0039](../adr/0039-an-arm-placed-by-hand.md),
[ADR-0045](../adr/0045-a-rest-pose-the-calibration-cannot-reach.md)). Both say "hold it" before
anything happens, which is the whole difference from `--by-hand`, where the release comes first
and the person's hands second.

> [!WARNING]
> Neither has run on an arm. Both are exercised against `lerobot:mock` and a fake arm in the test
> suite. What an SO-101 does in the moment the release reaches it away from its fold, how fast a
> shoulder held out at an angle drops, and whether one hand is enough to catch it, are for the
> bench to say.

### What is driven, and what is not

Only the **five body joints** are ever driven. The gripper is recorded, printed and stored, and
never commanded, for the same reason `stop` omits it: LeRobot writes only the keys it is given,
so leaving the gripper out keeps whatever squeeze is already commanded, and a rest move that
re-sent the gripper would open a hand that is holding something.

The rest move goes out **clipped** into the travel, like every other goal quackd sends. Until
2026-09-23 it was the one move sent unclipped, on the belief that a servo sent the angle of a
fold outside its recorded travel would drive the arm back into that fold. The bench showed it
does not: it clamps the goal to its limit and stops there
([A pose past the travel](#a-pose-past-the-travel)). The out-of-range refusal that guards
`move_joints` is still not in the rest move's way, and would have nothing to refuse: the goal is
the recorded pose already clipped into the travel from the same calibration. Record a pose you
are willing to have the arm driven toward from wherever a run ends.

The move itself is not paced the way `move_joints` is: it re-sends the goal at 10 Hz and runs
at the 5 degree step cap, on a budget
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

A pose past the travel is parked at the edge of it, and the row says `returned to it` or `at it
already` as it would for any pose, because the arm reached the pose it can be driven to and was
let go of there. What the fold means comes out as advice under the table, and the verdict stays
green. Captured on `lerobot:mock` registered with `shoulder_lift` at -118, against the mock's
travel of -100 to 100 for that joint (`quackd doctor --robot arm-01 --address mock://arm`):

```
│ rest pose         │ returned to it                                                              │
└───────────────────┴─────────────────────────────────────────────────────────────────────────────┘
shoulder_lift is recorded at -118 in the rest pose and this calibration lets its servo be driven to -100 and no further, so it parks there and is let go of there, free to settle the rest of the way on its own. Calibrate again with the arm folded (lerobot-calibrate) and record the pose again (quackd robot rest-pose arm-01) to make the fold reachable
```

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
| `lerobot real: connect failed 3 times: Could not connect on port ...` | the port is wrong, or something else already owns it. It is tried three times like any connect failure, since a port busy for a moment is as passing as a lost packet | LeRobot's own words come through, and they name `lerobot-find-port`, which is the way to be sure. The Feetech bus has one owner at a time, so close any teleoperation, recording or serial monitor still holding it, and on Linux check that your user can open the port (upstream's own line is `sudo chmod 666 /dev/ttyACM0`; the port's group, usually `dialout`, is the version that survives a reboot). No attempt opened the port, so nothing was written to a motor and the message says nothing about torque |
| `connect attempt 1 of 3 failed on <joint> (id <N>): Failed to write 'Lock' on id_=<N> ...`, and the session carries on | the bus lost a status packet on one of the torque writes LeRobot's connect makes (`Lock` or `Torque_Enable`), and quackd closed the port without writing anything and connected again. Seen three times on 2026-09-23, each on the first connect after the power had been off | nothing, once. The same joint named session after session is a cable to reseat: the one into that servo, and its connectors |
| `lerobot real: connect failed 3 times, the last on <joint> (id <N>): Failed to write ...` | every attempt failed, the last one on that servo. The torque writes may have stopped part way, so some motors can be holding and others limp, which the message says | keep a hand under the arm. Check that joint's cable and connectors, that the servo supply is on, and that nothing else has the port open, then connect again. A message that names no joint says to check the arm's cables and power instead |
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
| `move_joints: elbow_flex is at 12 with a goal of 45, and it has stopped moving` | a stall: five ticks of 0.1 s in which no watched joint moved more than half a step, counted once the move's ramp has handed the servo the goal itself | something is in the way, a mechanical limit the calibration does not know about, or a tripped servo. The arm is held first. A joint blocked partway through a long `duration_s` is only called stalled when that time is up. The same sentence ending `when the time ran out` means the joint was still moving when the verb's limit came: the time asked for, or the time the step cap needs if that is longer, plus 2.5 s, and never more than 18 s. A lowered `QUACKD_LEROBOT_MAX_STEP_DEG` on a long move gets there, and so does a joint creeping under a load |
| `the camera gave no frame: TimeoutError: ... too old` | the webcam stalled or was unplugged | only `observe` is affected, and a `pick` in flight. The arm carries on, and `report_state` starts saying `CAMERA DOWN:` with the reason, so a run that cannot call `observe` still records it |
| `cannot move_joints: the arm's torque is off, so a goal would reach a limp servo` | torque reads off | no verb can toggle torque either way. A fresh connect re-enables it, so torque still off after one points at a tripped servo or the supply. On a `--by-hand` run this is also what the arm reads like between the release and the moment quackd takes hold again, which is before the first turn |
| `cannot place: nothing is held: pick something first` | the `holding` precondition | holding is inferred from the gripper stopping short of shut, so an empty hand reads as nothing held. After a `--by-hand` start it is also what a pilot gets for the pencil you put between the jaws yourself: closing the gripper by hand sets a position and not a grip, and the pilot has to close on the object itself first |
| the run ends saying the arm did not answer | the heartbeat's round trip to the motors failed | the cable, the power, or a servo that has tripped. The arm holds its last goal under torque |
| the arm sags when the run ends | no rest pose is recorded, so LeRobot's `disconnect()` disables torque by its own default, at the end of every clean session | record one: `quackd robot rest-pose <name>`. Until you do, support it or fold it somewhere it can rest before you exit |
| `the arm is not at its rest pose (...), so torque was left on and it will not fall as it stands: hold it first, because connecting takes torque off every motor for a moment, then run quackd robot release NAME, or quackd doctor --robot NAME to park it, or cut its power` | the arm did not reach the pose you recorded, or the edge of its travel where the pose lies past it, so quackd kept torque rather than dropping it. A run at a terminal offered to release it first, and nobody pressed Enter | hold the arm before anything else, since both commands connect and connecting drops torque for a moment. Then run `quackd robot release NAME` to have it let go into your hands ([Releasing it where it stands](#releasing-it-where-it-stands)), or run `quackd doctor --robot NAME` to let the rest move try again from where it now is, or cut the servo supply. The parenthesis names the joints that fell short |
| `quackd cannot tell whether the arm is holding itself up (the arm did not answer: ...), so it kept whatever torque the arm has: hold it, and cut its power` | the arm did not answer the close's last read, so nothing says where it is or whether its servos are powered. Cutting the supply looks exactly like this, and so does a cable that came out in front of live servos | hold it, and cut the servo supply. No offer is made at the end of a run over an arm that went quiet |
| `torque still reads on for <joints>: cut the power` from `quackd robot release` | those motors kept their torque through the release | cut the servo supply while you hold the arm. The motors not named are limp, and the line after it says what the close then did ([Releasing it where it stands](#releasing-it-where-it-stands)) |

| a run aborts with `the arm did not reach its rest pose: ...` before any model call | the run could not start from the recorded pose | something is in the way, or the pose no longer matches the arm, which is what a new calibration does to a pose recorded before it. Record it again, or move whatever is blocking the fold. A fold past the calibrated travel no longer ends a run here: the arm parks at the edge |
| `<joint> is recorded at <angle> in the rest pose and this calibration lets its servo be driven to <limit> and no further, so it parks there and is let go of there ...` | the fold you recorded lies past the travel in your calibration file, so the arm parks at the edge of it and is let go of there. Not a fault: the run carries on | calibrate again with every joint taken all the way into the fold, then record the pose again ([A pose past the travel](#a-pose-past-the-travel)) |
| `report_state` says `<joint> reads <angle>, past the <limit> its servo can be driven to` | that joint was folded or placed past its travel with torque off, which is where a rest pose past the travel leaves it, or it was parked at its limit and has sagged a few degrees past it under its own weight | nothing. It is said so the pilot does not take the reading for a fault, and goals are still limited to the travel |

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
| `SOFollower.connect() refuses while the port is open` | `check_if_already_connected`, and `is_connected` is the port's flag. connect() opens the port first and configures last, and nothing shuts the port when configure() raises, so a retried connect has to close it first (`MotorsBus.disconnect(False)`) |
| `configure() switches torque off and on again with no retry` | `torque_disabled()` calls `disable_torque()` and `enable_torque()` with `num_retry` 0, each a `Torque_Enable` then a `Lock` write per motor, so one lost status packet fails the whole connect with the motors in two torque states. The bench arm did this on three connects on 2026-09-23; quackd connects again, up to three attempts |
| `SOFollower.bus is a FeetechMotorsBus` | the attribute registers are read through |
| `no deadman: nothing stops the arm when the client goes quiet` | the class has no thread, timer or timeout; a goal stands until the next write |
| `MotorsBus.disable_torque()` | never called on quackd's own initiative. The one call is `let_go()`, and it has two doors, each opened by a person at a terminal. `let_go()` is `quackd run --by-hand`'s, and refuses anywhere but the arm's recorded rest pose, the same condition `close()` uses to decide that letting go will not drop it. `let_go(anywhere=True)` is `quackd robot release`'s and the end-of-run offer's, after each has told the person to hold the arm, and releases wherever the arm stands, with or without a rest pose recorded. No verb reaches either and no model can ask for it. On a Feetech bus it writes `Torque_Enable` 0 then `Lock` 0 per motor, and quackd asks for `num_retry=5`, the count upstream's own `disconnect()` uses |
| `MotorsBus.enable_torque()` | called by `take_hold()`, to pick up an arm a person has just placed, with `num_retry=5` as for the release. It writes `Torque_Enable` 1 **and then `Lock` 1** per motor, two writes a motor rather than one |
| `MotorsBus.disconnect(disable_torque=True)` | the `disable_torque()` call is inside `if disable_torque`, so False closes the port and leaves every motor holding the goal it was last written: what an arm that missed its rest pose gets instead of falling, and how the port is closed between two connect attempts without a write to any motor |
| `MotorsBus.motors: name -> Motor(id, model, norm_mode)` | the table that gives each servo its bus address; a bus error's id is turned into a joint through it, never through an assumed order |
| `Failed to write '<register>' on id_=<N> with '<value>' after <k> tries. <result>` | what a single write or read that failed raises. quackd reads the id out of it to name the joint; a sync read or write names several and quackd names none |
| `MotorsBus.is_connected is port_handler.is_open` | a port flag, not a reply: why the heartbeat reads the arm |
| `FeetechMotorsBus.is_calibrated reads the motors back` | a missing, stale or foreign file all read as not calibrated |
| `write_calibration() is reached only through calibrate()` | quackd cannot move an arm's zero by accident |
| `MotorsBus.sync_read(data_name, motors=None, normalize=True, num_retry=0)` | one transaction for every motor named |
| `NORMALIZED_DATA is Goal_Position and Present_Position` | every other register comes back raw |
| `MotorCalibration(id, drive_mode, homing_offset, range_min, range_max)` | raw encoder ticks, not degrees |
| `degrees = (raw - mid) * 360 / 4095` | how a calibration file becomes a range in degrees, centred on zero |
| `a degrees goal is not clamped to the calibrated range` | the two 0..100 modes are clamped and DEGREES is not, so the servo's own clamp is what stops a goal past the travel, silently: why quackd refuses |
| `write_calibration() writes Min_Position_Limit and Max_Position_Limit, and the servo clamps Goal_Position to them` | the servo clamps every goal to the calibrated travel and never a reading, seen on an SO-101 on 2026-09-23: why a rest pose is clipped into the travel and a joint past it gets no goal |
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
| `TORQUE_ENABLE_HOLDS_PRESENT` | what a servo does with the goal it was last told when torque comes back on. `enable_torque()` writes `Torque_Enable` and then `Lock` on each motor, neither of them a goal, so whether the motor then holds where it is or drives to that stale goal is the firmware's business and is documented nowhere quackd can read. It matters because the goal last written before a hand-off is the rest pose the arm has since been lifted out of by hand, so a snap back to it would happen with somebody's fingers in the way. `take_hold()` writes the present position as the goal **before** enabling torque, writes it again after, and reads the arm back to check it stayed, so the assumption is never relied on in either direction. The one exception is a joint placed past its calibrated travel, which gets no goal at all because the servo would clamp it to the limit (`POSITION_LIMITS_CLAMP_GOALS`). For that joint this row is all there is, and the read-back is what says whether it moved |
| `GRIPPER_OPEN_VALUE` | 100 is assumed open; which end is open is how the arm was calibrated, and the checklist asks for it by hand |
| `HOLDING_INFERRED` | holding is the gripper told to close, settled, and short of shut; listed in `extras.assumptions`. A gripper a person closed by hand is a position and not a grip, so an arm placed with `--by-hand` reports nothing held until the pilot closes the gripper itself |
| `TEMPERATURE_C` | the register is read raw and treated as Celsius; the 60 °C refusal and the 70 °C cut-off are Feetech's numbers, not measured |
| `JOINT_RANGES` | each joint's travel is computed from the calibration file and a goal outside it is refused. It is not the mechanical limit on every arm: a calibration that never saw a joint folded all the way leaves the fold past it, as the bench arm's did on 2026-09-23, which is why a rest pose is clipped into the travel. Whether it is the mechanical limit on any given arm is unverified |
| `SERIAL_PORT` | `--address` is checked for shape and nothing more |
| `THREAD_SAFETY` | every call is serialised under one lock in a worker thread with a deadline; a blown deadline wedges the transport |
| `CAMERA_INDEX_MOVES` | an index is a scan position, not an identity: it can move on a replug or a reboot, and a laptop's own webcam usually holds 0. quackd records the index it opened and cannot tell you it is the camera you meant |
| `WINDOWS_CAMERA_BACKEND` | which backend a Windows machine needs for a given webcam is not knowable in advance, so quackd keeps upstream's ANY and gives the owner `?backend=msmf` |

## Status

`lerobot:mock` runs every arm verb through the executor in the test suite, including the
confirm gate on `pick`, the `holding` precondition on `place` and the heat refusal.
`lerobot:real` is exercised against a fake arm and a fake policy (verified method names, no
serial port), and it has now run on one real arm, on 2026-09-15 and again on 2026-09-23.

### What one afternoon proved, and what it did not

On 2026-09-15 an SO-101 follower, calibrated as `arm-01` and reached as `--robot lerobot:real
--address COM3` with no registered name, ran on Windows 11 with
Python 3.12.12, lerobot 0.6.1 and quackd 0.9.0, piloted by OpenAI's `gpt-6-astra`. This is the
only one of quackd's seven bodies that has been on hardware at all.

What ran:

- `lerobot-lookout`, with a real pilot and once with `--llm fake`, which is how you tell
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
  this page written after hardware rather than before it. It met the same arm on 2026-09-23 and
  could not reach a fold that lay past the calibrated travel, which is the box at the top of
  that section.
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
