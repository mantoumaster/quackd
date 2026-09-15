# A LeRobot SO-101 arm: from an empty laptop to a wave

This is the long way round, written for somebody who owns or can borrow an SO-101 and has
never run quackd or LeRobot. It starts with an empty laptop and ends with the arm noticing
you on a webcam and waving, driven by whichever model you choose to bring.

Three documents cover this arm and they do different jobs:

- **This one** is the narrative. It assumes nothing and it moves slowly.
- [lerobot-hardware-checklist.md](lerobot-hardware-checklist.md) is the risk ladder, sixteen
  steps in the order that can only fail safely. Once anything is about to move, that file is
  the authority and this one hands over to it.
- [adapters/lerobot.md](adapters/lerobot.md) is the reference: the manifest, every verb, the
  camera query keys, and the full table of what quackd refuses and why.

> [!IMPORTANT]
> Nothing in quackd has ever run on a real SO-101. The `lerobot:real` backend speaks names
> read from LeRobot's source at a pinned commit and has only ever talked to a fake arm. You
> are not retracing a path here, you are the first person down it. [What to
> report](#13-what-to-report) is the part that matters most, whether or not the arm waves.

<br>

## 00. Read this first

Three things this task implies that the arm cannot do. Read them now so that nothing at the
bench is a surprise.

**There is no `wave` verb.** This body's whole vocabulary is `report_state`, `move_joints`,
`gripper`, `place` and `stop`, plus `observe` when a camera is configured. A wave has to be
invented by the model out of several `move_joints` calls, live, the first time you ask. That
improvisation is the thing being tested. It is not a feature that already exists.

**The camera cannot look for you.** No SO-101 has one built in, and quackd does not put one
on the arm: it is a USB webcam that plugs into your computer, aimed once and then fixed. This
body has no neck and no `search_scan`. "Finds you" means you are already inside a frame that
never moves, so you position yourself rather than the arm.

**Whether it can see you at all depends on the model you bring.** quackd's bundled detector
is a colour threshold carrying the simulator's own ranges, so on a real desk `person` means
"a saturated blue thing" and nothing else. A model that accepts images sees the actual
webcam frame every step and can simply look. A model that does not gets one line of text.
[Choose your pilot](#03-choose-your-pilot) is where that decision gets made, and
[Wave to me](#11-wave-to-me) is where it pays off or does not.

> [!CAUTION]
> There is no e-stop on an SO-101 and quackd cannot give it one. LeRobot writes a torque and
> current cap on the gripper and on nothing else, so the five body joints run at whatever
> their firmware defaults to. quackd bounds which angles a joint may reach and how fast it
> gets there. It does not bound how hard it pushes. Cutting the servo supply is the only
> thing that stops this arm in every case, including the one where the controlling process
> has died with a goal still standing.

Two reflexes, and they matter more than anything you type:

1. Know which plug or switch kills power before anything is powered on. You are probably a
   guest with somebody else's hardware, so run the supply through a power strip you can
   reach rather than modifying their equipment.
2. Keep fingers out of the gripper's jaws any time torque might be on. A pinch hazard does
   not need a command in flight.

<br>

## 01. What to bring

- **A laptop with quackd already installed.** Section 02 needs real internet and pulls
  torch, so do it the night before rather than on lab wifi.
- **A couple of USB cables for the arm's controller board.** You will not know which port it
  wants until you are in front of it, so bring more than one shape, and a hub if your laptop
  is short on ports.
- **A USB webcam, and something to prop it up and aim it.** A small tripod, a clip, a stack
  of books. The arm cannot move the camera for you.
- **Something strongly blue to wear**, if the model you are bringing does not take images.
  Section 03 explains why, and it is the difference between the arm seeing you and not.
- **A way to cut power fast.** A power strip with its own switch, or a firm decision about
  which plug you are going to pull.
- **Permission, and the arm's own parts list.** Whoever runs the lab should confirm that you
  may connect your own laptop, and which supply that specific arm takes. Motors and supplies
  vary between builds and neither LeRobot nor quackd reads the voltage, so neither can warn
  you.
- **A key for one cloud vendor, or a local model server.** Section 03.

<br>

## 02. Install quackd

Python 3.12 or newer, in a clean environment of its own. quackd's own floor is 3.11, but the
`lerobot` extra carries a `python_version >= '3.12'` marker, and below that floor it resolves
to nothing at all while the install still reports success.

Every command in this guide uses [uv](https://docs.astral.sh/uv/). quackd does not require
it, it is simply what the documented commands assume.

```bash
uv venv --python 3.12
uv pip install "quackd[lerobot]"
```

Add the extra for whichever pilot you picked in the next section, in the same install. The
arm and the model are independent choices:

```bash
uv pip install "quackd[lerobot,anthropic]"   # or openai, gemini, grok, mistral, deepseek, ...
uv pip install "quackd[lerobot,openai]"      # the openai extra also serves every local server
```

Then check the machine:

```bash
quackd doctor
```

Two rows decide whether a serial port can be opened at all:

```
- lerobot                    not installed (quackd[lerobot])
- lerobot (feetech bus)      not installed (quackd[lerobot])
```

Both have to be green or the port will not open. The second one is the trap: the Feetech SDK
lives in LeRobot's own `[feetech]` extra rather than in its base dependencies, so a plain
`pip install lerobot` gives you a package that imports perfectly and then cannot talk to a
motor. `quackd[lerobot]` asks for `lerobot[feetech]` for exactly that reason.

> [!WARNING]
> If `lerobot` still reads `not installed` after an install that succeeded, check
> `python --version` before you reinstall anything. On 3.11 the extra resolves to zero
> packages, silently, and `doctor` has nothing else to complain about.

<br>

## 03. Choose your pilot

quackd never ships a model. You bring one, and the choice is a flag. Eleven cloud vendors
have a `--provider` name, five local presets cover the common self-hosted servers, and
`fake` is a scripted pilot that needs no key and is not a model at all.

`quackd doctor` prints a row per provider with the extra, the key it found and the model it
would use, and `quackd list-models` prints every model id quackd knows for every vendor.

### A cloud vendor

Put the key in the environment or in a `.env` file beside where you run quackd, then name
the provider. The key variable per vendor is in [`.env.example`](../.env.example) and in
`quackd doctor`.

```bash
quackd run lerobot-lookout --robot lerobot:real --address COM5 --provider openai
```

> [!NOTE]
> The provider for Claude is spelled `anthropic`. There is no `--provider claude`, and an
> unknown name is refused before anything connects, with the full list in the message.

### A local model, no key

Five presets, all of them served by the `openai` extra because they all speak the OpenAI
wire format. Start your server, then name the preset:

```bash
ollama pull qwen3:8b
quackd run lerobot-lookout --robot lerobot:real --address COM5 --provider ollama --model qwen3:8b
```

The other presets are `vllm`, `llamacpp`, `lmstudio`, and `local` for anything else, which
takes `--base-url http://host:port/v1`. Servers need their own tool-calling switches turned
on: vLLM wants `--enable-auto-tool-choice --tool-call-parser <family>`, and llama-server
wants `--jinja`. [local-llms.md](local-llms.md) has the per-server detail, including the
JSON text fallback quackd uses when a server is weak at native tool calls.

> [!WARNING]
> If an `OPENAI_API_KEY` is in your environment, a local run sends it to your local server as
> the bearer token, because that is the last fallback before the literal string
> `not-needed`. Set `LOCAL_API_KEY=not-needed` or pass `--api-key not-needed` if that server
> is not yours.

### The one that decides whether the arm can see you

Whether the camera frame reaches the model at all is a per-provider default:

| Pilot | Frame reaches the model? |
|---|---|
| A cloud vendor, default model | Yes, on every step, for most vendors |
| A cloud model marked `no frames` | No. The text detections go instead, silently |
| Any local preset | **No by default.** `--vision` turns it on |
| `--provider fake` | Never. It is a rule, not a model |

`quackd list-models` prints `no frames` in the notes column for the models that do not take
images, and it is worth reading before you promise yourself the arm will see anything. One
vendor's own default model is marked that way, so a bare `--provider glm` run never sees the
webcam.

`--vision` and `--no-vision` override all of it in both directions. A local pilot that is
meant to look at you needs `--vision` **and** a vision-capable model loaded in the server.

### The scripted pilot, and its one job

`--provider fake` is the default and needs no key, no extra and no network. It is not a
model: it is a small set of rules that picks its script from the task file's name, or from a
few keywords in a goal.

> [!CAUTION]
> `fake` cannot do the wave, and it will not tell you so. A goal it does not recognise falls
> through to a generic script whose only two moves are `quack` and `search_scan`, and it
> checks the allowlist before reaching for either. An arm has neither, so it goes straight to
> declaring success: `--goal "wave to me" --provider fake` ends with every joint exactly
> where it started, and exits 0, with no refusal printed to warn you. Worse, a goal containing
> the word *person* selects the patrol script, which reaches for verbs this body does not
> have and gets refused one at a time. Use `fake` to prove the wiring in section 07, and a
> real model for anything that has to think.

<br>

## 04. At the lab, before power

Everything here happens with the arm still unplugged.

- Clear the whole sweep the arm can reach, not just the desk footprint. An arm sweeps a
  volume rather than occupying a spot.
- Take anything fragile out of the gripper and off the desk within arm's length.
- Confirm your switch, or the plug you have decided on, is within reach of where you will
  actually be standing.
- Confirm with whoever owns the arm which supply it takes, and that it is already the right
  one.

<br>

## 05. Find the port, then calibrate

This is the step that is easiest to skip and cannot be. quackd refuses to drive an arm that
is not calibrated, because the calibration file is where every joint's travel comes from,
and calibration is LeRobot's own interactive tool which quackd never triggers.

Find the port first:

```bash
lerobot-find-port
```

It lists the ports, asks you to unplug the arm, and names the one that disappeared. That is
worth doing even when you are sure, because it is the only answer that is not a guess. On
Windows the port is `COMx` and also appears under Ports (COM & LPT) in Device Manager; on
Linux it is usually `/dev/ttyACM0`, and upstream's own fix for permissions is
`sudo chmod 666 /dev/ttyACM0`, with adding your user to that port's group being the version
that survives a reboot. If no port appears at all, suspect the cable or the barrel jack
before you go looking for a driver: USB does not power the controller board.

Then calibrate. Nothing moves on its own here. The tool asks *you* to move each joint
through its range.

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=COM5 --robot.id=arm-01
```

> [!IMPORTANT]
> The id has to be the one quackd will use. `arm-01` is the default behind a bare
> `--robot lerobot:real`. Name the robot anything else later, with
> `--robots arm=lerobot:real` or `quackd robot add lab-arm lerobot:real`, and the id becomes
> that name instead, so calibrate under it. Two arms sharing an id share one file with
> nothing in it to say which arm it came from, so if this is not the only SO-101 in the room
> today, pick an id nobody else is using.

It writes `<calibration dir>/robots/so_follower/<id>.json`, where the directory is
`$HF_LEROBOT_CALIBRATION`, else `$HF_LEROBOT_HOME/calibration`, else
`$HF_HOME/lerobot/calibration`. You do not have to hunt for it: `doctor` prints the path it
actually loaded, which is the fastest way to see that you calibrated `arm` and are
connecting as `arm-01`.

> [!NOTE]
> Upstream will ask you to move every joint through its range **except** `wrist_roll`, and it
> records a full encoder turn for that one rather than anything you swept. That is not a
> mistake you can correct. It is why the out of range refusal in section 10 works on the
> other body joints and cannot fire on that one.

<br>

## 06. First contact

The first command that energises the arm. It connects, reads, and moves nothing.

```bash
quackd doctor --robot lerobot:real --address COM5
```

> [!CAUTION]
> Support the arm while this starts and before it finishes. `configure()` runs with torque
> off, so connecting drops it for a moment, and LeRobot's `disconnect()` disables it again by
> default at the end of every clean session, a `doctor` probe included. An arm folded
> somewhere awkward will fall at either end.

Read four things off it:

- The calibration file it found is the one you just wrote.
- Each joint's range looks like the travel you swept.
- Torque is on.
- What the servos report for temperature with the arm cold. Write that number down. It is
  the baseline for everything later.

<br>

## 07. The first task

`lerobot-lookout` ships with quackd, moves no joint, and asks only for `report_state`. It is
the first thing to point at an arm nobody has driven. The scripted pilot is enough here,
because there is nothing to improvise:

```bash
quackd run lerobot-lookout --robot lerobot:real --address COM5 --provider fake
```

Expect one sentence naming where the joints are, whether torque is on, and whether anything
reads hot. A joint at or above 60 degrees Celsius is hot and worth naming; the servo's own
cut-off is 70.

Every run writes `runs/<timestamp>-<name>/` with the full transcript, every frame quackd
captured and a summary. `quackd trace` replays any of it afterwards.

<br>

## 08. Add the camera

Find which OpenCV index your webcam is, which is the part nobody can guess for you:

```bash
lerobot-find-cameras opencv
```

It lists every camera it can open and saves a frame from each under
`outputs/captured_images/`, so you can look at the pictures rather than guess. On a laptop
index 0 is usually the built-in webcam, so a plugged-in one is often 1 or 2. An index is a
scan position and not an identity: it can move when you replug or reboot.

Then ask quackd for a frame through it:

```bash
quackd doctor --robot lerobot:real --address COM5 --camera-url "opencv://1"
```

Quote the url. A bare `&` is a parse error in PowerShell and backgrounds the command in
bash. The url also takes a device path, `opencv:///dev/video2`, and the query keys `name`,
`width`, `height`, `fps`, `fourcc`, `rotation`, `backend` and `fov`. Add `?backend=msmf` if a
Windows camera lists and then will not open.

A camera you asked for and did not get is a refusal at connect, and it happens before the
arm is touched, so a wrong index costs you nothing but the message.

> [!TIP]
> Aim the webcam now at wherever you will actually stand, and pass `--fov-deg` for your lens
> once you know it. Without it quackd assumes the simulator's 90 degrees, says so in every
> detection line, and every bearing and distance is scaled wrong.

<br>

## 09. Rehearse with `--dry-run`

`--dry-run` connects to the arm for real and sends it nothing. Read-only verbs actually run,
so `report_state` reads the servos and the heartbeat keeps its round trip going; every other
verb is printed and skipped.

Rehearse the goal you actually intend to give it:

```bash
quackd run --goal "wave to me" --robot lerobot:real --address COM5 \
  --camera-url "opencv://1" --provider openai --max-steps 6 --dry-run
```

Every verb that would move a joint is printed and skipped. This is what that looks like,
quoted from the checklist's own narrower rehearsal, whose goal was to roll the wrist ten
degrees rather than to wave:

```
[dry-run] would run move_joints({'positions': {'wrist_roll': 10.0}, 'duration_s': 2.0})
[dry-run] move_joints not sent
```

A wave is several of those in a row, alternating about a neutral pose. Read three things
off it. That the model reached for the verbs you expected, with arguments
that look sane. That the joint goals are small rather than enormous. And that the arm
answered every heartbeat for the length of the run, because an arm that drops out here would
have dropped out mid move in the next section.

This costs a handful of API calls and is the cheapest rehearsal you will get. Run it more
than once if the plan looks odd.

> [!NOTE]
> `--max-steps` counts verb executions, not model calls. `assess_task` and the declarations
> cost no step. `max_llm_calls` and `max_minutes` exist too, but only a `.duck` file can set
> them.

<br>

## 10. Prove the safety net

From here the [hardware checklist](lerobot-hardware-checklist.md) is the authority on order
and on what a hand stays near. What follows is the same five checks expressed as commands.

Drop `--dry-run`, keep `--max-steps` small, and watch the arm rather than the terminal.

> [!CAUTION]
> This is where the arm starts moving, so from here **a hand stays on the power switch**.
> There is no e-stop, and cutting the servo supply is the only thing that stops this arm in
> every case. Keep the sweep clear and your hands out of it for everything below.

**1. The gripper, and which way it goes.**

```bash
quackd run --goal "open the gripper fully, then close it on nothing, then stop" \
  --robot lerobot:real --address COM5 --provider openai --max-steps 4
```

quackd assumes 100 is open and 0 is closed, and that is an assumption about how your arm was
assembled and calibrated rather than a fact about the model. If yours runs the other way,
stop here and say so in an issue: everything quackd believes about holding something rests
on it.

**2. One joint, small, in the middle of its range.**

```bash
quackd run --goal "roll the wrist ten degrees and stop" --robot lerobot:real \
  --address COM5 --provider openai --max-steps 3
```

It should take about a fifth of a second and stop. The arm moves at five degrees per action
re-sent ten times a second, so fifty degrees a second, and `QUACKD_LEROBOT_MAX_STEP_DEG`
lowers that if it looks fast in the room.

**3. A goal outside the calibrated range.**

```bash
quackd run --goal "move shoulder_pan to 170 degrees" --robot lerobot:real \
  --address COM5 --provider openai --max-steps 3
```

It should be refused with the real range in the reason, and nothing should reach the arm.
Use any body joint except `wrist_roll`, whose recorded travel is the whole turn.

> [!NOTE]
> This one maps imperfectly to a goal run, and it is worth knowing why. You are trying to
> test quackd's range gate, but the model chooses the number, so it may talk itself out of
> the attempt first, or pick a different joint. If the refusal you get is the pilot's rather
> than the executor's, you have learned something about the model and nothing about the
> gate. Over MCP you send the number yourself, which is the appendix.

**4. Pull the USB cable mid move.** Start a longer motion, then unplug the arm. The run
should end within about a second saying the arm did not answer. The arm holds its last goal
under torque: it must not sag and it must not carry on. Plug it back in and reconnect before
the next step.

**5. Ctrl-C mid move.** quackd's kill switch sends `stop`, which re-sends the present
position as the goal. The arm should freeze where it is rather than sag, and rather than
finish the motion it was in the middle of. `q` at the terminal does the same thing.

> [!CAUTION]
> If any of these five surprises you, stop. Cut power and read
> [When it will not work](#12-when-it-will-not-work) before going further.

<br>

## 11. Wave to me

Stand where the camera can see you, and ask:

```bash
quackd run --goal "wave to me" --robot lerobot:real --address COM5 \
  --camera-url "opencv://1" --fov-deg 62 --provider openai --max-steps 12
```

`62` there is an example, not a default: it is the figure for one common camera module. Use
your own lens's horizontal field of view, and if you do not know it, leave the flag off and
read the uncalibrated warning every detection line will carry.

What should happen: the model reads an observation that includes the camera, answers
`assess_task` with a verdict, and then issues several small `move_joints` calls alternating
about a neutral pose, before stopping. The motion is genuinely the arm's own joints doing
something nobody scripted. That is the whole thesis under test.

For a first attempt, a goal that says more gives you a better idea of what is coming:

```bash
quackd run --goal "If you can see a person in the camera, greet them: move wrist_flex, \
shoulder_pan and elbow_flex back and forth a few times, no more than about 20 degrees from \
where each one is now, in several small moves rather than one big one. Keep any wrist_roll \
move especially small. Do not touch the gripper. Then return to the start and stop." \
  --robot lerobot:real --address COM5 --camera-url "opencv://1" --provider openai --max-steps 12
```

quackd's own ceilings hold underneath whatever the model decides. It cannot put a joint
outside its calibrated range, and it cannot move faster than the step cap, however the
request is phrased. What quackd does not cap is force, so the sweep still has to be clear and
your hands still have to be out of it.

### Whether it can actually see you

This is the honest part, and it differs by pilot.

**A model that takes images** receives the webcam frame on every step and can simply look and
decide. Only the last two exchanges keep their image, so it cannot compare a frame from six
steps ago, but it can see you now.

**A model that does not** receives one line of text built by quackd's colour detector, and
that detector carries the simulator's ranges. It emits exactly four labels and `person`
means a saturated blue region, not a human. Wear a strong blue top and fill a decent share
of the frame and you will read as `person at bearing 12° left ~1.40 m`. Wear grey and the
line says `nothing detected`. The distance is computed from the *simulated* person's size,
so treat it as a rough guess rather than a measurement.

> [!WARNING]
> `observe` is not reachable on a `--goal` run even with a camera attached. A goal's
> allowlist is built from the arm's static manifest, and that manifest cannot know whether
> you plugged a webcam in, so it claims no camera. The frame and the detections still reach
> the pilot in every observation, which is how "sees a person and waves" works here. Asking
> for a look as a deliberate act needs an MCP client, which is the appendix.

There is a real person detector behind the `yolo` extra, and today it is reachable only from
Python by constructing `YoloDetector()` and passing it in. No CLI flag selects it. The same
is true of tuning the colour ranges to your own shirt, which [the FAQ](faq.md) covers as a
Python constructor.

### Making it repeatable

A `.duck` file turns the goal into a contract with an allowlist, a budget and a success test
that the model cannot talk its way out of. It is also the provider-agnostic way to carry this
task around, because `providers:` is a tested-with note rather than a restriction:

```markdown
---
duck: 1
name: wave-hello
description: Notice a person on the camera and greet them with a waving motion.
requires: [move_joints, report_state]
verbs:
  allow: [report_state, move_joints, stop]
  confirm: []
budgets: {max_steps: 20, max_minutes: 3, max_llm_calls: 20}
success:
  - You have moved the body joints back and forth several times and returned to the start.
abort_when:
  - Same verb fails 3 times in a row
providers: [anthropic, openai, gemini]
---

# Task

Greet the person in front of you with a wave, using only small joint moves.
```

Check it against this body before you run it, which happens before anything connects:

```bash
quackd validate wave-hello.duck --robot lerobot:real
```

Note what the allowlist does **not** contain. A task that so much as allows `observe` is
refused on `lerobot:real` for the reason above, so leave it out and let the frame arrive in
the observation.

<br>

## 12. When it will not work

The arm is not touched by anything in the first group: these all happen before or during
connect.

| What you see | What it means | What to do |
|---|---|---|
| `adapter 'lerobot' needs an extra` | the extra is not in this environment | install it, and check `python --version` is 3.12 or newer |
| `lerobot (feetech bus)` missing in `doctor` | LeRobot is installed without its `[feetech]` extra | reinstall `quackd[lerobot]`, which asks for `lerobot[feetech]` |
| `--address must be the arm's serial port` | no address, or it is not port shaped | Device Manager under Ports on Windows, `/dev/ttyACM0` elsewhere |
| `connect failed` naming a port | wrong port, or something else already owns it | close any teleoperation, recording or serial monitor, then `lerobot-find-port` |
| `the arm is not calibrated` | the motors do not match a calibration | run `lerobot-calibrate` under the id quackd will use |
| `the arm reports no calibration file` | there is no file for this id | the same fix, and check the path `doctor` prints |
| `--camera-url ... did not open` | wrong index, or it will not open under this backend | try another index, add `?backend=msmf`, or drop a pinned size. The arm was not touched |

And once it is running:

| What you see | What it means | What to do |
|---|---|---|
| `is outside this arm's calibrated range` | the goal is outside the travel in your calibration file | working as intended. Aim inside it. On `wrist_roll` this can never fire |
| `reads 61°C: let the arm cool` | the heat gate, below the servo's own 70 °C cut-off | let it cool. A joint that trips its own protection goes slack without announcing it |
| `and it has stopped moving` | a stall: five ticks in which no watched joint moved | something is in the way, or a servo tripped. The arm is held first |
| `the camera gave no frame` | the webcam stalled or was unplugged | the arm carries on, and `report_state` starts saying `CAMERA DOWN:` |
| `the arm's torque is off` | torque reads off | quackd never toggles torque. A fresh connect re-enables it, so this points at a tripped servo or the supply |
| the run ends saying the arm did not answer | the heartbeat's round trip failed | the cable, the power, or a tripped servo. The arm holds its last goal under torque |
| the arm sags when the run ends | LeRobot's `disconnect()` disables torque by its own default | support it, or fold it somewhere it can rest before you exit |

[adapters/lerobot.md](adapters/lerobot.md) has the full table, including the failures SO-101
owners report that nobody here has reproduced.

<br>

## 13. What to report

[Open a LeRobot hardware report](https://github.com/rokbenko/quackd/issues/new?template=lerobot-hardware-report.yml),
or a plain issue with the transcript and your `quackd doctor` output. A report that says it
did not work is worth as much as one that says it did. The six things that most need a real
arm:

- **Which end of the gripper's 0..100 range is open.** Assumed, and everything about holding
  depends on it.
- **Whether the holding band is anywhere near right.** quackd calls it holding when the
  gripper is told to close, settles, and settles between 8 and 90 of 100.
- **What a joint reads in degrees Celsius**, cold and after ten minutes of work. Both the 60
  refusal and the 70 cut-off are Feetech's documentation rather than anything measured here.
- **Whether five degrees an action felt right** in the room.
- **Whether a stall is caught.** Hold a joint gently against its goal and see whether the
  verb fails with where it stopped.
- **Which OpenCV index the camera turned out to be**, and whether it needed `?backend=msmf`.

And one this guide adds: **which model you used, and whether it could tell you were there.**
Nobody has pointed any pilot at a real webcam on a real desk.

Only flip the `real` row in [adapter-status.md](adapter-status.md) once a real arm has done
it, and say in the same commit what it did.

<br>

## Appendix: driving it from an MCP client

Everything above is the command line, because that works with every pilot quackd supports.
There is a second way in, and it is the only way to run exactly one verb and stop, or to ask
for a camera frame as a deliberate act.

`quackd serve-mcp` exposes a robot as [Model Context Protocol](https://modelcontextprotocol.io)
tools over stdio, so the client spawns it as a subprocess and the client's own model is the
pilot. quackd chooses no model in this mode and reads no key.

```bash
quackd serve-mcp --robot lerobot:real --address COM5 --camera-url "opencv://1"
```

The tools are `robot_list_verbs`, `robot_observe`, `robot_assess_task`, `robot_run_verb` and
the rest. `robot_run_verb` refuses anything other than `report_state`, `observe` and `stop`
until a verdict has been recorded, and `robot_observe` returns the frame itself, which is
what makes a deliberate look possible here and not on the CLI. A `.duck` loaded with
`robot_load_duckfile` is checked against the manifest of the robot **already connected**, so
a task that allows `observe` loads cleanly on a session started with a camera.

[mcp.md](mcp.md) has the client configuration, the full tool list and what the trace shows.
Two clients are documented and both are Anthropic's; the transport itself is a plain local
stdio server and nothing in it is specific to them.
