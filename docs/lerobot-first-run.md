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

> [!NOTE]
> One SO-101 has now been down this path. On 2026-09-15 an arm calibrated as `arm-01` ran
> `lerobot-lookout`, and then free-form `--goal` runs, on Windows 11 with Python
> 3.12.12, lerobot 0.6.1 and quackd 0.9.0, piloted by OpenAI's `gpt-6-astra`. It waved by
> rolling the wrist about 27 degrees either way, waved again with `shoulder_lift` at -39 and
> `elbow_flex` between 24 and 30, opened and closed the gripper, and in one run mimed a duck
> quacking with the jaws. A USB webcam on `opencv://1` gave it pictures. It also went limp and
> fell at the end of every one of those runs, which is what [section 07](#07-record-the-rest-pose)
> now exists to fix.
>
> Two things that account does not cover. It was reached as `--robot lerobot:real --address
> COM3`, with no registered name, so the registry steps below have not been run on hardware
> either; `arm-01` is both the id that run calibrated under and the id a bare
> `--robot lerobot:real` uses when you have not named the robot, which is why the two look
> alike. And the rest pose in section 07 was written after that day and has not been tried on
> an arm at all. So you are the second person down this path rather than the first, and what
> differs from that account is the part worth writing down. [What to
> report](#14-what-to-report) still matters most, whether or not the arm waves.

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
[Wave to me](#12-wave-to-me) is where it pays off or does not.

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

Put the key in the environment or in a `.env` file, then name the provider. The key variable
per vendor is in [`.env.example`](../.env.example) and in `quackd doctor`.

```bash
quackd run lerobot-lookout --robot lerobot:real --address COM5 --provider openai
```

> [!NOTE]
> The provider for Claude is spelled `anthropic`. There is no `--provider claude`, and an
> unknown name is refused before anything connects, with the full list in the message.

### Where the key goes

This cost time at the lab, so it gets its own subsection. quackd reads a `.env` from the folder
you are standing in when you type the command, and then from beside its own install, walking up
the folders above it, which is how it finds the one a `uv venv` user put in their venv root.
This is the layout that worked, verified on the machine that drove the arm:

```
D:\Development\lerobot-test\
├── .venv\
│   ├── Lib\
│   ├── Scripts\
│   ├── share\
│   ├── .env            <- the only entry here you create yourself
│   ├── .gitignore
│   ├── .lock
│   ├── CACHEDIR.TAG
│   └── pyvenv.cfg
├── outputs\
└── runs\
```

> [!NOTE]
> uv and the packages you install write everything else there, and a venv straight out of
> `uv venv --python 3.12` is shorter than the tree above: `Lib\`, `Scripts\`, `.gitignore`,
> `CACHEDIR.TAG` and `pyvenv.cfg`, and nothing more. `.lock` appears with the first install,
> and `share\` comes from a package that ships files of its own, which lerobot does. So a
> folder with fewer entries than this is the right folder, not a broken one.

and the file itself is one line:

```
OPENAI_API_KEY=sk-...
```

`quackd doctor` prints a row per provider with the key it found, masked down to its first four
and last two characters, which is the quickest way to see whether your file was read at all.

> [!NOTE]
> Either place works, so put the file wherever you will remember it: next to the command you
> type, or in the venv root as above. Neither file overrides a variable that is already in
> your shell, so a key exported by hand wins over both. Keep one file rather than two: the
> folder you are standing in is read first and nothing read afterwards replaces a name it
> already set, so two files that disagree resolve in an order you have to remember.

> [!WARNING]
> The variable name is case sensitive everywhere except Windows. The file at the lab read
> `OPENAI_API_Key`, which Windows happily resolves and macOS and Linux do not, so that same
> file would have found no key at all on either. Copy the name out of
> [`.env.example`](../.env.example) rather than typing it.

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
> have and gets refused one at a time. Use `fake` to prove the wiring in section 08, and a
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
> The id has to be the one quackd will use, which is why the command above says `arm-01`:
> [section 06](#06-first-contact) registers this arm under that name, and a registered name
> becomes the manifest id and therefore the calibration id quackd goes looking for. Register
> it as `lab-arm` instead and you have to calibrate as `lab-arm`. Two arms sharing an id share
> one file with nothing in it to say which arm it came from, so if this is not the only SO-101
> in the room today, pick a name nobody else is using and use that name in both places.

It writes `<calibration dir>/robots/so_follower/<id>.json`, where the directory is
`$HF_LEROBOT_CALIBRATION`, else `$HF_LEROBOT_HOME/calibration`, else
`$HF_HOME/lerobot/calibration`. You do not have to hunt for it: `doctor` prints the path it
actually loaded, which is the fastest way to see that you calibrated `arm` and are
connecting as `arm-01`.

> [!NOTE]
> Upstream will ask you to move every joint through its range **except** `wrist_roll`, and it
> records a full encoder turn for that one rather than anything you swept. That is not a
> mistake you can correct. It is why the out of range refusal in section 11 works on the
> other body joints and cannot fire on that one.

<br>

## 06. First contact

The first command that energises the arm. It connects, reads, and moves nothing.

```bash
quackd doctor --robot lerobot:real --address COM5
```

> [!CAUTION]
> Support the arm while this starts. `configure()` runs with torque off, so connecting drops
> it for a moment whatever else is true, and an arm folded somewhere awkward falls at that
> moment. Support it at the end too, for now: until [section 07](#07-record-the-rest-pose)
> has recorded a rest pose there is nothing for quackd to put the arm back to, so LeRobot's
> `disconnect()` disables torque where the arm stands, a `doctor` probe included. Once a pose
> is recorded, `doctor` returns the arm to it and leaves torque on if it cannot get there.

Read four things off it:

- The calibration file it found is the one you just wrote.
- Each joint's range looks like the travel you swept.
- Torque is on.
- What the servos report for temperature with the arm cold. Write that number down. It is
  the baseline for everything later.

### Name it

Every command from here on names the arm rather than respelling the backend and the port, and
the next section needs a name to keep a pose under. Register it with the same id you
calibrated:

```bash
quackd robot add arm-01 lerobot:real --address COM5 --provider openai
```

```
✓ added arm-01: lerobot:real at COM5
  quackd run <duck> --robot arm-01
```

`--provider` is whichever pilot you settled on in section 03, and it becomes this robot's
default, so a run that names no provider uses it. The registry file is `~/.quackd/robots.json`
and `quackd robot show arm-01` prints everything in it.

> [!IMPORTANT]
> The name you register is the manifest id, and the manifest id is the calibration id quackd
> looks for. `arm-01` here is the reason section 05 calibrated `arm-01`. If you registered a
> different name, calibrate that name instead, or `doctor` will report an arm with no
> calibration file and refuse to drive it.

Then prove the name resolves to the same arm:

```bash
quackd doctor --robot arm-01
```

It should print exactly what the command before it printed, with the address coming from the
registry instead of from your hand.

<br>

## 07. Record the rest pose

An SO-101 has no brake. It holds its own weight up because torque is on, and LeRobot's
`disconnect()` disables torque by its own default, which quackd keeps. So until quackd had a
rest pose, the arm went limp and fell at the end of every clean run, and at the end of every
`doctor` probe: that is what happened on the bench on 2026-09-15, on every run of the day.
Runs also began from wherever the last one left the arm, so no two started from the same
shape.

A rest pose fixes both. It is one folded pose, recorded once, that the arm can hold with
torque off because it is resting on itself or on the desk rather than held up.

**Fold the arm by hand first, with nothing connected.** An arm nobody has connected has no
torque on it, so it is limp and you can move it. Fold it low and compact, into the shape you
would leave it in overnight. Then record it:

```bash
quackd robot rest-pose arm-01
```

It connects, reads every joint, prints them, asks you whether that is the pose, and keeps the
answer in the registry beside the address and the camera. The capture below is the mock arm
with `--yes`, which answers the question for you, so the numbers are the mock's and yours will
be your own folded arm's:

```
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

> [!NOTE]
> That last line has one exception, and it is a flag rather than a fault. `--by-hand` starts a
> run from a pose you set with your own hands instead of from the recorded one. The recorded
> pose is still where the arm goes first, because it is the only place quackd will let go of
> it, and it is still where the arm folds back to at the end. [Or start from a pose you set by
> hand](#or-start-from-a-pose-you-set-by-hand), below, is the whole of it.

Without `--yes` the same joint table appears and then the question, and nothing is written
until you answer it. `--json` prints the whole registry entry instead and needs `--yes` with
it, because a script cannot answer a prompt: run with no terminal to ask on and the command
refuses with `no terminal to ask on: pass --yes to record it` rather than guessing.

> [!WARNING]
> A pose the arm cannot hold with torque off is a pose it will fall from, and quackd cannot
> tell the difference: it reads the angles you folded the arm into and believes you. Let go of
> the arm before you run the command and watch whether it stays there. If it sags, fold it
> lower and record again.

### What a run then does with it

| When | What happens |
|---|---|
| The start of a run | the arm is driven to the pose before the pilot is given control, so what a model improvises from is the same arm every time. A run that cannot get there aborts before a single LLM call is made |
| The end of a run | between the `stop` and the disconnect, on every exit path there is: success, failure, infeasible, out of budget, an abort, an error, and Ctrl-C |
| Torque, at the end | released only where the arm is known to have reached the pose. Where it has not, quackd turns LeRobot's disconnect flag off, leaves the arm holding itself up, and says so in one line |
| `--dry-run` | nothing. A dry run never moves the arm, at either end, so unless the arm happens to be at the pose already it is let go of with torque on |
| `quackd doctor` | a probe returns the arm to the pose too, and prints a `rest pose` row: `at it already`, `returned to it`, `not reached: ...`, or `none recorded (quackd robot rest-pose <name>)` |
| `quackd robot list --probe` | does not move the arm. It says `torque left on: not at its rest pose` when it had to keep holding it |
| `quackd run --by-hand` | the rest move still happens first, and torque comes off there instead of the pilot being given control. You lift the arm and set the starting pose yourself, quackd holds what you left, and the end of the run is the ordinary one |

Said plainly, because it is a change in behaviour rather than an addition: a probe or a dry
run on an arm that is away from its recorded rest pose now leaves torque **on**, where it used
to drop it. The arm stays up instead of falling, and it stays energised until you cut power or
run something that can put it down.

The line when it could not get there reads:

```
the arm is not at its rest pose (...), so torque was left on and it will not fall: hold the
arm and cut its power, or run again
```

Something is in the way, or a servo tripped. The arm is still energised and still holding
itself up, so hold it and cut power rather than walking away from it.

Two details worth knowing before they surprise you:

- **Only the five body joints are ever driven.** The gripper is recorded and printed, and it
  is never commanded, for the same reason `stop` leaves it alone: re-sending it would open a
  hand that is holding something.
- **The pose is sent without the range clamp.** A folded arm often sits outside the travel its
  calibration recorded, and the bench arm folded to `shoulder_lift` -113.5 against a
  calibrated range of about plus or minus 84.2. The usual out of range refusal would refuse to
  put the arm down, so the rest move does not go through it.

To forget the pose:

```bash
quackd robot rest-pose arm-01 --clear
```

```
✓ cleared arm-01's rest pose
  a run now leaves the arm where it stands, and torque drops there
```

> [!NOTE]
> Only the LeRobot arm is parked today. Every other body refuses a rest pose rather than
> accepting one and quietly ignoring it: a body with no joints says so, and a body with joints
> that quackd does not drive home says that only the LeRobot arm does this today.

### Or start from a pose you set by hand

The rest pose is one shape, and every run so far begins there. That is exactly what you want
when the question is what a model does with a known arm, and exactly what you do not want when
the work starts somewhere else: a pencil already between the jaws, the tip already on the
paper, the arm already over the corner of the desk where the task happens. Asking a model to
get there costs several moves, lands somewhere slightly different every time, and cannot put
anything into the gripper at all, because nobody is there to hand it over.

`--by-hand` is the other way in. You put the arm where the run should start, with your own
hands, and quackd holds it there:

```bash
quackd run --goal "draw a small circle" --robot arm-01 --provider openai --by-hand
```

What happens, in order, and the order is the whole of the feature:

1. **The arm drives to its recorded rest pose**, exactly as any other run does.
2. **quackd takes torque off, there and nowhere else.** This is the only call in quackd that
   de-energises a robot, and it refuses anywhere but the recorded pose, because an arm held up
   by torque alone falls the moment torque goes and your hands are not on it yet.
3. **It tells you the arm is yours, and waits.** There is no clock on this wait. The arm is
   resting on itself at the pose you watched it hold when you recorded it, so it can sit there
   while you go and find the pencil.
4. **You lift it, load the gripper, close the jaws on whatever it is with your fingers, and
   press Enter.**
5. **quackd takes hold of what you left.** It writes the position the arm is in as the goal
   *before* it enables torque, writes it again afterwards, waits a tick, reads back, and tells
   you that you can let go.
6. **The pilot runs**, from your pose rather than from the fold. The budget clock restarts
   here, so the time you spent finding the pencil is not taken out of the model's minutes.
7. **At the end the arm holds where it ended**, and quackd asks you to take out whatever is in
   the gripper before it opens the gripper, waiting up to two minutes for an answer.
8. **Then the ordinary teardown**: the rest move, and torque released there.

Everything quackd says to you, in the order it says it, with the arm placed at `shoulder_lift`
-20, `elbow_flex` 40, `wrist_flex` 15 and the gripper squeezed to 35:

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

[Section 08](#08-the-first-task) puts a whole run's trace around those three lines.

**Read the middle line rather than skimming it.** Those are the angles the arm reported after
torque came back on, not the angles you thought you left. quackd compares them against what it
measured a moment earlier and refuses if any joint has moved more than five degrees, because
the interesting failure here is silent: a heavy forearm sags a little as it takes its own
weight back, and a run that started five degrees below the pose you set would look exactly like
a run that started at it. The refusal names the joint and the gap, the run aborts before the
pilot's first turn, and torque stays on, because an arm that moved is still an arm that is
holding itself up.

**What the pilot is told.** A by-hand run adds a section to the system prompt, so the model is
not left to infer a strange starting shape from the joint angles alone:

```
## Where this run starts
A person placed this body by hand before your first turn, and quackd is holding it exactly
where they left it. This run does **not** start from the recorded rest pose, so do not assume
a folded arm or a known shape: read `report_state` and work from the joint angles it gives you. They
are where somebody decided the work should begin.

The gripper is where their fingers closed it, which is a position and not a grip. Nothing is
reported as held, and nothing should be: closing on an object is what makes this body say it is holding something, so if the task needs a firm hold on what is already between the jaws, call `gripper` to close on it before you lean on it.

When the run ends, the arm is handed back the same way: it holds where you left it, the person
is asked to take whatever is in the gripper, and only then does quackd fold the arm up.
```

**The gripper is a position, not a grip**, and that middle paragraph is the part worth
believing. Taking hold writes every joint's measured position back as its own goal, the
gripper's included, so the jaws are commanded to exactly the width your fingers left them at
and are never squeezed tighter. quackd then calls nothing held, and it is right not to:
holding on this arm is inferred from the gripper having been *told* to close and then settling
short of shut, and a goal of 35 out of 100 is not a close. So `report_state` reads `holding
nothing` with a pencil plainly in the jaws, and the observation the model reads says the same.
That is not a bug to work around. If the task needs a real hold rather than a resting width,
the pilot closes the gripper on what is already between the jaws with the `gripper` verb, and
from that moment the arm reports it as held.

**Ctrl-C while the arm is in your hands** ends the run the way any other abort does, which
means the teardown picks the arm back up before it folds it: torque comes on where you are
holding it, the arm travels to its rest pose, and only there does torque drop. Keep hold of it
and keep your fingers clear of the jaws until it has stopped. The run is recorded as aborted
and the pilot never gets a turn. [Section 11](#11-prove-the-safety-net) has this window and the
other new one in full.

**Five things `--by-hand` refuses**, none of which leave a run directory behind. The first
three are checked before quackd connects to anything, and the last two before the arm is
released:

```
✗ error: --by-hand is one person placing one arm, and this run has several robots
  drop --flock and --robots
✗ error: --by-hand and --dry-run ask for opposite things: one takes torque off the arm,
the other moves nothing
  rehearse the task with --dry-run, then run it again with --by-hand
✗ error: --by-hand waits for you to press Enter, and there is no terminal to ask on
  run it from a terminal, or drop the flag and start from the rest pose
✗ error: microduck:mock is not a body a person places by hand: only the LeRobot arm is
  quackd list-adapters
✗ error: --by-hand releases the arm at its recorded rest pose, and this arm has none
recorded
  quackd robot rest-pose arm-bare
```

The third one is the one that catches people. `--by-hand` is a conversation, so a run started
from a script, a scheduler or a CI job has nobody to talk to, and quackd says so up front
rather than releasing an arm into an empty room and waiting for an Enter that is never coming.

> [!WARNING]
> Everything in this subsection has been exercised against `lerobot:mock` and in the test
> suite, and not yet on a real arm. The two steps that need real servos are the release, where
> a pose you recorded but never let go of could turn out not to hold, and the re-energising,
> where nothing upstream documents what a servo does with the goal it was last told. Have a
> hand on the arm for both the first time, and see [what to report](#14-what-to-report).

<br>

## 08. The first task

`lerobot-lookout` ships with quackd, moves no joint, and asks only for `report_state`. It is
the first thing to point at an arm nobody has driven, and it is the first thing that ran on
the bench arm on 2026-09-15, both with a real pilot and once with `--provider fake`. The
scripted pilot is enough here, because there is nothing to improvise:

```bash
quackd run lerobot-lookout --robot arm-01 --provider fake
```

Expect one sentence naming where the joints are, whether torque is on, and whether anything
reads hot. A joint at or above 60 degrees Celsius is hot and worth naming; the servo's own
cut-off is 70.

The run also says where the arm is against the pose you recorded, once as it starts and again
as it finishes, so the four lines below are two pairs rather than one. This is the mock arm
again, which was already at its pose both times:

```
·  note    moving to the rest pose
·  note    already at the rest pose
·  note    moving to the rest pose
·  note    already at the rest pose
```

The arm was already folded there, so nothing moved. On a real arm the second line reads `at
the rest pose` when it had to travel to get there.

**The same command with `--by-hand` is the first hand-placed run worth making**, and for the
same reason it is the first run of any kind worth making: `lerobot-lookout` moves no joint of
its own. Whatever pose you put the arm in is the pose it is still in when the pilot has
finished, so the only thing under test is the hand-off itself, and a mistake costs you a
re-fold rather than a collision.

```bash
quackd run lerobot-lookout --robot arm-01 --provider fake --by-hand
```

The whole of it, on the mock arm, with the arm placed at `shoulder_lift` -20, `elbow_flex` 40,
`wrist_flex` 15 and the gripper squeezed to 35:

```
·  note    moving to the rest pose
·  note    already at the rest pose
·  hand    released: torque is off at the rest pose
the arm is yours: torque is off at its rest pose, so lift it, put whatever it needs in
the gripper, close the gripper on that, hold it where you want the run to start, and
press Enter
·  hand    held: holding the pose you set (elbow_flex 40, gripper 35, shoulder_lift -20, shoulder_pan 0, wrist_flex 15, wrist_roll 0)
holding the pose you set, you can let go. It is at elbow_flex 40, gripper 35,
shoulder_lift -20, shoulder_pan 0, wrist_flex 15, wrist_roll 0
▶  verb    report_state()
✓  result  report_state ok: shoulder_pan 0, shoulder_lift -20, elbow_flex 40, wrist_flex 15, wrist_roll 0, gripper 35; torque on; hottest shoulder_pan 30°C; holding nothing (0.0 s, 0 intents)
the run is over and the arm is holding where it ended. Take hold of whatever is in the
gripper and press Enter, and the gripper opens before the arm folds up. Leave it and
the arm folds up with the gripper shut
→  send    stop
·  hand    unloaded: opening the gripper
→  send    gripper(open=true)
·  note    moving to the rest pose
·  note    at the rest pose
```

Two lines there are worth checking against the arm in front of you. `report_state` reads back
the pose you set rather than the fold, which is the whole point of the flag, and it says
`holding nothing` with the gripper at 35, which is the inference described in [section
07](#or-start-from-a-pose-you-set-by-hand) rather than an empty hand.

Every run writes `runs/<timestamp>-<name>/` with the full transcript, every frame quackd
captured and a summary. `quackd trace` replays any of it afterwards.

<br>

## 09. Add the camera

Find which OpenCV index your webcam is, which is the part nobody can guess for you:

```bash
lerobot-find-cameras opencv
```

It lists every camera it can open and saves a frame from each under
`outputs/captured_images/`, so you can look at the pictures rather than guess. On a laptop
index 0 is usually the built-in webcam, so a plugged-in one is often 1 or 2. An index is a
scan position and not an identity: it can move when you replug or reboot. On the bench the
plugged-in webcam was `opencv://1` at first and `opencv://2` later, at 640x480, and it needed
no `?backend=` key on Windows.

Then ask quackd for a frame through it:

```bash
quackd doctor --robot arm-01 --camera-url "opencv://1"
```

Quote the url. A bare `&` is a parse error in PowerShell and backgrounds the command in
bash. The url also takes a device path, `opencv:///dev/video2`, and the query keys `name`,
`width`, `height`, `fps`, `fourcc`, `rotation`, `backend` and `fov`. Add `?backend=msmf` if a
Windows camera lists and then will not open.

A camera you asked for and did not get is a refusal at connect, and it happens before the
arm is touched, so a wrong index costs you nothing but the message.

Once the index is the right one, keep it in the registry so no later command has to carry it:

```bash
quackd robot edit arm-01 --camera-url "opencv://1"
```

```
✓ updated arm-01: camera-url
```

Every `--camera-url` on that command replaces the whole stored set, so naming a camera says
where the cameras are today, the way `--address` already does. `quackd robot edit arm-01
--clear camera-url` takes it away again.

> [!TIP]
> Aim the webcam now at wherever you will actually stand, and pass `--fov-deg` for your lens
> once you know it. Without it quackd assumes the simulator's 90 degrees, says so in every
> detection line, and every bearing and distance is scaled wrong.

### More than one camera

`--camera-url` repeats. One view of a desk is rarely enough to tell whether the gripper is
above the thing or in front of it, so the arm takes a second camera:

```bash
quackd robot edit arm-01 \
  --camera-url "opencv://1?name=top" --camera-url "opencv://2?name=side"
```

The same pair of flags works on `quackd robot add`, `quackd run`, `quackd doctor` and
`quackd serve-mcp`. This arm is the only body that reads more than one: every other robot
quackd drives refuses a second `--camera-url` with a message naming who takes several, rather
than opening the first and dropping the rest. The rules are few, and all of them are enforced
before the arm is energised:

| Rule | Why |
|---|---|
| With several, every url carries `?name=`, and the names differ | the name is the only thing telling two views apart, in what the model is shown, in a pick policy's observation, and in `frames/NNNN-<name>.png` |
| An index may not repeat | two handles on one webcam is not two views, it is a camera that will not open twice |
| The **first** url is the primary | it is the camera `--fov-deg` describes, the one the `camera:` detections line reports, and the only one the verbs that steer by sight read. Those run at 10 Hz, and fetching every camera there would blow the deadman window |
| A second camera that will not open refuses the whole connect | it happens before the arm is energised, and it lets go of the first camera on the way out |

Every frame reaches the model on every step, each one labelled with its camera name, on
Claude, both OpenAI APIs, Gemini, and any OpenAI-compatible local server with `--vision` on.

A camera that stalls later costs its own picture and nothing else: the others keep arriving,
and `report_state` and `doctor` name which one went, in a `camera <name>` row each. If the one
that died is the **primary**, the other views still reach the model, and the detections line
reports nothing seen, because a bearing read off a different lens would point somewhere else.

> [!NOTE]
> Two cameras is twice the pictures, and the bill is larger than that. The last two exchanges
> keep their images, so two cameras means four pictures in every request rather than two. Add
> the second one because you need the view, not because it is there.

> [!WARNING]
> A local server, or the model inside it, may accept only one image per message. If a server
> rejects a request that carries two frames, go back to a single `--camera-url`: nothing in
> quackd can make a one-image endpoint take two.

`robots.json` keeps a string when there is one camera and a list when there are several, so a
registry file written by quackd 0.9 loads unchanged.

<br>

## 10. Rehearse with `--dry-run`

`--dry-run` connects to the arm for real and sends it nothing. Read-only verbs actually run,
so `report_state` reads the servos and the heartbeat keeps its round trip going; every other
verb is printed and skipped. The rest move is skipped with them, at both ends, so a dry run
leaves the arm exactly where it found it.

Rehearse the goal you actually intend to give it:

```bash
quackd run --goal "wave to me" --robot arm-01 --provider openai --max-steps 6 --dry-run
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

Two of the bench's dry runs on 2026-09-15 ended early, and both endings were the rehearsal
doing its job. One aborted with `the arm did not answer: TimeoutError` when a single heartbeat
round trip failed, and it did not happen again that day or at all since. The other aborted
because the pilot answered `assess_task` with `uncertain` and the person at the keyboard
answered no.

**`--dry-run` and `--by-hand` are refused together**, for what they ask of the arm rather than
for how many robots a run has. A dry run's promise is that nothing reaches the arm at either
end, and taking torque off is not a command to the robot but a change to it, so there is no
honest way to do both:

```
✗ error: --by-hand and --dry-run ask for opposite things: one takes torque off the arm,
the other moves nothing
  rehearse the task with --dry-run, then run it again with --by-hand
```

Do what the hint says, in that order. Rehearse from the rest pose, read the verbs the model
reached for, and then make the real run the hand-placed one. The refusal lands before quackd
connects, so typing both out of habit costs you the message and nothing else.

> [!NOTE]
> `--max-steps` counts verb executions, not model calls. `assess_task` and the declarations
> cost no step. `max_llm_calls` and `max_minutes` exist too, but only a `.duck` file can set
> them.

<br>

## 11. Prove the safety net

From here the [hardware checklist](lerobot-hardware-checklist.md) is the authority on order
and on what a hand stays near. What follows is the same five checks expressed as commands.

Drop `--dry-run`, keep `--max-steps` small, and watch the arm rather than the terminal. The
first movement of each of these runs is not the model's: the arm travels to the rest pose you
recorded before the pilot is given control, and returns to it at the end.

> [!CAUTION]
> This is where the arm starts moving, so from here **a hand stays on the power switch**.
> There is no e-stop, and cutting the servo supply is the only thing that stops this arm in
> every case. Keep the sweep clear and your hands out of it for everything below.

**1. The gripper, and which way it goes.**

```bash
quackd run --goal "open the gripper fully, then close it on nothing, then stop" \
  --robot arm-01 --provider openai --max-steps 4
```

quackd assumes 100 is open and 0 is closed, and that is an assumption about how your arm was
assembled and calibrated rather than a fact about the model. The bench arm agreed: commanded
100 it reported 98 and stood open, and closed it settled at 3 with the jaws almost touching.
That is one arm. If yours runs the other way, stop here and say so in an issue: everything
quackd believes about holding something rests on it.

**2. One joint, small, in the middle of its range.**

```bash
quackd run --goal "roll the wrist ten degrees and stop" --robot arm-01 \
  --provider openai --max-steps 3
```

It should take about a fifth of a second and stop. The arm moves at five degrees per action
re-sent ten times a second, so fifty degrees a second, and `QUACKD_LEROBOT_MAX_STEP_DEG`
lowers that if it looks fast in the room.

**3. A goal outside the calibrated range.**

```bash
quackd run --goal "move shoulder_pan to 170 degrees" --robot arm-01 \
  --provider openai --max-steps 3
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
finish the motion it was in the middle of. `q` at the terminal does the same thing. Then watch
what follows, because Ctrl-C is an exit path like any other: the arm goes to its rest pose
before quackd lets go of it, and if it cannot get there it stays energised and says so rather
than dropping. Press Ctrl-C a second time and quackd quits at once, which is what the hint
under the header offers; if that lands while the arm is on its way to the pose, the process
exits without disconnecting at all and the arm holds where it stopped. That is the safe
direction and it is still a surprise, so expect it rather than pressing twice out of habit
([safety.md](safety.md)).

A [hand-placed run](#or-start-from-a-pose-you-set-by-hand) opens two more Ctrl-C windows, and
neither behaves like the one above. **During the placement wait**, with the arm limp in your
hands, Ctrl-C ends the run before the pilot has had a turn, and the teardown begins with a
`stop`, which on a released arm means taking hold of it again. Torque comes back on where you
are standing holding it, the arm then travels to its rest pose, and only there does torque
drop. Keep hold of it until it has stopped, and keep your fingers out of the jaws, because from
the arm's side that is an ordinary teardown and nothing about it is slower for being one.
**Inside the end-of-run hand-back**, where quackd is asking you to take whatever is in the
gripper, a second Ctrl-C means skip the gripper rather than abandon the run. The jaws stay
where they are, the arm still parks at its rest pose, the transport still closes properly, and
the trace says `the gripper was left as it is, and the arm still folds up`. That is a
deliberate exception to the rule in the paragraph above, and it exists because the first
version was not one: a second Ctrl-C there raised straight through the whole teardown, which
skipped the rest move, the close, the run's own end record and the summary, and left an
energised arm holding a pencil with nothing written down about the run that put it there. A
third press lands somewhere without that guard and still quits at once.

> [!CAUTION]
> If any of these five surprises you, stop. Cut power and read
> [When it will not work](#13-when-it-will-not-work) before going further.

<br>

## 12. Wave to me

Stand where the camera can see you, and ask:

```bash
quackd run --goal "wave to me" --robot arm-01 --fov-deg 62 --provider openai --max-steps 12
```

`62` there is an example, not a default: it is the figure for one common camera module. Use
your own lens's horizontal field of view, and if you do not know it, leave the flag off and
read the uncalibrated warning every detection line will carry.

What should happen: the model reads an observation that includes the camera, answers
`assess_task` with a verdict, and then issues several small `move_joints` calls alternating
about a neutral pose, before stopping. The motion is genuinely the arm's own joints doing
something nobody scripted. That is the whole thesis under test.

It is also the part that has now happened once. On 2026-09-15 `gpt-6-astra` answered a bare
`--goal` with wrist-roll waves of about 27 degrees either side of where the wrist sat, and in
a later run with a wider gesture: `shoulder_lift` at -39 and `elbow_flex` between 24 and 30,
the whole forearm moving rather than the wrist alone. Another run opened and closed the
gripper, and one mimed a duck quacking with it. None of that is a script in quackd. What your
model does with the same sentence is its own.

For a first attempt, a goal that says more gives you a better idea of what is coming:

```bash
quackd run --goal "If you can see a person in the camera, greet them: move wrist_flex, \
shoulder_pan and elbow_flex back and forth a few times, no more than about 20 degrees from \
where each one is now, in several small moves rather than one big one. Keep any wrist_roll \
move especially small. Do not touch the gripper. Then return to the start and stop." \
  --robot arm-01 --provider openai --max-steps 12
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

Aiming is the part that went wrong on the bench, and it is worth learning from. The webcam was
framed on the gripper, which cropped the raised arm out of the picture, so the model checked
its own waves against the joint angles in `report_state` rather than against anything it could
see. It still waved. It could not watch itself do it. Point the camera at the volume the
arm will move through, not at the end of it, or add a second view as in section 09.

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

### Give it a picture

The camera answers what the room looks like now. It cannot answer what the task is about, and
"draw what is in the picture" is not a sentence a robot's own webcam can be asked. `--image`
hands the task a file instead:

```bash
quackd run --goal "draw what is in the picture" --robot arm-01 \
  --provider openai --image sketch.png
```

The flag repeats, so several pictures can come with one task. **What the pilot receives** is
the picture attached to its first observation and to no other, labelled `task picture
sketch.png:` in front of the image itself and ahead of any camera frame in the same message.
Nothing trims it out of the history afterwards, which is the difference that matters: only the
last two exchanges keep their camera frame, and a task picture is still in front of the model
on the last step of a long run. The system prompt gains a section naming which pictures came
with the task and saying, in as many words, that they are not what the robot can see. The trace
counts them in the request line, so you can tell at a glance that they are still going:

```
llm>    step 0: 1 messages (1 with image, 1 task picture) to fake scripted:goal
```

**Where the copies are kept.** Every picture is re-encoded to PNG on the way in, brought under
1568 pixels on its longest edge and under one and a half megabytes, and written to
`runs/<timestamp>-<name>/images/00-sketch.png` beside the transcript, with a `task_image` line
in `transcript.jsonl` naming the file, the original name and the byte count. The copy is
therefore what the model was actually sent rather than the file you pointed at, which is what
makes it worth keeping. Two pictures with the same basename are numbered apart rather than
overwriting each other.

**A pilot that cannot see refuses the flag rather than dropping the pictures**, before a run
directory exists:

```
✗ error: fake scripted:goal does not take images, so it cannot be given 1 picture
  quackd list-models marks the models that take no frames; --vision overrides it where
the vendor does take them, and a local model needs --vision
```

Silently dropping them is the failure this refusal is here to prevent. A model handed "draw
what is in the picture" with no picture improvises something plausible, and the only trace of
why would be a drawing that has nothing to do with your sketch. The same rule covers the
pilots this page has already met. `--provider fake` never takes images, so the scripted pilot
needs `--vision` before it will accept one, and it still does nothing with it. A local model
needs `--vision`, or `QUACKD_VISION=1` in the environment, and it needs an actual
vision-capable model loaded in the server behind that flag. A cloud model marked `no frames`
in `quackd list-models` needs `--vision` too, and only where the vendor really does take them.

Both starts take a picture, and the second is the one this flag was written for:

```bash
quackd run --goal "draw what is in the picture on the paper in front of you" \
  --robot arm-01 --provider openai --image sketch.png --by-hand
```

Started from the rest pose, the model has the sketch and an empty gripper, and the first
problem it has to solve is getting hold of a pencil nobody gave it. Started
[by hand](#or-start-from-a-pose-you-set-by-hand), you put the pencil in the jaws yourself and
set the tip on the paper, so the model begins with the sketch, a known contact point and
nothing to improvise except the drawing. Neither of those has been tried on a real arm, and the
second one is where a pencil either stays put through a move or does not.

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
quackd validate wave-hello.duck --robot arm-01
```

Note what the allowlist does **not** contain. A task that so much as allows `observe` is
refused on `lerobot:real` for the reason above, so leave it out and let the frame arrive in
the observation.

<br>

## 13. When it will not work

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
| `it has no ?name= and 2 cameras were given` | several `--camera-url` and one of them is unnamed | name every url, `opencv://1?name=top --camera-url opencv://2?name=side`. Nothing was opened |

And once it is running:

| What you see | What it means | What to do |
|---|---|---|
| `is outside this arm's calibrated range` | the goal is outside the travel in your calibration file | working as intended. Aim inside it. On `wrist_roll` this can never fire |
| `reads 61°C: let the arm cool` | the heat gate, below the servo's own 70 °C cut-off | let it cool. A joint that trips its own protection goes slack without announcing it |
| `and it has stopped moving` | a stall: five ticks in which no watched joint moved | something is in the way, or a servo tripped. The arm is held first |
| `the camera gave no frame` | the webcam stalled or was unplugged | the arm carries on, and `report_state` starts saying `CAMERA DOWN:` |
| `the arm's torque is off` | torque reads off | no verb can toggle torque either way. A fresh connect re-enables it, so this points at a tripped servo or the supply |
| the run ends saying the arm did not answer | the heartbeat's round trip failed | the cable, the power, or a tripped servo. The arm holds its last goal under torque. Seen once on 2026-09-15, in a dry run, and not since |
| the arm sags when the run ends | no rest pose is recorded, so torque drops where the arm stands | `quackd robot rest-pose arm-01`, with the arm folded by hand first |
| `the arm is not at its rest pose (...), so torque was left on` | it could not get home: something is in the way, or a servo tripped. The run itself says `the arm did not reach its rest pose` | hold the arm, cut its power, clear whatever stopped it, and run again. It stays energised until you do |

[adapters/lerobot.md](adapters/lerobot.md) has the full table, including the failures SO-101
owners report that nobody here has reproduced.

<br>

## 14. What to report

[Open a LeRobot hardware report](https://github.com/rokbenko/quackd/issues/new?template=lerobot-hardware-report.yml),
or a plain issue with the transcript and your `quackd doctor` output. A report that says it
did not work is worth as much as one that says it did.

One arm has been down this path, so some of these questions have one answer and none of them
have two. The four that nobody has measured at all:

- **Whether the holding band is anywhere near right.** quackd calls it holding when the
  gripper is told to close, settles, and settles between 8 and 90 of 100. Nothing was held on
  the bench, so the band has never been tested against an object.
- **What a joint reads in degrees Celsius**, cold and after ten minutes of work. Both the 60
  refusal and the 70 cut-off are Feetech's documentation rather than anything measured here,
  and the bench run was too short to warm anything up.
- **Whether a stall is caught on purpose.** Hold a joint gently against its goal and see
  whether the verb fails with where it stopped. Nobody has deliberately tried it.
- **Whether five degrees an action felt right** in the room. One person has watched this arm
  move, and they did not write down an opinion on the speed.

And the two the bench answered once, where a second answer is what turns one arm's behaviour
into something true of the SO-101:

- **Which end of the gripper's 0..100 range is open.** On the bench, 100 is open: commanded
  100 it reported 98, and closed it settled at 3. Everything quackd believes about holding
  rests on this being the same on your arm.
- **Which model you used, and whether it could tell you were there.** `gpt-6-astra` waved, and
  it verified its own waves from joint readings rather than from the picture, because the
  camera was framed on the gripper. Whether a model can actually see you, on a camera aimed
  properly, is still open.

Two more arrived with [`--by-hand`](#or-start-from-a-pose-you-set-by-hand), and both of them
ask what a servo does rather than what quackd does, so neither the mock arm nor the test suite
can answer either:

- **Whether the arm stays where you put it when torque comes back on.** quackd writes the
  position it just measured as the goal before it enables torque, because nothing upstream
  documents what one of these servos does with the goal it was last told when it is
  re-energised, and the goal it was last told is the fold. It then writes the goal again,
  reads the arm back, and refuses if any joint has moved more than five degrees. Nobody knows
  whether that refusal ever fires on a real arm, or how far a loaded forearm sags in the
  moment it takes its own weight back. Say which joint moved and by how much.
- **Whether a hand-closed gripper keeps a pencil through a drawing move.** The jaws are left
  exactly where your fingers closed them and are never squeezed tighter, so whatever holds the
  object is the friction at the width you left. Whether that survives the arm actually moving,
  or whether the pilot has to close the gripper properly on it first with the `gripper` verb,
  is the difference between `--by-hand` being useful for a task with a tool in it and being a
  way to set a starting shape and nothing more.

The webcam question is closed enough to stop asking: the plugged-in camera was `opencv://1`
and later `opencv://2` at 640x480, and it needed no `?backend=` key on Windows. Say so anyway
if yours needed one, because that is the interesting case now.

If your arm does something this one did not, change the row in
[adapter-status.md](adapter-status.md) and say in the same commit what it did. One SO-101 is
one SO-101: six of quackd's seven bodies have still never been near hardware of any kind, so
most of that page is a description rather than a record.

<br>

## Appendix: driving it from an MCP client

Everything above is the command line, because that works with every pilot quackd supports.
There is a second way in, and it is the only way to run exactly one verb and stop, or to ask
for a camera frame as a deliberate act.

`quackd serve-mcp` exposes a robot as [Model Context Protocol](https://modelcontextprotocol.io)
tools over stdio, so the client spawns it as a subprocess and the client's own model is the
pilot. quackd chooses no model in this mode and reads no key.

```bash
quackd serve-mcp --robot arm-01
```

A session does the same thing with the rest pose that a run does, at both ends: it puts the
arm at the pose before the client's model gets a verb, and refuses to start at all if it
cannot get there.

The tools are `robot_list_verbs`, `robot_observe`, `robot_assess_task`, `robot_run_verb` and
the rest. `robot_run_verb` refuses anything other than `report_state`, `observe` and `stop`
until a verdict has been recorded, and `robot_observe` returns the frame itself, which is
what makes a deliberate look possible here and not on the CLI. A `.duck` loaded with
`robot_load_duckfile` is checked against the manifest of the robot **already connected**, so
a task that allows `observe` loads cleanly on a session started with a camera.

[mcp.md](mcp.md) has the client configuration, the full tool list and what the trace shows.
Two clients are documented and both are Anthropic's; the transport itself is a plain local
stdio server and nothing in it is specific to them.
