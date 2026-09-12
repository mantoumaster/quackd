# FAQ

**Which simulator should I use?** Both ship, and the cartoon is still the default. `sim2d`
starts in a second, needs no network, runs anywhere, and is what the three other bodies that have a simulator and
every CI sweep use. It tests the *agent loop* — search, approach, act, verify — and it will
never tell you whether a gait works, because it has no joints
([ADR-0007](adr/0007-sim2d-cartoon.md)). `--robot microduck:mujoco` is upstream's own Microduck
model in MuJoCo, walking on `alpha_walking.onnx`, the policy Pollen trained, at 50 Hz on the
CPU. The ball rolls, the duck undershoots what you asked for, and a pilot that works there has
met a robot that does not do what it is told. Same arena, same seeded layout for the duck and
the ball, same verbs, so a `.duck` written for one runs on the other
([ADR-0030](adr/0030-mujoco-physics-backend.md)). One exception, and it is the only one: the
cartoon stands a person in its arena and the physics world does not, so `follow-me`, whose
whole task is to follow somebody, is cartoon only.
Neither one installed? [`web/`](../web/README.md) is the same physics and the same two policies
in a page, and it does one thing neither Python simulator does: the sentence box and the
keyboard drive the same duck at the same time.

**How do I run the physics simulator, and what does it download?**
`uvx --from "quackd[mujoco]" quackd run find-and-kick --robot microduck:mujoco --provider fake`.
The first run fetches upstream's model, `robot_walk.xml` and 38 STL meshes, from `microduck_rl`
at a pinned commit, and `alpha_walking.onnx`, `alpha_stand.onnx` and their manifest from the
Hugging Face Hub at a pinned revision, into `~/.quackd/cache`. About 10 MB over the wire and 23
on disk. Every file is checked against a sha256 recorded when it was read, a run that gets a
different file fails rather than continues, and the licence notice is written beside them,
because the model files are CC BY-NC-SA and quackd ships none of them ([licenses.md](licenses.md)).
`QUACKD_MICRODUCK_ASSETS` points at a `microduck_rl` checkout of your own instead,
`QUACKD_CACHE_DIR` moves the cache, and `QUACKD_MUJOCO_BODY=puppet` runs a kinematic stand-in
that downloads nothing and is the body the tests build directly.

**How do I run the browser demo, and is it live anywhere?** `python web/serve.py`, then open
<http://localhost:8000/simulator/>. Nothing to build, and no quackd to install: that server is
one stdlib file. It still needs a server, for two reasons: browsers refuse ES modules over
`file://`, and the page expects to be mounted at `/simulator`, so every local reference in
`index.html` is absolute. That is why a plain `python -m http.server --directory web` no longer
works — it serves the HTML and then 404s the stylesheet and the script, because nothing answers
on `/simulator` at the root. `web/serve.py` takes an optional port. The page is live at
<https://www.quackd.org/simulator>, where the separate quackd-web project fetches this directory
into its own build at a pinned commit; `/simulator/source.json` says which one. The browser
fetches
about 45 MB the first time and caches it: MuJoCo's WebAssembly build, onnxruntime-web and
three.js from jsDelivr, upstream's model at the same pinned commit Python uses, and
`alpha_walking.onnx` and `alpha_stand.onnx` — the same two policies Python fetches, with the
kick a scripted impulse in both. It is more megabytes than the answer above because the browser
pulls `robot_walk.xml` and 38 separate meshes uncompressed, 22 MB of it, where Python pulls one
compressed archive. Nothing fetched is hash checked, which Python does and the page admits
([web/README.md](../web/README.md)).

**Can I drive the browser demo myself?** Yes, at the same time as the model, which is the
argument the page exists to make. The keyboard is never handed over because it is never taken
away: `W`/`S` walk, `A`/`D` turn, `Shift` with `A`/`D` strafes, `Q`/`E` look, `G` centres the
head, `Space` stops, `K` kicks, `R` stands the duck up, `O` prints the raw state the model's
observation is built from, `1`/`2` change camera and `Esc` hands the keyboard back to the arena.
A key that would *move* the robot takes it mid-run — the run is aborted and the request to the
model is aborted with it, so nothing keeps running against your key and no answer arrives after
you took the duck back, and the transcript records the handover with the key that did it. A key
that only reads (`O`, the camera keys) never interrupts. There is no key
for `say`, deliberately: a key carries a command, and a sentence needs something to read it.
The `quackd is on` switch decides only that last part — whether anything reads English — and no
longer decides whether you may drive.

**Why does the duck in the physics simulator not go the speed I asked for?** Because the walking
policy has a floor and quackd will not hide it. Under the model's own actuators the gait does
not start below about 0.22 m/s or 1.0 rad/s, and above that it achieves roughly 0.42 of what it
is asked. Both were measured here on one machine and are tagged UNVERIFIED in
`quackd/sim3d/upstream_api.py`. `move` defaults to 0.15 m/s, so a non-zero twist below the floor
is scaled up bodily, keeping the ratio between its axes so an arc stays an arc, and a twist
below a third of the floor is dropped to zero rather than turned into a lurch nobody asked for.
What was asked and what was sent are both in the state (`twist_commanded`, `twist_sent`,
`gait_floor`) and in the prompt. Upstream trains and deploys with a different actuator model, so
a real Microduck may track commands directly.

**Do I need a GPU for the physics simulator?** No. MuJoCo steps on the CPU and the policy runs
under onnxruntime's CPU provider. Upstream needs CUDA to *train* that policy, never to run it.
What the head camera needs is an OpenGL context to render into: a laptop has one, a bare server
may not, and the frames are what fails first there. On a headless Linux box, install `libosmesa6`
and set `MUJOCO_GL=osmesa`, which renders into process memory and needs no display, no GPU and no
`/dev/dri` — it is what CI's own physics job does. quackd names both in the error rather than
letting an OpenGL traceback out. Rendering is this backend's real cost, not physics.

**Does `uvx quackd run … --provider anthropic` work with no extras?** The default install
is light on purpose (no vendor SDKs). Use `uvx --from "quackd[anthropic]" quackd run …`, or
`uv pip install "quackd[anthropic]"`. Without the extra, quackd prints exactly that command.
`--provider fake` needs nothing.

**Which models are the defaults?** `anthropic` → `claude-opus-5`; `openai` → `gpt-5`;
`gemini` → `gemini-2.5-pro`; `grok` → `grok-4`. The non-Anthropic IDs were not verified
against vendor docs at release (no keys in CI) — override with `--model` or `QUACKD_MODEL`
if yours differs. Anthropic extras: `QUACKD_EFFORT` (default `medium`) and
`QUACKD_ANTHROPIC_FALLBACKS=0` to disable server-side refusal fallbacks.

**Are local LLMs supported (llama.cpp, vLLM, Ollama, LM Studio)?** Yes. They all speak
OpenAI's Chat Completions API, so `--provider ollama`, `vllm`, `llamacpp`, `lmstudio`, or
`local --base-url http://host:port/v1` works with no API key. Tool calling must be enabled
on the server (`llama-server --jinja`, `vllm serve --enable-auto-tool-choice
--tool-call-parser …`), vision is off unless you pass `--vision`, and a small model that
writes its tool call as plain JSON is still understood. Details: [local-llms.md](local-llms.md).

**How does the LLM "see"?** Providers with vision get the duck-cam PNG for the last two
turns; every provider gets a text line like `ball at bearing 12° left, ~0.80 m` from the
detector. Composite verbs steer on detections at 10 Hz and never wait for the model.

**How do I see what the model was told, what it thought, and what it sent the robot?**
You already do: the trace is on by default. `quackd run` narrates the whole run to stderr as
it happens, and every MCP tool call that reaches a robot comes back with a `trace` list of
the same lines. You get the system prompt once, then per turn the observation, the model's
reasoning where the provider returns any, the tool it chose, the tokens and latency, every
executor gate that fired, every intent that went to the robot (a steering loop's burst
collapsed into one line with its parameter ranges), and the result. `--no-trace` or
`QUACKD_TRACE=0` turns the views off; `runs/<ts>/transcript.jsonl` keeps everything either
way, uncapped. Details and the event list: [architecture.md](architecture.md#trace),
[ADR-0029](adr/0029-tracing.md).

**Can I read a run after it finished?** Yes. `quackd trace` replays the newest run under
`runs/` as the same lines it printed while it ran, and it takes a run name, a timestamp
prefix, a duck name or a transcript file if you want an older one. `--from-step N` starts
part way in, `--no-prompt` drops the system prompt, `--thinking all` shows every character
the model thought, and `--frames` adds a line per camera frame. It prints to stdout, so piping
it to a pager or a file is the point. A flock run replays every member, each line prefixed
with the robot that wrote it.

**Why is the thinking line empty for my model?** Because that model did not return any. Only
some do, and each in its own way: Claude returns a summary (quackd asks for one, since the
default is to send the blocks back empty), an OpenAI-compatible server may fill
`reasoning_content` or `reasoning`, Gemini returns thought parts when asked, and a local
server that separates nothing gets its `<think>` block split out of the answer. OpenAI's own
Chat Completions returns a reasoning token count and no text, so that is what the trace
shows. The scripted pilot has no reasoning either, but it does report which rule it followed
— what it saw, how the last verb ended, and the verb that fell out — on the same line, marked
`[scripted]` so it can never be mistaken for a model's own words. So a run with no API key
still shows you the shape of the trace.

**Does the robot need a powerful onboard computer?** No. quackd's own process, the part
that calls the LLM and runs the detector, never runs on the robot itself — you run
`quackd run` on a laptop or a server, a network hop away, and it talks to the robot (or the
simulator) from there. The Open Duck Mini's official target, a Raspberry Pi Zero 2 W, only
ever runs its existing 50 Hz walk policy plus two small daemons that do run on the Pi, a
bridge and a camera server, and neither does any perception or inference of its own: just
enough to swap the gamepad for a socket and to serve a JPEG ([`bridge/open_duck/`](../bridge/open_duck/README.md)). The Microduck's onboard
computer works the same way, through `robotd`. Nothing here needs an NPU or a bigger board
to keep up, because nothing model-shaped runs on the robot's own board in the first place.

**Does quackd use TOF or another depth sensor for obstacle avoidance?** Not yet. The only
sensing input today is a single colour camera: an HSV threshold (or optionally YOLO) gives
a bearing and an apparent-size distance to one named target, and `go_to` steers toward it —
there's no depth data, no occupancy grid and no general obstacle avoidance. The manifest
schema has a generic `tof` sensor slot for future adapters
([manifest-spec.md](manifest-spec.md)), and the Microduck's own `tofd` depth stream isn't
read yet either ([adapter-status.md](adapter-status.md)); the Open Duck Mini's official
build has no depth sensor at all.

**How do I tune the detector for a real orange ball?** `ColorBlobDetector` takes
`targets=(Target("ball", HSVRange(h_lo, h_hi, s_lo, v_lo), size_m=radius, round=True), …)`
in OpenCV HSV (H 0–180). Photograph the ball under your light, sample its hue, give ±8, and
pass `--fov-deg 62` to `quackd run` for the IMX219. Distance comes from apparent size:
measure the pixel radius at 1 m once and adjust `size_m` until it reads 1.00. Or install
`quackd[yolo]` and use `YoloDetector`.

**Does it remember anything between runs?** Since 0.6, a little, per robot. Each
`adapter:backend` has a JSONL file under `~/.quackd/memory/` holding two kinds of line: the
notes the pilot chose to keep with the `remember` tool, and one line per earlier run that
quackd writes itself (outcome, reason, the last few verb results). The newest twenty notes
and five episodes go into the next run's system prompt. It is deliberately not a memory
*system*: no embeddings, no search, newest wins, nothing shared between bodies, and the
executor never reads it, so a note can never widen an allowlist or lift a budget. The
scripted pilot has no `remember` in its script, so `--provider fake` accumulates run
outcomes and never a note. [memory.md](memory.md), [ADR-0025](adr/0025-memory-between-runs.md)

**Who decides the run succeeded?** The LLM, via `declare_success(reason)` — that is the
honest state of the art. In either simulator the run summary also records ground truth
(`ball_displacement_m`) and the tests check the claim against it.

**Why can't the duck say words?** Upstream has seven duck sounds and no TTS. `quack(text)`
maps your text to the closest tone (`greet`, `inquire`, `alarm`, `wheee`, …) and logs the
text.

**What does it cost?** A `find-and-kick` run is 3–8 model turns, each a few thousand input
tokens (mostly the system prompt and one image) and a short tool call. `transcript.jsonl`
records usage per turn.

**Windows?** Fully supported for sim, MCP and development. The real-robot `unix://` socket
is POSIX-only; forward it with `ssh -L 9870:/run/robotd.sock <robot>` and use
`--address tcp://127.0.0.1:9870`.

**Can I run it on my Microduck today?** `--robot microduck:jsonrpc` speaks the verified
`duck-ipc-proto` v23 vocabulary but has never touched hardware. Start with `--dry-run`,
read [adapter-status.md](adapter-status.md), and tell us what happened.

**Can I drive it from the Claude mobile app?** Not yet. `quackd serve-mcp` speaks `stdio`
only, so Claude Code and Claude Desktop spawn it as a local subprocess on the same machine
and talk to it over pipes. The mobile app reaches tools as remote connectors instead:
servers that run persistently at a network address with their own auth. quackd would need
an HTTP or SSE transport, a long-lived process, a reachable address and authentication
before a phone could talk to it. Roadmap, not shipped. The details are in
[mcp.md](mcp.md#why-not-from-my-phone-yet).

**Is control text-only?** Yes, from you — a `--goal` string, a `.duck` file, or a chat
message over MCP; there's no voice or GUI input. The loop isn't text-only end to end,
though: cloud providers also read the camera frame as an image each turn, and whatever
the model decides is always one of a fixed set of verbs (`walk_to`, `kick`, `quack`, …),
never a freeform command sent to the motors.

**Do I need to be near the robot to control it?** No, proximity is not the constraint.
`robotd`'s socket only accepts connections from the robot's own computer, so control always
goes through a network hop (see `Windows?` above), and that hop works the same across the room
or across the world. What matters is latency: the deadman expects `robot.move` roughly every
100 ms, so a slow or flaky link stops the robot outright, however close you are standing.

**Is quackd production-ready?** No — it's a research prototype built around one trusted
local operator, not a hardened multi-user product. There's almost no authentication anywhere;
`.duck` files with a `flock:` block are refused over MCP for exactly that reason ("one
pilot, a flock needs a coordinator"), and nothing arbitrates two sessions driving the same
robot at once. Every real-hardware transport is experimental and unverified end to end
([adapter-status.md](adapter-status.md)); the CLI and MCP server are both thin callers of
the same executor and verb registry, so a real client like a phone app would mean adding a
network-reachable server and auth on top, not rewriting the core.

**Can I control who's allowed to pilot my robot?** Barely, and only where quackd ships the
robot side. quackd
adds no login or accounts, so access is mostly whatever your OS and network enforce.
`robotd`'s socket can't be reached off the robot's own computer unless something bridges
it, so the real gate there is SSH's authentication (and your Wi-Fi's), not quackd's;
`quackd announce`/`discover` do broadcast a robot's identity, unauthenticated, to anyone on
the LAN ([lan.md](lan.md)), though that's identity only, not a way to drive it. The two
exceptions are the daemons quackd itself ships, for the Open Duck Mini and the ToddlerBot:
each binds loopback, and if a token is configured it checks one with `hmac.compare_digest`
before accepting a
handshake (`--token`, or `QUACKD_DUCK_TOKEN` for the duck and `QUACKD_TODDLERBOT_TOKEN` for the
ToddlerBot). The duck's camera server has no authentication at
all, so tunnel it. On a Microduck the physical gamepad preempts remote commands; on an Open
Duck it does not, because quackd's daemon *replaces* the gamepad the walk loop reads, which
makes the power switch the only thing that always wins ([safety.md](safety.md)).

**What stops the model itself from doing something dangerous?** The executor, not the
model's judgment: every verb call is checked against the loaded `.duck`'s allowlist,
budgets and confirm gates before anything is sent, and machine-enforced `abort_when` rules
and preconditions (not fallen, not sitting) run right after — a refusal is enforced code,
not a request the model can talk its way around. That's still only the software layer, and
what the body adds under it varies: the Microduck's `robotd` has fall detection, thermal
clamps and a deadman, while an Open Duck Mini v2 declares `none` and the watching human is
its fall detector — see [safety.md](safety.md).

**Does my data ever leave my machine?** Only if you choose a cloud provider. Claude,
OpenAI, Gemini and Grok each send the camera frame and prompt to that provider's API over
the network, under its own terms; the `fake` pilot and any local model (`ollama`, `vllm`,
`llamacpp`, `lmstudio`) never do and need no API key, though a local model is still served
over its own local HTTP endpoint, not literally air-gapped. Since 0.6 one thing also stays
behind on your machine: `~/.quackd/memory/<robot>.jsonl`, a plain text file of sentences
about where things are, which is then part of the prompt on the next run and so part of what
a cloud provider sees. `quackd memory show` prints it, `quackd memory clear` deletes it, and
`--no-memory` never writes it. See [memory.md](memory.md) and
[local-llms.md](local-llms.md).

**Can quackd drive something that is not a duck?** Since 0.4, yes: a robot is an adapter
that returns a manifest, and the verbs come from the manifest. `quackd list-adapters`
shows the seven that ship (Microduck, a LeRobot arm, any base over rosbridge, an Open
Duck Mini v2, an XLeRobot dual-arm cart, an AlohaMini with two arms on a lift, and a
ToddlerBot humanoid), `quackd list-verbs --robot microduck:sim2d` shows what one of them can do,
and `quackd validate your.duck --robot lerobot:mock` tells you, field by field, whether
your task fits that body. The rule never bends: a verb that is not in the manifest does
not exist on that robot. Writing one: [adapters.md](adapters.md).

**Why does `validate` say "requires kick, but arm-01 (lerobot-so101) does not provide
it"?** Because it is true. A `.duck` lists what it needs (`requires`, or for a `duck: 0`
file its whole allowlist) and an arm has no legs. Either pick a body that has the verb,
or write a task for the body you have.

**Can two different robots share a task?** In the simulator, `flock.roles` lets a `.duck`
declare capability-differentiated roles instead of one shared verb set — a spotter that
observes and judges, a kicker that goes to and kicks — but two Microducks are the only
pairing quackd ships an adapter for today, and no bundled starter exercises the split end
to end yet ([flock.md](flock.md)). On hardware, not yet: a flock across machines needs a
clock across machines, and nothing multi-robot has run on hardware.

**Why "quackd"?** Upstream names its daemons `robotd`, `mediad`, `padd`, `tofd`… the brain
daemon was missing. ([ADR-0002](adr/0002-name.md))
