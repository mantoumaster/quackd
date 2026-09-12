# Architecture

quackd is the brain daemon Microduck was missing, and a brain for any small robot
that has an adapter. This page is the map; the ADRs in [`adr/`](adr/) are the reasons.

## Three loops

| Loop | Rate | Where | Owner |
|---|---|---|---|
| Reflexes | the body's own (50 Hz on both ducks) | below quackd: `robotd` on a Microduck, quackd's own bridge daemon on an Open Duck, the position controller on an arm, the driver on a base | the robot's own controllers: RL policies (ONNX) for balance and gait on both ducks and for stand-up on the Microduck alone (a fallen Open Duck Mini needs a human), a learned pick policy on the arm when one is loaded. quackd writes none of this control code. It does *host* the loop on two bodies, both of which have no network API to talk to. On the Open Duck Mini it supplies only the seven numbers a gamepad would ([ADR-0024](adr/0024-open-duck-mini.md)). On the ToddlerBot it owns the fifty hertz loop outright, because that robot's `step()` is a no-op and a humanoid frozen mid-stride while a model thinks is a humanoid on the floor ([ADR-0028](adr/0028-toddlerbot.md)). Even there quackd writes no gait: the walk checkpoint is the robot's own. |
| Steering | 5–20 Hz | quackd process | perception + composite verbs. `go_to` (alias `walk_to`) closes the approach loop on detections. |
| Deliberation | ~0.2–1 Hz | LLM | reads frame summary + state + last result, picks one **verb**, judges success. |

Every design choice defends this separation: the LLM's only output is one tool call per
turn; composites never call the LLM; verbs send *intents*, never joint targets; the
adapter owns time, so the steering loop runs at sim speed in the simulator and in real
time on hardware without changing verb code. ([ADR-0003](adr/0003-three-loops.md))

One backend puts that first loop inside quackd's own process. `microduck:mujoco` steps
upstream's `alpha_walking.onnx` at the robot's own 50 Hz next to the agent loop, because a
simulated duck has no onboard computer to run it on. Nothing else changes: the transport
sends a twist and a head pose, quackd still writes no gait, and the boundary that is a
network hop on hardware is a function call here ([ADR-0030](adr/0030-mujoco-physics-backend.md)).

None of this needs to run on the robot's own computer. The only quackd code that ever runs
on a robot lives in `bridge/` (see Modules below): the Open Duck Mini's pair of daemons, a
bridge and a camera server; a host wrapper for the AlohaMini; and a daemon for the
ToddlerBot. None of it carries model or perception code of its own.

Since 0.4 the robot side is an **adapter** that declares a **manifest**: what body it has,
which intents and sensors, which verbs. The registry, the tool list, the allowlist universe
and the system prompt are all built from that manifest at connect time; a verb that is not
in it does not exist. The Microduck is the first adapter, wrapping the five transports
below unchanged. ([ADR-0017](adr/0017-robot-adapters-and-manifest.md),
[design/multi-robot.md](design/multi-robot.md))

```mermaid
sequenceDiagram
    participant L as LLM
    participant A as agent loop
    participant E as executor
    participant V as verb
    participant T as adapter
    participant P as perception
    A->>T: get_state / get_frame
    T-->>P: frame
    P-->>A: detections ("ball at bearing 12° left, ~0.8 m")
    A->>L: observation (text + image) + tool list
    L-->>A: exactly one tool call (e.g. go_to)
    A->>E: run_verb("go_to", params)
    E->>E: allowlist · confirm · budget · abort_when · preconditions · dry-run
    E->>V: execute(ctx, params) with timeout
    loop 10 Hz steering
        V->>T: get_frame → detect → send_intent(move)
    end
    V-->>E: VerbResult(ok, summary, data)
    E-->>A: result (→ transcript)
    A->>L: next observation
```

## Modules

| Path | Why it exists |
|---|---|
| `quackd/cli.py` | The front door: `run · validate · doctor · serve-mcp · list-verbs · list-adapters · record · trace · memory · discover · announce`. `--robot <adapter>:<backend>` everywhere, with `--address`, `--camera-url` and `--token` for a real robot. |
| `quackd/duckfile/` | The `.duck` contract (v0 and v1): strict pydantic frontmatter, parser, generated `schema.json`, `validate.py` (a task against one or more manifests). |
| `quackd/adapters/` | `RobotManifest` (data: what a robot is and can do), the `RobotAdapter` protocol, the factory behind `--robot`, and one package per robot: `microduck/` wraps the five transports and declares its manifest and extension verbs; `lerobot/` is a desktop arm (`mock`, `real`, [adapters/lerobot.md](adapters/lerobot.md)); `rosbridge/` is any wheeled base over rosbridge (`mock`, `ws`, [adapters/rosbridge.md](adapters/rosbridge.md)); `open_duck/` is an Open Duck Mini v2 (`sim2d`, `mock`, `bridge`, [adapters/open_duck.md](adapters/open_duck.md)), the first body whose robot side quackd also ships, in `bridge/open_duck/`, because its runtime has no network control API; `xlerobot/` is a dual-arm mobile manipulator (`mock`, `zmq`, [adapters/xlerobot.md](adapters/xlerobot.md)), the first body with both a base and arms, and the one quackd talks to by speaking its ZeroMQ host protocol rather than importing it, because upstream is not an installable package; `alohamini/` is two arms on a lift on a wheeled base (`mock`, `sim2d`, `zmq`, [adapters/alohamini.md](adapters/alohamini.md)), which quackd also reaches by speaking its ZeroMQ host protocol; `toddlerbot/` is a small humanoid (`mock`, `sim2d`, `bridge`, [adapters/toddlerbot.md](adapters/toddlerbot.md)), the third body whose robot side quackd ships, because upstream has no network API at all. Every SDK-touching package owns an `upstream_api.py` and a containment test. |
| `quackd/verbs/` | `core.py`: the verbs any robot can carry and what each requires; `aliases.py`: the one alias table; `registry.py`: built from a manifest at connect time; `learned.py`: the v2 interface. |
| `quackd/safety.py` | The layer that does not trust the LLM: `Executor`, `Budget`, `Heartbeat`, `KillSwitch`. Preconditions arrive from the adapter; the executor spells none. |
| `quackd/transport/` | The Microduck backend layer: the `DuckTransport` protocol; `sim2d`, `mujoco` (physics, needs `quackd[mujoco]`), `mock`, `jsonrpc` (experimental), `websocket` (stub); `upstream_api.py` is the only file allowed to spell a Microduck upstream method. |
| `quackd/sim2d/` | The cartoon world, two renders (top-down, duck-cam), the GIF recorder, the optional live window. |
| `quackd/sim3d/` | The physics world: the cartoon's arena minus its person, plus its seeds, deadman, kick cone and scoop, in MuJoCo. `world.py` steps a `Body`, and two exist, a kinematic puppet that needs no download and upstream's own Microduck model walking on upstream's own `alpha_walking.onnx` at 50 Hz. `assets.py` fetches the model and the policies at a pinned commit into `~/.quackd/cache` and checks every file against a recorded sha256; `upstream_api.py` is the only file allowed to spell a `microduck_rl` name ([ADR-0030](adr/0030-mujoco-physics-backend.md)). |
| `quackd/perception/` | `Detection` + `Detector`; the HSV colour-blob default; the lazy YOLO extra. |
| `quackd/agent/` | The loop, the prompts, the transcript, and one provider per vendor behind `LLMProvider`. |
| `quackd/trace.py` | The run narrating itself: `TraceEvent`, the `Tracer` that fans out to the transcript and to any number of views, the transport wrapper that turns every intent into an event, and the renderer both surfaces share ([ADR-0029](adr/0029-tracing.md)). |
| `quackd/memory.py` | What a robot keeps between runs: one JSONL file per `adapter:backend` with the notes the pilot saved (`remember`) and an episode per run; rendered into the prompt next time ([memory.md](memory.md), ADR-0025). |
| `quackd/mcp_server.py` | A robot, or a fleet (`--robots`), as MCP tools: eight `robot_*` tools through one executor per robot. |
| `bridge/toddlerbot/` | quackd's own ToddlerBot daemon: the fifty hertz loop upstream has no daemon for, plus the ten things it does not do at all, enumerated in the daemon's own docstring and in `bridge/toddlerbot/README.md` rather than a third time here. It owns the control loop rather than feeding one, which is true of no other body quackd drives. Standard library plus numpy, never imported by quackd, shipped in the sdist and never in the wheel ([ADR-0028](adr/0028-toddlerbot.md)). |
| `bridge/alohamini/` | quackd's own AlohaMini host: upstream's host loop with the arm torque its own `configure()` disables and never re-enables, plus three fields in every observation so quackd can tell this host from a stock one. Never imports quackd, ships in the sdist and never in the wheel ([ADR-0027](adr/0027-alohamini.md)). |
| `bridge/open_duck/` | **The first robot side quackd shipped**, and one of the three above. It has still never run on a duck, like everything else here. Two daemons for an Open Duck Mini v2's Raspberry Pi: the bridge, which is upstream's own walk loop with the gamepad it reads replaced by a socket, and the camera server, which serves one JPEG over HTTP. Standard library plus numpy, never imported by quackd, shipped in the sdist and never in the wheel ([ADR-0024](adr/0024-open-duck-mini.md)). |
| `web/` | The same loop in a browser, and the only quackd code that is not Python: MuJoCo compiled to WebAssembly, the same two policies (`alpha_walking`, `alpha_stand`) in onnxruntime-web, seven of the same verbs under the same allowlist-and-budget machinery, with the model and the policies fetched from the same pinned upstreams. The kick there is quackd's own scripted impulse, as it is in `sim3d`. What the Python loop has no equivalent of is the second pair of hands: the sentence box and the keyboard are both live at once, so `runtime.manual` is a lease on the twist rather than a mode, and a key that would *move* the robot takes it mid-run while a key that only reads does not. Mounted at `/simulator`, which is why `web/serve.py` — stdlib, and the one piece of Python in `web/` — runs it locally rather than `http.server`. Live at <https://www.quackd.org/simulator>, which the separate quackd-web project builds from this directory. It shares no code with the package, so it is kept in step by hand and `tests/test_web.py` holds the parts that can be checked from Python, printing what to paste when they drift. The mechanism, the key map and where it diverges from `sim3d` are in [`web/README.md`](../web/README.md) rather than a second time here ([ADR-0030](adr/0030-mujoco-physics-backend.md)). |
| `quackd/lan/` | LAN discovery over zeroconf (`_quackd._tcp.local.`): a pure TXT wire format, `announce`, `discover`; behind `quackd[lan]` ([lan.md](lan.md)). |
| `quackd/flock/` | Many robots on one task: the in-process `Bus`, the typed messages, the Contract Net `Auction` and the role auction, the deterministic coordinator, the scripted member FSM, the one-call planner and the runner that judges from ground truth ([flock.md](flock.md)). |
| `quackd/flock/mqtt_bus.py` | The flock `Bus` protocol over an MQTT broker, library only; the in-process bus stays the default. |
| `quackd/doctor.py` | What can run here and what we are assuming about the robot. |

## A turn, concretely

1. **Observe.** `transport.get_state()` → `DuckState`; `transport.get_frame()` → PIL image →
   `detector.detect()` → `[Detection]`. The frame is saved to `runs/<ts>/frames/`.
2. **Think.** The provider gets: the system prompt (contract in prose + the `.duck` body),
   the vendor-neutral history (`Exchange` = observation + decision), and the tool list
   (allowed verbs' JSON schemas + `declare_success` / `declare_failure`, plus `remember` when
   memory is on). With memory on the prompt also carries what this robot remembers from
   earlier runs. Only the last two observations keep their images. The provider must return
   one tool call.
3. **Enforce.** Zero tool calls → one re-prompt, then failure. Several → the first. Then
   `Executor.run_verb`: abort flag → allowlist → params → confirm → budget → machine-enforced
   `abort_when` → preconditions → dry-run → execute, racing the timeout against the abort.
   `stop` is exempt from the abort gate, so the brake still works after one.
4. **Act.** The verb runs; composites loop on the camera at 10 Hz; `move` re-sends its
   velocity every 100 ms to feed the robot's deadman.
5. **Record.** Every step above is a `TraceEvent`, and `transcript.jsonl` is the sink that
   never turns off (every kind it writes is in the table below); `summary.json` at the end; `run.gif` from the recorder in either simulator. With
   memory on, the run ends by appending one episode line to the robot's memory file
   ([memory.md](memory.md)). The terminal and the MCP tool results are views of the same
   stream (see [Trace](#trace)).

Step 0, before all of that: the loop calls `connect()` and, when an adapter answers with a
manifest, builds the registry from it (`registry_from_manifest`). A bare transport answers
`None` and gets the Microduck vocabulary.

Outcomes: `success` / `failure` (the LLM's claim via the meta tools), `budget`, `aborted`
(heartbeat, kill switch, `abort_when`) and `error`, which nobody chose: a provider that
failed, a transport that died mid-observation, a bug. In sim the run summary also carries ground truth
(`final_state.extras.ball_displacement_m`) so tests judge the claim.

## Transcript format

One JSON object per line: `{"t": seconds, "kind": ..., ...}`.

| Kind | What it records |
|---|---|
| `run_start` | contract, system prompt, tool names, robot manifest, how long connecting took |
| `observation` | what the model was shown this turn, and how long gathering it took |
| `llm_request` | how many messages went out, how many still carry an image, whether this is the re-prompt |
| `llm` | text, `thinking`, tool_calls, usage (this turn and the run's total), stop_reason, latency, or `error` when the call failed |
| `enforce` | zero tool calls (re-prompt) or several (first only) |
| `verb_start` | name as called, canonical name, params, source (`agent` · `mcp` · `cli`), whether it is nested inside a composite |
| `gate` | one per executor rule that fired: `abort` · `allowlist` · `unknown` · `params` · `confirm` · `budget` · `abort_when` · `precondition` · `dry_run` · `cancelled`, with the reason and, where it matters, the robot state that caused it |
| `intent` | every command sent to the robot: kind, params, whether it was accepted, and the robot's own clock when it has one |
| `verb_end` | outcome (`ok` · `fail` · `refused` · `denied` · `budget` · `aborted` · `preempted` · `error`), summary, wall seconds, the robot's own seconds on a simulator, and how many intents of each kind it sent |
| `verb` | the loop's own record of the call it made (name, params, ok, summary, data) |
| `declare`, `memory`, `note`, `frame`, `run_end` | the model's verdict, a saved note, a free-text line, a captured frame, the summary (with `trace_dropped`: events a view raised on and never showed) |

Example: [`assets/transcript-example.jsonl`](assets/transcript-example.jsonl), recorded
before the trace kinds existed.

## Trace

The transcript is one *sink* of an event stream, not a thing the loop writes directly
([ADR-0029](adr/0029-tracing.md)). The same events drive three live views, all on by default:

- **The terminal** (`quackd run`), on stderr, so `2> trace.log` keeps the outcome on screen.
  It shows the system prompt once, then per turn: the observation, what the model thought,
  the tool it called, tokens and latency, each gate that fired, each intent, and the result.
  A burst of intents from a steering loop is one line with its parameter ranges, because
  `go_to` recomputes its twist every 100 ms: `-> move x26 over 2.5 s (vx 0.1..0.2, vy 0, wz -0.01..0.88)`.
  A burst still going after two seconds is flushed as it stands and the next line continues
  it, so a long approach narrates itself instead of printing nothing until it ends.
- **The MCP tool result**, as a `trace` list on every call that reaches an executor, capped
  at thirty lines, with the uncapped version on the server's stderr. Over MCP the pilot is
  the client, so its reasoning and its token counts are not quackd's to show. What quackd can
  see it says: the verb, the gates, the intents, the result and the budget.
- **A flock's terminal**, one view per member with its name on every line and the
  coordinator's decisions under `flock`. Each robot's own transcript is its record.

`quackd trace` replays a finished run from its transcript afterwards, through the same
renderer, on stdout.

`--no-trace` or `QUACKD_TRACE=0` removes the views. The transcript is unaffected, because a
run that cannot be argued about afterwards is the thing this project cannot give up.
`--no-trace-prompt` or `QUACKD_TRACE_PROMPT=0` keeps the narration and drops the system
prompt, which is forty to seventy lines and worth reading once.
`QUACKD_TRACE_THINKING` is how much of the model's thinking each turn shows: a number of
characters, `all`, or `0`. The transcript always has all of it.

The trace shows intents as verbs issue them. A keepalive inside an adapter, a daemon's own
deadman resend and an adapter's stop-on-close are that adapter's business and appear only in
its logs.

## Where the seams are

- **Providers** — add a file under `agent/providers/`, one line in `factory.py`.
- **Detectors** — implement `detect(image) -> list[Detection]`; upstream's future feature
  stream becomes one more detector that reads a socket.
- **Robots** — a package under `adapters/` with `describe()` (the static manifest),
  `make()` (a `RobotAdapter`), `implementations()` (its own verbs) and `conditions()`
  (its named preconditions); keep upstream names in its own `upstream_api.py`.
- **Learned verbs** — `register_learned_verb(registry, spec, runner)`; see
  [learned-verbs.md](learned-verbs.md).
