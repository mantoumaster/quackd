# Adapters: write one in a day

An adapter is how a robot joins quackd. It answers one question, "what is this body and
what can it do", as a `RobotManifest`, and it moves the body through intents the robot's
own controllers execute. Everything else (the loop, the executor, the `.duck` contract,
the MCP server) is shared. quackd ships seven: `microduck`,
`lerobot`, `rosbridge`, `open_duck`, `xlerobot`, `alohamini` and `toddlerbot`. This page is the recipe; [ADR-0017](adr/0017-robot-adapters-and-manifest.md),
[ADR-0018](adr/0018-core-verbs-extensions-aliases.md) and
[ADR-0022](adr/0022-per-adapter-upstream-refs.md) are the reasons.

## The shape

```
quackd/adapters/<name>/
  __init__.py       # the manifest, the adapter class, and the four functions the factory calls
  verbs.py          # extension verbs and named preconditions (only if the robot has any)
  mock.py           # a backend that runs offline and does what the test says
  <real>.py         # the SDK backend, EXPERIMENTAL until run against the target
  upstream_api.py   # every SDK name you spell, VERIFIED or UNVERIFIED, with a pinned link
docs/adapters/<name>.md
tests/test_<name>_adapter.py
```

Some robots have no network API to talk to at all. The Open Duck Mini v2 is one: its
runtime reads a local gamepad and nothing else, so the adapter needed a companion daemon
that runs on the robot, in `bridge/<name>/`. Three rules if you find yourself there. It
must never import quackd, because quackd's dependencies do not belong on a 512 MB board.
It ships in the sdist and never in the wheel, so `packages` stays `["quackd"]`. And it
should be testable with no hardware, which in practice means a `--fake` mode and a pure
core the tests can drive directly.

The four module functions, the same on every adapter package:

```python
# static: no SDK import, no socket
def describe(backend: str, robot_id: str | None = None) -> RobotManifest: ...


# extension verbs and core overrides, keyed by canonical name
def implementations() -> dict[str, Verb]: ...


# named predicates the manifest references
def conditions() -> dict[str, Precondition]: ...


# the backend, imported lazily
def make(
    backend: str,
    *,
    robot_id=None,
    seed=None,
    address=None,
    live=False,
    camera_url=None,
    token=None,
) -> RobotAdapter: ...
```

`describe()` is what `quackd validate --robot`, `quackd list-verbs --robot`, `quackd
announce` and `doctor` use, so it must never import an SDK. `make()` imports the backend
module lazily. Add one row to `_ADAPTERS` in `quackd/adapters/factory.py` (backends,
status line, the pip extra, the module `doctor` probes by metadata) and the name works
everywhere `--robot` does.

## The manifest decides what exists

A verb that is not in the manifest does not exist: not in the registry, not in the MCP
tool list, not in `.duck` validation, not in the prompt. So the manifest is where honesty
lives. The rules, enforced by the model itself ([manifest-spec.md](manifest-spec.md)):

- **Core verbs need what they need.** `observe` needs a camera; `move` and `go_to` need
  the `twist` intent and mobility; `search_scan` needs a camera and either `twist` or
  `gaze`; `say` needs `sound`. Declare a core verb the body cannot support and the
  manifest refuses to build.
- **Extension verbs are the robot's own** (`kick`, `express`, `move_joints`). Declare
  them with `verb_spec(verb, core=False)` and supply the implementation from
  `implementations()`. Reusing a name another robot uses (`move_joints` on the LeRobot arm
  and on the XLeRobot) is how `requires: [move_joints]` is satisfied by both.
- **`stop` is universal**: present on every manifest, always allowed, never gated.
- **Aliases are not yours to declare.** `get_frame`, `walk_to` and `walk` live in
  `quackd/verbs/aliases.py`; a manifest names the canonical verb.
- **Preconditions are names**, with the predicate supplied by the adapter's
  `conditions()`: `{"kick": ["standing"]}` means the executor asks your `standing(state)`
  before every kick.
- **`safety_authority` says who stops the body when quackd goes quiet.** `native: none,
  deadman: false` is a legitimate answer; a wrong `deadman: true` is not.
- **`limits`** are what the core verbs clamp to (`max_vx`, `max_vy`, `max_wz`,
  `gaze_yaw_deg`); leave one out and the schema bound applies.
- **`digest()`** is the capability fingerprint discovery advertises; it ignores `id` and
  `backend`, so the same robot over `sim2d` and `mock` hashes the same.

## The adapter class

A `RobotAdapter` is a `DuckTransport` plus self-description. Wrap your backend and
delegate: `connect()` returns the manifest, `disconnect()`/`close()` release it,
`get_state()` returns a `DuckState` (`posture="unknown"` is fine for a body without
postures; `holding` is for grippers), `get_frame()` returns a PIL image or `None`,
`send_intent()` maps an `Intent` to the SDK, `health()` is informational and never
raises, `heartbeat()` is the watchdog and raises `HeartbeatError`. Copy
`quackd/adapters/rosbridge/__init__.py` for the smallest complete example.

Intents are the whole vocabulary between verbs and backends: `move` (a twist), `look`
(a gaze point), `sound`, `do` (a named skill, `antennas:wiggle`, `policy:pick:cup`),
`joint`, `gripper`, `enable`, `pose`, `stop`. A backend answers each with an `Ack`; a
refusal is data (`accepted=False, reason=...`), never an exception.

## Backends: mock first, then whichever route the robot leaves you

Write `mock` before anything else. It runs offline, records intents, serves a synthetic frame
if the body has a camera, and lets every verb, every executor gate and the detector run in the
test suite.

**A fake must never be kinder than the robot.** If the real backend cannot report a position,
the mock reports `None` too, even though it knows where it is. If a real stop cannot hold the
arms, the mock's cannot either. A mock that is easier than the body is a task that passes here
and fails there, and it is the only kind of bug this repository cannot catch for you.

Then the real backend, and the shape of that depends entirely on what the robot gives you.
Two of the seven adapters import an SDK. The others could not.

| Route | When | Who does it | What it costs |
|---|---|---|---|
| **Import the SDK** | upstream ships an installable package with a client | `lerobot:real`, `rosbridge:ws` | an optional extra, a lazy import inside `connect()`, a lock around a synchronous SDK |
| **Speak its wire** | upstream ships a host process but is not installable, or installing it would drag in torch and a Python floor | `xlerobot:zmq`, `alohamini:zmq` | reading the wire from source rather than the docs, and owning the framing yourself |
| **Ship the robot side** | upstream has no network API of any kind | `open_duck:bridge`, `toddlerbot:bridge` | a daemon in `bridge/`, a protocol you define at both ends, and everything below |

Whichever route: import upstream **inside `connect()`** and raise
`AdapterNotInstalled(name, "quackd[extra]")` on `ImportError`, so a machine without the extra
still validates, lists and mocks the robot. Serialise access under one lock in a worker thread
with a deadline unless you have read that it is thread-safe. Take injectable clients
(`client=`, `robot=`, `ros=`) so the tests drive the mapping with fakes. Add the extra to
`pyproject.toml`, run `uv lock`, and add the module to `doctor.py`'s `EXTRAS`. An adapter whose
robot side you ship needs no extra at all: `open_duck` and `toddlerbot` declare none.

**And never send the body's go-limp call.** `disable_motors`, `disable_torque`, `relax`, an
XLeRobot `disconnect()`: stop means stop, not collapse. This applies to teardown as much as to
`stop`, and upstream's own `disconnect()` is usually where the trap is — four of these robots
disable torque inside it (three by default, the ToddlerBot always), so `close()` has to stop
and hold rather than delegate.

### If you speak a wire

You own the framing, so you own the failure modes that come with it.

- **Nothing on a wire is timestamped unless you make it so.** Stamp on arrival, expose the age
  in `extras`, and turn "no observation lately" into a `HeartbeatError`, or a cached reading
  will be served as a fresh one and a stopped robot will look like a moving one.
- **A socket may drop your older message.** ZeroMQ's `CONFLATE` keeps only the newest, so two
  intents in one tick become one. quackd's answer on both ZeroMQ bodies is a single writer
  that re-sends the whole desired action, rather than a mirror of the robot's state.
- **A partial payload can mean something else entirely.** The AlohaMini's driver indexes three
  velocity keys with no `.get()`, so omitting one discards the whole action, arms included.
  Route every verb through one payload builder rather than composing dicts at each call site.
- **Refusal is data.** Whatever the socket raises when the host dies, the pilot should get an
  `Ack(accepted=False)` naming the address, not an exception through the executor's catch-all.

### If you ship the robot side

What the three rules above actually cost, beyond writing it:

- **A handshake that reports what is really there**, and a `connect()` that narrows the
  manifest from the answer. A capability the daemon reports is a verb quackd will offer, so it
  must report what loaded rather than what a flag claimed.
- **A protocol version, and a refusal on mismatch.** The daemon on the robot is one somebody
  installed months ago.
- **Whatever the robot's own runtime does not do.** The Open Duck's daemon feeds a loop that
  already exists. The ToddlerBot's owns the loop, because upstream's `step()` is a no-op, and
  that difference is most of the size difference between the two.
- **A keepalive, if silence means something.** Where the robot's deadman fires on silence,
  something has to say the client is still there while a long verb runs, because the executor
  sends one command and then waits.

### A simulator, if the cartoon world already draws the body

Four of the seven bodies here ship a `sim2d` backend. It is worth writing when the shared 2D world can represent the
body honestly and the robot has a task worth running end to end; it is not worth writing for a
body the world would have to lie about. It earns a ✅ only with a seeded acceptance sweep that
checks the world's ground truth, not merely a run that does not crash.

There is a second simulator, and it is not a general one. `quackd/sim3d/` holds one arena and a
`Body` protocol with two implementations, a kinematic puppet and the Microduck on its own
trained policy, so a physics body for a new robot means its MJCF, its own controller and a
reason the cartoon cannot serve, usually that you need to know whether a gait works. Nobody has
written a second one, and `sim2d` is what the shared world and the flock are built on.

## `upstream_api.py`: never guess a name

The traps that recur across bodies, and what each one cost, are collected in
[reading-robots.md](reading-robots.md). Read it before the first adapter you write against
an upstream you did not choose.

Every SDK name you spell lives in one file as an `UpstreamRef(name, status, source, note)`
with a permalink to a pinned commit and line. `VERIFIED` means you read it there;
`UNVERIFIED` means it is your assumption, and the note says what quackd does about it.
`tests/test_upstream_api.py` takes one row per adapter: the module, the files allowed to
touch its UNVERIFIED identifiers (the backend and `doctor.py`), and the source prefixes
every link must start with. `docs/adapters/<name>.md` must list every ref's name (a test
checks) and carry the pin and the word "never" until someone has run it for real.

## Status is a promise

The README's status table and [adapter-status.md](adapter-status.md) get ✅ only for what
was exercised against its real target by us. A new adapter arrives 🧪 for its SDK backend
and stays 🧪 until a human runs it on hardware and the transcript says so. Nothing in
this repository claims a robot moved unless one did.

## The checklist

1. `quackd/adapters/<name>/__init__.py` with the manifest, the adapter class and the four functions.
2. `mock.py`, and a test that runs every verb through an `Executor` on it.
3. `upstream_api.py` with pinned links; a row in `tests/test_upstream_api.py`.
4. The SDK backend, lazily imported, injectable, with a test on fakes and a test that the
   missing extra names itself.
5. A row in `_ADAPTERS`, the extra in `pyproject.toml` (`uv lock`), the module in `doctor.py`.
6. `docs/adapters/<name>.md` (every ref name, the pin, "never"), a row in the README status
   table and in `adapter-status.md`, a CHANGELOG entry, a `docs/architecture.md` mention.
7. The gate: `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest && uv run quackd validate ducks/*.duck`.
