# Contributing to quackd

Thanks for taking a toy duck seriously. Two kinds of contribution matter most: **new
`.duck` files** (the community funnel) and **new verbs** (the vocabulary). Both are small.

## Dev setup

```bash
git clone https://github.com/rokbenko/quackd && cd quackd
uv sync --extra dev            # the core and all seven robots, editable, no robot SDKs
uv sync --extra dev --extra mujoco   # the physics simulator, or its tests just skip
uv run pre-commit install
uv run pytest                  # the whole suite, a few minutes, no network, no keys
uv lock --check && uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run quackd validate ducks/*.duck
```

Add `--extra anthropic`, `--extra openai` or `--extra gemini` if you want a real provider.

This repository is a uv workspace: `quackd` is one distribution and each robot under
`adapters/` is another (`quackd-microduck`, `quackd-lerobot`, `quackd-rosbridge`,
`quackd-open-duck`, `quackd-xlerobot`, `quackd-alohamini`, `quackd-toddlerbot`). The dev
extra installs the core and all seven of them editable, so an edit anywhere is live in the
next test run with nothing to reinstall. None of the robot SDKs come with them, and none is
needed: the suite drives every adapter against a mock or a fake.

> [!IMPORTANT]
> What a user gets is not what you have. `uv pip install quackd` installs no robot at all,
> and every command that needs a body refuses with the list of extras. You are working with
> all seven present, so a message that only fires when one is missing is a message you will
> never see by accident. CI's `packaging` job is the one that checks it: it builds all eight
> packages, installs the core wheel alone, and asserts that `quackd doctor --json` reports
> seven adapters and none of them installed.

`uv lock --check` runs before anything else in CI. Eight packages share one lock file and
`uv sync` re-locks in silence when a `pyproject.toml` has drifted, which is how an unlocked
extra once reached a release. Change a dependency anywhere, run `uv lock`, and commit
`uv.lock` with the change that caused it.

Versions move together. The core and each adapter carry their own `__version__`, because an
adapter's sdist holds only its own source and cannot read the core's, and each one pins a
window on the core (`quackd>=0.9,<0.10`). `uv run python scripts/set_version.py X.Y.Z`
rewrites all eight and the windows that tie them together; nothing here is edited by hand. A
release then builds eight wheels and eight sdists with `uv build --all-packages`, core first
to PyPI because every adapter depends on it. The full order is in [PLAN.md](PLAN.md).

`uv run mypy` checks with whatever interpreter your venv has. CI runs it twice,
under 3.11 and 3.12, and `[tool.mypy]` pins no `python_version` on purpose (pinning 3.11
made mypy reject numpy's stubs under 3.12), so a clean local run is half of that gate.
`uv sync --python 3.12 --extra dev` and run it again for the other half.

Windows, macOS and Linux are all first-class. Tests must never touch the network. About a
third of that is the seeded acceptance sweeps, which CI holds at 10 of 10 by setting
`QUACKD_STRICT_SEEDS=1`; locally they pass at 8 of 10 so a slow machine does not block you.

## Where things live

| Path | What is there |
|---|---|
| `quackd/` | the core, and nothing that is one robot's: the loop, the executor, the verb registry, the `.duck` contract, the MCP server, the 2D arena (`quackd/sim2d/`) and the mock transport (`quackd/transport/mock.py`) |
| `quackd/adapters/` | what every adapter shares: `base.py` (the `RobotAdapter` protocol and its helpers), `manifest.py`, `catalogue.py` (the seven quackd publishes, as strings), `factory.py` (`--robot` to a body) |
| `adapters/<name>/` | one robot, one distribution: `pyproject.toml`, `README.md`, and the code in `src/quackd_<name>/` |
| `bridge/<name>/` | the daemon that runs on the robot itself (`open_duck`, `alohamini`, `toddlerbot`), which never imports quackd |
| `ducks/`, `docs/`, `tests/`, `web/`, `scripts/` | the starter task files, the documentation, the whole suite, the browser demo, `set_version.py` |

A robot's code is imported as `quackd_<name>`, never as `quackd.adapters.<name>`. The 2D
arena and the mock transport stayed in the core on purpose: three bodies subclass
`Sim2DTransport` and a fourth uses it as it is, and six of the seven mocks draw their frame
with `quackd/sim2d/render.py`, so neither was ever the duck's. `UpstreamRef`
lives in `quackd/upstream.py` for the same reason, since every adapter cites an upstream and
none of them should import a duck to do it.

Touching `adapters/microduck/src/quackd_microduck/sim3d/` or that package's
`transports/mujoco.py`? Install `--extra mujoco` or your work
is untested locally: both test modules start with `pytest.importorskip("mujoco")` and vanish
without it. CI's `physics` job installs the extra and runs them on the kinematic stand-in, which
touches no network. The tests marked `real_duck` need upstream's model in `~/.quackd/cache`, so
they skip until you have run `--robot microduck:mujoco` once, and a nightly job runs them there.
The gait arithmetic itself lives in that package's `sim3d/gait.py`, which imports no `mujoco`, so
`tests/test_sim3d_gait.py` runs whether you installed the extra or not.

Touching anything under `bridge/`? That is the code that runs on a robot, and there are
three lots of it now (`open_duck/`, `alohamini/`, `toddlerbot/`). It plays by different
rules: it must never import quackd (its dependencies do not belong on a 512 MB
Raspberry Pi), it ships in the sdist and never in the wheel, and it stays testable with no
hardware through its `--fake` mode and a pure core the tests drive directly. The
ToddlerBot daemon is the largest of the three, because it owns that robot's control
loop rather than feeding one, so it carries the most of its own safety machinery.

Touching `web/`? That is the browser demo, and the only quackd code that is not Python: plain
JavaScript modules, no build step, nothing to install. Run it with the server in the directory
itself, not with `http.server` ([web/README.md](web/README.md) says why):

```bash
python web/serve.py            # then open http://localhost:8000/simulator/
```

The mount matters here for a reason that is yours rather than the reader's. The live page at
<https://www.quackd.org/simulator> belongs to quackd-web, a separate project whose build fetches
this directory into its own `/simulator` at a pinned commit. `web/serve.py` mounts the directory
the way that deploy does, so the local mount is what you compare your change against, and what
you test there is what ships.

Merging is not shipping here. Because the deployed copy is pinned, a change to `web/` on `main`
does not reach a visitor until quackd-web builds again. Asking it to is the whole job of
`.github/workflows/refresh-the-simulator.yml`, which pings a Vercel deploy hook on pushes to
`main` that touch `web/` or the workflow file itself, and on a manual `workflow_dispatch`. The
hook is a secret (`VERCEL_DEPLOY_HOOK`), and until it exists the job says so and exits green
rather than failing. A fork never gets that far: the job is guarded on the repository name, so
it does nothing at all there. `/simulator/source.json` records which commit the live copy was
built from, which is how you tell whether your change is on it yet.

`tests/test_web.py` gates the directory from the ordinary suite with no browser: the ids the
script looks up, the mount, the assets, the key that is stored nowhere, the rule that a key
barges into a run if and only if it would move the robot, `node --check` on each module and the
argument validator executed under Node (those last two need node on your machine and skip
without it; CI's runners all have it — everything else is Python and always runs). It is the
floor, not the test: the page has been booted in a browser twice and a held `W` walks the duck,
but no model-driven run, no barge-in out of one and no recording has ever been watched. If you
open it, say what you saw in the PR. See [web/README.md](web/README.md).

Touching `quackd/lan/` or `quackd/flock/mqtt_bus.py`? Neither imports its library at module
level and neither is in the default install, so the tests run them on fakes: a fake zeroconf
registrar and a synchronous fake MQTT broker, no sockets. Keep it that way, and see
[docs/lan.md](docs/lan.md).

## Submit a `.duck`

1. Copy a starter from [`ducks/`](ducks/) and edit the frontmatter + body.
   Spec: [docs/duck-spec.md](docs/duck-spec.md).
2. `uv run quackd validate ducks/your-duck.duck` — it must pass.
3. Run it at least once: `uv run quackd run ducks/your-duck.duck --provider fake`
   (the scripted pilot only knows the starters, so for a new duck use a real provider if
   you have a key, or add a strategy to `quackd/agent/providers/fake.py`).
4. Open a PR. In the description say what it does, which providers you tried, and what
   failed. Ducks that mostly fail are still welcome if the file says so — that is data.

Checklist: `duck: 0` (or `duck: 1` if you use `requires`, `robots`, `flock.roles` or
`flock.allocation.method: pilots`, or `duck: 2` if you correct the robot's `datasheet` or a
role's `needs`) ·
slug name · `allow` lists only verbs the robot provides (`quackd list-verbs --robot ...`)
· `confirm` ⊆ `allow` · at least one `success` line · `abort_when` uses the two enforced
phrasings if you want them enforced · body starts with `# Task` · `quackd validate
your.duck --robot <adapter>:<backend>` passes for the robot you mean.

**Ask for a note.** Every solo starter except `hello-world` and the bring-up lookouts
ends its numbered strategy with a `remember` and carries a short *Memory* section saying what
is worth keeping for next time. A coordinator flock's duck has no `remember` step, because its
members are state machines with nothing to remember with; a pilot flock's members are whole
pilots and may.
Put the call in the strategy rather than only in a Memory section: a 14B local model read a
prompt-level hint and never wrote to memory, and followed the same instruction on its first
run once it was step 5. `remember` is offered automatically when memory is on and needs
nothing in your `allow` list. Skip it for a smoke test, the way `hello-world` does
([docs/memory.md](docs/memory.md)).

## Add a verb

1. Decide the kind. **Core** (`quackd/verbs/core.py`) = the same on every robot whose
   manifest meets a requirement (a camera, a `twist` intent, a `sound` intent); add its
   `Requirement` to `REQUIREMENTS`. **Extension** = one robot's own behaviour, in that
   adapter's `verbs.py` (Microduck: `adapters/microduck/src/quackd_microduck/verbs.py`; it
   needs a VERIFIED upstream method in that package's `upstream_api.py`). **Learned** = v2, see
   [docs/learned-verbs.md](docs/learned-verbs.md). If the thing you are adding never
   touches the body, it is probably not a verb at all: `remember` sits next to
   `declare_success` as a *meta tool* precisely so that the rule "the vocabulary comes from
   the manifest" keeps meaning something ([ADR-0025](docs/adr/0025-memory-between-runs.md)).
2. Write a pydantic params model (`extra="forbid"`, ranges on every number) and an
   `async def my_verb(ctx: VerbContext, p: MyParams) -> VerbResult`. Use
   `ctx.transport.send_intent(...)`, `ctx.transport.sleep(...)`, `ctx.detector`,
   `ctx.manifest` (to pick a strategy per body), and `ctx.on_frame(img, caption)` for the
   GIF. Never call an LLM from a verb.
3. Add a `Verb(...)` template with a one-line LLM-facing description, a `timeout_s`, a
   `safety_class` (`safe` · `confirm` · `dangerous`) and a `done_condition` to `CORE` or to
   the adapter's verb table, then a `VerbSpec` entry in the adapter's manifest (that is
   what makes the verb exist: a verb not in the manifest is not in the registry, the MCP
   tool list, `.duck` validation or the prompt). Preconditions are named in the manifest
   and supplied by the adapter's `conditions()`. Then classify it in `quackd/verdict.py`:
   `BEFORE_VERDICT` if it looks, speaks or brakes, `MOVES_THE_BODY` otherwise. The gate
   reads the first set and nothing else, so a verb in neither waits for a verdict like
   anything that moves the body, which is wrong for a verb the pilot needs in order to reach
   one, and `tests/test_verdict.py` fails until you have chosen.
4. Add a test: on `MockTransport` for intent sequences, on `Sim2DTransport` for behaviour, and on `MujocoTransport(body="puppet")` if the verb makes a claim about the body, because the cartoon cannot tell you whether one is true.
5. If the verb needs an upstream method we have not verified, add it to the adapter's
   `upstream_api.py` as `UNVERIFIED` with a note and a row in that adapter's page under
   `docs/adapters/` (the Microduck's table is in `docs/adapter-status.md`). Never invent
   one.
6. Mention it in `docs/architecture.md`, the README verb table (a test checks every
   registry name is backticked there) and `CHANGELOG.md` (Unreleased).
7. Nothing extra is needed for the trace: every intent your verb sends is already an event,
   and `ctx.log(...)` is already a `note`. If you emit a new event *kind*, add a row for it
   to the table in `docs/architecture.md`, because a test reads the kinds out of the code
   and fails when the docs do not name one.

Renaming a verb is not a rename: add the new name and keep the old one in
`quackd/verbs/aliases.py`, the only file that may spell an alias.

## Add a provider

A provider is one file under `quackd/agent/providers/` and six entries that have to
agree: its tuple in `CATALOGUE` in `providers/catalogue.py`, which is where the name, the
model ids and the default all come from and what puts a cloud vendor in `CLOUD_NAMES`, then
rows in `KEY_ENV`, `EXTRA_FOR` and `SDK_FOR` in `providers/factory.py`, then either a row in
`OPENAI_COMPATIBLE` or a branch in `make_provider`, and finally a `quackd[<name>]` extra in
`pyproject.toml`. `tests/test_catalogue.py` counts them now, under the heading
`one file and the entries that have to agree`: it walks `PROVIDER_NAMES` and fails on the
missing row, and it opens `pyproject.toml` to check the extra exists and installs the SDK
that provider imports. Before it existed a missing row was a `KeyError` out of `quackd
doctor`, which is the command people run when something is already wrong.

Then the browser, which has its own copy of the model list and its own reason to refuse one.
Run `python web/build_catalogue.py` and commit `web/src/catalogue.js`, or the generator-drift
test in `tests/test_web.py` fails. Then decide whether the page can call the vendor at all: it
calls from the visitor's browser, so a vendor that refuses a cross-origin preflight goes in
`NOT_FROM_A_BROWSER` in `web/src/providers.js` with the reason and the date, and a vendor that
answers one goes in `PROVIDERS` with its base URL, its key link and the `tool_choice` its own
docs allow. A test holds that pair to exactly the vendors `PROVIDERS` leaves out, so neither
half can be skipped quietly.

Four things the tracing depends on, none of them optional:

1. Fill `ProviderTurn.thinking` with the model's own reasoning when the API returns it, and
   `Usage.reasoning_tokens` with what it charged for. The trace shows the first and the
   transcript keeps all of it; a provider that drops them makes the run unarguable.
2. Degrade with exactly one retry. If the API refuses a request because it does not support
   thinking, turn thinking off, remember that, and retry once. Match the specific complaint,
   not the word: a 400 about a *replayed* thinking block is a different bug and retrying it
   loops.
3. Wrap every SDK exception in `ProviderError`. The loop treats one as a turn it can report
   and the run ends cleanly; anything else is a traceback in somebody's terminal.
4. Never let response parsing raise outside that wrapper. An empty `choices`, a usage field
   that is a string, a tool call with no name: all of it is `ProviderError`, and the test
   for it belongs in `tests/test_providers.py`.

## Add an adapter

A robot joins quackd as its own distribution, built from `adapters/<name>/` and imported as
`quackd_<name>`. It declares a `RobotManifest`, moves the body through intents its own
controllers execute, and announces itself to the core through the `quackd.adapters` entry
point group. That group is the whole of the contract, so an adapter for a robot nobody here
owns can be published to PyPI without a pull request against this repository. The recipe,
the rules the manifest enforces and the checklist are in
[docs/adapters.md](docs/adapters.md); the honesty rules are
[ADR-0022](docs/adr/0022-per-adapter-upstream-refs.md). In short: write `mock` first; put
every SDK name in the package's `upstream_api.py` with a pinned link and a row in
`tests/test_upstream_api.py`; import the SDK inside `connect()` behind that package's `[sdk]`
extra; never send the SDK's "go limp" call; write `docs/adapters/<name>.md` listing every ref;
and arrive 🧪 in the status tables until someone runs it against the real thing.

## Working agreements

- **Conventional Commits** (`feat:`, `fix:`, `docs:`, `chore:`, `test:`).
- Consequential decisions get a short ADR in `docs/adr/` (copy the shape of an existing one).
- Every module opens with a docstring saying *why it exists*.
- Keep the default install light: robots, provider SDKs and YOLO are all optional extras,
  and `uv pip install quackd` brings none of them.
- **Never commit an upstream asset.** No logos, meshes, CAD, MJCF, ONNX policies or videos,
  from Pollen Robotics or anyone else, in a commit, a test fixture or a docs asset. This got
  sharper in 0.8: a real `--robot microduck:mujoco` run puts upstream's `robot_walk.xml` and
  38 CC BY-NC-SA meshes in `~/.quackd/cache`. quackd's whole licence position is that it
  redistributes none of them, and a public history does not forget. `.gitignore` now catches
  `.stl` and `robot_walk.xml` as well as `.onnx`, but do not rely on it.
- Tone: confident, playful, honest about status.

## How your PR gets handled

Written down because 0.6 was the first release built on other people's pull requests, and
the way those two were handled is the way the next one will be.

**Your commits stay yours.** A PR is merged with `git merge` into a scratch integration
branch, never squashed, rebased or retyped, so your authorship survives verbatim and your
commits appear on `main` under your name. Not one line of your diff is edited in place.

**Corrections land separately.** Anything that needs fixing on top goes in its own
follow-up commit with its own message, so `git log` keeps the credit and the correction
distinguishable forever. You can read exactly what was changed after you and why, and
disagree with it.

**You get told everything that was found, not a verdict.** The review comment lists every
defect with the reasoning, including the ones that were nobody's fault. If something is
declined, the comment says why.

**You get credited** in the CHANGELOG entry, in the release note, and in the row of faces
in the README, which is generated from the contributor list and orders people by lines
added.

### For whoever is doing the merging

1. **Run the gate on the merged result, not on their branch.** A PR from a fork gets no CI
   here until a maintainer approves the run, so a green checklist in the description is
   usually not evidence about anything. Check how far behind `main` the branch is too: one
   of 0.6's contributions was 67 commits behind, from before robot adapters existed, so its
   ticked boxes had been measured against a repository two releases old.
2. **Check the claims, not only the code.** Both 0.6 contributions were well made and both
   asserted something false in prose. One said the README listed a gap among its
   limitations when it never had, and that sentence was about to ship in a permanent ADR.
   This project's credibility is that it does not say things that are not so, and a PR is
   where that leaks in.
3. **Fix it on top, in named commits.** One commit per theme reads better than one per
   defect and much better than one big one.
4. **Anything no test could see becomes a test.** That is the rule the whole repository
   runs on: a stale count in a docstring, a promise the code does not keep, a key two files
   have to agree on. If the review found it by reading, the next one should find it by
   failing.
5. **Expect the review to surface older breakage.** Auditing 0.6's two contributions turned
   up six claims that had gone stale on `main` before either arrived. Fix those in the same
   release and say so in the CHANGELOG rather than leaving them for later.
6. **Reply properly and say thank you.** Somebody spent their evening on this.

### How the reply is written

It goes out under a person's name, to a person, so it has to read like a person wrote it.
This section is written to its own rules, as the worked example.

**No dashes.** No em dash, no en dash, no hyphen standing in for a comma or a colon or an
aside, and no hyphen bullets. A hyphen belongs only inside something that genuinely has one:
`google-genai`, `gemini-3.8-flash`, `--provider`, a branch name, an identifier. Everything
else is a real sentence, or a colon, or a full stop. Prose here leans on em dashes heavily
and a reply must not, so the weight goes onto the colon and onto the short flat sentence
after a long one instead. This is the fastest way for a reply to look machine written, and
it is the last thing to check before posting.

**None of the assistant tells.** Not "Great work!", not "Let's dive in", not "Overall," or
"In summary," or "It's worth noting that". No three tidy parallel bullets where two uneven
sentences would do. No closing line that restates the paragraph above it. Hedging like "it
seems" or "it appears" means the checking did not happen, so go and check, then write the
answer.

**When something is declined, say what it lost to.** Usually it is timing and not judgement:
`main` moved, or the same problem got solved another way while the PR sat open. If they were
right when they wrote it, say that in as many words. Nobody should have to work out from
silence whether they were wrong.

**Thank them for the specific thing.** Not for "the contribution". For the evening, for the
provider nobody else was running, for the bug that could only be found by pointing a duck at
it and reading what came back.

**Length.** Something a person reads in one go. If it has grown headings to hold itself
together, it is too long.

## Reporting bugs and proposing verbs

Use the issue templates. `quackd doctor` output and the relevant `transcript.jsonl` lines
turn a vague bug into a fixable one.
