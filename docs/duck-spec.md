# The `.duck` file — spec v0 and v1 (normative)

A `.duck` file is a task for an LLM-piloted robot. It is deliberately **SKILL.md-shaped**:
YAML frontmatter between `---` fences, then a Markdown body. The frontmatter is a contract
the executor enforces; the body is the prompt. **The LLM is never trusted to self-police.**
`.duck` is the format name the way `Dockerfile` is: a task for a LeRobot arm is a `.duck`
too. `duck: 1` (quackd 0.4) adds what a multi-robot task needs; `duck: 0` files parse and
run unchanged ([ADR-0019](adr/0019-duck-spec-v1.md)).

Machine-readable schema: [`../quackd/duckfile/schema.json`](../quackd/duckfile/schema.json)
(generated from `quackd/duckfile/schema.py`; a test keeps them in sync).

## File shape

```
# optional comment lines above the first fence are allowed
---
<YAML mapping>
---
<Markdown body — must be non-empty>
```

Encoding UTF-8. The first non-blank, non-comment line must be `---`.

## Frontmatter fields

| Field | Type | Required | Enforced by | Meaning |
|---|---|---|---|---|
| `duck` | `0` or `1` | yes | parser | Spec version. `1` unlocks `requires`, `robots`, `flock.roles` and `flock.frame_hints`; using them under `0` is an error that names the fix. |
| `name` | slug `^[a-z0-9][a-z0-9-]{0,63}$` | yes | parser | Identifier; run directories and the fake pilot's strategies key on it. |
| `description` | string | yes | — | One human-facing line. Shown in the system prompt. |
| `author` | string | no | — | Credit. |
| `verbs.allow` | list of verb names, ≥ 1, unique | yes | **executor** | The only verbs the LLM may call. `stop` is always allowed. Unknown names fail `quackd validate`. |
| `verbs.confirm` | list ⊆ `allow` | no (default `[]`) | **executor** | Verbs that prompt a human y/N before running (`--yes` auto-accepts; MCP refuses unless `--yes`). |
| `budgets.max_steps` | int 1–1000 (default 40) | no | **executor** | Maximum verb executions. |
| `budgets.max_minutes` | number > 0 ≤ 180 (default 5) | no | **loop** | Robot-clock cap (sim time in both simulators, `sim2d` and `mujoco`, wall-clock on hardware). Checked before each model call and again the moment the model answers, so a provider that replies late cannot spend the overrun. A verb already running is not interrupted, so a run can overshoot by that verb's own timeout. |
| `budgets.max_llm_calls` | int 1–2000 (default 40) | no | **loop** | Maximum provider calls (re-prompts count). |
| `success` | list of strings, ≥ 1 | yes | LLM (+ ground truth in sim tests) | Criteria the model judges itself against via `declare_success(reason)`. |
| `abort_when` | list of strings | no | **executor** for two phrasings; LLM otherwise | See below. |
| `persona` | string | no | — | Tone. Inserted verbatim into the system prompt. |
| `providers` | list of strings | no | — | Tested-with, **not** a restriction. |
| `learned_verbs` | list of `{name, policy, description?, metadata?}` | no | `validate` rejects non-empty | Reserved for v2 ([learned-verbs.md](learned-verbs.md)). |
| `flock` | mapping, see below | no | **coordinator** | Cooperating robots (simulator only). Absent means a single robot. |
| `requires` | list of verb names ⊆ `allow` (v1) | no (default `[]`) | `validate --robot` | The verbs the task *needs*. Checked against each robot's manifest. For a v0 file every allowed verb is required. |
| `robots` | `<adapter>[:<backend>]`, or a mapping member → spec (v1) | no | CLI | The default robot(s), so `quackd run <duck>` needs no `--robot`. Flags win over the file. |

### `requires` and `robots` (v1)

`requires` is the honest minimum: a robot that lacks one of these verbs cannot do the task,
and `quackd validate <duck> --robot <adapter>:<backend>` says so with a field-level line
such as `requires kick, but arm-01 (lerobot-so101) does not provide it` (exit 1). Verbs in
`allow` that are not required are advisory: a robot may lack them and still qualify, and
`validate` reports them as a weaker `verbs.allow` line. For a solo task every listed robot
must provide every required verb; for a flock, the flock as a whole must (see the roles
below). Aliases count: a robot that provides `observe` satisfies `get_frame`.

`robots` names the default robot for a solo task (`robots: microduck:sim2d`) or one per
flock member (`robots: {duck-01: microduck:sim2d, duck-02: microduck:sim2d}`).

### `flock` — cooperating robots (simulator only)

A flock duck with a non-empty `verbs.confirm` fails `quackd validate` (there is no per-duck
terminal to prompt on). Full semantics: [flock.md](flock.md).

| Field | Type | Default | Enforced by | Meaning |
|---|---|---|---|---|
| `flock.members` | int 2–4, or list of 2–4 unique slugs | 3 | coordinator | Member count (named `duck-0`…) or explicit names. `--flock N` overrides. |
| `flock.allocation.method` | `auction` | `auction` | coordinator | Contract Net is still the only method. |
| `flock.allocation.bid` | `ball_distance` | `ball_distance` | coordinator | Lower camera-estimated distance wins. |
| `flock.allocation.tie_break` | `duck_id` | `duck_id` | coordinator | Lexicographic member name. |
| `flock.allocation.hysteresis_pct` | 0–100 | 20 | coordinator | A challenger must bid this much lower to unseat the current claimant. |
| `flock.allocation.claim_lease_s` | > 0 ≤ 60 | 6 | coordinator | Longest a claim may be held before re-auction (sim clock). A fixed fuse from the grant, not a progress timer. |
| `flock.safety.min_separation_m` | 0.1–2.0 | 0.4 | coordinator | Non-kickers keep at least this far from the action. |
| `flock.safety.one_claimant` | bool | true | coordinator | At most one robot approaches the ball. Always enforced, `false` is rejected at validation. |
| `flock.safety.per_duck_heartbeat_s` | > 0 ≤ 10 | 1.0 | coordinator | Bus heartbeat period; the watchdog presumes a duck dead after 3× this, or this plus 2.5 s, whichever is longer. |
| `flock.search.partition` | `heading` | `heading` | coordinator | Each duck owns a heading sector. |
| `flock.search.restart_s` | > 0 ≤ 120 | 8 | member | Re-scan the sector when nothing was found for this long. |
| `flock.roles` (v1) | mapping `{spotter: {requires: [...]}, kicker: {requires: [...]}}` | absent | coordinator | Heterogeneous roles. A robot bids only for a role whose `requires` its manifest satisfies. quackd knows exactly these two roles (both must be given), one robot each; `members` must then be a named list. Each role's `requires` ⊆ `allow`. |
| `flock.frame_hints` (v1) | `auto` · `on` · `off` | `auto` | runner | Share arena-frame target hints between robots. `auto` is on only when every member runs in `sim2d`; there is no shared frame on hardware ([flock.md](flock.md)). |

Unknown keys anywhere are errors (`extra="forbid"`).

### `abort_when` — what is enforced

Two phrasings are recognised (case-insensitive) and enforced by the executor:

- `Battery below N%` (also `under`, `<`) — before every verb, if the robot reports
  `battery_percent < N`, the run aborts. A body whose manifest has no `battery` sensor
  reports `None` and this rule can never fire on it, so it is silently unenforceable
  there rather than an error. An AlohaMini is the shipped example
  ([adapters/alohamini.md](adapters/alohamini.md)).
- `Same verb fails N times in a row` — N consecutive failed results of one verb abort the run.

Every other entry is handed to the LLM under *"Abort conditions you must respect yourself"*.
The spec says this plainly rather than pretending prose is policy.

### Verb names

Anything the robot's manifest provides (`quackd list-verbs`, or `list-verbs --robot`):
the core verbs `observe report_state stop say move go_to search_scan approach_and` on any
robot that meets their requirements, a robot's own extensions (Microduck: `sit stand
stand_up kick grab gaze quack`), plus any registered learned verb. The 0.3 names
`get_frame`, `walk_to` and `walk` are permanent aliases of `observe`, `go_to` and `move`;
a file may use either spelling but not both. `stop` may never appear in `confirm`. Params
and ranges come from the registry, not the `.duck` file ([ADR-0018](adr/0018-core-verbs-extensions-aliases.md)).

## Body

Free Markdown, non-empty, placed verbatim at the end of the system prompt under
*"Task file: `<name>` — `<description>`"*. Conventions the starters follow:

- `# Task` — one or two sentences of intent.
- `## Strategy` — a numbered plan naming verbs in backticks.
- `## Notes` — failure modes and what to do about them (verify-and-retry, when to give up).

The body cannot widen the contract: a verb mentioned in the body but absent from `allow`
is refused at runtime and the LLM is told so.

## Runtime semantics

- The loop ends with one of `success`, `failure` (the LLM's declaration), `budget`,
  `aborted` (heartbeat, kill switch, enforced `abort_when`), or `error` (a provider or
  transport that failed, or a bug). The robot is stopped in every case and its adapter
  closed.
- `--max-steps` on the CLI overrides `budgets.max_steps` for one run.
- `--dry-run` executes read-only verbs (`observe`, alias `get_frame`, and `report_state`) and logs everything else without
  sending an intent.

## Validation

`quackd validate <files or globs or bundled names>` prints a table and exits 1 on any
failure, with a path and a field-level reason. Checks: parse, schema, unknown verbs,
`learned_verbs` empty, no `confirm` in a flock. With `--robot <adapter>:<backend>` (one or
more) or `--robots name=spec,...`, the file is also checked against those robots' manifests:
`requires` (or, for v0, `allow`) per robot, and every flock role fillable by at least one
robot. Without a flag, the duck's own `robots:` default is used, then the Microduck.

## Resolution

`quackd run x` tries `x` as a path, then `x` / `x.duck` among the bundled starters
(`ducks/` in a checkout, `quackd/ducks/` inside the wheel).

## Versioning

`duck: 0` is the 0.1 to 0.3 contract ([ADR-0005](adr/0005-duck-spec-v0.md)); `duck: 1`
adds `requires`, `robots`, `flock.roles` and `flock.frame_hints`
([ADR-0019](adr/0019-duck-spec-v1.md)). v0 files keep parsing because the version is
explicit and the parser is strict; the only new rejections a v0 file can hit are two
contradictions no shipped file contains (a verb listed next to its alias, `stop` in
`confirm`). Older quackd versions refuse v1 files, which is the correct failure.
