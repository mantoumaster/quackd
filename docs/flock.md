# Flock mode

Several simulated robots cooperating on one task. Ships since v0.3, **simulator only**,
and labelled experimental.

```bash
uvx quackd run flock-kick --provider fake --seed 3               # ducks split the search, auction, kick
```

Ducks split the search for a ball, the one that bids the shortest camera distance wins the
kick, and everyone else keeps clear. One GIF, one `flock.jsonl` you can read the
cooperation out of, zero API keys.

## What a flock is

2 to 4 robots in one shared arena on one shared clock, each with its **own** safety
executor enforcing the same `.duck` contract: allowlist, budgets, machine enforced abort
rules, per robot transcript. Ducks come in the four colorways (Cream, Sky, Lavender,
Graphite). A deterministic **coordinator** referees. Add a `flock:` block to a `.duck` file
or pass `--flock N` to any duck; name the members' robots with `robots:` in the file or
`--robots name=<adapter>:<backend>,...`. Every member acts only through the verbs its own
manifest provides ([ADR-0020](adr/0020-heterogeneous-flocks.md)).

## The bus

All coordination crosses a tiny in process pub/sub bus, one message at a time, and every
message lands in `flock.jsonl`. Eight message kinds: `TASK` (the plan), `BID` (a sighting
with the bidder's own camera distance estimate and, with roles, the role it bids for and
the verbs it provides), `CLAIM` (the one kick permit, with the role assignments), `ROLE`
(SEARCH a heading sector, KICK, YIELD, STOP, and with roles SPOT and JUDGE), `HB`
(heartbeat for the watchdog), `RESULT` (kicked, miss, search empty, budget, aborted, and
with roles `kick_done`), `HINT` (an arena frame target estimate, sim only) and `VERDICT`
(the spotter's judgement). The bus is a small protocol so a LAN bus (MQTT) can slot in for
real robots. Only the in process implementation is used by default, and nobody ever awaits
the bus, which keeps the shared clock deadlock free.

## The auction, in one paragraph

Contract Net, the same shape RoboCup teams use. The first `BID` opens a window of 0.4 s
of sim time. When it closes, the lowest camera distance wins, ties break on the member
name, and a previous kicker keeps its claim unless a challenger undercuts it by the
hysteresis margin (20 % by default), which stops role oscillation. The claim carries a
lease (6 s), a fixed fuse from the moment it is granted. A miss or an expired lease
releases the claim and the failed duck sits out a cooldown, during which it may keep
searching but cannot bid. A lost heartbeat also releases the claim, but that duck is
presumed dead and excluded for good. Either way everyone re-scans the full circle (the
ball has moved) and the auction runs again. Ducks cannot fall in the 2D
simulator, which is the only place a flock runs, so fall handling is untested. A duck can fall in
`microduck:mujoco`, but every flock member has to be a `sim2d` robot, so nothing exercises
that path yet.

## Roles

- **SEARCH**: orient to your heading sector (if `walk` is allowed), `search_scan` inside
  it, quack on a sighting (the theatrical part), publish a `BID`.
- **KICK**: `walk_to` the target, `kick`, report the result. The contract's criterion is
  total ball displacement, so a rally of short kicks counts.
- **YIELD**: stop, and back away when the coordinator's ground truth check says you are
  inside the minimum separation ring, or as blind courtesy when your own last ball
  estimate was.
- **STOP**: the run is over.

Each role step is one verb through that duck's own executor. A role change mid verb
preempts it cleanly and does not count as a failure.

## Heterogeneous roles (0.4)

A `duck: 1` file may declare `flock.roles`, and quackd knows exactly two, `spotter` and
`kicker`:

```yaml
flock:
  members: [duck-01, duck-02]
  roles:
    spotter: {requires: [observe, gaze]}
    kicker: {requires: [go_to, kick]}
robots:
  duck-01: microduck:sim2d
  duck-02: microduck:sim2d
```

- **Capability aware bids.** A robot bids only for a role whose `requires` its manifest
  satisfies (aliases count: `get_frame` satisfies `observe`). The coordinator checks every
  bid again before counting it, so a bid from a robot we do not run cannot claim a role it
  cannot do. The cost stays the robot's own camera distance, so no shared map is needed.
- **One auction, several roles.** Roles are filled most constrained first (fewest eligible
  bidders, then role name), lowest own distance wins, ties break on the member name, and a
  previous kicker keeps its claim under the same hysteresis rule. The **spotter is held
  for the run** (its reference frame must not change between kicks); the kicker is
  re auctioned every cycle. A duck can spot too, so two ducks make a valid spotter and
  kicker pair, which is the only pairing quackd ships an adapter for today ([ADR-0020](adr/0020-heterogeneous-flocks.md)'s amendment).
- **SPOT**: gaze at the sighting, take a fresh frame, keep the target in view. The
  spotter's first sighting is its reference point.
- **KICK, with roles**: `go_to`, `kick`, step aside, and report `kick_done`. The actor
  never evaluates success.
- **JUDGE**: the spotter sweeps its gaze around the last sighting with a fresh frame at
  each look and publishes a `VERDICT`: `moved` when the target is more than the contract's
  0.3 m plus a judge margin from the reference, `not_moved`, or `lost`. The judge margin
  (0.15 m by default) exists because the size based distance estimate quantises in about
  0.2 m steps beyond 1.5 m: a strict spotter costs a re kick, a lenient one would let the
  world's veto fail the run. Only `moved` is a success; anything else sends the kicker
  back to search and kick again against the same reference, so a rally adds up.

The spotter judges, the world vetoes: `summary.json` only says `success` when the
spotter's verdict and the simulator's `ball_displacement_m` agree. No bundled starter
exercises this path today: it shipped alongside a stationary-head adapter that no longer
exists, so the mechanism above is real and tested at the unit level
([tests/test_flock_roles.py](../tests/test_flock_roles.py)) but currently undemonstrated
end to end. Nor can you simply write one: a `.duck` like the one above **validates** (`quackd
validate` reports a flock of 2) and then **fails at startup** with `no live robot can take the
spotter role`, because the coordinator checks who can fill each role before any member has
finished connecting and reported its vocabulary. Nothing in the shipped tree reaches the
role and auction path end to end, and that is the honest state of it.

## Frame of reference

There is no computable relative frame between two robots on hardware: the Microduck has
no absolute localisation, and quackd does not assume two robots share a coordinate frame
in general. Two consequences shape the design. The spotter judges displacement in its
**own camera frame**, against its own first sighting, which needs no shared frame at all;
that is why a stationary spotter would be the honest judge on hardware too. And **frame
hints** (`HINT` messages) are the spotter's arena frame estimate of the target, which only
exist in the simulator where every robot knows its pose: a receiver uses one solely to
choose which way to turn before its own `search_scan`, every approach and every kick uses
the kicker's own camera. `flock.frame_hints: auto` turns them on only when every member
runs in `sim2d`; on hardware they are off.

## What the LLM does, and does not do

At most **one** model call per run: the planner may tune task parameters (target label,
approach distance, scan step, timeout) through a single forced tool call. Numeric
parameters are clamped into the schema's ranges, an invalid field is dropped on its own
(the valid ones survive), and a missing or broken call falls back to deterministic
defaults, logged. With `--provider fake` even that call is skipped and the plan is a pure
function. The auction, the roles and the steering are deterministic code. Per duck LLM
pilots and LLM negotiated bids are still deliberately out of scope: they would cost N
times the tokens and latency, and the demo does not need them to be honest.
`summary.json` records `planner.llm_calls` (0 or 1) as proof.

## Ground truth

The outcome is judged by the coordinator from sim telemetry (`ball_displacement_m`), not
from any model's claim. A member reporting a kick the world did not record turns the run
into a failure. Duck to duck safety separation is watched from world ground truth: while
a claim is live the coordinator measures every other duck's true distance to the kicker
and orders an intruder to retreat, with the motion still running through that duck's own
executor. The kicker's ball approach uses perception only, exactly like a solo run.

## Reading a flock run

```
runs/<timestamp>-flock-kick/
  flock.jsonl          # the coordinator's log: every bus message, auction, verb
  summary.json         # outcome, kicker, auctions, bids, planner proof, per duck rollup;
                       # with roles also robots, roles, assignments, spotter and verdicts
  run.gif              # world view | the claimant's own camera, with phase captions
  ducks/duck-0/        # per robot transcript.jsonl and frames/ (no summary.json on purpose)
```

Three annotated lines from a real `flock.jsonl`:

```jsonc
{"sim_t": 2.4, "kind": "bus", "msg": {"kind": "BID", "src": "duck-1", "ball_dist_m": 0.62}}
{"sim_t": 2.8, "kind": "auction_decision", "kicker": "duck-1", "bids": {"duck-1": 0.62}, "tie": false}
{"sim_t": 8.6, "kind": "bus", "msg": {"kind": "RESULT", "src": "duck-1", "status": "kicked", "ball_moved_m": 0.59}}
```

## Watching a flock run

A flock is traced like a solo run, and on by default. Each robot gets its own view with its
name on every line, so three robots moving at once stay three readable columns rather than
one interleaving, and the coordinator's own decisions print under `flock`.

```
duck-2  verb    search_scan(target='ball', step_deg=45, max_steps=3)
duck-2  ->      look(x=1, y=0, z=0)
duck-2  <-      search_scan ok: ball found: ball at bearing 28° left ~0.81 m (after 2 turn steps) (1.8 s sim, 0.1 s wall, 19 intents)
flock   auction first bid duck-2 0.81 m
flock   claim   duck-2 (0.81 m)
duck-0  <-      search_scan PREEMPTED: duck-0: role change to YIELD (2.2 s sim, 0.1 s wall, 23 intents)
duck-2  end     stopped after 5 steps
```

That is a real `--seed 3` run, trimmed. The duck that wins the claim keeps searching, and the
two that lose are preempted mid-verb and yield, which is the moment a flock is hardest to
read from `flock.jsonl` alone. `PREEMPTED` is its own outcome rather than an error, because
a role change is the coordinator working, not a fault.

The `auction`, `claim` and `verdict` lines are the words the GIF captions use, so a line on
screen and a frame in `run.gif` say the same thing about the same moment.

Each robot's `ducks/<name>/transcript.jsonl` is its own record and gets every event whether
or not anyone is watching, exactly as a solo run's transcript does. `flock.jsonl` keeps the
coordinator's story as it always has, under its own names, so nothing is written twice.
`quackd trace <run>` replays those records afterwards, one member's transcript in full
after another rather than interleaved, and it does not read `flock.jsonl`, so no `flock`
line appears in a replay.

With a real provider the planner's one model call is traced under `flock` and recorded
in `flock.jsonl` as `llm_request` and `llm`. With `--provider fake`
there is no call to trace: the planner short circuits before it reaches a model.

`--no-trace` or `QUACKD_TRACE=0` removes the views and leaves every record intact.

## The shared clock

Sim time is a shared resource: the world advances one tick only while every participant
(each duck and the coordinator) is asleep, and it freezes while anyone thinks. A slow LLM
therefore costs zero sim time, and with `--provider fake` and a fixed seed a flock run is
reproducible. Wall clock heartbeat scheduling is the one nondeterministic input, and it
only influences failure path timing, as in solo runs.

## Which robots can join

Flock mode knows the **Microduck**. Any other adapter is refused when the run starts,
with the names it does know. An Open Duck Mini cannot join a flock yet, and that is a
limit of `quackd/flock/runner.py`, not of the robot: extending it is future work rather
than a hardware problem.

## Status and future work

Sim only. Nothing multi robot has run on hardware, and the acoustic channel stays
theatrical (a quack marks the sighting; Wi Fi would carry the real data). One
choreography ships, `flock-kick` (ducks), 10 of 10 seeds with scripted pilots and ground
truth checks. An MQTT bus implementing the same `Bus` protocol exists
([lan.md](lan.md)), library only and tested on a fake broker; a flock across machines
also needs a clock across machines, which is future work, as are hardware flocks once
Microducks ship. See [adapter-status.md](adapter-status.md) for the wider honesty
table.
