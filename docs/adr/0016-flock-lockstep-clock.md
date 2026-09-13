# ADR-0016: The flock clock is lockstep

**Status:** accepted, amended · **Date:** 2026-08-31

**Amended 2026-09-13 by [ADR-0034](0034-registered-robots-and-pilot-flocks.md):** this is
the *auction* flock's clock. A pilot flock has no `FlockClock`: its members are real or
simulated bodies on their own clocks, running concurrently in wall-clock time, so a slow model
costs real seconds there and a seed does not make the run reproducible. The "it may return as
a realtime mode" line below is what happened, for that kind only. Everything here still holds
for an auction, which is still the only flock with one shared world to advance.

## Context

In a solo sim run, whoever calls `transport.sleep()` steps the world. With N concurrent
member tasks that breaks: time would advance once per sleeper, in racy order, and
determinism under a seed (the repo's core promise) would be gone.

## Decision

`quackd/sim2d/clock.py` (`FlockClock`): sim time is a shared resource governed by a
barrier.

- Participants park with an **integer remaining-step count** (`round(seconds/DT)`),
  exactly reproducing the single-duck step arithmetic with no float drift.
- An advancer task steps the world one DT at a time **only while every registered
  participant is parked**, fires tick hooks per step, and wakes due sleepers in sorted
  participant order (deterministic).
- The world **freezes while anyone is awake**, so LLM latency costs zero sim time and the
  per-duck deadman keeps exactly its solo semantics.
- `unregister` (called by `transport.close()` and on member exit) cancels a parked future
  and re-evaluates the barrier, so a dead duck can never wedge time. Rule for
  participants, enforced by convention and tests: every await bottoms out in
  `clock.sleep`, and you unregister when you finish.
- The coordinator registers as participant `"coordinator"` with 0.05 s ticks, so bid
  windows, leases and the watchdog run on sim time.

The free-running alternative (a background task stepping continuously with command
latching) was analysed and rejected for v0.3: sim behaviour would depend on wall-clock
LLM latency, killing per-seed reproducibility, and there is no principled headless step
rate. It is the honest hardware-shaped model and may return as a `realtime` mode.

## Consequences

- `n_ducks=1` runs are bit-identical to the pre-flock code (verified by the unchanged
  seeded acceptance suite); flock runs with the fake planner are reproducible per seed.
- Wall-clock `Heartbeat` tasks interleave freely (the advancer yields to the loop every
  few steps); only failure-path timing is wall-clock dependent, as in solo runs.
- One subtlety, covered by tests: a participant that wakes and neither sleeps again nor
  unregisters freezes time for everyone. Members' idle roles are sleep loops on purpose.
