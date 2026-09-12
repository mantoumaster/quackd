# ADR-0019: `.duck` spec v1

**Status:** accepted · **Date:** 2026-09-02 · Implemented in Phase 1 of 0.4 ([design](../design/multi-robot.md))

## Context

ADR-0005 fixed `.duck` v0: an enforced YAML contract plus a Markdown body, strict parsing,
`duck: 0` as the only version. A multi-robot quackd needs a task to say which verbs it
*needs* (not only which it *allows*), which robots it expects, and, for a flock, which
roles exist and what each requires. The format name does not change: `.duck` is the format
the way `Dockerfile` is, whatever the robot.

## Decision

- **`duck: 1`** adds three top-level keys and two flock keys, all optional:
  - `requires: [verbs]`: the verbs the task needs. `quackd validate --robot <spec>` (one
    or more) checks them against each robot's manifest and reports field-level errors of
    the form `requires kick, but arm-01 (lerobot-so101) does not provide it`.
  - `robots:`: a string (`adapter:backend`, the solo default so `quackd run <duck>` needs
    no flags) or a mapping from flock member name to spec.
  - `flock.roles: {spotter: {requires: [...]}, kicker: {requires: [...]}}`. In 0.4 the
    role names are restricted to exactly `spotter` and `kicker`, because only those two
    behaviours exist in code; a requires-only role with no behaviour would be a fabricated
    capability. `count` is fixed at 1.
  - `flock.frame_hints: auto | on | off` (see ADR-0020).
- **`duck: 0` files parse and run unchanged.** Any v1 key under `duck: 0` is an error that
  names the fix (`requires needs duck: 1`). The only new rejections that can hit a v0 file
  are two contradictions no shipped duck contains: listing a verb and its alias in `allow`,
  and putting `stop` in `confirm`.
- **For `duck: 0`, `effective_requires` is `verbs.allow`.** A v0 task needs everything it
  allows; that is what makes `validate ducks/find-and-kick.duck --robot lerobot:mock`
  fail on `kick` with the wording above, and also makes `patrol-and-quack` fail on `walk`
  against a LeRobot arm, which is correct.
- Validation lives in one place, `quackd/duckfile/validate.py::validate_duck(duck,
  manifests)`, returning `Problem(field, robot, verb, message)` rows, and is shared by
  `quackd validate`, the MCP `robot_load_duckfile` tool and the flock runner.
- `validate` without `--robot` uses the duck's own `robots:` default, then the Microduck
  vocabulary, so CI's `quackd validate ducks/*.duck` keeps working for a starter with no
  `robots:` key, like `find-and-kick.duck`.
- The bundled starters stay `duck: 0`; only the two new ducks ship as v1.

## Consequences

- `quackd/duckfile/schema.json` is regenerated and the drift test covers the v1 keys.
- A `.duck` can be checked against a robot it has never run on, offline, which is the
  precondition for community ducks targeting robots the author does not own.
- Old quackd versions refuse v1 files (strict parsing), as they refuse `flock:` today; the
  spec documents this as intended.
