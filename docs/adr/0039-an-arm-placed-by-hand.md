# ADR-0039: The one place quackd lets go of a robot

**Status:** accepted · **Date:** 2026-09-18 · Extends [ADR-0036](0036-what-the-arm-does-not-say.md) (which said quackd never disables torque) and [ADR-0012](0012-safety-executor.md) (the kill switch, and what a second Ctrl-C means) · Implemented in `quackd/agent/loop.py`, `quackd/safety.py` and `adapters/lerobot/` ([page](../adapters/lerobot.md), [checklist](../lerobot-hardware-checklist.md))

## Context

Every run of an SO-101 has started the same way since the arm had an adapter: quackd drives it
to the pose recorded with `quackd robot rest-pose`, and the pilot takes it from there. That is
the right default and it stays the default. It is also the wrong start for most of what an arm
on a bench is actually for. Drawing needs a pencil already in the gripper. Stacking needs a
block already held. Anything over a sheet of paper needs the arm leaning over the paper, not
folded beside it.

There were two ways to get there and both are bad. A person can record a different rest pose,
which means teaching the arm a number for a position they could reach out and set in two
seconds, and which then becomes the pose every later run and every teardown folds to. Or the
pilot can jog there through `move_joints`, which spends model calls and budget getting to the
place where the task begins, one [ADR-0036](0036-what-the-arm-does-not-say.md) step cap per
tick, and ends somewhere approximate anyway.

The obstacle is that an arm a person can move is an arm with no torque in it, and on a
position-controlled arm torque is the only thing holding it up.
[ADR-0036](0036-what-the-arm-does-not-say.md) closed its decision with the flat sentence that
quackd never disables torque, and meant it: every gap that ADR closed was a gap between an LLM
and a servo. This is not that. A person standing at the arm with both hands on it is a
different situation from a model three seconds away, and it is the only situation in which
letting go is a service rather than a drop.

## Decision

- **The recorded rest pose is the only place an arm is released, on exactly the condition
  `close()` already trusts.** `let_go()` refuses anywhere else, naming what is out of place
  (`the arm is not at its rest pose (...), and an arm held up by torque alone falls when
  torque goes`). The test is `at_rest`, which is the same read `close()` makes before it lets
  `disable_torque_on_disconnect` stay true: quackd has been dropping torque at that pose at
  the end of every ordinary run since [ADR-0036](0036-what-the-arm-does-not-say.md), so it is
  the one pose this repository already treats as safe to leave an arm in with nothing holding
  it. So `--by-hand` drives to the rest pose first and releases there, and the person lifts
  the arm out of a pose it was standing in on its own. An arm with no pose recorded is refused
  before anything connects.
- **The present position is written as the goal before torque comes back on, and again after.**
  `enable_torque()` writes one register and nothing else, so what a servo does with the goal it
  was last told is the firmware's business and is documented nowhere at the pin: that is the
  new UNVERIFIED ref `TORQUE_ENABLE_HOLDS_PRESENT`. It matters because the goal last written
  before a hand-off is the rest pose the arm has since been lifted out of by hand, so a servo
  that drives to it drives back to the fold with somebody's fingers in the way. Writing where
  the arm is now, first, makes both readings of the undocumented behaviour end in the same
  place. Writing it again, sleeping a tick and reading back is how `take_hold()` answers with
  what happened rather than with what should have: the assumption is relied on in neither
  direction.
- **A slip refuses the run and still reports the arm as being in nobody's hands.** Those are
  two different facts and the code keeps them apart. `_in_hand` is cleared the moment torque
  reads on, before the pose is compared at all; only then does `take_hold()` refuse a joint
  that moved more than `TOL_DEG` (5.0 degrees), saying which joint and by how much. An arm
  that sagged as torque came on is holding itself perfectly well, just not where it was put,
  and the run stops because the pilot was promised a starting pose nobody now has. Reporting
  that arm as limp in somebody's hands would send a person to cut the power on a robot that
  needs nothing.
- **Enter is read through the kill switch, never with an `input()` of its own.** The switch
  already runs the only thread quackd points at stdin. A second reader would race it for the
  same keystroke and whichever lost would sit forever on a line the other had taken. So
  `KillSwitch` grew `entered`, `pressed`, a `presses` count and `wait_for_enter()`, and
  `_watch_keys` now loops until stdin ends instead of stopping at the first `q`: a run that
  hands the arm over waits for a person after the abort flag may already be set, and waits
  again inside its own teardown, so the reader has to outlive both. The CLI's
  `_TerminalHandOff` prints a line and waits on the switch, and is bound to it after the loop
  exists, because the switch is built from the loop's own abort event.
- **The end-of-run wait watches a fresh key press rather than the abort flag.** The first wait,
  while somebody places the arm, watches `abort`, which is how a Ctrl-C rescues a run whose
  operator walked away. The second, between the run's `stop` and the fold, cannot: on every run
  a person ended, the abort flag is already set before that wait begins, and watching it would
  skip the question on exactly the runs most likely to still have something in the gripper. It
  waits on `pressed` instead, which every waiter clears on the way in, so a fresh Ctrl-C ends
  it and a stale one does not. It is bounded at `AgentLoop.HAND_BACK_S` (120 s), because a run
  must still finish when the room is empty: the cost of waiting is an energised arm and the
  cost of not waiting is a pencil driven into the bench as the arm folds.
- **The second Ctrl-C is caught in that one window and nowhere else.**
  [ADR-0012](0012-safety-executor.md) hands SIGINT back after the first press on purpose, so a
  human who is not convinced the first one worked is never holding a key quackd has swallowed,
  and a second press raises straight through the teardown. `_hand_back()` catches
  `CancelledError` and `KeyboardInterrupt` on its own wait, because there the press means
  "skip the question", not "abandon the arm energised with no record written". The gripper is
  left exactly as the run left it, the arm still folds to its rest pose, the transport still
  closes, and the record still ends with a `run_end` and a summary. It buys one press: a third
  lands somewhere without a guard and quits at once.
- **A stop picks the arm back up before it holds it.** Every teardown begins with a stop, so a
  stop is what a Ctrl-C during the hand-off wait actually reaches, and sending a goal to a limp
  servo is a stop that stopped nothing. `_hold()` therefore calls `take_hold()` first whenever
  the arm is in a hand, which re-energises it where the person has it so that the rest move
  after can put it down. Where a run ends limp anyway, `close()` prints the new `LIMP_IN_HAND`
  note instead of the torque note, because telling somebody holding a limp arm that it is
  holding itself up is the single wrong answer that gets an arm dropped.
- **No verb asks for any of this, and no model can.** `let_go` and `take_hold` are not verbs,
  are not reachable over MCP, and are deliberately not on the `RobotAdapter` protocol, which is
  runtime-checkable and structural: a method there is a method all seven bodies must carry, and
  six of them are never handed to a person. A body that can be declares
  `supports_hand_off = True`, and the loop reaches it through `let_go_if_any` and
  `take_hold_if_any`, the same way a rest move reaches a bare transport. Each is called once,
  in a fixed order, only on a run a person started with `--by-hand` at a terminal. The pilot is
  told where its run begins (`build_system_prompt(..., by_hand=True)`) and nothing more: it
  cannot ask for torque to come off any more than it could before this change.

## Consequences

- Five refusals fire before anything connects and before a run directory exists: a body that is
  not the arm, an arm with no rest pose recorded, several robots, `--dry-run` (which moves
  nothing at either end and is the opposite ask), and no terminal to ask on. Each names the
  command that fixes it. A run without the flag is unchanged, in its prompt, its trace and its
  teardown.
- The trace gains `hand_off`, with `stage` one of `released`, `held`, `skipped` or `unloaded`,
  so a transcript says which of the two waits a run got through. `_hand_over()` restarts the
  budget after the person is done, because an arm can stand waiting while somebody goes to find
  a pencil and that used to be spent out of the `max_minutes` the model gets for the task.
- [ADR-0036](0036-what-the-arm-does-not-say.md) is amended rather than reversed, and carries
  the amendment. What that ADR said quackd never does is now one clause shorter and no more:
  the adapter still never calibrates, still refuses an arm with no calibration file, and still
  keeps LeRobot's default of dropping torque on `disconnect()`.
- **What is now UNVERIFIED, and only an arm can settle it.** Every release quackd has performed
  was against `lerobot:mock`, which cannot sag, cannot slip and cannot be lifted. Four things
  need a bench: what a Feetech servo does with its goal on re-enable
  (`TORQUE_ENABLE_HOLDS_PRESENT`); whether an SO-101 at its recorded fold really does stand
  with torque off, which is the assumption the whole of the first decision above rests on; how
  far a hand-placed arm settles between `enable_torque()` and the read a tick later, which is
  what 5.0 degrees was guessed against; and whether a released arm is light enough to place
  one-handed while the other hand loads the gripper. The route is the one
  [ADR-0036](0036-what-the-arm-does-not-say.md) left its own numbers on: the hardware
  checklist's *What to report*, answered by somebody with the arm rather than by anybody with a
  command.
- **The two ends read the same silence in opposite directions, on purpose.** A torque register
  that does not answer straight after the release is treated as a release that took: the other
  reading ends with `close()` telling a person holding a limp arm that it is holding itself up.
  A torque register that does not answer after `enable_torque()` is treated as no hold at all
  and refuses, because the other reading ends with quackd telling that same person they can let
  go. In both the rule is the same one: pick the answer that does not get the arm dropped, and
  never the one that sounds more decisive. Each costs something. A corrupt packet at the
  release can report an arm released that is still energised; a corrupt packet at the hold ends
  a run that might have been fine. Both are deliberate.
- **One bad register cannot speak for the other.** The status registers are read in a `try`
  each. They shared one until an adversarial pass found what that cost: `take_hold()`'s torque
  check was written to stand down whenever a register read had failed, so a corrupt
  *temperature* packet switched off the check on *torque*, and an arm that ignored
  `enable_torque()` was reported as holding the pose a person had just set. The line they were
  about to read said they could let go.
