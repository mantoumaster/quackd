# Jev: an optional discrete stepper in front of the model

quackd asks one question a turn — *which single tool call now* — and pays a frontier model's
full latency for it whether the answer is `report_state` or a six-joint pose. On the SO-101 run
at the top of [README.md](../README.md) that is **62.1 seconds of a 78.8 second run** spent
waiting on the model, against 12.2 seconds of the arm actually moving.

Some of those turns are not writing. They are choosing.

[TypeSafe's **Jev**](https://docs.typesafe.ai/introduction) is a *System One* model: you send it
a named state and typed questions, and it answers with a value and a probability distribution.
No text generation, no parsing, nothing to coerce into JSON.

**It is not another LLM, and it is not a smaller one.** That is the whole point of it, so it is
worth being precise about the difference before any of the rest makes sense. A language model
generates tokens: you ask it for a verb and it writes one, and everything quackd does around
that — the tool schemas, the one-call-per-turn rule, the re-prompt when it answers with prose —
exists to squeeze a text generator into a shape software can branch on. Jev generates nothing.
It scores a fixed set of options against a state and hands back which one, with a calibrated
probability for each. In TypeSafe's own words, *"LLMs produce words for people. Jev produces
typed decisions"*, and it is *"more like code: reliable, fast, self-consistent, and type-safe"*.

| | A language model | Jev |
|---|---|---|
| Output | tokens, which you parse | a typed value, already a value |
| Can it invent an option? | yes, and does | no. It can only score the ones you gave it |
| Can it write a joint angle? | yes | **no**, and not by rule: there is nowhere in the answer for a number to come from |
| Can it be used as a chatbot? | it is one | no, at all |
| What uncertainty looks like | a hedge in prose, or none | a number, per option, which your code reads |
| Where it sits in quackd | the pilot | in front of the pilot, for the turns that are a choice |

The practical consequence is the one this page is about. A frontier model takes about six
seconds to say `report_state` on this arm, because it takes about six seconds to say anything.
A classifier answers a six-way choice in a fraction of one, and charges for the question rather
than for the essay.

`--jev` is **off by default**, and stays off unless you ask for it. quackd installs no TypeSafe
code, needs no TypeSafe key, and never switches this on because it found one in your `.env`.

> [!IMPORTANT]
> The stepper cannot author a number and cannot author a sentence. Every joint angle, every
> velocity, every `reason` and every feasibility verdict is still the model's, on every body.
> What it can do is pick one of a handful of calls this robot already has.

**Contents**

- [What it is, and what it is not](#what-it-is-and-what-it-is-not)
- [Install and switch on](#install-and-switch-on)
- [The SO-101 arm, in four parts](#the-so-101-arm-in-four-parts)
  * [1. What the arm can and cannot hand over](#1-what-the-arm-can-and-cannot-hand-over)
  * [2. The hero run, call by call](#2-the-hero-run-call-by-call)
  * [3. Where the arm does hand over: the grip check](#3-where-the-arm-does-hand-over-the-grip-check)
  * [4. The other six bodies](#4-the-other-six-bodies)
- [How much faster, and how much cheaper](#how-much-faster-and-how-much-cheaper)
- [How a turn is decided](#how-a-turn-is-decided)
- [What it is never allowed to do](#what-it-is-never-allowed-to-do)
- [What the record says](#what-the-record-says)
- [Measuring it yourself](#measuring-it-yourself)
- [Configuration](#configuration)
- [Limitations](#limitations)

## What it is, and what it is not

| | |
|---|---|
| **It is** | a classifier that answers typed questions about a state: which of these calls, is this statement true, where on this rubric |
| **It is not** | a provider. `--provider` does not take it, `quackd list-models` does not list it, and it cannot pilot a robot on its own |
| **It decides** | which single verb, out of the ones this body's contract allows *and* whose parameters are a closed set |
| **The model decides** | every pose, every target, every sentence, every feasibility verdict, every `declare_success`, and every turn the stepper is not confident about |
| **The executor still decides** | whether the verb runs at all: the allowlist, the budgets, the confirm gates, the preconditions and the robot's own safety authority are untouched |

The division is the same one TypeSafe's own [smart-home demo](https://docs.typesafe.ai/demos/smart-home)
draws: the classifier picks the device and the action, and the language model does the language.
Here the classifier picks the verb and the model does the angles and the prose.

## Install and switch on

```bash
uv pip install "quackd[jev]"                     # the SDK on its own
uv pip install "quackd[lerobot,jev,openai]"      # the arm, the stepper and a model
```

Then a key, in the environment or a `.env` (see [`.env.example`](../.env.example)):

```
TYPESAFE_API_KEY=...
```

`quackd doctor` says whether both arrived:

```
discrete stepper (quackd run --jev; off unless you ask for it) ----------------
+-----------------------------------------------------------------------+
| stepper | extra                 | key                    | model      |
|---------+-----------------------+------------------------+------------|
| jev     | missing (quackd[jev]) | TYPESAFE_API_KEY unset | jev-1.13.0 |
+-----------------------------------------------------------------------+
It answers the turns that are a choice among calls this body can make. Every
pose, every sentence and every verdict is still the model's (docs/jev.md).
```

Three modes:

| `--jev` | What happens |
|---|---|
| `off` | the default, and quackd exactly as it has always been. No stepper is built and `typesafe_sdk` is never imported |
| `shadow` | it is asked on every turn and its answer is recorded beside the model's. **The run is unchanged**: the model still decides everything. This is the mode to start in, and the one that measures |
| `on` | it takes the turns it is confident about, and hands back the rest. Switching it on prints one line saying it has never been measured against a robot, because that is still true |

`QUACKD_JEV` does the same when the flag is absent; the flag wins.

**Asking for a stepper that cannot run does not stop the run.** The stepper is an optimisation
and the model is the pilot either way, so a script that always passes `--jev on` still drives
the robot on a machine with no key. It says so once, before anything is connected, and carries
on without it:

```
$ quackd run arm-grip-check --robot lerobot:mock --provider fake --jev shadow
! --jev shadow asked for, running without it: the stepper needs the optional extra quackd[jev] — run: uv pip install "quackd[jev]"
```

A mode nobody defined is a different matter, because that is a typo rather than a missing
install, and it stops the run:

```
$ quackd run arm-grip-check --robot lerobot:mock --provider fake --jev maybe
✗ error: unknown --jev mode 'maybe'; choose one of off, shadow, on
```

## The SO-101 arm, in four parts

The arm is the only body in this repository that has run on real hardware, so it is where this
has to be explained and where it has to be right.

> **On the SO-101, Jev reads the arm and works the gripper. Every angle is still the model's.**

### 1. What the arm can and cannot hand over

| | Tools | Concrete calls | Who answers |
|---|---|---|---|
| **A choice** | `report_state`, `stop`, `place`, `gripper`, `observe`¹ | **6** — `report_state`, `stop`, `place`, `gripper(open=true)`, `gripper(open=false)`, `observe` | the stepper, or the model |
| **A number** | `move_joints`, `pick` | — | **always the model** |
| **A sentence** | `assess_task`, `declare_success`, `declare_failure`, `remember` | — | **always the model** |

¹ `observe` is in the manifest only when a camera is configured, which is why
[`lerobot-lookout`](../ducks/lerobot-lookout.duck) leaves it out.

Nothing here is a special case for the arm. A verb is a **choice** when every parameter it has
is a closed set — an enum, a constant, or a boolean — or is *inert*, meaning it is optional,
defaults to null, and accepts null, so leaving it out chooses nothing. Everything else is a
number, and a number is the model's.

That one rule is why `move_joints` can never be the stepper's, on this arm or the two other
bodies that have one. Its `positions` is a required object. And there is a second reason that
is stronger than the first: **the joint names are not in the schema at all.** They live in a
validator, so there is nothing for a classifier to enumerate even in principle.

### 2. The hero run, call by call

The wave at the top of the README, exactly as it happened on 2026-09-15:

| Call | Tool | Whose turn this would be |
|---|---|---|
| 1 | `report_state` | **the stepper's** |
| 2 | `assess_task` | the model's — it needs a written reason |
| 3–8 | `move_joints` ×6 | the model's — every one is an angle |
| 9 | `stop` | **the stepper's** |
| 10 | `declare_success` | the model's — it needs a written reason |

**Two of the ten.** Six of that run's calls author joint angles and can never be the stepper's,
and two more are prose. That is the honest arithmetic on the one run this project has on real
hardware, and it is worth stating plainly rather than choosing a friendlier example.

The stepper is asked on every turn, so this run pays for ten questions to save two model calls.
That still comes out ahead, and by a good margin, but it is the least favourable task in the
repository and that is the reason to lead with it. What it works out to is in
[How much faster, and how much cheaper](#how-much-faster-and-how-much-cheaper).

### 3. Where the arm does hand over: the grip check

[`ducks/arm-grip-check.duck`](../ducks/arm-grip-check.duck) is the same arm and the opposite
shape: read the state, shut the gripper, read it again, release, stop. Every turn is a choice,
and nothing in it authors an angle, because `move_joints` is deliberately not in its allowlist.

```bash
quackd run arm-grip-check --robot lerobot:mock --jev on                  # no arm needed
quackd run arm-grip-check --robot arm-01 --by-hand --jev on              # with something in the gripper
```

Here is a real run of it on the mock arm, with a stub standing in for Jev, trimmed to the lines
that matter:

```text
step 0/12, llm calls 0/12, 0.0/3 min ------------------------------------------
   jev     report_state 0.97 >= 0.60 (0.00 s)
>  verb    report_state() from jev
+  result  report_state ok: shoulder_pan 0, shoulder_lift -90, ...; torque on;
           hottest shoulder_pan 30 degC; holding nothing (0.0 s, 0 intents)
step 1/12, llm calls 0/12, 1 by the stepper, 0.0/3 min ------------------------
   obs     While you were not asked, the stepper chose these (newest last):
           - report_state(): ok - ... holding nothing
   jev     escalate, to the model (0.00 s)
   llm>    step 1: 1 messages (0 with image) to fake scripted:arm-grip-check
   tool    assess_task(verdict='feasible', reason='...')
+  assess  feasible: ...
step 1/12, llm calls 1/12, 1 by the stepper, 0.0/3 min ------------------------
   jev     report_state 0.95 >= 0.60 (0.00 s)
>  verb    report_state() from jev
```

Four things in that are worth reading twice.

- **`from jev`** on the verb line. Who chose a verb is on the record, in the transcript and on
  the terminal, for every call.
- **`0.97 >= 0.60`** is the confidence against the floor for that class of verb. A read answers
  to a lower floor than a move; see [How a turn is decided](#how-a-turn-is-decided).
- **`escalate, to the model`** on the second turn. The stepper wanted the gripper, but no
  feasibility verdict had been recorded yet, so the gripper was not among the calls it was
  offered. It handed the turn back and the model recorded the verdict. It was not refused by a
  gate — it was never offered the option.
- **`While you were not asked, the stepper chose these`**. The model's history contains nothing
  the stepper did, because none of it is anything the model said. It is told instead, once, in
  the observation it is next shown, and told who chose them.
- **`0.00 s`** is a stub answering instantly in a test, not a measurement. See the note above.
  A run made today puts two more figures in that parenthesis, the tokens the question spent and
  what they cost; see [Measuring it yourself](#measuring-it-yourself). This transcript was
  recorded before quackd counted either, and replays exactly as it was written.

### 4. The other six bodies

The same loop and the same rule, with no per-body code anywhere.
[`microduck-lookout`](../ducks/microduck-lookout.duck) is the most discrete task quackd ships:
`gaze` in five directions, `observe`, `report_state`, `quack` and `stop` are all choices, and
only `say` is not, because it needs a sentence. `open-duck-lookout` is the identical allowlist
on an Open Duck Mini.

A body whose verbs are all numbers — a cart driving to a pose, an arm moving joints — escalates
every turn and the stepper costs it one question a step and nothing else. That is expected, not
broken, and the log says `not_offered` for it without a request being made at all.

## How much faster, and how much cheaper

> [!IMPORTANT]
> **Everything in this section is an estimate, and nobody has run Jev against a quackd robot.**
> It combines two measured things with one published one: quackd's own timings from the SO-101
> run on 2026-09-15, the size of the request quackd actually builds (measured against the mock
> arm), and TypeSafe's published per-call latency and price. Every input is named below so you
> can disagree with any of them. `--jev shadow` replaces the whole section with measurements
> from your own bench, and a number from there is worth more than this arithmetic.

### What the vendor claims, and what to do with it

TypeSafe's front page says **193.6× faster and 444.6× cheaper** than an LLM on System One tasks,
alongside a worked example of 0.114 s against 8.566 s and $0.000081 against $0.013880. Those two
things do not agree with each other: the example divides out to 75× and 171×, not 194× and 445×.
So the headline is a range over tasks rather than a constant, and it is theirs rather than
quackd's. Nothing in this repository repeats it as a quackd measurement.

What is usable is the underlying pair, which is consistent across their pages:

| Figure | Source |
|---|---|
| **0.114 s** per call | [typesafe.ai](https://typesafe.ai/), their worked example |
| **$0.042 per million input tokens**, output not charged | [their models page](https://docs.typesafe.ai/models) |
| **238× lower input price** than a frontier model | [typesafe.ai](https://typesafe.ai/), against Claude Fable 5.1 |

And what quackd brings to it, all measured:

| Figure | Source |
|---|---|
| **6.21 s** mean model call | the wave run: 62.1 s over 10 calls ([README](../README.md#what-happened-in-that-run)) |
| **49,096** input tokens over those 10 calls | the same run |
| **527 tokens** per Jev request on this arm | measured by hand before the estimator existed: 388 characters of state plus 1,721 of questions, on `lerobot:mock`. The shipped estimator counts the questions at 1,299 characters on that arm before the verdict clears and 1,454 after, because the criteria are one line per verb on offer, so it prints 421 and 460 where this row says 527 |

### One decision

On the turns it can answer, quackd's own baseline gives **6.21 s against 0.114 s, about 54×**.
That is lower than TypeSafe's headline because quackd's baseline is a slower model on a bigger
prompt, and it is the number that matters here.

On price, Jev's whole request is 527 tokens against a model call that averaged 4,910 input
tokens on that run, at a 238th of the price per token. That works out at **about a
two-thousandth of the cost of the call it replaces**, which is a wider gap than TypeSafe's own
445× rather than a narrower one, and for a reason worth knowing: quackd hands its model a large
prompt, nearly five thousand tokens a call once the contract, the datasheet, the memory and the
observations are in it, while the question it hands Jev is about a tenth of that. The stepper is
cheaper per token *and* asked a much smaller question.

That ratio assumes the model's input is priced like Claude Fable 5.1, which is the comparison
TypeSafe's own 238× is drawn against. `gpt-6-astra` drove the wave run, and the catalogue now
prices it at $10 per million input tokens, the same input rate as Claude Fable 5.1, so the
comparison holds for the model that actually did the driving. The token counts on both sides
are measured and both rates are the vendors' own, but a rate read off a page by hand on one day
is not an invoice: substitute yours where it differs.

### One run: it depends entirely on how many turns are a choice

This is the part a headline multiplier cannot tell you, and it is the honest centre of the
question. Jev is asked on every turn and only answers some of them, so the saving over a whole
run is capped by that share. With *f* the fraction of turns it answers, *M* the model's mean
call and *L* the stepper's, the think time goes from `N·M` to `N·L + (1−f)·N·M`, so:

**speedup ≈ 1 / (1 − f + L/M)**, and with quackd's numbers *L/M* is 0.018.

| Turns the stepper answers | Think time | Model spend |
|---|---|---|
| 20% | **1.2×** faster | about 20% less |
| 30% | **1.4×** faster | about 30% less |
| 50% | **1.9×** faster | about 50% less |
| 67% | **2.9×** faster | about 67% less |
| 80% | **4.6×** faster | about 80% less |

Two real runs, to put a number on *f* rather than guess at one. Driven on `lerobot:mock` with
the scripted pilot and a stub in Jev's place, `arm-grip-check` answered **4 of its 6 turns**
with the stepper and `lerobot-lookout` **3 of 6**, which is 67% and 50%. Neither reaches higher,
and on a short task neither can: the feasibility verdict and the closing `declare_success` are
sentences, so they are always the model's, and on a six-turn run that is a third of it before
anything else is counted. The share climbs with the length of the task, which is the opposite
of the usual intuition about where an optimisation pays.

The `L/M` term is what the stepper costs on the turns it *cannot* answer, and at 0.018 it is
almost nothing: even if Jev answered no turn at all, a run would only be about 2% slower. That
is the asymmetry the whole design rests on. Being wrong about a turn is cheap, and being right
is worth six seconds.

### The wave, end to end

The least favourable task here, worked through:

| | Measured, no stepper | Estimated, `--jev on` |
|---|---|---|
| Model calls | 10 | 8 |
| Stepper calls | 0 | 10 |
| Time spent thinking | 62.1 s | **≈ 49.2 s** |
| Whole run | 78.8 s | **≈ 65.9 s**, about 16% shorter |
| Model input tokens | 49,096 | roughly a fifth fewer |
| Stepper input tokens | 0 | ≈ 5,270, costing about 0.05% of what the model does |

The two calls that change hands are call 1 (`report_state`, which took 8.2 s) and call 9
(`stop`, 5.8 s). Ten stepper questions at 0.114 s add 1.14 s, so the net is about 12.9 seconds
off a 78.8 second run. On `arm-grip-check`, where two turns in three were a choice in the run
above, the same arithmetic gives close to three times less waiting.

### What would make this wrong

- **Jev's real latency against a robot's state.** 0.114 s is TypeSafe's figure on TypeSafe's
  task. quackd sends a different shape of request from a different network. The per-call timeout
  is one second, so the worst case is bounded, but the worst case is also where the saving goes.
- **How often it is confident enough.** Every turn below its floor escalates and costs the extra
  question with no saving. The tables above assume the turns it answers are the turns it can
  answer, which `--jev shadow` is how you find out.
- **Your verb mix.** `f` is the whole story and it is a property of the task and the body, not
  of Jev. An arm doing poses is a different number from an arm checking a grip.
- **Prices move.** Both sides of the ratio are somebody's rate card on a particular day.

## How a turn is decided

One request per turn carrying four questions. They run in parallel and in isolation, so the
fourth is nearly free:

| Question | Type | What it asks |
|---|---|---|
| `next_verb` | Choice | which single call now, out of this turn's options plus `escalate` |
| `done` | Noul | has everything under `success_when` already happened |
| `need_human` | Noul | must a person decide before anything else moves |
| `feasible` | Choice | can this body do this task at all — **recorded and never acted on**, see below |

`escalate` is always on the list. Without it a Choice always returns *something*, and the
confidence floor would be the only thing between "none of these is right" and a servo.

The gates are read in this order, and the two Nouls come first so a stepper that thinks the job
is finished never moves anything else:

1. any of the three probabilities is not a number at all → the model's turn. NaN is not a
   low confidence, it is no confidence, and it loses every comparison it is put through:
   `nan < 0.85` is False, so an unguarded one would clear a floor rather than miss it
2. `done` ≥ 0.5 → the model's turn
3. `need_human` ≥ 0.5 → the model's turn
4. the answer is `escalate`, or is not one of the calls offered this turn → the model's turn
5. the answer repeats the call the stepper made last turn → the model's turn
6. the stepper has answered 8 turns running → the model's turn
7. confidence is below the floor for that verb's class → the model's turn
8. otherwise, it is taken

An answer the router cannot read at all — a confidence that is a word, a `probabilities` that is
a list — costs that turn and nothing more. Reading the answer is part of the call, so it fails
the way the call does: the error is recorded and the model takes over.

A Noul carries no confidence, so 0.5 on the two of them is a raw probability meaning "more
likely than not". Escalating when the job is not in fact done costs one model call; not
escalating when it is costs a robot that carries on working after the task is over.

**Rules 4 and 5 are why a run always reaches the model.** Only the model can record a verdict,
declare an outcome or write a note, so a run that never reaches it can only end on a budget —
which is exactly what happened the first time this was tried: a stepper answering `report_state`
at 0.99 took all twelve turns of `lerobot-lookout` and the run died with nothing declared. A
reflex that fires twice identically is looping, not deciding.

The floors, every one of them a number [TypeSafe publish](https://docs.typesafe.ai/confidence):

| Verb class | Floor | Why that number |
|---|---|---|
| `stop` | **0.50** | the lowest in the system, on purpose. Below 0.5 is "genuinely unsure" in TypeSafe's words, and that is exactly where an unsure stepper should still be allowed to reach for the brake: a wrong `stop` costs one step, a wrong anything-else costs a move nobody chose |
| a read (`report_state`, `observe`, `introspect`) | **0.60** | sends no intent at all. TypeSafe's own universal floor for a cheap action |
| anything that sends an intent | **0.85** | their high-stakes number. Not 0.9, because their 0.9 is paired with "proceed with confirmation" and quackd expresses confirmation separately |
| a confirm-gated verb | **0.90** | literally their ">0.9, high stakes, proceed with confirmation" — and quackd's own confirm gate still asks a person on top of it |
| a `dangerous` verb | never offered | not a floor, a refusal |

TypeSafe say plainly that the right thresholds depend on your domain and your data. These are a
starting point, and `--jev shadow` is how you earn the right to move them.

**Why `assess_task` stays the model's.** `feasible|infeasible|uncertain` is a textbook Choice,
and the stepper is asked it on every turn — but its answer is only recorded, never acted on.
The verdict tool needs a written `reason`, which Jev does not write, and it needs `needs`, which
[`own_sheet_objection`](../quackd/verdict.py) reads to refuse a `feasible` verdict that names a
figure the datasheet does not publish. A verdict with an empty `needs` would pass that guard
trivially and silently disable it. And when a pilot answers `uncertain`, quackd puts its reason
to the person in the room; an empty question is not a question. Recording the Choice beside the
model's real verdict is the cheapest way to find out whether a future version should check the
model's verdict rather than write it.

## What it is never allowed to do

Four of these are structural rather than enforced, which is the stronger kind.

- **Author a number.** Not by rule but by construction: a verb with a free number in its schema
  is never a label, so there is no value for the stepper to choose.
- **Author a sentence.** Same reason. Every meta tool takes a required free string.
- **End a run.** `declare_success`, `declare_failure` and `assess_task` are all sentences, so
  the stepper cannot call any of them. Every ending goes through the model or through a budget.
- **Write to memory, or speak to a flock.** `remember` and `tell` are sentences too.
- **See anything.** Jev is documented as text only, so no camera frame ever reaches it. A turn
  that needs eyes escalates rather than guessing.
- **Get past a gate.** The allowlist, the budgets, the confirm gates, the preconditions and the
  robot's own safety authority are exactly what they were. A `.duck` binds the stepper the way
  it binds the model, because it binds the executor and both of them go through it.

> [!WARNING]
> None of this has been run against a real robot. The stepper has been exercised against the
> mock arm, the simulators and the test suite with a stub standing in for TypeSafe, and it has
> never driven hardware. That is the same standing as the rest pose and `--by-hand`
> ([README](../README.md#which-robots-work)).

## What the record says

Two new kinds in `transcript.jsonl`, both described in
[architecture.md](architecture.md#transcript-format).

`jev`, one per turn the stepper was asked, **identical in both modes** so the rows can be read
against each other: the labels it was offered, the one it chose, the whole probability
distribution, its confidence, the floor that applied, which gate fired, both Noul values, how
long it took, how large the state was and what was trimmed to fit, and, on the turns that
actually reached the network, what the question spent and what that cost.

`jev_shadow`, only in shadow mode and only after that step's `llm` record: what the stepper
would have chosen beside what the model actually chose, whether they agree, whether the stepper
cleared its floor, and what each of them cost. Agreement is about the whole call rather than
the verb's name, because `gripper(open=true)` and `gripper(open=false)` are opposite
instructions that share a word, and on `arm-grip-check` that is most of what there is to
compare. `same_verb` records the coarser reading beside it.

`summary.json` grows a `jev` block when there was a stepper, and nothing when there was not.
The budget line in every observation grows one clause once the stepper has answered:

```
step 2/12, llm calls 1/12, 2 by the stepper, 0.0/3 min
```

## Measuring it yourself

Shadow mode is the benchmark harness. It changes nothing about a run, so it is safe on
hardware, and it records the per-call token figures the hero run never kept.

```bash
# the grip loop on the mock arm: every turn is a choice, and no arm is needed
quackd run arm-grip-check --robot lerobot:mock \
  --provider openai --jev shadow --runs-dir runs/bench --no-log

# the same run with no stepper, as the baseline to read it against
quackd run arm-grip-check --robot lerobot:mock \
  --provider openai --runs-dir runs/bench --no-log

# the README's wave, shadowed: the counter-case, on the arm that ran it
quackd run --goal "Wave to the camera with an extended arm" --robot arm-01 \
  --max-steps 10 --jev shadow --runs-dir runs/bench
```

`--no-log` on the first two is about what you watch rather than what is kept. It stops the run
narrating itself on stderr, which is what you want when the point is the rows rather than the
watching, and it shortens `terminal.txt` in the run directory the same way, because that file
is the screen. `transcript.jsonl` and `summary.json` are written in full either way, and every
figure below is read from one of those two.

Then, per run directory:

| What | Read | From |
|---|---|---|
| **Jev's latency** against a robot's state, rather than TypeSafe's own 0.114 s on TypeSafe's own task | mean and max of `latency_s` | every `{"kind":"jev"}` |
| **State size** | `state_chars`, `state_tokens_est`, `trimmed` | every `{"kind":"jev"}` |
| **Coverage** | how many `gate` values are `taken`, out of all of them | every `{"kind":"jev"}` |
| **Agreement** | how often `agree` is true, and separately among rows where `would_have_acted` is true | every `{"kind":"jev_shadow"}` |
| **Calibration** | `confidence` bucketed against `agree` — the plot that earns the right to move a floor | join the two kinds on `step` |
| **What a stepper turn cost** | `usage` and `cost_usd`, with `usage_estimated` for whether that token count came from TypeSafe or from quackd's own arithmetic | every `{"kind":"jev"}` |
| **What the model cost** | `latency_s`, `usage`, `cost_usd` | every `{"kind":"llm"}` |
| **The two bills, on one turn** | `llm_cost_usd` beside `jev_cost_usd`: the ratio the section above could only reach by hand, now on the record for the turn that produced it | every `{"kind":"jev_shadow"}` |
| **The rollup** | `steps`, `llm_calls`, `elapsed_s`, `wall_s`, `usage`, `cost_usd`, `jev`, and `command` and `version` beside them, so a bench directory says which flags and which quackd produced the row rather than leaving it to your notes | `summary.json` |
| **The stepper's rollup** | `usage`, `cost_usd`, `cost_estimated`, and `price`, which is the rate that run was actually costed at rather than whatever the rate is when you read it back | the `jev` block of `summary.json` |

**quackd prices a stepper turn now, and marks the ones it had to guess at.** TypeSafe publish
Jev at **$0.042 per million input tokens, with output not charged**
([models](https://docs.typesafe.ai/models), re-read 2026-09-21), and that is the rate a turn is
costed at unless `QUACKD_JEV_PRICE` says otherwise. The turns that never reach the network,
`not_offered` and `state_too_large`, are charged nothing and carry no figure at all.

Which half of that is measured and which half is arithmetic is never left to be inferred. The
rate is TypeSafe's and published. The token count under it is measured only where their API
reports one: where it does not, quackd estimates the request as the state plus the questions at
four characters to the token, and flags the estimate three times over, with `usage_estimated`
on the turn, `cost_estimated` on the run's `jev` block, and a `~` in front of both the tokens
and the money on the log line and in front of the run's cost on the verdict panel.

That path is not a hypothetical. The SDK types both counts on `SystemOneResponse.usage` as
`int | None`, documented as "when the API did not report it", and a call that raised after its
request had already left the machine reports nothing at all while quite possibly still being
billed, so quackd charges that turn at the estimate rather than at nothing.

Here is the estimated reading, on the mock arm with a stub in Jev's place and no count coming
back from it. The `~` is the whole of the difference, and it sits on both numbers:

```text
   jev     report_state 0.97 >= 0.60 (0.00 s, ~992 tok ~$0.000042)
```

That is 992 tokens rather than the 527 quoted further up, and the arithmetic is on the record:
this turn's state was 2,669 characters and its questions 1,299, which is `(2669 + 1299) // 4`.
Both halves moved. The state grew because a real turn carries a real reading, and the questions
are smaller than the 527 row's 1,721 because that row was measured by hand before this code
existed and because the criteria are one line per verb on offer, which is two here and five
once the verdict clears. The record keeps `state_chars` beside the estimate for exactly that
reason: an estimate you cannot re-derive is a number you have to take on faith.

The model's half of the bill comes from quackd's own price catalogue, or from `--price` where
your rate is negotiated or your model is not in it. The two halves are kept apart **in the
record**, which is where the ratio this section exists to answer is read from: `cost_usd` on
every `llm` row against `cost_usd` on every `jev` row, and both of them on one `jev_shadow` row
for the same turn. The verdict panel is the one place they are added together, because a person
watching a run wants what it cost rather than a division; the exception is a model quackd has
no rate for, where the panel says `cost unpriced (stepper ~$0.0002)` rather than throw away the
half it does know.

None of this has been run against TypeSafe, so the measured path has never returned a real
token count here: every stepper figure in this repository came either from the estimate or from
a stub answering in its place. If you run any of it, the numbers are worth an issue.

## Configuration

| What | How |
|---|---|
| Mode | `--jev off\|shadow\|on`, or `QUACKD_JEV`. The flag wins. Default `off` |
| Key | `TYPESAFE_API_KEY`, in the environment or a `.env` |
| Model | `TYPESAFE_DEFAULT_MODEL`, default `jev-1.13.0`. Pinned rather than `jev-latest`, because a run whose stepper changed under it is a run whose transcript describes a model that is no longer the one that answered |
| Endpoint | `TYPESAFE_BASE_URL`, if it is not TypeSafe's own |
| Price | `QUACKD_JEV_PRICE`, written `in=0.042,out=0` in USD per million tokens, which is the same form `--price` takes for the model. quackd's own variable rather than the SDK's, and spelled that way on purpose: everything beginning `TYPESAFE_` is read by `typesafe_sdk` itself and quackd only ever checks that the key is there, while this one quackd reads and applies. Unset, a turn is costed at TypeSafe's published rate, and either way the rate used is written into the run |
| Install | `quackd[jev]`, which is not part of `quackd[all]` |

`--jev` is on `quackd run` and nowhere else. It is deliberately not on `quackd record`, which
makes the README's recordings and has to stay reproducible without a network call, and not on
`quackd serve-mcp`, where the model is the client and quackd has no think path to sit in front
of.

## Limitations

- **Text only.** No image, audio or video input ([models](https://docs.typesafe.ai/models)). A
  turn that needs to look escalates.
- **Hosted, and early access.** It is a network call in the middle of a robot's decision loop.
  It has a one-second timeout and one retry, and a failure costs that turn and never the run:
  the error is recorded and the model takes over.
- **English first.** Other languages are supported with lower accuracy, and a `.duck` written in
  one would be worth measuring before trusting.
- **Not the fast loops.** The steering loop runs at 10 Hz and the robot's own controllers faster
  than that. This sits in the slow loop, beside the model, and nothing about it changes what
  stops a body ([safety.md](safety.md)).
- **Confidence is calibrated over groups, not promised per answer.** A 0.93 is not a promise
  about that one answer; it is a statement about how a population of 0.93s behaves.
- **Unmeasured here.** No agreement rate, no calibration curve, no hardware run. The speed and
  cost section is an estimate built from TypeSafe's published figures and quackd's own measured
  ones, and it is labelled as one. Nothing on this page is a measurement of Jev driving a robot,
  because nobody has done that yet.

## Further reading

TypeSafe: [introduction](https://docs.typesafe.ai/introduction) ·
[System One](https://docs.typesafe.ai/concepts/system-one) ·
[state](https://docs.typesafe.ai/concepts/state) ·
[Choice](https://docs.typesafe.ai/primitives/choice) ·
[Noul](https://docs.typesafe.ai/primitives/noul) ·
[confidence](https://docs.typesafe.ai/confidence) ·
[speculative fan-out](https://docs.typesafe.ai/patterns/fan-out) ·
[confidence-gated routing](https://docs.typesafe.ai/patterns/confidence-routing) ·
[the smart-home demo](https://docs.typesafe.ai/demos/smart-home)

quackd: [architecture](architecture.md) · [safety](safety.md) ·
[the LeRobot arm](adapters/lerobot.md) · [ADR-0040](adr/0040-a-discrete-stepper-in-front-of-the-model.md)
