# Jev: an optional discrete stepper in front of the model

quackd asks one question a turn — *which single tool call now* — and pays a frontier model's
full latency for it whether the answer is `report_state` or a six-joint pose. On the SO-101 run
at the top of [README.md](../README.md) that is **62.1 seconds of a 78.8 second run** spent
waiting on the model, against 12.2 seconds of the arm actually moving.

Some of those turns are not writing. They are choosing.

[TypeSafe's **Jev**](https://docs.typesafe.ai/introduction) is a *System One* model: you send it
a named state and typed questions, and it answers with a value and a probability distribution.
No text generation, no parsing, nothing to coerce into JSON. It is not a smaller chatbot and it
cannot be used as one — it writes nothing at all.

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
| `on` | it takes the turns it is confident about, and hands back the rest |

`QUACKD_JEV` does the same when the flag is absent; the flag wins. Asking for a stepper that
cannot run is refused before the robot is connected, not four seconds into a serial handshake:

```
$ quackd run arm-grip-check --robot lerobot:mock --provider fake --jev on
✗ error: --jev on: the stepper needs the optional extra quackd[jev] — run: uv pip install "quackd[jev]"
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

It is also worth stating what it would cost. The stepper is asked on every turn, so a run like
this pays for ten questions to save two model calls. With *M* the model's mean answer time
(6.2 s on that run) and *L* the stepper's:

> On the wave, the stepper is a net loss unless **L < 1.24 s**.

That is why the per-call timeout is one second, and why `--jev shadow` exists and ships ahead of
`--jev on`: you measure before you switch it on.

> [!NOTE]
> TypeSafe publish no latency figure anywhere, and no one has run Jev against this arm. So this
> page states the arithmetic and not a result. [Measuring it yourself](#measuring-it-yourself)
> is how the blank gets filled, and a number from your bench is worth more than one from ours.

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

- **`from jev`** on the verb line. Who chose a verb is on the record, in the transcript and in
  the trace, for every call.
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

### 4. The other six bodies

The same loop and the same rule, with no per-body code anywhere.
[`microduck-lookout`](../ducks/microduck-lookout.duck) is the most discrete task quackd ships:
`gaze` in five directions, `observe`, `report_state`, `quack` and `stop` are all choices, and
only `say` is not, because it needs a sentence. `open-duck-lookout` is the identical allowlist
on an Open Duck Mini.

A body whose verbs are all numbers — a cart driving to a pose, an arm moving joints — escalates
every turn and the stepper costs it one question a step and nothing else. That is expected, not
broken, and the trace says `not_offered` for it without a request being made at all.

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

1. `done` ≥ 0.5 → the model's turn
2. `need_human` ≥ 0.5 → the model's turn
3. the answer is `escalate`, or is not one of the calls offered this turn → the model's turn
4. the answer repeats the call the stepper made last turn → the model's turn
5. the stepper has answered 8 turns running → the model's turn
6. confidence is below the floor for that verb's class → the model's turn
7. otherwise, it is taken

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
[architecture.md](architecture.md#transcript).

`jev`, one per turn the stepper was asked, **identical in both modes** so the rows can be read
against each other: the labels it was offered, the one it chose, the whole probability
distribution, its confidence, the floor that applied, which gate fired, both Noul values, how
long it took, how large the state was and what was trimmed to fit.

`jev_shadow`, only in shadow mode and only after that step's `llm` record: what the stepper
would have chosen beside what the model actually chose, whether they agree, whether the stepper
cleared its floor, and what each of them cost.

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
  --provider openai --jev shadow --runs-dir runs/bench --no-trace

# the same run with no stepper, as the baseline to read it against
quackd run arm-grip-check --robot lerobot:mock \
  --provider openai --runs-dir runs/bench --no-trace

# the README's wave, shadowed: the counter-case, on the arm that ran it
quackd run --goal "Wave to the camera with an extended arm" --robot arm-01 \
  --max-steps 10 --jev shadow --runs-dir runs/bench
```

Then, per run directory:

| What | Read | From |
|---|---|---|
| **Jev's latency**, the figure nobody publishes | mean and max of `latency_s` | every `{"kind":"jev"}` |
| **State size** | `state_chars`, `state_tokens_est`, `trimmed` | every `{"kind":"jev"}` |
| **Coverage** | how many `gate` values are `taken`, out of all of them | every `{"kind":"jev"}` |
| **Agreement** | how often `agree` is true, and separately among rows where `would_have_acted` is true | every `{"kind":"jev_shadow"}` |
| **Calibration** | `confidence` bucketed against `agree` — the plot that earns the right to move a floor | join the two kinds on `step` |
| **What the model cost** | `latency_s`, `usage` | every `{"kind":"llm"}` |
| **The rollup** | `steps`, `llm_calls`, `elapsed_s`, `usage`, `jev` | `summary.json` |

**Prices are yours to apply, not quackd's to print.** quackd counts tokens and has never
computed money. TypeSafe publish Jev at **$0.042 per million input tokens, with output not
charged** ([models](https://docs.typesafe.ai/models), read 2026-09-18); your model's rate card
is on your vendor's page. If you run any of this, the numbers are worth an issue.

## Configuration

| What | How |
|---|---|
| Mode | `--jev off\|shadow\|on`, or `QUACKD_JEV`. The flag wins. Default `off` |
| Key | `TYPESAFE_API_KEY`, in the environment or a `.env` |
| Model | `TYPESAFE_DEFAULT_MODEL`, default `jev-1.13.0`. Pinned rather than `jev-latest`, because a run whose stepper changed under it is a run whose transcript describes a model that is no longer the one that answered |
| Endpoint | `TYPESAFE_BASE_URL`, if it is not TypeSafe's own |
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
- **Unmeasured here.** No latency figure, no agreement rate, no hardware run. Everything on this
  page that is a number is either from TypeSafe's own documentation or from the arithmetic of a
  run that happened before any of this existed.

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
