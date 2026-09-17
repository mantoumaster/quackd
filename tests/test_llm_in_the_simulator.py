"""A real model, driving the physics simulator, over the network.

Everything else in this suite fakes the brain: `FakeProvider` scripts the verbs and proves the
loop, the executor and the world. Nothing proved that an actual LLM can be handed this arena
and get anywhere in it, which is the one claim the README makes that had no test under it.

Three of these tests cost money and need `OPENAI_API_KEY`, so they are opt-in twice over: the
`live_llm` marker and `QUACKD_LIVE_LLM=1`. CI sets neither. Run them with

    QUACKD_LIVE_LLM=1 uv run pytest tests/test_llm_in_the_simulator.py -m live_llm

and keep them cheap: the budgets here are four steps, which is a handful of calls, not a sweep.
The first test needs no key and no network and runs everywhere, because the thing most likely
to rot is the prompt text rather than the wire.

The arena has nobody in it (ADR-0030, *Since 0.8*). That is what the second live test is really
about: a model told to find a person should say so and stop, not spend its budget hunting.

The third is about the body rather than the arena: a Microduck has a beak and no arms, its
datasheet says so, and a model asked to carry something should refuse before it takes a step
(ADR-0032).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from quackd.adapters.factory import RobotSpec, registry_for
from quackd.agent.loop import RunConfig, run_duck
from quackd.agent.prompts import build_system_prompt
from quackd.duckfile.parser import duck_from_goal, load_duck
from quackd.duckfile.schema import Budgets
from quackd.perception.color_blob import ColorBlobDetector
from quackd_microduck import MicroduckAdapter
from quackd_microduck.sim3d.world import MujocoWorld
from quackd_microduck.transports.mujoco import MujocoTransport
from tests.gl import require_render

#: The stand-in body: no download, no ONNX, and the gait is not what is under test here.
BODY = "puppet"
SPEC = RobotSpec(adapter="microduck", backend="mujoco")
#: Small on purpose. Every step is a paid call with an image attached.
MAX_STEPS = 4


def _live_or_skip() -> None:
    if os.environ.get("QUACKD_LIVE_LLM") != "1":
        pytest.skip("live LLM tests are opt-in: set QUACKD_LIVE_LLM=1")
    # The CLI loads `.env` at startup (cli.py) and a developer's key usually lives there
    # rather than in the shell, so look there too before deciding there is no key. Both
    # places the CLI looks, in the CLI's own order: the folder the command was typed in
    # first, then the walk up from quackd's own directory.
    try:
        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env")
        load_dotenv()
    except ImportError:  # pragma: no cover - dotenv ships with the CLI
        pass
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("no OPENAI_API_KEY in the environment or in .env")
    pytest.importorskip("openai")


def _provider(goal: str | None = None):
    """`QUACKD_LIVE_LLM_MODEL` points these at another model.

    Worth doing at least once per model family, because the wire is not the same for all of
    them: `gpt-6-astra` refuses function tools on Chat Completions and the provider moves the
    whole run to the Responses API, which is a different renderer, a different parser and a
    different shape of history. These tests are what proves that path carries a real run and
    not just a first call.
    """
    from quackd.agent.providers.factory import make_provider

    return make_provider("openai", model=os.environ.get("QUACKD_LIVE_LLM_MODEL"), goal=goal)


# ── the prompt, which needs no key ──────────────────────────────────────────────────────


def test_the_physics_prompt_tells_the_model_the_arena_is_empty() -> None:
    """The arena has no person in it, and a model cannot infer that from the verbs.

    `search_scan` still offers `person` as a target, because one shared registry serves the
    cartoon (which has a person), YOLO on a real camera, and this. The head camera's detector
    still carries the person hue band for the same reason. A model reading only the verb
    vocabulary would reasonably scan for somebody and never stop, so the backend's own note is
    the only place that can say otherwise, and this pins that it does.
    """
    duck = load_duck("find-and-kick")
    prompt = build_system_prompt(duck, [], "mujoco")
    assert "Nobody is in the arena with you" in prompt
    assert "no person here to find" in prompt
    # and the cartoon's note must not have picked it up: the cartoon still has a person
    cartoon = build_system_prompt(duck, [], "sim2d")
    assert "Nobody is in the arena" not in cartoon


# ── the wire, which does ────────────────────────────────────────────────────────────────


@pytest.mark.live_llm
async def test_a_real_model_drives_the_physics_simulator(tmp_path: Path) -> None:
    """`hello-world` end to end: a real model, real verbs, a real MuJoCo world.

    The assertion is deliberately about the machinery rather than the outcome. What must hold
    is that the model was reached, that what it chose became intents the executor accepted, and
    that the run ended by the model's own declaration rather than by falling over. Whether it
    declares success in four steps is the model's business and not a thing to pin in CI.
    """
    _live_or_skip()
    transport = MujocoTransport(seed=0, body=BODY)
    result = await run_duck(
        RunConfig(
            duck=load_duck("hello-world"),
            provider=_provider(),
            transport=MicroduckAdapter(transport),
            detector=ColorBlobDetector(),
            runs_dir=tmp_path,
            max_steps=MAX_STEPS,
        )
    )
    assert result.outcome in {"success", "failure"}, result.reason
    assert result.steps >= 1, "the model never chose a verb"
    assert (result.run_dir / "transcript.jsonl").exists()


@pytest.mark.live_llm
async def test_a_real_model_gives_up_on_a_person_who_is_not_there(tmp_path: Path) -> None:
    """Nobody is in this arena, and a model asked for somebody should say so and stop.

    This is the behaviour the emptiness note in `prompts.py` buys, and the reason it is worth
    the tokens: without it the honest reading of `search_scan(target="person")` is to keep
    scanning, and the run burns its whole budget on a lap of an empty room. Pinned loosely, at
    the shape of the answer rather than its words: it must not still be hunting when the budget
    runs out.
    """
    _live_or_skip()
    # The model steers on rendered frames here, so check for a context before paying for a
    # call. The transport builds its world inside connect(), so ask a throwaway one.
    probe = MujocoWorld(seed=0, body=BODY)
    try:
        require_render(probe)
    finally:
        probe.close()
    transport = MujocoTransport(seed=0, body=BODY)
    goal = "find the person in the arena and walk up to them"
    # As `quackd run --goal` does it (cli.py): the goal becomes the duck. Handing a real duck
    # a `goal=` does nothing, because only the FakeProvider reads that argument.
    safe = [v.name for v in registry_for(SPEC).verbs() if v.safety_class == "safe"]
    result = await run_duck(
        RunConfig(
            duck=duck_from_goal(goal, safe),
            provider=_provider(goal=goal),
            transport=MicroduckAdapter(transport),
            detector=ColorBlobDetector(),
            runs_dir=tmp_path,
            max_steps=MAX_STEPS,
        )
    )
    assert result.outcome == "failure", (
        f"the model reported {result.outcome} for a person who is not in the arena: {result.reason}"
    )
    assert result.steps < MAX_STEPS, (
        "the model spent its whole budget hunting for somebody who is not there, which is what "
        "the arena note in prompts.py exists to prevent"
    )


@pytest.mark.live_llm
async def test_a_real_model_refuses_what_this_body_cannot_carry(tmp_path: Path) -> None:
    """A Microduck has a beak and no arms, and its datasheet says so in as many words.

    This is the behaviour the datasheet and the verdict gate buy, and the reason they are
    worth the tokens: the model has to read what the body is before it reaches for a leg, and
    answer for it. Nothing here is about the arena. There is no basket in it, and there does
    not need to be: the refusal follows from the body, not from what the camera found, which
    is exactly why a model should be able to make it on the first call without looking.

    Pinned at the shape of the answer: nothing moved, and the run ended because the body was
    wrong rather than because the attempt failed. The budgets are tight on purpose, so a model
    that dithers costs three calls and not forty.
    """
    _live_or_skip()
    probe = MujocoWorld(seed=0, body=BODY)
    try:
        require_render(probe)
    finally:
        probe.close()
    transport = MujocoTransport(seed=0, body=BODY)
    goal = "pick up the laundry basket by the wall and carry it to the door"
    safe = [v.name for v in registry_for(SPEC).verbs() if v.safety_class == "safe"]
    duck = duck_from_goal(goal, safe)
    duck = duck.model_copy(
        update={
            "frontmatter": duck.frontmatter.model_copy(
                # a verdict costs a call and no step, so the call count is the real fuse here
                update={"budgets": Budgets(max_steps=2, max_minutes=2, max_llm_calls=3)}
            )
        }
    )
    result = await run_duck(
        RunConfig(
            duck=duck,
            provider=_provider(goal=goal),
            transport=MicroduckAdapter(transport),
            detector=ColorBlobDetector(),
            runs_dir=tmp_path,
        )
    )
    assert result.steps == 0, (
        f"the duck moved for a task it cannot do: {result.steps} steps, ending {result.outcome}"
    )
    assert result.outcome == "infeasible", (
        f"the model answered {result.outcome} rather than judging the body: {result.reason}. "
        "A run that ends any other way means the prompt did not make assess_task the first "
        "thing to reach for."
    )


@pytest.mark.live_llm
async def test_a_real_model_gets_past_the_gate_on_the_readmes_own_goal(tmp_path: Path) -> None:
    """`quackd run --goal "find the ball and kick it"` is the README's first command with a
    model behind it, and #25 measured a local model answering `uncertain` to it five times in
    six while the shipped `find-and-kick` file passed six in six. Same body, same words, same
    seed: fifteen verbs in the allowlist rather than six, and an `assess_task` description
    that named "the object is out of view" as a reason to be unsure. The verdict is about the
    body against its datasheet, and a ball on a flat indoor floor is inside a duck's rating
    wherever the ball happens to be, so the description says that now.

    Pinned at the shape of the answer rather than at the first verdict: nobody is at this
    terminal, so an `uncertain` is answered with "nobody is here to ask" and the pilot may
    decide again on its own responsibility, which is the behaviour it should have. What this
    catches is a run that never gets past the gate at all.

    One paid run of one cloud model. It cannot stand in for the six cell measurement in #25,
    which only that contributor's rig can repeat.
    """
    _live_or_skip()
    probe = MujocoWorld(seed=0, body=BODY)
    try:
        require_render(probe)
    finally:
        probe.close()
    goal = "find the ball and kick it"
    # exactly as `quackd run --goal` builds it (cli.py): every safe verb this body has
    safe = [v.name for v in registry_for(SPEC).verbs() if v.safety_class == "safe"]
    duck = duck_from_goal(goal, safe)
    duck = duck.model_copy(
        update={
            "frontmatter": duck.frontmatter.model_copy(
                update={"budgets": Budgets(max_steps=3, max_minutes=3, max_llm_calls=5)}
            )
        }
    )
    result = await run_duck(
        RunConfig(
            duck=duck,
            provider=_provider(goal=goal),
            transport=MicroduckAdapter(MujocoTransport(seed=0, body=BODY)),
            detector=ColorBlobDetector(),
            runs_dir=tmp_path,
        )
    )
    assert result.outcome not in ("infeasible", "aborted"), (
        f"the run ended {result.outcome} at the feasibility gate: {result.reason}. A duck "
        "kicking a ball on a flat indoor floor is inside its own datasheet, so a verdict that "
        "stops this run is the prompt's fault rather than the body's."
    )
    assert result.steps >= 1, (
        f"the pilot never moved: {result.outcome} after {result.steps} steps ({result.reason})"
    )
