"""Every word the LLM reads, in one place.

The system prompt carries the contract (in prose the model can act on) and the `.duck`
body verbatim. Each turn's observation is compact and structured — features, not frames —
with the image attached separately for providers that can see. One tool call per turn is
stated here *and* enforced by the loop; saying it is not the same as trusting it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from quackd.duckfile.schema import DuckFile
from quackd.perception.base import Detection, summarize_detections
from quackd.transport.base import DuckState
from quackd.verbs.aliases import canonical
from quackd.verbs.registry import Verb, VerbResult
from quackd.verdict import BEFORE_VERDICT, needs_properties

if TYPE_CHECKING:
    from quackd.adapters.manifest import Datasheet, RobotManifest

DUCK_BLURB = "a small biped duck robot (25 cm, 800 g)"

ASSESS_TASK = {
    "name": "assess_task",
    "description": (
        "Your verdict on whether THIS body can do THIS task, judged against the datasheet in "
        "your prompt: the task's needs against the body's limits, not whether you can see the "
        "target yet. Required before the first verb that moves the body: until you have "
        "answered, a verb that moves the body is refused and told so, and only what looks, "
        "speaks or brakes runs. The Rules in your prompt name which of your verbs those are. "
        "Answer `feasible` when every need fits inside a limit you can point to; not having "
        "found the target yet is not by itself a doubt about the body, it is what the run is "
        "for. Answer "
        "`infeasible` when one need clearly exceeds a limit (a 3 kg basket on a 0.3 kg "
        "payload): the run ends at once and nothing moves, so name the limit and what you "
        "estimated. Answer `uncertain` only when the verdict itself turns on a figure you "
        "cannot judge from here: the mass or size of a thing that decides a limit and that you "
        "have not seen (out of view, or detections rather than an image), or a limit that "
        "matters listed as not published. A human is then asked, "
        "or you are told nobody is there to ask. Do not guess a mass or a size from the words "
        "of the task alone: look first, or say uncertain. You may call this again later, once "
        "you have seen the thing. It moves nothing and does not count as a step."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["feasible", "infeasible", "uncertain"],
                "description": (
                    "feasible: go. infeasible: the run ends before any motion. uncertain: a "
                    "human decides, or you are told nobody can."
                ),
            },
            "reason": {
                "type": "string",
                "description": (
                    "One or two sentences: the need, the limit it meets or exceeds, and how "
                    "you compared them."
                ),
            },
            "limits_consulted": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "The datasheet fields you read to decide (payload_kg, reach_m, mobility, "
                    "...). Empty when the task needs none of them."
                ),
            },
            "estimates": {
                "type": "array",
                "description": (
                    "What you guessed about the world to reach the verdict, so the record "
                    "shows it. One entry per object and quantity."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "object": {"type": "string"},
                        "quantity": {
                            "type": "string",
                            "enum": [
                                "mass_kg",
                                "size_m",
                                "distance_m",
                                "height_m",
                                "duration_min",
                                "count",
                                "other",
                            ],
                        },
                        "value": {"type": "number"},
                        "basis": {
                            "type": "string",
                            "enum": ["image", "detections", "task_text", "prior_knowledge"],
                            "description": (
                                "What the guess came from. task_text and prior_knowledge are "
                                "the weak ones."
                            ),
                        },
                        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    },
                    "required": ["object", "quantity", "value", "basis", "confidence"],
                    "additionalProperties": False,
                },
            },
            "needs": {
                "type": "object",
                "description": (
                    "What the task requires, in the datasheet's own field names. Numbers are "
                    "minimums (payload_kg: 3 means at least 3 kg). A matcher reads this to say "
                    "which other body could do the task, so fill it in even when the verdict "
                    "is feasible. It is also read back against this body's own datasheet: a "
                    "feasible verdict naming a need the sheet does not meet, or does not "
                    "publish, is refused and told which need, so name what the task turns on "
                    "and nothing it does not: a need this task does not actually rest on is "
                    "what gets a good verdict refused. A minimum of 0 asks for nothing and is "
                    "always met."
                ),
                "properties": needs_properties(),
                "additionalProperties": False,
            },
        },
        "required": ["verdict", "reason"],
        "additionalProperties": False,
    },
}
ASSESS_TASK_NAME = ASSESS_TASK["name"]


DECLARE_SUCCESS = {
    "name": "declare_success",
    "description": "Call when the success criteria are met. Say which criterion and what evidence you have.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Which criterion was met, and the evidence.",
            }
        },
        "required": ["reason"],
        "additionalProperties": False,
    },
}

DECLARE_FAILURE = {
    "name": "declare_failure",
    "description": (
        "Call when the task cannot be completed after trying (target not found, repeated "
        "failures, an abort condition). If nothing has moved yet and the body itself is the "
        "reason, call assess_task with infeasible instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
        "required": ["reason"],
        "additionalProperties": False,
    },
}

META_TOOLS = [ASSESS_TASK, DECLARE_SUCCESS, DECLARE_FAILURE]
META_TOOL_NAMES = {t["name"] for t in META_TOOLS}
DECLARE_NAMES = {str(DECLARE_SUCCESS["name"]), str(DECLARE_FAILURE["name"])}
"""The two that end a run. `assess_task` is a meta tool too, but it is a gate, not an ending."""

REMEMBER = {
    "name": "remember",
    "description": (
        "Save one short fact for FUTURE runs on this robot (where things usually are, what "
        "worked, what to avoid). It does not move the robot and does not count as a step. "
        "Do not repeat what the prompt already remembers."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "One sentence, concrete and reusable, e.g. 'the ball is usually near the left wall'.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional short labels (place, object, strategy).",
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    },
}
REMEMBER_NAME = REMEMBER["name"]

TELL = {
    "name": "tell",
    "description": (
        "Say one short thing to another pilot in your flock, or to all of them. It reaches "
        "them in their next observation. It moves nothing and does not count as a step, "
        "though it does use one of your calls. Say what you are about to do, what you have "
        "done, or what you need from them, so nobody waits on somebody who is not coming. "
        "You never hear your own words back."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": "A member name from `Your flock`, or `all` for everyone.",
            },
            "text": {
                "type": "string",
                "description": "One or two sentences, concrete: 'I have the ball, you spot'.",
            },
        },
        "required": ["to", "text"],
        "additionalProperties": False,
    },
}
TELL_NAME = TELL["name"]


COMPOSITE_VERBS = ("walk_to", "go_to", "search_scan", "approach_and")
"""The verbs that close their own loop on the camera, in the order the prompt prefers to name
one. A body that provides none of them is told about none of them."""


def goal_strategy(allow: Sequence[str]) -> str:
    """The `## Strategy` paragraph of a `--goal` task, in the verbs this body was granted.

    It used to name `observe`, `search_scan` and `go_to` whatever the body was. On 2026-09-15
    an SO-101 arm read all three in a prompt whose allowlist two lines above listed none of
    them, and whose executor would have refused each one: the arm has no camera verb it is
    granted, no scan and no composite. The model worked it out and looked with `report_state`
    instead, which is not a thing to rely on.

    Every verb named here is in `allow`. A body with neither looking verb is told to look with
    `report_state`, and one with no composite is told to prefer the verb that does the whole
    thing rather than being pointed at a verb that does not exist. A Microduck reads exactly
    what it read before."""
    names = set(allow)
    looking: list[str] = [verb for verb in ("observe", "search_scan") if verb in names]
    if not looking and "report_state" in names:
        looking = ["report_state"]
    look = " or ".join(f"`{verb}`" for verb in looking) if looking else "however this body can"
    composite = next((verb for verb in COMPOSITE_VERBS if verb in names), None)
    prefer = (
        f"prefer composite verbs like `{composite}`"
        if composite is not None
        else "prefer the verb that does the whole thing over micro-managing"
    )
    # a body with no camera verb has no frame to verify against, only its own joints
    fresh = "frame" if "observe" in names else "reading"
    return (
        f"Use the available verbs. Look before you act ({look}), {prefer}, verify with a "
        f"fresh {fresh}, then `remember` one fact and declare success."
    )


def before_verdict_clause(verbs: Sequence[Verb]) -> str:
    """`only `quack` and `stop` run`: which of THIS run's verbs run before the verdict.

    Read by the same rule the gate applies in `Executor._run_verb`, so the sentence cannot
    drift from the refusal: the canonical name is in `BEFORE_VERDICT`, or the verb's own
    adapter declared it `read_only`. It used to be one fixed list naming `observe`,
    `report_state`, `say` and the head verbs whatever the body was, which was wrong three ways
    at once. hello-world's pilot was told `observe` runs when its contract allows no such verb.
    An arm was told about head verbs no arm has. And since #26 a body quackd never shipped can
    bring its own sensing verb, which the gate lets through and this sentence could not name.

    Spelled as the allowlist spells it, so a duck that allows `get_frame` reads `get_frame`.
    `stop` is last and unconditional: the executor lets the brake through whether or not a
    contract listed it."""
    named = [
        verb.name
        for verb in verbs
        if (verb.read_only or (canonical(verb.name) in BEFORE_VERDICT and verb.kind != "learned"))
        and verb.name != "stop"
    ]
    spelled = [f"`{name}`" for name in dict.fromkeys(named)] + ["`stop`"]
    if len(spelled) == 1:
        return f"only {spelled[0]} runs"
    return f"only {', '.join(spelled[:-1])} and {spelled[-1]} run"


BODY_HEADING = "## Your body: what it can and cannot do"

_CONFIDENCE_KEY = (
    "official: the maker or a paper says so; estimate: one vendor, a community number or a "
    "reading off a photo; measured: somebody measured it and said how"
)


def _clamp_words(limits: dict[str, float]) -> str:
    """What quackd will let through, in the prompt's words.

    Read from `manifest.limits`, never from the datasheet: a clamp is quackd's own rule about
    what it sends, not a fact about the body. `control_hz` and `camera_fov_deg` are not clamps
    and are left out."""
    parts: list[str] = []
    if (vx := limits.get("max_vx")) is not None:
        parts.append(f"{vx:g} m/s forward")
        if (vy := limits.get("max_vy")) is not None:
            parts.append("no sideways motion" if vy == 0 else f"{vy:g} m/s sideways")
        if (wz := limits.get("max_wz")) is not None:
            parts.append(f"{wz:g} rad/s turning")
    if "lift_min_mm" in limits and "lift_max_mm" in limits:
        parts.append(f"lift {limits['lift_min_mm']:g} to {limits['lift_max_mm']:g} mm")
    if (yaw := limits.get("gaze_yaw_deg")) is not None:
        parts.append(f"gaze {yaw:g} degrees of yaw")
    if (deg := limits.get("joint_deg")) is not None:
        parts.append(f"joints within {deg:g} degrees")
    if (norm := limits.get("joint_norm")) is not None:
        parts.append(f"joints within a normalised {norm:g}")
    if (grip := limits.get("gripper")) is not None:
        parts.append(f"gripper 0 to {grip:g}")
    return ", ".join(parts)


def _hands(ds: Datasheet) -> str:
    if ds.manipulator == "beak":
        return (
            "a beak, no arms. It can scoop at an object on the floor right under it, and that "
            "is all"
        )
    if ds.manipulator == "gripper":
        arms = "one arm with a gripper" if ds.arms == 1 else f"{ds.arms} arms with a gripper each"
        return arms
    if ds.manipulator == "arms":
        return f"{ds.arms} arms and no gripper: it holds by closing both on a thing"
    return "none: nothing quackd can command touches an object"


def _power_and_ground(ds: Datasheet, *, mobile: bool) -> str:
    power = {
        True: "Mains powered, so nothing runs down",
        False: "Battery powered",
        None: "Power source not published",
    }[ds.tethered]
    if not mobile:
        return f"{power}. It does not move: no base and no legs"
    ground = {
        "indoor_flat": "Rated for a flat indoor floor",
        "indoor": "Rated for indoors",
        "outdoor": "Rated for outdoors",
    }.get(ds.terrain or "") or (
        "Terrain not published: assume a flat indoor floor and decline anything else"
    )
    rated = f". Not rated for {', '.join(ds.not_rated)}" if ds.not_rated else ""
    return f"{power}. {ground}{rated}"


def body_lines(manifest: RobotManifest) -> list[str]:
    """The body's own facts, as prompt lines.

    Every line starts with a word or a dash and a word, never with a backtick: the prompt
    spells an offered verb as a backticked name after a dash, and four adapter tests read the
    offered verbs back out of the prompt by that shape."""
    ds = manifest.datasheet
    if ds is None:
        return [
            f"The {manifest.model} adapter has published no datasheet. Treat every physical "
            "limit (weight, height, reach, endurance, terrain) as not published, and decline "
            "any task that hinges on one."
        ]
    mobile = manifest.mobility != "none"
    lines = [f"Each number says how sure quackd is of it and who says so ({_CONFIDENCE_KEY})."]
    lines += [f"- {label}: {fig.text(unit)}" for label, fig, unit in ds.known()]
    lines.append(f"- Manipulator: {_hands(ds)}.")
    if unknown := ds.unknown():
        lines.append(
            f"- Not published: {', '.join(unknown)}. Decline any task that hinges on any of them."
        )
    lines.append(f"- {_power_and_ground(ds, mobile=mobile)}.")
    limits = dict(manifest.limits)
    ranges: dict[str, Any] = manifest.extras.get("joint_range_deg") or {}
    if ranges:
        # an arm that has answered knows each joint's real travel, and a goal outside it is
        # refused; the schema's symmetric bound would tell the pilot a range it cannot use
        limits.pop("joint_deg", None)
    if clamps := _clamp_words(limits):
        lines.append(f"- quackd clamps you to {clamps}.")
    if ranges:
        travel = ", ".join(f"{joint} {lo:g} to {hi:g}" for joint, (lo, hi) in ranges.items())
        lines.append(
            f"- Each joint's travel in degrees, read from its own calibration, and the only "
            f"goals quackd will send: {travel}."
        )
    if manifest.sensors:
        lines.append(f"- Senses: {', '.join(manifest.sensors)}.")
    if ds.cannot:
        lines.append("Whatever the task says, this body cannot:")
        lines += [f"- {s}" for s in ds.cannot]
    if ds.notes:
        lines.append("Worth knowing:")
        lines += [f"- {s}" for s in ds.notes]
    return lines


def body_section(manifest: RobotManifest) -> str:
    """The `## Your body` block. Facts only: the rule that the pilot must judge the task
    against them before moving is one bullet in the prompt's Rules, where every other enforced
    rule is stated."""
    return f"\n{BODY_HEADING}\n" + "\n".join(body_lines(manifest)) + "\n"


def body_summary(manifest: RobotManifest) -> str:
    """The same facts as one paragraph, for a tool result (MCP has no system prompt)."""
    ds = manifest.datasheet
    if ds is None:
        return f"{manifest.model}: no datasheet published; treat every physical limit as unknown."
    facts = [f"{label.lower()} {fig.text(unit)}" for label, fig, unit in ds.known()]
    facts.append(_hands(ds))
    facts.append(_power_and_ground(ds, mobile=manifest.mobility != "none").lower())
    text = f"{manifest.model}: " + ", ".join(facts) + "."
    if unknown := ds.unknown():
        text += f" Not published: {', '.join(unknown)}."
    if ds.cannot:
        text += " Cannot: " + "; ".join(ds.cannot) + "."
    if clamps := _clamp_words(manifest.limits):
        text += f" Clamps: {clamps}."
    return text


def build_system_prompt(
    duck: DuckFile,
    verbs: list[Verb],
    transport_name: str,
    manifest: RobotManifest | None = None,
    memory_text: str | None = None,
    *,
    assumptions: list[str] | None = None,
    flock_text: str | None = None,
    task_images: Sequence[str] | None = None,
    by_hand: bool = False,
) -> str:
    """`memory_text` is what the robot remembers from earlier runs (`RobotMemory.recall`);
    None means memory is off for this run, "" means on but empty.

    `assumptions` is what the robot says quackd is standing in for on this backend, read from
    `state.extras` when the run connects. It belongs in the prompt rather than in every
    observation because the list is sentences and an observation line is a line."""
    fm = duck.frontmatter
    blurb = manifest.blurb if manifest is not None and manifest.blurb else DUCK_BLURB
    names = {v.name for v in verbs}
    # what this body actually has, not what a duck has. The fallback used to name
    # `search_scan` unconditionally, so an arm was told about a composite verb it does not
    # provide, in the same prompt whose allowlist does not list it.
    composite = next((v for v in COMPOSITE_VERBS if v in names), None)
    moves_itself = manifest is None or manifest.mobility != "none"
    controllers = "balance and gait" if moves_itself else "the motion"
    pilot_line = f"the robot's own controllers handle {controllers}"
    if composite is not None:
        pilot_line += (
            f", and composite\nverbs like `{composite}` close their own loops on the camera"
        )
    verb_lines = "\n".join(f"- `{v.name}`: {v.description}" for v in verbs)
    before_verdict = before_verdict_clause(verbs)
    success = "\n".join(f"- {s}" for s in fm.success)
    advisory = fm.advisory_abort_conditions
    abort_lines = (
        "\n".join(f"- {a}" for a in advisory) if advisory else "- (none beyond the enforced ones)"
    )
    body = body_section(manifest) if manifest is not None else ""
    cameras = list(manifest.extras.get("cameras") or []) if manifest is not None else []
    if len(cameras) > 1:
        # more than one, not merely present. `extras["cameras"]` is the adapter's own list and
        # an adapter may publish it for a single camera, which the LeRobot arm does not and
        # another body may; a pilot with one view told "this body has 1 cameras: forward,
        # every frame is labelled" is promised something the wire never does, because
        # `name_cameras` reads the same count and sends that one picture bare.
        listed = ", ".join(cameras)
        body += (
            f"\nThis body has {len(cameras)} cameras: {listed}. Every frame reaches you each "
            f"step, labelled with the name of the camera that took it. {cameras[0]} is the "
            "primary: the `camera:` line in your observation describes that view and no "
            "other, and the verbs that steer by sight read it alone. A camera that gave "
            "nothing this step is absent rather than blank, so read the labels to see which "
            "views you actually have, and do not assume a missing one is showing you an "
            "empty room.\n"
        )
    placed = ""
    if by_hand:
        reader = "`report_state`" if "report_state" in names else "your state line"
        # what this task was actually granted, not what the body has: a pilot told to close the
        # gripper by a prompt whose allowlist has no `gripper` is being sent at a refusal
        grip = (
            "closing on an object is what makes this body say it is holding something, so if "
            "the task needs a firm hold on what is already between the jaws, call `gripper` to "
            "close on it before you lean on it"
            if "gripper" in names
            else "this task cannot work the gripper, so whatever is between the jaws is held at "
            "the squeeze the person left and you cannot tighten it"
        )
        placed = f"""
## Where this run starts
A person placed this body by hand before your first turn, and quackd is holding it exactly
where they left it. This run does **not** start from the recorded rest pose, so do not assume
a folded arm or a known shape: read {reader} and work from the joint angles it gives you. They
are where somebody decided the work should begin.

The gripper is where their fingers closed it, which is a position and not a grip. Nothing is
reported as held, and nothing should be: {grip}.

When the run ends, the arm is handed back the same way: it holds where you left it, the person
is asked to take whatever is in the gripper, and only then does quackd fold the arm up.
"""
    pictures = ""
    if task_images:
        listed = ", ".join(f"`{name}`" for name in task_images)
        several = len(task_images) > 1
        pictures = f"""
## The picture{"s" if several else ""} that came with this task
{len(task_images)} picture{"s" if several else ""} {"were" if several else "was"} handed to this task on the command line: {listed}. {"Each is" if several else "It is"} in your first turn, labelled `task picture NAME:` in front of the picture itself, and {"they stay" if several else "it stays"} in front of you for the whole run.

{"These are" if several else "This is"} what the task is about. {"They are" if several else "It is"} not what the robot can see: a camera frame, where this body has one, is a separate picture of the room the robot is in right now, labelled with the name of its camera. When the task says "this picture", or "what you see in the image", it means {"these" if several else "this one"} and not the camera.
"""
    stand_ins = ""
    if assumptions:
        listed = "\n".join(f"- {a}" for a in assumptions)
        stand_ins = f"""
## What is a stand-in on this robot
The robot reported these itself, and they are in `extras.assumptions` in every observation.
Each is something quackd stands in for or assumes, not something this robot does. Do not
report one as the robot's own work, and do not read a success from one as evidence that the
real skill works.
{listed}
"""
    persona = f"\n## Persona\n{fm.persona}\n" if fm.persona else ""
    memory = ""
    if memory_text is not None:
        remembered = memory_text.strip() or "(nothing yet — this is the first run on this robot)"
        memory = f"""
## What you remember from earlier runs on this robot
{remembered}

Call `remember` (one short sentence) when you learn something worth keeping for next time:
where an object usually is, which strategy worked, what to avoid. It moves nothing and
costs no step, though it does use one of your calls. Do not save what is already listed
above.
"""
    # built by `quackd.flock.talk`, which knows the roster; this only decides where it goes
    flock = f"\n{flock_text.strip()}\n" if flock_text else ""
    sim_note = ""
    if transport_name == "sim2d":
        sim_note = (
            "\nYou are in the built-in 2D simulator: a cartoon top-down world. Distances are "
            "metres, the arena is about 2 m across, and the ball is orange.\n"
        )
    elif transport_name == "mujoco":
        # Deliberately only the arena. What the body is, and what its legs can do, differs
        # between the real duck and the kinematic stand-in behind this one backend name, and
        # both describe themselves in the stand-ins block below. Saying it here as well
        # contradicted one of them: this told the model it was driving "a real biped on its
        # own learned gait" even when it was driving the puppet.
        # Nobody is in this arena, and the model has to be told so outright. `search_scan`
        # still offers `person` as a target, because one shared verb registry serves the
        # cartoon (which has a person), a real camera through YOLO, and this; and the head
        # camera's colour detector still carries the person hue band for the same reason. So
        # a model left to infer it would reasonably scan for somebody and never stop.
        sim_note = (
            "\nYou are in the physics simulator (MuJoCo): a 2 m arena with low walls and an "
            "orange ball that rolls when kicked. Nobody is in the arena with you: there is no "
            "person here to find, follow or walk up to, so do not scan for one, and treat any "
            "`person` detection as scenery misread rather than somebody standing there. "
            "Distances are metres.\n"
        )
    return f"""You are the brain of {blurb}. You are a high-level pilot:
you choose ONE verb per turn; {pilot_line}. Do not micro-manage.

## Rules (enforced by the executor — not optional)
- Call exactly one tool per turn. Never zero, never two.
- Only these verbs are allowed: {", ".join(fm.verbs.allow)}. Anything else is refused.
- Budgets: {fm.budgets.max_steps} steps, {fm.budgets.max_minutes:g} minutes, {fm.budgets.max_llm_calls} LLM calls. The run stops when any is hit.
- Verbs marked confirm ({", ".join(fm.verbs.confirm) or "none"}) ask a human before running.
- Before the first verb that moves the body, call `assess_task` with your verdict on whether this body can do this task at all, judged against its datasheet below: `feasible`, `infeasible` (the run ends, nothing moves) or `uncertain` (a human is asked). The verdict is about the body, not the view: not having found the target yet is not by itself a reason for `uncertain`. It is one when a limit turns on that unseen thing's mass or size, or on a figure the datasheet does not publish. Until then {before_verdict}. Assess again later if what you see changes your mind.
- When a success criterion is met, call `declare_success`. If the task turns out impossible while doing it, call `declare_failure`.

## Success criteria
{success}

## Abort conditions you must respect yourself
{abort_lines}

## Verbs
{verb_lines}
{body}{placed}{pictures}{stand_ins}{persona}{memory}{flock}{sim_note}
## Task file: {fm.name} — {fm.description}

{duck.body}
"""


def inbox_lines(inbox: Sequence[Mapping[str, Any]], me: str | None) -> list[str]:
    """What the flock said to you since your last turn, oldest first.

    `to` is the addressee, so a message sent to everyone reads `all` and one sent to you reads
    `you`: a pilot deciding whether to answer needs to know which it was."""
    out = []
    for message in inbox:
        who = str(message.get("from", "?"))
        to = message.get("to")
        whom = "you" if (to is not None and to == me) else "all"
        out.append(f"- {who} -> {whom}: {str(message.get('text', '')).strip()}")
    return out


def build_observation_text(
    *,
    step: int,
    max_steps: int,
    state: DuckState,
    detections: list[Detection],
    last_verb: str | None,
    last_result: VerbResult | None,
    budget_status: str,
    inbox: Sequence[Mapping[str, Any]] | None = None,
    inbox_for: str | None = None,
    cameras: Sequence[str] | None = None,
    stepped: Sequence[str] | None = None,
) -> str:
    lines = [
        f"[step {step}/{max_steps} · {budget_status}]",
        f"state: {state.summary()}",
        f"camera: {summarize_detections(detections)}",
    ]
    if cameras and len(cameras) > 1:
        # only with more than one, and only as a pointer: the pictures are labelled where
        # they are, and the `camera:` line above is the primary's and says nothing of the rest
        rest = ", ".join(cameras[1:])
        lines.append(f"cameras: {cameras[0]} (detections above), {rest}")
    if last_verb is not None and last_result is not None:
        lines.append(
            f"last verb `{last_verb}`: {'ok' if last_result.ok else 'FAILED'} — {last_result.summary}"
        )
    if inbox:
        lines.append("Messages from your flock (newest last):")
        lines.extend(inbox_lines(inbox, inbox_for))
    if stepped:
        # Turns the discrete stepper answered while the model was not asked (`--jev on`).
        # None of it is in the model's history, because none of it is anything the model
        # said, so this line is the whole of what it knows about that time. It has to say
        # who chose them: a pilot that thinks it closed the gripper itself will not think
        # to check.
        lines.append("While you were not asked, the stepper chose these (newest last):")
        lines.extend(f"- {line}" for line in stepped)
    lines.append("Choose exactly one tool.")
    return "\n".join(lines)


def observation_features(
    *,
    state: DuckState,
    detections: list[Detection],
    last_verb: str | None,
    last_result: VerbResult | None,
    allowed: list[str],
    inbox: Sequence[Mapping[str, Any]] | None = None,
    flock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    # `inbox` and `flock` are absent rather than empty on a solo run, so every existing reader
    # of these features (the scripted strategies, the goldens) sees exactly what it always did
    extra: dict[str, Any] = {}
    if inbox is not None:
        extra["inbox"] = [dict(m) for m in inbox]
    if flock is not None:
        extra["flock"] = dict(flock)
    return {
        **extra,
        "state": state.model_dump(),
        "detections": [d.model_dump() for d in detections],
        "last_result": (
            {
                "verb": last_verb,
                "ok": last_result.ok,
                "summary": last_result.summary,
                "data": last_result.data,
            }
            if last_result is not None
            else None
        ),
        "allowed": allowed,
    }
