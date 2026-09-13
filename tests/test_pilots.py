"""`run_pilot_flock`: N agent loops on wall-clock time, and what happens when one of them
does not come back.

Everything here runs on `mock` backends, which is what a pilot flock has ever run on.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from quackd.adapters.factory import RobotSpec, describe
from quackd.agent.providers.base import Exchange, ProviderTurn, ToolCall, Usage
from quackd.agent.providers.fake import FakeProvider
from quackd.duckfile.parser import load_duck, parse_duck_text
from quackd.flock.pilots import (
    MAX_MEMBERS,
    SpecEntry,
    aggregate_outcome,
    roster_from_specs,
    run_pilot_flock,
    trim_contract,
    union_problems,
)
from quackd.memory import RobotMemory

MIXED = {
    "duck": RobotSpec("microduck", "mock", "duck"),
    "arm": RobotSpec("lerobot", "mock", "arm"),
}


def _roster(specs: dict[str, RobotSpec] | None = None) -> dict[str, SpecEntry]:
    return roster_from_specs(specs or MIXED)


def _providers(roster: dict[str, Any], duck_name: str = "flock-hello") -> dict[str, Any]:
    # one instance per member: a shared strategy closure would be shared state
    return {name: FakeProvider.for_duck(duck_name) for name in roster}


async def _run(tmp_path: Path, roster: dict[str, Any] | None = None, **kw: Any) -> Any:
    roster = roster if roster is not None else _roster()
    return await asyncio.wait_for(
        run_pilot_flock(
            load_duck("flock-hello"),
            roster,
            providers=kw.pop("providers", None) or _providers(roster),
            runs_dir=tmp_path,
            **kw,
        ),
        timeout=120,
    )


def _summary(result: Any) -> dict[str, Any]:
    return json.loads((result.run_dir / "summary.json").read_text(encoding="utf-8"))


def _flock_lines(result: Any) -> list[dict[str, Any]]:
    text = (result.run_dir / "flock.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ── the contract, per body ──────────────────────────────────────────────────────────────


def test_each_member_is_handed_the_half_of_the_contract_its_body_can_answer_for() -> None:
    fm = load_duck("flock-hello").frontmatter
    duck = trim_contract(fm, describe(MIXED["duck"]))
    arm = trim_contract(fm, describe(MIXED["arm"]))
    assert "say" in duck.verbs.allow and "say" not in arm.verbs.allow
    assert "move_joints" in arm.verbs.allow and "move_joints" not in duck.verbs.allow
    assert "report_state" in duck.verbs.allow and "report_state" in arm.verbs.allow
    assert duck.requires == ["report_state"] and arm.requires == ["report_state"]
    assert fm.verbs.allow == load_duck("flock-hello").frontmatter.verbs.allow, "the file is not"


def test_a_trimmed_contract_is_still_a_contract() -> None:
    """Rebuilt through the model, so every cross-field rule in the schema still holds."""
    text = (
        "---\nduck: 1\nname: t\ndescription: d\nverbs:\n  allow: [stop, say, move_joints]\n"
        "  confirm: [move_joints]\nsuccess: [x]\nrequires: [say]\n---\n# Task\nx\n"
    )
    fm = parse_duck_text(text).frontmatter
    duck = trim_contract(fm, describe(MIXED["duck"]))
    assert duck.verbs.confirm == [], "a gated verb this body lacks is not gated on it"
    assert duck.requires == ["say"]
    arm = trim_contract(fm, describe(MIXED["arm"]))
    assert arm.verbs.confirm == ["move_joints"] and arm.requires == []


def test_a_body_that_can_do_none_of_it_still_gets_stop() -> None:
    text = (
        "---\nduck: 1\nname: t\ndescription: d\nverbs:\n  allow: [move_joints]\n"
        "success: [x]\n---\n# Task\nx\n"
    )
    trimmed = trim_contract(parse_duck_text(text).frontmatter, describe(MIXED["duck"]))
    assert trimmed.verbs.allow == ["stop"]


def test_what_no_body_can_do_is_refused_and_what_one_can_is_not() -> None:
    text = (
        "---\nduck: 1\nname: t\ndescription: d\nverbs:\n  allow: [stop, kick, move_joints]\n"
        "success: [x]\nrequires: [kick]\n---\n# Task\nx\n"
    )
    manifests = {n: describe(s) for n, s in MIXED.items()}
    assert union_problems(parse_duck_text(text), manifests) == []
    nobody = text.replace("requires: [kick]", "requires: [kick, move_joints]").replace(
        "allow: [stop, kick, move_joints]", "allow: [stop, kick, move_joints, grab]"
    )
    assert union_problems(parse_duck_text(nobody), manifests) == []
    lacking = (
        "---\nduck: 1\nname: t\ndescription: d\nverbs:\n  allow: [stop, pick]\n"
        "success: [x]\nrequires: [pick]\n---\n# Task\nx\n"
    )
    only_duck = {"duck": manifests["duck"]}
    assert "requires pick" in union_problems(parse_duck_text(lacking), only_duck)[0]


# ── the outcome ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("members", "outcome"),
    [
        ({"a": ("success", "done"), "b": ("success", "done")}, "success"),
        ({"a": ("success", "done"), "b": ("failure", "no")}, "failure"),
        ({"a": ("budget", "out"), "b": ("failure", "no")}, "budget"),
        ({"a": ("infeasible", "too heavy"), "b": ("budget", "out")}, "infeasible"),
        ({"a": ("error", "boom"), "b": ("infeasible", "too heavy")}, "error"),
        # one member raised and the rest were stopped because it did: the cause is the headline
        ({"a": ("aborted", "stopped"), "b": ("error", "boom")}, "error"),
        # a person pressed something: nothing errored and everything aborted
        ({"a": ("aborted", "stopped"), "b": ("aborted", "stopped")}, "aborted"),
        ({}, "error"),
    ],
)
def test_the_worst_outcome_wins(members: dict[str, Any], outcome: str) -> None:
    assert aggregate_outcome(members)[0] == outcome


def test_the_reason_names_every_member_that_did_not_succeed_worst_first() -> None:
    _, reason = aggregate_outcome(
        {"a": ("failure", "no ball"), "b": ("success", "done"), "c": ("aborted", "stopped")}
    )
    assert reason == "c aborted: stopped; a failure: no ball"
    assert not reason.startswith("b "), "a member that succeeded is not in the reason"


def test_success_names_everyone() -> None:
    outcome, reason = aggregate_outcome({"a": ("success", "x"), "b": ("success", "y")})
    assert outcome == "success" and reason == "every member declared success: a, b"


# ── a whole run ─────────────────────────────────────────────────────────────────────────


async def test_two_different_bodies_talk_and_both_declare(tmp_path: Path) -> None:
    result = await _run(tmp_path)
    assert result.outcome == "success", result.reason
    assert result.messages == 2, "each said one thing"
    assert result.notices == 2, "and the runner said when each of them ended"
    assert {r["outcome"] for r in result.per_member.values()} == {"success"}
    assert result.per_member["arm"]["robot"] == "lerobot:mock"


async def test_two_bodies_of_one_kind_are_a_flock_too(tmp_path: Path) -> None:
    """The case the auction could always do and nothing else could: same body, twice."""
    twins = _roster(
        {
            "duck-a": RobotSpec("microduck", "mock", "duck-a"),
            "duck-b": RobotSpec("microduck", "mock", "duck-b"),
        }
    )
    result = await _run(tmp_path, twins)
    assert result.outcome == "success", result.reason
    assert list(result.per_member) == ["duck-a", "duck-b"]


async def test_the_artifacts_are_what_the_docs_say(tmp_path: Path) -> None:
    result = await _run(tmp_path)
    kinds = [line["kind"] for line in _flock_lines(result)]
    assert kinds[0] == "flock_start" and kinds[-1] == "flock_end"
    assert kinds.count("member_end") == 2
    talks = [
        line["msg"] for line in _flock_lines(result) if line.get("msg", {}).get("kind") == "TALK"
    ]
    assert len(talks) == 4, "two between the members, two from the runner"
    assert all("t" in line and "sim_t" not in line for line in _flock_lines(result)), (
        "a pilot flock runs on wall-clock seconds and must not stamp them as sim time"
    )
    for name in ("duck", "arm"):
        member = result.run_dir / "ducks" / name
        assert (member / "transcript.jsonl").exists()
        assert not (member / "summary.json").exists(), "a member is not a solo run"
    summary = _summary(result)
    assert summary["flock"] == {"members": ["duck", "arm"], "method": "pilots", "name": None}
    assert summary["transport"] == "mock"
    assert summary["messages"] == 2 and summary["notices"] == 2
    assert set(summary["per_member"]["duck"]) == {
        "outcome",
        "reason",
        "steps",
        "llm_calls",
        "usage",
        "provider",
        "model",
        "robot",
        "run_dir",
        "trace_dropped",
    }
    start = _flock_lines(result)[0]
    assert start["clock"] == "wall"
    assert start["contracts"]["arm"]["allow"] == ["report_state", "observe", "stop", "move_joints"]


async def test_a_mixed_flock_reports_its_backends_as_mixed(tmp_path: Path) -> None:
    roster = _roster(
        {
            "duck": RobotSpec("microduck", "sim2d", "duck"),
            "arm": RobotSpec("lerobot", "mock", "arm"),
        }
    )
    assert _summary(await _run(tmp_path, roster))["transport"] == "mixed"


async def test_the_runners_notice_reaches_the_survivors(tmp_path: Path) -> None:
    """A pilot waiting on somebody who has stopped has to be told, or it waits forever."""
    result = await _run(tmp_path)
    heard = [
        line["msg"]
        for line in _flock_lines(result)
        if line.get("msg", {}).get("kind") == "TALK" and line["msg"]["src"] == "flock"
    ]
    assert len(heard) == 2
    assert all("declared success" in m["text"] for m in heard)


# ── when one member does not come back ──────────────────────────────────────────────────


class _Boom:
    """A transport that will not connect."""

    name = "microduck"
    backend = "mock"

    async def connect(self) -> None:
        raise RuntimeError("no socket")

    async def close(self) -> None:
        return None

    def now(self) -> float:
        return 0.0


async def test_the_first_exception_stops_the_others_and_names_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quackd.adapters import factory

    real = factory.make_adapter

    def maybe_boom(spec: Any, **kw: Any) -> Any:
        return _Boom() if spec.name == "arm" else real(spec, **kw)

    monkeypatch.setattr(factory, "make_adapter", maybe_boom)
    monkeypatch.setattr("quackd.flock.pilots.make_adapter", maybe_boom)
    result = await _run(tmp_path)
    assert result.outcome == "error", result.reason
    assert result.per_member["arm"]["outcome"] == "error"
    assert "no socket" in result.per_member["arm"]["reason"]
    assert "arm" in result.reason
    others = [n for n in result.per_member if n != "arm"]
    assert all(result.per_member[n]["outcome"] in ("aborted", "success") for n in others)
    assert (result.run_dir / "ducks" / "arm" / "transcript.jsonl").exists()


async def test_a_provider_that_raises_is_that_members_error(tmp_path: Path) -> None:
    class _Angry:
        name, model, supports_vision = "angry", "none", False

        async def step(self, system: str, history: list[Exchange], tools: list[Any]) -> Any:
            raise RuntimeError("429 from the vendor")

    roster = _roster()
    providers: dict[str, Any] = _providers(roster)
    providers["arm"] = _Angry()
    result = await _run(tmp_path, roster, providers=providers)
    assert result.outcome == "error"
    assert "429 from the vendor" in result.per_member["arm"]["reason"]


async def test_the_kill_switch_reaches_every_member(tmp_path: Path) -> None:
    """One event, every executor: Ctrl-C stops every body, not the one in front.

    A pilot mid-verb is cancelled and sent a stop; one waiting on its model notices at its
    next turn, exactly as a solo run does."""

    roster = _roster()
    master = asyncio.Event()
    master.set()  # already pressed: mock bodies finish faster than any sleep could race
    result = await _run(
        tmp_path,
        roster,
        providers={n: FakeProvider(strategy=_always_report) for n in roster},
        abort=master,
    )
    assert result.outcome == "aborted", result.reason
    assert {r["outcome"] for r in result.per_member.values()} == {"aborted"}
    assert "kill switch" in result.reason
    assert all(r["steps"] == 0 for r in result.per_member.values()), "nothing moved"


# ── the flags ───────────────────────────────────────────────────────────────────────────


async def test_dry_run_reaches_every_member(tmp_path: Path) -> None:
    result = await _run(tmp_path, dry_run=True)
    assert _summary(result)["dry_run"] is True
    assert result.outcome == "success"


async def test_max_steps_applies_to_each_member(tmp_path: Path) -> None:
    roster = _roster()
    providers = {n: FakeProvider(strategy=_always_report) for n in roster}
    result = await _run(tmp_path, roster, providers=providers, max_steps=1)
    assert result.outcome == "budget", result.reason
    assert all(r["steps"] <= 1 for r in result.per_member.values())


def _always_report(obs: Any, step: int, history: list[Exchange]) -> ToolCall:
    return ToolCall(name="report_state", arguments={})


async def test_memory_is_keyed_by_the_name_or_the_body_and_can_be_off(tmp_path: Path) -> None:
    roster = _roster()
    memories = {name: RobotMemory(name, tmp_path / "mem") for name in roster}
    await _run(tmp_path, roster, memories=memories)
    assert (tmp_path / "mem" / "duck.jsonl").exists()
    assert (tmp_path / "mem" / "arm.jsonl").exists()
    off = tmp_path / "mem2"
    await _run(tmp_path, _roster(), memories=None)
    assert not off.exists()


async def test_a_flock_of_one_or_of_nine_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2 to 8 members"):
        await _run(tmp_path, _roster({"duck": MIXED["duck"]}))
    big = {f"duck-{i}": RobotSpec("microduck", "mock", f"duck-{i}") for i in range(MAX_MEMBERS + 1)}
    with pytest.raises(ValueError, match="2 to 8 members"):
        await _run(tmp_path, _roster(big))


async def test_a_member_with_no_pilot_is_refused(tmp_path: Path) -> None:
    roster = _roster()
    with pytest.raises(ValueError, match="no pilot for arm"):
        await _run(tmp_path, roster, providers={"duck": FakeProvider()})


async def test_a_task_no_body_can_do_refuses_before_anything_connects(tmp_path: Path) -> None:
    """Two arms asked to kick. Neither can, so nobody connects and nothing is written."""
    text = (
        "---\nduck: 1\nname: impossible\ndescription: d\nverbs:\n"
        "  allow: [stop, kick]\nsuccess: [x]\nrequires: [kick]\n"
        "flock:\n  members: [left, right]\n  allocation:\n    method: pilots\n"
        "---\n# Task\nx\n"
    )
    roster = _roster(
        {
            "left": RobotSpec("lerobot", "mock", "left"),
            "right": RobotSpec("lerobot", "mock", "right"),
        }
    )
    with pytest.raises(ValueError, match="requires kick"):
        await asyncio.wait_for(
            run_pilot_flock(
                parse_duck_text(text), roster, providers=_providers(roster), runs_dir=tmp_path
            ),
            timeout=30,
        )
    # checked off the event loop: the point is that the directory is empty, not when
    made = await asyncio.to_thread(lambda: list(tmp_path.glob("*flock*")))
    assert not made, "nothing on disk to explain away"


# ── the views ───────────────────────────────────────────────────────────────────────────


async def test_each_member_gets_its_own_view_and_the_flock_gets_one(tmp_path: Path) -> None:
    seen: dict[str, list[str]] = {}

    def trace(name: str) -> Any:
        seen.setdefault(name, [])
        return lambda event: seen[name].append(event.kind)

    await _run(tmp_path, trace=trace)
    assert set(seen) == {"duck", "arm", "flock"}
    assert "run_start" in seen["duck"] and "run_start" not in seen["flock"]
    assert seen["flock"] == ["talk", "talk"], "the flock view carries the runner's own notices"


def test_usage_is_summed_across_the_members(tmp_path: Path) -> None:
    class _Costly:
        name, model, supports_vision = "costly", "none", False

        async def step(self, system: str, history: list[Exchange], tools: list[Any]) -> Any:
            return ProviderTurn(
                tool_calls=[ToolCall(id="x", name="declare_success", arguments={"reason": "ok"})],
                usage=Usage(input_tokens=10, output_tokens=1),
                stop_reason="tool_use",
            )

    roster = _roster()
    result = asyncio.run(
        asyncio.wait_for(
            run_pilot_flock(
                load_duck("flock-hello"),
                roster,
                providers={n: _Costly() for n in roster},
                runs_dir=tmp_path,
            ),
            timeout=60,
        )
    )
    assert result.usage.input_tokens == 20 and result.usage.output_tokens == 2
    assert result.llm_calls == 2


async def test_a_cancelled_flock_records_every_member_rather_than_crashing(
    tmp_path: Path,
) -> None:
    """The second Ctrl-C: `asyncio.run` cancels the task. A member that never returned a
    result used to leave a `KeyError` where the interrupt should have been."""

    class _Slow:
        name, model, supports_vision = "slow", "none", False

        async def step(self, system: str, history: list[Exchange], tools: list[Any]) -> Any:
            await asyncio.sleep(30)
            raise AssertionError("never")

    roster = _roster()
    task = asyncio.ensure_future(_run(tmp_path, roster, providers={n: _Slow() for n in roster}))
    await asyncio.sleep(0.4)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # the record survives the interrupt, which is the whole point of writing it first
    run_dir = await asyncio.to_thread(lambda: sorted(tmp_path.glob("*flock-hello"))[-1])
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["outcome"] == "aborted"
    assert set(summary["per_member"]) == set(roster)
    assert all(m["outcome"] == "aborted" for m in summary["per_member"].values())
