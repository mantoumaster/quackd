"""Runners that need no torch: a Python function answers every request.

`ScriptedRunner` is what the tests drive the policy loop with, what the simulator rehearses a
segment with before a checkpoint is anywhere near the arm, and what a `PolicyLike` becomes when
it is handed to the backend: `ScriptedRunner.wrapping(policy)` asks it for one action per tick,
as `pick` always did. Everything here is deterministic. The same script, asked at the same ticks
with the same readings, answers the same chunks, which is what lets a test say that the
simulator's lockstep and a run on the wall's clock played the same trajectory.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from quackd_lerobot.policy.runner import Chunk, Features, Observation, PolicyLike

Script = Callable[[Observation, Mapping[str, float]], Sequence[Mapping[str, Any]] | None]
"""A scripted policy: an observation and the command last sent in, the chunk's actions out, one
per tick from the observation's, or None for a policy with nothing more to do."""


class ScriptedRunner:
    """A `PolicyRunner` whose answers come from `script`.

    `rate_hz` and `rate_source` are its declared rate and where that came from, as a runner
    loaded from a checkpoint declares them. `latency_ticks` is how many ticks it declares it
    takes to answer, which on the simulator holds every chunk back that many ticks before it is
    played, and which it never actually spends: a script answers as fast as Python does.
    `per_tick` makes it a runner asked every tick (tick mode). `instruction` and `resets` are
    what the last `reset` was told and how many there have been, for a test to read."""

    def __init__(
        self,
        script: Script,
        *,
        rate_hz: float,
        rate_source: str = "the script",
        latency_ticks: int = 0,
        per_tick: bool = False,
    ) -> None:
        if latency_ticks < 0:
            raise ValueError(f"latency_ticks={latency_ticks} must be 0 or more")
        self.script = script
        self.rate_hz = rate_hz
        self.rate_source = rate_source
        self.latency_ticks = latency_ticks
        self.per_tick = per_tick
        self.instruction = ""
        self.resets = 0
        self.requests = 0
        self.closed = False
        self._on_reset: Callable[[], None] | None = None

    @classmethod
    def wrapping(cls, policy: PolicyLike, *, rate_hz: float, rate_source: str) -> ScriptedRunner:
        """A `PolicyLike` as a runner: each request is one `act`, its answer a chunk of one
        action, and None a runner that is done. The policy's own `reset()`, where it has one,
        runs at every `reset`, and the instruction becomes the `task` every `act` is told."""
        runner: ScriptedRunner

        def act(observation: Observation, _sent: Mapping[str, float]) -> list[Any] | None:
            action = policy.act(dict(observation.reading), task=runner.instruction)
            return None if action is None else [action]

        runner = cls(act, rate_hz=rate_hz, rate_source=rate_source)
        reset = getattr(policy, "reset", None)
        if callable(reset):
            runner._on_reset = reset
        return runner

    def reset(self, instruction: str) -> None:
        self.instruction = instruction
        self.resets += 1
        if self._on_reset is not None:
            self._on_reset()

    def features(self) -> Features:
        return Features(self.rate_hz, self.rate_source, per_tick=self.per_tick)

    def next_chunk(self, observation: Observation, sent: Mapping[str, float]) -> Chunk:
        self.requests += 1
        actions = self.script(observation, sent)
        if actions is None:
            return Chunk(observation.tick, done=True)
        return Chunk(observation.tick, tuple(actions))

    def latency_s(self) -> float:
        # the declared ticks at the declared rate, so the loop's own rounding gives the ticks back
        if not (math.isfinite(self.rate_hz) and self.rate_hz > 0):
            return 0.0
        return self.latency_ticks / self.rate_hz

    def close(self) -> None:
        self.closed = True


__all__ = ["Script", "ScriptedRunner"]
