"""Laya, in this process: a decision LLM with no server, no network and no key.

The other presets are servers and share a client. This one is an encoder you import, and it
earns its own file for one reason: it is the proof that the seam here is a protocol rather
than an HTTP call. Everything quackd asks of a decision LLM it asks of this one too, and the
questions that go to a hosted API go to this unchanged, because Laya's own `predict` takes the
same `{"type": ..., "instructions": ..., "criteria": ...}` mapping the wire format does.

What it does not return is a token count, so every turn it answers is costed at the
self-hosted rate and marked estimated -- which is the truth: it billed you in electricity.

Imported only when a run has named it. `torch` is not something `quackd --help` may pay for.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from quackd.agent.decision.base import DecisionNotInstalled

if TYPE_CHECKING:
    from quackd.agent.decision.catalogue import DecisionSpec


class LayaLLM:
    """Laya's router, loaded on the turn that first needs it.

    The one preset built lazily, and deliberately so: the others construct an HTTP client in
    microseconds, and this one loads weights -- seven seconds cold, and a download the first
    time ever. Paying that while the CLI parses would stall every run that never reaches a
    turn. A load that fails is remembered, so a broken install costs one attempt rather than
    one per turn.
    """

    def __init__(self, spec: DecisionSpec, *, model: str | None = None):
        self.name = spec.name
        self.model = model or spec.model or "laya"
        self.url: str | None = None
        self._extra = spec.extra or "laya"
        self._router: Any = None
        self._broken: Exception | None = None

    def _load(self) -> Any:
        try:
            laya = importlib.import_module("laya")
        except ImportError as e:
            raise DecisionNotInstalled(self.name, self._extra) from e
        # `preload=True` because the alternative is rebuilding the model per call, which Laya's
        # own README measures at 7.4 seconds on a CPU against a timeout of one second.
        #
        # `default` so that what is preloaded is the checkpoint this run will actually ask for.
        # Laya holds one model at a time unless told otherwise, and its default is the plain
        # English one, so preloading blind and then asking for `typed-decisions` pays the load
        # twice and throws the first one away. Tried and then dropped rather than assumed,
        # because the keyword is younger than the class and quackd would rather load the wrong
        # checkpoint once than refuse to run at all.
        try:
            return laya.Router(preload=True, default=self.model)
        except TypeError:
            return laya.Router(preload=True, max_loaded=2)

    async def decide(
        self, state: Mapping[str, str], questions: Mapping[str, Mapping[str, Any]]
    ) -> Any:
        if self._broken is not None:
            raise self._broken
        if self._router is None:
            try:
                self._router = await asyncio.to_thread(self._load)
            except Exception as e:
                self._broken = e
                raise
        # `to_thread` because `predict` is a blocking forward pass and the loop it would block
        # is the one holding the robot's deadman.
        return await asyncio.to_thread(
            self._router.predict, dict(state), dict(questions), model=self.model
        )
