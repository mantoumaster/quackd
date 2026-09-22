"""A `laya` that loads no weights.

The in-process decision LLM is the one backend with no wire between quackd and the model, so
the only way to test the seam is to stand where the model would be. Installed into
`sys.modules` for the life of one test, because the real package pulls torch and a suite that
imported it would be testing a download.

What it stands in for is the shape rather than the answer: `Router(...)` takes its keywords and
remembers them, `predict(state, questions, model=...)` records what it was asked and hands back
whatever the test scripted, and the return is a plain mapping with an `answers` key and no
`usage` at all -- which is exactly the case quackd has to cost by estimate.
"""

from __future__ import annotations

import sys
import types
from typing import Any


class FakeRouter:
    """Laya's router, scripted.

    One instance stands for the class: `Router(...)` returns it, so a test can assert that a
    second turn did not build a second one. `loads` counts constructions, which is the whole
    question the lazy loader exists to answer.
    """

    def __init__(
        self,
        *,
        answers: dict[str, Any] | None = None,
        script: list[dict[str, Any]] | None = None,
        raises: Exception | None = None,
        routing: dict[str, Any] | None = None,
        rejects: tuple[str, ...] = (),
    ) -> None:
        self.answers = answers or {}
        self.script = list(script or [])
        self.raises = raises
        """Raised by `predict`, for the turn that has to become a gate rather than a crash."""
        self.routing = routing or {"model": "typed-decisions", "reason": "asked for"}
        self.rejects = rejects
        """Constructor keywords this pretend Laya has never heard of, so a test can play the
        older package the real loader falls back for."""
        self.loads = 0
        """How many times the class was constructed. The lazy loader promises once."""
        self.kwargs: dict[str, Any] = {}
        """What the last construction was handed, so a test can assert `preload=True`."""
        self.calls: list[tuple[dict[str, Any], dict[str, Any], Any]] = []
        """(state, questions, model) per turn."""

    def __call__(self, **kwargs: Any) -> FakeRouter:
        """`laya.Router(...)`, which is this instance whatever it is asked for."""
        for name in self.rejects:
            if name in kwargs:
                raise TypeError(f"Router() got an unexpected keyword argument {name!r}")
        self.loads += 1
        self.kwargs = dict(kwargs)
        return self

    def predict(
        self, state: dict[str, Any], questions: dict[str, Any], model: Any = None
    ) -> dict[str, Any]:
        self.calls.append((dict(state), dict(questions), model))
        if self.raises is not None:
            raise self.raises
        answers = self.script.pop(0) if self.script else self.answers
        # No `usage`: Laya reports no token count, which is what makes every turn it answers
        # an estimated one.
        return {"answers": dict(answers), "routing": dict(self.routing)}


def answer_choice(label: str, confidence: float = 0.99) -> dict[str, Any]:
    """One choice answer as Laya spells it: a mapping, and no `probabilities`."""
    return {"choice": label, "confidence": confidence}


def answer_noul(p: float) -> dict[str, Any]:
    return {"noul": p}


def turn(
    verb: str,
    confidence: float = 0.99,
    *,
    done: float = 0.0,
    need_human: float = 0.0,
    feasible: str = "feasible",
) -> dict[str, Any]:
    """One turn's four answers, in Laya's own mapping shape."""
    return {
        "next_verb": answer_choice(verb, confidence),
        "done": answer_noul(done),
        "need_human": answer_noul(need_human),
        "feasible": answer_choice(feasible, 0.9),
    }


def install(monkeypatch: Any, router: FakeRouter) -> FakeRouter:
    """Put `router` where `importlib.import_module("laya")` will find it, for one test."""
    module = types.ModuleType("laya")
    module.Router = router  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "laya", module)
    return router
