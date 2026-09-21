"""A `typesafe_sdk` that never leaves the machine.

Installed into `sys.modules` for the life of one test, because `quackd.agent.jev` imports the
real one lazily inside the turn that needs it and would otherwise find whatever is on the
developer's machine and nothing at all on CI, which is two different suites.

The shapes are the ones docs.typesafe.ai publishes: `Choice`, `Score` and `Noul` are carriers
for an instruction and its criteria, `system_one` takes a state and a mapping of question name
to question, and the result has an `answers` mapping whose members carry `choice`,
`confidence` and `probabilities`, or a bare `noul`. Nothing here is a model: the answers are
whatever the test scripted, which is the point.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any


class TypeSafeError(Exception):
    pass


class APITimeoutError(TypeSafeError):
    pass


class APIConnectionError(TypeSafeError):
    pass


class RateLimitError(TypeSafeError):
    pass


@dataclass
class RetryPolicy:
    max_retries: int = 1
    backoff_max: float = 0.2
    timeout: float = 1.0


@dataclass
class Choice:
    instructions: Any = ""
    criteria: Any = field(default_factory=dict)


@dataclass
class Score:
    instructions: Any = ""
    criteria: Any = field(default_factory=list)


@dataclass
class Noul:
    instructions: Any = ""
    criteria: Any = field(default_factory=dict)


@dataclass
class ChoiceAnswer:
    choice: str
    confidence: float = 1.0
    probabilities: dict[str, float] = field(default_factory=dict)


@dataclass
class NoulAnswer:
    noul: float = 0.0


@dataclass
class Usage:
    """What the API says the call cost. Both counts are `int | None` in the real SDK, which
    documents None as "when the API did not report it", and that None is the difference
    between a cost quackd measured and one it estimated."""

    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class SystemOneResult:
    answers: dict[str, Any]
    model: str = ""
    usage: Usage | None = None


class FakeJev:
    """One scripted stepper.

    `answers` is what every turn returns unless `script` has something left, which is how a
    test says "choose `report_state` first and then be done". `raises` makes the call fail,
    which is the path that has to cost a turn rather than a run.
    """

    def __init__(
        self,
        *,
        answers: dict[str, Any] | None = None,
        script: list[dict[str, Any]] | None = None,
        raises: Exception | None = None,
        usage: Usage | None = None,
    ) -> None:
        self.answers = answers or {}
        self.script = list(script or [])
        self.raises = raises
        self.usage = usage
        """What every turn reports spending, or None for an API that reported nothing, which
        is the path where quackd has to estimate the size of its own question."""
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []
        """(state, questions) per turn, so a test can assert what was and was not sent."""
        self.model = ""
        self.retry: RetryPolicy | None = None

    # the SDK's own constructor shape, so `Stepper._client()` builds this the same way
    def __call__(self, *, model: str = "", retry: RetryPolicy | None = None) -> FakeJev:
        self.model = model
        self.retry = retry
        return self

    async def system_one(self, state: Any, questions: dict[str, Any]) -> SystemOneResult:
        self.calls.append((dict(state), dict(questions)))
        if self.raises is not None:
            raise self.raises
        answers = self.script.pop(0) if self.script else self.answers
        return SystemOneResult(answers=dict(answers), model=self.model, usage=self.usage)


def choice(label: str, confidence: float = 0.99, **rest: float) -> ChoiceAnswer:
    spread = {label: confidence, **rest}
    return ChoiceAnswer(choice=label, confidence=confidence, probabilities=spread)


def turn(
    verb: str,
    confidence: float = 0.99,
    *,
    done: float = 0.0,
    need_human: float = 0.0,
    feasible: str = "feasible",
) -> dict[str, Any]:
    """One turn's four answers, in the shape `Stepper._route` reads them."""
    return {
        "next_verb": choice(verb, confidence),
        "done": NoulAnswer(done),
        "need_human": NoulAnswer(need_human),
        "feasible": choice(feasible, 0.9),
    }


def install(monkeypatch: Any, fake: FakeJev) -> FakeJev:
    """Put `fake` where `importlib.import_module("typesafe_sdk")` will find it, for one test."""
    module = types.ModuleType("typesafe_sdk")
    module.AsyncTypeSafeClient = fake  # type: ignore[attr-defined]
    module.TypeSafeClient = fake  # type: ignore[attr-defined]
    module.Choice = Choice  # type: ignore[attr-defined]
    module.Usage = Usage  # type: ignore[attr-defined]
    module.Score = Score  # type: ignore[attr-defined]
    module.Noul = Noul  # type: ignore[attr-defined]
    module.RetryPolicy = RetryPolicy  # type: ignore[attr-defined]
    module.TypeSafeError = TypeSafeError  # type: ignore[attr-defined]
    module.APITimeoutError = APITimeoutError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "typesafe_sdk", module)
    return fake
