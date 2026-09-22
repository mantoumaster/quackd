"""What quackd asks of a decision LLM, and the three ways it can be absent.

One method. A decision LLM is handed the state quackd built and the questions quackd wrote,
and hands back answers; everything else -- which turns are a choice, what the state may
contain, how confident an answer must be before it moves a servo -- is `stepper.py`'s and
stays there whatever answers them.

The errors are phrased like `providers/base.py`'s, deliberately: a reader who has met
`ProviderNotInstalled` should recognise its neighbour here without being told.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


class DecisionError(RuntimeError):
    """A decision LLM that cannot be built or cannot be reached. Raised while the CLI is still
    parsing, never mid-run: a turn that fails is a gate on the record, not an exception."""


class DecisionNotInstalled(DecisionError):
    def __init__(self, name: str, extra: str) -> None:
        super().__init__(
            f"--decision-llm {name} needs the optional extra quackd[{extra}] — "
            f'run: uvx --from "quackd[{extra}]" quackd ...  '
            f'or: uv pip install "quackd[{extra}]"'
        )


class DecisionMissingKey(DecisionError):
    def __init__(self, name: str, env_var: str) -> None:
        super().__init__(
            f"--decision-llm {name} needs {env_var} (set it in .env or the environment)"
        )


@runtime_checkable
class DecisionLLM(Protocol):
    """One decision LLM, however it is reached.

    `decide` is handed a flat mapping of short English strings -- the named state quackd built
    for this turn -- and a mapping of question name to a plain dict in the System One shape:
    `{"type": "choice" | "noul" | "score", "instructions": ..., "criteria": ...}`. Plain dicts
    rather than an SDK's own classes, so the same four questions go to a hosted API, a server
    on this machine and a model in this process without being rebuilt for each.

    What comes back may be anything `stepper._answer` can read: an object carrying `answers`,
    or a mapping with an `"answers"` key, whose members carry `choice` and `confidence` (and
    optionally `probabilities`), or a bare `noul`, as attributes or as keys. A `usage` with
    `input_tokens` is used where it is there and estimated where it is not, so a backend that
    counts nothing costs the run an estimate rather than an error.

    Raising is allowed and is not fatal: `Stepper.advise` catches everything, records the turn
    as `gate: error` and hands it to the model. A decision LLM may never end a run.
    """

    name: str
    """The preset that built it. What `--decision-llm` took and what the record says."""
    model: str
    """The id it answers as, for the record. Empty where the backend names none."""
    url: str | None
    """Where it was reached, for the record, or `None` for the hosted default and for one
    that runs in this process. Written through `command.redacted_url`, never raw."""

    async def decide(
        self, state: Mapping[str, str], questions: Mapping[str, Mapping[str, Any]]
    ) -> Any: ...
