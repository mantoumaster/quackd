"""Check a `.duck` against a vocabulary: the Microduck's by default, or one or more robot
manifests. One implementation, one wording, shared by `quackd validate`, the MCP
`robot_load_duckfile` tool and the flock runner (ADR-0019).

Parse and schema errors are the parser's; this module only judges a parsed contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from quackd.duckfile.schema import DuckFile

if TYPE_CHECKING:
    from quackd.adapters.manifest import RobotManifest
    from quackd.verbs.registry import VerbRegistry


@dataclass(frozen=True)
class Problem:
    field: str
    message: str
    robot: str | None = None
    verb: str | None = None

    def __str__(self) -> str:
        return f"{self.field}: {self.message}"


def validate_duck(
    duck: DuckFile,
    manifests: Sequence[RobotManifest] = (),
    *,
    registry: VerbRegistry | None = None,
) -> list[Problem]:
    """Every problem, in a stable order; an empty list means the contract is honourable.

    With manifests, `requires` (for a v0 file: every allowed verb) must be provided by each
    robot for a solo task, or by at least one robot of a flock; every flock role must be
    fillable by at least one robot. Without manifests the vocabulary is the registry."""
    fm = duck.frontmatter
    problems: list[Problem] = []
    if not manifests:
        if registry is None:
            from quackd.verbs.registry import default_registry

            registry = default_registry()
        unknown = registry.unknown(fm.verbs.allow)
        if unknown:
            problems.append(Problem("verbs.allow", f"unknown verbs: {', '.join(unknown)}"))
    if fm.learned_verbs:
        problems.append(
            Problem("learned_verbs", "must be empty (executing policies is a v2 feature)")
        )
    if fm.flock is not None and fm.verbs.confirm:
        problems.append(
            Problem("verbs.confirm", "a flock cannot prompt y/N per duck: empty verbs.confirm")
        )
    if not manifests:
        return problems

    if fm.datasheet is not None:
        # the merge is where a task file's corrections meet the body's own invariants: an
        # armless body handed a payload is a contradiction, and it is caught here rather than
        # after the robot has connected
        from pydantic import ValidationError

        from quackd.adapters.manifest import apply_datasheet_override

        for m in manifests:
            try:
                apply_datasheet_override(m, fm.datasheet)
            except ValidationError as e:
                why = "; ".join(str(err["msg"]).removeprefix("Value error, ") for err in e.errors())
                problems.append(Problem("datasheet", f"{m.id} ({m.model}): {why}", robot=m.id))

    reported: set[str] = set()
    if fm.flock is None:
        for m in manifests:
            for verb in fm.effective_requires:
                if not m.provides(verb):
                    reported.add(verb)
                    problems.append(
                        Problem(
                            "requires",
                            f"requires {verb}, but {m.id} ({m.model}) does not provide it",
                            robot=m.id,
                            verb=verb,
                        )
                    )
    else:
        ids = ", ".join(m.id for m in manifests)
        for verb in fm.effective_requires:
            if not any(m.provides(verb) for m in manifests):
                reported.add(verb)
                problems.append(
                    Problem(
                        "requires", f"requires {verb}, but none of {ids} provides it", verb=verb
                    )
                )
        for role, spec in (fm.flock.roles or {}).items():
            if not any(all(m.provides(v) for v in spec.requires) for m in manifests):
                problems.append(
                    Problem(
                        f"flock.roles.{role}",
                        f"no robot provides all of {', '.join(spec.requires)}",
                    )
                )
    # the weaker line: an allowed verb no robot has (a v1 task may allow more than it needs)
    for verb in fm.verbs.allow:
        if verb in reported or any(m.provides(verb) for m in manifests):
            continue
        who = (
            manifests[0].id
            if len(manifests) == 1
            else "any of " + ", ".join(m.id for m in manifests)
        )
        problems.append(Problem("verbs.allow", f"{verb} is not provided by {who}", verb=verb))
    return problems
