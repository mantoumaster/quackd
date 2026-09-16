"""Where an adapter's code lives, while some are their own packages and some are not.

The adapters are moving out of the core wheel one at a time, so for a while `lerobot` is
`quackd_lerobot` in `adapters/lerobot/src/` and `toddlerbot` is still
`quackd_toddlerbot` in `quackd/`. A test that cares which robot a file belongs to
should not also have to care which half of that move it is in, so it asks here instead.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[1]


def adapter_module(name: str, sub: str = "") -> ModuleType:
    """An adapter's module by robot name, wherever that adapter currently lives."""
    tail = f".{sub}" if sub else ""
    for root in (f"quackd_{name}", f"quackd.adapters.{name}"):
        try:
            return importlib.import_module(root + tail)
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError(f"no adapter package for {name!r}")


def source_roots() -> tuple[Path, ...]:
    """Every directory holding quackd's own Python: the core, then each adapter package."""
    return (REPO / "quackd", *sorted(p for p in (REPO / "adapters").glob("*/src") if p.is_dir()))


def canonical_rel(path: Path) -> str:
    """One name for a file whichever layout it is in, so a rule outlives the move.

    `quackd/adapters/lerobot/real.py` and `adapters/lerobot/src/quackd_lerobot/real.py` are
    the same file to anything that reasons about which robot owns it, and both read back as
    `adapters/lerobot/real.py`. Core files keep their path under `quackd/`.
    """
    resolved = path.resolve()
    core = REPO / "quackd"
    if resolved.is_relative_to(core):
        return resolved.relative_to(core).as_posix()
    packages = REPO / "adapters"
    if resolved.is_relative_to(packages):
        rel = resolved.relative_to(packages).as_posix()
        name, _, tail = rel.partition("/")
        # adapters/<name>/src/quackd_<name>/<file> -> adapters/<name>/<file>
        inner = tail.split("/", 2)
        return f"adapters/{name}/{inner[-1]}" if len(inner) >= 2 else f"adapters/{name}/{tail}"
    return resolved.as_posix()
