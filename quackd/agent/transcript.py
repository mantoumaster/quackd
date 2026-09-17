"""Every run leaves a paper trail: `runs/<timestamp>/transcript.jsonl`, frames, a summary.

A transcript exists so a run can be argued about after the fact — which prompt, which
tool call, which result, how many tokens — and so the golden tests can pin the loop's
behaviour without an LLM in the room.
"""

from __future__ import annotations

import io
import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from quackd.transport.base import CameraFrame

if TYPE_CHECKING:
    from quackd.trace import TraceEvent


def new_run_dir(base: str | Path = "runs", name: str | None = None) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    suffix = f"-{name}" if name else ""
    path = Path(base) / f"{stamp}{suffix}"
    i = 1
    while path.exists():
        path = Path(base) / f"{stamp}{suffix}-{i}"
        i += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class Transcript:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.path = run_dir / "transcript.jsonl"
        self.frames_dir = run_dir / "frames"
        self._fh = self.path.open("a", encoding="utf-8")
        self._t0 = time.monotonic()
        self.events = 0
        self.frame_count = 0

    def write(self, kind: str, **payload: Any) -> None:
        if self._fh.closed:
            # A verb task cancelled during teardown can narrate its last intent after the
            # record has closed. Dropping that line beats a `ValueError: I/O operation on
            # closed file` raised inside a task nobody is awaiting.
            return
        record = {"t": round(time.monotonic() - self._t0, 3), "kind": kind, **payload}
        self._fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        if kind != "intent":
            # An intent is written from inside `send_intent`, on the event loop, between two
            # deadman resends: a stalled filesystem there delays the next command past the
            # deadman and zeroes the velocity mid-stride. Every burst ends in a `verb_end`,
            # which flushes it, and the text layer's own buffer bounds what a hard kill could
            # lose to well under one verb's worth of lines.
            self._fh.flush()
        self.events += 1

    def sink(self, event: TraceEvent) -> None:
        """The transcript as a `Tracer` record: the event's kind and payload, on the
        transcript's own clock, so a `frame` written directly and an `observation` that came
        through the tracer never disagree about the time."""
        self.write(event.kind, **event.data)

    def save_frame(self, img: Image.Image, caption: str = "") -> Path:
        self.frames_dir.mkdir(exist_ok=True)
        path = self.frames_dir / f"{self.frame_count:04d}.png"
        img.save(path, format="PNG")
        self.write("frame", path=str(path.relative_to(self.run_dir)), caption=caption)
        self.frame_count += 1
        return path

    def save_frames(
        self, frames: Sequence[CameraFrame], caption: str = "", *, several: bool = False
    ) -> list[Path]:
        """Every camera's picture for one step, under one number.

        A body with one camera writes `0000.png` and a record with no camera in it, which is
        what every run before there could be two wrote. A body with several writes
        `0000-top.png` beside `0000-side.png` and names the camera in each record, so the
        file name says which view it is and the number still says which step.

        `several` is the body's camera count and not `len(frames)`, because a two-camera arm
        whose top lens stalls hands back one picture, and writing that as a bare `0000.png`
        leaves a file in the middle of a run with nothing anywhere saying which lens took it."""
        if len(frames) == 1 and not several:
            return [self.save_frame(frames[0].image, caption)]
        self.frames_dir.mkdir(exist_ok=True)
        paths: list[Path] = []
        for frame in frames:
            path = self.frames_dir / f"{self.frame_count:04d}-{frame.name}.png"
            frame.image.save(path, format="PNG")
            self.write(
                "frame",
                path=str(path.relative_to(self.run_dir)),
                caption=caption,
                camera=frame.name,
            )
            paths.append(path)
        # one step, one number, however many cameras were pointed at it
        self.frame_count += 1
        return paths

    def write_summary(self, summary: dict[str, Any]) -> Path:
        path = self.run_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return path

    def close(self) -> None:
        self._fh.close()

    @staticmethod
    def read(path: Path, *, lenient: bool = False) -> list[dict[str, Any]]:
        """Every record in the file. `lenient` skips the unparsable ones and counts them under
        the key `_skipped` on the last record: a run killed mid-write leaves a half line, and
        `quackd trace` should show the run that happened rather than a JSON error."""
        records: list[dict[str, Any]] = []
        skipped = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except ValueError:
                    if not lenient:
                        raise
                    skipped += 1
        if skipped and records:
            records[-1]["_skipped"] = skipped
        return records
