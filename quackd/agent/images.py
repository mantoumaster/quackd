"""Pictures that come with the task: `quackd run --image sketch.png`.

A camera frame is perception and arrives every step. One of these is neither: it is a file a
person named on the command line, it is fixed for the whole run, and it is what the task is
about. "Draw what is in the picture" is not a sentence a robot's own camera can answer.

Everything here is re-encoded to PNG before it goes anywhere. Every provider quackd speaks to
already takes a PNG, `NamedPng` promises one, and the copy kept beside the transcript is then
byte for byte what the model was sent rather than a source file it was derived from. The cost
is that a photograph is re-compressed on its way in, which is also where it is made small
enough to send.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from quackd.agent.providers.base import NamedPng

MAX_SIDE_PX = 1568
"""Longest edge a task picture is sent at. Above roughly this, the vendors that publish a
figure resize server side anyway and charge for the tokens; below it, a sketch is still
legible. Not a vendor's number for every vendor, so it is quackd's own choice."""

MAX_BYTES = 1_500_000
"""How large one encoded picture may be. Several of these ride in every single request for
the whole run, so the cap is on quackd's side of the wire rather than on the vendor's."""

SHRINK = 0.8
"""How much smaller to try when the encoded picture is still over the cap."""

MIN_SIDE_PX = 64
"""Stop shrinking here and refuse instead: a picture this small says nothing, and a loop that
kept halving would turn one bad file into a silent blank."""

FORMATS = ("PNG", "JPEG", "WEBP", "GIF", "BMP", "TIFF")
"""What PIL is allowed to have decoded. Named so the refusal can list them, and so a PDF or a
video handed to `--image` is one line rather than a traceback from inside a decoder."""


class TaskImageError(ValueError):
    """A `--image` that cannot be sent, in words that name the file and the fix."""


def _encode(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _fit(img: Image.Image, path: str) -> bytes:
    """One picture as PNG bytes, inside both caps.

    The long edge comes down first, which is what actually costs tokens, and only then the
    file size, which is what costs bandwidth. A sketch is mostly flat colour and lands far
    under the byte cap at full size; a photograph of a desk may not, and shrinks again."""
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX))
    png = _encode(img)
    while len(png) > MAX_BYTES:
        width = int(img.width * SHRINK)
        height = int(img.height * SHRINK)
        if min(width, height) < MIN_SIDE_PX:
            raise TaskImageError(
                f"--image {path}: this picture is still {len(png) // 1000} kB at "
                f"{img.width}x{img.height} and will not fit under {MAX_BYTES // 1000} kB "
                "without becoming unreadable; save a smaller or flatter copy"
            )
        img = img.resize((width, height))
        png = _encode(img)
    return png


def _names(paths: Sequence[str]) -> list[str]:
    """What each picture is called on the wire: its file name, and its place in the list when
    two files share one.

    The model is told a name and the task refers to the picture by what it shows, so the name
    has to be the one the person typed. Two directories with a `sketch.png` in each would
    otherwise both arrive as `sketch.png`, and a task naming one of them would be ambiguous in
    exactly the way a label exists to prevent."""
    bases = [Path(p).name for p in paths]
    return [f"{i + 1}-{base}" if bases.count(base) > 1 else base for i, base in enumerate(bases)]


def load_task_images(paths: Sequence[str]) -> list[NamedPng]:
    """Every `--image` as a PNG the providers can carry, in the order they were given.

    A file that is missing, unreadable, or not a picture is refused here, before the robot is
    connected and before a run directory exists. The alternative is a run that reaches the
    first request without the one thing the task is about."""
    names = _names(paths)
    out: list[NamedPng] = []
    for path, name in zip(paths, names, strict=True):
        file = Path(path)
        if not file.exists():
            raise TaskImageError(f"--image {path}: no such file")
        if file.is_dir():
            raise TaskImageError(f"--image {path}: that is a directory, not a picture")
        try:
            with Image.open(file) as img:
                if img.format not in FORMATS:
                    raise TaskImageError(
                        f"--image {path}: quackd sends {', '.join(FORMATS)} and this is "
                        f"{img.format or 'not a picture'}"
                    )
                # GIF and TIFF can hold several; the first frame is the one a person means
                img.seek(0)
                png = _fit(img, path)
        except TaskImageError:
            raise
        except UnidentifiedImageError as e:
            raise TaskImageError(
                f"--image {path}: this is not a picture quackd can read ({', '.join(FORMATS)})"
            ) from e
        except OSError as e:
            raise TaskImageError(f"--image {path}: could not be read: {e}") from e
        out.append(NamedPng(name=name, png=png))
    return out
