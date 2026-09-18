"""Pictures that come with the task: what loads, what is refused, and what each one is called.

`quackd run --image sketch.png` reads a file a person named, and everything it accepts has to
leave here as PNG bytes small enough to ride in every request of the whole run. So these tests
build real files with PIL and read the loader's answer back through PIL, rather than trusting
that a function that returned without raising produced something a provider could carry.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from quackd.agent import images
from quackd.agent.images import TaskImageError, load_task_images

PNG_MAGIC = b"\x89PNG"


def drawing(path: Path, size: tuple[int, int] = (320, 240), fmt: str | None = None) -> Path:
    """A flat picture: a red block on white, which encodes to a few kB at any size.

    The byte cap can never fire on one of these, so a test about the edge cap is about the
    edge cap and nothing else."""
    img = Image.new("RGB", size, "white")
    block = Image.new("RGB", (size[0] // 4, size[1] // 4), "red")
    img.paste(block, (size[0] // 8, size[1] // 8))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format=fmt)
    return path


def noise(path: Path, side: int) -> Path:
    """A square of random pixels, which is the one thing PNG cannot compress.

    Its encoded size therefore tracks its area, which is what makes the byte cap reachable
    from a file small enough for a test to write."""
    rng = np.random.default_rng(7)
    pixels = rng.integers(0, 256, (side, side, 3), dtype=np.uint8)
    Image.fromarray(pixels, "RGB").save(path)
    return path


def decoded(png: bytes) -> tuple[str, tuple[int, int]]:
    """The format and the size PIL reads back out of whatever the loader produced."""
    with Image.open(io.BytesIO(png)) as img:
        return img.format or "", img.size


def test_every_format_quackd_takes_comes_back_as_png(tmp_path: Path) -> None:
    """A JPEG is lossy, a WebP is a different codec and a GIF arrives in palette mode, and
    every provider quackd speaks to is handed a PNG. The magic bytes are the only proof that
    the re-encode happened rather than the source file being passed along."""
    paths = [
        str(drawing(tmp_path / "sketch.png")),
        str(drawing(tmp_path / "photo.jpg", fmt="JPEG")),
        str(drawing(tmp_path / "logo.webp", fmt="WEBP")),
        str(drawing(tmp_path / "loop.gif", fmt="GIF")),
    ]
    loaded = load_task_images(paths)

    assert [p.name for p in loaded] == ["sketch.png", "photo.jpg", "logo.webp", "loop.gif"]
    for picture in loaded:
        assert picture.png.startswith(PNG_MAGIC)
        assert decoded(picture.png)[0] == "PNG"


def test_a_picture_larger_than_the_cap_comes_back_smaller(tmp_path: Path) -> None:
    """The long edge is what costs tokens, so it is the long edge that comes down, and the
    shape has to survive it: a picture four times as wide as it is tall is still that when the
    model sees it."""
    wide = drawing(tmp_path / "wide.png", size=(2400, 600))

    png = load_task_images([str(wide)])[0].png

    assert decoded(png)[1] == (images.MAX_SIDE_PX, images.MAX_SIDE_PX // 4)


def test_a_picture_already_under_the_cap_keeps_its_own_size(tmp_path: Path) -> None:
    """Nothing is scaled up and nothing is scaled down for the sake of it: a sketch a person
    drew at 320x240 is legible at 320x240 and arrives at 320x240."""
    small = drawing(tmp_path / "small.png", size=(320, 240))

    png = load_task_images([str(small)])[0].png

    assert decoded(png)[1] == (320, 240)


def test_a_picture_too_heavy_to_send_is_shrunk_until_it_fits(tmp_path: Path) -> None:
    """A picture can be well inside the edge cap and still be megabytes, and these ride in
    every request for the whole run. A thousand pixels of random noise is already over the
    byte cap at a size no edge cap would touch, so only the second loop can save it."""
    heavy = noise(tmp_path / "heavy.png", 1000)
    assert heavy.stat().st_size > images.MAX_BYTES

    png = load_task_images([str(heavy)])[0].png

    assert len(png) <= images.MAX_BYTES
    width, height = decoded(png)[1]
    assert width == height  # a square went in, and shrinking keeps it square
    assert images.MIN_SIDE_PX <= width < 1000


def test_shrinking_stops_before_the_picture_is_unreadable_and_names_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal exists so that one bad file is one line rather than a blank thumbnail the
    model is asked to describe. No real picture stays over 1.5 MB at 64 pixels, so the cap is
    moved down to where a picture a test can write does, and the loop has to give up at
    MIN_SIDE_PX with the path in its message."""
    monkeypatch.setattr(images, "MAX_BYTES", 2_000)
    stubborn = noise(tmp_path / "stubborn.png", 200)

    with pytest.raises(TaskImageError) as refusal:
        load_task_images([str(stubborn)])

    said = str(refusal.value)
    assert str(stubborn) in said
    assert f"{images.MIN_SIDE_PX}x{images.MIN_SIDE_PX}" in said
    assert "unreadable" in said


def test_a_path_that_is_not_there_is_refused_by_name(tmp_path: Path) -> None:
    """Refused here, before the robot is connected and before a run directory exists: the
    alternative is a run that reaches its first request without the thing the task is about."""
    missing = tmp_path / "nope.png"

    with pytest.raises(TaskImageError) as refusal:
        load_task_images([str(missing)])

    assert str(missing) in str(refusal.value)
    assert "no such file" in str(refusal.value)


def test_a_directory_is_refused_by_name(tmp_path: Path) -> None:
    """A directory exists, so the missing-file check passes it straight through to PIL, which
    would raise something from inside a decoder about a permission error instead."""
    folder = tmp_path / "pictures"
    folder.mkdir()

    with pytest.raises(TaskImageError) as refusal:
        load_task_images([str(folder)])

    assert str(folder) in str(refusal.value)
    assert "that is a directory, not a picture" in str(refusal.value)


def test_a_file_that_is_not_a_picture_is_refused_by_name(tmp_path: Path) -> None:
    """PIL raises UnidentifiedImageError on a text file, and a traceback out of a decoder is
    not a sentence about the command line a person typed."""
    notes = tmp_path / "notes.txt"
    notes.write_bytes(b"this is not a picture, it is a shopping list\n")

    with pytest.raises(TaskImageError) as refusal:
        load_task_images([str(notes)])

    assert str(notes) in str(refusal.value)
    assert "not a picture quackd can read" in str(refusal.value)


def test_a_format_quackd_does_not_send_is_refused_and_lists_the_ones_it_does(
    tmp_path: Path,
) -> None:
    """PIL reads far more than quackd sends. A netpbm file decodes perfectly well and is still
    refused, because the refusal's job is to tell a person which file to convert."""
    netpbm = tmp_path / "raw.ppm"
    Image.new("RGB", (16, 16), "white").save(netpbm, format="PPM")

    with pytest.raises(TaskImageError) as refusal:
        load_task_images([str(netpbm)])

    said = str(refusal.value)
    assert str(netpbm) in said and "PPM" in said
    for fmt in images.FORMATS:
        assert fmt in said


def test_two_pictures_with_one_name_are_numbered_and_a_unique_name_is_left_alone(
    tmp_path: Path,
) -> None:
    """The model is told a name and the task refers to the picture by it, so two directories
    each holding a `sketch.png` must not both arrive as `sketch.png`. The file that clashes
    with nothing keeps the name the person typed, because a number there would be noise."""
    paths = [
        str(drawing(tmp_path / "left" / "sketch.png")),
        str(drawing(tmp_path / "right" / "sketch.png")),
        str(drawing(tmp_path / "cat.png")),
    ]

    loaded = load_task_images(paths)

    assert [p.name for p in loaded] == ["1-sketch.png", "2-sketch.png", "cat.png"]


def test_the_number_on_a_clashing_name_is_its_place_in_the_whole_list(tmp_path: Path) -> None:
    """The prefix is the picture's position among all of them, not a count of the clashes, so
    the same two files behind one unique file are `2-` and `3-`. That is what makes the number
    answer "which --image was that", which is the question a person asking has."""
    paths = [
        str(drawing(tmp_path / "cat.png")),
        str(drawing(tmp_path / "left" / "sketch.png")),
        str(drawing(tmp_path / "right" / "sketch.png")),
    ]

    loaded = load_task_images(paths)

    assert [p.name for p in loaded] == ["cat.png", "2-sketch.png", "3-sketch.png"]


def test_the_pictures_come_back_in_the_order_they_were_given(tmp_path: Path) -> None:
    """A task says "copy the first picture onto the second", and the names and the bytes both
    have to line up with the order of the flags for that sentence to mean anything."""
    sizes = {"wide.png": (240, 120), "tall.png": (120, 240), "square.png": (160, 160)}
    order = ["tall.png", "square.png", "wide.png"]
    for name, size in sizes.items():
        drawing(tmp_path / name, size=size)

    loaded = load_task_images([str(tmp_path / name) for name in order])

    assert [p.name for p in loaded] == order
    assert [decoded(p.png)[1] for p in loaded] == [sizes[name] for name in order]
