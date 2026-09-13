"""Draw `docs/assets/social-preview.png`, the 1280x640 card GitHub shows when a link is shared.

    uv run python docs/assets/social_preview.py        # writes docs/assets/social-preview.png

The card carried the one-liner, so it went stale the day the positioning changed
([ADR-0035](../adr/0035-one-cli-for-all-your-robots.md)). It had been made by hand and the
script was never committed, which is why `docs/assets/README.md` used to point at a commit
that does not contain one. This is that script, so the next person who changes a word here
does not have to redraw a duck.

Two things are drawn rather than loaded. The mark and the wordmark are `logo.svg`'s own
geometry, transcribed, because rasterising SVG would add a dependency for one asset. The two
panels on the right are real `sim2d` renders of a three robot arena, through the same
`render_topdown` and `render_duckcam` a run writes its GIF with, so what the card shows is
what the simulator draws.

Everything is drawn at `SS` times scale and resampled down, which is what gives the rounded
corners and the type their edges. Fonts are the one machine-dependent part: a bold and a
regular face are looked up from `_FACES` and the script stops with a readable error if
neither is found, rather than silently falling back to a bitmap font that would not fit.

Uploading it is still manual: GitHub has no API for Settings, Social preview.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from quackd.sim2d.render import render_duckcam, render_topdown
from quackd.sim2d.world import World

HERE = Path(__file__).resolve().parent
OUT = HERE / "social-preview.png"

W, H = 1280, 640
SS = 4  # supersample factor: draw big, resample once, keep the curves clean

# Sampled from the card this replaces, so the two sit side by side in a timeline without
# one of them looking like a different project. `logo.svg` is the source for the first four.
BG = (24, 24, 28)
YELLOW = (245, 197, 24)
YELLOW_DARK = (217, 165, 14)
LAVENDER = (183, 166, 223)
PURPLE = (156, 134, 214)
INK = (43, 43, 48)
GREY = (85, 85, 92)
WHITE = (255, 255, 255)
EYE = (26, 26, 30)
HEADLINE = (245, 245, 245)
BODY = (200, 200, 205)
MUTED = (140, 140, 150)
LABEL = (160, 160, 165)

#: Bold face, then regular. First readable path wins. Windows, macOS, then the usual Linux
#: packages; DejaVu is last because Pillow vendors it and it is the one that is always there.
_FACES: dict[str, tuple[str, ...]] = {
    "bold": (
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ),
    "regular": (
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ),
}


def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    """The first face on this machine, at `size` points of the supersampled canvas."""
    for path in _FACES[weight]:
        try:
            return ImageFont.truetype(path, size * SS)
        except OSError:
            continue
    raise SystemExit(
        f"no {weight} font found. Tried: {', '.join(_FACES[weight])}.\n"
        "Add the path of a TTF on this machine to _FACES and run it again."
    )


def s(x: float, y: float) -> tuple[float, float]:
    """A point in card coordinates, on the supersampled canvas everything is drawn on."""
    return x * SS, y * SS


# ── the mark, transcribed from logo.svg ─────────────────────────────────────────────────
#
# logo.svg draws in a 470x150 viewBox with the bird under `translate(8,6)`. Everything below
# is in those coordinates and `_mark` applies the scale, so a change upstream can be copied
# across by reading the two files next to each other.


def _bird_body(d: ImageDraw.ImageDraw, ox: float, oy: float, k: float) -> None:
    """Legs, feet and torso: the part of the group that is not rotated."""

    def p(x: float, y: float) -> tuple[float, float]:
        return ox + x * k, oy + y * k

    for pts in ((58, 96, 50, 116, 62, 130), (84, 96, 78, 116, 92, 130)):
        d.line(
            [p(pts[0], pts[1]), p(pts[2], pts[3]), p(pts[4], pts[5])],
            fill=INK,
            width=round(9 * k),
            joint="curve",
        )
        # `joint="curve"` rounds the corner but not the ends, which the SVG rounds too
        for cx, cy in ((pts[0], pts[1]), (pts[4], pts[5])):
            r = 4.5 * k
            d.ellipse(
                [p(cx, cy)[0] - r, p(cx, cy)[1] - r, p(cx, cy)[0] + r, p(cx, cy)[1] + r], fill=INK
            )
    for cx, cy in ((50, 116), (78, 116)):
        r = 5.5 * k
        d.ellipse(
            [p(cx, cy)[0] - r, p(cx, cy)[1] - r, p(cx, cy)[0] + r, p(cx, cy)[1] + r], fill=GREY
        )
    for x in (46, 78):  # feet
        d.rounded_rectangle([p(x, 126), p(x + 34, 139)], radius=6.5 * k, fill=YELLOW)
    for cx in (74, 106):
        r = 3 * k
        d.ellipse(
            [p(cx, 132.5)[0] - r, p(cx, 132.5)[1] - r, p(cx, 132.5)[0] + r, p(cx, 132.5)[1] + r],
            fill=INK,
        )
    d.rounded_rectangle([p(34, 60), p(100, 100)], radius=16 * k, fill=LAVENDER)
    d.rounded_rectangle([p(42, 66), p(60, 76)], radius=5 * k, fill=(70, 70, 76))
    d.rounded_rectangle([p(70, 34), p(81, 66)], radius=4 * k, fill=INK)  # neck
    for cy in (46, 58):
        r = 4 * k
        d.ellipse(
            [p(75.5, cy)[0] - r, p(75.5, cy)[1] - r, p(75.5, cy)[0] + r, p(75.5, cy)[1] + r],
            fill=GREY,
        )


def _bird_head(size: tuple[int, int], ox: float, oy: float, k: float) -> Image.Image:
    """Beak, head and eye on their own layer, so the SVG's `rotate(-8 92 26)` is one call."""
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    def p(x: float, y: float) -> tuple[float, float]:
        return ox + x * k, oy + y * k

    d.rounded_rectangle([p(72, 43), p(130, 56)], radius=6.5 * k, fill=YELLOW)
    d.line([p(80, 49.5), p(124, 49.5)], fill=YELLOW_DARK, width=round(2 * k))
    d.rounded_rectangle([p(54, 2), p(132, 48)], radius=23 * k, fill=LAVENDER)
    d.rounded_rectangle([p(64, 16), p(134, 46)], radius=15 * k, fill=INK)  # visor
    for r, fill in ((12.5, WHITE), (8.5, YELLOW), (4.0, EYE)):
        rr = r * k
        d.ellipse(
            [p(104, 30)[0] - rr, p(104, 30)[1] - rr, p(104, 30)[0] + rr, p(104, 30)[1] + rr],
            fill=fill,
        )
    rr = 1.6 * k
    d.ellipse(
        [
            p(106.5, 27.5)[0] - rr,
            p(106.5, 27.5)[1] - rr,
            p(106.5, 27.5)[0] + rr,
            p(106.5, 27.5)[1] + rr,
        ],
        fill=WHITE,
    )
    pivot = p(92, 26)
    return layer.rotate(-8, resample=Image.Resampling.BICUBIC, center=pivot)


def draw_mark(canvas: Image.Image, x: float, y: float, k: float) -> None:
    """The bird from `logo.svg`, its (8,6) group offset folded in, at `k` pixels per unit."""
    ox, oy = x + 8 * k, y + 6 * k
    _bird_body(ImageDraw.Draw(canvas), ox, oy, k)
    canvas.alpha_composite(_bird_head(canvas.size, ox, oy, k))


def draw_wordmark(d: ImageDraw.ImageDraw, x: float, y: float, size: int) -> None:
    """`quack` in yellow, `d` in purple, which is the one thing the wordmark must keep."""
    f = font("bold", size)
    d.text((x, y), "quack", font=f, fill=YELLOW, anchor="ls")
    d.text((x + d.textlength("quack", font=f), y), "d", font=f, fill=PURPLE, anchor="ls")


# ── the panels ──────────────────────────────────────────────────────────────────────────


def panels(side: int) -> tuple[Image.Image, Image.Image]:
    """One arena with three robots in it, and what one of them sees.

    A seeded `sim2d` world, rendered by the same two functions a run's GIF uses. Three ducks
    rather than one because the card is about a flock now, and they spawn in three of the
    four colorways, so the top-down view shows three robots and the duck cam shows two peers
    the way a teammate really appears to the detector.
    """
    world = World(seed=41, n_ducks=3)
    return (
        render_topdown(world, size=side).convert("RGBA"),
        render_duckcam(world, size=side).convert("RGBA"),
    )


def build() -> Image.Image:
    canvas = Image.new("RGBA", (W * SS, H * SS), (*BG, 255))
    d = ImageDraw.Draw(canvas)

    draw_mark(canvas, *s(78, 48), 2.02 * SS)
    d = ImageDraw.Draw(canvas)
    draw_wordmark(d, *s(408, 272), 112)

    d.text(
        s(70, 432),
        "One CLI for all your robots.",
        font=font("regular", 40),
        fill=HEADLINE,
        anchor="ls",
    )
    d.text(
        s(70, 476),
        "Connect them, command them, and let them work together.",
        font=font("regular", 25),
        fill=BODY,
        anchor="ls",
    )
    d.text(
        s(70, 514),
        "an LLM for a brain each  ·  one .duck file  ·  bundled simulator  ·  MCP",
        font=font("regular", 25),
        fill=BODY,
        anchor="ls",
    )
    d.text(
        s(70, 606),
        "github.com/rokbenko/quackd  ·  Apache 2.0  ·  unofficial, not affiliated with "
        "Pollen Robotics",
        font=font("regular", 17),
        fill=MUTED,
        anchor="ls",
    )

    side = 150 * SS
    small = font("regular", 15)
    for img, x, label in zip(panels(side), (950, 1110), ("world view", "duck cam"), strict=True):
        left, top = x * SS, 392 * SS
        canvas.alpha_composite(img, (left, top))
        d.rectangle([left, top, left + side - 1, top + side - 1], outline=(70, 62, 55), width=SS)
        d.text((left, top + side + 26 * SS), label, font=small, fill=LABEL, anchor="ls")

    return canvas.resize((W, H), Image.Resampling.LANCZOS).convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-o", "--out", type=Path, default=OUT, help=f"default: {OUT}")
    args = ap.parse_args()
    card = build()
    assert card.size == (W, H), card.size
    card.save(args.out, optimize=True)
    print(f"wrote {args.out} ({args.out.stat().st_size // 1024} KB)")
    print("upload it under Settings, Social preview: GitHub has no API for it")


if __name__ == "__main__":
    main()
