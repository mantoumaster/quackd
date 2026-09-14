"""Draw `docs/assets/social-preview.png`, the 1280x640 card GitHub shows when a link is shared.

    uv run python docs/assets/social_preview.py        # writes docs/assets/social-preview.png

Run it from a checkout. The sdist carries `docs/` and not `web/`, so the mark this pastes is
not in a published package, the same way `hero3d.py` needs upstream's model fetched first.

The card carried the one-liner, so it went stale the day the positioning changed
([ADR-0035](../adr/0035-one-cli-for-all-your-robots.md)). It had been made by hand and the
script was never committed, which is why `docs/assets/README.md` used to point at a commit
that does not contain one. This is that script, so the next person who changes a word here
does not have to redraw a duck.

The mark is loaded rather than drawn. It is `web/assets/duck-mark.png`, the duck head the
README opens with, the same art quackd.org shows in a browser tab, and it is pasted at its own
resolution after the card is resampled, so the one piece of finished art here is never
resized. Until 0.9 this script transcribed `logo.svg`'s geometry into Pillow instead, and
`logo.svg` is gone with it. The two panels on the right are real `sim2d` renders of a three
robot arena, through the same `render_topdown` and `render_duckcam` a run writes its GIF
with, so what the card shows is what the simulator draws.

Everything this script draws is drawn at `SS` times scale and resampled down, which is what
gives the type its edges. The mark is the exception above: it is art rather than geometry,
and it lands after that resample at the size it was exported. Fonts are the one
machine-dependent part: a bold and a regular face are looked up from `_FACES` and the script
stops with a readable error if neither is found, rather than silently falling back to a
bitmap font that would not fit.

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
#: The mark, kept once. `web/assets` serves it to the browser demo and the README opens with
#: it, so this reads that copy rather than taking a second one into `docs/assets`. The tab
#: icon beside it, `favicon-96.png`, is the same drawing exported small, not this file.
MARK = HERE.parent.parent / "web" / "assets" / "duck-mark.png"

W, H = 1280, 640
SS = 4  # supersample factor: draw big, resample once, keep the curves clean

# Sampled from the card this replaces, so the two sit side by side in a timeline without one
# of them looking like a different project. The brand colours are no longer among them: the
# mark brings its own and the name is set in the same white as the headline, because a
# wordmark that recolours two letters competes with a piece of art that did not need help.
BG = (24, 24, 28)
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


# ── the lockup ──────────────────────────────────────────────────────────────────────────
#
# Where the mark's own pixels land on the finished 1280x640 card. The name is placed off the
# mark rather than at a coordinate of its own, so moving one moves both.
MARK_XY = (57, 58)  # top left of the 256x256 PNG, chosen so its ink starts at the headline's x
MARK_GAP = 48  # from the mark's right-hand ink to the start of the name
WORDMARK_PT = 112


def load_mark() -> tuple[Image.Image, tuple[int, int, int, int]]:
    """The mark and the box its opaque pixels occupy on the finished card.

    The PNG is padded with transparency on three sides, so its file corner is not its visual
    corner. Everything is aligned to the ink, and the ink is measured rather than assumed, so
    redrawing the duck one day moves the name with it.
    """
    mark = Image.open(MARK).convert("RGBA")
    box = mark.getbbox() or (0, 0, *mark.size)
    x, y = MARK_XY
    return mark, (x + box[0], y + box[1], x + box[2], y + box[3])


def draw_wordmark(d: ImageDraw.ImageDraw, x: float, cy: float, size: int) -> None:
    """`quackd` in the headline's white, its ink centred on `cy`. Supersampled coordinates.

    Measured rather than placed. `_FACES` resolves to a different typeface on every machine,
    so a nominal point size is not an ink height, and a hardcoded baseline would sit the name
    at a different height on Windows than on CI.
    """
    f = font("bold", size)
    _, top, _, bottom = d.textbbox((x, 0.0), "quackd", font=f, anchor="ls")
    d.text((x, cy - (top + bottom) / 2), "quackd", font=f, fill=HEADLINE, anchor="ls")


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

    mark, ink = load_mark()
    draw_wordmark(d, *s(ink[2] + MARK_GAP, (ink[1] + ink[3]) / 2), WORDMARK_PT)

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

    # The mark goes on last, at its own resolution. Drawing it into the supersampled canvas
    # would mean scaling it up four times and back down once, which is two resamples of the
    # only art here that was not drawn by this script.
    card = canvas.resize((W, H), Image.Resampling.LANCZOS)
    card.alpha_composite(mark, MARK_XY)
    return card.convert("RGB")


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
