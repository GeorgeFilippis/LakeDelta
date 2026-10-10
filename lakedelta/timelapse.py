"""Label timelapse frames: year and season on top, the yearly area as a sparkline below."""

from __future__ import annotations

import io
import os
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont, ImageSequence

from .config import WATER

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")


def _font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(os.path.join(FONT_DIR, name), size)
    except OSError:
        return ImageFont.load_default(size)


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def annotate(gif: bytes | Sequence[Image.Image], years: Sequence[int], season: str,
             areas: dict[int, float], fps: float = 2.0) -> bytes:
    """Return a new GIF with a caption band and an area sparkline on every frame."""
    if isinstance(gif, (bytes, bytearray)):
        src = Image.open(io.BytesIO(gif))
        frames = [f.convert("RGB") for f in ImageSequence.Iterator(src)]
    else:
        frames = [f.convert("RGB") for f in gif]
    n = min(len(frames), len(years))
    if n == 0:
        return b""
    w, h = frames[0].size
    band = max(30, h // 14)
    spark_h = max(34, h // 10)
    title_font = _font("BarlowSemiCondensed-SemiBold.ttf", int(band * 0.58))
    vals = [areas.get(int(y)) for y in years[:n]]
    finite = [v for v in vals if v is not None]
    vmin, vmax = (min(finite), max(finite)) if finite else (0.0, 1.0)
    if vmax - vmin < 1e-9:
        vmax = vmin + 1.0
    water = _hex(WATER)

    out = []
    for i in range(n):
        frame = frames[i]
        canvas = Image.new("RGB", (w, h + band + spark_h), (15, 26, 33))
        canvas.paste(frame, (0, band))
        d = ImageDraw.Draw(canvas)
        d.text((12, band / 2), f"{years[i]}  {season}", font=title_font, fill=(240, 244, 246),
               anchor="lm")
        v = vals[i]
        if v is not None:
            label = f"{v:,.1f} km²" if v >= 10 else f"{v:,.2f} km²"
            d.text((w - 12, band / 2), label, font=title_font, fill=(240, 244, 246), anchor="rm")
        # sparkline of the yearly area, the current year highlighted
        y0 = band + h + 6
        y1 = band + h + spark_h - 8
        x0, x1 = 12, w - 12
        pts = []
        for j, vv in enumerate(vals):
            if vv is None:
                continue
            x = x0 + (x1 - x0) * (j / max(n - 1, 1))
            y = y1 - (y1 - y0) * ((vv - vmin) / (vmax - vmin))
            pts.append((x, y))
        if len(pts) >= 2:
            d.line(pts, fill=(120, 140, 152), width=2)
        if v is not None:
            x = x0 + (x1 - x0) * (i / max(n - 1, 1))
            y = y1 - (y1 - y0) * ((v - vmin) / (vmax - vmin))
            d.line([(x, y0 - 2), (x, y1 + 2)], fill=(80, 96, 106), width=1)
            d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=water, outline=(240, 244, 246), width=2)
        out.append(canvas)

    buf = io.BytesIO()
    paletted = [f.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
                for f in out]
    paletted[0].save(buf, format="GIF", save_all=True, append_images=paletted[1:],
                     duration=int(1000 / fps), loop=0, disposal=1)
    return buf.getvalue()
