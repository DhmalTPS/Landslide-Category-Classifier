"""App logo: a hill drawn as nested contour arcs; one arc breaks and its segment slides down the flank.

Shape: three concentric contour arcs (a topographic profile of a hill) with round caps; on the
right flank a segment of the outer contour (the slope surface) has broken away and slipped down-slope. Background
is fully transparent.
Colour: one continuous mint -> emerald -> deep-teal sweep across every contour (a single light
source from the crest), against a sunset amber -> coral -> rose for the moving mass
(complementary families, each graded analogously). One design, two formats:
  LOGO_SVG       vector, used in the page header and as the browser-tab icon
  png(size)      raster RGBA (Pillow), favicon.ico fallback and phone home-screen icon
"""
import io
import math

import numpy as np
from PIL import Image, ImageDraw

GROUND = [(0.0, (0xB4, 0xF0, 0xD3)), (0.42, (0x34, 0xB0, 0x8C)), (1.0, (0x08, 0x40, 0x45))]   # mint, emerald, deep teal
SIGNAL = [(0.0, (0xFF, 0xC8, 0x62)), (0.5, (0xFF, 0x74, 0x5E)), (1.0, (0xE0, 0x42, 0x82))]   # amber, coral, rose

C = (29.0, 45.0)                 # common centre of the contours (on the ground line), centred in 64x64
RADII = (23.0, 14.5, 6.0)        # outer (slope surface), middle, inner contour
W = 5.6                          # stroke width
BREAK = (18.0, 58.0)             # angular gap (degrees, 0 = right, 90 = top) in the outer contour
SHIFT = (5.6, 6.6)               # the broken segment slid down the flank by this offset
GROUND_AXIS = ((42, 17), (6, 50))


def _pt(r, deg, dx=0.0, dy=0.0):
    a = math.radians(deg)
    return (C[0] + r * math.cos(a) + dx, C[1] - r * math.sin(a) + dy)


def _arc_d(r, a0, a1, dx=0.0, dy=0.0):
    """SVG path of a contour arc from angle a0 down to a1 (counter-clockwise on screen = over the top)."""
    x0, y0 = _pt(r, a0, dx, dy)
    x1, y1 = _pt(r, a1, dx, dy)
    large = 1 if abs(a0 - a1) > 180 else 0
    return f"M{x0:.2f} {y0:.2f}A{r} {r} 0 {large} 1 {x1:.2f} {y1:.2f}"


def _stops(stops):
    return "".join(f'<stop offset="{o}" stop-color="#%02x%02x%02x"/>' % c for o, c in stops)


_seg0 = _pt(RADII[0], BREAK[1], *SHIFT)
_seg1 = _pt(RADII[0], BREAK[0], *SHIFT)
STROKE = f'fill="none" stroke-width="{W}" stroke-linecap="round"'

LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" aria-label="Landslide type check">'
    '<defs>'
    f'<linearGradient id="lg-ground" gradientUnits="userSpaceOnUse" x1="{GROUND_AXIS[0][0]}" y1="{GROUND_AXIS[0][1]}" '
    f'x2="{GROUND_AXIS[1][0]}" y2="{GROUND_AXIS[1][1]}">{_stops(GROUND)}</linearGradient>'
    f'<linearGradient id="lg-signal" gradientUnits="userSpaceOnUse" x1="{_seg0[0]:.2f}" y1="{_seg0[1]:.2f}" '
    f'x2="{_seg1[0]:.2f}" y2="{_seg1[1]:.2f}">{_stops(SIGNAL)}</linearGradient>'
    '</defs>'
    f'<path d="{_arc_d(RADII[0], 180, BREAK[1])}" stroke="url(#lg-ground)" {STROKE}/>'
    f'<path d="{_arc_d(RADII[0], BREAK[0], 0)}" stroke="url(#lg-ground)" {STROKE}/>'
    f'<path d="{_arc_d(RADII[1], 180, 0)}" stroke="url(#lg-ground)" {STROKE}/>'
    f'<path d="{_arc_d(RADII[2], 180, 0)}" stroke="url(#lg-ground)" {STROKE}/>'
    f'<path d="{_arc_d(RADII[0], BREAK[1], BREAK[0], *SHIFT)}" stroke="url(#lg-signal)" {STROKE}/>'
    '</svg>')


def _gradient(n, stops, p0, p1):
    """RGBA image (n x n) with a multi-stop linear gradient from p0 to p1 (pixel coordinates)."""
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    d = np.array(p1, np.float32) - np.array(p0, np.float32)
    t = np.clip(((xx - p0[0]) * d[0] + (yy - p0[1]) * d[1]) / float(d @ d), 0, 1)
    offs = np.array([o for o, _ in stops], np.float32)
    cols = np.array([c for _, c in stops], np.float32)
    rgb = np.stack([np.interp(t, offs, cols[:, i]) for i in range(3)], -1)
    return Image.fromarray(np.dstack([rgb, np.full((n, n), 255, np.float32)]).astype(np.uint8), "RGBA")


def _draw_arc(draw, k, r, a0, a1, dx=0.0, dy=0.0):
    """Round-capped arc from a0 down to a1 (degrees, math convention) into a mask."""
    cx, cy = (C[0] + dx) * k, (C[1] + dy) * k
    rr = r * k
    # PIL angles are clockwise from +x on screen: math angle a -> PIL -a
    draw.arc([cx - rr - W * k / 2, cy - rr - W * k / 2, cx + rr + W * k / 2, cy + rr + W * k / 2],
             start=-a0, end=-a1, fill=255, width=round(W * k))
    for a in (a0, a1):
        x, y = _pt(r, a, dx, dy)
        draw.ellipse([x * k - W * k / 2, y * k - W * k / 2, x * k + W * k / 2, y * k + W * k / 2], fill=255)


def png(size=180):
    """The same mark rasterised (4x supersampled) on a transparent background, as PNG bytes."""
    n = size * 4
    k = n / 64
    out = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    ground = Image.new("L", (n, n), 0)
    gd = ImageDraw.Draw(ground)
    _draw_arc(gd, k, RADII[0], 180, BREAK[1])
    _draw_arc(gd, k, RADII[0], BREAK[0], 0)
    _draw_arc(gd, k, RADII[1], 180, 0)
    _draw_arc(gd, k, RADII[2], 180, 0)
    out.paste(_gradient(n, GROUND, [v * k for v in GROUND_AXIS[0]], [v * k for v in GROUND_AXIS[1]]), (0, 0), ground)
    seg = Image.new("L", (n, n), 0)
    _draw_arc(ImageDraw.Draw(seg), k, RADII[0], BREAK[1], BREAK[0], *SHIFT)
    out.paste(_gradient(n, SIGNAL, [v * k for v in _seg0], [v * k for v in _seg1]), (0, 0), seg)
    buf = io.BytesIO()
    out.resize((size, size), Image.LANCZOS).save(buf, "PNG")
    return buf.getvalue()


if __name__ == "__main__":   # preview: python src/logo.py  -> results/logo_preview.png (transparent)
    from pathlib import Path
    Path(__file__).resolve().parent.parent.joinpath("results", "logo_preview.png").write_bytes(png(256))
    print("wrote results/logo_preview.png")
