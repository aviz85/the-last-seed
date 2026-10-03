"""
Pixel-perfect drawing primitives. All functions draw IN PLACE on an RGB
(h,w,3) or RGBA (h,w,4) numpy canvas and accept palette color names
('dust5'), '#hex' or (r,g,b). No anti-aliasing anywhere — ever.
"""
import numpy as np
from . import palette as P
from .core import dither_mask, bayer


def _put(img, mask, color):
    col = P.C(color)
    if img.shape[-1] == 4:
        img[mask] = (*col, 255)
    else:
        img[mask] = col


def pset(img, x, y, color):
    x, y = int(x), int(y)
    h, w = img.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        col = P.C(color)
        img[y, x, :3] = col
        if img.shape[-1] == 4:
            img[y, x, 3] = 255


def points(img, xs, ys, color):
    """Plot many points (arrays). Out-of-bounds points are skipped."""
    xs = np.asarray(xs).astype(np.int64).ravel()
    ys = np.asarray(ys).astype(np.int64).ravel()
    h, w = img.shape[:2]
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    col = P.C(color)
    img[ys[ok], xs[ok], :3] = col
    if img.shape[-1] == 4:
        img[ys[ok], xs[ok], 3] = 255


def rect(img, x, y, w, h, color):
    x, y, w, h = int(x), int(y), int(w), int(h)
    H_, W_ = img.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W_), min(y + h, H_)
    if x0 < x1 and y0 < y1:
        col = P.C(color)
        img[y0:y1, x0:x1, :3] = col
        if img.shape[-1] == 4:
            img[y0:y1, x0:x1, 3] = 255


def hline(img, x0, x1, y, color):
    if x1 < x0:
        x0, x1 = x1, x0
    rect(img, x0, y, x1 - x0 + 1, 1, color)


def vline(img, x, y0, y1, color):
    if y1 < y0:
        y0, y1 = y1, y0
    rect(img, x, y0, 1, y1 - y0 + 1, color)


def line_points(x0, y0, x1, y1):
    """Bresenham line pixel coordinates (lists)."""
    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    xs, ys = [], []
    while True:
        xs.append(x0); ys.append(y0)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy; x0 += sx
        if e2 <= dx:
            err += dx; y0 += sy
    return xs, ys


def line(img, x0, y0, x1, y1, color):
    xs, ys = line_points(x0, y0, x1, y1)
    points(img, xs, ys, color)


def disc_mask(h, w, cx, cy, r):
    """Boolean mask of a pixel-art filled circle (clean edges for small r)."""
    ys, xs = np.ogrid[:h, :w]
    return (xs - cx) ** 2 + (ys - cy) ** 2 <= (r + 0.35) ** 2


def disc(img, cx, cy, r, color):
    h, w = img.shape[:2]
    if r <= 0:
        pset(img, cx, cy, color)
        return
    x0, x1 = int(max(cx - r - 1, 0)), int(min(cx + r + 2, w))
    y0, y1 = int(max(cy - r - 1, 0)), int(min(cy + r + 2, h))
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.ogrid[y0:y1, x0:x1]
    m = (xs - cx) ** 2 + (ys - cy) ** 2 <= (r + 0.35) ** 2
    sub = img[y0:y1, x0:x1]
    _put(sub, m, color)


def ring(img, cx, cy, r, color, thickness=1):
    h, w = img.shape[:2]
    x0, x1 = int(max(cx - r - 2, 0)), int(min(cx + r + 3, w))
    y0, y1 = int(max(cy - r - 2, 0)), int(min(cy + r + 3, h))
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.ogrid[y0:y1, x0:x1]
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    m = (d2 <= (r + 0.35) ** 2) & (d2 > (r - thickness + 0.35) ** 2)
    _put(img[y0:y1, x0:x1], m, color)


def ellipse(img, cx, cy, rx, ry, color):
    h, w = img.shape[:2]
    x0, x1 = int(max(cx - rx - 1, 0)), int(min(cx + rx + 2, w))
    y0, y1 = int(max(cy - ry - 1, 0)), int(min(cy + ry + 2, h))
    if x0 >= x1 or y0 >= y1 or rx <= 0 or ry <= 0:
        return
    ys, xs = np.ogrid[y0:y1, x0:x1]
    m = ((xs - cx) / (rx + 0.35)) ** 2 + ((ys - cy) / (ry + 0.35)) ** 2 <= 1.0
    _put(img[y0:y1, x0:x1], m, color)


def thick_line(img, x0, y0, x1, y1, width, color):
    """Line with a round brush of given width (>=1)."""
    if width <= 1:
        return line(img, x0, y0, x1, y1, color)
    xs, ys = line_points(x0, y0, x1, y1)
    r = (width - 1) / 2.0
    for x, y in zip(xs, ys):
        disc(img, x, y, r, color)


def polygon_mask(h, w, pts):
    """Even-odd scanline fill mask for a polygon given as [(x,y), ...]."""
    pts = np.asarray(pts, dtype=np.float64)
    mask = np.zeros((h, w), bool)
    n = len(pts)
    ys = np.arange(h) + 0.5
    xs = np.arange(w) + 0.5
    for y_i, yc in enumerate(ys):
        xints = []
        for i in range(n):
            xa, ya = pts[i]
            xb, yb = pts[(i + 1) % n]
            if (ya <= yc < yb) or (yb <= yc < ya):
                xints.append(xa + (yc - ya) * (xb - xa) / (yb - ya))
        if len(xints) < 2:
            continue
        xints.sort()
        for k in range(0, len(xints) - 1, 2):
            mask[y_i] |= (xs >= xints[k]) & (xs < xints[k + 1])
    return mask


def polygon(img, pts, color):
    h, w = img.shape[:2]
    _put(img, polygon_mask(h, w, pts), color)


def fill_below(img, heights, color, x_offset=0):
    """Fill every column x from heights[x] down to the bottom (terrain).
    heights: 1D array (len >= canvas width) of y values (float ok)."""
    h, w = img.shape[:2]
    hs = np.asarray(heights, dtype=np.float64)[x_offset:x_offset + w]
    ys = np.arange(h)[:, None]
    m = ys >= np.round(hs)[None, :]
    _put(img, m, color)
    return m


def vgradient(img, y0, y1, colors, n=4, x0=0, x1=None, ox=0, oy=0, mask=None, band=0.5):
    """Vertical banded gradient through a list of palette colors between rows
    y0..y1, with ordered-dither transitions between bands (classic pixel sky).
    band = fraction of each band used for the dithered transition (1 = fully
    dithered, 0.3-0.6 = clean bands with dithered seams, 0 = hard bands)."""
    h, w = img.shape[:2]
    x1 = w if x1 is None else x1
    y0i, y1i = int(max(y0, 0)), int(min(y1, h))
    if y1i <= y0i:
        return
    k = len(colors)
    ys = np.arange(y0i, y1i)
    t = (ys - y0) / max(1e-6, (y1 - y0)) * (k - 1)          # 0..k-1
    th = bayer(y1i - y0i, x1 - x0, n, ox + x0, oy + y0i)
    fl = np.floor(t)[:, None]
    fr = (t[:, None] - fl)
    if band < 1.0:
        fr = np.clip((fr - 0.5) / max(band, 1e-3) + 0.5, 0, 1)
    q = (fl + (th < fr)).astype(np.int32)
    q = np.clip(q, 0, k - 1)
    sub = img[y0i:y1i, x0:x1]
    for i, c in enumerate(colors):
        m = q == i
        if mask is not None:
            m &= mask[y0i:y1i, x0:x1]
        _put(sub, m, c)


def radial_bands(img, cx, cy, radii, colors, n=4, soft=0.0, ox=0, oy=0):
    """Concentric bands: colors[i] inside radii[i] (radii descending order is
    NOT required — drawn from outermost to innermost). soft>0 dithers band
    edges over that many pixels."""
    h, w = img.shape[:2]
    order = np.argsort(radii)[::-1]
    ys, xs = np.ogrid[:h, :w]
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    th = bayer(h, w, n, ox, oy)
    for i in order:
        r = radii[i]
        if soft > 0:
            a = np.clip((r - d) / soft + 0.5, 0, 1)
            m = th < a
        else:
            m = d <= r
        _put(img, m, colors[i])


def outline_mask(mask, diagonal=False):
    """Pixels adjacent (4- or 8-neighbour) to mask but not in it."""
    m = mask
    o = np.zeros_like(m)
    o[1:, :] |= m[:-1, :]
    o[:-1, :] |= m[1:, :]
    o[:, 1:] |= m[:, :-1]
    o[:, :-1] |= m[:, 1:]
    if diagonal:
        o[1:, 1:] |= m[:-1, :-1]
        o[:-1, :-1] |= m[1:, 1:]
        o[1:, :-1] |= m[:-1, 1:]
        o[:-1, 1:] |= m[1:, :-1]
    return o & ~m
