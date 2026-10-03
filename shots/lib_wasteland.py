"""
lib_wasteland — shared art + helpers for the WASTELAND shots
(S1 shots/s01_wasteland.py, S4 shots/s04_journey.py). Owner: WASTELAND.

Everything here is deterministic; heavy art is built once at import by the
shot modules and cached. Per-frame helpers are pure functions of their args.
"""
import math
import numpy as np

from engine import *
from engine import draw, light, noise, ease, sprite, fx
from engine import palette as P
from engine.core import to_index, from_index, bayer, dither_levels

IDX = P.NAME_TO_IDX


def rgb(name):
    return np.array(P.C(name), np.uint8)


# =====================================================================
#  PALETTE-TRUE FOG / TINT LUTS
# =====================================================================
_GREY_RAMPS = ('ink', 'dust', 'ash', 'ochre', 'steel', 'night')


def _allowed(ramps):
    return np.array([i for r in ramps for i in P.RAMP_IDX[r]], np.int64)


class Fog:
    """Mix every palette colour toward `color` in OKLab, quantised to
    `levels` steps; results restricted to muted ramps (palette-true).
    apply(img, amount) with amount scalar or (h,w) in 0..1 (dithered)."""

    def __init__(self, color, levels=8, ramps=_GREY_RAMPS, gamma=1.0):
        allowed = _allowed(ramps)
        fog_lab = P.linear_to_oklab(P.srgb_to_linear(np.array(P.C(color), np.float64)))
        self.levels = levels
        lut = np.zeros((levels + 1, P.N), np.uint8)
        lut[0] = np.arange(P.N)
        for k in range(1, levels + 1):
            a = (k / levels) ** gamma
            mixed = P.PAL_LAB * (1 - a) + fog_lab[None, :] * a
            lut[k] = P.nearest_index(mixed, space='lab', allowed=allowed)
        self.lut = lut

    def apply(self, img, amount, band=0.5, n=4, ox=0, oy=0, mask=None):
        h, w = img.shape[:2]
        a = np.asarray(amount, np.float64)
        if a.ndim == 0:
            a = np.full((h, w), float(a))
        q = dither_levels(np.clip(a, 0, 1) * self.levels, self.levels + 1, n, ox, oy, band)
        if mask is not None:
            q = np.where(mask, q, 0)
        if not q.any():
            return img
        idx = to_index(np.ascontiguousarray(img[..., :3]))
        out = img.copy()
        out[..., :3] = from_index(self.lut[q, idx])
        return out


class Remap:
    """Arbitrary palette remap: {from_name: to_name}. Used for per-depth
    recolouring of silhouettes and for storm dimming of a layer."""

    def __init__(self, mapping):
        lut = np.arange(P.N, dtype=np.uint8)
        for a, b in mapping.items():
            lut[IDX[a]] = IDX[b]
        self.lut = lut

    def apply(self, img):
        idx = to_index(np.ascontiguousarray(img[..., :3]))
        out = img.copy()
        out[..., :3] = from_index(self.lut[idx])
        return out


# =====================================================================
#  MASK / SHADING HELPERS
# =====================================================================
def shift(m, dx, dy):
    """Shift a boolean mask by (dx,dy) with False fill."""
    h, w = m.shape
    o = np.zeros_like(m)
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    o[ys0:ys1, xs0:xs1] = m[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return o


def edge(m, dx, dy):
    """Pixels of m whose neighbour at (dx,dy) is outside m."""
    return m & ~shift(m, -dx, -dy)


def put(img, mask, color):
    col = P.C(color)
    if img.shape[-1] == 4:
        img[mask] = (*col, 255)
    else:
        img[mask] = col


def put_dither(img, mask, color, alpha, ox=0, oy=0, n=4):
    """Put color on mask pixels with ordered-dither coverage alpha (scalar or map)."""
    h, w = img.shape[:2]
    a = np.asarray(alpha, np.float64)
    th = bayer(h, w, n, ox, oy)
    put(img, mask & (th < a), color)


def shade_solid(img, mask, body, lit=None, dark=None, top=None, lit_w=1, dark_w=1):
    """Fill a silhouette with body colour, lit rim on its left edge (light from
    the upper-left), optional top rim, dark rim on the right edge."""
    put(img, mask, body)
    if dark is not None:
        e = np.zeros_like(mask)
        for k in range(1, dark_w + 1):
            e |= mask & ~shift(mask, -k, 0)
        put(img, e, dark)
    if lit is not None:
        e = np.zeros_like(mask)
        for k in range(1, lit_w + 1):
            e |= mask & ~shift(mask, k, 0)
        put(img, e, lit)
    if top is not None:
        put(img, mask & ~shift(mask, 0, 1), top)


def poly(img, pts, color):
    draw.polygon(img, pts, color)


def poly_mask(h, w, pts):
    """Polygon mask, computed only inside the polygon's bounding box."""
    a = np.asarray(pts, np.float64)
    x0 = int(max(0, math.floor(a[:, 0].min()) - 1)); x1 = int(min(w, math.ceil(a[:, 0].max()) + 2))
    y0 = int(max(0, math.floor(a[:, 1].min()) - 1)); y1 = int(min(h, math.ceil(a[:, 1].max()) + 2))
    m = np.zeros((h, w), bool)
    if x0 >= x1 or y0 >= y1:
        return m
    loc = draw.polygon_mask(y1 - y0, x1 - x0, a - np.array([x0, y0]))
    m[y0:y1, x0:x1] = loc
    return m


def line_mask(h, w, x0, y0, x1, y1):
    m = np.zeros((h, w), bool)
    xs, ys = draw.line_points(x0, y0, x1, y1)
    xs = np.asarray(xs); ys = np.asarray(ys)
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    m[ys[ok], xs[ok]] = True
    return m


def lines(img, segs, color):
    for (x0, y0, x1, y1) in segs:
        draw.line(img, x0, y0, x1, y1, color)


def catenary(img, x0, y0, x1, y1, sag, color, step=None, mask_out=None):
    """Hanging cable between two points (pixel-perfect, 1px)."""
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    s = np.linspace(0, 1, n * 2)
    xs = x0 + (x1 - x0) * s
    ys = y0 + (y1 - y0) * s + sag * 4 * s * (1 - s)
    xs = np.round(xs).astype(int); ys = np.round(ys).astype(int)
    pts = sorted(set(zip(xs.tolist(), ys.tolist())))
    # thin to 1px wide steps: keep one y per x when mostly horizontal
    if abs(x1 - x0) >= abs(y1 - y0):
        d = {}
        for x, y in zip(xs.tolist(), ys.tolist()):
            d.setdefault(x, y)
        pts = list(d.items())
    px = np.array([p[0] for p in pts]); py = np.array([p[1] for p in pts])
    if mask_out is not None:
        h, w = mask_out.shape
        ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        mask_out[py[ok], px[ok]] = True
    else:
        draw.points(img, px, py, color)


# =====================================================================
#  TERRAIN
# =====================================================================
def ridge(n, seed, base, amp, scale, octaves=4, x0=0.0, gain=0.5):
    xs = (np.arange(n) + x0) / scale
    return base - amp * (noise.fbm1(xs, seed, octaves, gain=gain) - 0.5) * 2.0


def terrain_mask(h, w, heights):
    ys = np.arange(h)[:, None]
    return ys >= np.round(np.asarray(heights)[None, :w])


def depth_below(h, w, heights):
    """Distance (px) of each pixel below the surface line (neg above)."""
    ys = np.arange(h)[:, None].astype(np.float64)
    return ys - np.asarray(heights, np.float64)[None, :w]


# =====================================================================
#  STAGE HELPERS
# =====================================================================
def scrolled(lay, offx, extra=2):
    """Crop a world-wide layer for a camera whose left screen edge sits at
    world x = offx. Returns (crop, frac, ix): add crop at x = -frac."""
    ix = int(math.floor(offx))
    fr = offx - ix
    ix = max(0, min(ix, lay.shape[1] - (W + extra)))
    return lay[:, ix:ix + W + extra], fr, ix


def cam_table(duration, vel_fn, dt=1.0 / 600):
    """Integrate a velocity profile once (import time) -> position(t) fn."""
    n = int(duration / dt) + 2
    ts = np.arange(n) * dt
    v = np.array([vel_fn(t) for t in ts])
    pos = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) * 0.5 * dt)])

    def pos_at(t):
        return float(np.interp(t, ts, pos))
    return pos_at


# =====================================================================
#  PROPS — every drawer paints into an RGB/RGBA canvas in place.
#  `pal` dicts name the colours for the current depth:
#     body, lit, dark, detail  (+ optional extras)
# =====================================================================
def rng(seed):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------- city
def building(img, x, base, w, h, kind, pal, r):
    """One ruined building. kind: block|skeleton|spire|broken|lean|stepped"""
    H_, W_ = img.shape[:2]
    m = np.zeros((H_, W_), bool)
    top = base - h
    if kind == 'lean':
        sh = r.choice([-1, 1]) * r.integers(2, 4)
        pts = [(x, base), (x + w, base), (x + w + sh, top), (x + sh, top)]
        m |= poly_mask(H_, W_, pts)
    elif kind == 'stepped':
        m[top + h // 3:base, x:x + w] = True
        m[top + h // 6:base, x + 1:x + w - 1] = True
        m[top:base, x + w // 2 - 1:x + w // 2 + 1] = True
    elif kind == 'spire':
        m[top + 3:base, x:x + w] = True
        cx = x + w // 2
        m[top - h // 3:top + 3, cx:cx + 1] = True
    else:
        m[top:base, x:x + w] = True
    # broken, jagged roofline
    if kind in ('block', 'broken', 'skeleton', 'lean', 'stepped'):
        col_drop = np.zeros(W_, int)
        k = x - 1
        while k < x + w + 4:
            run = int(r.integers(1, 4))
            col_drop[max(k, 0):max(k + run, 0)] = int(r.integers(0, 3))
            k += run
        for cx in range(max(x - 3, 0), min(x + w + 4, W_)):
            d = col_drop[cx]
            if d and top + d > 0:
                m[:top + d, cx] = False
    if kind == 'broken':
        # collapsed corner: diagonal bite out of one upper side
        bw = max(2, int(w * r.uniform(0.35, 0.6)))
        bh = max(3, int(h * r.uniform(0.25, 0.45)))
        if r.random() < 0.5:
            pts = [(x + w - bw, top - 1), (x + w + 1, top - 1), (x + w + 1, top + bh)]
        else:
            pts = [(x - 1, top - 1), (x + bw, top - 1), (x - 1, top + bh)]
        m &= ~poly_mask(H_, W_, pts)
    body = m.copy()
    if kind == 'skeleton':
        # gutted frame: columns + floors, holes show sky
        holes = np.zeros_like(m)
        for fy in range(top + 1, base - 1, 3):
            for fx in range(x + 1, x + w - 1, 3):
                if r.random() < 0.8:
                    holes[fy:fy + 2, fx:fx + 2] = True
        body &= ~holes
    put(img, body, pal['body'])
    put(img, body & ~shift(body, 1, 0), pal['lit'])
    if 'dark' in pal:
        put(img, body & ~shift(body, -1, 0) & (np.arange(W_)[None, :] > x + 1), pal['dark'])
    # windows: sparse regular grid, some missing (clusters by floor)
    if kind in ('block', 'broken', 'lean', 'stepped') and w >= 4:
        win = np.zeros_like(m)
        for fy in range(top + 2, base - 2, 3):
            if r.random() < 0.25:
                continue
            for fx in range(x + 1, x + w - 1, 2):
                if r.random() < 0.72:
                    win[fy, fx] = True
        put(img, win & body & ~(body & ~shift(body, 1, 0)), pal['detail'])
    return m


def city(img, x0, x1, base, seed, pal, hmax=38, hmin=6, env=None, kinds=None):
    """Row of ruined buildings between x0..x1 on baseline `base`."""
    r = rng(seed)
    kinds = kinds or ['block', 'block', 'broken', 'skeleton', 'spire', 'lean', 'stepped', 'broken']
    x = x0
    while x < x1:
        w = int(r.integers(4, 13))
        u = (x - x0) / max(1, (x1 - x0))
        e = env(u) if env else 1.0
        h = int(max(hmin, (hmin + (hmax - hmin) * r.uniform(0.35, 1.0)) * e))
        k = kinds[int(r.integers(0, len(kinds)))]
        if k == 'spire':
            w = int(r.integers(3, 6))
        building(img, x, base, w, h, k, pal, r)
        x += w + int(r.integers(-2, 3))


def susp_bridge(img, x0, x1, deck_y, tower_h, pal, broken_at=0.55):
    """Snapped suspension bridge (far). Two towers; the deck broken and
    hanging; one tower snapped."""
    w = x1 - x0
    t1 = x0 + int(w * 0.22)
    t2 = x0 + int(w * 0.78)
    top1 = deck_y - tower_h
    top2 = deck_y - int(tower_h * 0.45)          # snapped tower
    for tx, tt in ((t1, top1), (t2, top2)):
        draw.rect(img, tx, tt, 2, deck_y - tt + 10, pal['body'])
        draw.vline(img, tx, tt, deck_y + 9, pal['lit'])
        draw.hline(img, tx - 1, tx + 2, tt + 3, pal['body'])
    # jagged top of snapped tower
    draw.pset(img, t2 + 1, top2 - 1, pal['body'])
    # deck: left intact part, right broken sagging end
    bx = x0 + int(w * broken_at)
    draw.hline(img, x0, bx, deck_y, pal['body'])
    draw.hline(img, x0, bx - 2, deck_y + 1, pal['dark'])
    draw.line(img, bx, deck_y, bx + 3, deck_y + 7, pal['body'])      # drooping broken end
    draw.line(img, bx + 9, deck_y + 9, bx + 12, deck_y + 1, pal['body'])
    draw.hline(img, bx + 12, x1, deck_y + 1, pal['body'])
    draw.hline(img, bx + 13, x1, deck_y + 2, pal['dark'])
    # main cables
    catenary(img, x0, deck_y - 2, t1, top1 + 1, 2, pal['detail'])
    catenary(img, t1 + 1, top1 + 1, bx - 1, deck_y - 3, tower_h * 0.35, pal['detail'])
    # snapped cable hanging from tower 1 toward the gap
    catenary(img, bx + 12, deck_y - 1, t2, top2 + 1, 3, pal['detail'])
    catenary(img, t2 + 1, top2 + 1, x1, deck_y - 2, 3, pal['detail'])
    # hangers
    for hx in range(t1 + 4, bx - 2, 3):
        s = (hx - t1) / max(1, (bx - 1 - t1))
        cy = top1 + 1 + (deck_y - 3 - top1 - 1) * s + tower_h * 0.35 * 4 * s * (1 - s)
        draw.vline(img, hx, int(round(cy)) + 1, deck_y - 1, pal['faint'])


def dome_on_hill(img, cx, base, pal, hill_w=70, hill_h=11, dome_w=11, dome_h=6, draw_hill=True):
    """The greenhouse: tiny dome silhouette on a distant hill (subtle)."""
    H_, W_ = img.shape[:2]
    xs = np.arange(W_)
    u = (xs - cx) / (hill_w / 2)
    hh = base - hill_h * np.clip(1 - u * u, 0, 1) ** 0.8
    hm = terrain_mask(H_, W_, hh) & (np.abs(u) < 1.4)
    if draw_hill:
        put(img, hm, pal['hill'])
        put(img, hm & ~shift(hm, 0, 1) & (xs[None, :] < cx), pal['hill_lit'])
    top = int(round(base - hill_h))
    # rounded glasshouse dome: cupola + finial, hemispherical rows, ribs, drum
    widths = [3, 7, 9, 11, 11, 13, 13]
    n = len(widths)
    lit_right = pal.get('lit_side', 'left') == 'right'
    for j, wd in enumerate(widths):
        y = top - n + 1 + j
        x0 = cx - wd // 2
        for i in range(wd):
            x = x0 + i
            col = pal['dome']
            edge = (i == wd - 1) if lit_right else (i == 0)
            if edge and j >= 1:
                col = pal['dome_lit']
            # ribs: meridians at -3, 0, +3 (curving with the dome)
            if j == 0:
                col = pal['dome']                               # lantern
            elif j >= 2 and (x - cx) in ((-3 if j >= 3 else -2), 0, (3 if j >= 3 else 2)):
                col = pal.get('rib', pal['dome_lit'])
            if j == 5:
                col = pal.get('rib', pal['dome_lit'])          # drum cornice
            draw.pset(img, x, y, col)
    draw.hline(img, cx - 7, cx + 7, top + 1, pal['dome'])
    dome_h = n
    return (cx, top - dome_h // 2)


# ---------------------------------------------------------------- structures
def lattice(img, xl0, xr0, yb, xl1, xr1, yt, color, step=4, brace=True, lit=None):
    """Tapered lattice between two legs (bottom width xl0..xr0 at yb, top
    xl1..xr1 at yt) with zig-zag bracing."""
    draw.line(img, xl0, yb, xl1, yt, lit or color)
    draw.line(img, xr0, yb, xr1, yt, color)
    if not brace:
        return
    n = max(1, int(abs(yb - yt) / step))
    pts_l, pts_r = [], []
    for i in range(n + 1):
        s = i / n
        y = yb + (yt - yb) * s
        pts_l.append((xl0 + (xl1 - xl0) * s, y))
        pts_r.append((xr0 + (xr1 - xr0) * s, y))
    for i in range(n):
        a = pts_l[i] if i % 2 == 0 else pts_r[i]
        b = pts_r[i + 1] if i % 2 == 0 else pts_l[i + 1]
        if abs(a[0] - b[0]) >= 1.5:
            draw.line(img, a[0], a[1], b[0], b[1], color)


def radio_mast(img, x, base, h, lean, pal, guy=True):
    """Leaning lattice radio mast with guy wires (one snapped)."""
    top_x = x + lean
    w0 = max(3, h // 13)
    lattice(img, x - w0 // 2, x + w0 // 2 + 1, base, top_x - 1, top_x + 1, base - h,
            pal['body'], step=max(3, w0 + 1), lit=pal['lit'])
    # platforms
    for s in (0.45, 0.72):
        yy = base - h * s
        xx = x + lean * s
        ww = w0 * (1 - s) + 2
        draw.hline(img, xx - ww / 2 - 1, xx + ww / 2 + 1, yy, pal['body'])
    # antenna rod + dead beacon
    draw.line(img, top_x, base - h, top_x + lean * 0.12, base - h - h * 0.12, pal['body'])
    draw.pset(img, top_x + lean * 0.12, base - h - h * 0.12 - 1, pal['beacon'])
    if guy:
        # guy wires (faint), one snapped & dangling
        catenary(img, x + lean * 0.7, base - h * 0.7, x - h * 0.55, base, 2, pal['wire'])
        catenary(img, x + lean * 0.4, base - h * 0.4, x + h * 0.42, base, 1.5, pal['wire'])
        draw.line(img, x + lean * 0.7, base - h * 0.7, x + lean * 0.7 + 4, base - h * 0.7 + 14, pal['wire'])
        draw.line(img, x + lean * 0.7 + 4, base - h * 0.7 + 14, x + lean * 0.7 + 2, base - h * 0.7 + 22, pal['wire'])


def pylon(img, x, base, h, pal, lattice_on=True, tilt=0.0):
    """Power-line transmission tower. Returns cable attach points."""
    bw = h * 0.30
    ww = h * 0.10
    tw = h * 0.05
    wy = base - h * 0.55
    ty = base - h
    sx = lambda s: x + tilt * s        # lean helper, s = height fraction
    # legs
    lattice(img, sx(0) - bw / 2, sx(0) + bw / 2, base, sx(0.45) - ww / 2, sx(0.45) + ww / 2, wy,
            pal['body'], step=max(3, h / 9), brace=lattice_on and h > 22, lit=pal['lit'])
    lattice(img, sx(0.45) - ww / 2, sx(0.45) + ww / 2, wy, sx(1) - tw / 2, sx(1) + tw / 2, ty,
            pal['body'], step=max(3, h / 9), brace=lattice_on and h > 30, lit=pal['lit'])
    if h <= 22:
        # small: fill the silhouette solid-ish
        pts = [(sx(0) - bw / 2, base), (sx(0.45) - ww / 2, wy), (sx(1) - tw / 2, ty),
               (sx(1) + tw / 2, ty), (sx(0.45) + ww / 2, wy), (sx(0) + bw / 2, base)]
        if h <= 12:
            draw.polygon(img, pts, pal['body'])
    # cross arms
    attach = []
    for s, aw in ((0.66, 0.44), (0.84, 0.30)):
        ay = base - h * s
        ax = sx(s)
        draw.hline(img, ax - h * aw / 2, ax + h * aw / 2, ay, pal['body'])
        if h > 25:
            draw.line(img, ax - h * aw / 2, ay, ax - 1, ay - h * 0.06, pal['body'])
            draw.line(img, ax + h * aw / 2, ay, ax + 1, ay - h * 0.06, pal['body'])
        for side in (-1, 1):
            px = ax + side * h * aw / 2
            ins = max(1, int(h * 0.04))
            draw.vline(img, px, ay + 1, ay + ins, pal['body'])
            attach.append((px, ay + ins))
    draw.pset(img, sx(1), ty - 1, pal['body'])
    return attach


def overpass(img, x0, deck_y, ground_y, pal, seed=3):
    """Collapsed highway overpass: intact deck on pillars, a broken span
    tilting down into the sand, rebar, a bent guard rail."""
    r = rng(seed)
    H_, W_ = img.shape[:2]
    th = 4
    # intact deck
    x1 = x0 + 62
    m = np.zeros((H_, W_), bool)
    m[deck_y:deck_y + th, x0:x1] = True
    # broken end: jagged
    for k in range(th):
        m[deck_y + k, x1:x1 + int(r.integers(0, 3))] = True
    # fallen span: tilted slab from the break down to the ground
    fx0, fy0 = x1 + 3, deck_y + 2
    fx1, fy1 = x1 + 40, ground_y - 1
    span = poly_mask(H_, W_, [(fx0, fy0), (fx0 + 2, fy0 - 1), (fx1 + 2, fy1 - 3), (fx1, fy1 + 1)])
    m2 = span
    # pillars
    pil = np.zeros_like(m)
    for px in (x0 + 8, x0 + 36):
        pil[deck_y + th:ground_y, px:px + 3] = True
        pil[deck_y + th:deck_y + th + 1, px - 1:px + 4] = True
    # broken pillar stub under the fallen span
    pil[ground_y - 6:ground_y, x1 + 16:x1 + 19] = True
    pil[ground_y - 7, x1 + 17] = True
    put(img, pil, pal['dark'])
    put(img, pil & ~shift(pil, 1, 0), pal['body'])
    put(img, m | m2, pal['body'])
    put(img, (m | m2) & ~shift(m | m2, 0, 1), pal['lit'])       # top edge lit
    put(img, (m | m2) & ~shift(m | m2, 0, -1), pal['dark'])     # underside
    # guard rail along the deck (posts + rail), bent at the break
    draw.hline(img, x0, x1 - 2, deck_y - 2, pal['body'])
    for px in range(x0 + 2, x1 - 2, 4):
        draw.pset(img, px, deck_y - 1, pal['body'])
    draw.line(img, x1 - 2, deck_y - 2, x1 + 3, deck_y - 6, pal['body'])
    # rebar sticking out of the broken end
    for k, (dx, dy) in enumerate(((3, -2), (4, 1), (2, 3), (5, 2))):
        draw.line(img, x1 + 1, deck_y + k, x1 + 1 + dx, deck_y + k + dy, pal['detail'])
    # a hanging lamp post, bent
    draw.vline(img, x0 + 22, deck_y - 10, deck_y - 1, pal['body'])
    draw.line(img, x0 + 22, deck_y - 10, x0 + 26, deck_y - 12, pal['body'])
    draw.pset(img, x0 + 26, deck_y - 11, pal['body'])
    # a dangling road sign from the deck edge
    draw.rect(img, x0 + 44, deck_y + th, 7, 4, pal['sign'])
    draw.hline(img, x0 + 45, x0 + 49, deck_y + th + 1, pal['lit'])
    draw.vline(img, x0 + 50, deck_y + th, deck_y + th + 5, pal['sign'])
    return (x0, x1 + 42)


def blade_pts(hx, hy, ang, length, root_w=2):
    """Polygon for a tapered turbine blade from hub at angle ang (rad)."""
    ca, sa = math.cos(ang), math.sin(ang)
    px, py = -sa, ca
    r0 = 1.5
    pts = [(hx + ca * r0 + px * root_w * 0.5, hy + sa * r0 + py * root_w * 0.5),
           (hx + ca * length * 0.35 + px * root_w * 0.75, hy + sa * length * 0.35 + py * root_w * 0.75),
           (hx + ca * length, hy + sa * length),
           (hx + ca * r0 - px * root_w * 0.4, hy + sa * r0 - py * root_w * 0.4)]
    return pts


def blade(img, hx, hy, ang, length, color, width=2):
    """Pixel-clean blade: 2px root section then 1px to the tip."""
    ca, sa = math.cos(ang), math.sin(ang)
    x_mid, y_mid = hx + ca * length * 0.55, hy + sa * length * 0.55
    xt, yt = hx + ca * length, hy + sa * length
    if width >= 2:
        draw.thick_line(img, hx, hy, x_mid, y_mid, 2, color)
    else:
        draw.line(img, hx, hy, x_mid, y_mid, color)
    draw.line(img, x_mid, y_mid, xt, yt, color)


# ---------------------------------------------------------------- vegetation (dead)
def dead_tree(img, x, y, h, seed, color, lit=None, style='orchard', trunk_w=None):
    """Dead tree skeleton. style 'orchard' = pruned open-vase fruit tree,
    'wild' = gnarled, 'snag' = broken trunk with a couple of stubs."""
    r = rng(seed)
    tw = trunk_w if trunk_w is not None else (2 if h >= 12 else 1)
    if style == 'snag':
        th = h
        lean = r.uniform(-1.5, 1.5)
        for k in range(tw):
            draw.line(img, x + k, y, x + k + lean, y - th, color)
        if lit is not None:
            draw.line(img, x, y, x + lean, y - th, lit)
        for s_ in (0.55, 0.8):
            sy_ = y - th * s_
            sx_ = x + lean * s_
            d = r.choice([-1, 1])
            draw.line(img, sx_, sy_, sx_ + d * r.integers(2, 4), sy_ - r.integers(1, 3), color)
        return
    if style == 'orchard':
        # short trunk, crotch, 3-4 scaffold limbs in a clean V, twigs at tips
        trunk_h = max(3, int(round(h * 0.34)))
        lean = int(r.integers(-1, 2)) * (1 if h > 10 else 0)
        tx, ty = x + lean, y - trunk_h
        for k in range(tw):
            draw.line(img, x + k, y, tx + k, ty, color)
        if lit is not None and tw >= 2:
            draw.line(img, x, y, tx, ty, lit)
        top = h - trunk_h
        nlimb = 3 if r.random() < 0.5 else 4
        spread = [(-0.95, 0.95), (-0.35, 1.0), (0.35, 1.0), (0.95, 0.9)] if nlimb == 4 else \
                 [(-0.85, 0.95), (0.05, 1.0), (0.9, 0.92)]
        cx0 = tx + (tw - 1) * 0.5
        for (dx, ln) in spread:
            dxp = dx * top * 0.62 + r.uniform(-0.8, 0.8)
            ex, ey = cx0 + dxp, ty - top * ln * r.uniform(0.85, 1.0)
            # limb with a slight elbow
            mx, my = cx0 + dxp * 0.45, ty - (ty - ey) * 0.5
            draw.line(img, cx0, ty, mx, my, color)
            draw.line(img, mx, my, ex, ey, color)
            if h >= 13:
                # two twigs near the tip, fanning outward
                for sgn in (-1, 1):
                    if r.random() < 0.7:
                        l2 = max(1.5, top * 0.22)
                        draw.line(img, ex, ey + 1, ex + sgn * l2 * 0.8 + dx, ey - l2 * 0.6, color)
        return
    trunk_h = h * 0.45
    lean = r.uniform(-0.8, 0.8)
    tx, ty = x + lean, y - trunk_h
    for k in range(tw):
        draw.line(img, x + k, y, tx + k, ty, color)
    if lit is not None and tw >= 2:
        draw.line(img, x, y, tx, ty, lit)

    def branch(bx, by, ang, ln, depth):
        ex = bx + math.cos(ang) * ln
        ey = by + math.sin(ang) * ln
        draw.line(img, bx, by, ex, ey, color)
        if depth <= 0 or ln < 2.5:
            return
        nb = 2 if depth > 1 else int(r.integers(1, 3))
        for i in range(nb):
            da = r.uniform(0.35, 0.7) * (1 if i % 2 == 0 else -1)
            s_ = r.uniform(0.45, 0.9)
            branch(bx + (ex - bx) * s_, by + (ey - by) * s_, ang + da, ln * r.uniform(0.45, 0.62), depth - 1)

    a = -math.pi / 2 + r.uniform(-0.3, 0.3)
    branch(tx, ty, a, (h - trunk_h) * 0.9, 3)
    branch(tx, ty + trunk_h * 0.2, a - 0.8, (h - trunk_h) * 0.6, 2)
    branch(tx, ty + trunk_h * 0.3, a + 0.9, (h - trunk_h) * 0.55, 2)


# ---------------------------------------------------------------- ground detail
def stones(img, heights, x0, x1, count, seed, pal, ymin_off=0, ymax_off=8, size=(1, 3), cast=True):
    """Scatter small stones on a terrain (clusters: lit top-left, dark bottom-right)."""
    r = rng(seed)
    H_, W_ = img.shape[:2]
    for _ in range(count):
        x = int(r.integers(x0, x1))
        if x < 0 or x >= W_:
            continue
        y = int(heights[x] + r.integers(ymin_off, ymax_off + 1))
        s = int(r.integers(size[0], size[1] + 1))
        rx, ry = s, max(1, int(round(s * 0.6)))
        ys, xs = np.ogrid[0:H_, 0:W_]
        m = ((xs - x) / (rx + 0.35)) ** 2 + ((ys - y) / (ry + 0.35)) ** 2 <= 1.0
        if cast:
            sm = shift(m, 1, 1) & ~m
            put(img, sm, pal['shadow'])
        put(img, m, pal['body'])
        put(img, m & ~shift(m, 1, 0) | (m & ~shift(m, 0, 1)), pal['lit'])
        put(img, m & ~shift(m, -1, 0) & ~(m & ~shift(m, 0, 1)), pal['dark'])


def cracked_earth(h, w, y_h, seed, cw=6.6, K=846.0, cx=None, jitter=0.8):
    """Perspective dried-mud plates (ground plane seen from low eye height).
    y_h = horizon row. A cell is ~dy/cw px wide and ~dy^2/K px tall where
    dy = y - y_h. Returns (cell_id (h,w) int64, crack (h,w) bool, lip bool).
    crack = 1px boundary between plates; lip = plate pixel just below a crack
    (the curled, lit top edge of a mud plate)."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    dy = np.maximum(ys - y_h, 1.0)
    cx = w * 0.5 if cx is None else cx
    u = (xs - cx) * cw / dy
    v = K / dy
    iu, iv = np.floor(u), np.floor(v)
    best = np.full((h, w), 1e9)
    bid = np.zeros((h, w), np.int64)
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            cu = (iu + du).astype(np.int64)
            cv = (iv + dv).astype(np.int64)
            ju = noise.rand(cu, cv, seed)
            jv = noise.rand(cu, cv, seed + 1)
            px = cu + 0.5 + jitter * (ju - 0.5)
            pv = cv + 0.5 + jitter * (jv - 0.5)
            d = (u - px) ** 2 + ((v - pv) * 1.0) ** 2
            closer = d < best
            bid = np.where(closer, cu * 7919 + cv * 104729, bid)
            best = np.where(closer, d, best)
    crack = np.zeros((h, w), bool)
    crack[:, :-1] |= bid[:, :-1] != bid[:, 1:]
    crack[:-1, :] |= bid[:-1, :] != bid[1:, :]
    lip = shift(crack, 0, 1) & ~crack
    return bid, crack, lip


# ---------------------------------------------------------------- billboard
def shear_cols(spr, slope, pivot_x=0):
    """Pixel-clean tilt: shift each column vertically by round(slope*(x-pivot))."""
    h, w = spr.shape[:2]
    offs = np.round(slope * (np.arange(w) - pivot_x)).astype(int)
    lo, hi = min(0, offs.min()), max(0, offs.max())
    out = np.zeros((h + hi - lo, w, spr.shape[2]), np.uint8)
    for x in range(w):
        o = offs[x] - lo
        out[o:o + h, x] = spr[:, x]
    return out, -lo


def billboard_panel(w, h, pal):
    """Faded old-world advert: a green hill, a tree, a sun, a slogan.
    Returns RGBA panel incl. frame; torn section shows the struts."""
    img = layer(h, w)
    draw.rect(img, 0, 0, w, h, pal['frame'])
    iw, ih = w - 2, h - 2
    pn = layer(ih, iw)
    draw.rect(pn, 0, 0, iw, ih, pal['sky'])
    draw.vgradient(pn, 0, ih, [pal['sky'], pal['sky2']], band=0.5)
    draw.disc(pn, int(iw * 0.74), int(ih * 0.30), 2, pal['sun'])
    # rolling hill (faded green) + lighter crest
    xs = np.arange(iw)
    hill = ih * 0.62 - 3.0 * np.sin(xs / iw * math.pi * 1.2 + 0.4)
    hm = terrain_mask(ih, iw, hill)
    put(pn, hm, pal['hill'])
    put(pn, hm & ~shift(hm, 0, 1), pal['hill_lit'])
    # a round tree on the hill
    tx = int(iw * 0.3)
    ty = int(hill[tx]) - 1
    draw.vline(pn, tx, ty - 3, ty, pal['trunk'])
    draw.disc(pn, tx, ty - 5, 2, pal['tree'])
    draw.pset(pn, tx - 1, ty - 6, pal['hill_lit'])
    # slogan blocks (illegible)
    draw.hline(pn, 2, int(iw * 0.5), 1, pal['text'])
    draw.hline(pn, 2, int(iw * 0.36), 3, pal['text'])
    paste(img, pn, 1, 1)
    # torn away right part: holes show the struts behind
    tear = np.zeros((h, w), bool)
    r = rng(5)
    x0 = int(w * 0.62)
    for y in range(h):
        edge_x = x0 + int(3 * math.sin(y * 0.9)) + int(r.integers(0, 2))
        tear[y, edge_x:w - 1] = True
    tear[:2, :] &= False
    tear[:, :] &= ~((np.arange(h)[:, None] >= h - 4) & (np.arange(w)[None, :] > w - 8))
    img[tear] = 0
    for sx in range(x0 + 2, w - 1, 5):
        draw.vline(img, sx, 1, h - 2, pal['strut'])
    draw.hline(img, x0, w - 2, h // 2, pal['strut'])
    draw.rect(img, w - 1, 0, 1, h, pal['frame'])
    draw.hline(img, 0, w - 1, 0, pal['frame_lit'])
    return img, x0


def car_husk(img, x, y, pal, flip=False):
    """Rusted sedan husk half-buried in sand (x,y = left, ground line)."""
    shape = [(0, 0), (1, -3), (4, -4), (7, -8), (15, -8), (18, -4), (22, -3), (23, 0)]
    if flip:
        shape = [(23 - a, b) for a, b in shape]
    pts = [(x + a, y + b) for a, b in shape]
    H_, W_ = img.shape[:2]
    m = poly_mask(H_, W_, pts)
    put(img, m, pal['body'])
    put(img, m & ~shift(m, 0, 1), pal['lit'])
    # windows (empty holes)
    wins = [(8, -7, 3, 3), (12, -7, 3, 3)] if not flip else [(9, -7, 3, 3), (13, -7, 3, 3)]
    for (a, b, ww, hh) in wins:
        draw.rect(img, x + a, y + b, ww, hh, pal['hole'])
    # rust blotches
    for (a, b) in ((3, -2), (17, -2), (19, -3)):
        draw.pset(img, x + a, y + b, pal['rust'])
    draw.pset(img, x + 5, y - 1, pal['hole'])        # wheel arch
    draw.pset(img, x + 17, y - 1, pal['hole'])


def toppled_turbine(img, x_stump, x_nac, yg, pal):
    """Fallen wind turbine: snapped stump, tower lying in the sand, nacelle.
    Returns hub (x,y) for the per-frame blades."""
    # stump with jagged top
    draw.rect(img, x_stump, yg - 9, 3, 9, pal['body'])
    draw.vline(img, x_stump, yg - 9, yg - 1, pal['lit'])
    draw.pset(img, x_stump + 1, yg - 10, pal['body'])
    draw.pset(img, x_stump + 2, yg - 11, pal['body'])
    draw.pset(img, x_stump + 2, yg - 9, pal['dark'])
    # tower lying: long tube, thickens toward the (broken) base end
    H_, W_ = img.shape[:2]
    x0 = x_stump + 6
    pts = [(x0, yg - 4), (x_nac, yg - 5), (x_nac, yg - 2), (x0, yg)]
    m = poly_mask(H_, W_, pts)
    put(img, m, pal['body'])
    put(img, m & ~shift(m, 0, 1), pal['lit'])
    put(img, m & ~shift(m, 0, -1), pal['dark'])
    # rust bands / section joints
    for jx in range(x0 + 9, x_nac - 2, 12):
        draw.vline(img, jx, yg - 4, yg - 1, pal['joint'])
    draw.pset(img, x0 + 4, yg - 3, pal['rust'])
    draw.pset(img, x0 + 20, yg - 2, pal['rust'])
    # nacelle (box, rounded), hub nose
    nm = np.zeros((H_, W_), bool)
    nm[yg - 9:yg - 1, x_nac - 1:x_nac + 10] = True
    nm[yg - 10, x_nac + 1:x_nac + 8] = True
    put(img, nm, pal['body'])
    put(img, nm & ~shift(nm, 0, 1), pal['lit'])
    put(img, nm & ~shift(nm, 1, 0), pal['lit'])
    put(img, nm & ~shift(nm, 0, -1), pal['dark'])
    draw.hline(img, x_nac + 1, x_nac + 7, yg - 6, pal['joint'])
    draw.pset(img, x_nac + 3, yg - 4, pal['hole'])         # open hatch
    draw.pset(img, x_nac + 4, yg - 4, pal['hole'])
    hx, hy = x_nac + 11, yg - 5
    draw.disc(img, hx, hy, 2, pal['body'])
    draw.pset(img, hx - 1, hy - 2, pal['lit'])
    draw.pset(img, hx + 1, hy + 1, pal['dark'])
    return hx, hy


# ---------------------------------------------------------------- cloth
def cloth(img, ax, ay, length, width, t, pal, wind=1.0, seed=0, direction=-1):
    """Tattered rag tied at (ax,ay), streaming in `direction` (-1 = left).
    Animated flag wave (call with a stepped t for the 12fps pixel feel)."""
    for i in range(length):
        s = i / max(1, length - 1)
        amp = (0.4 + 1.8 * s) * wind
        yc = ay + 0.8 * s + amp * math.sin(i * 0.75 - t * 11.0 + seed) + 0.9 * s * s
        ww = width * (1.0 - 0.45 * s)
        # tatters at the tail: skip some rows
        top = int(round(yc - ww / 2))
        bot = int(round(yc + ww / 2))
        x = ax + direction * i
        for y in range(top, bot + 1):
            if s > 0.6 and ((y + i + seed) % 3 == 0) and y == bot:
                continue
            shade = math.sin(i * 0.75 - t * 11.0 + seed + 1.2)
            col = pal['lit'] if (y == top and shade > -0.2) else (pal['dark'] if (y == bot or shade < -0.55) else pal['body'])
            draw.pset(img, x, y, col)
    # knot
    draw.pset(img, ax, ay, pal['dark'])


# ---------------------------------------------------------------- giant fallen turbine (far-mid scale)
def big_blade_mask(h, w, hx, hy, ang, length, root=5.0, snap=None):
    """Tapered, slightly swept turbine blade polygon mask from the hub:
    narrow cylindrical root, max chord at ~18% span, long taper to a point.
    snap: fraction of span where the blade is broken off (jagged end)."""
    ca, sa = math.cos(ang), math.sin(ang)
    px, py = -sa, ca
    lead, trail = [], []
    n = 14
    s_end = 1.0 if snap is None else snap
    for i in range(n + 1):
        s = s_end * i / n
        d = 1.5 + (length - 1.5) * s
        if s < 0.06:
            wd = 2.6
        elif s < 0.2:
            wd = 2.6 + (root - 2.6) * ((s - 0.06) / 0.14) ** 0.6
        else:
            wd = 0.8 + (root - 0.8) * ((1 - s) / 0.8) ** 1.25
        bow = 1.0 * math.sin(math.pi * s)
        cx, cy = hx + ca * d + px * bow, hy + sa * d + py * bow
        # leading edge nearly straight, chord bulges to the trailing side
        lead.append((cx - px * wd * 0.3, cy - py * wd * 0.3))
        trail.append((cx + px * wd * 0.7, cy + py * wd * 0.7))
    if snap is not None:
        # jagged break
        ex, ey = lead[-1]
        tx, ty = trail[-1]
        mx, my = (ex + tx) / 2 + ca * 1.5, (ey + ty) / 2 + sa * 1.5
        pts = lead + [(mx, my)] + trail[::-1]
    else:
        pts = lead + trail[::-1]
    return poly_mask(h, w, pts)


def fallen_turbine(img, x_stump, x_nac, yg, pal):
    """Big toppled wind turbine for the far-mid layer. Returns hub (x,y)."""
    H_, W_ = img.shape[:2]
    # snapped stump, jagged top, rust streak
    st = np.zeros((H_, W_), bool)
    st[yg - 14:yg, x_stump:x_stump + 4] = True
    st[yg - 16:yg - 14, x_stump + 2:x_stump + 4] = True
    st[yg - 15, x_stump + 1] = True
    put(img, st, pal['body'])
    put(img, st & ~shift(st, 1, 0), pal['lit'])
    put(img, st & ~shift(st, -1, 0), pal['dark'])
    draw.vline(img, x_stump + 2, yg - 10, yg - 6, pal['rust'])
    # the tower lying on the dune, receding: thick near the stump, thinner away
    x0 = x_stump + 7
    tw = np.zeros((H_, W_), bool)
    tw |= poly_mask(H_, W_, [(x0, yg - 5), (x_nac + 1, yg - 8), (x_nac + 1, yg - 4), (x0, yg + 1)])
    put(img, tw, pal['body'])
    put(img, tw & ~shift(tw, 0, 1), pal['lit'])
    put(img, tw & ~shift(tw, 0, -1), pal['dark'])
    for jx in range(x0 + 12, x_nac - 3, 15):
        yy = int(round(yg - 5 + (jx - x0) / (x_nac - x0) * -3))
        draw.vline(img, jx, yy + 1, yy + 4, pal['joint'])
    for (a, b) in ((x0 + 6, 0), (x0 + 30, -1), (x0 + 31, -1)):
        draw.pset(img, a, yg - 2 + b, pal['rust'])
    # broken tower end at the stump side (jagged, dark interior)
    draw.vline(img, x0, yg - 4, yg, pal['hole'])
    draw.pset(img, x0 - 1, yg - 3, pal['body'])
    # nacelle
    nm = np.zeros((H_, W_), bool)
    ny = yg - 16
    nm[ny:ny + 9, x_nac:x_nac + 17] = True
    nm[ny - 1, x_nac + 2:x_nac + 14] = True
    nm[ny + 9, x_nac + 1:x_nac + 15] = True
    put(img, nm, pal['body'])
    put(img, nm & ~shift(nm, 0, 1), pal['lit'])
    put(img, nm & ~shift(nm, 1, 0), pal['lit'])
    put(img, nm & ~shift(nm, 0, -1), pal['dark'])
    put(img, nm & ~shift(nm, -1, 0), pal['dark'])
    draw.hline(img, x_nac + 2, x_nac + 13, ny + 4, pal['joint'])
    draw.rect(img, x_nac + 4, ny + 5, 3, 2, pal['hole'])      # torn hatch
    draw.pset(img, x_nac + 9, ny + 2, pal['rust'])
    draw.pset(img, x_nac + 10, ny + 2, pal['rust'])
    # spinner / hub nose
    hx, hy = x_nac + 19, ny + 4
    draw.ellipse(img, hx, hy, 2, 3, pal['body'])
    draw.pset(img, hx - 1, hy - 2, pal['lit'])
    draw.pset(img, hx + 2, hy + 1, pal['dark'])
    draw.pset(img, hx + 1, hy + 2, pal['dark'])
    return hx, hy


# =====================================================================
#  FX — pure functions of t
# =====================================================================
class VeilField:
    """Precomputed soft noise field for drifting dust veils. sample(x0, y0,
    w, h) returns the (h,w) float field at integer world offset x0."""

    def __init__(self, w, h, seed, sx=70.0, sy=9.0, octaves=3, lo=0.35, hi=0.75):
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
        n = noise.fbm2(xs / sx, ys / sy, seed=seed, octaves=octaves)
        self.f = np.clip((n - lo) / (hi - lo), 0, 1)
        self.w, self.h = w, h

    def sample(self, x0, y0, w, h):
        x0 = int(x0) % (self.w - w - 1)
        return self.f[y0:y0 + h, x0:x0 + w]


def band_profile(h, y0, y1, soft_top=6.0, soft_bot=4.0):
    ys = np.arange(h).astype(np.float64)
    a = np.clip((ys - y0) / soft_top, 0, 1) * np.clip((y1 - ys) / soft_bot, 0, 1)
    return a[:, None]


def motes(img, t, seed, n, area, vel, colors, weights=None, sway=(1.5, 0.3), ox=0.0):
    """Floating dust motes (1px). area in screen coords; ox shifts with a layer."""
    x, y, r = fx.field(t, n, seed, area, vel=vel, jitter=(abs(vel[0]) * 0.4 + 1, 1.5), sway=sway)
    fx.draw_field(img, x - ox, y, r, colors, weights)


def sand_streams(img, surf, x_off, t, seed, n, gust, colors, speed=(45, 70), length=(10, 34),
                 yoff=(0, 3), area_w=None, wave=0.6, amp=1.2, groups=4):
    """Wind-blown sand ribbons skimming a surface (right -> left).
    surf: surface heights indexed by WORLD x. x_off: world x of column 0.
    Ribbons come in gust bursts (`groups` families sharing a rhythm); each is
    a sinuous line with a bright solid head and a dotted, thinning tail.
    gust (0..1) sets how many are alive."""
    H_, W_ = img.shape[:2]
    cols = [P.C(c) for c in colors]
    for k in range(n):
        g = k % groups
        thr = noise.rand(k, 6, seed)
        if thr > gust:
            continue
        life = 1.1 + noise.rand(g, 5, seed) * 0.9            # family rhythm
        cyc = t / life + noise.rand(g, 1, seed) + 0.13 * noise.rand(k, 1, seed)
        age = cyc - math.floor(cyc)
        gen = int(math.floor(cyc))
        sp = speed[0] + noise.rand(k, 2, seed + gen) * (speed[1] - speed[0])
        ln = length[0] + noise.rand(k, 3, seed + gen) * (length[1] - length[0])
        yo = yoff[0] + noise.rand(k, 4, seed + gen) * (yoff[1] - yoff[0])
        x_start = x_off + W_ + 30 - noise.rand(k, gen * 3 + 1, seed + 9) * (W_ + 60)
        head = x_start - sp * age * life
        fade = min(1.0, age * 4.0) * min(1.0, (1 - age) * 3.0)
        L_ = int(ln * (0.35 + 0.65 * fade))
        if L_ < 3:
            continue
        ph = noise.rand(k, 7, seed) * 6.28
        for j in range(L_):
            wx = head + j
            cx = int(math.floor(wx - x_off))
            if cx < 0 or cx >= W_:
                continue
            s_ = j / L_
            if s_ > 0.5 and (j + k) % 2 == 1:
                continue
            if s_ > 0.78 and (j + k) % 4 != 0:
                continue
            wi = int(math.floor(wx)) % len(surf)
            yy = surf[wi] - 1 - yo - amp * math.sin(wx * 0.11 + t * 7.0 + ph) * (0.4 + s_)
            cy = int(round(yy))
            if 0 <= cy < H_:
                col = cols[0] if s_ < 0.22 else (cols[1] if s_ < 0.6 else cols[-1])
                img[cy, cx, :3] = col
                if img.shape[-1] == 4:
                    img[cy, cx, 3] = 255


def puff(img, x, y, age, seed, colors=('dust6', 'dust5'), wind=-10.0):
    """Tiny footstep dust puff (age 0..1)."""
    if age < 0 or age > 1:
        return
    n = 4
    for k in range(n):
        ang = math.pi + (k / (n - 1) - 0.5) * 1.6
        d = 1.0 + age * (2.5 + k * 0.5)
        px = x + math.cos(ang) * d + wind * age * 0.25
        py = y - 0.5 - abs(math.sin(ang)) * d * 0.45 - age * 1.0
        if (k + int(age * 6)) % 3 == 2 and age > 0.5:
            continue
        col = colors[0] if age < 0.45 else colors[1]
        draw.pset(img, px, py, col)


# =====================================================================
#  RAMP SHIFT — move pixels up/down their OWN ramp (warm stays warm)
# =====================================================================
def _ramp_step_luts():
    dark = np.arange(P.N, dtype=np.uint8)
    lite = np.arange(P.N, dtype=np.uint8)
    for rname, idxs in P.RAMP_IDX.items():
        for k, i in enumerate(idxs):
            dark[i] = idxs[max(0, k - 1)]
            lite[i] = idxs[min(len(idxs) - 1, k + 1)]
    # keep the saturated accents untouched (robot core / sprout)
    for rname in ('core', 'leaf'):
        for i in P.RAMP_IDX[rname]:
            dark[i] = i
            lite[i] = i
    luts = {0: np.arange(P.N, dtype=np.uint8)}
    d, l = luts[0].copy(), luts[0].copy()
    for k in range(1, 5):
        d = dark[d]
        l = lite[l]
        luts[-k] = d.copy()
        luts[k] = l.copy()
    return luts


RAMP_LUTS = _ramp_step_luts()


def ramp_shift(img, level, band=0.5, n=4, ox=0, oy=0, mask=None, rows=None):
    """Shift colours along their own ramps by a (dithered) float level map
    (+ = lighter, - = darker). rows=(y0,y1) restricts the work."""
    if rows is not None:
        y0, y1 = rows
        sub = img[y0:y1]
        lv = level[y0:y1] if np.ndim(level) == 2 else level
        mk = mask[y0:y1] if mask is not None else None
        img = img.copy()
        img[y0:y1] = ramp_shift(sub, lv, band, n, ox, oy + y0, mk)
        return img
    h, w = img.shape[:2]
    lv = np.asarray(level, np.float64)
    if lv.ndim == 0:
        lv = np.full((h, w), float(lv))
    lv = lv[:h, :w]
    fl = np.floor(lv)
    fr = lv - fl
    if band < 1.0:
        fr = np.clip((fr - 0.5) / max(band, 1e-3) + 0.5, 0, 1)
    q = (fl + (bayer(h, w, n, ox, oy) < fr)).astype(np.int32)
    q = np.clip(q, -4, 4)
    if mask is not None:
        q = np.where(mask, q, 0)
    if not q.any():
        return img
    idx = to_index(np.ascontiguousarray(img[..., :3]))
    out_idx = idx.copy()
    for k in (-4, -3, -2, -1, 1, 2, 3, 4):
        m = q == k
        if m.any():
            out_idx[m] = RAMP_LUTS[k][idx[m]]
    out = img.copy()
    out[..., :3] = from_index(out_idx)
    return out


def pool_local(img, cx, cy, radius, peak=2.0, mode='warm', sy=1.0, gamma=1.2, band=0.45,
               ox=0, oy=0, shadow='night'):
    """Light pool computed only in its bounding window; Bayer locked to the
    image's world origin (ox, oy) so the dither does not crawl."""
    h, w = img.shape[:2]
    rx, ry = radius + 1, radius * sy + 1
    x0, x1 = int(max(0, cx - rx)), int(min(w, cx + rx + 1))
    y0, y1 = int(max(0, cy - ry)), int(min(h, cy + ry + 1))
    if x0 >= x1 or y0 >= y1:
        return img
    lv = light.radial_level(y1 - y0, x1 - x0, cx - x0, cy - y0, radius, peak, gamma, sy)
    out = img.copy()
    out[y0:y1, x0:x1] = light.apply_light(img[y0:y1, x0:x1], lv, mode=mode, shadow=shadow,
                                          ox=ox + x0, oy=oy + y0, band=band)
    return out


# =====================================================================
#  LOW-SUN HELPERS (S4 dusk): rim light + long cast shadows
# =====================================================================
def long_shadow(mask, base_y, k_len=2.4, k_flat=0.1, direction=-1, max_h=None):
    """Project a standing silhouette onto the ground plane for a low sun.
    A pixel `h` px above base_y lands h*k_len px sideways (direction -1 =
    left) and h*k_flat px below the base line. Returns a bool mask."""
    h_, w_ = mask.shape
    out = np.zeros_like(mask)
    top = 0
    ys = np.where(mask.any(axis=1))[0]
    if len(ys) == 0:
        return out
    top = ys.min()
    for y in range(top, min(base_y + 1, h_)):
        row = mask[y]
        if not row.any():
            continue
        hgt = base_y - y
        if max_h is not None and hgt > max_h:
            continue
        ty = int(round(base_y + hgt * k_flat))
        if ty < 0 or ty >= h_:
            continue
        s0 = int(round(hgt * k_len)); s1 = int(round((hgt + 1) * k_len))
        for sft in range(s0, max(s1, s0 + 1)):
            dx = direction * sft
            if dx < 0:
                out[ty, :w_ + dx] |= row[-dx:]
            elif dx > 0:
                out[ty, dx:] |= row[:w_ - dx]
            else:
                out[ty] |= row
    return out


def rim_edges(mask, dx=1, dy=0, width=1):
    """Pixels on the side of `mask` facing direction (dx,dy)."""
    e = np.zeros_like(mask)
    for k in range(1, width + 1):
        e |= mask & ~shift(mask, -dx * k, -dy * k)
    return e


# =====================================================================
#  FOOTPRINTS — a trail in the sand behind the walker (world-anchored)
# =====================================================================
def footprints(img, x_off, xs_world, ys, colors, only_on=None, lit=None, w=1, keep=None):
    """Stamp prints at world x positions (list) on rows ys (list, same len).
    colors: print colour. only_on: set of RGB tuples the print may land on
    (e.g. only sand). lit: colour of a 1px lip above the print (sun side).
    keep: list of bools (skip prints the wind has erased)."""
    H_, W_ = img.shape[:2]
    col = P.C(colors)
    lcol = P.C(lit) if lit is not None else None
    allowed = None if only_on is None else {tuple(P.C(c)) for c in only_on}
    for k, (wx, y) in enumerate(zip(xs_world, ys)):
        if keep is not None and not keep[k]:
            continue
        cx = int(round(wx - x_off))
        for dx in range(w):
            xx = cx + dx
            if not (0 <= xx < W_ and 0 <= y < H_):
                continue
            if img.shape[-1] == 4 and img[y, xx, 3] == 0:
                continue
            if allowed is not None and tuple(img[y, xx, :3]) not in allowed:
                continue
            img[y, xx, :3] = col
            if lcol is not None and y - 1 >= 0 and dx == w - 1 and xx + 1 < W_:
                img[y - 1, xx + 1, :3] = lcol


def boulder(img, x0, yb, wd, hg, seed, body, rim=None, top=None, rim_dx=1, facets=None, n=7):
    """Angular rock silhouette standing on yb. rim = colour of the sun-side
    edge (rim_dx=+1: light from the right, -1: from the left). facets =
    colour of 2-3 inner facet lines."""
    r = rng(seed)
    pts = [(x0, yb + 1)]
    for i in range(1, n):
        u = i / n
        hh_ = hg * (math.sin(math.pi * u) ** 0.7) * r.uniform(0.75, 1.05)
        pts.append((x0 + wd * u + r.uniform(-1.5, 1.5), yb - hh_))
    pts.append((x0 + wd, yb + 1))
    h_, w_ = img.shape[:2]
    m = poly_mask(h_, w_, pts)
    put(img, m, body)
    if facets is not None:
        for k in range(2 + (wd > 40)):
            i = int(r.integers(2, n - 1))
            fx0, fy0 = pts[i]
            draw.line(img, fx0, fy0 + 1, fx0 + r.uniform(-wd * 0.15, wd * 0.15), yb, facets)
    if top is not None:
        put(img, m & ~shift(m, 0, 1), top)
    if rim is not None:
        put(img, rim_edges(m, rim_dx, 0), rim)
    return m


class RayField:
    """Precomputed geometry for light shafts from a fixed point (screen
    space): per frame only the angular falloff is evaluated."""

    def __init__(self, h, w, ox, oy):
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
        self.ang = np.arctan2(ys - oy, xs - ox)
        self.dist = np.sqrt((xs - ox) ** 2 + (ys - oy) ** 2)

    def level(self, angles, widths, length, peak=1.0, gamma=1.0, inner=0.0, inner_soft=1.0):
        lv = np.zeros_like(self.ang)
        for a, wd in zip(angles, widths):
            da = np.abs((self.ang - a + math.pi) % (2 * math.pi) - math.pi)
            np.maximum(lv, np.clip(1 - da / wd, 0, 1), out=lv)
        fall = np.clip(1 - self.dist / length, 0, 1) ** gamma
        out = peak * lv * fall
        if inner > 0:
            out *= np.clip((self.dist - inner) / inner_soft, 0, 1)
        return out


class MaskCache:
    """Deterministic memo for per-frame polygon masks keyed by quantised
    parameters (pure: same key -> same mask)."""

    def __init__(self, fn, maxn=256):
        self.fn, self.d, self.maxn = fn, {}, maxn

    def __call__(self, *key):
        m = self.d.get(key)
        if m is None:
            if len(self.d) > self.maxn:
                self.d.clear()
            m = self.fn(*key)
            self.d[key] = m
        return m


# =====================================================================
#  HAND-PLACED CLOUD / DUST STREAKS (lens shapes, tapered, gently lumpy)
# =====================================================================
def streak_shape(h, w, cx, cy, length, thick, seed=0, lump=0.35, flat_bottom=True, tilt=0.0):
    """A long tapered dust/cloud streak. Returns (body, top_edge, bottom_edge)
    boolean masks. thick = max thickness in px (centre). The upper outline is
    lumpy (billowing), the underside flatter (stratus)."""
    body = np.zeros((h, w), bool)
    x0, x1 = int(max(0, cx - length / 2 - 1)), int(min(w, cx + length / 2 + 2))
    if x0 >= x1:
        return body, body.copy(), body.copy()
    xs = np.arange(x0, x1).astype(np.float64)
    u = (xs - cx) / (length / 2)
    env = np.clip(1 - u * u, 0, 1) ** 0.75
    wob = noise.fbm1(xs / 9.0 + seed * 3.1, seed + 11, 3) - 0.5
    lumps = 1.0 + lump * 2.0 * (noise.fbm1(xs / 5.0 + seed * 1.7, seed + 23, 2) - 0.5)
    half = thick * env * lumps
    yc = cy + tilt * (xs - cx) + 0.8 * wob
    top = yc - half * (0.65 if flat_bottom else 0.5)
    bot = yc + half * (0.35 if flat_bottom else 0.5)
    ys = np.arange(h)[:, None].astype(np.float64)
    sub = (ys >= np.round(top)[None, :]) & (ys <= np.round(bot)[None, :]) & (half[None, :] >= 0.45)
    body[:, x0:x1] = sub
    top_e = body & ~shift(body, 0, 1)
    bot_e = body & ~shift(body, 0, -1)
    return body, top_e, bot_e
