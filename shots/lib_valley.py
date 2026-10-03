"""
lib_valley — the exterior world of the climax.  Owner: BLOOM.

Used by  shots/s08_bloom_ext.py  (S8b BLOOM OUTSIDE, the money shot)
and      shots/s09_valley.py     (S9a THE VALLEY, the afterglow).

ONE layout, painted in two states from the same geometry:
    DEAD   dust-choked overcast, grey-brown dunes and cracked earth, dead
           skeleton trees, pylons, a toppled turbine, a car husk, a ruined
           city + snapped bridge on the horizon, a dry riverbed
    ALIVE  meadows with composed flower drifts, leafy/blossoming trees, vines
           over the ruins, blue sky with cumulus, a stream reflecting the sky

Seven parallax layers (sky, far, v1, v2, hill, v3, fg) are authored and
composited in PALETTE-INDEX space (uint8 palette indices + binary alpha), so
lighting, fog and dead/alive compositing are palette-true by construction
and cheap.  Every layer carries an ARRIVAL-TIME map T (seconds on the S8b
clock): the moment the colour wave reaches that pixel (a pseudo-3D ground
plane distance from the tree, compressed for the far field).  Secondary
growth — grass blades growing up, flowers popping, vines climbing, trees
leafing out, the stream filling from its source — is authored as per-pixel
delays added to T.  The sky's T map is the tear in the dust.

Hero pieces: the burst Victorian dome (paned glass, iron ribs, the tree's
interior canopy behind the glass), the TREE (twisting bark strokes revealed
by time + a canopy of foliage MASSES built from z-buffered, shaded sphere
clumps with leafy silhouettes and blossom clusters; a live renderer for the
growth burst and a z-baked per-mass renderer for the settled canopy + sway),
roots and glowing cracks racing over the hill.

Coordinates: 'rest-screen' px = the screen at camera rest (S8b t < 3).
A layer's pixel (i, j) sits at rest-screen (x0 + j, y0 + i) and moves with
the camera by its parallax factor p (camera(): shared crane path, S9a
continues it on T = 5 + t so the move flows through the dissolve).
"""
import math
import numpy as np

from engine import *
from engine import draw, light, noise, ease, fx
from engine import palette as P
from engine.core import bayer as _bayer

IDX = P.NAME_TO_IDX
PAL = P.PAL
NP = P.N
TAU = 2 * math.pi


def ci(name):
    return IDX[name]


def cis(names):
    return np.array([IDX[n] for n in names], np.uint8)


# =====================================================================
#  CACHED DITHER MAPS
# =====================================================================
_BAY = {}


def bay(h, w, n=4, ox=0, oy=0):
    k = (h, w, n, ox % n, oy % n)
    b = _BAY.get(k)
    if b is None:
        b = _bayer(h, w, n, ox, oy)
        _BAY[k] = b
    return b


def quant(v, band=0.45, h=None, w=None, n=4, ox=0, oy=0):
    """Float field -> int levels with ordered-dither seams of width `band`."""
    v = np.asarray(v, np.float64)
    fl = np.floor(v)
    fr = v - fl
    if band < 1.0:
        fr = np.clip((fr - 0.5) / max(band, 1e-3) + 0.5, 0, 1)
    hh, ww = v.shape
    return (fl + (bay(hh, ww, n, ox, oy) < fr)).astype(np.int32)


# =====================================================================
#  PALETTE LUTS (index -> index)
# =====================================================================
def _ramp_lut(k, overflow=None, underflow=None):
    """Step k places along each colour's own ramp (clamped).  overflow /
    underflow: {ramp: [extra names beyond the end]} to continue a ramp."""
    lut = np.arange(NP, dtype=np.uint8)
    for i in range(NP):
        r = P.RAMP_OF[i]
        ids = list(P.RAMP_IDX[r])
        if overflow and r in overflow:
            ids = ids + [IDX[n] for n in overflow[r]]
        if underflow and r in underflow:
            ids = [IDX[n] for n in underflow[r]] + ids
        pos = ids.index(i)
        lut[i] = ids[int(np.clip(pos + k, 0, len(ids) - 1))]
    return lut


_OVER = {'meadow': ['gold4'], 'leaf': [], 'sky': ['white1'], 'dust': ['white0'],
         'bark': ['ochre8'], 'glass': ['white0'], 'gold': ['white0'], 'bloom': ['white0'],
         'violet': ['bloom5'], 'red': ['bloom5'], 'ash': ['dust9'], 'white': []}
_UNDER = {'meadow': ['ink0', 'leaf0'], 'leaf': ['ink0'], 'sky': ['night1'], 'dust': ['ink0'],
          'bark': ['ink0'], 'violet': ['night1'], 'bloom': ['violet0'], 'red': ['bark0'],
          'gold': ['bark1', 'ochre2'], 'ash': ['ink1'], 'glass': ['ink1'], 'paint': ['ink1'],
          'ochre': ['ink0'], 'rust': ['ink0'], 'steel': ['ink0'], 'white': ['sky6', 'sky7']}
RAMP = {k: _ramp_lut(k, _OVER, _UNDER) for k in range(-4, 5)}


def ramp_light(idx, level, band=0.45, ox=0, oy=0, sel=None):
    """Light/darken by whole ramp steps (dithered).  level scalar or map."""
    lv = np.asarray(level, np.float64)
    if lv.ndim == 0:
        lv = np.full(idx.shape, float(lv))
    q = np.clip(quant(lv, band, ox=ox, oy=oy), -4, 4)
    out = idx.copy()
    for k in range(-4, 5):
        if k == 0:
            continue
        m = q == k
        if sel is not None:
            m &= sel
        if m.any():
            out[m] = RAMP[k][idx[m]]
    return out


_ELUT = {}


def prewarm():
    for m in ('sun', 'green', 'flash'):
        elut(m, 'night')


def elut(mode, shadow='night'):
    k = (mode, shadow)
    if k not in _ELUT:
        _ELUT[k] = light._lut(mode, shadow)
    return _ELUT[k]


def tint_light(idx, level, mode='sun', shadow='night', band=0.45, ox=0, oy=0, sel=None):
    """Engine light modes (tinted), in index space."""
    lv = np.asarray(level, np.float64)
    if lv.ndim == 0:
        lv = np.full(idx.shape, float(lv))
    q = np.clip(quant(lv, band, ox=ox, oy=oy), -light.MAXL, light.MAXL)
    if sel is not None:
        q = np.where(sel, q, 0)
    return elut(mode, shadow)[q + light.MAXL, idx]


def mix_lut(target, levels=6, ramps=None, gamma=1.0):
    """LUT[k] mixes every colour k/levels of the way toward `target` in OKLab
    (nearest palette colour, optionally restricted to some ramps)."""
    allowed = None
    if ramps is not None:
        allowed = np.array([i for r in ramps for i in P.RAMP_IDX[r]], np.int64)
    tl = P.linear_to_oklab(P.srgb_to_linear(np.array(P.C(target), np.float64)))
    lut = np.zeros((levels + 1, NP), np.uint8)
    lut[0] = np.arange(NP)
    for k in range(1, levels + 1):
        a = (k / levels) ** gamma
        m = P.PAL_LAB * (1 - a) + tl[None, :] * a
        lut[k] = P.nearest_index(m, space='lab', allowed=allowed)
    return lut


def apply_mix(idx, lut, amount, band=0.5, ox=0, oy=0):
    lv = lut.shape[0] - 1
    a = np.asarray(amount, np.float64)
    if a.ndim == 0:
        a = np.full(idx.shape, float(a))
    q = np.clip(quant(np.clip(a, 0, 1) * lv, band, ox=ox, oy=oy), 0, lv)
    return lut[q, idx]


# =====================================================================
#  MASK HELPERS
# =====================================================================
def shift(m, dx, dy, fill=False):
    h, w = m.shape
    o = np.full_like(m, fill)
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    if ys0 < ys1 and xs0 < xs1:
        o[ys0:ys1, xs0:xs1] = m[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return o


def edge(m, dx, dy):
    """Pixels of m whose neighbour at (dx, dy) is outside m."""
    return m & ~shift(m, -dx, -dy)


def grow(m, r=1, diag=True):
    o = m.copy()
    for _ in range(r):
        n = o.copy()
        n |= shift(o, 1, 0) | shift(o, -1, 0) | shift(o, 0, 1) | shift(o, 0, -1)
        if diag:
            n |= shift(o, 1, 1) | shift(o, -1, -1) | shift(o, 1, -1) | shift(o, -1, 1)
        o = n
    return o


def runs_min(m, k):
    """Keep only horizontal runs of length >= k."""
    keep = np.zeros_like(m)
    h, w = m.shape
    for start in range(-(k - 1), 1):
        win = np.ones_like(m)
        for j in range(k):
            win &= shift(m, -(start + j), 0)
        # win True at x means m[x+start .. x+start+k-1] all True
        for j in range(k):
            keep |= shift(win, start + j, 0)
    return keep & m


# =====================================================================
#  INDEX CANVAS
# =====================================================================
class Paint:
    """Index-space canvas with binary alpha, in rest-screen coordinates."""

    def __init__(self, x0, y0, w, h):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.i = np.zeros((h, w), np.uint8)
        self.a = np.zeros((h, w), bool)
        self.X = np.broadcast_to((x0 + np.arange(w))[None, :].astype(np.float64), (h, w))
        self.Y = np.broadcast_to((y0 + np.arange(h))[:, None].astype(np.float64), (h, w))

    def put(self, mask, name):
        self.i[mask] = IDX[name] if isinstance(name, str) else name
        self.a[mask] = True

    def put_levels(self, mask, q, names):
        ids = cis(names)
        qq = np.clip(q, 0, len(names) - 1)
        self.i[mask] = ids[qq[mask]]
        self.a[mask] = True

    def px(self, x, y, name):
        j, i = int(round(x - self.x0)), int(round(y - self.y0))
        if 0 <= i < self.h and 0 <= j < self.w:
            self.i[i, j] = IDX[name]
            self.a[i, j] = True

    def pts(self, xs, ys, name):
        j = np.round(np.asarray(xs) - self.x0).astype(int).ravel()
        i = np.round(np.asarray(ys) - self.y0).astype(int).ravel()
        ok = (i >= 0) & (i < self.h) & (j >= 0) & (j < self.w)
        self.i[i[ok], j[ok]] = IDX[name]
        self.a[i[ok], j[ok]] = True

    def line(self, x0, y0, x1, y1, name):
        xs, ys = draw.line_points(x0, y0, x1, y1)
        self.pts(xs, ys, name)

    def mask_line(self, x0, y0, x1, y1):
        m = np.zeros((self.h, self.w), bool)
        xs, ys = draw.line_points(x0 - self.x0, y0 - self.y0, x1 - self.x0, y1 - self.y0)
        xs, ys = np.asarray(xs), np.asarray(ys)
        ok = (ys >= 0) & (ys < self.h) & (xs >= 0) & (xs < self.w)
        m[ys[ok], xs[ok]] = True
        return m

    def mask_poly(self, pts):
        return draw.polygon_mask(self.h, self.w, [(x - self.x0, y - self.y0) for x, y in pts])

    def mask_disc(self, cx, cy, r):
        return (self.X - cx) ** 2 + (self.Y - cy) ** 2 <= (r + 0.35) ** 2

    def mask_ellipse(self, cx, cy, rx, ry):
        return ((self.X - cx) / (rx + 0.35)) ** 2 + ((self.Y - cy) / (ry + 0.35)) ** 2 <= 1

    def copy(self):
        o = Paint(self.x0, self.y0, self.w, self.h)
        o.i[:] = self.i
        o.a[:] = self.a
        return o


PALA = np.concatenate([PAL, np.full((NP, 1), 255, np.uint8)], 1)


def rgba_of(idx, a):
    out = PALA[idx]
    out[..., 3][~a] = 0
    return out


def rgb_of(idx):
    return PAL[idx]


# =====================================================================
#  NOISE TEXTURES (precomputed, tileable enough for our use)
# =====================================================================
_NT = 256
_ys, _xs = np.mgrid[0:_NT, 0:_NT].astype(np.float64)
NTEX_LEAF = noise.fbm2(_xs / 2.2, _ys / 2.2, seed=901, octaves=2)      # leafy 2-3 px clusters
NTEX_EDGE = noise.fbm2(_xs / 1.7, _ys / 1.7, seed=902, octaves=2)      # jagged silhouettes
NTEX_BIG = noise.fbm2(_xs / 9.0, _ys / 9.0, seed=903, octaves=3)
del _ys, _xs


def ntex(tex, xs, ys):
    return tex[np.asarray(ys, np.int64) % _NT, np.asarray(xs, np.int64) % _NT]


# =====================================================================
#  SPHERE-CLUSTER RENDERER (foliage, cumulus, bushes)
# =====================================================================
LIGHT3 = np.array([-0.52, -0.66, 0.54])
LIGHT3 = LIGHT3 / np.linalg.norm(LIGHT3)


def spheres_zbuffer(h, w, ox, oy, S, jag=1.3, seed=0):
    """Render a union of spheres with a z-buffer.
    S: float array (N, >=4): cx, cy, cz, r  (rest-screen coords; ox, oy =
    rest-screen position of the canvas's top-left).
    Returns (sid (h,w) int32, -1 = empty, zbuf)."""
    sid = np.full((h, w), -1, np.int32)
    zb = np.full((h, w), -1e9)
    for k in range(len(S)):
        cx, cy, cz, r = S[k, 0] - ox, S[k, 1] - oy, S[k, 2], S[k, 3]
        if r <= 0.3:
            continue
        R = r + jag + 1
        x0, x1 = int(max(math.floor(cx - R), 0)), int(min(math.ceil(cx + R) + 1, w))
        y0, y1 = int(max(math.floor(cy - R), 0)), int(min(math.ceil(cy + R) + 1, h))
        if x0 >= x1 or y0 >= y1:
            continue
        ys = np.arange(y0, y1)[:, None]
        xs = np.arange(x0, x1)[None, :]
        dx = xs - cx
        dy = ys - cy
        d2 = dx * dx + dy * dy
        # jagged, leafy silhouette: radius perturbed by a texture locked to the sphere
        tx = (xs - int(cx) + 37 * k + seed * 11)
        ty = (ys - int(cy) + 53 * k + seed * 7)
        rr = r + jag * (ntex(NTEX_EDGE, tx, ty) - 0.5) * 2.0
        inside = d2 <= rr * rr
        z = cz + np.sqrt(np.clip(r * r - d2, 0, None))
        sub_z = zb[y0:y1, x0:x1]
        upd = inside & (z > sub_z)
        sub_z[upd] = z[upd]
        sid[y0:y1, x0:x1][upd] = k
    return sid, zb


def spheres_shade(sid, zb, ox, oy, S, light3=LIGHT3, tex_amp=0.22, seed=0):
    """Diffuse term in [-1, 1] per pixel from the visible sphere's normal,
    with a leafy texture locked to each sphere."""
    h, w = sid.shape
    m = sid >= 0
    out = np.zeros((h, w))
    if not m.any():
        return out, m
    yy, xx = np.nonzero(m)
    k = sid[m]
    cx = S[k, 0] - ox
    cy = S[k, 1] - oy
    r = np.maximum(S[k, 3], 0.5)
    nx = (xx - cx) / r
    ny = (yy - cy) / r
    nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0.0, 1))
    d = nx * light3[0] + ny * light3[1] + nz * light3[2]
    tx = xx - np.floor(cx).astype(int) + 29 * k + seed
    ty = yy - np.floor(cy).astype(int) + 41 * k + seed
    d = d + tex_amp * (ntex(NTEX_LEAF, tx, ty) - 0.5) * 2
    out[m] = d
    return out, m


# =====================================================================
#  WORLD GEOMETRY (rest-screen coordinates)
# =====================================================================
HORIZON = 96
XL, XR = -24, 360            # all layers span this x-range
LW = XR - XL

DOME_X, DOME_BASE, DOME_RX = 150, 114, 33
WALL_H = 4
DOME_Y0 = DOME_BASE - WALL_H             # glass starts on top of the wall (110)
DOME_HG = 36                             # glass height
APEX_Y = DOME_Y0 - DOME_HG               # 74
TREE_X = DOME_X
CROWN = (151.0, 57.0)
CANOPY_C = (157.0, 30.0)
CANOPY_R = (92.0, 24.0)

SUN = (40.0, 15.0)                       # in SKY-layer rest coords
TEAR_C = (86.0, -2.0)           # bursts open just above the canopy's upper-left edge
SUN_OPEN = 2.34

# parallax factors (crane-up: nearer layers move more)
PAR = {'sky': 0.30, 'far': 0.42, 'v1': 0.55, 'v2': 0.72, 'hill': 1.0, 'v3': 1.28, 'fg': 1.75}

# pseudo-3D ground plane for the colour-wave arrival times
F_LEN = 220.0
EYE_H = 10.0
Z_TREE = 50.0
X_TREE = (TREE_X - 160) * Z_TREE / F_LEN
WAVE_T0 = 0.6


def plane_depth(y):
    return F_LEN * EYE_H / np.maximum(np.asarray(y, np.float64) - HORIZON, 0.35)


def dist_to_T(D):
    """World distance from the tree -> wave arrival (S8b local seconds).
    Far distances are compressed so the wave reaches the horizon ruins by
    ~3.1 s (cue: the wave spans 48.6-51.5)."""
    D = np.asarray(D, np.float64)
    D = D / (1.0 + D / 900.0)
    return WAVE_T0 + 0.58 * np.log1p(D / 10.0)


def ground_T(X, Y, extra=10.0, seed=0, wobble=0.06):
    z = plane_depth(Y)
    wx = (X - 160) * z / F_LEN
    D = np.sqrt((wx - X_TREE) ** 2 + (z - Z_TREE) ** 2) + extra
    T = dist_to_T(D)
    T = T + wobble * (noise.fbm2(X / 23.0, Y / 9.0, seed=seed + 77, octaves=3) - 0.5) * 2
    return T


def terr(xs, seed, base, amp, scale, octaves=4, gain=0.5):
    return base - amp * (noise.fbm1(np.asarray(xs, np.float64) / scale, seed, octaves, gain=gain) - 0.5) * 2


class Layer:
    """A parallax layer in two states + arrival-time map."""

    def __init__(self, name, y0, h, opaque=False):
        self.name = name
        self.p = PAR[name]
        self.x0, self.y0, self.w, self.h = XL, y0, LW, h
        self.dead = Paint(XL, y0, LW, h)
        self.alive = Paint(XL, y0, LW, h)
        self.T = np.full((h, LW), 99.0)
        self.opaque = opaque
        self.flag = np.zeros((h, LW), np.uint8)      # 1 = flower, 2 = water

    @property
    def X(self):
        return self.dead.X

    @property
    def Y(self):
        return self.dead.Y


# =====================================================================
#  SMALL PAINT PRIMITIVES
# =====================================================================
def rnd(i, j=0, seed=0):
    return noise.rand(i, j, seed)


def rndxy(X, Y, seed):
    return noise.rand(np.asarray(X).astype(np.int64), np.asarray(Y).astype(np.int64), seed)


def surface_rows(top, Y):
    return Y - np.asarray(top)[None, :]


def paint_tufts(Pn, top, seed, density, cols, maxh=3, T=None, Tbase=None, grow=0.035,
                lean=0, x_ok=None):
    """Grass blades sticking up from a per-column surface line `top`,
    grouped in tufts. Blade pixels grow up after the wave."""
    w = Pn.w
    tuft = noise.fbm1(np.arange(w) / 3.0, seed + 3, 2)
    for j in range(w):
        if x_ok is not None and not x_ok[j]:
            continue
        if rnd(j, 1, seed) > density * (0.4 + 1.2 * tuft[j]):
            continue
        hgt = 1 + int(rnd(j, 2, seed) * maxh * (0.5 + tuft[j]))
        hgt = min(hgt, maxh)
        y0 = int(round(top[j]))
        for k in range(1, hgt + 1):
            lx = lean if (lean and k == hgt and hgt >= 3) else 0
            i = y0 - k - Pn.y0
            jj = j + lx
            if 0 <= i < Pn.h and 0 <= jj < w:
                c = cols[min(len(cols) - 1, k - 1 + (len(cols) - hgt if hgt < len(cols) else 0))]
                Pn.i[i, jj] = IDX[c]
                Pn.a[i, jj] = True
                if T is not None:
                    T[i, jj] = Tbase[j] + k * grow


def flower_sprite(kind, size):
    petal, centre = kind
    if size <= 1:
        return [(0, 0, petal)]
    if size == 2:
        return [(0, 0, centre), (0, 1, petal), (-1, 0, petal)]
    return [(-1, 0, petal), (0, -1, petal), (0, 1, petal), (1, 0, petal), (0, 0, centre)]


FLOWER_KINDS = [
    ('bloom4', 'bloom5'), ('white0', 'gold3'), ('gold3', 'gold4'), ('violet3', 'violet4'),
    ('red3', 'red4'), ('bloom3', 'bloom5'), ('white1', 'gold4'), ('violet2', 'violet4'),
]


def ripples(mask, seed, Pn, density=0.06, minlen=3, maxlen=7):
    """Horizontal dash clusters (sand ripples / grass strokes)."""
    h, w = mask.shape
    out = np.zeros_like(mask)
    r = rndxy(Pn.X, Pn.Y, seed)
    starts = mask & (r < density)
    ys, xs = np.nonzero(starts)
    for n in range(len(ys)):
        ln = minlen + int(rnd(n, 4, seed) * (maxlen - minlen + 1))
        i, j = ys[n], xs[n]
        out[i, j:min(w, j + ln)] = True
    return out & mask


def cracks(Pn, seed, cell=7.0, sy=0.45, width=0.9):
    """Voronoi crack network (cracked mud), squashed for ground perspective."""
    X, Y = Pn.X, Pn.Y
    gx = X / cell
    gy = Y / (cell * sy)
    ix, iy = np.floor(gx), np.floor(gy)
    d1 = np.full(X.shape, 9.0)
    d2 = np.full(X.shape, 9.0)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            a = (ix + ox).astype(np.int64)
            b = (iy + oy).astype(np.int64)
            cx = ix + ox + 0.15 + 0.7 * rnd(a, b, seed)
            cy = iy + oy + 0.15 + 0.7 * rnd(a, b, seed + 1)
            d = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
            nd2 = np.where(d < d1, d1, np.minimum(d2, d))
            d1 = np.minimum(d1, d)
            d2 = nd2
    return (d2 - d1) < width / cell * 1.1


def patchy(Pn, seed, sx=40.0, sy=12.0, thr=0.5, soft=0.06):
    n = noise.fbm2(Pn.X / sx, Pn.Y / sy, seed=seed, octaves=3)
    return n > thr


# =====================================================================
#  SKY
# =====================================================================
def build_sky():
    L = Layer('sky', -40, 158, opaque=True)
    X, Y = L.X, L.Y
    D, A = L.dead, L.alive
    ones = np.ones((L.h, L.w), bool)
    # ------------------------------------------------ DEAD: dust overcast
    ty = np.clip((Y + 40) / (HORIZON + 40), 0, 1.2)
    tone = 0.15 + 4.9 * ty ** 1.35
    dx = (X - SUN[0]) * 0.62
    dy = Y - SUN[1]
    de = np.sqrt(dx * dx + dy * dy)
    near = np.exp(-(de / 34.0) ** 1.2)
    tone += 2.3 * np.exp(-(de / 36.0) ** 1.15)
    tone += 0.7 * (noise.fbm2(X / 150.0, Y / 28.0, seed=5, octaves=2) - 0.5)
    heavy = noise.fbm2(X / 70.0, Y / 13.0, seed=31, octaves=3)
    hv = np.clip((heavy - 0.47) * 4.0, 0, 1) * np.clip((40 - Y) / 45.0, 0, 1)
    tone -= 1.3 * hv
    cols = ['dust3', 'dust4', 'dust5', 'dust6', 'dust7', 'dust8', 'dust9', 'dust10']
    tone = np.clip(tone + 1.0, 0, len(cols) - 1.001)
    q = quant(tone, band=0.5)
    n1 = noise.fbm2(X / 90.0, Y / 5.5, seed=21, octaves=3)
    n2 = noise.fbm2(X / 36.0 + 7.0, Y / 3.0, seed=22, octaves=2)
    cn = n1 * 0.8 + n2 * 0.2
    fade = np.clip((92 - Y) / 34.0, 0, 1)
    streak = (cn - 0.5) * fade > 0.075
    streak = runs_min(streak, 4)
    top = streak & ~shift(streak, 0, 1)
    bot = streak & ~shift(streak, 0, -1)
    body = np.clip(np.floor(tone - 1.1 + 0.7 * near), 0, 7).astype(int)
    q = np.where(streak, body, q)
    lit = np.clip(np.floor(tone + 0.4 + 1.3 * near), 0, 7).astype(int)
    q = np.where(top & (near > 0.15), lit, q)
    q = np.where(bot, np.clip(body - 1, 0, 7), q)
    D.put_levels(ones, q, cols)
    d = np.sqrt((X - SUN[0]) ** 2 + (Y - SUN[1]) ** 2)
    L.dead_nosun = D.i.copy()                  # the overcast without the smudge
    D.put(bay(L.h, L.w) < np.clip((6.4 - d) / 1.8, 0, 1), 'dust10')
    D.put(d <= 3.2, 'white0')
    L.smudge = (bay(L.h, L.w) < np.clip((6.4 - d) / 1.8, 0, 1)) | (d <= 3.2)
    L.smudge_i = D.i.copy()
    L.dens = np.clip((cn - 0.5) * fade * 5.0, 0, 1) + hv

    # ------------------------------------------------ ALIVE: blue sky
    ty2 = np.clip((Y + 40) / (HORIZON + 2 + 40), 0, 1)
    v = 0.1 + 5.2 * ty2 ** 1.45
    ds = np.sqrt(((X - SUN[0]) * 0.8) ** 2 + (Y - SUN[1]) ** 2)
    v += 2.1 * np.exp(-(ds / 30.0) ** 1.25)
    scol = ['sky3', 'sky4', 'sky5', 'sky6', 'sky7', 'sky8', 'white0']
    qa = quant(np.clip(v, 0, len(scol) - 1.001), band=0.55)
    A.put_levels(ones, qa, scol)
    clouds = _cloud_spheres()
    sid, zb = spheres_zbuffer(L.h, L.w, L.x0, L.y0, clouds, jag=1.0, seed=3)
    sh, cm = spheres_shade(sid, zb, L.x0, L.y0, clouds, tex_amp=0.10, seed=3)
    kk = np.maximum(sid, 0)
    base_y = clouds[kk, 4]
    cm &= Y <= base_y
    lowf = np.clip((Y - 40) / 50.0, 0, 1)
    under = np.clip((Y - (base_y - 5)) / 5.0, 0, 1)
    lvc = 1.5 + 2.3 * sh - 1.2 * under - 0.6 * lowf
    # contact shadow between puffs
    zs = shift(zb, 2, 2, fill=-1e9) if False else np.pad(zb, ((2, 0), (2, 0)), constant_values=-1e9)[:-2, :-2]
    lvc -= 1.0 * ((zs - zb) > 2.0)
    ctones = ['sky5', 'sky6', 'sky7', 'sky8', 'white0', 'white1']
    cq = quant(np.clip(lvc, 0, len(ctones) - 1.001), band=0.35)
    A.put_levels(cm, cq, ctones)
    paint_sun(A, X, Y, SUN, warm=False)
    L.clouds = clouds
    # ------------------------------------------------ tear arrival map
    dt = np.sqrt((X - TEAR_C[0]) ** 2 + ((Y - TEAR_C[1]) * 1.25) ** 2)
    T = 2.12 + 1.05 * (dt / 260.0) ** 0.75 + 0.30 * np.clip(L.dens, 0, 1) * np.clip(dt / 20.0, 0, 1)
    T += 0.10 * (noise.fbm2(X / 13.0, Y / 6.0, seed=44, octaves=3) - 0.5) * 2
    dsun = np.sqrt((X - SUN[0]) ** 2 + (Y - SUN[1]) ** 2)
    T = np.minimum(T, SUN_OPEN + dsun / 40.0)          # the sun bursts through its own gap
    L.T = T
    return L


def paint_sun(Pn, X, Y, c, warm=False):
    """A soft, bright sun disc with a warm dithered halo (no hard ring)."""
    d = np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2)
    th = bay(Pn.h, Pn.w, 4, 0, 0)
    rings = ([('gold4', 15.0, 7.0, 0.55), ('core6', 10.5, 5.0, 0.8), ('core7', 7.0, 2.6, 1.0)] if warm else
             [('sky8', 16.0, 7.0, 0.6), ('core7', 10.0, 4.5, 0.75), ('white0', 6.8, 2.2, 1.0)])
    for name, R, wdt, amax in rings:
        a = np.clip((R - d) / wdt, 0, 1) * amax
        Pn.put(th < a, name)
    Pn.put(d <= 4.6, 'white0')
    Pn.put(d <= 3.6, 'white1')


# ---------------------------------------------------------------- the parting
PART_C = (128.0, -4.0)          # the break opens above the tree (sky-layer coords)
PART_T0 = 2.2


class CloudParting:
    """The dust overcast parts like torn curtains: it is cut into ragged
    horizontal bands (following the strata); from the break each band splits
    and its halves slide apart (integer row shifts - no resampling), their
    leading edges fraying thin, gold-rimmed, revealing the blue behind."""

    def __init__(self, L, seed=451):
        self.L = L
        h = L.h
        rows = []
        y = 0
        k = 0
        while y < h:
            bh = 2 + int(rnd(k, 1, seed) * 4)
            rows += [k] * bh
            y += bh
            k += 1
        self.band = np.array(rows[:h])
        nb = k
        kk = np.arange(nb)
        self.jag = (rnd(kk, 2, seed) - 0.5) * 16.0
        self.spd = 0.75 + 0.55 * rnd(kk, 3, seed)
        self.dly = rnd(kk, 4, seed) * 0.06
        yb = np.array([L.y0 + np.mean(np.nonzero(self.band == b)[0]) for b in range(nb)])
        self.yb = yb
        self.topen = PART_T0 + np.abs(yb - PART_C[1]) / 205.0 + self.dly
        self.split = PART_C[0] + self.jag + 0.30 * (yb - PART_C[1])

    def frame(self, t, win):
        """-> (idx, dead_shown mask, rim light level) for the sky window."""
        L = self.L
        i0, i1, j0, j1 = win
        alive = L.alive.i[i0:i1, j0:j1]
        if t < PART_T0 - 0.01:
            dead = L.smudge_i[i0:i1, j0:j1]
            return dead.copy(), np.ones(dead.shape, bool), np.zeros(dead.shape)
        b = self.band[i0:i1]
        age = np.clip(t - self.topen[b], 0, None)
        u = np.clip(age / 0.95, 0, 1)
        Dm = 175.0 * self.spd[b] * (1 - (1 - u) ** 3) + 22.0 * self.spd[b] * np.clip(age - 0.95, 0, None)
        Dm = np.floor(Dm)[:, None]
        sp = np.round(self.split[b])[:, None]
        X = (L.x0 + j0 + np.arange(j1 - j0))[None, :].astype(np.float64)
        left = X < sp - Dm
        right = X >= sp + Dm
        src = np.where(left, X + Dm, np.where(right, X - Dm, -9999)) - L.x0
        valid = (left | right) & (src >= 0) & (src < L.w)
        srci = np.clip(src, 0, L.w - 1).astype(np.int64)
        rows = np.arange(i0, i1)[:, None]
        moved = L.dead_nosun[rows, srci]
        # erosion from the leading edge + a global thinning
        dist = np.where(left, (sp - Dm) - X, np.where(right, X - (sp + Dm), 0.0))
        opened = (age > 0)[:, None]
        ew = 3.0 + 24.0 * np.clip(age / 1.0, 0, 1)[:, None]
        a_edge = np.where(opened, np.clip(dist / ew, 0, 1), 1.0)
        a_glob = 1.0 - ease.smoothstep((t - 3.15) / 0.9)
        th = bay(i1 - i0, j1 - j0, 4, j0, i0)
        show = valid & (th < a_edge * a_glob)
        idx = np.where(show, moved, alive)
        # the pale sun smudge stays where the sun is, while cloud covers it
        sm = L.smudge[i0:i1, j0:j1] & show
        idx[sm] = L.smudge_i[i0:i1, j0:j1][sm]
        # light: gold on the fraying edges, warm backlight behind them
        ramp_in = np.clip((t - PART_T0) / 0.35, 0, 1)
        rim = np.where(show & opened & (dist < 3.0), 2.7, 0.0)
        rim = np.maximum(rim, np.where(show & opened & (dist >= 3.0) & (dist < 12.0),
                                       1.4 * (1 - (dist - 3.0) / 9.0), 0.0))
        rim += np.where(show, 0.6 * ramp_in, 0.0)
        return idx, show, rim * ramp_in


def _cloud_spheres():
    """Cumulus for the alive sky: (cx, cy, cz, r, base_y)."""
    out = []

    def cloud(cx, by, wdt, hgt, seed, n=14):
        for k in range(n):
            u = rnd(k, 1, seed)
            x = cx + (u - 0.5) * wdt
            prof = 1 - ((x - cx) / (wdt * 0.5 + 1e-6)) ** 2
            r = (0.35 + 0.65 * rnd(k, 2, seed)) * hgt * (0.45 + 0.55 * max(prof, 0))
            y = by - r * 0.55 - rnd(k, 3, seed) * hgt * 0.35 * max(prof, 0)
            out.append((x, y, rnd(k, 4, seed) * 4 + (1 - abs(u - 0.5) * 2) * 3, max(r, 2.0), by))

    for i, (cx, wd, hg) in enumerate([(-14, 40, 11), (30, 34, 9), (206, 30, 10), (246, 46, 14),
                                       (300, 44, 12), (346, 30, 10), (104, 26, 7)]):
        cloud(cx, 97, wd, hg, 100 + i, n=10)
    cloud(296, 50, 70, 22, 7, n=18)
    cloud(334, 30, 40, 12, 8, n=10)
    cloud(-6, 44, 30, 9, 9, n=8)
    cloud(226, 6, 34, 9, 10, n=8)
    return np.array(out, np.float64)


# =====================================================================
#  FAR: ruined city skyline + snapped bridge on the horizon
# =====================================================================
def _buildings():
    spec = [(196, 7, 88, 'flat'), (203, 9, 83, 'broken'), (212, 6, 86, 'flat'),
            (218, 11, 77, 'frame'), (229, 8, 84, 'broken'), (237, 5, 88, 'flat'),
            (242, 9, 72, 'spire'), (251, 12, 80, 'broken'), (263, 7, 69, 'lean'),
            (270, 10, 82, 'flat'), (280, 13, 64, 'frame'), (293, 8, 78, 'broken'),
            (301, 11, 71, 'broken'), (312, 6, 84, 'flat'), (318, 14, 75, 'frame'),
            (332, 9, 81, 'broken'), (341, 12, 70, 'broken'), (353, 8, 83, 'flat'),
            (-22, 9, 82, 'broken'), (-13, 7, 87, 'flat'), (98, 6, 89, 'broken'), (104, 8, 86, 'flat')]
    return spec


def build_far():
    L = Layer('far', 40, 80)
    X, Y = L.X, L.Y
    D, A = L.dead, L.alive
    h, w = L.h, L.w
    xs = X[0]
    mesa = terr(xs, 61, 86, 5, 60, 3)
    mesa = np.minimum(mesa, 91 - 7 * np.clip((xs - 150) / 90.0, 0, 1) * (noise.fbm1(xs / 25.0, 62, 2) > 0.45))
    mm = Y >= np.round(mesa)[None, :]
    D.put(mm, 'dust7')
    D.put(edge(mm, 0, -1) & mm, 'dust8')
    # distant ridges in morning haze: sky-lit crest, blue-green body, dark
    # bumps of far woodland along the crest, paler toward the base
    rdep = Y - np.round(mesa)[None, :]
    A.put(mm, 'paint5')
    A.put(mm & (rdep > 4), 'paint6')
    wood = mm & (rdep >= 1) & (rdep <= 3) & (noise.fbm2(X / 3.5, Y / 2.0, seed=64, octaves=2) > 0.5)
    A.put(wood, 'paint4')
    A.put(wood & edge(wood, 0, -1), 'paint5')
    A.put(edge(mm, 0, -1) & mm, 'sky7')
    gnd = Y >= HORIZON - 1
    D.put(gnd, 'dust7')
    A.put(gnd, 'meadow5')
    A.put(gnd & (rndxy(X * 5, Y, 65) < 0.18), 'paint4')
    solid = np.zeros((h, w), bool)
    lit = np.zeros((h, w), bool)
    dark = np.zeros((h, w), bool)
    win = np.zeros((h, w), bool)
    girders = np.zeros((h, w), bool)
    tops = []
    Xi, Yi = X.astype(int), Y.astype(int)
    for n, (bx, bw, ty, st) in enumerate(_buildings()):
        base = HORIZON + 1
        m = (X >= bx) & (X < bx + bw) & (Y >= ty) & (Y <= base)
        if st == 'broken':
            sl = (rnd(n, 1, 71) - 0.5) * 1.4
            jag = (rnd(xs.astype(int), n, 72) * 3).astype(int)
            cut = ty + sl * (xs - bx) + jag * (rnd(n, 2, 71) > 0.3)
            m &= Y >= cut[None, :]
        elif st == 'spire':
            m |= (X == bx + bw // 2) & (Y >= ty - 9) & (Y < ty)
            m |= (np.abs(X - (bx + bw // 2)) <= 1) & (Y >= ty - 3) & (Y < ty)
        elif st == 'lean':
            xx = X - (base - Y) * 0.22
            m = (xx >= bx) & (xx < bx + bw) & (Y >= ty) & (Y <= base)
            m &= Y >= ty + (X - bx) * 0.5 - 2
        elif st == 'frame':
            fr = (Y < ty + 7) & (Y >= ty)
            core = m & ~fr
            gx = ((X - bx) % 3 == 0) | ((Y - ty) % 3 == 0)
            girders |= m & fr & gx & (rndxy(X, Y, 73 + n) > 0.25)
            m = core
        solid |= m
        lit |= (m & ~shift(m, 1, 0)) | (m & ~shift(m, 2, 0) & (bw > 9))
        dark |= m & ~shift(m, -1, 0)
        wm = m & ((X - bx) % 3 == 1) & ((Y - ty) % 4 == 2) & (X > bx) & (X < bx + bw - 1)
        wm &= rndxy(X * 7, Y, 74 + n) > 0.35
        win |= wm
        tops.append((bx + bw * 0.5, ty, bw))
    bridge = np.zeros((h, w), bool)
    for tx in (14, 70):
        bridge |= (X >= tx) & (X <= tx + 2) & (Y >= 71) & (Y <= HORIZON)
        bridge |= (X >= tx - 1) & (X <= tx + 3) & ((Y == 74) | (Y == 80))
    deck_y = 88
    deckm = (Y >= deck_y) & (Y <= deck_y + 1) & (X >= -24) & (X <= 100) & ~((X > 36) & (X < 47))
    deckm |= (Y == deck_y + 2) & (X >= 33) & (X <= 36)
    deckm |= (Y == deck_y + 3) & (X >= 35) & (X <= 36)
    deckm |= (Y == deck_y + 2) & (X >= 47) & (X <= 49)
    bridge |= deckm
    cable = np.zeros((h, w), bool)
    for (x0, x1) in ((-24, 15), (17, 71), (73, 100)):
        xs_ = np.arange(x0, x1 + 1)
        s = (xs_ - x0) / max(1, x1 - x0)
        if x1 - x0 > 45:
            ys_ = 72 + (deck_y - 73) * (4 * s * (1 - s))
        elif x0 > 0:
            ys_ = 72 + (deck_y - 73) * s ** 1.5
        else:
            ys_ = 72 + (deck_y - 73) * (1 - s) ** 1.5
        for xx, yy in zip(xs_, ys_):
            if 36 < xx < 47:
                continue
            j, i = int(xx - L.x0), int(round(yy) - L.y0)
            if 0 <= i < h and 0 <= j < w:
                cable[i, j] = True
    hang = np.zeros((h, w), bool)
    for xx in range(-20, 100, 4):
        if 36 < xx < 47:
            continue
        s = (xx - 17) / 54.0
        if 0 <= s <= 1:
            yy = 72 + (deck_y - 73) * 4 * s * (1 - s)
            hang |= (X == xx) & (Y > yy) & (Y < deck_y)
    hang |= (X == 36) & (Y >= 84) & (Y <= 91)
    hang |= (X == 47) & (Y >= 83) & (Y <= 88)
    D.put(solid, 'ash6')
    D.put(dark, 'ash5')
    D.put(lit & solid, 'ash7')
    D.put(win & solid, 'ash5')
    D.put(girders, 'ash5')
    D.put(bridge, 'ash5')
    D.put(cable | hang, 'ash6')
    # ALIVE: soft atmospheric blue-grey, gentle greenery
    A.put(solid, 'ash6')
    A.put(dark, 'ash5')
    A.put(lit & solid, 'ash7')
    A.put(win & solid, 'ash5')
    A.put(girders, 'ash5')
    A.put(bridge, 'ash5')
    A.put(cable | hang, 'ash6')
    Tb = ground_T(X, np.full_like(Y, HORIZON + 0.6), extra=12, seed=3, wobble=0.1)
    Tb = Tb - 0.12 + 0.40 * np.clip(np.abs(X - TREE_X) / 210.0, 0, 1)
    L.T = Tb.copy()
    vine = np.zeros((h, w), bool)
    vn = noise.fbm1(xs / 3.0, 81, 2)
    for j in range(w):
        col = solid[:, j] | girders[:, j] | bridge[:, j]
        if not col.any():
            continue
        top_i = int(np.argmax(col))
        if vn[j] > 0.52:
            ln = int(3 + (vn[j] - 0.52) * 40)
            vine[top_i:top_i + ln, j] = col[top_i:top_i + ln]
        if vn[j] < 0.40:
            hgt = int(4 + (0.40 - vn[j]) * 45)
            bot = HORIZON - L.y0
            vine[max(0, bot - hgt):bot + 1, j] |= col[max(0, bot - hgt):bot + 1]
    A.put(vine, 'paint5')
    A.put(vine & (rndxy(X, Y, 82) < 0.45), 'paint4')
    L.T = np.where(vine, Tb + np.clip((HORIZON - Y) * 0.022, 0, 0.6) + 0.08, L.T)
    for n, (tx, ty, bw) in enumerate(tops):
        if rnd(n, 5, 83) < 0.55:
            S = []
            for k in range(3 + int(bw / 4)):
                S.append((tx + (rnd(k, 1, 84 + n) - 0.5) * bw, ty - 1 - rnd(k, 2, 84 + n) * 2,
                          rnd(k, 3, 84 + n), 1.2 + rnd(k, 4, 84 + n) * 1.8))
            S = np.array(S)
            sid, zb = spheres_zbuffer(h, w, L.x0, L.y0, S, jag=0.6, seed=n)
            sh, cm = spheres_shade(sid, zb, L.x0, L.y0, S, tex_amp=0.1)
            A.put(cm, 'paint4')
            A.put(cm & (sh > 0.35), 'paint5')
            L.T = np.where(cm, Tb + 0.35 + (ty - Y) * 0.01, L.T)
    L.T = np.where(A.a | D.a, L.T, 99.0)
    return L


# =====================================================================
#  V1: far valley hills (hazy)
# =====================================================================
V1_TREES = [22, 58, 236, 268, 310, 338]


def build_v1():
    L = Layer('v1', 78, 56)
    X, Y = L.X, L.Y
    D, A = L.dead, L.alive
    xs = X[0]
    top = 98.5 - 9 * (np.abs(xs - 150) / 210.0) ** 1.4 - 2.2 * (noise.fbm1(xs / 34.0, 91, 3) - 0.5) * 2
    m = Y >= np.round(top)[None, :]
    rim = m & ~shift(m, 0, 1)
    dep = surface_rows(top, Y)
    D.put(m, 'dust7')
    D.put(m & (dep > 4), 'dust6')
    D.put(ripples(m & (dep > 1.5), 93, D, 0.035, 3, 8), 'dust6')
    D.put(rim, 'dust8')
    trees_d = np.zeros_like(m)
    crowns = np.zeros_like(m)
    for n, tx in enumerate(V1_TREES):
        j = int(tx - L.x0)
        ty = int(round(top[j]))
        hh = 3 + int(rnd(n, 1, 94) * 3)
        trees_d |= (X == tx) & (Y < ty) & (Y >= ty - hh)
        trees_d |= (X == tx - 1) & (Y == ty - hh + 1)
        trees_d |= (X == tx + 1) & (Y == ty - hh + 2)
        crowns |= (X - tx) ** 2 + ((Y - (ty - hh)) * 1.3) ** 2 <= 4.5
    D.put(trees_d, 'dust5')
    # far fields: bands of meadow following the land, hedgerows, copses
    fld = noise.fbm2(X / 26.0, (Y - top[None, :]) / 3.0, seed=96, octaves=2)
    A.put(m, 'meadow5')
    A.put(m & (fld > 0.52), 'meadow4')
    A.put(m & (dep > 7), 'meadow4')
    hedge = m & (dep > 1.5) & (np.abs(((Y - top[None, :]) * 1.0 + X * 0.08) % 4.2 - 2.1) < 0.45)
    hedge &= noise.fbm2(X / 9.0, Y / 3.0, seed=97, octaves=2) > 0.47
    A.put(hedge, 'paint4')
    A.put(hedge & (rndxy(X, Y, 98) < 0.5), 'leaf3')
    A.put(rim, 'meadow6')
    A.put(rim & (rndxy(X, Y, 99) < 0.3), 'paint6')
    A.put(trees_d, 'paint3')
    A.put(crowns, 'paint4')
    A.put(crowns & (X < np.array(0)) , 'paint5')
    L.T = np.where(m | trees_d | crowns, ground_T(X, np.maximum(Y, top[None, :]) + 0.2, extra=10, seed=5), 99.0)
    L.T = np.where(crowns | trees_d, L.T + 0.2, L.T)
    L.top = top
    return L


# =====================================================================
#  V2: mid valley — dunes, dead trees, toppled turbine, pylons
# =====================================================================
def _skeleton(Pn, x, y, height, seed, spread=1.0, twig=True):
    """Dead tree skeleton (1px branches, 2px trunk). Returns (mask, tips)."""
    tips = []
    m = np.zeros((Pn.h, Pn.w), bool)
    lean = (rnd(seed, 1, 101) - 0.5) * 0.3
    x_top = x + lean * height
    m |= Pn.mask_line(x, y, x_top, y - height)
    if height > 9:
        m |= Pn.mask_line(x + 1, y, x_top + 1, y - height * 0.55)
    nb = 3 + int(rnd(seed, 2, 101) * 3)
    for b in range(nb):
        u = 0.35 + 0.6 * b / max(nb - 1, 1)
        bx = x + lean * height * u
        by = y - height * u
        side = -1 if b % 2 == 0 else 1
        ang = side * (0.5 + 0.6 * rnd(b, 3, seed)) * spread
        ln = height * (0.45 - 0.25 * u) * (0.7 + 0.6 * rnd(b, 4, seed))
        ex = bx + math.sin(ang) * ln
        ey = by - math.cos(ang) * ln * 0.8
        m |= Pn.mask_line(bx, by, ex, ey)
        tips.append((ex, ey, ln))
        if twig and ln > 4:
            ang2 = ang + side * 0.6
            ex2 = ex + math.sin(ang2) * ln * 0.4
            ey2 = ey - math.cos(ang2) * ln * 0.35
            m |= Pn.mask_line(ex, ey, ex2, ey2)
            tips.append((ex2, ey2, ln * 0.5))
            mx, my = (bx + ex) / 2, (by + ey) / 2
            m |= Pn.mask_line(mx, my, mx - side * ln * 0.3, my - ln * 0.35)
            tips.append((mx - side * ln * 0.3, my - ln * 0.35, ln * 0.4))
    tips.append((x_top, y - height, height * 0.3))
    return m, tips


def leaf_tree(L, tx, ty, height, seed, tips, Tbase, scale=1.0, blossom=0.0,
              tones=None, bloom_tones=None):
    """Foliage crown on skeleton tips (ALIVE canvas); pops outward."""
    A = L.alive
    S = []
    for k, (ex, ey, ln) in enumerate(tips):
        r = (1.6 + 0.35 * ln) * scale
        S.append((ex, ey - r * 0.2, rnd(k, 1, seed) * 3 + (ty - ey) * 0.1, r))
        if ln > 3:
            S.append((ex + (rnd(k, 2, seed) - 0.5) * r, ey - r * 0.6, rnd(k, 3, seed) * 3, r * 0.75))
    S = np.array(S)
    sid, zb = spheres_zbuffer(L.h, L.w, L.x0, L.y0, S, jag=0.9, seed=seed)
    sh, cm = spheres_shade(sid, zb, L.x0, L.y0, S, tex_amp=0.2, seed=seed)
    zs = np.pad(zb, ((1, 0), (1, 0)), constant_values=-1e9)[:-1, :-1]
    occ = (zs - zb) > 1.5
    tones = tones or ['leaf1', 'leaf2', 'leaf3', 'leaf4', 'leaf5', 'leaf6']
    lv = np.clip((sh + 0.55) / 0.3 - 1.0 * occ, 0, len(tones) - 1.001)
    q = quant(lv, band=0.3)
    A.put_levels(cm, q, tones)
    if blossom > 0:
        bl = cm & (rndxy(L.X, L.Y, seed + 9) < blossom) & (sh > -0.1)
        bl &= ~shift(bl, 1, 0) & ~shift(bl, 0, 1)
        bt = bloom_tones or ('bloom3', 'bloom5')
        A.put(bl, bt[0])
        A.put(bl & (sh > 0.45), bt[1])
    k = np.maximum(sid, 0)
    dd = np.sqrt((L.X - S[k, 0]) ** 2 + (L.Y - S[k, 1]) ** 2) / np.maximum(S[k, 3], 1)
    L.T = np.where(cm, Tbase + 0.12 + 0.22 * dd + 0.03 * (k % 5), L.T)
    return cm


def build_v2():
    L = Layer('v2', 88, 72)
    X, Y = L.X, L.Y
    D, A = L.dead, L.alive
    xs = X[0]
    top = 106.0 - 6 * (np.abs(xs - 160) / 190.0) ** 1.3 - 3.2 * (noise.fbm1(xs / 26.0, 111, 3) - 0.5) * 2
    m = Y >= np.round(top)[None, :]
    rim = m & ~shift(m, 0, 1)
    dep = surface_rows(top, Y)
    T = ground_T(X, np.maximum(Y, top[None, :] + 0.5), extra=10, seed=7)
    L.T = np.where(m, T, 99.0)
    D.put(m, 'dust6')
    D.put(m & (dep > 3), 'dust5')
    D.put(m & (dep > 10), 'dust4')
    D.put(ripples(m & (dep > 1.5), 113, D, 0.03, 3, 7), 'dust5')
    D.put(rim, 'dust7')
    A.put(m, 'meadow5')
    A.put(m & (dep > 2), 'meadow4')
    A.put(m & (dep > 9), 'meadow3')
    A.put(ripples(m & (dep > 1.5), 115, A, 0.03, 2, 6), 'meadow5')
    A.put(rim, 'meadow6')
    flower_drifts(A, m & (dep > 1), DRIFTS_V2, 117, T=L.T, flag=L.flag)
    Tcol = T[np.clip(np.round(top - L.y0).astype(int), 0, L.h - 1), np.arange(L.w)]
    paint_tufts(A, top, 118, 0.30, ['meadow4', 'meadow5'], maxh=2, T=L.T, Tbase=Tcol)
    for n, (tx, hh) in enumerate([(12, 11), (44, 8), (92, 9), (228, 10), (318, 12), (346, 8)]):
        j = int(tx - L.x0)
        ty = top[j] + 1
        tm, tips = _skeleton(D, tx, ty, hh, 120 + n)
        D.put(tm, 'dust3')
        A.put(tm, 'bark2')
        Tt = T[int(ty - L.y0), j]
        L.T = np.where(tm, Tt, L.T)
        leaf_tree(L, tx, ty, hh, 130 + n, tips, Tt, scale=0.8,
                  blossom=0.10 if n % 2 == 0 else 0.0,
                  tones=['leaf1', 'leaf2', 'leaf3', 'leaf4', 'leaf5'])
    # toppled wind turbine (right) -> overgrown
    bx = 262
    by = int(top[int(bx - L.x0)]) + 1
    tur = D.mask_line(bx, by, bx + 30, by - 9) | D.mask_line(bx + 1, by, bx + 31, by - 9)
    hub = (bx + 31, by - 10)
    tur |= D.mask_disc(hub[0], hub[1], 1.2)
    tur |= D.mask_line(hub[0], hub[1], hub[0] - 3, hub[1] - 14)
    tur |= D.mask_line(hub[0], hub[1], hub[0] + 11, hub[1] + 4)
    tur &= (Y <= np.round(top)[None, :] + 1) | (X > 280)
    D.put(tur, 'rust2')
    D.put(tur & edge(tur, 0, -1), 'rust3')
    A.put(tur, 'rust3')
    A.put(tur & edge(tur, 0, -1), 'rust4')
    vine = tur & (rndxy(X, Y, 121) < 0.45)
    A.put(vine, 'leaf3')
    A.put(vine & (rndxy(X, Y, 122) < 0.3), 'bloom4')
    Tt = T[int(by - L.y0), int(bx + 12 - L.x0)]
    L.T = np.where(tur, Tt, L.T)
    L.T = np.where(vine, Tt + 0.25 + (by - Y) * 0.02, L.T)
    # pylons
    pyl = np.zeros_like(m)
    ptops = []
    for n, px_ in enumerate([4, 60, 116]):
        j = int(px_ - L.x0)
        pb = int(round(top[j])) + 1
        ph = 13 - n
        pyl |= D.mask_line(px_ - 2, pb, px_, pb - ph) | D.mask_line(px_ + 2, pb, px_, pb - ph)
        pyl |= (Y == pb - ph + 2) & (np.abs(X - px_) <= 2)
        pyl |= (Y == pb - ph // 2) & (np.abs(X - px_) <= 1)
        ptops.append((px_, pb - ph + 2))
    cab = np.zeros_like(m)
    for (x0, y0), (x1, y1) in zip(ptops[:-1], ptops[1:]):
        for xx in range(x0 - 2, x1 + 3):
            s = (xx - x0 + 2) / (x1 - x0 + 4)
            yy = y0 + (y1 - y0) * s + 5 * 4 * s * (1 - s)
            j, i = int(xx - L.x0), int(round(yy) - L.y0)
            if 0 <= i < L.h and 0 <= j < L.w:
                cab[i, j] = True
    D.put(pyl, 'ash3')
    D.put(cab, 'ash4')
    A.put(pyl, 'steel3')
    A.put(cab, 'steel4')
    vine2 = pyl & (Y > np.round(top)[None, :] - 7)
    vine2 &= rndxy(X, Y, 123) < 0.6
    A.put(vine2, 'leaf3')
    L.T = np.where(pyl | cab, ground_T(X, np.full_like(Y, 104.0), extra=10, seed=7), L.T)
    L.T = np.where(vine2, L.T + (np.round(top)[None, :] - Y) * 0.04, L.T)
    L.top = top
    return L


# =====================================================================
#  HILL (the greenhouse hill): ground + dome baked in; roots/cracks dynamic
# =====================================================================
def hill_top(xs):
    xs = np.asarray(xs, np.float64)
    ax = np.maximum(np.abs(xs - DOME_X) - (DOME_RX + 4), 0)
    return DOME_BASE + 0.5 + ax ** 2 / 105.0 + 0.8 * (noise.fbm1(xs / 11.0, 141, 2) - 0.5) * 2 * np.clip(ax / 6, 0, 1)


def hill_T(X, Y):
    """Wave arrival over the hill surface (from the dome base outward)."""
    dx = (np.asarray(X, np.float64) - DOME_X) * (Z_TREE / F_LEN)
    dy = np.maximum(np.asarray(Y, np.float64) - DOME_BASE, 0) * 0.40
    D = np.sqrt(dx * dx + dy * dy)
    D = np.maximum(D - 6.5, 0)
    T = dist_to_T(D) + 0.04
    T = T + 0.05 * (noise.fbm2(np.asarray(X) / 9.0, np.asarray(Y) / 5.0, seed=142, octaves=2) - 0.5) * 2
    return T


T_DOME = float(dist_to_T(0.0)) + 0.06


def root_paths(seed=0, n=7, s9=False):
    """Roots that crack out of the hill around the dome base."""
    paths = []
    starts = [-31, -22, -12, 4, 14, 24, 32] if not s9 else [-33, -27, -20, -12, -4, 5, 13, 21, 28, 34]
    for k, sx in enumerate(starts):
        x = DOME_X + sx
        y = DOME_BASE + 0.5
        side = -1 if sx < 0 else 1
        ang = math.atan2(0.55, side * (0.4 + abs(sx) / 40.0))    # outward & downward
        pts = [(x, y)]
        ln = 20 + rnd(k, 1, seed + 401) * 30 + (12 if s9 else 0)
        s = 0.0
        while s < ln:
            ang += (noise.vnoise1(np.array([s / 7.0 + k * 10]), seed + 402)[0] - 0.5) * 0.5
            x += math.cos(ang)
            y += math.sin(ang) * 0.55
            y = max(y, float(hill_top([x])[0]) + 0.5)
            pts.append((x, y))
            s += 1
        paths.append((pts, 2.6 + rnd(k, 2, seed + 401) * 1.4 + (0.8 if s9 else 0), k))
    return paths


class HillFX:
    """Glowing cracks racing out of the dome base, then roots surfacing."""

    def __init__(self, s9=False):
        crack_px, crack_t = [], []
        root_px, root_t, root_c = [], [], []
        for pts, wd, k in root_paths(s9=s9):
            P_ = np.array(pts)
            n = len(P_)
            # crack continues beyond the root and branches once
            ext = [P_[-1] + (P_[-1] - P_[-2]) * j for j in range(1, int(n * 0.4))]
            cpts = list(P_) + ext
            br0 = int(n * 0.55)
            bdir = np.array([P_[br0, 0] - DOME_X, 3.0])
            bdir /= np.linalg.norm(bdir)
            bpts = [P_[br0] + np.array([bdir[0] * j + (-1 if j % 5 == 0 else 0), bdir[1] * j * 0.8]) for j in range(1, 12)]
            for j, (x, y) in enumerate(cpts):
                crack_px.append((int(round(x)), int(round(y))))
                crack_t.append(0.02 + j * 0.0055)
            for j, (x, y) in enumerate(bpts):
                crack_px.append((int(round(x)), int(round(y))))
                crack_t.append(0.02 + (br0 + j) * 0.0055)
            for j in range(n):
                s = j / max(n - 1, 1)
                r = max(0.5, wd * (1 - s) ** 0.8 / 2)
                x, y = P_[j]
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        d = math.hypot(dx, dy * 1.4)
                        if d <= r + 0.2:
                            col = 2
                            if dy < 0 and d > r - 1.0:
                                col = 4
                            elif dy > 0 and d > r - 1.0:
                                col = 1
                            elif dx < 0:
                                col = 3
                            root_px.append((int(round(x + dx)), int(round(y + dy))))
                            root_t.append(0.16 + j * 0.011)
                            root_c.append(col)
        self.cxy = np.array(crack_px)
        self.ct = np.array(crack_t)
        rp = np.array(root_px)
        rt = np.array(root_t)
        rc = np.array(root_c)
        # dedupe: keep earliest time / keep the lighter colour on overlap
        key = rp[:, 0] * 1000 + rp[:, 1]
        order = np.lexsort((-rc, rt, key))
        key_s = key[order]
        first = np.ones(len(order), bool)
        first[1:] = key_s[1:] != key_s[:-1]
        sel = order[first]
        self.rxy, self.rt, self.rc = rp[sel], rt[sel], rc[sel]
        self.rcol = cis(['bark1', 'bark1', 'bark2', 'bark3', 'bark4'])[self.rc]

    def draw(self, idx, alpha, ox, oy, t, T_alive=None, mossy=False, cracks=True):
        h, w = idx.shape
        # cracks
        x = self.cxy[:, 0] - ox
        y = self.cxy[:, 1] - oy
        ok = (x >= 0) & (x < w) & (y >= 0) & (y < h) & (t >= self.ct) & cracks
        age = t - self.ct
        if ok.any():
            xs, ys, ag = x[ok], y[ok], age[ok]
            col = np.full(len(xs), IDX['dust2'], np.uint8)
            col[ag < 0.9] = IDX['core3']
            col[ag < 0.55] = IDX['core5']
            col[ag < 0.30] = IDX['gold4']
            col[ag < 0.08] = IDX['white1']
            if T_alive is not None:
                al = T_alive[ys, xs] <= t
                col[al & (ag > 0.9)] = IDX['meadow1']
            idx[ys, xs] = col
            alpha[ys, xs] = True
        # roots
        x = self.rxy[:, 0] - ox
        y = self.rxy[:, 1] - oy
        ok = (x >= 0) & (x < w) & (y >= 0) & (y < h) & (t >= self.rt)
        if ok.any():
            c = self.rcol[ok].copy()
            if mossy:
                top = self.rc[ok] == 4
                c[top & (rnd(x[ok], y[ok], 777) < 0.6)] = IDX['leaf4']
            idx[y[ok], x[ok]] = c
            alpha[y[ok], x[ok]] = True


def build_hill():
    L = Layer('hill', -76, 276)
    X, Y = L.X, L.Y
    D, A = L.dead, L.alive
    xs = X[0]
    top = hill_top(xs)
    m = Y >= np.round(top)[None, :]
    dep = surface_rows(top, Y)
    rim = m & ~shift(m, 0, 1)
    slope = np.gradient(top)
    sl = np.broadcast_to(slope[None, :], X.shape)
    T = hill_T(X, Y)
    L.T = np.where(m, T, 99.0)
    lightv = -sl * 1.5 - np.clip(dep, 0, 30) / 30.0 * 0.7 + 0.35
    lightv += 0.16 * (NTEX_BIG[(Y.astype(int)) % 256, (X.astype(int)) % 256] - 0.5)
    # ---------------- DEAD: dusty earth (lighter than the near valley)
    dcols = ['dust4', 'dust5', 'dust6', 'dust7']
    dq = quant(np.clip(1.5 + lightv * 1.5, 0, 3.99), band=0.25)
    D.put_levels(m, dq, dcols)
    ck = cracks(D, 143, cell=6.0, sy=0.5) & m & (dep > 3) & patchy(D, 144, 30, 9, 0.52)
    D.put(ck, 'dust4')
    D.put(ripples(m & (dep > 1) & ~ck, 145, D, 0.02, 2, 5), 'dust5')
    D.put(rim, 'dust7')
    stones = np.zeros_like(m)
    for n in range(22):
        sx = DOME_X + (rnd(n, 1, 144) - 0.5) * 160
        sy_ = hill_top([sx])[0] + 2 + rnd(n, 2, 144) * 22
        if abs(sx - DOME_X) < DOME_RX + 5 and sy_ < DOME_BASE + 6:
            continue
        rr = 0.8 + rnd(n, 3, 144) * 1.5
        s = D.mask_ellipse(sx, sy_, rr * 1.3, rr * 0.8) & m
        stones |= s
        D.put(s, 'ash5')
        D.put(s & edge(s, 0, -1), 'ash7')
        D.put(s & edge(s, 0, 1), 'dust3')
        A.put(s, 'ash5')
        A.put(s & edge(s, 0, -1), 'meadow5')
        A.put(s & edge(s, 0, 1), 'meadow1')
    # rubble by the dome base: glass bits & iron scraps
    rub = m & (np.abs(X - DOME_X) < DOME_RX + 8) & (dep < 5) & (rndxy(X, Y, 146) < 0.10)
    D.put(rub, 'glass3')
    D.put(rub & (rndxy(X, Y, 147) < 0.35), 'rust2')
    # ---------------- ALIVE: grass, moss, flower drifts
    acols = ['meadow2', 'meadow3', 'meadow4', 'meadow5']
    aq = quant(np.clip(1.6 + lightv * 1.5, 0, 3.99), band=0.22)
    A.put_levels(m & ~stones, aq, acols)
    grass_tufts(A, m & (dep > 1) & ~stones, aq, 148, 0.035)
    A.put(rim & ~stones, 'meadow5')
    flower_drifts(A, m & (dep > 2) & ~stones, DRIFTS_HILL, 150, T=L.T, flag=L.flag)
    Tcol = T[np.clip(np.round(top - L.y0).astype(int), 0, L.h - 1), np.arange(L.w)]
    paint_tufts(A, top, 151, 0.55, ['meadow4', 'meadow5', 'meadow6'], maxh=3, T=L.T, Tbase=Tcol,
                x_ok=np.abs(xs - DOME_X) > DOME_RX + 2)
    L.top = top
    return L


# =====================================================================
#  V3: near valley floor — riverbed/stream, big tree, car husk
# =====================================================================
STREAM_PTS = [(170, 134.5), (163, 139), (168, 145), (186, 151), (203, 158), (209, 166),
              (224, 174), (246, 183), (268, 196), (290, 213), (302, 232), (306, 256)]


def v3_top(xs):
    xs = np.asarray(xs, np.float64)
    dip = 11.0 * np.exp(-((xs - 158) / 62.0) ** 2)
    return 124.0 + dip + 1.8 * (noise.fbm1(xs / 23.0, 161, 3) - 0.5) * 2 + 2.0 * np.clip((xs - 210) / 110, 0, 1)


def _polyline_dist(X, Y, pts, step=0.5):
    P_ = np.array(pts, np.float64)
    seg_len = np.sqrt(((P_[1:] - P_[:-1]) ** 2).sum(1))
    samples, arcs = [], []
    acc = 0.0
    for k in range(len(P_) - 1):
        n = max(2, int(seg_len[k] / step))
        for s in np.linspace(0, 1, n, endpoint=False):
            samples.append(P_[k] + (P_[k + 1] - P_[k]) * s)
            arcs.append(acc + seg_len[k] * s)
        acc += seg_len[k]
    samples.append(P_[-1])
    arcs.append(acc)
    S = np.array(samples)
    A_ = np.array(arcs)
    best = np.full(X.shape, 1e9)
    arc = np.zeros(X.shape)
    for k in range(len(S)):
        d = (X - S[k, 0]) ** 2 + (Y - S[k, 1]) ** 2
        u = d < best
        best = np.where(u, d, best)
        arc = np.where(u, A_[k], arc)
    return np.sqrt(best), arc


def build_v3():
    L = Layer('v3', 106, 150)
    X, Y = L.X, L.Y
    D, A = L.dead, L.alive
    xs = X[0]
    top = v3_top(xs)
    m = Y >= np.round(top)[None, :]
    dep = surface_rows(top, Y)
    rim = m & ~shift(m, 0, 1)
    T = ground_T(X, np.maximum(Y, top[None, :] + 0.5), extra=10.5, seed=9)
    L.T = np.where(m, T, 99.0)
    near = np.clip((Y - 130) / 60.0, 0, 1)
    lightv = 0.3 - np.clip(dep, 0, 16) / 16.0 * 0.55 + 0.5 * (noise.fbm2(X / 45.0, Y / 13.0, seed=162, octaves=2) - 0.5)
    lightv += 0.10 * (NTEX_BIG[(Y.astype(int) * 2) % 256, (X.astype(int)) % 256] - 0.5)
    sd, sarc = _polyline_dist(X, Y, STREAM_PTS)
    sw = 0.9 + np.clip(Y - 134, 0, None) * 0.20
    bed = (sd <= sw) & m
    bank = (sd <= sw + 1.3) & m & ~bed
    # ---------------- DEAD: dust, cracked patches, dry bed
    dcols = ['dust3', 'dust4', 'dust5', 'dust6']
    dq = quant(np.clip(1.9 + lightv * 1.6, 0, 3.99), band=0.25)
    D.put_levels(m, dq, dcols)
    ck = np.where(near > 0.45, cracks(D, 164, cell=12.0, sy=0.5), cracks(D, 163, cell=6.5, sy=0.45))
    ck &= m & (dep > 2) & patchy(D, 165, 34, 10, 0.53)
    D.put(ck, 'dust4')
    D.put(ck & (near > 0.6) & (dq < 2), 'dust3')
    D.put(ripples(m & (dep > 1) & ~ck, 166, D, 0.02, 2, 6), 'dust5')
    D.put(ripples(m & (dep > 1) & ~ck & (near > 0.3), 167, D, 0.012, 2, 5), 'dust4')
    D.put(rim, 'dust6')
    D.put(bed, 'dust4')
    D.put(bed & cracks(D, 168, cell=5.0, sy=0.6), 'dust3')
    D.put(bank & edge(bank | bed, 0, -1), 'dust6')
    D.put(edge(bed, 0, -1) & bed, 'dust2')
    D.put(bed & (rndxy(X, Y, 169) < 0.05), 'ash5')
    # ---------------- ALIVE: calm grass masses + tufts (flowers come in drifts)
    acols = ['leaf2', 'meadow2', 'meadow3', 'meadow4', 'meadow5']
    alv = 2.2 + lightv * 1.7 - 0.9 * near
    aq = quant(np.clip(alv, 0, 4.99), band=0.22)
    A.put_levels(m, aq, acols)
    aqq = np.clip(aq - 1, 0, 3)
    grass_tufts(A, m & (dep > 1) & (near < 0.35), aqq, 168, 0.030)
    grass_tufts(A, m & (dep > 1) & (near >= 0.35), aqq, 169, 0.045, big=0.5)
    A.put(rim, 'meadow5')
    # the stream: sky reflections, darker toward us, bank reflections
    wl = 3.4 - 1.6 * near + 0.6 * (noise.fbm2(X / 8.0, Y / 1.8, seed=172, octaves=2) - 0.5) * 2
    wcols = ['sky3', 'sky4', 'sky5', 'sky6', 'sky7']
    wq = quant(np.clip(wl, 0, 4.99), band=0.35)
    A.put_levels(bed, wq, wcols)
    A.put(edge(bed, 0, -1) & bed, 'meadow1')
    A.put(edge(bed, 0, -1) & shift(bed, 0, 1) & bed & ~edge(bed, 0, -1), 'sky3')
    A.put(bank, 'meadow2')
    A.put(bank & edge(bank | bed, 0, -1), 'meadow4')
    L.flag[bed] = 2
    L.sarc = sarc
    L.bed = bed
    L.T = np.where(bed, np.maximum(T, 0.95 + sarc * 0.0105) + 0.05, L.T)
    flower_drifts(A, m & (dep > 1) & ~bed & ~bank, DRIFTS_V3, 173, T=L.T, flag=L.flag)
    Tcol = T[np.clip(np.round(top - L.y0).astype(int), 0, L.h - 1), np.arange(L.w)]
    paint_tufts(A, top, 175, 0.6, ['meadow3', 'meadow4', 'meadow5'], maxh=3, T=L.T, Tbase=Tcol, lean=1)
    # ---------------- big dead tree (left) -> blossoming tree
    tx, tyb = 262, 140
    tm, tips = _skeleton(D, tx, tyb, 22, 181, spread=1.1)
    tm |= shift(tm, 1, 0) & (Y > tyb - 18)
    D.put(tm, 'dust2')
    D.put(tm & edge(tm, -1, 0), 'dust4')
    A.put(tm, 'bark2')
    A.put(tm & edge(tm, -1, 0), 'bark4')
    Tt = T[int(tyb - L.y0), int(tx - L.x0)]
    L.T = np.where(tm, Tt, L.T)
    leaf_tree(L, tx, tyb, 22, 182, tips, Tt, scale=1.0, blossom=0.18,
              tones=['leaf1', 'leaf2', 'leaf3', 'leaf4', 'leaf5', 'leaf6'])
    rf = D.mask_line(tx - 3, tyb + 1, tx, tyb - 1) | D.mask_line(tx + 4, tyb + 1, tx + 1, tyb - 1)
    D.put(rf, 'dust2')
    A.put(rf, 'bark2')
    # ---------------- car husk (right), half buried -> flowers growing out
    cx0, cy0 = 272, 146
    body = D.mask_poly([(cx0, cy0 + 6), (cx0 + 2, cy0 + 1), (cx0 + 7, cy0 - 3), (cx0 + 15, cy0 - 3),
                        (cx0 + 19, cy0 + 1), (cx0 + 24, cy0 + 2), (cx0 + 25, cy0 + 6)])
    body &= Y <= cy0 + 5
    wind = D.mask_poly([(cx0 + 8, cy0 - 2), (cx0 + 14, cy0 - 2), (cx0 + 17, cy0 + 1), (cx0 + 6, cy0 + 1)])
    wheel = D.mask_disc(cx0 + 20, cy0 + 5, 1.6)
    car = body | wheel
    D.put(car, 'rust2')
    D.put(car & edge(car, 0, -1), 'rust4')
    D.put(car & edge(car, -1, 0), 'rust3')
    D.put(wind & body, 'ink2')
    D.put(wheel, 'ink2')
    D.put((Y == cy0 + 5) & (X >= cx0 - 2) & (X <= cx0 + 27), 'dust5')
    A.put(car, 'rust3')
    A.put(car & edge(car, 0, -1), 'rust5')
    A.put(car & edge(car, -1, 0), 'rust4')
    A.put(wind & body, 'leaf2')
    A.put(wheel, 'ink2')
    A.put(car & (rndxy(X, Y, 183) < 0.3) & (Y > cy0), 'leaf3')
    Tc = T[int(cy0 + 4 - L.y0), int(cx0 + 12 - L.x0)]
    L.T = np.where(car, Tc, L.T)
    for k, (fx_, fy_, kk) in enumerate([(cx0 + 10, cy0 - 5, 0), (cx0 + 12, cy0 - 6, 1), (cx0 + 14, cy0 - 4, 4),
                                        (cx0 + 5, cy0 - 1, 2), (cx0 + 18, cy0 - 1, 0)]):
        A.px(fx_, fy_ + 2, 'leaf3')
        A.px(fx_, fy_ + 3, 'leaf2')
        L.T[int(fy_ + 2 - L.y0), int(fx_ - L.x0)] = Tc + 0.2
        L.T[int(fy_ + 3 - L.y0), int(fx_ - L.x0)] = Tc + 0.2
        for dy, dx, c in flower_sprite(FLOWER_KINDS[kk], 3):
            A.px(fx_ + dx, fy_ + dy, c)
            j, i = int(fx_ + dx - L.x0), int(fy_ + dy - L.y0)
            L.T[i, j] = Tc + 0.3 + 0.07 * k
            L.flag[i, j] = 1
    L.top = top
    return L


# =====================================================================
#  FG: foreground corners (fast parallax)
# =====================================================================
FG_BRANCH = [
    # pts, w0, w1  (rest-screen coords of the FG layer)
    ([(-30, 200), (-4, 181), (20, 165), (40, 151), (56, 137), (66, 124)], 11.0, 2.4),
    ([(22, 164), (17, 152), (18, 140)], 3.6, 1.2),
    ([(41, 150), (52, 152), (64, 146)], 3.2, 1.2),
    ([(56, 137), (49, 126), (47, 116)], 3.0, 1.0),
    ([(64, 128), (74, 125), (82, 119)], 2.4, 1.0),
    ([(6, 176), (-2, 160)], 3.0, 1.2),
]


def build_fg():
    L = Layer('fg', 100, 160)
    X, Y = L.X, L.Y
    D, A = L.dead, L.alive
    T = ground_T(X, np.clip(Y, 150, None) + 22, extra=8.0, seed=11)
    L.T = np.full((L.h, L.w), 99.0)
    # ---------------- the branch (wood) — dead silhouette / living bark
    Tb = 1.22
    # life runs up the branch from its base (off-screen, bottom-left) to the tips
    starts = [Tb, Tb + 0.16, Tb + 0.22, Tb + 0.28, Tb + 0.32, Tb + 0.05]
    bi, bt = paint_wood(L.x0, L.y0, L.w, L.h, [(pts, w0, w1, st, 0.34 if k == 0 else 0.14, 1.2)
                                                for k, ((pts, w0, w1), st) in enumerate(zip(FG_BRANCH, starts))])
    wood = bt < 90
    D.put(wood, 'dust1')
    D.put(wood & edge(wood, 0, -1), 'dust3')
    D.put(wood & edge(wood, -1, 0) & ~edge(wood, 0, -1), 'dust2')
    A.i[wood] = bi[wood]
    A.a[wood] = True
    L.T = np.where(wood, bt, L.T)
    # leaves + blossoms along the twigs (leaf out from the base to the tips)
    tips = []
    for pts, w0, w1 in FG_BRANCH:
        for sfr in (0.45, 0.75, 1.0):
            px, py = limb_point(pts, sfr)
            tips.append((px, py, sfr))
    S = []
    for k, (px, py, sfr) in enumerate(tips):
        if py > 172:
            continue
        for j in range(4):
            a_ = rnd(k * 5 + j, 1, 197) * TAU
            d_ = 3.5 * rnd(k * 5 + j, 2, 197)
            S.append((px + math.cos(a_) * d_, py + math.sin(a_) * d_ * 0.8 - 2, rnd(k * 5 + j, 3, 197) * 4,
                      2.6 + 2.8 * rnd(k * 5 + j, 4, 197)))
    S = np.array(S)
    sid, zb = spheres_zbuffer(L.h, L.w, L.x0, L.y0, S, jag=1.1, seed=198)
    sh, cm = spheres_shade(sid, zb, L.x0, L.y0, S, tex_amp=0.22, seed=198)
    zp = np.pad(zb, ((1, 0), (1, 0)), constant_values=-1e9)[:-1, :-1]
    occ = (zp - zb) > 1.3
    q = quant(np.clip((sh + 0.45) / 0.27 - 1.2 * occ, 0, 5.99), band=0.3)
    A.put_levels(cm, q, ['leaf0', 'leaf1', 'leaf2', 'leaf3', 'leaf4', 'leaf5'])
    k_ = np.maximum(sid, 0)
    dd = np.sqrt((X - S[k_, 0]) ** 2 + (Y - S[k_, 1]) ** 2) / np.maximum(S[k_, 3], 1)
    base_d = np.sqrt((S[k_, 0] + 30) ** 2 + (S[k_, 1] - 200) ** 2)
    L.T = np.where(cm, Tb + 0.12 + base_d * 0.0040 + 0.14 * dd, L.T)
    # cherry blossom clusters: florets + five-petal flowers, lit side heavier
    anchors = cm & (rndxy(X * 3, Y, 199) < 0.07) & (sh > -0.15)
    anchors &= ~shift(anchors, 1, 0) & ~shift(anchors, 0, 1) & ~shift(anchors, 2, 0)
    ys_, xs_ = np.nonzero(anchors)
    for n in range(len(ys_)):
        i, j = ys_[n], xs_[n]
        v = rnd(n, 1, 200)
        fam = BLOSSOM_FAM[int(v * 7.3) % len(BLOSSOM_FAM)]
        lvl = 0 if sh[i, j] > 0.45 else (1 if sh[i, j] > 0.15 else 2)
        hi, lo = fam[lvl], fam[lvl + 1]
        pat = BLOSSOM_PATS[int(rnd(n, 2, 200) * 97) % len(BLOSSOM_PATS)]
        for f_, (fx_, fy_) in enumerate(pat):
            q = int(rnd(n * 5 + f_, 3, 200) * 100)
            shape = FIVE_PETAL if (q < 30 and lvl < 2) else FLORETS[q % len(FLORETS)]
            for px_, py_, part in shape:
                ii, jj = i + fy_ + py_, j + fx_ + px_
                if 0 <= ii < L.h and 0 <= jj < L.w and cm[ii, jj]:
                    A.i[ii, jj] = IDX['gold3' if part == 'c' else (hi if part == 'h' else lo)]
                    L.T[ii, jj] = L.T[i, j] + 0.2 + 0.1 * rnd(n, 4, 200)
                    L.flag[ii, jj] = 1
    # ---------------- right corner: rock + tall grass / flower spikes
    rockR = D.mask_ellipse(322, 183, 18, 9) | D.mask_ellipse(300, 188, 9, 5)
    D.put(rockR, 'dust2')
    D.put(rockR & edge(rockR, 0, -1), 'dust4')
    A.put(rockR, 'ash3')
    A.put(rockR & edge(rockR, 0, -1), 'leaf3')
    A.put(rockR & edge(rockR, 0, -1) & (rndxy(X, Y, 196) < 0.4), 'leaf4')
    L.T = np.where(rockR, T, L.T)
    grass = np.zeros((L.h, L.w), bool)
    stalk_d = np.zeros((L.h, L.w), bool)
    Tg = np.full((L.h, L.w), 99.0)
    for k in range(46):
        gx = 262 + rnd(k, 1, 193) * 100
        gb = 196
        gh = 22 + rnd(k, 2, 193) ** 0.7 * 44
        bend = (rnd(k, 3, 193) - 0.3) * 9
        pts = [(gx, gb), (gx + bend * 0.3, gb - gh * 0.5), (gx + bend, gb - gh)]
        xs_, ys_, arc = _resample(pts, 0.5)
        for x_, y_ in zip(xs_, ys_):
            j, i = int(round(x_ - L.x0)), int(round(y_ - L.y0))
            if 0 <= i < L.h and 0 <= j < L.w:
                grass[i, j] = True
                Tg[i, j] = min(Tg[i, j], 1.3 + (gb - y_) * 0.006)
                if k % 3 == 0:
                    stalk_d[i, j] = True
    D.put(stalk_d, 'dust3')
    D.put(stalk_d & (rndxy(X, Y, 194) < 0.35), 'dust4')
    A.put(grass, 'leaf1')
    A.put(grass & (rndxy(X, Y, 194) < 0.45), 'leaf2')
    A.put(grass & edge(grass, -1, 0), 'leaf3')
    L.T = np.where(grass, Tg, L.T)
    # flower spikes (lupins): tapering spires of bells on leaning stems,
    # palmate leaves at their feet
    for k in range(8):
        gx = 272 + k * 10.5 + (rnd(k, 5, 195) - 0.5) * 6
        top_y = 124 + rnd(k, 6, 195) * 30
        lean = (rnd(k, 8, 195) - 0.5) * 0.12
        kind = [('violet3', 'violet4', 'violet1'), ('bloom3', 'bloom5', 'bloom1'),
                ('violet2', 'violet4', 'violet0'), ('white0', 'white1', 'ash5')][k % 4]
        n_b = 7 + int(rnd(k, 9, 195) * 5)
        for j in range(n_b):
            y_ = top_y + j * 1.6
            x_ = gx + lean * (y_ - top_y)
            wdt = 0 if j < 2 else (1 if j < n_b - 2 else 1)
            for dx in range(-wdt, wdt + 1):
                c = kind[1] if (dx < 0 or j < 2) else (kind[0] if j < n_b - 2 else kind[2])
                if (j + k) % 3 == 0 and dx > 0:
                    continue
                A.px(x_ + dx, y_, c)
                jj, ii = int(round(x_ + dx - L.x0)), int(round(y_ - L.y0))
                if 0 <= ii < L.h and 0 <= jj < L.w:
                    L.T[ii, jj] = 1.45 + 0.05 * j + 0.08 * rnd(k, 7, 195)
                    L.flag[ii, jj] = 1
        stem_top = top_y + n_b * 1.6
        for j in range(0, int(196 - stem_top)):
            y_ = stem_top + j
            x_ = gx + lean * (y_ - top_y)
            A.px(x_, y_, 'leaf2' if j % 5 else 'leaf3')
            jj, ii = int(round(x_ - L.x0)), int(round(y_ - L.y0))
            if 0 <= ii < L.h and 0 <= jj < L.w:
                L.T[ii, jj] = 1.35
        # palmate leaf fan near the base
        ly = 176 + rnd(k, 10, 195) * 8
        lx = gx + lean * (ly - top_y)
        for a_ in (-2.4, -1.9, -1.4, -0.9, -0.5):
            for r_ in range(1, 5):
                A.px(lx + math.cos(a_) * r_ * 1.2, ly + math.sin(a_) * r_ * 0.7, 'leaf3' if r_ < 4 else 'leaf4')
                jj, ii = int(round(lx + math.cos(a_) * r_ * 1.2 - L.x0)), int(round(ly + math.sin(a_) * r_ * 0.7 - L.y0))
                if 0 <= ii < L.h and 0 <= jj < L.w:
                    L.T[ii, jj] = 1.33
    return L


def build_world():
    return {'sky': build_sky(), 'far': build_far(), 'v1': build_v1(), 'v2': build_v2(),
            'hill': build_hill(), 'v3': build_v3(), 'fg': build_fg()}


ORDER = ['sky', 'far', 'v1', 'v2', 'hill', 'v3', 'fg']


# =====================================================================
#  THE GREENHOUSE DOME (Victorian wrought iron + glass), burst open
# =====================================================================
HOLE_R = 9.0
DOME_RINGS = (0.17, 0.35, 0.53, 0.71, 0.90, 1.10)
TRUNK_PTS = [(150, 113), (151, 99), (149, 86), (148, 74), (150.5, 65), (151, 57)]


def trunk_x(y):
    ys = np.array([p[1] for p in TRUNK_PTS][::-1], np.float64)
    xs = np.array([p[0] for p in TRUNK_PTS][::-1], np.float64)
    return np.interp(y, ys, xs)


def _dome_proj(theta, phi, rx=DOME_RX - 1, e=1.8):
    x = DOME_X + rx * np.sin(theta) * np.cos(phi)
    y = DOME_Y0 - DOME_HG * np.sin(phi) + e * np.cos(theta) * np.cos(phi)
    return x, y


def dome_masks(Pn):
    X, Y = Pn.X, Pn.Y
    h, w = Pn.h, Pn.w
    dy = (DOME_Y0 - Y) / DOME_HG
    half = (DOME_RX - 0.5) * np.sqrt(np.clip(1 - dy ** 2, 0, 1)) ** 0.92
    shell = (np.abs(X - DOME_X) <= half) & (Y <= DOME_Y0 + 1) & (Y >= APEX_Y)
    ang = np.arctan2(Y - (APEX_Y + 2), X - DOME_X)
    jag = 2.4 * (noise.fbm1((ang + math.pi) * 2.4, 301, 2) - 0.5) * 2
    hole = ((X - DOME_X) ** 2 + ((Y - (APEX_Y + 1)) * 1.2) ** 2) <= (HOLE_R + jag) ** 2
    hole &= Y <= APEX_Y + 12
    rib = np.zeros((h, w), bool)
    nr = 18
    ribs_front = []
    for k in range(nr):
        th = (k + 0.5) * TAU / nr - math.pi
        if math.cos(th) < 0.02:
            continue
        phis = np.linspace(0, math.pi / 2, 90)
        xs, ys = _dome_proj(th, phis)
        for x_, y_ in set(zip(np.round(xs).astype(int), np.round(ys).astype(int))):
            j, i = x_ - Pn.x0, y_ - Pn.y0
            if 0 <= i < h and 0 <= j < w:
                rib[i, j] = True
        ribs_front.append(th)
    ring = np.zeros((h, w), bool)
    for phi in DOME_RINGS:
        ths = np.linspace(-math.pi / 2, math.pi / 2, 200)
        xs, ys = _dome_proj(ths, phi)
        for x_, y_ in zip(xs, ys):
            j, i = int(round(x_)) - Pn.x0, int(round(y_)) - Pn.y0
            if 0 <= i < h and 0 <= j < w:
                ring[i, j] = True
    frame = (rib | ring) & shell
    bent = np.zeros((h, w), bool)
    for k, th in enumerate(ribs_front):
        x_, y_ = _dome_proj(th, 1.13)
        dxh = x_ - DOME_X
        ex = x_ + dxh * 0.6 + (1 if dxh >= 0 else -1) * 2
        ey = y_ - 7 - abs(math.cos(th)) * 3 - (k % 2) * 2
        bent |= Pn.mask_line(x_, y_, ex, ey)
    outl = (shell & ~shift(shell, 1, 0)) | (shell & ~shift(shell, -1, 0)) | (shell & ~shift(shell, 0, 1))
    outl &= ~hole
    glass = shell & ~hole
    hole_rim = grow(hole, 1) & glass
    broken = glass & (noise.fbm2(X / 4.5, Y / 3.2, seed=303, octaves=2) > 0.61) & ~frame
    wall = (np.abs(X - DOME_X) <= DOME_RX + 1) & (Y > DOME_Y0) & (Y <= DOME_BASE + 1)
    wall &= Y <= DOME_BASE + 0.5 + 1.5 * np.sqrt(np.clip(1 - ((X - DOME_X) / (DOME_RX + 1)) ** 2, 0, 1))
    door = (X >= 140) & (X <= 147) & (Y >= DOME_Y0 - 8) & (Y <= DOME_BASE + 1)
    door &= ~((Y == DOME_Y0 - 8) & ((X == 140) | (X == 147)))
    door &= ~((Y == DOME_Y0 - 8) & ((X == 141) | (X == 146)) & False)
    doorf = grow(door, 1, diag=False) & ~door & (Y <= DOME_BASE + 1)
    bricks = wall & (((Y - DOME_Y0) % 2 == 0) | (((X + (Y // 2) * 3) % 6) == 0))
    # trunk behind the glass
    tx = trunk_x(Y)
    tw = 6.5 + (Y - APEX_Y) / (DOME_BASE - APEX_Y) * 1.2
    behind = (np.abs(X - tx) <= tw) & shell & (Y > APEX_Y)
    return dict(shell=shell, hole=hole, frame=frame, bent=bent, outl=outl, glass=glass,
                hole_rim=hole_rim, broken=broken, wall=wall, door=door, doorf=doorf, bricks=bricks,
                behind=behind, tx=tx)


def dome_panes(X, Y):
    """Approximate (theta, phi) of the dome surface under each pixel ->
    pane ids (column, row) and in-pane coordinates for bevels."""
    sphi = np.clip((DOME_Y0 - Y + 0.5) / DOME_HG, 0, 0.999)
    phi = np.arcsin(sphi)
    rxp = (DOME_RX - 1) * np.cos(phi)
    sth = np.clip((X - DOME_X) / np.maximum(rxp, 0.5), -0.999, 0.999)
    th = np.arcsin(sth)
    nr = 18
    step = TAU / nr
    col_f = (th + math.pi) / step - 0.5
    col = np.floor(col_f).astype(int)
    rings = np.array([0.0] + list(DOME_RINGS) + [1.6])
    row = np.searchsorted(rings, phi) - 1
    u = col_f - col                                    # 0..1 across the pane
    nrg = len(rings) - 1
    lo = rings[np.clip(row, 0, nrg - 1)]
    hi = rings[np.clip(row + 1, 1, nrg)]
    v = (phi - lo) / np.maximum(hi - lo, 1e-3)         # 0 bottom .. 1 top
    return col, row, u, v, th, phi


def paint_dome(L, s9=False):
    """Bake the burst dome into the hill layer (dead + alive states)."""
    Ld, La = L.dead, L.alive
    X, Y = Ld.X, Ld.Y
    M = dome_masks(Ld)
    glass, frame, behind = M['glass'], M['frame'], M['behind']
    col, row, u, v, th, phi = dome_panes(X, Y)
    pid = col * 7 + row
    face = -np.sin(th) * 0.9 + np.sin(phi) * 0.45          # facing the upper-left light
    prand = rnd(pid, 1, 321)
    broken = glass & (rnd(pid, 2, 322) < (0.20 if not s9 else 0.26)) & ~frame
    # jagged shard left in some broken panes (upper edge)
    shard = broken & (v > 0.72 - 0.25 * rnd(pid, 3, 323)) & (u < 0.35 + 0.5 * rnd(pid, 4, 324))
    broken &= ~shard
    bevel = glass & ((u < 0.22) | (v > 0.80)) & ~broken      # lit upper-left pane edges
    lowgrime = np.clip(1 - np.sin(phi) * 2.2, 0, 1)
    tone = 1.9 + face * 1.25 + (prand - 0.5) * 0.9 - 0.8 * lowgrime
    tq = np.clip(np.floor(tone + 0.5), 0, 4).astype(int)
    M['broken'] = broken
    # the tree's interior canopy fills the upper dome, pressing on the glass
    lump = noise.fbm2(X / 5.0, Y / 4.0, seed=331, octaves=2)
    fol_e = ((X - DOME_X) / 29.0) ** 2 + ((Y - (APEX_Y + 15)) / 12.5) ** 2
    fol = (fol_e < 0.75 + 0.55 * (lump - 0.5) * 2) & M['shell'] & (Y < DOME_Y0 - 6)
    fol_lit = fol & (np.sin(th) < -0.15)
    # ================= DEAD: grimy glass, dark iron
    dcol = cis(['glass1', 'glass2', 'dust4', 'glass3', 'dust6'])
    Ld.i[glass] = dcol[tq[glass]]
    Ld.a[glass] = True
    b2 = bevel & (tq < 4)
    Ld.i[b2] = dcol[np.minimum(tq[b2] + 1, 4)]
    Ld.put(glass & behind & ~broken, 'glass1')
    Ld.put(glass & behind & ~broken & bevel, 'glass2')
    Ld.put(glass & fol & ~broken, 'leaf1')
    Ld.put(glass & fol & ~broken & (bevel | fol_lit & (lump > 0.5)), 'leaf2')
    Ld.put(broken, 'ink1')
    Ld.put(broken & behind, 'bark1')
    Ld.put(broken & behind & (X < M['tx']), 'bark2')
    Ld.put(shard, 'glass3')
    Ld.put(broken & fol, 'leaf2')
    Ld.put(broken & fol & (lump > 0.52), 'leaf3')
    Ld.put(broken & fol & (lump > 0.62) & fol_lit, 'leaf4')
    Ld.put(frame | M['outl'], 'ink2')
    Ld.put((frame | M['outl']) & (np.sin(th) < -0.1) & (rndxy(X, Y, 304) < 0.55), 'rust1')
    drip = glass & ~broken & shift(frame & M['glass'], 0, 1) & (rndxy(X, Y, 305) < 0.35) & ~frame & (row < 4)
    Ld.put(drip, 'rust1')
    Ld.put(M['bent'], 'rust1')
    Ld.put(M['bent'] & (rndxy(X, Y, 306) < 0.4), 'rust2')
    Ld.put(M['hole_rim'], 'ink2')
    Ld.put(M['wall'], 'ash4')
    Ld.put(M['bricks'], 'ash3')
    Ld.put(M['wall'] & edge(M['wall'], 0, -1), 'ash6')
    Ld.put(M['door'], 'ink0')
    Ld.put(M['doorf'], 'rust1')
    # ================= ALIVE: glass catches the blue sky, warm iron
    acol = cis(['glass1', 'glass2', 'sky4', 'glass3', 'sky6'])
    La.i[glass] = acol[tq[glass]]
    La.a[glass] = True
    La.i[b2] = acol[np.minimum(tq[b2] + 1, 4)]
    sheen = glass & ~broken & (np.abs((X - DOME_X) * 0.75 + (Y - DOME_Y0) * 0.65 + 15) < 1.3) & (np.sin(th) < 0.1)
    La.put(sheen, 'sky7')
    La.put(sheen & bevel, 'white0')
    La.put(glass & behind & ~broken, 'glass1')
    La.put(glass & behind & ~broken & bevel, 'glass2')
    La.put(glass & fol & ~broken, 'leaf2')
    La.put(glass & fol & ~broken & (bevel | fol_lit & (lump > 0.5)), 'leaf3')
    La.put(broken, 'ink1')
    La.put(broken & behind, 'bark2')
    La.put(broken & behind & (X < M['tx']), 'bark3')
    La.put(shard, 'glass4')
    La.put(broken & fol, 'leaf2')
    La.put(broken & fol & (lump > 0.52), 'leaf3')
    La.put(broken & fol & (lump > 0.62) & fol_lit, 'leaf5')
    La.put(broken & fol & (lump > 0.7) & fol_lit & (rndxy(X, Y, 332) < 0.3), 'bloom4')
    La.put(frame | M['outl'], 'rust1')
    La.put((frame | M['outl']) & (np.sin(th) < -0.1), 'rust3')
    La.put(drip, 'rust2')
    La.put(M['bent'], 'rust2')
    La.put(M['bent'] & (rndxy(X, Y, 306) < 0.4), 'rust3')
    La.put(M['hole_rim'], 'rust1')
    La.put(M['wall'], 'ash5')
    La.put(M['bricks'], 'ash3')
    La.put(M['wall'] & edge(M['wall'], 0, -1), 'ash7')
    La.put(M['door'], 'ink0')
    La.put(M['doorf'], 'rust2')
    # moss creeping over the wall; tendrils climbing the glass
    mn = noise.fbm2(X / 4.0, Y / 2.5, seed=307, octaves=2)
    moss = M['wall'] & (mn > (0.52 if not s9 else 0.38))
    La.put(moss, 'leaf3')
    La.put(moss & edge(moss, 0, -1), 'leaf4')
    La.put(moss & (rndxy(X, Y, 308) < 0.08), 'bloom4')
    L.T = np.where(moss, T_DOME + 0.12, L.T)
    Tv = L.T.copy()
    paths = vine_paths(11 if not s9 else 19, 309, 16 if not s9 else 34, s9)
    paint_vines(La, Tv, paths, 310, T_DOME + 0.15, flowers=0.10 if not s9 else 0.16,
                leafy=0.55 if not s9 else 0.8, dense=s9)
    vine = (Tv != L.T)
    L.T = np.where(vine, Tv, L.T)
    if s9:
        # ivy clusters and a skirt of leaves at the base
        S = []
        for k in range(26):
            th = (rnd(k, 1, 311) - 0.5) * 2.9
            phi = rnd(k, 2, 311) ** 1.7 * 0.8
            x_, y_ = _dome_proj(th, phi)
            S.append((float(x_), float(y_) + 1, rnd(k, 3, 311) * 3, 1.6 + rnd(k, 4, 311) * 2.4))
        S = np.array(S)
        sid, zb = spheres_zbuffer(L.h, L.w, L.x0, L.y0, S, jag=0.8, seed=312)
        sh, cm = spheres_shade(sid, zb, L.x0, L.y0, S, tex_amp=0.2)
        cm &= Y <= DOME_BASE + 1
        q = quant(np.clip((sh + 0.5) / 0.32, 0, 4.99), band=0.3)
        La.put_levels(cm, q, ['leaf1', 'leaf2', 'leaf3', 'leaf4', 'leaf5'])
        La.put(cm & (rndxy(X, Y, 313) < 0.05) & (sh > 0.2), 'white0')
        L.T = np.where(cm, T_DOME + 0.3, L.T)
    allm = M['shell'] | M['wall'] | M['bent'] | M['doorf'] | M['door']
    L.T = np.where(allm, T_DOME, L.T)
    L.T = np.where(vine, T_DOME + 0.15 + (DOME_BASE - Y) * 0.03, L.T)
    L.flag[M['glass'] & ~M['broken'] & ~M['frame']] = 3          # glass (inner glow)
    return M


# =====================================================================
#  THE TREE
# =====================================================================
BARK = ['bark1', 'bark2', 'bark3', 'bark4', 'bark5']
LEAF_T = ['leaf0', 'leaf1', 'leaf2', 'leaf3', 'leaf4', 'leaf5', 'leaf6', 'leaf7']


def _catmull(pts, per=12):
    P_ = np.array(pts, np.float64)
    if len(P_) < 3:
        return P_
    Q = [P_[0]] + list(P_) + [P_[-1]]
    out = []
    for k in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[k - 1], Q[k], Q[k + 1], Q[k + 2]
        for s in np.linspace(0, 1, per, endpoint=False):
            s2, s3 = s * s, s * s * s
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s2 +
                              (-p0 + 3 * p1 - 3 * p2 + p3) * s3))
    out.append(P_[-1])
    return np.array(out)


def _resample(pts, step=0.5):
    P_ = _catmull(pts)
    seg = np.sqrt(((P_[1:] - P_[:-1]) ** 2).sum(1))
    arc = np.concatenate([[0], np.cumsum(seg)])
    n = max(2, int(arc[-1] / step))
    a = np.linspace(0, arc[-1], n)
    return np.interp(a, arc, P_[:, 0]), np.interp(a, arc, P_[:, 1]), a


def _inv_out_cubic(v):
    v = np.clip(v, 0, 1)
    return 1 - (1 - v) ** (1 / 3.0)


def _blit_mask(idx, alpha, src_i, src_m, x0, y0):
    h, w = idx.shape
    sh, sw = src_i.shape
    X0, Y0 = max(0, x0), max(0, y0)
    X1, Y1 = min(w, x0 + sw), min(h, y0 + sh)
    if X0 >= X1 or Y0 >= Y1:
        return
    s_i = src_i[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    s_m = src_m[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    d_i = idx[Y0:Y1, X0:X1]
    d_i[s_m] = s_i[s_m]
    alpha[Y0:Y1, X0:X1] |= s_m


def paint_wood(x0, y0, w, h, strokes):
    """Render bark strokes into (idx, appear-time) images.
    strokes: list of (pts, w0, w1, t0, dur, twist)."""
    idx = np.zeros((h, w), np.uint8)
    best = np.full((h, w), 9.0)
    appear = np.full((h, w), 99.0)
    barkc = cis(BARK)
    for (pts, w0, w1, t0, dur, twist) in strokes:
        xs, ys, arc = _resample(pts)
        L_ = arc[-1]
        for n in range(len(xs)):
            s = arc[n] / max(L_, 1e-6)
            wd = w0 + (w1 - w0) * s
            r = wd / 2.0
            cx, cy = xs[n] - x0, ys[n] - y0
            X0, X1 = int(max(cx - r - 1, 0)), int(min(cx + r + 2, w))
            Y0, Y1 = int(max(cy - r - 1, 0)), int(min(cy + r + 2, h))
            if X0 >= X1 or Y0 >= Y1:
                continue
            yy = np.arange(Y0, Y1)[:, None]
            xx = np.arange(X0, X1)[None, :]
            ox_, oy_ = xx - cx, yy - cy
            d = np.sqrt(ox_ ** 2 + oy_ ** 2) / max(r + 0.35, 0.6)
            inside = d <= 1.0
            sub_b = best[Y0:Y1, X0:X1]
            upd = inside & (d < sub_b + 0.25)
            if not upd.any():
                continue
            ld = (ox_ * -0.72 + oy_ * -0.55) / max(r, 0.6)
            if n + 1 < len(xs):
                tx_, ty_ = xs[n + 1] - xs[n], ys[n + 1] - ys[n]
            else:
                tx_, ty_ = xs[n] - xs[n - 1], ys[n] - ys[n - 1]
            tl = math.hypot(tx_, ty_) + 1e-6
            across = (ox_ * (-ty_) + oy_ * tx_) / tl / max(r, 0.6)
            ph = arc[n] * 0.26 + across * twist
            groove = (np.sin(ph * math.pi) > 0.6) & (r > 1.8)
            ridge = (np.sin(ph * math.pi) < -0.85) & (r > 2.5) & (ld > -0.2)
            lv = 2.2 + 1.45 * ld - 0.95 * groove + 0.7 * ridge
            lv = np.where(d > 0.8, lv - 0.75, lv)
            q = np.clip(np.floor(lv + 0.5), 0, 4).astype(int)
            sub_i = idx[Y0:Y1, X0:X1]
            sub_i[upd] = barkc[q][upd]
            sub_b[upd] = np.minimum(sub_b[upd], d[upd])
            ap = t0 + float(_inv_out_cubic(s)) * dur
            sub_a = appear[Y0:Y1, X0:X1]
            sub_a[upd] = np.minimum(sub_a[upd], ap)
    return idx, appear


SCALES = np.array([0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.84, 0.93, 1.0, 1.07, 1.14, 1.21])


def _sphere_patch(r, k, seed, jag=1.3, tips=True, tex_amp=0.40):
    """Precomputed disc patch of a shaded sphere: (R, inside, zrel, dl)."""
    R = int(math.ceil(r + jag + 2))
    ys, xs = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float64)
    d2 = xs * xs + ys * ys
    e = ntex(NTEX_EDGE, xs + 37 * k + seed * 11, ys + 53 * k + seed * 7)
    rr = r + jag * (e - 0.5) * 2
    if tips and r > 2.5:
        # leaf tips: a few sectors poke out by 1-2 px (sparse, mostly upward)
        ang = np.arctan2(ys, xs)
        sec = np.floor((ang + math.pi) / TAU * 22).astype(int)
        tip = rnd(sec + 31 * k, 7, seed + 3)
        up = np.clip(-ys / max(r, 1) + 0.3, 0, 1)
        rr = rr + np.where(tip < 0.22 + 0.25 * up, 1.2 + (tip < 0.08), 0.0)
    inside = d2 <= rr * rr
    zrel = np.sqrt(np.clip(r * r - d2, 0, None))
    rs = max(r, 0.6)
    nx, ny = xs / rs, ys / rs
    nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
    dl = nx * LIGHT3[0] + ny * LIGHT3[1] + nz * LIGHT3[2]
    tex = (ntex(NTEX_LEAF, xs + 29 * k + seed, ys + 41 * k + seed) - 0.5) * 2
    dlc = 1.25 * dl + tex_amp * tex
    return R, inside, zrel, dlc


class Tree:
    """The great tree: bark strokes (reveal-by-time) + a canopy of clumped,
    z-buffered, shaded spheres (precomputed patches at growth scales) with
    blossom clusters; growth burst + sway."""

    def __init__(self, strokes, clumps, x0=-40, y0=-90, w=400, h=220, seed=11, clip=None,
                 env=(CANOPY_C[0], CANOPY_C[1], CANOPY_R[0], CANOPY_R[1]), blossoms=1.3,
                 tones=LEAF_T, base_lv=2.9, scales=True):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.seed = seed
        self.env = env
        self.tones = cis(tones)
        self.base_lv = base_lv
        # the trunk (first stroke) is clipped where the dome's glass is in
        # front of it; limbs & collar roots are painted over it
        ti, tt = paint_wood(x0, y0, w, h, strokes[:1])
        if clip is not None:
            tt = np.where(clip, 99.0, tt)
        ri, rt = paint_wood(x0, y0, w, h, strokes[1:])
        top = rt < 90
        self.wood_i = np.where(top, ri, ti)
        self.wood_t = np.where(top, np.minimum(rt, np.where(tt < 90, tt, 99.0)), tt)
        # clumps -> spheres (cx, cy, cz, r, birth, hostx, hosty)
        S = []
        clump_of = []
        self.mass_of_clump = np.array([cl[6] if len(cl) > 6 else 0 for cl in clumps])
        for c, cl in enumerate(clumps):
            cx, cy, R, birth, hx, hy = cl[:6]
            u = (cx - env[0]) / env[2]
            v = (cy - env[1]) / env[3]
            cz0 = 30 * math.sqrt(max(0.0, 1 - u * u - v * v)) + 4
            n = 5 + int(R / 3.5)
            for k in range(n):
                a = rnd(c * 17 + k, 1, seed) * TAU
                dd = R * 0.55 * math.sqrt(rnd(c * 17 + k, 2, seed))
                ox_, oy_ = math.cos(a) * dd, math.sin(a) * dd * 0.8 - R * 0.12
                r = R * (0.40 + 0.22 * rnd(c * 17 + k, 3, seed))
                cz = cz0 + (1 - dd / max(R, 1)) * R * 0.5 + rnd(c * 17 + k, 4, seed) * 2
                S.append((cx + ox_, cy + oy_, cz, r, birth + 0.05 * rnd(c * 17 + k, 5, seed), hx, hy))
                clump_of.append(c)
        self.S = np.array(S, np.float64)
        self.clump_of = np.array(clump_of)
        sc_list = SCALES if scales else np.array([1.0])
        self.scales = sc_list
        self.patches = [[_sphere_patch(self.S[k, 3] * sc, k, seed) for sc in sc_list]
                        for k in range(len(self.S))]
        # blossom clusters, gathered on some clumps, on the lit side
        B = []
        for k in range(len(self.S)):
            c = clump_of[k]
            dense = rnd(c, 9, seed + 5) < 0.55
            nb = int(blossoms * (1.25 if dense else 0.18) * (0.5 + rnd(k, 1, seed + 5)))
            for b in range(nb):
                a = -2.2 + (rnd(k * 7 + b, 2, seed + 5) - 0.5) * 3.0
                rr = 0.2 + 0.62 * rnd(k * 7 + b, 3, seed + 5)
                B.append((k, math.cos(a) * rr, math.sin(a) * rr, rnd(k * 7 + b, 4, seed + 5),
                          rnd(k * 7 + b, 6, seed + 5)))
        self.B = np.array(B, np.float64) if B else np.zeros((0, 5))

    def sphere_state(self, t, sway=0.0, sway_t=0.0, grow=True):
        S = self.S
        if not grow:
            cx, cy = S[:, 0].copy(), S[:, 1].copy()
            sc = np.ones(len(S))
        else:
            age = t - S[:, 4]
            # hold the burst until the trunk tip has (almost) reached the crown
            arrive = np.clip((t - (TRUNK_T0 + TRUNK_D * 0.62)) / 0.05, 0, 1)
            age_b = np.where(arrive > 0, t - np.maximum(S[:, 4], TRUNK_T0 + TRUNK_D * 0.62), 0.0)
            tau = np.clip(age_b / BURST_D, 0, 1)
            # position: out-back (overshoot past the limbs, settle = follow-through)
            s_ = 1.25
            tm = tau - 1
            E = tm * tm * ((s_ + 1) * tm + s_) + 1
            E = np.where(age <= 0, 0.0, E)
            hy = np.maximum(S[:, 6], trunk_tip_y(t) - 3.0)
            cx = S[:, 5] + (S[:, 0] - S[:, 5]) * E
            cy = hy + (S[:, 1] - hy) * E
            E = np.where(age_b <= 0, 0.0, E)
            cx = S[:, 5] + (S[:, 0] - S[:, 5]) * E
            cy = hy + (S[:, 1] - hy) * E
            # radius: a tight bud first, swelling as the canopy spreads
            g = np.clip(age / 0.14, 0, 1)
            sc = np.where(age <= 0, 0.0, (0.26 + 0.74 * np.clip(E, 0, 1.2)) * (1 - (1 - g) ** 2))
        if sway:
            ph = S[:, 0] * 0.045 + S[:, 1] * 0.03
            sw = sway * (np.sin(sway_t * 1.6 + ph) * 0.7 + np.sin(sway_t * 2.9 + ph * 2.3) * 0.3)
            reach = np.clip(np.abs(S[:, 0] - CROWN[0]) / 90.0 + (CROWN[1] - S[:, 1]) / 90.0, 0, 1)
            cx = cx + sw * (0.3 + 0.7 * reach)
            cy = cy + 0.35 * sw * np.sin(sway_t * 2.2 + ph)
        return cx, cy, sc

    def draw(self, idx, alpha, ox, oy, t, sway=0.0, sway_t=0.0, light_boost=0.0, grow=True,
             blossom_on=True):
        wm = (self.wood_t <= t) & (self.wood_t < 90)
        _blit_mask(idx, alpha, self.wood_i, wm, self.x0 - ox, self.y0 - oy)
        self._draw_spheres(idx, alpha, ox, oy, t, sway=sway, sway_t=sway_t, light_boost=light_boost,
                           grow=grow, blossom_on=blossom_on)

    def _draw_spheres(self, idx, alpha, ox, oy, t, sway=0.0, sway_t=0.0, light_boost=0.0, grow=True,
                      blossom_on=True, want_z=False):
        h, w = idx.shape
        cx, cy, sc = self.sphere_state(t, sway=sway, sway_t=sway_t, grow=grow)
        zb = np.full((h, w), -1e9)
        dlb = np.zeros((h, w))
        sid = np.full((h, w), -1, np.int32)
        cz = self.S[:, 2]
        ns = len(self.scales)
        for k in np.nonzero(sc * self.S[:, 3] > 0.45)[0]:
            si = int(np.argmin(np.abs(self.scales - sc[k]))) if ns > 1 else 0
            R, inside, zrel, dl = self.patches[k][si]
            px = int(round(cx[k] - ox)) - R
            py = int(round(cy[k] - oy)) - R
            X0, Y0 = max(px, 0), max(py, 0)
            X1, Y1 = min(px + 2 * R + 1, w), min(py + 2 * R + 1, h)
            if X0 >= X1 or Y0 >= Y1:
                continue
            sl = (slice(Y0 - py, Y1 - py), slice(X0 - px, X1 - px))
            z = zrel[sl] + cz[k]
            zs = zb[Y0:Y1, X0:X1]
            upd = inside[sl] & (z > zs)
            zs[upd] = z[upd]
            dlb[Y0:Y1, X0:X1][upd] = dl[sl][upd]
            sid[Y0:Y1, X0:X1][upd] = k
        m = sid >= 0
        if not m.any():
            return zb if want_z else None
        yy, xx = np.nonzero(m)
        ex, ey, erx, ery = self.env
        u = (xx + ox - ex) / erx
        v = (yy + oy - ey) / ery
        gz = np.sqrt(np.clip(1 - u * u - v * v, 0.04, 1))
        gn = np.sqrt(u * u + (v * 1.25) ** 2 + gz * gz)
        dg = (u * LIGHT3[0] + v * 1.25 * LIGHT3[1] + gz * LIGHT3[2]) / gn
        zpad = np.pad(zb, ((2, 0), (2, 0)), constant_values=-1e9)[:-2, :-2]
        occ = ((zpad - zb) > 2.2)[m]
        zpad1 = np.pad(zb, ((1, 0), (1, 0)), constant_values=-1e9)[:-1, :-1]
        occ1 = ((zpad1 - zb) > 1.2)[m]
        lv = self.base_lv + 1.9 * dg + dlb[m] - 1.35 * occ - 0.55 * occ1 + light_boost
        th = bay(h, w, 4, ox, oy)[m]
        q = np.clip(np.floor(lv + (th - 0.5) * 0.45 + 0.5), 0, len(self.tones) - 1).astype(int)
        idx[yy, xx] = self.tones[q]
        alpha[yy, xx] = True
        if blossom_on and len(self.B):
            lvmap = np.full((h, w), -9.0)
            lvmap[m] = lv
            self._blossoms(idx, sid, lvmap, cx, cy, sc, t, ox, oy, grow)
        return zb if want_z else None

    # ------------------------------------------------------------ baked
    def bake(self, pad=6):
        """Pre-render the settled canopy: one complete sprite per foliage
        mass WITH its z-buffer, so masses composite by per-pixel z-test
        (identical to the full render when still; cheap; masses can sway)."""
        mass_of_sphere = self.mass_of_clump[self.clump_of]
        nm = int(mass_of_sphere.max()) + 1
        full_S = self.S
        self.baked = []
        for mi in range(nm):
            sel = np.nonzero(mass_of_sphere == mi)[0]
            if not len(sel):
                continue
            Sx = full_S[sel]
            x0 = int(math.floor((Sx[:, 0] - Sx[:, 3]).min())) - pad
            x1 = int(math.ceil((Sx[:, 0] + Sx[:, 3]).max())) + pad
            y0 = int(math.floor((Sx[:, 1] - Sx[:, 3]).min())) - pad
            y1 = int(math.ceil((Sx[:, 1] + Sx[:, 3]).max())) + pad
            h, w = y1 - y0, x1 - x0
            idx = np.zeros((h, w), np.uint8)
            alpha = np.zeros((h, w), bool)
            keep = self.S
            self.S = full_S.copy()
            others = np.ones(len(full_S), bool)
            others[sel] = False
            self.S[others, 3] = 0.0
            zb = self._draw_spheres(idx, alpha, x0, y0, 99.0, grow=False, sway=0.0, sway_t=0.0, want_z=True)
            self.S = keep
            cxm = float(np.mean(Sx[:, 0]))
            cym = float(np.mean(Sx[:, 1]))
            self.baked.append((mi, x0, y0, idx, alpha, np.where(alpha, zb, -1e9), cxm, cym))

    def draw_baked(self, idx, alpha, ox, oy, rustle=0.0, rustle_t=0.0):
        """Composite the baked masses by z-test (locked to the layer: no
        rigid sway that would fight the sub-pixel camera).  `rustle` sends
        travelling ripples of lighter leaves through the canopy (wind)."""
        h, w = idx.shape
        wm = self.wood_t < 90
        _blit_mask(idx, alpha, self.wood_i, wm, self.x0 - ox, self.y0 - oy)
        zbuf = np.full((h, w), -1e9)
        mid = np.full((h, w), -1, np.int32)
        for (mi, x0, y0, si, sa, sz, cxm, cym) in self.baked:
            px = x0 - ox
            py = y0 - oy
            sh_, sw_ = si.shape
            X0, Y0 = max(px, 0), max(py, 0)
            X1, Y1 = min(px + sw_, w), min(py + sh_, h)
            if X0 >= X1 or Y0 >= Y1:
                continue
            sl = (slice(Y0 - py, Y1 - py), slice(X0 - px, X1 - px))
            z = sz[sl]
            zs = zbuf[Y0:Y1, X0:X1]
            upd = z > zs
            zs[upd] = z[upd]
            idx[Y0:Y1, X0:X1][upd] = si[sl][upd]
            mid[Y0:Y1, X0:X1][upd] = mi
        m = mid >= 0
        alpha |= m
        zp = np.pad(zbuf, ((2, 0), (2, 0)), constant_values=-1e9)[:-2, :-2]
        mp = np.pad(mid, ((2, 0), (2, 0)), constant_values=-1)[:-2, :-2]
        shd = m & (mp >= 0) & (mp != mid) & ((zp - zbuf) > 2.2)
        if shd.any():
            idx[shd] = RAMP[-1][idx[shd]]
        if rustle > 0 and m.any():
            self.rustle(idx, m, ox, oy, rustle, rustle_t)
        return m

    def rustle(self, idx, m, ox, oy, amount, t):
        """Wind turning the leaves: soft bands of lighter foliage travel
        across the canopy (world-locked, sub-pixel-free)."""
        h, w = idx.shape
        yy, xx = np.nonzero(m)
        X = xx + ox
        Y = yy + oy
        n = TEX_WIND[np.floor(Y * 1.5).astype(np.int64) % _TN,
                     np.floor(X * 1.1 - Y * 0.4 - 24.0 * t).astype(np.int64) % _TN]
        r = np.clip((n - 0.62) / 0.05, 0, 1) * amount
        th = bay(h, w, 4, ox, oy)[yy, xx]
        on = th < r
        if on.any():
            idx[yy[on], xx[on]] = RAMP[1][idx[yy[on], xx[on]]]

    def _blossoms(self, idx, sid, lvmap, cx, cy, sc, t, ox, oy, grow):
        """Blossom CLUSTERS: 1-4 florets each (1-4 px shapes, now and then a
        five-petal 3x3 flower), pink / light-pink / white families, lit by
        the canopy's light; they pop in white for two frames."""
        h, w = idx.shape
        Bk = self.B[:, 0].astype(int)
        bt0 = self.S[Bk, 4] + 0.42 + 0.55 * self.B[:, 3] if grow else np.full(len(Bk), -9.0)
        rr = self.S[Bk, 3] * sc[Bk]
        ok = (t >= bt0) & (rr > 1.0)
        bx = np.round(cx[Bk] + self.B[:, 1] * rr - ox).astype(int)
        by = np.round(cy[Bk] + self.B[:, 2] * rr - oy).astype(int)
        ok &= (bx >= 2) & (bx < w - 4) & (by >= 2) & (by < h - 4)
        n = np.nonzero(ok)[0]
        n = n[sid[by[n], bx[n]] == Bk[n]]
        for j in n:
            x0_, y0_ = bx[j], by[j]
            L_ = lvmap[y0_, x0_]
            v, u = self.B[j, 4], self.B[j, 3]
            fam = BLOSSOM_FAM[int(v * 7.3) % len(BLOSSOM_FAM)]
            lvl = 0 if L_ > 4.8 else (1 if L_ > 3.6 else (2 if L_ > 2.4 else 3))
            hi = IDX[fam[lvl]]
            lo = IDX[fam[min(lvl + 1, 4)]]
            if t - bt0[j] < 0.066:
                hi = lo = IDX['white1']
            pat = BLOSSOM_PATS[int(u * 97) % len(BLOSSOM_PATS)]
            for f_, (fx_, fy_) in enumerate(pat):
                q = int((v * 31 + f_ * 7) * 13) % 100
                if q < 9 and lvl < 3:
                    shape = FIVE_PETAL
                else:
                    shape = FLORETS[q % len(FLORETS)]
                for px_, py_, part in shape:
                    xx, yy = x0_ + fx_ + px_, y0_ + fy_ + py_
                    if 0 <= xx < w and 0 <= yy < h and sid[yy, xx] >= 0:
                        if part == 'c':
                            idx[yy, xx] = IDX['gold3' if lvl < 2 else 'bloom3']
                        else:
                            idx[yy, xx] = hi if part == 'h' else lo


# blossom colour families: lit -> shade, 5 steps
BLOSSOM_FAM = [
    ('bloom5', 'bloom4', 'bloom3', 'bloom2', 'bloom1'),      # pink
    ('white0', 'bloom5', 'bloom4', 'bloom3', 'bloom2'),      # light pink
    ('white1', 'white0', 'bloom5', 'bloom4', 'bloom3'),      # white
    ('bloom5', 'bloom4', 'bloom3', 'bloom2', 'bloom1'),      # pink (more often)
]
# floret shapes (h = light pixel, l = body)
FLORETS = [
    ((0, 0, 'h'),),
    ((0, 0, 'h'), (1, 0, 'l')),
    ((0, 0, 'h'), (0, 1, 'l')),
    ((0, 0, 'h'), (1, 0, 'l'), (0, 1, 'l')),
    ((0, 0, 'h'), (1, 0, 'h'), (0, 1, 'l'), (1, 1, 'l')),
    ((0, 0, 'h'), (1, 1, 'l')),
    ((1, 0, 'h'), (0, 1, 'h'), (1, 1, 'l')),
]
# a five-petal blossom seen face-on (3x3), gold heart
FIVE_PETAL = ((0, -1, 'h'), (-1, 0, 'h'), (1, 0, 'h'), (-1, 1, 'l'), (1, 1, 'l'), (0, 0, 'c'))
# where the florets of one cluster sit (relative to its anchor)
BLOSSOM_PATS = [
    ((0, 0),),
    ((0, 0), (2, 1)),
    ((0, 0), (-2, 1), (1, 2)),
    ((0, 0), (2, -1), (1, 2), (-2, 1)),
    ((0, 0), (3, 1)),
    ((0, 0), (1, 2), (3, 0)),
]


# ---------------------------------------------------------------- tree design
# Foliage MASSES (cx, cy, rx, ry) each fed by a limb from the crown.  Gaps
# between the masses show sky and limbs — a designed silhouette, not a blob.
MASSES_S8 = [
    # (cx, cy, rx, ry) rest-screen.  Framed so the whole crown - including
    # the burst's overshoot - stays inside the frame; the crane then leaves
    # generous sky above it.
    (153, 29, 45, 13),      # A crown top
    (105, 34, 30, 10),      # B upper left
    (202, 32, 31, 11),      # C upper right
    (74, 47, 23, 7),        # D lower-left pad
    (232, 50, 27, 8),       # E lower-right, drooping
    (158, 47, 27, 7),       # F low centre, around the crown
    (126, 46, 17, 6),       # G small, left of the crown
]
TRUNK_T0, TRUNK_D = -0.125, 0.36   # trunk tip pierces the apex before frame 0, crown at ~0.235
BURST_T0 = 0.10          # foliage bud appears on the rising tip, bursts at the crown
BURST_D = 0.62
LIMB_T = [(0.17, 0.52), (0.17, 0.52), (0.17, 0.52), (0.18, 0.55), (0.18, 0.55), (0.16, 0.40), (0.17, 0.44)]
COLLAR = [
    ([(146, 75), (139, 77), (133, 82), (129, 89)], 3.4, 1.0, 0.05, 0.32),
    ([(155, 75), (162, 77), (168, 83), (171, 91)], 3.4, 1.0, 0.06, 0.32),
    ([(148, 76), (144, 81), (142, 87)], 2.4, 1.0, 0.09, 0.25),
    ([(153, 76), (157, 82), (158, 88)], 2.4, 1.0, 0.10, 0.25),
]


def trunk_tip_y(t):
    """Height of the rising trunk tip (rest y) at S8b time t."""
    u = np.clip((t - TRUNK_T0) / TRUNK_D, 0, 1)
    sfr = 1 - (1 - u) ** 3
    y0, y1 = TRUNK_PTS[0][1], TRUNK_PTS[-1][1]
    return y0 + (y1 - y0) * sfr


def limb_point(pts, s):
    xs, ys, arc = _resample(pts)
    a = s * arc[-1]
    return float(np.interp(a, arc, xs)), float(np.interp(a, arc, ys))


def _limb_to(mass, crown, k):
    """A curving limb from the crown into a foliage mass (ends inside it)."""
    cx, cy, rx, ry = mass
    x0, y0 = crown
    ex, ey = cx + (x0 - cx) * 0.25, cy + ry * 0.35
    # bow upward like an umbrella rib, then out
    mx = x0 + (ex - x0) * 0.45
    my = y0 + (ey - y0) * 0.45 - abs(ex - x0) * 0.10 - 3
    q1 = (x0 + (mx - x0) * 0.5, y0 + (my - y0) * 0.5 + 1)
    return [(x0, y0), q1, (mx, my), (ex, ey)]


def tree_parts(masses, limb_t, crown=CROWN, trunk=None, fill=1.0, rbase=(8.0, 5.5), seed=71,
               grow=True, twig=True, width=1.0):
    trunk = trunk or (TRUNK_PTS, 13.5, 9.5, TRUNK_T0, TRUNK_D, 2.3)
    strokes = [trunk]
    clumps = []
    for k, (mass, (st, du)) in enumerate(zip(masses, limb_t)):
        pts = _limb_to(mass, (crown[0] + (mass[0] - crown[0]) * 0.02, crown[1] + 1), k)
        L_ = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
        w0 = (3.2 + min(L_, 70) / 22.0) * width
        strokes.append((pts, w0, 1.6 * width, st if grow else -9, du if grow else 1, 1.0))
        # a side twig or two (seen in the gaps)
        if twig and L_ > 30:
            for j, sgn in ((0.55, -1), (0.75, 1)):
                px, py = limb_point(pts, j)
                tx_ = px + (mass[0] - px) * 0.3 + sgn * 4
                ty_ = py - 7 - 4 * rnd(k, int(j * 10), seed)
                strokes.append(([(px, py), ((px + tx_) / 2 + sgn, (py + ty_) / 2), (tx_, ty_)],
                                1.8 * width, 1.0, (st + du * 0.5) if grow else -9, 0.2 if grow else 1, 0.6))
        # fill the mass with clumps (jittered grid, denser toward its top)
        cx, cy, rx, ry = mass
        n = int((rx * ry) / 70.0 * fill) + 4
        placed = []
        tries = 0
        while len(placed) < n and tries < 600:
            tries += 1
            u = rnd(tries, 1, seed + k * 13) * 2 - 1
            v = rnd(tries, 2, seed + k * 13) * 2 - 1
            if u * u + v * v > 1:
                continue
            x = cx + u * rx * 0.84
            y = cy + v * ry * 0.72
            R = rbase[0] + rbase[1] * rnd(tries, 3, seed + k * 13)
            R *= 1.0 - 0.25 * (u * u + v * v)
            if any(math.hypot(x - px_, (y - py_) * 1.3) < (R + pr) * 0.40 for px_, py_, pr in placed):
                continue
            placed.append((x, y, R))
        ltot = max(1.0, math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]) + max(rx, ry) * 0.6)
        for (x, y, R) in placed:
            # the whole canopy bursts out of the crown as one inflating mass:
            # host = crown, small stagger by distance (inner clumps first)
            dc = math.hypot(x - crown[0], (y - crown[1]) * 1.2)
            f = min(dc / ltot, 1.0)
            birth = (BURST_T0 + 0.07 * f + 0.02 * rnd(int(x * 7), int(y * 3), seed)) if grow else -9.0
            clumps.append((x, y, R, birth, crown[0], crown[1] - 3, k))
    return strokes, clumps


def s8_tree_parts():
    strokes, clumps = tree_parts(MASSES_S8, LIMB_T)
    for pts, w0, w1, st, du in COLLAR:
        strokes.append((pts, w0, w1, st, du, 0.8))
    return strokes, clumps


S8_ENV = (156.0, 33.0, 94.0, 24.0)


# =====================================================================
#  PER-FRAME COMPOSITING HELPERS
# =====================================================================
def window(L, sx, sy, pad=2):
    """Visible window (i0, i1, j0, j1) of layer L when its rest-screen origin
    sits at screen offset (sx, sy)."""
    j0 = int(math.floor(-L.x0 - sx)) - pad
    j1 = int(math.ceil(W - L.x0 - sx)) + pad
    i0 = int(math.floor(-L.y0 - sy)) - pad
    i1 = int(math.ceil(H - L.y0 - sy)) + pad
    j0, j1 = max(j0, 0), min(j1, L.w)
    i0, i1 = max(i0, 0), min(i1, L.h)
    if j0 >= j1 or i0 >= i1:
        return None
    return i0, i1, j0, j1


def comp_window(L, t, win, edge_w=0.07):
    i0, i1, j0, j1 = win
    T = L.T[i0:i1, j0:j1]
    b = bay(i1 - i0, j1 - j0, 4, j0, i0)
    al = (t - T) > (b - 0.5) * edge_w
    idx = np.where(al, L.alive.i[i0:i1, j0:j1], L.dead.i[i0:i1, j0:j1])
    a = np.where(al, L.alive.a[i0:i1, j0:j1], L.dead.a[i0:i1, j0:j1])
    return idx, a, al, T


def star(idx, x, y, core, arm, big=False):
    h, w = idx.shape
    x, y = int(x), int(y)
    if 0 <= x < w and 0 <= y < h:
        idx[y, x] = IDX[core]
    arms = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if big:
        arms = arms + ((2, 0), (-2, 0), (0, 2), (0, -2))
    for dx, dy in arms:
        xx, yy = x + dx, y + dy
        if 0 <= xx < w and 0 <= yy < h:
            idx[yy, xx] = IDX[arm]


def plot(idx, alpha, xs, ys, name):
    h, w = idx.shape
    xs = np.floor(np.asarray(xs, np.float64)).astype(int).ravel()
    ys = np.floor(np.asarray(ys, np.float64)).astype(int).ravel()
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    idx[ys[ok], xs[ok]] = IDX[name]
    if alpha is not None:
        alpha[ys[ok], xs[ok]] = True


# Hand-drawn bird sprites: (dx, dy, part) with part w = wing, b = body,
# u = lit underside.  3 wing frames: up / level / down (cycle up-mid-down-mid).
BIRD5 = [
    [(-2, -2, 'w'), (2, -2, 'w'), (-1, -1, 'w'), (1, -1, 'w'), (0, -1, 'b'), (0, 0, 'u')],
    [(-2, -1, 'w'), (-1, -1, 'w'), (0, -1, 'b'), (1, -1, 'w'), (2, -1, 'w'), (0, 0, 'u')],
    [(-1, -1, 'w'), (0, -1, 'b'), (1, -1, 'w'), (-2, 0, 'w'), (2, 0, 'w'), (0, 0, 'u')],
]
BIRD4 = [
    [(-2, -2, 'w'), (1, -2, 'w'), (-1, -1, 'w'), (0, -1, 'b'), (0, 0, 'u')],
    [(-2, -1, 'w'), (-1, -1, 'w'), (0, -1, 'b'), (1, -1, 'w'), (0, 0, 'u')],
    [(-1, -1, 'w'), (0, -1, 'b'), (-2, 0, 'w'), (1, 0, 'w'), (0, 0, 'u')],
]
BIRD3 = [
    [(-1, -1, 'w'), (1, -1, 'w'), (0, 0, 'b')],
    [(-1, 0, 'w'), (0, 0, 'b'), (1, 0, 'w')],
    [(0, 0, 'b'), (-1, 1, 'w'), (1, 1, 'w')],
]
BIRD_CYCLE = (0, 1, 2, 1)
BIRD_COLS = {'day': {'w': 'ash2', 'b': 'ink2', 'u': 'ash6'},
             'far': {'w': 'sky3', 'b': 'sky2', 'u': 'sky5'},
             'morning': {'w': 'ash2', 'b': 'ink2', 'u': 'dust8'},
             'morning_far': {'w': 'ash4', 'b': 'ash3', 'u': 'bloom4'}}


def draw_bird(idx, x, y, t, phase=0.0, size=5, cols='day', flap_hz=7.0, glide=False, alpha=None):
    """One bird at (x, y) (window px).  Flaps up-mid-down-mid at flap_hz,
    or glides (wings level) when glide=True."""
    spr = {5: BIRD5, 4: BIRD4, 3: BIRD3}[size]
    fr = 1 if glide else BIRD_CYCLE[int(t * flap_hz * 4 + phase * 4) % 4]
    cmap = BIRD_COLS[cols]
    h, w = idx.shape
    xi, yi = int(math.floor(x)), int(math.floor(y))
    for dx, dy, part in spr[fr]:
        xx, yy = xi + dx, yi + dy
        if 0 <= xx < w and 0 <= yy < h:
            idx[yy, xx] = IDX[cmap[part]]
            if alpha is not None:
                alpha[yy, xx] = True


def draw_flock(idx, birds, t, ox, oy, cols='day', alpha=None):
    """birds: list of dicts with x0, y0, vx, vy, t0, size, ph, bob, hz.
    Gentle bobbing, individual wingbeats, occasional glides."""
    for b in birds:
        a = t - b['t0']
        if a < 0:
            continue
        x = b['x0'] + b['vx'] * a + b.get('ax', 0.0) * a * a
        y = b['y0'] + b['vy'] * a + b['bob'] * math.sin(TAU * 0.55 * a + b['ph'] * 6)
        glide = (int(a * 0.9 + b['ph'] * 3) % 3) == 2
        far = b.get('far', b['size'] <= 3)
        cm = (cols + '_far' if cols + '_far' in BIRD_COLS else 'far') if far else cols
        draw_bird(idx, x - ox, y - oy, a, b['ph'], b['size'], cm, b['hz'], glide, alpha)


def make_flock(n, seed, x0, y0, spread, vx, vy, t0, size_mix=(5, 4), dt0=0.35, hz=(6.0, 8.5), far=False):
    birds = []
    for k in range(n):
        birds.append(dict(
            x0=x0 + (rnd(k, 1, seed) - 0.5) * spread[0] - k * 2.5,
            y0=y0 + (rnd(k, 2, seed) - 0.5) * spread[1],
            vx=vx * (0.85 + 0.3 * rnd(k, 3, seed)),
            vy=vy * (0.7 + 0.6 * rnd(k, 4, seed)),
            t0=t0 + rnd(k, 5, seed) * dt0,
            size=size_mix[int(rnd(k, 6, seed) * len(size_mix)) % len(size_mix)],
            ph=rnd(k, 7, seed), bob=0.8 + 1.2 * rnd(k, 8, seed),
            hz=hz[0] + (hz[1] - hz[0]) * rnd(k, 9, seed), far=far))
    return birds



# =====================================================================
#  SHARED WORLD CACHE (S8b and S9a live in the same render processes)
# =====================================================================
_CACHE = {}


def get_world():
    if 'world' not in _CACHE:
        wd = build_world()
        dome = paint_dome(wd['hill'])
        _CACHE['world'] = wd
        _CACHE['dome'] = dome
    return _CACHE['world'], _CACHE['dome']


DRIFTS_V3_S9 = [
    (110, 140, 40, 6, 1, 2, 0.30, 2), (60, 156, 40, 8, 2, 3, 0.26, 6), (170, 150, 26, 6, 5, 2, 0.28, 0),
    (220, 160, 34, 8, 3, 3, 0.24, 7), (300, 146, 34, 6, 0, 2, 0.30, 1), (150, 172, 30, 7, 4, 3, 0.2, 5),
    (20, 150, 26, 6, 3, 2, 0.28, 7), (270, 178, 30, 8, 1, 3, 0.2, 2),
]
DRIFTS_HILL_S9 = [
    (96, 122, 16, 4, 2, 2, 0.4, 6), (206, 124, 16, 4, 3, 2, 0.4, 7), (130, 128, 12, 4, 0, 2, 0.35, 5),
    (176, 130, 14, 4, 1, 2, 0.35, 2), (84, 132, 12, 4, 5, 2, 0.35, 0), (222, 134, 12, 4, 4, 2, 0.3, 0),
]


def get_hill_s9():
    """A copy of the hill layer with the overgrown S9a dome, in full bloom."""
    if 'hill9' not in _CACHE:
        wd, _ = get_world()
        H0 = build_hill()
        paint_dome(H0, s9=True)
        m = H0.alive.a & (H0.Y > hill_top(H0.X[0])[None, :] + 2) & (H0.flag == 0)
        m &= ~dome_masks(H0.alive)['shell']
        flower_drifts(H0.alive, m, DRIFTS_HILL_S9, 1150)
        _CACHE['hill9'] = H0
    return _CACHE['hill9']


def get_v3_s9():
    """The near valley, seasons later: far more flowers."""
    if 'v3_9' not in _CACHE:
        L = build_v3()
        m = L.alive.a & (L.flag == 0) & (L.Y > v3_top(L.X[0])[None, :] + 1)
        flower_drifts(L.alive, m, DRIFTS_V3_S9, 1160)
        _CACHE['v3_9'] = L
    return _CACHE['v3_9']


# ---------------------------------------------------------------- S9a tree
MASSES_S9 = [
    # seasons later: a vaster umbrella, still fully framed in S9a
    (152, 12, 60, 16),      # crown top
    (92, 22, 39, 13),       # upper left
    (214, 19, 41, 14),      # upper right
    (57, 40, 27, 9),        # lower-left pad
    (251, 42, 31, 10),      # lower-right, drooping
    (158, 40, 34, 9),       # low centre
    (117, 39, 21, 8),       # left of crown
    (191, 39, 20, 7),       # right of crown
]


def s9_tree_parts():
    """Seasons later: thicker trunk, longer limbs, a vast umbrella."""
    trunk = ([(150, 113), (151.5, 99), (149, 86), (147.5, 74), (150.5, 64), (151, 55)], 17.0, 12.5, -9, 1, 2.3)
    strokes, clumps = tree_parts(MASSES_S9, [(0, 1)] * len(MASSES_S9), crown=(151.0, 55.0), trunk=trunk,
                                 fill=1.05, rbase=(9.0, 6.0), seed=93, grow=False, width=1.35)
    for pts, w0, w1, st, du in COLLAR:
        strokes.append((pts, w0 + 1.0, w1 + 0.3, -9, 1, 0.8))
    for pts in ([(145, 76), (136, 80), (128, 88), (125, 97), (124, 106)],
                [(156, 76), (165, 80), (173, 89), (177, 99), (178, 108)],
                [(150, 77), (149, 86), (147, 95)]):
        strokes.append((pts, 3.0, 1.2, -9, 1, 0.8))
    env = (153.0, 26.0, 122.0, 30.0)
    return strokes, clumps, env


# =====================================================================
#  CAMERA (shared: S8b clock; S9a uses T = 5 + t so the move flows
#  continuously through the dissolve)
#
#  The path is authored FRAME-EXACT in output sub-pixel units (1/3 low-res
#  px = the Stage's hi-res step), so the hill layer (p = 1, the subject)
#  advances on EVERY frame once the move starts — never a stalled frame:
#    vertical crane  frames 90..168 (3.0 -> 5.6 s): 1 -> 1.25 -> 1 steps/frame
#    lateral drift   frames 150..   (5.0 s -> end of S9a): 1 step/frame
# =====================================================================
QSTEP = 1.0 / 3.0
_NF_CAM = 30 * 12


def _build_cam():
    yq = np.zeros(_NF_CAM + 1)
    xq = np.zeros(_NF_CAM + 1)
    acc = 0.0
    y = 0
    for k in range(91, 169):                     # moves between frame k-1 and k
        u = (k - 91) / 77.0
        rate = 1.0 + 0.25 * math.sin(math.pi * u) ** 2
        acc += rate
        stp = int(acc + 1e-9)
        acc -= stp
        y += stp
        yq[k] = y
    yq[169:] = y
    x = 0
    for k in range(151, _NF_CAM + 1):
        x += 1
        xq[k] = x
    return yq * QSTEP, xq * QSTEP


_CAM_Y, _CAM_X = _build_cam()
_CAM_K = np.arange(_NF_CAM + 1)


def camera(T):
    """(cam_x, cam_y) in hill-layer px (p = 1).  Exact 1/3-px values on
    frame times; linear in between (for off-frame previews)."""
    k = max(0.0, float(T) * 30.0)
    kr = round(k)
    if abs(k - kr) < 1e-6:
        k = kr
    return float(np.interp(k, _CAM_K, _CAM_X)), float(np.interp(k, _CAM_K, _CAM_Y))


# =====================================================================
#  SCREEN-SPACE MAPS (computed once per frame, sliced per layer window)
# =====================================================================
MARG = 10
SYS, SXS = np.mgrid[-MARG:H + MARG, -MARG:W + MARG].astype(np.float64)


def smap(M, ox_s, oy_s, h, w):
    """Slice screen-space map M for a window whose top-left pixel is at
    screen (ox_s, oy_s)."""
    px = int(round(ox_s)) + MARG
    py = int(round(oy_s)) + MARG
    Hm, Wm = M.shape
    if px >= 0 and py >= 0 and px + w <= Wm and py + h <= Hm:
        return M[py:py + h, px:px + w]
    out = np.zeros((h, w), M.dtype)
    X0, Y0 = max(px, 0), max(py, 0)
    X1, Y1 = min(px + w, Wm), min(py + h, Hm)
    if X0 < X1 and Y0 < Y1:
        out[Y0 - py:Y1 - py, X0 - px:X1 - px] = M[Y0:Y1, X0:X1]
    return out


_PG = 260          # polar grid half-size
_py, _px = np.mgrid[-_PG:_PG + H + 2 * MARG, -_PG:_PG + W + 2 * MARG].astype(np.float32)
_PANG = np.arctan2(_py, _px).astype(np.float32)
_PDIST = np.sqrt(_px * _px + _py * _py).astype(np.float32)
del _py, _px
_NB = 4096


def ray_map(sx, sy, angs, wids, amps, length):
    """Ray level map (screen + margin grid) radiating from (sx, sy):
    the angular profile is built in 1-D and looked up per pixel."""
    ix = int(round(sx)) + MARG
    iy = int(round(sy)) + MARG
    hh, ww = H + 2 * MARG, W + 2 * MARG
    x0 = _PG - ix
    y0 = _PG - iy
    x0 = int(np.clip(x0, 0, _PANG.shape[1] - ww))
    y0 = int(np.clip(y0, 0, _PANG.shape[0] - hh))
    ang = _PANG[y0:y0 + hh, x0:x0 + ww]
    dist = _PDIST[y0:y0 + hh, x0:x0 + ww]
    grid = np.linspace(-math.pi, math.pi, _NB, endpoint=False)
    prof = np.zeros(_NB, np.float32)
    for a, wd, am in zip(angs, wids, amps):
        da = np.abs((grid - a + math.pi) % TAU - math.pi)
        prof = np.maximum(prof, np.clip(1 - da / wd, 0, 1) * am)
    bi = ((ang + math.pi) * (_NB / TAU)).astype(np.int32) % _NB
    rl = prof[bi]
    fall = np.clip(1 - dist / length, 0, 1) * np.clip(dist / 14.0, 0, 1)
    return rl * fall, dist, ang


# scrolling textures (world-locked, drifting with the wind)
_TN = 256
_ty, _tx = np.mgrid[0:_TN, 0:_TN].astype(np.float64)
TEX_CLOUDSH = noise.fbm2(_tx / 24.0, _ty / 9.0, seed=881, octaves=3)
TEX_WIND = noise.fbm2(_tx / 13.0, _ty / 6.0, seed=882, octaves=2)
TEX_MIST = noise.fbm2(_tx / 30.0, _ty / 4.0, seed=883, octaves=2)
del _ty, _tx


def tex_sample(tex, X, Y, dx=0.0, sx=1.0, sy=1.0, shear=0.0):
    xi = np.floor(X * sx + Y * shear - dx).astype(np.int64) % _TN
    yi = np.floor(Y * sy).astype(np.int64) % _TN
    return tex[yi, xi]


# =====================================================================
#  VINES (tendrils climbing the dome), baked into the alive dome
# =====================================================================
def vine_paths(n, seed, reach, s9=False):
    paths = []
    for k in range(n):
        u = (k + 0.5) / n
        th = (u - 0.5) * 2.6 + (rnd(k, 1, seed) - 0.5) * 0.18
        phi = 0.0
        pts = []
        wig = (rnd(k, 2, seed) - 0.5) * 0.02
        length = reach * (0.45 + 0.75 * rnd(k, 3, seed))
        s = 0.0
        x, y = _dome_proj(th, phi)
        pts.append((float(x), float(y) + 3))
        while s < length:
            phi += 0.028
            th += wig + 0.06 * math.sin(s * 0.35 + k)
            x, y = _dome_proj(th, phi)
            pts.append((float(x), float(y)))
            s += 1.0
            if phi > 1.15:
                break
        paths.append(pts)
    return paths


def paint_vines(Pn, T, paths, seed, T0, flowers=0.12, leafy=0.55, dense=False):
    """1-px stems with alternating leaves (and a few flowers)."""
    for k, pts in enumerate(paths):
        prev = None
        for j, (x, y) in enumerate(pts):
            xi, yi = int(round(x)), int(round(y))
            if prev == (xi, yi):
                continue
            prev = (xi, yi)
            tt = T0 + j * 0.028
            Pn.px(xi, yi, 'leaf2')
            _setT(Pn, T, xi, yi, tt)
            if rnd(k * 131 + j, 1, seed) < leafy:
                side = 1 if (j // 2) % 2 == 0 else -1
                c = 'leaf4' if side < 0 else 'leaf3'
                Pn.px(xi + side, yi, c)
                _setT(Pn, T, xi + side, yi, tt + 0.03)
                if dense and rnd(k * 131 + j, 3, seed) < 0.5:
                    Pn.px(xi + side, yi - 1, 'leaf5' if side < 0 else 'leaf4')
                    _setT(Pn, T, xi + side, yi - 1, tt + 0.04)
            if rnd(k * 131 + j, 2, seed) < flowers:
                c = 'bloom4' if rnd(k * 131 + j, 4, seed) < 0.6 else 'white0'
                Pn.px(xi, yi - 1, c)
                _setT(Pn, T, xi, yi - 1, tt + 0.1)


def _setT(Pn, T, x, y, v):
    j, i = int(round(x - Pn.x0)), int(round(y - Pn.y0))
    if 0 <= i < Pn.h and 0 <= j < Pn.w:
        T[i, j] = v


# =====================================================================
#  MEADOW v3: composed flower drifts + grass tufts
# =====================================================================
TUFTS = [
    ((0, 0), (1, -1), (2, 0)),
    ((0, 0), (0, -1), (1, -2), (2, -1), (2, 0)),
    ((0, 0), (1, -1)),
    ((0, 0), (-1, -1)),
    ((0, 0), (0, -1)),
    ((0, 0), (1, -1), (1, -2)),
]


def grass_tufts(Pn, mask, lvq, seed, density, big=0.0, dark='meadow2', light_c='meadow4',
                darker='meadow1', lighter='meadow5'):
    """Small grass-blade stamps: darker blades on lit grass, lighter blades
    in shade (never noise: each stamp is a tiny drawn tuft)."""
    h, w = mask.shape
    r = rndxy(Pn.X * 5 + 3, Pn.Y, seed)
    pick = mask & (r < density)
    ys, xs = np.nonzero(pick)
    for n in range(len(ys)):
        i, j = ys[n], xs[n]
        v = rnd(n, 1, seed)
        k = int(v * (3 if big <= 0 else 6)) if rnd(n, 2, seed) > big else 1
        st = TUFTS[min(k, len(TUFTS) - 1)]
        lit = lvq[i, j] >= 2
        c = IDX[dark] if lit else IDX[lighter if lvq[i, j] <= 0 else light_c]
        if lit and lvq[i, j] >= 3:
            c = IDX['meadow3']
        for dx, dy in st:
            ii, jj = i + dy, j + dx
            if 0 <= ii < h and 0 <= jj < w and mask[ii, jj]:
                Pn.i[ii, jj] = c


def flower_drifts(Pn, mask, drifts, seed, T=None, flag=None, delay=(0.05, 0.4)):
    """drifts: list of (cx, cy, rx, ry, kind_index, size, density[, accent]).
    Dense at the centre, thinning at a noisy edge; one dominant kind."""
    h, w = mask.shape
    X, Y = Pn.X, Pn.Y
    edge_n = noise.fbm2(X / 9.0, Y / 4.0, seed=seed + 1, octaves=2)
    for d_i, dr in enumerate(drifts):
        cx, cy, rx, ry, ki, size, dens = dr[:7]
        accent = dr[7] if len(dr) > 7 else None
        e = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2
        f = np.clip(1.15 - e - 0.55 * (edge_n - 0.5) * 2, 0, 1)
        prob = dens * f ** 1.3
        r = rndxy(X * 7 + d_i * 131, Y * 3, seed + d_i)
        pick = mask & (r < prob)
        # spacing: no touching flower centres
        pick &= ~shift(pick, 1, 0) & ~shift(pick, 0, 1) & ~shift(pick, 1, 1) & ~shift(pick, -1, 1)
        ys, xs = np.nonzero(pick)
        order = np.argsort(ys, kind='stable')
        for n in order:
            i, j = ys[n], xs[n]
            kk = ki
            if accent is not None and rnd(n, 3, seed + d_i) < 0.14:
                kk = accent
            sz = size if rnd(n, 4, seed + d_i) > 0.3 else max(1, size - 1)
            spr = flower_sprite(FLOWER_KINDS[kk], sz)
            dl = delay[0] + (delay[1] - delay[0]) * rnd(n, 11, seed + d_i)
            if sz >= 3:
                for k2 in (1, 2):
                    if i + 1 + k2 < h and mask[i + 1 + k2, j]:
                        Pn.i[i + 1 + k2, j] = IDX['leaf2' if k2 > 1 else 'leaf3']
            for dy, dx, c in spr:
                ii, jj = i + dy, j + dx
                if 0 <= ii < h and 0 <= jj < w:
                    Pn.i[ii, jj] = IDX[c]
                    Pn.a[ii, jj] = True
                    if T is not None and T[i, j] < 90:
                        T[ii, jj] = T[i, j] + dl
                    if flag is not None:
                        flag[ii, jj] = 1


# kinds: 0 pink, 1 white daisy, 2 yellow, 3 violet, 4 red poppy, 5 rose, 6 white-gold, 7 deep violet
DRIFTS_V3 = [
    (84, 146, 44, 7, 2, 2, 0.30, 6), (40, 170, 34, 9, 4, 3, 0.20, 0), (236, 144, 40, 6, 3, 2, 0.30, 7),
    (292, 160, 36, 10, 3, 3, 0.18, 0), (190, 176, 26, 8, 1, 3, 0.18, 2), (132, 162, 22, 6, 0, 3, 0.20, 1),
    (254, 172, 24, 7, 2, 3, 0.18, 4), (18, 136, 30, 5, 1, 2, 0.26, 2), (330, 138, 30, 5, 2, 2, 0.26, 6),
]
DRIFTS_HILL = [
    (118, 124, 20, 5, 1, 2, 0.35, 2), (183, 125, 18, 5, 0, 2, 0.35, 5), (150, 131, 14, 4, 2, 2, 0.3, 6),
    (100, 133, 12, 4, 4, 2, 0.3, 0), (205, 134, 14, 4, 3, 2, 0.3, 7),
]
DRIFTS_V2 = [
    (40, 108, 50, 4, 2, 1, 0.45, 6), (250, 107, 60, 4, 3, 1, 0.45, 7), (130, 112, 30, 3, 0, 1, 0.4, 1),
    (320, 110, 30, 3, 2, 1, 0.4, 4),
]
