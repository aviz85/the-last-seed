"""
shots/lib_discovery.py — shared environment + helpers for S2 (THE EMPTY CAN)
and S3 (DISCOVERY).  Owner: DISCOVERY.

Everything here is either built once at import time (static art, cached) or
is a pure function of its arguments (per-frame FX), so render(t) stays pure.
"""
import math
import numpy as np
from scipy.spatial import Voronoi
from scipy import ndimage

from engine import W, H, C, RAMPS, canvas, layer, paste, bayer, dither_mask
from engine import draw, noise, light
from engine.sprite import from_ascii, blit, Sprite

# ------------------------------------------------------------------ colours
DUST = ['dust%d' % i for i in range(11)]
ASH = ['ash%d' % i for i in range(9)]


def pal_array(names):
    return np.array([C(n) for n in names], np.uint8)


DUST_RGB = pal_array(DUST)


def R(i, *keys, seed=0):
    """Vectorised deterministic random in [0,1) for integer index arrays."""
    x = np.asarray(i, np.int64)
    k = 0
    for q in keys:
        k = k * 131 + int(q) + 1
    return noise.rand(x, k, seed)


# ------------------------------------------------------------------ ground geometry
class GroundGeo:
    """Perspective-ish mapping of the ground plane.

    d = y - hy (rows below the horizon).  Plates of equal 'world' size get
    wider (~d^pw) and taller (~d) toward the camera.  (u, v) is the plate
    space in which the crack network is a jittered Voronoi diagram.
    """

    def __init__(self, hy, cx=160.0, cell_u=0.80, pw=0.83, cv=3.6):
        self.hy, self.cx, self.cu, self.pw, self.cv = hy, cx, cell_u, pw, cv

    def uv(self, x, y):
        d = np.maximum(np.asarray(y, float) - self.hy, 0.3)
        return self.cu * (np.asarray(x, float) - self.cx) / d ** self.pw, self.cv * np.log(d)

    def xy(self, u, v):
        d = np.exp(np.asarray(v, float) / self.cv)
        return self.cx + np.asarray(u, float) * d ** self.pw / self.cu, self.hy + d


def _seeds(geo, seed, w, h, dmin, jitter, drop):
    hy = geo.hy
    vmin = geo.cv * math.log(max(dmin, 0.5)) - 2.0
    vmax = geo.cv * math.log(h - hy + 3) + 2.0
    P = []
    for j in range(int(math.floor(vmin)), int(math.ceil(vmax)) + 1):
        dlo = math.exp(j / geo.cv)
        umax = geo.cu * (max(geo.cx, w - geo.cx) + 30) / max(dlo, 0.5) ** geo.pw
        n = int(math.ceil(umax)) + 2
        i = np.arange(-n, n + 1)
        key = (j + 500) * 4096 + (i + 2048)
        ju = (R(key, 1, seed=seed) - 0.5) * 2 * jitter
        jv = (R(key, 2, seed=seed) - 0.5) * 2 * jitter
        keep = R(key, 3, seed=seed) >= drop
        P.append(np.stack([i + 0.5 * (j & 1) + ju, j + 0.5 + jv], -1)[keep])
    return np.concatenate(P)


def crack_network(geo, seed, w=W, h=H, dmin=4.0, jitter=0.40, wob=0.5, drop=0.18, hair=0.45):
    """Rasterise a jittered-Voronoi mud-crack network.
    Returns an int map: 0 none, 1 primary crack, 2 secondary hairline."""
    hy = geo.hy
    pts = _seeds(geo, seed, w, h, dmin, jitter, drop)
    vor = Voronoi(pts)
    V = vor.vertices
    RV = np.array([rv for rv in vor.ridge_vertices if rv[0] >= 0 and rv[1] >= 0])
    order = np.zeros((h, w), np.uint8)
    VX, VY = geo.xy(V[:, 0], V[:, 1])
    rr = np.arange(len(RV))
    wobr = R(rr, 4, seed=seed)
    for k, (a, b) in enumerate(RV):
        xa, ya, xb, yb = VX[a], VY[a], VX[b], VY[b]
        if max(ya, yb) < hy + dmin or min(ya, yb) > h + 2:
            continue
        if max(xa, xb) < -4 or min(xa, xb) > w + 4:
            continue
        # far away: keep only the flatter cracks (vertical ones bunch up at the vanishing point)
        dd = 0.5 * (ya + yb) - hy
        if dd < 10 and abs(yb - ya) > 0.9 * abs(xb - xa) + 0.5:
            continue
        L = math.hypot(xb - xa, yb - ya)
        if L > 160 or not np.isfinite(L):
            continue                      # hull artefacts far outside the frame
        nseg = max(1, int(L / 4.0))
        us = np.linspace(0, 1, nseg + 1)
        pa, pb = V[a], V[b]
        px, py = geo.xy(pa[0] + (pb[0] - pa[0]) * us, pa[1] + (pb[1] - pa[1]) * us)
        if nseg > 1:
            nx, ny = -(yb - ya) / L, (xb - xa) / L
            o = np.sin(us * math.pi) * (wobr[k] - 0.5) * 2 * wob * min(1.0, dd / 12.0) * min(2.0, L / 8)
            px = px + nx * o
            py = py + ny * o
        for q in range(nseg):
            xs, ys = draw.line_points(px[q], py[q], px[q + 1], py[q + 1])
            xs = np.array(xs); ys = np.array(ys)
            ok = (xs >= 0) & (xs < w) & (ys >= hy) & (ys < h)
            order[ys[ok], xs[ok]] = 1
    # secondary hairline cracks: short dead-end cracks from a vertex into a plate
    vi = np.arange(len(V))
    sel = (R(vi, 9, seed=seed) < hair) & (VY - hy > 16) & (VY < h) & (VX > -2) & (VX < w + 2)
    for q in np.nonzero(sel)[0]:
        ang = R(q, 10, seed=seed) * 2 * math.pi
        ln = 0.22 + 0.28 * R(q, 11, seed=seed)
        u1, v1 = V[q, 0] + math.cos(ang) * ln, V[q, 1] + math.sin(ang) * ln * 0.8
        x1, y1 = geo.xy(u1, v1)
        xs, ys = draw.line_points(VX[q], VY[q], x1, y1)
        for x, y in zip(xs, ys):
            if 0 <= x < w and hy <= y < h and order[y, x] == 0:
                order[y, x] = 2
    return order


def build_ground(geo, seed, w=W, h=H, near_tone=4.0, far_tone=6.4, dmin=4.0,
                 sand=0.36, sand_scale=(0.22, 0.30)):
    """Paint the cracked-earth plane (rows >= hy) into an RGB image.

    Returns dict(img, tone(float map), crack(bool), sand(bool), depth, gmask)."""
    hy = geo.hy
    order = crack_network(geo, seed, w, h, dmin=dmin)
    crack = order > 0
    ys = np.arange(h, dtype=float)[:, None] * np.ones((1, w))
    xs = np.ones((h, 1)) * np.arange(w, dtype=float)[None, :]
    d = np.maximum(ys - hy, 0.0)
    gmask = ys >= hy
    u, v = geo.uv(xs, np.maximum(ys, hy + 0.3))
    # --- wind-blown sand drifts (in plate space -> perspective-correct patches)
    sn = noise.fbm2(u * sand_scale[0] + 17.3, v * sand_scale[1] + 3.1, seed + 40, 3)
    sn = sn + 0.18 * (1 - np.clip(d / 10, 0, 1))          # more sand toward the horizon
    sandm = gmask & (sn > 1 - sand)
    sandm = ndimage.binary_opening(sandm, np.ones((1, 3)))
    sandm = ndimage.binary_closing(sandm, np.ones((2, 3)))
    # plates (4-connected regions between cracks)
    lab, nlab = ndimage.label(gmask & ~crack)
    pid = np.arange(nlab + 1)
    cy = ndimage.mean(ys, lab, pid)
    cy = np.nan_to_num(np.asarray(cy, float), nan=hy)
    pd = np.maximum(cy - hy, 0)
    fd = np.clip((pd - 2.0) / 40.0, 0, 1) ** 0.75
    pbase = far_tone + (near_tone - far_tone) * fd
    r1, r2 = R(pid, 1, seed=seed + 5), R(pid, 2, seed=seed + 5)
    pvar = np.where(r1 < 0.20, 1.0, 0.0) - np.where(r2 < 0.16, 1.0, 0.0)
    pvar = pvar * np.clip((pd - 5) / 8, 0, 1)
    ptone = np.round(pbase + pvar)
    ptone[0] = far_tone
    tone = ptone[lab].astype(float)
    # --- relief: lit lip under a crack, shaded rim over it (near plates only)
    above = np.zeros_like(crack); above[1:] = crack[:-1]
    left = np.zeros_like(crack); left[:, 1:] = crack[:, :-1]
    below = np.zeros_like(crack); below[:-1] = crack[1:]
    right = np.zeros_like(crack); right[:, :-1] = crack[:, 1:]
    lit = (above | (left & ~below)) & ~crack & (d > 7)
    shd = (below | (right & ~above)) & ~crack & (d > 18) & ~lit
    tone = tone + lit * 1.0 - shd * 1.0
    # some near plates curl more: a 2-px lit lip
    above2 = np.zeros_like(crack); above2[2:] = crack[:-2]
    curl = (R(lab, 21, seed=seed) < 0.35) & above2 & ~crack & ~lit & (d > 26)
    tone = tone + curl * 1.0
    # small pits / flakes: 2-px horizontal clusters, one or two per near plate
    pit_r = noise.rand((xs // 2).astype(np.int64), ys.astype(np.int64), seed + 71)
    pits = (pit_r < 0.012) & (d > 22) & ~crack & ~lit & ~shd
    pits = pits | np.roll(pits, 1, axis=1)
    tone = tone - pits * 1.0
    # cracks: dark near, fading into haze far; hairlines lighter
    fdp = np.clip((d - 2.0) / 40.0, 0, 1) ** 0.75
    ctone = np.where(order == 2, 2.8, 1.0) + (1 - fdp) * 3.4
    tone = np.where(crack, np.round(ctone), tone)
    # --- sand fills cracks and smooths plates
    stone_ = np.round(far_tone + 0.6 + (near_tone + 1.4 - far_tone - 0.6) * fdp)
    sedge = sandm & ~ndimage.binary_erosion(sandm, np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], bool))
    tone = np.where(sandm, stone_, tone)
    tone = np.where(sedge & (d > 6), stone_ + 1, tone)     # lit crest on the drift's far edge
    # soft transition: cracks right at a drift edge become shallower
    near_sand = ndimage.binary_dilation(sandm, np.ones((3, 5))) & ~sandm & crack
    tone = np.where(near_sand, tone + 1, tone)
    tone = np.where(gmask, tone, 0)
    ti = np.clip(tone, 0, 10).astype(int)
    img = DUST_RGB[ti]
    # hue-shifted shadows: the deep cracks go cool (ash), not just darker brown
    deep = crack & (ti <= 2) & (d > 10) & ~sandm
    img[deep & (ti <= 1)] = C('ash1')
    img[deep & (ti == 2)] = C('ash2')
    return dict(img=img, tone=tone, crack=crack, sand=sandm, depth=d, gmask=gmask, plates=lab)


# ------------------------------------------------------------------ sky
SKY_TONES = ['ash3', 'ash4', 'ash5', 'ash6', 'dust7', 'dust8', 'dust9', 'dust10', 'white0']
SKY_RGB = pal_array(SKY_TONES)


def _interp_stops(y, stops):
    ys = np.array([s[0] for s in stops], float)
    vs = np.array([s[1] for s in stops], float)
    return np.interp(y, ys, vs)


def quant(tone, band=0.35, ox=0, oy=0):
    """Float tone field -> int levels, ordered-dither only in a narrow seam."""
    h, w = tone.shape
    fl = np.floor(tone)
    fr = np.clip((tone - fl - 0.5) / max(band, 1e-3) + 0.5, 0, 1)
    return (fl + (bayer(h, w, 4, ox, oy) < fr)).astype(np.int32)


class Sky:
    """Overcast dust sky: banded gradient, pale sun smudge (concentric halo
    rings), optional soft glow low behind the subject.  (Clouds: NoiseClouds.)"""

    def __init__(self, hy, seed, sun=(62, 32), stops=((0, 2.3), (45, 3.0), (90, 3.9), (115, 4.6), (126, 5.0)),
                 sun_rings=((2.4, 7.0), (4.6, 6.0), (8.5, 5.0)), strip=720, band=0.35,
                 tones=('dust3', 'dust4', 'dust5', 'dust6', 'dust7', 'dust8', 'dust9', 'dust10', 'white0'),
                 glow=None):
        self.hy, self.seed, self.sun, self.SW = hy, seed, sun, strip
        self.tones = list(tones)
        self.rgb = pal_array(self.tones)
        ys = np.arange(hy, dtype=float)[:, None] * np.ones((1, W))
        xs = np.ones((hy, 1)) * np.arange(W, dtype=float)[None, :]
        grad = _interp_stops(ys, stops)
        if glow is not None:                  # a soft brighter patch of haze low behind the subject
            gx, gy_, grx, gry, gamt = glow
            gd = np.sqrt(((xs - gx) / grx) ** 2 + ((ys - gy_) / gry) ** 2)
            grad = grad + gamt * np.clip(1 - gd, 0, 1) ** 1.6
        r = np.hypot(xs - sun[0], (ys - sun[1]))
        self.r = r
        # halo: piecewise tone by radius, merged with the gradient by max -> round rings
        self.rings = [tuple(q) for q in sun_rings]
        rr = np.array([q[0] for q in sun_rings] + [sun_rings[-1][0] * 1.8])
        tv = np.array([q[1] for q in sun_rings] + [0.0])
        halo = np.interp(r, rr, tv, left=tv[0], right=0.0)
        tone = np.maximum(grad, halo)
        self.grad = grad
        self.tone = tone
        self.sunlit = 1.6 * np.exp(-(r / 26.0) ** 2)      # back-lit haze near the sun
        q = quant(tone, band)
        q = np.clip(q, 0, len(self.tones) - 1)
        self.base_idx = q
        self.base = self.rgb[q]

    def render(self, img, t):
        img[:self.hy] = self.base
        return img


# ------------------------------------------------------------------ distant ruins
def skyline(seed, hy, x0, x1, hmin, hmax, body, lit, dark, win=None, gap=(2, 9), wid=(4, 13),
            envelope=None, lean_p=0.15, spire_p=0.2, broken_p=0.55, h=H, w=W):
    """RGBA layer with a row of ruined buildings standing on the horizon."""
    lay = layer(h, w)
    x = x0
    k = 0
    while x < x1:
        r = lambda q: float(noise.rand(k, q, seed))
        x += int(gap[0] + (gap[1] - gap[0]) * r(1))
        bw = int(wid[0] + (wid[1] - wid[0]) * r(2))
        env = 1.0 if envelope is None else float(envelope(x + bw / 2))
        bh = int((hmin + (hmax - hmin) * r(3) ** 1.4) * env)
        if bh < 3:
            x += bw
            k += 1
            continue
        lean = (r(4) - 0.5) * 0.5 if r(5) < lean_p else 0.0
        top = np.full(bw, float(hy - bh))
        if r(6) < broken_p:                            # broken / collapsed top
            cut = int(bw * (0.3 + 0.5 * r(7)))
            side = r(8) < 0.5
            depth = bh * (0.2 + 0.5 * r(9))
            for c in range(bw):
                cc = c if side else bw - 1 - c
                if cc < cut:
                    top[c] += depth * (1 - cc / max(cut, 1)) ** 0.8
                top[c] += int(noise.rand(k * 31 + c, 12, seed) * 2.4)
        for c in range(bw):
            ytop = int(round(top[c]))
            yb = hy + 2
            sh = int(round(lean * (hy - ytop)))
            for y in range(ytop, yb):
                xx = x + c + int(round(lean * (hy - y)))
                if 0 <= xx < w and 0 <= y < h:
                    col = body
                    if c == 0:
                        col = lit
                    elif c == bw - 1 and bw > 4:
                        col = dark
                    lay[y, xx, :3] = C(col)
                    lay[y, xx, 3] = 255
        # windows (dark holes) in rows
        if win is not None and bw >= 4:
            for y in range(int(top.min()) + 2, hy - 1, 3):
                for c in range(1, bw - 1, 2):
                    if y > top[c] + 1 and noise.rand(k * 97 + c, y, seed + 1) < 0.45:
                        xx = x + c + int(round(lean * (hy - y)))
                        if 0 <= xx < w:
                            lay[y, xx, :3] = C(win)
        # antenna / rebar
        if r(10) < spire_p:
            c = int(bw * r(11))
            ht = int(2 + 7 * r(12))
            yt = int(top[c])
            for y in range(yt - ht, yt):
                xx = x + c + int(round(lean * (hy - y)))
                if 0 <= xx < w and 0 <= y < h:
                    lay[y, xx, :3] = C(body)
                    lay[y, xx, 3] = 255
        x += bw
        k += 1
    return lay


def dunes(seed, hy, amp, freq, base_off, body, lit, shade, h=H, w=W, x_off=0.0):
    """Low dunes on the horizon.  Returns (RGBA layer, top profile array)."""
    xs = np.arange(w, dtype=float) + x_off
    prof = hy + base_off - amp * (noise.fbm1(xs * freq, seed, 3) - 0.35)
    prof = np.round(prof).astype(int)
    lay = layer(h, w)
    for x in range(w):
        y0 = prof[x]
        if y0 >= h:
            continue
        lay[y0:hy + 3, x, :3] = C(body)
        lay[y0:hy + 3, x, 3] = 255
        # light from upper-left: rising slope (toward the right) is lit, falling is shaded
        dl = prof[x] - prof[x - 1] if x > 0 else 0
        dr = prof[x + 1] - prof[x] if x < w - 1 else 0
        if dl < 0 or (dl == 0 and dr > 0 and False):
            lay[y0, x, :3] = C(lit)
        elif dl > 0:
            n = min(dl, 3)
            lay[y0:y0 + n + 1, x, :3] = C(shade)
        else:
            lay[y0, x, :3] = C(lit)
    return lay, prof


# ------------------------------------------------------------------ fence / posts
def post(img, x, yb, h, lean, width, cols, broken=0, seed=0):
    """Weathered fence post: base (x, yb), height h, top offset `lean` px.
    cols = (lit, body, dark).  broken>0 = jagged snapped top of that depth."""
    lit, body, dark = cols
    top = h
    for i in range(top):
        y = yb - i
        xx = x + int(round(lean * i / max(h, 1)))
        for c in range(width):
            if i >= top - broken and c != int(noise.rand(i, 3, seed) * width):
                continue
            col = lit if (c == 0 and width > 1) else (dark if (c == width - 1 and width > 2) else body)
            draw.pset(img, xx + c, y, col)
    return x + int(round(lean)), yb - h


def sag_wire(img, x0, y0, x1, y1, sag, color, gaps=(), seed=0):
    """1-px drooping wire (parabolic sag) from (x0,y0) to (x1,y1)."""
    n = max(2, int(abs(x1 - x0)))
    s = np.linspace(0, 1, n + 1)
    xs = x0 + (x1 - x0) * s
    ys = y0 + (y1 - y0) * s + sag * 4 * s * (1 - s)
    px, py = [], []
    for k in range(n):
        a, b = draw.line_points(xs[k], ys[k], xs[k + 1], ys[k + 1])
        px += a; py += b
    pts = sorted(set(zip(px, py)))
    for (x, y) in pts:
        f = (x - x0) / max(x1 - x0, 1)
        if any(g0 <= f <= g1 for g0, g1 in gaps):
            continue
        draw.pset(img, x, y, color)


# ------------------------------------------------------------------ lighting helpers
def light_box(img, cx, cy, radius, peak, mode='warm', sy=1.0, band=0.45, gamma=1.2, core=0.0, shadow='night'):
    """light_pool restricted to its bounding box (fast for small lights)."""
    r = int(math.ceil(radius)) + 2
    x0, x1 = max(int(cx) - r, 0), min(int(cx) + r + 1, img.shape[1])
    ry = int(math.ceil(radius * sy)) + 2
    y0, y1 = max(int(cy) - ry, 0), min(int(cy) + ry + 1, img.shape[0])
    if x0 >= x1 or y0 >= y1:
        return img
    sub = np.ascontiguousarray(img[y0:y1, x0:x1])
    lv = light.radial_level(y1 - y0, x1 - x0, cx - x0, cy - y0, radius, peak, gamma, sy, core)
    img[y0:y1, x0:x1] = light.apply_light(sub, lv, mode=mode, band=band, ox=x0, oy=y0, shadow=shadow)
    return img


def shadow_ellipse(img, cx, cy, rx, ry, steps=1.2, band=0.5, shadow='plain'):
    """Soft contact shadow: darken an ellipse by `steps` (down each colour's ramp)."""
    x0, x1 = max(int(cx - rx - 2), 0), min(int(cx + rx + 3), img.shape[1])
    y0, y1 = max(int(cy - ry - 2), 0), min(int(cy + ry + 3), img.shape[0])
    if x0 >= x1 or y0 >= y1:
        return img
    ys, xs = np.ogrid[y0:y1, x0:x1]
    dd = np.sqrt(((xs - cx) / max(rx, 0.5)) ** 2 + ((ys - cy) / max(ry, 0.5)) ** 2)
    lv = steps * np.clip(1.25 - dd, 0, 1) / 0.25
    lv = np.clip(lv, 0, steps)
    sub = np.ascontiguousarray(img[y0:y1, x0:x1])
    img[y0:y1, x0:x1] = ramp_darken(sub, lv, band=band, ox=x0, oy=y0)
    return img


# ------------------------------------------------------------------ atmosphere
class WindPath:
    """Cumulative wind displacement D(t) = integral of speed(t) dt, tabulated
    at import (so drifting particles move smoothly however the wind varies)."""

    def __init__(self, speed_fn, t0=-1.0, t1=12.0, dt=1 / 240):
        self.t0, self.dt = t0, dt
        ts = np.arange(t0, t1 + dt, dt)
        v = np.array([speed_fn(t) for t in ts])
        self.D = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) * 0.5 * dt)])
        self.ts = ts

    def __call__(self, t):
        return float(np.interp(t, self.ts, self.D))


def dust_motes(img, t, seed, n=22, area=(0, 20, W, 140), disp=0.0, colors=('dust9', 'dust8', 'dust10'),
               weights=(0.5, 0.35, 0.15), sway=(3.0, 0.35), vy=2.0):
    """Floating dust specks carried by the wind: disp = cumulative wind
    displacement (px, leftward), each speck with its own speed factor."""
    i = np.arange(n)
    ax, ay, aw, ah = area
    x0 = ax + noise.rand(i, 1, seed) * aw
    y0 = ay + noise.rand(i, 2, seed) * ah
    fac = 0.6 + 0.8 * noise.rand(i, 3, seed)
    ph = noise.rand(i, 5, seed)
    x = x0 - disp * fac
    y = y0 + vy * t * (noise.rand(i, 4, seed) - 0.3) + 1.5 * np.sin(2 * math.pi * (0.3 * t * (0.7 + 0.6 * ph) + ph))
    x = x + sway[0] * np.sin(2 * math.pi * (sway[1] * t * (0.7 + 0.6 * ph) + ph))
    x = ax + np.mod(x - ax, aw)
    y = ay + np.mod(y - ay, ah)
    r = noise.rand(i, 6, seed)
    xi, yi = np.floor(x).astype(int), np.floor(y).astype(int)
    edges = np.cumsum(weights) / np.sum(weights)
    lo = 0
    for c, hi in zip(colors, edges):
        m = (r >= lo) & (r < hi)
        draw.points(img, xi[m], yi[m], c)
        lo = hi
    big = r > 0.93
    draw.points(img, xi[big] + 1, yi[big], colors[0])
    return img


def sand_streams(img, t, seed, gy_fn, n=7, x_range=(-60, W + 60), speed=(70, 120), length=(14, 46),
                 hgt=(0, 6), colors=('dust7', 'dust8'), life=(1.2, 2.2), amp=1.4, strength=1.0):
    """Wind-blown sand threads skimming the ground right -> left.  Each is a
    solid, sinuous 1-px thread that grows out of nothing, streams and thins
    away (length envelope + a few ragged gaps at its tail - no regular dither)."""
    for i in range(n):
        per = life[0] + (life[1] - life[0]) * float(noise.rand(i, 1, seed))
        ph = float(noise.rand(i, 2, seed))
        cyc = math.floor(t / per + ph)
        a = (t / per + ph) - cyc                  # 0..1 age in this life
        key = i * 1000 + cyc
        sp = speed[0] + (speed[1] - speed[0]) * float(noise.rand(key, 3, seed))
        ln = length[0] + (length[1] - length[0]) * float(noise.rand(key, 4, seed))
        x0 = x_range[1] - (x_range[1] - x_range[0]) * float(noise.rand(key, 5, seed))
        hh = hgt[0] + (hgt[1] - hgt[0]) * float(noise.rand(key, 6, seed))
        col = colors[int(noise.rand(key, 7, seed) * len(colors)) % len(colors)]
        env = math.sin(math.pi * a) * strength     # grow in / thin out over its life
        cur = ln * env
        if cur < 2:
            continue
        head = x0 - sp * a * per
        xs = np.arange(int(head), int(head + cur))
        s = (xs - head) / max(cur, 1)
        ys = np.array([gy_fn(xx) for xx in xs]) - hh - amp * np.sin(xs * 0.19 + t * 9 + i * 2.1) * s
        # ragged tail: pixels drop out toward the end (hash, not a regular pattern)
        keep = noise.rand(xs.astype(np.int64), key, seed + 5) > np.clip((s - 0.55) * 1.6, 0, 0.8)
        draw.points(img, xs[keep], np.round(ys[keep]).astype(int), col)
    return img


def make_vignette(strength=1.2, power=2.6, cx=W / 2, cy=H / 2, sx=1.0, sy=1.0, bottom=0.0, top=0.0, hy=None):
    """Static vignette level map (negative light steps).  With hy given, the
    sky (rows < hy) is left alone (flat skies show dither seams)."""
    ys, xs = np.ogrid[:H, :W]
    d = np.sqrt(((xs - cx) / (W / 2 * sx)) ** 2 + ((ys - cy) / (H / 2 * sy)) ** 2) / math.sqrt(2)
    lv = -strength * np.clip(d, 0, 1.2) ** power * 1.4
    if bottom:
        lv = lv - bottom * np.clip((ys - (H - 40)) / 40.0, 0, 1) ** 1.5
    if top:
        lv = lv - top * np.clip((30 - ys) / 30.0, 0, 1) ** 1.5
    lv = lv * np.ones((H, W))
    if hy is not None:
        lv = lv * np.clip((ys - hy) / 14.0, 0, 1)
    return lv


# ---- hue-preserving darkening: one step down inside each colour's own ramp
from engine import palette as _P
from engine.core import to_index as _to_index, from_index as _from_index
_DOWN = np.arange(_P.N)
for _i in range(_P.N):
    _r, _p = _P.RAMP_OF[_i], _P.POS_OF[_i]
    if _p > 0:
        _DOWN[_i] = _P.RAMP_IDX[_r][_p - 1]
_DOWN_LUT = [np.arange(_P.N)]
for _k in range(1, 5):
    _DOWN_LUT.append(_DOWN[_DOWN_LUT[-1]])
_DOWN_LUT = np.stack(_DOWN_LUT)            # (5, N): k steps down


def ramp_darken(img, level, band=0.25, ox=0, oy=0):
    """Darken by `level` (float steps >= 0, scalar or (h,w)) moving each pixel
    down its own ramp (no hue shift).  Ordered dither in a narrow seam."""
    h, w = img.shape[:2]
    lv = np.asarray(level, float)
    if lv.ndim == 0:
        lv = np.full((h, w), float(lv))
    fl = np.floor(lv)
    fr = np.clip((lv - fl - 0.5) / max(band, 1e-3) + 0.5, 0, 1)
    q = np.clip((fl + (bayer(h, w, 4, ox, oy) < fr)).astype(np.int32), 0, 4)
    if not q.any():
        return img
    idx = _to_index(np.ascontiguousarray(img[..., :3]))
    out = img.copy()
    out[..., :3] = _from_index(_DOWN_LUT[q, idx])
    return out


# ------------------------------------------------------------------ dead trees
def dead_tree(img, x, y, height, seed, cols=('dust4', 'dust5', 'dust3'), lean=-0.15, spread=0.55,
              depth=4, trunk_w=2):
    """Skeletal dead tree drawn with pixel lines.  (x, y) = base.  cols =
    (trunk/branch colour, lit edge, dark accent).  Light from the upper-left."""
    body, lit, dark = cols

    def branch(x0, y0, ang, ln, lvl, k):
        x1 = x0 + math.sin(ang) * ln
        y1 = y0 - math.cos(ang) * ln
        wdt = trunk_w if lvl == 0 else 1
        xs, ys = draw.line_points(x0, y0, x1, y1)
        for (px, py) in zip(xs, ys):
            draw.pset(img, px, py, body)
            if wdt >= 2:
                draw.pset(img, px + 1, py, dark)
                draw.pset(img, px - 0 if False else px, py, lit if (py % 3) else body)
        if lvl >= depth or ln < 2.2:
            return
        nb = 2 if noise.rand(k, 1, seed) < 0.8 else 3
        for j in range(nb):
            r = float(noise.rand(k * 7 + j, 2, seed))
            da = (spread * (0.6 + 0.8 * r)) * (1 if j % 2 == 0 else -1) + lean * 0.5
            if j == 2:
                da = (r - 0.5) * 0.4
            nl = ln * (0.58 + 0.25 * float(noise.rand(k * 7 + j, 3, seed)))
            # branch starts a bit below the tip (not all from one node)
            f = 0.75 + 0.25 * float(noise.rand(k * 7 + j, 4, seed)) if lvl > 0 else 1.0
            bx = x0 + (x1 - x0) * f
            by = y0 + (y1 - y0) * f
            branch(bx, by, ang + da, nl, lvl + 1, k * 3 + j + 1)

    branch(x, y, lean, height * 0.42, 0, 1)


# ------------------------------------------------------------------ small ground props
_PEB = {'k': 'dust1', 'd': 'dust2', 'b': 'dust3', 'B': 'dust4', 'l': 'dust5', 'L': 'dust6', 'h': 'dust7'}
PEBBLES = [from_ascii(a, _PEB, {'b': b}) for a, b in (
    ("""
.lB.
lBbd
kddk
""", (1, 2)),
    ("""
.Lh.
LlBd
.dk.
""", (1, 2)),
    ("""
..lLB..
.lBBBbd
lBbbbdk
.kddkk.
""", (3, 3)),
    ("""
lB
dk
""", (0, 1)),
    ("""
.hL..
LlBb.
lBbdd
.kkk.
""", (2, 3)),
)]
_BONE = {'w': 'dust8', 'W': 'dust7', 'g': 'dust6', 'k': 'dust4', 'K': 'dust3'}
BONES = [from_ascii(a, _BONE, {'b': b}) for a, b in (
    ("""
w.....w
Wwwwwwg
g.....k
""", (3, 2)),
    ("""
.wW.
wg..
.k..
..W.
..gw
""", (2, 4)),
    ("""
..w...
.Wgk..
Wg.Ww.
k..gk.
""", (2, 3)),
)]
_TWIG = {'t': 'dust2', 'T': 'dust4'}
TWIGS = [from_ascii(a, _TWIG, {'b': b}) for a, b in (
    ("""
T.....
.tt...
...ttt
..t...
""", (3, 3)),
    ("""
.....T
..ttt.
tt....
""", (2, 2)),
)]


def scatter_props(img, geo, seed, n, sprites, x_range=(0, W), d_range=(14, 54), avoid=(), sand_tail=True):
    """Deterministically scatter small sprites on the ground (size by depth
    is baked into the sprite choice: small ones far, bigger near)."""
    placed = []
    for i in range(n):
        x = x_range[0] + (x_range[1] - x_range[0]) * float(noise.rand(i, 1, seed))
        dd = d_range[0] + (d_range[1] - d_range[0]) * float(noise.rand(i, 2, seed)) ** 0.8
        y = geo.hy + dd
        if any(ax0 <= x <= ax1 and ay0 <= y <= ay1 for ax0, ay0, ax1, ay1 in avoid):
            continue
        k = int(noise.rand(i, 3, seed) * len(sprites))
        spr = sprites[k]
        if spr.w > 4 and dd < 22:
            spr = sprites[min(k, 1)] if len(sprites) > 1 else spr
        # little sand tail on the lee (left) side - the wind blows from the right
        if sand_tail:
            for j in range(1, 3 + int(dd / 12)):
                draw.pset(img, int(x) - spr.w // 2 - j, int(y), 'dust6' if dd > 20 else 'dust7')
        shadow_ellipse(img, x + 0.8, y + 0.4, spr.w / 2 + 0.5, 0.9, steps=1.0, shadow='plain')
        blit(img, spr, x, y, anchor='b')
        placed.append((x, y))
    return placed


class NoiseClouds:
    """Streaky dust clouds from horizontally stretched fbm, thresholded into
    clean flat-toned shapes with lit upper rims.  Pre-rendered on a wide
    strip per parallax layer; render() crops with an integer offset."""

    def __init__(self, sky, seed, layers, strip=720):
        self.sky, self.SW = sky, strip
        hy = sky.hy
        self.layers = []
        for (lseed, y0, y1, sx, sy, thr, speed, dark) in layers:
            ys = np.arange(hy, dtype=float)[:, None] * np.ones((1, strip))
            xs = np.ones((hy, 1)) * np.arange(strip, dtype=float)[None, :]
            # periodic in x: blend two samples so the strip wraps seamlessly
            u = xs / strip
            n1 = noise.fbm2(xs / sx, ys / sy, lseed, 4)
            n2 = noise.fbm2((xs - strip) / sx, ys / sy, lseed, 4)
            n = n1 * (1 - u) + n2 * u
            env = np.clip(np.minimum((ys - y0) / 6.0, (y1 - ys) / 10.0), 0, 1)
            dens = (n - thr) * env
            body = dens > 0
            thick = dens > 0.075
            body = ndimage.binary_opening(body, np.ones((1, 3)))
            thick = ndimage.binary_opening(thick & body, np.ones((1, 3)))
            up = np.zeros_like(body); up[1:] = body[:-1]           # pixel above is cloud
            rim = body & ~up                                         # top edge
            dn = np.zeros_like(body); dn[:-1] = body[1:]
            belly = body & ~dn & thick                               # bottom edge of thick parts
            cls = np.zeros((hy, strip), np.int8)
            cls[body] = 1
            cls[thick] = 2
            cls[belly] = 3
            cls[rim] = 4
            self.layers.append((cls, speed, dark))

    def render(self, img, t):
        sky = self.sky
        hy = sky.hy
        g = sky.grad
        for li, (cls, speed, dark) in enumerate(self.layers):
            # staggered step phase per layer: the 1-px scroll steps never coincide
            off = int(math.floor(speed * t + 0.37 * li + 0.11))
            c = cls[:, (np.arange(W) + off) % self.SW]
            m = c > 0
            if not m.any():
                continue
            # tone relative to the local sky gradient (flat per row -> clean bands)
            base = g - dark
            tv = np.where(c == 1, base + 0.35, np.where(c == 2, base - 0.45, np.where(c == 3, base - 1.0, base + 1.1)))
            tv = tv + sky.sunlit * (c > 0)
            ti = np.clip(quant(tv, 0.5), 0, len(sky.tones) - 1).astype(np.int32)
            img[:hy][m] = sky.rgb[ti[m]]
        return img


# ------------------------------------------------------------------ discovery FX
def sparkle(img, x, y, size, cols=('leaf8', 'leaf7', 'leaf6', 'leaf5')):
    """4-point star twinkle.  size 0..4 (arm length); centre brightest."""
    x, y = int(x), int(y)
    if size <= 0:
        return
    c0, c1, c2, c3 = cols
    for k in range(1, size + 1):
        c = c1 if k == 1 else (c2 if k < size else c3)
        for dx, dy in ((k, 0), (-k, 0), (0, k), (0, -k)):
            draw.pset(img, x + dx, y + dy, c)
    if size >= 3:
        for dx, dy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            draw.pset(img, x + dx, y + dy, c3)
    draw.pset(img, x, y, c0)


def green_glow(img, x, y, amt=1.0, radius=9.0):
    """The sprout's faint life-glow (palette-true light pool)."""
    if amt <= 0.02:
        return img
    return light_box(img, x + 0.5, y + 0.5, radius, 1.05 * amt, mode='green', sy=0.62, band=0.45, gamma=1.3)


def dirt_bits(img, t, t0, n, seed, origin, speed=(20, 45), spread=(-2.6, -0.5), life=(0.35, 0.6),
              gravity=160.0, cols=('dust3', 'dust4', 'dust2')):
    """A small spray of soil clods (one-shot, pure function of t)."""
    from engine import fx
    x, y, age, alive, r = fx.bursts(t, n, seed, origin, speed=speed, life=life, gravity=gravity,
                                    spread=spread, t0=t0)
    for i in np.nonzero(alive)[0]:
        c = cols[int(r[i] * len(cols)) % len(cols)]
        draw.pset(img, int(x[i]), int(y[i]), c)


# ---- palette-true FOG: shift colours toward a haze colour in quantised steps
_FOG = {}


def fog_lut(fog='dust8', steps=4, ramps=('dust', 'ash', 'ochre', 'steel', 'paint', 'rust')):
    key = (fog, steps, ramps)
    if key in _FOG:
        return _FOG[key]
    allowed = [i for i in range(_P.N) if _P.RAMP_OF[i] in ramps]
    lab = _P.PAL_LAB
    f = _P.linear_to_oklab(_P.srgb_to_linear(np.array(C(fog), float)))
    lut = np.zeros((steps + 1, _P.N), np.int32)
    for k in range(steps + 1):
        a = k / steps
        tgt = lab * (1 - a) + f[None, :] * a
        d = ((tgt[:, None, :] - lab[None, allowed, :]) ** 2).sum(-1)
        lut[k] = np.array(allowed)[d.argmin(-1)]
        if k == 0:
            lut[k] = np.arange(_P.N)
    _FOG[key] = lut
    return lut


def apply_fog(img, level, fog='dust8', steps=4, band=0.35, ox=0, oy=0):
    """level (h,w) in 0..1 = fog density.  Quantised into `steps` palette-true
    haze steps with ordered dither only at the seams."""
    lut = fog_lut(fog, steps)
    h, w = img.shape[:2]
    lv = np.clip(np.asarray(level, float), 0, 1) * steps
    fl = np.floor(lv)
    fr = np.clip((lv - fl - 0.5) / max(band, 1e-3) + 0.5, 0, 1)
    q = np.clip((fl + (bayer(h, w, 4, ox, oy) < fr)).astype(np.int32), 0, steps)
    if not q.any():
        return img
    idx = _to_index(np.ascontiguousarray(img[..., :3]))
    out = img.copy()
    out[..., :3] = _from_index(lut[q, idx])
    return out


class DustVeil:
    """A gust dragging veils of dust across the frame right -> left.

    Built from streamlined WISPS (dense head, long thinning tail, wavy
    centre-line), a faint unifying haze between the first and last wisp,
    solid sand THREADS and a ground-hugging drift.  Coverage = ordered
    dither (palette-true translucency).  Pure function of t."""

    def __init__(self, t0, t1, gy, seed=0, n=11, height=48, spread=150.0, gain=1.6):
        self.t0, self.t1, self.gy, self.seed = t0, t1, gy, seed
        self.gain = gain
        self.n, self.height, self.spread = n, height, spread
        self.speed = (W + spread + 220) / (t1 - t0)
        self.w = []
        for i in range(n):
            r = [float(noise.rand(i, k, seed + 31)) for k in range(8)]
            self.w.append(dict(
                off=r[0] * spread,                          # distance behind the front
                y=gy + 2 - (r[1] ** 1.4) * height,          # centre height
                ln=80 + 150 * r[2],
                th=3.0 + 8.0 * r[3] * (1 - 0.4 * r[1]),
                amp=1.5 + 4.0 * r[4], wl=18 + 30 * r[5], ph=r[6] * 6.28,
                sp=1.0 + 0.25 * r[7], dens=0.55 + 0.4 * r[3]))

    def front(self, t):
        return W + 10 - self.speed * (t - self.t0)

    def coverage(self, t):
        """(H,W) float coverage in 0..1 (None when inactive)."""
        if t < self.t0 or t > self.t1 + 0.15:
            return None
        f = self.front(t)
        a = np.zeros((H, W))
        for q in self.w:
            head = f + q['off'] - (q['sp'] - 1.0) * self.speed * (t - self.t0) * 0.3
            xa, xb = max(int(head) - 1, 0), min(int(head + q['ln']) + 1, W)
            if xa >= xb:
                continue
            ext = q['th'] + q['amp'] + 4
            ya, yb = max(int(q['y'] - ext - 3), 0), min(int(q['y'] + ext + 1), H)
            if ya >= yb:
                continue
            xs = np.arange(xa, xb, dtype=float)
            ys = np.arange(ya, yb, dtype=float)[:, None]
            s = (xs - head) / q['ln']                               # 0 at head .. 1 at tail
            prof = np.clip(s * 7, 0, 1) * np.clip(1 - s, 0, 1) ** 0.7
            yc = q['y'] - 3.0 * s + q['amp'] * np.sin(xs / q['wl'] + q['ph'] + t * 3.0) * s
            th = np.maximum(q['th'] * prof, 0.01)
            dy = np.abs(ys - yc[None, :]) / th[None, :]
            v = np.clip(1 - dy, 0, 1) ** 0.8 * q['dens'] * ((prof > 0.05) & (s > -0.02) & (s < 1.0))[None, :]
            a[ya:yb, xa:xb] = np.maximum(a[ya:yb, xa:xb], v)
        # the dust mass: ragged soft front, billowing top, thinning toward the camera
        y0 = max(int(self.gy - self.height * 1.3), 0)
        y1 = min(int(self.gy + 28), H)
        xs = np.arange(W, dtype=float)
        ys = np.arange(y0, y1, dtype=float)[:, None]
        adv = self.speed * (t - self.t0)
        jag = 10.0 * noise.fbm1(ys[:, 0] / 7.0 + t * 3.0, self.seed + 21, 2) - 5.0
        rel = xs[None, :] - (f + jag[:, None])
        lead = np.clip(rel / 30.0, 0, 1) ** 1.3
        tail = np.exp(-np.clip(rel - self.spread, 0, None) / 80.0)
        htop = self.height * (0.55 + 0.6 * noise.fbm1((xs + adv * 0.9) / 22.0, self.seed + 23, 3))
        htop = htop * (0.65 + 0.55 * np.exp(-np.clip(xs - f, 0, None) / 110.0))
        above = np.clip((self.gy + 4 - ys) / htop[None, :], 0, None)          # 0 at ground .. 1 at top
        vz = np.clip(1 - above, 0, 1) ** 1.4
        below = np.clip((ys - (self.gy + 2)) / 26.0, 0, 1)                    # toward the camera
        vz = vz * (1 - below) ** 1.5
        live = (lead * tail) > 0.01
        cols = np.nonzero(live.any(0))[0]
        if len(cols):
            c0, c1 = cols[0], cols[-1] + 1
            tex = noise.fbm2((xs[None, c0:c1] + adv * 1.05) / 30.0, ys / 6.0, self.seed + 29, 3)
            mass = lead[:, c0:c1] * tail[:, c0:c1] * vz[:, c0:c1] * (0.35 + 1.0 * tex)
            sub = a[y0:y1, c0:c1]
            a[y0:y1, c0:c1] = np.maximum(sub * vz[:, c0:c1] ** 0.3, 0.5 * mass)
        return np.clip(a, 0, 1)

    def draw(self, img, t):
        a = self.coverage(t)
        if a is None:
            return img
        return apply_fog(img, np.clip(a * self.gain, 0, 0.92), fog='dust7', steps=4, band=0.4)

    def streaks(self, img, t, n=34):
        """Fast horizontal sand threads racing with the wisps."""
        if t < self.t0 or t > self.t1 + 0.15:
            return img
        f = self.front(t)
        for i in range(n):
            r1, r2, r3, r4 = (float(noise.rand(i, k, self.seed + 9)) for k in range(4))
            y = self.gy + 3 - (r1 ** 1.8) * self.height * 1.1
            x = f + 4 + r2 * (self.spread + 60) - 120 * (t - self.t0) * r3
            ln = 5 + 24 * r4
            if x > W or x + ln < 0:
                continue
            draw.hline(img, int(x), int(x + ln), int(y), 'dust9' if r4 > 0.55 else 'dust8')
        return img


# ------------------------------------------------------------------ foreground framing rocks
def fg_rock(img, x0, x1, base_y, height, seed, cols=('dust1', 'dust2', 'dust4', 'ink2'), tufts=3):
    """Dark foreground rock silhouette along the bottom edge (depth framing).
    Lumpy top, faint rim light on its upper-left facets, a few dry tufts."""
    body, mid, rim, deep = cols
    xs = np.arange(int(x0), int(x1))
    n = len(xs)
    if n <= 0:
        return
    s = (xs - x0) / max(x1 - x0, 1)
    env = np.clip(np.sin(np.pi * np.clip(s, 0, 1)) ** 0.6, 0, 1)
    lump = 0.75 + 0.35 * noise.fbm1(xs / 7.0, seed, 3)
    top = np.round(base_y - height * env * lump).astype(int)
    for k, x in enumerate(xs):
        if x < 0 or x >= img.shape[1]:
            continue
        y0 = top[k]
        if y0 >= img.shape[0]:
            continue
        img[max(y0, 0):img.shape[0], x] = C(body)
        # facet shading: rising edge (left side of lumps) catches light
        rising = k > 0 and top[k] < top[k - 1]
        flat = k > 0 and top[k] == top[k - 1]
        if rising or (flat and noise.rand(k, 3, seed) < 0.5):
            draw.pset(img, x, y0, rim)
            if rising:
                draw.pset(img, x, y0 + 1, mid)
        else:
            draw.pset(img, x, y0, mid)
    # dry grass tufts poking out of the rock's base / top
    for j in range(tufts):
        r = float(noise.rand(j, 7, seed))
        k = int(r * (n - 1))
        x = xs[k]
        y = top[k]
        for b in range(3):
            lean = -1 - b + int(noise.rand(j, b, seed) * 2)
            hh = 3 + int(noise.rand(j, b + 5, seed) * 4)
            draw.line(img, x + b - 1, y, x + b - 1 + lean, y - hh, mid if b % 2 else rim)


def calm_patch(img, cx, cy, rx, ry, seed, col='dust5', rim='dust6', shade='dust4', wob=0.28):
    """A wind-smoothed sheet of sand over the cracked plates: calm ground under
    the characters so the action reads.  Irregular edge, lit far rim, a few
    ripples.  Returns the boolean mask."""
    ys, xs = np.ogrid[:H, :W]
    ang = np.arctan2((ys - cy) / max(ry, 1e-3), (xs - cx) / max(rx, 1e-3))
    rr = 1.0 + wob * (noise.fbm1((ang + math.pi) * 2.2, seed, 3) - 0.5) * 2
    dd = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2
    inside = dd <= rr ** 2
    img[inside] = C(col)
    up = np.zeros_like(inside); up[1:] = inside[:-1]
    top_rim = inside & ~up
    img[top_rim] = C(rim)
    dn = np.zeros_like(inside); dn[:-1] = inside[1:]
    bottom = inside & ~dn
    img[bottom] = C(shade)
    # faint wind ripples: short lighter dashes across the sheet
    rip = inside & ~top_rim & ~bottom & (noise.rand((xs + ys * 3) // 5, ys, seed + 1) < 0.10) & ((ys + xs // 7) % 3 == 0)
    img[rip] = C(rim)
    return inside


# ------------------------------------------------------------------ broken farm windmill
def windmill_tower(lay, x, yb, h, cols=('dust4', 'dust5', 'dust3')):
    """Tapered lattice tower (farm water-pump windmill).  Returns hub (x, y)."""
    body, lit, dark = cols
    top_w, base_w = 2, 10
    xl0, xr0 = x - base_w // 2, x + base_w // 2
    xl1, xr1 = x - top_w // 2, x + top_w // 2
    ty = yb - h
    draw.line(lay, xl0, yb, xl1, ty, lit)            # legs (lit left leg)
    draw.line(lay, xr0, yb, xr1, ty, body)
    # cross bracing, every few px (X pattern), sparser near the top
    ys = list(range(yb, ty, -max(3, h // 6)))
    for a, b in zip(ys[:-1], ys[1:]):
        fa = (yb - a) / h
        fb = (yb - b) / h
        la = xl0 + (xl1 - xl0) * fa; ra = xr0 + (xr1 - xr0) * fa
        lb = xl0 + (xl1 - xl0) * fb; rb = xr0 + (xr1 - xr0) * fb
        draw.line(lay, la, a, rb, b, dark)
        draw.line(lay, ra, a, lb, b, dark)
        draw.line(lay, lb, b, rb, b, body)
    # platform + pump rod
    draw.hline(lay, xl1 - 2, xr1 + 2, ty + 2, body)
    draw.vline(lay, x, ty, yb, dark)
    return x, ty - 1


def windmill_wheel(img, hx, hy, r, angle, missing=(2, 5, 6), n=12, cols=('dust4', 'dust5', 'dust3'), aspect=0.5):
    """The rotor, facing the wind (right) so we see it in 3/4 (an ellipse);
    some blades are missing.  Blades on the near side are drawn last."""
    body, lit, dark = cols
    order = sorted(range(n), key=lambda k: math.cos(angle + 2 * math.pi * k / n))
    for k in order:
        if k in missing:
            continue
        a = angle + 2 * math.pi * k / n
        x1, y1 = hx + math.cos(a) * r * aspect + 1, hy + math.sin(a) * r
        x0, y0 = hx + math.cos(a) * 1.5 * aspect, hy + math.sin(a) * 1.5
        draw.line(img, x0, y0, x1, y1, lit if math.sin(a) < -0.3 else (dark if math.cos(a) < -0.5 else body))
    draw.pset(img, hx, hy, dark)
    draw.pset(img, hx + 1, hy, dark)
    # tail vane (to the right, pointing downwind = left?  wind from the right -> vane trails left)
    draw.line(img, hx + 1, hy, hx - 7, hy + 1, dark)
    draw.rect(img, hx - 11, hy - 2, 4, 4, body)
    draw.pset(img, hx - 11, hy - 2, lit)


# ------------------------------------------------------------------ dust-coated sprite
_DUSTY = {}
for _k in range(9):
    _DUSTY['leaf%d' % _k] = 'dust%d' % min(3 + _k // 2, 7)
    _DUSTY['ochre%d' % _k] = 'dust%d' % min(3 + _k // 2, 7)


def dusty(spr):
    """The same sprite coated in dust: green/ochre recoloured to dust greys."""
    from engine.sprite import recolor
    return recolor(spr, _DUSTY)


def vignette_plates(img, plates, lv, seed=0, jitter=0.35):
    """Bake a vignette into the static ground PLATE BY PLATE: each plate
    darkens as a whole (down its own ramp) by its mean level (+ a little
    per-plate jitter), so there are no dither seams or contours through the
    plates.  Non-plate pixels (cracks, props) use the rounded per-pixel level."""
    steps = np.clip(-lv, 0, 4)
    lab = plates
    n = int(lab.max()) + 1
    cnt = np.bincount(lab.ravel(), minlength=n).astype(float)
    tot = np.bincount(lab.ravel(), weights=steps.ravel(), minlength=n)
    mean = tot / np.maximum(cnt, 1)
    jit = (noise.rand(np.arange(n), 3, seed + 77) - 0.5) * jitter
    per = np.floor(mean + jit + 0.5)
    q = np.where(lab > 0, per[lab], np.floor(steps + 0.5)).astype(np.int32)
    q = np.clip(q, 0, 4)
    idx = _to_index(np.ascontiguousarray(img[..., :3]))
    out = img.copy()
    out[..., :3] = _from_index(_DOWN_LUT[q, idx])
    return out


# ------------------------------------------------------------------ foreground dry grass (animated)
def grass_tuft(img, x, y, t, seed, wind=0.6, n=5, h=(5, 11), cols=('dust5', 'dust6', 'dust4')):
    """A dark, dry grass tuft in the foreground, bending with the wind (from
    the right -> tips pushed left), animated on 12s.  Pure function of t."""
    ts = math.floor(t * 12) / 12.0
    for k in range(n):
        r1, r2, r3 = (float(noise.rand(k, q, seed)) for q in range(3))
        hh = h[0] + (h[1] - h[0]) * r1
        bx = x + (k - n / 2) * 1.2 + r2
        sway = wind * (1.5 + 2.5 * r3) + 0.9 * math.sin(ts * (4.0 + 3 * r2) + 6.28 * r3) * (0.4 + wind)
        tipx = bx - sway * (hh / h[1]) + (k - n / 2) * 0.7
        midx = bx - sway * 0.35 * (hh / h[1])
        col = cols[int(r3 * len(cols)) % len(cols)]
        draw.line(img, bx, y, midx, y - hh * 0.55, 'dust3')             # shaded base
        draw.line(img, midx, y - hh * 0.55, tipx, y - hh, col)         # bleached, sunlit upper blade


def sky_blend(img, sky_a, sky_b, k, region=None, n=8):
    """Blend two precomputed sky bases with an ordered-dither mask (k 0..1),
    optionally limited to a region mask (hy,W)."""
    if k <= 0:
        return img
    hy = sky_a.hy
    from engine.core import BAYER8
    th = np.tile(BAYER8, (hy // 8 + 1, W // 8 + 1))[:hy, :W]
    m = th < k
    if region is not None:
        m &= region
    img[:hy][m] = sky_b.base[m]
    return img


# ================================================================== LIFE (pass 2)
def sun_patch(sky, img, t, rings=None, breath=0.10, period=3.4, pad=3):
    """Re-render the sun's region with its halo rings breathing slowly
    (radius x (1 + breath*sin)); `rings` may override the ring set (e.g. an
    interpolated dimmer set).  Clean rings, dithered seams, screen-locked."""
    rings = rings if rings is not None else sky.rings
    rmax = rings[-1][0] * 1.8 * (1 + breath) + pad
    sx, sy = sky.sun
    x0, x1 = max(int(sx - rmax), 0), min(int(sx + rmax) + 1, W)
    y0, y1 = max(int(sy - rmax), 0), min(int(sy + rmax) + 1, sky.hy)
    if x0 >= x1 or y0 >= y1:
        return img
    s = 1.0 + breath * math.sin(2 * math.pi * t / period)
    r = sky.r[y0:y1, x0:x1] / s
    rr = np.array([q[0] for q in rings] + [rings[-1][0] * 1.8])
    tv = np.array([q[1] for q in rings] + [0.0])
    halo = np.interp(r, rr, tv, left=tv[0], right=0.0)
    tone = np.maximum(sky.grad[y0:y1, x0:x1], halo)
    q = quant(tone, 0.35, x0, y0)
    q = np.clip(q, 0, len(sky.tones) - 1)
    img[y0:y1, x0:x1] = sky.rgb[q]
    return img


def lerp_rings(ra, rb, k):
    return [(a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k) for a, b in zip(ra, rb)]


def dust_devil(img, t, x, yb, h, seed, amt=0.55, w0=2.0, w1=7.0):
    """A distant dust devil: a thin twisting funnel of dust (palette fog),
    wider at the top, its bands spiralling upward; leans and wobbles."""
    y0, y1 = max(int(yb - h), 0), min(int(yb) + 1, H)
    if y0 >= y1:
        return img
    ys = np.arange(y0, y1, dtype=float)[:, None]
    u = (yb - ys) / h                                            # 0 base .. 1 top
    lean = -4.0 * u ** 1.5 + 1.2 * np.sin(t * 1.7 + u * 5.0 + seed)
    cx = x + lean
    half = (w0 + (w1 - w0) * u ** 1.3) * (0.85 + 0.25 * np.sin(t * 3.1 + u * 9.0))
    x0, x1 = max(int(x - w1 - 8), 0), min(int(x + w1 + 8), W)
    if x0 >= x1:
        return img
    xs = np.arange(x0, x1, dtype=float)[None, :]
    d = np.abs(xs - cx) / np.maximum(half, 0.5)
    band = 0.55 + 0.45 * np.sin((xs - cx) * 1.3 - ys * 0.9 + t * 11.0)   # the spiral
    top = np.clip((1 - u) * 4, 0, 1) * np.clip(u * 6, 0, 1) + (u > 0.8) * 0.4
    a = np.clip(1 - d, 0, 1) ** 0.7 * band * amt * np.clip(top + 0.4, 0, 1)
    sub = np.ascontiguousarray(img[y0:y1, x0:x1])
    img[y0:y1, x0:x1] = apply_fog(sub, a, fog='dust6', steps=4, band=0.4, ox=x0, oy=y0)
    # a skirt of kicked-up sand at its foot
    for k in range(5):
        a2 = t * 7.0 + k * 1.3
        px = x + math.cos(a2) * (w0 + 2) * 1.2
        py = yb - 0.5 + math.sin(a2) * 0.6
        draw.pset(img, px, py, 'dust6' if k % 2 else 'dust7')
    return img


_TW = {'k': 'dust3', 'b': 'dust4', 'l': 'dust5', 'h': 'dust6'}
TUMBLE = [from_ascii(a, _TW, {'c': (3, 3)}) for a in ("""
..kbk..
.b.l.k.
k.hk.bk
bl.b.lb
k.bk.h.
.k.l.b.
..bkk..
""", """
..bkb..
.k.b.l.
bk.hk.k
.b.l.bk
kl.bk.b
.b.k.h.
..kkb..
""", """
..kkb..
.l.k.b.
b.kb.hk
k.lb.k.
bk.h.lb
.b.k.k.
..bbk..
""")]


def tumbleweed(img, t, t0, x0, y_ground, speed, seed, hop=3.0, spin=10.0):
    """A small tumbleweed bowling right->left along a far ground line, in
    hops, turning over (3 drawn frames).  Pure function of t."""
    a = t - t0
    if a < 0:
        return img
    x = x0 - speed * a
    if x < -10 or x > W + 10:
        return img
    ph = (a * 1.9 + noise.rand(seed, 1, 5)) % 1.0                    # hop cycle
    y = y_ground - 3 - hop * 4 * ph * (1 - ph)
    fr = TUMBLE[int(a * spin) % len(TUMBLE)]
    shadow_ellipse(img, x + 0.5, y_ground - 0.2, 3.0, 0.6, steps=1.0)
    blit(img, fr, x, y, anchor='c')
    return img


def shard_glint(img, x, y, t, times, dur=0.13):
    """A glass shard catching the sun now and then: a tiny twinkle."""
    for t0 in times:
        a = t - t0
        if 0 <= a < dur:
            k = int(a * 30)
            size = (1, 2, 1, 1)[min(k, 3)]
            sparkle(img, x, y, size, cols=('white0', 'dust10', 'dust9', 'dust8'))
            return img
    draw.pset(img, x, y, 'dust8')                                      # the shard itself
    return img
