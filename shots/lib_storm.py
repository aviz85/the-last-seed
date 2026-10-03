"""
shots/lib_storm.py — helpers for S5 THE STORM (owner: STORM / FX artist).

The storm is composited in PALETTE-INDEX space: frames are (h, w) uint8 arrays
of indices into engine.palette.PAL.  Fog, light, flashes and grading are then
cheap LUT lookups with ordered dithering between steps, and every pixel is
palette-true by construction (LUTs are restricted to the dead-world ramps, so
no stray meadow/gold/bark colours can sneak in).

Nothing here keeps per-frame state: every function is a pure function of its
arguments (static tables are built once at import).
"""
import math
import numpy as np

from engine import palette as P
from engine.core import BAYER4, BAYER8
from engine.draw import line_points, polygon_mask

# ---------------------------------------------------------------- palette
NAME = P.NAME_TO_IDX


def I(name):
    """Palette index of a colour name."""
    return NAME[name]


def RI(ramp):
    """Palette indices of a ramp (dark -> light)."""
    return list(P.RAMP_IDX[ramp])


STORM_RAMPS = ('ink', 'dust', 'ash', 'ochre', 'rust', 'steel', 'paint', 'core', 'leaf', 'white')
ALLOWED = np.array(sorted(sum((P.RAMP_IDX[r] for r in STORM_RAMPS), [])))
_ACCENT = np.zeros(P.N, bool)
for _r in ('core', 'leaf'):
    _ACCENT[P.RAMP_IDX[_r]] = True
PAL_LAB = P.PAL_LAB
PAL_LIN = P.PAL_LIN


def _nearest_lab(lab, allowed=ALLOWED):
    d = ((lab[..., None, :] - PAL_LAB[allowed]) ** 2).sum(-1)
    return allowed[d.argmin(-1)]


def lab_of(name):
    return P.linear_to_oklab(P.srgb_to_linear(np.array(P.C(name), np.float64)))


WARM_RAMPS = ('ink', 'dust', 'ochre', 'rust', 'core', 'white')
WARM = np.array(sorted(sum((P.RAMP_IDX[r] for r in WARM_RAMPS), [])))
_IS_WARM = np.zeros(P.N, bool)
for _r in ('ink', 'dust', 'ochre', 'rust'):
    _IS_WARM[P.RAMP_IDX[_r]] = True


def mix_lut(target, K=8, protect=0.3, allowed=ALLOWED, lift=0.0, warm_lock=True):
    """LUT[k, idx]: palette colour idx mixed k/K of the way toward `target`
    (in OKLab, perceptually even steps), snapped to the storm palette.
    Accent colours (core, leaf) only mix `protect` as much (they stay alive).
    warm_lock: warm inputs (ink/dust/ochre/rust) only map to warm ramps, so a
    darkened/fogged ochre never drifts into ash or steel blue-greys.
    lift: extra lightness added at full mix (for glow-type mixes)."""
    tl = lab_of(target)
    lut = np.zeros((K + 1, P.N), np.uint8)
    for k in range(K + 1):
        a = k / K
        aa = np.where(_ACCENT, a * protect, a)[:, None]
        lab = PAL_LAB * (1 - aa) + tl[None] * aa
        if lift:
            lab = lab.copy()
            lab[:, 0] += lift * a
        row = _nearest_lab(lab, allowed)
        if warm_lock:
            row = np.where(_IS_WARM, _nearest_lab(lab, WARM), row)
        lut[k] = row
    lut[0] = np.arange(P.N)
    return lut


def light_lut(tint='core4', mul=0.5, add=0.06, K=8, shadow='ochre0', sh_mul=0.64, sh_mix=0.35,
              allowed=ALLOWED, sat=1.0):
    """Signed light LUT like engine.light but restricted to the storm ramps.
    lut[k + K, idx] for k in -K..K (positive = lit by `tint`, negative =
    darker toward `shadow`)."""
    tint_l = P.srgb_to_linear(np.array(P.C(tint), np.float64))
    tint_n = tint_l / max(tint_l.max(), 1e-6)
    sh = P.srgb_to_linear(np.array(P.C(shadow), np.float64))
    shn = sh / max(sh.mean(), 1e-6)
    lut = np.zeros((2 * K + 1, P.N), np.uint8)
    for k in range(-K, K + 1):
        if k == 0:
            lut[K] = np.arange(P.N)
            continue
        if k > 0:
            c = PAL_LIN * (1 + mul * k * tint_n) + add * k * tint_n
        else:
            f = sh_mul ** (-k)
            c = PAL_LIN * f
            lum = c.mean(-1, keepdims=True)
            c = c * (1 - sh_mix) + lum * shn * sh_mix
        lab = P.linear_to_oklab(np.clip(c, 0, 1))
        if sat != 1.0:
            lab[:, 1:] *= sat
        lut[k + K] = _nearest_lab(lab, allowed)
    return lut


# ---------------------------------------------------------------- dithering
_BCACHE = {}


def bayer_tile(h, w, n=4, ox=0, oy=0):
    """Tiled Bayer thresholds (h,w) float32, pattern shifted by integer ox/oy
    (lock the pattern to a moving layer). Cached."""
    key = (h, w, n)
    if key not in _BCACHE:
        m = {4: BAYER4, 8: BAYER8}[n].astype(np.float32)
        _BCACHE[key] = np.tile(m, (h // n + 2, w // n + 2))
    big = _BCACHE[key]
    oy %= n
    ox %= n
    return big[oy:oy + h, ox:ox + w]


def quantize(level, K, ox=0, oy=0, band=0.45, n=4):
    """Float level map -> int steps 0..K with ordered-dither seams."""
    lv = np.asarray(level, np.float32)
    h, w = lv.shape
    fl = np.floor(lv)
    fr = lv - fl
    if band < 1.0:
        fr = np.clip((fr - 0.5) / max(band, 1e-3) + 0.5, 0.0, 1.0)
    q = fl + (bayer_tile(h, w, n, ox, oy) < fr)
    return np.clip(q, 0, K).astype(np.intp)


def apply_lut(idx, level, lut, ox=0, oy=0, band=0.45, n=4, signed=False, mask=None):
    """Apply a level LUT to an index image. level: scalar or (h,w) floats in
    [0, K] (or [-K, K] when signed). Returns a new index image."""
    K = (lut.shape[0] - 1) // 2 if signed else lut.shape[0] - 1
    h, w = idx.shape
    lv = np.asarray(level, np.float32)
    if lv.ndim == 0:
        lv = np.full((h, w), float(lv), np.float32)
    if signed:
        q = quantize(lv + K, 2 * K, ox, oy, band, n)
    else:
        q = quantize(lv, K, ox, oy, band, n)
    out = lut[q, idx]
    if mask is not None:
        out = np.where(mask, out, idx)
    return out


def dmask(alpha, ox=0, oy=0, n=4):
    """Dither coverage mask for alpha (h,w) in 0..1."""
    a = np.asarray(alpha, np.float32)
    return bayer_tile(a.shape[0], a.shape[1], n, ox, oy) < a


# ---------------------------------------------------------------- noise (3D, coarse)
def rand3(ix, iy, iz, seed=0):
    with np.errstate(over='ignore'):
        x = np.asarray(ix).astype(np.int64).astype(np.uint32)
        y = np.asarray(iy).astype(np.int64).astype(np.uint32)
        z = np.asarray(iz).astype(np.int64).astype(np.uint32)
        h = (x * np.uint32(374761393) + y * np.uint32(668265263) + z * np.uint32(3266489917)
             + np.uint32((seed * 2246822519 + 12345) & 0xFFFFFFFF))
        h = (h ^ (h >> np.uint32(13))) * np.uint32(1274126177)
        h = h ^ (h >> np.uint32(16))
        h = h * np.uint32(0x85EBCA6B)
        h = h ^ (h >> np.uint32(15))
    return h.astype(np.float32) * np.float32(1.0 / 4294967296.0)


def _fade(t):
    return t * t * (3 - 2 * t)


def vnoise3(x, y, z, seed=0):
    x = np.asarray(x, np.float32)
    y = np.asarray(y, np.float32)
    z = np.asarray(z, np.float32)
    xi, yi, zi = np.floor(x), np.floor(y), np.floor(z)
    fx, fy, fz = _fade(x - xi), _fade(y - yi), _fade(z - zi)
    xi, yi, zi = xi.astype(np.int64), yi.astype(np.int64), zi.astype(np.int64)
    c000 = rand3(xi, yi, zi, seed); c100 = rand3(xi + 1, yi, zi, seed)
    c010 = rand3(xi, yi + 1, zi, seed); c110 = rand3(xi + 1, yi + 1, zi, seed)
    c001 = rand3(xi, yi, zi + 1, seed); c101 = rand3(xi + 1, yi, zi + 1, seed)
    c011 = rand3(xi, yi + 1, zi + 1, seed); c111 = rand3(xi + 1, yi + 1, zi + 1, seed)
    a = c000 + (c100 - c000) * fx
    b = c010 + (c110 - c010) * fx
    c = c001 + (c101 - c001) * fx
    d = c011 + (c111 - c011) * fx
    e = a + (b - a) * fy
    f = c + (d - c) * fy
    return e + (f - e) * fz


def fbm3(x, y, z, seed=0, octaves=4, lac=2.0, gain=0.5, billow=False):
    """fBm of 3D value noise. billow=True uses |2n-1| (puffy, creased shapes)."""
    amp, tot, norm = 1.0, 0.0, 0.0
    x = np.asarray(x, np.float32); y = np.asarray(y, np.float32); z = np.asarray(z, np.float32)
    for o in range(octaves):
        n = vnoise3(x, y, z, seed + o * 131)
        if billow:
            n = 1.0 - np.abs(2.0 * n - 1.0)
        tot = tot + amp * n
        norm += amp
        amp *= gain
        x = x * lac + 17.3
        y = y * lac + 9.1
        z = z * 1.7
    return tot / norm


# ---------------------------------------------------------------- tileable 2D texture
def tile_noise(h, w, cell, seed, octaves=3):
    """Static value-noise texture (h,w) float32, tileable in x (period w)."""
    out = np.zeros((h, w), np.float32)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        c = max(1, int(round(cell / (2 ** o))))
        nx = max(1, w // c)
        cw = w / nx
        ys = np.arange(h, dtype=np.float32) / c
        xs = np.arange(w, dtype=np.float32) / cw
        xi = np.floor(xs).astype(np.int64); yi = np.floor(ys).astype(np.int64)
        fx = _fade(xs - xi)[None, :]; fy = _fade(ys - yi)[:, None]
        X0 = (xi % nx)[None, :]; X1 = ((xi + 1) % nx)[None, :]
        Y0 = yi[:, None]; Y1 = (yi + 1)[:, None]
        s = seed + o * 77
        a = rand3(X0, Y0, 0, s); b = rand3(X1, Y0, 0, s)
        cc = rand3(X0, Y1, 0, s); d = rand3(X1, Y1, 0, s)
        v = (a + (b - a) * fx) * (1 - fy) + (cc + (d - cc) * fx) * fy
        out += amp * v
        norm += amp
        amp *= 0.5
    return out / norm


# ---------------------------------------------------------------- fast bilinear upsampling
class Up:
    """Bilinear upsampler from a coarse node grid (spacing s px, nodes at
    screen coords (j-1)*s) to a full (h,w) field. Gather-based (no BLAS, so no
    thread spin-up when frames render in parallel worker processes)."""

    def __init__(self, h, w, s):
        self.h, self.w, self.s = h, w, s
        self.hc = h // s + 3
        self.wc = w // s + 3
        self.xc = ((np.arange(self.wc) - 1) * s).astype(np.float32)
        self.yc = ((np.arange(self.hc) - 1) * s).astype(np.float32)
        py = np.arange(h); px = np.arange(w)
        self.jy = py // s + 1
        self.fy = ((py % s) / float(s)).astype(np.float32)[:, None]
        self.jx = px // s + 1
        self.fx = ((px % s) / float(s)).astype(np.float32)[None, :]

    def __call__(self, coarse):
        c = np.asarray(coarse, np.float32)
        a = c[self.jy] * (1 - self.fy) + c[self.jy + 1] * self.fy
        return a[:, self.jx] * (1 - self.fx) + a[:, self.jx + 1] * self.fx


# ---------------------------------------------------------------- drawing (index space)
def put_points(img, xs, ys, val):
    xs = np.asarray(xs).astype(np.int64).ravel()
    ys = np.asarray(ys).astype(np.int64).ravel()
    h, w = img.shape[:2]
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    v = np.asarray(val)
    if v.ndim == 0:
        img[ys[ok], xs[ok]] = v
    else:
        img[ys[ok], xs[ok]] = np.broadcast_to(v.ravel() if v.size == xs.size else v, xs.shape)[ok]


def put_line(img, x0, y0, x1, y1, val):
    xs, ys = line_points(x0, y0, x1, y1)
    put_points(img, xs, ys, val)


def put_mask(img, mask, val, x=0, y=0):
    """Write val where mask (hm,wm) is True, mask top-left at (x,y)."""
    x, y = int(round(x)), int(round(y))
    hm, wm = mask.shape
    h, w = img.shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + wm, w), min(y + hm, h)
    if x0 >= x1 or y0 >= y1:
        return
    m = mask[y0 - y:y1 - y, x0 - x:x1 - x]
    sub = img[y0:y1, x0:x1]
    if np.ndim(val) == 0:
        sub[m] = val
    else:
        sub[m] = val[y0 - y:y1 - y, x0 - x:x1 - x][m]


def sprite_to_idx(rgba):
    """RGBA sprite -> (idx (h,w) uint8, mask (h,w) bool)."""
    from engine.core import to_index
    idx = to_index(np.ascontiguousarray(rgba[..., :3]))
    return idx, rgba[..., 3] > 127


def paste_idx(dst, sidx, smask, x, y):
    """Paste an index sprite (idx, mask) with top-left at (x, y)."""
    x, y = int(round(x)), int(round(y))
    hm, wm = smask.shape
    h, w = dst.shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + wm, w), min(y + hm, h)
    if x0 >= x1 or y0 >= y1:
        return
    m = smask[y0 - y:y1 - y, x0 - x:x1 - x]
    s = sidx[y0 - y:y1 - y, x0 - x:x1 - x]
    d = dst[y0:y1, x0:x1]
    d[m] = s[m]


def disc_mask(r):
    """Pixel-art disc mask of radius r (float), centred in a (2R+1)^2 box."""
    R = int(math.ceil(r)) + 1
    ys, xs = np.mgrid[-R:R + 1, -R:R + 1]
    return (xs * xs + ys * ys) <= (r + 0.35) ** 2, R


# ---------------------------------------------------------------- time integration
class Integral:
    """Pure-function time integral of a speed curve f(t) (pre-tabulated):
    X(t) = int_0^t f. Used so particles/clouds can move at gust-varying speed
    while staying a pure function of t."""

    def __init__(self, f, t0=-1.0, t1=8.0, dt=1.0 / 240):
        self.ts = np.arange(t0, t1 + dt, dt)
        v = np.array([f(t) for t in self.ts])
        c = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) * 0.5 * dt)])
        c -= np.interp(0.0, self.ts, c)
        self.c = c

    def __call__(self, t):
        return float(np.interp(t, self.ts, self.c))


# ================================================================ STORM ART
TEX = tile_noise(256, 1024, 6, 7, octaves=3)          # dusty edge / tone breakup texture
TEXF = tile_noise(256, 1024, 3, 19, octaves=2)        # finer grain


def make_puffs_row(t, off, seed, spacing, base_y, r0, r1, w, yjit=8.0, grow=0.12, hang=False,
                   squash=0.8, margin=80, rise=0.3, rate=(0.3, 0.6), gate=None):
    """Puffs of a horizontally wrapping bank (world-locked, slid by `off` px).
    gate(i_array) -> bool array: optional per-puff presence (gaps = surges).
    Returns list of (cx, cy, r, squash, key) in canvas coords."""
    i0 = int(math.floor((off - margin) / spacing))
    i1 = int(math.ceil((off + w + margin) / spacing))
    i = np.arange(i0, i1 + 1)
    rr = rand1(i, 1, seed); ry = rand1(i, 2, seed); ph = rand1(i, 3, seed); rx = rand1(i, 4, seed)
    r = r0 + (r1 - r0) * rr
    r = r * (1.0 + grow * np.sin(2 * np.pi * (t * (rate[0] + rate[1] * ph) + ph)))
    cx = i * spacing + (rx - 0.5) * spacing * 0.7 - off
    cy = base_y + (ry - 0.5) * 2 * yjit + (r * rise if hang else -r * rise)
    ok = np.ones(len(i), bool) if gate is None else gate(i)
    return [(float(a), float(b), float(c), squash, int(k)) for a, b, c, k, o in zip(cx, cy, r, i, ok) if o]


def rand1(i, j, seed):
    from engine.noise import rand
    return rand(i, j, seed)


def draw_puffs(img, puffs, tones, ox=0, light=(-0.5, -0.87), edge=0.15, texk=0.34, skew=0.25,
               tex_amp=0.5, lit_gain=1.5, lit_bias=1.3, band=0.3, order='y', hang=False,
               mask_out=None, clip=None, rim=None, rim_w=0.16, rim_dir=(-0.8, -0.6), rim_min=0.2,
               rim_tex=0.0):
    """Paint shaded dust puffs (painter's order) into an index image.
    tones: [shadow, body, light, highlight...] palette indices (dark->light).
    ox: world offset for texture locking. mask_out: optional bool canvas that
    receives the covered pixels. clip: optional (x_lo_fn) — pixels with
    x < clip(y) are skipped (used by the storm front)."""
    h, w = img.shape
    lx, ly = light
    tones = np.asarray(tones, np.uint8)
    nt = len(tones) - 1
    bt = bayer_tile(h, w, 4)
    if order == 'y':
        puffs = sorted(puffs, key=lambda p: (-p[1] if hang else p[1]) + p[2] * 0.1)
    puffs = list(puffs)
    for p in puffs:
        cx, cy, r, sq = p[0], p[1], p[2], p[3]
        rx, ry = r, r * sq
        x0, x1 = int(max(cx - rx - r * skew - 3, 0)), int(min(cx + rx + r * skew + 3, w))
        y0, y1 = int(max(cy - ry - 2, 0)), int(min(cy + ry + 3, h))
        if x0 >= x1 or y0 >= y1:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        ny = (ys - cy) / ry
        nx = (xs - cx + ny * ry * skew) / rx
        d = np.sqrt(nx * nx + ny * ny)
        tex = TEX[ys & 255, (xs + ox) & 1023]
        dd = d + (tex - 0.5) * texk
        m = dd + (bt[y0:y1, x0:x1] - 0.5) * edge < 1.0
        if clip is not None:
            m &= xs >= clip(ys)
        if not m.any():
            continue
        lit = (nx * lx + ny * ly) / np.maximum(d, 0.35)
        lit = lit * np.clip(d * 1.3, 0, 1) + (tex - 0.5) * tex_amp
        lvl = np.clip(lit_bias + lit * lit_gain, 0, nt)
        q = quantize(lvl, nt, band=band)
        sub = img[y0:y1, x0:x1]
        sub[m] = tones[q][m]
        if rim is not None:
            facing = (nx * rim_dir[0] + ny * rim_dir[1]) / np.maximum(d, 0.35)
            rm = m & (dd > 1.0 - rim_w) & (facing > rim_min)
            if rim_tex:
                rm &= TEXF[ys & 255, (xs + ox) & 1023] > rim_tex
            sub[rm] = rim
        if mask_out is not None:
            mask_out[y0:y1, x0:x1] |= m
    return img


def bank_body(img, puffs, tone, hang=False, mask_out=None, x_lo=None, y_to=None):
    """Solid body under (or above, hang) the curve through the puff centres,
    down to y_to (the bank's ground-contact line) or the canvas edge."""
    h, w = img.shape
    if not puffs:
        return None
    ps = sorted(puffs, key=lambda p: p[0])
    xs = np.array([p[0] for p in ps]); ys = np.array([p[1] for p in ps])
    yb = np.interp(np.arange(w), xs, ys)
    Y = np.arange(h)[:, None]
    m = (Y >= yb[None, :]) if not hang else (Y <= yb[None, :])
    if y_to is not None:
        m &= (Y <= y_to) if not hang else (Y >= y_to)
    if x_lo is not None:
        m &= np.arange(w)[None, :] >= x_lo
    img[m] = tone
    if mask_out is not None:
        mask_out |= m
    return m


def streak_field(img, t, n, seed, area, travel, length, colors, weights=None, slope=0.0,
                 wave=(0.0, 0.0, 0.0), density=None, dens_thresh=0.0, step=1, keep=None, thick=1):
    """Wind streaks: n particles wrapping in `area` (x, y, w, h), moved left by
    `travel` px (the integrated wind * layer speed, + per-particle speed jitter),
    each drawn as a horizontal-ish line of `length` (lo, hi) px trailing to the
    RIGHT (they fly leftward). wave = (amp_px, wavelength_px, speed) gives the
    streams a gentle undulation. density(x, y) -> 0..1 optional gating field so
    particles cluster into streams. keep(x, y) -> bool optional mask."""
    ax, ay, aw, ah = area
    i = np.arange(n)
    r = rand1(i, 9, seed)
    spd = 0.75 + 0.5 * rand1(i, 3, seed)
    x = ax + np.mod(rand1(i, 1, seed) * aw - travel * spd, aw)
    y = ay + rand1(i, 2, seed) * ah
    if wave[0]:
        y = y + wave[0] * np.sin((x + travel * wave[2]) / max(wave[1], 1e-3) * 2 * np.pi + rand1(i, 5, seed) * 6.28)
    L = length[0] + (length[1] - length[0]) * rand1(i, 4, seed) * (0.6 + 0.4 * spd)
    ok = np.ones(n, bool)
    if density is not None:
        ok &= density(x, y) > dens_thresh + 0.25 * rand1(i, 6, seed)
    if keep is not None:
        ok &= keep(x, y)
    x, y, L, r = x[ok], y[ok], L[ok], r[ok]
    if len(x) == 0:
        return
    k = len(colors)
    if weights is None:
        ci = np.minimum((r * k).astype(int), k - 1)
    else:
        edges = np.cumsum(weights) / np.sum(weights)
        ci = np.searchsorted(edges, r, side='right').clip(0, k - 1)
    cols = np.asarray(colors, np.uint8)[ci]
    Lmax = int(np.ceil(L.max()))
    h, w = img.shape
    for s in range(0, Lmax, step):
        live = L > s
        if not live.any():
            break
        px = (x[live] + s).astype(np.int64)
        py = np.floor(y[live] + slope * s).astype(np.int64)
        for dyy in range(thick):
            good = (px >= 0) & (px < w) & (py + dyy >= 0) & (py + dyy < h)
            if dyy and s == 0:
                continue
            img[py[good] + dyy, px[good]] = cols[live][good]


def rot_pts(pts, ang):
    c, s = math.cos(ang), math.sin(ang)
    p = np.asarray(pts, np.float64)
    return np.stack([p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c], 1)


def radial_level_c(h, w, cx, cy, radius, peak, gamma=1.0, sy=1.0, core=0.0):
    """Radial level map (h,w) float32: peak at (cx,cy) falling to 0 at radius."""
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    d = np.sqrt((xs - cx) ** 2 + ((ys - cy) / max(sy, 1e-6)) ** 2)
    d = np.maximum(d - core, 0)
    v = np.clip(1 - d / max(radius - core, 1e-6), 0, 1) ** gamma
    return (peak * v).astype(np.float32)


# ---------------------------------------------------------------- ramp stepping
def step_lut(k):
    """lut[idx] -> the colour k steps lighter (k>0) / darker (k<0) in its own ramp."""
    lut = np.arange(P.N, dtype=np.uint8)
    for i in range(P.N):
        rname, pos = P.RAMP_OF[i], P.POS_OF[i]
        ids = P.RAMP_IDX[rname]
        lut[i] = ids[int(np.clip(pos + k, 0, len(ids) - 1))]
    return lut


STEP = {k: step_lut(k) for k in (-3, -2, -1, 1, 2, 3)}


def streak_lut_field(img, n, seed, area, travel, length, lut, slope=0.0, wave=(0.0, 0.0, 0.0),
                     density=None, dens_thresh=0.0, keep=None, thick=1, taper=True):
    """Like streak_field but each streak re-maps the pixels under it through
    `lut` (e.g. STEP[1] = one ramp-step lighter): veils that adapt to the local
    tone instead of painting a fixed colour. taper: the tail (right end) of
    each streak is broken up with a dither so it thins out."""
    ax, ay, aw, ah = area
    i = np.arange(n)
    spd = 0.75 + 0.5 * rand1(i, 3, seed)
    x = ax + np.mod(rand1(i, 1, seed) * aw - travel * spd, aw)
    y = ay + rand1(i, 2, seed) * ah
    if wave[0]:
        y = y + wave[0] * np.sin((x + travel * wave[2]) / max(wave[1], 1e-3) * 2 * np.pi + rand1(i, 5, seed) * 6.28)
    Ls = length[0] + (length[1] - length[0]) * rand1(i, 4, seed) * (0.6 + 0.4 * spd)
    ok = np.ones(n, bool)
    if density is not None:
        ok &= density(x, y) > dens_thresh + 0.25 * rand1(i, 6, seed)
    if keep is not None:
        ok &= keep(x, y)
    x, y, Ls = x[ok], y[ok], Ls[ok]
    if len(x) == 0:
        return
    h, w = img.shape
    Lmax = int(np.ceil(Ls.max()))
    xs_all, ys_all = [], []
    for s in range(Lmax):
        live = Ls > s
        if taper:
            frac = s / np.maximum(Ls, 1)
            live &= (frac < 0.55) | (BAYER4[(np.floor(y).astype(int) & 3), (s + np.floor(x).astype(int)) & 3] > (frac - 0.55) / 0.45)
        if not live.any():
            continue
        px = (x[live] + s).astype(np.int64)
        py = np.floor(y[live] + slope * s).astype(np.int64)
        xs_all.append(px); ys_all.append(py)
        if thick > 1:
            xs_all.append(px); ys_all.append(py + 1)
    px = np.concatenate(xs_all); py = np.concatenate(ys_all)
    good = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    px, py = px[good], py[good]
    # de-duplicate so a pixel is stepped once
    key = np.unique(py * w + px)
    py, px = key // w, key % w
    img[py, px] = lut[img[py, px]]


def parabola_hops(t, t0, hops):
    """Height (px, >=0) of a bouncing object: hops = [(duration, height), ...]
    starting at t0; returns (height, index_of_hop, u)."""
    a = t - t0
    if a < 0:
        return 0.0, -1, 0.0
    for k, (d, hgt) in enumerate(hops):
        if a < d:
            u = a / d
            return 4 * hgt * u * (1 - u), k, u
        a -= d
    return 0.0, len(hops), 1.0


# ================================================================ DEBRIS (geometry, redrawn per frame — never resampled)
def _bush_geometry(seed=31, R=6.5):
    """A dead tumbleweed: a tangle of curved twigs inside a sphere of radius R.
    Returns (pts (n,2) float, shade (n,) int 0 dark/1 mid/2 light)."""
    pts, shade = [], []
    for k in range(15):
        r = R * (0.45 + 0.55 * float(rand1(k, 1, seed)))
        a0 = float(rand1(k, 2, seed)) * 2 * math.pi
        span = 1.2 + 2.0 * float(rand1(k, 3, seed))
        tilt = 0.35 + 0.65 * float(rand1(k, 4, seed))        # ellipse squash: arcs on a sphere
        rot = float(rand1(k, 5, seed)) * math.pi
        n = int(span * r * 2.2) + 2
        a = a0 + np.linspace(0, span, n)
        x = r * np.cos(a); y = r * np.sin(a) * tilt
        c, s = math.cos(rot), math.sin(rot)
        px, py = x * c - y * s, x * s + y * c
        pts.append(np.stack([px, py], 1))
        sh = 0 if k % 3 else 1
        shade += [sh] * n
    for k in range(7):                                       # spiky twigs poking out
        a = float(rand1(k, 7, seed)) * 2 * math.pi
        r0, r1 = R * 0.5, R * (1.05 + 0.3 * float(rand1(k, 8, seed)))
        n = 6
        rr = np.linspace(r0, r1, n)
        pts.append(np.stack([rr * math.cos(a), rr * math.sin(a)], 1))
        shade += [2 if k % 2 == 0 else 1] * n
    return np.concatenate(pts, 0), np.array(shade)


BUSH_PTS, BUSH_SHADE = _bush_geometry()


def draw_bush(img, cx, cy, ang, cols):
    """cols: [dark, mid, light] palette indices."""
    p = rot_pts(BUSH_PTS, ang)
    xs = np.floor(cx + p[:, 0]).astype(np.int64)
    ys = np.floor(cy + p[:, 1]).astype(np.int64)
    cols = np.asarray(cols, np.uint8)
    for s in (0, 1, 2):
        m = BUSH_SHADE == s
        put_points(img, xs[m], ys[m], cols[s])


def draw_sheet(img, cx, cy, phi, theta, cols, w=13.0, h=7.0, mask_out=None):
    """A tumbling sheet of corrugated metal. phi: rotation about its long axis
    (tumble, foreshortens), theta: in-plane spin. cols: [edge_dark, back, front_dark,
    front, front_light, glint] palette indices."""
    hw, hh = w / 2.0, h / 2.0
    cph = math.cos(phi)
    corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], np.float64)
    corners[:, 1] *= cph
    # slight bend (the sheet is buckled): push the right end down a touch
    c = rot_pts(corners, theta)
    c[:, 0] += cx; c[:, 1] += cy
    x0, y0 = int(math.floor(c[:, 0].min())) - 1, int(math.floor(c[:, 1].min())) - 1
    x1, y1 = int(math.ceil(c[:, 0].max())) + 2, int(math.ceil(c[:, 1].max())) + 2
    hbox, wbox = y1 - y0, x1 - x0
    if hbox <= 0 or wbox <= 0:
        return
    m = polygon_mask(hbox, wbox, c - [x0, y0])
    front = cph > 0
    ac = abs(cph)
    if ac < 0.22:
        # edge-on: a thin bright/dark line
        xs, ys = line_points(c[0, 0] * 0.5 + c[3, 0] * 0.5, c[0, 1] * 0.5 + c[3, 1] * 0.5,
                             c[1, 0] * 0.5 + c[2, 0] * 0.5, c[1, 1] * 0.5 + c[2, 1] * 0.5)
        put_points(img, xs, ys, cols[3] if front else cols[1])
        if mask_out is not None:
            put_points(mask_out, xs, ys, True)
        return
    if not m.any():
        return
    base = cols[3] if front else cols[1]
    sub = np.full((hbox, wbox), base, np.uint8)
    # corrugation stripes along the long axis
    for k, sy in enumerate(np.arange(-hh + 1.5, hh - 0.5, 2.0)):
        a = rot_pts([[-hw + 0.5, sy * cph], [hw - 0.5, sy * cph]], theta)
        xs, ys = line_points(a[0, 0] + cx - x0, a[0, 1] + cy - y0, a[1, 0] + cx - x0, a[1, 1] + cy - y0)
        put_points(sub, xs, ys, (cols[4] if front else cols[2]) if k % 2 == 0 else (cols[2] if front else cols[0]))
    # rim: darker outline of the plate
    from engine.draw import outline_mask
    rim = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    sub[rim] = cols[0] if not front else cols[2]
    # glint when it faces the viewer squarely
    if front and ac > 0.93:
        gx, gy = rot_pts([[-hw * 0.4, -hh * 0.4 * cph]], theta)[0]
        sub[int(np.clip(gy + cy - y0, 0, hbox - 1)), int(np.clip(gx + cx - x0, 0, wbox - 1))] = cols[5]
    tmp = img[max(y0, 0):y1, max(x0, 0):x1]
    mm = m[max(y0, 0) - y0:max(y0, 0) - y0 + tmp.shape[0], max(x0, 0) - x0:max(x0, 0) - x0 + tmp.shape[1]]
    ss = sub[max(y0, 0) - y0:max(y0, 0) - y0 + tmp.shape[0], max(x0, 0) - x0:max(x0, 0) - x0 + tmp.shape[1]]
    tmp[mm] = ss[mm]
    if mask_out is not None:
        mo = mask_out[max(y0, 0):y1, max(x0, 0):x1]
        mo[mm] = True


# antenna piece (matches chars/model.py: 'K' tip, 'KYK' bulb, 'S','S' rod)
ANT_PTS = np.array([[0, 0], [0, -1], [0, -2], [-1, -2], [1, -2], [0, -3]], np.float64)   # rod(2), bulb row(3), tip
ANT_ROLE = np.array([0, 0, 1, 2, 2, 3])        # 0 rod, 1 bulb, 2 bulb rim, 3 tip


def draw_antenna(img, x, y, ang, glow_col, rod_col, rim_col):
    p = rot_pts(ANT_PTS - [0, -1.5], ang)
    xs = np.round(x + p[:, 0]).astype(np.int64)
    ys = np.round(y + p[:, 1]).astype(np.int64)
    for role, col in ((0, rod_col), (2, rim_col), (3, rim_col), (1, glow_col)):
        m = ANT_ROLE == role
        put_points(img, xs[m], ys[m], col)


def sparks(img, t, t0, n, seed, origin, speed=(60, 220), life=(0.12, 0.55), gravity=320.0,
           drag_x=-160.0, spread=(0, 2 * math.pi), cols=None, tail=1.0, floor=None):
    """One-shot spark burst drawn as short motion streaks; colour cools with age.
    cols: list of palette indices hot -> cold. floor(x) -> ground y: sparks
    that reach the ground stop there and fizzle (1 px ember, cooling)."""
    a = t - t0
    if a < 0:
        return
    i = np.arange(n)
    ang = spread[0] + rand1(i, 11, seed) * (spread[1] - spread[0])
    sp = speed[0] + rand1(i, 12, seed) * (speed[1] - speed[0])
    lf = life[0] + rand1(i, 13, seed) * (life[1] - life[0])
    alive = a <= lf
    if not alive.any():
        return
    vx = np.cos(ang) * sp; vy = np.sin(ang) * sp

    def pos(tt):
        return (origin[0] + vx * tt + 0.5 * drag_x * tt * tt,
                origin[1] + vy * tt + 0.5 * gravity * tt * tt)
    x1, y1 = pos(a)
    x0, y0 = pos(max(a - tail / 30.0, 0))
    age = np.clip(a / lf, 0, 0.999)
    k = len(cols)
    ci = (age * k).astype(int)
    for j in np.nonzero(alive)[0]:
        if floor is not None:
            gy = floor(x1[j])
            if y1[j] >= gy:
                # landed: a cooling ember on the sand
                put_points(img, [x1[j]], [gy - 1], cols[min(ci[j] + 1, k - 1)])
                continue
        xs, ys = line_points(x0[j], y0[j], x1[j], y1[j])
        put_points(img, xs, ys, cols[ci[j]])


# ================================================================ wind-bent vegetation (forward kinematics, redrawn per frame)
def make_tree(seed=5):
    """A dead tree: list of segments (parent, length, rest_angle, flex, width).
    Angles in radians, 0 = up, positive = leaning right (clockwise)."""
    segs = [(-1, 11.0, -0.12, 0.04, 3), (0, 9.0, -0.05, 0.08, 2), (1, 8.0, 0.1, 0.14, 2)]
    # branches: (parent, length, rel_angle, flex, width)
    extra = [(0, 9.0, 0.95, 0.35, 1), (1, 10.0, -0.9, 0.3, 1), (1, 7.0, 0.8, 0.4, 1),
             (2, 7.0, -0.7, 0.45, 1), (2, 6.0, 0.6, 0.5, 1), (3, 5.0, 0.5, 0.6, 1),
             (4, 5.0, -0.6, 0.6, 1), (5, 4.0, 0.4, 0.7, 1), (2, 5.0, 0.05, 0.5, 1),
             (6, 4.0, 0.7, 0.8, 1), (7, 3.0, -0.5, 0.8, 1), (8, 3.5, 0.5, 0.8, 1)]
    return segs + extra


def tree_points(segs, bx, by, bend, flutter, detached=None):
    """Forward kinematics. bend: global wind bend (rad, negative = to the left);
    flutter(k) -> extra angle per segment. Returns list of (x0,y0,x1,y1,width,k)."""
    ang = [0.0] * len(segs)
    pos = [(0.0, 0.0)] * len(segs)
    out = []
    for k, (par, ln, rel, flex, wd) in enumerate(segs):
        if par < 0:
            a0, (x0, y0) = 0.0, (bx, by)
        else:
            a0 = ang[par]
            x0, y0 = pos[par]
        a = a0 + rel + bend * flex * 2.2 + flutter(k) * flex
        x1, y1 = x0 + math.sin(a) * ln, y0 - math.cos(a) * ln
        ang[k] = a
        pos[k] = (x1, y1)
        if detached is not None and k in detached:
            continue
        out.append((x0, y0, x1, y1, wd, k))
    return out, pos, ang


def draw_segments(img, segs, cols):
    """cols[width] -> palette index."""
    for (x0, y0, x1, y1, wd, k) in segs:
        xs, ys = line_points(x0, y0, x1, y1)
        xs = np.asarray(xs); ys = np.asarray(ys)
        c = cols.get(wd, cols[1])
        put_points(img, xs, ys, c)
        if wd >= 2:
            put_points(img, xs + 1, ys, c)
        if wd >= 3:
            put_points(img, xs - 1, ys + 1, c)


def tuft_blades(n, seed, h0, h1):
    out = []
    for k in range(n):
        out.append(((float(rand1(k, 1, seed)) - 0.5) * 4.0, h0 + (h1 - h0) * float(rand1(k, 2, seed)),
                    (float(rand1(k, 3, seed)) - 0.5) * 0.6, float(rand1(k, 4, seed))))
    return out


def draw_tuft(img, x, y, blades, bend, t, col, tip):
    """Dead grass tuft: blades bend left with the wind and flutter."""
    for (dx, h, a0, ph) in blades:
        a = a0 + bend * (0.8 + 0.4 * ph) + 0.25 * math.sin(t * (13 + 6 * ph) + ph * 6.3) * min(1.0, abs(bend))
        # quadratic bend: the tip leans more than the base
        pts_x, pts_y = [], []
        for s in range(int(h) + 1):
            f = s / max(h, 1)
            pts_x.append(x + dx + math.sin(a * f * 1.3) * s)
            pts_y.append(y - math.cos(a * f) * s)
        xs = np.floor(np.array(pts_x)).astype(int); ys = np.floor(np.array(pts_y)).astype(int)
        put_points(img, xs[:-2], ys[:-2], col)
        put_points(img, xs[-2:], ys[-2:], tip)


# ================================================================ fog density variation
TEXL = tile_noise(256, 1024, 44, 23, octaves=2)       # large, soft blobs (fog density variation)


def fog_map(h, w, base, amp, off=0.0, sx=2.0, sy=1.0, yoff=0):
    """Spatially varying fog level: base + amp*(noise-0.5), noise sampled from a
    large soft texture slid by `off` px (so fog banks drift with the wind).
    Varying levels keep step changes as organic contour bands instead of a
    frame-wide dither."""
    ys = (np.arange(h) // max(int(sy), 1) + yoff) & 255
    xs = ((np.arange(w) + int(off)) / sx).astype(np.int64) & 1023
    return base + amp * (TEXL[ys[:, None], xs[None, :]] - 0.5)


# ================================================================ pixel-width dither seams
def quantize_px(level, K, w_px=1.5, ox=0, oy=0, n=4, min_band=0.03):
    """Like quantize() but the dithered zone between steps is ~2*w_px PIXELS
    wide wherever it falls (band scaled by the local gradient): solid bands
    with thin hand-made-looking seams, never a wide checkerboard."""
    lv = np.asarray(level, np.float32)
    h, w = lv.shape
    gx = np.zeros_like(lv); gy = np.zeros_like(lv)
    gx[:, 1:-1] = (lv[:, 2:] - lv[:, :-2]) * 0.5
    gy[1:-1, :] = (lv[2:, :] - lv[:-2, :]) * 0.5
    g = np.abs(gx) + np.abs(gy)
    band = np.clip(2.0 * w_px * g, min_band, 1.0)
    fl = np.floor(lv)
    fr = np.clip((lv - fl - 0.5) / band + 0.5, 0.0, 1.0)
    q = fl + (bayer_tile(h, w, n, ox, oy) < fr)
    return np.clip(q, 0, K).astype(np.intp)


def apply_lut_px(idx, level, lut, w_px=1.5, ox=0, oy=0, n=4):
    """apply_lut with pixel-width dither seams (unsigned LUTs)."""
    K = lut.shape[0] - 1
    lv = np.asarray(level, np.float32)
    if lv.ndim == 0:
        lv = np.full(idx.shape, float(lv), np.float32)
    q = quantize_px(lv, K, w_px, ox, oy, n)
    return lut[q, idx]


# ================================================================ windowed radial light (fast)
def apply_radial(img, cx, cy, radius, peak, lut, gamma=1.0, sy=1.0, mod=None, band=0.45, signed=False):
    """Apply a radial light/fog LUT only inside the light's bounding box.
    mod(ys, xs) -> optional multiplier array for the window (e.g. dust texture,
    robot mask). Returns a new image."""
    h, w = img.shape
    r = radius
    x0, x1 = int(max(cx - r - 1, 0)), int(min(cx + r + 2, w))
    y0, y1 = int(max(cy - r * sy - 1, 0)), int(min(cy + r * sy + 2, h))
    out = img.copy()
    if x0 >= x1 or y0 >= y1 or peak <= 0:
        return out
    ys = np.arange(y0, y1, dtype=np.float32)[:, None]
    xs = np.arange(x0, x1, dtype=np.float32)[None, :]
    d = np.sqrt((xs - cx) ** 2 + ((ys - cy) / max(sy, 1e-6)) ** 2)
    lv = peak * np.clip(1 - d / max(r, 1e-6), 0, 1) ** gamma
    if mod is not None:
        lv = lv * mod(y0, y1, x0, x1)
    out[y0:y1, x0:x1] = apply_lut(img[y0:y1, x0:x1], lv, lut, ox=x0, oy=y0, band=band, signed=signed)
    return out


# ================================================================ VOLUMETRIC LOBES (v2 clouds)
# Unit-sphere shading tables, sampled by index (procedural lookup — no art is resampled).
_LS = 50.0                                   # table samples per unit radius
_LN = int(2 * 1.4 * _LS) + 1                 # covers [-1.4, 1.4] so dusty edge pixels shade too
_lu = np.linspace(-1.4, 1.4, _LN).astype(np.float32)
_LUX, _LUY = np.meshgrid(_lu, _lu)
LOBE_D2 = (_LUX ** 2 + _LUY ** 2).astype(np.float32)
_LNZ = np.sqrt(np.clip(1.0 - LOBE_D2, 0.0, 1.0)).astype(np.float32)


def lobe_shade_table(lx=-0.55, ly=-0.72, lz=0.42):
    """Lambert shading of a unit sphere seen head-on, light from (lx, ly, lz)
    (x right, y down, z toward the viewer). Values ~[-1, 1]."""
    n = math.sqrt(lx * lx + ly * ly + lz * lz)
    return ((_LUX * lx + _LUY * ly + _LNZ * lz) / n).astype(np.float32)


SHADE_UL = lobe_shade_table()                          # the dying sun, upper left
SHADE_UL_LOW = lobe_shade_table(-0.8, -0.35, 0.5)      # low, raking (the wall's churning top edge)


_MR_CACHE = {}


def _mass_rand(seed, n):
    key = (seed, n)
    v = _MR_CACHE.get(key)
    if v is None:
        k = np.arange(n)
        v = tuple(rand1(k, j, seed) for j in range(1, 8))
        if len(_MR_CACHE) > 20000:
            _MR_CACHE.clear()
        _MR_CACHE[key] = v
    return v


def mass_lobes(t, cx, cy, w, h, seed, n, r0, r1, roll=0.45, pulse=0.14, sq=0.82, top=0.35):
    """A billowing dust mass: n lobes scattered in a (w x h) ellipse around
    (cx, cy). Internal motion: every lobe circulates slowly around the mass
    centre (top lobes travel left = the mass rolls with the wind), swells and
    shrinks, and drifts in and out in depth — so lobes roll over each other
    instead of the mass sliding as a flat shape. Returns [(x, y, r, sq, hfac, z)].
    hfac: 0 at the top of the mass .. 1 at its underside (shadowing)."""
    R1, R2, R3, R4, R5, R6, R7 = _mass_rand(seed, n)
    a0 = R1 * 2 * np.pi
    rr = np.sqrt(R2) * 0.85
    om = roll * (0.6 + 0.8 * R3)
    th = a0 - om * t                                     # counter-clockwise on screen: top moves left
    ox = np.cos(th) * rr * w * 0.5
    oy = np.sin(th) * rr * h * 0.5
    # flatten the underside a little (masses sit on / hang from their bases)
    oy = np.where(oy > 0, oy * (1 - top * 0.5), oy)
    ph = R4 * 2 * np.pi
    r = r0 + (r1 - r0) * R5 * (1.0 - 0.4 * rr)          # bigger lobes at the core
    r = r * (1 + pulse * np.sin(t * (1.1 + 0.9 * R6) + ph))
    z = np.sin(t * (0.7 + 0.6 * R7) + ph * 1.7)
    hfac = np.clip((oy + h * 0.5) / max(h, 1e-3), 0, 1)
    return [(float(cx + a), float(cy + b), float(c), sq, float(d), float(e))
            for a, b, c, d, e in zip(ox, oy, r, hfac, z)]


def masses_row(t, off, seed, spacing, base_y, w0, w1, W, yjit=6.0, n=(5, 9), rfrac=(0.28, 0.5),
               hang=False, aspect=0.55, roll=0.45, gate=None, margin=90, hscale=None):
    """A horizontally wrapping row of billowing masses (world-locked, slid by
    `off` px). Returns the lobes of all masses, drawn back to front (masses
    ordered by their base; lobes within a mass by height + depth)."""
    i0 = int(math.floor((off - margin) / spacing))
    i1 = int(math.ceil((off + W + margin) / spacing))
    out = []
    for i in range(i0, i1 + 1):
        if gate is not None and not bool(gate(np.array([i]))[0]):
            continue
        ri = rand1(np.array([i]), np.arange(1, 8), seed + 1)
        w = w0 + (w1 - w0) * float(ri[0])
        h = w * aspect * (0.8 + 0.4 * float(ri[1])) * (hscale if hscale else 1.0)
        cx = i * spacing + (float(ri[2]) - 0.5) * spacing * 0.6 - off
        cy = base_y + (float(ri[3]) - 0.5) * 2 * yjit + (h * 0.3 if hang else -h * 0.3)
        nl = int(n[0] + (n[1] - n[0]) * float(ri[4]))
        r0, r1 = w * rfrac[0] * 0.5, w * rfrac[1] * 0.5
        lobes = mass_lobes(t, cx, cy, w, h, seed * 1000 + i, nl, r0, r1, roll=roll)
        if hang:
            lobes = [(x, y, r, s, 1.0 - hf * 0.6, z) for (x, y, r, s, hf, z) in lobes]
        key = (cy if not hang else -cy)
        out.append((key, sorted(lobes, key=lambda L: (L[1] if not hang else -L[1]) + 4 * L[5])))
    out.sort(key=lambda m: m[0])
    return [lb for _, m in out for lb in m]


def draw_lobes(img, lobes, tones, ox=0, table=None, texk=0.34, edge=0.14, skew=0.18, gain=2.3,
               bias=None, under=1.3, under_local=0.8, tex_amp=0.35, band=0.36, rim=None, rim_w=0.12,
               rim_min=0.6, rim_tex=0.0, clip_fn=None, mask_out=None, sqk=1.0):
    """Paint sphere-shaded dust lobes (in the given order, back to front).
    tones: palette indices dark -> light. Level per pixel:
        bias + gain * lambert  - under * hfac  - under_local * (lower half of the lobe)
        + dusty texture breakup, quantised to the ramp with dithered seams.
    rim: optional bright rim colour on the lit edge (the dying sun catching the
    billow tops). clip_fn(xs, ys) -> bool keep-mask (e.g. only below an edge)."""
    h, w = img.shape
    tbl = SHADE_UL if table is None else table
    tones = np.asarray(tones, np.uint8)
    nt = len(tones) - 1
    if bias is None:
        bias = nt * 0.5
    bt = bayer_tile(h, w, 4)
    for lb in lobes:
        cx, cy, r, sq, hf = lb[0], lb[1], lb[2], lb[3] * sqk, lb[4]
        if r < 0.8:
            continue
        rx, ry = r, r * sq
        x0 = int(max(cx - rx * (1.3 + skew) - 2, 0)); x1 = int(min(cx + rx * (1.3 + skew) + 3, w))
        y0 = int(max(cy - ry * 1.3 - 2, 0)); y1 = int(min(cy + ry * 1.3 + 3, h))
        if x0 >= x1 or y0 >= y1:
            continue
        ys = np.arange(y0, y1)[:, None]
        xs = np.arange(x0, x1)[None, :]
        v = (ys - cy) / ry
        u = (xs - cx + (ys - cy) * skew) / rx
        iu = np.clip((u * _LS + 1.4 * _LS + 0.5).astype(np.int32), 0, _LN - 1)
        iv = np.clip((v * _LS + 1.4 * _LS + 0.5).astype(np.int32), 0, _LN - 1)
        iu, iv = np.broadcast_arrays(iu, iv)
        tex = TEX[ys & 255, (xs + ox) & 1023]
        d2 = LOBE_D2[iv, iu] + (tex - 0.5) * texk
        m = d2 + (bt[y0:y1, x0:x1] - 0.5) * edge < 1.0
        if clip_fn is not None:
            m &= clip_fn(np.broadcast_to(xs, m.shape), np.broadcast_to(ys, m.shape))
        if not m.any():
            continue
        sh = tbl[iv, iu]
        lvl = bias + gain * sh - under * hf - under_local * np.clip(v, 0, 1) + (tex - 0.5) * tex_amp
        q = quantize(np.clip(lvl, 0, nt), nt, band=band, ox=0, oy=0)
        sub = img[y0:y1, x0:x1]
        sub[m] = tones[q][m]
        if rim is not None:
            rm = m & (d2 > 1.0 - rim_w) & (sh > rim_min)
            if rim_tex:
                rm &= TEXF[ys & 255, (xs + ox) & 1023] > rim_tex
            sub[rm] = rim
        if mask_out is not None:
            mask_out[y0:y1, x0:x1] |= m
    return img


# ================================================================ HEIGHT-FIELD MASSES (v3 clouds)
_LIGHT_UL = np.array([-0.55, -0.72, 0.42]) / np.linalg.norm([-0.55, -0.72, 0.42])


def build_mass(t, cx, cy, w, h, seed, n, r0, r1, roll=0.45, sq=0.8, pulse=0.14, hang=False):
    """A mass = its lobes (x, y, r, sq, zoff) + bbox. Lobes circulate / pulse /
    drift in depth (internal motion). zoff lifts some lobes toward the viewer so
    they bulge over their neighbours."""
    lobes = mass_lobes(t, cx, cy, w, h, seed, n, r0, r1, roll=roll, pulse=pulse, sq=sq)
    out = []
    for (x, y, r, s, hf, z) in lobes:
        out.append((x, y, r, s, 0.35 * r * z))
    pad = max(r1, 1.0) * (1.0 + pulse) + 3
    return dict(lobes=out, cx=cx, cy=cy, w=w, h=h, hang=hang,
                box=(cx - w * 0.5 - pad, cy - h * 0.5 - pad, cx + w * 0.5 + pad, cy + h * 0.5 + pad))


def masses_row_v3(t, off, seed, spacing, base_y, w0, w1, W, yjit=6.0, n=(6, 10), rfrac=(0.26, 0.5),
                  hang=False, aspect=0.55, roll=0.45, gate=None, margin=90, sq=0.8):
    """Horizontally wrapping row of billowing masses; returns masses back to front."""
    i0 = int(math.floor((off - margin) / spacing))
    i1 = int(math.ceil((off + W + margin) / spacing))
    ms = []
    ii = np.arange(i0, i1 + 1)
    RI = [rand1(ii, j, seed + 1) for j in range(1, 6)]
    ok = np.ones(len(ii), bool) if gate is None else np.asarray(gate(ii), bool)
    for j, i in enumerate(ii):
        if not ok[j]:
            continue
        i = int(i)
        w = w0 + (w1 - w0) * float(RI[0][j])
        h = w * aspect * (0.8 + 0.4 * float(RI[1][j]))
        cx = i * spacing + (float(RI[2][j]) - 0.5) * spacing * 0.6 - off
        cy = base_y + (float(RI[3][j]) - 0.5) * 2 * yjit + (h * 0.3 if hang else -h * 0.3)
        nl = int(n[0] + (n[1] - n[0]) * float(RI[4][j]))
        ms.append(build_mass(t, cx, cy, w, h, seed * 1000 + i, nl, w * rfrac[0] * 0.5, w * rfrac[1] * 0.5,
                             roll=roll, sq=sq, hang=hang))
    ms.sort(key=lambda m: (-m['cy'] if hang else m['cy']))
    return ms


def draw_masses(img, masses, tones, ox=0, gain=2.2, bias=None, under=1.6, under_pow=1.3, tex_amp=0.4,
                band=0.38, edge_px=1.6, fray=0.9, light=None, rim=None, rim_min=0.72, rim_tex=0.0,
                clip_fn=None, mask_out=None, crease=0.9, left_lit=0.5, bump=0.22):
    """Paint billowing dust masses from the union height field of their lobes.
    Surface normals come from the height field's gradient, so neighbouring lobes
    merge into one cauliflower surface with shaded creases; the whole mass is lit
    coherently (upper-left light, darker underside, stepped ramp with dithered
    seams); silhouettes fray into dithered dusty fringes."""
    H_, W_ = img.shape
    L_ = _LIGHT_UL if light is None else np.asarray(light) / np.linalg.norm(light)
    tones = np.asarray(tones, np.uint8)
    nt = len(tones) - 1
    if bias is None:
        bias = nt * 0.55
    for ms in masses:
        bx0, by0, bx1, by1 = ms['box']
        x0, y0 = int(max(math.floor(bx0), 0)), int(max(math.floor(by0), 0))
        x1, y1 = int(min(math.ceil(bx1), W_)), int(min(math.ceil(by1), H_))
        if x0 >= x1 - 2 or y0 >= y1 - 2:
            continue
        hb, wb = y1 - y0, x1 - x0
        Hf = np.full((hb, wb), -1.0, np.float32)
        for (lx, ly, r, s, zo) in ms['lobes']:
            if r < 0.8:
                continue
            ry = r * s
            ax0, ax1 = int(max(lx - r - 1, x0)), int(min(lx + r + 2, x1))
            ay0, ay1 = int(max(ly - ry - 1, y0)), int(min(ly + ry + 2, y1))
            if ax0 >= ax1 or ay0 >= ay1:
                continue
            yy = (np.arange(ay0, ay1, dtype=np.float32)[:, None] - ly) / s
            xx = np.arange(ax0, ax1, dtype=np.float32)[None, :] - lx
            q = r * r - xx * xx - yy * yy
            hz = np.sqrt(np.maximum(q, 0)) * s + zo
            if bump and r >= 4:
                # fluffy sub-billows that travel with the lobe (texture indexed lobe-relative)
                sy_ = (np.arange(ay0, ay1)[:, None] - int(ly) + int(lx * 7) ) & 255
                sx_ = (np.arange(ax0, ax1)[None, :] - int(lx) + int(ly * 13) + 101) & 1023
                hz = hz + bump * r * (TEX[sy_, sx_] - 0.5) * 2.0 * np.clip(q / (r * r + 1e-6) * 3, 0, 1)
            hz = np.where(q > 0, hz, -1.0)
            sub = Hf[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
            np.maximum(sub, hz, out=sub)
        ys = np.arange(y0, y1)[:, None]
        xs = np.arange(x0, x1)[None, :]
        tex = TEX[ys & 255, (xs + ox) & 1023]
        inside = Hf + (bayer_tile(hb, wb, 4, x0, y0) - 0.5) * edge_px + (tex - 0.5) * fray * 2.5 > 0.35
        if clip_fn is not None:
            inside &= clip_fn(np.broadcast_to(xs, inside.shape), np.broadcast_to(ys, inside.shape))
        if not inside.any():
            continue
        Hc = np.maximum(Hf, 0.0)
        gx = np.zeros_like(Hc); gy = np.zeros_like(Hc)
        gx[:, 1:-1] = (Hc[:, 2:] - Hc[:, :-2]) * 0.5
        gy[1:-1, :] = (Hc[2:, :] - Hc[:-2, :]) * 0.5
        gx = np.clip(gx, -3, 3); gy = np.clip(gy, -3, 3)
        nrm = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
        lam = (-gx * L_[0] - gy * L_[1] + L_[2]) * nrm
        # creases: where the surface dips between lobes (local height below its neighbourhood)
        if crease:
            nb = Hc.copy()
            np.maximum(nb[2:], Hc[:-2], out=nb[2:]); np.maximum(nb[:-2], Hc[2:], out=nb[:-2])
            np.maximum(nb[:, 2:], Hc[:, :-2], out=nb[:, 2:]); np.maximum(nb[:, :-2], Hc[:, 2:], out=nb[:, :-2])
            cav = np.clip((nb - Hc) * (1.0 / 3.0), 0, 1)
        else:
            cav = 0.0
        vy = np.clip((ys - (ms['cy'] - ms['h'] * 0.5)) / max(ms['h'], 1.0), 0, 1.2)
        vx = np.clip((xs - (ms['cx'] - ms['w'] * 0.5)) / max(ms['w'], 1.0), 0, 1)
        lvl = (bias + gain * lam - under * vy ** under_pow - crease * cav - left_lit * vx
               + (tex - 0.5) * tex_amp)
        q = quantize(np.clip(lvl, 0, nt), nt, band=band, ox=x0, oy=y0)
        sub = img[y0:y1, x0:x1]
        sub[inside] = tones[q][inside]
        if rim is not None:
            rm = inside & (lam > rim_min) & (Hc < 0.45 * max(ms['h'], 1)) & (vy < 0.55)
            if rim_tex:
                rm &= TEXF[ys & 255, (xs + ox) & 1023] > rim_tex
            sub[rm] = rim
        if mask_out is not None:
            mask_out[y0:y1, x0:x1] |= inside
    return img
