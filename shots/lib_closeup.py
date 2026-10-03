"""
lib_closeup.py — shared toolkit for the CLOSEUPS illustrations (c01–c04).

Everything here works in PALETTE-INDEX space: an image is an (h, w) uint8
array of palette indices; it becomes RGB only at the very end (PAL[idx]).
That keeps every pixel palette-true and makes lighting a cheap table lookup.

Main pieces
  ci / cis                 palette name -> index
  RAMP-WALK lighting       shift(idx, k): move k steps along each pixel's own
                           hue-shifted ramp (with hand-picked cross-ramp links
                           for the extremes: paint6 -> dust9 -> white, ...)
  quantize                 float field -> stepped ramp colors (optional narrow
                           ordered-dither seams)
  clean                    orphan-pixel cleanup (clusters, not noise)
  ell / ell_rho            pixel-true ellipse masks / normalised radius
  raycast_superellipsoid   tiny orthographic ray-caster for the robot's
                           rounded head / body (mask, depth, normals)
  ilight                   engine-compatible palette lighting in index space
  motes                    deterministic dust / pollen particles
"""
import math
import numpy as np
from engine import palette as P
from engine.core import bayer, to_index
from engine import light as _L
from engine import noise as N

PAL = P.PAL
NI = P.NAME_TO_IDX
NPAL = P.N


def ci(c):
    """palette name (or index) -> palette index"""
    if isinstance(c, (int, np.integer)):
        return int(c)
    return NI[c]


def cis(names):
    return np.array([ci(n) for n in names], np.uint8)


def to_rgb(idx):
    return PAL[idx]


def from_rgb(rgb):
    return to_index(np.ascontiguousarray(rgb[..., :3]))


# ------------------------------------------------------------------ ramp walk
_UP = np.arange(NPAL)
_DN = np.arange(NPAL)
for _r, _ids in P.RAMP_IDX.items():
    for _k, _i in enumerate(_ids):
        _UP[_i] = _ids[min(_k + 1, len(_ids) - 1)]
        _DN[_i] = _ids[max(_k - 1, 0)]
_XUP = {'paint6': 'dust9', 'dust9': 'dust10', 'dust10': 'white0', 'white0': 'white1',
        'steel6': 'white0', 'ash8': 'dust10', 'glass4': 'white0', 'rust6': 'ochre7',
        'ochre8': 'dust10', 'core7': 'white1', 'leaf8': 'white0', 'meadow6': 'gold4',
        'gold4': 'core6', 'bark6': 'ochre8', 'bloom5': 'white0', 'violet4': 'bloom5',
        'red4': 'bloom4', 'sky8': 'white1', 'ink2': 'steel1', 'night5': 'ash5'}
_XDN = {'paint0': 'night1', 'steel0': 'ink1', 'dust0': 'ink1', 'rust0': 'ink1', 'ash0': 'ink1',
        'night0': 'ink0', 'leaf0': 'night1', 'meadow0': 'leaf0', 'ochre0': 'ink1',
        'core0': 'rust0', 'bark0': 'ink1', 'glass0': 'night1', 'bloom0': 'violet0',
        'violet0': 'night1', 'red0': 'rust0', 'gold0': 'ochre1', 'sky0': 'night2', 'white0': 'dust10',
        'ink0': 'ink0'}
for a, b in _XUP.items():
    _UP[NI[a]] = NI[b]
for a, b in _XDN.items():
    _DN[NI[a]] = NI[b]
MAXS = 8
SHIFT = np.zeros((2 * MAXS + 1, NPAL), np.uint8)
SHIFT[MAXS] = np.arange(NPAL)
for _k in range(1, MAXS + 1):
    SHIFT[MAXS + _k] = _UP[SHIFT[MAXS + _k - 1]]
    SHIFT[MAXS - _k] = _DN[SHIFT[MAXS - _k + 1]]


def shift(idx, k, mask=None):
    """Move k steps along each pixel's ramp. k: int or int array (h,w)."""
    k = np.clip(np.asarray(k, np.int32), -MAXS, MAXS)
    out = SHIFT[k + MAXS, idx]
    if mask is not None:
        out = np.where(mask, out, idx)
    return out.astype(np.uint8)


def dshift(idx, level, n=4, band=0.5, ox=0, oy=0, mask=None):
    """Ramp-walk by a FLOAT level map, quantised with ordered dithering."""
    h, w = idx.shape
    lv = np.asarray(level, np.float64)
    if lv.ndim == 0:
        lv = np.full((h, w), float(lv))
    fl = np.floor(lv)
    fr = lv - fl
    if band < 1.0:
        fr = np.clip((fr - 0.5) / max(band, 1e-3) + 0.5, 0, 1)
    q = (fl + (bayer(h, w, n, ox, oy) < fr)).astype(np.int32)
    return shift(idx, q, mask)


def ilight(idx, level, mode='warm', shadow='night', n=4, band=0.45, ox=0, oy=0, mask=None):
    """engine.light.apply_light, but in index space (same LUTs => same look)."""
    h, w = idx.shape
    lv = np.asarray(level, np.float64)
    if lv.ndim == 0:
        lv = np.full((h, w), float(lv))
    fl = np.floor(lv)
    fr = lv - fl
    if band < 1.0:
        fr = np.clip((fr - 0.5) / max(band, 1e-3) + 0.5, 0, 1)
    q = (fl + (bayer(h, w, n, ox, oy) < fr)).astype(np.int32)
    q = np.clip(q, -_L.MAXL, _L.MAXL)
    if mask is not None:
        q = np.where(mask, q, 0)
    lut = _L._lut(mode, shadow)
    return lut[q + _L.MAXL, idx].astype(np.uint8)


def remap(idx, mapping, mask=None):
    """Recolor by name mapping {'paint3': 'rust2', ...}."""
    lut = np.arange(NPAL, dtype=np.uint8)
    for a, b in mapping.items():
        lut[ci(a)] = ci(b)
    out = lut[idx]
    if mask is not None:
        out = np.where(mask, out, idx)
    return out.astype(np.uint8)


# ------------------------------------------------------------------ quantize
def quantize(field, names, cuts=None, dith=0.0, n=4, ox=0, oy=0):
    """Float field -> palette indices of `names` (dark->light).
    cuts: thresholds (len(names)-1); default uniform over [0,1].
    dith: width (field units) of an ordered-dither seam at each cut (0 = hard)."""
    cols = cis(names)
    k = len(cols)
    f = np.asarray(field, np.float64)
    if cuts is None:
        cuts = np.linspace(0, 1, k + 1)[1:-1]
    lev = np.zeros(f.shape, np.int32)
    th = bayer(f.shape[0], f.shape[1], n, ox, oy) if dith > 0 else None
    for c in cuts:
        if dith > 0:
            a = np.clip((f - (c - dith / 2)) / dith, 0, 1)
            lev += (th < a)
        else:
            lev += (f >= c)
    return cols[np.clip(lev, 0, k - 1)]


def clean(idx, mask=None, passes=2):
    """Replace orphan pixels (no 4-neighbour of the same color) by the
    dominant neighbour color. Only inside `mask` if given."""
    out = idx.copy()
    for _ in range(passes):
        p = np.pad(out, 1, mode='edge')
        c = out
        up, dn, lf, rt = p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]
        orphan = (up != c) & (dn != c) & (lf != c) & (rt != c)
        if mask is not None:
            orphan &= mask
        if not orphan.any():
            break
        rep = np.where(up == dn, up, np.where(lf == rt, lf,
                       np.where((up == lf) | (up == rt), up, dn)))
        out = np.where(orphan, rep, c).astype(idx.dtype)
    return out


def clean_spurs(idx, mask=None):
    """Remove 1-px 'spurs': pixels with exactly one same-colored 4-neighbour
    that sit on a straight edge (reduces jaggy single steps)."""
    p = np.pad(idx, 1, mode='edge')
    c = idx
    up, dn, lf, rt = p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]
    same = (up == c).astype(int) + (dn == c) + (lf == c) + (rt == c)
    spur = same == 1
    if mask is not None:
        spur &= mask
    rep = np.where(up == dn, up, np.where(lf == rt, lf, c))
    return np.where(spur & (rep != c), rep, c).astype(idx.dtype)


# ------------------------------------------------------------------ geometry
def yx(h, w):
    ys, xs = np.mgrid[0:h, 0:w]
    return ys.astype(np.float64), xs.astype(np.float64)


def ell_rho(h, w, cx, cy, rx, ry):
    ys, xs = np.ogrid[:h, :w]
    return np.sqrt(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2)


def ell(h, w, cx, cy, rx, ry, grow=0.35):
    ys, xs = np.ogrid[:h, :w]
    return ((xs - cx) / (rx + grow)) ** 2 + ((ys - cy) / (ry + grow)) ** 2 <= 1.0


def disc(h, w, cx, cy, r, grow=0.35):
    ys, xs = np.ogrid[:h, :w]
    return (xs - cx) ** 2 + (ys - cy) ** 2 <= (r + grow) ** 2


def edge_in(mask):
    """Pixels of mask that touch a non-mask 4-neighbour."""
    p = np.pad(mask, 1, mode='constant', constant_values=False)
    inner = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return mask & ~inner


def edge_out(mask, diagonal=False):
    p = np.pad(mask, 1, mode='constant', constant_values=False)
    o = p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    if diagonal:
        o = o | p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
    return o & ~mask


def edge_side(mask, dx, dy):
    """Pixels of mask whose neighbour at (+dx,+dy) is outside the mask
    (e.g. dx=0,dy=1 -> bottom edge)."""
    h, w = mask.shape
    nb = np.zeros_like(mask)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    nb[ys0:ys1, xs0:xs1] = mask[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
    return mask & ~nb


def shifted(mask, dx, dy):
    """mask moved by (dx,dy) (no wrap)."""
    h, w = mask.shape
    out = np.zeros_like(mask)
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    out[ys0:ys1, xs0:xs1] = mask[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def dilate(mask, r=1, diagonal=True):
    out = mask.copy()
    for _ in range(r):
        out = out | edge_out(out, diagonal)
    return out


def erode(mask, r=1):
    out = mask.copy()
    for _ in range(r):
        out = out & ~edge_in(out)
    return out


def dist_to(mask, maxd=12):
    """Approx. chamfer distance (px) from each pixel to the nearest True of
    mask, capped at maxd."""
    d = np.where(mask, 0.0, float(maxd))
    cur = mask.copy()
    for k in range(1, maxd):
        nxt = dilate(cur, 1, diagonal=(k % 2 == 0))
        ring = nxt & ~cur
        d[ring] = k
        cur = nxt
        if cur.all():
            break
    return d


def polygon(h, w, pts):
    """Vectorised even-odd polygon mask (pixel centres)."""
    pts = np.asarray(pts, np.float64)
    ys, xs = np.mgrid[0:h, 0:w]
    xs = xs + 0.0
    ys = ys + 0.0
    inside = np.zeros((h, w), bool)
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        if y0 == y1:
            continue
        cond = (ys >= min(y0, y1)) & (ys < max(y0, y1))
        xint = x0 + (ys - y0) * (x1 - x0) / (y1 - y0)
        inside ^= cond & (xs < xint)
    return inside


def line_px(x0, y0, x1, y1):
    """Bresenham pixel list."""
    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    out = []
    while True:
        out.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return out


def put(idx, x, y, c):
    h, w = idx.shape
    x, y = int(x), int(y)
    if 0 <= x < w and 0 <= y < h:
        idx[y, x] = ci(c)


def puts(idx, pts, c):
    v = ci(c)
    h, w = idx.shape
    for x, y in pts:
        x, y = int(x), int(y)
        if 0 <= x < w and 0 <= y < h:
            idx[y, x] = v


def polyline_px(pts):
    out = []
    for i in range(len(pts) - 1):
        seg = line_px(*pts[i], *pts[i + 1])
        if out and seg and out[-1] == seg[0]:
            seg = seg[1:]
        out += seg
    return out


def bezier(p0, p1, p2, p3=None, n=24):
    ts = np.linspace(0, 1, n)
    p0, p1, p2 = np.array(p0, float), np.array(p1, float), np.array(p2, float)
    if p3 is None:
        pts = ((1 - ts) ** 2)[:, None] * p0 + (2 * (1 - ts) * ts)[:, None] * p1 + (ts ** 2)[:, None] * p2
    else:
        p3 = np.array(p3, float)
        pts = (((1 - ts) ** 3)[:, None] * p0 + (3 * (1 - ts) ** 2 * ts)[:, None] * p1 +
               (3 * (1 - ts) * ts ** 2)[:, None] * p2 + (ts ** 3)[:, None] * p3)
    return [tuple(p) for p in pts]


def curve_px(pts):
    """Pixel-perfect polyline through float points: rounds points, removes
    duplicates and L-corners (keeps 1-px lines clean)."""
    raw = polyline_px([(round(x), round(y)) for x, y in pts])
    out = []
    for p in raw:
        if out and out[-1] == p:
            continue
        out.append(p)
    # remove L-shaped doubles: if a,b,c where a and c are diagonal neighbours, drop b
    clean_pts = []
    i = 0
    while i < len(out):
        if 0 < i < len(out) - 1 and clean_pts:
            a, c = clean_pts[-1], out[i + 1]
            if abs(a[0] - c[0]) == 1 and abs(a[1] - c[1]) == 1:
                i += 1
                continue
        clean_pts.append(out[i])
        i += 1
    return clean_pts


# ------------------------------------------------------------------ 3D-ish
def normalize(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


def frame_yaw(beta, pitch=0.0, roll=0.0):
    """Robot local frame for yaw beta (0 = facing camera, +pi/2 = facing
    screen-right). Returns F (forward), U (up), S (side) as world vectors in
    screen space: X right, Y DOWN, Z toward the viewer."""
    F = np.array([math.sin(beta), 0.0, math.cos(beta)])
    U = np.array([0.0, -1.0, 0.0])
    S = np.cross(U, F)
    if pitch:
        # tilt forward (nod down): rotate F,U about S
        c, s = math.cos(pitch), math.sin(pitch)
        F, U = F * c - U * s, U * c + F * s
    if roll:
        c, s = math.cos(roll), math.sin(roll)
        U, S = U * c + S * s, S * c - U * s
    return F, U, S


def raycast_superellipsoid(h, w, C, F, U, S, a, n=4.0, zrange=(-400, 400), steps=240, x0=0, y0=0):
    """Orthographic ray-cast (rays along -Z) of a superellipsoid
    |f/af|^n + |u/au|^n + |s/as|^n <= 1 in the local frame (F,U,S) centred
    at C. Returns (mask, z, normal(h,w,3), local(h,w,3))."""
    af, au, as_ = a
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    xs += x0
    ys += y0
    dx = xs - C[0]
    dy = ys - C[1]

    def Fn(z):
        dz = z - C[2]
        qf = dx * F[0] + dy * F[1] + dz * F[2]
        qu = dx * U[0] + dy * U[1] + dz * U[2]
        qs = dx * S[0] + dy * S[1] + dz * S[2]
        return (np.abs(qf / af) ** n + np.abs(qu / au) ** n + np.abs(qs / as_) ** n) - 1.0

    zs = np.linspace(zrange[1], zrange[0], steps) + C[2]
    found = np.zeros((h, w), bool)
    zhit = np.full((h, w), np.nan)
    zprev = np.full((h, w), zs[0])
    for z in zs:
        inside = Fn(np.full((h, w), z)) <= 0
        new = inside & ~found
        zhit[new] = z
        found |= inside
        zprev = np.where(found, zprev, z)
    # refine by bisection between zprev (outside) and zhit (inside)
    lo = zhit.copy()          # inside
    hi = zprev.copy()         # outside (greater z)
    m = found
    for _ in range(18):
        mid = (lo + hi) / 2
        ins = Fn(np.where(m, mid, 0)) <= 0
        lo = np.where(m & ins, mid, lo)
        hi = np.where(m & ~ins, mid, hi)
    z = np.where(m, lo, np.nan)
    dz = np.where(m, z - C[2], 0)
    qf = dx * F[0] + dy * F[1] + dz * F[2]
    qu = dx * U[0] + dy * U[1] + dz * U[2]
    qs = dx * S[0] + dy * S[1] + dz * S[2]
    gf = n * np.sign(qf) * np.abs(qf) ** (n - 1) / af ** n
    gu = n * np.sign(qu) * np.abs(qu) ** (n - 1) / au ** n
    gs = n * np.sign(qs) * np.abs(qs) ** (n - 1) / as_ ** n
    nrm = gf[..., None] * F + gu[..., None] * U + gs[..., None] * S
    ln = np.linalg.norm(nrm, axis=-1, keepdims=True)
    nrm = nrm / np.maximum(ln, 1e-9)
    loc = np.stack([qf, qu, qs], -1)
    return m, z, nrm, loc


def lambert(nrm, Ldir, wrap=0.0):
    L = normalize(Ldir)
    d = (nrm * L).sum(-1)
    if wrap:
        d = (d + wrap) / (1 + wrap)
    return np.clip(d, 0, 1)


def spec(nrm, Ldir, power=20.0):
    L = normalize(Ldir)
    Hh = normalize(L + np.array([0, 0, 1.0]))
    return np.clip((nrm * Hh).sum(-1), 0, 1) ** power


# ------------------------------------------------------------------ particles
def field(t, n, seed, area, vel=(0.0, 0.0), jitter=(0.0, 0.0), sway=(0.0, 0.0)):
    i = np.arange(n)
    ax, ay, aw, ah = area
    x0 = ax + N.rand(i, 1, seed) * aw
    y0 = ay + N.rand(i, 2, seed) * ah
    vx = vel[0] + (N.rand(i, 3, seed) - 0.5) * 2 * jitter[0]
    vy = vel[1] + (N.rand(i, 4, seed) - 0.5) * 2 * jitter[1]
    ph = N.rand(i, 5, seed)
    x = x0 + vx * t
    y = y0 + vy * t
    if sway[0]:
        x = x + sway[0] * np.sin(2 * math.pi * (sway[1] * t * (0.7 + 0.6 * ph) + ph))
        y = y + 0.6 * sway[0] * np.cos(2 * math.pi * (sway[1] * 0.8 * t * (0.7 + 0.6 * ph) + 2 * ph))
    x = ax + np.mod(x - ax, aw)
    y = ay + np.mod(y - ay, ah)
    return x, y, N.rand(i, 6, seed)


def stamp(idx, x, y, pattern, colors):
    """Stamp a tiny ascii pattern (list of strings, '.' transparent, digits
    index into colors) with its top-left at (x,y)."""
    h, w = idx.shape
    for j, row in enumerate(pattern):
        for i, ch in enumerate(row):
            if ch == '.' or ch == ' ':
                continue
            px, py = int(x) + i, int(y) + j
            if 0 <= px < w and 0 <= py < h:
                idx[py, px] = ci(colors[int(ch)])


def sprite_from_ascii(art, legend):
    """ascii -> (idx(h,w) uint8, mask(h,w) bool). legend: char -> palette name."""
    lines = [ln for ln in art.strip('\n').split('\n')]
    ind = min((len(l) - len(l.lstrip(' ')) for l in lines if l.strip()), default=0)
    lines = [l[ind:] for l in lines]
    h = len(lines)
    w = max(len(l) for l in lines)
    idx = np.zeros((h, w), np.uint8)
    m = np.zeros((h, w), bool)
    for y, l in enumerate(lines):
        for x, ch in enumerate(l):
            if ch in '. ':
                continue
            idx[y, x] = ci(legend[ch])
            m[y, x] = True
    return idx, m


def blit_idx(dst, src, mask, x, y, flip=False):
    """Paste an index sprite (with bool mask) at integer top-left (x,y)."""
    if flip:
        src = src[:, ::-1]
        mask = mask[:, ::-1]
    x, y = int(round(x)), int(round(y))
    sh, sw = src.shape
    dh, dw = dst.shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + sw, dw), min(y + sh, dh)
    if x0 >= x1 or y0 >= y1:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    m = mask[y0 - y:y1 - y, x0 - x:x1 - x]
    d = dst[y0:y1, x0:x1]
    d[m] = s[m]


def step(t, fps=12.0):
    """Quantise time to an animation rate ('on twos'-style holds)."""
    return math.floor(t * fps + 1e-6) / fps


def damped(t, freq=3.0, decay=4.0):
    """Damped oscillation starting at t=0 (0 before): sin * exp."""
    if t <= 0:
        return 0.0
    return math.sin(2 * math.pi * freq * t) * math.exp(-decay * t)


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


# ------------------------------------------------------------------ thick curves (fingers, stems, vines)
def thick_curve(h, w, pts, r0, r1=None, cap=True):
    """Mask of a thick polyline through float points `pts` with radius
    tapering r0 -> r1. Returns (mask, along 0..1, across -1..1 (signed:
    + on the right of the direction of travel), dist)."""
    r1 = r0 if r1 is None else r1
    P_ = np.asarray(pts, np.float64)
    seg = P_[1:] - P_[:-1]
    L = np.sqrt((seg ** 2).sum(-1))
    cum = np.concatenate([[0], np.cumsum(L)])
    total = max(cum[-1], 1e-6)
    x0 = int(max(np.floor(P_[:, 0].min() - max(r0, r1) - 2), 0))
    x1 = int(min(np.ceil(P_[:, 0].max() + max(r0, r1) + 3), w))
    y0 = int(max(np.floor(P_[:, 1].min() - max(r0, r1) - 2), 0))
    y1 = int(min(np.ceil(P_[:, 1].max() + max(r0, r1) + 3), h))
    mask = np.zeros((h, w), bool)
    along = np.zeros((h, w))
    across = np.zeros((h, w))
    dist = np.full((h, w), 1e9)
    if x0 >= x1 or y0 >= y1:
        return mask, along, across, dist
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    best = np.full(xx.shape, 1e18)
    bt = np.zeros(xx.shape)
    bs = np.zeros(xx.shape)
    for i in range(len(seg)):
        ax, ay = P_[i]
        dx, dy = seg[i]
        l2 = max(dx * dx + dy * dy, 1e-9)
        tt = np.clip(((xx - ax) * dx + (yy - ay) * dy) / l2, 0, 1)
        px, py = ax + tt * dx, ay + tt * dy
        d2 = (xx - px) ** 2 + (yy - py) ** 2
        better = d2 < best
        best = np.where(better, d2, best)
        bt = np.where(better, (cum[i] + tt * L[i]) / total, bt)
        side = (xx - px) * (-dy) + (yy - py) * dx      # >0 : right of travel (screen coords)
        bs = np.where(better, np.sign(side), bs)
    d = np.sqrt(best)
    rad = r0 + (r1 - r0) * bt
    m = d <= rad + 0.3
    if not cap:
        m &= (bt > 0.0) & (bt < 1.0)
    mask[y0:y1, x0:x1] = m
    along[y0:y1, x0:x1] = bt
    across[y0:y1, x0:x1] = np.clip(bs * d / np.maximum(rad, 0.5), -1, 1)
    dist[y0:y1, x0:x1] = d
    return mask, along, across, dist


# ================================================================== HEAD RIG
# A reusable, relightable version of the c01/c04 head + camera-lens eye.
# A pose is ray-cast once (inside its bounding box) and returns LOCAL colours
# (albedo: paint, steel, rust... no lighting) + per-pixel normals, so a shot can
# relight it every frame (flickering porthole light, a sinking ember...).
# The lens is built exactly like c01/c04: collar, knurled barrel, bevelled
# bezel, dark marking ring, glass with a 7-blade iris over the pupil.
LENS_STD = dict(R_COL=24.0, R_BAR=21.8, D_FRONT=9.5, R_BEZ_IN=18.4, R_MARK_IN=16.8,
                R_GLASS=15.4, D_GLASS=8.6, D_IRIS=4.2, D_COL0=-2.0, D_COL1=1.5)


class LensGeom:
    """analytic orthographic geometry of the lens for one pose, inside a bbox."""

    def __init__(self, O, F, L, x0, y0, h, w):
        self.O, self.F, self.L = np.asarray(O, float), np.asarray(F, float), L
        self.x0, self.y0, self.h, self.w = x0, y0, h, w
        ys, xs = np.mgrid[y0:y0 + h, x0:x0 + w].astype(np.float64)
        self.xs, self.ys = xs, ys
        F = self.F
        self.NB = float(np.hypot(F[0], F[1]))
        E1 = np.array([F[0], F[1]]) / self.NB
        self.B3 = np.array([F[1], -F[0], 0.0]) / self.NB
        self.A3 = np.cross(self.B3, F)
        self.UA, self.VB = -self.A3, -self.B3
        self.FZ = F[2]
        px, py = xs - self.O[0], ys - self.O[1]
        self.P1 = px * E1[0] + py * E1[1]
        self.P2 = px * self.B3[0] + py * self.B3[1]

    def ring_local(self, d):
        a = (self.P1 - self.NB * d) / (-self.FZ)
        return -a, -self.P2

    def rho_ang(self, d):
        u, v = self.ring_local(d)
        return np.sqrt(u * u + v * v), np.arctan2(v, u)

    def screen_pt(self, d, r, ang):
        P_ = self.O + self.F * d + r * (self.UA * math.cos(ang) + self.VB * math.sin(ang))
        return P_[0], P_[1]

    def cyl_side(self, r, d0, d1):
        s = np.clip(self.P2 / r, -1, 1)
        c = np.sqrt(1 - s * s)
        d = (self.P1 + r * self.FZ * c) / self.NB
        m = (np.abs(self.P2) <= r + 0.3) & (d >= d0) & (d <= d1)
        nrm = self.A3[None, None, :] * c[..., None] + self.B3[None, None, :] * s[..., None]
        return m, d, nrm, np.arctan2(s, c)


def _quant_fixed(v, names, cuts):
    return quantize(v, names, cuts=cuts)


def head_pose(P, yaw, nod, roll=0.0, A=(40.0, 34.0, 45.0), n=6.0, L=None, u_off=2.0,
              pivot=(6.0, 1.08), damaged=True, knurl_phase=0.0, steps=150, ear=False):
    """Ray-cast one pose of the Gardener's head around the neck pivot P
    (screen x, y, z). Returns a dict with bbox-local arrays:
      'bbox' (x0,y0,w,h), 'mask', 'z', 'nrm', 'loc' (head local coords),
      'col' (local colour indices of head + lens metal), 'lens' (LensGeom),
      'glass' (glass mask), 'iris' (iris-plane mask), 'wall', 'O', 'F', 'U', 'S',
      'pupil' (screen xy of the iris centre), 'head' (head-only mask),
      'lensm' (lens metal mask), 'emis0' (pixels drawn per frame: glass)."""
    L = dict(LENS_STD if L is None else L)
    F, U, S = frame_yaw(yaw, nod, roll)
    P = np.asarray(P, float)
    C = P + F * pivot[0] + U * A[1] * pivot[1]
    O = C + F * A[0] * 0.985 + U * u_off
    R = max(A) * 1.35 + L['R_COL'] + 6
    x0 = int(math.floor(min(C[0], O[0]) - R))
    x1 = int(math.ceil(max(C[0], O[0]) + R))
    y0 = int(math.floor(min(C[1], O[1]) - R))
    y1 = int(math.ceil(max(C[1], O[1]) + R))
    h, w = y1 - y0, x1 - x0
    m, z, nrm, loc = raycast_se_fast(h, w, C, F, U, S, A, n=n, zrange=(-R * 1.5, R * 1.5),
                                     steps=steps, x0=x0, y0=y0)
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    qf, qu, qs = loc[..., 0], loc[..., 1], loc[..., 2]
    # ---------------- head local colours (albedo) + details
    col = np.full((h, w), ci('paint3'), np.uint8)
    # faded paint: lighter on the upper faces (sun-bleached), grime low
    col = np.where(m & ((nrm * U).sum(-1) > 0.55), ci('paint4'), col)
    col = np.where(m & (qu < -A[1] * 0.55), ci('paint2'), col)
    FACE_F = m & ((nrm * F).sum(-1) > 0.55)
    FACE_S = m & ((nrm * S).sum(-1) > 0.55)
    FACE_T = m & ((nrm * U).sum(-1) > 0.45)
    fs, fu = A[2] * 0.76, A[1] * 0.80
    g = (np.abs(qs / fs) ** 8 + np.abs(qu / fu) ** 8)
    inner = FACE_F & (g < 1.0)
    groove = FACE_F & edge_out(inner) & ~inner
    col = np.where(groove, ci('paint0'), col)
    sseam = FACE_S & (np.abs(qf - 0.2 * A[0]) < 0.6)
    col = np.where(sseam, ci('paint0'), col)
    col = np.where(shifted(sseam, 1, 0) & m & ~sseam, ci('paint4'), col)
    if damaged:
        # dent + chipped paint at the top-left of the head, storm scratches
        top_rows = np.nonzero(m.any(1))[0]
        left_cols = np.nonzero(m.any(0))[0]
        if len(top_rows) and len(left_cols):
            hx0, hy0 = xs[0, left_cols.min()], ys[top_rows.min(), 0]
            ch = m & (((xs - hx0 - A[2] * 0.38) / 6.0) ** 2 + ((ys - hy0 - A[1] * 0.55) / 3.4) ** 2 <= 1.0)
            col = np.where(ch, ci('steel3'), col)
            col = np.where(edge_out(ch) & m & (ys < hy0 + A[1] * 0.55), ci('paint0'), col)
        for (a0, b0, a1, b1) in ((0.10, 0.30, 0.34, 0.18), (0.14, 0.40, 0.30, 0.34)):
            # scratches in head-local side coordinates (on the side face)
            sc = FACE_S & (np.abs((qf / A[0] - a0) * (b1 - b0) - (qu / A[1] - b0) * (a1 - a0)) < 0.012) & \
                 (qf / A[0] > min(a0, a1)) & (qf / A[0] < max(a0, a1))
            col = np.where(sc, ci('paint5'), col)
    if ear:
        # the side 'ear': a hex bolt in a round socket (as in c01)
        ef, eu, er = -0.18 * A[0], -0.02 * A[1], 0.30 * A[1]
        dq = np.sqrt((qf - ef) ** 2 + (qu - eu) ** 2)
        sock = FACE_S & (dq <= er)
        col = np.where(sock, ci('paint2'), col)
        col = np.where(sock & (dq > er - 1.3), ci('paint0'), col)
        ha = np.arctan2(qu - eu, qf - ef)
        hexr = er * 0.62 / np.maximum(np.cos(((ha + math.pi / 6) % (math.pi / 3)) - math.pi / 6), 0.5)
        hx = sock & (dq <= hexr)
        col = np.where(hx, ci('steel2'), col)
        col = np.where(hx & (np.sin(ha) > 0.35), ci('steel3'), col)          # upper facets catch light
        col = np.where(hx & (np.sin(ha) < -0.45), ci('steel1'), col)
        col = np.where(hx & (dq <= hexr * 0.35), ci('steel1'), col)
        col = np.where(sock & ~hx & (dq <= hexr + 1.0), ci('paint0'), col)
    # rivets (4 on the faceplate corners)
    for (sgn_s, sgn_u) in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        d2 = (qs - sgn_s * (fs - 5)) ** 2 + (qu - sgn_u * (fu - 5)) ** 2 + (qf - A[0]) ** 2 * 0.2
        d2 = np.where(FACE_F & inner, d2, 1e9)
        k = int(np.argmin(d2))
        if d2.flat[k] < 30:
            ry, rx = divmod(k, w)
            for (dx, dy, c) in ((0, -1, 'paint5'), (-1, 0, 'paint4'), (0, 0, 'paint4'), (1, 0, 'paint2'), (0, 1, 'paint2')):
                if 0 <= ry + dy < h and 0 <= rx + dx < w and m[ry + dy, rx + dx]:
                    col[ry + dy, rx + dx] = ci(c)
    # ---------------- lens (metal parts as local colours, glass handled per frame)
    lg = LensGeom(O, F, L, x0, y0, h, w)
    bar_m, bar_d, bar_n, bar_t = lg.cyl_side(L['R_BAR'], 1.5, L['D_FRONT'])
    col_m, col_d, col_n, _ = lg.cyl_side(L['R_COL'], L['D_COL0'], L['D_COL1'])
    rho_f, ang_f = lg.rho_ang(L['D_FRONT'])
    front_m = rho_f <= L['R_BAR'] + 0.35
    colf_r, colf_a = lg.rho_ang(L['D_COL1'])
    colf_m = (colf_r <= L['R_COL'] + 0.35) & ~front_m
    rho_g, ang_g = lg.rho_ang(L['D_GLASS'])
    glass_m = rho_g <= L['R_GLASS'] + 0.3
    rho_i, ang_i = lg.rho_ang(L['D_IRIS'])
    iris_m = glass_m & (rho_i <= L['R_GLASS'] + 0.2)
    wall_m = glass_m & ~iris_m
    lensm = bar_m | col_m | colf_m | front_m
    # normals for the lens metal
    n_l = np.zeros((h, w, 3))
    n_l = np.where(col_m[..., None], col_n, n_l)
    n_l = np.where(colf_m[..., None], F[None, None, :], n_l)
    n_l = np.where(bar_m[..., None], bar_n, n_l)
    ca, sa = np.cos(ang_f), np.sin(ang_f)
    radial = lg.UA[None, None, :] * ca[..., None] + lg.VB[None, None, :] * sa[..., None]
    tilt = np.where(rho_f > L['R_BAR'] - 1.7, 1.5, np.where(rho_f > L['R_BEZ_IN'] + 1.2, 0.0, -1.2))
    nb_ = F[None, None, :] + tilt[..., None] * radial
    nb_ = nb_ / np.linalg.norm(nb_, axis=-1, keepdims=True)
    n_l = np.where(front_m[..., None], nb_, n_l)
    # lens local colours
    lc = np.zeros((h, w), np.uint8)
    lc = np.where(col_m | colf_m, ci('paint3'), lc)
    lc = np.where(colf_m & (colf_r < L['R_BAR'] + 1.2), ci('paint0'), lc)
    lc = np.where(bar_m, ci('steel2'), lc)
    kn = bar_m & (bar_d > 3.0) & (bar_d < 7.5) & ((((bar_t * 40 / (2 * math.pi)) + knurl_phase) % 1.0) < 0.5)
    lc = np.where(kn, ci('steel1'), lc)
    lc = np.where(bar_m & (np.abs(bar_d - 7.8) < 0.5), ci('steel0'), lc)
    lc = np.where(front_m, ci('steel3'), lc)
    lc = np.where(front_m & (rho_f > L['R_BAR'] - 0.7), ci('steel1'), lc)
    mk = front_m & (rho_f <= L['R_BEZ_IN']) & (rho_g > L['R_GLASS'])
    lc = np.where(mk, ci('steel0'), lc)
    # tick marks on the marking ring
    nt = 36
    fr_ = ((ang_f + knurl_phase * 0.2) * nt / (2 * math.pi)) % 1.0
    tick = mk & (np.minimum(fr_, 1 - fr_) * (2 * math.pi / nt) * rho_f < 0.55) & (rho_f < L['R_GLASS'] + 2.2)
    lc = np.where(tick, ci('steel3'), lc)
    lc = np.where(wall_m, ci('ink1'), lc)
    col = np.where(lensm, lc, col)
    nrm = np.where(lensm[..., None], n_l, nrm)
    full = m | lensm
    # pupil centre in screen coords
    pc = O + F * L['D_IRIS']
    return dict(bbox=(x0, y0, w, h), mask=full, head=m & ~lensm, lensm=lensm, z=z, nrm=nrm, loc=loc,
                col=col.astype(np.uint8), lens=lg, glass=glass_m, iris=iris_m, wall=wall_m,
                face_t=FACE_T & ~lensm, O=O, F=F, U=U, S=S, C=C, pupil=(pc[0], pc[1]), L=L,
                rho_i=rho_i, ang_i=ang_i, rho_g=rho_g, ang_g=ang_g, rho_f=rho_f, ang_f=ang_f,
                front=front_m, bar=bar_m)


def pose_light_level(pd, lights, ambient=-3.2, sky=0.0, sky_dir=(0.0, -1.0, 0.3)):
    """per-pixel light level (steps) for a pose: ambient + sum of point lights.
    lights: list of (pos(x,y,z), power, reach, wrap). Returns (h,w) float."""
    x0, y0, w, h = pd['bbox']
    ys, xs = np.mgrid[y0:y0 + h, x0:x0 + w].astype(np.float64)
    zs = np.where(np.isnan(pd['z']), 0.0, pd['z'])
    nrm = pd['nrm']
    lv = np.full((h, w), float(ambient))
    if sky:
        sd = normalize(sky_dir)
        lv += sky * np.clip((nrm * sd).sum(-1), 0, 1)
    for (pos, power, reach, wrap) in lights:
        dx, dy, dz = pos[0] - xs, pos[1] - ys, pos[2] - zs
        dist = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
        ln = (nrm[..., 0] * dx + nrm[..., 1] * dy + nrm[..., 2] * dz) / dist
        ln = np.clip((ln + wrap) / (1 + wrap), 0, 1)
        fall = np.clip(1 - dist / reach, 0, 1) ** 1.6
        lv += power * ln * fall
    return lv


# ================================================================== the sprout's leaf (shared with c02's design)
def raster_leaf(ang, L, W, droop, dark=0, pal=('leaf3', 'leaf4', 'leaf5', 'leaf6', 'leaf7'),
                edge=('leaf2', 'leaf7', 'leaf6', 'leaf8')):
    """Rasterise a cotyledon directly at angle `ang` (0 = right, +90deg = down),
    base at local (0,0). droop bends the midrib toward screen-down. Returns
    (idx, mask, (R,R) base pixel, tip offset). Same construction as c02."""
    R = int(L + W + 6)
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float64)
    ux, uy = math.cos(ang), math.sin(ang)
    ss = np.linspace(0.0, 1.0, 64)
    best = np.full(xx.shape, 1e9)
    bs = np.zeros(xx.shape)
    bt = np.zeros(xx.shape)
    for s_ in ss:
        mx = ux * s_ * L
        my = uy * s_ * L + droop * L * s_ * s_
        d2 = (xx - mx) ** 2 + (yy - my) ** 2
        better = d2 < best
        best = np.where(better, d2, best)
        bs = np.where(better, s_, bs)
        tx, ty = ux, uy + 2 * droop * s_
        tl = math.hypot(tx, ty)
        side = ((xx - mx) * (-ty / tl) + (yy - my) * (tx / tl))
        bt = np.where(better, side, bt)
    if ux < 0:
        bt = -bt
    dist = np.sqrt(best)
    s_ = bs
    blade = np.clip((s_ - 0.12) / 0.88, 0, 1)
    wprof = W * np.sin(np.pi * blade ** 0.85) ** 0.62
    wprof = np.where(s_ < 0.12, 0.6, wprof)
    t_ = np.sign(bt) * dist
    inside = dist <= wprof + 0.35
    inside &= ~((s_ >= 0.999) & (dist > 0.6))
    rel = t_ / np.maximum(wprof, 0.6)
    v = 0.66 - 0.36 * rel - 0.10 * (blade > 0.82) - 0.14 * dark
    col = quantize(v, list(pal), cuts=[0.30, 0.45, 0.62, 0.84])
    img = np.where(inside, col, 0).astype(np.uint8)
    img = clean(img, inside, 2)
    rib = inside & (np.abs(t_ - 0.2) < 0.5) & (blade > 0.05) & (blade < 0.72)
    img = np.where(rib, ci(pal[1] if not dark else pal[0]), img)
    ol = edge_in(inside)
    img = np.where(ol & (rel > -0.15), ci(edge[0]), img)
    img = np.where(ol & (rel <= -0.15) & (blade > 0.1), ci(edge[2] if dark else edge[1]), img)
    img = np.where(ol & (blade > 0.93), ci(pal[0]), img)
    hl = inside & ~ol & (rel < -0.40) & (blade > 0.25) & (blade < 0.52)
    img = np.where(hl, ci(edge[1] if dark else edge[3]), img)
    tip = (ux * L, uy * L + droop * L)
    return img, inside, (R, R), tip


# ================================================================== RAMP LIGHT (night + lamp, hand-picked ramps)
class RampLight:
    """An artist's two-light shading model in palette-index space.

    Every albedo colour belongs to a material FAMILY (paint, steel, rust, dust...)
    with a brightness OFFSET inside it. Each family has three hand-picked ramps
    of equal indexing: 'neutral' (ambient only), 'warm' (lamp light) and 'cold'
    (moon / sky light). A pixel's ramp position is
        B + offset + ambient_level + (warm or cold level)
    and the dominant light chooses which ramp is read. Levels are floats,
    quantised with ordered dithering only in narrow seams (band).
    Colours not in any family pass through unchanged."""

    def __init__(self, families, B=3):
        self.B = B
        names = list(families.keys())
        K = max(len(r) for f in families.values() for r in (f['neutral'], f['warm'], f['cold']))
        self.K = K
        nf = len(names) + 1
        self.FAM = np.zeros(NPAL, np.int32)
        self.OFF = np.zeros(NPAL, np.int32)
        self.LUT = np.zeros((3, nf, K), np.uint8)       # 0 neutral, 1 warm, 2 cold
        for fi, nm in enumerate(names, start=1):
            f = families[nm]
            for mi, key in enumerate(('neutral', 'warm', 'cold')):
                r = [ci(c) for c in f[key]]
                r = r + [r[-1]] * (K - len(r))
                self.LUT[mi, fi] = r
            for alb, o in f['members'].items():
                self.FAM[ci(alb)] = fi
                self.OFF[ci(alb)] = o
        self.fid = {nm: i for i, nm in enumerate(names, start=1)}

    @staticmethod
    def q(level, band=0.35, ox=0, oy=0, n=4):
        lv = np.asarray(level, np.float64)
        fl = np.floor(lv)
        fr = lv - fl
        if band < 1.0:
            fr = np.clip((fr - 0.5) / max(band, 1e-3) + 0.5, 0, 1)
        return (fl + (bayer(lv.shape[0], lv.shape[1], n, ox, oy) < fr)).astype(np.int32)

    def apply(self, alb, amb, warm, cold, band=0.35, amb_band=None, cold_band=None):
        """alb: (h,w) albedo indices; amb/warm/cold: float level maps (steps).
        amb_band: seam width for the ambient map (narrow = flat planes, few dithers)."""
        f = self.FAM[alb]
        aq = self.q(amb, band if amb_band is None else amb_band)
        wq = np.maximum(self.q(warm, band, 1, 2), 0)
        cq = np.maximum(self.q(cold, band if cold_band is None else cold_band, 2, 1), 0)
        use_w = (wq > 0) & (wq >= cq)
        use_c = (cq > wq)
        lv = np.where(use_w, wq, np.where(use_c, cq, 0))
        pos = np.clip(self.B + self.OFF[alb] + aq + lv, 0, self.K - 1)
        mode = np.where(use_w, 1, np.where(use_c, 2, 0))
        out = self.LUT[mode, f, pos]
        return np.where(f > 0, out, alb).astype(np.uint8)


def level_of(I, I0=0.02, r=1.5):
    """intensity (linear, >=0) -> light level in ramp steps (0 below I0/r)."""
    I = np.asarray(I, np.float64)
    return np.maximum(0.0, 1.0 + np.log(np.maximum(I, 1e-9) / I0) / math.log(r))


# ================================================================== the lens glass, for head_pose poses
def lens_glass(pd, rap=5.0, glow=0.0, glow_pal=('core1', 'core2', 'core3', 'core4', 'core5'),
               halo=('core0', 'rust0'), base=('ink0', 'night1', 'night2'), blade_phase=0.3,
               sheen_dir=(0.6, -0.8), sheen=('night3', 'glass2'), refl=(), glow_scale=1.0):
    """Paint the glass of a head_pose pose (same design as c01/c04: seven dark
    iris blades, a heptagonal aperture, a glowing sensor dot and a reflection
    crescent). Returns (idx, mask) in the pose bbox.
    rap: aperture radius; glow: 0..1 sensor glow; refl: list of
    (screen_dir(dx,dy), reach 0..1, colour, size) small reflections."""
    L = pd['L']
    rr, a = pd['rho_i'], pd['ang_i']
    glass = pd['glass']
    Rg = L['R_GLASS']
    nb = 7
    fb = ((a - blade_phase - 0.8 * np.clip((rr - rap) / max(Rg - rap, 1e-3), 0, 1)) * nb / (2 * math.pi)) % 1.0
    img = quantize(0.2 + 0.4 * fb, list(base), cuts=[0.33, 0.5])
    img = np.where((fb > 0.96) | (fb < 0.03), ci(base[0]), img)
    u, v = rr * np.cos(a), rr * np.sin(a)
    apo = rap * math.cos(math.pi / nb)
    inside = np.ones(rr.shape, bool)
    for k in range(nb):
        ang = blade_phase + 2 * math.pi * k / nb + math.pi / nb
        inside &= (u * math.cos(ang) + v * math.sin(ang)) <= apo
    img = np.where(inside, ci(base[0]), img)
    if glow > 0:
        pr = 0.7 + 1.9 * glow * glow_scale
        core = quantize(-rr + pr, list(glow_pal), cuts=[-1.0, 0.0, 0.9, 1.8], dith=0.5)
        c_in = inside & (rr < pr + 1.2)
        img = np.where(c_in, core, img)
        h1 = inside & (rr < pr + 2.4) & ~c_in
        img = np.where(h1, ci(halo[0]), img)
        h2 = (rr < pr + 3.6 + 2.0 * glow) & ~c_in & ~h1 & glass
        img = np.where(h2 & (TH2_like(rr.shape) < 0.5), ci(halo[1]), img)
    # the reflection crescent (screen space: toward sheen_dir, on the glass rim)
    x0, y0, w, h = pd['bbox']
    gy, gx = np.nonzero(glass)
    if len(gx):
        cx, cy = gx.mean(), gy.mean()
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        dx, dy = xx - cx, yy - cy
        rg = pd['rho_g']
        sd = np.array(sheen_dir, float)
        sd /= np.linalg.norm(sd)
        dd = (dx * sd[0] + dy * sd[1]) / np.maximum(np.hypot(dx, dy), 1e-6)
        cres = glass & (rg > Rg - 2.6) & (rg < Rg - 0.8) & (dd > 0.45)
        img = np.where(cres, ci(sheen[0]), img)
        img = np.where(cres & (dd > 0.85) & (rg > Rg - 1.9), ci(sheen[1]), img)
        # small reflections of things in front of the lens
        ex = max(gx.max() - gx.min(), 1) / 2.0
        ey = max(gy.max() - gy.min(), 1) / 2.0
        for it in refl:
            rd, reach, col, size = it[:4]
            core = it[4] if len(it) > 4 else None
            rd = np.array(rd, float)
            rd /= max(np.linalg.norm(rd), 1e-6)
            px, py = cx + rd[0] * ex * reach, cy + rd[1] * ey * reach
            rx, ry = (size, size) if np.isscalar(size) else size
            e = ((xx - px) / max(rx, 0.3)) ** 2 + ((yy - py) / max(ry, 0.3)) ** 2
            m = glass & (e <= 1.0 + 0.3 / max(rx * ry, 0.3))
            img = np.where(m, ci(col), img)
            if core is not None:
                img = np.where(glass & (e <= 0.3), ci(core), img)
    return img.astype(np.uint8), glass


def TH2_like(shape):
    return bayer(shape[0], shape[1], 2)


def raycast_se_fast(h, w, C, F, U, S, a, n=4.0, zrange=(-400, 400), steps=240, x0=0, y0=0):
    """Same contract as raycast_superellipsoid, faster: integer powers when n is an
    even integer, only candidate pixels are marched, found rays drop out."""
    af, au, as_ = a
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    xs += x0
    ys += y0
    dx = xs - C[0]
    dy = ys - C[1]
    R = max(a) * 1.75
    cand = (dx * dx + dy * dy) <= R * R
    ci_ = np.nonzero(cand.ravel())[0]
    bf = ((dx * F[0] + dy * F[1]) / af).ravel()[ci_]
    bu = ((dx * U[0] + dy * U[1]) / au).ravel()[ci_]
    bs = ((dx * S[0] + dy * S[1]) / as_).ravel()[ci_]
    kf, ku, ks = F[2] / af, U[2] / au, S[2] / as_
    ev = (abs(n - round(n)) < 1e-9) and (int(round(n)) % 2 == 0)
    ni = int(round(n))

    def pw(v):
        if ev:
            v2 = v * v
            out = v2
            for _ in range(ni // 2 - 1):
                out = out * v2
            return out
        return np.abs(v) ** n

    def Fq(bf_, bu_, bs_, dz):
        return pw(bf_ + dz * kf) + pw(bu_ + dz * ku) + pw(bs_ + dz * ks) - 1.0

    zs = np.linspace(zrange[1], zrange[0], steps) + C[2]
    nc = len(ci_)
    zhit = np.full(nc, np.nan)
    zprev = np.full(nc, zs[0])
    active = np.arange(nc)
    for i, z in enumerate(zs):
        if len(active) == 0:
            break
        dz = z - C[2]
        ins = Fq(bf[active], bu[active], bs[active], dz) <= 0
        hit = active[ins]
        zhit[hit] = z
        active = active[~ins]
        zprev[active] = z
    found = ~np.isnan(zhit)
    fi = np.nonzero(found)[0]
    lo = zhit[fi].copy()
    hi = zprev[fi].copy()
    for _ in range(18):
        mid = (lo + hi) / 2
        ins = Fq(bf[fi], bu[fi], bs[fi], mid - C[2]) <= 0
        lo = np.where(ins, mid, lo)
        hi = np.where(ins, hi, mid)
    zf = np.full(h * w, np.nan)
    zf[ci_[fi]] = lo
    z = zf.reshape(h, w)
    m = ~np.isnan(z)
    dz = np.where(m, z - C[2], 0)
    qf = dx * F[0] + dy * F[1] + dz * F[2]
    qu = dx * U[0] + dy * U[1] + dz * U[2]
    qs = dx * S[0] + dy * S[1] + dz * S[2]
    gf = n * np.sign(qf) * np.abs(qf) ** (n - 1) / af ** n
    gu = n * np.sign(qu) * np.abs(qu) ** (n - 1) / au ** n
    gs = n * np.sign(qs) * np.abs(qs) ** (n - 1) / as_ ** n
    nrm = gf[..., None] * F + gu[..., None] * U + gs[..., None] * S
    ln = np.linalg.norm(nrm, axis=-1, keepdims=True)
    nrm = nrm / np.maximum(ln, 1e-9)
    loc = np.stack([qf, qu, qs], -1)
    return m, z, nrm, loc


# ================================================================== CANONICAL CLOSE-UP HEAD (c01 reference)
def _plane_uv(P0, ax1, ax2, xs_, ys_):
    M2 = np.array([[ax1[0], ax2[0]], [ax1[1], ax2[1]]])
    Mi = np.linalg.inv(M2)
    dx, dy = xs_ - P0[0], ys_ - P0[1]
    return Mi[0, 0] * dx + Mi[0, 1] * dy, Mi[1, 0] * dx + Mi[1, 1] * dy


def _hexmask(a, b, R, rot=0.25):
    ok = np.ones_like(a, bool)
    for k in range(6):
        ang = k * math.pi / 3 + rot
        ok &= (a * math.cos(ang) + b * math.sin(ang)) <= R * math.cos(math.pi / 6)
    return ok


def head_pose_canon(P, yaw, nod, roll=0.0, A=(40.0, 34.0, 45.0), n=5.6, L=None, u_off=2.0,
                    pivot=(6.0, 1.08), knurl_phase=0.0, steps=120, stub=True, bolt=True, seed=11):
    """head_pose + the canonical close-up details (c01 is the reference), as
    ALBEDO + normals so the shot can relight it:
      * mint box head, generously rounded corners (n ~5.6 at close-up scale)
      * faceplate groove + side seam (with lit lips) + top cap seam; domed
        rivets on the faceplate and along the side seam
      * the HEX ear-bolt of c01 on the side face (washer + protruding hex head,
        facet normals)
      * damaged state: paint chipped at the top-left (bare steel, broken lips),
        a snapped antenna stub on the crown
      * edge wear: broken 1-px crest highlights on the box edges, steel nicks
      * rust ONLY along seams, at rivets and along the bottom edge — never
        streaks running down from the lens
    Returns head_pose's dict (same keys)."""
    pd = head_pose(P, yaw, nod, roll, A=A, n=n, L=L, u_off=u_off, pivot=pivot, damaged=False,
                   knurl_phase=knurl_phase, steps=steps, ear=False)
    x0, y0, w, h = pd['bbox']
    yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w].astype(np.float64)
    F, U, S, C = pd['F'], pd['U'], pd['S'], pd['C']
    qf, qu, qs = pd['loc'][..., 0], pd['loc'][..., 1], pd['loc'][..., 2]
    nrm = pd['nrm'].copy()
    hm = pd['head'].copy()
    col = pd['col'].copy()
    cf, cs, cu = (nrm * F).sum(-1), (nrm * S).sum(-1), (nrm * U).sum(-1)
    FF = hm & (cf > 0.55)
    FS = hm & (cs > 0.55)
    FT = hm & (cu > 0.55)
    # --- base: one mint albedo; the light makes the planes. Grime low.
    col = np.where(hm, ci('paint3'), col)
    col = np.where(hm & (qu < -A[1] * 0.62), ci('paint2'), col)
    # faceplate groove (rounded rectangle around the lens) with a lighter lip below/right
    fs_, fu_ = A[2] * 0.78, A[1] * 0.82
    g = (np.abs(qs / fs_) ** 8 + np.abs(qu / fu_) ** 8)
    inner = FF & (g < 1.0)
    groove = FF & edge_out(inner) & ~inner
    col = np.where(groove, ci('paint0'), col)
    col = np.where(shifted(groove, 1, 1) & inner & ~groove, ci('paint4'), col)
    # side panel seam (vertical, toward the front of the side face) + top cap seam
    sseam = FS & (np.abs(qf - 0.30 * A[0]) < 0.6)
    col = np.where(sseam, ci('paint0'), col)
    col = np.where(shifted(sseam, 1, 0) & FS & ~sseam, ci('paint4'), col)
    tseam = hm & ~FF & (np.abs(qu - 0.76 * A[1]) < 0.55) & (qf < 0.30 * A[0])
    col = np.where(tseam, ci('paint0'), col)
    col = np.where(shifted(tseam, 0, 1) & hm & ~tseam & ~FF, ci('paint4'), col)

    def dome(region, qf0=None, qu0=None, qs0=None):
        d2 = np.zeros((h, w))
        if qf0 is not None:
            d2 += (qf - qf0) ** 2
        if qu0 is not None:
            d2 += (qu - qu0) ** 2
        if qs0 is not None:
            d2 += (qs - qs0) ** 2
        d2 = np.where(region, d2, 1e9)
        k = int(np.argmin(d2))
        if d2.flat[k] > 25:
            return None
        ry, rx = divmod(k, w)
        for (dx, dy, c) in ((0, -1, 'paint5'), (-1, 0, 'paint5'), (0, 0, 'paint4'), (1, 0, 'paint2'),
                            (0, 1, 'paint2'), (1, 1, 'paint1')):
            if 0 <= ry + dy < h and 0 <= rx + dx < w and hm[ry + dy, rx + dx]:
                col[ry + dy, rx + dx] = ci(c)
        return rx, ry

    rivets = []
    for (sgn_s, sgn_u) in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        r = dome(FF & inner, qf0=A[0], qs0=sgn_s * (fs_ - 4.5), qu0=sgn_u * (fu_ - 4.5))
        if r:
            rivets.append(r)
    for qv in np.linspace(0.50, -0.72, 4) * A[1]:
        r = dome(FS, qf0=0.30 * A[0] - 3.5, qu0=qv)
        if r:
            rivets.append(r)
    # --- edge wear: broken crest highlights where two faces meet, a few bare nicks
    crest = np.zeros((h, w), bool)
    for a_, b_ in ((cf, cs), (cu, cs), (cu, cf)):
        crest |= hm & (np.abs(a_ - b_) < 0.10) & (a_ > 0.45) & (b_ > 0.45)
    n1 = N.fbm1((yy + xx) / 7.0, seed + 20, 3)
    col = np.where(crest & (n1 > 0.45), ci('paint5'), col)
    col = np.where(crest & (N.vnoise2(xx / 2.0, yy / 2.0, seed + 21) > 0.87), ci('steel3'), col)
    # --- damaged: paint chipped away at the TOP-LEFT (the back-top corner of the side face)
    cn = N.vnoise2(xx / 2.6, yy / 2.2, seed + 22)
    chip = hm & (qf < -0.30 * A[0]) & (qu > 0.30 * A[1]) & \
        (((qf + A[0]) / (0.42 * A[0])) ** 2 + ((A[1] - qu) / (0.55 * A[1])) ** 2 + 0.55 * cn < 1.0) & ~FF
    chip = chip & ~edge_in(chip) | (chip & (cn > 0.5))
    col = np.where(chip, ci('steel2'), col)
    col = np.where(chip & (N.vnoise2(xx / 1.5, yy / 1.5, seed + 23) > 0.7), ci('steel3'), col)
    lip_dn = hm & ~chip & shifted(chip, 0, 1)                      # broken paint edge under the chip
    lip_up = hm & ~chip & shifted(chip, 0, -1)
    col = np.where(lip_dn, ci('paint1'), col)
    col = np.where(lip_up & ~lip_dn, ci('paint5'), col)
    # --- rust, only where water sits: bottom edge, foot of the side seam, a rivet or two
    bott = hm & (qu < -A[1] * 0.90) & (N.vnoise1(xx / 3.0 + yy * 0.3, seed + 24) > 0.35)
    col = np.where(bott, ci('rust2'), col)
    col = np.where(bott & (N.vnoise1(xx / 2.0, seed + 25) > 0.72), ci('rust3'), col)
    sfoot = FS & (np.abs(qf - 0.30 * A[0]) < 1.3) & (qu < -A[1] * 0.45)
    col = np.where(sfoot & (N.vnoise1(yy / 2.0, seed + 26) > 0.3), ci('rust2'), col)
    for i, (rx, ry) in enumerate(rivets):
        if i in (3, 6):
            for (dx, dy, c) in ((1, 1, 'rust2'), (2, 1, 'rust1'), (1, 2, 'rust1'), (-1, 1, 'rust2')):
                if 0 <= ry + dy < h and 0 <= rx + dx < w and hm[ry + dy, rx + dx]:
                    col[ry + dy, rx + dx] = ci(c)
    # --- the HEX ear-bolt (c01): washer + hex head protruding along S
    mask = pd['mask'].copy()
    if bolt:
        P0 = C + F * (-0.10 * A[0]) + U * (-0.04 * A[1]) + S * A[2] * 0.995
        a0, b0 = _plane_uv(P0, F, U, xx, yy)
        wr_ = 0.31 * A[1]
        washer = (a0 ** 2 + b0 ** 2) <= wr_ ** 2
        hgt = 0.075 * A[2]
        at, bt = _plane_uv(P0 + S * hgt, F, U, xx, yy)
        R = 0.215 * A[1]
        top = _hexmask(at, bt, R)
        body = np.zeros((h, w), bool)
        for h_ in np.linspace(0, hgt, 6):
            a_, b_ = _plane_uv(P0 + S * h_, F, U, xx, yy)
            body |= _hexmask(a_, b_, R)
        side = body & ~top
        wreg = washer & hm
        col = np.where(wreg, ci('steel2'), col)
        col = np.where(wreg & edge_in(washer), ci('steel1'), col)
        nrm = np.where(wreg[..., None], S[None, None, :], nrm)
        ang = np.arctan2(bt, at)
        k = np.floor((ang - 0.25 + math.pi / 6) / (math.pi / 3)) % 6
        fang = k * math.pi / 3 + 0.25
        fn = np.cos(fang)[..., None] * F + np.sin(fang)[..., None] * U
        breg = body & (hm | shifted(hm, 0, 0))
        col = np.where(side & breg, ci('steel2'), col)
        nrm = np.where((side & breg)[..., None], fn, nrm)
        col = np.where(top & breg, ci('steel3'), col)
        col = np.where(top & breg & edge_side(top, 0, -1), ci('steel5'), col)
        nrm = np.where((top & breg)[..., None], S[None, None, :], nrm)
        col = np.where(breg & (edge_side(body, 1, 0) | edge_side(body, 0, 1)), ci('steel0'), col)
        ring = hm & edge_out(washer) & ~washer
        col = np.where(ring, ci('paint1'), col)
        col = np.where(ring & shifted(washer, -1, -1) & (N.vnoise1(xx / 1.5, seed + 27) > 0.5), ci('rust2'), col)
    # --- the snapped antenna stub on the crown (rusted, jagged top)
    if stub:
        base = C + U * A[1] * 0.97 - F * A[0] * 0.18
        tip = base + U * (0.22 * A[1]) + F * 1.0
        sm, al, ac, dd = thick_curve(h, w, [(base[0] - x0, base[1] - y0), (tip[0] - x0, tip[1] - y0)],
                                     0.075 * A[1], 0.06 * A[1])
        jag = sm & (al > 0.8) & (((xx.astype(int) * 7 + yy.astype(int) * 3) % 5) < 2)
        sm &= ~jag
        sm &= ~(pd['mask'] & ~pd['head'])
        c = np.where(ac < -0.3, ci('rust4'), ci('rust3'))
        c = np.where(ac > 0.5, ci('rust2'), c)
        c = np.where(al > 0.85, ci('rust4'), c)
        c = np.where(al < 0.18, ci('steel1'), c)
        col = np.where(sm, c, col)
        sn = np.stack([ac, np.zeros_like(ac), np.full_like(ac, 0.7)], -1)
        sn = sn / np.linalg.norm(sn, axis=-1, keepdims=True)
        nrm = np.where(sm[..., None], sn, nrm)
        mask |= sm
        hm |= sm
    pd['col'] = col.astype(np.uint8)
    pd['nrm'] = nrm
    pd['mask'] = mask
    pd['head'] = hm
    return pd


# ================================================================== c02's HAND, small (steel, 3 fingers, mint cuff)
def _rseg(xx, yy, x0, y0, ang, L, r, ends=0.55, p=5.0):
    ux, uy = math.cos(ang), math.sin(ang)
    a = (xx - x0) * ux + (yy - y0) * uy
    b = (xx - x0) * (-uy) + (yy - y0) * ux
    half = L / 2 + r * ends
    q = np.abs((a - L / 2) / half) ** p + np.abs(b / r) ** 2.4
    return q <= 1.0, np.clip(b / r, -1, 1), a / max(L, 1e-6)


def mech_hand(h, w, wx, wy, ang, s=1.0, curl=0.35, spread=1.0, cuff=True):
    """The Gardener's hand as drawn in c02, at close-up-insert scale: steel wrist
    ring, a MINT back plate over the palm (the cuff), three segmented steel
    fingers (index on top) with round knuckle caps. (wx, wy) = wrist centre,
    ang = pointing direction (radians, screen; 0 = right, +pi/2 = down).
    curl 0..1 bends the finger tips down. Returns (albedo, mask, normals, tip)
    as full (h, w) arrays (computed in a small box around the hand)."""
    R = int(math.ceil(26.0 * s)) + 4
    bx0, by0 = max(int(wx) - R, 0), max(int(wy) - R, 0)
    bx1, by1 = min(int(wx) + R + 1, w), min(int(wy) + R + 1, h)
    col_f = np.zeros((h, w), np.uint8)
    m_f = np.zeros((h, w), bool)
    n_f = np.zeros((h, w, 3))
    if bx0 >= bx1 or by0 >= by1:
        return col_f, m_f, n_f, (wx, wy)
    c_, m_, n_, tip = _mech_hand_box(by1 - by0, bx1 - bx0, wx - bx0, wy - by0, ang, s, curl, spread, cuff)
    col_f[by0:by1, bx0:bx1] = c_
    m_f[by0:by1, bx0:bx1] = m_
    n_f[by0:by1, bx0:bx1] = n_
    return col_f, m_f, n_f, (tip[0] + bx0, tip[1] + by0)


def _mech_hand_box(h, w, wx, wy, ang, s, curl, spread, cuff):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    col = np.zeros((h, w), np.uint8)
    m = np.zeros((h, w), bool)
    nrm = np.zeros((h, w, 3))
    ux, uy = math.cos(ang), math.sin(ang)
    px_, py_ = -uy, ux                                # 'across' axis (toward +1 = below the hand)

    def put_part(mask, across, c, dim=0):
        n_ = np.stack([px_ * across, py_ * across, np.sqrt(np.clip(1 - across ** 2, 0, 1)) + 0.25], -1)
        n_ = n_ / np.linalg.norm(n_, axis=-1, keepdims=True)
        new = mask & ~m
        col[new] = c[new] if isinstance(c, np.ndarray) else ci(c)
        nrm[new] = n_[new]
        m[new] = True

    def steel(across, dim=0):
        v = 0.56 - 0.40 * across - 0.12 * dim
        return quantize(v, ['steel1', 'steel2', 'steel3', 'steel4'], cuts=[0.30, 0.52, 0.80])

    # fingers first (they are in front of the palm): index (top), middle, ring (behind, darker)
    pl = 6.5 * s
    kx, ky = wx + (2.2 * s + pl) * ux, wy + (2.2 * s + pl) * uy          # knuckle line centre
    tip = (kx, ky)
    knuckles = []
    for k, (off, dang, dim) in enumerate(((-1.8, -0.30, 0), (0.2, 0.0, 1), (2.0, 0.28, 2))):
        fx, fy = kx + px_ * off * s * spread, ky + py_ * off * s * spread
        a1 = ang + dang * spread
        seg = [(4.0 * s, 1.35 * s, a1), (3.2 * s, 1.15 * s, a1 + curl * 1.0 + 0.1 * k)]
        x_, y_ = fx, fy
        for j, (L_, r_, aj) in enumerate(seg):
            fm, fc, fl = _rseg(xx, yy, x_, y_, aj, L_, r_)
            put_part(fm, fc, steel(fc, dim))
            knuckles.append((x_, y_))
            x_, y_ = x_ + L_ * math.cos(aj), y_ + L_ * math.sin(aj)
        if k == 0:
            tip = (x_, y_)
    for (jx, jy) in knuckles:
        jxi, jyi = int(round(jx)), int(round(jy))
        if 0 <= jyi < h and 0 <= jxi < w and m[jyi, jxi]:
            col[jyi, jxi] = ci('steel4')
    # palm: mint back plate on top (the cuff), steel underside
    pm, pc, pl_ = _rseg(xx, yy, wx + 2.2 * s * ux, wy + 2.2 * s * uy, ang, pl, 3.1 * s, 0.6)
    pv = 0.56 - 0.40 * pc
    pcol = np.where(pc < 0.30, quantize(pv, ['paint2', 'paint3', 'paint4', 'paint5'], cuts=[0.30, 0.52, 0.80]),
                    quantize(pv - 0.05, ['steel1', 'steel2', 'steel3'], cuts=[0.35, 0.6]))
    if not cuff:
        pcol = steel(pc)
    put_part(pm, pc, pcol)
    # a rivet on the back plate
    rx, ry = int(round(wx + (2.2 * s + pl * 0.45) * ux - px_ * 1.5 * s)), int(round(wy + (2.2 * s + pl * 0.45) * uy - py_ * 1.5 * s))
    if 0 <= ry < h and 0 <= rx < w and m[ry, rx]:
        col[ry, rx] = ci('paint5')
    # wrist ring (steel band, a little wider than the forearm)
    wm, wc, wl = _rseg(xx, yy, wx - 0.4 * s * ux, wy - 0.4 * s * uy, ang, 2.4 * s, 3.4 * s, 0.1)
    put_part(wm, wc, steel(wc))
    col[wm & (np.abs(wl - 0.5) > 0.42)] = ci('steel1')
    # outline
    ol = edge_out(m)
    col[ol] = ci('steel0')
    nrm[ol] = np.array([0.0, 0.0, 1.0])
    m |= ol
    return col, m, nrm, tip
