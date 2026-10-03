"""
c01 · THE EYE  (global 14.5–16.0, local 0–1.5)

Extreme close-up of the Gardener's camera-lens eye, 3/4 view facing right,
head nodded slightly down (it is looking at the sprout on the ground).
Mint faceplate with seams, rivets, rust; steel barrel with two knurled focus
rings; front bezel with an engraved distance scale; dark glass; 7-blade iris
over the amber pupil; a curved reflection of the grey sky with a tiny GREEN
point (the sprout) in it.

0.00–0.70  lens hunting (servo whirr): focus ring creeps, iris breathes,
           the green point is a soft, dim blur.
0.70       CLICK  (cue eye_focus 15.2): iris snaps smaller + rotates, rings jump,
           1-px servo shake.
0.80       click  (second, smaller step) + shake.
0.80–1.05  green point sharpens & brightens; rings settle (overshoot).
1.05–1.50  hold: the pupil glows, the green point twinkles, dust drifts.
"""
import math
import numpy as np
from engine import *
from engine import noise as NZ
from shots.lib_closeup import (ci, cis, shift, dshift, quantize, clean, ell, disc, edge_in, edge_out,
                               edge_side, shifted, dilate, erode, dist_to, frame_yaw,
                               raycast_superellipsoid, lambert, spec, normalize, puts, put,
                               curve_px, bezier, field, step, damped, smooth, PAL)
from shots.lib_closeup2 import rust_bloom, seam_rust, rivet_rust, edge_rust

DURATION = 1.5
HH, WW = 180, 320
ys, xs = np.mgrid[0:HH, 0:WW].astype(np.float64)
TH4 = bayer(HH, WW, 4)
TH2 = bayer(HH, WW, 2)

# ------------------------------------------------------------------ orientation
BETA = math.radians(34.0)       # yaw: 0 = facing camera, 90 = profile facing right
NOD = math.radians(9.0)         # head nodded down toward the sprout
F, U, S = frame_yaw(BETA, pitch=NOD)
LIGHT = normalize([-0.42, -0.64, 0.66])

O = np.array([171.0, 99.0, 0.0])          # lens base centre (world; screen xy)
NB = float(np.hypot(F[0], F[1]))           # |F.xy|
E1 = np.array([F[0], F[1]]) / NB           # screen dir of the lens axis
B3 = np.array([F[1], -F[0], 0.0]) / NB     # ring basis (no z)
A3 = np.cross(B3, F)                       # ring basis (toward viewer)
E2 = B3[:2]
FZ = F[2]
_px = xs - O[0]
_py = ys - O[1]
P1 = _px * E1[0] + _py * E1[1]             # along the axis (screen)
P2 = _px * E2[0] + _py * E2[1]             # across the axis (screen)


UA, VB = -A3, -B3                          # ring-plane basis: u ~ screen right, v ~ screen down


def ring_local(d):
    """local (u, v) coords of each pixel on the ring plane at depth d
    (u ~ right, v ~ down on screen; angle 0 = right, +90deg = down)."""
    a = (P1 - NB * d) / (-FZ)
    return -a, -P2


def rho_ang(d):
    u, v = ring_local(d)
    return np.sqrt(u * u + v * v), np.arctan2(v, u)


def screen_pt(d, r, ang):
    """screen xy of the ring point (depth d, radius r, local angle ang)."""
    P = O + F * d + r * (UA * math.cos(ang) + VB * math.sin(ang))
    return P[0], P[1]


def cyl_side(r, d0, d1):
    """visible side of the cylinder radius r between depths d0..d1:
    returns mask, depth map, theta map, normal (h,w,3)."""
    s = np.clip(P2 / r, -1, 1)
    c = np.sqrt(1 - s * s)
    d = (P1 + r * FZ * c) / NB
    m = (np.abs(P2) <= r + 0.3) & (d >= d0) & (d <= d1)
    nrm = A3[None, None, :] * c[..., None] + B3[None, None, :] * s[..., None]
    return m, d, np.arctan2(s, c), nrm, c


# lens construction (radius r in 3D units; depth d along F)
R_COL, D_COL0, D_COL1 = 67.0, -3.0, 3.0
R_BAR, D_BAR0, D_FRONT = 60.0, 3.0, 26.0
R_BEZ_IN, R_MARK_IN, R_RET_IN = 52.0, 45.5, 42.0
D_GLASS, D_IRIS, D_PUP = 24.5, 13.0, 9.0

STEEL = ['ink1', 'steel0', 'steel1', 'steel2', 'steel3', 'steel4', 'steel5', 'steel6']


# ------------------------------------------------------------------ background
def _background():
    th = TH4
    v = 1.0 - ys / 120.0 + 0.12 * (1 - xs / 320)
    cols = cis(['dust5', 'dust6', 'dust7', 'dust8'])
    lev = np.zeros((HH, WW), int)
    for c in (0.15, 0.52, 0.93):
        a = np.clip((v - (c - 0.09)) / 0.18, 0, 1)
        lev += th < a
    img = cols[np.clip(lev, 0, 3)]
    hz = 131 + 2.5 * np.sin(xs / 31.0) + 1.5 * np.sin(xs / 11.0 + 1)
    img = np.where(th < np.clip((ys - hz) / 7.0 + 0.5, 0, 1), ci('dust5'), img)
    img = np.where(th < np.clip((ys - hz - 12) / 14.0 + 0.5, 0, 1), ci('dust4'), img)
    return img.astype(np.uint8)


# ------------------------------------------------------------------ head
HEAD_A = (104.0, 96.0, 120.0)
HEAD_N = 7.0
U_OFF = 5.0
HEAD_C = O - F * HEAD_A[0] * 0.985 - U * U_OFF
PAINT = ['paint0', 'paint1', 'paint2', 'paint3', 'paint4', 'paint5']


def _head():
    m, z, nrm, loc = raycast_superellipsoid(HH, WW, HEAD_C, F, U, S, HEAD_A, n=HEAD_N,
                                            zrange=(-240, 240), steps=220)
    dif = lambert(nrm, LIGHT, wrap=0.15)
    sp = spec(nrm, LIGHT, 18.0)
    v = 0.10 + 0.78 * dif + 0.30 * sp + 0.12 * np.clip(-nrm[..., 1], 0, 1)
    # faces darken gently toward the bottom (grime + less sky light)
    v = v - 0.09 * np.clip((ys - 60) / 120, 0, 1)
    # ambient occlusion around the lens collar
    rc, _ = rho_ang(0.0)
    ao = np.clip((R_COL + 8 - rc) / 8.0, 0, 1) * (rc > R_COL - 1)
    v = v - 0.14 * ao
    v = np.where(m, v, 0)
    idx = quantize(v, PAINT, cuts=[0.16, 0.28, 0.46, 0.745, 0.905])
    idx = clean(idx, m, 3)
    # cast shadow of the lens barrel on the faceplate (light from upper-left)
    u0, v0 = ring_local(0.0)
    sh = np.sqrt((u0 - 9.0) ** 2 + (v0 - 11.0) ** 2) <= R_COL + 1.5
    rc_ = np.sqrt(u0 ** 2 + v0 ** 2)
    shadow = m & sh & (rc_ > R_COL)
    idx = np.where(shadow, shift(idx, -1), idx)
    return m, idx.astype(np.uint8), loc, nrm


BG = _background()
HEAD_M, HEAD_IDX, HEAD_LOC, HEAD_NRM = _head()
QF, QU, QS = HEAD_LOC[..., 0], HEAD_LOC[..., 1], HEAD_LOC[..., 2]
FACE_F = HEAD_M & ((HEAD_NRM * F).sum(-1) > 0.55)       # front face
FACE_S = HEAD_M & ((HEAD_NRM * S).sum(-1) > 0.55)       # side face
FACE_T = HEAD_M & ((HEAD_NRM * U).sum(-1) > 0.55)       # top face


def _sel_edges(img):
    """silhouette treatment: bright rim on top, dark line where the head
    turns away on the right, sky-lit edge on the far top-right."""
    m = HEAD_M
    top = edge_side(m, 0, -1)
    img[top] = ci('paint5')
    right = edge_side(m, 1, 0) & ~top
    img[right] = ci('paint1')
    return img


def _groove(img, mask, lit_dir=(1, 1)):
    """engraved seam: 1px dark line, 1px lighter lip on the lit side."""
    img[mask] = ci('paint1')
    lip = shifted(mask, lit_dir[0], lit_dir[1]) & HEAD_M & ~mask
    img[lip] = shift(img, 1)[lip]


def _rivet(img, x, y, big=False):
    """domed rivet in the surrounding paint: lit top-left, dark rim bottom-right,
    one crisp highlight pixel, contact shadow."""
    base = np.array([[img[y, x]]])
    c = {k: int(shift(base, k)[0, 0]) for k in (-3, -2, -1, 0, 1, 2)}
    pat = ['.bcb.',
           'bdcbA',
           'ccbbA',
           'bbbAA',
           '.AAA.']
    cmap = {'b': c[0], 'c': c[1], 'd': c[2], 'A': c[-2]}
    for j, row in enumerate(pat):
        for i, ch in enumerate(row):
            if ch == '.':
                continue
            px, py = x - 2 + i, y - 2 + j
            if 0 <= px < WW and 0 <= py < HH and HEAD_M[py, px]:
                img[py, px] = cmap[ch]
    for (dx, dy) in ((3, 0), (3, 1), (2, 2), (1, 3), (0, 3)):
        px, py = x + dx, y + dy
        if 0 <= px < WW and 0 <= py < HH and HEAD_M[py, px]:
            img[py, px] = c[-1]


def _streak(img, x, y, length, seed=0, width=2, mask=None):
    """rust run-off from (x,y): a bright rust bead at the source, then a
    subdued tapering stain (rust2 -> rust1) that dithers out into darkened
    paint. Slight meander; never a uniform stick."""
    mask = HEAD_M if mask is None else mask
    w0 = width + 0.5
    xx = float(x)
    for k in range(length):
        yy = y + k
        if not (0 <= yy < HH):
            break
        xx += (float(NZ.vnoise1(np.array([k / 6.0]), seed)[0]) - 0.5) * 0.5
        taper = 1 - k / length
        wd = max(1.0, w0 * taper ** 0.7)
        xa = xx - wd / 2
        p0 = int(math.floor(xa + 0.5))
        p1 = max(int(math.floor(xa + wd + 0.5)), p0 + 1)
        for px in range(p0, p1):
            if not (0 <= px < WW and mask[yy, px]):
                continue
            edge = (px == p1 - 1) and (p1 - p0) > 1
            if k <= 1:
                col = 'rust4' if (k == 0 and not edge) else 'rust3'
            elif taper > 0.55:
                col = 'rust1' if edge else 'rust2'
            elif taper > 0.3:
                if TH2[yy, px] >= 0.5:
                    img[yy, px] = shift(img[yy:yy + 1, px:px + 1], -1)[0, 0]
                    continue
                col = 'rust1'
            else:
                if TH4[yy, px] >= 0.3:
                    if TH4[yy, px] < 0.65:
                        img[yy, px] = shift(img[yy:yy + 1, px:px + 1], -1)[0, 0]
                    continue
                col = 'rust1'
            img[yy, px] = ci(col)


def _rust_patch(img, x, y, r, seed=0, mask=None):
    """small bloom of surface rust (clustered, not speckle)."""
    mask = HEAD_M if mask is None else mask
    x0, x1 = int(x - r - 2), int(x + r + 3)
    y0, y1 = int(y - r - 2), int(y + r + 3)
    for py in range(max(y0, 0), min(y1, HH)):
        for px in range(max(x0, 0), min(x1, WW)):
            d = math.hypot((px - x) * 0.8, py - y)
            n_ = float(NZ.vnoise2(np.array(px / 2.3), np.array(py / 2.3), seed))
            if d < r * (0.55 + 0.6 * n_) and mask[py, px]:
                img[py, px] = ci('rust2' if n_ > 0.55 else 'rust1')
                if n_ > 0.8:
                    img[py, px] = ci('rust3')


def _scratch(img, pts, mask=None):
    mask = HEAD_M if mask is None else mask
    px = curve_px(pts)
    for (x, y) in px:
        if 0 <= x < WW and 0 <= y + 1 < HH and mask[y + 1, x]:
            img[y + 1, x] = shift(img[y + 1:y + 2, x:x + 1], -2)[0, 0]
    for (x, y) in px:
        if 0 <= x < WW and 0 <= y < HH and mask[y, x]:
            img[y, x] = shift(img[y:y + 1, x:x + 1], 2)[0, 0]


def _find(qf=None, qu=None, qs=None, region=None):
    d2 = np.zeros((HH, WW))
    if qf is not None:
        d2 += (QF - qf) ** 2
    if qu is not None:
        d2 += (QU - qu) ** 2
    if qs is not None:
        d2 += (QS - qs) ** 2
    reg = HEAD_M if region is None else region
    d2 = np.where(reg, d2, 1e12)
    k = int(np.argmin(d2))
    y, x = divmod(k, WW)
    return x, y, d2.flat[k]


def _plane_local(P0, ax1, ax2):
    """screen pixel -> local coords on the plane through P0 spanned by ax1, ax2."""
    M2 = np.array([[ax1[0], ax2[0]], [ax1[1], ax2[1]]])
    Minv = np.linalg.inv(M2)
    dx, dy = xs - P0[0], ys - P0[1]
    return Minv[0, 0] * dx + Minv[0, 1] * dy, Minv[1, 0] * dx + Minv[1, 1] * dy


def _hexm(a, b, R, rot=0.25):
    ok = np.ones_like(a, bool)
    for k in range(6):
        ang = k * math.pi / 3 + rot
        ok &= (a * math.cos(ang) + b * math.sin(ang)) <= R * math.cos(math.pi / 6)
    return ok


def _ear_bolt(img):
    """the side ear-bolt (continuity with the model), at the far left edge:
    washer + hex head protruding along S, facets lit by facet normal."""
    P0 = HEAD_C + F * (-0.10 * HEAD_A[0]) + U * (-0.04 * HEAD_A[1]) + S * HEAD_A[2] * 0.995
    a0, b0 = _plane_local(P0, F, U)
    washer = (a0 ** 2 + b0 ** 2) <= 31 ** 2
    hgt = 8.0
    at, bt = _plane_local(P0 + S * hgt, F, U)
    top = _hexm(at, bt, 22.0)
    body = np.zeros_like(top)
    for h_ in np.linspace(0, hgt, 12):
        a_, b_ = _plane_local(P0 + S * h_, F, U)
        body |= _hexm(a_, b_, 22.0)
    side = body & ~top
    reg = HEAD_M & (washer | body)
    # washer: flat ring facing S, slight radial bevel lighting
    wr = np.sqrt(a0 ** 2 + b0 ** 2)
    wa = np.arctan2(b0, a0)
    nw = S[None, None, :] + 0.35 * (np.cos(wa)[..., None] * F + np.sin(wa)[..., None] * U) * (wr[..., None] > 26)
    nw = nw / np.linalg.norm(nw, axis=-1, keepdims=True)
    wv = 0.1 + 0.8 * lambert(nw, LIGHT, 0.2)
    wc = quantize(wv, ['steel1', 'steel2', 'steel3', 'steel4'], cuts=[0.3, 0.5, 0.68])
    img = np.where(HEAD_M & washer, wc, img)
    img = np.where(HEAD_M & washer & edge_in(washer), ci('steel1'), img)
    # hex facets: normal = outward direction of each facet in the (F,U) plane
    ang = np.arctan2(bt, at)
    k = np.floor((ang - 0.25 + math.pi / 6) / (math.pi / 3)) % 6
    fang = k * math.pi / 3 + 0.25
    fn = np.cos(fang)[..., None] * F + np.sin(fang)[..., None] * U
    fv = 0.12 + 0.8 * lambert(fn, LIGHT, 0.1)
    sc = quantize(fv, ['steel0', 'steel1', 'steel2', 'steel3', 'steel4'], cuts=[0.2, 0.35, 0.52, 0.7])
    img = np.where(HEAD_M & side, sc, img)
    tv = 0.1 + 0.8 * lambert(np.broadcast_to(S, (HH, WW, 3)), LIGHT) + 0.004 * (at - bt)
    tc = quantize(tv, ['steel3', 'steel4', 'steel5'], cuts=[0.58, 0.72])
    img = np.where(HEAD_M & top, tc, img)
    img = np.where(HEAD_M & top & edge_side(top, 0, -1), ci('steel6'), img)
    img = np.where(HEAD_M & body & edge_side(body, 1, 0), ci('steel0'), img)
    img = np.where(HEAD_M & body & edge_side(body, 0, 1), ci('steel0'), img)
    # rust bloom where the bolt meets the paint (bottom-right)
    ring = HEAD_M & edge_out(washer) & ~washer
    img = np.where(ring, ci('paint1'), img)
    return img


EMBLEM = [  # faded cream stencil: a sprout (the Gardener's unit emblem), foreshortened
    '.##..##',
    '####.##',
    '.###.#.',
    '...##..',
    '...#...',
    '...#...',
    '.#####.',
]


def _emblem(img, x0, y0):
    for j, row in enumerate(EMBLEM):
        for i, ch in enumerate(row):
            if ch != '#':
                continue
            x, y = x0 + i, y0 + j
            if not HEAD_M[y, x]:
                continue
            if (i, j) in ((1, 1), (5, 6), (2, 6)):
                continue            # a few chips in the stencil paint
            img[y, x] = ci('dust8')
    return img


def _dent(img, x, y, rx, ry):
    """small dent: upper-left inner rim in shadow, lower-right rim catches light."""
    for py in range(int(y - ry - 1), int(y + ry + 2)):
        for px in range(int(x - rx - 1), int(x + rx + 2)):
            if not (0 <= px < WW and 0 <= py < HH and HEAD_M[py, px]):
                continue
            d = ((px - x) / rx) ** 2 + ((py - y) / ry) ** 2
            if 0.45 < d <= 1.15:
                side = (px - x) / rx + (py - y) / ry
                if side < -0.3:
                    img[py, px] = shift(img[py:py + 1, px:px + 1], -1)[0, 0]
                elif side > 0.5:
                    img[py, px] = shift(img[py:py + 1, px:px + 1], 1)[0, 0]


def _wear(img):
    """edge wear as a crisp, broken 1-px crest highlight along the box edges
    (where the normal is halfway between two faces) + a few dust clusters
    settled on the top face near the back."""
    cf = (HEAD_NRM * F).sum(-1)
    cs = (HEAD_NRM * S).sum(-1)
    cu = (HEAD_NRM * U).sum(-1)
    crest = np.zeros((HH, WW), bool)
    for a_, b_ in ((cf, cs), (cu, cs), (cu, cf)):
        crest |= HEAD_M & (np.abs(a_ - b_) < 0.10) & (a_ > 0.45) & (b_ > 0.45)
    # thin it to 1 px: keep the pixel with the smallest |a-b| per row/col
    n1 = NZ.fbm1((ys + xs) / 9.0, 31, 3)
    dash = crest & (n1 > 0.42)
    img = np.where(dash & (ys < 160), ci('paint5'), img)
    # a few bare-metal nicks on the crest
    nick = crest & (NZ.vnoise2(xs / 2.0, ys / 2.0, 44) > 0.86)
    img = np.where(nick, ci('steel3'), img)
    # dust settled on the top face (warm, clustered, toward the back)
    return img


def _collar_rust(img):
    """rust IN the crevice under the collar (water sits there): a broken dark
    line hugging the collar with a few low, wide blooms — it spreads along
    the seam, it never runs down the face."""
    rc, ac = rho_ang(0.0)
    angd = np.degrees(ac)
    crev = HEAD_M & ~LENS_M & (rc > R_COL + 0.2) & (rc < R_COL + 1.7) & (angd > 38) & (angd < 150)
    img = seam_rust(img, crev, HEAD_M, seed=51, density=0.62, scale=3.0)
    # the lowest arc holds water longest: a solid 2-px rust line in the crevice
    low = HEAD_M & ~LENS_M & (rc > R_COL + 0.2) & (rc < R_COL + 2.4) & (angd > 72) & (angd < 122)
    img = seam_rust(img, low, HEAD_M, seed=52, density=0.9, scale=4.0)
    for (ang, rx, ry, sd) in ((64, 3.0, 1.2, 3), (90, 4.4, 1.6, 1), (106, 3.2, 1.4, 4), (121, 3.0, 1.3, 2),
                              (139, 2.0, 1.0, 7)):
        x, y = screen_pt(0.0, R_COL + 2.3, math.radians(ang))
        img = rust_bloom(img, x, y, rx, ry, seed=60 + sd, mask=HEAD_M & ~LENS_M)
    return img


def _weathering(img, riv, groove, seam):
    """rust only where water sits: the lower runs of the faceplate groove and
    the side seam, the feet of the lower rivets, the washer of the ear-bolt,
    and the rounded bottom-right edge of the head."""
    img = seam_rust(img, groove & (ys > 118), HEAD_M, seed=71, density=0.5)
    img = seam_rust(img, groove & (ys > 160), HEAD_M, seed=73, density=0.75)
    img = seam_rust(img, seam & (ys > 104), HEAD_M, seed=72, density=0.55)
    for i, (x, y) in enumerate(riv):
        if y > 96 and NZ.rand(i, 3, 77) > 0.25:
            img = rivet_rust(img, x, y, r=2.2, seed=80 + i, mask=HEAD_M)
            img = rust_bloom(img, x + 2.5, y + 2.2, 1.6, 0.9, seed=85 + i, mask=HEAD_M, crust=False)
    # where the groove turns under the lens a little rust has pooled
    img = rust_bloom(img, 103, 171, 2.8, 1.3, seed=90, mask=HEAD_M & ~LENS_M)
    # bottom-right edge: the rounded corner where the face turns under
    img = edge_rust(img, HEAD_M & (ys > 128), depth=2, seed=91, density=0.62, scale=4.5, side=(1, 0))
    img = edge_rust(img, HEAD_M & (ys > 156) & (xs > 238), depth=3, seed=92, density=0.6, scale=4.5, side=(1, 1))
    for (x, y, rx, ry, sd) in ((251, 160, 2.2, 1.2, 94), (246, 173, 2.6, 1.3, 95)):
        img = rust_bloom(img, x, y, rx, ry, seed=sd, mask=HEAD_M)
    # the ear-bolt's washer: rust at its lower-right rim
    img = rust_bloom(img, 24, 104, 2.4, 1.3, seed=93, mask=HEAD_M)
    return img


def _head_details(img):
    img = _sel_edges(img)
    # faceplate groove on the front face: rounded rectangle in (qs, qu)
    fs, fu = 92.0, 75.0
    g = (np.abs(QS / fs) ** 8 + np.abs(QU / fu) ** 8)
    inner = FACE_F & (g < 1.0)
    gm = FACE_F & edge_out(inner) & ~inner
    _groove(img, gm, lit_dir=(1, 1))
    # side panel seam (vertical) + top cap seam (horizontal)
    sm = FACE_S & (np.abs(QF - (0.30 * HEAD_A[0])) < 0.7)
    _groove(img, sm, lit_dir=(1, 0))
    tm = HEAD_M & (np.abs(QU - 0.76 * HEAD_A[1]) < 0.6) & ~FACE_F & (QF < 0.30 * HEAD_A[0])
    _groove(img, tm, lit_dir=(0, 1))
    riv = []
    for (qs_, qu_) in [(fs - 7, fu - 7), (-(fs - 7), fu - 7), (fs - 7, 0.0), (0.0, fu - 7),
                       (fs - 7, -(fu - 7)), (-(fs - 7), -(fu - 7))]:
        x, y, d2 = _find(qf=HEAD_A[0], qs=qs_, qu=qu_, region=FACE_F & inner)
        if d2 < 60 and 3 < x < WW - 3 and 3 < y < HH - 3:
            riv.append((x, y))
    for qu_ in np.linspace(0.55, -0.9, 6) * HEAD_A[1]:
        x, y, d2 = _find(qf=0.30 * HEAD_A[0] - 8, qu=qu_, region=FACE_S)
        if d2 < 60 and 3 < x < WW - 3 and 3 < y < HH - 3:
            riv.append((x, y))
    for (x, y) in riv:
        _rivet(img, x, y, big=True)
    img = _wear(img)
    img = _collar_rust(img)
    img = _weathering(img, riv, gm, sm)
    _dent(img, 30, 140, 3.5, 2.5)
    _dent(img, 232, 150, 2.5, 2.0)
    # a little rust bloom at the seam foot and near the lower faceplate rivet
    # scratches (few, deliberate)
    _scratch(img, [(40, 30), (50, 27), (57, 26)])
    _scratch(img, [(88, 141), (97, 138)])
    _scratch(img, [(247, 118), (253, 122), (256, 126)])
    img = _emblem(img, 40, 40)
    img = _ear_bolt(img)
    return img


# ------------------------------------------------------------------ lens static maps
BAR_M, BAR_D, BAR_T, BAR_N, BAR_C = cyl_side(R_BAR, D_BAR0, D_FRONT)
COL_M, COL_D, COL_T, COL_N, _ = cyl_side(R_COL, D_COL0, D_COL1)
RHO_F, ANG_F = rho_ang(D_FRONT)
FRONT_M = RHO_F <= R_BAR + 0.35
COLF_R, COLF_A = rho_ang(D_COL1)
COLF_M = (COLF_R <= R_COL + 0.35) & ~FRONT_M          # collar's front lip ring
RHO_G, ANG_G = rho_ang(D_GLASS)
GLASS_M = RHO_G <= R_RET_IN + 0.3
RHO_I, ANG_I = rho_ang(D_IRIS)
IRIS_M = GLASS_M & (RHO_I <= R_RET_IN + 0.2)
WALL_M = GLASS_M & ~IRIS_M
LENS_M = BAR_M | COL_M | COLF_M | FRONT_M
IA, IB = ring_local(D_IRIS)


def _collar_static():
    img = np.zeros((HH, WW), np.uint8)
    v = 0.12 + 0.75 * lambert(COL_N, LIGHT, 0.25) + 0.35 * spec(COL_N, LIGHT, 20)
    side = quantize(v, ['paint0', 'paint1', 'paint2', 'paint3', 'paint4'], cuts=[0.22, 0.38, 0.56, 0.76])
    img = np.where(COL_M, side, img)
    # collar front lip (flat ring facing F): lit a bit less than the top face
    fv = 0.45 + 0.25 * np.cos(COLF_A + 2.2)
    lip = quantize(fv, ['paint2', 'paint3', 'paint4'], cuts=[0.4, 0.62])
    img = np.where(COLF_M, lip, img)
    # dark crevice where barrel meets collar lip; bright edge at the lip's outer rim (top-left)
    img = np.where(COLF_M & (COLF_R > R_COL - 1.2) & (np.cos(COLF_A + 2.2) > 0.2), ci('paint5'), img)
    img = np.where(COLF_M & (COLF_R < R_BAR + 1.6), ci('paint0'), img)
    img = np.where(COL_M & (COL_D < D_COL0 + 1.0), ci('paint0'), img)
    # 6 hex screws on the collar lip
    for k in range(6):
        ang = math.radians(30 + 60 * k)
        x, y = screen_pt(D_COL1, (R_COL + R_BAR) / 2 + 0.5, ang)
        x, y = int(round(x)), int(round(y))
        for (dx, dy, c) in ((0, -1, 'steel5'), (-1, 0, 'steel4'), (0, 0, 'steel4'), (1, 0, 'steel2'),
                            (0, 1, 'steel2'), (1, 1, 'paint1'), (2, 0, 'paint1'), (0, 2, 'paint1')):
            if COLF_M[y + dy, x + dx]:
                img[y + dy, x + dx] = ci(c)
    return img


COL_IDX = _collar_static()


def _front_static():
    img = np.zeros((HH, WW), np.uint8)
    r, a = RHO_F, ANG_F
    ca, sa = np.cos(a), np.sin(a)
    radial = UA[None, None, :] * ca[..., None] + VB[None, None, :] * sa[..., None]

    def bevel_v(tilt):
        n = F[None, None, :] + tilt * radial
        n = n / np.linalg.norm(n, axis=-1, keepdims=True)
        return 0.06 + 0.66 * lambert(n, LIGHT, 0.15) + 0.9 * spec(n, LIGHT, 36)
    v = np.where(r > 57.5, bevel_v(1.5), np.where(r > 54.5, bevel_v(0.0), bevel_v(-1.3)))
    bez = quantize(v, STEEL[1:], cuts=[0.16, 0.28, 0.42, 0.58, 0.74, 0.95])
    img = np.where(r > R_BEZ_IN, bez, img)
    img = clean(img, FRONT_M & (r > R_BEZ_IN), 2)
    # silhouette + separation lines
    img = np.where((r > 59.3), shift(img, -2), img)
    img = np.where((np.abs(r - 54.5) < 0.5), shift(img, -1), img)
    # marking ring: dark anodised, faint sheen at upper-left
    mk = (r <= R_BEZ_IN) & (r > R_MARK_IN)
    sheen = np.cos(a + 2.4)
    mv = quantize(sheen, ['steel0', 'steel1'], cuts=[0.72])
    img = np.where(mk, mv, img)
    img = np.where(mk & (np.abs(r - R_BEZ_IN) < 0.7), ci('ink1'), img)
    # retainer ring (recessed): dark with a thin catch-light lower-right
    rr = RHO_G
    ret = (r <= R_MARK_IN) & (rr > R_RET_IN)
    img = np.where(ret, ci('ink1'), img)
    band = ret & (np.abs(rr - R_RET_IN - 1.4) < 0.6)
    img = np.where(band & (np.cos(ANG_G - 0.7) > 0.55), ci('steel2'), img)
    img = np.where(band & (np.cos(ANG_G + 2.4) > 0.7), ci('steel1'), img)
    return img


FRONT_IDX = _front_static()
MARK_M = FRONT_M & (RHO_F <= R_BEZ_IN - 0.5) & (RHO_F > R_MARK_IN + 0.5)


def _wall_static():
    d = RHO_I - R_RET_IN
    return quantize(np.clip(d / 8.0, 0, 1), ['steel1', 'steel0', 'ink1'], cuts=[0.3, 0.65])


WALL_IDX = _wall_static()

# ------------------------------------------------------------------ reflection on the glass
GA, GB = ring_local(D_GLASS)
_rr = np.sqrt(GA ** 2 + GB ** 2)
# crescent: inside the glass, outside a circle pushed toward lower-right (in local coords)
# local a axis points 'back' on screen (left); b axis points down-ish.
_cut = np.sqrt((GA - 14) ** 2 + (GB - 17) ** 2)
REFL_M = GLASS_M & (_rr <= R_RET_IN - 1.0) & (_cut > R_RET_IN + 1.0)
# reflected horizon (curved, fish-eye): sky above, ground below
_hz = GB - (-4.0 + 0.30 * (GA + 18.0) + 0.004 * (GA + 18.0) ** 2)
HORIZ = _hz > 0
REFL_SKY = REFL_M & ~HORIZ
REFL_GND = REFL_M & HORIZ
HLINE = REFL_M & HORIZ & ~shifted(HORIZ, 0, 1)


def _refl_static():
    img = np.zeros((HH, WW), np.uint8)
    edge = (R_RET_IN - 1.0) - _rr                  # 0 at the outer rim
    inner = _cut - (R_RET_IN + 1.0)                # 0 at the inner cut
    img = np.where(REFL_SKY, ci('ash5'), img)
    img = np.where(REFL_SKY & (edge < 6.0), ci('ash6'), img)
    img = np.where(REFL_SKY & (edge < 2.4), ci('ash7'), img)
    img = np.where(REFL_SKY & (inner < 2.0), ci('ash4'), img)
    img = np.where(REFL_GND, ci('night3'), img)
    img = np.where(REFL_GND & (edge < 2.2), ci('ash3'), img)
    img = np.where(REFL_GND & shifted(HLINE, 0, 1), ci('ash3'), img)
    img = np.where(HLINE, ci('dust6'), img)
    return img


REFL_IDX = _refl_static()
REFL_SOFT = REFL_M & (((_cut - (R_RET_IN + 1.0)) < 1.6))   # dithered inner edge


def _gp():
    """green point: left of the pupil on the dark glass, where the reflected
    horizon line points to."""
    x, y = screen_pt(D_IRIS, 25.0, math.radians(197))
    return int(round(x)), int(round(y))


GP = _gp()

# ------------------------------------------------------------------ static composite
BASE = np.where(HEAD_M, HEAD_IDX, BG).astype(np.uint8)
BASE = _head_details(BASE)


_vd = np.sqrt(((xs - 175) / 205.0) ** 2 + ((ys - 92) / 128.0) ** 2)
VIG = (_vd + 0.10 * (TH4 - 0.5)) > 1.0          # 1-step corner vignette with a dithered seam

# ------------------------------------------------------------------ timing
T_CLICK1, T_CLICK2 = 0.70, 0.80


def focus_state(t):
    tq = step(t, 24)
    if t < T_CLICK1:
        rap = 15.5 + 0.8 * math.sin(tq * 4.2 + 0.4)
        rot = 0.03 * math.sin(tq * 3.1)
        phA = 0.30 * tq + 0.05 * math.sin(tq * 9.0)
        phB = -0.08 * tq
    else:
        rap = 11.5 if t < T_CLICK2 else 8.0 - 1.2 * max(0.0, 1 - (t - T_CLICK2) / 0.07) + 0.45 * damped(t - 0.87, 5, 9)
        rot = 0.22 if t < T_CLICK2 else 0.36 + 0.03 * damped(t - T_CLICK2, 5, 9)
        base = 0.30 * T_CLICK1
        phA = base + 1.5 * smooth((t - T_CLICK1) / 0.13) + 0.22 * damped(t - 0.83, 4, 8)
        phB = -0.056 - 0.8 * smooth((t - T_CLICK2) / 0.15) - 0.12 * damped(t - 0.95, 4, 8)
    focus = smooth((t - 0.74) / 0.32)
    sx = sy = 0
    for t0 in (T_CLICK1, T_CLICK2):
        dt = t - t0
        if 0 <= dt < 1 / 30:
            sx, sy = 1, (1 if t0 == T_CLICK1 else 0)
        elif 1 / 30 <= dt < 2 / 30:
            sx, sy = -1, 0
    power = -1.2 if t < T_CLICK1 else (-0.4 if t < T_CLICK2 else 0.2)
    if T_CLICK1 <= t < T_CLICK1 + 1 / 30 or T_CLICK2 <= t < T_CLICK2 + 1 / 30:
        power = 2.2                      # one-frame flare on each click
    power += 0.35 * math.sin((t - 1.0) * 7.0) * (t > 1.0)
    return dict(rap=rap, rot=rot, phA=phA, phB=phB, focus=focus, sx=sx, sy=sy, power=power)


# ------------------------------------------------------------------ per-frame lens parts
_lys, _lxs = np.nonzero(dilate(LENS_M, 3))
SL = (slice(int(_lys.min()), int(_lys.max()) + 1), slice(int(_lxs.min()), int(_lxs.max()) + 1))

KN_A = (5.5, 12.0)
KN_B = (15.0, 22.5)
BAR_V = 0.08 + 0.62 * lambert(BAR_N, LIGHT, 0.25) + 0.55 * spec(BAR_N, LIGHT, 26)
BAR_BASE = quantize(BAR_V, STEEL[1:7], cuts=[0.18, 0.3, 0.45, 0.62, 0.82])


def _barrel(phA, phB):
    img = BAR_BASE[SL].copy()
    m = BAR_M[SL]
    d, th, c = BAR_D[SL], BAR_T[SL], BAR_C[SL]
    for (d0, d1), ph, nk in ((KN_A, phA, 80), (KN_B, phB, 60)):
        band = m & (d >= d0) & (d <= d1)
        f = (th * nk / (2 * math.pi) + ph) % 1.0
        mod = np.where(f < 0.25, 1, np.where(f < 0.5, 0, np.where(f < 0.75, -1, 0)))
        mod = np.where(c > 0.45, mod, np.where(f < 0.5, 0, -1))
        img = np.where(band, shift(img, mod), img)
        img = np.where(m & (np.abs(d - d1) < 0.6), shift(img, 1), img)
        img = np.where(m & (np.abs(d - d0) < 0.6), ci('steel0'), img)
    img = np.where(m & (np.abs(d - 13.5) < 0.55), shift(img, -1), img)
    img = np.where(m & (d > D_FRONT - 1.1), shift(img, -1), img)
    return img


def _marking(ph):
    angf, rhof, markm = ANG_F[SL], RHO_F[SL], MARK_M[SL]
    a = angf + ph * 0.16
    nt = 48
    f = (a * nt / (2 * math.pi)) % 1.0
    dist = np.minimum(f, 1 - f) * (2 * math.pi / nt) * rhof
    k = np.floor(a * nt / (2 * math.pi) + 0.5).astype(int)
    longt = (k % 6 == 0)
    tick = markm & (dist < 0.62) & ((rhof < R_MARK_IN + 3.0) | (longt & (rhof < R_BEZ_IN - 1.0)))
    lit = np.cos(angf + 2.4) > -0.3
    return tick, lit


def _iris(rap, rot, power):
    nb = 7
    u, v = IA[SL], IB[SL]
    r = RHO_I[SL]
    th2 = TH2[SL]
    a = np.arctan2(v, u)
    apo = rap * math.cos(math.pi / nb)
    inside = np.ones(u.shape, bool)
    for k in range(nb):
        ang = rot + 2 * math.pi * k / nb + math.pi / nb
        inside &= (u * math.cos(ang) + v * math.sin(ang)) <= apo
    R = R_RET_IN
    tt = np.clip((r - rap) / (R - rap), 0, 1)
    seam_ang = rot + 0.78 * tt
    fb = ((a - seam_ang) * nb / (2 * math.pi)) % 1.0
    bid = np.floor((a - seam_ang) * nb / (2 * math.pi)).astype(int) % nb
    lit = 0.5 + 0.5 * np.cos(ANG_I[SL] + 2.3)
    bv = 0.12 + 0.50 * fb * (0.6 + 0.4 * lit) + 0.10 * (bid % 2) - 0.12 * tt
    img = quantize(bv, ['night1', 'night2', 'night3', 'steel1', 'steel2'], cuts=[0.16, 0.3, 0.44, 0.56])
    seam = (fb > 0.965) | (fb < 0.03)
    img = np.where(seam & (tt > 0.03), ci('ink0'), img)
    lead = (fb > 0.90) & (fb <= 0.965) & (tt > 0.08) & (tt < 0.8) & (lit > 0.45)
    img = np.where(lead, ci('steel3'), img)
    # inside the aperture: a dark cavity with internal lens-element arcs lit
    # amber, and the glowing sensor core at the centre ('power' = brightness)
    rr = np.sqrt(u ** 2 + v ** 2)
    ang = np.arctan2(v, u)
    cav = quantize(-rr, ['ink0', 'night1', 'rust0'], cuts=[-12.5, -9.8], dith=1.0)
    arc1 = (np.abs(rr - 12.0) < 0.6) & (np.cos(ang + 2.3) > 0.25)
    arc2 = (np.abs(rr - 14.5) < 0.6) & (np.cos(ang - 0.8) > 0.45)
    cav = np.where(arc1, ci('core1'), cav)
    cav = np.where(arc2, ci('rust1'), cav)
    glow = quantize(-rr + 1.4 * power, ['night1', 'core0', 'core1', 'core2', 'core3', 'core4', 'core5', 'core6', 'core7'],
                    cuts=[-10.2, -9.0, -7.9, -6.9, -5.6, -4.3, -2.8, -1.3], dith=0.9)
    pupil = np.where(rr < 10.5 + 1.4 * max(power, 0), glow, cav)
    img = np.where(inside, pupil, img)
    rim = dilate(inside, 1) & ~inside
    rim2 = dilate(inside, 2) & ~inside & ~rim
    litrim = np.cos(ang_rim := np.arctan2(v, u)) * 0.6 + np.sin(ang_rim) * 0.8 > -0.2
    img = np.where(rim & litrim, ci('core1'), img)
    img = np.where(rim & ~litrim, ci('rust0'), img)
    img = np.where(rim2 & litrim & (power > -0.5) & (th2 < 0.5), ci('core0'), img)
    return img, inside


def _green(focus, t):
    """the sprout's reflection: a soft dim blur that pulls into a crisp,
    bright green point with a twinkle."""
    img = np.zeros((HH, WW), np.uint8)
    m = np.zeros((HH, WW), bool)
    gx, gy = GP
    g = GLASS_M
    if focus < 0.97:
        rb = 3.8 * (1 - focus) + 1.0
        outer = disc(HH, WW, gx + 0.5, gy - 0.5, rb, 0.2) & g
        mo = outer & (TH4 < 0.28 + 0.2 * focus)
        img[mo] = ci('leaf2')
        m |= mo
        inner = disc(HH, WW, gx + 0.5, gy - 0.5, max(rb - 1.8, 0.6), 0.2) & g & (TH2 < 0.5 + 0.5 * focus)
        img[inner] = ci('leaf3' if focus < 0.5 else 'leaf4')
        m |= inner
    if focus > 0.35:
        # crisp 2x2 core
        cols = ('leaf5', 'leaf5', 'leaf4', 'leaf4') if focus < 0.7 else ('leaf8', 'leaf7', 'leaf6', 'leaf6')
        for (x, y), c in zip(((gx, gy - 1), (gx + 1, gy - 1), (gx, gy), (gx + 1, gy)), cols):
            img[y, x] = ci(c)
            m[y, x] = True
    if focus > 0.8:
        # soft glow ring + twinkle arms
        ring = disc(HH, WW, gx + 0.5, gy - 0.5, 3.2, 0.2) & ~disc(HH, WW, gx + 0.5, gy - 0.5, 1.2, 0.2) & g & ~m
        mr = ring & (TH2 < 0.5)
        img[mr] = ci('leaf1')
        m |= mr
        for (x, y) in ((gx - 1, gy - 1), (gx + 2, gy - 1), (gx, gy - 2), (gx + 1, gy + 1)):
            img[y, x] = ci('leaf4')
            m[y, x] = True
        tw = NZ.vnoise1(np.array([t * 7.0]), 9)[0]
        if tw > 0.45:
            for k, c in ((2, 'leaf6'), (3, 'leaf4')):
                for (x, y) in ((gx - k, gy - 1), (gx + 1 + k, gy - 1)):
                    img[y, x] = ci(c)
                    m[y, x] = True
            for k, c in ((2, 'leaf5'),):
                for (x, y) in ((gx, gy - 1 - k), (gx + 1, gy + k)):
                    img[y, x] = ci(c)
                    m[y, x] = True
    return img, m


GLINT = [  # hand-drawn glint (top-left of the glass): W=white1, w=white0, a=ash7, b=ash6
    '..WWWw.',
    '.WWWWWa',
    'WWWWWwa',
    'WWWWwab',
    '.wwaab.',
]
GLINT2 = ['.w.',
          'wWa',
          '.a.']


def _glint(img):
    gx, gy = screen_pt(D_GLASS, 28.5, math.radians(216))
    gx, gy = int(round(gx)) - 3, int(round(gy)) - 2
    cmap = {'W': 'white1', 'w': 'white0', 'a': 'ash7', 'b': 'ash6'}
    for j, row in enumerate(GLINT):
        for i, ch in enumerate(row):
            if ch != '.':
                img[gy + j, gx + i] = ci(cmap[ch])
    hx, hy = screen_pt(D_GLASS, 30.0, math.radians(38))
    hx, hy = int(round(hx)), int(round(hy))
    for j, row in enumerate(GLINT2):
        for i, ch in enumerate(row):
            if ch != '.':
                img[hy + j, hx + i] = ci(cmap[ch])
    # specular streak hugging the rim, upper-left (the dome's edge)
    pts = [screen_pt(D_GLASS, R_RET_IN - 2.4, math.radians(a_)) for a_ in np.linspace(186, 232, 16)]
    for (x, y) in curve_px(pts):
        img[y, x] = ci('white0')
    # lower-right rim catch-light on the glass edge
    for ang in np.linspace(math.radians(-10), math.radians(55), 22):
        x, y = screen_pt(D_GLASS, R_RET_IN - 1.6, ang)
        img[int(round(y)), int(round(x))] = ci('ash4')
    return img


# ------------------------------------------------------------------ dust
def _dust(img, t):
    x, y, r = field(t, 24, 101, (-10, -10, 340, 200), vel=(-8.0, 2.5), jitter=(4, 3), sway=(2.0, 0.4))
    for xi, yi, ri in zip(x, y, r):
        xi, yi = int(xi), int(yi)
        if 0 <= xi < WW and 0 <= yi < HH:
            img[yi, xi] = ci('dust9' if ri > 0.55 else 'dust8')
            if ri > 0.85 and xi + 1 < WW:
                img[yi, xi + 1] = ci('dust7')
    x, y, r = field(t, 6, 202, (-24, -24, 368, 228), vel=(-15.0, 3.0), jitter=(5, 3), sway=(3.0, 0.3))
    for xi, yi, ri in zip(x, y, r):
        rad = 2.2 + 2.6 * ri
        x0, x1 = int(max(xi - rad - 1, 0)), int(min(xi + rad + 2, WW))
        y0, y1 = int(max(yi - rad - 1, 0)), int(min(yi + rad + 2, HH))
        if x0 >= x1 or y0 >= y1:
            continue
        sub = img[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dd = (xx - xi) ** 2 + (yy - yi) ** 2 <= (rad + 0.3) ** 2
        chk = TH2[y0:y1, x0:x1] < 0.5
        sub[dd & chk] = shift(sub, 1)[dd & chk]
    return img


# ------------------------------------------------------------------ render
def render(t):
    t = min(max(t, 0.0), DURATION + 1.0)
    st = focus_state(t)
    img = BASE.copy()

    # ---- the lens assembly, built inside its bounding box only
    th2 = TH2[SL]
    lay = np.zeros(TH2[SL].shape, np.uint8)
    lay = np.where(COL_M[SL] | COLF_M[SL], COL_IDX[SL], lay)
    lay = np.where(BAR_M[SL], _barrel(st['phA'], st['phB']), lay)
    lay = np.where(FRONT_M[SL], FRONT_IDX[SL], lay)
    tick, lit = _marking(st['phA'])
    lay = np.where(tick & lit, ci('steel4'), lay)
    lay = np.where(tick & ~lit, ci('steel2'), lay)
    iris, inside = _iris(st['rap'], st['rot'], st['power'])
    lay = np.where(IRIS_M[SL], iris, lay)
    wall = WALL_M[SL]
    lay = np.where(wall, WALL_IDX[SL], lay)
    near = wall & (dist_to(inside, 5) < 4)
    lay = np.where(near & (th2 < 0.5), ci('rust0'), lay)
    lay = np.where(REFL_M[SL], REFL_IDX[SL], lay)
    lay = np.where(REFL_SOFT[SL] & (th2 < 0.5), iris, lay)
    g, gm = _green(st['focus'], t)
    lay = np.where(gm[SL], g[SL], lay)
    full = np.zeros((HH, WW), np.uint8)
    full[SL] = lay
    # fixed amber index mark (top of bezel)
    mx, my = screen_pt(D_FRONT, (R_BEZ_IN + R_MARK_IN) / 2, math.radians(-90))
    mx, my = int(round(mx)), int(round(my))
    full[my - 1:my + 2, mx] = ci('core4')
    full[my - 1, mx] = ci('core5')
    full = _glint(full)

    m = LENS_M
    if st['sx'] or st['sy']:
        full = np.roll(np.roll(full, st['sy'], 0), st['sx'], 1)
        m = np.roll(np.roll(m, st['sy'], 0), st['sx'], 1)
    img = np.where(m, full, img)
    img = _dust(img, t)
    img = np.where(VIG, shift(img, -1), img)
    return PAL[img]
