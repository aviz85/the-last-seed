"""
c04 · REST  (global 55.0–60.0, local 0–5.0)

Time has passed. The Gardener sits where it gave its heart, now part of the
garden: the canonical close-up head (c01's box head, knurled lens barrel,
HEX ear-bolt; damaged: antenna stub on the crown, chipped paint top-left),
mint paint in warm morning sun — moss (darker, saturated) on its head and
shoulders, tiny flowers, a vine around its neck, a flower growing from its
open chest porthole. The air moves: leaf shadows drift over it in the breeze,
light shafts breathe, motes glint in them, petals fall. Its lens is dark.

0.05–0.50  a songbird (~20 px) flies in from the upper right over the sunlit
           canopy, flares its wings and lands on the robot's mossy head
0.50       (bird_land 55.5) touch-down: it crouches, settles, a blink
1.20       (eye_glow 56.2) the lens flickers ... and lights up soft GREEN;
           steady from 1.40, slowly brightening, breathing
1.40–3.20  THE HOLD (56.4–58.2): everything fully visible
  1.58       the bird hops once toward the glowing lens
  1.82       it tilts its head down at the lens — a green catch-light in its eye
  2.05       the robot's head lifts 1 px toward the bird (tiny, earned: awake)
  2.30       (57.3) a chirp: the beak opens twice
  2.58       it tilts its head the other way; 3.12 a blink
3.00       (title 58.0) the title overlay begins; from ~3.2 the compositor darkens
           the picture under it while the title keeps the green glow at EYE_POS

EYE_POS = pupil centre (screen px) for the title's green glow — constant from
T_LIFT (2.05 s) on, so it is exact for the whole title (local >= 3.0).
EYE_R = radius of the glowing pupil core (px). eye_pos(t) gives the pupil
centre at any local time (1 px lower before the lift).
"""
import math
import numpy as np
from engine import *
from engine import noise as NZ
from shots.lib_closeup import (ci, cis, shift, ilight, quantize, clean, ell, disc, edge_in, edge_out, edge_side,
                               shifted, dilate, erode, frame_yaw, raycast_superellipsoid, lambert, normalize, put,
                               curve_px, bezier, field, smooth, blit_idx, thick_curve, head_pose_canon, PAL)
from shots.lib_closeup2 import RampSnap, LUM, tone1

DURATION = 5.0
HH, WW = 180, 320
ys, xs = np.mgrid[0:HH, 0:WW].astype(np.float64)
TH4 = bayer(HH, WW, 4)
TH2 = bayer(HH, WW, 2)
TH8 = bayer(HH, WW, 8)


def _n3(x, y, z):
    v = np.stack(np.broadcast_arrays(x, y, z), -1).astype(np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


# ------------------------------------------------------------------ pose (slumped, nodded, a little rolled)
BETA = math.radians(32.0)
NOD = math.radians(11.0)
ROLL = math.radians(-2.5)
F, U, S = frame_yaw(BETA, pitch=NOD, roll=ROLL)
HEAD_A = (40.0, 34.0, 45.0)
O = np.array([204.0, 112.0, 0.0])              # lens base centre -> pupil (206, 113)
PIVOT = O - F * HEAD_A[0] * 0.985 - U * 2.0 - F * 6.0 - U * HEAD_A[1] * 1.08
SUN = normalize([-0.30, -0.70, 0.65])          # warm morning sun: upper left, a little frontal
KSUN, KSKY = 0.92, 0.40                        # sun key, sky + foliage bounce fill


# ================================================================== BACKGROUND
def _clump(img, cx, cy, rad, seed, cols, leaf_r=(2.2, 4.2), n=None, lit=(-0.6, -0.8), sq=0.85):
    """a foliage clump: small leaf ellipses packed in a rough disc, each shaded
    by the clump's form (lit upper-left) with its own lit edge. cols dark->light."""
    n = n or int(rad * rad * 0.22)
    for k in range(n):
        a = NZ.rand(k, 1, seed) * 2 * math.pi
        rr = rad * math.sqrt(NZ.rand(k, 2, seed))
        lx, ly = cx + rr * math.cos(a), cy + rr * math.sin(a) * sq
        lr = leaf_r[0] + NZ.rand(k, 3, seed) * (leaf_r[1] - leaf_r[0])
        ang = NZ.rand(k, 4, seed) * math.pi
        x0, x1 = int(max(lx - lr - 1, 0)), int(min(lx + lr + 2, WW))
        y0, y1 = int(max(ly - lr - 1, 0)), int(min(ly + lr + 2, HH))
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        ca, sa = math.cos(ang), math.sin(ang)
        u = ((xx - lx) * ca + (yy - ly) * sa) / lr
        v = (-(xx - lx) * sa + (yy - ly) * ca) / (lr * 0.55)
        m = u * u + v * v <= 1.0
        fl = 0.5 + 0.5 * ((lx - cx) * lit[0] + (ly - cy) * lit[1]) / max(rad, 1)
        lvl = int(np.clip(fl * len(cols) + (NZ.rand(k, 5, seed) - 0.5) * 0.9, 0, len(cols) - 1))
        sub = img[y0:y1, x0:x1]
        sub[m] = ci(cols[lvl])
        edge = m & ~np.roll(m, 1, 0) & (yy < ly)
        if lvl < len(cols) - 1:
            sub[edge] = ci(cols[lvl + 1])


TRUNK_X = lambda y: np.where(y < 128, 270 + 0.0085 * (128 - y) ** 2, 270 - 0.22 * (y - 128))
CANOPY_Y = lambda x: (66 + 7 * np.sin(x / 23.0 + 0.6) + 4 * np.sin(x / 9.0 + 2.0) - 12 * np.exp(-((x - 60) / 45.0) ** 2)
                      + 13 * np.exp(-((x - 228) / 20.0) ** 2))


EDGE = np.zeros((HH, WW), bool)          # canopy leaves hanging in front of the trunk's top


def _background():
    # 1. the sunlit canopy high up (bright, soft, low contrast; the title and the bird's sky)
    v = 1.0 - ys / 120.0 + 0.10 * (1 - xs / 320) + 0.25 * (NZ.fbm2(xs / 40.0, ys / 26.0, 71, 3) - 0.5)
    img = quantize(v, ['meadow3', 'meadow4', 'meadow5'], cuts=[0.18, 0.55], dith=0.05)
    img = clean(img, None, 2)
    for k in range(13):
        cx = -20 + NZ.rand(k, 1, 5) * 360
        cy = -12 + NZ.rand(k, 2, 5) * 70
        _clump(img, cx, cy, 16 + 13 * NZ.rand(k, 3, 5), 100 + k, ('meadow3', 'meadow4', 'meadow4', 'meadow5'),
               leaf_r=(3.5, 6.0))
    # sun through the leaves: irregular bright gaps (bokeh of the sky)
    for k in range(12):
        cx = NZ.rand(k, 5, 77) * 320
        cy = 3 + NZ.rand(k, 6, 77) * 46
        rr = 1.5 + NZ.rand(k, 7, 77) * 2.6
        m = ell(HH, WW, cx, cy, rr * 1.4, rr, 0.2) & ~ell(HH, WW, cx + rr, cy + rr * 0.6, rr, rr * 0.8, 0.2)
        img = np.where(m, ci('gold4'), img)
        img = np.where(m & ell(HH, WW, cx - 0.4, cy - 0.3, rr * 0.6, rr * 0.45, 0.2), ci('white0'), img)
    # 2. the shade under the canopy, behind the robot: deep greens (the pale mint pops against them)
    shade = ys > CANOPY_Y(xs) + 2.0 * NZ.fbm1(xs / 3.0, 89, 2)
    dv = 0.55 - (ys - 70) / 260.0 + 0.30 * (NZ.fbm2(xs / 30.0, ys / 22.0, 81, 3) - 0.5)
    deep = quantize(dv, ['leaf0', 'leaf1', 'leaf2'], cuts=[0.30, 0.52], dith=0.04)
    img = np.where(shade, deep, img)
    for k in range(16):
        cx = -30 + NZ.rand(k, 1, 9) * 360
        cy = 70 + NZ.rand(k, 2, 9) * 120
        _clump(img, cx, cy, 16 + 14 * NZ.rand(k, 3, 9), 300 + k, ('leaf0', 'leaf1', 'leaf2', 'leaf3'),
               leaf_r=(3.0, 5.5))
    # the canopy's lower edge: a row of sunlit leaf clusters hanging over the shade
    # (and in front of the trunk's top); a big sunlit cluster behind the bird's perch
    edge = np.zeros((HH, WW), bool)
    for k in range(20):
        cx = -10 + k * 18 + 8 * NZ.rand(k, 1, 13)
        cy = float(CANOPY_Y(np.array(cx))) + 2 + 5 * NZ.rand(k, 2, 13)
        before = img.copy()
        _clump(img, cx, cy, 8 + 5 * NZ.rand(k, 3, 13), 500 + k, ('meadow2', 'meadow3', 'meadow4', 'meadow5'),
               leaf_r=(2.5, 4.5), sq=0.6)
        edge |= img != before
    for k, (cx, cy, rad) in enumerate(((222.0, 76.0, 13.0), (240.0, 70.0, 10.0), (206.0, 70.0, 9.0))):
        before = img.copy()
        _clump(img, cx, cy, rad, 560 + k, ('meadow3', 'meadow4', 'meadow5', 'meadow5'), leaf_r=(2.5, 4.5), sq=0.75)
        edge |= img != before
    EDGE[:] = edge
    img = clean(img, None, 1)
    # blossoms in the canopy (the same tree as the valley shot)
    for k in range(38):
        cx = int(NZ.rand(k, 1, 808) * 330) - 5
        cy = int(NZ.rand(k, 2, 808) * 150) - 4
        big = NZ.rand(k, 3, 808) > 0.55
        dark = cy > CANOPY_Y(np.array(float(cx))) + 4
        c4, c5, c3 = ('bloom3', 'bloom4', 'bloom2') if dark else ('bloom4', 'bloom5', 'bloom3')
        pat = ((0, 0, c4), (1, 0, c5), (0, 1, c3), (1, 1, c4), (-1, 1, c3), (2, 1, c3), (1, -1, c4)) if big \
            else ((0, 0, c5), (1, 0, c4), (0, 1, c3))
        for (dx, dy, c) in pat:
            put(img, cx + dx, cy + dy, c)
    return img.astype(np.uint8)


def TRUNK_LEFT(y):
    """left flank of the giant trunk behind the robot (a buttress root flares
    out toward the lower left)."""
    return (112 + 7 * np.sin(y / 37.0 + 1.0) + 4 * np.sin(y / 13.0 + 0.4)
            - 34 * np.clip((y - 138) / 42.0, 0, 1) ** 1.6)


def _trunk_alb():
    """the tree it gave its heart to: its vast trunk rises right behind the
    robot (in the canopy's shade), up into the leaves. Warm bark behind cool
    mint. Albedo + normals: it takes the same sun, flecked through the canopy."""
    xl = TRUNK_LEFT(ys)
    rag = 5.0 * NZ.fbm1(xs / 5.0, 88, 3)                      # the trunk vanishes into the leaf mass
    m = (xs > xl) & (ys > CANOPY_Y(xs) + 5.0 + rag) & ~EDGE
    rel = xs - xl
    wob = 4.0 * np.sin(ys / 23.0 + xs / 31.0) + 1.6 * np.sin(ys / 9.0 + xs / 17.0)
    ridge = np.sin((xs + wob) / 4.6) + 0.25 * np.sin((xs + 0.5 * wob) / 2.3 + 1.0)
    alb = np.where(ridge > 0.55, ci('bark4'), np.where(ridge > -0.45, ci('bark3'), ci('bark2')))
    # deep fissures, a few long ones
    fis = (np.abs(np.sin((xs + 1.7 * wob) / 13.0)) < 0.07) & (NZ.vnoise1(xs / 9.0, 3) > 0.4)
    alb = np.where(fis, ci('bark1'), alb)
    t = np.clip(rel / 34.0, 0, 1)
    nrm = _n3(-(1 - t) ** 1.5 * 0.95, -0.08, 0.30 + 0.70 * t)
    # moss and lichen on the bark: on the ridges, heavier low and on the shaded side
    mn = NZ.fbm2(xs / 7.0, ys / 6.0, 61, 3)
    mo = m & (mn > 0.70 - 0.16 * np.clip((ys - 120) / 60, 0, 1)) & (ridge > -0.2)
    alb = np.where(mo, ci('meadow2'), alb)
    alb = np.where(mo & (mn > 0.72), ci('meadow3'), alb)
    return m, alb.astype(np.uint8), nrm


BG = _background()
TRUNK_M, TRUNK_ALB, TRUNK_N = _trunk_alb()


# ================================================================== THE ROBOT (albedo + normals)
HEAD = head_pose_canon(PIVOT, BETA, NOD, ROLL, A=HEAD_A, n=5.6, pivot=(6.0, 1.08), u_off=2.0, steps=150)
_HEAD_NOSTUB = head_pose_canon(PIVOT, BETA, NOD, ROLL, A=HEAD_A, n=5.6, pivot=(6.0, 1.08), u_off=2.0, steps=150,
                               stub=False)
HX0, HY0, HW, HH_ = HEAD['bbox']


def _full(a, fill=0, dtype=None):
    """head-pose bbox array -> full frame (clipped)."""
    shp = (HH, WW) + a.shape[2:]
    out = np.full(shp, fill, dtype=dtype or a.dtype)
    xa, ya = max(HX0, 0), max(HY0, 0)
    xb, yb = min(HX0 + HW, WW), min(HY0 + HH_, HH)
    out[ya:yb, xa:xb] = a[ya - HY0:yb - HY0, xa - HX0:xb - HX0]
    return out


HEAD_M = _full(HEAD['mask'], False)
HEAD_COL = _full(HEAD['col'], 0)
# the canonical albedo uses paint3 as its mid; this shot's light model wants the
# real paint colour: walk the head's paint up two steps (contrast ratios kept)
_PUP = np.arange(256, dtype=np.uint8)
for _k in range(7):
    _PUP[ci(f'paint{_k}')] = ci(f'paint{min(_k + 2, 6)}')
HEAD_COL = _PUP[HEAD_COL]
HEAD_N = _full(HEAD['nrm'], 0.0)
HEAD_Z = _full(np.nan_to_num(HEAD['z'], nan=0.0), 0.0)
LENS_M = _full(HEAD['mask'] & ~HEAD['head'], False)
GLASS_M = _full(HEAD['glass'], False)
IRIS_M = _full(HEAD['iris'], False)
RHO_I = _full(HEAD['rho_i'], 99.0)
ANG_I = _full(HEAD['ang_i'], 0.0)
RHO_G = _full(HEAD['rho_g'], 99.0)
ANG_G = _full(HEAD['ang_g'], 0.0)
RHO_F = _full(HEAD['rho_f'], 99.0)
ANG_F = _full(HEAD['ang_f'], 0.0)
STUB_M = HEAD_M & ~_full(_HEAD_NOSTUB['mask'], False)
_lc = HEAD['loc']
FACE_T = _full(HEAD['face_t'], False)
FACE_F = _full(HEAD['head'] & ((HEAD['nrm'] * F).sum(-1) > 0.55), False)
QS = _full(_lc[..., 2], 0.0)
QU = _full(_lc[..., 1], 0.0)
LENS = HEAD['L']
PUPIL = HEAD['pupil']

# ---- body (shoulders + chest), neck
FB, UB, SB = frame_yaw(BETA, pitch=math.radians(-4.0))
BODY_A = (44.0, 56.0, 52.0)
BODY_C = np.array([186.0, 212.0, -60.0])
BODY_M, BODY_Z, BODY_N, BODY_L = raycast_superellipsoid(HH, WW, BODY_C, FB, UB, SB, BODY_A, n=5.0,
                                                        zrange=(-140, 140), steps=160)
BODY_Z = np.nan_to_num(BODY_Z, nan=-60.0)
BFACE_T = BODY_M & ((BODY_N * UB).sum(-1) > 0.5)
PORT_C3 = BODY_C + FB * BODY_A[0] * 0.99 + UB * (BODY_A[1] * 0.50) - SB * (BODY_A[2] * 0.05)
PORT_R = 19.0


def _plane_local(P0, ax1, ax2):
    M2 = np.array([[ax1[0], ax2[0]], [ax1[1], ax2[1]]])
    Minv = np.linalg.inv(M2)
    dx, dy = xs - P0[0], ys - P0[1]
    return Minv[0, 0] * dx + Minv[0, 1] * dy, Minv[1, 0] * dx + Minv[1, 1] * dy


PU, PV = _plane_local(PORT_C3, -SB, -UB)
PORT_RR = np.sqrt(PU ** 2 + PV ** 2)
PORT_A = np.arctan2(PV, PU)
PORT_HOLE = BODY_M & (PORT_RR <= PORT_R - 4)
PORT_RING = BODY_M & (PORT_RR <= PORT_R) & ~PORT_HOLE

_NT = HEAD['C'] + HEAD['U'] * (-HEAD_A[1] * 0.8)
_NB = BODY_C + UB * BODY_A[1] * 0.92
NECK_M, NECK_AL, NECK_AC, _ = thick_curve(HH, WW, [(_NT[0], _NT[1]), (_NB[0], _NB[1])], 11.0, 12.5)
NECK_M &= ~HEAD_M


# ================================================================== MOSS (albedo + clump normals: it lights with the sun)
def _moss_region(face_mask, extra=4.0, seed=1, drip=3.0):
    """moss = the top face + a bumpy cushion rising above the silhouette + a
    ragged fringe hanging over the edges."""
    reg = face_mask.copy()
    cols_ = np.nonzero(reg.any(0))[0]
    cushion = np.zeros_like(reg)
    for x in cols_:
        ytop = np.nonzero(reg[:, x])[0].min()
        bump = abs(math.sin(x * 0.55 + seed)) ** 0.6 * 0.5 + NZ.fbm1(np.array([x / 6.0]), seed, 3)[0]
        hgt = int(round(extra * (0.35 + 0.9 * bump)))
        cushion[max(ytop - hgt, 0):ytop, x] = True
    reg |= cushion
    fringe = np.zeros_like(reg)
    for x in cols_:
        ybot = np.nonzero(reg[:, x])[0].max()
        hang = int(round(drip * NZ.fbm1(np.array([x / 3.0]), seed + 5, 3)[0] ** 2 * 2.2))
        fringe[ybot:min(ybot + hang, HH), x] = True
    return reg | fringe


def _moss_paint(reg, seed, base_nrm):
    """cushion moss built from rounded clumps (r 1.6-3.8 px): each clump a small
    dome (its own normal) with a slightly varied albedo; dark gaps between."""
    alb = np.where(reg, ci('leaf1'), 0).astype(np.uint8)
    nrm = np.where(reg[..., None], base_nrm, np.array([0.0, 0.0, 1.0]))
    ys_, xs_ = np.nonzero(reg)
    n = max(len(xs_) // 9, 1)
    cand = []
    for k in range(n):
        j = int(NZ.rand(k, 1, seed) * len(xs_))
        cand.append((ys_[j], xs_[j], k))
    cand.sort()
    for (y, x, k) in reversed(cand):
        r = 1.6 + NZ.rand(k, 2, seed) * 2.2
        tone = NZ.rand(k, 3, seed)
        x0, x1 = int(max(x - r - 1, 0)), int(min(x + r + 2, WW))
        y0, y1 = int(max(y - r - 1, 0)), int(min(y + r + 2, HH))
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        dx, dy = (xx - x) / (r + 0.4), ((yy - y) * 1.15) / (r + 0.4)
        m = (dx * dx + dy * dy <= 1.0) & reg[y0:y1, x0:x1]
        nz = np.sqrt(np.clip(1 - dx * dx - dy * dy, 0, 1)) + 0.3
        dn = _n3(dx * 0.9, dy * 0.9 - 0.35, nz)
        sub_n = nrm[y0:y1, x0:x1]
        sub_n[m] = dn[m]
        sub_a = alb[y0:y1, x0:x1]
        sub_a[m] = ci('meadow1' if tone < 0.28 else ('meadow3' if tone > 0.86 else 'meadow2'))
    # the underside of the cushion: a dark line where it meets the paint
    alb = np.where(reg & edge_side(reg, 0, 1), ci('leaf1'), alb)
    return alb, nrm


MOSS_HEAD = _moss_region(FACE_T & ~STUB_M, 5.0, seed=3, drip=3.0) & ~LENS_M & ~STUB_M
MOSS_BODY = _moss_region(BFACE_T & (ys < 175), 3.0, seed=7, drip=2.0) & ~HEAD_M
_up = _n3(0.0, -1.0, 0.35)
MOSS_H_ALB, MOSS_H_N = _moss_paint(MOSS_HEAD, 3, _up)
MOSS_B_ALB, MOSS_B_N = _moss_paint(MOSS_BODY, 7, _up)


# ================================================================== COMPOSITION (albedo, normals)
def _compose(lift):
    """the robot's albedo / normal maps (full frame) with the head group raised
    `lift` px (0 or 1)."""
    alb = np.zeros((HH, WW), np.uint8)
    nrm = np.zeros((HH, WW, 3))
    nrm[..., 2] = 1.0
    m = np.zeros((HH, WW), bool)

    def put_(mask, a, n):
        nonlocal alb, nrm, m
        alb = np.where(mask, a, alb).astype(np.uint8)
        nrm = np.where(mask[..., None], n, nrm)
        m = m | mask
    # trunk (behind everything)
    put_(TRUNK_M, TRUNK_ALB, TRUNK_N)
    # body: mint, the open porthole (steel ring, dark empty cavity)
    put_(BODY_M, ci('paint5'), BODY_N)
    rn = _n3(np.cos(PORT_A) * 0.5 * (PORT_RR - PORT_R + 2) / 2.0, np.sin(PORT_A) * 0.5, 0.85)
    put_(PORT_RING, ci('steel4'), rn)
    put_(PORT_HOLE, ci('ink1'), _n3(0.0, 0.0, 1.0))
    alb = np.where(PORT_HOLE & (PORT_RR > PORT_R - 6), ci('steel1'), alb)
    alb = np.where(PORT_RING & (PORT_RR > PORT_R - 0.7), ci('steel1'), alb)
    put_(MOSS_BODY, MOSS_B_ALB, MOSS_B_N)
    # neck (corrugated steel)
    rib = (np.floor(NECK_AL * 9) % 2) == 0
    nn = _n3(-NECK_AC * 0.8, 0.0, np.sqrt(np.clip(1 - NECK_AC ** 2, 0.05, 1)))
    put_(NECK_M, np.where(rib, ci('steel3'), ci('steel4')).astype(np.uint8), nn)
    alb = np.where(NECK_M & edge_in(NECK_M), ci('steel1'), alb)
    # the head group
    def up(a):
        if not lift:
            return a
        o = np.zeros_like(a)
        o[:-1] = a[1:]
        return o
    hm = up(HEAD_M)
    put_(hm, up(HEAD_COL), up(HEAD_N))
    mh = up(MOSS_HEAD)
    put_(mh, up(MOSS_H_ALB), up(MOSS_H_N))
    sm = up(STUB_M)
    put_(sm, up(HEAD_COL), up(HEAD_N))
    return alb, nrm, m


COMP = [_compose(0), _compose(1)]

# ================================================================== LIGHT
F_PAINT, F_STEEL, F_RUST, F_MOSS, F_BARK, F_INK = 1, 2, 3, 4, 5, 6
RSNAP = RampSnap({
    F_PAINT: dict(cold='ink1 paint0-4', warm='paint2-6 dust9'),
    F_STEEL: dict(cold='ink0 steel0-4', warm='steel1-6 white0'),
    F_RUST: dict(cold='ink1 rust0-3', warm='rust1-5'),
    F_MOSS: dict(cold='ink0 leaf0-3', warm='leaf1 meadow1-5'),
    F_BARK: dict(cold='ink0 bark0-3', warm='bark1-6'),
    F_INK: dict(cold='ink0-2', warm='ink0-2 steel1'),
}, {'paint': F_PAINT, 'steel': F_STEEL, 'rust': F_RUST, 'leaf': F_MOSS, 'meadow': F_MOSS, 'bark': F_BARK,
    'ink': F_INK, 'night': F_INK, 'ochre': F_RUST})


def light_robot(alb, nrm, mask, dap):
    """sun (warm key) x dapple + sky fill -> lightness + warm share -> ramps."""
    kd = np.clip(nrm[..., 0] * SUN[0] + nrm[..., 1] * SUN[1] + nrm[..., 2] * SUN[2], 0, 1)
    sky = np.clip(0.55 - 0.45 * nrm[..., 1], 0.1, 1.0)
    Ew = KSUN * kd * dap
    Ec = KSKY * sky
    ya = LUM[alb]
    Y = ya * (Ec + Ew)
    wsh = ya * Ew / np.maximum(Y, 1e-6)
    L = np.cbrt(tone1(Y))
    fam = np.where(mask, RSNAP.family(alb), 0)
    return RSNAP.snap(fam, L, wsh, band={0: 0.30, F_PAINT: 0.14, F_STEEL: 0.2, F_BARK: 0.10, F_MOSS: 0.22}, wband=0.10, keep=alb)


# ================================================================== THE LENS GLASS (c01's lens: blades, aperture; dark, dusty)
R_GLASS = LENS['R_GLASS']
_gy, _gx = np.nonzero(dilate(GLASS_M, 2))
GSL = (slice(int(_gy.min()), int(_gy.max()) + 1), slice(int(_gx.min()), int(_gx.max()) + 1))


def _glass_static():
    rr, a = RHO_I, ANG_I
    nb, rap = 7, 5.6
    fb = ((a - 0.3 - 0.8 * np.clip((rr - rap) / (R_GLASS - rap), 0, 1)) * nb / (2 * math.pi)) % 1.0
    img = quantize(0.2 + 0.4 * fb, ['ink0', 'night1', 'night2'], cuts=[0.33, 0.5])
    img = np.where((fb > 0.96) | (fb < 0.03), ci('ink0'), img)
    u, v = rr * np.cos(a), rr * np.sin(a)
    apo = rap * math.cos(math.pi / nb)
    inside = np.ones((HH, WW), bool)
    for k in range(nb):
        ang = 0.3 + 2 * math.pi * k / nb + math.pi / nb
        inside &= (u * math.cos(ang) + v * math.sin(ang)) <= apo
    img = np.where(inside, ci('ink0'), img)
    # glass surface (in front of any glow): a sheen crescent, settled dust, the sun's glint
    top = np.zeros((HH, WW), np.uint8)
    rg, ag = RHO_G, ANG_G
    ug, vg = rg * np.cos(ag), rg * np.sin(ag)
    sheen = GLASS_M & (rg < R_GLASS - 1.2) & (np.sqrt((ug - 4) ** 2 + (vg - 5) ** 2) > R_GLASS - 0.2)
    top = np.where(sheen, ci('night3'), top)
    top = np.where(sheen & (np.sqrt((ug - 4) ** 2 + (vg - 5) ** 2) > R_GLASS + 1.0), ci('ash3'), top)
    settle = np.clip((vg + 2.0) / 12.0, 0, 1)
    dn = NZ.vnoise2(xs / 1.6, ys / 1.2, 12)
    dust = GLASS_M & (dn > 0.93 - 0.20 * settle)
    top = np.where(dust, ci('dust4'), top)
    top = np.where(dust & (NZ.vnoise2(xs / 1.1, ys / 1.1, 19) > 0.7), ci('dust6'), top)
    return img.astype(np.uint8), inside, top


GLASS_BASE, APERTURE, GLASS_TOP = _glass_static()
# the sun's glint on the glass, upper left (c01's glint, smaller: the lens is dusty)
_gpts = np.nonzero(GLASS_M)
_gcx, _gcy = float(_gpts[1].mean()), float(_gpts[0].mean())
GLINT_XY = (int(round(_gcx - 7)), int(round(_gcy - 7)))
GLINT = ['.wW.', 'wWWa', '.aa.']


def glass(glow):
    """dark, dusty glass; iris blades faintly visible. glow 0..1 (green)."""
    img = GLASS_BASE.copy()
    if glow > 0:
        sub = img[GSL]
        rr = RHO_I[GSL]
        inside = APERTURE[GSL]
        th2 = TH2[GSL]
        pr = 1.6 + 2.2 * glow
        core = quantize(-rr + pr, ['leaf2', 'leaf4', 'leaf6', 'leaf7', 'leaf8'], cuts=[-1.2, 0.0, 1.0, 2.2], dith=0.6)
        c_in = inside & (rr < pr + 1.4)
        sub = np.where(c_in, core, sub)
        h1 = (rr < pr + 1.4 + 1.2 + 1.5 * glow) & ~c_in
        h2 = (rr < pr + 1.4 + 2.6 + 3.5 * glow) & ~c_in & ~h1
        h3 = (rr < pr + 1.4 + 3.8 + 5.5 * glow) & ~c_in & ~h1 & ~h2
        sub = np.where(h1, ci('leaf2' if glow > 0.55 else 'leaf1'), sub)
        sub = np.where(h2, ci('leaf1' if glow > 0.55 else 'leaf0'), sub)
        sub = np.where(h3 & (th2 < 0.5), ci('leaf0'), sub)
        rimi = dilate(inside, 1) & ~inside
        sub = np.where(rimi & (rr < pr + 6), ci('leaf1'), sub)
        img[GSL] = sub
    img = np.where(GLASS_TOP > 0, GLASS_TOP, img)
    gx, gy = GLINT_XY
    for j, row in enumerate(GLINT):
        for i, ch in enumerate(row):
            if ch != '.' and GLASS_M[gy + j, gx + i]:
                img[gy + j, gx + i] = ci({'W': 'white0', 'w': 'ash7', 'a': 'ash5'}[ch])
    return img


# ================================================================== LIFE ON IT (painted details: flowers, grass, vine)
FLOWER = {'white': ('white0', 'gold3'), 'violet': ('violet3', 'gold4'), 'pink': ('bloom4', 'gold3'),
          'gold': ('gold3', 'gold1'), 'blue': ('sky5', 'white0')}


def _flower(det, x, y, kind, size=1):
    pc, cc = FLOWER[kind]
    if size == 0:
        put(det, x, y, pc)
        return
    for (dx, dy) in ((0, -1), (-1, 0), (1, 0), (0, 1)):
        put(det, x + dx, y + dy, pc)
    put(det, x, y, cc)
    put(det, x + 1, y + 1, 'leaf1')


def _details(lift):
    """flowers in the moss, grass + flowers in the faceplate seam, the vine
    around the neck. Returns an index layer (0 = empty)."""
    det = np.zeros((HH, WW), np.uint8)
    # vine spiralling around the neck
    top, bot = _NT, _NB
    n = 60
    pts = []
    for i in range(n + 1):
        u = i / n
        cx = top[0] + (bot[0] - top[0]) * u
        cy = top[1] + (bot[1] - top[1]) * u + 2
        ph = u * 2.6 * 2 * math.pi + 0.8
        pts.append((cx + 12.5 * math.cos(ph), cy + 2.2 * math.sin(ph), math.sin(ph) > -0.15, math.cos(ph), i))
    for i in range(n):
        x0, y0, f0, c0, _ = pts[i]
        x1, y1, _, _, _ = pts[i + 1]
        for (x, y) in curve_px([(x0, y0), (x1, y1)]):
            if 0 <= x < WW and 0 <= y < HH and (f0 or not NECK_M[y, x]):
                det[y, x] = ci('leaf4' if f0 else 'leaf1')
                if f0 and y + 1 < HH:
                    det[y + 1, x] = ci('leaf2')
    for (x, y, f, c, i) in pts:
        if i % 7 == 3 and f:
            xi, yi = int(round(x)), int(round(y))
            d = 1 if c > 0 else -1
            for (dx, dy, col) in ((d, -1, 'leaf5'), (2 * d, -1, 'leaf4'), (d, -2, 'leaf6'), (2 * d, -2, 'leaf5'),
                                  (3 * d, -2, 'leaf4')):
                put(det, xi + dx, yi + dy, col)
    # grass + flowers in the faceplate groove, low on the head
    fs_, fu_ = HEAD_A[2] * 0.78, HEAD_A[1] * 0.82
    g = (np.abs(QS / fs_) ** 8 + np.abs(QU / fu_) ** 8)
    inner = FACE_F & (g < 1.0)
    gm = FACE_F & edge_out(inner) & ~inner & (ys > O[1] + 4)
    ys_, xs_ = np.nonzero(gm)
    order = np.argsort(xs_)
    for k, j in enumerate(order[::9]):
        x, y = xs_[j], ys_[j]
        r_ = NZ.rand(k, 1, 44)
        if r_ < 0.45:
            hgt = 2 + int(NZ.rand(k, 2, 44) * 3)
            for i in range(hgt):
                put(det, x + (i // 2) * (1 if r_ < 0.2 else -1), y - i, 'leaf4' if i < hgt - 1 else 'leaf5')
        elif r_ < 0.75:
            _flower(det, x, y - 1, ('white', 'violet', 'gold')[k % 3], 1)
        else:
            put(det, x, y, 'leaf3')
            put(det, x, y - 1, 'leaf5')
    # flowers scattered in the moss
    for reg, seed, nf in ((MOSS_HEAD, 5, 9), (MOSS_BODY, 9, 7)):
        ys_, xs_ = np.nonzero(reg & ~edge_in(reg))
        for k in range(nf):
            j = int(NZ.rand(k, 1, seed) * len(xs_))
            kind = ('white', 'violet', 'pink', 'gold', 'white', 'blue')[k % 6]
            _flower(det, xs_[j], ys_[j], kind, 1 if k % 3 else 0)
    if lift:
        mv = np.zeros_like(det)
        mv[:-1] = det[1:]
        hg = dilate(HEAD_M | MOSS_HEAD, 3)
        det = np.where(hg, mv, det)
    return det


DETAILS = [_details(0), _details(1)]


def _chest_flower(img, t):
    bx, by = PORT_C3[0], PORT_C3[1] - 6
    sway = 1.2 * math.sin(t * 1.3) + 0.6 * math.sin(t * 2.9 + 1)
    pts = []
    for i in range(27):
        u = i / 26
        pts.append((bx + 3.0 * math.sin(u * 2.2) + sway * u * u, by - u * 26))
    for (x, y) in pts:
        xi, yi = int(round(x)), int(round(y))
        put(img, xi, yi, 'leaf4')
        put(img, xi + 1, yi, 'leaf2')
    for (i, d) in ((9, -1), (15, 1)):
        x, y = pts[i]
        xi, yi = int(round(x)), int(round(y))
        for (dx, dy, c) in ((d, 0, 'leaf5'), (2 * d, -1, 'leaf5'), (3 * d, -1, 'leaf6'), (4 * d, -2, 'leaf5'),
                            (2 * d, 0, 'leaf3'), (3 * d, 0, 'leaf4'), (5 * d, -2, 'leaf4')):
            put(img, xi + dx, yi + dy, c)
    x, y = pts[-1]
    xi, yi = int(round(x)), int(round(y))
    BL = ['..aab..', '.abbba.', 'abbcbba', 'abcdcbb', '.bbcbb.', '..bba..']
    cols = {'a': 'bloom5', 'b': 'bloom4', 'c': 'gold4', 'd': 'gold2'}
    for j, row in enumerate(BL):
        for i2, ch in enumerate(row):
            if ch != '.':
                put(img, xi - 3 + i2, yi - 5 + j, cols[ch])
    put(img, xi - 3, yi - 1, 'bloom2')
    put(img, xi + 3, yi, 'bloom2')
    return img


# ================================================================== MOVING LIGHT: leaf shadows, shafts, motes
def breeze(t):
    """the canopy's sway (px): a slow drift + two beats of the breeze."""
    return (3.2 * math.sin(t * 0.85) + 1.3 * math.sin(t * 2.1 + 1.0) + 1.6 * t,
            1.4 * math.sin(t * 0.63 + 2.0) + 0.5 * t)


_DAP_BOX = (slice(0, HH), slice(0, WW))


def dapple(t):
    """sunlight through the canopy, 0.35 (leaf shadow) .. 1 (sun), soft-edged:
    leaf-sized shadows (6-12 px) in two layers drifting at different speeds,
    so the pattern both slides and slowly re-forms, as leaves do in a breeze."""
    bx, by = breeze(t)
    a = NZ.fbm2((xs + bx) / 8.5, (ys + by) / 6.5, 707, 2)
    b = NZ.fbm2((xs + 0.6 * bx - 2.5 * t) / 5.5, (ys + 0.45 * by) / 4.5, 709, 2)
    n = 0.6 * a + 0.4 * b
    return 1.0 - 0.65 * np.clip((n - 0.62) / 0.05 + 0.5, 0.0, 1.0)


def flecks(t):
    """sun flecks on the shaded bark: small bright images of the sun through
    gaps in the leaves, drifting with the canopy's sway."""
    bx, by = breeze(t)
    n = NZ.fbm2((xs + 0.8 * bx) / 7.0, (ys + 0.8 * by) / 5.0, 717, 2)
    f = np.clip((n - 0.64) / 0.025 + 0.5, 0.0, 1.0)
    return 0.34 + 0.66 * f


SHAFTS = ((34.0, 10.0, 0.55, 0.0), (96.0, 7.0, 0.50, 1.7), (152.0, 12.0, 0.60, 3.1), (218.0, 6.0, 0.58, 4.4))


def shafts(t):
    """god rays falling through the canopy (upper left -> lower right); each
    breathes slowly (width + strength) as the leaves above open and close."""
    sh = np.zeros((HH, WW))
    for (x0_, w_, a_, ph) in SHAFTS:
        br = 0.75 + 0.25 * math.sin(t * 1.35 + ph) + 0.10 * math.sin(t * 3.1 + 2 * ph)
        wob = 1.5 * math.sin(t * 0.8 + ph)
        d_ = np.abs((xs - x0_ - wob) - ys * a_)
        wd = w_ * (0.85 + 0.25 * br)
        sh = np.maximum(sh, br * np.clip(1 - d_ / wd, 0, 1) * np.clip(1.15 - ys / 230, 0, 1))
    return sh


# ================================================================== THE SONGBIRD (a robin, ~21 px; faces LEFT)
BIRD_COL = dict(K='bark0', b2='bark2', b3='bark3', b4='bark4', b5='bark5', r1='red1', r2='red2', r3='red3',
                r4='red4', c7='dust7', c8='dust8', c9='dust9', w0='white0', bk='bark1', eye='ink0', hl='white1')
_BIRD_CACHE = {}


def _rot(x, y, a):
    c, s = math.cos(a), math.sin(a)
    return x * c - y * s, x * s + y * c


def bird_sprite(pose):
    """one pose of the bird -> (idx, mask, eye (x, y), feet anchor (x, y)).
    pose: (body_dy, body_ang deg, head_ang deg, beak 0/1/2, blink, wings, legs, tail_ang deg)
    wings: 'fold' | 'up' | 'down' | 'flare'   legs: 'stand' | 'bend' | 'tuck' | 'reach'"""
    if pose in _BIRD_CACHE:
        return _BIRD_CACHE[pose]
    body_dy, body_ang, head_ang, beak, blink, wings, legs, tail_ang = pose
    W_, H_ = 40, 34
    AX, AY = 18, 27                                          # feet anchor inside the box
    yy, xx = np.mgrid[0:H_, 0:W_].astype(np.float64)
    X, Y = xx - AX, yy - AY                                  # local, feet at (0, 0)
    ba = math.radians(body_ang)
    reg = np.zeros((H_, W_), np.int8)                        # region ids
    # --- body (ellipse), centre raised by the legs
    leg_h = {'stand': 3.6, 'bend': 2.2, 'tuck': 1.2, 'reach': 3.0}[legs]
    bcx, bcy = 1.4, -(leg_h + 4.4) + body_dy
    u, v = _rot(X - bcx, Y - bcy, -ba)
    body = (u / 6.3) ** 2 + (v / 4.7) ** 2 <= 1.0
    # --- tail: a tapered wedge from the body's back, angled up
    ta = math.radians(tail_ang) + ba
    tx0, ty0 = bcx + 5.0 * math.cos(ba), bcy + 5.0 * math.sin(ba) - 0.8
    tu = (X - tx0) * math.cos(ta) + (Y - ty0) * math.sin(ta)
    tv = -(X - tx0) * math.sin(ta) + (Y - ty0) * math.cos(ta)
    tail = (tu > -1.0) & (tu < 5.6) & (np.abs(tv + 0.2) <= 1.6 - 0.12 * np.clip(tu, 0, 6)) & ~body
    # --- head, rotating about the neck
    nx_, ny_ = bcx - 4.2 * math.cos(ba) + 0.6, bcy - 4.2 * math.sin(ba) - 1.6
    hoff = _rot(-1.3, -2.6, math.radians(head_ang))
    hcx, hcy = nx_ + hoff[0], ny_ + hoff[1]
    head = (X - hcx) ** 2 + (Y - hcy) ** 2 <= 3.95 ** 2
    ha = math.radians(head_ang)
    # beak: pointing left, rotated with the head. Closed: a short wedge. Open: two
    # mandibles in a clear V (the gap is kept free of outline), a dark-red mouth
    bxa, bya = _rot(-3.5, 0.5, ha)
    bx, by = hcx + bxa, hcy + bya

    def ray(ang, L_, w0, w1):
        du = (X - bx) * math.cos(ang) + (Y - by) * math.sin(ang)
        dv = -(X - bx) * math.sin(ang) + (Y - by) * math.cos(ang)
        return (du > -0.7) & (du < L_) & (np.abs(dv) <= w0 + (w1 - w0) * np.clip(du / L_, 0, 1))
    gap = np.zeros((H_, W_), bool)
    mouth = np.zeros((H_, W_), bool)
    if beak == 0:
        bk_up = ray(math.pi + ha - math.radians(4.0), 3.0, 1.3, 0.3) & ~head
        bk_lo = np.zeros_like(bk_up)
    else:
        op = math.radians(24.0 if beak == 1 else 36.0)
        a_up, a_lo = math.pi + ha - op, math.pi + ha + op * 0.8
        bk_up = ray(a_up, 3.3, 0.9, 0.4) & ~head
        bk_lo = ray(a_lo, 2.8, 0.8, 0.4) & ~head & ~bk_up
        wedge_all = ray(math.pi + ha + op * 0.1, 3.6, 0.6, 3.2) & ~head
        mouth = wedge_all & ~bk_up & ~bk_lo & ray(math.pi + ha + op * 0.1, 1.4, 0.8, 1.2)
        gap = wedge_all & ~bk_up & ~bk_lo & ~mouth
    # --- wing
    if wings == 'fold':
        wu, wv = _rot(X - (bcx + 1.8), Y - (bcy - 1.0), -ba)
        wing = ((wu / 5.4) ** 2 + (wv / 2.9) ** 2 <= 1.0) & body
        wing |= ((wu > 3.0) & (wu < 7.4) & (np.abs(wv + 0.4) < 1.5 - 0.25 * (wu - 3.0))) & ~head
    else:
        # an open wing: a long rounded blade from the shoulder, raised / lowered / flared
        wang = {'up': -62.0, 'down': 38.0, 'flare': -80.0}[wings]
        wa = math.radians(wang) + ba
        sx0, sy0 = bcx + 0.5, bcy - 1.8
        wu = (X - sx0) * math.cos(wa) + (Y - sy0) * math.sin(wa)
        wv = -(X - sx0) * math.sin(wa) + (Y - sy0) * math.cos(wa)
        L_ = 10.5 if wings != 'down' else 8.5
        wing = (wu > -1.0) & (wu < L_) & (np.abs(wv - 0.8) <= 2.8 - 0.14 * np.clip(wu, 0, L_))
        wing |= (wu > -1.0) & (wu < L_ * 0.55) & (np.abs(wv - 0.8) <= 3.4)
    # --- legs: two thin legs (1 px) with a gap, little forward toes
    legm = np.zeros((H_, W_), bool)
    if legs in ('stand', 'bend', 'reach'):
        fx = -1.5 if legs == 'reach' else 0.0
        for lx in (-1.2, 1.4):
            top_y = bcy + 3.2
            for (x, y) in curve_px([(lx + 0.3 + fx * 0.4, top_y), (lx + fx, -1.0)]):
                if 0 <= x + AX < W_ and 0 <= y + AY < H_:
                    legm[y + AY, x + AX] = True
            for (x, y) in ((lx + fx - 1, 0), (lx + fx, 0), (lx + fx + 1, 0)):
                xi, yi = int(round(x)) + AX, int(round(y)) + AY
                if 0 <= xi < W_ and 0 <= yi < H_:
                    legm[yi, xi] = True
    # --- colour regions
    lit = -(X - bcx) * 0.55 - (Y - bcy) * 0.85             # sun from the upper left
    idx = np.zeros((H_, W_), np.uint8)
    C_ = {k: ci(v) for k, v in BIRD_COL.items()}
    # body: back brown (top/right), breast red (front, left), belly cream (low)
    bu, bv = _rot(X - bcx, Y - bcy, -ba)
    breast = body & (bu < 1.2 - 0.6 * bv) & (bv > -2.6)
    belly = body & ~breast & (bv > 1.4)
    back = body & ~breast & ~belly
    idx[back] = np.where(lit[back] > 2.0, C_['b4'], C_['b3'])
    idx[breast] = np.where(lit[breast] > 1.0, C_['r3'], np.where(lit[breast] > -2.0, C_['r2'], C_['r1']))
    idx[belly] = np.where(lit[belly] > -3.0, C_['c9'], C_['c8'])
    idx[tail] = C_['b3']
    idx[tail & (tv > 0.6)] = C_['b2']
    # head: brown crown + nape, the red face / throat
    hu, hv = _rot(X - hcx, Y - hcy, -ha)
    face = head & (hu < 0.8) & (hv > -1.2)
    crown = head & ~face
    idx[crown] = np.where((hu + hv)[crown] < -2.5, C_['b5'], C_['b4'])
    idx[face] = np.where((hu + hv)[face] < -0.5, C_['r3'], C_['r2'])
    idx[head & (hu > 2.4) & (hv > 0.5)] = C_['b3']
    # wing (in front of the body) with feather rows
    wing_only = wing & ~head
    idx[wing_only] = C_['b3']
    if wings == 'fold':
        wu2, wv2 = _rot(X - (bcx + 1.8), Y - (bcy - 1.0), -ba)
        idx[wing_only & (wv2 < -1.4)] = C_['b4']
        rows = wing_only & ((np.floor(wu2 + 0.6 * wv2) % 3) == 0) & (wu2 > 0.5)
        idx[rows] = C_['b2']
        idx[wing_only & (wv2 > 1.6)] = C_['b2']
    else:
        idx[wing_only & (wv < 0.2)] = C_['b4']
        idx[wing_only & (wu > 4.0) & ((np.floor(wu) % 2) == 0) & (wv > 1.2)] = C_['b2']
        idx[wing_only & (wu > 6.0) & (wv > 2.0)] = C_['c8']
    m = body | tail | head | wing | bk_up | bk_lo | mouth
    idx[bk_up] = C_['bk']
    idx[bk_lo] = C_['b2']
    idx[mouth] = C_['r1']
    # eye (2x2 dark, a catch-light upper-left) or a blink (a lid line)
    ex_, ey_ = _rot(-1.3, -0.7, ha)
    ex, ey = int(round(hcx + ex_)) + AX, int(round(hcy + ey_)) + AY
    if blink:
        idx[ey, ex - 1:ex + 1] = C_['b2']
        idx[ey - 1, ex - 1:ex + 1] = C_['r2']
    else:
        idx[ey - 1:ey + 1, ex - 1:ex + 1] = C_['eye']
        idx[ey - 1, ex - 1] = C_['hl']
    # outline (selective: warm brown, not black); the thin legs stay unoutlined
    ol = edge_out(m) & ~m & ~legm & ~gap
    idx[ol] = C_['K']
    legs_only = legm & ~m & ~ol
    idx[legs_only] = C_['bk']
    mm = m | ol | legs_only
    # a lighter rim on the sunlit top edge (the back / crown catch the sun)
    top = m & edge_side(m, 0, -1) & ~legm & ~bk_up & ~bk_lo
    idx[top & (idx == C_['b3'])] = C_['b5']
    idx[top & (idx == C_['b4'])] = C_['b5']
    out = (idx, mm, (ex, ey), (AX, AY))
    _BIRD_CACHE[pose] = out
    return out


# pose table: (body_dy, body_ang, head_ang, beak, blink, wings, legs, tail_ang)
P_PERCH = (0.0, -12.0, 0.0, 0, 0, 'fold', 'stand', -18.0)
P_BLINK = (0.0, -12.0, 0.0, 0, 1, 'fold', 'stand', -18.0)
P_CROUCH = (1.4, -6.0, 6.0, 0, 0, 'fold', 'bend', -10.0)
P_HOPAIR = (-2.5, -18.0, -4.0, 0, 0, 'fold', 'tuck', -26.0)
P_TILT = (0.0, -8.0, -36.0, 0, 0, 'fold', 'stand', -14.0)          # peering down at the lens
P_TILT_B = (0.0, -8.0, -36.0, 0, 1, 'fold', 'stand', -14.0)
P_TILT2 = (0.0, -14.0, 22.0, 0, 0, 'fold', 'stand', -20.0)        # cocked the other way
P_CHIRP1 = (-0.6, -20.0, 20.0, 1, 0, 'fold', 'stand', -34.0)
P_CHIRP2 = (-0.8, -22.0, 28.0, 2, 0, 'fold', 'stand', -38.0)
P_FLY_UP = (-4.0, 4.0, -4.0, 0, 0, 'up', 'tuck', -4.0)
P_FLY_DN = (-4.0, 4.0, -4.0, 0, 0, 'down', 'tuck', -4.0)
P_FLARE = (-3.0, -28.0, -10.0, 0, 0, 'flare', 'reach', 18.0)

P_GLIDE = (-4.0, 6.0, -4.0, 0, 0, 'fold', 'tuck', -2.0)


# ================================================================== TIMING
T_FLY0, T_LAND = 0.05, 0.50                          # bird_land 55.5
T_GLOW = 1.20                                        # eye_glow 56.2: flicker, steady from 1.40
T_STEADY = T_GLOW + 0.20
T_HOP = 1.58                                         # one hop toward the glowing lens
T_TILT = 1.82                                        # head tilted down at the lens
T_LIFT = 2.05                                        # the robot's head lifts 1 px toward the bird
T_CHIRP = 2.30                                       # chirp (57.3): the beak opens twice
T_TILT2 = 2.58                                       # tilts the other way
T_TITLE = 3.00                                       # title 58.0 (picture darkens from ~3.2)


def moss_top(x, lift=0):
    """y of the moss surface under the bird's feet (feet sink 1 px into it)."""
    x = int(round(x))
    return int(MOSS_TOPS[x - 2:x + 3].min()) + 1 - lift


MOSS_TOPS = np.array([np.nonzero(MOSS_HEAD[:, x])[0].min() if MOSS_HEAD[:, x].any() else 999 for x in range(WW)])
LAND_X = 229
HOP_X = 221


def head_lift(t):
    return 1 if t >= T_LIFT else 0


def eye_glow(t):
    """flicker (1.20-1.40) ... then steady soft green, slowly brightening, breathing."""
    if t < T_GLOW - 1e-6:
        return 0.0
    f = int(round((t - T_GLOW) * 30.0))              # frame-exact flicker: on on . on . . | steady from 1.40
    if f < 6:
        return (0.30, 0.30, 0.0, 0.42, 0.0, 0.0)[f]
    u = min(max((t - T_STEADY) / 1.75, 0.0), 1.0)
    g = 0.42 + 0.58 * (1.0 - (1.0 - u) ** 2)
    breath = 0.05 * math.sin((t - T_STEADY) * 2.3) * smooth((t - 2.0) / 0.8)
    return min(g + breath, 1.05)


def _flight(t):
    """position of the feet anchor + pose during the fly-in (bounding flight:
    flap - glide - flap, then a flare with the legs reaching)."""
    u = (t - T_FLY0) / (T_LAND - T_FLY0)
    e = 1 - (1 - u) ** 2.2                            # fast in, braking into the landing
    x0, y0 = 338.0, 6.0
    x1, y1 = float(LAND_X), float(moss_top(LAND_X))
    x = x0 + (x1 - x0) * e
    y = y0 + (y1 - y0) * (e ** 1.25) - 7.0 * math.sin(math.pi * min(e * 1.15, 1.0)) \
        + 2.5 * math.sin(e * math.pi * 3.0) * (1 - e)          # the undulation of bounding flight
    f = int(round(t * 30))
    if u > 0.80:
        return P_FLARE, x, y
    if 7 <= f <= 9:                                   # a short glide between the wing bursts
        return P_GLIDE, x, y
    # 4-frame wingbeat, downstrokes on the sound's flutters (local 0.067, 0.200, 0.333)
    return (P_FLY_DN if f % 4 in (2, 3) else P_FLY_UP), x, y


def _perch_head(t):
    """a perched bird is never still: quick head turns held for a moment."""
    k = int(t * 3.1)
    return (0.0, -7.0, 4.0, 0.0, 8.0, -4.0, 0.0, 5.0)[k % 8]


def bird_state(t):
    """(pose, feet x, feet y in the unlifted head frame) or None before it arrives."""
    if t < T_FLY0:
        return None
    if t < T_LAND:
        return _flight(t)
    x, y = LAND_X, moss_top(LAND_X)
    dt = t - T_LAND
    if dt < 0.07:
        return P_CROUCH, x, y
    if t < T_HOP:
        if 0.86 <= t < 0.91 or 1.27 <= t < 1.32:          # settles; blinks at the eye's flicker
            return P_BLINK, x, y
        if 1.32 <= t < T_HOP:                             # it has seen the light: very still, looking
            return (0.0, -12.0, -16.0, 0, 0, 'fold', 'stand', -18.0), x, y
        h = _perch_head(t)
        return (0.0, -12.0, h, 0, 0, 'fold', 'stand', -18.0), x, y
    # one hop toward the lens (left), airborne 0.12 s, onto the moss surface there
    x2, y2 = HOP_X, moss_top(HOP_X)
    if t < T_HOP + 0.05:
        return P_CROUCH, x, y
    if t < T_HOP + 0.17:
        u = (t - T_HOP - 0.05) / 0.12
        return P_HOPAIR, x + (x2 - x) * u, y + (y2 - y) * u - 4.0 * math.sin(math.pi * u)
    x, y = x2, y2
    if t < T_HOP + 0.22:
        return P_CROUCH, x, y
    if T_TILT <= t < T_CHIRP - 0.04:
        if 2.12 <= t < 2.17:
            return P_TILT_B, x, y
        return P_TILT, x, y                              # peering down at the glowing lens
    if T_CHIRP <= t < T_CHIRP + 0.07:
        return P_CHIRP1, x, y
    if T_CHIRP + 0.13 <= t < T_CHIRP + 0.21:
        return P_CHIRP2, x, y
    if T_CHIRP - 0.04 <= t < T_CHIRP:
        return (-0.4, -18.0, 14.0, 0, 0, 'fold', 'stand', -28.0), x, y      # draws breath, head up
    if T_CHIRP + 0.07 <= t < T_CHIRP + 0.13:
        return (-0.6, -20.0, 18.0, 0, 0, 'fold', 'stand', -32.0), x, y
    if T_CHIRP + 0.21 <= t < T_CHIRP + 0.27:
        return (-0.3, -16.0, 10.0, 0, 0, 'fold', 'stand', -24.0), x, y
    if T_CHIRP - 0.04 <= t < T_TILT2:
        return (0.0, -14.0, 6.0, 0, 0, 'fold', 'stand', -20.0), x, y
    if T_TILT2 <= t < 3.02:
        return P_TILT2, x, y
    if 3.12 <= t < 3.17 or 4.40 <= t < 4.45:
        return P_BLINK, x, y
    if 3.62 <= t < 4.20:
        return P_TILT, x, y
    return P_PERCH, x, y


# ================================================================== PARTICLES
_PET = [['.aab.', 'abbbc', '.bcc.'], ['.ab', 'abb', 'bbc', 'bc.'], ['aab.', '.bbc'], ['.a.', 'abb', 'bbc', '.c.']]
_PET_COL = {'a': 'bloom5', 'b': 'bloom4', 'c': 'bloom3'}
HERO_PETALS = [(0.55, 36.0, -6.0, -3.0, 44.0, 6.0, 2.1, 0.0),
               (1.52, 122.0, -6.0, -6.0, 52.0, 7.0, 2.6, 1.3),
               (2.18, 302.0, -6.0, -20.0, 46.0, 6.0, 2.3, 2.2),
               (2.95, 176.0, -6.0, -9.0, 40.0, 5.0, 1.9, 0.7)]


def _hero_petals(img, t):
    for (t0, x0, y0, vx, vy, amp, rate, ph) in HERO_PETALS:
        dt = t - t0
        if dt < 0:
            continue
        x = x0 + vx * dt + amp * math.sin(dt * rate + ph)
        y = y0 + vy * dt + 1.5 * math.sin(dt * rate * 2 + ph)
        if y > HH + 4:
            continue
        fr = _PET[int(dt * 7.5 + ph * 3) % len(_PET)]
        xi, yi = int(round(x)), int(round(y))
        for j, row in enumerate(fr):
            for i, ch in enumerate(row):
                if ch != '.':
                    put(img, xi + i, yi + j, _PET_COL[ch])
    return img


def _puff(img, t, t0, x, y, seed):
    """a touch-down: a few pollen grains and moss bits jump up and settle."""
    dt = t - t0
    if dt < 0 or dt > 0.30:
        return img
    for i in range(6):
        ang = math.radians(-160 + 140 * NZ.rand(i, 1, seed))
        sp = 14 + 16 * NZ.rand(i, 2, seed)
        px_ = x + math.cos(ang) * sp * dt
        py_ = y + math.sin(ang) * sp * dt + 60 * dt * dt
        col = 'gold4' if i % 3 == 0 else ('meadow5' if i % 3 == 1 else 'dust9')
        if dt > 0.2 and i % 2:
            continue
        put(img, int(round(px_)), int(round(py_)), col)
    return img


def _motes(img, t, sh):
    """dust and pollen glinting only where a light shaft catches them."""
    x, y, r = field(t, 70, 919, (-10, -10, 340, 200), vel=(3.5, 4.5), jitter=(2.5, 2.5), sway=(2.5, 0.35))
    for xi, yi, ri in zip(x, y, r):
        xi, yi = int(xi), int(yi)
        if 0 <= xi < WW and 0 <= yi < HH and sh[yi, xi] > 0.45:
            tw = (int(t * 9 + ri * 17) % 4) != 0
            if tw:
                img[yi, xi] = ci('white0' if (ri > 0.7 and sh[yi, xi] > 0.7) else ('gold4' if ri > 0.35 else 'dust9'))
    return img


# ================================================================== RENDER
ROBOT_M = [COMP[0][2] & ~TRUNK_M, COMP[1][2] & ~TRUNK_M]
_GREEN_OK = [None, None]


def _up(a, k):
    if not k:
        return a
    o = np.zeros_like(a)
    o[:-1] = a[1:]
    return o


def render(t):
    t = min(max(t, 0.0), DURATION + 1.0)
    lift = head_lift(t)
    alb, nrm, rm = COMP[lift]
    dap = dapple(t)
    sh = shafts(t)
    # ---- background: light shafts breathe through the canopy air; sun flecks on the low leaves
    img = BG.copy()
    shaft_on = (sh > 0.42 + 0.30 * (TH4 - 0.5)) & ~rm & (ys < 165)
    img = np.where(shaft_on, shift(img, 1), img)
    shaft_tr = (sh > 0.42 + 0.30 * (TH4 - 0.5)) & TRUNK_M & ~ROBOT_M[lift]
    # sun flecks drifting over the low leaves (the 'ground' of this frame)
    fl = flecks(t)
    low = (ys > CANOPY_Y(xs) + 8) & ~rm
    img = np.where(low & (fl > 0.9), shift(img, 1), img)
    img = np.where(low & (fl > 0.62) & (fl <= 0.9) & (TH2 < 0.5), shift(img, 1), img)
    # ---- the robot (+ trunk): sun x leaf shadows, sky fill
    dap_r = np.where(TRUNK_M & ~ROBOT_M[lift], flecks(t), dap)
    lit = light_robot(alb, nrm, rm, dap_r)
    lit = np.where(shaft_tr, shift(lit, 1), lit)
    img = np.where(rm, lit, img)
    # ---- the lens
    glow = eye_glow(t)
    gm = _up(GLASS_M, lift)
    img = np.where(gm, _up(glass(glow), lift), img)
    # ---- life on it: flowers, seam grass, the vine (dappled)
    det = DETAILS[lift]
    dm = det > 0
    img = np.where(dm, np.where(dap < 0.5, shift(det, -1), det), img)
    img = _chest_flower(img, t)
    # ---- the bird
    bs = bird_state(t)
    bird_m = np.zeros((HH, WW), bool)
    if bs is not None:
        pose, bx, by = bs
        bidx, bm, (ex, ey), (ax, ay) = bird_sprite(pose)
        bidx = bidx.copy()
        if t >= T_LAND:
            by = by - lift
        x0 = int(round(bx)) - ax
        y0 = int(round(by)) - ay
        # dapple on the bird: a leaf shadow dims it one step
        hh_, ww_ = bidx.shape
        ya_, yb_ = max(y0, 0), min(y0 + hh_, HH)
        xa_, xb_ = max(x0, 0), min(x0 + ww_, WW)
        if ya_ < yb_ and xa_ < xb_:
            dsub = np.ones((hh_, ww_))
            dsub[ya_ - y0:yb_ - y0, xa_ - x0:xb_ - x0] = dap[ya_:yb_, xa_:xb_]
            bidx = np.where(bm & (dsub < 0.5) & (bidx != ci('bark0')), shift(bidx, -1), bidx)
        if glow > 0.25 and t >= T_LAND:
            # the green light from below catches the bird's breast and belly edge ...
            body = bm & np.isin(bidx, cis(['dust9', 'dust8', 'red2', 'red3', 'red1']))
            low_ = body & ~np.roll(body, -1, 0)
            bidx = np.where(low_, ci('leaf5' if glow > 0.7 else 'leaf4'), bidx)
            # ... and a green catch-light in its eye (lower edge, toward the lens)
            if glow > 0.45 and pose[4] == 0:
                bidx[ey, ex - 1] = ci('leaf7' if pose[2] < -20 else 'leaf6')
                if pose[2] < -20 and glow > 0.6:
                    bidx[ey, ex] = ci('leaf5')
        if t >= T_LAND and pose not in (P_HOPAIR,):
            for dxs in range(-2, 4):                       # a soft contact shadow in the moss
                X_, Y_ = int(round(bx)) + dxs, int(round(by)) + 1
                if 0 <= X_ < WW and 0 <= Y_ < HH and rm[Y_, X_]:
                    img[Y_, X_] = shift(img[Y_:Y_ + 1, X_:X_ + 1], -1)[0, 0]
        blit_idx(img, bidx, bm, x0, y0)
        tmp = np.zeros((HH, WW), np.uint8)
        blit_idx(tmp, bm.astype(np.uint8), bm, x0, y0)
        bird_m = tmp > 0
    # ---- the eye's green light on the faceplate, the moss rim, the bird
    if glow > 0:
        ex, ey = eye_pos(t)
        d = np.sqrt((xs - ex) ** 2 + (ys - ey) ** 2)
        rad = 14 + 22 * glow
        g_lv = np.clip(1 - d / rad, 0, 1) ** 1.2 * (0.6 + 0.9 * glow)
        hm_ = _up(HEAD_M & ~LENS_M, lift) | _up(MOSS_HEAD, lift)
        g_lv = np.where(hm_ & ~gm, g_lv, 0.0)
        img = ilight(img, g_lv, mode='green', band=0.0)
        rb = _up((np.abs(RHO_F - (LENS['R_BEZ_IN'] + 0.8)) < 0.6) & (np.cos(ANG_F - 0.9) > 0.35), lift)
        img = np.where(rb & (glow > 0.45), ci('leaf4' if glow < 0.8 else 'leaf5'), img)
    # ---- petals + pollen + motes in the shafts
    img = _hero_petals(img, t)
    x, y, r = field(t, 9, 515, (-20, -20, 360, 220), vel=(9.0, 13.0), jitter=(3, 3), sway=(5.0, 0.35))
    for k, (xi, yi, ri) in enumerate(zip(x, y, r)):
        xi, yi = int(xi), int(yi)
        flip = int(t * 6 + ri * 7) % 3
        col = 'bloom4' if ri > 0.5 else 'bloom3'
        pts = ((0, 0), (1, 0)) if flip == 0 else (((0, 0), (0, 1)) if flip == 1 else ((0, 0), (1, 1)))
        for (dx, dy) in pts:
            put(img, xi + dx, yi + dy, col)
        put(img, xi, yi, 'bloom5' if flip != 1 else col)
    img = _motes(img, t, sh)
    img = _puff(img, t, T_LAND, LAND_X, moss_top(LAND_X) - 1, 31)
    img = _puff(img, t, T_HOP + 0.17, HOP_X, moss_top(HOP_X) - 1, 32)
    return PAL[img]


EYE_POS_0 = (int(round(PUPIL[0])), int(round(PUPIL[1])))       # before the 1-px lift: (206, 113)
EYE_POS = (EYE_POS_0[0], EYE_POS_0[1] - 1)                      # from T_LIFT (2.05 s) on: (206, 112)
EYE_R = 4.0


def eye_pos(t):
    return EYE_POS if t >= T_LIFT else EYE_POS_0
