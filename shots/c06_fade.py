"""
c06 · THE LAST LIGHT  (global 43.4–45.0, local 0–1.6; the film plays it from 43.6
with offset 0.2, so the screen shows local 0.2–1.6)

Grey dawn. The Gardener has just pressed its heart into the planter soil (c08).
It kneels facing right, head bowed toward the soil, one hand (c02's hand) on the
planter rim. Close-up of its face: the canonical head (c01 reference: mint box
head, hex ear-bolt, knurled lens, amber pupil, white glint), 3/4 facing right;
the dying eye sits where c04's reborn eye will open. Lower right: the stone
coping, dark soil, the tiny planted sprout and the warm ember of the buried
heart, whose glow lights the face from below-right and sinks.

It OPENS at the dawn exposure of the S7b wide and c08 (mean ~90): cold grey-blue
dawn falls through the broken glass from above RIGHT (ash / glass ramps), plus the
warm under-light from the soil.

0.20      (43.60, first frame on screen) eye lit amber; the under-light sinking.
0.25/0.5  (43.65 / 43.9) the pupil flickers out and back, weaker each time.
0.9       (44.3 eye_off) the eye goes dark for good — final click; the head sags
          2 px as the servos let go, and the world steps down with it.
0.9–1.6   palette-stepped darkening (every colour walks down its ramp, like the
          compositor's fades): near-black by 1.6 (mean ~15) to meet S8a's dark
          opening; only the ember remains in the soil, steady and faint.
"""
import math
import numpy as np
from engine import *
from engine import ease
from engine import noise as NZ
from engine import light as ELIGHT
from shots.lib_closeup import (ci, quantize, clean, ell, disc, edge_in, edge_out, edge_side, shifted, dilate,
                               erode, frame_yaw, raycast_se_fast, normalize, bezier, thick_curve, head_pose_canon,
                               mech_hand, raster_leaf, RampLight, level_of, lens_glass, PAL)

DURATION = 1.6
HH, WW = 180, 320
ys, xs = np.mgrid[0:HH, 0:WW].astype(np.float64)
TH2 = bayer(HH, WW, 2)
TH4 = bayer(HH, WW, 4)
LUM = (PAL[:, :3].astype(np.float64) @ np.array([0.30, 0.55, 0.15]))

# ------------------------------------------------------------------ timing (local = global - 43.4)
T_FLICK1, T_FLICK2 = 0.25, 0.50      # pupil flickers (43.65, 43.9)
T_OFF = 0.90                         # eye_off 44.3: final click, eye dark, head sags, world steps down
T_EMBER = 1.10                       # the warm under-light has sunk to a faint ember
DAWN = normalize([0.58, -0.78, 0.22])   # grey-blue dawn through the broken glass, from above RIGHT

# canonical lit mint: lit planes paint4-paint6, shade paint2-paint3
RL = RampLight({
    'paint': dict(
        neutral=['ink0', 'paint0', 'paint1', 'paint2', 'paint3', 'paint3', 'paint4', 'paint4'],
        warm=['ink0', 'paint0', 'bark1', 'ochre2', 'ochre3', 'rust3', 'ochre4', 'rust4', 'ochre5', 'core3',
              'ochre6', 'core4'],
        cold=['ink0', 'paint0', 'paint1', 'paint2', 'paint3', 'paint4', 'paint5', 'paint6', 'dust10', 'white0'],
        members={'paint0': -2, 'paint1': -1, 'paint2': -1, 'paint3': 0, 'paint4': 1, 'paint5': 2, 'paint6': 2}),
    'steel': dict(
        neutral=['ink0', 'steel0', 'steel1', 'steel1', 'steel2', 'steel3', 'steel3', 'steel4'],
        warm=['ink0', 'steel0', 'bark1', 'rust1', 'bark2', 'rust2', 'ochre3', 'rust3', 'ochre4', 'rust4', 'core3'],
        cold=['ink0', 'steel0', 'steel1', 'steel1', 'steel2', 'steel3', 'steel4', 'steel5', 'steel6', 'white0'],
        members={'steel0': -2, 'steel1': -1, 'steel2': 0, 'steel3': 1, 'steel4': 2, 'steel5': 2, 'steel6': 3}),
    'rust': dict(
        neutral=['ink0', 'rust0', 'rust1', 'rust2', 'rust3', 'rust3', 'rust4'],
        warm=['ink0', 'rust0', 'rust1', 'rust2', 'core1', 'rust3', 'core2', 'rust4', 'core3', 'core4'],
        cold=['ink0', 'rust0', 'rust1', 'rust2', 'rust3', 'dust6', 'rust4', 'dust7', 'rust5'],
        members={'rust0': -2, 'rust1': -1, 'rust2': -1, 'rust3': 0, 'rust4': 1, 'rust5': 2, 'rust6': 2}),
    'stone': dict(
        neutral=['ink0', 'dust1', 'dust2', 'dust3', 'dust4', 'dust5', 'dust5', 'dust6'],
        warm=['ink0', 'dust1', 'bark1', 'ochre2', 'ochre3', 'ochre4', 'ochre5', 'ochre6', 'core4'],
        cold=['ink0', 'dust1', 'dust2', 'dust3', 'ash4', 'ash5', 'ash6', 'ash7', 'dust9', 'ash8'],
        members={'dust0': -3, 'dust1': -2, 'dust2': -1, 'dust3': 0, 'dust4': 1, 'dust5': 2}),
    'brick': dict(
        neutral=['ink0', 'bark0', 'bark1', 'bark2', 'bark3', 'bark4', 'bark4'],
        warm=['ink0', 'bark0', 'rust1', 'bark2', 'rust2', 'ochre3', 'rust3', 'ochre4', 'rust4', 'core3'],
        cold=['ink0', 'bark0', 'bark1', 'bark2', 'bark3', 'dust5', 'bark4', 'dust6'],
        members={'bark1': -2, 'bark2': -1, 'bark3': 0, 'bark4': 1, 'bark5': 2}),
    'soil': dict(
        neutral=['ink0', 'ink1', 'bark0', 'bark1', 'bark2', 'bark2', 'bark3'],
        warm=['ink0', 'ink1', 'bark0', 'rust0', 'bark1', 'rust1', 'core0', 'rust2', 'core1', 'core2', 'core3',
              'core4'],
        cold=['ink0', 'ink1', 'bark0', 'bark1', 'bark2', 'bark3', 'dust4', 'dust5'],
        members={'ochre0': -1, 'ochre1': 0, 'ochre2': 1}),
})


def _nz(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


# ================================================================== THE PLANTER (brick rim, coping, soil)
# The camera looks from the planter side: the soil (with the buried heart and the
# sprout) is in the foreground, lower right; the stone coping of the rim runs as a
# diagonal band; the kneeling robot is behind it, bowed over the soil.
def cop_top(x):
    return 181.0 - 0.26 * (x - 60.0) + 0.00012 * (x - 60.0) ** 2


def cop_w(x):
    return 7.5 - 2.5 * np.clip((x - 60.0) / 260.0, 0, 1)


def _planter():
    yt = cop_top(xs)
    yb = yt + cop_w(xs)
    cop = (ys >= yt) & (ys < yb)
    inner = (ys >= yb) & (ys < yb + 2.6)                             # the rim's inner face above the soil
    soil = ys >= yb + 2.6
    col = np.zeros((HH, WW), np.uint8)
    nrm = np.zeros((HH, WW, 3))
    u = xs - 60.0
    # coping: worn stone slabs with joints, a rounded lip toward the soil
    col = np.where(cop, ci('dust3'), col)
    col = np.where(cop & (ys < yt + 1.2), ci('dust4'), col)
    col = np.where(cop & (np.abs(((u + 11) % 31.0) - 15.5) > 14.7), ci('dust1'), col)
    col = np.where(cop & (NZ.vnoise2(xs / 3.5, ys / 2.0, 81) > 0.74), ci('dust2'), col)
    col = np.where(cop & (ys >= yb - 1.2), ci('dust2'), col)
    nrm = np.where(cop[..., None], _nz(np.array([0.15, -0.9, 0.4])), nrm)
    nrm = np.where((cop & (ys >= yb - 1.2))[..., None], _nz(np.array([0.3, -0.2, 0.93])), nrm)
    # inner face: a sliver of old brick above the soil line
    col = np.where(inner, ci('bark3'), col)
    col = np.where(inner & (np.abs(((u + 3) % 9.0) - 4.5) > 4.0), ci('bark1'), col)
    nrm = np.where(inner[..., None], _nz(np.array([0.25, 0.1, 0.96])), nrm)
    # soil: dark, lumpy clods, calmer far away
    col = np.where(soil, ci('ochre1'), col)
    cl = NZ.vnoise2(xs / 5.0, ys / 3.2, 85)
    col = np.where(soil & (cl > 0.60), ci('ochre2'), col)
    col = np.where(soil & (cl < 0.30), ci('ochre0'), col)
    # clod shadows: the lower edge of each lump
    lump = soil & (cl > 0.60)
    col = np.where(soil & ~lump & shifted(lump, 0, 1), ci('ochre0'), col)
    nrm = np.where(soil[..., None], _nz(np.array([0.1, -0.9, 0.42])), nrm)
    nrm = np.where(lump[..., None], _nz(np.array([0.2, -0.8, 0.55])), nrm)
    return cop, inner, soil, col, nrm


COP_M, INNER_M, SOIL_M, PLANT_COL, PLANT_N = _planter()
PLANT_M = COP_M | INNER_M | SOIL_M

EMBER = np.array([236.0, float(cop_top(236.0) + cop_w(236.0)) + 11.0, 30.0])
SPROUT_B = (251.0, float(cop_top(251.0) + cop_w(251.0)) + 7.0)


def _ember_geom():
    d = np.sqrt(((xs - EMBER[0]) / 2.0) ** 2 + ((ys - EMBER[1]) / 1.0) ** 2)
    cracks = (NZ.vnoise2(xs / 2.0, ys / 1.3, 91) > 0.52) & (d < 7.5) & SOIL_M
    return d, cracks


EMB_D, EMB_CRACK = _ember_geom()


# ================================================================== BACKGROUND (the dome at dawn)
def _background():
    """final colours: the far glass wall of the dome in grey-blue dawn, brighter toward
    the broken panes at the upper right; dark iron ribs; soft shafts of dawn light
    falling from the upper right; the far floor behind the robot."""
    v = 0.39 - 0.22 * (ys / HH) + 0.26 * (xs / WW) + 0.05 * NZ.vnoise2(xs / 34.0, ys / 22.0, 7)
    bg = quantize(v, ['ash3', 'ash4', 'ash5', 'glass2', 'ash6'], cuts=[0.30, 0.40, 0.50, 0.58], dith=0.02)
    # panes: each a slightly different value (old glass), a few broken ones open to the sky
    ribx = [96.0, 150.0, 206.0, 262.0, 316.0]

    def rib_x(x0, y):
        return x0 - (x0 - 205.0) * 0.00035 * np.clip(118.0 - y, 0, None) ** 1.25

    pane_i = np.zeros((HH, WW), np.int32)
    for x0 in ribx:
        pane_i += (xs > rib_x(x0, ys)).astype(np.int32)
    row = (ys > 38 - 0.0008 * (xs - 240) ** 2).astype(np.int32) + (ys > 80 - 0.0005 * (xs - 240) ** 2).astype(np.int32)
    ph = (pane_i * 5 + row * 3) % 7
    bg = np.where((ph == 2) & (ys < 116), np.where(LUM[bg] > LUM[ci('ash4')], ci('ash4'), bg), bg)
    broken = ((pane_i == 4) & (row == 0)) | ((pane_i == 5) & (row <= 1)) | ((pane_i == 3) & (row == 0) & (xs > 238))
    broken &= (ys < 116)
    # an open, broken pane: flat pale sky; only its border keeps jagged shards of old glass
    inner_b = erode(broken, 3)
    jag = NZ.vnoise2(xs / 2.5, ys / 2.5, 12)
    sky = inner_b | (broken & (jag > 0.62))
    bg = np.where(sky, ci('ash6'), bg)
    bg = np.where(sky & (ys < 36), ci('ash7'), bg)
    bg = np.where(broken & ~sky & edge_out(sky), ci('glass3'), bg)            # the broken glass edge glints
    # soft shafts of dawn falling from the upper right (one step lighter, dithered edges)
    sh = np.zeros((HH, WW))
    for (x0, w_) in ((250.0, 10.0), (292.0, 7.0), (214.0, 5.0)):
        d_ = np.abs((xs - x0) + (ys - 0) * 0.62)
        sh = np.maximum(sh, np.clip(1 - d_ / w_, 0, 1))
    shaft = (sh > 0.35 + 0.25 * TH4) & (ys < 150)
    up = {ci('ash3'): 'ash4', ci('ash4'): 'ash5', ci('ash5'): 'ash6', ci('glass2'): 'glass3', ci('ash6'): 'ash7'}
    for k_, v_ in up.items():
        bg = np.where(shaft & (bg == k_), ci(v_), bg)
    # ribs and mullions (iron, dark), a cold glint on their right edges
    for x0 in ribx:
        rx = rib_x(x0, ys)
        r = (np.abs(xs - rx) < 1.3) & (ys < 118)
        bg = np.where(r, ci('ash1'), bg)
        bg = np.where((np.abs(xs - (rx + 1.6)) < 0.5) & (ys < 118) & (NZ.vnoise1(ys / 8.0 + x0, 3) > 0.55),
                      ci('ash5'), bg)
    for y0, k_ in ((38, 0.0008), (80, 0.0005)):
        mm = (np.abs(ys - (y0 - k_ * (xs - 240) ** 2)) < 0.6) & (ys < 118)
        bg = np.where(mm, ci('ash2'), bg)
    # the far floor behind the robot: warm-grey dusty tiles, darker toward us
    fl = ys >= 118
    Z = 30.0 / np.maximum(ys - 104.0, 1.0)
    X = (xs - 160.0) * Z / 150.0
    chk = ((np.floor(X / 0.12) + np.floor(Z / 0.12)) % 2) == 0
    bg = np.where(fl, np.where(chk, ci('dust4'), ci('dust3')), bg)
    bg = np.where(fl & (ys < 121), ci('ash3'), bg)
    bg = np.where(fl & (ys > 150), np.where(chk, ci('dust3'), ci('dust2')), bg)
    return bg


BG = _background()

# ================================================================== ROBOT: head (bowed), torso bent over the planter, arm
HEAD_A = (48.0, 40.8, 54.0)
LENS = {k: v * 1.2 for k, v in dict(R_COL=24.0, R_BAR=21.8, D_FRONT=9.5, R_BEZ_IN=18.4, R_MARK_IN=16.8,
                                       R_GLASS=15.4, D_GLASS=8.6, D_IRIS=4.2, D_COL0=-2.0, D_COL1=1.5).items()}
BOW = (37.0, 31.0, -7.0)                     # yaw, nod, roll: bowed toward the soil
_F, _U, _S = frame_yaw(math.radians(BOW[0]), math.radians(BOW[1]), math.radians(BOW[2]))
HEAD_C = np.array([176.0, 84.0, 0.0])
PIVOT = HEAD_C - _F * 4.0 - _U * HEAD_A[1] * 1.12

BYAW = math.radians(34.0)
FB, UB, SB = frame_yaw(BYAW, pitch=math.radians(42.0))     # bent forward over the planter
BODY_A = (34.0, 44.0, 40.0)
BODY_C = PIVOT - UB * BODY_A[1] * 0.97 - FB * 10.0        # the neck reaches forward out of the bent torso


def _body():
    m, z, nrm, loc = raycast_se_fast(HH, WW, BODY_C, FB, UB, SB, BODY_A, n=4.6, zrange=(-110, 110), steps=110)
    col = np.full((HH, WW), ci('paint3'), np.uint8)
    qf, qu, qs = loc[..., 0], loc[..., 1], loc[..., 2]
    front = m & ((nrm * FB).sum(-1) > 0.5)
    side = m & ((nrm * SB).sum(-1) > 0.6)
    fs = front & (np.abs(qs - BODY_A[2] * 0.70) < 0.6)
    col = np.where(fs, ci('paint0'), col)
    col = np.where(shifted(fs, 1, 0) & front & ~fs, ci('paint4'), col)
    ss = side & (np.abs(qf + BODY_A[0] * 0.05) < 0.6)
    col = np.where(ss, ci('paint0'), col)
    col = np.where(shifted(ss, 1, 0) & side & ~ss, ci('paint4'), col)
    for k in range(4):                                             # domed rivets along the side seam
        d2 = (qf + BODY_A[0] * 0.05 + 3.0) ** 2 + (qu - BODY_A[1] * (0.6 - 0.3 * k)) ** 2
        rv = side & (d2 < 1.2)
        col = np.where(rv, ci('paint5'), col)
        col = np.where(shifted(rv, 1, 1) & side & ~rv, ci('paint1'), col)
    # rust only where water sits: the seams' feet and the bottom edge
    col = np.where((fs | ss) & (qu < -BODY_A[1] * 0.78), ci('rust2'), col)
    col = np.where(m & (qu < -BODY_A[1] * 0.86) & (NZ.vnoise1(xs / 3.0 + ys * 0.2, 5) > 0.35), ci('rust2'), col)
    return m, z, nrm, col


BODY_M, BODY_Z, BODY_N, BODY_COL = _body()
NECK_M = ell(HH, WW, PIVOT[0] + 4.0, PIVOT[1] - 3.0, 15.0, 7.0, 0.2)
SHOULDER = BODY_C + SB * BODY_A[2] * 0.92 + UB * BODY_A[1] * 0.40 - FB * BODY_A[0] * 0.05
WRIST = (133.0, float(cop_top(133.0)) - 3.5)


def _arm():
    """near arm reaching down to the planter rim; c02's hand lies on the coping, its three
    steel fingers curled over the lip toward the soil (mint cuff on the back of the hand)."""
    col = np.zeros((HH, WW), np.uint8)
    m = np.zeros((HH, WW), bool)
    nrm = np.zeros((HH, WW, 3))
    sh = (SHOULDER[0], SHOULDER[1])
    wx, wy = WRIST
    el = (sh[0] + 0.45 * (wx - sh[0]) - 9.0, sh[1] + 0.55 * (wy - sh[1]) + 7.0)
    # the hand first (in front of the forearm)
    fa = math.atan2(wy - el[1], wx - el[0])
    hc, hm_, hn, tip = mech_hand(HH, WW, wx, wy, fa + math.radians(-8.0), s=1.15, curl=0.95, spread=0.9)
    col = np.where(hm_, hc, col)
    nrm = np.where(hm_[..., None], hn, nrm)
    m |= hm_
    for (pts, r0, r1) in (([sh, el], 4.4, 4.0), ([el, (wx, wy)], 3.8, 3.3)):
        mm, al, ac, d = thick_curve(HH, WW, pts, r0, r1)
        mm &= ~m
        dx, dy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
        L_ = math.hypot(dx, dy)
        px_, py_ = -dy / L_, dx / L_
        nn = _nz(np.stack([px_ * ac, py_ * ac, np.sqrt(np.clip(1 - ac ** 2, 0, 1)) + 0.2], -1))
        col = np.where(mm, np.where(ac * np.sign(py_ if abs(py_) > 1e-3 else 1.0) < -0.35, ci('steel3'), ci('steel2')), col)
        nrm = np.where(mm[..., None], nn, nrm)
        m |= mm
    ej = disc(HH, WW, el[0], el[1], 4.2, 0.2) & ~hm_
    col = np.where(ej, ci('steel1'), col)
    col = np.where(ej & disc(HH, WW, el[0] - 1.0, el[1] - 1.2, 2.0, 0.1), ci('steel3'), col)
    nrm = np.where(ej[..., None], _nz(np.stack([(xs - el[0]) / 4.2, (ys - el[1]) / 4.2, np.full_like(xs, 0.9)], -1)), nrm)
    m |= ej
    return m, col, nrm


ARM_M, ARM_COL, ARM_N = _arm()


def _pose(p):
    return head_pose_canon(PIVOT, math.radians(p[0]), math.radians(p[1]), math.radians(p[2]), A=HEAD_A,
                           L=LENS, pivot=(4.0, 1.12), steps=120)


HEAD = _pose(BOW)

# ================================================================== LIGHT
def ember_unit(nrm, P):
    """the buried heart's glow, leaking up through the soil (unit power)."""
    dx, dy, dz = EMBER[0] - P[0], EMBER[1] - P[1], EMBER[2] - P[2]
    dist = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
    lx, ly, lz = dx / dist, dy / dist, dz / dist
    ln = np.clip((nrm[..., 0] * lx + nrm[..., 1] * ly + nrm[..., 2] * lz + 0.15) / 1.15, 0, 1)
    up = np.clip(0.35 + 0.8 * ly, 0, 1)                        # it shines UP out of the ground
    return ln * (0.25 + 0.75 * up) / (1.0 + (dist / 40.0) ** 2)


def key_level(nrm, gain=3.0):
    """the dawn key from the upper right: lit planes two to three steps up (paint4-5)."""
    return gain * np.clip((nrm * DAWN).sum(-1), 0, 1) ** 0.9


def rim_level(M, strong=4.0, second=3.4):
    """the dawn catches the top / right silhouette edges (paint6)."""
    r1 = edge_side(M, 0, -1) | edge_side(M, 1, -1) | edge_side(M, 1, 0)
    inner = M & ~r1
    r2 = (edge_side(inner, 0, -1) | edge_side(inner, 1, -1)) & ~r1
    lv = np.zeros(M.shape)
    lv[r1] = strong
    lv[r2] = second
    return lv


def _compose_static():
    alb = np.zeros((HH, WW), np.uint8)
    nrm = np.zeros((HH, WW, 3))
    z = np.full((HH, WW), -200.0)
    alb = np.where(BODY_M, BODY_COL, alb)
    nrm = np.where(BODY_M[..., None], BODY_N, nrm)
    z = np.where(BODY_M, np.nan_to_num(BODY_Z, nan=-30.0), z)
    nk = NECK_M & ~BODY_M
    alb = np.where(nk, ci('steel1'), alb)
    nrm = np.where(nk[..., None], _nz(np.array([0.3, -0.5, 0.8])), nrm)
    z = np.where(nk, -10.0, z)
    alb = np.where(ARM_M, ARM_COL, alb)
    nrm = np.where(ARM_M[..., None], ARM_N, nrm)
    z = np.where(ARM_M, 20.0, z)
    front = PLANT_M & ~ARM_M
    alb = np.where(front, PLANT_COL, alb)
    nrm = np.where(front[..., None], PLANT_N, nrm)
    z = np.where(front, 30.0, z)
    return alb, nrm, z


S_ALB, S_N, S_Z = _compose_static()
S_ACT = PLANT_M | BODY_M | NECK_M | ARM_M
S_I = np.where(S_ACT, ember_unit(S_N, (xs, ys, S_Z)), 0.0)
S_I = np.where(SOIL_M, 0.55 * np.clip(1 - EMB_D / 17.0, 0, 1) ** 2.0, S_I)
S_I = np.where(INNER_M, S_I * 0.25, S_I)
S_I = np.where(BODY_M & ~ARM_M & ~PLANT_M, S_I * 0.15, S_I)
S_KEY = np.where(S_ACT, key_level(S_N), 0.0)
S_KEY = np.where(SOIL_M, S_KEY * 0.45, S_KEY)
S_KEY = np.maximum(S_KEY, np.where(BODY_M & ~PLANT_M & ~ARM_M, rim_level(BODY_M | NECK_M, 3.6, 3.0), 0.0))
S_KEY = np.maximum(S_KEY, np.where(ARM_M, rim_level(ARM_M, 3.0, 2.4), 0.0))
S_AMB = np.zeros((HH, WW))
S_AMB = np.where(SOIL_M, -0.6, S_AMB)
S_AMB = np.where(SOIL_M & (ys > 170), -1.0, S_AMB)
VIGNETTE = -1.0 * np.clip(np.sqrt(((xs - 175) / 190) ** 2 + ((ys - 100) / 120) ** 2) - 0.50, 0, 1) ** 1.3 / 0.50 ** 1.3

# head light maps (the pose is static; it sags 2 px at T_OFF)
_x0, _y0, _w, _h = HEAD['bbox']
_hy, _hx = np.mgrid[_y0:_y0 + _h, _x0:_x0 + _w].astype(np.float64)
_hz = np.where(np.isnan(HEAD['z']), 0.0, HEAD['z'])
HEAD['I'] = np.where(HEAD['mask'], 0.42 * ember_unit(HEAD['nrm'], (_hx, _hy, _hz)), 0.0)
HEAD['sky'] = np.where(HEAD['mask'], np.maximum(key_level(HEAD['nrm']), rim_level(HEAD['mask'])), 0.0)
HEAD['amb'] = np.where(HEAD['mask'] & ((HEAD['nrm'] * HEAD['U']).sum(-1) < -0.5), -1.0, 0.0)


# ================================================================== TIMING CURVES
def ember(t):
    """under-light power: bright as the heart goes in, sinking, a faint ember from ~1.1."""
    if t >= T_EMBER:
        return 0.16
    a = 1.0 - ease.smoothstep(ease.prog(t, 0.0, T_EMBER))
    return 0.16 + 0.84 * (0.35 + 0.65 * a) * a ** 0.5


def eye_glow(t):
    """the sensor's amber: dimming with the power-down whine, two flickers, dark at T_OFF."""
    if t >= T_OFF:
        return 0.0
    g = 0.85 - 0.45 * ease.prog(t, 0.0, T_OFF)
    if T_FLICK1 <= t < T_FLICK1 + 2 / 30:                         # flicker 1: out for two frames
        return 0.0
    if t >= T_FLICK1 + 2 / 30:
        g *= 0.85
    if T_FLICK2 <= t < T_FLICK2 + 1 / 30:                         # flicker 2: out, half, out, back weaker
        return 0.0
    if T_FLICK2 + 1 / 30 <= t < T_FLICK2 + 2 / 30:
        return g * 0.5
    if T_FLICK2 + 2 / 30 <= t < T_FLICK2 + 4 / 30:
        return 0.0
    if t >= T_FLICK2 + 4 / 30:
        g *= 0.75
    if T_OFF - 1 / 30 <= t < T_OFF:                               # the last breath before the click
        g *= 1.35
    return g


def aperture(t):
    """the iris relaxes open as power fails."""
    return 3.6 + 2.8 * ease.smoothstep(ease.prog(t, 0.1, T_OFF + 0.1))


def sag(t):
    """head drop when the servos let go (px): 0 -> 1 -> 2 (settled)."""
    if t < T_OFF:
        return 0
    return 1 if t < T_OFF + 1 / 30 else 2


FADE_KEYS = [(0.90, 2), (1.00, 3), (1.10, 4), (1.22, 5), (1.34, 6), (1.46, 7)]


def fade_steps(t):
    """after the click the world steps down, palette-fade style, to near-black at 1.6."""
    k = 0
    for (t0, s) in FADE_KEYS:
        if t >= t0:
            k = s
    return k


_FADE_LUT = ELIGHT._lut('neutral', 'plain')


def darken(idx, k):
    if k <= 0:
        return idx
    return _FADE_LUT[ELIGHT.MAXL - int(k), idx].astype(np.uint8)


# ================================================================== the sprout (tiny, planted) + the ember
def _draw_ember(img, t, e):
    """the heart's glow leaking through the cracks of the soil over the burial spot."""
    v = e * (1.15 - EMB_D / 7.5)
    hot = EMB_CRACK & (v > 0.06)
    c = quantize(v, ['rust0', 'core0', 'core1', 'core2', 'core3', 'core4'], cuts=[0.12, 0.27, 0.45, 0.66, 0.9])
    img = np.where(hot, c, img)
    return img


SPROUT_PALS = {   # stem (lit, shade), right leaf pal/edge, left leaf pal/edge — dawn light from the upper right
    'dawn': (('leaf4', 'leaf2'), ('leaf2', 'leaf3', 'leaf4', 'leaf5', 'leaf6'), ('leaf1', 'leaf6', 'leaf3', 'leaf7'),
             ('leaf1', 'leaf2', 'leaf3', 'leaf4', 'leaf5'), ('leaf1', 'leaf5', 'leaf2', 'leaf6')),
    'warm': (('leaf4', 'leaf2'), ('leaf2', 'leaf3', 'leaf4', 'leaf5', 'leaf6'), ('meadow2', 'leaf6', 'leaf3', 'leaf7'),
             ('leaf1', 'leaf2', 'leaf3', 'leaf4', 'leaf5'), ('meadow2', 'leaf5', 'leaf2', 'leaf6')),
}


def _draw_sprout(img, t, e):
    """the tiny planted sprout beside the buried heart: dawn on its upper edges, the
    ember's warmth under its leaves while it is strong."""
    stem, palR, edgeR, palL, edgeL = SPROUT_PALS['warm' if e > 0.5 else 'dawn']
    bx, by = SPROUT_B
    pts = [(bx, by), (bx - 0.5, by - 6), (bx + 0.5, by - 11), (bx - 0.5, by - 15)]
    curve = bezier(*pts, n=20)
    mm, al, ac, d = thick_curve(HH, WW, curve, 1.1, 0.8)
    img = np.where(mm, np.where(ac > 0.2, ci(stem[0]), ci(stem[1])), img)
    tip = pts[-1]
    for (ang, L, W, dr, pal, edge) in ((math.radians(-24), 7.5, 2.8, 0.18, palR, edgeR),
                                       (math.radians(204), 6.8, 2.5, 0.35, palL, edgeL)):
        li, lm, (R, _), tp = raster_leaf(ang, L, W, dr, 0, pal=pal, edge=edge)
        x0 = int(round(tip[0])) - R
        y0 = int(round(tip[1])) - R
        h, w = lm.shape
        ya, yb = max(y0, 0), min(y0 + h, HH)
        xa, xb = max(x0, 0), min(x0 + w, WW)
        sub = img[ya:yb, xa:xb]
        lmm = lm[ya - y0:yb - y0, xa - x0:xb - x0]
        sub[lmm] = li[ya - y0:yb - y0, xa - x0:xb - x0][lmm]
    return img


# ================================================================== RENDER
def render(t):
    t = min(max(t, 0.0), DURATION + 0.5)
    e = ember(t)
    k = fade_steps(t)
    alb = S_ALB.copy()
    I = S_I.copy()
    key = S_KEY.copy()
    amb = S_AMB.copy()
    # ---- head (static pose; a 2 px sag when the servos let go)
    dy = sag(t)
    pd = HEAD
    x0, y0, w, h = pd['bbox']
    y0 += dy
    ya, yb = max(y0, 0), min(y0 + h, HH)
    xa, xb = max(x0, 0), min(x0 + w, WW)
    sl = (slice(ya, yb), slice(xa, xb))
    psl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
    hm = pd['mask'][psl]
    front = (ARM_M | PLANT_M)[sl]                       # the hand and the rim stay in front of the head
    hm = hm & ~front
    alb[sl][hm] = pd['col'][psl][hm]
    I[sl][hm] = pd['I'][psl][hm]
    key[sl][hm] = pd['sky'][psl][hm]
    amb[sl][hm] = pd['amb'][psl][hm]
    # ---- light: grey dawn from the upper right + the ember's under-light (the face cools
    #      faster than the soil)
    I = np.where(SOIL_M | INNER_M | COP_M, I * e, I * e * e)
    img = RL.apply(alb, amb + VIGNETTE, level_of(I, 0.02, 1.5), key, band=0.35, amb_band=0.12, cold_band=0.15)
    dyn = np.zeros((HH, WW), bool)
    dyn[sl] |= hm
    img = np.where(~S_ACT & ~dyn, BG, img)
    # ---- the planted sprout
    img = _draw_sprout(img, t, e)
    # ---- the lens (c01 design): dying amber sensor; the ember reflected until the eye dies;
    #      the white glint of the dawn sky
    g = eye_glow(t)
    refl = []
    if t < T_OFF:
        refl.append(((0.75, 0.62), 0.62, 'core0' if e < 0.6 else 'core1', (0.8 + 1.2 * e, 0.5 + 0.6 * e),
                     'core2' if e > 0.55 else None))
    refl.append(((0.55, -0.8), 0.62, 'white0', 0.5))
    gi, gm = lens_glass(pd, rap=aperture(t), glow=g, sheen_dir=(0.55, -0.85), base=('ink0', 'night1', 'night2'),
                        sheen=('ash3', 'glass2'), refl=refl, glow_scale=1.3)
    sub = img[sl]
    gmm = gm[psl] & ~front
    sub[gmm] = gi[psl][gmm]
    # ---- after the click the world steps down (palette fade), to near-black
    img = darken(img, k)
    # ---- the last light: the heart's glow through the soil cracks (steady, faint at the end)
    img = _draw_ember(img, t, e)
    # the filament cools: two frames of afterglow in the dead sensor, then nothing
    if T_OFF <= t < T_OFF + 2 / 30:
        px, py = pd['pupil']
        xi, yi = int(round(px)), int(round(py)) + dy
        img[yi, xi] = ci('core0') if t < T_OFF + 1 / 30 else ci('rust0')
        if t < T_OFF + 1 / 30:
            for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if gm[yi + ddy - y0, xi + ddx - x0]:
                    img[yi + ddy, xi + ddx] = ci('rust0')
    return PAL[img]
