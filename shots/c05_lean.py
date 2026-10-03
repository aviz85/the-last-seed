"""
c05 · THE LEAN  (global 34.3–38.0, local 0–3.7; the compositor fades it to black 37.7–38.0)

Night inside the ruined greenhouse. A two-shot close-up: the Gardener (damaged;
the canonical close-up head — mint box head, hex ear-bolt, antenna stub, chipped
paint top-left, rust only at seams / rivets / bottom edges; c02's hand) sits
against the rubble on the LEFT, facing right; its chest porthole is the only
light — a dim, failing amber heart. The watering can stands right against its
leg (as in S6), the wilted sprout hanging over the rim.

The sprout REACHES FOR THE ROBOT'S LIGHT, cause and effect: its reach is
integrated from the heart's light (at import) — it only reaches while the light
is steady, sags when the heart stutters, and stretches further in the heart's
swell. The light path is visible: a soft cone of warm haze thrown forward from
the porthole with motes glinting in it, and a warm rim-light on the sprout's side
facing the porthole that grows as it turns (leaves catching amber).

0.0–0.2   stillness; the sprout droops to the right over the can's rim, as in S6.
0.2       (34.5 sprout_lean) the heart steadies: the sprout begins to lift.
0.56–0.74 the heart stutters: the sprout sags back; then it reaches again,
          continuously, straightening and leaning left into the light (~1.5).
1.2       (35.5 robot_notice) the head turns down toward the sprout with a small
          questioning tilt; the iris refocuses (click); a green reflection in the lens.
1.93      a short dip in the heart: the sprout sags a little, recovers; the tilt settles.
2.3       (36.6 look_chest) the NOD: the head pitches down to its own chest (pure
          pitch about the neck, hand-drawn keys: a breath up, the drop, an overshoot,
          a settle; it sinks 2 px into its collar); the lens tilts toward the heart.
2.62–3.35 the heart swells once (realisation); the sprout stretches toward it;
          amber in the lens; the hand drifts up into the glow.
"""
import math
import numpy as np
from engine import *
from engine import ease
from engine import noise as NZ
from shots.lib_closeup import (ci, cis, quantize, clean, ell, disc, edge_in, edge_out, edge_side, shifted,
                               dilate, erode, frame_yaw, raycast_se_fast as raycast_superellipsoid, normalize, bezier,
                               thick_curve, head_pose, head_pose_canon, mech_hand, raster_leaf, RampLight, level_of,
                               lens_glass, PAL)

DURATION = 3.7
HH, WW = 180, 320
ys, xs = np.mgrid[0:HH, 0:WW].astype(np.float64)
TH2 = bayer(HH, WW, 2)
TH4 = bayer(HH, WW, 4)

# ------------------------------------------------------------------ timing (local = global - 34.3)
T_LEAN0, T_LEAN1 = 0.20, 1.40        # sprout_lean 34.5
T_NOTICE = 1.20                      # robot_notice 35.5
T_CHEST = 2.30                       # look_chest 36.6

# ------------------------------------------------------------------ light
MOON = normalize([0.62, -0.62, -0.48])       # cold light from the upper right, from BEHIND
I0, RSTEP = 0.02, 1.5                        # intensity of warm step 1, ratio per step

RL = RampLight({
    'paint': dict(
        neutral=['ink0', 'night1', 'night2', 'paint0', 'paint1', 'paint1', 'glass1', 'paint2'],
        warm=['ink0', 'night1', 'night2', 'bark1', 'ochre1', 'ochre2', 'ochre3', 'rust3', 'ochre4',
              'rust4', 'ochre5', 'core3', 'core4'],
        cold=['ink0', 'night1', 'night2', 'paint0', 'paint1', 'glass1', 'paint2', 'glass2', 'paint3', 'glass3'],
        members={'paint0': -2, 'paint1': -1, 'paint2': -1, 'paint3': 0, 'paint4': 1, 'paint5': 2, 'paint6': 2}),
    'steel': dict(
        neutral=['ink0', 'night0', 'night1', 'steel0', 'steel1', 'steel1', 'steel2', 'steel3'],
        warm=['ink0', 'night0', 'night1', 'bark0', 'bark1', 'rust1', 'bark2', 'rust2', 'bark3', 'rust3',
              'ochre4', 'rust4', 'core3'],
        cold=['ink0', 'night0', 'night1', 'steel0', 'night3', 'steel1', 'night4', 'steel2', 'night5',
              'steel3', 'ash5'],
        members={'steel0': -2, 'steel1': -1, 'steel2': 0, 'steel3': 1, 'steel4': 2, 'steel5': 2, 'steel6': 3}),
    'rust': dict(
        neutral=['ink0', 'night0', 'ink1', 'bark0', 'rust0', 'bark1', 'rust1', 'rust1'],
        warm=['ink0', 'night0', 'ink1', 'rust0', 'bark1', 'rust1', 'core0', 'rust2', 'core1', 'rust3',
              'core2', 'rust4', 'core3', 'core4'],
        cold=['ink0', 'night0', 'ink1', 'bark0', 'bark1', 'bark2', 'dust2', 'dust3', 'ash4'],
        members={'rust0': -2, 'rust1': -1, 'rust2': -1, 'rust3': 0, 'rust4': 1, 'rust5': 2, 'rust6': 2}),
    'dust': dict(
        neutral=['ink0', 'night0', 'night1', 'ink1', 'ink2', 'dust0', 'dust1', 'dust2'],
        warm=['ink0', 'night0', 'night1', 'bark0', 'bark1', 'ochre1', 'bark2', 'ochre2', 'ochre3', 'ochre4',
              'ochre5', 'ochre6'],
        cold=['ink0', 'night0', 'night1', 'ink1', 'night2', 'night3', 'ash2', 'night4', 'ash3', 'ash4'],
        members={'dust0': -2, 'dust1': -1, 'dust2': 0, 'dust3': 1, 'dust4': 1, 'dust5': 2}),
})


LUM = (PAL[:, :3].astype(np.float64) @ np.array([0.30, 0.55, 0.15]))


def _nz(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


# ================================================================== BODY (torso), sitting, facing right
BYAW = math.radians(50.0)
FB, UB, SB = frame_yaw(BYAW, pitch=math.radians(-6.0))     # leaning back against the rubble
DX = 24.0                                           # the whole robot group, placed for balance
BODY_C = np.array([86.0 + DX, 136.0, -20.0])
BODY_A = (32.0, 41.0, 37.0)
PORT_R = 12.0
PORT_RIN = 9.0


def _plane_local(P0, ax1, ax2):
    M2 = np.array([[ax1[0], ax2[0]], [ax1[1], ax2[1]]])
    Minv = np.linalg.inv(M2)
    dx, dy = xs - P0[0], ys - P0[1]
    return Minv[0, 0] * dx + Minv[0, 1] * dy, Minv[1, 0] * dx + Minv[1, 1] * dy


PORT_C3 = BODY_C + FB * BODY_A[0] * 1.0 + UB * (-BODY_A[1] * 0.06)
PU, PV = _plane_local(PORT_C3, -SB, -UB)            # porthole plane coords (u ~ screen right, v ~ down)
PORT_RR = np.sqrt(PU ** 2 + PV ** 2)
PORT_A = np.arctan2(PV, PU)
LIGHT_POS = PORT_C3 + FB * 3.0                     # the heart shines forward through the glass


def _body():
    m, z, nrm, loc = raycast_superellipsoid(HH, WW, BODY_C, FB, UB, SB, BODY_A, n=4.6,
                                            zrange=(-90, 90), steps=120)
    qf, qu, qs = loc[..., 0], loc[..., 1], loc[..., 2]
    col = np.full((HH, WW), ci('paint3'), np.uint8)
    top = m & ((nrm * UB).sum(-1) > 0.5)
    col = np.where(top, ci('paint4'), col)
    front = m & ((nrm * FB).sum(-1) > 0.55)
    side = m & ((nrm * SB).sum(-1) > 0.55)
    # front plate: two vertical seams (lit lip) with domed rivets, as on the canonical model
    for sgn in (-1, 1):
        sm = front & (np.abs(qs - sgn * BODY_A[2] * 0.70) < 0.55)
        col = np.where(sm, ci('paint0'), col)
        col = np.where(shifted(sm, 1, 0) & front & ~sm, ci('paint4'), col)
        for k in range(5):
            qv = BODY_A[1] * (0.62 - 0.30 * k)
            d2 = (qs - sgn * BODY_A[2] * 0.80) ** 2 + (qu - qv) ** 2
            rv = front & (d2 < 1.1)
            col = np.where(rv, ci('paint5'), col)
            col = np.where(shifted(rv, 1, 1) & front & ~rv, ci('paint1'), col)
            if k == 4:                                               # water sits at the lowest rivets
                col = np.where(shifted(rv, 1, 2) & front & ~rv, ci('rust2'), col)
        # rust at the foot of each seam
        foot = sm & (qu < -BODY_A[1] * 0.55)
        col = np.where(foot, ci('rust2'), col)
    # side panel seam + the shoulder socket
    sseam = side & (np.abs(qf + BODY_A[0] * 0.05) < 0.55) & (qu < BODY_A[1] * 0.25)
    col = np.where(sseam, ci('paint0'), col)
    col = np.where(sseam & (qu < -BODY_A[1] * 0.55), ci('rust2'), col)
    # grime low; rust only along the bottom edge (never streaks)
    low = m & (qu < -BODY_A[1] * 0.62)
    col = np.where(low, ci('paint2'), col)
    bott = m & (qu < -BODY_A[1] * 0.86) & (NZ.vnoise1(xs / 3.0 + ys * 0.2, 5) > 0.35)
    col = np.where(bott, ci('rust2'), col)
    col = np.where(bott & (NZ.vnoise1(xs / 2.0, 6) > 0.75), ci('rust3'), col)
    # the porthole seat: a darker ring of paint around the bezel
    seat = m & (PORT_RR <= PORT_R + 1.6) & (PORT_RR > PORT_R)
    col = np.where(seat, ci('paint2'), col)
    return m, z, nrm, loc, col


BODY_M, BODY_Z, BODY_N, BODY_L, BODY_COL = _body()
PORT_ALL = BODY_M & (PORT_RR <= PORT_R + 0.35)
PORT_RING = PORT_ALL & (PORT_RR > PORT_RIN)
PORT_GLASS = PORT_ALL & (PORT_RR <= PORT_RIN)
_pgy, _pgx = np.nonzero(PORT_GLASS)
PORT_XY = (float(_pgx.mean()), float(_pgy.mean()))


def _porthole_static():
    """bezel (steel, bevelled, 8 bolts), hinge on the near side, latch on the far side."""
    col = np.zeros((HH, WW), np.uint8)
    nrm = np.zeros((HH, WW, 3))
    r = PORT_RR
    col = np.where(PORT_RING, ci('steel2'), col)
    col = np.where(PORT_RING & (r > PORT_R - 1.0), ci('steel1'), col)
    col = np.where(PORT_RING & (r < PORT_RIN + 1.0), ci('steel1'), col)
    col = np.where(PORT_RING & (r > PORT_RIN + 1.0) & (r < PORT_R - 1.0) & (np.sin(PORT_A) < -0.3), ci('steel3'), col)
    # bevel normals: outward on the outer half, inward on the inner lip
    radial = (-SB)[None, None, :] * np.cos(PORT_A)[..., None] + (-UB)[None, None, :] * np.sin(PORT_A)[..., None]
    tilt = np.where(r > (PORT_R + PORT_RIN) / 2, 0.9, -0.9)
    n = FB[None, None, :] + tilt[..., None] * radial
    nrm = np.where(PORT_RING[..., None], _nz(n), nrm)
    for k in range(8):
        a = math.radians(22.5 + 45 * k)
        bu, bv = (PORT_R - 1.6) * math.cos(a), (PORT_R - 1.6) * math.sin(a)
        b = PORT_RING & ((PU - bu) ** 2 + (PV - bv) ** 2 < 0.9)
        col = np.where(b, ci('steel4'), col)
    # hinge: two knuckles on the near (screen-left) side
    for vv in (-5.0, 4.0):
        hk = BODY_M & (np.abs(PV - vv) < 2.0) & (PU < -PORT_R + 1.0) & (PU > -PORT_R - 2.4)
        col = np.where(hk, ci('steel2'), col)
        col = np.where(hk & (PV < vv - 1.0), ci('steel3'), col)
        col = np.where(hk & (PV > vv + 1.2), ci('steel0'), col)
        nrm = np.where(hk[..., None], _nz(FB - 0.6 * SB), nrm)
    # latch tab on the far side
    lt = BODY_M & (np.abs(PV + 0.5) < 1.6) & (PU > PORT_R - 0.5) & (PU < PORT_R + 2.0)
    col = np.where(lt, ci('steel2'), col)
    nrm = np.where(lt[..., None], FB, nrm)
    return (col > 0), col, nrm


PH_M, PH_COL, PH_N = _porthole_static()


# ================================================================== LEGS (steel boxes) + BOOTS (rust)
def _frame(f, u):
    f = normalize(f)
    u = normalize(np.asarray(u, float) - f * np.dot(u, f))
    s = np.cross(u, f)
    return f, u, s


def _legs():
    m = np.zeros((HH, WW), bool)
    col = np.zeros((HH, WW), np.uint8)
    nrm = np.zeros((HH, WW, 3))
    zb = np.full((HH, WW), -1e9)
    parts = [
        # (centre, forward, up, half-sizes, n, albedo)  far leg first, near leg last
        ((142.0 + DX, 170.0, -52.0), (0.93, 0.02, 0.36), (0.0, -0.93, 0.36), (26.0, 6.5, 7.5), 4.0, 'steel2'),
        ((171.0 + DX, 163.0, -44.0), (0.30, -0.05, 0.95), (0.0, -1.0, 0.0), (3.6, 7.5, 6.5), 3.0, 'rust3'),
        ((136.0 + DX, 178.0, -6.0), (0.93, 0.06, 0.36), (0.0, -0.93, 0.36), (28.0, 7.0, 8.5), 4.0, 'steel2'),
        ((166.0 + DX, 170.0, 2.0), (0.30, -0.05, 0.95), (0.0, -1.0, 0.0), (4.0, 8.0, 7.0), 3.0, 'rust3'),
    ]
    for (c, f, u, a, n, al) in parts:
        F_, U_, S_ = _frame(f, u)
        mm, z, nn, loc = raycast_superellipsoid(HH, WW, np.array(c), F_, U_, S_, a, n=n, zrange=(-60, 60), steps=70)
        z = np.nan_to_num(z, nan=-1e9)
        vis = mm & (z > zb)
        cc = np.full((HH, WW), ci(al), np.uint8)
        if al == 'steel2':
            cc = np.where((nn * U_).sum(-1) > 0.6, ci('steel3'), cc)                    # top faces
            cc = np.where(np.abs(loc[..., 0] - a[0] * 0.15) < 0.6, ci('steel0'), cc)     # knee joint seam
        else:
            cc = np.full((HH, WW), ci('rust2'), np.uint8)
            cc = np.where(loc[..., 1] < -a[1] * 0.45, ci('rust1'), cc)                  # sole
            cc = np.where((loc[..., 1] < -a[1] * 0.45) & (loc[..., 1] > -a[1] * 0.62), ci('rust1'), cc)
            cc = np.where((nn * U_).sum(-1) > 0.7, ci('rust3'), cc)
        col = np.where(vis, cc, col)
        nrm = np.where(vis[..., None], nn, nrm)
        zb = np.where(vis, z, zb)
        m |= vis
    return m, col, nrm, zb


LEG_M, LEG_COL, LEG_N, LEG_Z = _legs()


# ================================================================== the near ARM (per frame: drifts)
SHOULDER = BODY_C + SB * BODY_A[2] * 0.93 + UB * BODY_A[1] * 0.40 - FB * BODY_A[0] * 0.10


def _arm(drift=0.0):
    """near (right) arm: hangs from the shoulder, forearm forward, the hand (c02's
    design: steel, three fingers, a mint cuff) resting by the hip; `drift` (0..1)
    lifts it toward the chest, into the heart's glow, the fingers opening."""
    col = np.zeros((HH, WW), np.uint8)
    m = np.zeros((HH, WW), bool)
    nrm = np.zeros((HH, WW, 3))
    sh = (SHOULDER[0], SHOULDER[1])
    el = (sh[0] + 4.0 + 5.0 * drift, sh[1] + 30.0 - 3.0 * drift)
    wr = (sh[0] + 21.0 + 11.0 * drift, sh[1] + 44.0 - 12.0 * drift)
    # shoulder socket
    sk = disc(HH, WW, sh[0], sh[1], 5.2, 0.2)
    col = np.where(sk, ci('steel1'), col)
    col = np.where(sk & disc(HH, WW, sh[0] - 0.8, sh[1] - 0.8, 3.0, 0.2), ci('steel2'), col)
    nrm = np.where(sk[..., None], _nz(np.stack([(xs - sh[0]) / 5.2, (ys - sh[1]) / 5.2, np.full_like(xs, 0.8)], -1)), nrm)
    m |= sk
    for (pts, r0, r1) in (([sh, el], 3.6, 3.2), ([el, wr], 3.1, 2.7)):
        mm, al, ac, d = thick_curve(HH, WW, pts, r0, r1)
        dx, dy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
        L_ = math.hypot(dx, dy)
        px_, py_ = -dy / L_, dx / L_
        nn = np.stack([px_ * ac, py_ * ac, np.sqrt(np.clip(1 - ac ** 2, 0, 1))], -1) + np.array([0.0, 0.0, 0.2])
        c = np.where(ac * np.sign(py_ if abs(py_) > 1e-3 else 1) < -0.35, ci('steel3'), ci('steel2'))
        col = np.where(mm, c, col)
        nrm = np.where(mm[..., None], _nz(nn), nrm)
        m |= mm
    # elbow joint
    ej = disc(HH, WW, el[0], el[1], 3.4, 0.2)
    col = np.where(ej, ci('steel1'), col)
    col = np.where(ej & disc(HH, WW, el[0] - 0.7, el[1] - 0.9, 1.6, 0.1), ci('steel3'), col)
    nrm = np.where(ej[..., None], _nz(np.stack([(xs - el[0]) / 3.4, (ys - el[1]) / 3.4, np.full_like(xs, 0.9)], -1)), nrm)
    m |= ej
    # the hand, c02's design (in front of the forearm)
    fa = math.atan2(wr[1] - el[1], wr[0] - el[0])
    ang = fa + math.radians(20.0 - 55.0 * drift)
    hc, hm_, hn, tip = mech_hand(HH, WW, wr[0], wr[1], ang, s=0.95, curl=0.60 - 0.35 * drift)
    col = np.where(hm_, hc, col)
    nrm = np.where(hm_[..., None], hn, nrm)
    m |= hm_
    return m, col, nrm, wr


# ================================================================== the watering CAN (rust / steel, dented)
# the can stands right against the robot's near boot (in S6 the can touches its leg)
CAN_RX, CAN_RY, CAN_H = 17.0, 5.0, 26.0
_LEG_RIGHT = int(np.nonzero(LEG_M[150:178].any(0))[0].max())
CAN_C = (float(_LEG_RIGHT) + CAN_RX - 1.5, 162.0)
CAN_TOP = CAN_C[1] - CAN_H / 2
CAN_BOT = CAN_C[1] + CAN_H / 2


def _can():
    col = np.zeros((HH, WW), np.uint8)
    m = np.zeros((HH, WW), bool)
    nrm = np.zeros((HH, WW, 3))
    cx, cy = CAN_C
    top, bot = CAN_TOP, CAN_BOT
    rel = (xs - cx) / CAN_RX
    body = (np.abs(rel) <= 1.0) & (ys >= top) & (ys <= bot + CAN_RY * np.sqrt(np.clip(1 - rel ** 2, 0, 1)))
    col = np.where(body, ci('rust3'), col)
    col = np.where(body & (NZ.vnoise2(xs / 4.0, ys / 3.0, 17) > 0.58), ci('rust2'), col)
    col = np.where(body & (NZ.vnoise2(xs / 2.5, ys / 2.5, 18) > 0.80), ci('steel2'), col)      # bare-metal scuffs
    col = np.where(body & (np.abs(ys - (top + 6 + 1.2 * rel ** 2)) < 0.7), ci('rust1'), col)  # rolled band
    col = np.where(body & (np.abs(ys - (top + 7 + 1.2 * rel ** 2)) < 0.5), ci('rust4'), col)
    nx = np.clip(rel, -1, 1)
    nn = np.stack([nx, np.zeros_like(nx), np.sqrt(np.clip(1 - nx ** 2, 0, 1))], -1)
    # a shallow dent: a darker crescent on its upper-left, a lit lip on its lower-right
    dr = np.sqrt(((xs - (cx - 5)) / 4.2) ** 2 + ((ys - (cy + 4)) / 3.2) ** 2)
    col = np.where(body & (dr < 1.0) & ((xs - (cx - 5)) + (ys - (cy + 4)) < -1.0), ci('rust1'), col)
    col = np.where(body & (dr < 1.15) & (dr > 0.75) & ((xs - (cx - 5)) + (ys - (cy + 4)) > 1.5), ci('rust4'), col)
    nrm = np.where(body[..., None], nn, nrm)
    m |= body
    # rim (rolled lip) + soil inside the mouth
    rim = ell(HH, WW, cx, top, CAN_RX + 0.5, CAN_RY + 0.3, 0.2)
    inner = ell(HH, WW, cx, top + 0.5, CAN_RX - 1.8, CAN_RY - 1.3, 0.2)
    col = np.where(rim, ci('rust4'), col)
    col = np.where(rim & (ys > top + 1.5), ci('rust3'), col)
    col = np.where(inner, ci('dust2'), col)
    col = np.where(inner & (ys < top - 0.5), ci('rust1'), col)                             # the back inner wall
    col = np.where(inner & (ys >= top - 0.5) & (NZ.vnoise2(xs / 1.5, ys / 1.2, 23) > 0.6), ci('dust3'), col)
    nrm = np.where(rim[..., None], _nz(np.stack([(xs - cx) / CAN_RX * 0.6, np.full_like(xs, -0.8), np.full_like(xs, 0.4)], -1)), nrm)
    nrm = np.where(inner[..., None], np.array([0.0, -1.0, 0.15]), nrm)
    m |= rim
    soil = inner & (ys >= top - 0.5)
    # handle: an arch over the top, from back-left to back-right
    pts = bezier((cx - 13, top - 0.5), (cx - 6, top - 15), (cx + 8, top - 15), (cx + 13, top - 1.5), n=28)
    hm, al, ac, d = thick_curve(HH, WW, pts, 1.7, 1.7)
    behind = hm & inner & (ys > top - 1.0)
    hm &= ~behind
    col = np.where(hm & ~m, ci('rust3'), col)
    col = np.where(hm & ~m & (ac < -0.2), ci('rust4'), col)
    nrm = np.where((hm & ~m)[..., None], _nz(np.stack([np.zeros_like(ac), -np.abs(ac) - 0.2, np.full_like(ac, 0.7)], -1)), nrm)
    m |= hm
    # spout: tapered tube from the lower right, up to a sprinkler rose
    sp = [(cx + CAN_RX - 3, cy + 6), (cx + 27, cy - 2), (cx + 39, cy - 14)]
    smk, al, ac, d = thick_curve(HH, WW, sp, 3.0, 1.6)
    smk &= ~body
    col = np.where(smk, ci('rust3'), col)
    col = np.where(smk & (ac < -0.3), ci('rust4'), col)
    nrm = np.where(smk[..., None], _nz(np.stack([ac * 0.5, ac * 0.8, np.full_like(ac, 0.6)], -1)), nrm)
    m |= smk
    rose = ell(HH, WW, cx + 41, cy - 16, 2.8, 3.8, 0.2)
    col = np.where(rose, ci('steel2'), col)
    col = np.where(rose & ell(HH, WW, cx + 41.6, cy - 16, 1.4, 2.4, 0.2), ci('steel1'), col)
    nrm = np.where(rose[..., None], _nz(np.array([0.8, -0.2, 0.55])), nrm)
    m |= rose
    return m, col, nrm, soil


CAN_M, CAN_COL, CAN_N, CAN_SOIL = _can()
SPROUT_BASE = (CAN_C[0] - 3.0, CAN_TOP + 0.5)

# ================================================================== BACKGROUND: the dark greenhouse
FLOOR_Y0 = 112
HOR_Y = 96.0
TILE = 0.07


def _floor_coords():
    Z = 40.0 / np.maximum(ys - HOR_Y, 1.0)          # depth
    X = (xs - 190.0 - DX) * Z / 216.0
    return X / TILE, Z / TILE


FX, FZ = _floor_coords()


def _background():
    """final night colours for the far glass wall (not ramp-lit) + albedo for floor/rubble/rib."""
    # the dome's far glass: near-black, a breath of navy toward the moon (upper right)
    v = 0.55 * (xs / WW) + 0.35 * (1 - ys / FLOOR_Y0) - 0.2 + 0.04 * NZ.vnoise2(xs / 40.0, ys / 30.0, 4)
    bg = quantize(v, ['night0', 'night1', 'night2'], cuts=[0.42, 0.63], dith=0.10)
    # far ribs of the dome (iron, black on the faint navy) + a mullion or two
    def rib_x(x0, y):
        return x0 - (x0 - 200.0) * 0.00035 * np.clip(FLOOR_Y0 - y, 0, None) ** 1.25
    for x0 in (176, 236, 296):
        rx = rib_x(x0, ys)
        r = (np.abs(xs - rx) < 1.1) & (ys < FLOOR_Y0)
        bg = np.where(r, ci('ink0'), bg)
        # a cold rim on the right edge of the rib, only in moonlit gaps
        rim = (np.abs(xs - (rx + 1.5)) < 0.5) & (ys < FLOOR_Y0 - 6) & (NZ.vnoise1(ys / 7.0 + x0, 7) > 0.74)
        bg = np.where(rim, ci('night3'), bg)
    for yy0 in (30, 66):
        mm = (np.abs(ys - (yy0 + 0.0010 * (xs - 230) ** 2)) < 0.55) & (xs > 150)
        bg = np.where(mm, ci('ink0'), bg)
    # glints: a few cold sparks on broken glass far behind
    for (gx, gy, big) in ((238, 31, 1), (297, 58, 0), (178, 18, 0), (265, 87, 0)):
        bg[gy, gx] = ci('glass3' if big else 'night5')
        if big:
            for dx_, dy_ in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                bg[gy + dy_, gx + dx_] = ci('night4')
    # ---- floor (albedo, ramp-lit): checker tiles in perspective
    fl = ys >= FLOOR_Y0
    chk = ((np.floor(FX) + np.floor(FZ)) % 2) == 0
    alb = np.where(fl, np.where(chk, ci('dust2'), ci('dust1')), 0).astype(np.uint8)
    gx_ = np.abs((FX % 1.0) - 0.5) > 0.5 - 0.9 / np.maximum(216.0 * TILE / np.maximum(FZ * TILE, 1e-3), 1.0) * 1.0
    gz_ = np.abs((FZ % 1.0) - 0.5) > 0.47
    alb = np.where(fl & (gx_ | gz_), ci('dust0'), alb)
    # cracks and dirt
    alb = np.where(fl & (NZ.vnoise2(FX * 1.3, FZ * 1.3, 41) > 0.80), ci('dust0'), alb)
    # ---- rubble behind the robot (left): broken masonry — angular blocks with flat faces
    rub = np.zeros((HH, WW), bool)
    rn = np.zeros((HH, WW, 3))
    blocks = [(10 + DX, 136, 19, 12, 0.10), (33 + DX, 150, 15, 10, -0.25), (4 + DX, 168, 22, 14, 0.05),
              (28 + DX, 174, 17, 9, 0.30), (2, 120, 16, 13, -0.2), (-6, 150, 18, 16, 0.25), (8, 96, 12, 9, 0.1),
              (18 + DX, 112, 15, 9, -0.35), (42 + DX, 126, 11, 8, 0.45), (-4, 94, 14, 11, 0.20),
              (38 + DX, 102, 9, 6, -0.15), (26, 76, 13, 9, 0.3)]
    for i, (sx, sy, rx, ry, rot) in enumerate(blocks):
        c, s_ = math.cos(rot), math.sin(rot)
        u = ((xs - sx) * c + (ys - sy) * s_) / rx
        v = (-(xs - sx) * s_ + (ys - sy) * c) / ry
        e = np.abs(u) ** 6 + np.abs(v) ** 6 + 0.18 * NZ.vnoise2(xs / 3.0, ys / 3.0, 50 + i)
        st = (e < 1.0) & ~rub
        # flat faces: the top face (toward -v) faces up, the side faces the viewer
        top = v < -0.55
        n = np.where(top[..., None], np.array([0.0, -0.95, 0.3]), np.array([-0.2, 0.0, 0.98]))
        rn = np.where(st[..., None], n, rn)
        rub |= st
        alb_i = np.where(top, ci('dust3'), ci('dust2'))
        if i % 3 == 1:                                         # a chunk of old brick wall
            brick = (np.abs(((u * rx + 3 * (np.floor(v * ry / 4) % 2)) % 7.0) - 3.5) > 3.0) | ((v * ry) % 4.0 < 0.8)
            alb_i = np.where(~top & brick, ci('dust1'), alb_i)
        rub_prev = rub & ~st
        alb = np.where(st, alb_i, alb)
    alb = np.where(rub & edge_in(rub) & (rn[..., 1] > -0.5), ci('dust0'), alb)
    # (the fallen rib of the wide is out of frame here: nothing crosses the robot's silhouette)
    rib = np.zeros((HH, WW), bool)
    return bg, alb, fl, rub, rib, rn


BG, BG_ALB, FLOOR_M, RUB_M, RIB_M, BG_N = _background()

# ================================================================== static composite (albedo + normals + z)
STATIC_ALB = BG_ALB.copy()
STATIC_N = BG_N.copy()
STATIC_N[FLOOR_M & ~RUB_M & ~RIB_M] = np.array([0.0, -1.0, 0.25])
STATIC_Z = np.where(FLOOR_M, -60.0 + (ys - FLOOR_Y0) * 1.0, -120.0)
STATIC_Z = np.where(RUB_M | RIB_M, -70.0, STATIC_Z)
ACTOR_STATIC = BODY_M | LEG_M | CAN_M
STATIC_ALB = np.where(BODY_M, BODY_COL, STATIC_ALB)
STATIC_N = np.where(BODY_M[..., None], BODY_N, STATIC_N)
STATIC_Z = np.where(BODY_M, np.nan_to_num(BODY_Z, nan=-20.0), STATIC_Z)
STATIC_ALB = np.where(PH_M, PH_COL, STATIC_ALB)
STATIC_N = np.where(PH_M[..., None], PH_N, STATIC_N)
_legfront = LEG_M & ~(BODY_M & (np.nan_to_num(BODY_Z, nan=-1e9) > LEG_Z))
STATIC_ALB = np.where(_legfront, LEG_COL, STATIC_ALB)
STATIC_N = np.where(_legfront[..., None], LEG_N, STATIC_N)
STATIC_Z = np.where(_legfront, LEG_Z, STATIC_Z)
STATIC_ALB = np.where(CAN_M, CAN_COL, STATIC_ALB)
STATIC_N = np.where(CAN_M[..., None], CAN_N, STATIC_N)
STATIC_Z = np.where(CAN_M, -30.0, STATIC_Z)
PIVOT = BODY_C + UB * BODY_A[1] * 0.97 - FB * 3.0
NECK_M = ell(HH, WW, PIVOT[0] + 2.0, PIVOT[1] - 3.0, 13.0, 6.0, 0.2) & (ys < PIVOT[1] + 1.0)
STATIC_ALB = np.where(NECK_M, ci('steel1'), STATIC_ALB)
STATIC_ALB = np.where(NECK_M & (np.abs(ys - (PIVOT[1] - 4.0)) < 0.6), ci('steel0'), STATIC_ALB)
STATIC_N = np.where(NECK_M[..., None], _nz(np.stack([(xs - PIVOT[0] - 2.0) / 13.0, np.full_like(xs, -0.3),
                                                      np.full_like(xs, 0.8)], -1)), STATIC_N)
STATIC_Z = np.where(NECK_M, -18.0, STATIC_Z)
BG_ONLY = ~(FLOOR_M | RUB_M | RIB_M | BODY_M | LEG_M | CAN_M | NECK_M)


# ================================================================== LIGHT maps (unit power)
def warm_unit(nrm, P):
    """porthole light (unit power) on surfaces with normals nrm at positions P (x,y,z arrays)."""
    dx, dy, dz = LIGHT_POS[0] - P[0], LIGHT_POS[1] - P[1], LIGHT_POS[2] - P[2]
    dist = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
    lx, ly, lz = dx / dist, dy / dist, dz / dist
    ln = nrm[..., 0] * lx + nrm[..., 1] * ly + nrm[..., 2] * lz
    ln = np.clip((ln + 0.2) / 1.2, 0, 1)
    # emission: the glass faces forward; little light goes sideways/back
    ce = -(lx * FB[0] + ly * FB[1] + lz * FB[2])
    em = (0.05 + 0.95 * np.clip(ce, 0, 1)) ** 1.3
    fall = 1.0 / (1.0 + (dist / 48.0) ** 2)
    near = 0.5 * np.clip(1 - dist / 32.0, 0, 1) ** 1.5         # the glowing glass, seen from the side
    return ln * (em * fall + near)


def bounce_unit(nrm, P):
    """warm light bounced up from the lit floor pool, on down-facing surfaces."""
    cx, cy = LIGHT_POS[0] + 30.0, 166.0
    d = np.sqrt(((P[0] - cx) / 1.6) ** 2 + ((P[1] - cy) / 1.0) ** 2)
    down = np.clip(nrm[..., 1], 0, 1) ** 2
    return 0.06 * down * np.clip(1 - d / 70.0, 0, 1) ** 1.5


def moon_level(nrm, gain=1.3):
    """broad, faint moonlight on surfaces that face it (the crisp rim comes from rim_level)."""
    nl = np.clip((nrm * MOON).sum(-1), 0, 1)
    return gain * nl ** 1.5


def rim_level(M, strong=4.0, second=2.3):
    """the cold rim: the 1-2 px of a silhouette that face the moon (top / right edges)."""
    r1 = edge_side(M, 0, -1) | edge_side(M, 1, 0) | edge_side(M, 1, -1)
    inner = M & ~r1
    r2 = (edge_side(inner, 0, -1) | edge_side(inner, 1, -1)) & ~r1
    lv = np.zeros(M.shape)
    lv[r1] = strong
    lv[r2] = second
    return lv


def _floor_warm():
    """the warm pool on the floor (hand-shaped: an elongated ellipse thrown forward),
    with the can's shadow cut out of it."""
    lf = np.array([LIGHT_POS[0] + 22.0, 174.0])
    dx, dy = xs - lf[0], (ys - lf[1]) * 3.2
    d = np.sqrt(dx * dx + dy * dy)
    ang = np.arctan2(dy, dx)
    fwd = np.clip(0.25 + 0.75 * np.cos(ang), 0, 1)
    I = 0.19 * fwd / (1.0 + (d / 62.0) ** 2)
    # the can's shadow: a wedge behind it, away from the light
    cb = np.array([CAN_C[0], CAN_BOT - 1.0])
    cdx, cdy = cb[0] - lf[0], (cb[1] - lf[1]) * 3.2
    cd = math.hypot(cdx, cdy)
    ca = math.atan2(cdy, cdx)
    half = math.atan2(CAN_RX * 0.95, cd)
    da = np.abs(np.angle(np.exp(1j * (ang - ca))))
    sh = (da < half) & (d > cd - 2.0)
    soft = np.clip((half - da) / (half * 0.35), 0, 1)
    I = np.where(sh, I * (1 - 0.85 * soft), I)
    # the robot's own legs shade the floor right under them
    ao = dilate(LEG_M, 2) & ~LEG_M & (ys > 150)
    I = np.where(ao, I * 0.5, I)
    return I


def _static_levels():
    P = (xs, ys, STATIC_Z)
    I = np.zeros((HH, WW))
    act = BODY_M | LEG_M | CAN_M | PH_M | NECK_M
    Iw = warm_unit(STATIC_N, P) + bounce_unit(STATIC_N, P)
    I = np.where(act, Iw, I)
    fl = FLOOR_M & ~act & ~RUB_M & ~RIB_M
    I = np.where(fl, _floor_warm(), I)
    I = np.where(RUB_M | RIB_M, 0.25 * warm_unit(STATIC_N, P), I)
    # the porthole shines forward: the torso behind its own plane only gets a glow halo
    on_body = BODY_M & ~PH_M
    halo = np.clip(1 - (PORT_RR - PORT_R) / 5.0, 0, 1) ** 1.6
    I = np.where(on_body, np.maximum(I * 0.35, 0.05 * halo), I)
    lip = PORT_RR < PORT_RIN + 1.2
    I = np.where(PH_M, np.where(lip, 0.13, 0.004), I)
    mo = moon_level(STATIC_N)
    mo = np.where(FLOOR_M & ~RUB_M & ~RIB_M, 0.0, mo)
    # crisp rims on each silhouette (what is in front hides the rims behind)
    front = BODY_M | NECK_M | LEG_M | CAN_M
    rubv = RUB_M & ~RIB_M & ~front
    mo = np.maximum(mo, np.where(rubv, rim_level(RUB_M | front, 2.2, 1.2), 0.0))
    for M, s1, s2 in ((RIB_M & ~front, 3.0, 1.5),
                      (BODY_M, 4.0, 2.3), (LEG_M & ~BODY_M, 3.0, 1.6), (CAN_M, 3.6, 2.0), (NECK_M, 3.0, 1.5)):
        mo = np.maximum(mo, np.where(M, rim_level(M, s1, s2), 0.0))
    mo = np.where(PH_M, np.minimum(mo, 1.0), mo)
    # ambient: occlusion & vignette & depth
    amb = np.zeros((HH, WW))
    amb = np.where(FLOOR_M, -np.clip((FZ * TILE - 0.62) / 0.7, 0, 2.2), amb)
    amb = np.where(RUB_M, -0.6, amb)
    amb = np.where(BODY_M & (BODY_L[..., 1] < -BODY_A[1] * 0.45), -1.0, amb)
    amb = np.where(BODY_M & ((BODY_N * SB).sum(-1) > 0.6), amb - 1.0, amb)           # side face: turned from the light
    return I, mo, amb


STATIC_I, STATIC_MOON, STATIC_AMB = _static_levels()
VIGNETTE = -1.3 * np.clip(np.sqrt(((xs - 150 - DX * 0.5) / 190) ** 2 + ((ys - 105) / 120) ** 2) - 0.45, 0, 1) ** 1.3 / 0.55 ** 1.3

def _cone():
    """visible-light field of the porthole in the dusty air: forward cone, soft edges."""
    ax = np.array([FB[0], FB[1] * 0.5 - 0.03])
    ax = ax / np.linalg.norm(ax)
    dx, dy = xs - PORT_XY[0], ys - PORT_XY[1]
    d = np.sqrt(dx * dx + dy * dy) + 1e-6
    cth = (dx * ax[0] + dy * ax[1]) / d
    th = np.arccos(np.clip(cth, -1, 1))
    cone = np.clip(1 - th / math.radians(27.0), 0, 1) ** 1.4
    fall = np.clip(1 - d / 150.0, 0, 1) ** 1.2
    near = np.clip(1 - d / 26.0, 0, 1) ** 1.5                  # the glow right at the glass
    return np.maximum(0.62 * cone * fall * np.clip(d / 12.0, 0, 1), 0.55 * near)


CONE = _cone()

# ================================================================== HEAD POSES (canonical head, c01 reference)
HEAD_A = (40.0, 34.0, 45.0)
REST = (61.0, 13.0, -4.0)
LOOK = (57.0, 30.0, 8.0)          # the notice: down toward the sprout, a questioning tilt
WATCH = (56.0, 30.0, 2.0)         # the question settles
# the nod to its own chest: PURE PITCH about the neck, hand-drawn keys on twos —
# a small breath up, the drop, a slight overshoot, a settle; the head sinks into
# its collar (px) as the weight goes forward
NOD_KEYS = [(2.300, 27.0, 0), (2.367, 30.0, 0), (2.467, 35.0, 0), (2.567, 40.0, 1), (2.667, 44.0, 1),
            (2.767, 47.0, 2), (2.900, 45.0, 2)]


def _mix(a, b, s):
    return tuple(a[i] + (b[i] - a[i]) * s for i in range(3))


# servo schedule for the notice: (start time, fraction) on twos
_MOVE_A = [(1.200, 0.18), (1.267, 0.44), (1.333, 0.71), (1.400, 0.92), (1.467, 1.05), (1.567, 1.0)]
_MOVE_W = [(1.933, 0.5), (2.000, 1.0)]


def head_key(t):
    """(key, (yaw, nod, roll), sink px) of the held pose at time t."""
    k, pose, sink = 'rest', REST, 0
    for i, (ts, s) in enumerate(_MOVE_A):
        if t >= ts:
            k, pose = 'a%d' % i, _mix(REST, LOOK, s)
    for i, (ts, s) in enumerate(_MOVE_W):
        if t >= ts:
            k, pose = 'w%d' % i, _mix(LOOK, WATCH, s)
    for i, (ts, pitch, sk) in enumerate(NOD_KEYS):
        if t >= ts:
            k, pose, sink = 'n%d' % i, (WATCH[0], pitch, WATCH[2]), sk
    return k, pose, sink


def _build_pose(p):
    pd = head_pose_canon(PIVOT, math.radians(p[0]), math.radians(p[1]), math.radians(p[2]), A=HEAD_A,
                         pivot=(4.0, 1.12), steps=100)
    x0, y0, w, h = pd['bbox']
    yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w].astype(np.float64)
    zz = np.where(np.isnan(pd['z']), 0.0, pd['z'])
    nrm = pd['nrm']
    m = pd['mask']
    # the head sits ABOVE the porthole: only scattered glow + bounce reach it
    Iw = 0.45 * warm_unit(nrm, (xx, yy, zz)) + 0.5 * bounce_unit(nrm, (xx, yy, zz))
    pd['I'] = np.where(m, Iw, 0.0)
    pd['moon'] = np.where(m, np.maximum(moon_level(nrm, 1.3), rim_level(m, 4.0, 2.3)), 0.0)
    # the underside of the head is darker; the side face is turned from the light
    amb = np.where(m & ((nrm * pd['U']).sum(-1) < -0.5), -1.0, 0.0)
    amb = np.where(m & ((nrm * pd['S']).sum(-1) > 0.6), amb - 1.0, amb)
    pd['amb'] = amb
    gy, gx = np.nonzero(pd['glass'])
    pd['gc'] = (gx.mean() + x0, gy.mean() + y0) if len(gx) else pd['pupil']
    return pd


POSES = {}
for _t in sorted(set([0.0] + [a for a, _ in _MOVE_A] + [a for a, _ in _MOVE_W] + [a for a, _, _ in NOD_KEYS])):
    _k, _p, _s = head_key(_t + 1e-4)
    if _k not in POSES:
        POSES[_k] = _build_pose(_p)


# ================================================================== the heart
def core_power(t):
    """the failing heart (~0.3). Weak and unsteady until 0.2, then it steadies — the
    sprout begins to reach. A stutter 0.56-0.74 (the sprout sags), steady again (it
    reaches, continuously), a short dip at 1.93, the realisation swell 2.62-3.35
    (the sprout reaches further), a last dip at 3.30 (the compositor fades out
    from 3.4)."""
    f = ease.flicker(t + 2.3, seed=61, rate=9.0)
    f2 = ease.flicker(t + 2.3, seed=67, rate=23.0)
    g = 0.30 * (0.90 + 0.16 * f) * (0.96 + 0.06 * f2)
    g *= 0.70 + 0.30 * ease.smoothstep(ease.prog(t, 0.08, 0.26))
    if 0.56 <= t < 0.74:                                   # stutter: out, half, out
        u = (t - 0.56) / 0.18
        g *= 0.40 if (u < 0.36 or u > 0.64) else 0.72
    for tc, w in ((1.93, 0.045), (3.30, 0.05)):
        if abs(t - tc) < w:
            g *= 0.55
    return g * (1.0 + 0.60 * pulse(t))


def pulse(t):
    return ease.smoothstep(ease.prog(t, 2.62, 2.92)) * (1 - ease.smoothstep(ease.prog(t, 3.0, 3.35)))


def _heart(g, t):
    """the core seen through the porthole glass: warm orb, dark rim, crack glint."""
    r = PORT_RR
    k = g / 0.3
    cu, cv = 0.6, 1.0                                         # the orb sits a little low in its cradle
    ro = np.sqrt((PU - cu) ** 2 + ((PV - cv) * 1.0) ** 2)
    v = 5.2 * k ** 0.8 - ro
    out = quantize(v, ['rust0', 'core0', 'core1', 'core2', 'core3', 'core4', 'core5'],
                   cuts=[-3.2, -1.6, 0.0, 1.4, 2.6, 3.8], dith=0.7)
    # dark inner shadow at the top of the glass (the bezel lip)
    out = np.where((r > PORT_RIN - 1.3) & (PV < -2.0), ci('core0'), out)
    # the old crack across the glass catches the light
    cr = (np.abs((PU + 1.0) - 0.8 * (PV + 3.5) + 0.05 * (PV + 3.5) ** 2) < 0.45) & (PV < 1.5) & (PV > -8.5)
    out = np.where(cr & (v > -2.5), ci('core4'), out)
    out = np.where(cr & (v <= -2.5), ci('core2'), out)
    # glass glint (cold, from the moon) at the upper right of the glass
    gl = (np.abs(r - (PORT_RIN - 1.6)) < 0.55) & (np.cos(PORT_A + 0.9) > 0.7)
    out = np.where(gl, ci('core5') if k > 1.2 else ci('core3'), out)
    return out


# ================================================================== the SPROUT — it reaches for the robot's light
# Cause and effect: the sprout's reach s(t) is integrated from the heart's light
# (at import, deterministic): it can only reach while the light is steady, sags
# when the heart stutters, and reaches further in the realisation swell.
# s: 0 = wilted hook over the can's rim (as in S6), 0.45 upright, 1 leaning to
# the porthole, 1.35 stretched toward it.
STEM_W = (1.35, 0.95)
LEAF_R = dict(L=10.5, W=3.6)
LEAF_L = dict(L=9.5, W=3.2)
# stem control points relative to the base, per key pose
SPROUT_KEYS = [
    (0.00, [(0, 0), (0.5, -15), (6.0, -26), (16.5, -12.0)]),
    (0.45, [(0, 0), (0.3, -10), (0.0, -20), (-1.0, -29.5)]),
    (1.00, [(0, 0), (-1.0, -10), (-7.0, -19), (-17.0, -24.0)]),
    (1.35, [(0, 0), (-1.5, -10), (-10.0, -18), (-22.5, -21.0)]),
]
# cotyledons (angle deg, droop) right / left, per key
LEAF_KEYS = [
    (0.00, ((78.0, 0.55), (112.0, 0.60))),
    (0.45, ((-30.0, 0.12), (208.0, 0.38))),
    (1.00, ((-62.0, 0.10), (190.0, 0.22))),
    (1.35, ((-72.0, 0.06), (182.0, 0.14))),
]


def _key_interp(keys, s):
    s = min(max(s, keys[0][0]), keys[-1][0])
    for (s0, a), (s1, b) in zip(keys[:-1], keys[1:]):
        if s <= s1:
            u = (s - s0) / (s1 - s0)
            return a, b, u
    return keys[-1][1], keys[-1][1], 0.0


def _reach_table(dt=1.0 / 240.0, t1=DURATION + 1.0):
    ts = np.arange(0.0, t1 + dt, dt)
    s = 0.0
    out = np.zeros(len(ts))
    for i, t in enumerate(ts):
        will = ease.smoothstep(ease.prog(t, 0.18, 1.10)) ** 0.6   # its strength returns from 0.2
        lt = min(max(core_power(t) / 0.30, 0.0), 1.8)
        resp = min(max((lt - 0.35) / 0.60, 0.0), 2.0)             # it needs the light: a stutter starves it
        target = min(will * resp, 1.35)
        k = 3.4 if target > s else 7.5                            # reaches steadily, sags fast
        s += (target - s) * min(k * dt, 1.0)
        out[i] = s
    return ts, out


_RT, _RS = _reach_table()


def reach(t):
    return float(np.interp(t, _RT, _RS))


def sprout_state(t):
    s = reach(t)
    a, b, u = _key_interp(SPROUT_KEYS, s)
    pts = [(a[i][0] + (b[i][0] - a[i][0]) * u, a[i][1] + (b[i][1] - a[i][1]) * u) for i in range(4)]
    # alive: a slow sway once it is up, and a small tremor while the light flickers
    if t > 0.3:
        k = min(1.0, (t - 0.3) / 0.8)
        sw = k * (0.55 * math.sin((t - 0.3) * 2.1) + 0.25 * math.sin((t - 0.3) * 5.3 + 1.0))
        pts[3] = (pts[3][0] + sw, pts[3][1] + 0.3 * sw)
        pts[2] = (pts[2][0] + 0.4 * sw, pts[2][1])
    return s, pts


def leaf_angles(s):
    a, b, u = _key_interp(LEAF_KEYS, s)
    return [(math.radians(a[i][0] + (b[i][0] - a[i][0]) * u), a[i][1] + (b[i][1] - a[i][1]) * u) for i in range(2)]


LEAFPALS = [('leaf0', 'leaf1', 'leaf2', 'leaf3', 'leaf3'),
            ('leaf1', 'leaf2', 'leaf3', 'leaf4', 'leaf4'),
            ('leaf1', 'leaf2', 'leaf3', 'leaf4', 'leaf5'),
            ('leaf1', 'leaf2', 'leaf3', 'meadow4', 'meadow5')]          # leaves catching the amber
STEMPALS = [('leaf1', 'leaf2', 'leaf0'), ('leaf2', 'leaf3', 'leaf1'), ('leaf2', 'leaf4', 'leaf1'),
            ('leaf2', 'leaf4', 'leaf1')]                                 # mid, lit, shade
WARM_RIM = ['meadow3', 'meadow4', 'meadow5', 'gold2', 'gold3']            # the heart's light on the green


def _draw_sprout(img, t, g):
    """the sprout in its own faint light plus the heart's warm light on the side
    that faces the porthole — a rim that grows as it turns toward it."""
    s, pts = sprout_state(t)
    bx, by = SPROUT_BASE
    P = [(bx + p[0], by + p[1]) for p in pts]
    curve = bezier(P[0], P[1], P[2], P[3], n=30)
    lv = 0 if s < 0.25 else (1 if s < 0.6 else (2 if s < 1.02 else 3))
    lt = min(max(g / 0.30, 0.0), 1.8)
    rim = min(s, 1.3) * lt                                    # how much of the heart's light it catches
    ri = int(min(max((rim - 0.25) / 0.28, -1), len(WARM_RIM) - 1))
    # the life-glow on the soil around the stem
    if lv >= 1:
        rr = np.hypot(xs - bx, (ys - by) * 1.8)
        img = np.where(CAN_SOIL & (rr < 5.5), ci('leaf0'), img)
        if lv >= 2:
            img = np.where(CAN_SOIL & (rr < 2.6), ci('leaf1'), img)
    mm, al, ac, d = thick_curve(HH, WW, curve, STEM_W[0], STEM_W[1])
    st = STEMPALS[lv]
    stem_c = np.where(ac < -0.25, ci(st[1]), ci(st[0]))      # left of travel = toward the heart
    stem_c = np.where(ac > 0.5, ci(st[2]), stem_c)
    if ri >= 0:
        rimw = -0.55 + 0.25 * min(rim, 1.2)                  # the lit band widens as it turns
        stem_c = np.where(ac < rimw, ci(WARM_RIM[ri]), stem_c)
    img = np.where(mm, stem_c, img)
    tip = P[3]
    tips = []
    (aR, dR), (aL, dL) = leaf_angles(s)
    pal = LEAFPALS[lv]
    for (ang, spec, dr, sc) in ((aR, LEAF_R, dR, 1.0), (aL, LEAF_L, dL, 0.92 + 0.12 * min(s, 1.3))):
        if ri >= 0:
            edge = ('leaf1', WARM_RIM[ri], pal[2], WARM_RIM[max(ri - 1, 0)])
        else:
            edge = ('meadow0', 'meadow2', 'leaf2', 'meadow3') if lv == 0 else ('leaf1', 'leaf4', 'leaf3', 'leaf5')
        li, lm, (R, R2), tp = raster_leaf(ang, spec['L'] * sc, spec['W'], dr, 0, pal=pal, edge=edge)
        x0 = int(round(tip[0])) - R
        y0 = int(round(tip[1])) - R
        h, w = lm.shape
        ya, yb = max(y0, 0), min(y0 + h, HH)
        xa, xb = max(x0, 0), min(x0 + w, WW)
        if ya < yb and xa < xb:
            sub = img[ya:yb, xa:xb]
            lmm = lm[ya - y0:yb - y0, xa - x0:xb - x0]
            sub[lmm] = li[ya - y0:yb - y0, xa - x0:xb - x0][lmm]
        tips.append((tip[0] + tp[0], tip[1] + tp[1]))
    return img, P, tips, s


# ================================================================== the arm, animated on twos (states cached)
T_DRIFT0, T_DRIFT1 = 2.75, 3.70


def arm_key(t):
    """held arm state: 0 at rest, then one state per 1/15 s of the drift."""
    if t < T_DRIFT0:
        return 0
    return 1 + int(math.floor((min(t, T_DRIFT1 + 0.2) - T_DRIFT0) * 15.0))


def _arm_state(k):
    t = T_DRIFT0 + (k - 1) / 15.0 if k > 0 else 0.0
    drift = ease.in_out_sine(ease.prog(t, T_DRIFT0, T_DRIFT1)) if k > 0 else 0.0
    am, acol, anrm, hand = _arm(drift)
    P = (xs, ys, np.full((HH, WW), 8.0))
    aI = np.where(am, warm_unit(anrm, P) + bounce_unit(anrm, P), 0.0)
    amoon = np.where(am, moon_level(anrm, 2.4), 0.0)
    return am, acol, aI, amoon


ARM_STATES = {k: _arm_state(k) for k in range(0, arm_key(DURATION + 1.0) + 1)}


# ================================================================== RENDER
def eye_power(t, g):
    """dim amber sensor; breathes with the failing heart."""
    return 0.45 * (0.75 + 0.25 * g / 0.3)


def aperture(t):
    """iris opening: dilated in the dark; a snap refocus as it notices the sprout;
    widens as it looks at its own heart."""
    r = 5.0
    if t >= 1.60:
        r = 3.2 if t < 1.667 else 4.2                       # click: close, then settle on the sprout
    if t >= 2.9:
        r = 4.2 + 1.4 * ease.smoothstep(ease.prog(t, 2.9, 3.4))
    return r


def render(t):
    t = min(max(t, 0.0), DURATION + 0.5)
    g = core_power(t)
    k = g / 0.3
    alb = STATIC_ALB.copy()
    nrmI = STATIC_I.copy()
    moon = STATIC_MOON.copy()
    amb = STATIC_AMB.copy()
    # ---- arm (in front of the torso side, behind nothing)
    am, acol, aI, amoon = ARM_STATES[arm_key(t)]
    alb = np.where(am, acol, alb)
    nrmI = np.where(am, aI, nrmI)
    moon = np.where(am, amoon, moon)
    amb = np.where(am, 0.0, amb)
    # ---- head (held servo pose)
    key, _, sink = head_key(t)
    pd = POSES[key]
    x0, y0, w, h = pd['bbox']
    y0 += sink                                        # the head settles into its collar as it nods
    ya, yb = max(y0, 0), min(y0 + h, HH)
    xa, xb = max(x0, 0), min(x0 + w, WW)
    sl = (slice(ya, yb), slice(xa, xb))
    psl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
    hm = pd['mask'][psl]
    alb[sl][hm] = pd['col'][psl][hm]
    nrmI[sl][hm] = pd['I'][psl][hm]
    moon[sl][hm] = pd['moon'][psl][hm]
    amb[sl][hm] = pd['amb'][psl][hm]
    # ---- light
    warm_lv = level_of(nrmI * k, I0, RSTEP)
    img = RL.apply(alb, amb + VIGNETTE, warm_lv, moon + VIGNETTE * 0.5, band=0.35, amb_band=0.12)
    # background glass wall: untouched night colours (where no actor covers it)
    dyn = am.copy()
    dyn[sl] |= hm
    img = np.where(BG_ONLY & ~dyn, BG, img)
    # ---- emissives: the heart behind the porthole glass
    img = np.where(PORT_GLASS, _heart(g, t), img)
    # ---- the light path: the heart's glow made visible in the dusty air — a soft cone
    #      thrown forward from the porthole, across to the sprout (it only brightens
    #      what is behind it, and flickers, dips and swells with the heart)
    air = (BG_ONLY | FLOOR_M) & ~dyn & ~ACTOR_STATIC
    hz = CONE * k + (TH4 - 0.5) * 0.10
    for thr, cname in ((0.20, 'bark0'), (0.34, 'rust0'), (0.55, 'core0')):
        c = ci(cname)
        img = np.where(air & (hz > thr) & (LUM[img] < LUM[c]), c, img)
    # ---- the lens glass (dim amber sensor, reflections)
    rap = aperture(t)
    refl = []
    if key[0] in 'aw':
        refl.append(((0.75, 0.65), 0.55, 'leaf6', 0.6))               # the sprout, tiny and green
    if t >= 2.55:
        a_ = ease.prog(t, 2.55, 2.95)
        refl.append(((-0.55, 0.85), 0.58, 'core1' if a_ < 0.5 else 'core2', (0.8 + 1.6 * a_, 0.5 + 0.9 * a_),
                     'core4' if a_ > 0.6 else None))
    refl.append(((0.62, -0.78), 0.62, 'white0', 0.5))                  # the glint: the eye is alive
    gi, gm = lens_glass(pd, rap=rap, glow=eye_power(t, g), sheen_dir=(0.55, -0.85),
                        sheen=('night3', 'glass2'), refl=refl)
    sub = img[sl]
    gmm = gm[psl]
    sub[gmm] = gi[psl][gmm]
    # ---- the sprout (drawn in its own light) and its faint life-glow on the soil / rim
    img, sp_pts, tips, s = _draw_sprout(img, t, g)
    # ---- shimmer: green sparks rising from the sprout and drifting toward the heart while it leans
    if T_LEAN0 - 0.05 < t < T_LEAN1 + 1.3:
        a_ = ease.prog(t, T_LEAN0 - 0.05, T_LEAN0 + 0.35) * (1 - ease.prog(t, T_LEAN1 + 0.3, T_LEAN1 + 1.3))
        tipx, tipy = sp_pts[3]
        for i in range(14):
            ph = float(NZ.rand(i, 1, 531))
            per = 0.9 + 0.5 * float(NZ.rand(i, 4, 531))
            cyc = ((t - T_LEAN0) / per + ph) % 1.0
            gen = int(math.floor((t - T_LEAN0) / per + ph))
            jx = (float(NZ.rand(i, 10 + gen, 531)) - 0.5) * 14
            px = tipx + jx - 16 * cyc * (0.3 + 0.7 * s)
            py = tipy + 7 - cyc * 20 + (float(NZ.rand(i, 20 + gen, 531)) - 0.5) * 6
            if NZ.rand(i, 30 + gen, 532) > 0.75 * a_ or cyc > 0.9:
                continue
            xi, yi = int(round(px)), int(round(py))
            if not (1 <= xi < WW - 1 and 1 <= yi < HH - 1):
                continue
            c = ci('leaf8') if cyc < 0.3 else (ci('leaf7') if cyc < 0.6 else ci('leaf5'))
            img[yi, xi] = c
            # the brightest ones twinkle as a tiny cross, briefly
            tw = (t * 12.0 + i * 3.7) % 5.0
            if cyc < 0.45 and tw < 1.0 and i % 3 == 0:
                for dx_, dy_ in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    img[yi + dy_, xi + dx_] = ci('leaf5')
    # ---- dust motes turning in the heart's light: they glint only inside the cone
    for i in range(40):
        sx = float(NZ.rand(i, 1, 541))
        sy = float(NZ.rand(i, 2, 541))
        px = PORT_XY[0] + 4 + (sx * 110 + t * (1.4 + 1.2 * float(NZ.rand(i, 3, 541)))) % 110
        py = PORT_XY[1] - 42 + (sy * 80 - t * (0.8 + 0.6 * float(NZ.rand(i, 4, 541))) + 1.5 * math.sin(t * 0.8 + i)) % 80
        xi, yi = int(px), int(py)
        if not (0 <= xi < WW and 0 <= yi < HH) or not air[yi, xi]:
            continue
        tw = 0.6 + 0.4 * math.sin(t * (3.0 + 2.0 * float(NZ.rand(i, 5, 541))) + i * 1.7)
        e = CONE[yi, xi] * k * tw * 1.9
        if e > 0.95:
            img[yi, xi] = ci('core3')
        elif e > 0.65:
            img[yi, xi] = ci('core2')
        elif e > 0.40:
            img[yi, xi] = ci('rust2')
    return PAL[img]
