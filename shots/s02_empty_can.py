"""
S2 · THE EMPTY CAN  (global 7.0–12.0, DURATION 5.0)       owner: DISCOVERY
Medium, static.  The robot stands left of centre by a dead sunflower; tries to
water it; ONE drop; the earth drinks it; it droops; a dead leaf snaps off and
tumbles away on the wind (right -> left, past the robot).

Local timeline (global = local + 7.0):
  0.0–1.2 idle, looking at the stalk        1.2 'water' (raise)   1.8 tilt
  1.95 a bead swells on the spout           2.2 the drop falls    2.6 hit: splash + wet spot
  2.6–3.2 the earth drinks it               3.4 'droop' -> hold   3.6 weak core flicker
  3.6–4.0 a gust builds, the leaf strains   4.0 the leaf snaps and tumbles away left
"""
import math
import numpy as np
from engine import *
from engine import draw, noise, light, ease
from engine.sprite import blit, anchor_pos, from_ascii, flip
from chars import robot, props
from shots import lib_discovery as L

DURATION = 5.0

HY = 126            # horizon
GY = 144            # ground line (robot feet)
PLANT_X = 171       # dead sunflower stem base x
PLANT_Y = GY + 3    # the plant (and its little bed) stand a step nearer the camera
T_WATER, T_DROP, T_HIT, T_DROOP, T_LEAF = 1.2, 2.2, 2.6, 3.4, 4.0

# ------------------------------------------------------------------ robot placement
_spr_tilt = robot.frame('water', 1.0, can='empty')     # held full tilt (see WATER_WARP)
_fx, _fy = _spr_tilt.anchor('feet')
_sx, _sy = _spr_tilt.anchor('spout')
SPOUT_DX, SPOUT_DY = _sx - _fx, _sy - _fy
DROP_X = PLANT_X - 6                                   # lands on bare earth just left of the tuft
ROBOT_X = int(min(DROP_X - SPOUT_DX, PLANT_X - 17))

# ------------------------------------------------------------------ static world
SKY = L.Sky(HY, seed=21, sun=(62, 30), glow=(163, 126, 95, 42, 0.9))
# as the robot droops the pale sun sinks behind thicker haze (the world dims a shade)
SKY_DIM = L.Sky(HY, seed=21, sun=(62, 30), glow=(163, 126, 95, 42, 0.6),
                sun_rings=((2.0, 5.9), (4.0, 5.2), (7.0, 4.4)))
_yy, _xx = np.mgrid[:HY, :W]
DIM_REGION = np.hypot((_xx - 163) / 95.0, (_yy - 126) / 42.0) < 1.0      # the horizon glow behind the pair
CLOUDS = L.NoiseClouds(SKY, 5, [
    # seed, y0, y1, stretch x, stretch y, threshold, speed px/s, darkness
    (201, -10, 60, 70.0, 6.0, 0.52, 5.2, 0.9),
    (202, 44, 96, 60.0, 3.4, 0.585, 3.5, 0.7),
    (203, 92, 114, 80.0, 2.0, 0.64, 2.2, 0.5),
])
GEO = L.GroundGeo(HY, cx=150, cell_u=0.95, cv=4.0)
GROUND = L.build_ground(GEO, seed=11, near_tone=3.1, far_tone=5.9, sand=0.24)


def gy_at(x):
    return GY + 0.0 * x


# a broken farm fence far behind the pair (low contrast), with a gap behind them
FENCE = [  # x, base y, height, lean(px at top), width, broken-top depth
    (6, 136, 12, -1.0, 2, 0), (28, 135, 11, 0.5, 1, 0), (48, 134.5, 10, -1.5, 1, 2), (67, 134, 10, 1.0, 1, 0),
    (85, 133.5, 9, -0.5, 1, 0), (102, 133, 9, 2.0, 1, 3), (118, 132.5, 8, -1.0, 1, 0),
    (198, 131.5, 7, 1.5, 1, 0), (212, 131, 7, -0.5, 1, 2), (226, 131, 6, 0.5, 1, 0), (240, 130.5, 6, -1.0, 1, 0)]
FENCE_WIRE_BREAKS = {2, 5, 6, 7}


def _build_front():
    """Everything static in front of the sky: far hills, ruins, dunes, ground,
    fence, dead tree, pebbles & bones."""
    img = np.zeros((H, W, 3), np.uint8)
    msk = np.zeros((H, W), bool)
    g = GROUND
    img[HY:] = g['img'][HY:]
    msk[HY:] = True

    def put(lay):
        m = lay[..., 3] > 0
        img[m] = lay[..., :3][m]
        msk[m] = True

    # far flat mesas (barely there)
    mes = layer()
    xs = np.arange(W)
    top = HY - 2 - 6 * np.clip(1 - np.abs(xs - 40) / 70, 0, 1) ** 0.4 - 3 * np.clip(1 - np.abs(xs - 118) / 30, 0, 1) ** 0.5
    top = np.round(top + noise.fbm1(xs * 0.2, 3, 2) * 1.2).astype(int)
    for x in range(W):
        if top[x] < HY:
            mes[top[x]:HY + 1, x] = (*C('dust8'), 255)
    put(mes)
    put(L.skyline(31, HY + 1, 198, 332, 4, 14, 'ash8', 'dust9', 'ash7', win=None,
                  envelope=lambda x: 0.35 + 0.65 * math.exp(-((x - 252) / 55) ** 2)))
    put(L.skyline(32, HY + 2, 204, 320, 3, 19, 'ash8', 'dust9', 'ash7', win='ash7',
                  envelope=lambda x: 0.15 + 0.85 * math.exp(-((x - 270) / 30) ** 2)))
    dl, prof = L.dunes(41, HY, 4, 0.014, 1.5, 'dust7', 'dust8', 'dust6')
    put(dl)
    # dead tree, mid-distance, left  (props that rise above the horizon go on an RGBA layer)
    L.shadow_ellipse(img, 57, 138.5, 5, 0.8, steps=0.8, shadow='plain')
    lay = layer()
    L.dead_tree(lay, 56, 138, 30, seed=5, cols=('dust4', 'dust5', 'dust3'), lean=-0.12, spread=0.6, depth=4)
    # fence: posts + sagging wires (broken in places)
    tops = []
    for i, (x, yb, hh, lean, wd, br) in enumerate(FENCE):
        cols = ('dust5', 'dust4', 'dust4') if yb > 133 else ('dust6', 'dust5', 'dust5')
        L.post(lay, x, int(yb), hh, lean, wd, cols, broken=br, seed=i)
        tops.append((x + int(round(lean * 0.85)), int(yb) - hh + 2 + br))
    for a in range(len(FENCE) - 1):
        if a in FENCE_WIRE_BREAKS or FENCE[a + 1][0] - FENCE[a][0] > 40:
            continue
        (xa, ya), (xb, yb_) = tops[a], tops[a + 1]
        L.sag_wire(lay, xa, ya, xb, yb_, 1.6, 'dust5' if FENCE[a][1] > 133 else 'dust6')
        L.sag_wire(lay, xa, ya + 3, xb, yb_ + 3, 1.2, 'dust5' if FENCE[a][1] > 133 else 'dust6')
    # dangling wire ends at the breaks
    for a in (2, 5):
        xa, ya = tops[a]
        draw.line(lay, xa + 1, ya, xa + 3, ya + 3, 'dust5')
        draw.line(lay, xa + 1, ya + 3, xa + 2, ya + 6, 'dust5')
    put(lay)
    # ground clutter: a few pebbles, bleached bones, twigs (kept away from the pair)
    avoid = [(ROBOT_X - 14, GY - 6, PLANT_X + 16, GY + 5)]
    L.scatter_props(img, GEO, 77, 16, L.PEBBLES, x_range=(0, W), d_range=(8, 54), avoid=avoid)
    L.scatter_props(img, GEO, 78, 2, L.BONES, x_range=(20, 300), d_range=(24, 50), avoid=avoid, sand_tail=False)
    L.scatter_props(img, GEO, 79, 3, L.TWIGS, x_range=(0, W), d_range=(20, 50), avoid=avoid, sand_tail=False)
    # calm, wind-smoothed ground under the pair so the action reads
    L.calm_patch(img, (ROBOT_X + PLANT_X) / 2 + 2, GY + 2.0, 34, 5.4, seed=17, col='dust4', rim='dust5', shade='dust3')
    # the robot's little garden bed: tilled soil ringed with placed pebbles
    bx, by, rx, ry = PLANT_X - 1.0, PLANT_Y + 0.5, 10.0, 2.6
    ys_, xs_ = np.ogrid[:H, :W]
    inside = ((xs_ - bx) / rx) ** 2 + ((ys_ - by) / ry) ** 2 <= 1.0
    img[inside] = C('dust5')
    clod = inside & (noise.rand(xs_ // 2 * 7 + ys_ * 131, 3, 91) < 0.25)
    img[clod] = C('dust4')
    lip = inside & ~np.roll(inside, 1, axis=0)                    # far rim of the bed catches light
    img[lip] = C('dust6')
    for k in range(22):
        ang = 2 * math.pi * k / 22 + 0.13
        px = bx + math.cos(ang) * (rx + 1.2)
        py = by + math.sin(ang) * (ry + 0.9)
        pxi, pyi = int(round(px)), int(round(py))
        near = py > by
        draw.pset(img, pxi, pyi, 'dust7' if near else 'dust6')
        draw.pset(img, pxi + 1, pyi, 'dust6' if near else 'dust5')
        if near:
            draw.pset(img, pxi, pyi + 1, 'dust3')
            draw.pset(img, pxi + 1, pyi + 1, 'dust3')
    L.fg_rock(img, -6, 34, 181, 11, 3)
    L.fg_rock(img, 292, 330, 181, 8, 4, tufts=2)
    return img, msk


FRONT, FRONT_M = _build_front()
VIG = L.make_vignette(0.8, 2.6, cx=160, cy=118, sx=1.05, sy=1.1, bottom=0.45, hy=HY)
FRONT = L.vignette_plates(FRONT, GROUND['plates'], VIG, seed=3)      # baked, plate by plate

# ------------------------------------------------------------------ the dead sunflower
LV = {'g': 'ash5', 'G': 'ash6', 'k': 'dust2', 'K': 'dust2', 'h': 'ash7'}
# limp, crumpled dead leaves hanging from their stalks (attach point 'a' = top)
LEAF1 = ["""
...K
..kK
.kgk
kgGk
kGgk
.kgk
..k.
""", """
...K
.kkK
kgGk
kGgk
kgk.
kk..
""", """
...K
..kK
.kGk
.kgk
kgk.
kk..
"""]
LEAF2 = ["""
K...
Kk..
kgk.
kGgk
kgGk
kgk.
.k..
""", """
K...
Kkk.
kGgk
kgGk
.kgk
..kk
""", """
K...
Kk..
kGk.
kgk.
.kgk
..kk
"""]
FLY = ["""
.kk..
kgGgk
kGggk
.kkk.
""", """
kgGgk
""", """
.kkk.
kggkk
kgkkk
..kk.
""", """
.k
kG
kg
.k
"""]
PLH = {'s': 'dust2', 'S': 'dust3', 'q': 'dust3', 'c': 'dust3', 'C': 'dust5', 'k': 'dust1', 'd': 'dust2',
       'e': 'dust3', 'p': 'dust4', 'P': 'dust5', 'o': 'ochre3', 'O': 'ochre4', 'x': 'dust6'}
# bowed dead sunflower, head in 3/4 facing down-right; dark spiky petal fringe
# (reads against the pale sky); calyx lit top-left; 2px stalk arcing over.
HEAD_R1 = """
...ssS........
..sS..sS......
.sS....sqp....
Ss...pqCcqp...
Ss..pqCCccqPp.
Ss.pPcCcckkkp.
Ss.pkkdddekkp.
Ss.pkddeddddkp
Ss.pkdedddedkp
...pkkdddddkp.
...ppkkddkkpp.
....p.pk.kp.p.
.....p..p..p..
"""
HEAD_R2 = """
...ssS........
..sS..sS......
.sS....sqp....
Ss...pqCcqp...
Ss..pqCCccqp..
Ss.pPcCcckkkPp
Ss.pkkdddekkp.
Ss.pkdeddddkkp
Ss.pkdddeddkp.
...pkkddddkkpp
...ppkkdkkpp..
....p.p.kp.p..
......p..p....
"""
_HEAD_R = from_ascii(HEAD_R1, PLH, {'stem': (0, 8)})
_HEAD2_R = from_ascii(HEAD_R2, PLH, {'stem': (0, 8)})
LEAF_HANG = [from_ascii(a, LV, {'a': (3, 0)}) for a in LEAF1]
LEAF2_HANG = [from_ascii(a, LV, {'a': (0, 0)}) for a in LEAF2]
LEAF_FLY = [from_ascii(a, LV, {'c': (1, 1)}) for a in FLY[:3]] + [from_ascii(FLY[3], LV, {'c': (0, 1)})]
STEM_H = 17


def stem_pts(bend):
    """Centreline (x float, y int) of the straight stem, base -> top, for a wind
    bend in px (+ = top pushed left, toward the robot)."""
    pts = []
    for i in range(STEM_H + 1):
        s = i / STEM_H
        x = PLANT_X - 1.4 * math.sin(math.pi * s) * (1 - 0.3 * s) - bend * s ** 1.8 + 1.2 * s
        pts.append((x, PLANT_Y - i))
    return pts


def wind(t):
    """0..~1.3 wind strength; a gust builds 3.6-4.1 (it tears the leaf off)."""
    base = 0.45 + 0.35 * float(noise.fbm1(np.array([t * 1.3]), 5, 3)[0])
    g = 0.7 * math.exp(-((t - 4.05) / 0.35) ** 2)
    return base + g


WIND_D = L.WindPath(lambda t: 10 + 10 * wind(t))


def plant_pose(t):
    """Discrete stem bend (animated on 12s): wind push + a slow natural sway,
    and the recoil when the leaf snaps off."""
    ts = ease.step_time(t, 12)
    b = 0.6 + 2.4 * wind(ts) + 1.3 * math.sin(ts * 2 * math.pi * 0.55 + 0.4) + 0.5 * math.sin(ts * 2 * math.pi * 1.3)
    if ts >= T_LEAF:
        a = ts - T_LEAF
        b -= 1.8 * math.exp(-a * 4.5) * math.cos(a * 20)       # springs back when released
    return float(np.clip(round(b * 2) / 2, 0, 4.0))


def draw_plant(img, t, with_leaf1=True):
    bend = plant_pose(t)
    pts = stem_pts(bend)
    L.shadow_ellipse(img, PLANT_X + 0.5, PLANT_Y + 0.6, 4.5, 1.1, steps=1.3, shadow='plain')
    # base tuft of dead grass
    for k, (dx, hh, lean) in enumerate(((-3, 2, -1), (-2, 4, -2), (2, 3, -1), (3, 2, -1), (-1, 3, -1), (1, 4, -2), (4, 1, -1))):
        x0 = PLANT_X + dx
        draw.line(img, x0, PLANT_Y, x0 + lean - (1 if bend > 2 else 0), PLANT_Y - hh, 'dust3' if k % 2 else 'dust2')
    # stem: woody & 2 px (lit left / dark right) low, 1 px high
    for i, (x, y) in enumerate(pts):
        xi = int(round(x))
        draw.pset(img, xi, y, 'dust4' if (i % 5) else 'dust3')
        draw.pset(img, xi + 1, y, 'dust2')
    # broken leaf-stalk spurs (it was once leafy)
    for sfrac, side in ((0.28, -1), (0.62, 1), (0.72, -1)):
        x, y = pts[int(STEM_H * sfrac)]
        xi = int(round(x)) + (2 if side > 0 else -1)
        draw.pset(img, xi, y, 'dust2')
        draw.pset(img, xi + side, y - 1, 'dust3')
    tx, ty = pts[-1]
    head = _HEAD_R if (int(t * 12) // 3) % 2 == 0 or bend < 2 else _HEAD2_R
    blit(img, head, int(round(tx)), ty, anchor='stem')
    # leaves: fluttering, faster before the snap
    w = wind(t)
    rate = 9 + 12 * w + (12 if T_LEAF - 0.45 < t < T_LEAF else 0) + (6 if t >= T_LEAF else 0)
    tq = ease.step_time(t, 12)                                     # dead leaves rattle on 12s
    fi = int(abs(noise.fbm1(np.array([tq * rate * 0.3]), 9, 2)[0]) * 7.9) % 3
    fi2 = int(abs(noise.fbm1(np.array([tq * rate * 0.26 + 3]), 12, 2)[0]) * 7.9) % 3
    a1 = pts[int(STEM_H * 0.86)]
    a2 = pts[int(STEM_H * 0.46)]
    a1 = (int(round(a1[0])), a1[1])
    a2 = (int(round(a2[0])), a2[1])
    blit(img, LEAF2_HANG[fi2], a2[0] + 1, a2[1], anchor='a')
    if with_leaf1:
        blit(img, LEAF_HANG[fi], a1[0] - 1, a1[1], anchor='a')
    return pts, a1


# ------------------------------------------------------------------ the drop
def draw_drop(img, t, sx, sy):
    """Bead swells on the spout, falls (stretching), splashes, wet spot fades."""
    gy = PLANT_Y
    if T_DROP - 0.25 <= t < T_DROP:
        k = (t - (T_DROP - 0.25)) / 0.25
        draw.pset(img, sx, sy + 1, 'glass3')
        if k > 0.45:
            draw.pset(img, sx, sy + 2, 'glass4')
    elif T_DROP <= t < T_HIT:
        k = (t - T_DROP) / (T_HIT - T_DROP)
        y = sy + 2 + (gy - 1 - (sy + 2)) * k * k
        x = sx - int(round(1.0 * k * k))
        tail = 1 + int(k * 2.2)
        for j in range(1, tail + 1):
            draw.pset(img, x, int(y) - j, 'glass3' if j < tail else 'glass2')
        draw.pset(img, x, int(y), 'glass4')
        if k < 0.6:
            draw.pset(img, x, int(y) - 1, 'white0')
    return DROP_X


def draw_splash(img, t, x):
    """1-frame crown + 1 frame of falling specks, then a dark wet spot that fades."""
    gy = PLANT_Y
    a = t - T_HIT
    if a < 0:
        return
    if a < 1 / 30:
        for dx, dy, c in ((-2, -1, 'glass3'), (-1, -2, 'glass4'), (1, -2, 'glass4'), (2, -1, 'glass3'), (0, -1, 'white0')):
            draw.pset(img, x + dx, gy + dy, c)
    elif a < 2 / 30:
        for dx, dy, c in ((-3, -1, 'glass3'), (3, 0, 'glass3'), (-1, -3, 'glass2'), (1, -3, 'glass3')):
            draw.pset(img, x + dx, gy + dy, c)
    # wet spot: darker earth that shrinks away as the ground drinks it (hand-keyed)
    WET = [  # (until a, [(dx, dy, steps), ...])
        (0.08, [(-2, 0, 1), (-1, 0, 2), (0, 0, 2), (1, 0, 2), (2, 0, 1), (-1, 1, 1), (0, 1, 1), (1, 1, 1)]),
        (0.18, [(-2, 0, 1), (-1, 0, 2), (0, 0, 2), (1, 0, 1), (0, 1, 1)]),
        (0.27, [(-1, 0, 1), (0, 0, 2), (1, 0, 1)]),
        (0.34, [(-1, 0, 1), (0, 0, 1)]),
        (0.40, [(0, 0, 1)]),
    ]
    for until, px in WET:
        if a < until:
            for dx, dy, st in px:
                xx, yy = x + dx, gy + dy
                sub = np.ascontiguousarray(img[yy:yy + 1, xx:xx + 1])
                img[yy:yy + 1, xx:xx + 1] = L.ramp_darken(sub, st, band=0.01)
            break


LEAF_KEYS = [(0.00, (0, 0)), (0.16, (-11, -8)), (0.36, (-29, -15)), (0.56, (-52, -16)),
             (0.76, (-78, -11)), (1.00, (-110, -5)), (1.40, (-160, 4)), (2.00, (-240, 10))]


def leaf_flight(t, a1):
    """Position + frame of the snapped leaf: lifts, sails over the robot's
    head, then flutters down-left (tau = t - T_LEAF)."""
    tau = t - T_LEAF
    dx, dy = ease.keyframes(tau, LEAF_KEYS, ease=ease.in_out_sine)
    dx += 1.5 * math.sin(tau * 11.0)
    dy += 2.0 * math.sin(tau * 8.0 + 0.8) * min(1.0, tau * 4)
    fr = int(ease.step_time(tau, 12) * 12) % 4
    return a1[0] - 2 + dx, a1[1] + 1 + dy, LEAF_FLY[fr]


# ------------------------------------------------------------------ robot
def core_level(t):
    """0.7 steady; one weak, faltering flicker around 3.6."""
    k = int(round((t - 3.53) * 30))
    seq = [0.35, 0.12, 0.45, 0.2, 0.55, 0.62]
    if 0 <= k < len(seq):
        return seq[k]
    return 0.7


# The c07 close-up ('the last drop') is cut in over local 1.9-3.1.  The 'water'
# anim is time-warped so the cut back at 3.1 still finds the can at full tilt,
# pouring, exactly as c07 leaves it; it then lowers over 3.1-3.4 and hands off
# to 'droop' at 3.4.  (anim times: 0.6 = full tilt, 0.7-1.0 coax, 1.0-1.4 hold
# tilt3, 1.4 tilt2, 1.5 tilt1, 1.6 mid, 1.8 low, 1.9 carry = droop's first pose)
T_CUT_OUT, T_CUT_IN = 1.9, 3.1
WATER_WARP = [(T_WATER, 0.0), (T_CUT_OUT, T_CUT_OUT - T_WATER),   # on screen: real time
              (T_CUT_IN, 1.35),                                    # hidden: coax, then hold the pour
              (T_DROOP, 1.9)]                                      # back on screen: the can lowers


def water_t(t):
    ts, vs = zip(*WATER_WARP)
    return float(np.interp(t, ts, vs))


def robot_sprite(t):
    c = core_level(t)
    if t < T_WATER:
        return robot.frame('idle', t, can='empty', core=c)
    if t < T_DROOP:
        return robot.frame('water', water_t(t), can='empty', core=c)
    return robot.frame('droop', t - T_DROOP, can='empty', core=c)


# ------------------------------------------------------------------ render


def render(t):
    t = max(0.0, t)
    img = np.empty((H, W, 3), np.uint8)
    SKY.render(img, t)
    kd = ease.smoothstep(ease.prog(t, 3.3, 4.4))
    L.sky_blend(img, SKY, SKY_DIM, kd, DIM_REGION)
    L.sun_patch(SKY, img, t, rings=L.lerp_rings(SKY.rings, SKY_DIM.rings, kd), breath=0.11, period=3.6)
    CLOUDS.render(img, t)
    img[FRONT_M] = FRONT[FRONT_M]
    # far away, a thin dust devil wanders along the horizon (in front of the dead city)
    L.dust_devil(img, t, 292 - 7.0 * t, HY + 3, 20, seed=2, amt=0.7)

    # ---- wind-blown sand threads (behind the characters)
    w = wind(t)
    sw = min(1.0, 0.55 + 0.4 * w)
    L.sand_streams(img, t, 71, lambda x: HY + 6, n=8, x_range=(-30, W + 40), speed=(28, 48), length=(8, 22),
                   hgt=(0, 3), colors=('dust7', 'dust8'), amp=0.8, strength=sw)             # far plain
    L.sand_streams(img, t, 7, gy_at, n=13, x_range=(-40, W + 80), speed=(80, 130), length=(20, 64),
                   hgt=(0, 5), colors=('dust7', 'dust6', 'dust8'), amp=2.0, strength=sw)    # at their feet (behind)

    # ---- the plant (behind the robot) + the drop
    pts, a1 = draw_plant(img, t, with_leaf1=t < T_LEAF)
    spr = robot_sprite(t)
    # robot contact shadow
    L.shadow_ellipse(img, ROBOT_X + 0.5, GY + 0.5, 7.5, 1.4, steps=1.4, shadow='plain')
    if T_HIT <= t:
        draw_splash(img, t, DROP_X - 1)          # the drop drifts 1 px downwind as it falls
    blit(img, spr, ROBOT_X, GY, anchor='feet')
    sx, sy = anchor_pos(spr, ROBOT_X, GY, 'feet', 'spout')
    draw_drop(img, t, sx, sy)

    # ---- core glow (the only warm light in the world)
    cx, cy = anchor_pos(spr, ROBOT_X, GY, 'feet', 'core')
    c = core_level(t)
    L.light_box(img, cx + 0.5, cy + 0.5, 4.0 + 3.0 * c, 1.6 * c, mode='warm', band=0.5)

    # ---- a glass shard in the dirt glints now and then
    L.shard_glint(img, 262, 136, t, (0.55, 1.45, 3.95, 4.85))

    # ---- the snapped leaf
    if t >= T_LEAF:
        x, y, fr = leaf_flight(t, a1)
        blit(img, fr, x, y, anchor='c')
        if t - T_LEAF < 0.25:     # a crumb falls from the break
            a = t - T_LEAF
            draw.pset(img, a1[0] - 1, a1[1] + int(40 * a * a) + 1, 'dust3')

    # ---- atmosphere (near sand threads skim the foreground, below the characters)
    w = wind(t)
    L.sand_streams(img, t, 72, lambda x: 158 + 0.06 * (x - 160), n=7, x_range=(-60, W + 120), speed=(150, 230),
                   length=(30, 90), hgt=(0, 4), colors=('dust6', 'dust7'), amp=2.6, strength=min(1.0, 0.5 + 0.45 * w))
    # dry grass in the foreground, bending in the wind
    for (gx, gy_, sd) in ((14, 172, 1), (22, 175, 2), (302, 174, 3)):
        L.grass_tuft(img, gx, gy_, t, sd, wind=min(1.2, w))
    L.dust_motes(img, t, 3, n=26, area=(0, 30, W, 140), disp=WIND_D(t))
    return img
