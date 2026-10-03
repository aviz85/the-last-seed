"""
S3 · DISCOVERY  (module 8.0 s; on screen local 0–2.5 = global 12.0–14.5 and
local 5.5–8.0 = global 17.5–20.0; 2.5–5.5 is covered by c01/c02)   owner: DISCOVERY

Part A  0.0–1.8 the robot trudges right into the wind (empty can), leaving
        boot prints; 0.4–1.2 a gust drags a veil of dust across the frame
        right->left and blows away a little sand drift ahead (x~230);
        1.4 a GREEN glint twinkles there, then the tiny sprout stays visible;
        1.8 the robot halts: 'startle' (head snaps toward it); hold.
hidden  2.5–5.5 it hurries over (prints) and kneels on the LEFT of the sprout.
Part B  5.5 kneeling; 5.9 'scoop' — ground sprout until 5.9+SCOOP_LIFT_T, then a
        dug hole; the anim carries clump + can; 7.0 into the can; 7.3 'rise_hold'
        (hugging the can); 7.5 happy beat: tiny bounce, pupil glint, antenna blink.
The sprout's faint green glow follows it everywhere.
"""
import math
import numpy as np
from engine import *
from engine import draw, noise, light, ease
from engine.sprite import blit, anchor_pos, from_ascii
from chars import robot, props
from shots import lib_discovery as L

DURATION = 8.0

HY = 124            # horizon
GY = 146            # ground line (robot feet, sprout base)
SPROUT_X = 230

# ------------------------------------------------------------------ timing
T_GUST0, T_GUST1 = 0.4, 1.2
T_GLINT = 1.4
T_STOP = 1.8
T_SCOOP = 5.9
T_LIFT = T_SCOOP + robot.SCOOP_LIFT_T
T_INCAN = T_SCOOP + robot.SCOOP_INTO_CAN_T
T_RISE = 7.1        # rise_hold's own little hop (t=0.4) lands on the 7.5 happy beep
T_HAPPY = 7.5

WS = float(getattr(robot, 'WALK_SPEED', 9.0))       # px/s ground speed of 'walk'
X_HALT = 134.0
WALK_PH = 0.1                                   # walk phase: the halt lands on a near-passing frame
X0 = X_HALT - (robot.travel('walk', T_STOP + WALK_PH) - robot.travel('walk', WALK_PH))
KNEEL_X = SPROUT_X - robot.SCOOP_SPROUT_OFFSET[0]
KNEEL_Y = GY - robot.SCOOP_SPROUT_OFFSET[1]
SPROUT_Y = GY
# hidden stretch: startle hold -> hurried walk -> kneel
T_HURRY0 = 2.75
KNEEL_DUR = robot.duration('kneel')
T_KNEEL0 = 4.85
T_HURRY1 = T_KNEEL0

# ------------------------------------------------------------------ static world
SKY = L.Sky(HY, seed=33, sun=(74, 28), stops=((0, 2.4), (44, 3.1), (88, 4.0), (113, 4.7), (124, 5.1)),
            glow=(214, 124, 105, 44, 0.7))
# from the glint on, the haze low above the sprout brightens a little (hope)
SKY_HOPE = L.Sky(HY, seed=33, sun=(74, 28), stops=((0, 2.4), (44, 3.1), (88, 4.0), (113, 4.7), (124, 5.1)),
                 glow=(222, 124, 110, 48, 1.35))
_yy, _xx = np.mgrid[:HY, :W]
HOPE_REGION = np.hypot((_xx - 222) / 110.0, (_yy - 124) / 48.0) < 1.0
CLOUDS = L.NoiseClouds(SKY, 6, [
    (301, -10, 56, 75.0, 6.5, 0.53, 5.6, 0.9),
    (302, 40, 94, 60.0, 3.4, 0.585, 3.7, 0.7),
    (303, 90, 112, 80.0, 2.0, 0.64, 2.4, 0.5),
])
GEO = L.GroundGeo(HY, cx=200, cell_u=0.95, cv=4.0)
GROUND = L.build_ground(GEO, seed=23, near_tone=3.1, far_tone=5.9, sand=0.22)

_POLE = {'k': 'dust2', 'b': 'dust3', 'l': 'dust4', 'i': 'dust6'}


def _utility_pole(lay, x, yb, h, lean, cols=('dust4', 'dust3', 'dust2')):
    lit, body, dark = cols
    for i in range(h):
        xx = x + int(round(lean * i / h))
        draw.pset(lay, xx, yb - i, lit)
        draw.pset(lay, xx + 1, yb - i, dark if i % 5 else body)
    tx = x + int(round(lean))
    ty = yb - h
    # cross-arm (tilted) with insulators
    draw.line(lay, tx - 5, ty + 4, tx + 6, ty + 3, body)
    for dx, dy in ((-5, 3), (-1, 3), (3, 2), (6, 2)):
        draw.pset(lay, tx + dx, ty + dy, 'dust6')
    return tx, ty


WM_HUB = []
CABLE = []


def _build_front():
    img = np.zeros((H, W, 3), np.uint8)
    msk = np.zeros((H, W), bool)
    img[HY:] = GROUND['img'][HY:]
    msk[HY:] = True

    def put(lay):
        m = lay[..., 3] > 0
        img[m] = lay[..., :3][m]
        msk[m] = True

    # far city ruins, left
    put(L.skyline(61, HY + 1, -10, 78, 4, 14, 'ash8', 'dust9', 'ash7', win=None,
                  envelope=lambda x: 0.3 + 0.7 * math.exp(-((x - 30) / 40) ** 2)))
    put(L.skyline(62, HY + 2, -4, 62, 3, 17, 'ash8', 'dust9', 'ash7', win='ash7',
                  envelope=lambda x: 0.2 + 0.8 * math.exp(-((x - 26) / 26) ** 2)))
    # a far low ridge, right
    rid = layer()
    xs = np.arange(W)
    top = HY - 1 - 5 * np.clip(1 - np.abs(xs - 285) / 60, 0, 1) ** 0.6 - 2 * np.clip(1 - np.abs(xs - 205) / 40, 0, 1)
    top = np.round(top + noise.fbm1(xs * 0.15, 8, 2) * 1.2).astype(int)
    for x in range(W):
        if top[x] < HY:
            rid[top[x]:HY + 1, x] = (*C('dust8'), 255)
    put(rid)
    dl, prof = L.dunes(71, HY, 4, 0.013, 1.5, 'dust7', 'dust8', 'dust6', x_off=300)
    put(dl)
    lay = layer()
    # the old farm windmill (its rotor is animated), mid-distance left
    L.shadow_ellipse(img, 84, 136.5, 7, 0.9, steps=0.9, shadow='plain')
    WM_HUB.append(L.windmill_tower(lay, 84, 136, 30, cols=('dust4', 'dust5', 'dust3')))
    # a small dead shrub by its foot
    L.dead_tree(lay, 97, 136, 11, seed=12, cols=('dust4', 'dust5', 'dust3'), lean=-0.25, spread=0.7, depth=3, trunk_w=1)
    # line of old utility poles receding right (cables snapped)
    poles = [(300, 142, 30, 2.5), (268, 134, 16, -1.0), (250, 130, 9, 0.5), (240, 128, 6, 0.0)]
    tops = []
    for i, (x, yb, h, lean) in enumerate(poles):
        d = yb - HY
        cols = ('dust4', 'dust3', 'dust2') if d > 14 else ('dust5', 'dust4', 'dust4') if d > 7 else ('dust6', 'dust5', 'dust5')
        L.shadow_ellipse(img, x + 1, yb + 0.5, 2 + d / 8, 0.7, steps=0.9, shadow='plain')
        tops.append(_utility_pole(lay, x, yb, h, lean, cols))
    (ax, ay), (bx, by) = tops[0], tops[1]
    L.sag_wire(lay, ax - 5, ay + 3, bx - 3, by + 3, 4.0, 'dust3')
    (cx_, cy_) = tops[2]
    L.sag_wire(lay, bx + 3, by + 3, cx_ + 2, cy_ + 2, 1.5, 'dust5')
    CABLE.append((ax + 6, ay + 3))            # snapped cable end: animated in render()
    put(lay)
    avoid = [(X0 - 20, GY - 5, SPROUT_X + 14, GY + 6)]
    L.scatter_props(img, GEO, 87, 16, L.PEBBLES, x_range=(0, W), d_range=(8, 54), avoid=avoid)
    L.scatter_props(img, GEO, 88, 2, L.BONES, x_range=(10, 310), d_range=(26, 52), avoid=avoid, sand_tail=False)
    L.scatter_props(img, GEO, 89, 3, L.TWIGS, x_range=(0, W), d_range=(22, 50), avoid=avoid, sand_tail=False)
    # calm sand sheets where the action happens (so it reads), a crack for the sprout
    L.calm_patch(img, 124, GY + 1.5, 40, 4.6, seed=31, col='dust4', rim='dust5', shade='dust3')
    L.calm_patch(img, 222, GY + 1.5, 27, 4.8, seed=32, col='dust4', rim='dust5', shade='dust3')
    for (xa, ya, xb, yb) in ((222, GY + 3, 227, GY + 1), (227, GY + 1, 233, GY), (233, GY, 240, GY - 1)):
        draw.line(img, xa, ya, xb, yb, 'dust1')
        draw.line(img, xa, ya + 1, xb, yb + 1, 'dust5')
    L.fg_rock(img, -8, 26, 181, 9, 13, tufts=2)
    L.fg_rock(img, 268, 332, 181, 12, 14)
    return img, msk


FRONT, FRONT_M = _build_front()
VIG = L.make_vignette(0.8, 2.6, cx=185, cy=118, sx=1.05, sy=1.1, bottom=0.45, hy=HY)
FRONT = L.vignette_plates(FRONT, GROUND['plates'], VIG, seed=3)      # baked, plate by plate

# ------------------------------------------------------------------ the sand drift over the sprout
_MND = {'a': 'dust6', 'b': 'dust7', 'c': 'dust8', 'd': 'dust5'}
MOUND = [from_ascii(a, _MND, {'b': b}) for a, b in (
    ("""
...ccc....
.cbbbbba..
cbbbbbbbad
""", (5, 2)),
    ("""
...cc....
.cbbbba..
cbbbbbad.
""", (5, 2)),
    ("""
..cbb..
cbbbbad
""", (4, 1)),
    ("""
.cba.
cbbad
""", (3, 1)),
    ("""
cbad
""", (2, 0)),
)]
T_MOUND0 = 0.66     # the veil reaches x=230 about here


def wind(t):
    """Wind strength: steady breeze, the gust (0.4-1.2), then a hush around the
    glint (the world holds its breath), easing back to a breeze by part B."""
    base = 0.5 + 0.3 * float(noise.fbm1(np.array([t * 1.2]), 17, 3)[0])
    g = 0.9 * ease.smoothstep(ease.prog(t, 0.35, 0.6)) * (1 - ease.smoothstep(ease.prog(t, 0.9, 1.35)))
    lull = 0.55 * ease.smoothstep(ease.prog(t, 1.1, 1.45)) * (1 - ease.smoothstep(ease.prog(t, 2.6, 5.0)))
    return max(0.18, base + g - lull)


WIND_D = L.WindPath(lambda t: 6 + 12 * wind(t) + 22 * max(0.0, wind(t) - 0.6))
VEIL = L.DustVeil(0.38, 1.22, GY, seed=5, n=12, height=54, spread=150.0, gain=1.45)


# ------------------------------------------------------------------ robot choreography
def robot_state(t):
    """-> (anim, anim_t, x, y, can)"""
    if t < T_STOP:
        x = X0 + robot.travel('walk', t + WALK_PH) - robot.travel('walk', WALK_PH)
        return 'walk', t + WALK_PH, x, GY, 'empty'
    if t < T_HURRY0:
        return 'startle', t - T_STOP, X_HALT, GY, 'empty'
    if t < T_HURRY1:
        k = (t - T_HURRY0) / (T_HURRY1 - T_HURRY0)
        x = X_HALT + (KNEEL_X - X_HALT) * ease.in_out_sine(k)
        return 'walk', (x - X_HALT) / WS, x, GY, 'empty'
    if t < T_SCOOP:
        return 'kneel', t - T_KNEEL0, KNEEL_X, KNEEL_Y, 'empty'
    if t < T_RISE:
        return 'scoop', t - T_SCOOP, KNEEL_X, KNEEL_Y, 'empty'
    return 'rise_hold', t - T_RISE, KNEEL_X, KNEEL_Y, 'sprout'


def core_level(t):
    """0.7 steady; the heart swells warmer for a moment at the happy beat."""
    a = t - (T_HAPPY - 0.05)
    if 0 <= a < 0.6:
        return 0.7 + 0.28 * math.sin(math.pi * min(1.0, a / 0.6)) ** 0.7
    return 0.7


def bulb_pos(spr, x, y):
    """Screen position of the antenna bulb: the topmost core-coloured pixel."""
    img = spr.img
    a = img[..., 3] > 0
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return None
    rgb = img[ys, xs, :3].astype(np.int32)
    key = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    core = np.isin(key, _CORE_KEYS)
    if not core.any():
        return None
    i = np.argmin(np.where(core, ys, 10 ** 6))
    fx, fy = spr.anchor('feet')
    return int(round(x)) - fx + int(xs[i]), int(round(y)) - fy + int(ys[i])


_CORE_KEYS = np.array([(r << 16) | (g << 8) | b for (r, g, b) in (C('core%d' % i) for i in range(2, 8))], np.int32)


def sprout_visible(t):
    return t >= T_GLINT and t < T_LIFT


def glint_size(t):
    k = int(math.floor((t - T_GLINT) * 30))
    seq = [2, 4, 3, 1]                   # a 4-frame twinkle
    return seq[k] if 0 <= k < len(seq) else 0


def sprout_light_pos(t, spr, x, y):
    if t < T_LIFT:
        return SPROUT_X, SPROUT_Y - 3
    if t < T_INCAN:
        return anchor_pos(spr, x, y, 'feet', 'hand')
    mx, my = anchor_pos(spr, x, y, 'feet', 'can_mouth')
    return mx, my - 2


# ------------------------------------------------------------------ render
def render(t):
    t = max(0.0, t)
    img = np.empty((H, W, 3), np.uint8)
    SKY.render(img, t)
    L.sky_blend(img, SKY, SKY_HOPE, 0.9 * ease.smoothstep(ease.prog(t, 1.35, 2.6)), HOPE_REGION)
    L.sun_patch(SKY, img, t, breath=0.11, period=3.9)
    CLOUDS.render(img, t)
    img[FRONT_M] = FRONT[FRONT_M]
    w = wind(t)
    # the snapped cable swings from the near pole (animated on 12s)
    tq = ease.step_time(t, 12)
    cx0, cy0 = CABLE[0]
    sway = -1.2 * min(1.3, w) - 0.9 * math.sin(tq * 4.3) * (0.4 + 0.6 * min(1.0, w))
    pts = [(cx0 + sway * (k / 6.0) ** 1.4 + 0.6 * math.sin(k * 0.9), cy0 + 2 * k) for k in range(7)]
    for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
        draw.line(img, xa, ya, xb, yb, 'dust3')
    # far behind (part B), a tumbleweed bowls across the plain on the wind
    L.tumbleweed(img, t, 5.2, 338, HY + 10, 30.0, seed=3)
    # the windmill's tail vane flashes in the sun now and then (part B only)
    L.shard_glint(img, WM_HUB[0][0] - 11, WM_HUB[0][1] - 2, t, (6.35, 7.85)) if t > 5.0 else None
    # the windmill's broken rotor creaks round in the wind (faster in the gust), on 12s
    ts = ease.step_time(t, 12)
    ang = 0.35 * ts + 5.5 * max(0.0, min(ts, 1.6) - 0.45) ** 1.6 * (1.0 if ts < 1.6 else 1.0) + 0.9
    hx, hy = WM_HUB[0]
    L.windmill_wheel(img, hx, hy, 7, ang, cols=('dust4', 'dust5', 'dust3'))

    anim, at, rx, ry, can = robot_state(t)
    cl = core_level(t)
    spr = robot.frame(anim, at, can=can, core=cl)
    by = 0                                   # (rise_hold's own hop lands on 7.5)


    # the sand drift hiding the sprout, blown away by the gust
    if t < T_MOUND0 + 0.55:
        k = int(max(0.0, t - T_MOUND0) / 0.11)
        if k < len(MOUND):
            blit(img, MOUND[k], SPROUT_X, SPROUT_Y, anchor='b')
    if T_MOUND0 <= t < T_MOUND0 + 0.9:
        L.dirt_bits(img, t, T_MOUND0, 14, 41, (SPROUT_X, SPROUT_Y - 2), speed=(40, 95),
                    spread=(math.pi - 0.5, math.pi + 0.15), life=(0.3, 0.8), gravity=30.0,
                    cols=('dust7', 'dust8', 'dust6'))

    # the sprout (ground) / the dug hole
    if T_MOUND0 + 0.45 <= t < T_GLINT:
        # uncovered but still dust-coated (grey on grey) until the glint
        blit(img, L.dusty(props.sprout('ground', t, wind=0.0)), SPROUT_X, SPROUT_Y, anchor='base')
    if sprout_visible(t):
        sw = 0.0 if t >= T_SCOOP + 0.45 else min(1.0, 0.3 + 0.5 * w)
        s = props.sprout('ground', t, wind=sw)
        blit(img, s, SPROUT_X, SPROUT_Y, anchor='base')
    if t >= T_LIFT:
        # the dug hole (same pixels the scoop anim draws) + the soil heaped beside it
        for dx, c in ((-2, 'dust3'), (-1, 'dust2'), (0, 'ink1'), (1, 'dust2'), (2, 'dust3')):
            draw.pset(img, SPROUT_X + dx, SPROUT_Y, c)
        for dx, dy, c in ((3, 0, 'dust4'), (4, 0, 'dust3'), (3, -1, 'dust5'), (-3, 0, 'dust4'), (-4, 0, 'dust3')):
            draw.pset(img, SPROUT_X + dx, SPROUT_Y + dy, c)

    # wind-blown sand threads (far plain, and at their feet behind the characters)
    sst = min(1.0, 0.2 + 0.8 * w)
    L.sand_streams(img, t, 81, lambda x: HY + 6, n=int(3 + 6 * min(1.0, w)), x_range=(-30, W + 40),
                   speed=(28, 50), length=(8, 22), hgt=(0, 3), colors=('dust7', 'dust8'), amp=0.8, strength=sst)
    L.sand_streams(img, t, 11, lambda x: GY - 1, n=int(4 + 10 * min(1.0, w) + 8 * min(1, max(0, w - 0.8))),
                   x_range=(-40, W + 80), speed=(90, 170), length=(18, 64), hgt=(0, 6),
                   colors=('dust7', 'dust6', 'dust8'), amp=2.0, strength=sst)

    # robot (+ contact shadow)
    L.shadow_ellipse(img, rx + 0.5, ry + 0.5, 7.5, 1.4, steps=1.4, shadow='plain')
    blit(img, spr, rx, ry + by, anchor='feet')

    # digging spray + crumbs from the lifted clump
    if T_SCOOP <= t < T_LIFT + 0.3:
        for j, t0 in enumerate((T_SCOOP + 0.1, T_SCOOP + 0.32, T_SCOOP + 0.52)):
            L.dirt_bits(img, t, t0, 5, 60 + j, (SPROUT_X - 1, SPROUT_Y - 1), speed=(18, 34),
                        spread=(-2.7, -0.4), life=(0.25, 0.45), gravity=170.0)
    if T_LIFT <= t < T_INCAN + 0.2:
        hx, hy_ = anchor_pos(spr, rx, ry + by, 'feet', 'hand')
        L.dirt_bits(img, t, T_LIFT + 0.05, 4, 70, (hx, hy_ + 1), speed=(2, 8), spread=(1.2, 1.9),
                    life=(0.3, 0.5), gravity=150.0)

    # near sand threads skim the foreground, below the characters
    L.sand_streams(img, t, 82, lambda x: 160 + 0.05 * (x - 160), n=int(2 + 6 * min(1.0, w)), x_range=(-60, W + 120),
                   speed=(150, 240), length=(30, 90), hgt=(0, 4), colors=('dust6', 'dust7'), amp=2.6,
                   strength=min(1.0, 0.15 + 0.8 * w))

    # atmosphere (drawn before the lights, so drifting motes catch the glows)
    # dry grass in the foreground, bending in the wind
    for (gx, gy_, sd) in ((10, 174, 4), (284, 170, 5), (296, 173, 6)):
        L.grass_tuft(img, gx, gy_, t, sd, wind=min(1.3, w))
    L.dust_motes(img, t, 13, n=26, area=(0, 24, W, 140), disp=WIND_D(t))

    # lights: the robot's core, the sprout's glow, the glint
    cx, cy = anchor_pos(spr, rx, ry + by, 'feet', 'core')
    L.light_box(img, cx + 0.5, cy + 0.5, 4.0 + 3.5 * cl, 1.85 * cl, mode='warm', band=0.5)
    if t >= T_GLINT:
        gx, gy = sprout_light_pos(t, spr, rx, ry + by)
        a = t - T_GLINT
        swell = min(1.0, a / 0.07)                                  # rises with the twinkle
        pulse = 1.0 + 0.55 * math.exp(-max(0.0, a - 0.07) / 0.22) * swell
        breathe = 0.08 * math.sin(t * 2.4)
        L.green_glow(img, gx, gy, (0.95 + breathe) * pulse * (0.35 + 0.65 * swell), radius=7.5 + 2.5 * (pulse - 1))
    gs = glint_size(t)
    if gs:
        L.sparkle(img, SPROUT_X, SPROUT_Y - 4, gs)

    # the green point reflected in its lens (bridges into the c01 eye close-up)
    if T_STOP + 0.1 <= t < T_HURRY0 + 0.3:
        ex, ey = anchor_pos(spr, rx, ry + by, 'feet', 'eye')
        px, py = ex - 1, ey + 1
        if 0 <= px < W and 0 <= py < H:
            r_, g_, b_ = (int(v) for v in img[py, px])
            if r_ + g_ + b_ < 150:                                   # only on the dark glass
                draw.pset(img, px, py, 'leaf6')

    # happy beat: pupil glint + antenna bulb blink
    if T_HAPPY <= t < T_HAPPY + 0.3:
        ex, ey = anchor_pos(spr, rx, ry + by, 'feet', 'eye')
        k = int((t - T_HAPPY) * 30)
        draw.pset(img, ex, ey, 'core7')
        draw.pset(img, ex + 1, ey, 'core6')
        if k in (1, 2, 3, 4):
            L.sparkle(img, ex - 2, ey - 2, (1, 2, 1, 1)[k - 1], cols=('white1', 'white0', 'core7', 'core6'))
        bp = bulb_pos(spr, rx, ry + by)
        if bp is not None:
            draw.pset(img, bp[0], bp[1], 'core7')
            L.light_box(img, bp[0] + 0.5, bp[1] + 0.5, 3.5, 1.5, mode='warm', band=0.5)

    # the gust veil; the amber heart shines through the dust
    cov = VEIL.coverage(t)
    if cov is not None:
        img = L.apply_fog(img, np.clip(cov * VEIL.gain, 0, 0.92), fog='dust7', steps=4, band=0.4)
        cxi, cyi = int(cx), int(cy)
        k = float(cov[min(max(cyi, 0), H - 1), min(max(cxi, 0), W - 1)])
        if k > 0.08:
            L.light_box(img, cx + 0.5, cy + 0.5, 5.0 + 3 * k, 1.2 + 1.5 * k, mode='warm', band=0.5)
            draw.pset(img, cxi, cyi, 'core5')
        VEIL.streaks(img, t)
    return img
