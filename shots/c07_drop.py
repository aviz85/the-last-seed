"""
c07 · THE LAST DROP  (global 8.9–10.1, local 0–1.2)   — insert cut into S2

Ground-level close-up (c02's lens and ground construction) at S2's place:
cracked mud plates; the dead sunflower's backlit stalk on the right third,
its limp grey leaf hanging toward the fall line; S2's pale dead city far
right; a hazy warm-grey sky lit from the upper left (off-screen sun). The
watering can's big rusty, dented spout enters from the upper left, tilted
down to the right. Everything is palette-index art; the only water colours
are ash / glass / white.

Local timeline (frames at 30 fps; global = local + 8.9):
  f0–f8   (0.00–0.27) the robot coaxes: S2's arm jiggles (f0, f6 — the can
          lifts a hair) each followed by a flick down that pushes water out;
          the bead swells under the lip (a star-glint as it catches the sun,
          f5–f7), and at the big flick (f8) it sags into a neck
  f9      (0.30, cue drop_fall 9.2) the spout snaps back up and the heavy bead
          lets go: the thread springs back into a wet bump on the lip, a tiny
          satellite droplet follows two frames behind
  f9–f20  it falls, accelerating (y ~ k^2), round, then two stretch frames with
          a trail; a soft contact shadow gathers under it (f18–f20)
  f21     (0.70, cue drop_hit 9.6) it slaps flat with a flash; f22 a see-through
          crown with seven points and beads; f23 the crown slumps into a broken
          ring; four droplets arc out and leave wet specks; a puff of dust
  f24–f28 one more little shake (hope): the bump on the lip swells a hair
          (f26–f28) ... and sinks back (f29). Nothing comes.
  f24–f33 the little pool soaks away; the dark stain shrinks, gone by 1.10
  0.74–   one mote of dust drifts down past the empty mouth, where the bead hung
  f32–f35 (1.07–1.17) the arm gives up: the spout sags, heavy and reluctant
          (S2 has the can lowered at 10.1)
"""
import math
import numpy as np
from engine import *
from engine import noise as NZ
from engine.ease import keyframes
from shots import lib_drop as D
from shots.lib_drop import ci, PAL, put, blit_idx, shift

DURATION = 1.2
FPS_ = 30

T_DETACH, T_HIT = 0.30, 0.70

# ------------------------------------------------------------------ static world
SKY = D.build_sky()
G = D.build_ground(79, 0.4)
BG = np.where(D.ys > D.Y_H + 0.5, G['img'], SKY).astype(np.uint8)
STALK_BASE = (208.0, 147.0)
BG = D.pebbles(BG, G['plate'], seed=91, n=60, avoid=((118, 146, 172, 166), (190, 138, 232, 152)))
STALK_M = D.build_stalk(BG, STALK_BASE, node_y=66.0)
LEAVES = {k: D.hanging_leaf(math.radians(k)) for k in range(-10, 11, 2)}
NODE = (int(round(D.stalk_cx(66.0, STALK_BASE) - 5)), 67)

# ------------------------------------------------------------------ the spout: rotation variants
ANG0 = 32.0
TIP0 = np.array([152.0, 54.0])
PIVOT_L = 230.0                                   # the robot's grip, far up-left off-screen
_d0 = np.array([math.cos(math.radians(ANG0)), math.sin(math.radians(ANG0))])
PIVOT = TIP0 - PIVOT_L * _d0
DANG = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
SPOUTS = {}
for _da in DANG:
    _a = math.radians(ANG0 + _da)
    _tip = PIVOT + PIVOT_L * np.array([math.cos(_a), math.sin(_a)])
    SPOUTS[_da] = D.raster_spout(tuple(_tip), _a, r_tip=10.5, grow=0.045, flare=2.6, band_sd=(86.0, 94.0))
    SPOUTS[_da]['low'] = SPOUTS[_da]['geom'].lip_low()

# the drop leaves from where the bead hung at the big flick (f8), and lands below
LOW0 = SPOUTS[1.5]['low']
LAND_Y = 156


# ------------------------------------------------------------------ timing helpers
def frame_of(t):
    return int(math.floor(t * FPS_ + 1e-6))


def age(t, t0):
    """seconds since t0, measured on the 30 fps frame grid (robust to float
    noise like t = 0.6999999999999993 for the frame at 0.7)."""
    return (frame_of(t) - frame_of(t0)) / FPS_


# The robot's hand, keyed per frame (30 fps): (tilt offset in 0.5 deg steps, + = tip
# down;  dy px).  f0 and f6 are S2's arm jiggles (the can lifts a hair), each followed by
# a coaxing flick down that pushes water into the bead; on the rebound after the big
# flick (f8 -> f9) the heavy bead lets go.  After the hit: one more little shake (f24-28),
# a held stillness, then the reluctant sag as the arm starts to lower the can (S2).
POSE_KEYS = {
    0: (-0.5, 0), 1: (0.5, 0), 2: (1.0, 0), 3: (0.5, 0), 4: (-0.5, 0), 5: (0.0, 0),
    6: (-0.5, 0), 7: (1.0, 0), 8: (1.5, 0), 9: (-0.5, 0), 10: (0.5, 0), 11: (0.0, 0),
    24: (0.5, 0), 25: (1.5, 0), 26: (-0.5, 0), 27: (1.0, 0), 28: (0.0, 0),
    32: (0.0, 1), 33: (-0.5, 1), 34: (-0.5, 2), 35: (-1.0, 3), 36: (-1.0, 4), 37: (-1.0, 5),
}
T_LAST = 24 / 30.0                                # the last, hopeful shake


def spout_pose(t):
    """(tilt offset (deg), dx, dy) for the frame at time t (holds between keys)."""
    f = frame_of(t)
    da, dy = POSE_KEYS[max(k_ for k_ in POSE_KEYS if k_ <= f)]
    return da, 0, dy


# the bead's growth (r, height, contact half-width) and the moment it lets go
BEAD_ST = [D.bead_sprite(1.4, 2.0, 1.5), D.bead_sprite(2.2, 3.0, 2.0), D.bead_sprite(3.0, 5.0, 2.4),
           D.bead_sprite(3.8, 7.0, 2.6), D.bead_sprite(4.6, 9.0, 2.6), D.bead_sprite(5.3, 11.0, 2.6),
           D.bead_sprite(5.9, 13.0, 2.5), D.bead_sprite(6.2, 15.0, 2.4),
           D.bead_sprite(6.1, 19.0, 2.2, neck=(4, 1.5))]
BEAD_SEQ = [1, 1, 2, 3, 4, 4, 5, 6, 8]            # frames 0..8: each flick pushes water in; f8 = the neck
# what stays on the lip: the snapped thread, the bump springing back, over, settled
REM_ST = [D.bead_sprite(1.5, 5.0, 1.5, neck=(2, 1.5)), D.bead_sprite(2.4, 4.0, 2.2),
          D.bead_sprite(2.6, 2.0, 2.6), D.bead_sprite(2.3, 3.0, 2.2)]
DROP_R = 6.0
DROP_ROUND = D.drop_sprite(DROP_R, 1.0)
DROP_ST1 = D.drop_sprite(DROP_R, 1.22)
DROP_ST2 = D.drop_sprite(DROP_R * 0.95, 1.42)
SAT = D.drop_sprite(1.4, 1.0)                    # the satellite droplet from the snapped thread


def bead_stage(t):
    return BEAD_SEQ[min(frame_of(t), len(BEAD_SEQ) - 1)]


def remnant(t):
    """what hangs on the lip after the drop: the snapped thread, then a small wet
    bump; at the last shake it swells a hair (hope) ... and shrinks back."""
    f = frame_of(t) - frame_of(T_DETACH)
    if f < 3:
        return REM_ST[f]
    g = frame_of(t) - frame_of(T_LAST)
    if 2 <= g <= 4:
        return BEAD_ST[2] if g == 3 else BEAD_ST[1]     # it gathers ...
    if g == 5:
        return REM_ST[2]                                # ... and sinks back: nothing comes
    return REM_ST[3]


# ------------------------------------------------------------------ the fall
def drop_state(t, delay=0):
    """(sprite, x of its centre column, y of its bottom row) of the falling drop, or None.
    delay > 0: the satellite droplet, born from the snapped thread `delay` frames later."""
    k = frame_of(t) - frame_of(T_DETACH) - delay
    n = frame_of(T_HIT) - frame_of(T_DETACH)      # 12
    if k < 0:
        return None
    y0 = LOW0[1] + 1 + BEAD_ST[8][1].shape[0]      # its bottom as it lets go
    if delay:
        y0 = LOW0[1] + 8.0
        q = k / n
        y = y0 + (LAND_Y - 12 - y0) * q * q * 1.05
        if y >= LAND_Y - 12:
            return None
        return SAT, LOW0[0] - 2.0 * q * q, y
    if k >= n:
        return None
    q = k / n
    y = y0 + (LAND_Y - y0) * q * q
    x = LOW0[0] - 2.0 * q * q                      # the wind nudges it left
    spr = DROP_ROUND
    if k == n - 2:
        spr = DROP_ST1
    elif k == n - 1:
        spr = DROP_ST2
    return spr, x, y


# ------------------------------------------------------------------ impact
LAND_X = int(round(LOW0[0] - 2.0))                 # where the drop meets the plate
_yy, _xx = D.ys, D.xs
_de = np.sqrt(((_xx - LAND_X - 0.3) / 15.5) ** 2 + ((_yy - LAND_Y + 0.6) / 5.6) ** 2)
_n = NZ.fbm2(_xx / 4.0, _yy / 2.0, 77, 3)
WET_D = _de + 0.30 * (_n - 0.5)                     # irregular wet footprint (< 1 = wet at max)
WET_REGION = (WET_D < 1.3) & G['plate']


def wet_radius(t):
    """normalised radius of the dark wet patch: spreads fast, then the dry earth
    drinks it from the rim inward (gone by ~1.10)."""
    a = age(t, T_HIT)
    if a < 0:
        return 0.0
    if a < 0.10:
        return 0.40 + 0.60 * math.sin(0.5 * math.pi * a / 0.10)
    if a < 0.13:
        return 1.0
    u = (a - 0.13) / 0.27
    return max(0.0, 1.0 - u ** 0.8)


# droplets flung from the crown's points:
# (vx, vy px/s, x0 offset, ground depth offset, size, launch height above the plate, launch delay in frames)
FLING = [(-42.0, -60.0, -13, 2, 2, 9, 0), (-16.0, -80.0, -6, -2, 1, 14, 0), (18.0, -70.0, 4, -3, 2, 15, 1),
         (46.0, -52.0, 13, 3, 2, 8, 0)]
GRAV = 1900.0


def droplets(img, t):
    """the crown's beads, flung up and out; each lands and leaves a tiny wet speck."""
    for i, (vx, vy, x0, dz, sz, lh, dl) in enumerate(FLING):
        a = age(t, T_HIT) - (2.0 + dl) / FPS_
        if a < 0:
            continue
        gy = LAND_Y + dz                                  # its own bit of ground
        y0 = LAND_Y - lh
        A, B, Cc = 0.5 * GRAV, vy, y0 - gy
        tl = (-B + math.sqrt(B * B - 4 * A * Cc)) / (2 * A)
        x = LAND_X + x0 + vx * min(a, tl)
        if a < tl:
            y = y0 + vy * a + 0.5 * GRAV * a * a
            vyy = vy + GRAV * a
            put(img, x, y, 'white0' if i % 2 == 0 else 'glass4')
            if sz > 1:
                put(img, x, y + 1, 'glass3')
                put(img, x + 1, y, 'glass4')
            sp = math.hypot(vx, vyy)
            if sp > 50:
                put(img, x - vx / sp * 1.2, y - vyy / sp * 1.2, 'glass3')
        elif a < tl + 0.16:
            xi, yi = int(round(x)), int(round(gy))
            for ddx in range(sz):
                if 0 <= xi + ddx < D.WW and 0 <= yi < D.HH and G['plate'][yi, xi + ddx]:
                    img[yi, xi + ddx] = shift(img[yi:yi + 1, xi + ddx:xi + ddx + 1], -1)[0, 0]
    return img


def dust_puff(img, t):
    """the dry dust the drop knocks up: a few motes lifting and drifting left."""
    a = age(t, T_HIT)
    if a < 0 or a > 0.40:
        return img
    for i in range(10):
        ang = math.pi * (0.10 + 0.8 * NZ.rand(i, 1, 303))
        sp = 16 + 26 * NZ.rand(i, 2, 303)
        x = LAND_X + math.cos(ang) * sp * a * 1.6 - 9.0 * a - 1
        y = LAND_Y - 1 - math.sin(ang) * sp * a + 6.0 * a * a
        life = 0.18 + 0.20 * NZ.rand(i, 3, 303)
        if a < life:
            put(img, x, y, 'dust8' if a < life * 0.5 else 'dust7')
    return img


def splash(img, t):
    f = frame_of(t) - frame_of(T_HIT)
    if f < 0:
        return img
    # the wet patch: two steps darker in its heart, one at its rim
    r = wet_radius(t)
    if r > 0:
        heart = WET_REGION & (WET_D < r * 0.70)
        rim = WET_REGION & (WET_D < r) & ~heart
        img = np.where(heart, shift(img, -2), img)
        img = np.where(rim, shift(img, -1), img)
    if f < len(D.SPLASH_BIG):
        bi, bm, (ox, oy), clear = D.SPLASH_BIG[f]
        x0, y0 = LAND_X - ox, LAND_Y - oy
        if clear is not None:
            # a sheet of water: the plate behind it shows through, brightened and cooled
            h, w = bi.shape
            sub = img[y0:y0 + h, x0:x0 + w]
            see = shift(sub, 2)
            bi = np.where(clear, see, bi)
        blit_idx(img, bi, bm, x0, y0)
    elif f - 3 < len(D.PUDDLES):
        bi, bm, (ox, oy) = D.PUDDLES[f - 3]
        blit_idx(img, bi, bm, LAND_X - ox, LAND_Y - oy - 1)
    img = droplets(img, t)
    img = dust_puff(img, t)
    return img


# ------------------------------------------------------------------ the air: dust
def wind(t):
    return 0.55 + 0.45 * float(NZ.fbm1(np.array([t * 1.6 + 3.0]), 5, 3)[0])


def grains(img, t):
    """dust grains skittering over the plates, right -> left, with little hops."""
    n = 30
    i = np.arange(n)
    X0 = (NZ.rand(i, 1, 61) - 0.5) * 3.6
    Z0 = 0.8 + NZ.rand(i, 2, 61) * 1.6
    spd = 0.8 + NZ.rand(i, 3, 61) * 1.3
    Xc = ((X0 - spd * (t + 8.9) + 2.0) % 4.0) - 2.0
    hop = np.abs(np.sin((t * (3 + 3 * NZ.rand(i, 4, 61)) + NZ.rand(i, 5, 61)) * math.pi)) * 0.025
    x = D.X_C + D.FOC * Xc / Z0
    y = D.Y_H + D.FH / Z0 - hop * D.FOC / Z0
    for xi, yi, zi in zip(x, y, Z0):
        xi, yi = int(round(xi)), int(round(yi))
        if 0 <= xi < D.WW and D.Y_H + 2 < yi < D.HH and not STALK_M[yi, xi]:
            img[yi, xi] = ci('dust8' if zi < 1.3 else 'dust7')
            if zi < 1.0 and xi + 1 < D.WW:
                img[yi, xi + 1] = ci('dust6')
    return img


def motes(img, t):
    """airborne dust drifting with the wind (behind the spout)."""
    xm, ym, rm = D.field(t + 8.9, 20, 71, (-10, 8, 340, 120), vel=(-26.0, 2.0), jitter=(8, 3), sway=(1.5, 0.5))
    for xi, yi, ri in zip(xm, ym, rm):
        xi, yi = int(xi), int(yi)
        if 0 <= xi < D.WW and 0 <= yi < D.HH:
            img[yi, xi] = ci('dust9' if ri > 0.6 else 'dust8')
    return img


# the one mote that drifts down past the empty mouth, where the bead hung:
# dust, where there should be water
HERO = [(0.74, (186.0, 42.0)), (0.95, (174.0, 56.0)), (1.10, (165.0, 67.0)), (1.30, (153.0, 79.0))]


def hero_mote(img, t):
    if t < HERO[0][0] or t > HERO[-1][0]:
        return img
    x, y = keyframes(t, [(a, p) for a, p in HERO], ease=lambda u: u)
    x += 1.0 * math.sin(t * 6.0)
    y += 0.7 * math.sin(t * 4.3 + 1.0)
    xi, yi = int(round(x)), int(round(y))
    # it turns as it drifts, catching the light now and then
    tw = (frame_of(t) // 3) % 3
    put(img, xi, yi, 'dust10' if tw != 1 else 'dust9')
    put(img, xi + 1, yi, 'dust9' if tw == 0 else 'dust8')
    put(img, xi, yi + 1, 'dust9' if tw == 2 else 'dust8')
    put(img, xi + 1, yi + 1, 'dust7')
    return img


def leaf_sway(t):
    """the dead leaf trembles in the wind (12 fps), a key of LEAVES."""
    t12 = math.floor(t * 12 + 1e-6) / 12
    w = wind(t12)
    g = float(NZ.fbm1(np.array([t12 * 1.7]), 9, 2)[0]) - 0.5
    ang = 1.0 + 4.0 * w * (0.5 + g) + 2.0 * math.sin(t12 * 5.0)
    return int(max(-10, min(10, 2 * round(ang / 2))))


def bead_glint(img, t, bi, bm, x0, y0):
    """the full bead catches the hazy sun: a tiny star on its specular for three
    frames — the moment of most hope."""
    f = frame_of(t)
    if f not in (5, 6, 7):
        return img
    ys_, xs_ = np.nonzero(bi == ci('white1'))
    if not len(xs_):
        return img
    sx, sy = x0 + int(xs_.min()), y0 + int(ys_.min())
    arm = {5: 1, 6: 2, 7: 1}[f]
    put(img, sx, sy, 'white1')
    for a in range(1, arm + 1):
        c = 'white0' if a == 1 else 'glass4'
        for (ddx, ddy) in ((a, 0), (-a, 0), (0, -a)):
            put(img, sx + ddx, sy + ddy, c)
    return img


def drop_shadow(img, t):
    """the soft, sky-lit contact shadow that gathers under the drop just
    before it lands (overcast light: small and diffuse)."""
    k = frame_of(t) - frame_of(T_DETACH)
    n = frame_of(T_HIT) - frame_of(T_DETACH)
    if k < n - 3 or k >= n + 1:
        return img
    rx = {n - 3: 2.5, n - 2: 3.5, n - 1: 4.5, n: 6.0}[k]
    x0 = LAND_X
    sub_y0, sub_y1 = LAND_Y - 2, LAND_Y + 3
    for yy in range(sub_y0, sub_y1):
        for xx in range(int(x0 - rx - 1), int(x0 + rx + 2)):
            e = ((xx - x0) / rx) ** 2 + ((yy - LAND_Y + 0.3) / (rx * 0.38)) ** 2
            if e <= 1.0 and G['plate'][yy, xx]:
                img[yy, xx] = shift(img[yy:yy + 1, xx:xx + 1], -1)[0, 0]
    return img


def sp_mask_at(sp, x, y):
    """is (x, y) (screen, for the unshifted pose) on the spout sprite's outline?"""
    xi, yi = int(x) - sp['x0'], int(y) - sp['y0']
    m = sp['mask']
    return 0 <= yi < m.shape[0] and 0 <= xi < m.shape[1] and m[yi, xi]


# ------------------------------------------------------------------ render
def render(t):
    t = min(max(t, 0.0), DURATION + 1.0)
    t = frame_of(t) / FPS_                        # every beat lives on the 30 fps frame grid
    img = BG.copy()
    img = motes(img, t)
    img = grains(img, t)
    D.draw_grass(img, t, STALK_BASE)
    li, lm, (ox, oy) = LEAVES[leaf_sway(t)]
    blit_idx(img, li, lm, NODE[0] - ox, NODE[1] - oy)

    da, dx, dy = spout_pose(t)
    sp = SPOUTS[da]
    blit_idx(img, sp['idx'], sp['mask'], sp['x0'] + dx, sp['y0'] + dy)
    lx, ly = sp['low'][0] + dx, sp['low'][1] + dy
    lxi, lyi = int(round(lx)), int(math.floor(ly)) + 1

    if frame_of(t) < frame_of(T_DETACH):
        bi, bm, cx = BEAD_ST[bead_stage(t)]
    else:
        bi, bm, cx = remnant(t)
    blit_idx(img, bi, bm, lxi - cx, lyi)
    img = bead_glint(img, t, bi, bm, lxi - cx, lyi)
    # the lip is wet where the water gathers: a film over its edge, one glint
    for ddx, c in ((-2, 'glass2'), (-1, 'glass2'), (0, 'glass3'), (1, 'glass2')):
        if sp_mask_at(sp, lxi + ddx - dx, lyi - 1 - dy):
            put(img, lxi + ddx, lyi - 1, c)
    if sp_mask_at(sp, lxi - 1 - dx, lyi - 2 - dy):
        put(img, lxi - 1, lyi - 2, 'glass4')

    img = drop_shadow(img, t)
    img = splash(img, t)
    # the satellite droplet trails the drop by two frames; the drop itself
    for spr, dl in ((SAT, 2), (None, 0)):
        ds = drop_state(t, dl)
        if ds is None:
            continue
        s_, x, y = ds
        s_ = s_ if spr is None else spr
        di, dm, dcx, bot = s_
        blit_idx(img, di, dm, int(round(x)) - dcx, int(round(y)) - bot)
        if spr is None and s_ is DROP_ST2:
            # speed: two faint beads of the trail above it
            put(img, int(round(x)), int(round(y)) - bot - 3, 'glass3')
            put(img, int(round(x)), int(round(y)) - bot - 6, 'glass2')
    img = hero_mote(img, t)
    return PAL[img]
