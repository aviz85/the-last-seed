"""
S8b · BLOOM, OUTSIDE (global 48.0–53.0) — owner: BLOOM.  THE MONEY SHOT.

Exterior wide: the greenhouse hill in the dead valley (local t = global - 48).
  0.000  (48.0 cut_s8b_climax)  IMPACT FRAME: tree + dome punched out as a
         black silhouette against a white-gold flash with light spikes;
         camera jolt + decaying shake (second bump at the canopy burst).
  0.03-0.30  the dome glows gold from inside like a lantern; glass shards,
         the lantern cap and iron scraps burst out; the trunk tip rises
         through the apex carrying a leaf bud; cracks glow white-gold across
         the hill, roots surface along them.
  0.24-0.90  the CANOPY EXPLODES: the bud bursts into foliage masses that
         inflate outward with an ease-out (overshoot + settle), limbs inside;
         loose leaves & petals fly out; blossoms pop (white frame -> pink).
  0.60  (48.6 color_wave)  a heartbeat of light at the dome's foot releases
         the WAVE: a perspective ring races out over the land (warm pre-glow,
         white-hot edge, gold band, green afterglow, sparkles, rising motes);
         grass grows up, flowers pop, trees leaf out, the stream fills with
         light from its source (~0.95-1.7), the foreground branch comes alive
         base -> tips (~1.2-1.7), the ruins' vines climb (~3.0-3.8).
  2.20  (50.2 sun_break)  the dust tears open just above the canopy's edge:
         gold rim light, blue sky spreading, god-rays; the sun bursts through
         its own gap at ~2.38; sunlight sweeps the land, the canopy's dappled
         shadow falls over the hill and dome, cloud shadows drift.
  3.00  (51.0 birds)  a flock bursts out from the canopy into the open sky;
         a far flock crosses the other way.
  3.0-5.0  slow eased crane-up / pull-back (sub-pixel Stage parallax) reveals
         the reborn valley; wind ripples the new grass; petals drift.
render(t) is valid for t up to 5.6+ (the S9a dissolve samples past 5.0);
the camera path continues into S9a.
"""
import math
import numpy as np

from engine import *
from engine import draw, light, noise, ease, fx
from engine import palette as P
from shots import lib_valley as V

DURATION = 5.0

WORLD, DOME = V.get_world()
HILL = WORLD['hill']
DOME_ALL = DOME['shell'] | DOME['wall'] | DOME['bent'] | DOME['doorf'] | DOME['door']
HILLFX = V.HillFX()
_strokes, _clumps = V.s8_tree_parts()
_TP = V.Paint(-40, -90, 400, 220)
_DM = V.dome_masks(_TP)
TREE = V.Tree(_strokes, _clumps, x0=-40, y0=-90, w=400, h=220, seed=11,
              clip=(_DM['shell'] & ~_DM['hole']) | _DM['wall'] | (_TP.Y > V.DOME_Y0), env=V.S8_ENV)
PARTING = V.CloudParting(WORLD['sky'])
TREE.bake()
BAKE_T = 1.36                 # growth & blossom pops done -> baked canopy
V.prewarm()

camera = V.camera


def shake(t):
    if t < 0 or t > 1.5:
        return 0.0, 0.0
    amp = 3.2 * math.exp(-t / 0.20) + 1.2 * math.exp(-((t - 0.40) / 0.13) ** 2) + 0.35 * math.exp(-t / 0.8)
    dx, dy = fx.shake(t, amp, seed=5, freq=24)
    jolt = 2.4 * math.exp(-t / 0.06)
    return dx, dy + jolt


# ------------------------------------------------------------------ lighting
SUN_REVEAL = 2.55
IMPACT_F = 1.0 / 30 - 1e-4      # the single impact frame
IMPACT_RAYS = [(-math.pi / 2 + (noise.rand(k, 1, 541) - 0.5) * 2.9, 0.022 + noise.rand(k, 2, 541) * 0.035,
                0.6 + noise.rand(k, 3, 541) * 0.4) for k in range(13)]
RAY_W = [0.030, 0.050, 0.026, 0.060, 0.024, 0.042, 0.030, 0.050]


def rays_strength(t):
    return 1.8 * ease.out_cubic(ease.prog(t, 2.25, 2.95)) - 0.5 * ease.in_out_sine(ease.prog(t, 3.5, 5.5))


# the canopy's dappled shadow over the hill & dome (hill-layer coords)
def _tree_shadow():
    X, Y = HILL.X, HILL.Y
    e = ((X - (V.DOME_X + 36)) / 64.0) ** 2 + ((Y - (V.DOME_BASE + 6)) / 13.0) ** 2
    dap = V.tex_sample(V.TEX_CLOUDSH, X, Y, sx=3.0, sy=5.0)
    sh = np.clip((1 - e) * 2.2, 0, 1) * (dap < 0.62)
    sh = np.where(Y >= V.APEX_Y + 8, sh, 0.0)
    return sh


TREE_SH = _tree_shadow()


def frame_maps(t, sunxy, apex, tip, base=(0, 0)):
    """Screen-space light maps shared by all layers this frame."""
    M = {}
    if t >= 2.2:
        rs = rays_strength(t)
        angs = [0.26 + 0.04 * math.sin(t * 0.7) + k * 0.155 + 0.025 * math.sin(t * 1.3 + k) for k in range(8)]
        amps = [0.72 + 0.28 * math.sin(t * 2.1 + k * 1.7) for k in range(8)]
        rl, dist, ang = V.ray_map(sunxy[0], sunxy[1], angs, RAY_W, amps, 430.0)
        M['rays'] = rs * rl
        M['sunlit'] = np.clip((t - (2.42 + dist * 0.0042)) / 0.30, 0, 1)
        vis = ease.prog(t, SUN_REVEAL - 0.05, SUN_REVEAL + 0.25)
        burst = 2.2 * math.exp(-max(t - SUN_REVEAL, 0) / 0.35) * vis
        M['sunglow'] = ((1.3 * vis + burst) * np.clip(1 - dist / (44 + 34 * burst), 0, 1) ** 1.3 *
                        np.clip((dist - 12.0) / 8.0, 0, 1))
    if t < 0.9:
        dx = V.SXS - apex[0]
        dy = V.SYS - apex[1]
        d = np.sqrt(dx * dx + (dy * 1.15) ** 2)
        core = 4.5 * math.exp(-t / 0.06) + 1.3 * math.exp(-t / 0.28)
        lv = core * np.clip(1 - d / (46 + 90 * min(t / 0.05, 1)), 0, 1) ** 1.5
        if t < 0.35:
            ang = np.arctan2(dy, dx)
            sp = np.zeros(d.shape)
            for (ak, wk, pk) in IMPACT_RAYS:
                da = np.abs((ang - ak + math.pi) % (2 * math.pi) - math.pi)
                sp = np.maximum(sp, np.clip(1 - da / wk, 0, 1) * pk)
            ln = 200 + 280 * min(t / 0.10, 1)
            lv = lv + 3.2 * math.exp(-t / 0.09) * sp * np.clip(1 - d / ln, 0, 1)
        M['impact'] = lv
        if t < 0.07:
            rad = 78.0 if t < IMPACT_F else 40.0
            M['flash'] = 3.4 * np.clip(1 - d / rad, 0, 1) ** 0.8
    if tip is not None:
        d = np.sqrt((V.SXS - tip[0]) ** 2 + (V.SYS - tip[1]) ** 2)
        M['tip'] = tip[2] * np.clip(1 - d / 20.0, 0, 1) ** 1.3
    if 0.56 <= t < 1.1:
        # the wave is released: one heartbeat of light at the dome's foot
        bx, by = base
        d = np.sqrt((V.SXS - bx) ** 2 + ((V.SYS - by) * 2.2) ** 2)
        pk = 2.6 * math.exp(-max(t - 0.6, 0) / 0.13) * min(1.0, (t - 0.56) / 0.04)
        M['pulse'] = pk * np.clip(1 - d / 75.0, 0, 1) ** 1.4
    return M


def layer_light(name, idx, al, T, t, M, ox_s, oy_s, glass=None, lit_mask=None, shade=None, dj=0, di=0,
                sky_rim=None):
    """All lighting for one layer window (ox_s, oy_s: its screen position)."""
    h, w = idx.shape
    a = t - T
    sky = name == 'sky'
    S = lambda k: V.smap(M[k], ox_s, oy_s, h, w)
    # ---- 1. overcast until the sunlight front passes; then shadows
    if not sky:
        m = al if lit_mask is None else (al | lit_mask)
        sunlit = S('sunlit') if 'sunlit' in M else 0.0
        env = -(1 - sunlit)
        if shade is not None:
            env = env - shade * sunlit
        env = np.where(m, env, 0.0)
        if lit_mask is not None and t < 1.8:
            # the freshly burst tree carries its own life-glow for a moment
            env = env + np.where(lit_mask, 1.0 - ease.smoothstep((t - 0.45) / 1.15), 0.0)
        if (env < -0.01).any():
            idx = V.ramp_light(idx, env, band=0.6, ox=dj, oy=di)
    # ---- 2. trailing green glow of the wave
    if not sky:
        g = np.where((a > 0.08) & (a < 1.1), 1.7 * np.exp(-np.clip(a - 0.08, 0, 5) / 0.30), 0.0)
        sel = g > 0.25
        if sel.any():
            idx = V.tint_light(idx, g, 'green', band=0.55, sel=sel, ox=dj, oy=di)
    # ---- 3. golden light
    lv = np.zeros((h, w))
    if not sky:
        far = name in ('far', 'v1')
        lv += np.where((a >= -0.22) & (a < 0), (0.9 if far else 1.8) * np.clip(1 + a / 0.22, 0, 1) ** 1.2, 0.0)
        lv += np.where((a >= 0.05) & (a < 0.5), (1.6 if far else 3.0) * np.exp(-np.clip(a - 0.05, 0, 5) / 0.12), 0.0)
    else:
        if sky_rim is not None:
            lv += sky_rim
        if 'sunglow' in M:
            lv += S('sunglow')
    if 'rays' in M:
        lv += S('rays') * {'sky': 0.75, 'far': 0.35, 'v1': 0.6}.get(name, 1.0)
    if 'impact' in M:
        lv += S('impact')
        if glass is not None:
            lv += np.where(glass, 3.4 * math.exp(-t / 0.40), 0.0)
    if 'tip' in M:
        lv += S('tip')
    if 'pulse' in M and not sky:
        lv += S('pulse')
    if (lv > 0.05).any():
        idx = V.tint_light(idx, lv, 'sun', band=0.5, ox=dj, oy=di)
    # ---- 4. white-hot leading edge + the impact frame
    fl = np.zeros((h, w))
    if not sky:
        fl += np.where((a >= 0) & (a < 0.06), 3.2 if name not in ('far', 'v1') else 1.4, 0.0)
        if glass is not None:
            fl = np.where(glass, 0.0, fl)
    if 'flash' in M:
        fl += S('flash')
    if (fl > 0.05).any():
        idx = V.tint_light(idx, fl, 'flash', band=0.5, sel=fl > 0.05, ox=dj, oy=di)
    return idx


# ------------------------------------------------------------------ particles
N_SHARD = 110
_si = np.arange(N_SHARD)
SH_ANG = -math.pi / 2 + (noise.rand(_si, 1, 501) - 0.5) * 2.8
SH_SPD = 40 + noise.rand(_si, 2, 501) ** 0.7 * 150
SH_X0 = V.DOME_X + (noise.rand(_si, 3, 501) - 0.5) * 16
SH_Y0 = V.APEX_Y + 3 + (noise.rand(_si, 4, 501) - 0.5) * 6
SH_LIFE = 0.9 + noise.rand(_si, 5, 501) * 1.2
SH_KIND = noise.rand(_si, 6, 501)
SH_SPIN = 7 + noise.rand(_si, 7, 501) * 16


def draw_shards(idx, alpha, t, ox, oy):
    if t > 2.5 or t < 0:
        return
    age = t + 0.035
    drag = (1 - math.exp(-age * 0.9)) / 0.9
    x = SH_X0 + np.cos(SH_ANG) * SH_SPD * drag
    y = SH_Y0 + np.sin(SH_ANG) * SH_SPD * drag + 0.5 * 175.0 * age * age
    ok = (age < SH_LIFE) & (y < V.hill_top(x) + 1)
    ph = ((age * SH_SPIN + SH_KIND * 10) % 4).astype(int)
    for k in np.nonzero(ok)[0]:
        xx, yy = x[k] - ox, y[k] - oy
        kind = SH_KIND[k]
        if kind < 0.12:
            V.plot(idx, alpha, [xx, xx + 1], [yy, yy], 'ink2')
            continue
        glint = ph[k] == 0
        c = 'white1' if (glint and kind > 0.5) else ('glass4' if kind > 0.72 else ('glass3' if kind > 0.4 else 'sky7'))
        if ph[k] % 2 == 0:
            V.plot(idx, alpha, [xx, xx + 1], [yy, yy + 1], c)
        else:
            V.plot(idx, alpha, [xx], [yy], c)
        if glint and kind > 0.82 and age < 1.3:
            V.star(idx, xx, yy, 'white1', 'glass4')


N_LEAF = 80
_li = np.arange(N_LEAF)
_side = np.where(noise.rand(_li, 7, 561) < 0.5, 0.0, math.pi)
_elev = -0.75 + noise.rand(_li, 1, 561) * 1.0
LF_ANG = np.where(_side == 0.0, _elev, math.pi - _elev)
LF_SPD = 330 + noise.rand(_li, 2, 561) ** 0.7 * 260
LF_T0 = 0.24 + noise.rand(_li, 3, 561) * 0.22
LF_LIFE = 0.8 + noise.rand(_li, 4, 561) * 0.9
LF_KIND = noise.rand(_li, 5, 561)


def draw_leaf_burst(idx, alpha, t, ox, oy):
    """Loose leaves & petals flung out as the canopy explodes."""
    if t < 0.24 or t > 2.0:
        return
    age = t - LF_T0
    ok = (age > 0) & (age < LF_LIFE)
    if not ok.any():
        return
    drag = (1 - np.exp(-np.clip(age, 0, None) * 3.2)) / 3.2
    x = V.CROWN[0] + np.cos(LF_ANG) * LF_SPD * drag * 1.15 + 9 * np.sin(age * 5 + _li) * age
    y = V.CROWN[1] - 10 + np.sin(LF_ANG) * LF_SPD * drag * 0.6 + 30 * age * age
    fl = ((age * 12 + _li) // 1).astype(int) % 3
    for k in np.nonzero(ok)[0]:
        kd = LF_KIND[k]
        if kd < 0.62:
            c = ('leaf6', 'leaf4', 'leaf5')[fl[k]]
        else:
            c = ('bloom5', 'bloom4', 'white0')[fl[k]]
        xx, yy = x[k] - ox, y[k] - oy
        if fl[k] == 0:
            V.plot(idx, alpha, [xx, xx + 1], [yy, yy], c)
        else:
            V.plot(idx, alpha, [xx], [yy], c)


def draw_lantern(idx, alpha, t, ox, oy):
    """The dome's lantern cap, blown off, tumbling up and away."""
    if t < 0 or t > 1.7:
        return
    x = V.DOME_X + 3 + 42 * t
    y = V.APEX_Y - 5 - 125 * t + 0.5 * 190 * t * t
    if y > V.hill_top([x])[0]:
        return
    fr = int(t * 14) % 4
    shapes = [[(0, 0), (1, 0), (2, 0), (1, -1), (1, -2), (0, 1), (2, 1)],
              [(0, 0), (0, 1), (0, 2), (-1, 1), (-2, 1), (1, 0), (1, 2)],
              [(0, 0), (1, 0), (2, 0), (1, 1), (1, 2), (0, -1), (2, -1)],
              [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (-1, 0), (-1, 2)]]
    for dx, dy in shapes[fr]:
        V.plot(idx, alpha, [x - ox + dx], [y - oy + dy], 'ink2')
    V.plot(idx, alpha, [x - ox + 1], [y - oy], 'rust3')


N_DUST = 90
_di = np.arange(N_DUST)
DU_ANG = noise.rand(_di, 1, 511) * math.tau
DU_SPD = 20 + noise.rand(_di, 2, 511) * 46


def draw_dust(idx, alpha, t, ox, oy):
    """Dust shaken off the hill by the impact: a low ring puffing outward."""
    if t < 0 or t > 1.4:
        return
    r = DU_SPD * (1 - math.exp(-t * 2.4)) / 2.4 * 2.4
    x = V.DOME_X + np.cos(DU_ANG) * (V.DOME_RX + 3 + r)
    y = V.DOME_BASE + 1 + np.sin(DU_ANG) * (3 + r * 0.22) - 7 * t * (0.5 + noise.rand(_di, 6, 511))
    on = (t < 0.5 + noise.rand(_di, 3, 511) * 0.9) & (noise.rand(_di, 4, 511) < 1 - t / 1.4)
    V.plot(idx, alpha, x[on] - ox, y[on] - oy, 'dust8')
    on2 = on & (noise.rand(_di, 5, 511) < 0.55)
    V.plot(idx, alpha, x[on2] - ox + 1, y[on2] - oy, 'dust7')


def draw_motes(idx, Tfull, t, seed, j0, i0, n=110):
    """Motes of light rising from freshly-alive ground (gold / green);
    world-locked to the layer."""
    h, w = idx.shape
    H_, W_ = Tfull.shape
    k = np.arange(n)
    px = (noise.rand(k, 1, seed) * W_).astype(int)
    py = (noise.rand(k, 2, seed) * H_).astype(int)
    T0 = Tfull[py, px]
    t0 = T0 + 0.05 + noise.rand(k, 3, seed) * 0.35
    age = t - t0
    ok = (age > 0) & (age < 1.4) & (T0 < 50)
    yy = py - i0 - age * (9 + noise.rand(k, 4, seed) * 14)
    xx = px - j0 + np.sin(age * 5 + k) * 1.3 - age * 3
    ok &= ((t * 12 + k * 3) // 1).astype(int) % 6 != 0
    gold = noise.rand(k, 5, seed) < 0.5
    V.plot(idx, None, xx[ok & gold], yy[ok & gold], 'gold4')
    V.plot(idx, None, xx[ok & ~gold], yy[ok & ~gold], 'leaf7')


N_DD = 140
_dd = np.arange(N_DD)


def draw_dead_dust(idx, Tfull, t, seed, j0, i0):
    """Dust motes skimming the still-dead ground, blown right -> left;
    each one vanishes once the wave has reached where it floats."""
    h, w = idx.shape
    H_, W_ = Tfull.shape
    x = (noise.rand(_dd, 1, seed) * W_ - (26 + 22 * noise.rand(_dd, 2, seed)) * t) % W_
    y = noise.rand(_dd, 3, seed) * H_ + np.sin(t * 2.3 + _dd) * 1.2
    xi = x.astype(int)
    yi = np.clip(y.astype(int), 0, H_ - 1)
    Tl = Tfull[yi, xi]
    ok = (Tl < 90) & (t < Tl - 0.05)
    xw, yw = xi - j0, yi - i0
    ok &= (xw >= 0) & (xw < w) & (yw >= 0) & (yw < h)
    c = np.where(noise.rand(_dd, 4, seed) < 0.5, V.ci('dust8'), V.ci('dust7'))
    idx[yw[ok], xw[ok]] = c[ok]
    tail = ok & (noise.rand(_dd, 5, seed) < 0.55) & (xw + 1 < w)
    idx[yw[tail], xw[tail] + 1] = V.ci('dust6')


def draw_water_glints(idx, flag, al, t, j0, i0):
    ys, xs = np.nonzero((flag == 2) & al)
    if len(xs) == 0:
        return
    fr = int(t * 12)
    r = noise.rand((xs + j0) * 57 + (ys + i0) * 3, fr, 973)
    pk = r < 0.04
    for x_, y_, v in zip(xs[pk], ys[pk], r[pk] / 0.04):
        idx[y_, x_] = IDX_W1
        if v < 0.4 and x_ + 1 < idx.shape[1] and flag[y_, x_ + 1] == 2:
            idx[y_, x_ + 1] = V.ci('sky8')


def draw_sparkles(idx, T, t, seed, dj=0, di=0):
    """Twinkling sparkles riding the wavefront band."""
    a = t - T
    band = (a >= -0.03) & (a < 0.22)
    if not band.any():
        return
    fr = int(t * 15)
    ys, xs = np.nonzero(band)
    r = noise.rand((xs + dj) * 131 + (ys + di) * 7, fr, seed)
    pick = r < 0.085
    xs1, ys1 = xs[pick], ys[pick]
    rr = r[pick] / 0.085
    big = rr < 0.14
    for x_, y_ in zip(xs1[big], ys1[big]):
        V.star(idx, x_, y_, 'white1', 'gold4')
    mid = (rr >= 0.10) & (rr < 0.55)
    idx[ys1[mid], xs1[mid]] = IDX_W1
    lo = rr >= 0.55
    idx[ys1[lo], xs1[lo]] = IDX_G4


IDX_W1 = V.ci('white1')
IDX_G4 = V.ci('gold4')

# petals drifting from the canopy
N_PET = 50
_pi = np.arange(N_PET)
PE_X0 = V.CANOPY_C[0] + (noise.rand(_pi, 1, 521) - 0.5) * 160
PE_Y0 = 8 + noise.rand(_pi, 2, 521) * 40
PE_T0 = 0.9 + noise.rand(_pi, 3, 521) * 6.0
PE_VX = 9 + noise.rand(_pi, 4, 521) * 10
PE_VY = 8 + noise.rand(_pi, 5, 521) * 8


def draw_petals(idx, alpha, t, ox, oy):
    age = (t - PE_T0) % 3.4
    on = t >= PE_T0
    x = PE_X0 + PE_VX * age + 3 * np.sin(age * 3.1 + _pi)
    y = PE_Y0 + PE_VY * age + 1.5 * np.sin(age * 4.3 + _pi * 2)
    flip = ((age * 8 + _pi) // 1).astype(int) % 3
    for c, m in (('bloom4', on & (flip == 0)), ('bloom5', on & (flip == 1)), ('white0', on & (flip == 2))):
        V.plot(idx, alpha, x[m] - ox, y[m] - oy, c)


# flock A bursts out from behind the canopy's right edge (cue 'birds' 51.0);
# flock B, far and small, crosses the other way low over the horizon
FLOCK_A = V.make_flock(9, 551, 246, 34, (16, 20), 54.0, -9.0, 2.98, size_mix=(5, 5, 4), dt0=0.4)
FLOCK_B = V.make_flock(6, 552, 334, 58, (22, 8), -30.0, -2.0, 3.05, size_mix=(4, 3), dt0=0.3, hz=(8.0, 10.0), far=True)


def draw_birds(idx, t, ox, oy):
    if t < 2.95:
        return
    V.draw_flock(idx, FLOCK_B, t, ox, oy, cols='day')
    V.draw_flock(idx, FLOCK_A, t, ox, oy, cols='day')


# ------------------------------------------------------------------ render
def tip_glow(t):
    """Light at the rising trunk tip."""
    if t > 0.8 or t < 0:
        return None
    s = min(t / 0.26, 1.0)
    y = V.APEX_Y - (V.APEX_Y - V.CROWN[1]) * (1 - (1 - s) ** 3)
    return (V.CROWN[0], y, 2.6 * (1 - ease.smoothstep((t - 0.2) / 0.6)))


CLOUD_SY = {'v2': 2.6, 'hill': 1.7, 'v3': 1.15, 'fg': 0.9}


def render(t):
    t = float(t)
    cx, cy = camera(t)
    shx, shy = shake(t)
    st = Stage(bg='dust6')
    ps = V.PAR['sky']
    sunxy = (V.SUN[0] - cx * ps + shx, V.SUN[1] + cy * ps + shy)
    apex = (V.DOME_X - cx + shx, V.APEX_Y + cy + shy)
    tg = tip_glow(t)
    tip = None if tg is None else (tg[0] - cx + shx, tg[1] + cy + shy, tg[2])
    base = (V.DOME_X - cx + shx, V.DOME_BASE + cy + shy)
    M = frame_maps(t, sunxy, apex, tip, base)
    cloud_on = ease.smoothstep((t - 2.9) / 0.8)
    for name in V.ORDER:
        L = WORLD[name]
        p = L.p
        sx = -cx * p + shx
        sy = cy * p + shy
        win = V.window(L, sx, sy)
        if win is None:
            continue
        i0, i1, j0, j1 = win
        sky_rim = None
        if name == 'sky':
            idx, dshow, sky_rim = PARTING.frame(t, win)
            a = np.ones(idx.shape, bool)
            al = ~dshow
            T = np.full(idx.shape, 99.0)
        else:
            idx, a, al, T = V.comp_window(L, t, win)
            idx = idx.copy()
            a = a.copy()
        ox = L.x0 + j0
        oy = L.y0 + i0
        ox_s, oy_s = ox + sx, oy + sy
        glass = None
        treem = None
        Tl = T
        shade = None
        if name == 'hill':
            HILLFX.draw(idx, a, ox, oy, t, T_alive=T)
            tidx = np.zeros_like(idx)
            ta = np.zeros_like(a)
            if t < BAKE_T:
                TREE.draw(tidx, ta, ox, oy, t)
            else:
                TREE.draw_baked(tidx, ta, ox, oy, rustle=0.9 * ease.smoothstep((t - BAKE_T) / 1.0), rustle_t=t)
            idx[ta] = tidx[ta]
            a |= ta
            treem = ta
            Tl = np.where(ta, 99.0, T)
            draw_shards(idx, a, t, ox, oy)
            draw_leaf_burst(idx, a, t, ox, oy)
            draw_lantern(idx, a, t, ox, oy)
            draw_dust(idx, a, t, ox, oy)
            glass = HILL.flag[i0:i1, j0:j1] == 3
            Tl = np.where(DOME_ALL[i0:i1, j0:j1] & ~ta, 99.0, Tl)
            shade = np.where(ta, 0.0, TREE_SH[i0:i1, j0:j1])
        if name in ('hill', 'v3') and t > 2.6:
            # wind begins to ripple through the new grass
            Xw = L.X[i0:i1, j0:j1]
            Yw = L.Y[i0:i1, j0:j1]
            n = V.tex_sample(V.TEX_WIND, Xw, Yw, dx=30.0 * t, sx=0.8, sy=1.6, shear=-0.6)
            sheen = np.clip((n - 0.62) / 0.06, 0, 1) * 0.85 * ease.smoothstep((t - 2.6) / 1.0)
            sheen *= al & (L.flag[i0:i1, j0:j1] == 0)
            if name == 'hill':
                sheen = np.where(DOME_ALL[i0:i1, j0:j1], 0.0, sheen)
            idx = V.ramp_light(idx, sheen, band=0.35, ox=j0, oy=i0)
        if name in CLOUD_SY and cloud_on > 0:
            X = L.X[i0:i1, j0:j1]
            Y = L.Y[i0:i1, j0:j1]
            cs = V.tex_sample(V.TEX_CLOUDSH, X, Y, dx=7.0 * t, sx=0.55, sy=CLOUD_SY[name])
            csh = np.clip((cs - 0.60) / 0.03, 0, 1) * cloud_on
            if treem is not None:
                csh = np.where(treem, 0.0, csh)
            shade = csh if shade is None else np.maximum(shade, csh)
        idx = layer_light(name, idx, al, Tl, t, M, ox_s, oy_s, glass=glass, lit_mask=treem, shade=shade,
                          dj=j0, di=i0, sky_rim=sky_rim)
        if name in ('v2', 'v3', 'fg', 'hill'):
            draw_dead_dust(idx, L.T, t, 800 + V.ORDER.index(name), j0, i0)
        if name == 'v3' and t > 1.0:
            draw_water_glints(idx, L.flag[i0:i1, j0:j1], al, t, j0, i0)
        if name != 'sky':
            draw_sparkles(idx, Tl, t, 600 + V.ORDER.index(name), j0, i0)
            if name in ('v2', 'hill', 'v3', 'fg'):
                draw_motes(idx, L.T, t, 700 + V.ORDER.index(name), j0, i0)
        else:
            draw_birds(idx, t, ox, oy)
        if name == 'hill':
            draw_petals(idx, a, t, ox, oy)
            if t < IMPACT_F:
                # impact frame: tree + dome punched out as a dark silhouette
                sil = treem | DOME_ALL[i0:i1, j0:j1]
                idx[sil] = V.ci('ink0')
                # a hair of rim light where the flash wraps the silhouette
                rim = sil & ~V.shift(sil, 0, 1) | sil & ~V.shift(sil, 1, 0) | sil & ~V.shift(sil, -1, 0)
                idx[rim] = V.ci('ink2')
        if L.opaque:
            st.add(V.rgb_of(idx), ox_s, oy_s)
        else:
            st.add(V.rgba_of(idx, a), ox_s, oy_s)
    return st
