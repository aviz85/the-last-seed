"""
S9a · THE VALLEY (global 53.0–55.0, dissolving in from S8b over 0.6 s)
— owner: BLOOM.

Seasons later: the same world fully alive in soft morning light.  The
giant tree's vast umbrella canopy over the overgrown dome, meadows rippling
in the wind, the stream sparkling, mist in the valley, birds, butterflies,
drifting petals; the camera keeps drifting on from S8b's crane (same
camera path, so the dissolve reads as time passing in one place).
render(t) is valid for any t >= 0 (tested to DURATION + 1).
"""
import math
import numpy as np

from engine import *
from engine import draw, light, noise, ease, fx
from engine import palette as P
from shots import lib_valley as V

DURATION = 2.0
T_OFF = 5.0                   # camera clock continues from S8b

WORLD, _DOME = V.get_world()
HILL9 = V.get_hill_s9()
V3_9 = V.get_v3_s9()
HILLFX9 = V.HillFX(s9=True)
_st, _cl, _env = V.s9_tree_parts()
_TP = V.Paint(-60, -110, 440, 240)
_DM = V.dome_masks(_TP)
TREE9 = V.Tree(_st, _cl, x0=-60, y0=-110, w=440, h=240, seed=21, env=_env, blossoms=2.0,
               clip=(_DM['shell'] & ~_DM['hole']) | _DM['wall'] | (_TP.Y > V.DOME_Y0), base_lv=3.0,
               scales=False)
TREE9.bake()
V.prewarm()

SUN9 = (14.0, 50.0)           # low morning sun (sky-layer rest coords)


def _build_sky9():
    """Morning sky: blue above, a warm peach-gold glow around the low sun on
    the left, pink-lit clouds."""
    L = WORLD['sky']
    X, Y = L.X, L.Y
    idx = np.zeros((L.h, L.w), np.uint8)
    ty = np.clip((Y + 40) / (V.HORIZON + 2 + 40), 0, 1)
    v = 0.2 + 4.1 * ty ** 1.35
    ds = np.sqrt(((X - SUN9[0]) * 0.6) ** 2 + ((Y - SUN9[1]) * 1.1) ** 2)
    v += 0.9 * np.exp(-(ds / 40.0) ** 1.2)
    cols = ['sky4', 'sky5', 'sky6', 'sky7', 'sky8', 'white0']
    q = V.quant(np.clip(v, 0, len(cols) - 1.001), band=0.6)
    idx[:] = V.cis(cols)[np.clip(q, 0, len(cols) - 1)]
    # warm glow: peach -> gold toward the sun, hugging the horizon
    warm = np.exp(-(ds / 52.0) ** 1.4) * np.clip((Y + 10) / 70.0, 0, 1) * 4.2
    wl = np.clip(V.quant(np.clip(warm, 0, 3.99), band=0.55), 0, 3)
    wcol = V.cis(['sky7', 'bloom5', 'gold4', 'white0'])
    sel = wl >= 1
    idx[sel] = wcol[wl[sel]]
    # a thin band of pink along the horizon
    pinkb = (np.abs(Y - 80) < 7) & (V.bay(L.h, L.w) < np.clip(1 - np.abs(Y - 80) / 7.0, 0, 1) * 0.55) & (idx == V.ci('sky7'))
    idx[pinkb] = V.ci('bloom5')
    clouds = L.clouds
    sid, zb = V.spheres_zbuffer(L.h, L.w, L.x0, L.y0, clouds, jag=1.0, seed=3)
    L3 = np.array([-0.85, -0.2, 0.5])
    L3 = L3 / np.linalg.norm(L3)
    sh, cm = V.spheres_shade(sid, zb, L.x0, L.y0, clouds, light3=L3, tex_amp=0.10, seed=3)
    kk = np.maximum(sid, 0)
    base_y = clouds[kk, 4]
    cm &= Y <= base_y
    under = np.clip((Y - (base_y - 5)) / 5.0, 0, 1)
    lvc = 1.7 + 2.2 * sh - 1.1 * under
    ct = ['sky5', 'sky6', 'bloom4', 'bloom5', 'white0', 'white1']
    cq = np.clip(V.quant(np.clip(lvc + 0.5, 0, len(ct) - 1.001), band=0.35), 0, len(ct) - 1)
    idx[cm] = V.cis(ct)[cq[cm]]
    Pn = V.Paint(L.x0, L.y0, L.w, L.h)
    Pn.i[:] = idx
    V.paint_sun(Pn, X, Y, SUN9, warm=True)
    return Pn.i


SKY9 = _build_sky9()


# ------------------------------------------------------------------ motion
def wind_sheen(X, Y, t):
    """Travelling bands of wind across the meadow (0..1), world-locked."""
    n = V.tex_sample(V.TEX_WIND, X, Y, dx=30.0 * t, sx=0.8, sy=1.6, shear=-0.6)
    return np.clip((n - 0.62) / 0.06, 0, 1) * 0.85


def draw_water_glints(idx, flag, t, seed, j0=0, i0=0):
    ys, xs = np.nonzero(flag == 2)
    if len(xs) == 0:
        return
    fr = int(t * 12)
    r = noise.rand((xs + j0) * 57 + (ys + i0) * 3, fr, seed)
    pk = r < 0.045
    for x_, y_, v in zip(xs[pk], ys[pk], r[pk] / 0.045):
        idx[y_, x_] = V.ci('white1')
        if v < 0.4 and x_ + 1 < idx.shape[1] and flag[y_, x_ + 1] == 2:
            idx[y_, x_ + 1] = V.ci('sky8')
        if v < 0.12 and x_ - 1 >= 0 and flag[y_, x_ - 1] == 2:
            idx[y_, x_ - 1] = V.ci('sky8')


def draw_dew(idx, al, t, seed, j0, i0, dens=0.0011):
    """A few slow twinkles of dew catching the low sun (world-locked)."""
    ys, xs = np.nonzero(al)
    if len(xs) == 0:
        return
    X_, Y_ = xs + j0, ys + i0
    r = noise.rand(X_ * 31 + Y_ * 17, 0, seed)
    pk = r < dens
    ph = noise.rand(X_[pk] * 7 + Y_[pk], 1, seed + 1)
    on = ((t * 1.6 + ph) % 1.0) < 0.28          # each glint lights ~0.18 s
    idx[ys[pk][on], xs[pk][on]] = V.ci('white0')


BUTTERFLIES = [
    # x0, y0, colour pair (open, body), drift speed, seed  (hill-layer rest coords)
    (92, 112, ('white0', 'dust7'), 7.0, 1),
    (214, 116, ('gold4', 'gold2'), -6.0, 2),
    (64, 130, ('bloom4', 'bloom2'), 5.0, 3),
    (246, 128, ('violet3', 'violet1'), -7.0, 4),
    (178, 122, ('white1', 'dust7'), 4.0, 5),
    (126, 124, ('gold3', 'gold1'), 6.0, 6),
]


def draw_butterflies(idx, alpha, t, ox, oy, TT):
    """Fluttering butterflies: 3x3 'X' wings open / closed, erratic path."""
    for (x0, y0, (c1, c2), sp, sd) in BUTTERFLIES:
        u = TT - 4.6
        x = x0 + sp * u + 6 * math.sin(u * 1.3 + sd) + 2.5 * math.sin(u * 3.7 + sd * 2)
        y = y0 - 4 * math.sin(u * 0.9 + sd) - 3.0 * abs(math.sin(u * 4.6 + sd))
        fr = int(TT * 13 + sd * 3) % 3
        xi, yi = int(round(x - ox)), int(round(y - oy))
        if fr == 0:          # wings open
            pts = [(-1, -1, c1), (1, -1, c1), (-1, 1, c1), (1, 1, c1), (-1, 0, c1), (1, 0, c1), (0, 0, c2)]
        elif fr == 1:        # half
            pts = [(-1, -1, c1), (1, -1, c1), (0, 0, c2), (0, 1, c2)]
        else:                # closed (edge-on)
            pts = [(0, -1, c1), (0, 0, c2), (0, 1, c2)]
        for dx, dy, c in pts:
            V.plot(idx, alpha, [xi + dx], [yi + dy], c)


N_PET = 70
_pi = np.arange(N_PET)
PE_X0 = V.CANOPY_C[0] + (noise.rand(_pi, 1, 921) - 0.5) * 220
PE_Y0 = 0 + noise.rand(_pi, 2, 921) * 55
PE_T0 = noise.rand(_pi, 3, 921) * 4.0
PE_VX = 8 + noise.rand(_pi, 4, 921) * 9
PE_VY = 7 + noise.rand(_pi, 5, 921) * 7


def draw_petals(idx, alpha, TT, ox, oy):
    age = (TT - PE_T0) % 4.0
    x = PE_X0 + PE_VX * age + 3 * np.sin(age * 3.1 + _pi)
    y = PE_Y0 + PE_VY * age + 1.5 * np.sin(age * 4.3 + _pi * 2)
    flip = ((age * 8 + _pi) // 1).astype(int) % 3
    for c, k in (('bloom4', 0), ('bloom5', 1), ('white0', 2)):
        m = flip == k
        V.plot(idx, alpha, x[m] - ox, y[m] - oy, c)


N_NP = 16
_ni = np.arange(N_NP)
NP_X0 = -30 + noise.rand(_ni, 1, 931) * 300
NP_Y0 = 60 + noise.rand(_ni, 2, 931) * 60
NP_T0 = noise.rand(_ni, 3, 931) * 3.0
NP_VX = 22 + noise.rand(_ni, 4, 931) * 16
NP_VY = 12 + noise.rand(_ni, 5, 931) * 10


def draw_near_petals(idx, alpha, TT, ox, oy):
    """A few big petals tumbling past close to the camera (FG layer)."""
    age = (TT - NP_T0) % 3.0
    x = NP_X0 + NP_VX * age + 5 * np.sin(age * 2.3 + _ni)
    y = NP_Y0 + NP_VY * age + 3 * np.sin(age * 3.7 + _ni * 2)
    fr = ((age * 9 + _ni) // 1).astype(int) % 4
    shapes = [((0, 0), (1, 0), (0, 1)), ((0, 0), (1, 0)), ((0, 0), (0, 1)), ((0, 0), (1, 1))]
    cols = [('bloom5', 'bloom4'), ('white0', 'bloom5'), ('bloom4', 'bloom3'), ('bloom5', 'bloom3')]
    for k in range(N_NP):
        sh = shapes[fr[k]]
        c = cols[(fr[k] + k) % 4]
        for j, (dx, dy) in enumerate(sh):
            V.plot(idx, alpha, [x[k] - ox + dx], [y[k] - oy + dy], c[0] if j == 0 else c[1])


N_BM = 90
_bm = np.arange(N_BM)
BM_X = noise.rand(_bm, 1, 941) * 330 - 5
BM_Y = noise.rand(_bm, 2, 941) * 150 - 20
BM_VX = 1.5 + noise.rand(_bm, 3, 941) * 3
BM_VY = -0.8 + noise.rand(_bm, 4, 941) * 1.6


def draw_beam_motes(idx, M, TT, ox, oy, ox_s, oy_s):
    """Pollen / dust motes, visible only where they drift through a beam."""
    h, w = idx.shape
    x = BM_X + BM_VX * TT + 2.0 * np.sin(TT * 0.7 + _bm)
    y = BM_Y + BM_VY * TT + 1.5 * np.sin(TT * 0.9 + _bm * 1.7)
    xi = np.floor(x - ox).astype(int)
    yi = np.floor(y - oy).astype(int)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    R = V.smap(M['rays'], ox_s, oy_s, h, w)
    lit = np.zeros(N_BM, bool)
    lit[ok] = R[yi[ok], xi[ok]] > 0.5
    tw = ((TT * 3 + _bm * 0.37) % 1.0) < 0.75
    m = ok & lit & tw
    idx[yi[m], xi[m]] = np.where(noise.rand(_bm[m], 5, 941) < 0.5, V.ci('gold4'), V.ci('white0'))


FLOCK9_A = V.make_flock(2, 961, 238, 26, (12, 5), 9.0, -1.5, 4.9, size_mix=(5, 4), dt0=0.1, hz=(5.0, 6.0))
FLOCK9_B = V.make_flock(5, 962, 318, 46, (26, 10), -14.0, -1.0, 4.9, size_mix=(4, 3), dt0=0.2, hz=(8.0, 10.0), far=True)
FLOCK9_C = V.make_flock(4, 963, -6, 34, (18, 8), 16.0, -2.0, 5.1, size_mix=(4,), dt0=0.3, hz=(7.0, 9.0), far=True)


def draw_birds(idx, TT, ox, oy):
    V.draw_flock(idx, FLOCK9_B, TT, ox, oy, cols='morning')
    V.draw_flock(idx, FLOCK9_C, TT, ox, oy, cols='morning')
    V.draw_flock(idx, FLOCK9_A, TT, ox, oy, cols='morning')


# ------------------------------------------------------------------ light
RAY9_W = [0.024, 0.040, 0.020, 0.046, 0.026, 0.034]


def frame_maps9(TT, sunxy):
    angs = [-0.06 + k * 0.075 + 0.02 * math.sin(TT * 0.9 + k) for k in range(6)]
    amps = [0.75 + 0.25 * math.sin(TT * 1.7 + k * 1.3) for k in range(6)]
    rl, dist, ang = V.ray_map(sunxy[0], sunxy[1], angs, RAY9_W, amps, 420.0)
    rl = rl * np.clip((dist - 12.0) / 10.0, 0, 1)
    return {'rays': 1.9 * rl,
            'glow_sky': 1.1 * np.clip(1 - dist / 70.0, 0, 1) ** 1.5 * np.clip((dist - 15.0) / 8.0, 0, 1),
            'glow_gnd': 0.9 * np.clip(1 - dist / 150.0, 0, 1) ** 1.6}


def light9(name, idx, M, ox_s, oy_s, ground, dj=0, di=0):
    """Soft low morning sun from the left: long rays, warm near the sun."""
    h, w = idx.shape
    lv = V.smap(M['rays'], ox_s, oy_s, h, w) * {'far': 0.3, 'v1': 0.45, 'v2': 0.75}.get(name, 1.0)
    if name == 'sky':
        lv += V.smap(M['glow_sky'], ox_s, oy_s, h, w)
    else:
        lv += V.smap(M['glow_gnd'], ox_s, oy_s, h, w) * ground
    if (lv > 0.05).any():
        idx = V.tint_light(idx, lv, 'sun', band=0.55, ox=dj, oy=di)
    return idx




def valley_mist(name, idx, L, i0, i1, j0, j1, TT):
    """Morning mist lying in the valley folds: a luminous lift within each
    colour's own ramp (hue kept - no grey), pooled just above each nearer
    ridge and thinning upward, in soft drifting patches."""
    X = L.X[i0:i1, j0:j1]
    Y = L.Y[i0:i1, j0:j1]
    if name == 'far':
        shape = np.clip((Y - 88.0) / 8.0, 0, 1) ** 1.4 * 1.6
    else:
        dep = Y - L.top[None, j0:j1]
        a, b, amp = {'v1': (1.5, 6.0, 1.7), 'v2': (4.0, 13.0, 1.5)}[name]
        shape = np.clip((dep - a) / (b - a), 0, 1) ** 1.2 * amp
    n = V.tex_sample(V.TEX_CLOUDSH, X, Y, dx=4.0 * TT, sx=0.8, sy=2.2)
    patch = np.clip((n - 0.40) / 0.22, 0, 1)
    amt = shape * patch
    if not (amt > 0.1).any():
        return idx
    return V.ramp_light(idx, amt, band=0.5, ox=j0, oy=i0)


# ------------------------------------------------------------------ render
def render(t):
    t = float(t)
    TT = T_OFF + t
    cx, cy = V.camera(TT)
    st = Stage(bg='sky6')
    ps = V.PAR['sky']
    sunxy = (SUN9[0] - cx * ps, SUN9[1] + cy * ps)
    M = frame_maps9(TT, sunxy)
    for name in V.ORDER:
        L = HILL9 if name == 'hill' else (V3_9 if name == 'v3' else WORLD[name])
        p = L.p
        sx = -cx * p
        sy = cy * p
        win = V.window(L, sx, sy)
        if win is None:
            continue
        i0, i1, j0, j1 = win
        if name == 'sky':
            idx = SKY9[i0:i1, j0:j1].copy()
            a = np.ones(idx.shape, bool)
        else:
            idx = L.alive.i[i0:i1, j0:j1].copy()
            a = L.alive.a[i0:i1, j0:j1].copy()
        ox = L.x0 + j0
        oy = L.y0 + i0
        Xs = np.broadcast_to((ox + sx) + np.arange(j1 - j0)[None, :], idx.shape)
        Ys = np.broadcast_to((oy + sy) + np.arange(i1 - i0)[:, None], idx.shape)
        ground = a.copy()
        flag = L.flag[i0:i1, j0:j1]
        if name in ('hill', 'v3', 'v2'):
            # wind rippling through the meadow: lighter travelling bands
            Xw = L.X[i0:i1, j0:j1]
            Yw = L.Y[i0:i1, j0:j1]
            sheen = wind_sheen(Xw, Yw, TT) * (flag == 0) * a
            idx = V.ramp_light(idx, sheen, band=0.35, ox=j0, oy=i0)
        if name == 'hill':
            HILLFX9.draw(idx, a, ox, oy, 99.0, mossy=True, cracks=False)
            tidx = np.zeros_like(idx)
            ta = np.zeros_like(a)
            TREE9.draw_baked(tidx, ta, ox, oy, rustle=1.0, rustle_t=TT)
            # morning rim: leaves whose left neighbour is open sky catch the sun
            rim = ta & ~V.shift(ta, 1, 0) & (tidx != V.ci('ink0'))
            rim2 = ta & ~V.shift(ta, 2, 0) & ~rim
            tidx[rim] = V.RAMP[2][tidx[rim]]
            tidx[rim2] = V.RAMP[1][tidx[rim2]]
            idx[ta] = tidx[ta]
            a |= ta
            ground &= ~ta
        idx = light9(name, idx, M, ox + sx, oy + sy, ground, dj=j0, di=i0)
        if name == 'hill':
            # the low sun warms the sun-side of the canopy
            tm = a & ~ground
            warm = np.clip(1.0 - (Xs - sunxy[0]) / 190.0, 0, 1) ** 1.6 * 1.3 * tm
            if warm.any():
                idx = V.tint_light(idx, warm, 'sun', band=0.5, sel=warm > 0.05, ox=j0, oy=i0)
            draw_beam_motes(idx, M, TT, ox, oy, ox + sx, oy + sy)
        if name in ('far', 'v1', 'v2'):
            idx = valley_mist(name, idx, L, i0, i1, j0, j1, TT)
        if name == 'v3':
            draw_water_glints(idx, flag, TT, 971, j0, i0)
        if name == 'v3':
            draw_dew(idx, ground & (flag == 0), TT, 981 + len(name), j0, i0)
        if name == 'sky':
            draw_birds(idx, TT, ox, oy)
        if name == 'hill':
            draw_butterflies(idx, a, t, ox, oy, TT)
            draw_petals(idx, a, TT, ox, oy)
        if name == 'fg':
            draw_near_petals(idx, a, TT, ox, oy)
        if L.opaque:
            st.add(V.rgb_of(idx), ox + sx, oy + sy)
        else:
            st.add(V.rgba_of(idx, a), ox + sx, oy + sy)
    return st
