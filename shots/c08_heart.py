"""
c08 · THE HEART INTO THE SOIL  (global 42.8–43.6, local 0–0.8)   — insert in S7b

The film's central act, close: the Gardener's two hands (c02's design: steel
three-finger hands, mint back plates) bring its heart (c03's amber orb in its
copper cradle, still tethered by its cables) down from the upper left and press
it into the planter's dark, dry, cracked soil, a hand's width left of the
wilting sprout. Cold grey-blue dawn falls from the upper right (c06's light and
materials, so the cut into c06 is seamless).

Local timeline (frames at 30 fps; global = local + 42.8):
  f0–f5   (0.00–0.17) the orb descends; its warm pool on the soil grows brighter
          and tightens as it nears the ground
  f6      (0.20, cue core_in_soil 43.0) the hands press it in: it sinks halfway,
          soil grains hop, light bursts out along the cracks (f6–f10, racing
          outward), the sprout's leaves tremble in the warm light
  f7–f16  (0.23–0.53) it sinks fully; the glow sinks into the ground; the cracks
          pulse once like a heartbeat (lub f12, dub f14)
  f18     (0.60, cue power_down 43.4) the fingers linger on the soil ...
  f19–f23 (0.63–0.77) ... and slacken; the light settles to a smouldering ember
          (c06 opens on it); the hands go still
"""
import math
import numpy as np
from engine import *
from engine import noise as NZ
from shots import lib_drop as D
from shots.lib_drop import ci, PAL, put, blit_idx, shift
from shots.lib_closeup import mech_hand, level_of, thick_curve, bezier, edge_out, edge_in, shifted, disc, quantize

DURATION = 0.8
FPS_ = 30
HH, WW = D.HH, D.WW
xs, ys = D.xs, D.ys


def frame_of(t):
    return int(math.floor(t * FPS_ + 1e-6))


F_PRESS, F_GONE, F_LUB, F_DUB, F_LINGER, F_LAST = 6, 16, 12, 14, 18, 23

# ------------------------------------------------------------------ the place
HEART = (148.0, 116.0)                 # where the heart goes into the soil (screen)
SPROUT = (190.0, 110.0)                # the sprout's foot, a hand's width to the right
R_ORB = 13.5
S = D.c08_planter(HEART, SPROUT)
S0 = np.array([HEART[0], HEART[1], D.soil_z(HEART[1])])          # the soil point under the heart
ZS = D.soil_z(ys)
_src = S['soil'] & (S['rad'] < 7.0)
CRACK_D = D.crack_geodesic(S['cracks'], _src, maxd=70.0)         # the light's road along the cracks
GLOW_NET = np.isfinite(CRACK_D)

# ------------------------------------------------------------------ the heart's path
H_DESC = [26.0, 19.0, 13.0, 8.0, 3.9, 1.2]                          # gap under the orb, f0-f5
HC_SINK = {6: 0.55, 7: 0.05, 8: -0.12, 9: -0.28, 10: -0.42, 11: -0.56, 12: -0.68, 13: -0.80, 14: -0.90,
           15: -0.98}                                                # centre height / r while it sinks


def orb_centre(f):
    if f < F_PRESS:
        hc = H_DESC[max(f, 0)] + R_ORB
    elif f in HC_SINK:
        hc = HC_SINK[f] * R_ORB
    else:
        hc = -1.25 * R_ORB
    return S0 + D.UP3 * hc, hc


# ------------------------------------------------------------------ the hands (12 fps pose holds)
# per hand: (wrist offset from the anchor, pointing angle deg, curl, scale, elbow offset from the wrist)
POSES = {
    'grip':   {'near': ((-41.0, 6.0), -6.0, 1.10, 2.4, (-36.0, 16.0)), 'far': ((-37.0, -16.0), 26.0, 1.15, 2.2, (-30.0, -26.0))},
    'press':  {'near': ((-39.0, 7.0), 2.0, 0.85, 2.4, (-36.0, 14.0)), 'far': ((-28.0, -22.0), 50.0, 0.80, 2.2, (-26.0, -28.0))},
    'linger': {'near': ((-37.0, 8.0), 6.0, 0.55, 2.4, (-36.0, 13.0)), 'far': ((-25.0, -19.0), 44.0, 0.50, 2.2, (-27.0, -27.0))},
    'slack':  {'near': ((-37.0, 9.0), 10.0, 0.30, 2.4, (-36.0, 12.0)), 'far': ((-26.0, -18.0), 40.0, 0.28, 2.2, (-28.0, -26.0))},
}
SHOULDER = {'near': (-40.0, 96.0), 'far': (-6.0, -40.0)}


def hand_state(f):
    """(pose, anchor = where the hands hold / rest (screen xy), sag px)."""
    if f <= F_PRESS + 1:
        C, _ = orb_centre(f)
        return ('grip' if f < F_PRESS else 'press'), (C[0], C[1]), 0
    C8, _ = orb_centre(F_PRESS + 2)
    base = (C8[0] - 13.0, min(C8[1], HEART[1] - 6.0))     # the fingers settle at the hole's edge
    if f < F_GONE:
        return 'press', (base[0], base[1] + min(f - F_PRESS - 2, 3) * 0.7), 0
    if f < F_LINGER + 1:
        return 'linger', (base[0], base[1] + 2.0), 0
    return 'slack', (base[0], base[1] + 2.0), (1 if f < F_LINGER + 3 else 2)


_HAND_CACHE = {}


def hands(f):
    """[(albedo, mask, normals)] for the far and the near arm+hand (full frame),
    cached by (pose, integer anchor, sag)."""
    pose, anc, sag = hand_state(f)
    key = (pose, int(round(anc[0])), int(round(anc[1])), sag)
    if key in _HAND_CACHE:
        return _HAND_CACHE[key]
    out = []
    for which in ('far', 'near'):
        woff, ang, curl, sc, eoff = POSES[pose][which]
        wx, wy = anc[0] + woff[0], anc[1] + woff[1] + sag
        el = (wx + eoff[0], wy + eoff[1])
        ac_, am, an = D.c08_arm(SHOULDER[which], el, (wx, wy), r_up=6.2 if which == 'near' else 5.4,
                                r_fore=5.2 if which == 'near' else 4.6)
        hc, hm, hn, tip = mech_hand(HH, WW, wx, wy, math.radians(ang), s=sc, curl=curl, spread=0.9)
        am &= ~hm
        out.append((np.where(hm, hc, ac_), am | hm, np.where(hm[..., None], hn, an)))
    _HAND_CACHE[key] = out
    return out


# ------------------------------------------------------------------ light curves (per frame)
POOL = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.15, 0.55, 0.36, 0.26, 0.20, 0.16, 0.22, 0.15, 0.18, 0.12,
        0.08, 0.06, 0.05, 0.045, 0.04, 0.04, 0.04, 0.04]
GLOW_R = {6: 6.0, 7: 18.0, 8: 31.0, 9: 42.0, 10: 50.0, 11: 52.0, 12: 66.0, 13: 58.0, 14: 62.0, 15: 52.0,
          16: 45.0, 17: 38.0, 18: 33.0, 19: 28.0, 20: 24.0, 21: 21.0, 22: 19.0, 23: 18.0}
GLOW_I = {6: 1.25, 7: 1.15, 8: 1.05, 9: 0.95, 10: 0.86, 11: 0.78, 12: 1.12, 13: 0.80, 14: 0.94, 15: 0.72,
          16: 0.64, 17: 0.58, 18: 0.54, 19: 0.50, 20: 0.47, 21: 0.45, 22: 0.44, 23: 0.44}
WARM_CAP = 5.0                                    # the soil tops out at amber: the heart stays brightest
# the under-glow over the buried heart: a flattened, lumpy patch on the soil
_ue = np.sqrt(((xs - HEART[0] - 2.0) / 25.0) ** 2 + ((ys - HEART[1]) / 10.5) ** 2)
SUB_FALL = np.clip(1 - _ue, 0, 1) ** 1.0 * (0.70 + 0.6 * NZ.fbm2(xs / 3.2, ys / 2.0, 93, 2))
SUB = {7: 0.35, 8: 0.50, 9: 0.60, 10: 0.66, 11: 0.64, 12: 0.95, 13: 0.70, 14: 0.82, 15: 0.66, 16: 0.62,
       17: 0.60, 18: 0.58, 19: 0.56, 20: 0.55, 21: 0.54, 22: 0.53, 23: 0.52}
# micro-cracks in the crust over the heart, where its light leaks through (c06's ember)
EMB_SPECK = S['soil'] & (NZ.fbm2(xs / 2.1, ys / 1.3, 97, 3) > 0.56) & (_ue < 0.85)


def blur3(a, n=2):
    for _ in range(n):
        p = np.pad(a, 1, mode='edge')
        a = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
             p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]) / 9.0
    return a


# the glowing line is the crack's floor: one pixel wide (its lower edge), not the whole crack
GLOW_LINE = GLOW_NET & ~shifted(GLOW_NET, 0, -1)
GLOW_LINE |= GLOW_NET & (S['rad'] < 9.0)


def glow_field(f):
    """emissive intensity along the cracks (0 where dark)."""
    if f < F_PRESS:
        return np.zeros((HH, WW))
    R = GLOW_R[f]
    I0 = GLOW_I[f]
    d = np.where(GLOW_LINE, CRACK_D, np.inf)
    base = I0 * np.clip(1 - d / R, 0, 1) ** 0.55
    # while the light is still racing out (and on each beat) a bright front runs ahead
    front = {6: 1.0, 7: 1.0, 8: 1.0, 9: 0.9, 10: 0.6, 12: 0.9, 14: 0.6}.get(f, 0.0)
    head = front * np.exp(-((d - R * 0.92) / 5.0) ** 2)
    return np.where(d < R + 2.0, np.maximum(base, head), 0.0)


# ------------------------------------------------------------------ the crater / the closed mound
def crater(f, C, hc):
    """(ring, inner, shadow, lip) masks of the soil pushed up round the sinking orb:
    the ring of raised soil, the hole, the band just outside the ring that the ring
    shades from the heart's light, and the ring's front lip (drawn over the heart)."""
    if f < F_PRESS or f >= F_GONE:
        return None, None, None, None
    rho = math.sqrt(max(R_ORB ** 2 - hc ** 2, 0.0)) + 0.8
    e = np.sqrt(((xs - HEART[0]) / rho) ** 2 + ((ys - HEART[1]) / (rho * D.SIN35)) ** 2)
    w = 3.4 / max(rho, 3.0)
    ring = S['soil'] & (e >= 1.0) & (e < 1.0 + w)
    inner = S['soil'] & (e < 1.0)
    shadow = S['soil'] & (e >= 1.0 + w) & (e < 1.0 + 3.2 * w)
    lip = ring & (ys > HEART[1] + 0.5)
    return ring, inner, shadow, lip


MOUND = S['soil'] & (np.sqrt(((xs - HEART[0]) / 10.0) ** 2 + ((ys - HEART[1]) / 4.6) ** 2) < 1.0)
MOUND_CRUMB = MOUND & (NZ.fbm2(xs / 1.8, ys / 1.4, 71, 2) > 0.52)


# ------------------------------------------------------------------ the sprout
SPROUTS = {}
for _pal in ('dawn', 'warm', 'hot'):
    for _k, (_dl, _dr, _ln) in enumerate(((0.0, 0.0, 0.0), (0.10, -0.08, 1.0), (-0.07, 0.07, -0.5), (0.05, -0.04, 0.5))):
        SPROUTS[(_pal, _k)] = D.c08_sprout(_pal, _ln, _dl, _dr)


def sprout_state(f):
    """(palette, tremble key): warm in the heart's light, trembling on the impact and the beat."""
    pal = 'warm' if f < F_PRESS else ('hot' if f < F_PRESS + 2 else ('warm' if f < F_GONE + 1 else 'dawn'))
    if F_PRESS <= f < F_PRESS + 5:
        k = (1, 2, 3, 2, 3)[f - F_PRESS]
    elif F_LUB <= f < F_LUB + 4:
        k = (1, 3, 2, 3)[f - F_LUB]
    else:
        k = 0
    return pal, k


# ------------------------------------------------------------------ grains knocked up by the press
GRAINS = [(-9.0, -2.0, -34.0, -58.0), (-5.0, 1.0, -18.0, -74.0), (7.0, 0.0, 22.0, -66.0), (11.0, -1.0, 36.0, -50.0),
          (-12.0, 2.0, -44.0, -40.0), (3.0, 2.0, 10.0, -80.0), (-2.0, -3.0, -6.0, -62.0), (13.0, 2.0, 46.0, -34.0)]


def grains(img, f, glow_on):
    a = (f - F_PRESS) / FPS_
    if a < 0 or a > 0.24:
        return img
    for i, (x0, y0, vx, vy) in enumerate(GRAINS):
        x = HEART[0] + x0 + vx * a
        y = HEART[1] + y0 + vy * a + 0.5 * 900.0 * a * a
        if y > HEART[1] + y0 + 0.5:
            continue
        # grains flung up beside the heart catch its light; the far ones stay dark
        near = abs(x - HEART[0]) < 16
        put(img, x, y, ('core3' if i % 2 else 'rust4') if near else ('rust1' if i % 2 else 'bark1'))
        if i % 2 == 0:
            put(img, x + 1, y, 'rust2' if near else 'bark0')
    return img


# ------------------------------------------------------------------ the tether (c03's cables)
_CABLE_CACHE = {}


def _cable_masks(cx, cy):
    key = (int(round(cx)), int(round(cy)))
    if key not in _CABLE_CACHE:
        out = []
        for k, (ex, ey) in enumerate(((-20.0, 30.0), (6.0, -30.0))):
            p0 = (key[0] - R_ORB * 0.55, key[1] - R_ORB * 0.35 + 4.0 * k)
            mid = ((p0[0] + ex) / 2 + 6.0, max(p0[1], ey) + 22.0 - 8.0 * k)
            pts = bezier((ex, ey), mid, p0, n=14)
            m, al, ac, dd = thick_curve(HH, WW, pts, 1.3, 1.1)
            out.append((m, m & (ac > 0.2) & (al > 0.55)))
        _CABLE_CACHE[key] = out
    return _CABLE_CACHE[key]


def cables(img, C, f, hidden):
    """two cables from the back of the orb up to the chest (off frame, upper left);
    dark, their undersides warm where the heart lights them."""
    for m, lit in _cable_masks(C[0], C[1]):
        img = np.where(m & ~hidden, ci('ink1'), img)
        img = np.where(lit & ~hidden, ci('rust2' if f < F_GONE else 'rust1'), img)
    return img


# ------------------------------------------------------------------ render
def render(t):
    t = min(max(t, 0.0), DURATION + 1.0)
    f = min(frame_of(t), F_LAST)
    C, hc = orb_centre(f)
    alb = S['alb'].copy()
    nrm = S['nrm'].copy()
    zz = ZS.copy()
    ring, inner, cshadow, lip = crater(f, C, hc)
    if ring is not None:
        alb = np.where(ring, ci('ochre2'), alb)
        alb = np.where(ring & shifted(inner, 0, 1) & ~inner, ci('ochre1'), alb)
        rn = D._nz3(np.stack([(HEART[0] - xs) * 0.08, (HEART[1] - ys) * 0.08 - 0.6, np.full_like(xs, 0.7)], -1))
        nrm = np.where(ring[..., None], rn, nrm)
    elif f >= F_GONE:
        alb = np.where(MOUND, ci('ochre1'), alb)
        alb = np.where(MOUND_CRUMB, ci('ochre2'), alb)
    # ---- the hands and arms (far, near) into the lighting buffers
    hh = hands(f)
    hmask = np.zeros((HH, WW), bool)
    for (col, m, n) in hh:
        alb = np.where(m, col, alb)
        nrm = np.where(m[..., None], n, nrm)
        zz = np.where(m, C[2] + 12.0, zz)
        hmask |= m
    # ---- warm light: the heart above ground, then what leaks from the hole + the cracks
    Lc = C if hc > R_ORB * 0.6 else S0 + D.UP3 * max(hc + R_ORB * 0.7, 2.0)
    L = np.stack([Lc[0] - xs, Lc[1] - ys, Lc[2] - zz], -1)
    d2 = (L * L).sum(-1)
    dist = np.sqrt(d2) + 1e-6
    ln = np.clip((nrm * L).sum(-1) / dist, 0, 1)
    # inverse square: high up the pool is wide and soft; near the ground, tight and fierce
    I = 1000.0 * POOL[f] * ln / (d2 + 16.0)
    I = np.where(hmask, I * 0.2, I)                       # the hands are lit mostly where they touch it
    g = glow_field(f)
    halo = blur3(g, 2) * 2.2
    I = I + np.where(hmask, 0.0, halo * 0.5)
    if cshadow is not None:
        I = np.where(cshadow, I * 0.22, I)                   # the raised rim shades the ground beyond it
    warm = level_of(I, 0.02, 2.0)
    warm = np.where(S['soil'] | S['ctop'], np.minimum(warm, WARM_CAP), warm)
    warm = np.where(hmask & (alb == ci('steel0')), np.minimum(warm, 1.5), warm)   # outlines stay lines
    key = 3.0 * np.clip((nrm * D.DAWN3).sum(-1), 0, 1) ** 0.9
    key = np.where(S['soil'], key * (0.55 + 0.25 * np.clip(ys / 90.0, 0, 1)), key)
    amb = np.zeros((HH, WW))
    img = D.C08_RL.apply(alb, amb, warm, key, band=0.35, amb_band=0.12, cold_band=0.15)
    lit = img.copy()
    near_m = hh[1][1]
    # ---- the ember's micro-cracks (emissive), then the glowing cracks (the hands stay in front)
    if f >= F_PRESS + 1:
        ev = SUB[f] * 1.25 * np.clip(1 - _ue / 0.85, 0, 1) ** 0.8
        em = EMB_SPECK & ~hmask & (ev > 0.08)
        img = np.where(em, quantize(ev, ['rust2', 'core1', 'core2', 'core3', 'core4', 'core5'],
                                    cuts=[0.16, 0.30, 0.46, 0.64, 0.86]), img)
    hot = (g > 0.05) & ~hmask
    gc = quantize(g, ['rust1', 'core0', 'core1', 'core2', 'core3', 'core4', 'core5'],
                  cuts=[0.12, 0.24, 0.38, 0.54, 0.72, 0.92])
    img = np.where(hot, gc, img)
    # ---- the tether and the heart (emissive), hidden under the soil
    if hc > -R_ORB:
        oi, om, R = D.c08_orb(R_ORB, 1.3)
        cx, cy = int(round(C[0])), int(round(C[1]))
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float64)
        zf = C[2] + np.sqrt(np.clip(R_ORB ** 2 - xx ** 2 - yy ** 2, 0, None))
        P3 = np.stack([C[0] + xx - S0[0], C[1] + yy - S0[1], zf - S0[2]], -1)
        above = (P3 * D.UP3).sum(-1) > 0.0
        img = cables(img, C, f, near_m)
        blit_idx(img, oi, om & above, cx - R, cy - R)
        orb_vis = np.zeros((HH, WW), bool)
        ya, xa = cy - R, cx - R
        orb_vis[max(ya, 0):ya + 2 * R + 1, max(xa, 0):xa + 2 * R + 1] = \
            (om & above)[max(-ya, 0):, max(-xa, 0):][:HH - max(ya, 0), :WW - max(xa, 0)]
        if hc > -R_ORB * 0.3:
            clip = lambda x, y: (np.array([x - S0[0], y - S0[1], C[2] - S0[2]]) @ D.UP3) < 0 or near_m[y, x]
            img = D.c08_cradle(img, C[0], C[1], R_ORB, clip=clip)
    else:
        img = cables(img, C, f, near_m | ~(ys < HEART[1] - 2))
    if lip is not None:
        # the crater's front lip covers the heart's lower edge: a lit top, a dark face
        top_row = lip & ~shifted(lip, 0, 1)
        img = np.where(lip, ci('bark1'), img)
        img = np.where(top_row, ci('rust4' if f < F_PRESS + 3 else 'rust3'), img)
    img = np.where(near_m, lit, img)                      # the near hand is in front of it all
    # c03's language: where the fingers border the glowing heart, their rims burn amber
    if hc > -R_ORB:
        glow_src = orb_vis & ~near_m
        n1 = hmask & (shifted(glow_src, 1, 0) | shifted(glow_src, -1, 0) | shifted(glow_src, 0, 1) |
                      shifted(glow_src, 0, -1))
        n2 = hmask & ~n1 & (shifted(n1, 1, 0) | shifted(n1, -1, 0) | shifted(n1, 0, 1) | shifted(n1, 0, -1))
        hot_ = POOL[f] > 0.5
        img = np.where(n2 & (alb != ci('steel0')), ci('rust4' if hot_ else 'rust3'), img)
        img = np.where(n1, ci('core3' if hot_ else 'rust4'), img)
    # ---- grains, the sprout
    img = grains(img, f, f >= F_PRESS)
    pal, k = sprout_state(f)
    si, sm, (bx, by) = SPROUTS[(pal, k)]
    blit_idx(img, si, sm, int(SPROUT[0]) - bx, int(SPROUT[1]) - by)
    return PAL[img]
