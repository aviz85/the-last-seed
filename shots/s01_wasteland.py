"""
S1 · THE DEAD WORLD (global 0.0–7.0) — owner: WASTELAND.
Ultra-wide establishing shot, 7 parallax layers, slow truck right.
"""
import math
import numpy as np

from engine import *
from engine import draw, light, noise, ease, sprite, fx
from engine import palette as P
from engine.core import dither_levels, bayer
from engine.sprite import blit, anchor_pos
from chars import robot
from shots import lib_wasteland as L

DURATION = 7.0

# ------------------------------------------------------------------ camera
S_CAM = 14.0                     # truck distance at the robot's depth (px)
F = dict(sky=0.02, far=0.08, fmid=0.25, mback=0.55, mid=1.0, near=1.7, fg=4.6)


def _vel(t):
    return ease.smoothstep(t / 1.4) * (1.0 - ease.smoothstep((t - 3.6) / 3.4))


_cam_raw = L.cam_table(DURATION + 1.2, _vel)
_CAM_END = _cam_raw(DURATION)


def cam(t):
    return S_CAM * _cam_raw(min(max(t, 0.0), DURATION + 1.0)) / _CAM_END


def lw(factor, extra=10):
    return W + int(math.ceil(S_CAM * factor)) + extra


# ------------------------------------------------------------------ sky
SKY_W, SKY_H = 332, 112
SUN = (74.0, 33.0)
SKY_COLS = ['dust4', 'dust5', 'dust6', 'dust7', 'dust8', 'dust9', 'dust10']


# hand-placed overcast streaks: (cx, cy, length, thickness, seed, tilt)
SKY_STREAKS = [
    (58, 6, 150, 4.5, 1, 0.010), (222, 11, 210, 6.0, 2, -0.004), (304, 27, 110, 4.0, 3, 0.000),
    (160, 24, 96, 3.0, 4, 0.006), (26, 47, 96, 3.0, 5, 0.000), (206, 44, 170, 4.5, 6, 0.004),
    (112, 57, 120, 2.6, 7, 0.000), (284, 60, 110, 3.0, 8, 0.000), (58, 70, 110, 2.0, 9, 0.000),
    (198, 76, 150, 2.4, 10, 0.000), (302, 84, 80, 2.0, 11, 0.000), (126, 86, 96, 1.6, 12, 0.000),
]


def _build_sky():
    ys, xs = np.mgrid[0:SKY_H, 0:SKY_W].astype(np.float64)
    ty = np.clip(ys / 100.0, 0, 1.2)
    base = 0.55 + 3.3 * ty ** 1.25
    dx = (xs - SUN[0]) * 0.62
    dy = (ys - SUN[1])
    de = np.sqrt(dx * dx + dy * dy)
    # haze rings: the glow is quantised into clean bands with narrow seams
    glow = 2.9 * np.exp(-(de / 34.0) ** 1.15)
    tone = base + glow
    tone += 0.45 * (noise.fbm2(xs / 150.0, ys / 26.0, seed=5, octaves=2) - 0.5)
    tone = np.clip(tone, 0, len(SKY_COLS) - 1.001)
    q = dither_levels(tone, len(SKY_COLS), n=4, band=0.3)
    near = np.exp(-(de / 30.0) ** 1.2)
    # hand-placed streaks of dust-cloud: darker bodies, the edge facing the
    # sun lit (brighter the nearer the sun), a darker opposite edge
    for (cx, cy, ln, th, sd, tl) in SKY_STREAKS:
        body, top_e, bot_e = L.streak_shape(SKY_H, SKY_W, cx, cy, ln, th, seed=sd, tilt=tl)
        b_idx = np.clip(np.floor(tone - 0.9 + 0.6 * near), 0, 6).astype(int)
        q[body] = b_idx[body]
        lit_e, dark_e = (top_e, bot_e) if cy > SUN[1] else (bot_e, top_e)
        li = np.clip(np.floor(tone + 0.3 + 1.3 * near), 0, 6).astype(int)
        sel = lit_e & (near > 0.14)
        q[sel] = li[sel]
        q[dark_e & ~sel] = np.clip(b_idx[dark_e & ~sel] - 1, 0, 6)
    img = np.zeros((SKY_H, SKY_W, 3), np.uint8)
    for i, c in enumerate(SKY_COLS):
        img[q == i] = P.C(c)
    # the sun: a pale smudge (soft dithered rim), small hot centre
    d = np.sqrt((xs - SUN[0]) ** 2 + (ys - SUN[1]) ** 2)
    th = bayer(SKY_H, SKY_W, 4)
    img[(d <= 5.6)] = P.C('dust10')
    img[(d > 5.6) & (d <= 7.0) & (th < 0.5)] = P.C('dust10')
    img[d <= 3.2] = P.C('white0')
    return img


# ------------------------------------------------------------------ L1 far
HORIZON = 99
S1_RAMPS = ('ink', 'dust', 'ash', 'ochre')
FOG_FAR = L.Fog('dust8', levels=6, ramps=S1_RAMPS)
FOG_MID = L.Fog('dust7', levels=6, ramps=S1_RAMPS)


def fog_rows(img, fog, y0, y1, a0, a1, band=0.32):
    h, w = img.shape[:2]
    ys = np.arange(h)[:, None].astype(np.float64)
    a = np.clip((ys - y0) / max(1e-6, (y1 - y0)), 0, 1) * (a1 - a0) + a0
    a = np.broadcast_to(a, (h, w))
    op = img[..., 3] > 0 if img.shape[-1] == 4 else None
    img[...] = fog.apply(img, a, band=band, mask=op)


def _build_far():
    w = lw(F['far'])
    img = layer(H, w)
    back = dict(body='ash7', lit='ash8', detail='ash6', dark='ash6')
    front = dict(body='ash6', lit='ash7', detail='ash5', dark='ash5')
    env = lambda u: 0.3 + 0.7 * math.sin(math.pi * min(max(u * 0.95 + 0.05, 0), 1)) ** 0.6
    L.city(img, -4, 150, HORIZON, seed=7, pal=back, hmax=34, hmin=5, env=env)
    L.city(img, 4, 128, HORIZON + 1, seed=8, pal=front, hmax=42, hmin=6, env=env)
    L.susp_bridge(img, 146, 210, 91, 20,
                  dict(body='ash6', lit='ash7', dark='ash5', detail='ash6', faint='ash7'))
    # far power line marching to the dome
    pal_py = dict(body='ash6', lit='ash7')
    specs = [(228, 101, 23), (250, 100.5, 16), (266, 100, 11), (277, 99.5, 8), (285, 99.2, 6)]
    atts = [L.pylon(img, x, int(round(b)), h, pal_py, lattice_on=True) for (x, b, h) in specs]
    for a, b in zip(atts[:-1], atts[1:]):
        for (x0, y0), (x1, y1) in zip(a[:len(b)], b):
            L.catenary(img, x0, y0, x1, y1, 1.0 + abs(x1 - x0) * 0.05, 'ash7')
    dome = L.dome_on_hill(img, 297, HORIZON + 1,
                          dict(hill='ash7', hill_lit='ash8', dome='ash5', dome_lit='ash6', rib='ash6'),
                          hill_w=60, hill_h=8, dome_w=11, dome_h=6)
    draw.rect(img, 0, HORIZON + 1, w, H - HORIZON - 1, 'dust8')
    fog_rows(img, FOG_FAR, HORIZON - 14, HORIZON + 1, 0.0, 0.85)
    # the dome itself stays crisp (low contrast) above the haze line
    L.dome_on_hill(img, 297, HORIZON + 1, dict(hill='ash7', hill_lit='ash8', dome='ash5', dome_lit='ash6',
                                               rib='ash6'), hill_w=60, hill_h=8, draw_hill=False)
    return img, dome


# ------------------------------------------------------------------ L2 far-mid
TURB = {}


def _build_fmid():
    w = lw(F['fmid'])
    img = layer(H, w)
    hh = L.ridge(w, seed=31, base=106.0, amp=2.5, scale=50.0, octaves=3)
    hh = hh - 3.0 * np.exp(-((np.arange(w) - 190) / 40.0) ** 2)        # dune under the turbine
    L.radio_mast(img, 16, 107, 70, 5, dict(body='ash5', lit='ash6', wire='ash6', beacon='rust2'))
    pal_t = dict(body='ash6', lit='ash7', dark='ash5', joint='ash5', rust='ochre3', hole='ash4')
    hx, hy = L.fallen_turbine(img, 150, 200, 105, pal_t)
    TURB.update(hub=(hx, hy), pal=pal_t)
    # snapped-off third blade lying on the dune in front
    bm = L.big_blade_mask(H, w, 238, 106, math.radians(176), 40, root=2.4)
    L.put(img, bm, 'ash6')
    L.put(img, bm & ~L.shift(bm, 0, 1), 'ash7')
    m = L.terrain_mask(H, w, hh)
    L.put(img, m, 'dust6')
    L.put(img, m & ~L.shift(m, 0, 1), 'dust7')
    fog_rows(img, FOG_MID, 98, 114, 0.0, 0.75)
    fog_rows(img, FOG_FAR, 30, 104, 0.3, 0.0)
    return img


def _blade_local(ang_q, length, root, snap):
    """Blade mask in a local frame (hub at (40, 60)); cached by angle."""
    return L.big_blade_mask(100, 100, 40, 60, ang_q / 10000.0, length, root=root, snap=snap)


BLADE = L.MaskCache(_blade_local)


def turbine_rotor(img, ox, t):
    """Per-frame creaking rotor (drawn on the far-mid crop, ox = crop origin).
    The raised blade hangs loose in its pitch bearing and rocks in the wind
    in little stick-slip jerks (8 fps)."""
    hx, hy = TURB['hub']
    hx -= ox
    pal = TURB['pal']
    tq = ease.step_time(t, 8.0)
    ph = tq / 3.3 * 2 * math.pi
    rock = math.radians(1.5 * math.sin(ph) + 0.45 * math.sin(2.7 * ph + 1.0))
    a0 = math.radians(-122)
    h_, w_ = img.shape[:2]

    def paint_blade(m):
        L.put(img, m, pal['body'])
        L.put(img, m & ~L.shift(m, -1, 0), pal['dark'])
        L.put(img, m & ~L.shift(m, 0, -1), pal['dark'])
        L.put(img, m & ~L.shift(m, 1, 0), pal['lit'])

    def place(local):
        m = np.zeros((h_, w_), bool)
        x0, y0 = int(round(hx)) - 40, int(round(hy)) - 60
        xs0, ys0 = max(0, x0), max(0, y0)
        xs1, ys1 = min(w_, x0 + 100), min(h_, y0 + 100)
        if xs0 < xs1 and ys0 < ys1:
            m[ys0:ys1, xs0:xs1] = local[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
        return m

    q = lambda a: int(round(a * 10000))
    # right blade snapped mid-span, lower-left blade buried in the dune
    paint_blade(place(BLADE(q(a0 + math.radians(120)), 50, 5.0, 0.42)))
    paint_blade(place(BLADE(q(a0 + math.radians(240)), 50, 5.0, 0.32)))
    bm = place(BLADE(q(a0 + rock), 56, 5.2, None))
    paint_blade(bm)
    # grime streaks on the raised blade
    ca, sa = math.cos(a0 + rock), math.sin(a0 + rock)
    for d in (14, 15, 27, 36):
        draw.pset(img, hx + ca * d + 1, hy + sa * d + 1, pal['dark'])
    # spinner (nose cone)
    draw.disc(img, hx, hy, 2, pal['body'])
    draw.pset(img, hx - 1, hy - 2, pal['lit'])
    draw.pset(img, hx - 2, hy - 1, pal['lit'])
    draw.pset(img, hx + 1, hy + 2, pal['dark'])
    draw.pset(img, hx + 2, hy + 1, pal['dark'])


# ------------------------------------------------------------------ L3 mid-back
def _overpass(img, xb, deck_y, gy):
    """Collapsed highway overpass (left): deck on pillars, broken span."""
    H_, W_ = img.shape[:2]
    c = dict(body='ash5', lit='ash6', dark='ash4', deep='ash3')
    deck = np.zeros((H_, W_), bool)
    deck[deck_y:deck_y + 6, 0:xb] = True
    r = L.rng(4)
    for k in range(6):
        deck[deck_y + k, xb:xb + int(r.integers(0, 3))] = True
    # pillars with caps
    pil = np.zeros_like(deck)
    for px in (6, 40):
        pil[deck_y + 6:gy, px:px + 5] = True
        pil[deck_y + 6:deck_y + 8, px - 1:px + 6] = True
    L.put(img, pil, c['body'])
    L.put(img, pil & ~L.shift(pil, 1, 0), c['lit'])
    L.put(img, pil & ~L.shift(pil, -1, 0), c['dark'])
    L.put(img, pil & ~L.shift(pil, -2, 0), c['dark'])
    # fallen span: thick slab from the break down into the sand
    x0, y0 = xb + 2, deck_y + 2
    x1, y1 = xb + 40, gy - 1
    span = L.poly_mask(H_, W_, [(x0, y0), (x0 + 3, y0 - 2), (x1 + 3, y1 - 5), (x1, y1 + 1)])
    allm = deck | span
    L.put(img, allm, c['body'])
    L.put(img, allm & ~L.shift(allm, 0, 1), c['lit'])
    L.put(img, allm & ~L.shift(allm, 0, -1), c['deep'])
    draw.hline(img, 0, xb - 1, deck_y + 2, c['dark'])          # parapet seam
    draw.line(img, x0 + 2, y0, x1 + 1, y1 - 3, c['dark'])
    # shadow under the deck
    draw.hline(img, 0, xb - 1, deck_y + 6, c['deep'])
    # rebar from the broken end
    for k, (dx, dy) in enumerate(((4, -3), (5, 0), (3, 2), (6, 3), (2, 4))):
        draw.line(img, xb + 1, deck_y + 1 + k, xb + 1 + dx, deck_y + 1 + k + dy, 'dust3')
    # bent lamp post on the deck
    draw.vline(img, 22, deck_y - 12, deck_y - 1, c['dark'])
    draw.line(img, 22, deck_y - 12, 27, deck_y - 14, c['dark'])
    draw.line(img, 27, deck_y - 14, 29, deck_y - 12, c['dark'])
    # faded green highway sign hanging off the deck
    draw.rect(img, 49, deck_y + 6, 9, 5, 'paint1')
    draw.hline(img, 49, 57, deck_y + 6, 'paint2')
    draw.hline(img, 51, 55, deck_y + 8, 'paint2')
    draw.pset(img, 56, deck_y + 11, 'paint1')
    # broken pillar stub under the gap
    draw.rect(img, xb + 12, gy - 7, 4, 7, c['body'])
    draw.vline(img, xb + 12, gy - 7, gy - 1, c['lit'])
    draw.pset(img, xb + 13, gy - 8, c['body'])


def _build_mback():
    w = lw(F['mback'])
    img = layer(H, w)
    hh = L.ridge(w, seed=41, base=115.0, amp=1.8, scale=38.0, octaves=3)
    _overpass(img, 58, 83, 118)
    L.car_husk(img, 286, int(hh[286]) + 3, dict(body='dust5', lit='dust6', hole='dust3', rust='rust2'))
    m = L.terrain_mask(H, w, hh)
    L.put(img, m, 'dust5')
    L.put(img, m & ~L.shift(m, 0, 1), 'dust6')
    # sand banked against the overpass pillars / span
    for (cx, rx, ry) in ((8, 9, 3), (42, 10, 3), (98, 12, 3)):
        draw.ellipse(img, cx, 116, rx, ry, 'dust5')
        draw.hline(img, cx - rx + 3, cx + rx - 3, 116 - ry, 'dust6')
    # dead orchard: two receding rows of pruned fruit trees
    rows = [(118.5, 10, 9.0, 'dust5', None, 4.0, 136, 226), (123.0, 16, 13.0, 'dust4', 'dust5', 0.0, 128, 226)]
    for ri, (yb, th, sp, col, lit, off, xa, xb_) in enumerate(rows):
        x = xa + off
        k = 0
        while x < xb_:
            hk = th + ((k * 7 + ri * 3) % 3) - 1
            if ri == 1 and k == 3:                 # a fallen tree in the front row
                draw.line(img, int(x) - 3, int(yb), int(x) + 10, int(yb) - 2, col)
                draw.line(img, int(x) + 6, int(yb) - 1, int(x) + 9, int(yb) - 6, col)
                draw.line(img, int(x) + 8, int(yb) - 2, int(x) + 13, int(yb) - 5, col)
                draw.pset(img, int(x) - 3, int(yb) - 1, col)
            elif ri == 0 and k == 5:               # a snapped trunk
                L.dead_tree(img, int(x), int(yb), 5, seed=9, color=col, style='snag')
            else:
                L.dead_tree(img, int(x), int(yb), hk, seed=100 + ri * 50 + k, color=col, lit=lit, style='orchard')
            x += sp
            k += 1
    # irrigation line along the front row (muted rust), with little supports
    draw.hline(img, 122, 226, 125, 'ochre1')
    draw.hline(img, 122, 226, 124, 'ochre2')
    for px in range(125, 226, 13):
        draw.pset(img, px + 6, 124, 'rust1')
    draw.line(img, 226, 124, 229, 122, 'ochre2')
    # billboard, tilted & half-buried
    pan, tear_x = L.billboard_panel(36, 15, dict(frame='dust3', frame_lit='dust5', sky='ash6', sky2='dust6',
                                                   sun='dust8', hill='paint2', hill_lit='paint3', trunk='dust3',
                                                   tree='paint1', text='dust7', strut='dust3'))
    pan_s, oy = L.shear_cols(pan, 0.09)
    bx, by = 234, 97
    draw.rect(img, bx + 6, by + 14, 2, 12, 'dust3')
    draw.rect(img, bx + 27, by + 16, 2, 10, 'dust3')
    paste(img, pan_s, bx, by - oy)
    draw.ellipse(img, bx + 17, 124, 16, 3, 'dust5')
    draw.hline(img, bx + 4, bx + 30, 121, 'dust6')
    fog_rows(img, FOG_MID, 104, 126, 0.0, 0.6)
    return img, (bx, by - oy, tear_x)


# ------------------------------------------------------------------ L4 robot plane (old road)
ROAD_Y = 129


def _build_mid():
    w = lw(F['mid'])
    img = layer(H, w)
    xs = np.arange(w)
    top = 124.6 + 0.6 * np.sin(xs / 23.0) + 0.6 * (noise.fbm1(xs / 16.0, 3, 3) - 0.5)
    m = L.terrain_mask(H, w, top)
    L.put(img, m, 'dust4')
    L.put(img, m & ~L.shift(m, 0, 1), 'dust5')
    # asphalt: low-contrast, broken, mostly buried
    road = np.zeros((H, w), bool)
    road[127:131, :] = True
    L.put(img, road, 'dust3')
    draw.hline(img, 0, w, 131, 'dust2')
    r = L.rng(9)
    for x in range(2, w, 10):
        if r.random() < 0.75:
            draw.hline(img, x, x + 2, 129, 'dust5')
    for _ in range(30):
        x = int(r.integers(0, w)); y = int(r.integers(127, 131))
        draw.line(img, x, y, x + int(r.integers(-3, 4)), y + 1, 'dust2')
    # sand drifts creeping over the road
    for cx, wd, hgt in ((14, 16, 5.5), (47, 7, 3.5), (131, 22, 6.0), (163, 9, 4.0), (214, 14, 5.0),
                        (236, 6, 3.0), (268, 25, 6.5), (309, 10, 4.0), (333, 12, 5.0)):
        pts = [(cx - wd * 0.45, 132), (cx - wd * 0.2, 131 - hgt * 0.75), (cx, 131.5 - hgt),
               (cx + wd * 0.3, 131 - hgt * 0.8), (cx + wd * 0.75, 131 - hgt * 0.35), (cx + wd * 1.3, 132)]
        pm = L.poly_mask(H, w, pts)
        L.put(img, pm, 'dust4')
        L.put(img, pm & ~L.shift(pm, 0, 1), 'dust5')
    # embankment shading below the road
    ys = np.arange(H)[:, None] + np.zeros((1, w))
    L.put(img, m & (ys >= 132) & (ys < 134), 'dust3')
    # embankment: dry stubble tufts, pebbles, a slope shading band
    L.put(img, m & (ys >= 138) & (ys < 140) & (bayer(H, w, 4) < 0.5), 'dust3')
    for k in range(70):
        x = int(r.integers(0, w)); y = int(r.integers(134, 146))
        kind = r.random()
        if kind < 0.55:
            hgt = int(r.integers(1, 3))
            draw.vline(img, x, y - hgt, y, 'dust3')
            if r.random() < 0.5:
                draw.pset(img, x + 1, y - 1, 'dust3')
            draw.pset(img, x - 1, y, 'dust5')
        else:
            draw.pset(img, x, y, 'dust5')
            draw.pset(img, x + 1, y, 'dust3')
    # old roadside reflector posts along the near shoulder (some fallen)
    for (px, kind) in ((24, 0), (63, 2), (148, 0), (188, 1), (231, 0), (272, 0), (318, 2)):
        if kind == 2:                      # fallen in the sand
            draw.hline(img, px, px + 4, 134, 'dust6')
            draw.pset(img, px + 1, 134, 'dust2')
            continue
        lean = 1 if kind == 1 else 0
        draw.line(img, px, 135, px + lean, 130, 'dust6')
        draw.pset(img, px + lean, 131, 'dust2')
        draw.pset(img, px, 136, 'dust3')
    # broken guard-rail posts, ahead of the robot
    for px in (182, 193, 205, 216):
        draw.vline(img, px, 122, 125, 'dust3')
    draw.hline(img, 182, 206, 122, 'dust3')
    draw.line(img, 206, 122, 211, 125, 'dust3')
    # leaning road sign
    draw.line(img, 246, 125, 248, 113, 'dust2')
    sg = layer(6, 9)
    draw.rect(sg, 0, 0, 9, 6, 'ash5')
    draw.rect(sg, 0, 0, 9, 1, 'ash6')
    draw.hline(sg, 2, 5, 3, 'ash7')
    draw.pset(sg, 6, 2, 'ash7'); draw.pset(sg, 6, 4, 'ash7'); draw.pset(sg, 7, 3, 'ash7')
    sgs, oy = L.shear_cols(sg, -0.2)
    paste(img, sgs, 244, 108 - oy)
    return img


# ------------------------------------------------------------------ L5 near ground
POLE = (40, 131)


def _build_near():
    w = lw(F['near'])
    img = layer(H, w)
    xs = np.arange(w)
    top = 144.0 + 2.5 * np.sin(xs / 41.0 + 1.0) + 1.5 * (noise.fbm1(xs / 20.0, 6, 3) - 0.5) * 2
    m = L.terrain_mask(H, w, top)
    L.put(img, m, 'dust3')
    cid, crack, lip = L.cracked_earth(H, w, HORIZON, seed=17, cw=5.6, K=820.0, jitter=0.9)
    tone = noise.rand(cid, 0, 3)
    below = L.depth_below(H, w, top)
    ys, xs2 = np.mgrid[0:H, 0:w]
    # wind-blown sand lying over the mud in soft drifts (clusters, not noise)
    sandn = noise.fbm2(xs2 / 38.0, ys / 7.0, seed=61, octaves=3) + np.clip(8 - below, 0, 8) * 0.03
    sand = m & (sandn > 0.56)
    mud = m & (below > 2) & ~sand
    L.put(img, mud & (tone < 0.12), 'dust2')
    L.put(img, mud & (tone > 0.88), 'dust4')
    L.put(img, mud & lip & ~crack, 'dust4')
    L.put(img, mud & crack, 'dust2')
    big = mud & crack & (noise.rand(cid, 1, 5) < 0.35)
    L.put(img, big, 'dust1')
    L.put(img, sand, 'dust4')
    L.put(img, sand & ~L.shift(sand, 0, 1), 'dust5')
    rip = sand & (np.sin(xs2 / 5.0 + ys * 2.1 + 3 * noise.fbm2(xs2 / 20.0, ys / 3.0, seed=62, octaves=2)) > 0.9)
    L.put(img, rip & ~(sand & ~L.shift(sand, 0, 1)), 'dust3')
    L.put(img, m & ~L.shift(m, 0, 1), 'dust4')
    L.put(img, m & (below >= 1) & (below < 2), 'dust2')
    L.stones(img, top, 0, w, 18, seed=23,
             pal=dict(body='dust3', lit='dust5', dark='dust2', shadow='dust1'), ymin_off=3, ymax_off=26, size=(1, 3))
    # rusted irrigation main (muted), broken joint
    for (x0, y0, x1, y1) in ((126, 164, 200, 158), (205, 158, 300, 151)):
        draw.thick_line(img, x0, y0, x1, y1, 3, 'ochre1')
        draw.line(img, x0, y0 - 1, x1, y1 - 1, 'ochre2')
        draw.line(img, x0, y0 + 2, x1, y1 + 2, 'dust1')
    draw.rect(img, 199, 156, 2, 5, 'rust1')
    draw.rect(img, 204, 155, 2, 5, 'rust1')
    draw.vline(img, 256, 146, 152, 'ochre1')
    draw.line(img, 256, 146, 259, 144, 'ochre2')
    # the pole for the rag
    px, py = POLE
    draw.line(img, px, 168, px + 2, py, 'dust1')
    draw.line(img, px + 1, 168, px + 3, py, 'dust2')
    draw.pset(img, px + 2, py - 1, 'dust1')
    return img


# ------------------------------------------------------------------ L6 foreground silhouettes
def _build_fg():
    w = lw(F['fg'], extra=24)
    img = layer(H, w)
    # left: angular rock outcrop, faceted, rim-lit from the upper left,
    # a gnarled dead tree clinging to it, dry grass at its foot
    m1 = L.boulder(img, -6, 181, 40, 30, 11, 'ink2', rim='dust1', rim_dx=-1, facets='ink1', n=8)
    m2 = L.boulder(img, 22, 181, 36, 18, 12, 'ink2', rim='dust1', rim_dx=-1, facets='ink1', n=6)
    L.put(img, (m1 | m2) & ~L.shift(m1 | m2, 0, 1), 'dust1')
    L.dead_tree(img, 17, 156, 40, seed=5, color='ink2', style='wild', trunk_w=3)
    draw.line(img, 16, 157, 12, 150, 'ink2')
    draw.line(img, 12, 150, 7, 148, 'ink2')
    for k, (dx, hgt, lean) in enumerate(((38, 9, -2), (41, 12, 1), (44, 7, 3), (47, 10, -1), (52, 6, 2))):
        draw.line(img, dx, 174, dx + lean, 174 - hgt, 'ink2')
    # middle: a tuft of dead grass blades rising from the bottom edge
    for k, (dx, h, lean) in enumerate(((0, 16, -3), (3, 22, -1), (5, 13, 2), (8, 18, 4), (-3, 11, -5))):
        draw.line(img, 176 + dx, 180, 176 + dx + lean, 180 - h, 'ink2')
    # right: tilted fence post with sagging barbed wire + a concrete chunk
    draw.thick_line(img, 352, 180, 358, 118, 4, 'ink2')
    draw.line(img, 355, 180, 360, 120, 'dust1')
    L.catenary(img, 354, 128, 290, 142, 5, 'ink2')
    L.catenary(img, 353, 138, 292, 152, 4, 'ink2')
    for bx in range(296, 352, 6):
        s = (bx - 290) / 64.0
        yy = 128 + (142 - 128) * (1 - s) + 5 * 4 * s * (1 - s)
        draw.pset(img, bx, int(round(yy)) - 1, 'ink2')
        draw.pset(img, bx, int(round(yy)) + 1, 'ink2')
    pts = [(300, 180), (310, 166), (332, 162), (348, 170), (352, 180)]
    cm = L.poly_mask(H, w, pts)
    L.put(img, cm, 'ink2')
    L.put(img, cm & ~L.shift(cm, 0, 1), 'dust1')
    return img


BUILT = {}


def build():
    if BUILT:
        return BUILT
    BUILT['sky'] = _build_sky()
    BUILT['far'], BUILT['dome'] = _build_far()
    BUILT['fmid'] = _build_fmid()
    BUILT['mback'], BUILT['bb'] = _build_mback()
    BUILT['mid'] = _build_mid()
    BUILT['near'] = _build_near()
    BUILT['fg'] = _build_fg()
    return BUILT


build()

# ------------------------------------------------------------------ robot
WALK_V = float(getattr(robot, 'WIDE_WALK_SPEED', 4.0))
ROBOT_X0 = 92.0                  # world x (L4) at t = 0
STEP_PERIOD = 0.5                # one footfall every half walk cycle (wide cycle = 1.0 s)
CONTACTS = ((0.4, 0), (0.9, 2))  # (phase in the 1 s wide cycle, foot x offset from 'feet')


def robot_x(t):
    """Continuous world x at the wide walk speed. The camera never trucks
    faster than 3.04 px/s at this depth, so the robot's SCREEN x rises
    monotonically (>= 0.96 px/s): no step-forward / slide-back. (The wide
    walk's own frame stepping stays within 0.6 px of this line, sub-visible
    at 11 px tall.)"""
    return ROBOT_X0 + WALK_V * t


def contacts_before(t, t_from=-14.0):
    """All footfalls (time, world x of the planted foot) up to time t."""
    out = []
    n0 = int(math.floor(t_from))
    for n in range(n0, int(math.floor(t)) + 1):
        for ph, off in CONTACTS:
            tc = n + ph
            if tc > t:
                break
            x = (robot_x(tc) if tc >= 0 else ROBOT_X0 + WALK_V * tc) + off
            out.append((tc, x, off))
    return out


# ------------------------------------------------------------------ FX setup
VEIL_A = L.VeilField(1000, H, seed=91, sx=120.0, sy=12.0, octaves=2, lo=0.42, hi=0.66)
VEIL_B = L.VeilField(1000, H, seed=92, sx=95.0, sy=7.0, octaves=2, lo=0.45, hi=0.66)
BAND_A = L.band_profile(H, 58, 104, soft_top=18, soft_bot=6)
BAND_B = L.band_profile(H, 99, 127, soft_top=7, soft_bot=3)
SURF_ROAD = 130.0 + 0.8 * np.sin(np.arange(2000) / 13.0)
SURF_NEAR = 150.0 + np.zeros(2000)


SHADOW_F = L.VeilField(1600, H, seed=97, sx=150.0, sy=9.0, octaves=2, lo=0.50, hi=0.68)
YS = np.arange(H)[:, None].astype(np.float64)


RAYS = L.RayField(H, W + 2, SUN[0], SUN[1])


def ray_map(t, ox=0.0):
    """Crepuscular rays fanning down-right from the sun smudge (screen space)."""
    sway = 0.02 * math.sin(t * 0.7)
    angs = [0.55 + sway, 0.86 - sway * 0.5, 1.18 + sway * 0.7]
    wid = [0.09, 0.06, 0.08]
    return RAYS.level(angs, wid, 240, peak=1.5, gamma=0.8, inner=14, inner_soft=30.0)


def sun_cover(t):
    """How much a thin drifting cloud streak veils the sun (0..1)."""
    x = 128.0 - 16.0 * t                   # streak centre drifts left across the disc
    return float(np.exp(-((x - SUN[0]) / 22.0) ** 2))


_SY, _SX = np.mgrid[0:SKY_H, 0:W + 2]


def streak_over_sun(sky, t, ox):
    """A thin, drifting strand of dust passing in front of the sun smudge."""
    x = 128.0 - 16.0 * t - ox
    h_, w_ = sky.shape[:2]
    ys, xs = _SY[:h_, :w_], _SX[:h_, :w_]
    u = (xs - x) / 34.0
    thick = np.clip(1 - u * u, 0, 1) ** 0.7 * (2.2 + 0.8 * np.sin(xs * 0.21 + t))
    yc = SUN[1] + 1.5 + 2.0 * u
    m = np.abs(ys - yc) < thick
    amt = np.where(m, 0.45 + 0.35 * (np.abs(ys - yc) < thick * 0.45), 0.0)
    y0, y1 = int(SUN[1] - 10), int(SUN[1] + 14)
    out = sky.copy()
    out[y0:y1] = FOG_MID.apply(sky[y0:y1], amt[y0:y1], band=0.3, ox=ox, oy=y0)
    return out


def shadow_map(t, c, par):
    f = SHADOW_F.sample(200 + 5.0 * t + c * par, 0, W + 2, H)
    return -0.85 * f


def gust(t):
    """Wind in gusts: a lull, a gust peaking ~2.4 s, a lull, a gust ~5.7 s."""
    g = 0.12 + 0.95 * math.exp(-((t - 2.4) / 0.95) ** 2) + 0.85 * math.exp(-((t - 5.7) / 0.85) ** 2)
    g += 0.08 * math.sin(t * 5.1)
    return float(np.clip(g, 0, 1))


VFOG_A = L.Fog('dust8', levels=3, ramps=S1_RAMPS)
VFOG_B = L.Fog('dust7', levels=3, ramps=S1_RAMPS)


def veil_maps(t, c):
    a = 0.60 * VEIL_A.sample(40 + 3.2 * t + c * 0.1, 0, W + 2, H) * BAND_A
    b = 0.50 * VEIL_B.sample(10 + 8.5 * t + c * 0.5, 0, W + 2, H) * BAND_B
    return a, b


def fog_band(crop, fog, amt, y0, y1, ox):
    sub = crop[y0:y1]
    crop[y0:y1] = fog.apply(sub, amt[y0:y1, :sub.shape[1]], band=0.12, ox=ox, oy=y0,
                            mask=(sub[..., 3] > 0) if sub.shape[-1] == 4 else None)


def render(t):
    B = BUILT
    c = cam(t)
    g = gust(t)
    va, vb = veil_maps(t, c)
    st = Stage(bg='dust8')

    # ---- sky
    sky, fr, ix = L.scrolled(B['sky'], c * F['sky'])
    sky = sky.copy()
    fog_band(sky, VFOG_A, va, 58, 104, ix)
    fog_band(sky, VFOG_B, vb, 99, 112, ix)
    cover = sun_cover(t)
    rays = ray_map(t) * (1.0 - 0.45 * cover)
    sky = L.ramp_shift(sky, rays[:SKY_H, :sky.shape[1]] * 0.8, band=0.35, ox=ix, rows=(20, SKY_H))
    sky = streak_over_sun(sky, t, ix)
    sky_fr = fr

    # ---- far city (composited onto the sky: both are nearly static)
    crop, fr, ix = L.scrolled(B['far'], c * F['far'])
    crop = crop.copy()
    fog_band(crop, VFOG_A, va, 58, 104, ix)
    fog_band(crop, VFOG_B, vb, 99, 127, ix)
    crop = L.ramp_shift(crop, rays * 0.55, band=0.35, ox=ix, rows=(40, 104))
    L.motes(crop, t, 11, 14, (0, 62, W, 44), (-4.0, 0.3), ['dust9'], sway=(1.0, 0.2), ox=-fr)
    back = np.empty((H, crop.shape[1], 3), np.uint8)
    back[:] = P.C('dust8')
    back[:SKY_H] = sky[:, :crop.shape[1]]
    paste(back, crop, 0, 0)
    st.add(back, -fr, 0)

    # ---- far-mid (turbine rotor creaks)
    crop, fr, ix = L.scrolled(B['fmid'], c * F['fmid'])
    crop = crop.copy()
    turbine_rotor(crop, ix, t)
    fog_band(crop, VFOG_B, vb, 99, 127, ix)
    crop = L.ramp_shift(crop, rays * 0.55 + shadow_map(t, c, 0.25) * 0.6, band=0.38, ox=ix, rows=(30, 116))
    st.add(crop, -fr, 0)

    # ---- mid-back (billboard flap)
    crop, fr, ix = L.scrolled(B['mback'], c * F['mback'])
    crop = crop.copy()
    bx, by, tear = B['bb']
    tq = ease.step_time(t, 12.0)
    L.cloth(crop, bx + tear + 1 - ix, by + 2, 7, 2, tq * 0.8,
            dict(body='dust6', lit='dust7', dark='ash5'), wind=0.8, seed=2.1)
    fog_band(crop, VFOG_B, vb, 99, 127, ix)
    crop = L.ramp_shift(crop, rays * 0.4 + shadow_map(t, c, 0.55) * 0.8, band=0.38, ox=ix, rows=(80, 128))
    L.motes(crop, t, 12, 16, (0, 84, W + 40, 60), (-9.0, 0.8), ['dust8', 'dust9'], sway=(1.5, 0.3), ox=-fr)
    st.add(crop, -fr, 0)

    # ---- robot plane
    crop, fr, ix = L.scrolled(B['mid'], c * F['mid'])
    crop = crop.copy()
    crop = L.ramp_shift(crop, shadow_map(t, c, 1.0), band=0.35, ox=ix, rows=(118, 150))
    L.sand_streams(crop, SURF_ROAD, ix, t, seed=5, n=14, gust=g, colors=['dust7', 'dust6', 'dust5'],
                   speed=(40, 64), length=(22, 50), yoff=(-1, 3), amp=0.8)
    rx = robot_x(t)
    # its trail: prints in the drifted sand (only where sand lies on the road)
    cts = contacts_before(t)
    pxs = [x for (_, x, _) in cts]
    pys = [ROAD_Y - (1 if off else 0) for (_, _, off) in cts]
    keep = [noise.rand(int(round(tc * 10)), 3, 44) > (rx - x) / 80.0 for (tc, x, _) in cts]
    L.footprints(crop, ix, pxs, pys, 'dust3', only_on=('dust4', 'dust5'), keep=keep)
    spr = robot.frame('walk', t, can='empty', scale='wide', core=0.8)
    fax, fay = spr.anchor('feet')
    cax, cay = spr.anchor('core')
    core_cx = rx + (cax - fax) - ix           # exact (fractional) crop coords
    core_cy = ROAD_Y + (cay - fay)
    # footstep puffs at each real footfall (world-anchored, blown left)
    for (tc, x, off) in cts[-2:]:
        age = (t - tc) / 0.55
        if 0 <= age <= 1:
            L.puff(crop, x - ix, ROAD_Y, age, seed=int(tc * 10), colors=('dust6', 'dust5'))
    crop = L.pool_local(crop, core_cx + 3, ROAD_Y, 9, peak=2.6, mode='warm', sy=0.36, gamma=1.8, band=0.45,
                        ox=ix)
    crop = L.pool_local(crop, core_cx + 1, core_cy + 1, 3.5, peak=1.5, mode='warm', sy=1.0, gamma=1.3, band=0.5,
                        ox=ix)
    st.add(crop, -fr, 0)
    # the robot on its own layer (with its contact shadow attached)
    rl = layer(24, 24)
    draw.hline(rl, 12 - 3, 12 + 2, 23, 'dust2')
    blit(rl, spr, 12, 22, anchor='feet')
    st.add(rl, rx - (c * F['mid']) - 12, ROAD_Y - 22)

    # ---- near ground (rag on the pole)
    crop, fr, ix = L.scrolled(B['near'], c * F['near'])
    crop = L.ramp_shift(crop, shadow_map(t, c, 1.7), band=0.35, ox=ix, rows=(136, H))
    L.sand_streams(crop, SURF_NEAR, ix, t, seed=8, n=16, gust=g, colors=['dust6', 'dust5', 'dust5'],
                   speed=(70, 110), length=(30, 70), yoff=(-26, 4), amp=1.5)
    L.cloth(crop, POLE[0] + 2 - ix, POLE[1] + 1, 14, 5, tq,
            dict(body='ash4', lit='ash5', dark='ash3'), wind=0.7 + 0.5 * g, seed=1.3)
    L.motes(crop, t, 13, 9, (0, 60, W + 80, 110), (-24.0, 2.0), ['dust7', 'dust8'], sway=(3.0, 0.4), ox=-fr)
    st.add(crop, -fr, 0)

    # ---- foreground silhouettes
    crop, fr, ix = L.scrolled(B['fg'], c * F['fg'])
    st.add(crop, -fr, 0)
    return st
