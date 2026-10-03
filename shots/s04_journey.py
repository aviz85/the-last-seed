"""
S4 · THE JOURNEY (global 20.0–26.0) — owner: WASTELAND.
Dusk tracking shot. The Gardener (medium, can='sprout') is PINNED at integer
screen x 100 (feet y 140) on its own layer while the world glides under it
on 6 parallax layers (camera eases in windows centred on each stop/start, so
its lag nets to zero; feet skate <= 0.375 px only inside those windows).
A dull red-orange sun low on the right rim-lights everything and throws long
shadows toward the camera-left.

Local beats (global = local + 20.0):
  0.00  walking (footfalls every 0.4 s: 0.0, 0.4 ... 2.4)
  2.50  stops on a contact frame, 'look_down' at the sprout (cue 22.5); the
        sprout's green glow swells; 3.20 head comes back up; 3.50 walks on
        (footfalls 3.5, 3.9, 4.3, 4.7)
  4.00  the wind rises (streams, spindrift); the dust wall rises on the
        horizon ahead (cue storm_rise 24.0); ~4.5 a tumbleweed bounds in
  4.60-4.82  the wall's crest climbs over the sun and swallows it; the hidden
        sun's glow lingers above the burning crest, then the sky dims outward
  4.55-5.55  the storm's shadow sweeps the land right->left (robot ~5.05)
  4.80  stops (idle); 5.00 'look_up' at the storm (cue 25.0); hold
  5.25  a low rumble (horizontal camera tremble, builds to 0.55 px)
  6.00  cut to S5
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

DURATION = 6.0

# ------------------------------------------------------------------ timing
T_STOP1, T_LOOK_UP1, T_GO2 = 2.50, 3.20, 3.50
T_STORM, T_STOP2, T_LOOKUP = 4.00, 4.80, 5.00
WALK_V = float(getattr(robot, 'WALK_SPEED', 10.0))
WALK_PH1 = 0.4          # walk phase at t=0: ends on a contact frame at 2.5 (and starts on one)
WALK_PH2 = 0.0          # second walk starts on a contact frame and ends on one at 4.8

SCREEN_X = 100.0             # robot feet x on screen (walking)
GROUND_Y = 140               # robot feet y
F = dict(sky=0.01, far=0.05, fmid=0.22, mid=0.5, ground=1.0, fg=3.4)
SUN = (262.0, 95.0)
SUN_R = 9


def _trav(ta):
    if hasattr(robot, 'travel'):
        return float(robot.travel('walk', max(ta, 0.0)))
    return WALK_V * ease.step_time(max(ta, 0.0), 10.0)


def walk_dist(t):
    """Robot's travelled distance (world px) — the animator's exact per-frame
    travel, so the planted foot never slides."""
    d1 = _trav(WALK_PH1 + min(t, T_STOP1 - 1e-3)) - _trav(WALK_PH1)
    d2 = _trav(WALK_PH2 + min(max(t - T_GO2, 0.0), T_STOP2 - T_GO2 - 1e-3)) - _trav(WALK_PH2)
    return d1 + d2


T_WIN = 0.4       # camera ease window, CENTRED on each stop / start


def _cam_vel(t):
    """The camera tracks the pinned robot. Around each stop/start it eases
    over a window centred on the event: smoothstep is point-symmetric, so the
    camera's lag integrates back to exactly zero when the window ends (the
    feet skate at most WALK_V*T_WIN*0.094 = 0.375 px, only inside windows)."""
    h = T_WIN / 2
    s = 1.0 - ease.smoothstep((t - (T_STOP1 - h)) / T_WIN)
    s += ease.smoothstep((t - (T_GO2 - h)) / T_WIN) * (1.0 - ease.smoothstep((t - (T_STOP2 - h)) / T_WIN))
    return WALK_V * s


_cam = L.cam_table(DURATION + 1.2, _cam_vel)


def cam(t):
    return _cam(min(max(t, 0.0), DURATION + 1.0))


def lw(f, extra=16):
    return W + int(math.ceil(WALK_V * DURATION * f)) + extra


# ------------------------------------------------------------------ light state
def dim_amount(t):
    """0 = low sun shining, 1 = sun swallowed by the dust wall."""
    return ease.smoothstep((t - 4.55) / 0.95)


def wind_amount(t):
    return 0.25 + 0.75 * ease.smoothstep((t - T_STORM) / 0.8)


# ------------------------------------------------------------------ SKY
SKY_H = 132
SKY_COLS = ['rust0', 'ochre0', 'rust1', 'ochre1', 'rust2', 'ochre2', 'rust3', 'ochre3', 'rust4', 'ochre4', 'ochre5']
MUTED = ('ochre', 'rust')          # dusk: fogs stay in the warm ramps (no grey intrusions)
FOG_DIM = L.Fog('ochre0', levels=6, ramps=MUTED)


def dim_twin(img, fog=0.18, steps=-1.0):
    """Sun gone: one ramp step darker (texture kept) + a little murk.
    Scalar amounts are snapped to exact fog levels: a uniform amount that
    falls between two levels would screen-door the whole layer."""
    op = img[..., 3] > 0 if img.shape[-1] == 4 else None
    out = L.ramp_shift(img, float(round(steps)), band=0.5, mask=op)
    lv = FOG_DIM.levels
    out = FOG_DIM.apply(out, round(fog * lv) / lv, mask=op)
    return out


# hand-placed dust streaks: (cx, cy, length, thickness, seed, tilt)
SKY_STREAKS = [
    (70, 12, 170, 5.0, 1, 0.000), (236, 20, 190, 6.0, 2, -0.008), (150, 33, 130, 3.5, 3, 0.004),
    (300, 41, 120, 4.0, 4, 0.000), (36, 47, 110, 3.0, 5, 0.006), (182, 55, 170, 4.0, 6, -0.004),
    (292, 63, 80, 2.5, 7, 0.000), (96, 69, 150, 3.0, 8, 0.000), (236, 76, 140, 2.5, 9, 0.000),
    (150, 86, 120, 2.0, 10, 0.000), (40, 90, 70, 1.8, 11, 0.000), (318, 101, 64, 1.6, 12, 0.000),
]
# one warm step darker (sun band / dust in front of the halo); core ramp included
_BAND_DARK = {'core4': 'core3', 'core3': 'core2', 'rust5': 'rust4', 'ochre5': 'ochre4', 'ochre4': 'ochre3',
              'rust4': 'rust3', 'ochre3': 'ochre2', 'rust3': 'rust2', 'ochre6': 'ochre5'}


def _band_darken(img, mask):
    for a, b in _BAND_DARK.items():
        m = mask & np.all(img == np.array(P.C(a), np.uint8), axis=-1)
        img[m] = P.C(b)


def _build_sky(lit=True):
    w = W + 8
    ys, xs = np.mgrid[0:SKY_H, 0:w].astype(np.float64)
    ty = np.clip(ys / 112.0, 0, 1.2)
    tone = 0.4 + 7.4 * ty ** 1.55
    dx = (xs - SUN[0]) * 0.42
    dy = (ys - SUN[1]) * 0.95
    de = np.sqrt(dx * dx + dy * dy)
    near = np.exp(-(de / 34.0) ** 1.1)                  # sun proximity 0..1
    glow = 2.6 * np.exp(-(de / 30.0) ** 1.2)
    tone += glow if lit else 0.35 * glow - 0.6
    tone += 0.45 * (noise.fbm2(xs / 140.0, ys / 22.0, seed=305, octaves=2) - 0.5)
    tone = np.clip(tone, 0, 9.55)                      # never brighter than the halo
    base_q = dither_levels(tone, len(SKY_COLS), n=4, band=0.3)
    q = base_q.copy()
    # hand-placed dust streaks: darker bodies; undersides lit by the low sun
    for (cx, cy, ln, th, sd, tl) in SKY_STREAKS:
        body, top_e, bot_e = L.streak_shape(SKY_H, w, cx, cy, ln, th, seed=sd, tilt=tl)
        b_idx = np.clip(np.floor(tone - 1.5), 0, len(SKY_COLS) - 1).astype(int)
        q[body] = b_idx[body]
        if lit:
            li = np.clip(np.floor(tone + 0.4 + 2.0 * near), 0, len(SKY_COLS) - 1).astype(int)
            sel = bot_e & (near > 0.10)
            q[sel] = li[sel]
            q[top_e] = np.clip(b_idx[top_e] - 1, 0, None)
    img = np.zeros((SKY_H, w, 3), np.uint8)
    for i, c in enumerate(SKY_COLS):
        img[q == i] = P.C(c)
    if lit:
        th4 = bayer(SKY_H, w, 4)
        d = np.sqrt((xs - SUN[0]) ** 2 + ((ys - SUN[1]) * 1.04) ** 2)
        # warm dithered halo: spread sideways along the horizon haze and
        # ragged (noise), luminance falling off monotonically from the disc
        hn = 1.0 + 0.22 * (noise.fbm2(xs / 7.0, ys / 4.0, seed=333, octaves=2) - 0.5) * 2
        dh = np.sqrt(((xs - SUN[0]) * 0.72) ** 2 + ((ys - SUN[1]) * 1.18) ** 2) / hn
        lv = np.clip((SUN_R + 15.0 - dh) / 15.0, 0, 1) ** 1.3 * 3.0
        hq = dither_levels(lv, 4, n=4, band=0.35)
        for lvl, col in ((1, 'ochre4'), (2, 'ochre5'), (3, 'rust5')):
            sel = (hq == lvl) & (q <= SKY_COLS.index('ochre4') + (lvl - 1))   # never darken
            img[sel] = P.C(col)
        # the disc: dull red-orange, no outline, a soft dithered limb and a
        # small, faint warm heart
        img[d <= SUN_R - 0.6] = P.C('core3')
        limb = (d > SUN_R - 0.6) & (d <= SUN_R + 0.7)
        img[limb & (th4 < 0.5)] = P.C('core3')
        img[limb & (th4 >= 0.5)] = P.C('rust5')
        img[(d <= 2.3)] = P.C('core4')
        img[(d > 2.3) & (d <= 3.9) & (th4 < 0.45)] = P.C('core4')
        # ONE thin dust band across the lower disc: a clean 1px row whose
        # tapered ends dissolve into single pixels (dithered), plus a faint
        # broken wisp above the heart
        yb = int(round(SUN[1] + 3.0))
        for xx in range(int(SUN[0] - 15), int(SUN[0] + 18)):
            u = (xx - (SUN[0] + 1.5)) / 16.5
            keep = 1.0 - u * u
            if keep <= 0:
                continue
            if keep < 0.35 and th4[yb, xx] >= keep / 0.35 * 0.75:
                continue
            _band_darken(img[yb:yb + 1, xx:xx + 1], np.ones((1, 1), bool))
        yw = int(round(SUN[1] - 4.0))
        wisp = np.zeros((SKY_H, w), bool)
        for xx in range(int(SUN[0] - 1), int(SUN[0] + 11)):
            if xx % 3 != 1:                      # broken: two on, one off
                wisp[yw, xx] = True
        _band_darken(img, wisp)
    else:
        img = dim_twin(img, fog=0.18, steps=-1.0)
    return img


# ------------------------------------------------------------------ FAR (ruins, dome)
HORIZON = 112
FOG_HZ = L.Fog('ochre5', levels=6, ramps=MUTED)
DOME = {}


def _build_far(lit=True):
    w = lw(F['far'])
    img = layer(H, w)
    back = dict(body='ochre3', lit='ochre4', detail='rust3', dark='ochre3')
    front = dict(body='rust2', lit='rust3', detail='ochre1', dark='rust2')
    env = lambda u: 0.25 + 0.75 * math.sin(math.pi * min(max(u, 0), 1)) ** 0.8
    L.city(img, -6, 150, HORIZON, seed=17, pal=back, hmax=24, hmin=4, env=env)
    L.city(img, 8, 120, HORIZON + 1, seed=18, pal=front, hmax=30, hmin=4, env=env)
    pal_d = dict(hill='rust2', hill_lit='rust3', dome='ochre0', dome_lit='rust2' if lit else 'ochre1',
                 rib='rust1', lit_side='right')
    DOME['pos'] = L.dome_on_hill(img, 222, HORIZON + 1, pal_d, hill_w=62, hill_h=9)
    for (x, h) in ((168, 14), (186, 11), (201, 8)):
        L.pylon(img, x, HORIZON + 1, h, dict(body='rust2', lit='rust2'), lattice_on=False)
    for (x0, y0), (x1, y1) in (((164, HORIZON - 8), (184, HORIZON - 6)), ((190, HORIZON - 6), (200, HORIZON - 4))):
        L.catenary(img, x0, y0, x1, y1, 1.2, 'rust2')
    draw.rect(img, 0, HORIZON + 1, w, H - HORIZON - 1, 'ochre3')
    ys = np.arange(H)[:, None].astype(np.float64)
    xsr = np.arange(w)[None, :].astype(np.float64)
    near_sun = np.clip(1.0 - np.abs(xsr - SUN[0]) / 230.0, 0, 1) ** 1.5
    a = np.clip((ys - (HORIZON - 14)) / 15.0, 0, 1) * (0.35 + 0.5 * near_sun)
    img[...] = FOG_HZ.apply(img, a, band=0.3, mask=img[..., 3] > 0)
    if not lit:
        img[...] = dim_twin(img, fog=0.25)
    return img


# ------------------------------------------------------------------ dunes
def dune_field(w, seed, base, amp, period, slip=0.28, jitter=0.35):
    """Transverse dunes seen side-on, wind from the RIGHT: a long gentle
    windward slope descending to the right of each crest, a short steep slip
    face on its left. Returns surface heights (screen y)."""
    r = L.rng(seed)
    x = np.arange(w).astype(np.float64)
    hh = np.full(w, base, np.float64)
    cx = -period * 0.3
    while cx < w + period:
        a = amp * r.uniform(0.55, 1.15)
        wl = period * r.uniform(0.45, 0.75) * slip / 0.28      # lee (slip) width
        ww = period * r.uniform(0.9, 1.5)                    # windward width
        u = x - cx
        prof = np.where(u < 0, np.clip(1 + u / wl, 0, 1) ** 1.6, np.clip(1 - u / ww, 0, 1) ** 1.35)
        # rounded crest
        prof = np.where(np.abs(u) < 2, prof * (0.94 + 0.06 * (1 - np.abs(u) / 2)), prof)
        hh = np.minimum(hh, base - a * prof)
        cx += period * r.uniform(1 - jitter, 1 + jitter)
    return hh


def shade_dunes(img, hh, pal, lit=True, rim=True, mask_extra=None):
    """Sunlit windward slopes (facing the low sun on the right), slip faces
    in shadow, a bright rim on sunlit crests, darker trough lines."""
    h_, w_ = img.shape[:2]
    m = L.terrain_mask(h_, w_, hh)
    slope = np.gradient(hh)
    face_sun = slope > 0.05
    below = L.depth_below(h_, w_, hh)
    L.put(img, m, pal['body'])
    if lit:
        L.put(img, m & ~face_sun[None, :] & (below < 7), pal['shadow'])
        L.put(img, m & face_sun[None, :] & (below < 2.5), pal['lit'])
        if rim:
            L.put(img, m & ~L.shift(m, 0, 1) & face_sun[None, :], pal['rim'])
    else:
        L.put(img, m & ~face_sun[None, :] & (below < 7), pal['shadow'])
    return m


def _build_fmid(lit=True):
    w = lw(F['fmid'])
    img = layer(H, w)
    # a half-buried leaning ruin (left of the robot's path, far from the sun)
    tw = L.poly_mask(H, w, [(26, 124), (38, 124), (46, 90), (36, 88)])
    L.put(img, tw, 'ochre1')
    for fy in range(94, 122, 4):
        x0 = 36 - (fy - 88) * 0.3
        for fx_ in range(int(x0) + 2, int(x0) + 9, 3):
            draw.pset(img, fx_ + 1, fy, 'ochre0')
    if lit:
        L.put(img, L.rim_edges(tw, 1, 0), 'rust3')
    back = dune_field(w, seed=401, base=121.0, amp=8.0, period=72.0)
    front = dune_field(w, seed=402, base=128.0, amp=12.0, period=100.0)
    xs = np.arange(w)
    valley = np.exp(-((xs - 95) / 55.0) ** 4)
    front = front * (1 - valley) + 128.0 * valley
    back = back * (1 - valley * 0.7) + 121.5 * valley * 0.7
    tmp = layer(H, w)
    shade_dunes(tmp, back, dict(body='ochre3', shadow='ochre2', lit='ochre4', rim='ochre6'), lit=lit)
    paste(img, tmp, 0, 0)
    tmp = layer(H, w)
    m = shade_dunes(tmp, front, dict(body='ochre2', shadow='ochre1', lit='rust3', rim='ochre5'), lit=lit)
    paste(img, tmp, 0, 0)
    if lit:
        sh = L.long_shadow(tw, 124, k_len=2.2, k_flat=0.14) & (img[..., 3] > 0) & ~tw
        img[...] = L.ramp_shift(img, np.where(sh, -1.0, 0.0), band=0.5)
    ys = np.arange(H)[:, None].astype(np.float64)
    a = np.clip((ys - 110) / 22.0, 0, 1) * 0.4
    img[...] = FOG_HZ.apply(img, np.broadcast_to(a, img.shape[:2]) * np.clip((126 - ys) / 6.0, 0, 1),
                            band=0.3, mask=img[..., 3] > 0)
    if not lit:
        img[...] = dim_twin(img, fog=0.25)
    return img, front


# ------------------------------------------------------------------ L3 mid: wrecks, dead tree, walls
def boulder(img, x0, yb, wd, hg, seed, body, rim=None, top=None):
    """Angular rock silhouette standing on yb; rim light on its sun side."""
    r = L.rng(seed)
    n = 7
    pts = [(x0, yb + 1)]
    for i in range(1, n):
        u = i / n
        hh_ = hg * (math.sin(math.pi * u) ** 0.7) * r.uniform(0.75, 1.05)
        pts.append((x0 + wd * u + r.uniform(-1.5, 1.5), yb - hh_))
    pts.append((x0 + wd, yb + 1))
    h_, w_ = img.shape[:2]
    m = L.poly_mask(h_, w_, pts)
    L.put(img, m, body)
    if top is not None:
        L.put(img, m & ~L.shift(m, 0, 1), top)
    if rim is not None:
        L.put(img, L.rim_edges(m, 1, 0) | (m & ~L.shift(m, 0, 1) & L.shift(L.rim_edges(m, 1, 0), 0, 0)), rim)
    return m


def _build_mid(lit=True):
    w = lw(F['mid'])
    img = layer(H, w)
    xs = np.arange(w)
    hh = 133.0 + 1.5 * np.sin(xs / 37.0) + 1.2 * (noise.fbm1(xs / 21.0, 55, 3) - 0.5) * 2
    sil = layer(H, w)
    # big gnarled dead tree (behind-left of the robot at the start)
    L.dead_tree(sil, 22, int(hh[22]) + 1, 40, seed=12, color='ochre0', style='wild', trunk_w=3)
    # a car wreck on its roof, half in the sand
    car = L.poly_mask(H, w, [(172, 134), (174, 129), (179, 127), (199, 127), (203, 130), (205, 134)])
    L.put(sil, car, 'ochre0')
    draw.rect(sil, 184, 129, 5, 2, 'ochre1')
    draw.rect(sil, 191, 129, 5, 2, 'ochre1')
    draw.pset(sil, 177, 135, 'ochre0'); draw.pset(sil, 199, 135, 'ochre0')
    # broken wall ruin with a window hole
    wm = np.zeros((H, w), bool)
    wm[118:134, 300:314] = True
    wm[114:118, 300:306] = True
    wm[116:118, 306:309] = True
    wm[122:127, 305:309] = False
    L.put(sil, wm, 'ochre0')
    # leaning power pole with a sagging cable
    px_ = 150
    yb = int(hh[px_]) + 1
    draw.line(sil, px_, yb, px_ - 5, yb - 30, 'ochre0')
    draw.line(sil, px_ + 1, yb, px_ - 4, yb - 30, 'ochre0')
    draw.hline(sil, px_ - 10, px_ + 2, yb - 27, 'ochre0')
    L.catenary(sil, px_ - 10, yb - 26, px_ - 40, yb - 3, 4, 'ochre0')
    L.catenary(sil, px_ + 2, yb - 26, px_ + 26, yb - 1, 5, 'ochre0')
    # rocks
    boulder(sil, 128, int(hh[128]) + 1, 14, 6, 3, 'ochre0')
    boulder(sil, 355, int(hh[355]) + 1, 20, 9, 4, 'ochre0')
    objs = sil[..., 3] > 0
    ground = L.terrain_mask(H, w, hh)
    L.put(img, ground, 'ochre1')
    L.put(img, ground & ~L.shift(ground, 0, 1), 'ochre3' if lit else 'ochre2')
    ys, xg = np.mgrid[0:H, 0:w]
    ph = xg / 9.0 + ys * 1.6 + 2.2 * noise.fbm2(xg / 28.0, ys / 4.0, seed=57, octaves=2)
    rip = ground & (np.sin(ph) > 0.93) & (L.depth_below(H, w, hh) > 2)
    L.put(img, rip, 'ochre0')
    if lit:
        L.put(img, L.shift(rip, 1, 0) & ground & ~rip, 'ochre2')
        base = int(np.median(hh)) + 1
        sh = L.long_shadow(objs, base, k_len=2.4, k_flat=0.22) & ground
        img[...] = L.ramp_shift(img, np.where(sh, -1.0, 0.0), band=0.5)
    paste(img, sil, 0, 0)
    if lit:
        L.put(img, L.rim_edges(objs, 1, 0), 'rust3')
    else:
        img[...] = dim_twin(img, fog=0.25)
    return img, hh


# ------------------------------------------------------------------ L4 ground: the robot's path
def _build_ground(lit=True):
    w = lw(F['ground'])
    img = layer(H, w)
    xs = np.arange(w)
    top = 138.0 + 1.0 * np.sin(xs / 29.0 + 0.5) + 0.7 * (noise.fbm1(xs / 13.0, 61, 3) - 0.5) * 2
    m = L.terrain_mask(H, w, top)
    ys, xg = np.mgrid[0:H, 0:w]
    below = np.clip(L.depth_below(H, w, top), 0, None)
    L.put(img, m, 'ochre2')
    # very broad, soft swells (a single ramp step, big clusters)
    sw = noise.fbm2(xg / 90.0, below / 9.0, seed=66, octaves=2)
    L.put(img, m & (sw > 0.62) & (below > 4), 'ochre3' if lit else 'ochre2')
    L.put(img, m & (sw < 0.36) & (below > 4), 'ochre1')
    # near band falls into shade
    L.put(img, m & (below >= 31), 'ochre1')
    L.put(img, m & (below >= 28) & (below < 31) & (bayer(H, w, 4) < 0.5), 'ochre1')
    # hand-placed ripple dashes on perspective-spaced lines
    r = L.rng(64)
    k = 0
    yk = 2.2
    while yk < 44:
        k += 1
        x = -r.uniform(0, 12)
        amp = 0.35 + 0.05 * yk
        while x < w:
            ln = r.uniform(5, 14 + yk * 0.5)
            gap = r.uniform(3, 10 + yk * 0.4)
            ph = r.uniform(0, 6.28)
            for xx in range(int(x), int(min(w, x + ln))):
                yy = int(round(top[xx] + yk + amp * math.sin(xx / (4.0 + yk * 0.25) + ph)))
                if 0 <= yy < H and m[yy, xx]:
                    base = img[yy, xx, :3].copy()
                    img[yy, xx, :3] = P.C('ochre1') if tuple(base) != P.C('ochre1') else P.C('ochre0')
                    if lit and yy - 1 >= 0 and m[yy - 1, xx] and (xx + k) % 3 != 0:
                        img[yy - 1, xx, :3] = P.C('ochre3')
            x += ln + gap
        yk = yk * 1.28 + 1.4
    L.put(img, m & ~L.shift(m, 0, 1), 'ochre4' if lit else 'ochre3')
    L.put(img, m & ~L.shift(m, 0, 1) & ~L.shift(m, 0, 2) & ~(m & ~L.shift(m, 0, 1)), 'ochre3' if lit else 'ochre2')
    # pebbles with long shadows (toward the camera-left)
    stones = np.zeros((H, w), bool)
    for _ in range(22):
        x = int(r.integers(4, w - 4)); y = int(top[x] + r.integers(3, 26))
        s_ = int(r.integers(1, 3))
        stones[y - s_ + 1:y + 1, x:x + s_ + 1] = True
    if lit:
        sh = np.zeros_like(stones)
        for k2 in range(1, 8):
            sh |= L.shift(stones, -k2, k2 // 4)
        img[...] = L.ramp_shift(img, np.where(sh & m & ~stones, -1.0, 0.0), band=0.5)
    L.put(img, stones, 'ochre0')
    if lit:
        L.put(img, L.rim_edges(stones, 1, 0), 'rust3')
    else:
        img[...] = dim_twin(img, fog=0.25)
    return img, top


# ------------------------------------------------------------------ L5 foreground rocks
def _build_fg(lit=True):
    w = lw(F['fg'], extra=60)
    img = layer(H, w)
    specs = [(40, 46, 22, 1), (150, 26, 12, 2), (262, 70, 34, 3), (372, 34, 15, 4), (452, 58, 26, 5),
             (536, 22, 10, 6), (590, 48, 20, 7)]
    for (x0, wd, hg, sd) in specs:
        m = boulder(img, x0, 181, wd, hg, sd, 'dust0')
        # a few facets: darker planes, a lit top edge, rim on the sun side
        r = L.rng(sd + 40)
        for k in range(2):
            fx0 = x0 + wd * r.uniform(0.25, 0.7)
            draw.line(img, fx0, 181, fx0 + r.uniform(-6, 6), 181 - hg * r.uniform(0.4, 0.8), 'ink1')
        L.put(img, m & ~L.shift(m, 0, 1), 'dust1')
        if lit:
            L.put(img, L.rim_edges(m, 1, 0), 'rust2')
            L.put(img, m & ~L.shift(m, 0, 1) & L.shift(L.rim_edges(m, 1, 0), 1, 0), 'rust2')
    # dry twigs sticking out of the second big rock
    for (x0, y0, x1, y1) in ((290, 150, 284, 132), (284, 132, 280, 127), (286, 138, 293, 130), (296, 149, 300, 136),
                             (470, 160, 466, 146), (466, 146, 468, 141), (478, 158, 484, 149)):
        draw.line(img, x0, y0, x1, y1, 'dust0')
    return img


# ------------------------------------------------------------------ build all
BUILT = {}


def build():
    if BUILT:
        return BUILT
    for lit in (True, False):
        k = 'L' if lit else 'D'
        BUILT['sky' + k] = _build_sky(lit)
        BUILT['far' + k] = _build_far(lit)
        BUILT['fmid' + k], BUILT['fmid_h'] = _build_fmid(lit)
        BUILT['mid' + k], BUILT['mid_h'] = _build_mid(lit)
        BUILT['ground' + k], BUILT['ground_h'] = _build_ground(lit)
        BUILT['fg' + k] = _build_fg(lit)
    return BUILT


build()
YS = np.arange(H)[:, None].astype(np.float64)


_RAYS = {}


def sun_rays(t, w):
    """Shafts of low sunlight fanning up/left from the disc through the dust."""
    if w not in _RAYS:
        _RAYS[w] = L.RayField(SKY_H, w, SUN[0], SUN[1])
    sway = 0.012 * math.sin(t * 0.6)
    angs = [math.pi + 0.16 + sway, math.pi + 0.52 - sway, -math.pi / 2 - 0.55 + sway, -0.35]
    wid = [0.13, 0.16, 0.14, 0.12]
    return _RAYS[w].level(angs, wid, 170, peak=1.25, gamma=1.3, inner=SUN_R + 4, inner_soft=20.0)


def scrolled_m(lay, offx):
    """Like L.scrolled but with a 1px margin on both sides (camera rumble)."""
    ix = int(math.floor(offx)) - 1
    fr = offx - (ix + 1)
    ix = max(0, min(ix, lay.shape[1] - (W + 4)))
    return lay[:, ix:ix + W + 4], fr + 1.0, ix


def blend_ld(name, off, dmask):
    """Crop the lit and dim twins of a layer and dither-dissolve between them."""
    a, fr, ix = scrolled_m(BUILT[name + 'L'], off)
    b, _, _ = scrolled_m(BUILT[name + 'D'], off)
    if dmask is None:
        return a.copy(), fr, ix
    if dmask is True:
        return b.copy(), fr, ix
    h_, w_ = a.shape[:2]
    m = dmask[:h_, :w_] if dmask.shape[1] >= w_ else np.pad(dmask, ((0, 0), (0, w_ - dmask.shape[1])), mode='edge')[:h_]
    out = np.where(m[..., None], b, a)
    return out, fr, ix


_B8 = bayer(H, W + 40, 8)
_XS = np.arange(W + 40)[None, :].astype(np.float64)


def shade_front(t):
    """x position of the storm's shadow edge sweeping over the land (right
    -> left) as the wall climbs over the sun."""
    return 400.0 - 560.0 * ease.in_out_sine((t - 4.55) / 1.0)


_YSF = np.arange(H)[:, None].astype(np.float64)
# ragged edge of the storm's shadow (fixed shape that rides with the front)
_RAG = 14.0 * (noise.fbm1(np.arange(H) / 9.0, 404, 3)[:, None] - 0.5) * 2.0


_DSUN = np.sqrt(((_XS - SUN[0]) * 0.8) ** 2 + ((np.arange(H)[:, None] - SUN[1]) * 1.0) ** 2)


def sky_dissolve_mask(t):
    """The sky loses its glow only once the wall's crest has covered the
    disc (~4.82): the dimming spreads outward from the hidden sun."""
    r = 560.0 * ease.in_out_sine((t - 4.78) / 1.0) - 30.0
    if r <= -29.0:
        return None
    if r > 700:
        return True
    a = np.clip((r - _DSUN) / 34.0 + 0.5, 0, 1)
    return _B8 < a


def dissolve_mask(d, ox=0, t=None):
    if d <= 0.001:
        return None
    if d >= 0.999:
        return True
    xf = shade_front(t)
    # slanted: the far ground (near the horizon) falls into shadow first
    edge = xf + (_YSF - 150.0) * 0.9 + _RAG
    a = np.clip((_XS - edge) / 30.0 + 0.5, 0, 1)
    return _B8 < a


# ------------------------------------------------------------------ the storm wall
WALL_FOOT = HORIZON + 3
_r = L.rng(505)
_LOBES = []
for i in range(150):
    v = _r.uniform(0.0, 1.0) ** 1.6              # most lobes crowd the crest
    _LOBES.append(dict(u=_r.uniform(-0.02, 1.08), v=v, r=_r.uniform(0.5, 1.0),
                       ph=_r.uniform(0, 6.28), sp=_r.uniform(0.7, 1.3)))
_LOBES.sort(key=lambda b: b['v'])            # v: 0 = crest (back) ... 1 = low front
_YY, _XX = np.mgrid[0:H, 0:W + 4]


def wall_state(t):
    u = max(0.0, (t - 3.92) / 2.1)
    hmax = 150.0 * min(u, 1.35) ** 1.45
    lead = 328.0 - 190.0 * ease.out_quad(min(u, 1.3) / 1.3)
    return hmax, lead


def wall_profile(xs, t, hmax, lead):
    env = np.clip((xs - lead) / 26.0, 0, 1) ** 0.5 * (0.62 + 0.38 * np.clip((xs - lead) / 150.0, 0, 1) ** 0.8)
    roll = 0.07 * np.sin(xs / 21.0 + t * 0.8) + 0.05 * np.sin(xs / 8.5 - t * 1.5)
    return WALL_FOOT - hmax * env * (0.93 + roll), env


def storm_wall(t, w):
    """Colossal haboob: a steep advancing front of dust lobes, dark, with a
    hot silver lining where the swallowed sun sits behind it."""
    img = layer(H, w)
    hmax, lead = wall_state(t)
    if hmax < 0.6:
        return img, None, lead
    xs = np.arange(w).astype(np.float64)
    top, env = wall_profile(xs, t, hmax, lead)
    yy, xx = _YY[:, :w], _XX[:, :w]
    body = (yy >= top[None, :]) & (yy <= WALL_FOOT + 2) & (env[None, :] > 0.01)
    L.put(img, body, 'ochre0')
    sun_behind = min(1.0, max(0.0, (t - 4.30) / 0.45)) * max(0.25, 1.0 - (t - 5.4) / 1.2)
    th = bayer(H, w, 4)
    for b in _LOBES:
        x = lead - 6 + b['u'] * (w + 30 - lead) - 5.0 * t * b['sp']
        if x < lead - 10 or x > w + 14:
            continue
        xi = int(min(max(x, 0), w - 1))
        local_h = WALL_FOOT - top[xi]
        if local_h < 4:
            continue
        rr = (2.8 + 0.085 * local_h * b['r'] + 2.0 * b['r']) * (1.0 + 0.08 * math.sin(t * 1.5 * b['sp'] + b['ph']))
        cy = top[xi] + rr * 0.75 + b['v'] * (local_h - rr * 1.6) * 0.8
        x0, x1 = int(max(0, x - rr - 1)), int(min(w, x + rr + 2))
        y0, y1 = int(max(0, cy - rr - 1)), int(min(H, cy + rr + 2))
        if x0 >= x1 or y0 >= y1:
            continue
        dx = xx[y0:y1, x0:x1] - x
        dy = yy[y0:y1, x0:x1] - cy
        rn = np.sqrt(dx * dx + dy * dy) / rr
        disc = (rn <= 1.0 + 0.3 / rr) & (yy[y0:y1, x0:x1] <= WALL_FOOT + 1)
        if not disc.any():
            continue
        body[y0:y1, x0:x1] |= disc
        sub = img[y0:y1, x0:x1]
        crest = (cy - rr) <= top[xi] + 2.5
        # backlit mass: dark face; faint ambient sky-light on lobe tops;
        # a thin darker crescent under each lobe separates the billows
        upper = (cy - (WALL_FOOT - hmax)) < 0.6 * hmax
        L.put(sub, disc, 'ochre0')
        if crest or (upper and b['r'] > 0.72):
            L.put(sub, disc & (dy < -0.5 * rr) & (rn > 0.62) & (rn < 1.0 - 0.3 / rr + 0.3), 'ochre1')
        if upper:
            L.put(sub, disc & (rn > 1.0 - 1.05 / rr) & (dy > 0.1 * rr) & (dx < 0.5 * rr), 'dust0')
        if x < lead + 22:
            # the front bulges catch the last western glow on their left flanks
            L.put(sub, disc & (rn > 1.0 - 1.2 / rr) & (dx < -0.55 * rr) & (dy < 0.3 * rr), 'rust1')
        if crest:
            edge = disc & (rn > 1.0 - 1.3 / rr)
            lam = (dx * 0.75 - dy * 0.65) / rr
            dsun = math.hypot((x - SUN[0]) * 0.8, cy - rr - SUN[1])
            heat = math.exp(-(dsun / (30.0 + 44.0 * sun_behind)) ** 2) * (0.3 + 1.0 * sun_behind)
            rim = edge & (lam > 0.2) & (dy < 0.25 * rr)
            if heat > 0.75:
                L.put(sub, rim, 'rust4')
                L.put(sub, rim & (lam > 0.7) & (rn > 1.0 - 0.7 / rr), 'core2')
            elif heat > 0.42:
                L.put(sub, rim, 'rust3')
            elif heat > 0.14:
                L.put(sub, rim, 'rust2')
            else:
                L.put(sub, rim & (dy < -0.4 * rr), 'ochre1')
    # dust thickening toward the ground catches the last light: lighter foot
    fz = np.clip((yy - (WALL_FOOT - 0.40 * hmax)) / max(8.0, 0.40 * hmax), 0, 1)
    fq = dither_levels(np.broadcast_to(fz * 2.2, (H, w)), 3, n=4, band=0.35)
    L.put(img, body & (fq == 1), 'ochre1')
    L.put(img, body & (fq == 2), 'ochre2')
    # dusty lighter foot (lit dust near the ground) and a soft top of the skirt
    foot = body & (yy > WALL_FOOT - 9)
    L.put(img, foot & (th < 0.5), 'ochre2')
    L.put(img, body & (yy > WALL_FOOT - 4), 'ochre2')
    img[~body] = 0
    return img, top, lead


def gust_front_fog(t, lead, w, ox_world, factor):
    """Amount map: the low dust front racing ahead of the wall along the
    ground (engulfs the distance first)."""
    hmax, _ = wall_state(t)
    if hmax < 2:
        return None
    xs = np.arange(w)[None, :].astype(np.float64)
    ys = np.arange(H)[:, None].astype(np.float64)
    ahead = np.clip(1.0 - (lead - 40 - xs) / 90.0, 0, 1)
    v = np.clip((ys - (WALL_FOOT - 8 - 0.12 * hmax)) / 12.0, 0, 1) * np.clip((142 - ys) / 14.0, 0, 1)
    return np.clip(ahead * v * min(1.0, hmax / 40.0), 0, 0.6)


FOG_GUST = L.Fog('ochre2', levels=6, ramps=MUTED)


# ------------------------------------------------------------------ wind FX
def gust(t):
    base = wind_amount(t)
    return float(np.clip(base * (0.75 + 0.35 * noise.vnoise1(np.array([t * 0.9 + 3.0]), 81)[0]), 0, 1))


def spindrift(img, hh, x_off, t, amount, seed, color):
    """Sand smoking off dune crests, blown left (drawn on a layer crop)."""
    if amount <= 0.02:
        return
    h_, w_ = img.shape[:2]
    crests = np.where(np.diff(np.sign(np.gradient(hh))) > 0)[0]      # local minima of y = crest tops
    col = P.C(color)
    for ci in crests:
        cxw = ci
        cx0 = cxw - x_off
        if cx0 < -40 or cx0 > w_ + 5:
            continue
        cy0 = hh[ci]
        for k in range(int(7 * amount) + 1):
            ph = noise.rand(ci, k, seed)
            cyc = t * (1.6 + ph) + ph * 5
            age = cyc - math.floor(cyc)
            dist = age * (18 + 30 * amount)
            px = cx0 - dist - k * 1.5
            py = cy0 - 1 - age * (3 + 3 * ph) + 0.8 * math.sin(age * 9 + k)
            if (int(age * 10) + k) % 3 == 2 and age > 0.5:
                continue
            ix_, iy_ = int(round(px)), int(round(py))
            if 0 <= ix_ < w_ and 0 <= iy_ < h_:
                img[iy_, ix_, :3] = col
                img[iy_, ix_, 3] = 255
                if age < 0.4 and ix_ + 1 < w_:
                    img[iy_, ix_ + 1, :3] = col
                    img[iy_, ix_ + 1, 3] = 255


BUSH = []
_rb = L.rng(909)
for i in range(15):
    a1 = _rb.uniform(0, 2 * math.pi)
    a2 = a1 + _rb.uniform(1.6, 3.6)
    BUSH.append((a1, a2, _rb.uniform(0.75, 1.0), _rb.uniform(0.75, 1.0), i % 4 == 0))


def tumble_bush(img, t, x_off):
    """A dry tumbleweed (pale bleached twigs) bounding in from the right as
    the wind rises — a foretaste of S5's tumbling bush."""
    t0 = T_STORM + 0.15
    if t < t0:
        return
    u = t - t0
    wx = 392.0 + WALL_OFF - 80.0 * u - 8.0 * u * u
    cx = wx - x_off
    ph = (u * 2.6) % 1.0                                  # hop rhythm
    bounce = 4 * ph * (1 - ph) * (7.0 - min(3.0, u * 1.2))
    R = 5.5
    cy = GROUND_Y + 3 - R - bounce
    rot = -u * 9.0
    if cx < -20 or cx > img.shape[1] + 20:
        return
    for (a1, a2, r1, r2, inner) in BUSH:
        x1, y1 = cx + math.cos(a1 + rot) * R * r1, cy + math.sin(a1 + rot) * R * r1 * 0.92
        x2, y2 = cx + math.cos(a2 + rot) * R * r2, cy + math.sin(a2 + rot) * R * r2 * 0.92
        draw.line(img, x1, y1, x2, y2, 'ochre1' if inner else 'ochre3')
    # a few stray twig ends poking out
    for k in range(3):
        a = rot * 1.0 + k * 2.1
        draw.pset(img, cx + math.cos(a) * (R + 1.5), cy + math.sin(a) * (R + 1.2), 'ochre3')
    # its shadow on the sand, shrinking as it hops up
    sw = max(1, int(4 - bounce * 0.4))
    draw.hline(img, cx - sw, cx + sw, GROUND_Y + 3, 'ochre1')


WALL_OFF = 0.0


def backlit_motes(img, t, amount_lit, wind, ox=0.0):
    x, y, r = fx.field(t, 34, 77, (0, 60, W + 60, 90), vel=(-6.0 - 40.0 * wind, 0.6),
                       jitter=(4.0, 1.5), sway=(1.5, 0.35))
    x = x - ox
    near_sun = np.exp(-(((x - SUN[0]) / 120.0) ** 2 + ((y - SUN[1]) / 60.0) ** 2))
    bright = (r < 0.5 + 0.5 * near_sun * amount_lit)
    fx.draw_field(img, x[bright], y[bright], r[bright], ['ochre6', 'ochre5', 'ochre4'])
    dark = ~bright
    fx.draw_field(img, x[dark], y[dark], r[dark], ['ochre3'])


# ------------------------------------------------------------------ robot staging
def robot_pose(t):
    """(anim, t_anim) for local time t."""
    if t < T_STOP1:
        return 'walk', t + WALK_PH1
    if t < T_LOOK_UP1:
        return 'look_down', t - T_STOP1
    if t < T_GO2:
        return 'look_down', max(0.0, 0.4 - (t - T_LOOK_UP1) * (0.4 / (T_GO2 - T_LOOK_UP1)))
    if t < T_STOP2:
        return 'walk', t - T_GO2 + WALK_PH2
    if t < T_LOOKUP:
        return 'idle', t - T_STOP2
    return 'look_up', t - T_LOOKUP


def robot_world_x(t):
    """The robot is pinned at integer screen x SCREEN_X; the world glides
    under it, so its world position is simply SCREEN_X + camera."""
    return SCREEN_X + cam(t)


def contact_times(t_to):
    """Footfalls (walk contact frames every 0.4 s of walk time) up to t_to."""
    out = []
    k = 0
    while True:                                  # segment 1 (and before the shot)
        tc = -16.0 + 0.4 * k - (WALK_PH1 % 0.4)
        k += 1
        if tc >= T_STOP1 or tc > t_to:
            break
        out.append(tc)
    k = 0
    while True:                                  # segment 2
        tc = T_GO2 + 0.4 * k
        k += 1
        if tc >= T_STOP2 or tc > t_to:
            break
        out.append(tc)
    return out


# ------------------------------------------------------------------ render
def render(t):
    return render_stage(t)


def rumble(t):
    a = 0.55 * ease.smoothstep((t - 5.25) / 0.75)
    if a <= 0.0:
        return 0.0
    return fx.shake(t, a, seed=24, freq=13.0)[0]


def render_stage(t):
    c = cam(t)
    d = dim_amount(t)
    dm = dissolve_mask(d, t=t)
    st = Stage(bg='ochre1')
    st.shake = (rumble(t), 0.0)
    crop, fr, ix = blend_ld('sky', c * F['sky'], sky_dissolve_mask(t))
    if d < 0.98:
        rays = sun_rays(t, crop.shape[1]) * (1.0 - d)
        crop = L.ramp_shift(crop, rays, band=0.35, ox=ix, rows=(0, SKY_H))
    st.add(crop, -fr, 0)
    crop, fr, ix = blend_ld('far', c * F['far'], dm)
    st.add(crop, -fr, 0)
    wall, wtop, lead = storm_wall(t, W + 4)
    st.add(wall, 0, 0)
    g = gust(t)
    wind = wind_amount(t)
    for name in ('fmid', 'mid'):
        crop, fr, ix = blend_ld(name, c * F[name], dm)
        gf = gust_front_fog(t, lead, crop.shape[1], ix, F[name])
        if gf is not None:
            crop = FOG_GUST.apply(crop, gf * (1.0 if name == 'fmid' else 0.6), band=0.3, ox=ix,
                                  mask=crop[..., 3] > 0)
        hh = BUILT['fmid_h'] if name == 'fmid' else BUILT['mid_h']
        spindrift(crop, hh, ix, t, max(0.0, (wind - 0.35) / 0.65), 31 if name == 'fmid' else 32,
                  'ochre4' if d < 0.5 else 'ochre3')
        if name == 'mid':
            L.sand_streams(crop, BUILT['mid_h'], ix, t, seed=41, n=10, gust=g * 0.8,
                           colors=['ochre4', 'ochre3', 'ochre3'], speed=(50, 80), length=(18, 40),
                           yoff=(-6, 1), amp=0.8)
        if name == 'mid':
            backlit_motes(crop, t, 1.0 - d, wind, ox=-fr)
        st.add(crop, -fr, 0)

    # ---- ground + robot. The robot is PINNED at an integer screen x on its
    # own layer; the world glides under it (no 10 Hz sub-pixel vibration).
    crop, fr, ix = blend_ld('ground', c * F['ground'], dm)
    anim, ta = robot_pose(t)
    spr = robot.frame(anim, ta, can='sprout', scale='medium', core=0.6)
    RX = int(SCREEN_X)
    rxw = robot_world_x(t)
    L.sand_streams(crop, BUILT['ground_h'], ix, t, seed=43, n=20, gust=g,
                   colors=['ochre5' if d < 0.5 else 'ochre4', 'ochre4', 'ochre3'],
                   speed=(60 + 50 * wind, 100 + 60 * wind), length=(26, 70), yoff=(-34, 1), amp=1.4)
    tumble_bush(crop, t, ix)
    # footprints: one print per real footfall, trailing behind in the sand
    pxs, pys, keep = [], [], []
    for j, tc in enumerate(contact_times(t)):
        wx = (robot_world_x(tc) if tc >= 0 else SCREEN_X + WALK_V * tc) + 2
        pxs.append(wx)
        pys.append(GROUND_Y + (j % 2))
        keep.append(noise.rand(j, 5, 91) > ((rxw - wx) / 160.0) ** 2)
    L.footprints(crop, ix, pxs, pys, 'ochre0', lit='ochre3' if d < 0.5 else None, w=2, keep=keep)
    # the chest porthole faces right: its warm light falls on the sand in
    # front of the robot (stronger as the daylight dies). Exact fractional
    # centre in ground-crop coords, so the pool stays put while the sand slides.
    cx_s, cy_s = anchor_pos(spr, RX, GROUND_Y, 'feet', 'core')
    crop = L.pool_local(crop, cx_s + 7 + fr, GROUND_Y + 1, 17 + 5 * d, peak=0.8 + 1.4 * d, mode='warm',
                        sy=0.3, gamma=1.5, band=0.45, ox=ix)
    st.add(crop, -fr, 0)

    # robot sprite layer (integer position)
    rl = layer(44, 56)
    RLX, RLY = RX - 24, GROUND_Y - 40
    blit(rl, spr, 24, 40, anchor='feet')
    rm = rl[..., 3] > 0

    # long shadow to the left, on its own layer pinned with the robot; its
    # pixels are the ground under it (sampled from the gliding crop) two
    # ramp steps darker, fading toward the tip (gone once the sun is).
    if d < 0.999:
        big = np.zeros((H, W + 8), bool)
        big[RLY:RLY + 44, RLX:RLX + 56] = rm
        shd = L.long_shadow(big, GROUND_Y, k_len=2.2, k_flat=0.26) & ~big
        ys_, xs_ = np.nonzero(shd)
        ok = xs_ < W
        ys_, xs_ = ys_[ok], xs_[ok]
        kx = np.floor(xs_ + fr + 0.5).astype(int)
        ok = (kx >= 0) & (kx < crop.shape[1])
        ys_, xs_, kx = ys_[ok], xs_[ok], kx[ok]
        opaque = crop[ys_, kx, 3] > 0
        ys_, xs_, kx = ys_[opaque], xs_[opaque], kx[opaque]
        if len(xs_):
            sl = layer(H, W)
            sl[ys_, xs_, :3] = crop[ys_, kx, :3]
            sl[ys_, xs_, 3] = 255
            fade = np.clip(1.0 - (RX - np.arange(W)[None, :]) / 70.0, 0, 1)
            lv = np.where(sl[..., 3] > 0, (-1.0 - 1.3 * fade) * (1 - d), 0.0)
            sl = L.ramp_shift(sl, lv, band=0.35, rows=(GROUND_Y - 2, GROUND_Y + 10))
            st.add(sl, 0, 0)

    # warm rim on its sun side, the sprout's green glow, the core's glow
    if d < 0.999:
        rim = L.rim_edges(rm, 1, 0)
        rim &= bayer(44, 56, 4) >= d
        L.put(rl, rim, 'rust3')
        top = rm & ~L.shift(rm, 0, 1) & (L.shift(rim, -1, 0) | L.shift(rim, -1, 1)) & (bayer(44, 56, 4) >= d)
        L.put(rl, top, 'rust4')
        L.put(rl, rim & ~L.shift(rm, 0, 1), 'rust4')
    mx, my = anchor_pos(spr, 24, 40, 'feet', 'can_mouth')
    # the sprout's tiny green glow swells while the robot gazes down at it
    tender = ease.smoothstep((t - (T_STOP1 + 0.2)) / 0.35) * (1.0 - ease.smoothstep((t - T_LOOK_UP1) / 0.4))
    rl = light.light_pool(rl, mx, my - 2, 2.6 + 1.4 * tender, peak=1.3 + 0.8 * tender, mode='green',
                          gamma=1.2, band=0.5)
    ex, ey = anchor_pos(spr, 24, 40, 'feet', 'core')
    rl = light.light_pool(rl, ex, ey, 6, peak=1.2, mode='warm', gamma=1.3, band=0.5)
    rl[..., 3] = np.where(rm, 255, 0).astype(np.uint8)
    st.add(rl, RLX, RLY)

    crop, fr, ix = blend_ld('fg', c * F['fg'], dm)
    st.add(crop, -fr, 0)
    return st
