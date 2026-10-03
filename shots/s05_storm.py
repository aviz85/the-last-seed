"""
S5 — THE STORM (global 26.0-32.0, local t 0-6).  Owner: STORM / FX artist.

The film's action peak.  A haboob slams into the Gardener on a dune ridge; it
braces, turns its back and curls around the sprout, is struck by a flying
sheet of metal (antenna ripped off), its heart stutters, and it rises and
trudges on, uphill into the wind, as the dust thickens into a whiteout.

Staging: low camera; the robot (medium, 26 px) stands on a dune crest at the
left third, silhouetted against the churning storm sky; sand smokes off the
ridge; the dark dune face fills the bottom of frame.

Beats (local t = global - 26.0):
  0.00  cut (continuous with S4's last shot): a colossal, horizon-wide haboob.
        Its far face spans the horizon behind the robot, the churning top edge
        lit by the dying sun from the upper left, low on the left and climbing
        out of frame to the right; it rises as the wall arrives. Its foot — a
        low rolling surge of billows — sweeps in along the ridge from the right.
  0.33  the surge rolls over the robot (shake, shove); darkness behind it  'brace'
  1.28  the gust rips a branch off the dead tree on the ridge
  1.42  a huge gust blasts through (speed lines, haze wave, shake)
  1.50  -> 'shield' (back to the wind, curled over the can)            cue shield 27.5
  2.10  the metal sheet enters, tumbling
  2.50  it strikes the head: 2 impact frames + 2-frame hit-stop, heavy
        shake, sparks sprayed down-left; 'damaged'. The antenna pops off
        straight up (slow first frames), glows, arcs away left with an amber
        trail and fades                                                cue antenna_rip 28.5
  3.60  core stutters (3 drop-outs, stub sparks) -> dim 0.35           cue core_flicker 29.6
  4.00  rises: 'unshield', turns back into the wind                    cue trudge 30.0
  4.50  trudges on uphill ('brace', damaged); a pale wall of dust rolls
        in from the right, swallows it (~5.1) and it glimmers on as a
        ghost: amber heart + green sprout, the only colour
  5.65  (film fades to black) — frames are already deep in the whiteout

Clouds are billowing MASSES (lib_storm.draw_masses): the union height field of
circulating, pulsing lobes, lit from the upper left with shadowed undersides
and creases, quantised to stepped ramps with dithered seams.

Depth stack (back -> front), each block followed by its own palette-true fog:
  murk backdrop, haze veils | far rolling banks | mid surges, sky streaks |
  dune face, spindrift, sand ribbons | [pre-storm world left of the front] |
  robot, debris, sheet, antenna, sand spill | amber core light in the dust |
  sparks, stones, near grit, roof & floor banks | whiteout + pale front |
  impact frames.
Composited in palette-index space (lib_storm) on an overscanned canvas, so the
camera shake never shows an edge; returned as a Stage (sub-pixel shake).
"""
import math
import numpy as np

from engine import *
from engine import ease, fx
from engine import palette as PP
from engine.noise import rand, vnoise1, fbm1
from chars import robot
from shots import lib_storm as L
from shots.lib_storm import I, STEP
from engine.draw import line_points

DURATION = 6.0

# ------------------------------------------------------------------ layout (canvas coords)
M = 8                                   # overscan margin for camera shake
CW, CH = W + 2 * M, H + 2 * M           # 336 x 196
HY = 150 + M                            # true horizon: hidden behind the dune (low camera)
GY = 140 + M                            # dune crest height under the robot's feet
RX = 104 + M                            # robot feet x
YS = np.arange(CH, dtype=np.float32)[:, None]
XS = np.arange(CW, dtype=np.float32)[None, :]


def _crest():
    """The dune ridge: rises gently to the right (uphill into the wind)."""
    x = np.arange(CW, dtype=np.float64)
    wig = 2.0 * (fbm1(x / 55.0, 21, 3) - float(fbm1(np.array([RX / 55.0]), 21, 3)[0]))
    return (GY - 0.075 * (x - RX) + wig).astype(np.float32)


CREST = _crest()
BELOW = YS >= np.round(CREST)[None, :]          # dune face mask (static)


def crest_y(x):
    return float(np.interp(x, np.arange(CW), CREST))


# ------------------------------------------------------------------ beats
FRAME = 1.0 / FPS
T_FRONT = 0.33
T_GUST = 1.42
T_SHIELD = 1.5
T_SHEET_IN = 2.10
T_IMPACT = 2.5
T_FLICK = 3.6
T_RISE = 4.0
T_BRACE2 = 4.5                           # T_RISE + duration('unshield')
HITSTOP = 2                              # frames frozen at the impact
EPS = 1e-4                               # beat comparisons are robust to float time (k/30 - 26.0)


F_IMPACT = int(round(T_IMPACT * FPS))    # frame 75: impact frame; 76: flash frame


def frame_no(t):
    return int(math.floor(t * FPS + 1e-4))


def world_time(t):
    """Hit-stop: the world freezes for 2 frames at the impact."""
    if F_IMPACT <= frame_no(t) < F_IMPACT + HITSTOP:
        return T_IMPACT
    return t


def _gauss(t, c, w):
    return math.exp(-((t - c) / w) ** 2)


def _wind_raw(t):
    """Wind strength (~1 = storm). Pure function of t."""
    base = 1.0 + 0.4 * (float(fbm1(np.array([t * 1.4]), 41, 3)[0]) - 0.5)
    calm = ease.smoothstep((t - (T_FRONT - 0.28)) / 0.3)
    w = 0.35 + (base - 0.35) * calm
    w += 0.6 * _gauss(t, T_FRONT + 0.06, 0.13)                        # the front slams in
    g = ease.smoothstep((t - (T_GUST - 0.08)) / 0.18) * (1 - ease.smoothstep((t - 1.75) / 0.9))
    w += 1.0 * g                                                      # the huge gust
    w += 0.5 * ease.smoothstep((t - 3.9) / 2.0)                       # whiteout build
    return w


_WT = np.arange(-1.0, 8.0, 1.0 / 480)
_WV = np.array([_wind_raw(float(x)) for x in _WT])


def wind(t):
    return float(np.interp(t, _WT, _WV))


WINDX = L.Integral(wind)


def fog_level(t):
    """0 = clear air, 1 = total whiteout."""
    f = 0.14 + 0.14 * ease.smoothstep((t - 0.35) / 0.7)
    f += 0.12 * ease.smoothstep((t - T_GUST) / 0.4)
    f += 0.12 * ease.smoothstep((t - 2.2) / 2.0)                    # visibility keeps collapsing
    f += 0.5 * ease.smoothstep((t - 4.15) / 1.7)
    return min(f, 1.0)


def core_level(t):
    """0.6 -> three drop-outs at 3.6 (each re-ignition overshoots) -> dim 0.35
    (continuity: S6 0.3)."""
    if t < T_FLICK - EPS:
        return 0.6
    a = t - T_FLICK
    seq = [(0.0, 0.02), (0.066, 0.95), (0.10, 0.6), (0.133, 0.02), (0.20, 0.85), (0.233, 0.03),
           (0.30, 0.7), (0.333, 0.45), (0.366, 0.2), (0.40, 0.4), (0.5, 0.35)]
    v = 0.6
    for ts, lv in seq:
        if a >= ts - 1e-6:
            v = lv
    return v


def eye_level(t):
    if t < T_FLICK - EPS:
        return 1.0
    if t - T_FLICK < 0.34:
        return 0.15 if core_level(t) < 0.1 else 0.9
    return 0.8


def shake(t):
    w = wind(t)
    amp = 0.25 + 1.1 * max(0.0, w - 0.9)
    dx, dy = fx.shake(t, amp, seed=5, freq=12.0)
    dy *= 0.6
    a = t - T_FRONT                                                   # front slam kick
    if a > -0.02:
        k = 3.2 * math.exp(-max(a, 0) * 9)
        dx += -k * math.cos(2 * math.pi * 7 * max(a, 0))
        dy += 0.5 * k * math.sin(2 * math.pi * 9 * max(a, 0))
    a = t - T_IMPACT                                                  # the impact
    if a >= 0:
        k = 5.5 * math.exp(-a * 6.5)
        dx += -k * math.cos(2 * math.pi * 8.5 * a)
        dy += 0.6 * k * math.sin(2 * math.pi * 11 * a + 0.6)
    lim = M - 1
    return max(-lim, min(lim, dx)), max(-lim, min(lim, dy))


# ------------------------------------------------------------------ robot choreography
RX0 = RX - 4                        # start x: it gains ~6 px in the brace, then holds


def _shield_faces_left_natively():
    """Does the character anim's held 'shield' pose already face left (its back
    to the wind from the right)? Decided from the anchors, so the shot adapts
    to whatever the character artist delivers."""
    spr = robot.frame('shield', robot.duration('shield') + 0.3, can='sprout')
    return spr.anchor('eye')[0] < spr.anchor('feet')[0]


SHIELD_FLIP = not _shield_faces_left_natively()     # mirror only if the anim doesn't turn itself
HAS_UNSHIELD = 'unshield' in robot.ANIMS


def robot_state(t):
    """-> (anim, t_anim, variant, face_left, x, y) canvas coords (feet).
    Positions use robot.travel() so the feet never slide while stepping; the
    wind's shoves (front, gust, sheet) are brief sideways slides on the sand."""
    variant = 'damaged' if t >= T_IMPACT - EPS else 'normal'
    shove = 0.0
    a = t - T_FRONT
    if a > 0:
        shove -= 2.4 * min(1.0, a * 15) * math.exp(-a * 3.5)          # shoved by the front
    g = t - T_GUST
    if 0 < g < 0.2:
        shove -= 1.0 * math.sin(math.pi * g / 0.2)
    if t < T_SHIELD - EPS:
        x = RX0 + robot.travel('brace', t) + shove
        return 'brace', t, variant, False, x, crest_y(x)
    x_hold = RX0 + robot.travel('brace', T_SHIELD)
    if t < T_RISE - EPS:
        a = t - T_IMPACT
        if a >= 0:
            shove -= 2.0 * math.exp(-a * 8)                             # knocked by the sheet
        x = x_hold + shove
        return 'shield', max(0.0, t - T_SHIELD), variant, SHIELD_FLIP, x, crest_y(x)
    if t < T_BRACE2 - EPS:
        if HAS_UNSHIELD:                                                # rise: turn back into the wind
            return 'unshield', max(0.0, t - T_RISE), variant, False, x_hold, crest_y(x_hold)
        d = robot.duration('shield')                                    # (fallback: shield played back)
        ta = max(0.0, d - (t - T_RISE) * d / (T_BRACE2 - T_RISE))
        return 'shield', ta, variant, SHIELD_FLIP, x_hold, crest_y(x_hold)
    x = x_hold + robot.travel('brace', max(0.0, t - T_BRACE2))
    return 'brace', max(0.0, t - T_BRACE2), variant, False, x, crest_y(x)


def robot_sprite(t, with_levels=True):
    anim, ta, variant, face_left, x, y = robot_state(t)
    x = float(int(round(x)))
    if with_levels:
        spr = robot.frame(anim, ta, can='sprout', variant=variant, scale='medium',
                          core=core_level(t), eye=eye_level(t), face_left=face_left)
    else:
        spr = robot.frame(anim, ta, can='sprout', variant=variant, scale='medium', face_left=face_left)
    ax, ay = spr.anchor('feet')
    return spr, int(round(x)) - ax, int(round(y)) - ay, face_left


def robot_anchor(t, name):
    spr, px, py, _ = robot_sprite(t, False)
    qx, qy = spr.anchor(name)
    return px + qx, py + qy


# ------------------------------------------------------------------ LUTs
FOG_FAR = L.mix_lut('dust5', K=8, protect=0.2)
FOG_MID = L.mix_lut('dust4', K=8, protect=0.2)
FOG_WHITE = L.mix_lut('dust7', K=8, protect=0.35)
GLOW = L.mix_lut('core5', K=8, protect=1.0)
GREEN = L.mix_lut('leaf5', K=8, protect=1.0, warm_lock=False,
                  allowed=np.array(L.RI('leaf') + L.RI('ochre') + L.RI('dust') + L.RI('ink')))
FLASH = L.mix_lut('white0', K=8, protect=1.0)
DARKEN = L.mix_lut('ochre0', K=8, protect=0.5)
DUSK = L.mix_lut('rust1', K=8, protect=0.3)
UP4 = L.Up(CH, CW, 4)


# ------------------------------------------------------------------ static art
def _dune_strip():
    """The dune face below the crest, tileable 1024 wide (scrolled very slowly:
    ripples migrate). Lighter just under the crest, darker toward the viewer,
    wind ripples parallel to the ridge, spaced in perspective."""
    Wt = 1024
    xs = np.arange(Wt, dtype=np.float32)[None, :]
    crest = np.tile(CREST, 4)[:Wt][None, :]
    dy = np.clip(YS - crest, 0, None)                         # px below the crest
    depth = np.clip(dy / (CH - GY + 10), 0, 1)
    cols = [I('ochre0'), I('ochre1'), I('ochre2'), I('ochre3'), I('ochre4')]
    lv = 2.75 - 2.4 * depth ** 0.7
    z = np.sqrt(dy + 2.0)
    wob = L.TEX[YS.astype(int) & 255, (xs.astype(int) // 2) & 1023]
    ph = z * 1.35 + 0.9 * wob + xs / 140.0
    rip = np.sin(ph * 2 * np.pi)
    lv = lv + np.where(rip > 0.62, 0.8, 0.0) - np.where(rip < -0.72, 0.5, 0.0)
    lv[dy < 1.0] += 0.6                                      # crisp lit lip of the crest
    lv = np.clip(lv, 0, len(cols) - 1)
    q = L.quantize(lv, len(cols) - 1, band=0.3)
    return np.array(cols, np.uint8)[q]


DUNE = _dune_strip()


def _pre_storm():
    """The dim dusk world just before the wall hits (continuity with S4's end):
    a dust-choked rust-ochre dusk, faint ruined towers far away above the
    ridge, the dune in dusk shadow. Darkening toward the storm on the right."""
    cols = [I('ochre0'), I('rust0'), I('rust1'), I('rust2'), I('ochre2'), I('ochre3')]
    lv = np.clip((YS - 4) / (GY - 2), 0, 1) ** 1.35 * 4.9 + 0.2
    lv = lv - 1.8 * np.clip((XS - 60) / 260.0, 0, 1) * np.clip(YS / GY, 0.25, 1)
    lv = lv + 0.35 * (L.TEX[(YS.astype(int) // 4) & 255, (XS.astype(int) // 6) & 1023] - 0.5)
    img = np.array(cols, np.uint8)[L.quantize(np.clip(lv + 0 * XS, 0, len(cols) - 1), len(cols) - 1, band=0.55)]
    # far ruined towers: one step darker than the sky behind them, small
    sil = np.zeros((CH, CW), bool)
    for bx, bw, top in ((22, 3, GY - 17), (26, 1, GY - 11), (34, 2, GY - 8), (63, 2, GY - 13),
                        (66, 1, GY - 9), (88, 1, GY - 6)):
        sil[int(top):int(GY + 12), bx:bx + bw] = True
    sil[int(GY - 14):int(GY - 13), 20:26] = True           # a broken girder
    sil[int(GY - 5):int(GY + 12), 40:62] = True             # a low ruin line
    sil[int(GY - 7):int(GY - 5), 44:52] = True
    img[sil] = STEP[-1][img[sil]]
    dune = DUNE[:, :CW]
    img[PYLON] = STEP[-1][img[PYLON]]
    _PRE_SIL[:] = (sil | PYLON) & ~BELOW
    img = np.where(BELOW, L.apply_lut(dune, 2.5, DUSK, band=1.0), img)
    return img



# ------------------------------------------------------------------ layers
def backdrop(t, X, fog):
    """Storm murk: heavy dark roof, glowing band low behind the ridge, streaky
    haze veils sliding left."""
    xc = UP4.xc[None, :]
    yc = UP4.yc[:, None]
    z = t * 0.4
    veils = L.fbm3((xc + X * 24.0) / 160.0, yc / 10.0, z, 5, 3)
    swirl = L.fbm3((xc + X * 44.0) / 75.0, yc / 28.0, z * 1.7, 9, 3)
    base = np.clip((yc - 4) / (GY - 12), 0, 1) ** 1.35 * 4.3 + 0.2
    glow = 0.9 * np.exp(-(((xc - 250) / 150.0) ** 2 + ((yc - GY + 12) / 40.0) ** 2))
    glow *= 1 - ease.smoothstep((t - 2.0) / 2.5)
    lvc = base + glow + (veils - 0.5) * 2.0 + (swirl - 0.5) * 1.1
    lvc = lvc * (1 - 0.35 * fog) + 3.6 * 0.35 * fog                 # the murk flattens
    lv = UP4(lvc)
    cols = [I('ochre0'), I('ochre1'), I('ochre2'), I('ochre3'), I('ochre4'), I('ochre5')]
    q = L.quantize_px(np.clip(lv, 0, len(cols) - 1), len(cols) - 1, w_px=1.2, ox=int(X * 24.0))
    return np.array(cols, np.uint8)[q]


SKY_T = [I('ochre0'), I('ochre1'), I('ochre2'), I('ochre3'), I('ochre4')]
BANK_T = [I('ochre2'), I('ochre3'), I('ochre4'), I('ochre5'), I('ochre6')]


def sky_billows(img, t, X, fog):
    """Huge dust masses churning through the middle of the sky: billowing
    lobes rolling over each other, lit from the upper left, dark undersides."""
    ms = L.masses_row_v3(t, X * 62 + 300, 404, 175, 70, 160, 240, CW, yjit=20, n=(13, 17),
                         rfrac=(0.16, 0.3), aspect=0.36, sq=0.7, roll=0.3,
                         gate=lambda i: L.rand1(i, 6, 404) > 0.25)
    L.draw_masses(img, ms, SKY_T, ox=int(X * 62), band=0.45, bump=0.16, under=1.3, gain=1.5, bias=1.9,
                  tex_amp=0.3, crease=0.6)


def far_banks(img, t, X):
    """Two rows of rolling dust banks along the hidden horizon behind the ridge."""
    ms = L.masses_row_v3(t, X * 28, 101, 50, GY - 14, 46, 84, CW, yjit=7, n=(5, 8), rfrac=(0.3, 0.52),
                         aspect=0.55, roll=0.5)
    L.bank_body(img, [(m['cx'], m['cy'] + m['h'] * 0.15, 1, 1, 0) for m in ms], I('ochre4'), y_to=HY)
    L.draw_masses(img, ms, BANK_T[:4], ox=int(X * 28), band=0.42, bump=0.2, under=1.1, gain=1.8, bias=1.9)
    ms2 = L.masses_row_v3(t, X * 52 + 17, 202, 56, GY - 2, 40, 76, CW, yjit=5, n=(5, 7),
                          rfrac=(0.28, 0.5), aspect=0.55, roll=0.6, gate=lambda i: L.rand1(i, 9, 202) > 0.3)
    L.bank_body(img, [(m['cx'], m['cy'] + m['h'] * 0.15, 1, 1, 0) for m in ms2], I('ochre3'), y_to=HY)
    L.draw_masses(img, ms2, BANK_T[:4], ox=int(X * 52), band=0.42, bump=0.2, under=1.4, gain=1.9, bias=1.6)


def far_veils(img, t, X):
    L.streak_lut_field(img, 60, 505, (0, 14, CW, GY - 20), X * 80, (24, 70), STEP[1],
                       wave=(3, 150, 0.2),
                       density=lambda x, y: L.TEX[(y.astype(int) * 3) & 255, ((x + X * 30) / 2).astype(int) & 1023],
                       dens_thresh=0.34)


def mid_surges(img, t, X, fog):
    """Dust surges rolling along just behind the ridge (bottoms hidden by it)."""
    k = 0.8 + 0.7 * fog + 0.4 * _gauss(t, T_GUST + 0.2, 0.35)
    ms = L.masses_row_v3(t, X * 110 + 60, 303, 44, GY + 2, 22 * k, 40 * k, CW, yjit=3, n=(4, 6),
                         rfrac=(0.3, 0.55), aspect=0.6, roll=0.9,
                         gate=lambda i: np.sin(i * 0.9 + L.rand1(i, 8, 303) * 2.0) > -0.2)
    if not ms:
        return
    for m in ms:                                       # sit them on the ridge line
        dyc = crest_y(np.clip(m['cx'], 0, CW - 1)) - GY
        m['cy'] += dyc
        m['lobes'] = [(x, y + dyc, r, s_, z) for (x, y, r, s_, z) in m['lobes']]
        b = m['box']; m['box'] = (b[0], b[1] + dyc, b[2], b[3] + dyc)
    L.bank_body(img, [(m['cx'], m['cy'] + m['h'] * 0.2, 1, 1, 0) for m in ms], I('ochre3'), y_to=HY)
    L.draw_masses(img, ms, BANK_T[:4], ox=int(X * 110), band=0.4, bump=0.22, under=1.3, gain=1.9, bias=1.7)


def sky_streaks(img, t, X):
    w = wind(t)
    L.streak_field(img, t, int(22 + 30 * w), 606, (0, 24, CW, GY - 20), X * 230, (4 * w, 14 * w),
                   [I('ochre5'), I('dust7'), I('ochre4')], weights=[4, 1, 2],
                   wave=(2.5, 90, 0.25), slope=0.06,
                   density=lambda x, y: L.TEX[(y.astype(int) * 2 + 40) & 255, ((x + X * 60) / 3).astype(int) & 1023],
                   dens_thresh=0.36)


def dune(img, t, X):
    off = int(X * 3) % 1024
    cols = (np.arange(CW) + off) % 1024
    d = DUNE[:, cols]
    img[BELOW] = d[BELOW]


def spindrift(img, t, X):
    """Sand smoking off the ridge line: a dithered haze hugging the crest whose
    density streams left with the wind, plus grains lifted off the crest."""
    w = wind(t)
    dy = np.clip(CREST[None, :] - YS, 0, None)                      # px above the crest
    ok = (dy > 0) & (dy < 10 + 14 * w)
    band = L.TEX[(YS.astype(int) // 3 + 11) & 255, ((XS + X * 260) / 3).astype(int) & 1023]
    alpha = np.clip((band - 0.35) * 2.2, 0, 1) * np.exp(-dy / (2.0 + 4.0 * w)) * min(1.0, 0.4 + 0.5 * w)
    m = ok & L.dmask(alpha, ox=int(X * 260))
    img[m] = STEP[1][img[m]]
    m2 = ok & L.dmask(alpha * 0.55 - 0.15, ox=int(X * 260) + 2, oy=1)
    img[m2] = STEP[1][img[m2]]
    n = 70                                                          # grains leaving the crest
    i = np.arange(n)
    rate = 1.6 + 1.2 * L.rand1(i, 1, 616)
    ph = t * rate + L.rand1(i, 2, 616)
    cyc = np.floor(ph)
    age = (ph - cyc) * 0.5
    x0 = L.rand1(i * 131 + cyc.astype(np.int64), 3, 616) * (CW + 60)
    x = x0 - (150 + 120 * L.rand1(i, 4, 616)) * age * w
    y = np.interp(np.clip(x0, 0, CW - 1), np.arange(CW), CREST) - 1 - 22 * age * (0.4 + L.rand1(i, 5, 616)) - 3 * age
    ln = (1 + 3 * w * L.rand1(i, 6, 616)).astype(int)
    for s in range(int(ln.max())):
        live = ln > s
        L.put_points(img, x[live] + s, y[live], I('dust6'))


def sand_ribbons(img, t, X):
    """Saltation ribbons racing along the dune face."""
    w = wind(t)
    L.streak_lut_field(img, int(40 + 25 * w), 404, (0, GY - 6, CW, CH - GY + 6), X * 280, (12 * w, 40 * w), STEP[1],
                       wave=(1.2, 70, 0.3),
                       keep=lambda x, y: BELOW[np.clip(y.astype(int), 0, CH - 1), np.clip(x.astype(int), 0, CW - 1)])


def robot_plane_sand(img, t, X):
    """Grains skimming past at the robot's depth (1-4 px dashes)."""
    w = wind(t)
    L.streak_field(img, t, int(25 + 25 * w), 707, (0, GY - 26, CW, 30), X * 330, (1, 5 * w),
                   [I('ochre4'), I('ochre5'), I('dust6')], weights=[3, 3, 1], slope=0.1,
                   wave=(2.0, 40, 0.2))


def near_streaks(img, t, X, fog):
    """Near grit: short fat dashes of sand clumps whipping past the lens."""
    L.streak_field(img, t, 12, 808, (0, 0, CW, CH), X * 720, (4, 12),
                   [I('ochre0'), I('ochre1'), I('dust6')], weights=[2, 2, 1],
                   wave=(5, 220, 0.1), slope=0.12,
                   density=lambda x, y: L.TEX[(y.astype(int) + 90) & 255, ((x + X * 90) / 4).astype(int) & 1023],
                   dens_thresh=0.25, thick=1)
    L.streak_field(img, t, 5, 809, (0, 0, CW, CH), X * 760, (3, 6),
                   [I('ochre0'), I('ochre1')], wave=(5, 220, 0.1), slope=0.12, thick=2)


ROOF_T = [I('ochre0'), I('ochre0'), I('ochre1'), I('ochre2'), I('ochre3')]


def near_banks(img, t, X, fog, inside=None):
    """The storm's roof (hanging billows racing overhead) and dark near dust
    rolling past the bottom of frame."""
    k = 1.0 + 0.3 * fog + 0.3 * _gauss(t, T_GUST + 0.15, 0.3)
    ms = L.masses_row_v3(t, X * 190, 818, 64, M - 4, 80 * k, 150 * k, CW, yjit=9, n=(8, 11),
                         rfrac=(0.2, 0.4), aspect=0.36, sq=0.66, roll=0.6, hang=True)
    clip = None if inside is None else (lambda xs, ys: inside[np.clip(ys, 0, CH - 1), np.clip(xs, 0, CW - 1)])
    body = np.zeros((CH, CW), bool)
    L.bank_body(np.zeros((CH, CW), np.uint8), [(m['cx'], m['cy'] - m['h'] * 0.1, 1, 1, 0) for m in ms], 1,
                hang=True, mask_out=body)
    if inside is not None:
        body &= inside
    img[body] = I('ochre0')
    L.draw_masses(img, ms, ROOF_T, ox=int(X * 190), band=0.45, bump=0.12, under=-0.2, gain=1.3, bias=1.35,
                  left_lit=0.25, crease=0.5, tex_amp=0.3, clip_fn=clip)
    ms2 = L.masses_row_v3(t, X * 340, 919, 90, CH + 10, 34, 60, CW, yjit=4, n=(4, 6),
                          rfrac=(0.28, 0.5), aspect=0.45, roll=0.9, gate=lambda i: L.rand1(i, 5, 919) > 0.45)
    if ms2:
        L.draw_masses(img, ms2, [I('ochre0'), I('ochre0'), I('ochre1'), I('ochre1')], ox=int(X * 340),
                      band=0.5, bump=0.06, under=0.8, gain=0.9, bias=1.2, crease=0.2, edge_px=2.4, fray=1.4,
                      clip_fn=clip)


T_GUST_HIT = 1.47          # the gust sheet reaches the robot
GUST_V = 900.0


def gust_x(t):
    return RX + GUST_V * (T_GUST_HIT - t)


def gust_edge(t):
    """Per-row x of the gust's leading edge (leans forward at the top, billowy)."""
    y = np.arange(CH, dtype=np.float32)
    bil = 22.0 * (L.fbm3(0.0 * y + 3.1, y / 13.0, t * 3.0, 88, 2) - 0.5)
    return gust_x(t) - 0.3 * (GY - y) + bil


def gust_sheet(img, t, X, behind=False):
    """The huge gust: a haze wave of blasting dust with a billowy leading
    edge, washing over everything right -> left. Drawn twice: behind the robot
    (denser) and in front of it (thinner, plus a burst of long speed lines)
    so it reads as volume."""
    a = t - (T_GUST_HIT - 0.32)
    if a < 0 or a > 1.1:
        return img
    xe = gust_edge(t)
    d = XS - xe[:, None]
    streaky = L.TEX[(YS.astype(int) + (60 if behind else 140)) & 255, ((XS + X * 420) / 9).astype(int) & 1023]
    body = np.clip(d / 24.0, 0, 1) * np.clip(1 - d / 260.0, 0, 1)
    fade = 1 - ease.smoothstep((a - 0.5) / 0.55)
    hgt = np.clip((YS - 10) / 100.0, 0, 1)
    k = 0.9 if behind else 0.45
    alpha = body * np.clip((streaky - 0.3) * 2.2, 0, 1) * fade * hgt * k
    ox = int(X * 420)
    m = L.dmask(alpha, ox=ox)
    img[m] = STEP[1][img[m]]
    m2 = L.dmask(alpha - 0.45, ox=ox + 1, oy=2)
    img[m2] = STEP[1][img[m2]]
    if not behind and fade > 0.05:
        L.streak_field(img, t, 70, 1313, (0, 0, CW, CH), X * 700, (22, 64),
                       [I('ochre5'), I('dust6'), I('ochre4'), I('dust7')], weights=[3, 2, 2, 1],
                       wave=(3, 160, 0.1), slope=0.04,
                       keep=lambda x, y: (x > xe[np.clip(y.astype(int), 0, CH - 1)] - 4) &
                                         (x < xe[np.clip(y.astype(int), 0, CH - 1)] + 60 + 240 * fade))
    return img


# ------------------------------------------------------------------ the haboob (0 - ~1.1)
# Continuous with S4's last shot: a colossal, horizon-wide wall of billowing
# dust. Its far face spans the whole horizon behind the robot, the lit,
# churning top edge low on the left and climbing out of frame to the right;
# the edge rises as the wall arrives. Its foot — a low, wide rolling surge —
# sweeps in along the ground from the right and engulfs the robot at T_FRONT;
# behind it everything is the dark inside of the storm.
SURGE_V = 600.0
WALL_EDGE_T = [I('ochre0'), I('ochre1'), I('ochre2'), I('ochre3'), I('ochre4'), I('ochre5')]
WALL_DEEP_T = [I('ochre0'), I('ochre0'), I('ochre1'), I('ochre2'), I('ochre3')]
SURGE_T = [I('ochre0'), I('ochre1'), I('ochre2'), I('ochre3'), I('ochre4')]


def front_x(t):
    """x of the surge's nose at the crest height (canvas); reaches the robot at T_FRONT."""
    return RX + SURGE_V * (T_FRONT - t)


def wall_edge(t):
    """Per-column y (canvas) of the far wall's top edge."""
    x = np.arange(CW, dtype=np.float32)
    rise = 150.0 * ease.in_out_sine(min(max(t / 0.7, 0.0), 1.0))
    return 112.0 - rise - 175.0 * (x / CW) ** 0.85


def surge_edge(t):
    """Per-row x of the inside boundary: the rolling surge along the ground (its
    billowing face slopes back with height), then the near wall body whose
    boundary tilts far to the right higher up."""
    y = np.arange(CH, dtype=np.float32)
    h = GY - y
    bil = 12.0 * (L.fbm3(0.0 * y + 5.3, y / 14.0, t * 2.5, 77, 2) - 0.5)
    return front_x(t) + np.where(h < 46, 0.2 * np.maximum(h, -10), 9 + 1.05 * (h - 46)) + bil


def _on_edge(ms, edge_y, dy):
    """Move masses so they sit on a per-column edge line (+dy)."""
    for m in ms:
        cx = int(np.clip(m['cx'], 0, CW - 1))
        off = float(edge_y[cx]) + dy - m['cy']
        m['cy'] += off
        m['lobes'] = [(x, y + off, r, s_, z) for (x, y, r, s_, z) in m['lobes']]
        b = m['box']
        m['box'] = (b[0], b[1] + off, b[2], b[3] + off)
    return ms


def draw_far_wall(out, t, X):
    """The far face of the haboob on the outside plate (behind the robot)."""
    ye = wall_edge(t)
    face = (YS > ye[None, :]) & ~BELOW
    dy = YS - ye[None, :]
    base = np.clip((YS - (GY - 50)) / 50.0, 0, 1)             # the hazy, dust-kicking foot
    tex = L.TEX[(YS.astype(int) // 2 + 17) & 255, ((XS + X * 70) / 3).astype(int) & 1023]
    lv = 1.1 - 0.9 * np.clip(dy / 34.0, 0, 1) + 2.1 * base ** 1.3 + 0.7 * (tex - 0.5)
    tones = np.array([I('ochre0'), I('ochre1'), I('ochre2'), I('ochre3')], np.uint8)
    q = L.quantize_px(np.clip(lv + 0 * XS, 0, 3).astype(np.float32), 3, w_px=1.2, ox=int(X * 70))
    out[face] = tones[q][face]
    keep = lambda xs, ys: (ys > ye[np.clip(xs, 0, CW - 1)] - 2) & ~BELOW[np.clip(ys, 0, CH - 1), np.clip(xs, 0, CW - 1)]
    # embedded billows, rows following the edge down the face (darker with depth)
    for row, (dyr, seed, gain) in enumerate(((74, 611, 2.0), (44, 612, 2.3))):
        ms = L.masses_row_v3(t, X * (60 + 12 * row) + 40 * row, seed, 84, 0, 80, 130, CW, yjit=9, n=(6, 9),
                             rfrac=(0.2, 0.38), aspect=0.5, roll=0.7)
        _on_edge(ms, ye, dyr)
        L.draw_masses(out, ms, WALL_DEEP_T, ox=int(X * 60), band=0.42, bump=0.24, under=1.5, gain=gain,
                      bias=1.6 + 0.2 * row, clip_fn=keep)
    # the lit, churning top edge (the dying sun rakes it from the upper left)
    ms = L.masses_row_v3(t, X * 75 + 11, 613, 84, 0, 96, 150, CW, yjit=6, n=(11, 15), rfrac=(0.17, 0.34),
                         aspect=0.5, roll=1.0, sq=0.78)
    _on_edge(ms, ye, 16)
    for m in ms:                                      # perspective: far (left) billows are smaller
        k = 0.5 + 0.65 * float(np.clip(m['cx'] / CW, 0, 1))
        cx, cy = m['cx'], m['cy']
        m['lobes'] = [(cx + (x - cx) * k, cy + (y - cy) * k, r * k, s_, z * k) for (x, y, r, s_, z) in m['lobes']]
        m['w'] *= k; m['h'] *= k
        b = m['box']
        m['box'] = (cx + (b[0] - cx) * k, cy + (b[1] - cy) * k, cx + (b[2] - cx) * k, cy + (b[3] - cy) * k)
    L.draw_masses(out, ms, WALL_EDGE_T, ox=int(X * 75), band=0.4, bump=0.26, under=1.9, gain=2.3, bias=2.3,
                  light=(-0.75, -0.45, 0.5), rim=I('rust3'), rim_min=0.62, rim_tex=0.35,
                  clip_fn=lambda xs, ys: ~BELOW[np.clip(ys, 0, CH - 1), np.clip(xs, 0, CW - 1)])
    # far silhouettes stand in front of the wall's hazy foot
    out[PRE_SIL] = I('ochre0')


def draw_surge(img, t, X, xe):
    """The haboob's foot: a low, wide roll of churning billows hugging the
    ground from its nose to beyond the right edge of frame, rolling over the
    ridge; streamers of dust race ahead of it."""
    x0 = float(front_x(t))
    x = x0
    ms = []
    k = 0
    while x < x0 + 150 and k < 6:                       # the surge's head; it thins into murk behind
        f = (x - x0) / 150.0
        w = ((80 if k == 0 else 60) + 24 * float(L.rand1(k, 1, 71))) * (1 - 0.45 * f)
        h = w * (0.66 if k == 0 else 0.56) * (1 - 0.35 * f)
        cx = x + w * 0.4
        cy = crest_y(np.clip(cx, 0, CW - 1)) + 10 - h * 0.5
        ms.append(L.build_mass(t, cx, cy, w, h, 7100 + k, 9, w * 0.13, w * 0.27, roll=1.4, sq=0.8))
        x += w * 0.6
        k += 1
    # the roll drags along the ground: a dust body under the billows down to the sand
    body = np.zeros((CH, CW), bool)
    L.bank_body(np.zeros((CH, CW), np.uint8), [(m['cx'] - m['w'] * 0.25, m['cy'], 1, 1, 0) for m in ms] +
                [(ms[-1]['cx'] + 10, ms[-1]['cy'] + 8, 1, 1, 0)], 1, y_to=GY + 14,
                x_lo=int(max(x0 + 6, 0)), mask_out=body)
    body &= XS < ms[-1]['cx'] + 12
    img[body] = I('ochre1')
    L.draw_masses(img, ms[::-1], SURGE_T, ox=int(X * 300), band=0.4, bump=0.26, under=1.7, gain=2.2,
                  bias=1.8, light=(-0.7, -0.6, 0.45), rim=I('ochre5'), rim_min=0.72, rim_tex=0.4)
    ahead = []
    for k in range(9):                                    # dust streamers racing ahead, low
        age = (t * 2.5 + float(L.rand1(k, 3, 56))) % 1.0
        x = front_x(t) - 8 - age * 90
        y = crest_y(np.clip(x, 0, CW - 1)) - 1 + 10 * float(L.rand1(k, 4, 56)) - age * 7
        r = (3 + 4 * float(L.rand1(k, 5, 56))) * (1 - age * 0.5)
        ahead.append((x, y, r * 1.6, 0.45, k))
    L.draw_puffs(img, ahead, [I('ochre1'), I('ochre2'), I('ochre3'), I('ochre4')],
                 light=(-0.9, -0.45), lit_bias=1.2, texk=0.6, edge=0.6)


# ------------------------------------------------------------------ the dead tree, grass, far pylon
TREE = L.make_tree()
TREE_X = 290
TREE_SNAP = 1.28                          # the gust reaches the tree: a branch rips off
TREE_LOST = (5, 10)                       # the windward branch (+ its twig)


def _tree_flutter(t):
    w = wind(t)
    def f(k):
        return 0.5 * (float(vnoise1(np.array([t * (7.0 + k * 0.7) + k * 3.1]), 71)[0]) - 0.5) * w
    return f


def tree_pose(t):
    w = wind(t)
    bend = -(0.22 + 0.34 * w)
    return L.tree_points(TREE, TREE_X, crest_y(TREE_X) + 1, bend, _tree_flutter(t),
                         detached=set(TREE_LOST) if t >= TREE_SNAP else None)


_SNAP = {}


def draw_tree(img, t):
    segs, pos, ang = tree_pose(t)
    L.draw_segments(img, segs, {3: I('ochre0'), 2: I('ochre0'), 1: I('ochre1')})
    if t >= TREE_SNAP:
        if 'segs' not in _SNAP:
            s0, _, _ = tree_pose(TREE_SNAP - 1e-3)
            _SNAP['segs'] = [sg for sg in s0 if sg[5] in TREE_LOST]
        a = t - TREE_SNAP
        sg = _SNAP['segs']
        cx = np.mean([0.5 * (q[0] + q[2]) for q in sg]); cy = np.mean([0.5 * (q[1] + q[3]) for q in sg])
        nx = cx - 180 * a - 160 * a * a
        ny = cy - 40 * a + 120 * a * a
        rot = -9.0 * a
        moved = []
        for (x0, y0, x1, y1, wd, k) in sg:
            p = L.rot_pts([[x0 - cx, y0 - cy], [x1 - cx, y1 - cy]], rot)
            moved.append((nx + p[0, 0], ny + p[0, 1], nx + p[1, 0], ny + p[1, 1], wd, k))
        if nx > -20:
            L.draw_segments(img, moved, {1: I('ochre1')})


TUFTS = [(x, L.tuft_blades(5, 300 + i, 3, 6)) for i, x in enumerate((64, 84, 142, 166, 214, 248, 268))]
NEAR_TUFTS = [(x, dy, L.tuft_blades(7, 400 + i, 7, 13)) for i, (x, dy) in enumerate(((24, 40), (182, 44), (318, 36)))]


def draw_grass(img, t, near=False):
    w = wind(t)
    bend = -(0.35 + 0.55 * w)
    if not near:
        for x, bl in TUFTS:
            L.draw_tuft(img, x, crest_y(x) + 1, bl, bend, t, I('ochre1'), I('ochre3'))
    else:
        for x, dy, bl in NEAR_TUFTS:
            L.draw_tuft(img, x, crest_y(x) + dy, bl, bend * 1.1, t + x, I('ochre0'), I('ochre1'))


def _pylon():
    """A far lattice pylon (hazy silhouette) — it dissolves as visibility collapses."""
    m = np.zeros((CH, CW), bool)
    x, base, top = 44.0, crest_y(44) + 6, crest_y(44) - 40
    lean = -2.5
    def P(f, side):
        y = base + (top - base) * f
        half = 5.5 * (1 - f) + 1.2
        return x + lean * f + side * half, y
    segs = [(P(0, -1), P(1, -1)), (P(0, 1), P(1, 1))]
    for k in range(7):
        f0, f1 = k / 7.0, (k + 1) / 7.0
        segs.append((P(f0, -1), P(f1, 1)))
        segs.append((P(f0, 1), P(f1, -1)))
    ax, ay = P(0.86, 0)
    segs.append(((ax - 9, ay), (ax + 9, ay)))
    segs.append(((ax - 9, ay), (ax - 7, ay + 2)))
    segs.append(((ax + 9, ay), (ax + 7, ay + 2)))
    bx, by = P(0.7, 0)
    segs.append(((bx - 6, by), (bx + 6, by)))
    segs.append(((ax + 9, ay + 2), (ax + 34, ay + 14)))       # a snapped cable drooping
    for (a, b) in segs:
        xs, ys = line_points(a[0], a[1], b[0], b[1])
        L.put_points(m, xs, ys, True)
    return m


PYLON = _pylon()
_PRE_SIL = np.zeros((CH, CW), bool)
PRE = _pre_storm()
PRE_SIL = _PRE_SIL


# ------------------------------------------------------------------ debris
BUSH_T0, BUSH_V = 0.45, 290.0
BUSH_HOPS = [(0.20, 6), (0.28, 11), (0.18, 4), (0.32, 13), (0.22, 5), (0.30, 9), (0.26, 6), (0.3, 8)]
BUSH_R = 6.5


def draw_bush_at(img, t):
    if t < BUSH_T0 or t > BUSH_T0 + 2.2:
        return
    x = CW + 10 - BUSH_V * (t - BUSH_T0) - 25 * (WINDX(t) - WINDX(BUSH_T0) - (t - BUSH_T0))
    hgt, k, u = L.parabola_hops(t, BUSH_T0, BUSH_HOPS)
    gy = crest_y(np.clip(x, 0, CW - 1)) + 9
    y = gy - BUSH_R - hgt
    ang = (x - CW) / BUSH_R
    L.draw_bush(img, x, y, ang, [I('ochre0'), I('dust2'), I('dust4')])


STONES = [  # (t0, dy below crest, speed, size, seed)
    (0.62, 22, 330.0, 2, 1), (0.95, 33, 390.0, 3, 2), (1.55, 17, 360.0, 2, 3),
    (2.05, 27, 420.0, 2, 4), (3.05, 38, 380.0, 3, 5), (3.7, 20, 350.0, 2, 6),
    (4.4, 29, 400.0, 2, 7), (5.0, 24, 420.0, 3, 8),
]


STONE_HOPS = {sd: [(0.16 + 0.1 * float(L.rand1(j, 1, sd)), 3 + 6 * float(L.rand1(j, 2, sd))) for j in range(10)]
              for (_, _, _, _, sd) in STONES}


def draw_stones(img, t):
    for (t0, dyc, v, s, sd) in STONES:
        a = t - t0
        if a < 0 or a > 1.3:
            continue
        x = CW + 6 - v * a
        if x < -10:
            continue
        gy = int(crest_y(np.clip(x, 0, CW - 1)) + dyc)
        hgt, k, u = L.parabola_hops(t, t0, STONE_HOPS[sd])
        y = gy - hgt
        xi, yi = int(x), int(y)
        img[max(yi - s + 1, 0):max(yi + 1, 0), max(xi, 0):max(xi + s, 0)] = I('ochre0')
        if 0 <= yi - s + 1 < CH and 0 <= xi < CW:
            img[yi - s + 1, xi] = I('dust3')
        if k >= 1 and u < 0.35:                               # kicked-up puff at touch-down
            px = int(x + 2 + u * 30)
            for dx, dy in ((0, 0), (1, 0), (2, -1), (3, 0), (1, -1)):
                if 0 <= gy + dy < CH and 0 <= px + dx < CW and (dx + int(u * 10)) % 2 == 0:
                    img[gy + dy, px + dx] = I('dust5')


_ANT = {}


def antenna_geometry():
    """The antenna exactly as the character model draws it at the moment of
    impact: the pixels present in the 'normal' sprite but not the 'damaged' one
    (stem + bulb; the damaged stub stays on the head). Returns (root_xy canvas,
    pts (n,2) relative to the piece's centre, palette idx per pt, centre_xy).
    Falls back to model-sheet geometry if the diff is empty."""
    if 'g' in _ANT:
        return _ANT['g']
    t = T_IMPACT - 1e-3
    anim, ta, var, fl, x, y = robot_state(t)
    x = float(int(round(x)))
    sn = robot.frame(anim, ta, can='sprout', variant='normal', scale='medium', face_left=fl)
    sd = robot.frame(anim, ta, can='sprout', variant='damaged', scale='medium', face_left=fl)
    ax, ay = sn.anchor('feet')
    px, py = int(round(x)) - ax, int(round(crest_y(x))) - ay
    diff = (sn.img[..., 3] > 127) & ~(sd.img[..., 3] > 127)
    ys, xs = np.nonzero(diff)
    if len(ys):
        idx = L.sprite_to_idx(sn.img)[0][ys, xs]
        root = (px + int(xs[ys.argmax()]), py + int(ys.max()) + 1)
        cx, cy = xs.mean(), ys.mean()
        pts = np.stack([xs - cx, ys - cy], 1).astype(np.float64)
        centre = (px + cx, py + cy)
    else:                                            # fallback: model-sheet antenna above the head
        m = sn.img[..., 3] > 127
        rows = np.nonzero(m.sum(1) >= 4)[0]
        r0 = rows[0] if len(rows) else 0
        c = int(np.nonzero(m[r0])[0].mean())
        root = (px + c, py + r0)
        pts = L.ANT_PTS - [0, -1.5]
        idx = np.array([I('steel3'), I('steel3'), I('core5'), I('steel0'), I('steel0'), I('steel0')])
        centre = (root[0], root[1] - 2.5)
    _ANT['g'] = (root, pts, idx, centre)
    return _ANT['g']


def impact_point():
    """Where the sheet strikes: the antenna root on top of the head."""
    return antenna_geometry()[0]


def sheet_state(t):
    """-> (x, y, phi, theta) or None."""
    if t < T_SHEET_IN:
        return None
    hx, hy = impact_point()
    if t <= T_IMPACT:
        u = (t - T_SHEET_IN) / (T_IMPACT - T_SHEET_IN)
        p0 = np.array([CW + 14.0, GY - 64.0])
        p2 = np.array([hx + 5.0, hy - 1.0])
        p1 = np.array([hx + 95.0, GY - 84.0])
        p = (1 - u) ** 2 * p0 + 2 * (1 - u) * u * p1 + u * u * p2
        return float(p[0]), float(p[1]), 0.5 + 9.3 * u, -0.6 + 3.2 * u
    a = t - T_IMPACT
    return hx + 6 - 240 * a - 120 * a * a, hy - 2 - 170 * a + 230 * a * a, 9.8 + 24 * a, 2.6 + 15 * a


SHEET_COLS = [I('steel0'), I('rust1'), I('rust2'), I('steel3'), I('steel4'), I('steel6')]


def draw_sheet_at(img, t, mask=None):
    s = sheet_state(t)
    if s is None:
        return
    x, y, phi, th = s
    if x < -20 or y < -20:
        return
    L.draw_sheet(img, x, y, phi, th, SHEET_COLS, w=16.0, h=9.0, mask_out=mask)


# The ripped antenna: a touch bigger than the model's 1-px stalk so it reads —
# a 2x2 amber bulb, a steel rod, a rust break. (x, y) relative to the bulb.
ANT_PIECE = [((0, -1), 'steel0'), ((1, -1), 'steel0'),
             ((-1, 0), 'steel0'), ((0, 0), 'core6'), ((1, 0), 'core5'), ((2, 0), 'steel0'),
             ((-1, 1), 'steel0'), ((0, 1), 'core5'), ((1, 1), 'core4'), ((2, 1), 'steel0'),
             ((0, 2), 'steel0'), ((1, 2), 'steel5'), ((2, 2), 'steel0'),
             ((0, 3), 'steel0'), ((1, 3), 'steel3'), ((2, 3), 'steel0'),
             ((1, 4), 'rust3')]
ANT_XY = np.array([p for p, _ in ANT_PIECE], np.float64) - [0.5, 1.5]
ANT_COL = np.array([I(c) for _, c in ANT_PIECE])
ANT_IS_BULB = np.array([c.startswith('core') for _, c in ANT_PIECE])
T_ANT0 = T_IMPACT + HITSTOP * FRAME          # it starts to move when the hit-stop releases
ANT_V0, ANT_G, ANT_VX, ANT_AX = 80.0, 170.0, 20.0, 60.0


def _ant_air(a, x0, y0):
    return x0 - ANT_VX * a - ANT_AX * a * a, y0 - ANT_V0 * a + 0.5 * ANT_G * a * a


def _ant_landing():
    """(age, x) where the arc meets the ridge (pure; cached)."""
    if 'land' not in _ANT:
        root = antenna_geometry()[0]
        x0, y0 = root[0] + 0.5, root[1] - 5.0
        a = 0.05
        while a < 3.0:
            x, y = _ant_air(a, x0, y0)
            if y >= crest_y(np.clip(x, 0, CW - 1)) - 3:
                break
            a += 0.002
        _ANT['land'] = (a, x0, y0)
    return _ANT['land']


def antenna_state(t):
    """Flight of the snapped antenna: it pops straight up (slow, readable first
    frames: ~2 px per frame), arcs over to the left as the wind takes it, drops
    onto the sand and skitters away. Returns (x, y, angle, age) of the bulb."""
    if t < T_IMPACT - EPS:
        return None
    a_land, x0, y0 = _ant_landing()
    a = max(0.0, t - T_ANT0)
    if a <= a_land:
        x, y = _ant_air(a, x0, y0)
        return x, y, -0.25 - 10.0 * a, a
    xl, _ = _ant_air(a_land, x0, y0)
    b = a - a_land
    x = xl - 75 * b - 40 * b * b
    gy = crest_y(np.clip(x, 0, CW - 1)) - 3
    hop = 5.0 * abs(math.sin(b * 8.0)) * math.exp(-b * 3.5)
    return x, gy - hop, -0.25 - 10.0 * a_land - 6.5 * b, a


def draw_antenna_at(img, t):
    """The antenna piece, rotated as points (never a resampled bitmap), its
    bulb glowing amber with a short comet trail and dying as it goes."""
    s_ = antenna_state(t)
    if s_ is None:
        return img
    x, y, ang, a = s_
    if x < -12:
        return img
    life = 1.0 - ease.smoothstep((a - 0.7) / 0.6)              # the bulb fades after ~0.7 s
    if life > 0.15:
        img = L.apply_radial(img, x + 0.5, y + 0.5, 7.5, 4.0 * life, GLOW, gamma=1.3, band=0.5)
        for k, col in ((5, 'core2'), (4, 'core2'), (3, 'core3'), (2, 'core4'), (1, 'core5')):
            sp = antenna_state(max(T_ANT0, t - k * FRAME * 0.7))
            if sp is None or (k > 2 and life < 0.6):
                continue
            L.put_points(img, [round(sp[0])], [round(sp[1])], I(col))
    q = L.rot_pts(ANT_XY, ang)
    cols = ANT_COL.copy()
    if life < 0.999:
        dim = [I('core6'), I('core5'), I('core4'), I('core3'), I('core2'), I('rust2')]
        k = int(round((1 - life) * (len(dim) - 1)))
        cols[ANT_IS_BULB] = dim[min(k + 0, len(dim) - 1)]
    if 0.35 < a < 0.8 and frame_no(t) % 5 == 0:                 # it sputters
        cols[ANT_IS_BULB] = I('core2')
    L.put_points(img, np.round(x + q[:, 0]), np.round(y + q[:, 1]), cols)
    return img


SPARK_COLS = [I('white1'), I('core7'), I('core6'), I('core5'), I('core4'), I('core3'), I('core2')]


def draw_sparks(img, t):
    hx, hy = impact_point()
    L.sparks(img, t, T_IMPACT + FRAME, 30, 91, (hx, hy), speed=(60, 220),
             life=(0.15, 0.75), gravity=340, drag_x=-260,
             spread=(math.pi * 0.6, math.pi * 1.32), cols=SPARK_COLS, tail=1.4,
             floor=lambda x: crest_y(min(max(x, 0), CW - 1)) + 2)
    for j, tf in enumerate((T_FLICK, T_FLICK + 0.133, T_FLICK + 0.233)):
        if 0 <= t - tf < 0.25:
            s = impact_point()                        # the snapped stub (the curl is held)
            L.sparks(img, t, tf, 6, 120 + j, s, speed=(30, 90), life=(0.06, 0.2), gravity=200,
                     drag_x=-200, spread=(math.pi, 2 * math.pi), cols=SPARK_COLS[1:], tail=1.0)
            c = robot_anchor(t, 'core')
            L.sparks(img, t, tf + 0.066, 5, 140 + j, c, speed=(25, 70), life=(0.05, 0.16), gravity=160,
                     drag_x=-150, spread=(0, 2 * math.pi), cols=SPARK_COLS[1:5], tail=1.0)


# ------------------------------------------------------------------ feet puffs, gust debris
# sand kicked up at the feet: the front's hit, the gust shove, the sheet's knock, standing up,
# and every heavy 'brace' footfall (the anim lands its weight on frames 7 and 16 of 18 @12 fps)
_LAND = (7 / 12.0, 16 / 12.0)
PUFF_TIMES = ([T_FRONT + 0.02, T_GUST_HIT + 0.03, T_IMPACT + 0.07, T_RISE + 0.35] +
              [c * 1.5 + l for c in range(1) for l in _LAND] +
              [T_BRACE2 + c * 1.5 + l for c in range(2) for l in _LAND])


def foot_puffs(img, t, X):
    """Sand kicked up at the feet (the front's hit, the gust, the impact knock,
    each trudging step) — blown away to the left and thinning out."""
    for j, t0 in enumerate(PUFF_TIMES):
        a = t - t0
        if a < 0 or a > 0.38:
            continue
        fx, fy = robot_anchor(t0, 'feet')
        big = 1.3 if j < 3 else 0.9
        puffs = []
        for k in range(4):
            r = (2.0 + 2.5 * float(L.rand1(k, 1, 700 + j))) * big * (1 + a * 2.2)
            px = fx + 3 - (60 + 50 * float(L.rand1(k, 2, 700 + j))) * a * wind(t) - k * 2
            py = fy - 1 - r * 0.4 - 10 * a * float(L.rand1(k, 3, 700 + j))
            puffs.append((px, py, r, 0.6, k))
        fade = 1 - a / 0.38
        L.draw_puffs(img, puffs, [I('ochre2'), I('ochre3'), I('ochre4'), I('dust5')], ox=int(X * 300),
                     light=(-0.5, -0.87), lit_bias=1.6, texk=0.7, edge=0.5 + 1.5 * (1 - fade), skew=0.6)


GUST_BITS = [(1.30 + 0.06 * k, 30 + 23 * k, 0.8 + 0.3 * (k % 3)) for k in range(6)]


def gust_debris(img, t):
    """Bits of dead stuff (leaves, twigs, scraps) whipped through by the gust."""
    for k, (t0, y0, sz) in enumerate(GUST_BITS):
        a = t - t0
        if a < 0 or a > 1.2:
            continue
        x = CW + 5 - 520 * a
        if x < -8:
            continue
        y = y0 + 40 * math.sin(a * 5 + k) + 30 * a
        ang = a * (14 + 5 * k)
        ln = 2 + sz * 1.5
        dx, dy = math.cos(ang) * ln, math.sin(ang) * ln * 0.6
        L.put_line(img, x - dx, y - dy, x + dx, y + dy, I('ochre0') if k % 2 else I('ochre1'))


def vignette(img):
    return L.apply_lut(img, VIGN, DARKEN, band=0.5)


_vx = (XS - (RX + 30)) / (CW * 0.62)
_vy = (YS - (GY - 30)) / (CH * 0.75)
VIGN = np.clip(np.sqrt(_vx ** 2 + _vy ** 2) - 0.45, 0, None) ** 1.5 * 3.2


# ------------------------------------------------------------------ the drift against its back
DRIFT_T0, DRIFT_T1 = 1.85, 4.0


_DRIFT = {}


def _drift_x0():
    """Right edge of the curled silhouette's lower half (its windward back)."""
    if 'x' not in _DRIFT:
        spr, px, py, _ = robot_sprite(3.0, False)
        m = spr.img[..., 3] > 127
        rows = np.nonzero(m.any(1))[0]
        low = m[rows[-1] - 6:rows[-1] + 1]
        cols = np.nonzero(low.any(0))[0]
        _DRIFT['x'] = px + int(cols[-1]) - 1
    return _DRIFT['x']


def sand_drift(img, t):
    """Sand piling up against the curled robot's windward back (right side)
    during the shield hold; left behind as a small drift when it moves on."""
    if t < DRIFT_T0:
        return
    g = ease.smoothstep((min(t, DRIFT_T1) - DRIFT_T0) / (DRIFT_T1 - DRIFT_T0))
    if g <= 0.02:
        return
    x0 = _drift_x0()
    hmax = 1.0 + 4.0 * g
    wid = 4 + 8 * g
    grain = int(t * 24)
    for dx in range(-1, int(wid) + 1):
        f = max(dx, 0) / max(wid, 1)
        hh = hmax * (1 - f) ** 1.4
        if hh < 0.6:
            continue
        cx = x0 + dx
        if not 0 <= cx < CW:
            continue
        base = int(round(crest_y(cx)))
        top = int(round(base - hh))
        col = img[top + 1:base + 1, cx]
        img[top + 1:base + 1, cx] = STEP[-2][col]                   # mound in its own shade
        img[top, cx] = STEP[1][img[top, cx]] if (dx + grain) % 6 else STEP[2][img[top, cx]]   # wind-lit crest


# ------------------------------------------------------------------ robot + light
def draw_robot(img, t):
    spr, px, py, face_left = robot_sprite(t)
    sidx, smask = L.sprite_to_idx(spr.img)
    # contact shadow: the sand darkened one step under and around the feet
    rows = np.nonzero(smask.any(1))[0]
    if len(rows):
        fy = rows[-1]
        fx = np.nonzero(smask[fy])[0]
        y0 = py + fy + 1
        if 0 <= y0 < CH:
            xa, xb = max(px + fx[0] - 1, 0), min(px + fx[-1] + 2, CW)
            img[y0, xa:xb] = STEP[-2][img[y0, xa:xb]]
            if y0 + 1 < CH:
                img[y0 + 1, xa + 1:xb - 1] = STEP[-1][img[y0 + 1, xa + 1:xb - 1]]
    L.paste_idx(img, sidx, smask, px, py)
    full = np.zeros((CH, CW), bool)
    L.put_mask(full, smask, True, px, py)
    anc = {k: (px + v[0], py + v[1]) for k, v in spr.anchors.items()}
    return anc, full, (sidx, smask), (px, py), face_left


def _spill_emitters(smask):
    """Leeward (left) edges where sand pours off: the head's top-left corner,
    the back of the head, and the shoulder. From the sprite silhouette."""
    h, w = smask.shape
    rows = smask.sum(1)
    head_rows = np.nonzero(rows >= 4)[0]
    if len(head_rows) == 0:
        return []
    r0 = head_rows[0]
    em = []
    xs = np.nonzero(smask[r0])[0]
    em.append((xs[0], r0))                                   # top-left of the head
    for r in range(r0 + 1, min(r0 + 7, h)):
        xs = np.nonzero(smask[r])[0]
        if len(xs):
            em.append((xs[0], r + 2))                           # back/left of the head
            break
    # shoulder: first row after the neck whose width jumps up again
    prev = rows[r0]
    narrow = False
    for r in range(r0 + 1, h):
        if rows[r] < 0.6 * rows[r0]:
            narrow = True
        elif narrow and rows[r] >= 0.8 * rows[r0]:
            xs = np.nonzero(smask[r])[0]
            em.append((xs[0], r))
            break
    return em


def sand_spill(img, t, smask, pos):
    """Sand pouring off the head and shoulders on the leeward (left) side in
    thin animated streams, and skittering over the top edge."""
    px, py = pos
    h, w = smask.shape
    ww = wind(t)
    em = _spill_emitters(smask)
    for e, (ex, ey) in enumerate(em):
        ln = int(5 + 7 * min(ww, 1.6) * (1.0 if e < 2 else 0.7))
        v = 55 + 25 * e
        drop = 0.07 + 0.03 * e
        for s_ in range(ln):
            if (s_ + int(t * v) + 2 * e) % 4 >= 2:
                continue
            x = px + ex - 1 - s_
            y = py + ey + int(drop * s_ * s_ + 0.25 * s_)
            if 0 <= x < CW and 0 <= y < CH:
                if not (0 <= y - py < h and 0 <= x - px < w and smask[y - py, x - px]):
                    img[y, x] = I('dust6') if (s_ + e) % 5 == 0 else (I('ochre1') if s_ < ln * 0.6 else I('ochre2'))
    # sand skittering over the top edge of the silhouette (leftward)
    cols_any = smask.any(0)
    xs = np.nonzero(cols_any)[0]
    if len(xs) == 0:
        return
    top = np.where(cols_any, smask.argmax(0), 99)
    sh = int(t * 40 * ww)
    for c in xs:
        if (c + sh) % 4 == 0:
            yy = py + top[c] - 1
            if 0 <= yy < CH and 0 <= px + c < CW:
                img[yy, px + c] = I('ochre2')


def core_pool(img, t, X, anc, face_left, robot_mask, fog, air=None, radius_k=1.0, peak_k=1.0, lut=None):
    """The amber core light scattered in the streaming dust (and pooled on the
    sand). air: optional per-pixel dust multiplier (clear air scatters less)."""
    cl = core_level(t)
    cx, cy = anc['core']
    cx += -3 if face_left else 3
    radius = (12 + 62 * cl * (0.85 + 0.3 * fog)) * radius_k
    peak = (0.4 + 6.0 * cl) * peak_k

    def mod(y0, y1, x0, x1):
        ys = np.arange(y0, y1)[:, None]
        xs = np.arange(x0, x1)[None, :]
        dust = L.TEX[(ys // 2 + 30) & 255, ((xs + X * 140) / 2).astype(int) & 1023]
        m = np.where(BELOW[y0:y1, x0:x1], 0.9, 0.45 + 1.1 * dust)
        if air is not None:
            m = m * air[y0:y1, x0:x1]
        return np.where(robot_mask[y0:y1, x0:x1], m * 0.3, m)
    return L.apply_radial(img, cx, cy, radius, peak, GLOW if lut is None else lut, gamma=1.7, sy=0.9,
                          mod=mod, band=0.45)


def sprout_glow(img, anc, robot_mask, strength=1.0):
    """The sprout's faint living green, leaking from the hugged can: the only
    other colour in the murk."""
    if 'can_mouth' not in anc or strength <= 0.02:
        return img
    sx, sy = anc['can_mouth']
    return L.apply_radial(img, sx, sy - 2, 6.5, 4.6 * strength, GREEN, gamma=1.4,
                          mod=lambda y0, y1, x0, x1: np.where(robot_mask[y0:y1, x0:x1], 0.55, 1.0), band=0.4)


# ------------------------------------------------------------------ impact frames
def impact_frame(img, rmask, sheet_mask):
    """Impact frame: two-tone silhouette flash with a starburst."""
    out = np.full_like(img, I('dust9'))
    lum = PP.PAL_LAB[img][..., 0]
    out[lum < 0.36] = I('dust7')
    out[rmask | sheet_mask] = I('ink0')
    hx, hy = impact_point()
    for k in range(12):
        a = k / 12 * 2 * math.pi + 0.3
        r = 9 + 9 * (k % 2)
        L.put_line(out, hx, hy, hx + math.cos(a) * r, hy + math.sin(a) * r * 0.85, I('white1'))
    return out


# ------------------------------------------------------------------ the pale whiteout front (4.2 - 6.0)
def white_edge(t):
    """Per-row x of the whiteout's leading edge: it rolls in from the right
    with the wind (~260 px/s), reaching the robot at ~5.05."""
    y = np.arange(CH, dtype=np.float32)
    base = RX + 420.0 * (0.5 - (t - 4.4) / 1.55)
    bil = 26.0 * (L.fbm3(0.0 * y + 7.7, y / 18.0, t * 1.2, 99, 2) - 0.5)
    return base - 0.25 * (GY - y) + bil


PALE_T = [I('dust5'), I('dust6'), I('dust7'), I('dust8')]


def white_lip(img, t, X, xw):
    """Pale billows along the whiteout's leading edge (a ghost of the haboob
    that opened the shot): volumetric masses rolling in the wind."""
    ms = []
    y = CH + 14.0
    k = 0
    while y > -40 and k < 14:
        w = 46 + 44 * float(L.rand1(k, 1, 66))
        h = w * 0.62
        yy = int(np.clip(y, 0, CH - 1))
        cx = float(xw[yy]) + w * 0.18 + (float(L.rand1(k, 2, 66)) - 0.5) * 20
        ms.append(L.build_mass(t, cx, y, w, h, 6600 + k, 8, w * 0.14, w * 0.28, roll=0.9, sq=0.8))
        y -= h * (0.62 + 0.3 * float(L.rand1(k, 3, 66)))
        k += 1
    L.draw_masses(img, ms, PALE_T, ox=int(X * 300), band=0.42, bump=0.24, under=1.1, gain=1.6, bias=1.6,
                  tex_amp=0.3)


# ------------------------------------------------------------------ render
def render(t):
    tw = world_time(t)
    X = WINDX(tw)
    fog = fog_level(tw)
    front_on = tw < 1.05
    wt = ease.smoothstep((tw - 4.4) / 1.5)                  # whiteout progress

    # ---------------- far: murk, churning billows, rolling banks, veils
    img = backdrop(tw, X, fog)
    sky_billows(img, tw, X, fog)
    far_banks(img, tw, X)
    img[PYLON] = I('ochre2')
    far_veils(img, tw, X)
    lv = L.fog_map(CH, CW, 0.4 + 4.6 * fog, 1.7, off=X * 40, yoff=17)
    img = L.apply_lut_px(img, lv, FOG_FAR, w_px=1.2)

    # ---------------- mid: surges behind the ridge, sky streaks
    mid_surges(img, tw, X, fog)
    sky_streaks(img, tw, X)
    lv = L.fog_map(CH, CW, 0.3 + 2.8 * fog, 2.0, off=X * 90, yoff=83)
    img = L.apply_lut_px(img, lv, FOG_MID, w_px=1.2)

    # ---------------- the dune, sand in the air, the gust (behind the robot)
    dune(img, tw, X)
    sand_ribbons(img, tw, X)
    spindrift(img, tw, X)
    img = gust_sheet(img, tw, X, behind=True)

    # ---------------- the dust-wall front: pre-storm world left of it
    if front_on:
        xf = surge_edge(tw)
        d = XS - xf[:, None]
        soft = np.where(YS < GY - 46, 40.0, 6.0)                     # soft above the surge band
        storm = L.dmask(np.clip(d / soft + 0.5, 0, 1))
        dz = np.clip(1 - d / 260.0, 0, 1) * (d > -20) * np.where(BELOW, 0.55, 1.0)
        img = L.apply_lut(img, 7.2 * dz ** 0.6, DARKEN, band=0.5)
        if tw < 0.75:
            out = PRE.copy()
            draw_far_wall(out, tw, X)
            img = np.where(storm, img, out)

    # ---------------- robot plane
    draw_tree(img, tw)
    draw_grass(img, tw)
    anc, rmask, (sidx, smask), spos, face_left = draw_robot(img, tw)
    sand_spill(img, tw, smask, spos)
    foot_puffs(img, tw, X)
    robot_plane_sand(img, tw, X)
    draw_bush_at(img, tw)
    sheet_mask = np.zeros((CH, CW), bool)
    draw_sheet_at(img, tw, sheet_mask)
    img = gust_sheet(img, tw, X)
    near_k = np.where(BELOW, np.clip(1.15 - (YS - GY) / 45.0, 0.35, 1.0), 1.0)
    lv = L.fog_map(CH, CW, 0.2 + 2.2 * fog, 1.6, off=X * 160, yoff=151) * (near_k + (1 - near_k) * wt)
    fogged = L.apply_lut_px(img, lv, FOG_MID, w_px=1.2)
    rl = int(round(float(np.median(lv[rmask])) * 0.75)) if rmask.any() else 0
    fogged = np.where(rmask, FOG_MID[min(rl, 8)][img], fogged)
    if front_on:
        img = np.where(storm, fogged, img)
        draw_surge(img, tw, X, xf)
    else:
        img = fogged
    img = core_pool(img, tw, X, anc, face_left, rmask, fog,
                    air=np.where(storm, 1.0, 0.3) if front_on else None)
    img = sprout_glow(img, anc, rmask, 1.0 - 0.6 * fog)
    sand_drift(img, tw)
    if T_FLICK <= tw < T_FLICK + 0.45 and core_level(tw) < 0.1:      # the key light drops out
        cx, cy = anc['core']
        img = L.apply_radial(img, cx, cy, 64, 1.3, DARKEN, gamma=1.3, band=0.5)
    draw_sparks(img, t)
    img = draw_antenna_at(img, t)

    # ---------------- near
    draw_grass(img, tw, near=True)
    draw_stones(img, tw)
    gust_debris(img, tw)
    near_streaks(img, tw, X, fog)
    near_banks(img, tw, X, fog, inside=storm if front_on else None)
    img = vignette(img)

    # ---------------- whiteout: streaky density bands racing left
    if tw > 4.05:
        xw = white_edge(tw)
        d = (XS - xw[:, None]) / 150.0 + 0.5                       # 0.5 at the pale front
        sweep = np.clip(d, 0, 1)
        sweep = sweep * sweep * (3 - 2 * sweep)
        lvw = L.fog_map(CH, CW, 0.0, 2.6 * min(1.0, wt * 2.5 + 0.3), off=X * 380, sx=2.5, yoff=201) + 7.4 * sweep
        rsw = float(np.clip((RX - float(np.median(white_edge(tw)))) / 150.0 + 0.5, 0, 1))
        rw = int(round(7.4 * (rsw * rsw * (3 - 2 * rsw)) * 0.66))
        robot_px = img[rmask]
        img = L.apply_lut_px(img, np.clip(lvw, 0, 7.6), FOG_WHITE, w_px=1.2)
        is_core = np.isin(robot_px, L.RI('core'))                   # the heart and the pupil stay warm
        img[rmask] = np.where(is_core, FOG_WHITE[min(rw, 2)][robot_px], FOG_WHITE[min(rw, 8)][robot_px])
        white_lip(img, tw, X, xw)
        img = core_pool(img, tw, X, anc, face_left, rmask, 0.0, radius_k=0.62, peak_k=1.45 * wt)

    # ---------------- impact frames (during the hit-stop)
    fn = frame_no(t)
    if fn == F_IMPACT:
        img = impact_frame(img, rmask, sheet_mask)
    elif fn == F_IMPACT + 1:
        hx, hy = impact_point()
        img = L.apply_radial(img, hx, hy, 64, 4.5, FLASH, gamma=1.8, band=0.5)

    st = Stage(bg='ochre2')
    st.add(PP.PAL[img], -M, -M)
    st.shake = shake(t)
    return st
