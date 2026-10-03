"""
THE LAST SEED — end title overlay (owner: TITLE).  Global 58.0-60.0, DURATION 2.0

    render(t) -> {'layer': (180,320,4) uint8 RGBA (binary alpha, palette colours),
                  'under_fade': float 0..1}
The compositor darkens the picture under the layer by round(under_fade * 8)
palette ramp-steps (no dither); the film's global fade (59.5-60.0 = local
1.5-2.0) is the same kind of stepped fade and takes everything.

DESIGN
  * A hand-built pixel Roman face (cap height 14; 2 px stems, 1 px hairlines,
    1 px serifs; pointed A, serifed E arm), set wide (tracking 4) like a film
    title. Colour: one warm ramp - warm-white cap line -> cream -> gold (the pale
    end of the robot's heart light), a 1 px ochre under-shadow for an engraved
    feel and a 1 px ink outline for legibility over the picture.
  * Reveal (0.05-0.75 s): the letters GROW up from the baseline one after
    another, each crowned by a bright growing tip that settles into the cap-line
    highlight; a tiny twinkle catches each letter's serif as it completes.
  * An ornament rule under the title (0.2-0.8 s): a small diamond warms up and
    a hairline grows outward from it, tapering into dashes and dots.
  * The living detail: a seedling grows out of the S of SEED (0.6-1.0 s),
    unfurls two leaves with a little overshoot, and gets a sparkle.
  * The robot's eye stays the last light. The layer re-renders the real c04
    close-up around EYE_POS and lights it palette-true: darkened by the same
    ramp-steps as the compositor at the rim of the pool (seamless), less toward
    the pupil, plus green light on the lens - so the lens itself glows in the
    dark. Eye and seedling share one heartbeat: a pulse as the leaves open, then
    a slow swell while the gold title recedes one step before the global fade.
  * One slow diagonal gleam crosses the gold (1.0-1.45 s).
  * under_fade: 0 for the first 0.2 s, then 0 -> 0.85 over 0.2-1.2 s.

EYE_POS / EYE_R: read from shots.c04_rest (EYE_POS a tuple, or eye_pos(t));
default (160, 70), pupil radius 3. Eye high in frame (y < 100) -> title in the
lower third, otherwise in the upper third.
"""
import importlib
import math

import numpy as np

from engine import *
from engine import draw, ease
from engine import light as LIGHT
from engine.stage import to_lowres

DURATION = 2.0

# =====================================================================  FONT
# Hand-built pixel Roman capitals, cap height 14. Stems 2 px, hairlines 1 px,
# 1 px serifs. '#' = ink.
CAP = 14
_FONT = {
    'T': """
##########
#...##...#
....##....
....##....
....##....
....##....
....##....
....##....
....##....
....##....
....##....
....##....
....##....
...####...
""",
    'H': """
####....####
.##......##.
.##......##.
.##......##.
.##......##.
.##......##.
.##########.
.##......##.
.##......##.
.##......##.
.##......##.
.##......##.
.##......##.
####....####
""",
    'E': """
#########
.##.....#
.##......
.##......
.##......
.##...#..
.######..
.##...#..
.##......
.##......
.##......
.##......
.##.....#
#########
""",
    'L': """
####.....
.##......
.##......
.##......
.##......
.##......
.##......
.##......
.##......
.##......
.##......
.##......
.##.....#
#########
""",
    'A': """
......#......
......##.....
.....#.##....
.....#.##....
.....#.##....
....#...##...
....#...##...
....#...##...
...#.....##..
...########..
...#.....##..
..#.......##.
..#.......##.
.###.....####
""",
    'S': """
..#####..
.##....##
##......#
##.......
.##......
..###....
...####..
.....###.
......##.
.......##
.......##
#......##
#.....##.
.#####...
""",
    'D': """
#######.....
.##....##...
.##.....##..
.##......##.
.##.......##
.##.......##
.##.......##
.##.......##
.##.......##
.##.......##
.##......##.
.##.....##..
.##....##...
#######.....
""",
}


def _parse(src):
    rows = src.strip('\n').split('\n')
    w = max(len(r) for r in rows)
    m = np.zeros((len(rows), w), bool)
    for y, r in enumerate(rows):
        for x, c in enumerate(r):
            m[y, x] = c == '#'
    assert m.shape[0] == CAP, (src, m.shape)
    return m


GLYPH = {k: _parse(v) for k, v in _FONT.items()}

# =====================================================================  LAYOUT
TEXT = 'THE LAST SEED'
TRACK = 4                       # px between letters (airy, cinematic)
SPACE = 8                       # extra advance of a word space (on top of TRACK)
KERN = {('L', 'A'): -3, ('A', 'S'): -1, ('S', 'T'): -1}


def layout(text=TEXT):
    """-> list of (char, x) for the non-space glyphs, total width."""
    x = 0
    out = []
    prev = None
    for ch in text:
        if ch == ' ':
            x += SPACE
            prev = None
            continue
        if prev is not None:
            x += KERN.get((prev, ch), 0)
        out.append((ch, x))
        x += GLYPH[ch].shape[1] + TRACK
        prev = ch
    return out, x - TRACK


# =====================================================================  COLOUR
# One warm light ramp; letters, rule and sparkles are all expressed in steps
# of it (0 = transparent), so "brighter / dimmer" is just +/- levels.
LV = [None, 'ochre1', 'ochre2', 'ochre3', 'ochre4', 'ochre5', 'ochre6', 'ochre7',
      'ochre8', 'dust10', 'core7', 'white1']
NLV = len(LV)
LV_RGB = np.array([(0, 0, 0)] + [C(n) for n in LV[1:]], np.uint8)
ROW_LEVEL = np.array([10] + [9] * 4 + [8] * 5 + [7] * 4, float)   # per glyph row, cap line first
SHADOW_LEVEL = 2                            # ochre2 under-shadow
PAD = 2                                     # art margin (outline + shadow)

# green light ramp (the seedling, the self-drawn eye glow fallback)
GL = [None, 'leaf1', 'leaf2', 'leaf3', 'leaf4', 'leaf5', 'leaf6', 'leaf7', 'leaf8']
GL_RGB = np.array([(0, 0, 0)] + [C(n) for n in GL[1:]], np.uint8)

INK = np.array(C('ink0'), np.uint8)
_B4 = bayer(H, W, 4)


def _quant(v, th, band=0.5):
    """Float level field -> int levels, ordered-dithered with threshold map th."""
    fl = np.floor(v)
    fr = v - fl
    if band < 1.0:
        fr = np.clip((fr - 0.5) / band + 0.5, 0.0, 1.0)
    return (fl + (th < fr)).astype(np.int32)


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# =====================================================================  TIMING
LET_T0, LET_STEP, LET_DUR = 0.05, 0.04, 0.30      # letter i grows over [T0 + i*STEP, +DUR]
RULE_T = (0.20, 0.34, 0.30, 0.80)                  # diamond blooms / hairline grows
SPROUT_T0 = 0.60                                   # the seedling grows 0.60-1.00
FADE_T = (0.20, 1.20)                              # under_fade ramp
FADE_HOLD = 0.85
GLINT_T = (1.00, 1.45)                             # slow gleam across the gold
GROW_TIP = 2.4                                     # extra light levels at a growing tip


def under_fade(t):
    """Picture fully held for 0.2 s, then 0 -> 0.85 over 0.2-1.2 s."""
    return max(0.0, FADE_HOLD * ease.in_out_sine(ease.prog(t, *FADE_T)))


def breath(t):
    """The shared heartbeat (0..1) of the robot's eye and the seedling: one
    pulse answering the seedling as its leaves open, then a slow swell while
    the gold title recedes, so green life is the last light of the film."""
    pulse = (ease.smoothstep(ease.prog(t, 0.88, 1.05))
             * (1.0 - ease.smoothstep(ease.prog(t, 1.05, 1.38))))
    swell = 0.85 * ease.smoothstep(ease.prog(t, 1.32, 1.72))
    return max(pulse, swell)


def title_dim(t):
    """Light levels the gold title recedes by at the very end."""
    return 1.2 * ease.smoothstep(ease.prog(t, 1.32, 1.58))


# =====================================================================  TITLE ART
def _build_title():
    """Static per-pixel maps of the title art (letters, shadow, outline)."""
    lay, w = layout()
    AH, AW = CAP + 2 * PAD, w + 2 * PAD
    let = np.full((AH, AW), -1, np.int16)     # letter index of each ink pixel
    shd = np.full((AH, AW), -1, np.int16)     # ... of each under-shadow pixel
    oln = np.full((AH, AW), -1, np.int16)     # ... of each outline pixel
    for i, (ch, x) in enumerate(lay):
        g = GLYPH[ch]
        m = np.zeros((AH, AW), bool)
        m[PAD:PAD + CAP, PAD + x:PAD + x + g.shape[1]] = g
        s = np.zeros_like(m)
        s[1:] = m[:-1]
        s &= ~m
        o = draw.outline_mask(m | s)
        let[m] = i
        shd[s & (shd < 0)] = i
        oln[o & (oln < 0)] = i
    M = let >= 0
    shd[M] = -1
    S = shd >= 0
    oln[M | S] = -1
    lvl = np.zeros((AH, AW))
    lvl[PAD:PAD + CAP] = ROW_LEVEL[:, None]
    rb = np.broadcast_to(((PAD + CAP - 1) - np.arange(AH, dtype=float))[:, None], (AH, AW)).copy()
    ys, xs = np.mgrid[0:AH, 0:AW]
    return dict(lay=lay, w=w, AH=AH, AW=AW, let=let, shd=shd, oln=oln,
                M=M, S=S, O=oln >= 0, lvl=lvl,
                rb=rb,                       # row index counted up from the baseline
                gx=xs + 0.6 * ys)            # diagonal coordinate for the gleam


TA = _build_title()
NL = len(TA['lay'])


def _glint(t):
    """+levels of a soft diagonal band of light sweeping left -> right."""
    u = ease.prog(t, *GLINT_T)
    if u <= 0.0 or u >= 1.0:
        return 0.0
    c = -12 + (TA['AW'] + 24) * ease.in_out_sine(u)
    return 1.3 * np.clip(1.0 - np.abs(TA['gx'] - c) / 3.2, 0.0, 1.0)


def _title_into(L, t, ax, ay):
    """Draw the title (art top-left at screen ax, ay) into layer L. Each letter
    grows up from its baseline; the rows just below the growing front glow."""
    AH, AW = TA['AH'], TA['AW']
    p = [ease.prog(t, LET_T0 + i * LET_STEP, LET_T0 + i * LET_STEP + LET_DUR) for i in range(NL)]
    front = np.array([ease.out_cubic(x) for x in p]) * (CAP + 0.6)      # rows grown
    settle = np.array([ease.smoothstep(ease.prog(x, 0.55, 1.0)) for x in p])
    th = bayer(AH, AW, 4, ax, ay)                                       # screen-locked
    M, S, O, rb = TA['M'], TA['S'], TA['O'], TA['rb']
    q = np.zeros((AH, AW), np.int32)
    # letters: visible below the (dithered) front, bright tip fading as they settle
    li = TA['let'][M]
    g = front[li] - rb[M]
    tip = GROW_TIP * np.clip(1.0 - g / 2.3, 0.0, 1.0) * (1.0 - settle[li])
    glint = _glint(t)
    lv = TA['lvl'][M] + tip - title_dim(t) + (glint[M] if np.ndim(glint) else 0.0)
    q[M] = np.where(g > 1.6 * th[M], _quant(lv, th[M], 0.6), 0)
    # under-shadow follows the row above it
    q[S] = np.where(front[TA['shd'][S]] - (rb[S] + 1) > th[S], SHADOW_LEVEL, 0)
    q = np.clip(q, 0, NLV - 1)
    # outline: around the rows already grown
    ol = np.zeros((AH, AW), bool)
    ol[O] = front[TA['oln'][O]] - np.clip(rb[O], 0, CAP - 1) > 0.5
    sub = L[ay:ay + AH, ax:ax + AW]
    on = q > 0
    sub[ol & ~on, :3] = INK
    sub[ol & ~on, 3] = 255
    sub[on, :3] = LV_RGB[q[on]]
    sub[on, 3] = 255


# =====================================================================  ORNAMENT RULE
# One side of the hairline, from the diamond outward: light level per pixel
# (0 = gap). A deliberate taper: bright line -> softer line -> dashes -> dots
# -> a small terminal dot, like a pen stroke lifting off the page.
_RULE_SIDE = ([0] * 4 + [7] * 12 + [6] * 11 + [5] * 5 +
              [4, 4, 0, 4, 4, 0, 0, 4, 0, 0, 3, 0, 0, 0, 5])
RULE_LEN = len(_RULE_SIDE) - 1
_DIAMOND = [            # light levels (hex), centred on the rule
    "..6..",
    ".696.",
    "69a96",
    ".696.",
    "..6..",
]


def _rule_into(L, cx, y, t):
    k = ease.prog(t, RULE_T[0], RULE_T[1])
    if k <= 0:
        return
    dim = title_dim(t)
    bw = RULE_LEN + 2
    R = layer(7, 2 * bw + 1)                 # local layer centred on (cx, y)
    ox, oy = cx - bw, y - 3
    # the diamond blooms out from its centre (bright as it opens, then settles)
    grown = k * 2.99
    for r, row in enumerate(_DIAMOND):
        for c, ch in enumerate(row):
            if ch == '.' or abs(r - 2) + abs(c - 2) > grown:
                continue
            v = int(ch, 16) + 1.5 * (1.0 - k) - dim
            x, yy = cx - 2 + c, y - 2 + r
            q = int(v) + (1 if _B4[yy, x] < (v - int(v)) else 0)
            if q > 0:
                R[yy - oy, x - ox, :3] = LV_RGB[min(q, NLV - 1)]
                R[yy - oy, x - ox, 3] = 255
    # the hairline grows outward from the diamond, bright at its growing tip
    e = ease.out_cubic(ease.prog(t, RULE_T[2], RULE_T[3]))
    reach = 3 + e * (RULE_LEN - 3 + 0.99)
    for side in (-1, 1):
        for d in range(4, min(int(reach), RULE_LEN) + 1):
            lv = _RULE_SIDE[d]
            if lv == 0:
                continue
            v = lv - dim + 2.2 * max(0.0, 1.0 - (reach - d) / 3.0) * (1.0 - e)
            x = cx + side * d
            q = min(int(v) + (1 if _B4[y, x] < (v - int(v)) else 0), NLV - 1)
            if q > 0:
                R[y - oy, x - ox, :3] = LV_RGB[q]
                R[y - oy, x - ox, 3] = 255
    on = R[..., 3] > 0
    ol = draw.outline_mask(on)
    sub = L[oy:oy + R.shape[0], ox:ox + R.shape[1]]
    free = ol & (sub[..., 3] == 0)
    sub[free, :3] = INK
    sub[free, 3] = 255
    sub[on] = R[on]


# =====================================================================  SEEDLING
_SPR_LEG = {'H': 'leaf8', 'h': 'leaf7', 'g': 'leaf6', 'G': 'leaf5', 'd': 'leaf4', 's': 'leaf4'}
_BRIGHTER = {'g': 'h', 'h': 'H', 'G': 'g', 'd': 'd', 's': 's', 'H': 'H'}
# Growth frames, 11 wide, stem in column 5, bottom row = base (sits on the S):
# shoot -> bud -> leaves part -> open -> overshoot -> settle (asymmetric pair,
# the right leaf a little larger and higher).
_SPR = [
    ".....g.....",
    ".....g.....|.....s.....",
    ".....g.....|.....s.....|.....s.....",
    ".....h.....|.....g.....|.....s.....|.....s.....",
    ".....h.....|....gG.....|.....d.....|.....s.....|.....s.....|.....s.....",
    "....h.h....|....gdgh...|.....d.....|.....s.....|.....s.....|.....s.....",
    "......hh...|..hh.hgg...|..ggd.dG...|....dsd....|.....s.....|.....s.....|.....s.....",
    ".........hh|.hh.....hgg|hggh..hgggG|.GgGd.dGGd.|...ddsd....|.....s.....|.....s.....|.....s.....",
    "........hh.|.hh....hggh|hggh..hgggG|.GgGd.dGGd.|...ddsd....|.....s.....|.....s.....|.....s.....",
]
_SPR_T = [0.00, 0.04, 0.08, 0.12, 0.17, 0.22, 0.26, 0.32, 0.40]     # seconds after SPROUT_T0
_SPR_FRAMES = [f.split('|') for f in _SPR]
_SPR_FINAL = _SPR_FRAMES[-1]
_SPR_LIFT = _SPR_FRAMES[-2]                  # leaves lifted a touch
SPROUT_LIFT_T = 1.55                         # ... reaching for the light in the final swell


def _sprout_rows(t):
    u = t - SPROUT_T0
    if u < 0:
        return None, 0.0
    k = max(i for i, tt in enumerate(_SPR_T) if u >= tt)
    if k < len(_SPR_T) - 1:
        return _SPR_FRAMES[k], 0.0
    return (_SPR_LIFT if t >= SPROUT_LIFT_T else _SPR_FINAL), breath(t)


def _sprout_into(L, bx, by, t):
    rows, b = _sprout_rows(t)
    if rows is None:
        return
    h, w = len(rows), len(rows[0])
    ox, oy = bx - 5, by - (h - 1)
    m = np.zeros((h + 2, w + 2), bool)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            m[r + 1, c + 1] = ch != '.'
    ol = draw.outline_mask(m)
    sub = L[oy - 1:oy + h + 1, ox - 1:ox + w + 1]
    free = (sub[..., 3] == 0) | np.all(sub[..., :3] == INK, axis=-1)
    sub[ol & free] = (*INK, 255)
    th = _B4[oy:oy + h, ox:ox + w]
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == '.':
                continue
            if b > 0 and th[r, c] < 0.8 * b:          # leaves brighten with the heartbeat
                ch = _BRIGHTER[ch]
            L[oy + r, ox + c, :3] = C(_SPR_LEG[ch])
            L[oy + r, ox + c, 3] = 255


# =====================================================================  SPARKLES
_SPARK = [                                   # 4-point star, grows and fades
    ["a"],
    [".a.", "aba", ".a."],
    ["..c..", "..a..", "cabac", "..a..", "..c.."],
    [".a.", "aba", ".a."],
    ["c"],
]
_SPARK_T = [0.0, 0.04, 0.09, 0.21, 0.28, 0.34]
_TWINK = [["a"], [".a.", "aba", ".a."], ["c"]]   # the small per-letter twinkle
_TWINK_T = [0.0, 0.035, 0.1, 0.15]
_SPARK_COL = {'a': 'core7', 'b': 'white1', 'c': 'ochre6'}
# where a twinkle catches each letter as it completes: (x, y) in glyph pixels
_TWINK_AT = {'T': (0, 0), 'H': (11, 0), 'E': (8, 0), 'L': (0, 0), 'A': (6, 0), 'S': (8, 1), 'D': (8, 1)}
SPARK_DIAMOND_T = 0.30
SPARK_SPROUT_T = SPROUT_T0 + 0.35


def _sparkle_into(L, x, y, t, t0, frames=_SPARK, times=_SPARK_T):
    u = t - t0
    if u < 0 or u >= times[-1]:
        return
    k = max(i for i in range(len(frames)) if u >= times[i])
    fr = frames[k]
    n = len(fr)
    for r, row in enumerate(fr):
        for c, ch in enumerate(row):
            if ch == '.':
                continue
            xx, yy = x - n // 2 + c, y - n // 2 + r
            if 0 <= xx < W and 0 <= yy < H:
                L[yy, xx, :3] = C(_SPARK_COL[ch])
                L[yy, xx, 3] = 255


# =====================================================================  THE EYE
EYE_DEFAULT = (160, 70)
EYE_R_DEFAULT = 3.0
EYE_OVERRIDE = None           # tests only: (x, y)
EYE_R_OVERRIDE = None         # tests only: pupil radius
BASE_OVERRIDE = None          # tests only: callable(t) -> the picture under the title


def _c04():
    try:
        return importlib.import_module('shots.c04_rest')
    except Exception:
        return None


_TIMING = {}


def _timing():
    """(c04 local time at title t=0, compositor fade steps), from film.py when
    available so a retime of the overlay or the close-up is picked up."""
    if not _TIMING:
        off, steps = 3.0, 8
        try:
            import film
            s = [x for x in film.TIMELINE if x.module == 'shots.c04_rest'][-1]
            o = [x for x in film.OVERLAYS if x.module == __name__][0]
            off = o.start - s.start + s.offset
            steps = int(getattr(film, 'FADE_STEPS', steps))
        except Exception:
            pass
        _TIMING.update(off=off, steps=steps)
    return _TIMING['off'], _TIMING['steps']


def eye_pos(t=0.0):
    """Pupil centre in screen px, from shots.c04_rest (eye_pos(t) or EYE_POS)."""
    if EYE_OVERRIDE is not None:
        return float(EYE_OVERRIDE[0]), float(EYE_OVERRIDE[1])
    m = _c04()
    try:
        lt = _timing()[0] + t
        f = getattr(m, 'eye_pos', None) if m is not None else None
        ep = f(lt) if callable(f) else getattr(m, 'EYE_POS', EYE_DEFAULT) if m is not None else EYE_DEFAULT
        if callable(ep):
            ep = ep(lt)
        return float(ep[0]), float(ep[1])
    except Exception:
        return EYE_DEFAULT


def eye_radius():
    """Radius (px) of the close-up's glowing pupil (shots.c04_rest.EYE_R)."""
    r = EYE_R_OVERRIDE
    if r is None:
        m = _c04()
        r = getattr(m, 'EYE_R', None) if m is not None else None
    try:
        return float(np.clip(float(r), 1.5, 9.0)) if r is not None else EYE_R_DEFAULT
    except Exception:
        return EYE_R_DEFAULT


def _picture(t):
    """The picture under the title (the c04 close-up, un-darkened), or None."""
    if BASE_OVERRIDE is not None:
        return BASE_OVERRIDE(t)
    m = _c04()
    if m is None:
        return None
    try:
        lt = _timing()[0] + t
        d = float(getattr(m, 'DURATION', 1e9))
        return to_lowres(m.render(min(max(lt, 0.0), d - 1e-4)))
    except Exception:
        return None


def _eye_into(L, t, U):
    """The robot's eye stays the last light: a pool around the pupil where the
    real close-up is darkened by fewer ramp-steps than the rest of the picture
    (exactly as many at the rim, so it joins seamlessly) and lit green."""
    k = int(round(min(max(U, 0.0), 1.0) * _timing()[1]))
    kg = ease.smoothstep(ease.prog(t, FADE_T[0], FADE_T[1]))      # glow takes over as it darkens
    if kg <= 0 or k <= 0:
        return
    b = breath(t)
    er = eye_radius()
    ex, ey = eye_pos(t)
    ex, ey = int(round(ex)), int(round(ey))
    base = _picture(t)
    if base is None:
        _glow_into(L, ex, ey, t, er)
        return
    R1 = 2.4 * er + 12.0                       # the lit pool (steps return to -k at R1)
    R2 = 1.2 * er + 5.0                        # green light on the pupil and inner glass
    R = int(math.ceil(max(R1, R2))) + 1
    x0, x1 = max(ex - R, 0), min(ex + R + 1, W)
    y0, y1 = max(ey - R, 0), min(ey + R + 1, H)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.ogrid[y0:y1, x0:x1]
    d = np.sqrt((xs - ex) ** 2 + (ys - ey) ** 2)
    sub = base[y0:y1, x0:x1]
    # 1) fewer darkening steps toward the pupil (integer -k at the rim)
    f1 = _smooth(1.0 - (d - er) / (R1 - er)) ** 1.3
    img = LIGHT.apply_light(sub, -k * (1.0 - f1), mode='neutral', shadow='plain',
                            ox=x0, oy=y0, band=0.4)
    # 2) green light spilling from the pupil over the glass
    f2 = np.clip(1.0 - (d - 0.5 * er) / (R2 - 0.5 * er), 0.0, 1.0) ** 1.6
    img = LIGHT.apply_light(img, (0.6 + 1.2 * kg + 0.9 * b) * f2, mode='green', shadow='plain',
                            ox=x0, oy=y0, band=0.4)
    m = d < R1
    Ls = L[y0:y1, x0:x1]
    Ls[m, :3] = img[m]
    Ls[m, 3] = 255
    # 3) the hot centre of the pupil
    mm = d <= 0.4 + 0.25 * er * kg
    Ls[mm, :3] = GL_RGB[8]
    Ls[mm, 3] = 255
    # 4) from just before the film's global fade, nothing in the eye is paler
    #    than leaf7: leaf7 and below darken along green ramps (leaf8 and the
    #    whites would walk through greys), so the last light stays green
    c = ease.prog(t, HOT_CLAMP_T[0], HOT_CLAMP_T[1])
    if c > 0:
        hot = (_LUMA[to_index(np.ascontiguousarray(Ls[..., :3]))] > _LUMA_LEAF7 + 1e-6) & (Ls[..., 3] > 0)
        hot &= _B4[y0:y1, x0:x1] < c
        Ls[hot, :3] = GL_RGB[7]


HOT_CLAMP_T = (1.40, 1.50)
_LUMA = palette.PAL_LAB[:, 0]
_LUMA_LEAF7 = float(_LUMA[palette.NAME_TO_IDX['leaf7']])
_GLOW_CORE = [(2.6, 5), (1.6, 6), (1.0, 7), (0.4, 8)]   # solid rings: (radius, GL index)


def _glow_into(L, ex, ey, t, er):
    """Fallback when the close-up can't be rendered here: a self-drawn green
    lens light (solid core rings + a soft halo with narrow dithered seams)."""
    k = ease.smoothstep(ease.prog(t, FADE_T[0], FADE_T[1]))
    b = breath(t)
    s = (0.35 + 0.65 * k) * (0.93 + 0.17 * b) * (er / 3.0)
    R = int(math.ceil(11 * er / 3.0))
    x0, x1 = max(ex - R, 0), min(ex + R + 1, W)
    y0, y1 = max(ey - R, 0), min(ey + R + 1, H)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.ogrid[y0:y1, x0:x1]
    d = np.sqrt((xs - ex) ** 2 + (ys - ey) ** 2)
    sub = L[y0:y1, x0:x1]
    rc = 2.6 * s
    hv = (3.3 * k + 0.3 * b) * np.clip(1.0 - (d - rc) / (6.5 * s), 0.0, 1.0) ** 1.8
    hq = np.clip(_quant(hv, _B4[y0:y1, x0:x1], 0.4), 0, 4)
    m = (hq > 0) & (d > rc)
    sub[m, :3] = GL_RGB[hq[m]]
    sub[m, 3] = 255
    for rad, ci in _GLOW_CORE:
        m = d <= rad * s + 0.25
        sub[m, :3] = GL_RGB[ci if (ci < 8 or k >= 0.6) else 7]
        sub[m, 3] = 255


# =====================================================================  PLACEMENT
_CACHE = {}


def _static(eye):
    """Layout for an eye position (cached)."""
    key = (int(round(eye[0])), int(round(eye[1])))
    if key in _CACHE:
        return _CACHE[key]
    w = TA['w']
    x0 = int(round(160 - w / 2.0))
    lower = key[1] < 100              # eye high in frame -> title in the lower third
    y_cap = 131 if lower else 30
    rule_y = y_cap + CAP + 6
    # the S of SEED (the second S) -> seedling base on its cap line, above the top arc
    s_i = [i for i, (ch, _) in enumerate(TA['lay']) if ch == 'S'][1]
    s_x = TA['lay'][s_i][1]
    tw = [(x0 + x + _TWINK_AT[ch][0], y_cap + _TWINK_AT[ch][1]) for ch, x in TA['lay']]
    st = dict(y_cap=y_cap, rule_y=rule_y, lower=lower,
              ax=x0 - PAD, ay=y_cap - PAD, cx=x0 + w // 2,
              sprout=(x0 + s_x + 4, y_cap - 1), twinkles=tw)
    _CACHE[key] = st
    return st


# =====================================================================  RENDER
def render(t):
    t = float(t)
    L = layer()
    if t < 0:
        return {'layer': L, 'under_fade': 0.0}
    t = min(t, DURATION + 1.0)
    st = _static(eye_pos(0.0))
    U = under_fade(t)
    # 1. the robot's eye: the last light
    _eye_into(L, t, U)
    # 2. ornament rule
    _rule_into(L, st['cx'], st['rule_y'], t)
    # 3. the letters, each caught by a twinkle as it completes
    _title_into(L, t, st['ax'], st['ay'])
    for i, (x, y) in enumerate(st['twinkles']):
        _sparkle_into(L, x, y, t, LET_T0 + i * LET_STEP + 0.62 * LET_DUR, _TWINK, _TWINK_T)
    # 4. the seedling on the S of SEED
    bx, by = st['sprout']
    _sprout_into(L, bx, by, t)
    # 5. sparkles: the rule's diamond, the opened seedling
    _sparkle_into(L, st['cx'], st['rule_y'], t, SPARK_DIAMOND_T)
    _sparkle_into(L, bx + 4, by - 9, t, SPARK_SPROUT_T)
    return {'layer': L, 'under_fade': U}
