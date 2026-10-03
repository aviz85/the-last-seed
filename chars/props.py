"""
chars/props.py — the SPROUT and the WATERING CAN as stand-alone props.   (CHAR-A)

sprout(state, t=0.0, scale='medium', lean=0.0, wind=0.0) -> Sprite
    anchor 'base' = where the stem meets the soil (blit it on the ground line).
    state: 'ground'   growing from a crack in dead earth, split seed husk at its base,
                      2 small leaves, one drooping
           'wilt'     drooping, leaves hanging (dying)
           'lean'     stem bends toward a light; lean in [-1, 1] (-1 = fully left, +1 = fully right)
           'planted'  upright, a little healthier (no husk)
    wind: 0..1 trembling amount (animates with t)
    medium scale: ~7 px tall. The ONLY saturated green in the dead world ('leaf' ramp).

can(state='empty', scale='medium') -> Sprite
    the same small dented rusty watering can the robot carries (spout right, handle left).
    anchors: 'base' (bottom centre, on the ground line), 'mouth' (top opening, where the
    sprout sits), 'spout' (tip of the rose), 'grip' (the handle).
    state: 'empty' | 'sprout' | 'sprout_wilt' | 'sprout_lean' (the sprout in a bit of soil)

clump(state='sprout', t=0.0) -> Sprite  (anchor 'base' = same as sprout('ground'))
    the dug-up soil clump with the sprout, used inside robot 'scoop'.

can_pose(pose, state, scale) -> Sprite   (internal, used by chars/robot.py)
    poses: 'up', 'up_su', 'up_sd' (walk sway), 'tilt1'..'tilt3' (watering), 'up_l'
    (standing with the spout LEFT), 'front' (mid-swing, spout toward camera).
"""
import math
import numpy as np
from engine import palette as P
from engine.sprite import Sprite, from_ascii, blit

# ================================================================== can
_CAN_LEG = {
    'K': 'steel0', 'a': 'steel1', 's': 'steel2', 'S': 'steel3', 't': 'steel4', 'T': 'steel5',
    'u': 'rust1', 'r': 'rust2', 'R': 'rust3', 'o': 'rust4', 'O': 'rust5',
    'd': 'dust2', 'D': 'dust3', 'e': 'dust4',
}


def _rows(art):
    return [l for l in art.strip('\n').split('\n')]


def _shear_cols(art, anchors, col, dy):
    """columns >= col moved by dy rows (pure pixel translation)."""
    rows = _rows(art)
    h, w = len(rows), max(len(r) for r in rows)
    pad = abs(dy)
    g = [['.'] * w for _ in range(h + 2 * pad)]
    for x in range(w):
        d = dy if x >= col else 0
        for y in range(h):
            ch = rows[y][x] if x < len(rows[y]) else '.'
            if ch != '.':
                g[y + pad + d][x] = ch
    anc = {}
    for k, (x, y) in anchors.items():
        anc[k] = (x, y + pad + (dy if x >= col else 0))
    return '\n'.join(''.join(r) for r in g), anc


CAN_ART = {
    # upright, spout right, handle loop on the back (left)
    'up': ("""
...........KK
...KKKK...KoK
.KKtaaSK.KoK.
K.KToRRKKoK..
K.KoRRRooK...
.KKrRrrKK....
...KKKK......
""", {'grip': (0, 3), 'mouth': (4, 2), 'spout': (11, 1), 'base': (4, 6)}),
    # ~20 deg forward tilt (rotating about the grip)
    'tilt1': ("""
...KK........
..KtaKK......
.KKToaSK...KK
K.KoRRRKKKKoK
K.KRRRRRoooK.
.KKKrRrRKKK..
....KKrrK....
......KK.....
""", {'grip': (0, 3), 'mouth': (4, 1), 'spout': (11, 3), 'base': (6, 7)}),
    # ~45 deg: pouring
    'tilt2': ("""
..KKK........
.K.taK.......
K.KToaK......
K.KoRaSK.....
.KKRRRRK.....
..KrRRRRKK...
...KrRRoooK..
....KrKKKKooK
.....K.....KK
""", {'grip': (0, 2), 'mouth': (4, 1), 'spout': (11, 7), 'base': (6, 7)}),
    # ~60 deg: steep pour (full tilt)
    'tilt3': ("""
..KK.......
.K.tK......
K.KTaK.....
K.KoaaK....
.KRRRaSK...
.KrRRRRK...
..KrRRRoK..
...KrRKoK..
....KK.KoK.
........KoK
.........K.
""", {'grip': (0, 2), 'mouth': (4, 2), 'spout': (9, 9), 'base': (5, 8)}),
}
# mirrored (spout LEFT, handle right) — re-lit from the upper left, not a plain flip
CAN_ART['up_l'] = ("""
KK...........
KoK...KKKK...
.KoK.KtaaSKK.
..KoKKToRRK.K
...KoooRRRK.K
....KKrrRrKK.
......KKKK...
""", {'grip': (12, 3), 'mouth': (8, 2), 'spout': (1, 1), 'base': (8, 6)})
# seen at 3/4 (mid-swing, spout foreshortened toward the camera)
CAN_ART['front'] = ("""
..KKKK...
.KtaaSK..
KToRRRRK.
KoRRRRrKK
KRRRRRKOK
KrRRrrKK.
.KKKKKK..
""", {'grip': (0, 3), 'mouth': (4, 1), 'spout': (7, 4), 'base': (4, 6)})
CAN_ART['up_su'] = _shear_cols(*CAN_ART['up'], 10, -1)
CAN_ART['up_sd'] = _shear_cols(*CAN_ART['up'], 10, 1)
CAN_POSES = tuple(CAN_ART)

# ================================================================== sprout
_SP_LEG = {
    '1': 'leaf1', '2': 'leaf2', '3': 'leaf3', '4': 'leaf4', '5': 'leaf5', '6': 'leaf6', '7': 'leaf7', '8': 'leaf8',
    'h': 'ochre3', 'H': 'ochre5', 'j': 'ochre6', 'k': 'ochre2',
    'd': 'dust1', 'D': 'dust2', 'e': 'dust3', 'E': 'dust4', 'b': 'bark1', 'B': 'bark2',
}


def _sp(art):
    """sprout art; the anchor 'base' is the lowest stem pixel ('2'/'3' in the last row)."""
    rows = art.strip('\n').split('\n')
    last = rows[-1]
    bx = next(i for i, ch in enumerate(last) if ch in '1234')
    return (art, {'base': (bx, len(rows) - 1)})


# growing from a crack: split seed husk at the base, right leaf up, left leaf drooping
SP_GROUND = [_sp("""
....677.
...45667
.55.3...
5643....
4...3...
...32...
...2....
..h2Hj..
"""), _sp("""
...677..
..45667.
.55.3...
5643....
4...3...
...32...
...2....
..h2Hj..
"""), _sp("""
...677..
..5667..
.55.3...
564.3...
4...3...
...32...
...2....
..h2Hj..
"""), _sp("""
....667.
...45677
..5.3...
.5643...
.4..3...
...32...
...2....
..h2Hj..
""")]
# upright & a little healthier (planted in the greenhouse soil)
SP_PLANTED = [_sp("""
.67..76.
5665.564
..4564..
...43...
...3....
...3....
...2....
"""), _sp("""
.67...7.
5665.566
..4564..
...43...
...3....
...3....
...2....
"""), _sp("""
67...76.
665..564
.45564..
...43...
...3....
...3....
...2....
""")]
# wilting -> reaching toward a light on the RIGHT (index 0 = wilted .. 4 = reaching)
SP_LEAN_R = [_sp("""
..3443..
.3...33.
.3...342
.3...3.3
.3....3.
.2......
.2......
"""), _sp("""
..34443.
.3....43
.3...343
.3...3.3
.3......
.2......
.2......
"""), _sp("""
...3445.
..3..456
..3...45
.3....3.
.3......
.3......
.2......
"""), _sp("""
.....567
...34566
..3...5.
..3.....
.3......
.3......
.2......
"""), _sp("""
......78
....5677
...4.667
..43..5.
..3.....
.3......
.2......
""")]
# wilting (drooped to the right) -> straightening -> reaching toward a light on the LEFT
SP_LEAN_L = [SP_LEAN_R[0], _sp("""
..344...
.3...4..
.3..343.
.3...3..
.3......
.2......
.2......
"""), _sp("""
.45.54.
4..3..4
...3...
...3...
...3...
...2...
...2...
"""), _sp("""
765.....
66543...
.5...3..
.....3..
......3.
......3.
......2.
"""), _sp("""
87......
7765....
766.4...
.5..34..
.....3..
......3.
......2.
""")]
SP_WILT = [SP_LEAN_R[0], _sp("""
..3443..
.3...33.
.3...34.
.3...3.2
.3....3.
.2......
.2......
""")]
# wide scale (~3 px): a green glint
SP_WIDE = {'ground': _sp("""
.6
5.
.3
"""), 'wilt': _sp("""
33
3.
2.
"""), 'planted': _sp("""
5.6
.4.
.3.
"""), 'lean': _sp("""
6.
.4
3.
""")}


def _spr(art, leg, anchors):
    return from_ascii(art, leg, dict(anchors))


def _wind_index(t, wind, n, seed=0):
    """deterministic flutter: which of n wind frames to show at time t (12 fps)."""
    if wind <= 0.02 or n <= 1:
        return 0
    k = int(math.floor(t * 12 + 1e-6))
    # hashed per-frame gust strength
    h = ((k * 2654435761 + seed * 97) >> 7) & 0xFFFF
    g = (h / 65535.0) * 0.6 + 0.4 * (0.5 + 0.5 * math.sin(t * 5.3 + seed))
    if g < 1.0 - wind:
        return 0
    return 1 + (k % (n - 1))


def sprout(state='ground', t=0.0, scale='medium', lean=0.0, wind=0.0):
    if scale == 'wide':
        art, anc = SP_WIDE.get(state, SP_WIDE['ground'])
        return _spr(art, _SP_LEG, anc)
    if state == 'ground':
        frames = SP_GROUND
    elif state == 'planted':
        frames = SP_PLANTED
    elif state == 'wilt':
        frames = SP_WILT
    elif state == 'lean':
        seq = SP_LEAN_L if lean < 0 else SP_LEAN_R
        i = int(round(min(abs(lean), 1.0) * 4))
        art, anc = seq[i]
        return _spr(art, _SP_LEG, anc)
    elif state == 'can':
        frames = SP_GROUND
    else:
        frames = SP_GROUND
    art, anc = frames[_wind_index(t, wind, len(frames))]
    return _spr(art, _SP_LEG, anc)


CLUMP_SOIL = [(1, 4, 'e'), (2, 4, 'D'), (4, 4, 'D'), (5, 4, 'e'),
              (0, 5, 'e'), (1, 5, 'D'), (2, 5, 'd'), (4, 5, 'd'), (5, 5, 'D'), (6, 5, 'e'),
              (1, 6, 'D'), (2, 6, 'd'), (4, 6, 'd'), (5, 6, 'D')]


def clump(state='sprout', t=0.0):
    """The dug-up soil clump with the sprout (+ its seed husk), as carried in the
    robot's hands during 'scoop'. Anchor 'base' = the sprout's base, identical
    to sprout('ground')'s 'base' so the hand-over at SCOOP_LIFT_T has no pop."""
    s = sprout('ground', t)
    img = s.img.copy()
    for x, y, ch in CLUMP_SOIL:
        img[y, x, :3] = P.C(_SP_LEG[ch])
        img[y, x, 3] = 255
    return Sprite(img, {'base': s.anchors['base'], 'hold': (3, 6)})


def _can_sprout(state, t=0.0, wind=0.0, lean=None):
    if state == 'sprout_wilt':
        return sprout('wilt', t, wind=wind)
    if state == 'sprout_lean':
        return sprout('lean', t, lean=-1.0 if lean is None else lean)
    return sprout('ground', t, wind=wind)


def can_pose(pose='up', state='empty', scale='medium', t=0.0, wind=0.0, lean=None):
    art, anc = CAN_ART[pose]
    s = _spr(art, _CAN_LEG, anc)
    if state != 'empty':
        sp = _can_sprout(state, t, wind, lean)
        mx, my = s.anchors['mouth']
        pad = 7
        img = np.zeros((s.h + pad, s.w, 4), np.uint8)
        img[pad:] = s.img
        anc2 = {k: (x, y + pad) for k, (x, y) in s.anchors.items()}
        s = Sprite(img, anc2)
        for dx, c in ((-1, 'dust3'), (0, 'dust2'), (1, 'dust3')):
            if s.img[my + pad, mx + dx, 3]:
                s.img[my + pad, mx + dx, :3] = P.C(c)
        # the husk rides along in the clump for the healthy state; plain soil otherwise
        blit(s.img, sp, mx, my + pad, anchor='base')
    return s


def can(state='empty', scale='medium', t=0.0, wind=0.0, lean=None):
    """see module doc. Optional t/wind animate the sprout inside; lean (-1..1) overrides
    the 'sprout_lean' amount (default -1 = fully toward the left)."""
    if scale == 'wide':
        img = np.zeros((3, 3, 4), np.uint8)
        for (x, y, c) in ((0, 1, 'rust3'), (1, 1, 'rust4'), (0, 2, 'rust2'), (1, 2, 'rust3'), (2, 1, 'rust2')):
            img[y, x, :3] = P.C(c); img[y, x, 3] = 255
        if state != 'empty':
            img[0, 1, :3] = P.C('leaf5'); img[0, 1, 3] = 255
        return Sprite(img, {'base': (1, 2), 'mouth': (1, 0), 'spout': (2, 1), 'grip': (0, 1)})
    return can_pose('up', state, scale, t, wind, lean)
