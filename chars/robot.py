"""
chars/robot.py — THE GARDENER: animation API used by every shot.   (CHAR-A)

    from chars import robot
    from engine.sprite import blit, anchor_pos
    spr = robot.frame('walk', t_anim, can='empty', variant='normal', scale='medium')
    blit(img, spr, x, ground_y, anchor='feet', flipx=False)
    ex, ey = anchor_pos(spr, x, ground_y, 'feet', 'eye')    # eye glow position
    cx, cy = anchor_pos(spr, x, ground_y, 'feet', 'core')   # chest core light source

frame(anim, t, can=None, variant='normal', scale='medium', core=1.0, eye=1.0, face_left=False)
  anim     animation name (see ANIMS). t = seconds since THIS animation began
           (the shot controls timing; one-shot anims hold their last pose
           when t > duration — and the holds keep breathing).
  can      None | 'empty' | 'sprout' | 'sprout_wilt' | 'sprout_lean'
           (watering can in hand, with/without the sprout in it)
  variant  'normal' | 'damaged'  (damaged = after the storm: antenna snapped
           to a stub, dent + paint chip top-left of head, cracked backpack)
  scale    'medium' (26 px tall, full anim set) | 'wide' (~11 px, subset)
  core     0..1 chest-core glow brightness (0 = dark / removed)
  eye      0..1 lens pupil glow (0 = powered off)
  face_left  mirror the sprite (anchors mirrored too)

Every returned Sprite has anchors:
  'feet' (ground contact, bottom-centre), 'eye', 'core', 'hand',
  'can_mouth' (top opening of the can, where the sprout sits; = hand if no can)
  'spout'     (tip of the can's spout, when a can is held — the S2 water drop falls from here)
  'core_held' (only in open_chest/place_core: the removed core's centre, for the scene's light)

duration(anim) -> nominal seconds for one-shot anims (loops: one cycle).

MOVING THE ROBOT WITHOUT FOOT SLIDING
  WALK_SPEED = 10.0 px/s  (medium 'walk': 1 px per 10-fps frame, 8 px per 0.8 s cycle)
  WALK_TIRED_SPEED = 6.0 px/s, BRACE_SPEED = 4.0 px/s, WIDE_WALK_SPEED = 4.0 px/s
  travel(anim, t, scale='medium') -> exact ground distance (px, integer, stepped with the
      anim frames) covered after t seconds. For zero foot slide, place the robot at
          x = x0 + robot.travel('walk', t)            (not x0 + WALK_SPEED * t)
      (x0 + SPEED * t is within 1 px of it if you prefer a Stage sub-pixel glide.)

PROP-HANDOFF CONSTANTS (contract between character + shot artists):
  SCOOP_SPROUT_OFFSET = (13, 0)  sprout base relative to the KNEELING robot's feet
  SCOOP_LIFT_T  = 0.75           s into 'scoop' when the soil clump leaves the ground
                                 (the shot stops drawing the ground sprout after this;
                                 from 0.75 the anim draws the clump+sprout in the hands,
                                 pixel-identical to props.sprout('ground') at that moment)
  SCOOP_INTO_CAN_T = 1.1         s into 'scoop' when the sprout is in the can
                                 (from then on the anim draws the can with the sprout)
  SCOOP_CAN_OFFSET = (-12, 0)    'scoop'/'kneel'(with can)/'rise_hold' draw the can
                                 themselves, standing on the ground at the kneeling robot's
                                 back hip on the camera side, fully visible (base anchor at
                                 feet + offset, spout pointing left, handle toward the robot).
                                 The clump flies into it in a 15-fps arc 0.75-1.1; rise_hold
                                 picks it up and swings it round to the belly.
  WATER_TILT_T = 0.6             s into 'water' when the can is fully tilted
In chars/robot_acts.py (CHAR-B):
  SIT_CAN_DOWN_T = 0.45          s into 'sit' when the can is set on the ground;
  CAN_SET_OFFSET = (13, 0)       where (relative to the SITTING robot's feet) — after
                                 this the SHOT draws props.can() + props.sprout() there.
  PLANT_T = 1.2, PLANT_OFFSET = (15, -11)   'dig_plant': sprout placed in soil (relative
                                 to KNEELING feet; planter soil surface ~11 px above floor)
  CORE_PLACE_T = 0.4, CORE_PLACE_OFFSET = (11, -11)   'place_core': core touches soil
  POWER_EYE_OFF_T = 0.9          s into 'power_down' when the eye goes dark

KNEEL POSE (end of 'kneel', start pose of CHAR-B's dig_plant/open_chest/place_core):
  see the KNEEL comment block below (pixel layout + anchors).

ANIMATION REFERENCE (medium, 10 fps unless noted; t in s from the anim start)
  idle       loop 2.0 (20 fr). breath: body down 1 px 0.9-1.6, head drags 1.7, spring antenna.
             even loops: iris blink 0.3-0.6, a small nod 1.1-1.5; odd loops: glances up at
             the sky 0.3-0.7. The antenna bulb blinks like a status LED.
  walk       loop 0.8 = 8 fr (contact/down/pass/up x2), 1 px per frame = 10 px/s.
             can None/'empty': calm plod (1-px bob, nod, spring antenna, can swings, spout rocks)
             can 'sprout*'  : buoyant bounce (2 px, head tips up proudly), can hugged, sprout bobs.
  walk_tired loop 1.0 = 10 fr, 6 px. 0.1-0.3 painful hitch onto the hurt near leg (hunch, dip,
             wince), 0.4-0.9 good leg carries while the hurt foot DRAGS on its toe (dust).
  brace      12 fps. one heavy 3-px step per 0.75 s (1.5 s double cycle), 4 px/s. Leaning ~18 deg
             into the wind from the RIGHT, head bowed, iris squinting, antenna whipping back,
             can pressed to the belly, sprout thrashing.
  shield     0.0 flinch (eyes shut) 0.2 turning (crouched front view, can + sprout hugged
             to the chest, pupil clear) 0.3 slams down curled,
             FACING LEFT, backpack to the wind 0.4 settle; >= 0.5 hold: 2-frame tremble loop,
             antenna flutter, the iris squeezes shut now and then.
  unshield   EXTRA 0.5 (not in the original list): shield hold -> turn -> brace pose.
  water      0.1 anticipation dip, 0.2-0.3 can swung up in front, 0.4 tilt ~20, 0.5 ~45,
             0.6 FULL TILT ~60 (WATER_TILT_T), 0.7-1.0 coaxing 1-px jiggle, hold to 1.4, 1.4-1.5
             un-tilt, 1.6-1.9 lower, 2.0 settle dip, 2.1 carry. 'spout' = rose tip on every frame.
             (raised arms are routed under / over the porthole: the core is never covered)
  droop      0.1 head sinks, 0.2 shoulders slump + can lowered onto the dirt, 0.3 sink, 0.4
             settle bounce; hold: slow 2.4 s breathing, antenna sagging, a tired half-blink.
  look_down  0.1-0.2 head tips down (pupil low), 0.3 overshoot; hold breathes.
  look_up    0.1-0.2 head tips up, torso leans back (awe), 0.3 overshoot; hold, a slow blink.
  startle    0.0 anticipation squash, 0.1 hop 2 px + head snaps up/forward + pupil flares,
             0.3 landing squash, 0.4 settle leaning in; hold: bright stare, antenna quiver.
  kneel      0.1 front foot steps forward, 0.2 drops, 0.3 overshoot, >= 0.4 KNEEL POSE.
             with a can: swings it round and sets it down behind (SCOOP_CAN_OFFSET).
  scoop      0.1 lean in, 0.2-0.7 dig strokes at 0.3/0.5 with dirt flicked back, 0.75 clump
             (SCOOP_LIFT_T), 0.817-1.067 the clump arcs up past the chest and down into the
             can at its back hip (15 fps in-betweens), 1.1 in the can (SCOOP_INTO_CAN_T),
             1.2-1.3 pat pat, 1.4 bulb blinks.
  rise_hold  0.1 grab the can behind, 0.2 swing it round (3/4 view) while rising, 0.3 up,
             hugging, 0.4 happy hop, 0.5-0.7 settle; hold: tender sway (2 s), head nuzzling.
  wide       idle, walk (10 fr/1.0 s, 4 px/s), walk_tired (6 fps), brace (4 fps), kneel —
             11 px tall: lens glint + amber pupil, core pixel, antenna bulb, 3-px rust can.
"""
import math
import numpy as np
from engine import palette as P
from engine.sprite import Sprite, flip
from engine.draw import line_points
from chars import model, props

CW, CH = 48, 36
OX, OY = 12, 10
FEET = (8, 25)

_LEG = {
    'K': 'steel0',
    '0': 'paint0', '1': 'paint1', '2': 'paint2', '3': 'paint3', '4': 'paint4', '5': 'paint5', '6': 'paint6',
    'u': 'rust1', 'r': 'rust2', 'R': 'rust3', 'o': 'rust4', 'O': 'rust5',
    'a': 'steel1', 's': 'steel2', 'S': 'steel3', 't': 'steel4', 'T': 'steel5', 'W': 'steel6',
    'g': 'ink2', 'n': 'ink1',
    'G': 'ash5', 'b': 'ash3', 'B': 'ash6', 'c': 'ash4',
    'V': 'core6', 'Y': 'core5', 'y': 'core3', 'w': 'core2',
    'd': 'dust2', 'D': 'dust3', 'e': 'dust4', 'E': 'dust5', 'f': 'dust6', 'F': 'dust7',
    'l': 'leaf3', 'L': 'leaf5',
}
_DMG = {'x': ('paint5', 'steel2'), 'X': ('paint5', 'steel3'), 'z': ('paint5', 'steel1'),
        'Z': ('ash5', 'steel1'), 'j': ('steel3', 'steel1'), 'k': ('steel0', 'steel3')}
_CORE_TAB = [
    ('steel2', 'steel1', 'steel1', 'ink2'),
    ('core1', 'core0', 'core0', 'steel1'),
    ('core2', 'core1', 'core1', 'core0'),
    ('core3', 'core2', 'core2', 'core1'),
    ('core4', 'core3', 'core3', 'core2'),
    ('core5', 'core4', 'core4', 'core2'),
    ('core5', 'core5', 'core4', 'core3'),
    ('core6', 'core5', 'core5', 'core3'),
    ('core6', 'core5', 'core5', 'core3'),
]
_EYE_TAB = [
    ('steel1', 'ink2', 'ink2'),
    ('core1', 'core0', 'core0'),
    ('core2', 'core1', 'core1'),
    ('core3', 'core2', 'core1'),
    ('core4', 'core3', 'core2'),
    ('core5', 'core4', 'core2'),
    ('core5', 'core4', 'core3'),
    ('core6', 'core5', 'core3'),
    ('core6', 'core5', 'core3'),
    ('core6', 'core6', 'core4'),
]
_NCODE = 256 + P.N
_LUTS = {}


def _rgba(name):
    r, g, b = P.C(name)
    return (r, g, b, 255)


def _lut(variant, core, eye):
    key = (variant, core, eye)
    lut = _LUTS.get(key)
    if lut is not None:
        return lut
    lut = np.zeros((_NCODE, 4), np.uint8)
    for ch, nm in _LEG.items():
        lut[ord(ch)] = _rgba(nm)
    dmg = 1 if variant == 'damaged' else 0
    for ch, pair in _DMG.items():
        lut[ord(ch)] = _rgba(pair[dmg])
    for ch, nm in zip('()[]', _CORE_TAB[core]):
        lut[ord(ch)] = _rgba(nm)
    for ch, nm in zip('PQq', _EYE_TAB[min(eye, 9)]):
        lut[ord(ch)] = _rgba(nm)
    lut[ord('h')] = _rgba('white0' if eye >= 3 else ('steel5' if eye >= 1 else 'steel4'))
    lut[ord('*')] = _rgba('core5' if core >= 4 else ('core4' if core >= 2 else 'core2'))
    lut[ord('&')] = _rgba('core7' if core >= 4 else ('core5' if core >= 2 else 'core3'))
    for i in range(P.N):
        lut[256 + i] = (*P.PAL[i], 255)
    lut[0] = (0, 0, 0, 0)
    _LUTS[key] = lut
    return lut


class _Part:
    __slots__ = ('g', 'anc', 'w', 'h')

    def __init__(self, art, anc):
        lines = art.strip('\n').split('\n')
        ind = min((len(l) - len(l.lstrip(' ')) for l in lines if l.strip()), default=0)
        lines = [l[ind:].rstrip() for l in lines]
        self.h = len(lines)
        self.w = max(len(l) for l in lines)
        g = np.zeros((self.h, self.w), np.uint16)
        for y, l in enumerate(lines):
            for x, ch in enumerate(l):
                if ch not in '. ':
                    g[y, x] = ord(ch)
        self.g = g
        self.anc = dict(anc)


def _part(art, **anc):
    return _Part(art, anc)


def _shear_rows(art, shifts):
    rows = [l for l in art.strip('\n').split('\n')]
    return '\n'.join('.' * s + r for s, r in zip(shifts, rows))


_TORSO_ART = """
.....KsSsK....
KKKKKKKKKKKKK.
KSZK45544KK43K
KSaK4444K()K3K
KjGK3444K[]K2K
KsGK33333KK32K
KssK2r3333R22K
KKKK22r22Rr21K
....KKKKKKKKK.
"""
_TA = dict(neck=(7, -1), shoulder=(6, 3), core=(9, 3), hip=(8, 8), fshoulder=(9, 3))


def _lean_anc(shifts):
    out = {}
    for k, (x, y) in _TA.items():
        r = min(max(y, 0), len(shifts) - 1)
        out[k] = (x + shifts[r], y)
    return out


_LEANS = {'n': [0] * 9, 'l1': [1, 1, 1, 1, 1, 1, 0, 0, 0], 'l2': [2, 2, 1, 1, 1, 1, 0, 0, 0],
          'lb': [3, 3, 2, 2, 2, 2, 1, 1, 0],
          'lk': [-1, -1, 0, 0, 0, 0, 0, 0, 0],          # leaning back (awe)
          'l3': [3, 3, 2, 2, 2, 2, 1, 0, 0]}
TORSO = {k: _Part(_shear_rows(_TORSO_ART, v), _lean_anc(v)) for k, v in _LEANS.items()}

HEAD = {
    'n': _part("""
..KKkKKKKK.....
.KxX544444KKK..
Kz5444443KtTSK.
K5444443KthggSK
K44TS433KSgPQsK
K44Ss333KSgQqaK
K33r33322KssaK.
.K222r2222KKK..
..KKKKKKKK.....
""", pivot=(5, 8), ant=(4, -1), eye=(11, 4), glass=(10, 3)),
    'd1': _part("""
..KKk..........
.KxX5KKKKK.....
Kz54444444KKK..
K54444443KtTSK.
K44TS443KthggSK
K44Ss433KSgPQsK
K33r3333KSgQqaK
.K2223322KssaK.
..KKKr2222KKK..
.....KKKKK.....
""", pivot=(5, 8), ant=(4, -1), eye=(11, 5), glass=(10, 4)),
    'u1': _part("""
.....KKKKK.....
..KKk44444KKK..
.KxX54443KtTSK.
Kz544443KthggSK
K5444433KSgPQsK
K44TS333KSgQqaK
K44Ss3322KssaK.
K33r3r2222KKK..
.K222KKKKK.....
..KKK..........
""", pivot=(5, 9), ant=(4, 0), eye=(11, 4), glass=(10, 3)),
    'd2': _part("""
..KKk..........
.KxX5KKK.......
Kz544444KK.....
K544444444KKK..
K44TS4433KtTSK.
K44Ss433KthggSK
K33r3333KSgPQsK
.K222332KSgQqaK
..KKKr222KssaK.
.....KKK22KKK..
........KK.....
""", pivot=(5, 8), ant=(4, -1), eye=(11, 6), glass=(10, 5)),
    'u2': _part("""
........KK.....
.....KKK44KKK..
..KKk4443KtTSK.
.KxX5444KthggSK
Kz544443KSgPQsK
K5444433KSgQqaK
K44TS3332KssaK.
K44Ss33222KKK..
K33r3r22KK.....
.K222KKK.......
..KKK..........
""", pivot=(5, 10), ant=(4, 1), eye=(11, 4), glass=(10, 3)),
}

GLASS = {
    'fwd':    ['hgg', 'gPQ', 'gQq'],
    'down':   ['hgg', 'ggg', 'gPQ'],      # lowered gaze: the iris narrows to a low slit
    'back':   ['hgg', 'PQg', 'Qqg'],
    'up':     ['hPQ', 'gQq', 'ggg'],
    'half':   ['hsS', 'sPS', 'SSs'],
    'closed': ['hsS', 'sSs', 'Ssa'],
    'wide':   ['hgg', 'gPP', 'gPQ'],
}

# kneel legs (local (0,0) = model (0,19)): front (near) knee up, boot planted
# at x 14..19; back (far) knee down under the hips, its boot lying behind.
KNEEL_LEGS = _part("""
....................
...............KKK..
..............KtTSK.
..............KSStK.
..KK..........KSStKK
.KrRKKKK......KRooRK
.KKKKKKK......KKKKKK
""", feet=(8, 6))


# crouch (shield): both boots planted, knees folded under the lowered torso
CROUCH_LEGS = _part("""
.KKK....KKKK.
KsaK...KStSK.
KrRrK..KRooRK
KKKKK..KKKKKK
""", feet=(8 - 1, 3))

# FRONT VIEW (facing the camera) — the mid-turn frame of 'shield' / 'unshield'.
# A deep crouch: head bowed low over the sprout, shoulders hunched, knees splayed,
# both arms wrapped round the can (3/4 view, spout toward camera) with the sprout
# pressed to its chest.  Same height as the curl around it (top y = -21).
# FRONT_HEAD: local (0,0); 'pivot' = chin centre.  Pupil low (looking at the sprout).
FRONT_HEAD = _part("""
.......K*K........
........S.........
...KKKKKkKKKKKK...
..Kx544444444322K.
.KTK54KKKKK4432KSK
.KsK4KstTSaK432KsK
.K.K4KthgggK322K.K
..K44KSgggaK32K...
..K44KSgPQaK32K...
..K33KssaaaK22K...
..K333KKKKK222K...
...KK2222222KKK...
""", pivot=(8, 11), eye=(8, 8), glass=(7, 6))
# FRONT_BODY: shoulders + porthole (hidden behind the sprout) + splayed crouch legs;
# 'feet' = bottom centre.  Arms are drawn separately, over the can.
FRONT_BODY = _part("""
.KKK...............KKK.
KSSSK.............KSStK
KS4KKKKKKKKKKKKKKKK32SK
KSK4455444444443322KSK.
.KK44KK()KK4443322KK...
..K44K[]]K44433322K....
..K3333KKKK33332222K...
.KK2222222222222222KK..
KSSK...............KSSK
KStSK.............KSStK
KrRRK.............KRooK
KKKKK.............KKKKK
""", feet=(11, 11), core=(7, 4))


class _G:
    def __init__(self):
        self.g = np.zeros((CH, CW), np.uint16)
        self.anc = {}

    def put(self, part, anchor, mx, my):
        ax, ay = part.anc[anchor]
        self._paste(part.g, OX + mx - ax, OY + my - ay)

    def _paste(self, src, x0, y0):
        h, w = src.shape
        X0, Y0 = max(x0, 0), max(y0, 0)
        X1, Y1 = min(x0 + w, CW), min(y0 + h, CH)
        if X0 >= X1 or Y0 >= Y1:
            return
        s = src[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
        d = self.g[Y0:Y1, X0:X1]
        m = s != 0
        d[m] = s[m]

    def put_sprite(self, spr, anchor, mx, my):
        ax, ay = spr.anchor(anchor)
        self._paste(_sprite_codes(spr), OX + mx - ax, OY + my - ay)
        return mx - ax, my - ay

    def px(self, mx, my, ch):
        x, y = OX + mx, OY + my
        if 0 <= x < CW and 0 <= y < CH:
            self.g[y, x] = ord(ch) if isinstance(ch, str) else ch

    def get(self, mx, my):
        x, y = OX + mx, OY + my
        if 0 <= x < CW and 0 <= y < CH:
            return int(self.g[y, x])
        return 0

    def outline_pts(self, pts, ch='K'):
        k = ord(ch)
        for (x, y) in pts:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if self.get(x + dx, y + dy) == 0:
                    self.px(x + dx, y + dy, k)


_RGB2IDX = {(int(r) << 16) | (int(g) << 8) | int(b): i for i, (r, g, b) in enumerate(P.PAL)}
_SC_CACHE = {}


def _sprite_codes(spr):
    img = spr.img
    key = img.tobytes()
    c = _SC_CACHE.get(key)
    if c is not None:
        return c
    codes = np.zeros(img.shape[:2], np.uint16)
    m = img[..., 3] > 127
    if m.any():
        rgb = img[..., :3][m].astype(np.int64)
        keys = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
        codes[m] = np.array([256 + _RGB2IDX.get(int(k), 0) for k in keys], np.uint16)
    _SC_CACHE[key] = codes
    return codes


def _poly(pts):
    out = []
    for i in range(len(pts) - 1):
        xs, ys = line_points(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
        seg = list(zip(xs, ys))
        if out:
            seg = seg[1:]
        out += seg
    if len(pts) == 1:
        out = [tuple(pts[0])]
    return out


def _leg(G, near, top, hip_x, dx, lift):
    shin = 'KStK' if near else 'KasK'
    boot_y = 24 - lift
    n = boot_y - top
    for i in range(max(n, 0)):
        y = top + i
        f = (i + 1) / (n + 1)
        x = hip_x + int(round(dx * f))
        for j, ch in enumerate(shin):
            G.px(x + j, y, ch)
    bx = hip_x + dx
    if near:
        boot, sole, bo = 'KRooRK', 'KKKKKK', 0
    else:
        boot, sole, bo = 'KrRrK', 'KKKKK', -1
    for j, ch in enumerate(boot):
        G.px(bx + bo + j, boot_y, ch)
    for j, ch in enumerate(sole):
        G.px(bx + bo + j, boot_y + 1, ch)


def _legs(G, spec, by):
    kind = spec[0]
    top = 22 + by
    if kind == 'stand':
        lift = spec[1] if len(spec) > 1 else 0
        _leg(G, False, top, 4, 0, lift)
        _leg(G, True, top, 9, 0, lift)
    elif kind == 'walk':
        _, ndx, nl, fdx, fl = spec[:5]
        hip = spec[5] if len(spec) > 5 else 0          # hips shifted (lean): feet stay put
        _leg(G, False, top, 4 + hip, fdx - hip, fl)
        _leg(G, True, top, 9 + hip, ndx - hip, nl)
    elif kind == 'crouch':
        G.put(CROUCH_LEGS, 'feet', FEET[0], FEET[1])
    elif kind == 'kneel':
        G.put(KNEEL_LEGS, 'feet', FEET[0], FEET[1])


def _arm(G, pts, near=True):
    pix = _poly(pts)
    for i, (x, y) in enumerate(pix):
        if near:
            ch = 't' if i == 0 else ('S' if i % 2 else 's')
        else:
            ch = 's' if i % 2 else 'a'
        G.px(x, y, ch)
    return pix


_DEFAULT = dict(fingers=None, swind=0.0, _t=0.0, flash=False, mirror=False, front=False, canstate=None, legs=('stand',), bx=0, by=0, torso='n', hx=0, hy=0, head='n', eye='fwd', eb=0,
                ant='up', arm=None, farm=None, can=None, clump=None, fx=None, farm_front=False)

ANT = {
    # stem pixels relative to the base, bulb position (a spring: it stretches & squashes)
    'up3':   ([(0, 0), (0, -1), (0, -2)], (0, -3)),
    'up1':   ([(0, 0)], (0, -1)),
    'back3': ([(0, 0), (0, -1), (-1, -2)], (-1, -3)),
    'back1': ([(0, 0)], (-1, -1)),
    'fwd3':  ([(0, 0), (0, -1), (1, -2)], (1, -3)),
    'fwd1':  ([(0, 0)], (1, -1)),
    'up':    ([(0, 0), (0, -1)], (0, -2)),
    'back':  ([(0, 0), (-1, -1)], (-1, -2)),
    'fwd':   ([(0, 0), (1, -1)], (1, -2)),
    'back2': ([(0, 0), (-1, -1)], (-2, -2)),
    'fwd2':  ([(0, 0), (1, -1)], (2, -2)),
    'low':   ([(0, 0), (1, 0)], (2, -1)),
}

REST_ARM = [(0, 0), (0, 2), (1, 3), (1, 5)]


def _compose(pose, can, variant):
    p = dict(_DEFAULT)
    p.update(pose)
    if p.get('front'):
        return _compose_front(p, can, variant)
    G = _G()
    bx, by = p['bx'], p['by']
    torso = TORSO[p['torso']]
    head = HEAD[p['head']]
    tx, ty = 8 + bx, 21 + by
    tl = (tx - torso.anc['hip'][0], ty - torso.anc['hip'][1])

    def T(name):
        a = torso.anc[name]
        return tl[0] + a[0], tl[1] + a[1]
    sh, fsh = T('shoulder'), T('fshoulder')
    canspec = p['can'] if (can is not None or p.get('canstate')) else None
    if p.get('canstate'):
        can = p['canstate']
    elif canspec is not None and can is None:
        can = 'empty'
    cspr = None
    can_at = None
    if canspec is not None:
        mode = canspec[0]
        if mode == 'hand':
            _, cpose, cdx, cdy = canspec[:4]
            cspr = props.can_pose(cpose, can, t=p.get('_t', 0.0), wind=p.get('swind', 0.0))
        else:                         # ('abs', pose, x, y, anchor, z)
            cspr = props.can_pose(canspec[1], can, t=p.get('_t', 0.0), wind=p.get('swind', 0.0))
    # far arm
    if p['farm'] is not None and not p['farm_front']:
        pix = _arm(G, [(fsh[0] + x, fsh[1] + y) for x, y in p['farm']], near=False)
        G.outline_pts(pix)
    if canspec is not None and canspec[0] == 'abs' and len(canspec) > 5 and canspec[5] == 'back':
        _, cpose, cx, cy, canc = canspec[:5]
        can_at = G.put_sprite(cspr, canc, cx, cy)
    _legs(G, p['legs'], by)
    G.put(torso, 'hip', tx, ty)
    nk = T('neck')
    hpx, hpy = nk[0] + p['hx'], nk[1] + p['hy']
    htl = (hpx - head.anc['pivot'][0], hpy - head.anc['pivot'][1])

    def H(name):
        a = head.anc[name]
        return htl[0] + a[0], htl[1] + a[1]
    ab = H('ant')
    if variant == 'damaged':
        G.px(ab[0], ab[1], 'r')
    else:
        stem, bulb = ANT[p['ant']]
        for (x, y) in stem:
            G.px(ab[0] + x, ab[1] + y, 'S')
        bx_, by_ = ab[0] + bulb[0], ab[1] + bulb[1]
        for dx, dy in ((0, -1), (-1, 0), (1, 0)):
            if G.get(bx_ + dx, by_ + dy) == 0:
                G.px(bx_ + dx, by_ + dy, 'K')
        G.px(bx_, by_, '&' if p.get('flash') else '*')
    G.put(head, 'pivot', hpx, hpy)
    gl = H('glass')
    for j, row in enumerate(GLASS[p['eye']]):
        for i, ch in enumerate(row):
            G.px(gl[0] + i, gl[1] + j, ch)
    arm = p['arm'] if p['arm'] is not None else REST_ARM
    pts = [(sh[0] + x, sh[1] + y) for x, y in arm]
    hand = pts[-1]
    if canspec is not None:
        if canspec[0] == 'hand':
            can_at = G.put_sprite(cspr, 'grip', hand[0] + cdx, hand[1] + cdy)
        elif not (len(canspec) > 5 and canspec[5] == 'back'):
            _, cpose, cx, cy, canc = canspec[:5]
            can_at = G.put_sprite(cspr, canc, cx, cy)
        ca = cspr.anchors
        G.anc['can_mouth'] = (can_at[0] + ca['mouth'][0], can_at[1] + ca['mouth'][1])
        G.anc['spout'] = (can_at[0] + ca['spout'][0], can_at[1] + ca['spout'][1])
    if p['clump'] is not None:
        cx, cy = p['clump'][:2]
        G.put_sprite(props.clump(), 'base', cx, cy)
    if p.get('fingers') and canspec is not None:
        mx_, my_ = G.anc['can_mouth']
        for (dx, dy, ch) in p['fingers']:
            G.px(mx_ + dx, my_ + dy, ch)
    if p['farm'] is not None and p['farm_front']:
        pix = _arm(G, [(fsh[0] + x, fsh[1] + y) for x, y in p['farm']], near=False)
        G.outline_pts(pix)
    pix = _arm(G, pts, near=True)
    G.outline_pts(pix[1:])
    if p['fx']:
        for (x, y, ch) in p['fx']:
            G.px(x, y, ch)
    G.anc.update(feet=FEET, eye=H('eye'), core=T('core'), hand=hand)
    G.anc.setdefault('can_mouth', hand)
    G.anc.setdefault('spout', hand)
    G.anc.setdefault('core_held', T('core'))
    if p.get('mirror'):
        _mirror(G)
    return G, p


def _mirror(G):
    """flip the composed grid about the feet column (the robot turns to face left
    while standing on the same spot)."""
    fx = OX + FEET[0]
    new = np.zeros_like(G.g)
    for X in range(CW):
        X2 = 2 * fx - X
        if 0 <= X2 < CW:
            new[:, X2] = G.g[:, X]
    G.g = new
    G.anc = {k: (2 * FEET[0] - x, y) for k, (x, y) in G.anc.items()}


def _compose_front(p, can, variant):
    """front-view crouch, hugging the can (+ sprout) in front of the chest."""
    G = _G()
    fx, fy = FEET
    G.put(FRONT_BODY, 'feet', fx, fy)
    btl = (fx - FRONT_BODY.anc['feet'][0], fy - FRONT_BODY.anc['feet'][1])
    # head sunk between the hunched shoulders (turtling): chin at y = -9
    hx, hy = fx, fy - 9
    G.put(FRONT_HEAD, 'pivot', hx, hy)
    htl = (hx - FRONT_HEAD.anc['pivot'][0], hy - FRONT_HEAD.anc['pivot'][1])
    if variant == 'damaged':
        for (x, y) in ((7, 0), (8, 0), (9, 0), (8, 1)):
            G.px(htl[0] + x, htl[1] + y, 0)
        G.px(htl[0] + 8, htl[1] + 1, 'r')
        G.px(htl[0] + 8, htl[1] + 2, 'S')
    st = can if can is not None else None
    cmx, cmy = fx, fy - 5
    if st is not None:
        cspr = props.can_pose('front', st, t=p.get('_t', 0.0), wind=p.get('swind', 0.0))
        at = G.put_sprite(cspr, 'base', fx + 3, fy)     # lagging a little to the right mid-turn
        ca = cspr.anchors
        cmx, cmy = at[0] + ca['mouth'][0], at[1] + ca['mouth'][1]
        G.anc['can_mouth'] = (cmx, cmy)
        G.anc['spout'] = (at[0] + ca['spout'][0], at[1] + ca['spout'][1])
    # arms: from both shoulders, round the can, hands meeting at its front
    left = [(btl[0] + 2, btl[1] + 3), (btl[0] + 3, btl[1] + 7), (fx, fy - 3)]
    right = [(btl[0] + 20, btl[1] + 3), (btl[0] + 19, btl[1] + 7), (fx + 5, fy - 3)]
    for pts, near in ((left, False), (right, True)):
        pix = _poly(pts)
        for i, (x, y) in enumerate(pix):
            ch = ('t' if i == 0 else ('S' if i % 2 else 's')) if near else ('s' if i % 2 else 'a')
            G.px(x, y, ch)
        G.outline_pts(pix[1:])
    eye = (htl[0] + FRONT_HEAD.anc['eye'][0], htl[1] + FRONT_HEAD.anc['eye'][1])
    core = (btl[0] + FRONT_BODY.anc['core'][0], btl[1] + FRONT_BODY.anc['core'][1])
    G.anc.update(feet=FEET, eye=eye, core=core, hand=(fx + 5, fy - 3))
    G.anc.setdefault('can_mouth', (fx, fy - 5))
    G.anc.setdefault('spout', G.anc['can_mouth'])
    G.anc['core_held'] = core
    return G, p


# ======================================================================
#  animation tables
# ======================================================================
def _step(keys, t):
    """keys: list of (t0, pose); returns pose of last key with t0 <= t."""
    cur = keys[0][1]
    for t0, pz in keys:
        if t + 1e-6 >= t0:
            cur = pz
        else:
            break
    return cur


def _carry_arm(hdx=0, hy=5):
    return [(0, 0), (0, 2), (1 + hdx // 2, 3), (1 + hdx, hy)]


# WALK: 8 frames @10 fps, 1 px per frame (10 px/s).  contact / down / pass / up x2.
# The prop sets the mood: with the EMPTY can (routine, lonely) it plods; carrying
# the SPROUT (life!) it bounces, head tipping up proudly at the top of each stride.
_WK_N = [(2, 0), (1, 0), (0, 0), (-1, 0), (-2, 0), (-1, 1), (1, 2), (2, 1)]
_WK = {
    'calm': dict(body=[0, 1, 0, 0, 0, 1, 0, 0], head=['n', 'd1', 'n', 'n', 'n', 'd1', 'n', 'n'],
                 hy=[0, 0, 1, 0, 0, 0, 1, 0], ant=['back', 'back3', 'back', 'up', 'back', 'back3', 'back', 'up']),
    'happy': dict(body=[0, 1, 0, -1, 0, 1, 0, -1], head=['n', 'd1', 'n', 'u1', 'n', 'd1', 'n', 'u1'],
                  hy=[0, 0, 1, 0, 0, 0, 1, 0], ant=['back', 'back3', 'back', 'back1', 'back', 'back3', 'back', 'back1']),
}


def _walk_pose(t, can):
    k = int(math.floor(t * 10 + 1e-6)) % 8
    n = _WK_N[k]
    f = _WK_N[(k + 4) % 8]
    W = _WK['happy' if (can is not None and can != 'empty') else 'calm']
    body = W['body'][k]
    pose = dict(legs=('walk', n[0], n[1], f[0], f[1]), by=body, head=W['head'][k], hy=W['hy'][k],
                ant=W['ant'][k])
    if can is None:
        sw = [-2, -1, 0, 1, 2, 1, 0, -1][k]
        pose['arm'] = [(0, 0), (0, 2), (1 + sw // 2, 3), (1 + sw, 5)]
    elif can == 'empty':
        # can hangs at the side, lagging the bounce, spout rocking (follow-through)
        hdx = [-1, -1, 0, 1, 1, 1, 0, -1][k]
        cy = [0, 0, -1, 0, 0, 0, -1, 0][k]
        pose['arm'] = _carry_arm(hdx, 5 - body + cy)
        pose['can'] = ('hand', ['up_su', 'up', 'up_sd', 'up', 'up_su', 'up', 'up_sd', 'up'][k], 0, 0)
    else:
        pose.update(_hug(body))
        pose['swind'] = 0.35
    return pose


def _hug(by=0, dx=0, dy=0, pose='up'):
    """both arms hug the can against the belly; can mouth right of the porthole."""
    return dict(arm=[(0, 0), (1, 3), (4, 5), (8, 5)],
                farm=[(0, 0), (4, 3), (7, 4)],
                fingers=[(3, 1, 's'), (3, 2, 'S')],          # far hand wrapping the can's front
                can=('abs', pose, 14 + dx, 20 + by + dy, 'mouth'))


def _idle_pose(t, can):
    """2 s breathing loop; even loops blink, odd loops glance up at the sky.
    The antenna bulb blinks like a status LED."""
    n = int(math.floor(t / 2.0))
    k = int(math.floor((t - 2.0 * n) * 10 + 1e-6))          # 0..19
    by = 1 if 9 <= k <= 16 else 0
    hy = 1 if k == 17 else 0
    head, eye, ant = 'n', 'fwd', 'up'
    if k == 9:
        ant = 'up3'
    elif k == 10:
        ant = 'fwd'
    elif k == 17:
        ant = 'up1'
    elif k == 18:
        ant = 'back'
    flash = k in (1, 2)
    if n % 2 == 0:
        if k in (3, 6):
            eye = 'half'
        elif k in (4, 5):
            eye = 'closed'
        elif 11 <= k <= 15:
            head = 'd1'
    else:
        if 3 <= k <= 7:
            eye = 'up'
            head = 'u1' if 4 <= k <= 6 else 'n'
            ant = 'back' if k == 4 else ant
        flash = k in (12, 13)
    pose = dict(by=by, hy=hy, head=head, eye=eye, ant=ant, flash=flash)
    if can == 'empty':
        pose['arm'] = _carry_arm(0, 5 - by)
        pose['can'] = ('hand', 'up_sd' if k == 9 else ('up_su' if k == 17 else 'up'), 0, 0)
    elif can is not None:
        pose.update(_hug(by))
        pose['swind'] = 0.15
    return pose


# --- water: raise (0-.6), full tilt at .6, coax (.6-1.4), lower (1.4-2.2)
# raised arms are routed under / over the porthole so the core never gets covered
A_LOW = [(0, 0), (0, 2), (4, 3)]                 # hand (10,19): can swung forward, low
A_MID = [(0, 0), (1, 3), (6, 1)]                 # hand (12,17): under the porthole
A_POUR = [(0, 0), (1, -2), (4, -2), (7, -1)]     # hand (13,15): over the porthole
A_POUR_UP = [(0, 0), (1, -2), (4, -2), (7, -2)]  # coax jiggle


def _water_pose(t, can):
    H = lambda pz, dy=0: ('hand', pz, 0, dy)
    A = [(0.0, dict(arm=_carry_arm(0, 5), can=H('up'))),
         (0.1, dict(by=1, head='d1', arm=_carry_arm(0, 4), can=H('up'), ant='fwd')),
         (0.2, dict(arm=A_LOW, can=H('up'), ant='back')),
         (0.3, dict(arm=A_MID, can=H('up'), head='n', ant='back')),
         (0.4, dict(arm=A_POUR, can=H('tilt1'), head='d1', ant='up')),
         (0.5, dict(arm=A_POUR, can=H('tilt2'), head='d1', eye='down', ant='fwd')),
         (0.6, dict(arm=A_POUR, can=H('tilt3'), head='d2', eye='down', ant='fwd')),
         (0.7, dict(arm=A_POUR_UP, can=H('tilt3'), head='d2', eye='down', ant='up')),
         (0.8, dict(arm=A_POUR, can=H('tilt3'), head='d2', eye='down', ant='fwd')),
         (0.9, dict(arm=A_POUR_UP, can=H('tilt3'), head='d2', eye='down', ant='up')),
         (1.0, dict(arm=A_POUR, can=H('tilt3'), head='d2', eye='down', ant='up')),
         (1.4, dict(arm=A_POUR, can=H('tilt2'), head='d2', eye='fwd')),
         (1.5, dict(arm=A_POUR, can=H('tilt1'), head='d1')),
         (1.6, dict(arm=A_MID, can=H('up'), head='d1')),
         (1.8, dict(arm=A_LOW, can=H('up'), head='n')),
         (1.9, dict(arm=_carry_arm(0, 5), can=H('up'), head='n', ant='back')),
         (2.0, dict(by=1, arm=_carry_arm(0, 4), can=H('up'), ant='fwd')),
         (2.1, dict(arm=_carry_arm(0, 5), can=H('up'))),
         ]
    return _step(A, t)


def _droop_pose(t, can):
    """arm lowers the can till it touches the dirt, head sinks, shoulders slump
    forward, small settle bounce; the hold breathes slowly."""
    def arm(by, drop=0):
        if can == 'empty':
            return dict(arm=_carry_arm(0, 5 - by + drop), can=('hand', 'up', 0, 0))
        if can is not None:
            return _hug(by)
        return dict(arm=[(0, 0), (0, 2), (1, 3), (1, 5 - by + drop)])
    A = [(0.0, dict(ant='up', **arm(0))),
         (0.1, dict(head='d1', ant='fwd', **arm(0))),
         (0.2, dict(by=1, torso='l1', head='d2', eye='down', ant='fwd3', **arm(1, 1))),
         (0.3, dict(by=2, torso='l1', head='d2', hy=1, eye='down', ant='fwd', **arm(2, 1))),
         (0.4, dict(by=1, torso='l1', head='d2', hy=1, eye='down', ant='fwd1', **arm(1, 1))),
         (0.5, dict(by=1, torso='l1', head='d2', hy=1, eye='down', ant='low', **arm(1, 1))),
         ]
    if t < 0.8:
        return _step(A, t)
    u = (t - 0.8) % 2.4                             # slow, heavy breathing
    by = 1 if u < 1.4 else 2
    hy = 1
    ant = 'low' if u < 1.4 else ('fwd1' if u < 1.6 else 'low')
    eye = 'half' if 1.9 < u < 2.1 else 'down'
    return dict(by=by, torso='l1', head='d2', hy=hy, eye=eye, ant=ant, **arm(by, 1))


def _startle_pose(t, can):
    def hold(by):
        if can == 'empty':
            return dict(arm=_carry_arm(0, 5 - by), can=('hand', 'up', 0, 0))
        if can is not None:
            return _hug(by)
        return dict(arm=[(0, 0), (0, 2), (1, 3), (1, 5)])
    A = [(0.0, dict(by=1, head='d1', ant='fwd', **hold(1))),
         (0.1, dict(by=-2, legs=('stand', 2), head='u1', hx=1, eye='wide', eb=1, ant='back2', **hold(-2))),
         (0.2, dict(by=-1, legs=('stand', 1), head='n', hx=1, eye='wide', eb=1, ant='back', **hold(-1))),
         (0.3, dict(by=1, head='n', hx=1, hy=1, eye='wide', eb=1, ant='fwd', fx=PUFF(6, 25, 1, 3), **hold(1))),
         (0.4, dict(by=0, head='n', hx=1, eye='wide', eb=1, ant='up', **hold(0))),
         ]
    if t < 0.5:
        return _step(A, t)
    u = t - 0.5
    ant = 'up' if int(u * 10) % 6 else 'fwd'
    eye = 'wide' if int((u % 1.6) * 10) != 9 else 'fwd'
    return dict(by=0, head='n', hx=1, eye=eye, eb=1, ant=ant, **hold(0))


# ----------------------------------------------------------------- kneel
# KNEEL POSE = the hold of 'kneel' (t >= 0.4) and the START POSE of CHAR-B's
# dig_plant / open_chest / place_core.  Rig values: legs KNEEL_LEGS, torso upright
# ('n') at bx=+1, by=+2 (hips 1 px forward, 2 px lower than standing), head 'd2'
# (looking down at the ground ahead), pupil low, antenna sagging forward, near hand
# resting on the front knee.  Anchors relative to 'feet' (the same ground point as
# when standing — the robot steps its front foot forward as it kneels):
#     eye (+6,-13)  core (+2,-7)  hand (+3,-3)   bbox x -7..+11, y -23..0
# Exact pixels (normal variant; K=steel0 outline, 1-5 paint1-5, r/R/o rust2-4,
# a/s/S/t/T steel1-5, g ink2 glass, h glint, P/Q/q pupil core6/5/3 at eye=1,
# ( ) [ ] porthole core6/5/5/3 at core=1, * bulb core5; x/X/z/Z/j/k = paint5/ash5/
# steel pixels that change in the damaged variant):
#   -23  .......K...........
#   -22  ......K*K..........
#   -21  .......S...........
#   -20  ......S............
#   -19  ....KKk............
#   -18  ...KxX5KKK.........
#   -17  ..Kz544444KK.......
#   -16  ..K544444444KKK....
#   -15  ..K44TS4433KtTSK...
#   -14  ..K44Ss433KthggSK..
#   -13  ..K33r3333KSgPQsK..
#   -12  ...K222332KSPQqaK..
#   -11  ....KKKr222KssaK...
#   -10  .....KsKKK22KKK....
#    -9  KKKKKKKKKKKKK......
#    -8  KSZK45544KK43K.....
#    -7  KSaK44t4K()K3K.....
#    -6  KjGK344SK[]K2K.....
#    -5  KsGK333s3KK32KKKK..
#    -4  KssK2r33SsR22KtTSK.      <- front (near) knee up, shin x +7..+10
#    -3  KKKK22r22RS21KSStK.
#    -2  .KK.KKKKKKKKKKSStKK
#    -1  KrRKKKK......KRooRK      <- back boot lying behind / front boot planted x +6..+11
#    +0  KKKKKKK......KKKKKK
#               ^ feet anchor column (x = 0)
KNEEL = dict(legs=('kneel',), bx=1, by=2, torso='n', head='d2', eye='down', ant='fwd',
             arm=[(0, 0), (1, 2), (4, 4)])
SCOOP_CAN_OFFSET = (-12, 0)         # can 'base' relative to the kneeling feet: stands at the robot's back hip
                                    # (near side, drawn in front of the body), spout pointing left, handle toward it


def _ground_can(state='empty'):
    return ('abs', 'up_l', FEET[0] + SCOOP_CAN_OFFSET[0], FEET[1] + SCOOP_CAN_OFFSET[1], 'base')


def PUFF(x, y, stage, w=2):
    """a tiny dust puff at ground contact (x, y = ground point), stage 1 or 2."""
    if stage == 1:
        return [(x - w, y - 1, 'F'), (x - w - 1, y, 'f'), (x + w + 5, y - 1, 'f'), (x + w + 6, y, 'F')]
    return [(x - w - 1, y - 2, 'f'), (x - w - 2, y - 1, 'E'), (x + w + 6, y - 2, 'E'), (x + w + 7, y - 1, 'f')]


def _kneel_pose(t, can):
    gc = dict(can=_ground_can()) if can is not None else {}
    hc = dict(arm=_carry_arm(0, 5), can=('hand', 'up', 0, 0)) if can is not None else {}
    swing = dict(arm=[(0, 0), (-1, 2), (-3, 4)], can=('abs', 'front', -1, 24, 'base')) if can is not None else {}
    A = [(0.0, dict(by=1, head='d1', ant='fwd', **hc)),
         (0.1, dict(by=1, bx=1, legs=('walk', 2, 1, -1, 0), head='d1', ant='back', **swing)),
         (0.2, dict(by=2, bx=2, legs=('kneel',), torso='l1', head='d1', ant='back', arm=[(0, 0), (-1, 2), (-2, 4)], **gc)),
         (0.3, dict(by=3, bx=2, legs=('kneel',), torso='l1', head='d2', hy=1, eye='down', ant='fwd',
                    arm=[(0, 0), (1, 2), (3, 4)], fx=PUFF(3, 25, 1), **gc)),
         (0.35, dict(by=3, bx=2, legs=('kneel',), torso='l1', head='d2', hy=1, eye='down', ant='fwd',
                     arm=[(0, 0), (1, 2), (3, 4)], fx=PUFF(3, 25, 2), **gc)),
         (0.4, dict(KNEEL, ant='low', **gc)),
         (0.5, dict(KNEEL, **gc)),
         ]
    return _step(A, t)


def _scoop_pose(t, can):
    SX, SY = FEET[0] + SCOOP_SPROUT_OFFSET[0], FEET[1] + SCOOP_SPROUT_OFFSET[1]
    gc = _ground_can()
    MX, MY = FEET[0] + SCOOP_CAN_OFFSET[0], FEET[1] - 4          # can mouth (model)
    reach = dict(legs=('kneel',), bx=3, by=2, torso='l3', head='d2', eye='down', ant='fwd')
    d1 = [(SX - 3, SY - 2, 'D'), (SX - 4, SY - 3, 'd'), (SX + 3, SY - 2, 'D')]
    d2 = [(SX - 5, SY - 4, 'd'), (SX - 6, SY - 3, 'D'), (SX + 4, SY - 4, 'd'), (SX - 2, SY - 1, 'D')]
    d3 = [(SX - 7, SY - 3, 'D'), (SX - 8, SY - 1, 'd'), (SX + 5, SY - 2, 'D')]
    hole = [(SX - 1, SY, 'd'), (SX, SY, 'n'), (SX + 1, SY, 'd'), (SX - 2, SY, 'D'), (SX + 2, SY, 'D')]
    sh = (6 + 3 + 2, 16 + 2)                                      # leaned shoulder (model)
    NH = lambda x, y, ex=3, ey=2: [(0, 0), (ex, ey), (x - sh[0], y - sh[1])]
    FH = lambda x, y: [(0, 0), (3, 1), (x - sh[0] - 3, y - sh[1])]

    def carry(cx, cy, bx, torso, head, eye, ant, by=2):
        """both hands under the clump at (cx, cy) (its base), body rocking back."""
        shx = 6 + bx + {'n': 0, 'l1': 1, 'l2': 1, 'l3': 2}[torso]
        shy = 16 + by
        return dict(legs=('kneel',), bx=bx, by=by, torso=torso, head=head, eye=eye, ant=ant,
                    arm=[(0, 0), ((cx - 1 - shx) // 2, (cy + 1 - shy) // 2 + 1), (cx - 1 - shx, cy + 1 - shy)],
                    farm=[(0, 0), ((cx + 2 - shx - 3) // 2, 1), (cx + 2 - shx - 3, cy + 1 - shy)],
                    can=gc, clump=(cx, cy), fx=hole)
    back = dict(KNEEL, bx=0, head='d1', eye='back')
    shb = (6, 18)
    BH = lambda x, y: [(0, 0), (-1, 1), (x - shb[0], y - shb[1])]
    A = [(0.0, dict(KNEEL, can=gc)),
         (0.1, dict(reach, bx=2, torso='l2', head='d1', arm=[(0, 0), (2, 2), (6, 3)], farm=[(0, 0), (3, 1), (6, 2)], can=gc)),
         (0.2, dict(reach, arm=NH(19, 24), farm=FH(23, 24), can=gc)),
         (0.3, dict(reach, hy=1, arm=NH(17, 23), farm=FH(23, 24), can=gc, fx=d1, ant='up')),
         (0.4, dict(reach, arm=NH(19, 24), farm=FH(24, 23), can=gc, fx=d2)),
         (0.5, dict(reach, hy=1, arm=NH(17, 23), farm=FH(23, 24), can=gc, fx=d1 + d3, ant='up')),
         (0.6, dict(reach, arm=NH(20, 24), farm=FH(22, 24), can=gc, fx=d2)),
         (0.7, dict(reach, head='d1', arm=NH(20, 24), farm=FH(22, 24), can=gc, fx=d3)),
         (0.75, dict(reach, head='d1', arm=NH(20, 24), farm=FH(22, 24), can=gc, clump=(SX, SY), fx=hole, ant='up')),
         # the carry: a smooth arc (15 fps) up past the chest and down into the can
         (0.817, carry(SX - 2, SY - 4, 3, 'l2', 'd1', 'fwd', 'back')),
         (0.883, carry(SX - 7, SY - 7, 2, 'l1', 'n', 'fwd', 'back2')),
         (0.95, carry(SX - 13, SY - 8, 1, 'l1', 'n', 'back', 'back')),
         (1.017, carry(MX + 3, SY - 7, 0, 'n', 'd1', 'back', 'fwd')),
         (1.067, carry(MX, MY - 2, 0, 'n', 'd2', 'back', 'fwd')),
         (1.1, dict(back, head='d2', arm=BH(MX + 1, MY - 1), can=gc, canstate='sprout', fx=hole, ant='fwd')),
         (1.2, dict(back, head='d2', arm=BH(MX + 1, MY - 2), can=gc, canstate='sprout', fx=hole)),
         (1.3, dict(back, head='d2', arm=BH(MX + 1, MY - 1), can=gc, canstate='sprout', fx=hole, ant='low')),
         (1.4, dict(back, head='d2', arm=BH(MX + 2, MY), can=gc, canstate='sprout', fx=hole, flash=True)),
         (1.5, dict(back, head='d2', arm=BH(MX + 2, MY), can=gc, canstate='sprout', fx=hole)),
         ]
    return _step(A, t)


def _rise_pose(t, can):
    MX, MY = FEET[0] + SCOOP_CAN_OFFSET[0], FEET[1] - 4
    gc = ('abs', 'up_l', FEET[0] + SCOOP_CAN_OFFSET[0], FEET[1], 'base')
    back = dict(KNEEL, bx=0, head='d2', eye='back')
    A = [(0.0, dict(back, arm=[(0, 0), (-1, 1), (MX + 2 - 6, MY - 18)], can=gc)),
         (0.1, dict(back, head='d1', arm=[(0, 0), (-1, 1), (MX + 3 - 6, MY - 1 - 18)],
                    can=('abs', 'up_l', MX + 1, FEET[1] - 1, 'base'))),
         (0.2, dict(legs=('kneel',), bx=1, by=2, head='d1', eye='fwd', ant='back', arm=[(0, 0), (1, 3), (3, 5)],
                    can=('abs', 'front', FEET[0] + 3, FEET[1] - 1, 'base'))),
         (0.3, dict(legs=('walk', 1, 0, -1, 0), by=1, head='d1', ant='back', **_hug(1, 0, 1))),
         (0.4, dict(by=-1, legs=('stand', 1), head='n', ant='back2', **_hug(-1))),
         (0.5, dict(by=0, head='d1', ant='fwd', **_hug(0))),
         (0.6, dict(by=1, head='d1', ant='fwd', **_hug(1))),
         (0.7, dict(by=0, head='d1', eye='down', ant='up', **_hug(0))),
         ]
    if t < 0.9:
        return _step(A, t)
    u = (t - 0.9) % 2.0
    k = int(u * 10)
    hx = 1 if 5 <= k < 12 else 0
    head = 'd2' if 6 <= k < 11 else 'd1'
    ant = 'fwd' if k in (5, 6) else ('back' if k in (12, 13) else 'up')
    return dict(by=0, hx=hx, head=head, eye='down', ant=ant, **_hug(0))


def _look_pose(t, can, up):
    def hold(by):
        if can == 'empty':
            return dict(arm=_carry_arm(0, 5 - by), can=('hand', 'up', 0, 0))
        if can is not None:
            return _hug(by)
        return {}
    if up:
        A = [(0.0, dict(head='n', ant='up', **hold(0))),
             (0.1, dict(head='u1', eye='up', ant='back', **hold(0))),
             (0.2, dict(head='u2', eye='up', torso='lk', ant='back2', **hold(0))),
             (0.3, dict(head='u2', eye='up', torso='lk', hy=-1, ant='back3', **hold(0))),
             ]
        if t < 0.4:
            return _step(A, t)
        k = int(((t - 0.4) % 1.6) * 10)
        return dict(head='u2', eye='up' if k != 9 else 'half', torso='lk', ant='back' if k < 8 else 'up', **hold(0))
    A = [(0.0, dict(head='n', ant='up', **hold(0))),
         (0.1, dict(head='d1', eye='down', ant='fwd', **hold(0))),
         (0.2, dict(head='d2', eye='down', ant='fwd2', **hold(0))),
         (0.3, dict(head='d2', eye='down', hy=1, ant='fwd', **hold(0))),
         ]
    if t < 0.4:
        return _step(A, t)
    k = int(((t - 0.4) % 2.0) * 10)
    return dict(head='d2', eye='down', hy=1 if k < 10 else 0, ant='fwd' if k < 12 else 'up', **hold(0))


# ----------------------------------------------------------------- storm
# BRACE: 12 fps, one heavy 3-px step every 0.75 s (1.5 s double cycle) = 4 px/s.
_BR_N = [(1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (-1, 0), (-1, 0), (-1, 0), (-2, 0),
         (-2, 0), (-2, 0), (-3, 0), (-3, 1), (-2, 1), (-1, 1), (1, 1), (2, 0), (1, 0)]
_BR_F = [(-2, 0), (-2, 0), (-3, 0), (-3, 1), (-2, 1), (-1, 1), (1, 1), (2, 0), (1, 0),
         (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (-1, 0), (-1, 0), (-1, 0), (-2, 0)]
_BR_ADV = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6]


def _brace_hug(by, k):
    """can pressed to the belly under the bowed head; sprout whipping in the wind."""
    return dict(arm=[(0, 0), (1, 3), (5, 5), (9, 5)], farm=[(0, 0), (4, 3), (8, 3)],
                fingers=[(3, 1, 's'), (3, 2, 'S')],
                can=('abs', 'up', 17, 20 + by, 'mouth'), swind=1.0)


def _brace_pose(t, can):
    k = int(math.floor(t * 12 + 1e-6)) % 18
    n, f = _BR_N[k], _BR_F[k]
    by = 1 if k in (7, 8, 16, 17) else 0            # weight lands
    hy = 1 if k in (8, 9, 17, 0) else 0             # head lags the landing
    ant = ['back3', 'back2', 'back', 'back2', 'back3', 'back2'][k % 6]
    eye = 'half' if k % 9 in (3, 4, 5) else 'fwd'   # squinting into the dust
    pose = dict(legs=('walk', n[0], n[1], f[0], f[1], 1), bx=1, by=by, torso='lb', head='d1', hx=1, hy=hy,
                eye=eye, ant=ant)
    if can is not None:
        pose.update(_brace_hug(by, k))
    else:
        pose.update(arm=[(0, 0), (1, 3), (4, 5), (8, 4)], farm=[(0, 0), (4, 3), (7, 3)])
    return pose


def _curl(tremble=0, by=3):
    """crouched, curled over the can (composed facing right, then mirrored)."""
    return dict(legs=('crouch',), bx=1 + tremble, by=by, torso='l3', head='d2', hx=1, hy=1, eye='down',
                ant='fwd2' if tremble else 'fwd', mirror=True,
                arm=[(0, 0), (1, 3), (4, 5), (8, 4)], farm=[(0, 0), (4, 3), (7, 4)],
                can=('abs', 'up', 15 + tremble, 20 + by, 'mouth'))


def _shield_pose(t, can):
    A = [(0.0, dict(_brace_pose(0.0, can), head='d2', hy=1, eye='closed', ant='back2')),   # flinch
         (0.1, dict(_brace_pose(0.0, can), by=2, head='d2', hy=1, eye='closed', ant='back', torso='l2')),
         (0.2, dict(front=True)),                                                           # turning
         (0.3, dict(_curl(0, 4), fx=PUFF(8, 25, 1, 4))),                                    # slam down
         (0.4, dict(_curl(0, 3), fx=PUFF(8, 25, 2, 4))),
         ]
    if t < 0.5:
        pz = _step(A, t)
    else:
        k = int(math.floor((t - 0.5) * 10 + 1e-6))
        pz = _curl(k % 2, 3)
        if k % 7 == 3:
            pz['eye'] = 'closed'
    if can is None and not pz.get('front'):
        pz = dict(pz, can=None)
    return pz


def _unshield_pose(t, can):
    A = [(0.0, dict(_curl(0, 3))),
         (0.1, dict(_curl(0, 2), head='d1')),
         (0.2, dict(front=True)),
         (0.3, dict(_brace_pose(0.0, can), by=1, hy=1, ant='back2')),
         (0.4, dict(_brace_pose(0.0, can))),
         ]
    pz = _step(A, t)
    if can is None and not pz.get('front'):
        pz = dict(pz, can=None)
    return pz


# WALK_TIRED: 10 fps, 1.0 s cycle, 6 px: quick step on the good (far) leg, then the
# hurt near leg DRAGS forward along the ground (its toe scrapes — intended slide).
_WT_ADV = [0, 1, 2, 3, 3, 3, 4, 4, 5, 5]
_WT_N = [(2, 0), (1, 0), (0, 0), (-1, 0), (-1, 0), (0, 0), (0, 0), (1, 0), (1, 0), (2, 0)]
_WT_F = [(-4, 0), (-2, 1), (-1, 1), (-1, 0), (-1, 0), (-1, 0), (-2, 0), (-2, 0), (-3, 0), (-3, 0)]


def _tired_pose(t, can):
    """limp: a painful hitch onto the hurt NEAR leg (hunch + dip + head drop),
    then the good leg carries while the hurt foot drags forward on its toe."""
    k = int(math.floor(t * 10 + 1e-6)) % 10
    n, f = _WT_N[k], _WT_F[k]
    by = [1, 2, 2, 2, 1, 1, 1, 1, 1, 1][k]
    torso = 'l2' if k in (1, 2, 3) else 'l1'
    head = ['d1', 'd1', 'd2', 'd2', 'd2', 'd1', 'd1', 'd1', 'd1', 'd1'][k]
    hy = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0][k]
    ant = ['low', 'low', 'fwd1', 'fwd1', 'low', 'low', 'fwd', 'low', 'low', 'fwd'][k]
    eye = 'half' if k in (2, 3) else 'down'           # wince
    toe = 9 + n[0] + 5
    dust = {5: [(toe, 24, 'D')], 6: [(toe - 1, 23, 'd')], 8: [(toe, 24, 'D')], 9: [(toe - 1, 23, 'd')]}.get(k)
    pose = dict(legs=('walk', n[0], n[1], f[0], f[1]), by=by, torso=torso, head=head, hy=hy, eye=eye,
                ant=ant, fx=dust)
    if can is not None:
        pose.update(_hug(by))
        pose['swind'] = 0.2
    else:
        pose['arm'] = [(0, 0), (1, 2), (1, 3), (2, 5 - (by - 1))]
    return pose


def _pose(anim, t, can):
    f = {'walk': _walk_pose, 'idle': _idle_pose, 'water': _water_pose, 'droop': _droop_pose,
         'startle': _startle_pose, 'kneel': _kneel_pose, 'scoop': _scoop_pose, 'rise_hold': _rise_pose,
         'look_up': lambda t, c: _look_pose(t, c, True), 'look_down': lambda t, c: _look_pose(t, c, False),
         'walk_tired': _tired_pose, 'brace': _brace_pose, 'shield': _shield_pose,
         'unshield': _unshield_pose}.get(anim)
    if f is None:
        return {}
    return f(t, can)


ANIMS = {
    # --- CHAR-A (chars/robot.py) ---------------------------------------
    'idle':        (True, 2.0, ('medium', 'wide')),
    'walk':        (True, 0.8, ('medium', 'wide')),   # 8 frames @10fps, 10 px/s (medium), 4 px/s (wide)
    'walk_tired':  (True, 1.0, ('medium', 'wide')),   # limping, stooped, 6 px/s
    'brace':       (True, 0.75, ('medium', 'wide')),  # storm: leaning into wind (wind from RIGHT), can hugged, 4 px/s
    'shield':      (False, 0.5, ('medium',)),         # storm: turns back to wind & curls around the can -> hold
    'water':       (False, 2.2, ('medium',)),         # raise can, tilt over plant, hold, lower
    'droop':       (False, 0.8, ('medium',)),         # sad: head sinks, shoulders slump -> hold
    'look_down':   (False, 0.4, ('medium',)),         # -> hold
    'look_up':     (False, 0.4, ('medium',)),         # -> hold
    'startle':     (False, 0.5, ('medium',)),         # small jolt + head snaps forward, noticing -> hold
    'kneel':       (False, 0.6, ('medium', 'wide')),  # down on one knee -> hold
    'scoop':       (False, 1.5, ('medium',)),         # kneeling: dig around sprout, lift soil clump into can
    'rise_hold':   (False, 0.9, ('medium',)),         # stand up hugging the can to its chest -> hold
    'unshield':    (False, 0.5, ('medium',)),         # EXTRA: from the shield hold, turn back into the wind -> brace pose
    # --- CHAR-B (chars/robot_acts.py) -----------------------------------
    'sit':         (False, 0.8, ('medium', 'wide')),
    'sit_idle':    (True, 3.0, ('medium', 'wide')),
    'sit_look_can': (False, 0.5, ('medium',)),
    'sit_look_chest': (False, 0.6, ('medium',)),
    'dig_plant':   (False, 2.0, ('medium',)),
    'open_chest':  (False, 1.2, ('medium',)),
    'place_core':  (False, 1.0, ('medium',)),
    'power_down':  (False, 1.6, ('medium',)),
    'dead_sit':    (True, 1.0, ('medium', 'wide')),
}

SCOOP_SPROUT_OFFSET = (13, 0)
SCOOP_LIFT_T = 0.75
SCOOP_INTO_CAN_T = 1.1
WATER_TILT_T = 0.6
WALK_SPEED = 10.0          # px/s, medium
WALK_TIRED_SPEED = 6.0     # px/s, medium
BRACE_SPEED = 4.0          # px/s, medium
WIDE_WALK_SPEED = 4.0      # px/s, wide


def duration(anim):
    return ANIMS[anim][1]


# ---------------------------------------------------------------- travel
_WIDE_ADV = [0, 1, 1, 2, 2, 2, 3, 3, 4, 4]


def travel(anim, t, scale='medium'):
    """Ground distance (px) the feet have covered t s into a locomotion anim;
    integer, stepped with the animation frames (zero foot slide)."""
    t = max(t, 0.0)
    if scale == 'wide':
        if anim == 'walk':
            k = int(math.floor(t * 10 + 1e-6))
            return 4 * (k // 10) + _WIDE_ADV[k % 10]
        if anim in ('walk_tired', 'brace'):
            fps = 6.0 if anim == 'walk_tired' else 4.0
            k = int(math.floor(t * fps + 1e-6))
            return 4 * (k // 10) + _WIDE_ADV[k % 10]
        return 0
    if anim == 'walk':
        return int(math.floor(t * 10 + 1e-6))
    if anim == 'walk_tired':
        k = int(math.floor(t * 10 + 1e-6))
        return 6 * (k // 10) + _WT_ADV[k % 10]
    if anim == 'brace':
        k = int(math.floor(t * 12 + 1e-6))
        return 6 * (k // 18) + _BR_ADV[k % 18]
    return 0


def _q(v, n=8):
    return int(round(min(max(v, 0.0), 1.0) * n))


# ======================================================================
#  WIDE scale (~11 px): hand-pixelled mini version of the same robot
# ======================================================================
WCW, WCH = 20, 16
WOX, WOY = 6, 3
WFEET = (4, 10)
# 11 px tall; head:body ~ 40:60 like the model. 'Q' = amber pupil, 'h' = glint,
# 't' = antenna stem joint, '(' = core, '*' = bulb.
_WIDE_UPPER = _part("""
..*......
.KtKKK...
Kx544KKK.
K443KhQK.
K3322KKK.
.KKKKK...
KZK4(K...
KaK32K...
.KKKKK...
""", top=(0, 0))


def _wide_frame(anim, t, can, variant):
    g = np.zeros((WCH, WCW), np.uint16)

    def px(x, y, ch):
        X, Y = WOX + x, WOY + y
        if 0 <= X < WCW and 0 <= Y < WCH:
            g[Y, X] = ord(ch) if isinstance(ch, str) else ch
    hb, lean, low, bob = 0, 0, 0, 0
    bulb_dx = 0
    eye = 'open'
    legs = None
    cdx = 0
    if anim in ('walk', 'walk_tired', 'brace'):
        fps = {'walk': 10.0, 'walk_tired': 6.0, 'brace': 4.0}[anim]
        k = int(math.floor(t * fps + 1e-6)) % 10
        nx = [1, 0, 0, -1, -1, -1, 0, 0, 1, 1][k]
        nl = [0, 0, 0, 0, 0, 1, 1, 1, 1, 0][k]
        fx = [-1, 0, 0, 1, 1, 1, 0, 0, -1, -1][k]
        fl = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0][k]
        bob = 1 if k in (0, 5) else 0                   # weight lands
        hb = 1 if k in (1, 6) else 0                    # head follows a frame later
        bulb_dx = -1 if k in (0, 1, 5, 6) else 0         # bulb lags back on the bounce
        cdx = -1 if k in (1, 2, 6, 7) else 0             # can swings back after the step
        legs = ((3 + fx, fl, 's', 'r'), (5 + nx, nl, 't', 'o'))
        if anim == 'brace':
            lean, bulb_dx = 1, -1
        if anim == 'walk_tired':
            low, hb = 1, 1 if k in (1, 2, 3) else 0
    elif anim == 'kneel':
        low = 1 if t >= 0.2 else 0
        legs = ((1, 0, 's', 'r'), (6, 0, 't', 'o')) if t >= 0.2 else None
        bob = 1 if 0.2 <= t < 0.3 else 0
    else:   # idle and anything else
        n = int(math.floor(t / 2.0))
        k = int(math.floor((t - 2.0 * n) * 10 + 1e-6))
        hb = 1 if 9 <= k <= 16 else 0
        bulb_dx = 1 if k == 9 else (-1 if k == 17 else 0)
        if n % 2 == 0 and k in (4, 5):
            eye = 'closed'
    if legs is None:
        legs = ((3, 0, 's', 'r'), (5, 0, 't', 'o'))
    for (x, lift, sh, bt) in legs:
        if lift:
            px(x, 9, bt); px(x - 1, 9, 'K'); px(x + 1, 9, 'K')
        else:
            if low == 0 and bob == 0:
                px(x, 9, sh); px(x - 1, 9, 'K'); px(x + 1, 9, 'K')
            px(x, 10, bt); px(x - 1, 10, 'K'); px(x + 1, 10, 'K')
    up = _WIDE_UPPER.g
    body_dy = low + bob
    head_dy = body_dy + hb
    for y in range(up.shape[0]):
        for x in range(up.shape[1]):
            c = up[y, x]
            if not c:
                continue
            if y == 0:                       # bulb: sways
                continue
            if y <= 5:
                px(x + lean, y + head_dy - (1 if hb and y == 5 and bob == 0 and low == 0 else 0), int(c))
            else:
                px(x, y + body_dy, int(c))
    if variant == 'damaged':
        px(2 + lean, 1 + head_dy, 'r')
    else:
        px(2 + lean + bulb_dx, 0 + head_dy, '*')
    if eye == 'closed':
        px(5 + lean, 3 + head_dy, 's'); px(6 + lean, 3 + head_dy, 'S')
    if can is not None:
        cx, cy = 6 + cdx, 7 + body_dy
        px(cx, cy, 'R'); px(cx + 1, cy, 'o'); px(cx, cy + 1, 'r'); px(cx + 1, cy + 1, 'R')
        px(cx + 2, cy, 'o')
        if can != 'empty':
            px(cx, cy - 1, 'L')
    anc = {'feet': WFEET, 'eye': (6 + lean, 3 + head_dy), 'core': (4, 6 + body_dy), 'hand': (6 + cdx, 7 + body_dy)}
    anc['can_mouth'] = (6 + cdx, 6 + body_dy)
    anc['spout'] = (8 + cdx, 7 + body_dy)
    anc['core_held'] = anc['core']
    return g, {k: (x + WOX, y + WOY) for k, (x, y) in anc.items()}


# ======================================================================
#  public entry point
# ======================================================================
_CACHE = {}


def frame(anim, t, can=None, variant='normal', scale='medium', core=1.0, eye=1.0, face_left=False):
    if anim not in ANIMS:
        raise KeyError(anim)
    if anim in _B_HANDLES():
        return _frame(anim, t, can, variant, scale, core, eye, face_left)
    # poses are step functions of time (10/12 fps + a few exact cue times): cache
    key = (anim, int(math.floor(max(float(t), 0.0) * 120 + 1e-6)), can, variant, scale,
           _q(core), _q(eye), bool(face_left))
    hit = _CACHE.get(key)
    if hit is None:
        spr = _frame(anim, t, can, variant, scale, core, eye, face_left)
        if len(_CACHE) > 20000:
            _CACHE.clear()
        _CACHE[key] = (spr.img, dict(spr.anchors))
        return spr
    return Sprite(hit[0].copy(), dict(hit[1]))


def _B_HANDLES():
    try:
        from chars import robot_acts
        if hasattr(robot_acts, 'frame'):
            return getattr(robot_acts, 'HANDLES', ())
    except ImportError:
        pass
    return ()


def _frame(anim, t, can, variant, scale, core, eye, face_left):
    # CHAR-B animations are implemented in chars/robot_acts.py
    try:
        from chars import robot_acts
        if hasattr(robot_acts, 'frame') and anim in getattr(robot_acts, 'HANDLES', ()):
            spr = robot_acts.frame(anim, t, can=can, variant=variant, scale=scale, core=core, eye=eye)
            return flip(spr) if face_left else spr
    except ImportError:
        pass
    t = max(float(t), 0.0)
    if scale == 'wide':
        g, anc = _wide_frame(anim, t, can, variant)
        e = _q(eye)
        img = _lut(variant, _q(core), e)[g]
        spr = Sprite(img, anc)
        return flip(spr) if face_left else spr
    pose = dict(_pose(anim, t, can))
    pose['_t'] = t
    G, p = _compose(pose, can, variant)
    e = _q(eye)
    if p.get('eb') and e > 0:
        e = min(9, e + 1)
    img = _lut(variant, _q(core), e)[G.g]
    anc = {k: (x + OX, y + OY) for k, (x, y) in G.anc.items()}
    spr = Sprite(img, anc)
    return flip(spr) if face_left else spr


def kneel_parts():
    """For CHAR-B: the KNEEL pose as a dict usable with this rig."""
    return dict(KNEEL)
