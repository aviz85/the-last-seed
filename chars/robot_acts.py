"""
chars/robot_acts.py — the Gardener's GREENHOUSE acting (owned by CHAR-B).

Called through chars/robot.py (which handles face_left):
    frame(anim, t, can=None, variant='damaged', scale='medium', core=1.0, eye=1.0) -> Sprite
    core_orb(t, brightness=1.0) -> Sprite   the removed heart, anchor 'center', ~1.2 Hz heartbeat
    dig_can(state='empty') -> Sprite         the can exactly as dig_plant draws it (see below)

Every Sprite carries anchors feet, eye, core, hand, can_mouth, spout, core_held (+ 'back'
on the sitting poses: the backpack's outermost column, x = SIT_BACK_X).  All offsets are
relative to the FEET anchor (x right, y down; y = 0 is the floor row; nothing is ever
drawn below it).  Medium = 26 px robot; wide (~11 px) for sit / sit_idle / dead_sit.

BUILD. A pixel-exact rig of the canonical model (chars/model.py IDLE_DAMAGED; the parts
re-assemble the model with 0 px difference): hand-drawn rotated variants (heads pitched
~15/30/45 deg, torso reclined / leaning 14 / 28 deg, bent-leg sets), CHAR-A's own parts
where the two animators meet (their walk_tired stance: 'l1' torso + 'd1' head; their
KNEEL_LEGS), 1-px steel arms textured like the model (t,S,s..) that get an outline only
against the background or across the head.  Eye / core colours use CHAR-A's tables, so
the same `eye` / `core` values glow identically in both animators' frames.  Poses are
keyed frame by frame at 12 fps with holds (the *_FR timelines); holds keep breathing.

ANIMS (medium, t in s from the anim start; one-shots hold their last pose)
  sit 0.8          f0 == CHAR-A walk_tired stance (hugging the can) -> tired sag -> bends,
                   sets the can down with both hands (touch 1/3 s, released SIT_CAN_DOWN_T)
                   -> pushes back -> drops onto the floor, back against the rubble on its
                   LEFT, legs forward (IMPACT SIT_IMPACT_T=0.583: dust) -> head rebounds,
                   arm flops late -> settle 0.8 (== sit_idle / sit_look_can frame 0).
  sit_idle 3.0     loop: exhale sink 0.75-1.33, heavier sink + nod 2.08-2.67 (lens dims 2.42).
  sit_look_can 0.5 perks up 0.08 -> turns to the can with overshoot 0.17 -> settle 0.33 ->
                   iris refocus + curious lift 0.42 -> 0.5 hold (breathes every 1.25 s).
  sit_look_chest   (starts from look_can's end) breaks off 0.08 -> eyes drop 0.25 -> bows
     0.6           over its glowing porthole 0.33 -> hand rises 0.42-0.58, stops halfway;
                   hold: hesitates 0.75, then rises 1 px closer (1.0): decided; breathes.
  dig_plant 2.0    leans in, digs with scrapes at DIG_SCRAPE_TS (soil flecks) -> turns back
                   to the can BEHIND it, lifts the wilted sprout out (DIG_TAKE_T) -> carries
                   it past its chest and up over the rim -> sets it in (PLANT_T) -> firms the
                   soil -> pats DIG_PAT_TS -> sits up, looking at it (2.0).
  open_chest 1.2   bows to its chest, fingers on the latch 0.17 -> click 0.25 -> the door
                   swings open 0.33-0.5 (hinged on its left edge) -> the bare core blazes ->
                   reaches in 0.58 -> lifts the heart out (CORE_OUT_T 0.75) -> both hands
                   cup it in front of the chest 0.92 -> hold ('core_held' = orb centre).
  place_core 1.0   leans in, offers the heart up at eye level 0.17 -> presses it into the
                   soil (touch 0.33, gone at CORE_PLACE_T) -> both hands linger flat 0.4-0.58
                   -> far hand lifts, the NEAR hand slides down to rest on the rim just below
                   the heart and stays there -> upright end pose at PLACE_END_T (0.83).
  power_down 1.6   (== place_core end) sags back onto its heels AWAY from the planter while
                   the hand stays on the rim; eye: flicker 0.25, flicker 0.5, fades 0.58-0.83,
                   dark at POWER_EYE_OFF_T with a 1-px head drop ('click'); the arm's weight
                   settles at POWER_STILL_T (1.0); utterly still after.
  dead_sit         == power_down's last frame: on its heels, head bowed toward the buried
                   heart, eye dark, porthole open & empty (door edge-on at its left), one
                   hand resting on the rim right below the heart, fingers toward the sprout.

HAND-OFF CONTRACT (what the SHOT draws itself)
  sit          t < SIT_CAN_DOWN_T: the can (state `can`) is in the sprite (hugged).
               t >= SIT_CAN_DOWN_T: not in the sprite -> shot draws props.can(state) with
               its 'base' at feet + CAN_SET_OFFSET.  Start 'sit' at (cue - SIT_IMPACT_T) to
               land the clank on the sit_down cue.
  dig_plant    the can is INSIDE the sprite, BEHIND the robot: dig_can(state) with its
               'base' at feet + DIG_CAN_OFFSET ('sprout_wilt' until DIG_TAKE_T, then
               'empty'; can=None is treated as 'sprout_wilt').  After dig_plant the shot
               draws dig_can('empty') there (NOT props.can(), whose spout points right).
               Sprout in the hand DIG_TAKE_T..PLANT_T; from PLANT_T the shot draws
               props.sprout('planted') at feet + PLANT_OFFSET, BEHIND the robot sprite
               (the robot hunches over it until ~1.8 and then reveals it).
  open_chest   orb in the sprite from CORE_OUT_T (always blazing: `core` only drives the
               porthole while the heart is inside).
  place_core   orb in the sprite until CORE_PLACE_T, then gone: the shot draws the buried
               glow at feet + CORE_PLACE_OFFSET (core_orb() is available); the robot's hand
               rests at feet + RIM_HAND from PLACE_END_T on (beside the glow).
  power_down   drives its own eye (x `eye`); dead_sit's eye is always dark.
"""
import math
import numpy as np

from engine import palette as P
from engine.sprite import Sprite, from_ascii
from chars import model

HANDLES = ('sit', 'sit_idle', 'sit_look_can', 'sit_look_chest', 'dig_plant', 'open_chest',
           'place_core', 'power_down', 'dead_sit')

# ---- contract constants (agreed defaults, unchanged) -------------------------
SIT_CAN_DOWN_T = 0.45
CAN_SET_OFFSET = (13, 0)
PLANT_T = 1.2
PLANT_OFFSET = (15, -11)
CORE_PLACE_T = 0.4
CORE_PLACE_OFFSET = (11, -11)
POWER_EYE_OFF_T = 0.9
# ---- extra timing / placement info for the shot + sound (read-only) ----------
SIT_IMPACT_T = 7 / 12            # butt hits the floor (clank + dust puff)
SIT_BACK_X = -10                 # sitting: backpack's outer pixel column (rubble goes left of it)
DIG_CAN_OFFSET = (-8, 0)         # dig_plant: can 'base' BEHIND the kneeling robot (props.can_pose('up_l'))
DIG_TAKE_T = 11 / 12             # dig_plant: sprout lifted out of the can (can -> 'empty')
DIG_SCRAPE_TS = (0.5, 0.75)      # dig_plant: soil scrape contacts
DIG_PAT_TS = (1.6, 1.75)         # dig_plant: pats
CORE_OUT_T = 0.75                # open_chest: orb leaves the porthole
POWER_STILL_T = 1.0              # power_down: final settle done (utterly still after)
PLACE_END_T = 10 / 12            # place_core: end pose reached (== power_down's first frame)
HEARTBEAT_HZ = 1.2

FPS_CHAR = 12.0

# ----------------------------------------------------------------------------
# legend: model legend + extra ramp steps + SENTINELS (placeholder colours
# replaced per frame by the eye / core brightness).
LEG = dict(model.LEGEND)
LEG.update({
    '6': 'paint6', '0': 'paint0', 'u': 'steel6', 'O': 'rust5', 'q': 'rust1',
    'k': 'ink1', 'z': 'ink0', 'b': 'ash3', 'n': 'ash4',
})
_SENT = {
    'E': (251, 3, 1), 'e': (251, 3, 2), 'f': (251, 3, 3), 'H': (251, 3, 4),   # eye: hi, mid, lo, glint
    'C': (251, 3, 5), 'c': (251, 3, 6), 'B': (251, 3, 8), 'd': (251, 3, 7),    # porthole TL, TR, BL, BR
}
LEG.update(_SENT)
_SENT_KEYS = {k: (v[0] << 16) | (v[1] << 8) | v[2] for k, v in _SENT.items()}


def _grid(s):
    lines = [l for l in s.strip('\n').split('\n')]
    w = max(len(l) for l in lines)
    return [l.ljust(w, '.') for l in lines]


def _art(s, pivot, **anchors):
    """ASCII part -> dict(img rgba, pv pivot, anc anchors, txt rows)."""
    spr = from_ascii(s, LEG)
    return {'img': spr.img, 'pv': pivot, 'anc': anchors, 'txt': _grid(s)}


_GLASS_W = ('Hgg', 'ggg', 'gEe')          # CHAR-A's 'down' gaze (mirrored live below)


def _mirror_chara():
    """Mirror the pieces CHAR-A and CHAR-B share at the walk_tired -> sit hand-over from
    chars/robot.py at import time (their 'down' glass, their 'd1' head art, their eye/core
    colour tables), so later tweaks on their side can't re-introduce a pop. Falls back to
    the static copies in this file if anything is missing."""
    global _GLASS_W
    out = {}
    try:
        from chars import robot as A
    except Exception:
        return out
    tr = {'h': 'H', 'P': 'E', 'Q': 'e', 'q': 'f', 'x': 's', 'X': 'S', 'z': 'a', 'Z': 'a', 'k': 'S',
          'n': 'k', 'W': 'u', 'j': 'a'}
    try:
        g = A.GLASS['down']
        if len(g) == 3 and all(len(r) == 3 for r in g):
            _GLASS_W = tuple(''.join(tr.get(ch, ch) for ch in r) for r in g)
    except Exception:
        pass
    try:
        prt = A.HEAD['d1']
        rows = [''.join(tr.get(chr(c), chr(c)) if c else '.' for c in row) for row in prt.g]
        ax, ay = prt.anc['ant']
        w = max(len(r) for r in rows)
        stub = ['.' * w]
        if ay == -1 and 0 <= ax < w:
            stub = ['.' * ax + 'r' + '.' * (w - ax - 1)]
        if set(''.join(rows)) <= set(LEG) | {'.'}:
            pvx, pvy = prt.anc['pivot']
            ex, ey = prt.anc['eye']
            gx, gy = prt.anc['glass']
            out['a_d1'] = ('\n'.join(stub + rows), (pvx, pvy + 2), dict(eye=(ex, ey + 1), glass=(gx, gy + 1),
                                                                    stub=(ax, 0)))
    except Exception:
        pass
    try:
        if len(A._EYE_TAB) == 10 and len(A._CORE_TAB) == 9:
            out['eye_tab'] = [tuple(r) for r in A._EYE_TAB]
            out['core_tab'] = [tuple(r) for r in A._CORE_TAB]
    except Exception:
        pass
    return out


_CHARA = _mirror_chara()


def _with_pupil(part, mode):
    """Copy of a head part with the pupil drawn in the 3x3 glass for a gaze mode.
    R = forward/down-right (model), L = down-left/back (own chest), S = iris contracted."""
    gx, gy = part['anc']['glass']
    rows = [list(r) for r in part['txt']]
    pat = {'R': ('Hgg', 'gEe', 'gef'), 'L': ('Hgg', 'Eeg', 'efg'), 'S': ('Hgg', 'gEg', 'ggg'),
           'D': ('Hgg', 'ggg', 'gEe'), 'W': _GLASS_W}[mode]
    for j in range(3):
        for i in range(3):
            rows[gy + j][gx + i] = pat[j][i]
    s = '\n'.join(''.join(r) for r in rows)
    return _art(s, part['pv'], **part['anc'])


# ============================================================================
# PARTS  (pose space: feet = (0,0); model feet = (8,25))
# ----------------------------------------------------------------------------
# HEADS (damaged).  pivot = neck-top pixel under the head.  'glass' = glint pixel
# (top-left of the 3x3 lens glass); 'stub' = antenna stub pixel.
_HEADS = {
    'lvl': _art("""
....r..........
..KKSKKKKK.....
.KsS544444KKK..
Ka5444443KtTSK.
K5444443KtHggSK
K44TS433KSgEesK
K44Ss333KSgefaK
K33r33322KssaK.
.K222r2222KKK..
..KKKKKKKK.....
""", (5, 10), eye=(11, 5), glass=(10, 4), stub=(4, 0)),
    # pitched down ~15 deg
    'dn1': _art("""
....r...........
..KKSKKK........
.KsS5444KK......
Ka544444443K....
K544444443KKKK..
K444TS4333KtTSK.
K443Ss333KtHggSK
K333r3322KSgEesK
.K2222r22KSgefaK
..KKKKK222KssaK.
.......KKKKKKK..
""", (5, 10), eye=(12, 7), glass=(11, 6), stub=(4, 0)),
    # pitched down ~30 deg
    'dn2': _art("""
.......r.......
....KKS........
..KsS55KK......
.K5444444KK....
K5444444443K...
K44TS433333K...
K44Ss3333KKK...
K33r3332KtTSK..
.K222r2KtHggSK.
.KK2221KSgEesK.
...K221KSgefaK.
....KKKKKssaK..
.........KKK...
""", (3, 10), eye=(10, 9), glass=(9, 8), stub=(7, 0)),
    # CHAR-A's 'd1' head (chars/robot.py HEAD['d1'], damaged colours), used where CHAR-A's
    # walk_tired hands over to 'sit' so the first frame matches theirs pixel for pixel.
    'a_d1': _art("""
....r..........
..KKS..........
.KsS5KKKKK.....
Ka54444444KKK..
K54444443KtTSK.
K44TS443KtHggSK
K44Ss433KSgEesK
K33r3333KSgefaK
.K2223322KssaK.
..KKKr2222KKK..
.....KKKKK.....
""", (5, 10), eye=(11, 6), glass=(10, 5), stub=(4, 0)),
    # pitched down ~45 deg (bowed / drooping)
    'dn3': _art("""
....KKKK.......
...K55sSK.r....
..K5444444S....
.K54444444K....
K544TS44433K...
K44rSs433333K..
K3333333332K...
.K23r3332KKK...
..K22222KtTSK..
...K221KtHggSK.
....K21KSgEesK.
.....K1KSgefaK.
......KKKssaK..
.........KKK...
""", (3, 9), eye=(10, 10), glass=(9, 9), stub=(10, 1)),
}
if 'a_d1' in _CHARA:
    _art_s, _pv, _anc_d = _CHARA['a_d1']
    _HEADS['a_d1'] = _art(_art_s, _pv, **_anc_d)
HEADS = {}
for _k, _p in _HEADS.items():
    for _m in ('R', 'L', 'S', 'D', 'W'):
        HEADS[(_k, _m)] = _with_pupil(_p, _m)

NECK = _art("KsSsK", (2, 0))

# BODIES (torso + backpack, no near arm). pivot = neck attach (pixel above the top
# outline); sh_n / sh_f = near / far shoulder; port = porthole's top-left core pixel;
# hip = pixel under the bottom centre where the legs attach.
BODIES = {
    'up': _art("""
KKKKKKKKKKKKK.
KSaK45544KK43K
KSaK4444KCcK3K
KaGK3444KBdK2K
KsGK33333KK32K
KssK2r3333R22K
KKKK22r22Rr21K
....KKKKKKKKK.
""", (7, -1), sh_n=(6, 2), sh_f=(9, 2), port=(9, 2), hip=(8, 8)),
    # reclined ~8 deg (top rows shifted back): leaning against something behind
    'rec': _art("""
KKKKKKKKKKKKK..
KSaK45544KK43K.
KSaK4444KCcK3K.
KaGK3444KBdK2K.
KsGK33333KK32K.
.KssK2r3333R22K
.KKKK22r22Rr21K
.....KKKKKKKKK.
""", (7, -1), sh_n=(6, 2), sh_f=(9, 2), port=(9, 2), hip=(9, 8)),
    # CHAR-A's 'l1' stoop: top rows sheared forward 1px (their walk_tired torso)
    'l1': _art("""
.KKKKKKKKKKKKK.
.KSaK45544KK43K
.KSaK4444KCcK3K
.KaGK3444KBdK2K
.KsGK33333KK32K
KssK2r3333R22K.
KKKK22r22Rr21K.
....KKKKKKKKK..
""", (8, -1), sh_n=(7, 2), sh_f=(10, 2), port=(10, 2), hip=(8, 8)),
    # leaning forward ~14 deg (rotated about the hip)
    'f1': _art("""
..KKKK..........
..KSaKKKKK......
.KSaK45544KKKK..
.KSaK444444443KK
.KaGK34444KK443K
.KsGK3344KCcK33K
KssK23333KBdK32K
KKKK2r3333KK32K.
....K22r333R22K.
.....KKKK2Rr21K.
.........KKKKK..
""", (9, 0), sh_n=(7, 4), sh_f=(10, 4), port=(10, 5), hip=(8, 10)),
    # leaning forward ~28 deg: the backpack rides up like a shell
    'f2': _art("""
...KKK...........
...KSaKK.........
..KSaaK5KK.......
..KaGGK455KK.....
.KsGGK445444KK...
.KssGK3344KK44KK.
.KssK3334KCcK443K
..KKK2r33KBdK43K.
....K2r333KK332K.
....KK2333R221K..
......KK2Rr221K..
........KK221K...
..........KKK....
""", (10, 2), sh_n=(8, 5), sh_f=(11, 5), port=(10, 6), hip=(7, 11)),
}

# LEGS. pivot = the feet anchor (0,0) of the pose.  'back' is drawn behind the
# torso; some sets also have a 'front' layer drawn over it.
LEGS = {
    'stand': {'back': _art("""
.KasK.KStK..
.KasK.KStK..
KrRrK.KRooRK
KKKKK.KKKKKK
""", (5, 3))},
    # CHAR-A's walk_tired stance (body 1px lower): one row of shin above the boots
    'stand1': {'back': _art("""
.KasK.KStK..
KrRrK.KRooRK
KKKKK.KKKKKK
""", (5, 2))},
    # knees bent, body 1px lower, boots where they stood
    'crouch': {'back': _art("""
.KasK..KStK..
KrRrK.KSRooRK
KKKKK.KKKKKKK
""", (5, 2))},
    # lunge (weight forward, back leg extended) for reaching far ahead
    'lunge': {'back': _art("""
........KStK..
.KKKK..KSStK..
KrRasK.KRooRK.
KKKKKK.KKKKKK.
""", (6, 3))},
    # legs stretched forward, toes up (body bottom on the floor); far boot peeks behind
    'sit': {'back': _art("""
........KK
........rKK
.....KKKRRK
.....KKKKoK
KKKKKKtttoK
KKKKKKSSSRK
KKKKKKKKKKK
""", (2, 6))},
    # legs half out (landing frame of the sit)
    'sitland': {'back': _art("""
.......KK..
......KtTK.
KKKKKKSStK.
KKKKKKKSRoK
KKKKKKKKKKK
""", (2, 4))},
    # one-knee kneel = CHAR-A's KNEEL_LEGS (chars/robot.py) translated +2 px: front knee up,
    # boot planted ahead; back knee down under the hips, its boot lying behind.
    'kneel': {'back': _art("""
...............KKK..
..............KtTSK.
..............KSStK.
..KK..........KSStKK
.KrRKKKK......KRooRK
.KKKKKKK......KKKKKK
""", (6, 5))},
    # front knee coming down (kneel -> heels transition)
    'kneel2': {'back': _art("""
..............KKK...
..KK.........KtTSKK.
.KrRKKKK.....KSSRoRK
.KKKKKKK.....KKKKKKK
""", (6, 3))},
    # sitting back on the heels (body bottom at y=-1): knees down in front, boots tucked behind
    'heels': {'back': _art("""
.KK..........KKKK.
KrRKKK......KtTSSK
KKKKKKKKKKKKKKKKKK
""", (8, 2))},
}


# body hip offsets: pivot = hip - (hip - pv)
def _body_at(name, hx, hy):
    b = BODIES[name]
    return (name, hx - (b['anc']['hip'][0] - b['pv'][0]), hy - (b['anc']['hip'][1] - b['pv'][1]))


# ============================================================================
# compositing buffer
BW, BH = 80, 56
RX, RY = 34, 44          # feet anchor inside the work buffer


class _Buf:
    __slots__ = ('img', 'head')

    def __init__(self):
        self.img = np.zeros((BH, BW, 4), np.uint8)
        self.head = np.zeros((BH, BW), bool)      # pixels covered by the head (arm outline goes over them)

    def put_head(self, part, x, y):
        img = part['img']
        px, py = part['pv']
        X = RX + int(x) - px
        Y = RY + int(y) - py
        h, w = img.shape[:2]
        a = img[..., 3] > 0
        self.img[Y:Y + h, X:X + w][a] = img[a]
        self.head[Y:Y + h, X:X + w] |= a

    def on_head(self, x, y):
        X, Y = RX + int(x), RY + int(y)
        return 0 <= X < BW and 0 <= Y < BH and bool(self.head[Y, X])

    def put(self, part, x, y, only_empty=False):
        img = part['img']
        px, py = part['pv']
        X = RX + int(x) - px
        Y = RY + int(y) - py
        h, w = img.shape[:2]
        a = img[..., 3] > 0
        sub = self.img[Y:Y + h, X:X + w]
        if only_empty:
            a = a & (sub[..., 3] == 0)
        sub[a] = img[a]

    def put_sprite(self, spr, x, y, anchor):
        ax, ay = spr.anchors[anchor]
        X = RX + int(x) - ax
        Y = RY + int(y) - ay
        h, w = spr.img.shape[:2]
        x0, y0 = max(X, 0), max(Y, 0)
        x1, y1 = min(X + w, BW), min(Y + h, BH)
        if x0 >= x1 or y0 >= y1:
            return
        s = spr.img[y0 - Y:y1 - Y, x0 - X:x1 - X]
        a = s[..., 3] > 0
        self.img[y0:y1, x0:x1][a] = s[a]

    def pset(self, x, y, color, only_empty=False):
        X, Y = RX + int(x), RY + int(y)
        if 0 <= X < BW and 0 <= Y < BH:
            if only_empty and self.img[Y, X, 3]:
                return
            self.img[Y, X, :3] = P.C(color)
            self.img[Y, X, 3] = 255

    def empty(self, x, y):
        X, Y = RX + int(x), RY + int(y)
        return 0 <= X < BW and 0 <= Y < BH and self.img[Y, X, 3] == 0


def _anc(part, x, y, name):
    px, py = part['pv']
    ax, ay = part['anc'][name]
    return (int(x) - px + ax, int(y) - py + ay)


def _line_pts(x0, y0, x1, y1):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    out = []
    while True:
        out.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return out


# ----------------------------------------------------------------------------
# arms + hands
# hand shapes: (dx, dy, tone) relative to the wrist point, dx mirrored by `dir`
# tone 0 = lit (steel4 / far steel3), 1 = mid, 2 = shade
_HANDS = {
    'none': (),
    'down': ((0, 1, 1),),                       # relaxed, fingers down
    'rest': ((1, 0, 1),),                       # lying on a surface, fingers forward
    'reach': ((1, 0, 0), (2, 0, 1)),            # open, fingers forward
    'up': ((0, -1, 0), (1, -1, 1)),             # fingers up (toward the chest)
    'claw': ((1, 0, 0), (1, 1, 1)),             # curled, digging
    'grip': ((1, 0, 0), (0, 1, 1), (1, 1, 2)),  # closed fist
    'flat': ((1, 0, 0), (2, 0, 1), (-1, 0, 1)),  # palm down, patting / pressing
    'pinch': ((1, 0, 0), (1, -1, 1)),           # holding a small stem
    'cup': ((1, 0, 1), (1, -1, 0)),             # cupped under / around the orb
}
_TONES_N = ('steel4', 'steel3', 'steel2')
_TONES_F = ('steel3', 'steel2', 'steel1')


def _arm(buf, pts, hand='none', far=False, d=1, outline=True):
    """pts: [shoulder, elbow, ..., wrist] absolute pose coords.  Texture follows the
    model (and CHAR-A): near arm t,S,s,S,s..; far arm s,a,s,a.. (darker)."""
    px = []
    for a, b in zip(pts[:-1], pts[1:]):
        for q in _line_pts(a[0], a[1], b[0], b[1]):
            if not px or px[-1] != q:
                px.append(q)
    cols = {}
    for i, q in enumerate(px):
        if far:
            cols[q] = 'steel2' if i % 2 else 'steel1'
        else:
            cols[q] = 'steel4' if i == 0 else ('steel3' if i % 2 else 'steel2')
    tones = _TONES_F if far else _TONES_N
    wx, wy = pts[-1]
    for dx, dy, tn in _HANDS[hand]:
        cols[(wx + d * dx, wy + dy)] = tones[tn]
    ol = set()
    if outline:
        skip = set() if far else {tuple(pts[0])}
        for (x, y) in cols:
            if (x, y) in skip:
                continue
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (nx, ny) not in cols and (buf.empty(nx, ny) or (not far and buf.on_head(nx, ny))):
                    ol.add((nx, ny))
    for (x, y) in ol:            # outline first: against the background and across the head
        buf.pset(x, y, 'steel0')
    for (x, y), c in cols.items():
        buf.pset(x, y, c)
    return px[-1] if px else (wx, wy)


# ----------------------------------------------------------------------------
# props
def _can_sprite(state, cpose='up'):
    from chars import props
    st = state or 'empty'
    if cpose != 'up' and hasattr(props, 'can_pose'):
        try:
            return props.can_pose(cpose, st)
        except Exception:
            pass
    try:
        return props.can(st, scale='medium')
    except TypeError:
        return props.can(st)


def _sprout_sprite(state, t):
    from chars import props
    try:
        return props.sprout(state, t=t, scale='medium')
    except TypeError:
        return props.sprout(state)


# the removed core: heartbeat frames (sentinel-free, real colours)
_ORB_LEG = {'1': 'core1', '2': 'core2', '3': 'core3', '4': 'core4', '5': 'core5', '6': 'core6',
            '7': 'core7', 'W': 'white1'}
_ORB = {
    'rest': from_ascii("""
465
676
465
""", _ORB_LEG, {'center': (1, 1)}),
    'mid': from_ascii("""
.56.
5776
6776
.65.
""", _ORB_LEG, {'center': (1, 1)}),
    'beat': from_ascii("""
4664
6WW6
6WW7
4664
""", _ORB_LEG, {'center': (1, 1)}),
}


def _orb_phase(t):
    """Heartbeat (~1.2 Hz): lub (beat), dub (mid-beat), long rest."""
    p = (t * HEARTBEAT_HZ) % 1.0
    if p < 0.1:
        return 'beat'
    if p < 0.2:
        return 'mid'
    if p < 0.3:
        return 'beat'
    if p < 0.42:
        return 'mid'
    return 'rest'


def _dim_orb(spr, brightness):
    steps = int(round((1.0 - max(0.0, min(1.0, brightness))) * 5))
    if steps <= 0:
        return spr
    s = spr.copy()
    ramp = [P.C(n) for n in ('core0', 'core1', 'core2', 'core3', 'core4', 'core5', 'core6', 'core7')]
    ramp_idx = {c: i for i, c in enumerate(ramp)}
    wh = P.C('white1')
    a = s.img[..., 3] > 0
    ys, xs = np.where(a)
    for y, x in zip(ys, xs):
        c = tuple(int(v) for v in s.img[y, x, :3])
        i = 7 if c == wh else ramp_idx.get(c, 3)
        j = i - steps
        if j < 0:
            s.img[y, x, 3] = 0 if i < 5 else 255
            s.img[y, x, :3] = P.C('rust1') if i >= 5 else s.img[y, x, :3]
        else:
            s.img[y, x, :3] = ramp[j]
    return s


def dig_can(state='empty'):
    """The can exactly as dig_plant draws it (CHAR-A's props.can_pose('up_l', state): spout
    LEFT, handle toward the robot, like the S3 scoop). Shots that draw the can after dig_plant
    (open_chest / place_core / power_down / dead_sit, S8a) should blit THIS with its 'base'
    at feet + DIG_CAN_OFFSET, so the can doesn't flip across the cut."""
    return _can_sprite(state, 'up_l')


def core_orb(t, brightness=1.0):
    """The removed power core as a stand-alone prop (anchor 'center'), pulsing like a
    slow heartbeat (~1.2 Hz). brightness 0..1 steps it down the core ramp (0 = cold ember)."""
    spr = _ORB[_orb_phase(t)].copy()
    return _dim_orb(spr, brightness)


# ----------------------------------------------------------------------------
# porthole hatch states, relative to the port anchor (px,py) = top-left core pixel.
# ring: (0,-1)(1,-1) (-1,0)(2,0) (-1,1)(2,1) (0,2)(1,2)
def _hatch(buf, px, py, state, core, has_core):
    inner = ((0, 0), (1, 0), (0, 1), (1, 1))
    if state == 'closed':
        return
    if state == 'unlatch':
        buf.pset(px + 2, py, 'steel4')          # latch pops: the rim edge catches light
        buf.pset(px + 2, py + 1, 'steel3')
        return
    # from here the porthole is (partly) open: redraw the socket
    cc = _core_cols(core, open_=True) if has_core else _OPEN_TAB[0]
    for (dx, dy), c in zip(inner, cc):
        buf.pset(px + dx, py + dy, c)
    if state == 'ajar':                          # ~45 deg: the glass door covers the left half
        buf.pset(px, py, 'steel4')
        buf.pset(px, py + 1, 'steel3')
        buf.pset(px - 1, py - 1, 'steel0')
        buf.pset(px - 2, py, 'steel0')
        buf.pset(px - 2, py + 1, 'steel0')
    elif state == 'edge':                        # ~90 deg: door edge-on at the hinge
        for dy, c in ((-1, 'steel0'), (0, 'steel5'), (1, 'steel4'), (2, 'steel0')):
            buf.pset(px - 1, py + dy, c)
        buf.pset(px - 2, py, 'steel0')
        buf.pset(px - 2, py + 1, 'steel0')
    elif state == 'open':                        # swung past 90 deg: the door stands out toward
        for dy, c in ((-1, 'steel0'), (0, 'steel5'), (1, 'steel3'), (2, 'steel0')):   # the viewer,
            buf.pset(px - 2, py + dy, c)         # seen nearly edge-on just left of the ring
        buf.pset(px - 3, py, 'steel0')
        buf.pset(px - 3, py + 1, 'steel0')


# ----------------------------------------------------------------------------
# eye / core colour ramps (brightness 0..1 -> sentinel replacement)
# Identical to CHAR-A's tables (chars/robot.py _EYE_TAB / _CORE_TAB) so the same
# eye/core values glow the same in both animators' frames (no pop at hand-overs).
_EYE_TAB = [('steel1', 'ink2', 'ink2'), ('core1', 'core0', 'core0'), ('core2', 'core1', 'core1'),
            ('core3', 'core2', 'core1'), ('core4', 'core3', 'core2'), ('core5', 'core4', 'core2'),
            ('core5', 'core4', 'core3'), ('core6', 'core5', 'core3'), ('core6', 'core5', 'core3'),
            ('core6', 'core6', 'core4')]
_CORE_TAB = [('steel2', 'steel1', 'steel1', 'ink2'), ('core1', 'core0', 'core0', 'steel1'),
             ('core2', 'core1', 'core1', 'core0'), ('core3', 'core2', 'core2', 'core1'),
             ('core4', 'core3', 'core3', 'core2'), ('core5', 'core4', 'core4', 'core2'),
             ('core5', 'core5', 'core4', 'core3'), ('core6', 'core5', 'core5', 'core3'),
             ('core6', 'core5', 'core5', 'core3')]
if 'eye_tab' in _CHARA:
    _EYE_TAB, _CORE_TAB = _CHARA['eye_tab'], _CHARA['core_tab']
_OPEN_TAB = [('ink0', 'ink1', 'ink1', 'steel1'),                      # open & empty socket
             ('core3', 'core2', 'core2', 'core1'), ('core4', 'core3', 'core3', 'core2'),
             ('core5', 'core4', 'core4', 'core3'), ('core6', 'core5', 'core5', 'core4'),
             ('core7', 'core6', 'core6', 'core5')]


def _q(v, n=8):
    return int(round(min(max(float(v), 0.0), 1.0) * n))


def _eye_cols(e):
    k = _q(e)
    pq = _EYE_TAB[min(k, 9)]
    glint = 'white0' if k >= 3 else ('steel5' if k >= 1 else 'steel4')
    return (pq[0], pq[1], pq[2], glint)


def _core_cols(c, open_=False):
    """4 porthole pixels (TL, TR, BL, BR). open_: the hatch is open, the bare core blazes."""
    if open_:
        return _OPEN_TAB[1 + min(4, _q(c, 4))]
    return _CORE_TAB[_q(c)]


def _resolve(img, eye, core):
    rgb = img[..., :3].astype(np.int32)
    key = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    ec = _eye_cols(eye)
    cc = _core_cols(core)
    rep = {'E': ec[0], 'e': ec[1], 'f': ec[2], 'H': ec[3], 'C': cc[0], 'c': cc[1], 'B': cc[2], 'd': cc[3]}
    op = img[..., 3] > 0
    for k, kv in _SENT_KEYS.items():
        m = (key == kv) & op
        if m.any():
            img[m, :3] = P.C(rep[k])


# ----------------------------------------------------------------------------
# FX pixels (dust puffs, soil flecks) -- tiny, deliberate clusters
_FX = {
    # (dx, dy, colour)
    'puff1': ((0, 0, 'dust4'), (-1, -1, 'dust3'), (1, -1, 'dust4')),
    'puff2': ((-1, -1, 'dust3'), (1, -2, 'dust3'), (-2, 0, 'dust2'), (2, 0, 'dust2')),
    'puff3': ((-2, -2, 'dust2'), (2, -2, 'dust2'), (-3, 0, 'dust2')),
    # dry soil flicked up and forward past the lens: pale dust catching the light reads
    # against the dark planter (tested in the real S7 dawn lighting)
    'soil1': ((0, 0, 'dust4'), (1, -1, 'dust5'), (2, -1, 'dust3')),
    'soil2': ((1, -2, 'dust5'), (3, -3, 'dust4'), (2, -5, 'dust5'), (4, -1, 'dust3'), (0, -4, 'dust4')),
    'soil3': ((3, -1, 'dust4'), (5, -2, 'dust3'), (4, 1, 'dust4'), (2, 1, 'dust3')),
    'pat': ((-1, -1, 'dust4'), (3, -1, 'dust4'), (-2, 0, 'dust3')),
}


# ============================================================================
# POSE RENDERER
def _render(p, can_state, core, eye, t):
    b = _Buf()
    anchors = {'feet': (0, 0)}
    bname, bx, by = p['body']
    body = BODIES[bname]
    hname, hx, hy = p['head']
    head = HEADS[(hname, p.get('pupil', 'R'))]
    legs = LEGS[p['legs'][0]]
    lx, ly = p['legs'][1], p['legs'][2]
    cx, cy = _anc(body, bx, by, 'port')
    anchors['core'] = (cx, cy)
    sfx, sfy = _anc(body, bx, by, 'sh_f')
    snx, sny = _anc(body, bx, by, 'sh_n')

    # 1. far arm (behind everything)
    af = p.get('arm_f')
    if af:
        _arm(b, [(sfx, sfy)] + list(af[0]), af[1] if len(af) > 1 else 'none', far=True)
    # 2. legs (back layer), body, neck, head
    b.put(legs['back'], lx, ly)
    b.put(body, bx, by)
    nk = p.get('neck', 1)
    if nk:
        b.put(NECK, bx, by)
        if nk == 2:
            b.put(NECK, bx, by - 1)
    # 3. porthole state (drawn before the head: a bowed head may overlap the chest)
    has_core = p.get('port', 'core') == 'core'
    hatch = p.get('hatch', 'closed')
    if hatch != 'closed' or not has_core:
        _hatch(b, cx, cy, hatch if hatch != 'closed' else 'open', core, has_core)
    sp = p.get('sprout')
    if sp:                      # carried sprout: in front of the torso, behind the head
        spr = _sprout_sprite('wilt', t)
        b.put_sprite(spr, sp[0], sp[1], 'base')
        for dx, dy, c in ((-1, 0, 'dust2'), (0, 0, 'dust3'), (1, 0, 'dust2'), (0, 1, 'dust1')):
            b.pset(sp[0] + dx, sp[1] + dy, c)
    b.put_head(head, hx, hy)
    if 'front' in legs:
        b.put(legs['front'], lx, ly)
    ex, ey = _anc(head, hx, hy, 'eye')
    anchors['eye'] = (ex, ey)
    # 4. props held / standing in the sprite (behind the near hand)
    hand = p.get('arm_n')
    cs = p.get('can')
    if cs:
        # ('mouth'|'base'|'grip', (x, y), can_pose, state_override)
        mode, (qx, qy) = cs[0], cs[1]
        cpose = cs[2] if len(cs) > 2 else 'up'
        st = cs[3] if len(cs) > 3 and cs[3] else can_state
        spr = _can_sprite(st, cpose)
        b.put_sprite(spr, qx, qy, mode)
        ax, ay = spr.anchors[mode]
        for k in ('mouth', 'spout'):
            if k in spr.anchors:
                kx, ky = spr.anchors[k]
                anchors['can_mouth' if k == 'mouth' else 'spout'] = (qx + kx - ax, qy + ky - ay)
    orb = p.get('orb')
    if orb:
        o = _dim_orb(_ORB[_orb_phase(t)], max(core, 0.0) if p.get('orb_dim') else 1.0)
        b.put_sprite(o, orb[0], orb[1], 'center')
        anchors['core_held'] = (orb[0], orb[1])
    # 5. near arm (in front)
    if hand:
        _arm(b, [(snx, sny)] + list(hand[0]), hand[1] if len(hand) > 1 else 'none',
             far=False, d=hand[2] if len(hand) > 2 else 1)
        anchors['hand'] = tuple(hand[0][-1])
    else:
        anchors['hand'] = (snx, sny + 5)
    for fx in p.get('fx', ()):
        name, fx_x, fx_y = fx
        for dx, dy, c in _FX[name]:
            b.pset(fx_x + dx, fx_y + dy, c, only_empty=True)
    anchors.setdefault('can_mouth', anchors['hand'])
    anchors.setdefault('spout', anchors['hand'])
    anchors.setdefault('core_held', anchors['core'])
    if 'back' in p:
        anchors['back'] = (p['back'], 0)
    e = eye * p.get('eye_mul', 1.0)
    _resolve(b.img, e, core if has_core else 0.0)
    return b, anchors


def _crop(b, anchors):
    b.img[RY + 1:] = 0          # nothing may hang below the floor row (y = 0)
    a = b.img[..., 3] > 0
    ys, xs = np.where(a)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    img = np.ascontiguousarray(b.img[y0:y1 + 1, x0:x1 + 1])
    anc = {k: (int(RX + v[0] - x0), int(RY + v[1] - y0)) for k, v in anchors.items()}
    return Sprite(img, anc)


def render_pose(p, can_state=None, core=1.0, eye=1.0, t=0.0):
    b, anc = _render(p, can_state, core, eye, t)
    return _crop(b, anc)


def pose(base=None, **kw):
    d = dict(base) if base else {}
    d.update(kw)
    return d


# ============================================================================
# KEY POSES  (all coordinates relative to the feet anchor)
# ----------------------------------------------------------------------------
# standing, stooped, hugging the can against the belly with both arms == CHAR-A's
# walk_tired stance (chars/robot.py _tired_pose + _hug at by=1: torso 'l1', head 'd1',
# glass 'down', can mouth at (6,-4)), so walk_tired -> sit hands over without a pop.
STAND = dict(body=_body_at('l1', 0, -2), head=('a_d1', 0, -11), pupil='W', legs=('stand1', 0, 0),
             arm_n=([(0, -5), (3, -3), (7, -3)], 'none'), arm_f=([(6, -5), (9, -4)], 'none'),
             can=('mouth', (6, -4)))

# seated against something on its LEFT, legs forward: body bottom on the floor
_SIT_B = _body_at('rec', -1, 1)                     # -> ('rec', -3, -8)
SIT_REST = dict(body=_SIT_B, head=('dn1', -2, -8), legs=('sit', 0, 0),
                arm_n=([(-4, -3), (-3, -1)], 'down'), back=SIT_BACK_X)
_SIT_UP = _body_at('up', -1, 1)                     # leaning off the wall (curious)

# kneeling at the planter (planter on the RIGHT: soil y=-11) == CHAR-A's KNEEL + 2px
_KN_B = _body_at('up', 3, -1)                       # -> ('up', 2, -10): bottom outline y=-2
KNEEL = dict(body=_KN_B, head=('dn1', 2, -10), legs=('kneel', 0, 0),
             arm_n=([(2, -5), (5, -3)], 'none'))
_KN_F1 = _body_at('f1', 3, -1)
_KN_F1f = _body_at('f1', 4, -1)                     # weight rocked forward onto the front knee
_KN_F2 = _body_at('f2', 4, -1)
_KN_F2f = _body_at('f2', 5, -1)
_KN_REC = _body_at('rec', 2, -1)                    # turned/leaning back toward the can


# ============================================================================
# TIMELINES — (t_start, pose) at 12 fps with holds.  f(n) = n/12 s.
def f(n):
    return n / FPS_CHAR


def _hug_can(mx, my, body, head, legs=('stand', 0, 0), near=None, far=None, **kw):
    """both hands on the can whose mouth is at (mx,my): near hand on the handle side,
    far hand steadying the front."""
    return pose(STAND, body=body, head=head, legs=legs,
                arm_n=(near or [(mx - 3, my + 3), (mx - 1, my + 1)], 'none'),
                arm_f=(far or [(mx + 1, my - 1), (mx + 2, my)], 'none'),
                can=('mouth', (mx, my)), **kw)


# ---------------------------------------------------------------- sit (0.8 s)
# hug -> tired sag -> bends & sets the can down with both hands at +13 (touch 1/3 s,
# released at 0.45) -> pushes back -> drops heavily onto the floor (impact 7/12)
# -> head / arm overlap -> settle -> hold.
SIT_FR = [
    (f(0), STAND),
    # anticipation: tired sag, knees give a little, head drops a notch
    (f(1), pose(STAND, body=_body_at('l1', 0, -1), head=('dn1', 0, -10), pupil='R', legs=('crouch', 0, 0),
                arm_n=([(0, -4), (3, -2), (7, -2)], 'none'), arm_f=([(6, -4), (9, -3)], 'none'),
                can=('mouth', (6, -3)))),
    # leans in: the can swings forward, lifted clear of the floor
    (f(2), _hug_can(9, -5, _body_at('f1', 1, -2), ('dn2', 3, -12), ('crouch', 0, 0),
                    near=[(3, -5), (6, -3)], far=[(7, -6), (11, -5)])),
    # lunge + deep bend: can coming down
    (f(3), _hug_can(12, -5, _body_at('f2', 3, -2), ('dn2', 6, -11), ('lunge', 0, 0),
                    near=[(6, -5), (9, -3)], far=[(10, -6), (14, -5)])),
    # touches down at CAN_SET_OFFSET (base (13,0)): careful, holds it a beat
    (f(4), _hug_can(13, -4, _body_at('f2', 3, -2), ('dn2', 6, -11), ('lunge', 0, 0),
                    near=[(6, -5), (10, -3)], far=[(10, -6), (15, -4)], fx=[('puff1', 10, 0)])),
    # SIT_CAN_DOWN_T: lets go (the shot draws the can from here), starts to push back
    (SIT_CAN_DOWN_T, pose(STAND, body=_body_at('f1', 1, -2), head=('dn2', 3, -12), legs=('crouch', 0, 0),
                          arm_n=([(3, -6), (7, -6)], 'none'), arm_f=([(7, -7), (10, -7)], 'none'), can=None)),
    # falls back: torso upright, legs sliding out, head lags forward
    (f(6), pose(SIT_REST, body=_body_at('up', -1, -1), head=('dn2', 0, -10), legs=('sitland', 0, 0),
                arm_n=([(0, -6), (3, -7)], 'none'))),
    # IMPACT: butt + backpack hit (squash: head sinks into the neck), dust
    (SIT_IMPACT_T, pose(SIT_REST, head=('dn2', -1, -7), neck=0, arm_n=([(-2, -4), (1, -5)], 'none'),
                        fx=[('puff1', -6, 0), ('puff1', 9, 0)])),
    # overlap: head rebounds (neck stretches 1px), arm flops down late
    (f(8), pose(SIT_REST, head=('dn1', -2, -9), neck=2, arm_n=([(-3, -3), (-1, -2)], 'none'),
                fx=[('puff2', -6, 0), ('puff2', 10, 0)])),
    (f(9), pose(SIT_REST, arm_n=([(-4, -3), (-2, -1)], 'none'), fx=[('puff3', -6, 0)])),
    (0.8, SIT_REST),
]

# ---------------------------------------------------------------- sit_idle (3 s loop)
_SIT_EXH = pose(SIT_REST, head=('dn1', -2, -7), neck=0)             # head sinks 1px (exhale)
_SIT_NOD = pose(SIT_REST, head=('dn1', -1, -7), neck=0)             # sinks + sways forward
SIT_IDLE_FR = [
    (0.0, SIT_REST),
    (0.75, _SIT_EXH),
    (1.33, SIT_REST),
    (2.08, _SIT_NOD),
    (2.42, pose(_SIT_NOD, eye_mul=0.45)),                            # drowsy lens flicker
    (2.50, _SIT_NOD),
    (2.67, SIT_REST),
]

# ---------------------------------------------------------------- sit_look_can (0.5 s)
_LC = pose(SIT_REST, body=_SIT_UP, head=('dn2', -1, -8), pupil='R')
SIT_LOOK_CAN_FR = [
    (f(0), SIT_REST),
    (f(1), pose(SIT_REST, head=('lvl', -2, -8))),                        # perks up
    (f(2), pose(_LC, head=('dn2', 0, -8))),                              # turns to the can (overshoot)
    (f(4), _LC),                                                          # settle
    (f(5), pose(_LC, pupil='S', head=('dn2', -1, -9), neck=2)),          # iris refocuses: a curious lift
    (0.5, _LC),
]
_LC_BREATH = pose(_LC, head=('dn2', -1, -7), neck=0)

# ---------------------------------------------------------------- sit_look_chest (0.6 s)
_CH = pose(SIT_REST, body=_SIT_UP, head=('dn3', -1, -8), pupil='L')
SIT_LOOK_CHEST_FR = [
    (f(0), _LC),
    (f(1), pose(_LC, head=('dn1', -1, -8), pupil='R')),                  # breaks off the sprout
    (f(3), pose(_LC, head=('dn2', -1, -8), pupil='L')),                  # eyes drop to its chest
    (f(4), _CH),                                                          # bows: looks at its heart
    (f(5), pose(_CH, arm_n=([(-4, -3), (-2, -2)], 'up'))),               # the hand starts to rise
    (f(6), pose(_CH, arm_n=([(-3, -2), (0, -2)], 'up'))),
    (f(7), pose(_CH, arm_n=([(-3, -2), (1, -3)], 'up'))),                # halfway: stops
]
_CH_HOLD = pose(_CH, arm_n=([(-3, -2), (1, -3)], 'up'))
_CH_BREATH = pose(_CH_HOLD, head=('dn3', -1, -7), neck=0)
_CH_DIP = pose(_CH, arm_n=([(-3, -2), (1, -2)], 'up'))              # hesitates...
_CH_RESOLVE = pose(_CH, arm_n=([(-3, -3), (1, -4)], 'up'))          # ...then a pixel closer: decided
_CH_RESOLVE_BREATH = pose(_CH_RESOLVE, head=('dn3', -1, -7), neck=0)


def _chest_hold(t):
    """sit_look_chest hold (t >= 0.6): the hand hovers halfway, hesitates (0.75-0.9), then
    rises one pixel closer to the heart (1.0: decided) and stays; the head keeps breathing.
    (S6 cuts at t=1.4, so the decision is on screen for its last 0.4 s.)"""
    if t < 0.75:
        return _CH_HOLD
    if t < 0.9:
        return _CH_DIP
    if t < 1.0:
        return _CH_HOLD
    u = (t - 1.0) % 1.4
    return _CH_RESOLVE_BREATH if 0.55 <= u < 0.97 else _CH_RESOLVE


# ---------------------------------------------------------------- dig_plant (2.0 s)
# The can stands BEHIND the robot (spout left, like CHAR-A's scoop in S3): it digs,
# turns back and lifts the sprout out of the can, swings it forward over its knee and
# sets it into the hole at PLANT_OFFSET, firms the soil and pats it twice.
_DCAN = ('base', DIG_CAN_OFFSET, 'up_l')
_DCAN_E = ('base', DIG_CAN_OFFSET, 'up_l', 'empty')
_K1 = pose(KNEEL, body=_KN_F1, head=('dn2', 4, -11), can=_DCAN)
_K1f = pose(KNEEL, body=_KN_F1f, head=('dn2', 5, -11), can=_DCAN)
_K2 = pose(KNEEL, body=_KN_F2f, head=('lvl', 8, -10), pupil='D', can=_DCAN_E)
# soil work: torso leans in, head stays LEVEL with the pupil looking down, so the lens sits
# just above the soil and the hands work visibly beneath it
_KD = pose(KNEEL, body=_KN_F1f, head=('lvl', 6, -11), pupil='D', can=_DCAN)
DIG_FR = [
    (f(0), pose(KNEEL, can=_DCAN)),
    (f(1), pose(_KD, head=('dn1', 5, -11), pupil='R', arm_n=([(5, -9), (9, -12)], 'claw'))),   # reaches over the rim
    (f(2), pose(_KD, arm_n=([(7, -10), (12, -12)], 'claw'))),           # hand raised over the spot
    (f(3), pose(_KD, arm_n=([(8, -9), (14, -11)], 'claw'))),            # digs in
    (f(5), pose(_KD, arm_n=([(7, -9), (12, -10)], 'claw'), fx=[('soil1', 15, -12)])),
    (DIG_SCRAPE_TS[0], pose(_KD, head=('lvl', 5, -11), arm_n=([(6, -8), (10, -10)], 'claw'),
                            fx=[('soil2', 15, -12)])),                                           # SCRAPE 1
    (f(7), pose(_KD, arm_n=([(7, -10), (13, -12)], 'claw'), fx=[('soil3', 15, -14)])),
    (f(8), pose(_KD, arm_n=([(8, -9), (14, -11)], 'claw'))),
    (DIG_SCRAPE_TS[1], pose(_KD, head=('lvl', 5, -11), arm_n=([(6, -8), (10, -10)], 'claw'),
                            fx=[('soil2', 15, -12)])),                                           # SCRAPE 2
    # turns back to the can: leans back, lens looks back, hand reaches behind
    (f(10), pose(KNEEL, body=_KN_REC, head=('dn1', 1, -10), pupil='L', can=_DCAN,
                 arm_n=([(-2, -5), (-5, -6)], 'reach', -1), fx=[('soil3', 15, -14)])),
    # DIG_TAKE_T: lifts the wilted sprout (with its soil) out of the can
    (DIG_TAKE_T, pose(KNEEL, body=_KN_REC, head=('dn1', 1, -10), pupil='L', can=_DCAN_E,
                      arm_n=([(-3, -6), (-7, -6)], 'pinch', -1), sprout=(-8, -5))),
    # swings it forward in front of its chest (past its heart), then up over the rim
    (f(12), pose(KNEEL, head=('dn2', 2, -10), pupil='L', can=_DCAN_E,
                 arm_n=([(1, -3), (4, -4)], 'pinch'), sprout=(5, -4))),
    (f(13), pose(_K1f, head=('dn1', 5, -11), can=_DCAN_E, arm_n=([(7, -9), (11, -12)], 'pinch'), sprout=(12, -12))),
    (f(14), pose(_K2, arm_n=([(10, -9), (14, -12)], 'pinch'), sprout=(15, -12))),
    # PLANT_T: set into the hole (the shot draws it from here); hand firms the soil
    (PLANT_T, pose(_K2, arm_n=([(10, -9), (13, -11)], 'flat'))),
    (f(15), pose(_K2, arm_n=([(9, -9), (12, -11)], 'flat'))),
    (f(16), pose(_K2, arm_n=([(10, -10), (13, -13)], 'flat'))),
    (f(18), pose(_K2, arm_n=([(10, -11), (13, -14)], 'flat'))),
    (DIG_PAT_TS[0], pose(_K2, arm_n=([(10, -9), (13, -11)], 'flat'), fx=[('pat', 14, -11)])),       # PAT
    (f(20), pose(_K2, arm_n=([(10, -10), (13, -13)], 'flat'))),
    (DIG_PAT_TS[1], pose(_K2, arm_n=([(10, -9), (13, -11)], 'flat'), fx=[('pat', 14, -11)])),       # PAT
    (f(22), pose(_K1f, can=_DCAN_E, arm_n=([(7, -9), (10, -10)], 'none'))),
    (f(23), pose(KNEEL, can=_DCAN_E, arm_n=([(3, -6), (7, -8)], 'none'))),
    (2.0, pose(KNEEL, can=_DCAN_E)),
]

# ---------------------------------------------------------------- open_chest (1.2 s)
# porthole of the kneeling torso: core pixel at (4,-7); ring (3..6, -8..-5)
_OC = pose(KNEEL, head=('dn2', 2, -10), pupil='L')
OPEN_FR = [
    (f(0), KNEEL),
    (f(1), pose(_OC, arm_n=([(2, -4), (4, -4)], 'up'))),                  # hand rises
    (f(2), pose(_OC, arm_n=([(3, -4), (7, -6)], 'claw', -1))),            # fingers on the latch
    (f(3), pose(_OC, arm_n=([(3, -4), (7, -7)], 'claw', -1), hatch='unlatch')),   # click
    (f(4), pose(_OC, arm_n=([(3, -5), (5, -7)], 'claw', -1), hatch='ajar')),      # swings...
    (f(5), pose(_OC, arm_n=([(2, -5), (2, -8)], 'claw', -1), hatch='edge')),
    (f(6), pose(_OC, arm_n=([(3, -4), (6, -5)], 'none'), hatch='open')),          # open: the heart blazes
    (f(7), pose(_OC, arm_n=([(3, -4), (5, -6)], 'claw', -1), hatch='open')),      # reaches in
    (CORE_OUT_T, pose(_OC, arm_n=([(3, -4), (6, -6)], 'cup'), hatch='open', port='empty',
                      orb=(6, -7))),                                                # lifts it out
    (f(10), pose(_OC, pupil='R', arm_n=([(3, -4), (7, -6)], 'cup'), hatch='open',
                 port='empty', orb=(8, -8))),
    (f(11), pose(_OC, head=('dn1', 2, -10), pupil='R', arm_n=([(3, -4), (7, -5)], 'cup'), hatch='open',
                 port='empty', orb=(8, -7), arm_f=([(7, -5), (10, -6)], 'cup'))),
    (f(13), pose(_OC, pupil='R', arm_n=([(3, -4), (7, -5)], 'cup'), hatch='open',
                 port='empty', orb=(8, -7), arm_f=([(7, -5), (10, -6)], 'cup'))),
]
_HOLD_CORE = OPEN_FR[-1][1]

# ---------------------------------------------------------------- place_core (1.0 s)
# offers the heart up at eye level, presses it into the soil (CORE_PLACE_T), both hands
# linger flat on the soil; then the far hand withdraws while the NEAR hand slides down to
# rest on the rim, just below the buried heart -- it stays there through power_down.
_PC = dict(hatch='open', port='empty')
_KP1 = dict(body=_KN_F1f, head=('dn2', 5, -11))
_KP2 = dict(body=_KN_F2f, head=('dn2', 7, -10))
RIM_HAND = (10, -9)                     # the hand that never lets go (on the rim's front edge)


def _rim_arm(sh):
    """near arm from shoulder `sh` down along the body (clear of the open porthole),
    elbow low, forearm rising to rest on the rim at RIM_HAND."""
    sx, sy = sh
    ey = min(sy + 3, -2)
    return ([(sx, sy + 2), (sx + 1, ey + 1), (sx + 3, ey), (6, -6), RIM_HAND], 'rest')


def _sh(body):
    return _anc(BODIES[body[0]], body[1], body[2], 'sh_n')


PLACE_FR = [
    (f(0), _HOLD_CORE),
    (f(1), pose(_HOLD_CORE, **_KP1, orb=(9, -9),
                arm_n=([(6, -6), (8, -8)], 'cup'), arm_f=([(9, -7), (11, -8)], 'cup'))),
    (f(2), pose(_HOLD_CORE, body=_KN_F1f, head=('dn1', 5, -11), orb=(10, -13),
                arm_n=([(7, -9), (9, -12)], 'cup'), arm_f=([(10, -10), (12, -12)], 'cup'))),   # offered up
    (f(3), pose(_HOLD_CORE, **_KP2, orb=(11, -13),
                arm_n=([(9, -9), (10, -12)], 'cup'), arm_f=([(11, -10), (13, -12)], 'cup'))),
    (f(4), pose(_HOLD_CORE, **_KP2, orb=(11, -11),
                arm_n=([(9, -8), (10, -10)], 'cup'), arm_f=([(11, -9), (13, -10)], 'cup'))),   # touches soil
    # CORE_PLACE_T: pressed in, gone from the sprite; both hands flat on the soil (tender)
    (CORE_PLACE_T, pose(KNEEL, **_KP2, **_PC,
                        arm_n=([(9, -8), (10, -10)], 'flat'), arm_f=([(12, -9), (13, -10)], 'flat'))),
    (f(6), pose(KNEEL, **_KP2, **_PC,                                                              # relax
                arm_n=([(9, -8), (10, -11)], 'flat'), arm_f=([(12, -9), (13, -11)], 'flat'))),
    (f(7), pose(KNEEL, **_KP2, **_PC,
                arm_n=([(9, -8), (10, -11)], 'flat'), arm_f=([(12, -10), (13, -12)], 'flat'))),   # far hand lifts first
    (f(8), pose(KNEEL, **_KP1, **_PC, arm_n=([(8, -7), (10, -10)], 'rest'))),                      # near hand slides to the rim
    (f(9), pose(KNEEL, body=_KN_F1f, head=('dn1', 5, -11), **_PC, arm_n=_rim_arm(_sh(_KN_F1f)))),
    (f(10), pose(KNEEL, head=('dn1', 2, -10), **_PC, arm_n=_rim_arm(_sh(_KN_B)))),                 # end pose (0.83)
]
_AFTER_PLACE = PLACE_FR[-1][1]

# ---------------------------------------------------------------- power_down (1.6 s)
# eye: two flickers, then fades and dies at POWER_EYE_OFF_T. body: sags back onto its heels,
# away from the planter, while the near hand stays on the rim below its heart; the head
# bows toward the glow; last settle (head drops a pixel with the 'click', then the arm's
# weight) -> utterly still from POWER_STILL_T.
_B_HEEL = _body_at('up', 0, 0)
_DEAD_ARM = _rim_arm(_sh(_B_HEEL))
DEAD = pose(KNEEL, body=_B_HEEL, head=('dn2', -1, -9), legs=('heels', 0, 0), **_PC, pupil='D',
            arm_n=_DEAD_ARM, arm_f=None)
_b1 = _body_at('up', 2, 0)
_b2 = _body_at('up', 1, 0)
POWER_FR = [
    (f(0), _AFTER_PLACE),
    (f(1), pose(_AFTER_PLACE, head=('dn1', 2, -9), neck=0)),                          # first sag
    (f(2), pose(DEAD, body=_b1, head=('dn1', 1, -9), pupil='R', legs=('kneel2', 0, 0),
                arm_n=_rim_arm(_sh(_b1)))),                                             # knee comes down
    (f(4), pose(DEAD, body=_b2, head=('dn2', 0, -10), pupil='R', neck=2,
                arm_n=_rim_arm(_sh(_b2)))),                                             # onto its heels
    (f(6), pose(DEAD, head=('dn2', -1, -10), pupil='R', neck=2)),                      # sags back, head lags
    (f(8), pose(DEAD, head=('dn2', -1, -10), pupil='D', neck=2)),
    (POWER_EYE_OFF_T, pose(DEAD, arm_n=([(-2, -4), (-1, -3), (1, -4), (6, -6), RIM_HAND], 'rest'))),  # click: head drops
    (POWER_STILL_T, DEAD),                                                             # the arm's weight settles
]
# eye envelope inside power_down (x caller's eye): two flickers, a fading glow, dark
_POWER_EYE = [(0.0, 1.0), (0.25, 0.0), (1 / 3, 0.8), (0.5, 0.0), (7 / 12, 0.5), (2 / 3, 0.3),
              (0.75, 0.2), (5 / 6, 0.1), (POWER_EYE_OFF_T, 0.0)]

# ============================================================================
_ANIMS = {
    'sit': SIT_FR, 'sit_look_can': SIT_LOOK_CAN_FR, 'sit_look_chest': SIT_LOOK_CHEST_FR,
    'sit_idle': SIT_IDLE_FR, 'dig_plant': DIG_FR, 'open_chest': OPEN_FR, 'place_core': PLACE_FR,
    'power_down': POWER_FR, 'dead_sit': [(0.0, DEAD)],
}


def _pick(fr, t):
    k = 0
    for i, (ts, _) in enumerate(fr):
        if t + 1e-6 >= ts:
            k = i
    return fr[k][1]


def _step(t):
    return math.floor(t * FPS_CHAR + 1e-6) / FPS_CHAR


def _breathing_hold(t, t_end, still, breath, period=1.25, dur=0.42, first=0.55):
    """Holds that still breathe: 1px sink for `dur` every `period` seconds."""
    if t < t_end + first:
        return still
    ph = (t - t_end - first) % period
    return breath if ph < dur else still


def _pose_for(anim, t):
    if anim == 'sit_idle':
        return _pick(SIT_IDLE_FR, t % 3.0)
    if anim == 'sit_look_can' and t >= 0.5:
        return _breathing_hold(t, 0.5, _LC, _LC_BREATH)
    if anim == 'sit_look_chest' and t >= 0.6:
        return _chest_hold(t)
    return _pick(_ANIMS[anim], t)


def _power_eye(t):
    v = 1.0
    for ts, e in _POWER_EYE:
        if t + 1e-6 >= ts:
            v = e
    return v


def frame(anim, t, can=None, variant='damaged', scale='medium', core=1.0, eye=1.0):
    if anim not in HANDLES:
        raise KeyError(anim)
    t = max(0.0, float(t))
    if scale == 'wide':
        return _wide_frame(anim, t, core, eye, variant)
    # keyframes are authored on the 12 fps grid (+ exact hand-off times), so poses are
    # picked with the raw t; the orb heartbeat is stepped at 12 fps too.
    p = _pose_for(anim, t)
    if anim == 'dig_plant' and can not in ('sprout', 'sprout_wilt', 'sprout_lean'):
        can = 'sprout_wilt'        # the brief: the can holds the wilted sprout until DIG_TAKE_T
    elif can is None:
        can = 'empty'              # (like chars/robot.py: a held can with no state is empty)
    e = eye
    if anim == 'power_down':
        e = eye * _power_eye(t)
    elif anim == 'dead_sit':
        e = 0.0
    b, anc = _render(p, can, core, e, _step(t))
    if variant == 'normal':
        _antenna(b, p, anc)
    return _crop(b, anc)


def _antenna(b, p, anc):
    """variant='normal' (not used in the greenhouse): the intact antenna on the stub.
    Stem 2px + amber bulb, tilted with the head's pitch."""
    hname, hx, hy = p['head']
    head = HEADS[(hname, p.get('pupil', 'R'))]
    sx, sy = _anc(head, hx, hy, 'stub')
    lean = {'dn2': 1, 'dn3': 1}.get(hname, 0)
    b.pset(sx, sy, 'steel3')
    b.pset(sx + lean, sy - 1, 'steel3')
    bx, by = sx + 2 * lean, sy - 2
    for dx, dy in ((0, -1), (-1, 0), (1, 0)):
        b.pset(bx + dx, by + dy, 'steel0', only_empty=True)
    b.pset(bx, by, 'core5')


# ============================================================================
# WIDE scale (~11 px standing): sit / sit_idle / dead_sit.
# Built on CHAR-A's hand-pixelled wide robot (chars/robot.py _WIDE_UPPER, damaged):
# same head + torso pixels, legs forward (sit) or folded on the heels (dead).
_W_HEAD = from_ascii("""
.KrKKK...
Ks544KKK.
K443KHEK.
K3322KKK.
.KKKKK...
""", LEG)
_W_SIT = from_ascii("""
.......K.
KaK4CKKRK
KaK32KtoK
.KKKKKKKK
""", LEG)
_W_HEELS = from_ascii("""
.........
KaKCKK...
KaK32KK..
.KKKKKtK.
""", LEG)
_W_FEET = (4, 8)            # feet anchor in the 12x10 wide canvas (floor row = 8)


def _wide_img(body, hdx, hdy, eye, core, dark_core=False, variant='damaged'):
    img = np.zeros((10, 12, 4), np.uint8)
    a = body.img[..., 3] > 0
    img[5:5 + body.h, 0:body.w][a] = body.img[a]
    a = _W_HEAD.img[..., 3] > 0
    img[1 + hdy:1 + hdy + _W_HEAD.h, hdx:hdx + _W_HEAD.w][a] = _W_HEAD.img[a]
    if variant == 'normal':                     # intact antenna (CHAR-A's wide art)
        img[1 + hdy, 2 + hdx] = (*P.C('steel3'), 255)
        img[hdy, 2 + hdx] = (*P.C('core5'), 255)
    _resolve(img, eye, 0.0 if dark_core else core)
    anc = {'feet': _W_FEET, 'eye': (6 + hdx, 3 + hdy), 'core': (4, 6), 'hand': (5, 7),
           'back': (0, _W_FEET[1])}
    anc['can_mouth'] = anc['hand']
    anc['spout'] = anc['hand']
    anc['core_held'] = anc['core']
    return Sprite(img, anc)


def _wide_frame(anim, t, core, eye, variant='damaged'):
    v = variant
    if anim in ('dead_sit', 'power_down'):
        if anim == 'power_down' and t < POWER_EYE_OFF_T:
            return _wide_img(_W_HEELS, 0, 0, eye * _power_eye(t), 0.0, dark_core=True, variant=v)
        return _wide_img(_W_HEELS, 1, 1, 0.0, 0.0, dark_core=True, variant=v)
    if anim == 'sit' and t < SIT_IMPACT_T:
        # still standing: CHAR-A's own wide idle frame, so the mini sits down from their pose
        try:
            from chars import robot
            return robot.frame('idle', 0.0, can=None, variant=v, scale='wide', core=core, eye=eye)
        except Exception:
            pass
    if anim in ('sit', 'sit_look_can', 'sit_look_chest'):
        return _wide_img(_W_SIT, 0, 0, eye, core, variant=v)
    if anim in ('dig_plant', 'open_chest', 'place_core'):
        # medium-only anims: a kneeling-at-work mini as a safe fallback
        return _wide_img(_W_HEELS, 1, 0, eye, core if anim != 'place_core' else 0.0,
                         dark_core=anim == 'place_core', variant=v)
    # sit_idle (3 s loop): head sways 1 px forward twice (never covers the 1-px core)
    u = t % 3.0
    sway = 1 if (0.75 <= u < 1.33) or (2.08 <= u < 2.67) else 0
    return _wide_img(_W_SIT, sway, 0, eye * (0.45 if 2.42 <= u < 2.5 else 1.0), core, variant=v)
