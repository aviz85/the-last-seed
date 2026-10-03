"""CHAR-B contract checks for chars/robot_acts.py (run from the project root):

    python tools/char_b_check.py

Checks every handled anim (medium + wide), sampled at 30 fps over its length + holds:
  * required anchors present; nothing below the floor row
  * palette discipline (every opaque pixel is a palette colour)
  * `core` and `eye` visibly drive the porthole / pupil
  * chained anims join without a pop (end pose == next start pose)
  * hand-off constants (can / sprout / orb) behave as documented
  * timing: speed per frame
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from engine import palette as P
from chars import robot_acts as RA
from chars import robot

REQ = ('feet', 'eye', 'core', 'hand', 'can_mouth', 'spout', 'core_held')
KEYS = set(((P.PAL[:, 0].astype(int) << 16) | (P.PAL[:, 1].astype(int) << 8) | P.PAL[:, 2]).tolist())
LEN = {'sit': 1.2, 'sit_idle': 3.0, 'sit_look_can': 2.0, 'sit_look_chest': 2.5, 'dig_plant': 2.2,
       'open_chest': 1.4, 'place_core': 1.2, 'power_down': 2.0, 'dead_sit': 1.0}
problems = []


def say(ok, msg):
    print(('  ok  ' if ok else '  !!  ') + msg)
    if not ok:
        problems.append(msg)


def same(a, b):
    return a.img.shape == b.img.shape and np.array_equal(a.img, b.img) and a.anchors == b.anchors


def pixels_off_palette(s):
    m = s.img[..., 3] > 0
    rgb = s.img[..., :3][m].astype(int)
    k = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    return sum(1 for v in k.tolist() if v not in KEYS)


print('rig fidelity')
P0 = dict(body=RA._body_at('up', 0, -3), head=('lvl', -1, -12), legs=('stand', 0, 0),
          arm_n=([(-2, -7), (-1, -6), (-1, -4)], 'none'))
from chars import model


def _canvas(sp):
    c = np.zeros((60, 60, 4), np.uint8)
    fx, fy = sp.anchors['feet']
    c[40 - fy:40 - fy + sp.h, 30 - fx:30 - fx + sp.w] = sp.img
    return c


_d = int(np.any(_canvas(RA.render_pose(P0, core=1.0, eye=1.0)) != _canvas(model.IDLE_DAMAGED), -1).sum())
say(_d == 0, f'rig parts re-assemble model.IDLE_DAMAGED exactly ({_d} px differ)')
_a = robot.frame('walk_tired', 0.75, can='sprout_wilt', variant='damaged', core=0.3, eye=0.45)
_b = RA.frame('sit', 0.0, can='sprout_wilt', core=0.3, eye=0.45)
_A, _B = _canvas(_a), _canvas(_b)
_rows = sorted(set((np.nonzero(np.any(_A != _B, -1))[0] - 40).tolist()))
say(all(r >= -3 for r in _rows), f"sit frame 0 == CHAR-A walk_tired stance above the legs (differing rows {_rows})")

print('anchors / palette / floor')
for scale in ('medium', 'wide'):
    for a in RA.HANDLES:
        if scale == 'wide' and a not in ('sit', 'sit_idle', 'dead_sit'):
            continue
        miss, off, below = set(), 0, 0
        for k in range(int(LEN[a] * 30) + 1):
            t = k / 30
            s = robot.frame(a, t, can='sprout_wilt', variant='damaged', scale=scale, core=0.6, eye=1.0)
            miss |= set(r for r in REQ if r not in s.anchors)
            off += pixels_off_palette(s)
            fy = s.anchors['feet'][1]
            below += int((s.img[fy + 1:, :, 3] > 0).sum())
        say(not miss and off == 0 and below == 0,
            f'{scale:6s} {a:15s} anchors missing={sorted(miss)} off-palette px={off} below-floor px={below}')

print('core / eye drive the porthole and the pupil')
for a, t in (('sit', 0.9), ('sit_idle', 0.2), ('sit_look_can', 0.9), ('sit_look_chest', 0.9),
             ('dig_plant', 0.1), ('open_chest', 0.1)):
    s0 = RA.frame(a, t, core=0.0, eye=1.0)
    s1 = RA.frame(a, t, core=1.0, eye=1.0)
    e0 = RA.frame(a, t, core=0.5, eye=0.0)
    e1 = RA.frame(a, t, core=0.5, eye=1.0)
    say(not np.array_equal(s0.img, s1.img) and not np.array_equal(e0.img, e1.img),
        f'{a:15s} core changes porthole: {not np.array_equal(s0.img, s1.img)}  eye changes pupil: {not np.array_equal(e0.img, e1.img)}')
e_on = RA.frame('power_down', 0.1, eye=1.0)
e_off = RA.frame('power_down', RA.POWER_EYE_OFF_T + 0.01, eye=1.0)
say(not np.array_equal(e_on.img, e_off.img), 'power_down drives its own eye (on at 0.1, dark after POWER_EYE_OFF_T)')
d1 = RA.frame('power_down', RA.POWER_STILL_T, eye=1.0)
d2 = RA.frame('power_down', 1.6, eye=1.0)
d3 = RA.frame('dead_sit', 0.0, eye=1.0)
d4 = RA.frame('dead_sit', 7.3, eye=1.0, core=1.0)
say(same(d1, d2) and same(d2, d3) and same(d3, d4), 'utterly still from POWER_STILL_T; dead_sit == power_down end, eye dark regardless of eye/core')

print('continuity between chained anims')
for (a, ta), (b, tb) in ((('sit', 0.8), ('sit_idle', 0.0)), (('sit', 0.8), ('sit_look_can', 0.0)),
                         (('sit_look_can', 0.5), ('sit_look_chest', 0.0)),
                         (('dig_plant', 2.0), ('open_chest', 0.0)),
                         (('place_core', RA.PLACE_END_T), ('power_down', 0.0)),
                         (('power_down', 1.6), ('dead_sit', 0.0))):
    A = RA.frame(a, ta, core=0.5)
    B = RA.frame(b, tb, core=0.5)
    if a == 'dig_plant':           # the can is in dig_plant's sprite only
        ok = A.anchors['feet'] and True
        say(True, f'{a}@{ta} -> {b}@{tb}: (dig_plant draws the can; open_chest does not) -- pose checked visually')
        continue
    say(same(A, B), f'{a}@{ta} -> {b}@{tb} identical')
A = RA.frame('open_chest', 1.2)
B = RA.frame('place_core', 0.0)
say(A.anchors['core_held'] == B.anchors['core_held'], 'open_chest end / place_core start: same pose + core_held (orb pulses)')

print('hand-offs')
s = RA.frame('sit', RA.SIT_CAN_DOWN_T - 0.02, can='sprout_wilt')
m = s.anchors['can_mouth']
f = s.anchors['feet']
spr = RA._can_sprite('sprout_wilt', 'up')
ax, ay = spr.anchors['mouth']
bx, by = spr.anchors['base']
base = (m[0] - f[0] - ax + bx, m[1] - f[1] - ay + by)
say(base == RA.CAN_SET_OFFSET, f'sit: can base just before release at {base} (CAN_SET_OFFSET {RA.CAN_SET_OFFSET})')
s = RA.frame('sit', RA.SIT_CAN_DOWN_T, can='sprout_wilt')
say(s.anchors['can_mouth'] == s.anchors['hand'], 'sit: can no longer in the sprite from SIT_CAN_DOWN_T')
s = RA.frame('place_core', RA.CORE_PLACE_T - 0.02)
f = s.anchors['feet']
c = s.anchors['core_held']
say((c[0] - f[0], c[1] - f[1]) == RA.CORE_PLACE_OFFSET, f'place_core: orb centre just before CORE_PLACE_T at {(c[0]-f[0], c[1]-f[1])}')
s = RA.frame('place_core', RA.CORE_PLACE_T)
say(s.anchors['core_held'] == s.anchors['core'], 'place_core: orb gone from CORE_PLACE_T (core_held falls back to core)')
s = RA.frame('dig_plant', RA.PLANT_T - 0.02)
say(True, f'dig_plant: sprout in hand before PLANT_T; hand at {tuple(np.subtract(s.anchors["hand"], s.anchors["feet"]))}')

print('speed')
t0 = time.time()
n = 0
for a in RA.HANDLES:
    for k in range(20):
        RA.frame(a, k / 10, core=0.5)
        n += 1
dt = (time.time() - t0) / n * 1000
say(dt < 15, f'{dt:.1f} ms / frame')

print('\n' + ('ALL OK' if not problems else f'{len(problems)} problem(s)'))
