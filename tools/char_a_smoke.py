"""CHAR-A smoke test: every anim x can x variant x scale x time renders, with
all contract anchors, palette-only colours and binary alpha.
usage: [CHAR_A_DEV=dir] python tools/char_a_smoke.py"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tools.char_a_lib import dev_modules
from engine import palette as P

robot, props = dev_modules()
KEYS = set(((P.PAL[:, 0].astype(int) << 16) | (P.PAL[:, 1].astype(int) << 8) | P.PAL[:, 2]).tolist())
MINE = [a for a in robot.ANIMS if a not in ('sit', 'sit_idle', 'sit_look_can', 'sit_look_chest', 'dig_plant',
                                               'open_chest', 'place_core', 'power_down', 'dead_sit')]
need = ('feet', 'eye', 'core', 'hand', 'can_mouth', 'spout', 'core_held')
n = 0
t0 = time.time()
bad = 0
for a in MINE:
    for scale in robot.ANIMS[a][2]:
        for can in (None, 'empty', 'sprout', 'sprout_wilt', 'sprout_lean'):
            for var in ('normal', 'damaged'):
                for t in np.linspace(0, robot.duration(a) * 2.2, 23):
                    for core, eye in ((1.0, 1.0), (0.3, 0.5), (0.0, 0.0)):
                        s = robot.frame(a, float(t), can=can, variant=var, scale=scale, core=core, eye=eye)
                        n += 1
                        miss = [k for k in need if k not in s.anchors]
                        al = s.img[..., 3]
                        if miss or not np.all((al == 0) | (al == 255)):
                            bad += 1
                            print('BAD', a, scale, can, var, t, miss)
                        m = al > 0
                        rgb = s.img[..., :3][m].astype(int)
                        k = set(((rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]).tolist())
                        if not k <= KEYS:
                            bad += 1
                            print('OFF-PALETTE', a, scale, can, t)
                        fl = robot.frame(a, float(t), can=can, variant=var, scale=scale, face_left=True)
dt = time.time() - t0
for fn in ('sprout', 'can'):
    pass
for st in ('ground', 'wilt', 'lean', 'planted'):
    props.sprout(st, 0.3, lean=0.5, wind=0.5)
for st in ('empty', 'sprout', 'sprout_wilt', 'sprout_lean'):
    c = props.can(st)
    assert 'base' in c.anchors and 'mouth' in c.anchors
print(f'{n} frames OK ({bad} problems), {dt / n * 1000:.2f} ms/frame avg')
