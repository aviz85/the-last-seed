"""CHAR-A: follow the robot inside real shots — crops of consecutive frames.
usage: python tools/char_a_follow.py <shot> <t0> <n> [step_frames] [scale]"""
import os
import sys
import importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from engine.stage import to_lowres
from engine.core import upscale
from engine import text as TX
from tools.char_a_lib import save, stack

name = sys.argv[1]
t0 = float(sys.argv[2]); n = int(sys.argv[3])
step = int(sys.argv[4]) if len(sys.argv) > 4 else 3
sc = int(sys.argv[5]) if len(sys.argv) > 5 else 4
m = importlib.import_module('shots.' + name)


def where(t):
    if name == 's02_empty_can':
        return m.ROBOT_X, m.GY
    if name == 's03_discovery':
        if t < 2.5:
            return m.X0 + m.robot.travel('walk', t + m.WALK_PH) - m.robot.travel('walk', m.WALK_PH), m.GY
        return m.KNEEL_X, m.GY
    if name == 's04_journey':
        return m.SCREEN_X, m.GROUND_Y
    if name == 's05_storm':
        spr, px, py, fl = m.robot_sprite(t, False)
        ax, ay = spr.anchor('feet')
        return px + ax, py + ay
    if name == 's06_night':
        anim, ta, x, y = m.robot_pose(t)
        return x, y
    return 160, 120


tiles = []
for k in range(n):
    t = t0 + k * step / 30.0
    im = to_lowres(m.render(t))
    x, y = where(t)
    x0, y0 = int(x) - 26, int(y) - 34
    x0 = max(0, min(320 - 56, x0)); y0 = max(0, min(180 - 42, y0))
    crop = im[y0:y0 + 42, x0:x0 + 56].copy()
    TX.text(crop, f'{t:.2f}', 1, 1, 'white0')
    tiles.append(upscale(crop, sc))
cols = 6
rows = []
for i in range(0, len(tiles), cols):
    r = tiles[i:i + cols]
    while len(r) < cols:
        r.append(np.zeros_like(tiles[0]))
    rows.append(np.concatenate(r, axis=1))
print(save(stack(rows), f'char_a_follow_{name}.png'))
