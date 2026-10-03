"""CHAR-A: tight crops around the robot inside real shots.
usage: python tools/char_a_crop.py <module> x y w h scale t1 t2 ... -> review/char_a_crop_<module>.png"""
import os
import sys
import importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from engine.stage import to_lowres
from engine.core import upscale
from engine import text as TX
from tools.char_a_lib import save, stack

mod = importlib.import_module(sys.argv[1])
x, y, w, h, sc = [int(v) for v in sys.argv[2:7]]
ts = [float(a) for a in sys.argv[7:]]
tiles = []
for t in ts:
    im = to_lowres(mod.render(t))[y:y + h, x:x + w].copy()
    TX.text(im, f'{t:.2f}', 1, 1, 'white0')
    tiles.append(upscale(im, sc))
cols = 4
rows = []
for i in range(0, len(tiles), cols):
    r = tiles[i:i + cols]
    while len(r) < cols:
        r.append(np.zeros_like(tiles[0]))
    rows.append(np.concatenate(r, axis=1))
name = sys.argv[1].split('.')[-1]
print(save(stack(rows), f'char_a_crop_{name}.png'))
