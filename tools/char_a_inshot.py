"""CHAR-A: stills of the robot inside the real shots (crop around the robot, 3x).
usage: python tools/char_a_inshot.py shots.s02_empty_can 0.5 1.8 2.2 ... [--full]"""
import os
import sys
import importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from engine.stage import to_lowres
from engine.core import upscale
from tools.char_a_lib import save, stack

mod = importlib.import_module(sys.argv[1])
full = '--full' in sys.argv
ts = [float(a) for a in sys.argv[2:] if not a.startswith('--')]
imgs = [to_lowres(mod.render(t)) for t in ts]
if full:
    rows = [upscale(i, 2) for i in imgs]
    out = stack(rows)
else:
    # crop: 160x90 window centred on the biggest non-background change region -> just use centre band
    tiles = [upscale(i, 3) for i in imgs]
    out = stack(tiles)
name = sys.argv[1].split('.')[-1]
print(save(out, f'char_a_inshot_{name}.png'))
