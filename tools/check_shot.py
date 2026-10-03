"""Validate a shot module against the contract.
usage: python tools/check_shot.py shots.s05_storm
Checks: DURATION vs shot list, output type/shape, determinism, palette discipline, speed."""
import sys, os, time, importlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from engine import palette as P
from engine.stage import Stage, to_lowres
import film

name = sys.argv[1]
m = importlib.import_module(name)
d = float(m.DURATION)
exp = None
for s in film.TIMELINE:
    if s.module == name:
        exp = max(exp or 0, s.offset + (s.end - s.start))
print(f'{name}: DURATION={d}  timeline needs >= {exp}')
if exp is not None and d + 1e-6 < exp:
    print('  !! DURATION shorter than the timeline needs')

keys = set(((P.PAL[:, 0].astype(int) << 16) | (P.PAL[:, 1].astype(int) << 8) | P.PAL[:, 2]).tolist())
ts = np.linspace(0, d - 1 / 30, 10).tolist() + [d + 0.5]
worst = 0.0
tt = 0.0
for t in ts:
    t0 = time.time()
    f = m.render(t)
    tt += time.time() - t0
    if isinstance(f, Stage):
        raw = f.lowres()
    else:
        assert f.dtype == np.uint8 and f.shape[:2] == (180, 320), f'bad frame shape {f.shape} {f.dtype}'
        raw = f[..., :3]
    k = ((raw[..., 0].astype(int) << 16) | (raw[..., 1].astype(int) << 8) | raw[..., 2]).ravel()
    off = np.mean([v not in keys for v in k[::7]])
    worst = max(worst, off)
f1 = to_lowres(m.render(d * 0.5)); f2 = to_lowres(m.render(d * 0.5))
print(f'  deterministic: {np.array_equal(f1, f2)}')
print(f'  off-palette pixels (before snap), worst frame: {worst * 100:.1f}%  (aim for ~0%)')
print(f'  speed: {tt / len(ts) * 1000:.0f} ms/frame (budget 150)')
