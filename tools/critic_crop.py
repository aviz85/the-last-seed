"""critic_crop — crops of cached low-res film frames (review/critic/lr.npy).
usage: python tools/critic_crop.py OUT --t 9.0 9.1 ... | --f0 F --n N [--step S]
          [--crop x y w h] [--scale 4] [--cols 4]
Frames are addressed by global time (rounded to the nearest frame) or frame index."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from critic_lib import load_lr, grid

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--t', type=float, nargs='*')
ap.add_argument('--f0', type=int, default=None)
ap.add_argument('--n', type=int, default=8)
ap.add_argument('--step', type=int, default=1)
ap.add_argument('--crop', type=int, nargs=4, default=None)
ap.add_argument('--scale', type=int, default=4)
ap.add_argument('--cols', type=int, default=4)
ap.add_argument('--margin', action='store_true', help='labels in a strip above each frame')
a = ap.parse_args()
lr = load_lr()
if a.t:
    ks = [int(round(t * 30)) for t in a.t]
else:
    ks = [a.f0 + i * a.step for i in range(a.n)]
fr = []
for k in ks:
    f = lr[k]
    if a.crop:
        x, y, w, h = a.crop
        f = f[y:y + h, x:x + w]
    fr.append(f)
labels = [f'{k / 30:.3f}' for k in ks]
print(grid(fr, labels, cols=a.cols, scale=a.scale, path=a.out if a.out.endswith('.png') else a.out + '.png',
           margin=a.margin))
