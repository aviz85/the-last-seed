"""Film stills at given global times, low-res composited, upscaled.
usage: python tools/stills.py 34.0 35.0 --scale 3 [--grid 2] [--tag x]"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
import film
from engine.core import upscale, W, H
from engine import text as T

ap = argparse.ArgumentParser()
ap.add_argument('times', type=float, nargs='+')
ap.add_argument('--scale', type=int, default=3)
ap.add_argument('--grid', type=int, default=0, help='combine into a grid with N columns')
ap.add_argument('--tag', default='still')
ap.add_argument('--hires', action='store_true')
ap.add_argument('--crop', type=int, nargs=4, default=None, help='x y w h in low-res px')
a = ap.parse_args()
imgs = []
for t in a.times:
    if a.hires:
        f = film.frame(t, hires=True)
        if a.crop:
            x, y, w, h = a.crop
            f = f[y * 6:(y + h) * 6, x * 6:(x + w) * 6]
        img = f
    else:
        f = film.frame(t, hires=False).copy()
        if a.crop:
            x, y, w, h = a.crop
            f = f[y:y + h, x:x + w]
        T.text(f, f'{t:.2f}', 2, 2, 'white1', shadow='ink0')
        img = upscale(f, a.scale)
    imgs.append(img)
if a.grid:
    cols = a.grid
    rows = (len(imgs) + cols - 1) // cols
    h, w = imgs[0].shape[:2]
    out = np.zeros((rows * (h + 4), cols * (w + 4), 3), np.uint8)
    for k, im in enumerate(imgs):
        r, c = divmod(k, cols)
        out[r * (h + 4):r * (h + 4) + h, c * (w + 4):c * (w + 4) + w] = im
    p = f'review/{a.tag}_grid.png'
    Image.fromarray(out).save(p)
    print(p)
else:
    for t, im in zip(a.times, imgs):
        p = f'review/{a.tag}_{t:05.2f}.png'
        Image.fromarray(im).save(p)
        print(p)
