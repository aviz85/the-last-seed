"""Render sprites large on a checker + scene backdrop for review.
usage: python tools/spriteview.py module.attr [module.attr ...] --scale 8 --out review/x.png
attr may be a Sprite or a list of Sprites."""
import sys, os, importlib, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
from engine import *
from engine.sprite import Sprite


def collect(spec):
    mod, attr = spec.rsplit('.', 1)
    m = importlib.import_module(mod)
    v = getattr(m, attr)
    if callable(v) and not isinstance(v, Sprite):
        v = v()
    if isinstance(v, dict):
        v = list(v.values())
    return v if isinstance(v, (list, tuple)) else [v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('specs', nargs='+')
    ap.add_argument('--scale', type=int, default=8)
    ap.add_argument('--out', default='review/sprites.png')
    ap.add_argument('--bg', default='dust5')
    a = ap.parse_args()
    sprites = []
    for s in a.specs:
        sprites += collect(s)
    pad = 4
    w = sum(s.w for s in sprites) + pad * (len(sprites) + 1)
    h = max(s.h for s in sprites) + pad * 2
    img = canvas(a.bg, h=h, w=w)
    # checker behind for transparency read
    x = pad
    for s in sprites:
        blit(img, s, x, pad) if False else paste(img, s.img, x, h - pad - s.h)
        x += s.w + pad
    Image.fromarray(upscale(img, a.scale)).save(a.out)
    print(a.out)


if __name__ == '__main__':
    main()
