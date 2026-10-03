"""Preview the title's small sprites (sprout growth frames, sway, sparkles, eye glow)
at 12x on black.   python tools/title_sprites.py -> review/title_sprites.png"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from PIL import Image
from engine import *
import shots.title as TT


def main():
    sc = 10
    cell_w, cell_h = 26, 26
    frames = []
    # sprout growth sequence sampled at its keyframe times (+ sway)
    for tt in list(TT._SPR_T) + [TT._SPR_T[-1] + 0.8]:
        t = TT.SPROUT_T0 + tt + 1e-3
        L = layer(cell_h, cell_w)
        big = layer()
        TT._sprout_into(big, 12, 18, t)
        L[:] = big[:cell_h, :cell_w]
        frames.append(L)
    # sparkles
    for u in (0.01, 0.05, 0.1, 0.22, 0.29):
        big = layer()
        TT._sparkle_into(big, 12, 12, u, 0.0)
        frames.append(big[:cell_h, :cell_w].copy())
    # the self-drawn eye glow (fallback when c04 can't be rendered) at a few times
    for t in (0.3, 0.7, 1.1, 1.6):
        big = layer()
        TT._glow_into(big, 12, 12, t, 3.0)
        frames.append(big[:cell_h, :cell_w].copy())
    n = len(frames)
    cols = 7
    rows = (n + cols - 1) // cols
    out = canvas('ink0', h=rows * (cell_h + 1), w=cols * (cell_w + 1))
    out[:] = (40, 40, 48)
    for k, f in enumerate(frames):
        r, c = divmod(k, cols)
        cell = canvas('ink0', h=cell_h, w=cell_w)
        paste(cell, f, 0, 0)
        out[r * (cell_h + 1):r * (cell_h + 1) + cell_h, c * (cell_w + 1):c * (cell_w + 1) + cell_w] = cell
    img = upscale(out, sc)
    p = os.path.join(ROOT, 'review', 'title_sprites.png')
    Image.fromarray(img).save(p)
    print(p, img.shape)


if __name__ == '__main__':
    main()
