"""Glyph design review for the end-title font.
usage: python tools/title_glyphs.py            -> review/title_glyphs.png
Top: each glyph big on a pixel grid. Bottom: the finished title (final colours,
t = 2.05) at 6x on black."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from PIL import Image
from engine import *
from engine import palette as P
import shots.title as TT


def glyph_panel(scale=12):
    chars = list(TT.GLYPH.keys())
    pad = 3
    w = sum(TT.GLYPH[c].shape[1] for c in chars) + pad * (len(chars) + 1)
    h = TT.CAP + pad * 2
    img = canvas('ink1', h=h, w=w)
    x = pad
    for c in chars:
        g = TT.GLYPH[c]
        gh, gw = g.shape
        img[pad:pad + gh, x:x + gw][g] = P.C('dust10')
        x += gw + pad
    big = upscale(img, scale)
    big[::scale, :, :] = (big[::scale, :, :] * 0.6 + 20).astype(np.uint8)
    big[:, ::scale, :] = (big[:, ::scale, :] * 0.6 + 20).astype(np.uint8)
    return big


def final_title(scale=6):
    TT.EYE_OVERRIDE = (4, 4)
    TT.GLINT_T = (9.0, 9.5)          # no gleam in the static record
    st = TT._static((4, 4))
    L = TT.render(1.25)['layer']
    img = canvas('ink0')
    m = L[..., 3] > 0
    img[m] = L[..., :3][m]
    TT.EYE_OVERRIDE = None
    return upscale(img[st['ay'] - 12:st['rule_y'] + 5, 60:260], scale)


def main():
    out = [glyph_panel(), final_title()]
    W_ = max(o.shape[1] for o in out)
    H_ = sum(o.shape[0] + 10 for o in out)
    sheet = np.zeros((H_, W_, 3), np.uint8)
    sheet[:] = (40, 40, 48)
    y = 0
    for o in out:
        sheet[y:y + o.shape[0], :o.shape[1]] = o
        y += o.shape[0] + 10
    p = os.path.join(ROOT, 'review', 'title_glyphs.png')
    Image.fromarray(sheet).save(p)
    print(p, sheet.shape)


if __name__ == '__main__':
    main()
