"""Compare typographic variants of the end title (tracking / glyph tweaks).
python tools/title_typo.py -> review/title_typo.png"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from PIL import Image
from engine import *
from engine import text as T
import shots.title as TT

BASE = {k: v for k, v in TT._FONT.items()}
VARIANTS = {
    'current': ({}, 3, 7),
    'track4': ({}, 4, 7),
    'A_point': ({'A': """
......#......
......##.....
.....#.##....
.....#.##....
.....#.##....
....#...##...
....#...##...
....#...##...
...#.....##..
...########..
...#.....##..
..#.......##.
..#.......##.
.###.....####
"""}, 3, 7),
    'E_serif': ({'E': """
#########
.##.....#
.##......
.##......
.##......
.##...#..
.######..
.##...#..
.##......
.##......
.##......
.##......
.##.....#
#########
"""}, 3, 7),
}


def render_variant(glyphs, track, space):
    TT.GLYPH = {k: TT._parse(v) for k, v in {**BASE, **glyphs}.items()}
    TT.TRACK, TT.SPACE = track, space
    TT.TA = TT._build_title()
    TT.NL = len(TT.TA['lay'])
    TT._CACHE.clear()
    TT.EYE_OVERRIDE = (4, 4)
    TT.GLINT_T = (9.0, 9.5)          # no gleam in the static record
    st = TT._static((4, 4))
    L = TT.render(1.25)['layer']
    img = canvas('ink0')
    m = L[..., 3] > 0
    img[m] = L[..., :3][m]
    return img[st['ay'] - 12:st['rule_y'] + 5, 60:260]


rows = []
for name, (g, tr, sp) in VARIANTS.items():
    im = render_variant(g, tr, sp).copy()
    T.text(im, name, 1, 1, 'dust6')
    rows.append(np.pad(im, ((1, 1), (0, 0), (0, 0)), constant_values=40))
out = upscale(np.concatenate(rows, 0), 5)
p = os.path.join(ROOT, 'review', 'title_typo.png')
Image.fromarray(out).save(p)
print(p, out.shape)
