"""Stand-in for the c04 close-up (overgrown robot head, bird, green eye), used
only to judge the title overlay until the real shot lands.

    standin(t, eye=(200, 72)) -> (180,320,3) uint8   (t = c04 local time)
"""
import os
import sys
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from engine import *
from engine import draw, light, noise


def _bg(seed=3):
    img = canvas('leaf2')
    ys, xs = np.mgrid[0:H, 0:W]
    n = noise.fbm2(xs / 26.0, ys / 20.0, seed, 4)
    lv = (n - 0.5) * 6
    cols = ['leaf0', 'leaf1', 'leaf2', 'meadow2', 'meadow3', 'meadow4', 'meadow5']
    q = np.clip((n - 0.25) / 0.55 * (len(cols) - 1), 0, len(cols) - 1)
    q = dither_levels(q, len(cols), band=0.4)
    for i, c in enumerate(cols):
        fill(img, q == i, c)
    # sky gaps upper right
    sk = noise.fbm2(xs / 18.0, ys / 14.0, seed + 9, 3)
    fill(img, (sk > 0.66) & (ys < 70), 'sky6')
    fill(img, (sk > 0.72) & (ys < 70), 'sky7')
    return img


BG = _bg()


def standin(t, eye=(200, 72)):
    img = BG.copy()
    ex, ey = eye
    # head: big rounded block, lens on the front-right
    hx0, hy0, hx1, hy1 = ex - 110, ey - 44, ex + 34, ey + 40
    m = np.zeros((H, W), bool)
    draw._put  # noqa
    ys, xs = np.ogrid[:H, :W]
    r = 16
    inner = (xs >= hx0 + r) & (xs <= hx1 - r) & (ys >= hy0) & (ys <= hy1)
    inner |= (xs >= hx0) & (xs <= hx1) & (ys >= hy0 + r) & (ys <= hy1 - r)
    for cx, cy in [(hx0 + r, hy0 + r), (hx1 - r, hy0 + r), (hx0 + r, hy1 - r), (hx1 - r, hy1 - r)]:
        inner |= (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
    m = inner
    # body / shoulders
    bm = (xs >= ex - 130) & (xs <= ex + 40) & (ys >= hy1 + 12)
    nm = (xs >= ex - 60) & (xs <= ex - 25) & (ys >= hy1) & (ys < hy1 + 14)
    for mm, base in ((bm, 'paint3'), (nm, 'steel2'), (m, 'paint4')):
        o = draw.outline_mask(mm)
        fill(img, o, 'steel0')
        fill(img, mm, base)
    # simple shading: top-left lighter
    lv = np.clip((ex - 40 - xs) / 120.0 + (ey - ys) / 90.0, -1.5, 1.5)
    img = light.apply_light(img, lv * 1.2, mode='sun', mask=m | bm)
    # rust streaks
    for k in range(6):
        x = int(hx0 + 14 + k * 21)
        draw.vline(img, x, hy1 - 12 - (k % 3) * 4, hy1 - 1, 'rust3')
    # moss on top of head + shoulders
    mx = np.arange(hx0 - 2, hx1 + 3)
    mh = (noise.fbm1(mx / 7.0, 5) * 9).astype(int)
    for x, hh in zip(mx, mh):
        draw.vline(img, x, hy0 - hh // 2, hy0 + 2 + hh // 3, 'meadow3')
        draw.pset(img, x, hy0 - hh // 2, 'meadow5')
    for k in range(18):
        fx = int(hx0 + 5 + noise.rand(k, 1, 7) * (hx1 - hx0 - 10))
        fy = int(hy0 - 3 + noise.rand(k, 2, 7) * 5)
        c = ['bloom4', 'violet3', 'white0', 'gold3'][k % 4]
        draw.pset(img, fx, fy, c)
    # lens
    draw.disc(img, ex, ey, 22, 'steel0')
    draw.disc(img, ex, ey, 20, 'steel3')
    draw.ring(img, ex, ey, 20, 'steel5')
    draw.disc(img, ex, ey, 15, 'steel0')
    draw.disc(img, ex, ey, 14, 'ink1')
    draw.ring(img, ex, ey, 10, 'ash1')
    # green pupil (eye glow from c04 t=1.8)
    g = min(max((t - 1.8) / 0.8, 0), 1)
    if g > 0:
        draw.disc(img, ex, ey, 5, 'leaf2' if g < 0.5 else 'leaf3')
        draw.disc(img, ex, ey, 3, 'leaf4' if g < 0.5 else 'leaf5')
        draw.disc(img, ex, ey, 1, 'leaf6' if g < 0.8 else 'leaf7')
    draw.pset(img, ex - 7, ey - 8, 'white0')
    draw.pset(img, ex - 6, ey - 8, 'glass4')
    # porthole with flower
    px, py = ex - 20, hy1 + 40
    draw.disc(img, px, py, 13, 'steel0')
    draw.disc(img, px, py, 11, 'steel3')
    draw.disc(img, px, py, 8, 'bark1')
    draw.line(img, px, py, px + 2, py - 10, 'leaf4')
    draw.disc(img, px + 2, py - 12, 2, 'bloom3')
    draw.pset(img, px + 2, py - 12, 'gold3')
    # bird on head (hops)
    bx = hx0 + 60 + (4 if t > 2.6 else 0)
    by = hy0 - 6
    draw.ellipse(img, bx, by, 5, 3, 'bark3')
    draw.disc(img, bx + 5, by - 3, 2, 'bark3')
    draw.pset(img, bx + 8, by - 3, 'core4')
    draw.pset(img, bx + 6, by - 4, 'ink0')
    draw.line(img, bx - 5, by, bx - 9, by - 2, 'bark2')
    draw.ellipse(img, bx - 1, by + 1, 3, 1, 'red3')
    # dappled light drifting
    ys2, xs2 = np.mgrid[0:H, 0:W]
    d = noise.fbm2(xs2 / 22.0 + t * 0.25, ys2 / 18.0, 11, 3)
    img = light.apply_light(img, np.clip((d - 0.55) * 8, 0, 1.5), mode='sun')
    # petals drifting
    for k in range(5):
        x = (40 + k * 61 + t * 14) % W
        y = (20 + k * 33 + t * 9 + 5 * math.sin(t * 2 + k)) % H
        draw.pset(img, x, y, 'bloom4')
        draw.pset(img, x + 1, y, 'bloom5')
    return img
