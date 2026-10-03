"""
Sprites: RGBA pixel images with named anchor points.

Author sprites as ASCII art + a legend (char -> palette color name):

    ART = '''
    ..KK..
    .KabK.
    ..KK..
    '''
    spr = from_ascii(ART, {'K': 'steel0', 'a': 'paint3', 'b': 'paint5'},
                     anchors={'feet': (3, 3)})

'.' and ' ' are transparent. Anchors are (x, y) in sprite pixel coords.
blit() places a sprite so a chosen anchor lands on a canvas position.
"""
import numpy as np
from . import palette as P
from .core import paste


class Sprite:
    __slots__ = ('img', 'anchors')

    def __init__(self, img, anchors=None):
        self.img = img                        # (h,w,4) uint8, binary alpha
        self.anchors = dict(anchors or {})
        self.anchors.setdefault('origin', (0, 0))

    @property
    def h(self):
        return self.img.shape[0]

    @property
    def w(self):
        return self.img.shape[1]

    def anchor(self, name):
        return self.anchors.get(name, self.anchors['origin'])

    def copy(self):
        return Sprite(self.img.copy(), dict(self.anchors))


def from_ascii(art, legend, anchors=None):
    lines = [ln for ln in art.strip('\n').split('\n')]
    # strip common indentation
    ind = min((len(l) - len(l.lstrip(' ')) for l in lines if l.strip()), default=0)
    lines = [l[ind:] for l in lines]
    h = len(lines)
    w = max(len(l) for l in lines)
    img = np.zeros((h, w, 4), np.uint8)
    cache = {}
    for y, l in enumerate(lines):
        for x, ch in enumerate(l):
            if ch in '. ':
                continue
            if ch not in legend:
                raise KeyError(f'sprite char {ch!r} not in legend (row {y})')
            if ch not in cache:
                cache[ch] = P.C(legend[ch])
            img[y, x, :3] = cache[ch]
            img[y, x, 3] = 255
    return Sprite(img, anchors)


def from_array(rgba, anchors=None):
    return Sprite(np.ascontiguousarray(rgba), anchors)


def flip(spr):
    """Mirror horizontally (anchors mirrored too)."""
    w = spr.w
    anc = {k: (w - 1 - x, y) for k, (x, y) in spr.anchors.items()}
    return Sprite(spr.img[:, ::-1].copy(), anc)


def blit(dst, spr, x, y, anchor='origin', flipx=False, mask=None):
    """Draw sprite so that its `anchor` lands at canvas (x, y). Returns the
    top-left (px, py) used. Works on RGB and RGBA canvases."""
    if flipx:
        spr = flip(spr)
    ax, ay = spr.anchor(anchor)
    px, py = int(round(x)) - ax, int(round(y)) - ay
    paste(dst, spr.img, px, py, mask=mask)
    return px, py


def anchor_pos(spr, x, y, anchor_at, anchor_query, flipx=False):
    """Canvas position of anchor `anchor_query` when the sprite is blitted
    with `anchor_at` placed at (x,y)."""
    if flipx:
        spr = flip(spr)
    ax, ay = spr.anchor(anchor_at)
    qx, qy = spr.anchor(anchor_query)
    return int(round(x)) - ax + qx, int(round(y)) - ay + qy


def outline(spr, color, diagonal=False):
    """New sprite with a 1px outline around opaque pixels (grows by 1px)."""
    h, w = spr.h, spr.w
    img = np.zeros((h + 2, w + 2, 4), np.uint8)
    img[1:-1, 1:-1] = spr.img
    a = img[..., 3] > 0
    o = np.zeros_like(a)
    o[1:, :] |= a[:-1, :]; o[:-1, :] |= a[1:, :]
    o[:, 1:] |= a[:, :-1]; o[:, :-1] |= a[:, 1:]
    if diagonal:
        o[1:, 1:] |= a[:-1, :-1]; o[:-1, :-1] |= a[1:, 1:]
        o[1:, :-1] |= a[:-1, 1:]; o[:-1, 1:] |= a[1:, :-1]
    o &= ~a
    img[o] = (*P.C(color), 255)
    anc = {k: (x + 1, y + 1) for k, (x, y) in spr.anchors.items()}
    return Sprite(img, anc)


def silhouette(spr, color):
    s = spr.copy()
    m = s.img[..., 3] > 0
    s.img[m, :3] = P.C(color)
    return s


def recolor(spr, mapping):
    """mapping: {from_color_name_or_rgb: to_color_name_or_rgb}"""
    s = spr.copy()
    for a, b in mapping.items():
        ca, cb = np.array(P.C(a), np.uint8), P.C(b)
        m = (s.img[..., 3] > 0) & np.all(s.img[..., :3] == ca, axis=-1)
        s.img[m, :3] = cb
    return s


def mask_of(spr):
    return spr.img[..., 3] > 0


def pick(frames, t, fps=10.0, loop=True, start=0.0):
    """Choose a frame from a list for time t playing at fps."""
    import math
    k = int(math.floor(max(0.0, t - start) * fps + 1e-6))
    n = len(frames)
    return frames[k % n] if loop else frames[min(k, n - 1)]
