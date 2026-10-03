"""CHAR-A review helpers (owned by CHAR-A).

    from tools.char_a_lib import dev_modules, strip, save
    robot, props = dev_modules()          # real chars.robot / chars.props, or the
                                          # scratch copies if CHAR_A_DEV=<dir> is set
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from PIL import Image

from engine import palette as P
from engine.core import upscale, paste
from engine import text as T


def _load_as(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def dev_modules():
    d = os.environ.get('CHAR_A_DEV')
    import chars  # noqa
    if d:
        props = _load_as('chars.props', os.path.join(d, 'props.py'))
        robot = _load_as('chars.robot', os.path.join(d, 'robot.py'))
    else:
        from chars import props, robot
    return robot, props


def solid(h, w, color):
    img = np.empty((h, w, 3), np.uint8)
    img[:] = P.C(color)
    return img


def crop_union(sprites, margin=1):
    """Crop all sprites to the union bbox of their opaque pixels (anchors kept)."""
    from engine.sprite import Sprite
    m = np.zeros(sprites[0].img.shape[:2], bool)
    for s in sprites:
        if s.img.shape[:2] == m.shape:
            m |= s.img[..., 3] > 0
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return sprites
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, m.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, m.shape[1])
    out = []
    for s in sprites:
        out.append(Sprite(s.img[y0:y1, x0:x1].copy(), {k: (x - x0, y - y0) for k, (x, y) in s.anchors.items()}))
    return out


def strip(sprites, labels=None, scale=8, bg='dust5', pad=3, grid=False, ground=None,
          label_color='white0', cols=None, crop=True):
    """Lay out sprites in a grid (each cell bottom-aligned); returns an upscaled
    RGB image with optional pixel grid; labels are drawn at 2x above each cell."""
    if crop and len({s.img.shape for s in sprites}) == 1:
        sprites = crop_union(sprites)
    n = len(sprites)
    cols = cols or n
    rows = (n + cols - 1) // cols
    cw = max(s.w for s in sprites) + pad * 2
    chh = max(s.h for s in sprites) + pad * 2
    img = solid(rows * chh, cols * cw, bg)
    for k, s in enumerate(sprites):
        r, c = divmod(k, cols)
        x0, y0 = c * cw + pad, r * chh + pad + (chh - pad * 2 - s.h)
        if ground is not None:
            gy = y0 + (s.anchor('feet')[1] if 'feet' in s.anchors else s.h - 1)
            img[gy + 1:gy + 2, c * cw:(c + 1) * cw] = P.C(ground)
        paste(img, s.img, x0, y0)
    big = upscale(img, scale)
    if grid and scale >= 6:
        big[::scale, :] = (big[::scale, :].astype(np.int32) * 7 // 8).astype(np.uint8)
        big[:, ::scale] = (big[:, ::scale].astype(np.int32) * 7 // 8).astype(np.uint8)
    if labels:
        lab_h = 18
        out = np.empty((big.shape[0] + rows * lab_h, big.shape[1], 3), np.uint8)
        out[:] = (30, 30, 36)
        for r in range(rows):
            band = solid(lab_h // 2, big.shape[1] // 2, 'ink1')
            for c in range(cols):
                k = r * cols + c
                if k < n:
                    T.text(band, str(labels[k])[:max(2, cw * scale // 12)], c * cw * scale // 2 + 2, 1, label_color)
            out[r * (chh * scale + lab_h):r * (chh * scale + lab_h) + lab_h, :band.shape[1] * 2] = upscale(band, 2)
            out[r * (chh * scale + lab_h) + lab_h:(r + 1) * (chh * scale + lab_h)] = big[r * chh * scale:(r + 1) * chh * scale]
        big = out
    return big


def stack(images, pad=6, bg=(30, 30, 36)):
    w = max(i.shape[1] for i in images)
    h = sum(i.shape[0] for i in images) + pad * (len(images) - 1)
    out = np.empty((h, w, 3), np.uint8)
    out[:] = bg
    y = 0
    for i in images:
        out[y:y + i.shape[0], :i.shape[1]] = i[..., :3]
        y += i.shape[0] + pad
    return out


def save(img, name):
    p = name if os.path.isabs(name) else os.path.join(ROOT, 'review', name)
    Image.fromarray(np.ascontiguousarray(img[..., :3])).save(p)
    return p
