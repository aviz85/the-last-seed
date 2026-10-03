"""CHAR-B helper: render sprites huge with a pixel grid + anchor marks (for pixel work).
usage (from project root):
    python tools/char_b_view.py              # model reference
Import: from tools.char_b_view import grid_image
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from PIL import Image
from engine import palette as P
from engine import text as T


def grid_image(sprites, scale=14, bg='dust5', labels=None, anchors=True, pad=3, gridcol=(0, 0, 0)):
    """sprites: list of Sprite. Returns an RGB uint8 array."""
    bgc = np.array(P.C(bg), np.uint8)
    ws = [s.w for s in sprites]
    hmax = max(s.h for s in sprites)
    W_ = sum(ws) + pad * (len(sprites) + 1)
    H_ = hmax + pad * 2 + 3
    img = np.zeros((H_, W_, 3), np.uint8)
    img[:] = bgc
    x = pad
    boxes = []
    for s in sprites:
        y = pad + hmax - s.h
        a = s.img[..., 3] > 127
        sub = img[y:y + s.h, x:x + s.w]
        sub[a] = s.img[..., :3][a]
        boxes.append((x, y, s))
        x += s.w + pad
    big = np.repeat(np.repeat(img, scale, 0), scale, 1)
    # grid
    for gx in range(0, big.shape[1], scale):
        big[:, gx] = (big[:, gx].astype(int) * 3 // 4).astype(np.uint8)
    for gy in range(0, big.shape[0], scale):
        big[gy, :] = (big[gy, :].astype(int) * 3 // 4).astype(np.uint8)
    if anchors:
        cols = {'feet': (255, 0, 255), 'eye': (0, 255, 255), 'core': (255, 255, 0), 'hand': (0, 255, 0),
                'can_mouth': (255, 128, 0), 'spout': (255, 0, 0), 'core_held': (255, 255, 255)}
        for (bx, by, s) in boxes:
            for k, (ax, ay) in s.anchors.items():
                if k not in cols:
                    continue
                cx = (bx + ax) * scale + scale // 2
                cy = (by + ay) * scale + scale // 2
                c = cols[k]
                r = max(2, scale // 5)
                big[max(cy - r, 0):cy + r + 1, max(cx - 1, 0):cx + 2] = c
                big[max(cy - 1, 0):cy + 2, max(cx - r, 0):cx + r + 1] = c
    if labels:
        for (bx, by, s), lab in zip(boxes, labels):
            T.text(big, str(lab), bx * scale + 2, 2, color='white1', shadow='ink0')
    return big


if __name__ == '__main__':
    from chars import model
    im = grid_image([model.IDLE, model.IDLE_DAMAGED], scale=16, labels=['IDLE', 'DAMAGED'])
    p = os.path.join(ROOT, 'review', 'char_b_model_grid.png')
    Image.fromarray(im).save(p)
    print(p)
