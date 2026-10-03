"""
Stage: layered frame with SUB-PIXEL layer positions (smooth camera moves).

A shot's render(t) may return either
  (a) an (H, W, 3) uint8 image  — everything on the pixel grid, or
  (b) a Stage                   — layers composited at 1920x1080 with each
                                  layer offset by fractions of a pixel, so
                                  slow pans/parallax glide instead of judder.

    st = Stage(bg='dust3')
    st.add(sky_rgb, 0, 0)                      # opaque full-screen layer
    st.add(mountains_rgba, -cam_x * 0.3, 20)   # float positions (low-res px)
    st.add(ground_rgba, -cam_x, 110)
    return st

Rules:
  * Each layer is an RGB (opaque) or RGBA (binary alpha) low-res image.
  * Positions are the layer's top-left, in low-res pixels, floats allowed.
    Sub-pixel positions are quantised to 1/3 px (2 output px) on output.
  * Do per-layer lighting/dithering yourself BEFORE adding (palette-true).
  * Whole-frame camera shake: st.shake = (dx, dy) in low-res px (floats);
    give layers some overscan margin if you shake.
"""
import numpy as np
from .core import W, H, SCALE, snap, upscale
from . import palette as P


class Stage:
    def __init__(self, bg='ink0'):
        self.bg = P.C(bg)
        self.layers = []
        self.shake = (0.0, 0.0)

    def add(self, img, x=0.0, y=0.0):
        self.layers.append((img, float(x), float(y)))
        return self

    # --------------------------------------------------------- low-res flatten
    def lowres(self):
        out = np.empty((H, W, 3), np.uint8)
        out[:] = self.bg
        sx, sy = self.shake
        for img, x, y in self.layers:
            _paste_lr(out, img, int(round(x + sx)), int(round(y + sy)))
        return out

    # --------------------------------------------------------- hi-res composite
    def hires(self, scale=SCALE):
        OH, OW = H * scale, W * scale
        out = np.empty((OH, OW, 3), np.uint8)
        out[:] = self.bg
        sx, sy = self.shake
        q = 2 if scale % 2 == 0 else 1
        for img, x, y in self.layers:
            X = int(round((x + sx) * scale / q)) * q      # output-pixel position
            Y = int(round((y + sy) * scale / q)) * q
            lh, lw = img.shape[:2]
            # visible low-res source rect
            c0 = max(0, (-X) // scale)
            r0 = max(0, (-Y) // scale)
            c1 = min(lw, -(-(OW - X) // scale))
            r1 = min(lh, -(-(OH - Y) // scale))
            if c0 >= c1 or r0 >= r1:
                continue
            crop = img[r0:r1, c0:c1]
            crop = snap(crop)
            big = upscale(crop, scale)
            bx, by = X + c0 * scale, Y + r0 * scale
            x0, y0 = max(bx, 0), max(by, 0)
            x1, y1 = min(bx + big.shape[1], OW), min(by + big.shape[0], OH)
            if x0 >= x1 or y0 >= y1:
                continue
            s = big[y0 - by:y1 - by, x0 - bx:x1 - bx]
            if s.shape[-1] == 4:
                m = s[..., 3] > 127
                out[y0:y1, x0:x1][m] = s[..., :3][m]
            else:
                out[y0:y1, x0:x1] = s
        return out


def _paste_lr(dst, img, x, y):
    lh, lw = img.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + lw, W), min(y + lh, H)
    if x0 >= x1 or y0 >= y1:
        return
    s = img[y0 - y:y1 - y, x0 - x:x1 - x]
    if s.shape[-1] == 4:
        m = s[..., 3] > 127
        dst[y0:y1, x0:x1][m] = s[..., :3][m]
    else:
        dst[y0:y1, x0:x1] = s


def to_lowres(frame):
    """Shot output -> (H,W,3) palette-snapped low-res image."""
    if isinstance(frame, Stage):
        return snap(frame.lowres())
    return snap(np.ascontiguousarray(frame[..., :3]))


def to_hires(frame, scale=SCALE):
    """Shot output -> (H*scale, W*scale, 3) final image."""
    if isinstance(frame, Stage):
        return frame.hires(scale)
    return upscale(snap(np.ascontiguousarray(frame[..., :3])), scale)
