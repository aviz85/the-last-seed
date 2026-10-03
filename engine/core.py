"""
Core constants, canvases, dithering, palette snapping and compositing.

Canvas convention: numpy uint8 arrays, shape (H, W, 3) for opaque images and
(h, w, 4) for sprites/layers (alpha is BINARY: 0 or 255 — pixel art has no
soft alpha; use dithering for transparency).
Coordinates: x to the right, y DOWN, origin top-left, integer pixels.
"""
import os
import numpy as np
from . import palette as P

W, H = 320, 180          # native pixel-art resolution
FPS = 30                 # film frame rate
SCALE = 6                # 320x180 -> 1920x1080
OUT_W, OUT_H = W * SCALE, H * SCALE

# ----------------------------------------------------------------- canvases

def canvas(color='ink0', h=H, w=W):
    """New opaque canvas filled with a palette color (name or rgb)."""
    c = np.empty((h, w, 3), np.uint8)
    c[:] = P.C(color)
    return c


def layer(h=H, w=W):
    """New fully transparent RGBA layer."""
    return np.zeros((h, w, 4), np.uint8)


# ----------------------------------------------------------------- dithering
def _bayer(n):
    m = np.array([[0]])
    while m.shape[0] < n:
        m = np.block([[4 * m + 0, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m + 0.5) / (n * n)   # thresholds strictly inside (0,1)


BAYER2 = _bayer(2)
BAYER4 = _bayer(4)
BAYER8 = _bayer(8)


def bayer(h=H, w=W, n=4, ox=0, oy=0):
    """Tiled Bayer threshold map (h,w) in (0,1). ox/oy shift the pattern (use
    integer world offsets to lock dithering to a moving layer)."""
    m = {2: BAYER2, 4: BAYER4, 8: BAYER8}[n]
    ys = (np.arange(h) + oy) % n
    xs = (np.arange(w) + ox) % n
    return m[ys[:, None], xs[None, :]]


def dither_mask(alpha, n=4, ox=0, oy=0):
    """Boolean mask: True where a pixel should be 'on' for coverage alpha (0..1).
    alpha may be a scalar or an (h,w) array."""
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim == 0:
        return bayer(H, W, n, ox, oy) < alpha
    h, w = alpha.shape
    return bayer(h, w, n, ox, oy) < alpha


def dither_levels(value, levels, n=4, ox=0, oy=0, band=0.5):
    """Quantise a float field (h,w) with values in [0, levels-1] to integer
    levels with ordered dithering at the boundaries (band = width of the
    dithered zone between levels, 0..1)."""
    value = np.asarray(value, dtype=np.float64)
    h, w = value.shape
    th = bayer(h, w, n, ox, oy)
    fl = np.floor(value)
    fr = value - fl
    if band < 1.0:
        fr = np.clip((fr - 0.5) / max(band, 1e-3) + 0.5, 0, 1)
    q = (fl + (th < fr)).astype(np.int32)
    return np.clip(q, 0, levels - 1)


# ----------------------------------------------------------------- palette snap
_LUT_PATH = os.path.join(os.path.dirname(__file__), '.snap_lut_v1.npy')
_SNAP_LUT = None
_PAL_KEYS = (P.PAL[:, 0].astype(np.uint32) << 16) | (P.PAL[:, 1].astype(np.uint32) << 8) | P.PAL[:, 2]
_KEY_ORDER = np.argsort(_PAL_KEYS)
_KEYS_SORTED = _PAL_KEYS[_KEY_ORDER]


def _build_lut():
    global _SNAP_LUT
    if _SNAP_LUT is not None:
        return _SNAP_LUT
    if os.path.exists(_LUT_PATH):
        lut = np.load(_LUT_PATH)
        if lut.max() < P.N:
            _SNAP_LUT = lut
            return lut
    v = (np.arange(64) * 4 + 2).astype(np.uint8)
    r, g, b = np.meshgrid(v, v, v, indexing='ij')
    cols = np.stack([r.ravel(), g.ravel(), b.ravel()], -1)
    lut = np.empty(len(cols), np.uint8)
    step = 16384
    for i in range(0, len(cols), step):
        lab = P.linear_to_oklab(P.srgb_to_linear(cols[i:i + step]))
        d = ((lab[:, None, :] - P.PAL_LAB[None]) ** 2).sum(-1)
        lut[i:i + step] = d.argmin(-1)
    np.save(_LUT_PATH, lut)
    _SNAP_LUT = lut
    return lut


def to_index(img):
    """RGB image -> palette index image (h,w) uint8 (non-palette colors snap)."""
    lut = _build_lut()
    flat = img.reshape(-1, 3)
    key = (flat[:, 0].astype(np.uint32) << 16) | (flat[:, 1].astype(np.uint32) << 8) | flat[:, 2]
    pos = np.searchsorted(_KEYS_SORTED, key)
    pos = np.clip(pos, 0, len(_KEYS_SORTED) - 1)
    exact = _KEYS_SORTED[pos] == key
    idx = np.empty(len(flat), np.uint8)
    idx[exact] = _KEY_ORDER[pos[exact]]
    ne = ~exact
    if ne.any():
        f = flat[ne]
        bins = ((f[:, 0] >> 2).astype(np.int32) << 12) | ((f[:, 1] >> 2).astype(np.int32) << 6) | (f[:, 2] >> 2).astype(np.int32)
        idx[ne] = lut[bins]
    return idx.reshape(img.shape[:2])


def from_index(idx):
    return P.PAL[idx]


def snap(img):
    """Snap any RGB (or RGBA — alpha kept) image to the master palette."""
    if img.shape[-1] == 4:
        out = img.copy()
        out[..., :3] = from_index(to_index(np.ascontiguousarray(img[..., :3])))
        return out
    return from_index(to_index(img))


# ----------------------------------------------------------------- compositing
def paste(dst, src, x, y, mask=None):
    """Paste src (RGB or RGBA with binary alpha) onto dst at integer (x,y)
    (top-left). Clips to dst bounds. Optional extra boolean mask (src-shaped)."""
    x, y = int(round(x)), int(round(y))
    sh, sw = src.shape[:2]
    dh, dw = dst.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + sw, dw), min(y + sh, dh)
    if x0 >= x1 or y0 >= y1:
        return dst
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    d = dst[y0:y1, x0:x1]
    m = None
    if src.shape[-1] == 4:
        m = s[..., 3] > 127
    if mask is not None:
        mm = mask[y0 - y:y1 - y, x0 - x:x1 - x]
        m = mm if m is None else (m & mm)
    if dst.shape[-1] == 4:
        if m is None:
            d[..., :3] = s[..., :3]
            d[..., 3] = 255
        else:
            d[..., :3][m] = s[..., :3][m]
            d[..., 3][m] = 255
    else:
        if m is None:
            d[:] = s[..., :3]
        else:
            d[m] = s[..., :3][m]
    return dst


def over(dst_rgb, layer_rgba):
    """Composite a full-size RGBA layer over an RGB canvas (in place)."""
    m = layer_rgba[..., 3] > 127
    dst_rgb[m] = layer_rgba[..., :3][m]
    return dst_rgb


def fill(dst, mask, color):
    """Set pixels where mask is True to a palette color (name/rgb)."""
    col = P.C(color)
    if dst.shape[-1] == 4:
        dst[mask] = (*col, 255)
    else:
        dst[mask] = col
    return dst


def fill_dither(dst, alpha, color, n=4, ox=0, oy=0):
    """Dithered 'transparent' fill: alpha (h,w) float coverage 0..1."""
    m = dither_mask(alpha, n, ox, oy)
    return fill(dst, m, color)


def upscale(img, s=SCALE):
    """Nearest-neighbour integer upscale."""
    return np.repeat(np.repeat(img, s, axis=0), s, axis=1)
