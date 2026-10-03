"""
Palette-true lighting.

Light is expressed in STEPS: +1 = one ramp-step brighter (tinted by the
light's color), -1 = one step darker (shifted toward cool shadow). A float
'level map' is quantised with ordered dithering, so light pools get the
classic hand-dithered ring edges, and the result is ALWAYS palette colors.

    img = apply_light(img, level_map, mode='warm')
    img = light_pool(img, cx, cy, radius=40, peak=3, mode='warm')
    img = vignette(img, 1.5)

Modes (tint of positive light):
  warm  – the robot's amber core / lamp light
  sun   – golden sunlight
  cool  – grey-blue dawn / sky light
  moon  – blue night light
  green – life glow (sprout, veins of light)
  flash – white flash (impacts, lightning)
Negative levels darken toward `shadow` (default cool night blue).
"""
import numpy as np
from . import palette as P
from .core import to_index, from_index, bayer

MAXL = 8
_MODES = {
    'warm':  ('core4', 0.50, 0.060),
    'sun':   ('gold3', 0.42, 0.050),
    'cool':  ('ash7', 0.34, 0.045),
    'moon':  ('sky5', 0.36, 0.040),
    'green': ('leaf6', 0.48, 0.055),
    'flash': ('white1', 0.65, 0.120),
    'neutral': ('dust9', 0.38, 0.045),
}
_SHADOWS = {
    'night': ('night2', 0.62, 0.30),
    'warmdark': ('ochre0', 0.64, 0.25),
    'plain': ('ink1', 0.64, 0.10),
}
_LUTS = {}


def _lut(mode, shadow):
    key = (mode, shadow)
    if key in _LUTS:
        return _LUTS[key]
    tint_name, mul, add = _MODES[mode]
    sh_name, sh_mul, sh_mix = _SHADOWS[shadow]
    tint = P.srgb_to_linear(P.C(tint_name))
    tint_n = tint / max(tint.max(), 1e-6)
    sh = P.srgb_to_linear(P.C(sh_name))
    base = P.PAL_LIN
    lut = np.zeros((2 * MAXL + 1, P.N), np.uint8)
    for k in range(-MAXL, MAXL + 1):
        if k == 0:
            lut[k + MAXL] = np.arange(P.N)
            continue
        if k > 0:
            c = base * (1 + mul * k * tint_n) + add * k * tint_n
        else:
            f = sh_mul ** (-k)
            c = base * f
            lum = c.mean(-1, keepdims=True)
            shn = sh / max(sh.mean(), 1e-6)
            c = c * (1 - sh_mix) + lum * shn * sh_mix
        idx = P.nearest_index(np.clip(c, 0, 1.0))
        lut[k + MAXL] = idx
    _LUTS[key] = lut
    return lut


def apply_light(img, level, mode='warm', shadow='night', n=4, ox=0, oy=0, mask=None, band=0.45):
    """Return a new image lit by `level` (float steps, scalar or (h,w)).
    band: width (0..1) of the dithered zone between light steps. 1 = fully
    dithered smooth ramp, 0.3-0.5 = clean hand-drawn rings with dithered
    edges (default), 0 = hard posterised rings."""
    h, w = img.shape[:2]
    lv = np.asarray(level, dtype=np.float64)
    if lv.ndim == 0:
        lv = np.full((h, w), float(lv))
    fl = np.floor(lv)
    fr = lv - fl
    if band < 1.0:
        fr = np.clip((fr - 0.5) / max(band, 1e-3) + 0.5, 0, 1)
    q = (fl + (bayer(h, w, n, ox, oy) < fr)).astype(np.int32)
    q = np.clip(q, -MAXL, MAXL)
    if mask is not None:
        q = np.where(mask, q, 0)
    if not q.any():
        return img.copy()
    rgb = img[..., :3]
    idx = to_index(np.ascontiguousarray(rgb))
    lut = _lut(mode, shadow)
    out_idx = lut[q + MAXL, idx]
    out = img.copy()
    out[..., :3] = from_index(out_idx)
    return out


def radial_level(h, w, cx, cy, radius, peak=3.0, gamma=1.0, sy=1.0, core=0.0):
    """Radial level map: `peak` at center falling to 0 at `radius`.
    sy squashes vertically (sy<1 = flatter ellipse, for ground pools).
    core: radius (px) of a flat full-intensity centre."""
    ys, xs = np.ogrid[:h, :w]
    d = np.sqrt((xs - cx) ** 2 + ((ys - cy) / max(sy, 1e-6)) ** 2)
    d = np.maximum(d - core, 0)
    v = np.clip(1 - d / max(radius - core, 1e-6), 0, 1) ** gamma
    return peak * v


def light_pool(img, cx, cy, radius, peak=3.0, mode='warm', gamma=1.2, sy=1.0, core=0.0,
               shadow='night', n=4, mask=None, band=0.45):
    lv = radial_level(img.shape[0], img.shape[1], cx, cy, radius, peak, gamma, sy, core)
    return apply_light(img, lv, mode=mode, shadow=shadow, n=n, mask=mask, band=band)


def darken(img, steps=1.0, mask=None, shadow='night'):
    return apply_light(img, -abs(steps), mode='neutral', shadow=shadow, mask=mask)


def vignette(img, strength=1.6, power=2.2, shadow='night', cx=None, cy=None):
    h, w = img.shape[:2]
    cx = w / 2 if cx is None else cx
    cy = h / 2 if cy is None else cy
    ys, xs = np.ogrid[:h, :w]
    d = np.sqrt(((xs - cx) / (w / 2)) ** 2 + ((ys - cy) / (h / 2)) ** 2) / np.sqrt(2)
    return apply_light(img, -strength * np.clip(d, 0, 1) ** power * 1.4, mode='neutral', shadow=shadow)


def grade(img, steps, mode='neutral', shadow='night'):
    """Uniform grade of the whole frame by `steps` (e.g. -1 for 'dimmer')."""
    return apply_light(img, float(steps), mode=mode, shadow=shadow)
