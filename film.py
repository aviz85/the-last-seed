"""
THE LAST SEED — timeline & compositor.

The film is a list of SHOTS on a global clock (seconds). Each entry plays a
shot module's render(local_t) where local_t = (t - start) + offset. A module
can appear several times (e.g. a medium shot interrupted by close-up inserts
keeps running 'off-screen' — its clock continues through the cutaway).

Transitions are ordered-dither (Bayer 8x8) dissolves on the 320x180 grid,
so they stay pixel-true.
"""
import importlib
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.core import W, H, FPS, SCALE, BAYER8, upscale
from engine.stage import to_lowres, to_hires
from engine import palette as P

DURATION = 60.0


@dataclass
class Shot:
    start: float
    end: float
    module: str
    offset: float = 0.0
    tin: Optional[Tuple[str, float]] = None     # ('fade'|'dissolve'|'white', seconds)
    tout: Optional[Tuple[str, float]] = None    # ('fade'|'white', seconds)


@dataclass
class Overlay:
    start: float
    end: float
    module: str


TIMELINE = [
    Shot(0.0, 7.0, 'shots.s01_wasteland', 0.0, tin=('fade', 1.6)),
    Shot(7.0, 8.9, 'shots.s02_empty_can', 0.0),
    Shot(8.9, 10.1, 'shots.c07_drop', 0.0),
    Shot(10.1, 12.0, 'shots.s02_empty_can', 3.1),
    Shot(12.0, 14.5, 'shots.s03_discovery', 0.0, tin=('dissolve', 0.5)),
    Shot(14.5, 16.0, 'shots.c01_eye', 0.0),
    Shot(16.0, 17.5, 'shots.c02_sprout', 0.0),
    Shot(17.5, 20.0, 'shots.s03_discovery', 5.5),
    Shot(20.0, 26.0, 'shots.s04_journey', 0.0),
    Shot(26.0, 32.0, 'shots.s05_storm', 0.0, tout=('fade', 0.35)),
    Shot(32.0, 34.3, 'shots.s06_night', 0.0, tin=('fade', 0.8)),
    Shot(34.3, 38.0, 'shots.c05_lean', 0.0, tout=('fade', 0.3)),
    Shot(38.0, 40.0, 'shots.s07_gift', 0.0, tin=('fade', 0.6)),
    Shot(40.0, 42.1, 'shots.c03_core', 0.0),
    Shot(42.1, 42.8, 'shots.s07_gift', 4.1),
    Shot(42.8, 43.6, 'shots.c08_heart', 0.0),
    Shot(43.6, 45.0, 'shots.c06_fade', 0.2),
    Shot(45.0, 48.0, 'shots.s08_bloom_int', 0.0),
    Shot(48.0, 53.0, 'shots.s08_bloom_ext', 0.0),
    Shot(53.0, 55.0, 'shots.s09_valley', 0.0, tin=('dissolve', 0.6)),
    Shot(55.0, 60.0, 'shots.c04_rest', 0.0),
]

OVERLAYS = [
    Overlay(58.0, 60.0, 'shots.title'),
]

_mods = {}


def mod(name):
    if name not in _mods:
        _mods[name] = importlib.import_module(name)
    return _mods[name]


def reload_all():
    for k in list(_mods):
        importlib.reload(_mods[k])


def shot_at(t):
    for i, s in enumerate(TIMELINE):
        if s.start <= t < s.end:
            return i, s
    return len(TIMELINE) - 1, TIMELINE[-1]


def _snap_t(x):
    """Snap a time to the exact 30 fps frame grid (kills float drift such as
    9.6 - 8.9 = 0.69999...)."""
    return int(round(x * FPS)) / FPS


def _local(s, t, clamp=True):
    m = mod(s.module)
    lt = (int(round(t * FPS)) - int(round(s.start * FPS)) + int(round(s.offset * FPS))) / FPS
    d = float(getattr(m, 'DURATION', 1e9))
    if clamp:
        lt = min(max(lt, 0.0), d - 1e-4)
    return m, lt


def _render_shot(s, t, hires, clamp=True):
    m, lt = _local(s, t, clamp)
    f = m.render(lt)
    return to_hires(f) if hires else to_lowres(f)


def _mask(alpha):
    """Low-res dither mask (H,W) bool: True = show incoming / visible."""
    th = np.tile(BAYER8, (H // 8 + 1, W // 8 + 1))[:H, :W]
    return th < alpha


def _apply_mask(a, b, alpha, hires):
    m = _mask(alpha)
    if hires:
        m = upscale(m[..., None].astype(np.uint8), SCALE)[..., 0].astype(bool)
    out = b.copy()
    out[m] = a[m]
    return out


FADE_STEPS = 8


def _darken(img, steps):
    """Palette-true stepped fade: every color walks `steps` ramp-steps toward
    black (classic console palette fade, no dither screen-door)."""
    from engine.core import to_index
    from engine import light as L
    k = int(round(steps))
    if k <= 0:
        return img
    if k >= FADE_STEPS:
        out = np.empty_like(img)
        out[:] = P.C('ink0')
        return out
    lut = L._lut('neutral', 'plain')[L.MAXL - k]
    idx = to_index(np.ascontiguousarray(img))
    return P.PAL[lut[idx]]


FADE_LEVELS = 16
_FADE_LUTS = {}


def _fade_lut(level):
    """Palette LUT for a hue-preserving fade toward black. level 0..FADE_LEVELS.
    Every color is scaled in OKLab (lightness AND chroma, hue kept) and snapped
    only to colors of its OWN ramp (+ the ink ramp), so nothing changes hue on
    the way down — the way a colourist would hand-build a console palette fade."""
    if level in _FADE_LUTS:
        return _FADE_LUTS[level]
    f = level / FADE_LEVELS
    lut = np.zeros(P.N, np.int64)
    ink = P.RAMP_IDX['ink']
    for p in range(P.N):
        r = P.RAMP_OF[p]
        cand = list(P.RAMP_IDX[r]) + list(ink)
        if r == 'white':
            cand += list(P.RAMP_IDX['dust'])
        L, a, b = P.PAL_LAB[p]
        tgt = np.array([L * (1 - f), a * (1 - f), b * (1 - f)])
        c = np.array(cand)
        d = ((P.PAL_LAB[c] - tgt) ** 2).sum(-1)
        lut[p] = c[int(d.argmin())]
    _FADE_LUTS[level] = lut
    return lut


def _fade(img, f):
    """Fade an RGB palette image toward black by fraction f (0 = none, 1 = black)."""
    from engine.core import to_index
    lv = int(round(min(max(f, 0.0), 1.0) * FADE_LEVELS))
    if lv <= 0:
        return img
    if lv >= FADE_LEVELS:
        out = np.empty_like(img)
        out[:] = P.C('ink0')
        return out
    idx = to_index(np.ascontiguousarray(img))
    return P.PAL[_fade_lut(lv)[idx]]


def _solid(color, hires):
    h, w = (H * SCALE, W * SCALE) if hires else (H, W)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = P.C(color)
    return img


def frame(t, hires=True):
    """Final composited frame at global time t."""
    i, s = shot_at(t)
    img = _render_shot(s, t, hires)

    # incoming transition
    if s.tin is not None:
        kind, d = s.tin
        a = (_snap_t(t) - s.start) / d
        if a < 1.0:
            if kind == 'fade':
                img = _fade(img, 1.0 - max(a, 0.0))
            elif kind == 'white':
                img = _apply_mask(img, _solid('white0', hires), max(a, 0.0), hires)
            elif kind == 'dissolve' and i > 0:
                prev = TIMELINE[i - 1]
                pimg = _render_shot(prev, t, hires, clamp=False)
                img = _apply_mask(img, pimg, max(a, 0.0), hires)
    # outgoing transition
    if s.tout is not None:
        kind, d = s.tout
        a = (s.end - _snap_t(t)) / d
        if a < 1.0:
            if kind == 'fade':
                img = _fade(img, 1.0 - max(a, 0.0))
            else:
                img = _apply_mask(img, _solid('white0', hires), max(a, 0.0), hires)

    # overlays (title)
    for o in OVERLAYS:
        if o.start <= t < o.end:
            om = mod(o.module)
            res = om.render((int(round(t * FPS)) - int(round(o.start * FPS))) / FPS)
            lay = res.get('layer') if isinstance(res, dict) else res
            fade = res.get('under_fade', 0.0) if isinstance(res, dict) else 0.0
            if fade > 0:
                img = _darken(img, min(fade, 1.0) * FADE_STEPS)
            if lay is not None:
                from engine.core import snap
                lay = snap(lay)
                if hires:
                    lay = upscale(lay, SCALE)
                m = lay[..., 3] > 127
                img[m] = lay[..., :3][m]

    # global end fade
    if t > DURATION - 0.5:
        a = (DURATION - t) / 0.5
        img = _fade(img, 1.0 - max(a, 0.0))
    return img
