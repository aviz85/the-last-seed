"""
Deterministic hash-based randomness and value noise (numpy vectorised).
Everything is a pure function of its inputs, so any frame can be rendered
independently (frames are rendered out of order / in parallel).
"""
import numpy as np

_M = np.uint32(0x9E3779B1)


def hash_u32(x, y=0, seed=0):
    """Integer hash -> uint32. x, y ints or int arrays."""
    with np.errstate(over='ignore'):
        x = np.asarray(x).astype(np.int64).astype(np.uint32)
        y = np.asarray(y).astype(np.int64).astype(np.uint32)
        h = x * np.uint32(374761393) + y * np.uint32(668265263) + np.uint32(seed) * np.uint32(2246822519)
        h = (h ^ (h >> np.uint32(13))) * np.uint32(1274126177)
        h = h ^ (h >> np.uint32(16))
        h = h * _M
        h = h ^ (h >> np.uint32(15))
    return h


def rand(x, y=0, seed=0):
    """Uniform [0,1) from integer coords."""
    return hash_u32(x, y, seed).astype(np.float64) / 4294967296.0


def rand_range(lo, hi, x, y=0, seed=0):
    return lo + (hi - lo) * rand(x, y, seed)


def _fade(t):
    return t * t * (3 - 2 * t)


def vnoise1(x, seed=0):
    """Smooth 1D value noise in [0,1)."""
    x = np.asarray(x, dtype=np.float64)
    i = np.floor(x).astype(np.int64)
    f = _fade(x - i)
    a = rand(i, 0, seed)
    b = rand(i + 1, 0, seed)
    return a + (b - a) * f


def fbm1(x, seed=0, octaves=4, lac=2.0, gain=0.5):
    x = np.asarray(x, dtype=np.float64)
    amp, tot, norm = 1.0, 0.0, 0.0
    for o in range(octaves):
        tot = tot + amp * vnoise1(x, seed + o * 101)
        norm += amp
        amp *= gain
        x = x * lac
    return tot / norm


def vnoise2(x, y, seed=0):
    """Smooth 2D value noise in [0,1). x, y float arrays (broadcastable)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    fx = _fade(x - xi)
    fy = _fade(y - yi)
    a = rand(xi, yi, seed)
    b = rand(xi + 1, yi, seed)
    c = rand(xi, yi + 1, seed)
    d = rand(xi + 1, yi + 1, seed)
    return (a + (b - a) * fx) * (1 - fy) + (c + (d - c) * fx) * fy


def fbm2(x, y, seed=0, octaves=4, lac=2.0, gain=0.5):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    amp, tot, norm = 1.0, 0.0, 0.0
    for o in range(octaves):
        tot = tot + amp * vnoise2(x, y, seed + o * 131)
        norm += amp
        amp *= gain
        x = x * lac
        y = y * lac
    return tot / norm


def grid(h, w, x0=0.0, y0=0.0, sx=1.0, sy=1.0):
    """Coordinate grids (ys, xs) for an (h,w) canvas: world = (x0 + i*sx)."""
    ys = y0 + np.arange(h)[:, None] * sy
    xs = x0 + np.arange(w)[None, :] * sx
    return np.broadcast_to(ys, (h, w)), np.broadcast_to(xs, (h, w))
