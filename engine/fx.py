"""
Deterministic effects: particle fields, streaks, rays, shake.
All are pure functions of time t (no state), so frames can render in any order.
"""
import math
import numpy as np
from . import palette as P
from .noise import rand, vnoise1, fbm1
from .draw import points, line_points
from .light import apply_light
from .core import bayer


def field(t, n, seed, area, vel=(0.0, 0.0), jitter=(0.0, 0.0), sway=(0.0, 0.0), wrap=True):
    """A wrapping field of n particles drifting with velocity vel (+/- jitter,
    px/s), optional sinusoidal sway (amp_px, freq_hz). area=(x,y,w,h).
    Returns (x, y, r) arrays; r in [0,1) is a stable per-particle random for
    choosing size/color/brightness."""
    i = np.arange(n)
    ax, ay, aw, ah = area
    x0 = ax + rand(i, 1, seed) * aw
    y0 = ay + rand(i, 2, seed) * ah
    vx = vel[0] + (rand(i, 3, seed) - 0.5) * 2 * jitter[0]
    vy = vel[1] + (rand(i, 4, seed) - 0.5) * 2 * jitter[1]
    ph = rand(i, 5, seed)
    x = x0 + vx * t
    y = y0 + vy * t
    if sway[0]:
        x = x + sway[0] * np.sin(2 * math.pi * (sway[1] * t * (0.7 + 0.6 * ph) + ph))
    if wrap:
        x = ax + np.mod(x - ax, aw)
        y = ay + np.mod(y - ay, ah)
    return x, y, rand(i, 6, seed)


def draw_field(img, x, y, r, colors, weights=None):
    """Plot particles; colors chosen by r thresholds (weights = cumulative
    probability per color, default uniform)."""
    k = len(colors)
    if weights is None:
        edges = np.linspace(0, 1, k + 1)[1:]
    else:
        edges = np.cumsum(weights) / np.sum(weights)
    lo = 0.0
    for c, hi in zip(colors, edges):
        m = (r >= lo) & (r < hi)
        points(img, np.floor(x[m]), np.floor(y[m]), c)
        lo = hi


def streaks(img, x, y, dx, dy, color):
    """Short motion streaks from (x,y) to (x+dx, y+dy) (arrays or scalars)."""
    x = np.atleast_1d(x); y = np.atleast_1d(y)
    dx = np.broadcast_to(dx, x.shape); dy = np.broadcast_to(dy, x.shape)
    for a, b, c, d in zip(x, y, dx, dy):
        xs, ys = line_points(a, b, a + c, b + d)
        points(img, xs, ys, color)


def bursts(t, n, seed, origin, speed=(20, 60), life=(0.5, 1.2), gravity=0.0, spread=(0, 2 * math.pi), t0=0.0):
    """One-shot radial burst starting at time t0 from origin. Returns (x, y,
    age_frac, alive, r). age_frac in [0,1] over each particle's life."""
    i = np.arange(n)
    ang = spread[0] + rand(i, 11, seed) * (spread[1] - spread[0])
    sp = speed[0] + rand(i, 12, seed) * (speed[1] - speed[0])
    lf = life[0] + rand(i, 13, seed) * (life[1] - life[0])
    age = t - t0
    alive = (age >= 0) & (age <= lf)
    a = np.clip(age, 0, None)
    x = origin[0] + np.cos(ang) * sp * a
    y = origin[1] + np.sin(ang) * sp * a + 0.5 * gravity * a * a
    return x, y, np.clip(a / lf, 0, 1), alive, rand(i, 14, seed)


def rays_level(h, w, ox, oy, angles, widths, length, peak=2.0, gamma=1.0):
    """Light-shaft level map radiating from (ox,oy). angles in radians
    (0 = right, pi/2 = down). widths = angular half-widths (radians).
    Use with light.apply_light(img, level, 'sun')."""
    ys, xs = np.ogrid[:h, :w]
    dx = xs - ox
    dy = ys - oy
    ang = np.arctan2(dy, dx)
    dist = np.sqrt(dx * dx + dy * dy)
    lv = np.zeros((h, w))
    for a, wd in zip(angles, widths):
        da = np.abs((ang - a + math.pi) % (2 * math.pi) - math.pi)
        lv = np.maximum(lv, np.clip(1 - da / wd, 0, 1))
    fall = np.clip(1 - dist / length, 0, 1) ** gamma
    return peak * lv * fall


def shake(t, amp, seed=0, freq=18.0):
    """Camera shake offset (dx, dy) in px; amp may vary over time."""
    dx = (vnoise1(np.array([t * freq]), seed)[0] - 0.5) * 2 * amp
    dy = (vnoise1(np.array([t * freq]), seed + 7)[0] - 0.5) * 2 * amp
    return float(dx), float(dy)


def wind_gust(t, seed=0, base=1.0, gust=1.0, rate=0.6):
    """Smoothly varying wind strength."""
    return base + gust * float(fbm1(np.array([t * rate]), seed, 3)[0])
