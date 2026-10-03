"""Timing helpers: easing, remapping, keyframes."""
import math
import numpy as np


def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def lerp(a, b, t):
    return a + (b - a) * t


def remap(x, a, b, c=0.0, d=1.0, clip=True):
    """Map x from [a,b] to [c,d]."""
    t = (x - a) / (b - a) if b != a else 0.0
    if clip:
        t = clamp(t)
    return c + (d - c) * t


def prog(t, start, end):
    """Normalised progress of t through [start,end], clamped to 0..1."""
    return clamp((t - start) / (end - start)) if end > start else float(t >= start)


def smoothstep(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)


def smootherstep(t):
    t = clamp(t)
    return t * t * t * (t * (t * 6 - 15) + 10)


def in_quad(t): t = clamp(t); return t * t
def out_quad(t): t = clamp(t); return 1 - (1 - t) * (1 - t)
def in_out_quad(t): t = clamp(t); return 2 * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2
def in_cubic(t): t = clamp(t); return t ** 3
def out_cubic(t): t = clamp(t); return 1 - (1 - t) ** 3
def in_out_cubic(t): t = clamp(t); return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2
def in_out_sine(t): t = clamp(t); return -(math.cos(math.pi * t) - 1) / 2
def out_expo(t): t = clamp(t); return 1.0 if t >= 1 else 1 - 2 ** (-10 * t)
def in_expo(t): t = clamp(t); return 0.0 if t <= 0 else 2 ** (10 * t - 10)


def out_back(t, s=1.70158):
    t = clamp(t) - 1
    return t * t * ((s + 1) * t + s) + 1


def out_elastic(t):
    t = clamp(t)
    if t in (0.0, 1.0):
        return t
    return 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * (2 * math.pi / 3)) + 1


def keyframes(t, keys, ease=smoothstep):
    """Piecewise interpolation. keys: list of (time, value) sorted by time.
    value may be a number or tuple. ease: easing fn applied per segment
    (or a list of fns, one per segment)."""
    if t <= keys[0][0]:
        return keys[0][1]
    if t >= keys[-1][0]:
        return keys[-1][1]
    for i in range(len(keys) - 1):
        t0, v0 = keys[i]
        t1, v1 = keys[i + 1]
        if t0 <= t <= t1:
            e = ease[i] if isinstance(ease, (list, tuple)) else ease
            u = e((t - t0) / (t1 - t0)) if t1 > t0 else 1.0
            if isinstance(v0, (tuple, list)):
                return tuple(a + (b - a) * u for a, b in zip(v0, v1))
            return v0 + (v1 - v0) * u
    return keys[-1][1]


def frame_index(t, fps, n_frames, loop=True, start=0.0):
    """Which frame of an n-frame animation playing at `fps` to show at time t."""
    k = int(math.floor(max(0.0, t - start) * fps + 1e-6))
    return k % n_frames if loop else min(k, n_frames - 1)


def step_time(t, fps):
    """Quantise time to an animation rate (e.g. 12fps 'on twos' feel)."""
    return math.floor(t * fps + 1e-6) / fps


def pulse(t, period, duty=0.5):
    return ((t / period) % 1.0) < duty


def flicker(t, seed=0, rate=18.0):
    """Deterministic irregular flicker in [0,1] (for failing lights)."""
    from .noise import vnoise1
    return float(vnoise1(np.array([t * rate]), seed)[0])
