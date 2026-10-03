"""
THE LAST SEED - percussion & orchestral FX (timpani, taiko, toms, cymbals,
swells, heartbeat, air, rumble). Mono arrays unless noted.
"""
import numpy as np
from scipy import signal

from audio.music_dsp import (SR, ns, hz, db2a, rng_for, rcos, fade, lp, hp, bp, noise,
                             smooth_rand, peq)


# ------------------------------------------------------------- timpani
def timpani_hit(midi, vel=0.8, seed=0, length=3.5):
    """Kettle drum: near-harmonic membrane modes (air-loaded), strike pitch
    glide, a fast-dying (0,1) thud mode and felt-mallet noise."""
    rng = rng_for('timp', midi, seed, round(vel, 2))
    f = float(hz(midi))
    n = ns(length)
    t = np.arange(n) / SR
    glide = 1 + 0.018 * vel * np.exp(-t / 0.06)
    ratios = [1.0, 1.504, 1.742, 2.0, 2.245, 2.494, 2.8, 2.98]
    adb = [0, -5, -15, -9, -17, -15, -21, -23]
    T60 = [3.2, 2.0, 1.1, 1.4, 0.9, 0.8, 0.6, 0.5]
    y = np.zeros(n)
    for r, a, T in zip(ratios, adb, T60):
        ph = 2 * np.pi * np.cumsum(f * r * glide) / SR + rng.uniform(0, 6.28)
        br = vel ** (0.6 * (r - 1))
        y += db2a(a) * br * np.sin(ph) * np.exp(-6.9 * t / T)
    # (0,1) mode thud
    ph0 = 2 * np.pi * np.cumsum(f * 0.62 * (1 + 0.05 * np.exp(-t / 0.02))) / SR
    y += 0.9 * np.sin(ph0) * np.exp(-6.9 * t / 0.28)
    k = ns(0.04)
    y[:k] += 0.35 * vel * lp(rng.standard_normal(k), 700 + 2600 * vel) * np.exp(-np.arange(k) / (0.005 * SR))
    a = ns(0.0015)
    y[:a] *= rcos(a)
    return fade(y * vel, 0, 0.3)


def timpani_roll(midi, dur, v0=0.15, v1=0.9, seed=0, rate=18.0, curve=1.8, tail=2.5):
    """Roll = train of mallet strokes convolved with soft/hard stroke IRs
    (crescendo from v0 to v1 over `dur`)."""
    rng = rng_for('roll', midi, seed, dur)
    n = ns(dur + tail)
    tr_s = np.zeros(n)
    tr_h = np.zeros(n)
    t = 0.0
    while t < dur:
        v = v0 + (v1 - v0) * (t / dur) ** curve
        i = ns(t)
        amp = v * rng.uniform(0.82, 1.0)
        tr_s[i] += amp * (1 - v)
        tr_h[i] += amp * v
        t += (1.0 / rate) * (1 + 0.1 * rng.standard_normal())
    irs = timpani_hit(midi, 0.3, seed, 3.0)
    irh = timpani_hit(midi, 0.95, seed + 1, 3.0)
    y = signal.oaconvolve(tr_s, irs)[:n] + signal.oaconvolve(tr_h, irh)[:n]
    return fade(y * 0.55, 0, tail * 0.5)


# ------------------------------------------------------------- taiko / toms
def taiko(vel=0.9, f0=55.0, seed=0, decay=0.9, skin=0.6, length=1.8):
    rng = rng_for('taiko', f0, seed, round(vel, 2))
    n = ns(length)
    t = np.arange(n) / SR
    f = f0 * (1 + 1.1 * np.exp(-t / 0.025))
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-6.9 * t / decay)
    f2 = 1.58 * f0 * (1 + 0.6 * np.exp(-t / 0.02))
    y += 0.35 * np.sin(2 * np.pi * np.cumsum(f2) / SR + 1.0) * np.exp(-6.9 * t / (decay * 0.4))
    k = ns(0.12)
    tk = np.arange(k) / SR
    y[:k] += skin * vel * bp(rng.standard_normal(k), 140, 1800) * np.exp(-tk / 0.018)
    y[:k] += 0.18 * vel * hp(rng.standard_normal(k), 2500) * np.exp(-tk / 0.0025)
    a = ns(0.001)
    y[:a] *= rcos(a)
    return fade(y * vel, 0, 0.25)


def tom(vel=0.8, f0=110.0, seed=0):
    return taiko(vel, f0, seed, decay=0.45, skin=0.45, length=0.9)


# ------------------------------------------------------------- cymbals
def _metal(n, rng, count=220, lo=700.0, hi=15000.0, tmin=0.8, tmax=3.2, decay=True):
    """dense inharmonic partial cloud weighted towards 3-7 kHz (cymbal-like,
    not bell-like)."""
    t = np.arange(n) / SR
    fr = np.exp(rng.uniform(np.log(lo), np.log(hi), count))
    w = np.where(fr < 4500, (fr / 4500.0) ** 1.2, (4500.0 / fr) ** 0.8)
    am = rng.uniform(0.4, 1.0, count) * w
    T = rng.uniform(tmin, tmax, count) * (3000.0 / fr) ** 0.3
    y = np.zeros(n)
    for f, a, TT in zip(fr, am, T):
        s = np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
        y += a * (s * np.exp(-6.9 * t / TT) if decay else s)
    return y / np.sqrt(count) * 2.0


def cymbal(vel=0.8, length=4.0, seed=0, bright=1.0, soft=False):
    """Suspended cymbal / crash: HF noise wash over a dense partial cloud.
    soft=True: felt-mallet stroke (no stick click, 12 ms bloom)."""
    rng = rng_for('cym', seed, round(vel, 2))
    n = ns(length)
    t = np.arange(n) / SR
    y = _metal(n, rng) * 0.7
    y += 1.0 * bp(rng.standard_normal(n), 2800 * bright, 15000) * np.exp(-6.9 * t / 2.6)
    y += 0.35 * bp(rng.standard_normal(n), 900, 3000) * np.exp(-6.9 * t / 0.7)
    if soft:
        a = ns(0.012)
        y = lp(y, 9000, 2)
    else:
        k = ns(0.01)
        y[:k] += 1.2 * hp(rng.standard_normal(k), 3000) * np.exp(-np.arange(k) / (0.0015 * SR))
        a = ns(0.001)
    y[:a] *= rcos(a)
    return fade(y * vel * 0.35, 0, length * 0.3)


def cymbal_swell(dur, peak=0.8, seed=0, curve=2.4, tail=1.5):
    """Soft-mallet cymbal roll swelling to `dur`, then ringing out."""
    rng = rng_for('cswell', seed, dur)
    n = ns(dur + tail)
    t = np.arange(n) / SR
    body = _metal(n, rng, 160, 900, 14000, decay=False) * 0.6
    sheen = hp(rng.standard_normal(n), 3500)
    mid = bp(rng.standard_normal(n), 1200, 4500)
    x = np.clip(t / dur, 0, 1) ** curve
    rolls = 1 + 0.15 * np.sin(2 * np.pi * 16 * t) * (1 - x)
    after = np.clip(t - dur, 0, None)
    env = np.where(t < dur, x, np.exp(-6.9 * after / tail)) * rolls
    bright = np.clip(0.3 + 0.7 * env, 0, 1)
    y = (0.9 * body + 0.5 * mid * (1 - 0.5 * bright) + 0.7 * sheen * bright) * env
    return fade(y * peak * 0.4, 0.02, tail * 0.4)


def reverse_cymbal(dur, peak=0.8, seed=0):
    """Reversed crash: rises to its end point (hard stop, 6 ms)."""
    c = cymbal(1.0, max(dur + 0.3, 1.0), seed)
    r = c[::-1][-ns(dur):]
    r = fade(r, dur * 0.3, 0.006)
    return r / (np.max(np.abs(r)) + 1e-9) * peak * 0.5


# ------------------------------------------------------------- low FX
def heartbeat(vel=1.0, f=46.0, seed=0, gap=0.3):
    """'lub-dub': two felt-mallet sub thumps."""
    rng = rng_for('hb', seed)
    n = ns(gap + 1.2)
    y = np.zeros(n)
    for off, v in ((0.0, 1.0), (gap, 0.72)):
        i = ns(off)
        k = ns(0.9)
        t = np.arange(k) / SR
        fr = f * (1 + 0.7 * np.exp(-t / 0.018))
        s = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-6.9 * t / 0.42)
        s += 0.25 * lp(rng.standard_normal(k), 260) * np.exp(-t / 0.03)
        s[:ns(0.004)] *= rcos(ns(0.004))
        y[i:i + k] += v * s
    return fade(y * vel, 0, 0.2)


def rumble(dur, seed=0, peak=0.6, lo=28.0, hi=120.0, att=None, rel=0.6):
    """distant low rumble swell (stereo, decorrelated)."""
    rng = rng_for('rumble', seed, dur)
    n = ns(dur)
    out = np.zeros((2, n))
    t = np.arange(n) / SR
    att = att or dur * 0.7
    env = np.clip(t / att, 0, 1) ** 1.8 * (1 + 0.25 * smooth_rand(n, 1.2, rng))
    for ch in range(2):
        x = bp(noise(n, rng, 'brown'), lo, hi, 2)
        out[ch] = x / (np.std(x) + 1e-9)
    out *= env * peak * 0.3
    return fade(out, 0.05, rel)


def air(dur, seed=0, peak=0.3, lo=350.0, hi=2200.0, rate=0.18):
    """wind-like 'air' texture: band noise whose colour and level drift."""
    rng = rng_for('air', seed, dur)
    n = ns(dur)
    out = np.zeros((2, n))
    for ch in range(2):
        w = rng.standard_normal(n)
        b1 = bp(w, lo, lo * 2.2, 2)
        b2 = bp(w, lo * 2.2, hi, 2)
        b3 = reson(w, (lo * hi) ** 0.5 * 1.3, 90.0) * 3.0   # faint whistle
        m1 = 0.5 + 0.5 * np.tanh(smooth_rand(n, rate, rng))
        m2 = np.clip(0.25 + 0.3 * smooth_rand(n, rate * 0.7, rng), 0, 1)
        lvl = np.clip(0.75 + 0.35 * smooth_rand(n, rate, rng), 0.2, 1.4)
        out[ch] = (b1 * m1 + b2 * (1 - m1) * 0.7 + b3 * m2 * 0.25) * lvl
    out /= np.std(out) + 1e-9
    return out * peak * 0.1


def reson(x, f0, bw):
    from audio.music_dsp import reson as _r
    return _r(x, f0, bw)


def sub_boom(vel=1.0, f=41.0, length=2.5):
    """clean sine sub drop for the big downbeats."""
    n = ns(length)
    t = np.arange(n) / SR
    fr = f * (1 + 0.5 * np.exp(-t / 0.05))
    y = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-6.9 * t / length)
    y[:ns(0.003)] *= rcos(ns(0.003))
    return fade(y * vel, 0, 0.3)
