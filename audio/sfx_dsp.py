"""
THE LAST SEED - sound design DSP primitives (numpy / scipy only).

Everything is synthesized: noise shaping, (time-varying) filters, modal
resonators, FM / additive oscillators, stick-slip friction, granular
crackle, synthetic convolution reverbs, a look-ahead peak limiter.
All randomness goes through rng(seed) so a render is fully deterministic.
"""
import zlib

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import minimum_filter1d, uniform_filter1d

SR = 48000
TAU = 2.0 * np.pi
NYQ = SR / 2.0


# ----------------------------------------------------------------------------- basics
def n_of(t):
    return int(round(float(t) * SR))


def tvec(n):
    return np.arange(n) / SR


def rng(seed):
    if isinstance(seed, str):
        seed = zlib.crc32(seed.encode('utf8'))
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)


def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-12))


def undb(d):
    return 10.0 ** (np.asarray(d, float) / 20.0)


def normalize(x, peak=1.0):
    m = np.max(np.abs(x)) if x.size else 0.0
    return x * (peak / m) if m > 0 else x


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


# ----------------------------------------------------------------------------- noise
def noise(n, r, color='white'):
    """Unit-std noise. color: white | pink (1/f) | brown (1/f^2, HP 12 Hz) | blue."""
    x = r.standard_normal(n)
    if color == 'white':
        return x
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    f[0] = f[1]
    if color == 'pink':
        X /= np.sqrt(f)
    elif color == 'brown':
        X /= f
        X *= 1.0 / np.sqrt(1.0 + (12.0 / f) ** 4)       # 2nd-order-ish HP at 12 Hz
    elif color == 'blue':
        X *= np.sqrt(f)
    y = np.fft.irfft(X, n)
    return y / (np.std(y) + 1e-12)


def shaped_noise(n, r, mag_fn):
    """Noise with an arbitrary static magnitude response mag_fn(freqs_hz)."""
    X = np.fft.rfft(r.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / SR)
    y = np.fft.irfft(X * mag_fn(f), n)
    return y / (np.std(y) + 1e-12)


# ----------------------------------------------------------------------------- static filters
def _sos(kind, f, order):
    if kind in ('low', 'high'):
        f = float(np.clip(f, 5.0, NYQ * 0.98))
    else:
        f = [float(np.clip(f[0], 5.0, NYQ * 0.97)), float(np.clip(f[1], 6.0, NYQ * 0.98))]
    return signal.butter(order, f, kind, fs=SR, output='sos')


def lp(x, fc, order=2):
    return signal.sosfilt(_sos('low', fc, order), x, axis=-1)


def hp(x, fc, order=2):
    return signal.sosfilt(_sos('high', fc, order), x, axis=-1)


def bp(x, f1, f2, order=2):
    return signal.sosfilt(_sos('bandpass', (f1, f2), order), x, axis=-1)


def rbj(kind, f0, q=0.707, gain_db=0.0):
    """RBJ cookbook biquad -> (b, a)."""
    f0 = float(np.clip(f0, 10.0, NYQ * 0.95))
    A = 10.0 ** (gain_db / 40.0)
    w = TAU * f0 / SR
    cw, sw = np.cos(w), np.sin(w)
    al = sw / (2.0 * q)
    if kind == 'lp':
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + al, -2 * cw, 1 - al]
    elif kind == 'hp':
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + al, -2 * cw, 1 - al]
    elif kind == 'bp':                    # 0 dB peak
        b = [al, 0.0, -al]
        a = [1 + al, -2 * cw, 1 - al]
    elif kind == 'peak':
        b = [1 + al * A, -2 * cw, 1 - al * A]
        a = [1 + al / A, -2 * cw, 1 - al / A]
    elif kind == 'lowshelf':
        s = 2 * np.sqrt(A) * al
        b = [A * ((A + 1) - (A - 1) * cw + s), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - s)]
        a = [(A + 1) + (A - 1) * cw + s, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - s]
    elif kind == 'highshelf':
        s = 2 * np.sqrt(A) * al
        b = [A * ((A + 1) + (A - 1) * cw + s), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - s)]
        a = [(A + 1) - (A - 1) * cw + s, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - s]
    else:
        raise ValueError(kind)
    b = np.asarray(b, float) / a[0]
    a = np.asarray(a, float) / a[0]
    return b, a


def biquad(x, kind, f0, q=0.707, gain_db=0.0):
    b, a = rbj(kind, f0, q, gain_db)
    return signal.lfilter(b, a, x, axis=-1)


def peak_eq(x, f0, gain_db, q=1.0):
    return biquad(x, 'peak', f0, q, gain_db)


def tvf(x, f0, q=0.707, kind='bp', gain_db=0.0, block=128):
    """Time-varying RBJ biquad (coefficients updated every `block` samples,
    filter state carried across blocks). f0 / q: scalar or per-sample arrays."""
    n = len(x)
    f0 = np.broadcast_to(np.asarray(f0, float), (n,))
    q = np.broadcast_to(np.asarray(q, float), (n,))
    y = np.empty(n)
    zi = np.zeros(2)
    for s in range(0, n, block):
        e = min(n, s + block)
        m = (s + e) // 2
        b, a = rbj(kind, f0[m], q[m], gain_db)
        y[s:e], zi = signal.lfilter(b, a, x[s:e], zi=zi)
    return y


def onepole(x, fc):
    a = np.exp(-TAU * fc / SR)
    return signal.lfilter([1 - a], [1, -a], x, axis=-1)


def dc_block(x, fc=15.0):
    return hp(x, fc, 1)


def tilt(x, db_per_oct_hi, pivot=1000.0):
    """Crude spectral tilt with two shelves."""
    y = biquad(x, 'highshelf', pivot * 2, 0.6, db_per_oct_hi * 2)
    return biquad(y, 'lowshelf', pivot / 2, 0.6, -db_per_oct_hi * 2)


# ----------------------------------------------------------------------------- envelopes / curves
def env_pts(n, pts, smooth=True):
    """Envelope through (t_sec, value) points. smooth=True -> monotone cubic."""
    tp = np.array([p[0] for p in pts], float)
    vp = np.array([p[1] for p in pts], float)
    t = tvec(n)
    if smooth and len(pts) >= 3:
        y = PchipInterpolator(tp, vp, extrapolate=False)(t)
        y = np.where(t < tp[0], vp[0], y)
        y = np.where(t > tp[-1], vp[-1], y)
        return np.nan_to_num(y, nan=vp[-1])
    return np.interp(t, tp, vp)


def ar(n, attack, release, hold=0.0, curve=2.0):
    """Attack (raised-cosine) / optional hold / exponential-ish release to exactly 0."""
    t = tvec(n)
    a = max(attack, 1e-4)
    e = np.ones(n)
    ia = t < a
    e[ia] = 0.5 - 0.5 * np.cos(np.pi * t[ia] / a)
    tr = t - a - hold
    ir = tr > 0
    if release > 0:
        x = np.clip(tr[ir] / release, 0, 1)
        e[ir] = (1 - x) ** curve
    return e


def exp_dec(n, t60, attack=0.0005):
    t = tvec(n)
    e = np.exp(-6.9078 * t / max(t60, 1e-4))
    if attack > 0:
        e *= np.clip(t / attack, 0, 1) ** 0.5
    return e


def fades(x, fin=0.002, fout=0.005):
    x = np.array(x, float, copy=True)
    n = x.shape[-1]
    a = min(n_of(fin), n // 2)
    b = min(n_of(fout), n // 2)
    if a > 0:
        x[..., :a] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(a) / a)
    if b > 0:
        x[..., n - b:] *= 0.5 + 0.5 * np.cos(np.pi * (np.arange(b) + 1) / b)
    return x


def smooth_rand(n, hz, r, lo=0.0, hi=1.0):
    """Smooth random curve in [lo, hi] with ~hz control points per second."""
    k = int(n / SR * hz) + 5
    tp = (np.arange(k) - 2) / hz
    y = CubicSpline(tp, r.random(k))(tvec(n))
    return lo + (hi - lo) * np.clip(y, 0.0, 1.0)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def ramp(n, t0, t1, v0=0.0, v1=1.0, shape='smooth'):
    t = tvec(n)
    x = np.clip((t - t0) / max(t1 - t0, 1e-6), 0, 1)
    if shape == 'smooth':
        x = smoothstep(x)
    elif shape == 'cos':
        x = 0.5 - 0.5 * np.cos(np.pi * x)
    return v0 + (v1 - v0) * x


# ----------------------------------------------------------------------------- oscillators
def phase_of(f):
    return TAU * np.cumsum(np.asarray(f, float)) / SR


def sine(f, n=None, ph0=0.0):
    if np.isscalar(f):
        return np.sin(TAU * f * tvec(n) + ph0)
    return np.sin(phase_of(f) + ph0)


def additive(f, amps, ph=None, rng_=None):
    """Harmonic additive oscillator with anti-alias fade. f: Hz array; amps: list per harmonic."""
    f = np.asarray(f, float)
    base = phase_of(f)
    y = np.zeros_like(f)
    for k, a in enumerate(amps, start=1):
        if a == 0:
            continue
        g = np.clip((0.45 * SR - k * f) / (0.05 * SR), 0, 1)       # fade near Nyquist
        p0 = 0.0 if rng_ is None else rng_.random() * TAU
        y += a * g * np.sin(k * base + p0)
    return y


def fm(fc, ratio, index, ph_mod=0.0):
    """Two-operator FM. fc Hz array, index scalar/array."""
    pc = phase_of(fc)
    return np.sin(pc + index * np.sin(ratio * pc + ph_mod))


# ----------------------------------------------------------------------------- modal / impacts
def modal(dur, freqs, t60s, amps, r=None, attack=0.0003, detune=0.0):
    n = n_of(dur)
    t = tvec(n)
    y = np.zeros(n)
    for f, d, a in zip(freqs, t60s, amps):
        if f >= NYQ * 0.95 or a == 0:
            continue
        if r is not None and detune:
            f = f * (1 + detune * r.uniform(-1, 1))
        ph = 0.0 if r is None else r.random() * TAU
        y += a * np.sin(TAU * f * t + ph) * np.exp(-6.9078 * t / d)
    if attack > 0:
        y *= np.clip(t / attack, 0, 1)
    return y


def resonate(x, freqs, t60s, gains):
    """Excite a bank of 2-pole resonators with signal x (unit-amplitude modes)."""
    y = np.zeros_like(x)
    for f, d, g in zip(freqs, t60s, gains):
        if f >= NYQ * 0.95:
            continue
        rr = np.exp(-6.9078 / (d * SR))
        w = TAU * f / SR
        y += signal.lfilter([g * np.sin(w)], [1.0, -2 * rr * np.cos(w), rr * rr], x)
    return y


def impulse_train(n, rate, r, jitter=0.1, amp_sigma=0.3):
    """Irregular impulses at a (time-varying) rate in Hz -> sparse array."""
    rate = np.broadcast_to(np.asarray(rate, float), (n,))
    ph = np.cumsum(rate * (1 + jitter * r.standard_normal(n) * 0.5).clip(0.1, None)) / SR
    idx = np.nonzero(np.diff(np.floor(ph)) > 0)[0] + 1
    x = np.zeros(n)
    x[idx] = np.minimum(r.lognormal(0, amp_sigma, len(idx)), np.exp(2.1 * amp_sigma))
    return x


def poisson_impulses(n, rate, r, amp_sigma=0.6, signed=True):
    rate = np.broadcast_to(np.asarray(rate, float), (n,))
    hit = r.random(n) < (rate / SR)
    x = np.zeros(n)
    k = int(hit.sum())
    if k:
        # log-normal grain sizes, tail capped (~98th pct) so no rare grain pops 25 dB out of a texture
        a = np.minimum(r.lognormal(0, amp_sigma, k), np.exp(2.1 * amp_sigma))
        if signed:
            a *= r.choice([-1.0, 1.0], k)
        x[hit] = a
    return x


def crackle(n, rate, r, band=(1500, 8000), grain_ms=1.5, amp_sigma=0.7, order=2):
    """Granular crackle / grit / rustle texture (normalized to ~unit RMS at full rate)."""
    x = poisson_impulses(n, rate, r, amp_sigma)
    a = np.exp(-1.0 / (max(grain_ms, 0.05) * 1e-3 * SR))
    x = signal.lfilter([1.0], [1.0, -a], x * r.standard_normal(n).clip(-3, 3) * 0.5 + x)
    y = bp(x, band[0], band[1], order)
    return y


# ----------------------------------------------------------------------------- stick-slip friction
def friction(dur, rate, modes, r, jitter=0.15, grit=0.3, amp_sigma=0.35, env=None):
    """Creak / squeak / groan: irregular slip impulses at `rate` (Hz, array or scalar)
    exciting resonant `modes` = (freqs, t60s, gains)."""
    n = n_of(dur)
    x = impulse_train(n, rate, r, jitter, amp_sigma)
    if grit > 0:
        # each slip also releases a tiny burst of friction noise
        k = np.exp(-tvec(n_of(0.002)) / 0.0004)
        g = signal.fftconvolve(x, k)[:n] * r.standard_normal(n)
        x = x + grit * g
    if env is not None:
        x = x * env
    return resonate(x, *modes)


# ----------------------------------------------------------------------------- stereo helpers
def pan(x, p):
    """Mono -> stereo (2, n), constant-power. p in [-1, 1] (scalar or array)."""
    th = (np.clip(np.asarray(p, float), -1, 1) + 1) * (np.pi / 4)
    return np.stack([x * np.cos(th), x * np.sin(th)])


def width(st, w):
    """Mid/side width: 0 = mono, 1 = unchanged, >1 wider. w scalar/array."""
    m = 0.5 * (st[0] + st[1])
    s = 0.5 * (st[0] - st[1]) * w
    return np.stack([m + s, m - s])


def balance(st, p):
    """Balance a stereo signal toward p (-1 left .. +1 right), keeps width."""
    th = (np.clip(np.asarray(p, float), -1, 1) + 1) * (np.pi / 4)
    return np.stack([st[0] * np.cos(th) * np.sqrt(2), st[1] * np.sin(th) * np.sqrt(2)])


def decorrelate(x, r, amount=1.0, taps=12, spread_ms=8.0):
    """Mono -> stereo via two different sparse all-pass-ish tap sets (subtle width)."""
    n = len(x)
    out = []
    for ch in range(2):
        k = np.zeros(n_of(spread_ms / 1000) + 2)
        k[0] = 1.0
        for _ in range(taps):
            k[r.integers(1, len(k))] += r.normal(0, 0.25 * amount)
        k /= np.sqrt(np.sum(k * k))
        out.append(signal.fftconvolve(x, k)[:n])
    return np.stack(out)


# ----------------------------------------------------------------------------- reverb
def make_ir(dur, t60=(1.5, 1.2, 0.8), xover=(350.0, 3000.0), predelay=0.01,
            er=(), density=None, lp_hz=12000.0, hp_hz=80.0, onset=0.008,
            metal=None, seed='ir', diffuse_gain=1.0):
    """Synthetic stereo impulse response.
    t60 = (low, mid, high) decay times; density=None -> gaussian (dense),
    else velvet noise with `density` impulses/s (sparse, grainy).
    er = [(time_s, gain, pan)] discrete early reflections (pan -1..1).
    metal = (count, fmin, fmax, t60min, t60max, level) adds ringing modes."""
    n = n_of(dur)
    t = tvec(n)
    chans = []
    for ch in range(2):
        r = rng(f'{seed}-{ch}')
        bands = []
        for bi in range(3):
            if density is None:
                x = r.standard_normal(n)
            else:
                x = np.zeros(n)
                k = int(density * dur)
                idx = r.integers(0, n, k)
                x[idx] = r.choice([-1.0, 1.0], k) * r.uniform(0.5, 1.0, k)
                x *= np.sqrt(SR / max(density, 1))
            if bi == 0:
                x = lp(x, xover[0], 2)
            elif bi == 1:
                x = bp(x, xover[0], xover[1], 1)
            else:
                x = hp(x, xover[1], 2)
            bands.append(x * np.exp(-6.9078 * t / t60[bi]))
        y = sum(bands) * diffuse_gain
        y *= 1.0 - np.exp(-t / max(onset, 1e-4))            # diffuse build-up
        if metal is not None:
            cnt, f0, f1, d0, d1, lvl = metal
            fr = np.exp(r.uniform(np.log(f0), np.log(f1), cnt))
            y += lvl * modal(dur, fr, r.uniform(d0, d1, cnt), r.uniform(0.3, 1.0, cnt) / np.sqrt(cnt), r)
        for (tt, g, p) in er:
            i = n_of(tt)
            if i < n:
                side = np.cos((p + 1) * np.pi / 4) if ch == 0 else np.sin((p + 1) * np.pi / 4)
                y[i] += g * side * np.sqrt(2) * np.sqrt(SR) * 0.02
        y = lp(y, lp_hz, 2)
        y = hp(y, hp_hz, 1)
        pd = n_of(predelay)
        y = np.concatenate([np.zeros(pd), y])[:n]
        chans.append(y)
    ir = np.stack(chans)
    ir /= np.sqrt(np.sum(ir ** 2) / 2.0)
    return ir


def convolve_stereo(send, ir, cross=0.3):
    """Stereo send (2,n) through stereo IR (2,m). Keeps some L/R image, decorrelated tail."""
    n = send.shape[1]
    inL = send[0] * (1 - cross) + send[1] * cross
    inR = send[1] * (1 - cross) + send[0] * cross
    outL = signal.oaconvolve(inL, ir[0])[:n]
    outR = signal.oaconvolve(inR, ir[1])[:n]
    return np.stack([outL, outR])


# ----------------------------------------------------------------------------- dynamics / master
def soft_clip(x, drive=1.0):
    return np.tanh(drive * x) / np.tanh(drive)


def limiter_gain(st, ceiling_db=-1.3, look_ms=4.0, release_ms=80.0, os=4):
    """Look-ahead TRUE-PEAK limiter gain curve (1-D, same length). Multiply both channels by it.
    Peaks are detected on a 4x-oversampled copy, so inter-sample overs are caught too."""
    if os > 1:
        n = st.shape[1]
        up = signal.resample_poly(st, os, 1, axis=-1)
        pk = np.max(np.abs(up), axis=0)[:n * os].reshape(n, os).max(1)
        pk = np.maximum(pk, np.max(np.abs(st), axis=0))
    else:
        pk = np.max(np.abs(st), axis=0)
    c = undb(ceiling_db)
    g = np.minimum(1.0, c / np.maximum(pk, 1e-12))
    if np.all(g >= 1.0):
        return np.ones_like(g)
    L = max(1, n_of(look_ms / 1000))
    g1 = minimum_filter1d(g, size=2 * L + 1, mode='nearest')
    R = max(1, n_of(release_ms / 1000))
    # hold the reduction forward in time for R samples (window [i-R, i])
    g2 = minimum_filter1d(g1, size=R + 1, origin=R // 2, mode='nearest')
    g3 = uniform_filter1d(g2, size=L + 1, mode='nearest')
    g3 = uniform_filter1d(g3, size=L + 1, mode='nearest')
    return np.minimum(g3, g1)


def true_peak_db(st, os=4):
    y = signal.resample_poly(st, os, 1, axis=-1)
    return float(db(np.max(np.abs(y))))
