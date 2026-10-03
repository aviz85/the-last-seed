"""
THE LAST SEED - score DSP toolkit (numpy / scipy only).

Filters, envelopes, stereo buses, synthetic convolution reverb, SNES-style
echo, look-ahead limiter and ITU-R BS.1770 loudness metering.
"""
import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d

SR = 48000
_NOTE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


# ----------------------------------------------------------------- basics
def m(name):
    """'C#5' / 'Bb3' / 'D5' -> midi number (C4 = 60). ints pass through."""
    if not isinstance(name, str):
        return name
    s = name.strip()
    v = _NOTE[s[0].upper()]
    i = 1
    while i < len(s) and s[i] in '#b':
        v += 1 if s[i] == '#' else -1
        i += 1
    return 12 * (int(s[i:]) + 1) + v


def hz(midi):
    return 440.0 * 2.0 ** ((np.asarray(midi, dtype=float) - 69.0) / 12.0)


def ns(sec):
    return int(round(sec * SR))


def db2a(d):
    return 10.0 ** (np.asarray(d, dtype=float) / 20.0)


def a2db(a):
    return 20.0 * np.log10(np.maximum(np.abs(a), 1e-12))


def cents(c):
    return 2.0 ** (np.asarray(c, dtype=float) / 1200.0)


def rng_for(*keys):
    """Deterministic RNG from arbitrary hashable keys."""
    h = 1469598103934665603
    for k in keys:
        for ch in repr(k):
            h = ((h ^ ord(ch)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return np.random.default_rng(h)


# ----------------------------------------------------------------- filters
def _cl(f):
    return float(np.clip(f, 10.0, SR * 0.47))


def lp(x, fc, order=2):
    return signal.sosfilt(signal.butter(order, _cl(fc), 'low', fs=SR, output='sos'), x, axis=-1)


def hp(x, fc, order=2):
    return signal.sosfilt(signal.butter(order, _cl(fc), 'high', fs=SR, output='sos'), x, axis=-1)


def bp(x, lo, hi, order=2):
    return signal.sosfilt(signal.butter(order, [_cl(lo), _cl(hi)], 'band', fs=SR, output='sos'), x, axis=-1)


def _rbj(kind, f0, gain_db=0.0, q=0.707):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * _cl(f0) / SR
    cw, sw = np.cos(w0), np.sin(w0)
    al = sw / (2 * q)
    if kind == 'peak':
        b = [1 + al * A, -2 * cw, 1 - al * A]
        a = [1 + al / A, -2 * cw, 1 - al / A]
    elif kind == 'lowshelf':
        sa = 2 * np.sqrt(A) * al
        b = [A * ((A + 1) - (A - 1) * cw + sa), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sa)]
        a = [(A + 1) + (A - 1) * cw + sa, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sa]
    elif kind == 'highshelf':
        sa = 2 * np.sqrt(A) * al
        b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sa)]
        a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    elif kind == 'bandpass':   # constant 0 dB peak gain
        b = [al, 0.0, -al]
        a = [1 + al, -2 * cw, 1 - al]
    else:
        raise ValueError(kind)
    b = np.array(b) / a[0]
    a = np.array(a) / a[0]
    return b, a


def peq(x, f0, gain_db, q=1.0):
    b, a = _rbj('peak', f0, gain_db, q)
    return signal.lfilter(b, a, x, axis=-1)


def shelf(x, f0, gain_db, kind='low', q=0.707):
    b, a = _rbj('lowshelf' if kind == 'low' else 'highshelf', f0, gain_db, q)
    return signal.lfilter(b, a, x, axis=-1)


def reson(x, f0, bw):
    """2-pole resonator (bandpass, 0 dB peak) with bandwidth bw Hz."""
    b, a = _rbj('bandpass', f0, 0.0, max(f0 / max(bw, 1.0), 0.3))
    return signal.lfilter(b, a, x, axis=-1)


def onepole_lp(x, fc):
    a = np.exp(-2 * np.pi * fc / SR)
    return signal.lfilter([1 - a], [1, -a], x, axis=-1)


# ----------------------------------------------------------------- shapes
def rcos(n):
    """raised-cosine ramp 0->1 over n samples."""
    if n <= 0:
        return np.ones(0)
    return 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)


def fade(x, fin=0.0, fout=0.0):
    """raised-cosine fade in/out (seconds) on last axis, in place-safe copy."""
    y = np.array(x, dtype=float, copy=True)
    n = y.shape[-1]
    a, b = min(ns(fin), n), min(ns(fout), n)
    if a > 0:
        y[..., :a] *= rcos(a)
    if b > 0:
        y[..., n - b:] *= rcos(b)[::-1]
    return y


def env_asr(n, att, hold, rel, curve=1.0, att_shape='cos'):
    """Attack (s) -> sustain level 1 until `hold` (s from start) -> release (s).
    Smooth (raised-cosine / exponential) segments; length n samples."""
    e = np.zeros(n)
    a = max(ns(att), 1)
    h = max(ns(hold), a)
    r = max(ns(rel), 1)
    ia = min(a, n)
    if att_shape == 'cos':
        e[:ia] = rcos(a)[:ia]
    else:  # exponential-ish swell
        e[:ia] = (np.linspace(0, 1, a) ** 2.2)[:ia]
    if h > a:
        e[a:min(h, n)] = 1.0
    if h < n:
        m_ = min(r, n - h)
        lvl = e[h - 1] if h > 0 else 1.0
        # exponential release that reaches ~-60 dB at r, then cos to 0
        t = np.arange(m_) / r
        rel_curve = np.exp(-6.9 * t * curve) * (0.5 + 0.5 * np.cos(np.pi * t))
        e[h:h + m_] = lvl * rel_curve
    return e


def pan_gains(p):
    """equal-power pan, p in [-1, 1] -> (gL, gR)."""
    th = (np.clip(p, -1, 1) + 1) * np.pi / 4
    return np.cos(th), np.sin(th)


def noise(n, rng, color='white'):
    w = rng.standard_normal(n)
    if color == 'white':
        return w
    if color == 'pink':   # Paul Kellet's refined approximation via filter
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        return signal.lfilter(b, a, w) * 8.0
    if color == 'brown':
        y = signal.lfilter([1.0], [1.0, -0.995], w)
        return y / (np.std(y) + 1e-9)
    raise ValueError(color)


def smooth_rand(n, rate_hz, rng, order=2):
    """band-limited random curve ~N(0,1) varying at about rate_hz."""
    k = max(int(n * rate_hz / SR) + 4, 4)
    pts = rng.standard_normal(k)
    x = np.linspace(0, k - 1, n)
    y = np.interp(x, np.arange(k), pts)
    if order > 1:  # smooth the linear interpolation corners
        w = max(int(SR / rate_hz / 2), 1)
        y = uniform_filter1d(y, w, mode='nearest')
        y /= (np.std(y) + 1e-9)
    return y


# ----------------------------------------------------------------- oscillators
def saw_blep(freq, phase0=0.0):
    """band-limited (polyBLEP) sawtooth; freq may be scalar or per-sample array."""
    freq = np.asarray(freq, dtype=float)
    dt = freq / SR
    ph = (phase0 + np.cumsum(dt)) % 1.0
    y = 2.0 * ph - 1.0
    dtv = np.broadcast_to(dt, ph.shape)
    m1 = ph < dtv
    x = ph[m1] / dtv[m1]
    y[m1] -= x + x - x * x - 1.0
    m2 = ph > 1.0 - dtv
    x = (ph[m2] - 1.0) / dtv[m2]
    y[m2] -= x * x + x + x + 1.0
    return y


def phase_of(freq, phase0=0.0):
    freq = np.asarray(freq, dtype=float)
    return 2 * np.pi * (phase0 + np.cumsum(freq) / SR)


# ----------------------------------------------------------------- buses
class Bus:
    """stereo accumulation buffer."""

    def __init__(self, n):
        self.x = np.zeros((2, n))
        self.n = n

    def add(self, sig, t0, pan=0.0, gain=1.0):
        """mono sig (1-D) or stereo (2, k) at time t0 seconds."""
        i0 = ns(t0)
        sig = np.asarray(sig)
        if sig.ndim == 1:
            gl, gr = pan_gains(pan)
            st = np.vstack([sig * gl, sig * gr])
        else:
            st = sig
            if pan != 0.0:
                gl, gr = pan_gains(pan)
                st = np.vstack([st[0] * gl * np.sqrt(2), st[1] * gr * np.sqrt(2)])
        k = st.shape[1]
        a, b = max(i0, 0), min(i0 + k, self.n)
        if b <= a:
            return
        self.x[:, a:b] += gain * st[:, a - i0:b - i0]

    def mix(self, other, gain=1.0):
        self.x += gain * (other.x if isinstance(other, Bus) else other)


def width(st, w):
    """stereo width: 0 = mono, 1 = unchanged, >1 wider."""
    mid = 0.5 * (st[0] + st[1])
    side = 0.5 * (st[0] - st[1]) * w
    return np.vstack([mid + side, mid - side])


# ----------------------------------------------------------------- reverb
def make_ir(t60=2.8, pre=0.028, t60_hf=None, damp_hz=5000.0, lf_mult=1.15,
            er=True, seed=7, length=None, diffuse_fade=0.012, early_ms=90.0, early_xf=220.0):
    """Synthetic stereo hall IR: band-split decaying noise (HF damped),
    pre-delay and sparse early reflections. Unit energy per channel."""
    rng = np.random.default_rng(seed)
    length = length or (t60 * 1.15 + pre)
    n = ns(length)
    t = np.arange(n) / SR
    edges = [20, 180, 500, 1200, 2800, 6000, 11000, 22000]
    out = np.zeros((2, n))
    for ch in range(2):
        w = rng.standard_normal(n)
        acc = np.zeros(n)
        for lo, hi in zip(edges[:-1], edges[1:]):
            fc = np.sqrt(lo * hi)
            if fc < 500:
                tb = t60 * lf_mult
            else:
                tb = t60 / (1.0 + (fc / damp_hz) ** 1.3)
            band = bp(w, lo, min(hi, SR * 0.45), order=3)
            acc += band * np.exp(-6.9 * t / tb)
        # diffuse build-up after pre-delay
        dly = ns(pre)
        tail = np.zeros(n)
        tail[dly:] = acc[:n - dly]
        ramp = np.clip((t - pre) / diffuse_fade, 0, 1) ** 1.5
        tail *= ramp
        if er:
            k = 14
            ref = np.max(np.abs(tail))
            taps = np.sort(rng.uniform(0.004, pre + 0.045, k))
            for i, tp in enumerate(taps):
                idx = ns(tp)
                if idx + 8 < n:
                    amp = 0.45 * (1 - i / k) * rng.uniform(0.5, 1.0) * rng.choice([-1, 1])
                    # tiny smeared tap (not a single-sample spike)
                    tail[idx:idx + 8] += amp * ref * np.hanning(8)
        out[ch] = tail / np.sqrt(np.sum(tail ** 2))
    # early field common to both sides (no image shift on sustained pure
    # tones), late tail fully decorrelated (width)
    w = np.clip((t - (pre + early_ms / 1000.0)) / (early_xf / 1000.0), 0, 1)
    w = 0.5 - 0.5 * np.cos(np.pi * w)             # 0 early -> 1 late
    common = 0.5 * (out[0] + out[1])
    for ch in range(2):
        y = common * (1 - w) + out[ch] * w
        out[ch] = y / np.sqrt(np.sum(y ** 2))
    return out


def freq_shift(x, df):
    """single-sideband frequency shift by df Hz (Hilbert transform)."""
    n = x.shape[-1]
    nf = 1 << int(np.ceil(np.log2(n)))
    a = signal.hilbert(x, N=nf)[..., :n]
    return np.real(a * np.exp(2j * np.pi * df * np.arange(n) / SR))


def convolve_reverb(st, ir, mono_in=True, shift=0.25):
    """st: (2, n) dry send -> (2, n) wet (truncated to n). The send is
    frequency-shifted by `shift` Hz so the tail never locks into a static comb
    with sustained pure tones (modulated-hall behaviour); the early field is
    common to both sides (see make_ir) so images never drift."""
    n = st.shape[1]
    if mono_in:
        src = 0.5 * (st[0] + st[1])
        sl = freq_shift(src, shift) if shift else src
        wl = signal.oaconvolve(sl, ir[0])[:n]
        wr = signal.oaconvolve(sl, ir[1])[:n]
    else:
        wl = signal.oaconvolve(st[0], ir[0])[:n] * 0.8 + signal.oaconvolve(st[1], ir[0])[:n] * 0.2
        wr = signal.oaconvolve(st[1], ir[1])[:n] * 0.8 + signal.oaconvolve(st[0], ir[1])[:n] * 0.2
    return np.vstack([wl, wr])


def snes_echo(st, delay_l=0.33, delay_r=0.44, fb=0.38, taps=5, lp_hz=3200.0):
    """SNES-flavoured echo: progressively darker repeats, cross-fed stereo."""
    n = st.shape[1]
    out = np.zeros_like(st)
    cur = st.copy()
    for k in range(1, taps + 1):
        cur = lp(cur, lp_hz, order=1)
        g = fb ** k
        dl, dr = ns(delay_l * k), ns(delay_r * k)
        src_l, src_r = (cur[1], cur[0]) if k % 2 else (cur[0], cur[1])
        if dl < n:
            out[0, dl:] += g * src_l[:n - dl]
        if dr < n:
            out[1, dr:] += g * src_r[:n - dr]
    return out


# ----------------------------------------------------------------- dynamics
def limiter(x, ceiling_db=-1.2, lookahead=0.006, release=0.12):
    """Look-ahead brickwall limiter on (2, n). Gain is multiplicative, so
    digital silence stays silent."""
    c = float(db2a(ceiling_db))
    pk = np.max(np.abs(x), axis=0)
    g = np.minimum(1.0, c / np.maximum(pk, 1e-12))
    la = max(ns(lookahead), 1)
    g = minimum_filter1d(g, size=2 * la + 1, mode='nearest')
    g = uniform_filter1d(g, size=la + 1, mode='nearest')
    # release smoothing (instant attack already provided by look-ahead ramp)
    idx = np.nonzero(g < 0.99999)[0]
    if idx.size:
        coef = np.exp(-1.0 / (release * SR))
        out = g.copy()
        start = idx[0]
        stop = min(idx[-1] + ns(release * 8), len(g))
        prev = 1.0
        seg = g[start:stop]
        res = np.empty_like(seg)
        for i in range(seg.size):
            v = seg[i]
            if v < prev:
                prev = v
            else:
                prev = v + (prev - v) * coef
            res[i] = prev
        out[start:stop] = res
        g = out
    return x * g[None, :], g


def compress(x, thresh_db=-18.0, ratio=2.0, att=0.02, rel=0.25, knee_db=6.0, makeup_db=0.0):
    """gentle RMS-ish bus compressor (feed-forward, stereo linked)."""
    lvl = np.sqrt(uniform_filter1d(np.mean(x ** 2, axis=0), ns(0.01), mode='nearest') + 1e-12)
    ldb = a2db(lvl)
    over = ldb - thresh_db
    gr = np.where(over <= -knee_db / 2, 0.0,
                  np.where(over >= knee_db / 2, over * (1 - 1 / ratio),
                           (1 - 1 / ratio) * (over + knee_db / 2) ** 2 / (2 * knee_db)))
    # smooth gain reduction (attack/release) with two one-poles on dB
    a_a = np.exp(-1 / (att * SR))
    a_r = np.exp(-1 / (rel * SR))
    fast = signal.lfilter([1 - a_a], [1, -a_a], gr)
    slow = signal.lfilter([1 - a_r], [1, -a_r], gr)
    grs = np.maximum(fast, slow * 0.999)
    return x * db2a(-grs + makeup_db)[None, :], grs


# ----------------------------------------------------------------- metering
_KB1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
_KA1 = [1.0, -1.69065929318241, 0.73248077421585]
_KB2 = [1.0, -2.0, 1.0]
_KA2 = [1.0, -1.99004745483398, 0.99007225036621]


def k_weight(x):
    y = signal.lfilter(_KB1, _KA1, x, axis=-1)
    return signal.lfilter(_KB2, _KA2, y, axis=-1)


def _block_ms(y, win, hop):
    p = np.sum(y ** 2, axis=0)
    cs = np.concatenate([[0.0], np.cumsum(p)])
    w, h = ns(win), ns(hop)
    starts = np.arange(0, len(p) - w + 1, h)
    return (cs[starts + w] - cs[starts]) / w, (starts + w / 2) / SR


def lufs_integrated(x):
    y = k_weight(x)
    ms, _ = _block_ms(y, 0.4, 0.1)
    L = -0.691 + 10 * np.log10(ms + 1e-20)
    g = ms[L > -70]
    if g.size == 0:
        return -np.inf
    lr = -0.691 + 10 * np.log10(g.mean()) - 10
    g2 = ms[(L > -70) & (L > lr)]
    return -0.691 + 10 * np.log10(g2.mean())


def loudness_curve(x, win=3.0, hop=0.05):
    y = k_weight(x)
    ms, tc = _block_ms(y, win, hop)
    return tc, -0.691 + 10 * np.log10(ms + 1e-20)


def true_peak_db(x, os=4):
    up = signal.resample_poly(x, os, 1, axis=-1)
    return float(a2db(np.max(np.abs(up))))
