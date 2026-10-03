"""
THE LAST SEED - hand-built tonal instruments (additive / subtractive).

Single notes return mono float arrays starting at the onset.
Line instruments (legato ensembles) take a list of notes
    (t, dur, midi, vel)  - absolute seconds
and return (t_start, stereo_or_mono_array).
"""
import numpy as np
from scipy.ndimage import uniform_filter1d

from audio.music_dsp import (SR, ns, hz, db2a, cents, rng_for, rcos, fade, lp, hp, bp,
                             reson, saw_blep, phase_of, smooth_rand, pan_gains, onepole_lp)


# ============================================================ metallophones
def _partials(f, parts, T, n, rng, att_ms=0.5):
    t = np.arange(n) / SR
    y = np.zeros(n)
    for r, adb, tf in parts:
        fr = f * r
        if fr > 18500:
            continue
        ph = 0.0 if r == 1.0 else rng.uniform(0, 2 * np.pi)
        y += db2a(adb) * np.sin(2 * np.pi * fr * t + ph) * np.exp(-6.9 * t / (T * tf))
    a = max(int(att_ms * SR / 1000), 2)
    y[:a] *= rcos(a)
    return y


def musicbox(midi, vel=0.75, seed=0, bright=1.0, ring=None, decay=1.0):
    """Steel-comb music box: pure fundamental + inharmonic cantilever modes
    (6.27x, 17.55x) that die fast, a torsional twin partial that makes the
    note shimmer, and a tiny pin-pluck click."""
    rng = rng_for('mbox', midi, seed)
    f = float(hz(midi))
    T = float(np.clip(3.4 * (523.25 / f) ** 0.55, 0.9, 6.0)) * decay
    L = ring if ring else T * 0.9
    n = ns(L)
    tw = 1.0 + rng.uniform(0.0007, 0.0014)
    parts = [(1.0, 0.0, 1.0), (tw, -15.0, 0.85), (2.0, -31.0, 0.25), (3.0, -42.0, 0.15),
             (6.267, -12.0 + 6 * (bright - 1), 0.10), (6.267 * 1.0035, -21.0, 0.08),
             (17.55, -30.0, 0.03)]
    y = _partials(f, parts, T, n, rng, att_ms=1.2)
    k = ns(0.006)
    clk = bp(rng.standard_normal(k), 1800, 7000) * np.exp(-np.arange(k) / (0.0012 * SR)) * rcos(k)[::-1]
    y[:k] += 0.012 * clk
    y *= vel ** 1.3
    return fade(y, 0, min(0.2, L * 0.3) if ring else min(0.06, L * 0.2))


def celesta(midi, vel=0.7, seed=0, bright=1.0, ring=None):
    """Celesta: felt hammer on steel plate over a tuned wooden resonator."""
    rng = rng_for('cel', midi, seed)
    f = float(hz(midi))
    T = float(np.clip(2.8 * (523.25 / f) ** 0.6, 0.6, 5.0))
    n = ns(min(T * 0.85, ring) if ring else T * 0.85)
    parts = [(1.0, 0.0, 1.0), (1.0009, -16.0, 0.9), (2.0, -26.0, 0.35),
             (2.756, -21.0 + 5 * (bright - 1), 0.2), (5.404, -31.0, 0.09), (8.933, -40.0, 0.045)]
    y = _partials(f, parts, T, n, rng, att_ms=1.5)
    k = ns(0.012)
    y[:k] += 0.02 * lp(rng.standard_normal(k), 2200) * np.exp(-np.arange(k) / (0.002 * SR))
    return fade(y * vel ** 1.2, 0, 0.15 if ring else 0.05)


def glock(midi, vel=0.6, seed=0, ring=None):
    """Bright bell / glockenspiel sparkle."""
    rng = rng_for('glk', midi, seed)
    f = float(hz(midi))
    T = float(np.clip(4.5 * (1046.5 / f) ** 0.4, 1.2, 6.0))
    n = ns(min(T * 0.8, ring) if ring else T * 0.8)
    parts = [(1.0, 0.0, 1.0), (1.0006, -12.0, 0.95), (2.756, -9.0, 0.3),
             (5.404, -16.0, 0.13), (8.933, -24.0, 0.06)]
    y = _partials(f, parts, T, n, rng, att_ms=0.3)
    k = ns(0.003)
    y[:k] += 0.03 * hp(rng.standard_normal(k), 4000) * np.exp(-np.arange(k) / (0.0006 * SR))
    return fade(y * vel ** 1.2, 0, 0.15 if ring else 0.05)


# ============================================================ felt piano
def piano(midi, vel=0.55, dur=2.0, seed=0, pedal=True, felt=0.75, rel=None):
    """Soft felt piano: stiff-string inharmonicity, per-partial two-stage
    decay, 1-3 detuned unison strings, felt-softened hammer spectrum,
    hammer/felt thump. `dur` = how long the damper stays off."""
    rng = rng_for('pno', midi, seed, round(float(vel), 3))
    f0 = float(hz(midi))
    B = float(np.clip(0.00030 * 2 ** ((midi - 60) / 12 * 0.95), 5e-5, 0.02))
    Tl = float(np.clip(15.0 * (65.4 / f0) ** 0.62, 1.6, 18.0))
    rel = rel if rel is not None else (0.25 if not pedal else 0.8)
    hold = min(dur, Tl * 1.05)
    L = hold + rel + 0.02
    n = ns(L)
    t = np.arange(n) / SR
    N = int(max(1, min(48, 15000 / f0)))
    k = np.arange(1, N + 1)
    fk = k * f0 * np.sqrt(1 + B * k ** 2)
    ok = fk < 17000
    k, fk = k[ok], fk[ok]
    cut = (350 + 2400 * vel ** 1.7) * felt + f0 * 1.5
    amp = (k ** -0.9) * np.exp(-(fk - f0) / cut) * (0.25 + 0.75 * np.abs(np.sin(np.pi * k / 8.3)))
    if f0 < 110:
        amp[0] *= 0.55
    Tk = np.maximum(Tl / (1.0 + (fk / 1600.0) ** 1.35), 0.12)
    c = 0.32
    if midi >= 45:
        det = [-0.9, 0.15, 0.85]
    elif midi >= 34:
        det = [-0.6, 0.6]
    else:
        det = [0.0]
    det = [d * rng.uniform(0.6, 1.3) for d in det]
    muls = [float(cents(d)) for d in det]
    y = np.zeros(n)
    for kk, fr, a, T in zip(k, fk, amp, Tk):
        e = (1 - c) * np.exp(-6.9 * t / (T * 0.28)) + c * np.exp(-6.9 * t / T)
        s = np.zeros(n)
        for mu in muls:
            ph = rng.uniform(0, 2 * np.pi) if kk > 1 else 0.0
            s += np.sin(2 * np.pi * fr * mu * t + ph)
        y += a * e * s
    y /= len(muls)
    a3 = ns(0.003)
    y[:a3] *= rcos(a3)
    kn = min(ns(0.03), n)
    tn = t[:kn]
    y[:kn] += 0.012 * vel * bp(rng.standard_normal(kn), 400, 3500) * np.exp(-tn / 0.004)
    y[:kn] += 0.03 * vel * np.sin(2 * np.pi * 70 * tn) * np.exp(-tn / 0.012) * rcos(kn)[::-1]
    # damper
    h = ns(hold)
    if h < n:
        tr = np.arange(n - h) / SR
        y[h:] *= np.exp(-6.9 * tr / rel) * (0.5 + 0.5 * np.cos(np.pi * np.minimum(tr / rel, 1)))
    return y * vel ** 1.4


# ============================================================ plucked strings
def _plucked(f0, vel, rng, T1, beta, cut, inh=2e-5, L=None, att_ms=1.2, noise_lvl=0.01, nlo=1500, nhi=6000,
             damp=0.05):
    L = L or T1 * 0.8
    n = ns(L)
    t = np.arange(n) / SR
    N = int(max(1, min(30, 12000 / f0)))
    k = np.arange(1, N + 1)
    fk = k * f0 * np.sqrt(1 + inh * k ** 2)
    ok = fk < 17000
    k, fk = k[ok], fk[ok]
    amp = np.abs(np.sin(np.pi * k * beta)) / k ** 1.15 * np.exp(-fk / cut)
    Tk = np.maximum(T1 / (1 + (fk / 1300.0) ** 1.4), 0.06)
    y = np.zeros(n)
    for kk, fr, a, T in zip(k, fk, amp, Tk):
        ph = 0.0 if kk == 1 else rng.uniform(0, 2 * np.pi)
        y += a * np.sin(2 * np.pi * fr * t + ph) * np.exp(-6.9 * t / T)
    y /= np.abs(np.sin(np.pi * beta)) + 1e-3
    a = max(int(att_ms * SR / 1000), 2)
    y[:a] *= rcos(a)
    kn = min(ns(0.006), n)
    y[:kn] += noise_lvl * bp(rng.standard_normal(kn), nlo, nhi) * np.exp(-np.arange(kn) / (0.0012 * SR))
    return fade(y * vel ** 1.3, 0, min(damp, L * 0.3))


def harp(midi, vel=0.65, seed=0, bright=1.0, ring=None, decay=1.0):
    rng = rng_for('harp', midi, seed)
    f0 = float(hz(midi))
    T1 = float(np.clip(8.0 * (98.0 / f0) ** 0.6, 1.0, 10.0)) * decay
    return _plucked(f0, vel, rng, T1, 0.28 + 0.04 * rng.random(), 2200 + 2600 * vel * bright,
                    L=min(ring, T1 * 0.8) if ring else min(T1 * 0.8, 6.0), damp=0.12 if ring else 0.05)


def pizz(midi, vel=0.7, seed=0, bright=1.0):
    rng = rng_for('pizz', midi, seed)
    f0 = float(hz(midi))
    T1 = float(np.clip(1.3 * (98.0 / f0) ** 0.35, 0.3, 1.5))
    return _plucked(f0, vel, rng, T1, 0.18, 3000 + 2000 * vel * bright, inh=4e-5,
                    L=T1 * 1.0, att_ms=0.8, noise_lvl=0.02, nlo=2000, nhi=7000)


# ============================================================ line machinery
def phrases(notes, gap=0.035):
    """split sorted notes (t, dur, midi, vel) into legato phrases."""
    notes = sorted(notes, key=lambda x: x[0])
    out, cur = [], []
    for nt in notes:
        if cur and nt[0] - (cur[-1][0] + cur[-1][1]) > gap:
            out.append(cur)
            cur = []
        cur.append(nt)
    if cur:
        out.append(cur)
    return out


def phrase_curves(ph, tA, n, glide, lag, att, rel, dip=0.0, dip_w=0.035, vel_smooth=0.06):
    """per-sample midi, velocity, envelope, time-since-note for one phrase."""
    t = tA + np.arange(n) / SR
    st = np.array([p[0] for p in ph]) + lag
    idx = np.clip(np.searchsorted(st, t, side='right') - 1, 0, len(ph) - 1)
    mid = np.array([p[2] for p in ph], float)[idx]
    vel = np.array([p[3] for p in ph], float)[idx]
    tn = t - st[idx]
    g = max(ns(glide), 1)
    mid = uniform_filter1d(uniform_filter1d(mid, g, mode='nearest'), g, mode='nearest')
    vel = uniform_filter1d(vel, max(ns(vel_smooth), 1), mode='nearest')
    on = st[0]
    off = ph[-1][0] + ph[-1][1] + lag
    x = np.clip((t - on) / max(att, 1e-3), 0, 1)
    env = 0.5 - 0.5 * np.cos(np.pi * x)
    after = np.clip((t - off) / max(rel, 1e-3), 0, 1)
    env *= np.exp(-5.0 * after) * (0.5 + 0.5 * np.cos(np.pi * after))
    if dip > 0 and len(ph) > 1:
        for s in st[1:]:
            env *= 1 - dip * np.exp(-((t - s) / dip_w) ** 2)
    return mid, vel, env, tn


def _render_phrases(notes, voice_fn, pre, rel, extra=0.0, gap=0.035):
    """helper: calls voice_fn(ph, tA, n) per phrase; returns (t_start, stereo)."""
    phs = phrases(notes, gap)
    t_start = min(p[0][0] for p in phs) - pre
    t_end = max(p[-1][0] + p[-1][1] for p in phs) + rel + extra
    out = np.zeros((2, ns(t_end - t_start) + 16))
    for ph in phs:
        tA = ph[0][0] - pre
        tB = ph[-1][0] + ph[-1][1] + rel + extra
        n = ns(tB - tA)
        y = voice_fn(ph, tA, n)
        i0 = ns(tA - t_start)
        if y.ndim == 1:
            y = np.vstack([y, y]) * np.sqrt(0.5)
        out[:, i0:i0 + y.shape[1]] += y[:, :out.shape[1] - i0]
    return t_start, out


def trem_am(n, rng, rate=13.0):
    r = rate * (1 + 0.12 * smooth_rand(n, 2.0, rng))
    ph = np.cumsum(r) / SR + rng.random()
    s = np.abs(np.sin(np.pi * ph))
    return 0.3 + 0.7 * s ** 0.7


# ============================================================ bowed strings
def strings_line(notes, voices=6, att=0.25, rel=0.6, glide=0.06, vib_rate=5.6, vib_cents=14.0,
                 bright=1.0, trem=0.0, trem_rate=13.0, seed=0, dip=0.18, spread=0.6, pan=0.0,
                 tone='saw', detune=5.0, gap=0.035, bend=None):
    """Section of `voices` bowed players on one legato part. Each player has
    its own tuning, drift, vibrato and timing lag (smears note changes like a
    real section). Dynamic brightness follows the envelope/velocity."""
    rng0 = rng_for('str', seed, len(notes), notes[0][0], notes[0][2])
    pre = 0.05

    def voice_fn(ph, tA, n):
        st = np.zeros((2, n))
        dyn = np.zeros(n)
        tt = np.arange(n) / SR
        for v in range(voices):
            rng = np.random.default_rng(rng0.integers(1 << 31))
            lag = rng.uniform(-0.012, 0.024)
            midc, vel, env, tn = phrase_curves(ph, tA, n, glide * rng.uniform(0.8, 1.3), lag,
                                               att * rng.uniform(0.85, 1.2), rel * rng.uniform(0.9, 1.1), dip)
            vd = vib_cents * rng.uniform(0.7, 1.25)
            ve = uniform_filter1d(np.clip((tn - 0.18) / 0.4, 0, 1), ns(0.08), mode='nearest')
            vib = vd * ve * np.sin(2 * np.pi * vib_rate * rng.uniform(0.92, 1.08) * tt + rng.uniform(0, 6.28))
            c = rng.normal(0, detune) + 2.5 * smooth_rand(n, 0.5, rng) + vib
            if bend:
                c = c + np.interp(tA + tt, [b[0] for b in bend], [b[1] for b in bend])
            f = hz(midc) * cents(c)
            if tone == 'saw':
                osc = saw_blep(f, rng.random())
            else:   # 'harm' : flautando / harmonics - nearly pure
                p = phase_of(f, rng.random())
                osc = np.sin(p) + 0.12 * np.sin(2 * p) + 0.04 * np.sin(3 * p)
            a = env * vel ** 1.2 * (1 + 0.05 * smooth_rand(n, 3.0, rng))
            if trem > 0:
                a *= (1 - trem) + trem * trem_am(n, rng, trem_rate)
            y = osc * a
            gl, gr = pan_gains(np.clip(pan + spread * (rng.random() - 0.5) * 2, -1, 1))
            st[0] += gl * y
            st[1] += gr * y
            dyn += env * vel
        dyn = np.clip(dyn / voices, 0, 1)
        if tone == 'saw':
            fc_lo = 1300 + 1000 * bright
            fc_hi = 3000 + 3200 * bright
            w = dyn ** 1.3
            st = lp(st, fc_lo, 4) * (1 - w) + lp(st, fc_hi, 4) * w
            # rosin / bow hiss riding the bowing envelope
            nz = bp(rng0.standard_normal((2, n)), 2500, 9000, 2)
            st += 0.035 * np.sqrt(voices) * nz * dyn[None, :] * (0.6 + 0.4 * bright)
        return st / np.sqrt(voices)

    return _render_phrases(notes, voice_fn, pre, rel, gap=gap)


def string_body(x, sect='vln'):
    """static body / bridge EQ applied on a string section bus."""
    from audio.music_dsp import peq, shelf
    if sect in ('vln', 'vla'):
        y = hp(x, 150 if sect == 'vln' else 110, 2)
        y = peq(y, 290, 3.0, 1.4)
        y = peq(y, 560, 2.0, 1.6)
        y = peq(y, 1500, -3.0, 1.2)
        y = peq(y, 2700, 3.0, 1.3)
        y = shelf(y, 5000, -6.0, 'high')
    else:  # celli / basses
        y = hp(x, 32, 2)
        y = peq(y, 220, 2.5, 1.2)
        y = peq(y, 600, -2.0, 1.0)
        y = peq(y, 1800, 1.5, 1.2)
        y = shelf(y, 3500, -8.0, 'high')
    return y


# ============================================================ choir
FORMANTS = {  # vowel 'a' : (freqs, dB gains, bandwidths)
    's': ([800, 1150, 2900, 3900, 4950], [0, -6, -32, -20, -50], [80, 90, 120, 130, 140]),
    'a': ([800, 1150, 2800, 3500, 4950], [0, -4, -20, -36, -60], [80, 90, 120, 130, 140]),
    't': ([650, 1080, 2650, 2900, 3250], [0, -6, -7, -8, -22], [80, 90, 120, 130, 140]),
    'b': ([600, 1040, 2250, 2450, 2750], [0, -7, -9, -9, -20], [60, 70, 110, 120, 130]),
    'o': ([450, 800, 2830, 3500, 4950], [0, -9, -16, -28, -55], [70, 80, 100, 130, 135]),  # soft 'oh'
}


def choir_line(notes, voices=6, att=0.35, rel=0.7, glide=0.07, vib_rate=5.2, vib_cents=20.0,
               seed=0, breath=0.05, spread=0.8, pan=0.0, dip=0.12):
    """Choir part: glottal-ish sources (saw -> tilt) for `voices` singers.
    Returns the raw source; formant-filter it per register with `formant()`."""
    rng0 = rng_for('choir', seed, len(notes), notes[0][0], notes[0][2])

    def voice_fn(ph, tA, n):
        st = np.zeros((2, n))
        tt = np.arange(n) / SR
        for v in range(voices):
            rng = np.random.default_rng(rng0.integers(1 << 31))
            lag = rng.uniform(-0.02, 0.03)
            midc, vel, env, tn = phrase_curves(ph, tA, n, glide * rng.uniform(0.8, 1.25), lag,
                                               att * rng.uniform(0.8, 1.3), rel * rng.uniform(0.85, 1.15), dip)
            ve = uniform_filter1d(np.clip((tn - 0.25) / 0.5, 0, 1), ns(0.1), mode='nearest')
            vib = vib_cents * rng.uniform(0.6, 1.3) * ve * np.sin(
                2 * np.pi * vib_rate * rng.uniform(0.9, 1.1) * tt + rng.uniform(0, 6.28))
            c = rng.normal(0, 8.0) + 4.0 * smooth_rand(n, 0.6, rng) + vib
            f = hz(midc) * cents(c)
            src = saw_blep(f, rng.random())
            a = env * vel ** 1.2 * (1 + 0.08 * smooth_rand(n, 2.5, rng))
            y = src * a + breath * rng.standard_normal(n) * a
            gl, gr = pan_gains(np.clip(pan + spread * (rng.random() - 0.5) * 2, -1, 1))
            st[0] += gl * y
            st[1] += gr * y
        return onepole_lp(st, 1600.0) / np.sqrt(voices)

    return _render_phrases(notes, voice_fn, 0.05, rel)


def formant(x, reg='s', mix_direct=0.05):
    """vowel formant bank; on stereo input each side gets slightly different
    formants (different singers) for a wider, less synthetic choir."""
    fr, gdb, bw = FORMANTS[reg]

    def bank(v, scale):
        y = np.zeros_like(v)
        for f, g, b in zip(fr, gdb, bw):
            y += db2a(g) * reson(v, f * scale, b * scale)
        return y + mix_direct * hp(v, 3000)
    if x.ndim == 2:
        return np.vstack([bank(x[0], 0.975), bank(x[1], 1.03)]) * 2.2
    return bank(x, 1.0) * 2.2


# ============================================================ flute / ocarina
def flute_line(notes, kind='flute', att=0.07, rel=0.35, glide=0.025, vib_rate=5.0, vib_cents=13.0,
               trem=0.07, breath=0.05, seed=0, dip=0.35, bright=1.0, gap=0.035):
    """Monophonic flute/ocarina: sine + few harmonics, delayed vibrato with
    amplitude vibrato, breath noise shaped around the pitch, tongue chiff."""
    rng = rng_for('flute', kind, seed, notes[0][0])

    def voice_fn(ph, tA, n):
        tt = np.arange(n) / SR
        midc, vel, env, tn = phrase_curves(ph, tA, n, glide, 0.0, att, rel, dip)
        ve = uniform_filter1d(np.clip((tn - 0.22) / 0.45, 0, 1), ns(0.08), mode='nearest')
        vr = np.sin(2 * np.pi * vib_rate * tt + rng.uniform(0, 6.28))
        f = hz(midc) * cents(vib_cents * ve * vr + 3.0 * smooth_rand(n, 0.8, rng))
        p = phase_of(f)
        A = env * vel ** 1.1 * (1 + trem * ve * vr)
        b = (0.6 + 0.8 * vel) * bright
        if kind == 'flute':
            hs = [1.0, 0.30 * b, 0.13 * b, 0.06 * b, 0.03 * b]
        else:  # ocarina: almost pure, hollow
            hs = [1.0, 0.09 * b, 0.025 * b, 0.01 * b]
        tone = np.zeros(n)
        for k, h in enumerate(hs, 1):
            if np.max(f) * k < 17000:
                tone += h * np.sin(k * p + (0 if k == 1 else rng.uniform(0, 6.28)))
        fm = float(np.median(hz(midc)))
        nz = rng.standard_normal(n)
        br = reson(nz, fm, fm * 0.12) * 1.5 + 0.25 * bp(nz, 2500, 9000)
        y = tone * A + breath * br * A * (0.7 + 0.3 * vel)
        # chiff on phrase start and tongued notes
        starts = [ph[0][0]] + ([p_[0] for p_ in ph[1:]] if dip > 0.2 else [])
        for s in starts:
            i = ns(s - tA)
            k = ns(0.05)
            if 0 <= i < n - k:
                y[i:i + k] += 0.05 * vel[i] * bp(rng.standard_normal(k), 1800, 7000) * np.exp(-np.arange(k) / (0.01 * SR))
        return y

    return _render_phrases(notes, voice_fn, 0.03, rel, gap=gap)


# ============================================================ horn
def horn_line(notes, voices=3, att=0.12, rel=0.45, glide=0.06, vib_cents=6.0, seed=0, bright=1.0,
              spread=0.3, pan=0.0, dip=0.25):
    """Warm French-horn section: saw through a dynamics-tracking low-pass."""
    rng0 = rng_for('horn', seed, notes[0][0], notes[0][2])

    def voice_fn(ph, tA, n):
        tt = np.arange(n) / SR
        st = np.zeros((2, n))
        for v in range(voices):
            rng = np.random.default_rng(rng0.integers(1 << 31))
            midc, vel, env, tn = phrase_curves(ph, tA, n, glide, rng.uniform(-0.01, 0.02), att, rel, dip)
            ve = np.clip((tn - 0.3) / 0.5, 0, 1)
            c = rng.normal(0, 4.0) + 2.0 * smooth_rand(n, 0.5, rng) + vib_cents * ve * np.sin(
                2 * np.pi * 4.8 * tt + rng.uniform(0, 6.28))
            f = hz(midc) * cents(c)
            src = saw_blep(f, rng.random())
            a = env * vel ** 1.3
            w = np.clip(a, 0, 1) ** 1.4
            y = lp(src, 450 * bright, 2) * (1 - w) + lp(src, 2400 * bright, 2) * w
            y *= a
            gl, gr = pan_gains(np.clip(pan + spread * (rng.random() - 0.5) * 2, -1, 1))
            st[0] += gl * y
            st[1] += gr * y
        return lp(st, 5000, 2) / np.sqrt(voices)

    return _render_phrases(notes, voice_fn, 0.05, rel)


# ============================================================ pads
def pad_line(notes, voices=3, att=1.2, rel=1.6, glide=0.25, seed=0, cutoff=900.0, spread=0.9,
             pan=0.0, shape='saw', detune=7.0):
    """Slow synth pad (cold: low cutoff, no vibrato, slow drift)."""
    rng0 = rng_for('pad', seed, notes[0][0], notes[0][2])

    def voice_fn(ph, tA, n):
        st = np.zeros((2, n))
        for v in range(voices):
            rng = np.random.default_rng(rng0.integers(1 << 31))
            midc, vel, env, tn = phrase_curves(ph, tA, n, glide, rng.uniform(-0.05, 0.05), att, rel, 0.0)
            c = rng.normal(0, detune) + 3.0 * smooth_rand(n, 0.25, rng)
            f = hz(midc) * cents(c)
            if shape == 'saw':
                osc = saw_blep(f, rng.random())
            else:  # soft triangle-ish (odd harmonics)
                p = phase_of(f, rng.random())
                osc = np.sin(p) + 0.11 * np.sin(3 * p) + 0.04 * np.sin(5 * p)
            y = osc * env * vel
            gl, gr = pan_gains(np.clip(pan + spread * (rng.random() - 0.5) * 2, -1, 1))
            st[0] += gl * y
            st[1] += gr * y
        return lp(st, cutoff, 2) / np.sqrt(voices)

    return _render_phrases(notes, voice_fn, 0.05, rel)
