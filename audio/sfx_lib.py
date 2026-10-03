"""
THE LAST SEED - sound library. Every function returns a freshly synthesized
mono (n,) or stereo (2, n) float array starting at t=0 (onset at the first
samples unless documented), with per-instance randomization from `r`.
"""
import numpy as np
from scipy import signal

from sfx_dsp import (SR, TAU, n_of, tvec, rng, noise, lp, hp, bp, biquad, peak_eq, tvf,
                     onepole, env_pts, ar, exp_dec, fades, smooth_rand, smoothstep, phase_of,
                     additive, fm, modal, resonate, poisson_impulses, crackle, friction,
                     impulse_train, pan, width, decorrelate, normalize, undb, soft_clip, ramp)

# =============================================================================
# ROBOT - body
# =============================================================================
# The Gardener's body: a small hollow mint-painted steel shell. One shared set of
# modal frequencies (per-instance jitter) so every clank sounds like the same robot.
BODY_MODES = np.array([612.0, 947.0, 1391.0, 1876.0, 2533.0, 3317.0, 4410.0, 5870.0])
BODY_T60 = np.array([0.16, 0.13, 0.11, 0.09, 0.07, 0.055, 0.04, 0.03])
CAN_MODES = np.array([486.0, 1143.0, 1712.0, 2398.0, 3261.0, 4180.0])      # dented tin watering can
CAN_T60 = np.array([0.35, 0.28, 0.22, 0.16, 0.12, 0.09])


def fit(x, n):
    """Pad with zeros / truncate the last axis to n samples."""
    m = x.shape[-1]
    if m >= n:
        return x[..., :n]
    pad = [(0, 0)] * (x.ndim - 1) + [(0, n - m)]
    return np.pad(x, pad)


def _jit(r, a, amt):
    return a * (1 + r.uniform(-amt, amt, len(a)))


def body_clank(r, level=1.0, bright=1.0, dur=0.35, rattle=0.3, size=1.0):
    """The robot shell ringing briefly (+ optional loose-part rattle)."""
    fr = _jit(r, BODY_MODES / size, 0.035)
    amps = np.array([1.0, 0.85, 0.7, 0.6, 0.5, 0.35, 0.25, 0.18]) * r.uniform(0.35, 1.0, 8)
    amps *= bright ** np.linspace(-0.5, 1.5, 8)
    y = modal(dur, fr, _jit(r, BODY_T60, 0.25) * size, amps, r)
    # soften the strike slightly (felt-ish contact through the dirt)
    k = np.hanning(n_of(0.0012) + 2)
    y = signal.fftconvolve(y, k / k.sum())[:len(y)]
    if rattle > 0 and r.random() < 0.85:
        n = len(y)
        for _ in range(r.integers(1, 4)):
            d = r.uniform(0.012, 0.06)
            i = n_of(d)
            if i < n:
                m = modal(dur - d, _jit(r, BODY_MODES[3:] * r.uniform(1.1, 1.6), 0.05),
                          BODY_T60[3:] * 0.5, r.uniform(0.2, 1.0, 5), r)
                y[i:] += rattle * r.uniform(0.2, 0.6) * m[:n - i]
    return level * y


def can_clink(r, level=1.0, damp=1.0, dur=0.6):
    fr = _jit(r, CAN_MODES, 0.02)
    amps = np.array([0.6, 1.0, 0.8, 0.6, 0.4, 0.25]) * r.uniform(0.3, 1.0, 6)
    y = modal(dur, fr, CAN_T60 / damp, amps, r)
    k = np.hanning(n_of(0.0008) + 2)
    return level * signal.fftconvolve(y, k / k.sum())[:len(y)]


def footstep(r, surface='dirt', weight=1.0, clank=1.0, grit=1.0, can=0.0, drag=0.0, glass=0.0):
    """One small clanky robot footfall. Onset (contact) at t = drag lead-in (0 if no drag).
    surface: 'dirt' | 'sand' | 'stone'."""
    lead = 0.0
    if drag > 0:
        lead = r.uniform(0.09, 0.14)
    dur = 0.42 + lead
    n = n_of(dur)
    y = np.zeros(n)
    i0 = n_of(lead)
    m = n - i0
    t = tvec(m)
    # 1. contact thud (pitch-dropping sine + low noise)
    f0 = r.uniform(88, 118) * (1.15 - 0.15 * weight)
    fcur = f0 * (1 + 0.8 * np.exp(-t / 0.005))
    thud = np.sin(phase_of(fcur)) * np.exp(-t / (0.018 + 0.012 * weight))
    thud *= np.clip(t / 0.0015, 0, 1)
    thud += 0.7 * lp(noise(m, r), 380) * np.exp(-t / 0.012)
    y[i0:] += (0.55 + 0.45 * weight) * thud
    # 2. ground texture
    if surface in ('dirt', 'sand'):
        dens = (5000 if surface == 'dirt' else 9000) * np.exp(-t / (0.018 if surface == 'dirt' else 0.035))
        g = crackle(m, dens, r, band=(700, 7500), grain_ms=0.5)
        g += 0.5 * bp(noise(m, r), 500, 3000) * np.exp(-t / 0.012)
        y[i0:] += grit * 0.14 * g / (np.std(g[:n_of(0.02)]) + 1e-9) * (0.6 + 0.4 * weight)
    else:  # stone: hard short clack
        c = bp(noise(m, r), 1200, 6500) * np.exp(-t / 0.0025)
        c += 0.5 * modal(m / SR, _jit(r, np.array([2210.0, 3480.0, 5120.0]), 0.1),
                         [0.02, 0.015, 0.01], [1, 0.7, 0.5], r)
        y[i0:] += grit * 0.55 * c
        g = crackle(m, 2500 * np.exp(-t / 0.01), r, band=(1500, 9000), grain_ms=0.3)
        y[i0:] += grit * 0.05 * g / (np.std(g[:n_of(0.01)]) + 1e-9)
    # 3. the shell clanks a few ms after contact
    dly = n_of(r.uniform(0.004, 0.011))
    bc = body_clank(r, clank * 0.3, bright=1.0 + 0.3 * (surface == 'stone'), dur=0.3, rattle=0.35)
    k = min(len(bc), n - i0 - dly)
    y[i0 + dly:i0 + dly + k] += bc[:k]
    # 4. the watering can swinging in the hand
    if can > 0 and r.random() < 0.7:
        d2 = n_of(r.uniform(0.02, 0.05))
        cc = can_clink(r, can * r.uniform(0.08, 0.2), damp=1.5, dur=0.35)
        k = min(len(cc), n - i0 - d2)
        y[i0 + d2:i0 + d2 + k] += cc[:k]
    # 5. drag / scuff before contact (tired or heavy feet)
    if drag > 0:
        dn = i0 + n_of(0.01)
        tt = tvec(dn)
        e = np.sin(np.pi * np.clip(tt / (dn / SR), 0, 1)) ** 1.5
        sc = crackle(dn, 7000 * e, r, band=(600, 5000), grain_ms=0.8) + 0.6 * bp(noise(dn, r), 400, 2500)
        y[:dn] += drag * 0.1 * sc * e / (np.std(sc) + 1e-9)
    # 6. tiny glass crunch under foot (greenhouse floor)
    if glass > 0:
        gl = glass_crunch(r, level=glass, dur=0.25)
        k = min(len(gl), n - i0)
        y[i0:i0 + k] += gl[:k]
    return fades(y, 0.0005, 0.03)


# ----------------------------------------------------------------------------- servos
def servo(r, dur, f0=150.0, f1=380.0, shape='bell', level=1.0, bright=1.0, gear=0.5,
          whine=0.35, grit=0.15, formant=1800.0, jerk=0.0):
    """Small geared DC servo ('whirr'): pitch follows motor speed.
    shape: bell | up | down | halt | flat."""
    n = n_of(dur)
    x = np.linspace(0, 1, n)
    if shape == 'bell':
        v = np.sin(np.pi * x) ** 0.7
    elif shape == 'up':
        v = smoothstep(x * 1.6) * (1 - smoothstep((x - 0.85) / 0.15))
    elif shape == 'down':
        v = (1 - smoothstep(x)) * smoothstep(x * 8)
    elif shape == 'halt':          # fast spin-up then abrupt stop
        v = smoothstep(x * 3.0) * (x < 0.93) + (x >= 0.93) * np.clip((1 - x) / 0.07, 0, 1) ** 2
    else:
        v = np.ones(n)
    v = np.clip(v, 0, 1)
    wob = 1 + 0.025 * (smooth_rand(n, 30, r) - 0.5) + jerk * 0.1 * (smooth_rand(n, 12, r) - 0.5)
    f = (f0 + (f1 - f0) * v) * wob
    # soft pulse-ish motor tone, steep harmonic roll-off (a whirr, not a buzz)
    amps = [abs(np.sin(np.pi * k * 0.3)) / k ** 1.45 for k in range(1, 26)]
    y = additive(f, amps, rng_=r)
    y *= 1 + gear * 0.5 * np.sin(phase_of(f * 3.5) + r.random() * TAU)
    y = peak_eq(y, formant * bright, 7.0, 1.8)
    y = hp(y, 160, 2)
    y = lp(y, 3800 * bright, 2)
    y = y / (np.std(y) + 1e-9)
    # gear-mesh whirr: band noise at a few kHz, amplitude-modulated by the tooth rate
    gn = bp(noise(n, r), 1200 * bright, 4200 * bright, 2)
    gn *= 0.55 + 0.45 * np.sin(phase_of(f * 2.0))
    y += 0.45 * gn / (np.std(gn) + 1e-9)
    # characteristic high whine tracking the speed (quiet)
    wf = np.minimum(f * r.uniform(8.0, 10.0), 9000.0)
    y += whine * 0.8 * np.sin(phase_of(wf)) * (0.2 + 0.8 * v)
    y += grit * bp(noise(n, r), 2500, 7000) * v
    e = (0.2 + 0.8 * v) * ar(n, 0.01, 0.025, hold=max(dur - 0.035, 0))
    return fades(level * 0.22 * y * e, 0.005, 0.015)


def micro_click(r, level=1.0, f=(3100.0, 5200.0, 7900.0), t60=0.02, body=0.3):
    """A tiny crisp mechanical click (relay, detent, latch)."""
    fr = np.array(f) * (1 + r.uniform(-0.05, 0.05, len(f)))
    y = modal(0.08, fr, [t60 * 1.3, t60, t60 * 0.7][:len(f)], r.uniform(0.5, 1.0, len(f)), r)
    t = tvec(len(y))
    y += 1.2 * hp(noise(len(y), r), 2000) * np.exp(-t / 0.0006)
    y += body * np.sin(TAU * r.uniform(700, 1100) * t) * np.exp(-t / 0.006)
    return level * fades(y, 0.0001, 0.01)


def clicks(r, times, level=1.0, **kw):
    n = n_of(max(times) + 0.1)
    y = np.zeros(n)
    for k, t in enumerate(times):
        c = micro_click(r, level * (1.0 if k == 0 else r.uniform(0.6, 0.95)), **kw)
        i = n_of(t)
        m = min(len(c), n - i)
        y[i:i + m] += c[:m]
    return y


# ----------------------------------------------------------------------------- the VOICE
# PITCH PLAN - every pitched SFX is tuned to the score's material (composer's auditions: D minor / F major,
# G-major choir & horn at the climax). All pitched sounds use the tones F major and G major share,
# {C, D, E, G, A}:  robot motif G-E-D-G (pentatonic, no third), heart hum on D, shimmers on open fifths.
PITCH_SHIFT = 0.0              # semitones: transposes EVERY pitched SFX together if the score changes key

_NOTE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def note(name):
    """'G5', 'C#6', 'Bb4' -> Hz (A4 = 440), including PITCH_SHIFT."""
    k = _NOTE[name[0].upper()]
    i = 1
    while i < len(name) and name[i] in '#b':
        k += 1 if name[i] == '#' else -1
        i += 1
    midi = 12 * (int(name[i:]) + 1) + k
    return 440.0 * 2 ** ((midi - 69 + PITCH_SHIFT) / 12.0)


VOICE_ROOT_HZ = note('G5')     # 784 Hz
MOTIF = [0, 9, 7, 12]          # G5  E6  D6  G6   - the robot's 'name' motif (bi-DEE-doo-DEEE)
HUM_HZ = note('D2')            # 73.4 Hz - the robot's heart


def voice(notes, r, root=VOICE_ROOT_HZ, index=1.3, bright=1.0, whisper=0.0, damage=0.0,
          level=1.0, tail=0.25):
    """Tiny melodic robot voice. notes: list of dicts
       semi, dur, gap(after), amp, scoop (semitones below, 0..), glide_to (semi at end),
       bend (semitones added over last 40%), vib (cents), vib_rate."""
    total = sum(nt['dur'] + nt.get('gap', 0.0) for nt in notes) + tail
    n = n_of(total)
    t = tvec(n)
    semi = np.zeros(n)
    amp = np.zeros(n)
    vib = np.zeros(n)
    attack_emph = np.zeros(n)
    mouth = np.zeros(n)
    cur = 0
    prev_end = None
    for k, nt in enumerate(notes):
        d = nt['dur']
        g = nt.get('gap', 0.0)
        i0, i1 = cur, cur + n_of(d)
        tt = tvec(i1 - i0)
        s0 = nt['semi']
        s1 = nt.get('glide_to', s0)
        sc = nt.get('scoop', 1.0)
        curve = s0 + (s1 - s0) * smoothstep(tt / d) ** 1.2
        curve = curve - sc * np.exp(-tt / 0.012) + 0.18 * sc * np.exp(-((tt - 0.03) / 0.012) ** 2)
        b = nt.get('bend', 0.0)
        if b:
            curve = curve + b * smoothstep((tt / d - 0.55) / 0.45)
        semi[i0:i1] = curve
        a = nt.get('amp', 1.0)
        att = nt.get('att', 0.006)
        rel = nt.get('rel', 0.022)
        e = ar(i1 - i0, att, rel, hold=max(d - att - rel, 0.0), curve=1.6)
        e *= 1.0 - 0.18 * (tt / d)             # slight sag across the note
        amp[i0:i1] = np.maximum(amp[i0:i1], a * e)
        attack_emph[i0:i1] = np.exp(-tt / 0.03)
        # 'mouth': the formant opens quickly and closes toward the end of each note (a vocal 'wah')
        mouth[i0:i1] = np.sin(np.pi * np.clip(tt / d, 0, 1)) ** 0.6 * (1 - 0.3 * (tt / d))
        vc = nt.get('vib', 0.0)
        if vc:
            vr = nt.get('vib_rate', 6.8)
            ve = smoothstep((tt - 0.05) / 0.08)
            vib[i0:i1] = vc / 100.0 * ve * np.sin(TAU * vr * tt)
        # portamento through the gap toward next note's scooped start
        j1 = i1 + n_of(g)
        if k + 1 < len(notes) and j1 > i1:
            nx = notes[k + 1]['semi'] - notes[k + 1].get('scoop', 1.0)
            semi[i1:j1] = np.linspace(curve[-1], nx, j1 - i1)
        elif j1 > i1:
            semi[i1:j1] = curve[-1]
        cur = j1
    if cur < n:
        semi[cur:] = semi[cur - 1] if cur > 0 else 0
    f = root * 2 ** ((semi + vib) / 12.0)
    f *= 1 + 0.003 * (smooth_rand(n, 18, r) - 0.5)                 # tiny living pitch jitter (+-2.5 cents)
    if damage > 0:
        f *= 1 + damage * 0.01 * (smooth_rand(n, 40, r) - 0.5)
    idx = index * (0.65 + 0.7 * attack_emph)
    ph = phase_of(f)
    # ratio-2 FM core (strong fundamental + odd partials: warm, ocarina / 16-bit-ish) with a touch of
    # ratio-1 modulation for a few even partials; brighter at each note's attack
    y = np.sin(ph + 0.75 * idx * np.sin(2 * ph) + 0.22 * idx * np.sin(ph))
    y *= amp
    # tiny speaker with a moving 'mouth' formant + band limits + gentle saturation
    fm_c = (1050 + 1350 * onepole(mouth, 60)) * bright
    y = tvf(y, fm_c, 1.4, 'peak', gain_db=7.0, block=64)
    y = hp(y, 320, 2)
    y = lp(y, 6500 * bright, 2)
    y = soft_clip(y * 1.4, 1.2) / 1.4
    if whisper > 0:
        br = noise(n, r)
        b1 = tvf(br, f * 1.0, 6.0, 'bp')          # pitched breath
        b2 = bp(br, 1500, 6000)
        breath = (0.7 * b1 + 0.3 * b2)
        breath = breath / (np.std(breath) + 1e-9)
        y = (1 - 0.35 * whisper) * y + whisper * 0.22 * breath * amp ** 1.3
    if damage > 0:
        # weak, flickery speaker: random micro-dropouts + faint crackle
        drop = 1 - damage * 0.6 * (smooth_rand(n, 25, r) > 0.82)
        y *= onepole(drop, 200)
        y += damage * 0.05 * crackle(n, 300 * (amp > 0.05), r, band=(2000, 8000), grain_ms=0.3)
    return level * fades(y, 0.001, 0.02)


def motif_notes(style='happy'):
    s = MOTIF
    if style == 'happy':      # bright, bouncy
        return [dict(semi=s[0], dur=0.075, gap=0.028, amp=0.82, scoop=1.4),
                dict(semi=s[1], dur=0.068, gap=0.028, amp=0.74, scoop=0.9),
                dict(semi=s[2], dur=0.078, gap=0.036, amp=0.84, scoop=0.9),
                dict(semi=s[3], dur=0.215, gap=0.0, amp=1.0, scoop=1.8, bend=1.4, vib=32, vib_rate=7.2)]
    if style == 'whisper':    # SAME motif: slower, softer, warmer - an echo
        return [dict(semi=s[0], dur=0.17, gap=0.075, amp=0.78, scoop=0.9, att=0.03, rel=0.06),
                dict(semi=s[1], dur=0.15, gap=0.075, amp=0.70, scoop=0.7, att=0.03, rel=0.06),
                dict(semi=s[2], dur=0.17, gap=0.09, amp=0.80, scoop=0.7, att=0.03, rel=0.06),
                dict(semi=s[3], dur=0.62, gap=0.0, amp=0.95, scoop=1.0, bend=0.5, vib=22, vib_rate=5.2,
                     att=0.05, rel=0.35)]
    if style == 'soft':       # look_down_can: tender two-note fragment
        return [dict(semi=s[2], dur=0.11, gap=0.05, amp=0.8, scoop=0.8, att=0.012, rel=0.04),
                dict(semi=s[1], dur=0.24, gap=0.0, amp=0.9, scoop=0.8, bend=-0.8, vib=18, vib_rate=5.5,
                     att=0.015, rel=0.12)]
    if style == 'question':   # robot_notice: 'hm?' rising
        return [dict(semi=s[0], dur=0.1, gap=0.06, amp=0.75, scoop=0.8, att=0.01, rel=0.04),
                dict(semi=s[0] + 2, glide_to=s[1] + 3, dur=0.34, gap=0.0, amp=0.95, scoop=0.5,
                     vib=14, vib_rate=6.0, att=0.02, rel=0.1)]
    raise ValueError(style)


# =============================================================================
# SMALL FOLEY
# =============================================================================
def glass_tink(r, level=1.0, f0=None, dur=0.35, hard=1.0):
    """One tiny glass shard ringing."""
    if f0 is None:
        f0 = np.exp(r.uniform(np.log(2600), np.log(6500)))
    ratios = np.array([1.0, 1.58, 2.37, 3.11, 4.06]) * (1 + r.uniform(-0.04, 0.04, 5))
    t60 = np.array([0.22, 0.14, 0.1, 0.07, 0.05]) * r.uniform(0.6, 1.5) * (dur / 0.35)
    y = modal(dur, f0 * ratios, t60, r.uniform(0.3, 1.0, 5) * [1, .8, .6, .5, .4], r)
    t = tvec(len(y))
    y += hard * 0.6 * hp(noise(len(y), r), 4000) * np.exp(-t / 0.0005)
    return level * fades(y, 0.0001, 0.01)


def glass_crunch(r, level=1.0, dur=0.25):
    n = n_of(dur)
    y = np.zeros(n)
    for _ in range(r.integers(3, 7)):
        i = n_of(r.uniform(0, 0.04))
        g = glass_tink(r, r.uniform(0.2, 0.7), dur=r.uniform(0.05, 0.15))
        k = min(len(g), n - i)
        y[i:i + k] += g[:k]
    t = tvec(n)
    y += 0.3 * crackle(n, 20000 * np.exp(-t / 0.01), r, band=(3000, 12000), grain_ms=0.2)
    return level * y


def water_drip(r, level=1.0):
    """Drop leaving the spout: tiny liquid 'plip' + faint hollow can resonance."""
    n = n_of(0.35)
    t = tvec(n)
    f = 1350 * (1 + 1.1 * (1 - np.exp(-t / 0.012)))
    y = np.sin(phase_of(f)) * np.exp(-t / 0.018) * np.clip(t / 0.001, 0, 1)
    y += 0.12 * fit(can_clink(r, 1.0, damp=2.5, dur=0.35), n)
    y += 0.3 * fit(micro_click(r, 0.4, f=(3800, 6100, 8900), t60=0.01, body=0.0), n)
    return level * fades(y, 0.0003, 0.02)


def drop_hit(r, level=1.0):
    """The single drop hits dry earth: tiny dry 'tk' + faint soaking hiss (~0.7 s)."""
    n = n_of(0.9)
    t = tvec(n)
    tk = bp(noise(n, r), 900, 5500) * np.exp(-t / 0.0012)
    tk += 0.5 * np.sin(TAU * 260 * t) * np.exp(-t / 0.004)
    tk += 0.35 * np.sin(TAU * 1850 * t) * np.exp(-t / 0.003)
    # soaking: fine fizz + a few micro-crackles as the dust drinks it
    e = np.clip(t / 0.03, 0, 1) * np.exp(-t / 0.22)
    fizz = bp(noise(n, r), 3500, 11000) * e
    cr = crackle(n, 700 * e, r, band=(3500, 10000), grain_ms=0.15)
    hiss = 0.03 * fizz / (np.std(fizz[:n_of(0.3)]) + 1e-9) + 0.025 * cr / (np.std(cr[:n_of(0.3)]) + 1e-9)
    return level * fades(tk + hiss, 0.0002, 0.05)


def leaf_snap(r, level=1.0):
    """Brittle dead leaf snapping off the stalk (+ stalk quiver)."""
    n = n_of(0.45)
    t = tvec(n)
    y = np.zeros(n)
    for k in range(r.integers(2, 4)):
        i = n_of(r.uniform(0, 0.008)) if k else 0
        m = n - i
        c = bp(noise(m, r), 1800, 9000) * np.exp(-tvec(m) / 0.0015) * r.uniform(0.5, 1.0)
        c += 0.5 * modal(m / SR, np.array([2900.0, 4700.0, 6800.0]) * r.uniform(0.9, 1.1),
                         [0.012, 0.009, 0.006], [1, .7, .5], r)
        y[i:] += c
    # stalk quiver: small dry wooden rattles
    for _ in range(4):
        i = n_of(r.uniform(0.02, 0.2))
        m = n - i
        y[i:] += r.uniform(0.05, 0.15) * modal(m / SR, np.array([1150.0, 1930.0, 3100.0]) * r.uniform(0.9, 1.1),
                                               [0.02, 0.015, 0.01], [1, .6, .4], r)
    return level * fades(y, 0.0001, 0.03)


def leaf_skitter(r, dur=1.4, level=1.0):
    """Dead leaf tumbling over dry cracked ground; recedes (level/LP) over dur."""
    n = n_of(dur)
    t = tvec(n)
    y = np.zeros(n)
    tt = 0.03
    while tt < dur - 0.05:
        i = n_of(tt)
        slide = r.random() < 0.25
        ln = r.uniform(0.04, 0.11) if slide else r.uniform(0.006, 0.02)
        m = min(n_of(ln), n - i)
        e = np.sin(np.pi * np.linspace(0, 1, m)) ** (0.8 if slide else 2)
        g = crackle(m, 7000 if slide else 15000, r, band=(r.uniform(1500, 2800), r.uniform(6000, 9500)),
                    grain_ms=0.3)
        y[i:i + m] += g * e * r.uniform(0.4, 1.0) / (np.std(g) + 1e-9)
        tt += r.uniform(0.035, 0.13)
    dist = np.exp(-t / (dur * 0.55))
    y = tvf(y * dist, 9000 * dist + 1500, 0.7, 'lp')
    return level * 0.3 * fades(y, 0.002, 0.08)


def soil_scrape(r, dur=0.2, level=1.0, bright=1.0):
    """Metal fingers raking dry soil: distinct grains over a low 'shff'."""
    n = n_of(dur)
    t = tvec(n)
    x = t / dur
    e = np.sin(np.pi * np.clip(x, 0, 1)) ** 0.6 * (1 + 0.35 * np.sin(TAU * r.uniform(16, 28) * t))
    g = crackle(n, 4200 * e, r, band=(900 * bright, 8000 * bright), grain_ms=0.9, amp_sigma=1.0)
    g = g / (np.std(g) + 1e-9)
    body = tvf(noise(n, r, 'pink'), 380 + 600 * np.sin(np.pi * x), 1.4, 'bp')
    body = body / (np.std(body) + 1e-9)
    tick = micro_click(r, 0.35, f=(2400, 3900, 5600), t60=0.03, body=0.1)
    y = 0.55 * g * e + 0.3 * body * e
    y[:len(tick)] += tick[:n]
    return level * 0.35 * fades(y, 0.004, 0.03)


def dirt_trickle(r, dur=0.5, level=1.0):
    n = n_of(dur)
    t = tvec(n)
    dens = 1200 * np.exp(-t / (dur * 0.35))
    g = crackle(n, dens, r, band=(900, 7000), grain_ms=0.4)
    return level * 0.25 * fades(g / (np.std(g[:n_of(0.1)]) + 1e-9), 0.003, 0.05)


def soil_thunk(r, level=1.0, can=True):
    """Soil clump with the sprout set into the tin can: muffled thud + hollow tin + grit."""
    n = n_of(0.7)
    t = tvec(n)
    f = 135 * (1 + 0.5 * np.exp(-t / 0.006))
    y = np.sin(phase_of(f)) * np.exp(-t / 0.035) * np.clip(t / 0.002, 0, 1)
    y += 0.6 * lp(noise(n, r), 600) * np.exp(-t / 0.02)
    if can:
        y += 0.5 * can_clink(r, 1.0, damp=3.0, dur=0.7)
    g = crackle(n, 6000 * np.exp(-t / 0.03), r, band=(800, 6000), grain_ms=0.5)
    y += 0.15 * g / (np.std(g[:n_of(0.05)]) + 1e-9)
    return level * fades(y, 0.0005, 0.05)


def soil_pat(r, level=1.0):
    n = n_of(0.3)
    t = tvec(n)
    y = lp(noise(n, r), 700) * np.exp(-t / 0.014) * 1.2
    y += 0.8 * np.sin(TAU * r.uniform(95, 120) * t) * np.exp(-t / 0.025)
    g = crackle(n, 5000 * np.exp(-t / 0.015), r, band=(900, 5000), grain_ms=0.5)
    y += 0.1 * g / (np.std(g[:n_of(0.03)]) + 1e-9)
    y += 0.15 * fit(micro_click(r, 1.0, f=(2600, 4100, 6300), t60=0.015, body=0.0), n)
    return level * fades(y, 0.0005, 0.05)


def sub_thump(r, f0=62.0, f1=38.0, dur=0.9, level=1.0, knock=0.3, body=0.4, mid=0.0):
    """Deep soft pulse (heartbeat / soil pulse): pitch-falling sine + soft body.
    mid > 0 adds a short 150-600 Hz 'presence' thud so the pulse translates on small speakers."""
    n = n_of(dur)
    t = tvec(n)
    f = f1 + (f0 - f1) * np.exp(-t / 0.05)
    y = np.sin(phase_of(f)) * ar(n, 0.012, dur - 0.012, curve=2.5)
    y += body * np.sin(phase_of(f * 2.02)) * np.exp(-t / 0.06) * np.clip(t / 0.006, 0, 1)
    y += knock * lp(noise(n, r), 260) * np.exp(-t / 0.02) * np.clip(t / 0.004, 0, 1)
    if mid > 0:
        m = bp(noise(n, r, 'pink'), 150, 600, 2) * np.exp(-t / 0.028) * np.clip(t / 0.005, 0, 1)
        m += 0.6 * np.sin(phase_of(f * 4.0)) * np.exp(-t / 0.045) * np.clip(t / 0.006, 0, 1)
        y += mid * m / (np.max(np.abs(m)) + 1e-9)
    return level * fades(y, 0.002, 0.06)


# =============================================================================
# helpers
# =============================================================================
def unit(x):
    return x / (np.std(x) + 1e-9)


def lag(x, k):
    k = int(k)
    if k <= 0:
        return x
    return np.concatenate([np.full(k, x[0]), x[:-k]])


def add_at(dst, src, t, sr=SR):
    """Add src (mono or stereo) into dst at time t (s); clips at the end."""
    i = n_of(t)
    if i >= dst.shape[-1]:
        return dst
    if i < 0:
        src = src[..., -i:]
        i = 0
    m = min(src.shape[-1], dst.shape[-1] - i)
    if m > 0:
        dst[..., i:i + m] += src[..., :m]
    return dst


# =============================================================================
# WIND / WEATHER
# =============================================================================
def wind_bed(n, seed, g, lag_s=0.35, body=1.0, howl=0.22, hiss=0.3, rumble=0.25,
             body_f=(240.0, 820.0), body_q=0.9, howl_f=(380.0, 820.0), howl_q=7.0,
             hiss_band=(2800.0, 11000.0), hiss_rate=(1500.0, 14000.0), hiss_floor=0.12, hiss_pow=1.8,
             body_lp=None):
    """Band-passed noise wind with gusts (stereo). g = gust intensity (0..~1.5).
    Right channel leads (wind blows from screen right)."""
    out = []
    for ch in range(2):
        r = rng(f'{seed}-{ch}')
        gg = np.clip(g if ch == 1 else lag(g, n_of(lag_s)), 0, 2.5)
        bd = tvf(noise(n, r, 'pink'), body_f[0] + (body_f[1] - body_f[0]) * np.minimum(gg, 1.6), body_q, 'bp')
        if body_lp:
            bd = lp(bd, body_lp, 2)
        y = body * unit(bd) * (0.25 + 0.75 * gg) ** 1.5
        if howl > 0:
            hf = howl_f[0] + (howl_f[1] - howl_f[0]) * np.minimum(gg, 1.6) + 50 * (smooth_rand(n, 0.6, r) - 0.5)
            y += howl * unit(tvf(noise(n, r), hf, howl_q, 'bp')) * gg ** 2.2
        if hiss > 0:
            cr = crackle(n, hiss_rate[0] + hiss_rate[1] * gg ** 2, r, band=hiss_band, grain_ms=0.25, amp_sigma=0.8)
            y += hiss * unit(cr) * (hiss_floor + (1 - hiss_floor) * gg ** hiss_pow)
        if rumble > 0:
            y += rumble * unit(lp(noise(n, r, 'brown'), 110, 4)) * (0.5 + 0.5 * gg)
        out.append(y)
    return np.stack(out)


def whistle(n, r, f0, g=None, q=45.0, drift=0.04):
    """Wind whistling through a gap in the ruins: narrowband noise, pitch rides the gust."""
    wob = 1 + drift * (smooth_rand(n, 0.8, r) - 0.5) + 0.012 * (smooth_rand(n, 6, r) - 0.5)
    f = f0 * wob
    if g is not None:
        f = f * (1 + 0.12 * (np.clip(g, 0, 1.5) - 0.5))
    x = noise(n, r)
    return unit(tvf(tvf(x, f, q, 'bp'), f, q * 0.6, 'bp'))


def metal_creak(r, dur=1.2, base=1.0, rate=(10.0, 45.0, 18.0), level=1.0, bright=1.0, tail=0.8, grit=0.4):
    """Rusty structure groaning (stick-slip through metal modes)."""
    n = n_of(dur + tail)
    nd = n_of(dur)
    rc = np.full(n, rate[-1], float)
    rc[:nd] = env_pts(nd, [(0, rate[0]), (dur * 0.45, rate[1]), (dur, rate[2])])
    fr = np.array([173.0, 297.0, 452.0, 668.0, 931.0, 1310.0, 1795.0, 2460.0]) * base * (1 + r.uniform(-0.04, 0.04, 8))
    t60 = np.array([0.9, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2])
    gains = np.array([0.6, 1.0, 0.8, 0.7, 0.5, 0.35, 0.25, 0.15]) * bright ** np.linspace(-1, 1.5, 8)
    env = np.zeros(n)
    env[:nd] = np.sin(np.pi * np.linspace(0, 1, nd)) ** 0.6
    y = friction((dur + tail), rc, (fr, t60, gains), r, jitter=0.3, grit=grit, env=env)
    y = hp(y, 80)
    return level * fades(normalize(y), 0.01, 0.15)


def hinge_creak(r, dur=0.45, level=1.0):
    """Small rusty hinge squeak: fast stick-slip, nasal resonances, pitch rises then falls."""
    n = n_of(dur + 0.2)
    nd = n_of(dur)
    rc = np.full(n, 150.0)
    rc[:nd] = env_pts(nd, [(0, 90), (dur * 0.3, 310), (dur * 0.7, 380), (dur, 140)])
    fr = np.array([1180.0, 1720.0, 2650.0, 3480.0, 4620.0]) * r.uniform(0.95, 1.05)
    y = friction(dur + 0.2, rc, (fr, [0.05, 0.04, 0.035, 0.03, 0.02], [1, .8, .7, .45, .3]), r, jitter=0.12,
                 grit=0.15, amp_sigma=0.25,
                 env=np.concatenate([np.sin(np.pi * np.linspace(0, 1, nd)) ** 0.5, np.zeros(n - nd)]))
    y = hp(y, 400)
    return level * fades(normalize(y), 0.005, 0.05)


def wood_groan(r, dur=1.2, level=1.0, rate=(20.0, 70.0, 30.0), base=1.0, tail=0.3):
    """Living wood straining / creaking (lower Q than metal)."""
    n = n_of(dur + tail)
    nd = n_of(dur)
    rc = np.full(n, rate[-1], float)
    rc[:nd] = env_pts(nd, [(0, rate[0]), (dur * 0.5, rate[1]), (dur, rate[2])])
    fr = np.array([142.0, 251.0, 397.0, 611.0, 890.0, 1270.0, 1810.0]) * base * (1 + r.uniform(-0.05, 0.05, 7))
    env = np.zeros(n)
    env[:nd] = np.sin(np.pi * np.linspace(0, 1, nd)) ** 0.5
    y = friction(dur + tail, rc, (fr, [0.14, 0.12, 0.1, 0.08, 0.06, 0.045, 0.035], [0.7, 1, .9, .7, .5, .35, .2]),
                 r, jitter=0.4, grit=0.6, env=env)
    return level * fades(normalize(hp(y, 70)), 0.01, 0.08)


def wood_crackle(r, dur, rate_env, level=1.0):
    """Wood fibres cracking / snapping: sharp bursts through short resonances."""
    n = n_of(dur)
    x = poisson_impulses(n, rate_env, r, amp_sigma=1.1)
    y = resonate(x, np.array([780.0, 1340.0, 2210.0, 3350.0, 5100.0]) * r.uniform(0.9, 1.1),
                 [0.02, 0.016, 0.012, 0.009, 0.006], [0.6, 1, .9, .7, .5])
    y += 0.5 * hp(signal.lfilter([1.0], [1, -0.6], x), 1500)
    return level * fades(normalize(y), 0.002, 0.05)


def whoosh(r, dur, peak=0.45, f=(400.0, 1600.0, 600.0), q=1.1, pan_path=(0.0, 0.0), pass_w=0.25,
           hiss=0.35, low=0.25, air=0.3, w=0.5, level=1.0, rel_curve=3.0):
    """Swept band-noise whoosh (stereo). Pan moves fastest around the peak."""
    n = n_of(dur)
    t = tvec(n)
    x = t / dur
    e = env_pts(n, [(0, 0.0), (dur * peak * 0.55, 0.22), (dur * peak, 1.0),
                    (dur * (peak + (1 - peak) * 0.3), 0.45), (dur, 0.0)])
    e = np.clip(e, 0, 1) ** 1.2
    fc = env_pts(n, [(0, f[0]), (dur * peak, f[1]), (dur, f[2])])
    body = unit(tvf(noise(n, r, 'pink'), fc, q, 'bp'))
    y = body * e
    if air > 0:
        y += air * unit(tvf(noise(n, r), np.minimum(fc * 2.6, 15000), 3.0, 'bp')) * e ** 1.5
    if hiss > 0:
        eh = lag(e, n_of(0.06))
        y += hiss * unit(crackle(n, 2000 + 20000 * eh, r, band=(3500, 12000), grain_ms=0.2)) * eh
    if low > 0:
        y += low * unit(lp(noise(n, r, 'brown'), 160, 2)) * e
    p = pan_path[0] + (pan_path[1] - pan_path[0]) * (0.5 + 0.5 * np.tanh((x - peak) / max(pass_w, 1e-3) * 2.0))
    st = pan(y, p)
    if w > 0:
        dec = decorrelate(y, r, 1.0)
        st = st * (1 - 0.35 * w) + 0.35 * w * dec * np.array([[1], [1]]) * 0.7
    return level * fades(st, 0.005, 0.04)


# =============================================================================
# ROBOT - electrical / core
# =============================================================================
def zap(r, dur=0.3, level=1.0, f0=118.0):
    """Electrical arc: unstable buzzing discharge, gated in jagged bursts."""
    n = n_of(dur)
    t = tvec(n)
    f = f0 * (1 + 0.15 * (smooth_rand(n, 25, r) - 0.5))
    buzz = additive(f, [1.0 / k ** 0.6 for k in range(1, 90)], rng_=r)
    nz = hp(noise(n, r), 900)
    gate = onepole((smooth_rand(n, 90, r) > 0.4).astype(float), 900)
    e = np.clip(t / 0.002, 0, 1) * np.exp(-t / (dur * 0.45))
    y = (0.55 * unit(buzz) + 0.7 * unit(nz)) * gate * e
    y = soft_clip(y * 2.2, 1.5)
    y += 0.5 * unit(crackle(n, 5000 * e, r, band=(2500, 14000), grain_ms=0.15)) * e
    y = peak_eq(y, 3200, 5, 1.0)
    return level * fades(normalize(y), 0.0003, 0.02)


def sparks(r, dur=0.9, rate=40.0, level=1.0, pan_from=0.0, pan_to=-0.6):
    """Spark crackles: sharp ticks with tiny sizzles, thinning out; drift in stereo."""
    n = n_of(dur)
    y = np.zeros((2, n))
    tt = 0.0
    while True:
        lam = rate * np.exp(-tt / (dur * 0.4))
        tt += r.exponential(1.0 / max(lam, 1e-3))
        if tt > dur - 0.05:
            break
        m = n_of(r.uniform(0.005, 0.04))
        e = np.exp(-tvec(m) / (m / SR / 3))
        s = hp(noise(m, r), r.uniform(2500, 5000)) * e
        s[:n_of(0.0004)] += r.choice([-1, 1]) * 2.5
        s *= r.lognormal(0, 0.5) * np.exp(-tt / (dur * 0.6))
        p = pan_from + (pan_to - pan_from) * (tt / dur) + r.uniform(-0.25, 0.25)
        add_at(y, pan(s, p), tt)
    return level * fades(y, 0.0002, 0.03)


def core_hum(r, dur, f0=55.0, level=1.0, bright=0.5, detune=0.6, env=None, air=0.15):
    """The robot's heart: warm harmonic hum with a slow chorus beat."""
    n = n_of(dur)
    t = tvec(n)
    drift = 1 + 0.0025 * (smooth_rand(n, 2.5, r) - 0.5)
    amps = [1.0, 0.62, 0.48, 0.34, 0.24, 0.17, 0.12, 0.085, 0.06, 0.04]
    amps = [a * (bright ** (0.6 * k)) if k > 2 else a for k, a in enumerate(amps)]
    f = f0 * drift
    y = additive(f, amps, rng_=r) + 0.8 * additive(f * (1 + detune / f0), amps, rng_=r)   # 'detune' Hz beat
    y = unit(y)
    if air > 0:
        a = unit(tvf(noise(n, r), 1800 + 600 * np.sin(TAU * 0.3 * t), 2.0, 'bp'))
        y += air * a
    if env is not None:
        y *= env
    return level * fades(y, 0.02, 0.05)


def heartbeat(r, level=1.0, dub=0.62, gap=0.21, f=(68.0, 44.0), mid=0.35):
    y = np.zeros(n_of(0.95))
    add_at(y, sub_thump(r, f[0], f[1], 0.55, knock=0.35, body=0.55, mid=mid), 0.0)
    add_at(y, dub * sub_thump(r, f[0] * 1.1, f[1] * 1.08, 0.5, knock=0.25, body=0.45, mid=mid * 0.8), gap)
    return level * y


def power_down(r, dur=0.9, level=1.0, f_start=1150.0, f_end=48.0, stutters=(0.36, 0.55)):
    """Descending power-down whine (flywheel spinning down) with stutter dropouts."""
    n = n_of(dur)
    t = tvec(n)
    x = t / dur
    f = f_start * (f_end / f_start) ** (x ** 0.85)
    hic = np.zeros(n)
    gate = np.ones(n)
    for s in stutters:
        i0, i1 = n_of(s * dur), n_of(s * dur + 0.038)
        gate[i0:i1] = 0.04
        hic += 0.12 * np.exp(-((t - s * dur - 0.045) / 0.02) ** 2)
    f = f * (1 + hic)
    gate = onepole(gate, 350)
    amps = [1.0, 0.0, 0.33, 0.0, 0.2, 0.05, 0.12, 0.04, 0.08]
    y = unit(additive(f, amps, rng_=r))
    y += 0.4 * unit(servo(r, dur, 90, 260, 'down', whine=0.0, grit=0.05)) * (1 - x)
    y = tvf(y, np.maximum(f * 5, 250), 0.7, 'lp')
    e = (1 - 0.65 * x ** 1.5) * ar(n, 0.01, 0.1, hold=dur - 0.11)
    y = y * e * gate
    return level * fades(normalize(y), 0.003, 0.03)


# =============================================================================
# MAGIC: shimmers, sparkles, energy
# =============================================================================
def sparkles(r, dur, rate_env, fband=(3000.0, 9000.0), gdur=(0.006, 0.035), rise=0.0, spread=0.8, level=1.0):
    """Tiny sine grains (glints of light). rate_env: grains/s (array len n or scalar).
    rise: semitones the band climbs over the event."""
    n = n_of(dur)
    rate_env = np.broadcast_to(np.asarray(rate_env, float), (n,))
    y = np.zeros((2, n))
    cum = np.cumsum(rate_env) / SR
    k = int(cum[-1])
    if k == 0:
        return y
    targets = np.sort(r.uniform(0, cum[-1], k))
    idx = np.searchsorted(cum, targets)
    for i in idx:
        tt = i / SR
        sh = 2 ** (rise * (tt / dur) / 12.0)
        f0 = np.exp(r.uniform(np.log(fband[0]), np.log(fband[1]))) * sh
        gd = r.uniform(*gdur)
        m = n_of(gd)
        tg = tvec(m)
        g = np.sin(TAU * f0 * tg + r.random() * TAU) * np.sin(np.pi * tg / gd) ** 2
        g *= r.lognormal(0, 0.5)
        add_at(y, pan(g, r.uniform(-spread, spread)), tt)
    return level * y


def shimmer(r, dur, f0, ratios=(1.0, 2.0, 2.76, 3.5, 4.2, 5.4), glide=2.0, att=0.35, rel=0.6,
            twinkle=(3.0, 11.0), spread=0.7, detune=0.004, grains=0.0, grain_band=(3500.0, 9000.0),
            level=1.0, amp_tilt=0.8):
    """Glassy shimmering partials (stereo) with a pitch glide + optional sparkle grains."""
    n = n_of(dur)
    t = tvec(n)
    x = t / dur
    gl = 2 ** (glide * smoothstep(x) / 12.0)
    env = ar(n, att, rel, hold=max(dur - att - rel, 0), curve=1.8)
    y = np.zeros((2, n))
    for k, ra in enumerate(ratios):
        for d in (-1, 1):
            fk = f0 * ra * (1 + d * detune * r.uniform(0.5, 1.5)) * gl
            if np.max(fk) > 0.45 * SR:
                continue
            tw = 0.55 + 0.45 * np.sin(TAU * r.uniform(*twinkle) * t + r.random() * TAU)
            a = (amp_tilt ** k) * tw
            s = np.sin(phase_of(fk) + r.random() * TAU) * a
            p = ((-1) ** k) * spread * (0.4 + 0.6 * r.random()) * d
            y += pan(s, p)
    y *= env / (len(ratios) ** 0.5)
    if grains > 0:
        y += grains * sparkles(r, dur, 60 * env / env.max(), grain_band, rise=glide) * 0.5
    return level * fades(y, 0.01, 0.05)


def riser(r, dur, f=(600.0, 7000.0), q=2.5, level=1.0, curve=2.0, w=0.6):
    """Filtered-noise riser (stereo)."""
    n = n_of(dur)
    x = np.linspace(0, 1, n)
    fc = f[0] * (f[1] / f[0]) ** (x ** 1.2)
    e = x ** curve
    out = []
    for ch in range(2):
        rr = rng(int(r.integers(1 << 30)))
        out.append(unit(tvf(noise(n, rr, 'pink'), fc, q, 'bp')) * e)
    st = np.stack(out)
    st = width(st, w)
    return level * fades(st, 0.01, 0.01)


# =============================================================================
# GLASS / IMPACTS
# =============================================================================
def glass_burst(r, dur=3.2, level=1.0, n_shards=460):
    """A glass dome apex exploding outward: crack, dense crash, then a long rain of shards."""
    n = n_of(dur)
    y = np.zeros((2, n))
    t = tvec(n)
    # initial crack / shatter body
    crack = hp(noise(n_of(0.2), r), 700) * np.exp(-tvec(n_of(0.2)) / 0.012)
    crack[:n_of(0.001)] *= 3
    add_at(y, pan(crack, 0.0) * 1.3, 0.0)
    body = crackle(n_of(0.6), 40000 * np.exp(-tvec(n_of(0.6)) / 0.08), r, band=(2500, 16000), grain_ms=0.3)
    add_at(y, decorrelate(unit(body) * 0.35 * np.exp(-tvec(n_of(0.6)) / 0.12), r), 0.0)
    for k in range(n_shards):
        if r.random() < 0.55:
            tt = r.exponential(0.07)                       # the burst
            lvl = r.lognormal(0, 0.5) * 0.8
            p = r.uniform(-0.6, 0.6)
        else:
            tt = 0.25 + r.gamma(2.2, 0.35)                 # falling & landing
            lvl = r.lognormal(0, 0.6) * 0.45 * np.exp(-(tt - 0.25) / 1.4)
            p = r.uniform(-1.0, 1.0)
        if tt > dur - 0.2:
            continue
        g = glass_tink(r, lvl, f0=np.exp(r.uniform(np.log(2200), np.log(11000))), dur=r.uniform(0.06, 0.55),
                       hard=r.uniform(0.3, 1.0))
        add_at(y, pan(g, p), tt)
    return level * fades(y, 0.0003, 0.2)


def boom(r, dur=3.0, level=1.0, f=(72.0, 27.0)):
    """Deep impact boom: falling sub + air pressure body + a little crack for definition."""
    n = n_of(dur)
    t = tvec(n)
    fr = f[1] + (f[0] - f[1]) * np.exp(-t / 0.18)
    sub = np.sin(phase_of(fr)) * np.exp(-t / 0.75) * np.clip(t / 0.004, 0, 1)
    bodyn = lp(noise(n, r, 'brown'), 220, 2)
    bodyn = unit(bodyn) * np.exp(-t / 0.35) * np.clip(t / 0.003, 0, 1)
    mid = unit(bp(noise(n, r, 'pink'), 120, 700)) * np.exp(-t / 0.09)
    crk = unit(hp(noise(n, r), 1800)) * np.exp(-t / 0.006)
    y = 1.0 * sub + 0.45 * bodyn + 0.35 * mid + 0.25 * crk
    return level * fades(normalize(y), 0.0005, 0.3)


def leaf_burst(r, dur=2.2, level=1.0, att=0.04):
    """A canopy of leaves exploding outward / thrashing (stereo)."""
    n = n_of(dur)
    t = tvec(n)
    e = np.clip(t / att, 0, 1) * np.exp(-t / (dur * 0.33))
    out = []
    for ch in range(2):
        rr = rng(int(r.integers(1 << 30)))
        c = crackle(n, 3000 + 60000 * e, rr, band=(1100, 10000), grain_ms=0.35, amp_sigma=0.9)
        c = unit(c) * (e * (0.8 + 0.2 * smooth_rand(n, 25, rr)))
        c += 0.5 * unit(tvf(noise(n, rr, 'pink'), 900 + 1500 * e, 0.9, 'bp')) * e
        out.append(c)
    return level * fades(np.stack(out), 0.003, 0.2)


def rustle(n, seed, g, band=(1200.0, 9500.0), dens=(1500.0, 26000.0), body=0.35, flutter=6.0):
    """Leaves in a breeze (stereo bed): granular rustle, density & level follow g (0..1)."""
    out = []
    for ch in range(2):
        r = rng(f'{seed}-{ch}')
        gg = np.clip(g, 0, 1.5) * (0.85 + 0.3 * smooth_rand(n, flutter, r))
        c = crackle(n, dens[0] + dens[1] * gg, r, band=band, grain_ms=0.3, amp_sigma=0.9)
        y = unit(c) * gg ** 1.3
        if body > 0:
            y += body * unit(tvf(noise(n, r, 'pink'), 500 + 700 * np.minimum(gg, 1.2), 0.8, 'bp')) * gg ** 1.5
        out.append(y)
    return np.stack(out)


# =============================================================================
# WATER
# =============================================================================
def stream(n, seed, rate=260.0, level=1.0, flow=0.35, splash=0.25):
    """Babbling brook: rising-pitch bubble resonances (van den Doel) + flow + splash."""
    r = rng(seed)
    y = np.zeros((2, n))
    dur = n / SR
    k = int(rate * dur)
    tt = np.sort(r.uniform(0, dur, k))
    gurgle = smooth_rand(n, 1.3, r, 0.35, 1.0)
    for t0 in tt:
        if r.random() > gurgle[min(n - 1, n_of(t0))]:
            continue
        rad = np.exp(r.uniform(np.log(0.0007), np.log(0.006)))      # m
        f0 = 3.0 / rad
        d = 0.043 * f0 + 0.0014 * f0 ** 1.5
        L = min(5.0 / d, 0.08)
        m = n_of(L)
        tb = tvec(m)
        fb = f0 * (1 + 0.1 * d * tb)
        a = (rad / 0.003) ** 0.9 * r.lognormal(0, 0.4)
        s = a * np.sin(phase_of(fb)) * np.exp(-d * tb)
        add_at(y, pan(fades(s, 0.0003, 0.002), r.uniform(-0.7, 0.7)), t0)
    y = unit(y) * 1.0
    for ch in range(2):
        rr = rng(f'{seed}-flow-{ch}')
        y[ch] += flow * unit(lp(noise(n, rr, 'pink'), 1400, 2)) * smooth_rand(n, 0.9, rr, 0.6, 1.0)
        y[ch] += splash * unit(crackle(n, 900 * gurgle, rr, band=(3000, 11000), grain_ms=0.25))
    return level * y


# =============================================================================
# BIRDS
# =============================================================================
def _syl(r, fpts, dur, h2=0.07, fm_rate=0.0, fm_depth=0.0, am_rate=0.0, att=0.15, rel=0.35, level=1.0):
    n = max(n_of(dur), 16)
    t = tvec(n)
    f = env_pts(n, [(p[0] * dur, p[1]) for p in fpts])
    if fm_rate:
        f = f + fm_depth * np.sin(TAU * fm_rate * t + r.random() * TAU)
    f = np.clip(f, 200, 11000)
    ph = phase_of(f)
    y = np.sin(ph) + h2 * np.sin(2 * ph + 0.4)
    x = t / (n / SR)
    a = np.sin(0.5 * np.pi * np.clip(x / att, 0, 1)) * np.cos(0.5 * np.pi * np.clip((x - (1 - rel)) / rel, 0, 1))
    if am_rate:
        a *= 0.55 + 0.45 * np.sin(TAU * am_rate * t)
    return level * y * a


def bird_syllable(r, kind, s=1.0):
    """Songbird syllables built from FM / pitch envelopes. s scales pitch."""
    u = r.uniform
    if kind == 'tsip':
        return _syl(r, [(0, 7400 * s * u(.9, 1.1)), (1, 4700 * s)], u(0.028, 0.045))
    if kind == 'chip':
        return _syl(r, [(0, 6800 * s), (0.3, 5400 * s), (1, 3700 * s)], u(0.018, 0.03), h2=0.15, att=0.08)
    if kind == 'tweet':
        a = u(3000, 3600) * s
        return _syl(r, [(0, a), (0.45, a * 1.6), (1, a * 1.35)], u(0.07, 0.11))
    if kind == 'whistle':
        a = u(2600, 3900) * s
        return _syl(r, [(0, a), (1, a * u(0.9, 1.12))], u(0.14, 0.28), fm_rate=u(20, 30), fm_depth=a * 0.012,
                    att=0.25, rel=0.4)
    if kind == 'seee':
        a = u(6800, 8200) * s
        return _syl(r, [(0, a), (1, a * u(0.96, 1.03))], u(0.18, 0.35), att=0.3, rel=0.5, h2=0.02)
    if kind == 'warble':
        a = u(3300, 4500) * s
        return _syl(r, [(0, a * 0.95), (1, a * 1.05)], u(0.12, 0.26), fm_rate=u(28, 46), fm_depth=u(350, 700) * s)
    if kind == 'churr':
        a = u(3900, 4600) * s
        return _syl(r, [(0, a), (1, a * 0.93)], u(0.12, 0.2), am_rate=u(55, 80), h2=0.15)
    if kind == 'trill':
        k = int(r.integers(7, 15))
        rate = u(13, 22)
        hi, lo = u(5200, 6400) * s, u(3600, 4400) * s
        out = []
        for j in range(k):
            out.append(_syl(r, [(0, hi), (1, lo)], 0.7 / rate, att=0.1, rel=0.5,
                            level=0.75 + 0.25 * np.sin(np.pi * j / k)))
            out.append(np.zeros(n_of(0.3 / rate)))
        return np.concatenate(out)
    if kind == 'teacher':
        a = u(4900, 5500) * s
        return np.concatenate([_syl(r, [(0, a), (1, a * 0.97)], 0.075), np.zeros(n_of(0.035)),
                               _syl(r, [(0, a * 0.78), (1, a * 0.7)], 0.095)])
    if kind == 'flute':          # mellow blackbird-like note with a glide
        a = u(1900, 2800) * s
        return _syl(r, [(0, a * 0.92), (0.3, a * 1.05), (1, a * u(0.85, 1.1))], u(0.2, 0.38),
                    fm_rate=u(5, 8), fm_depth=a * 0.01, att=0.2, rel=0.45, h2=0.12)
    if kind == 'twee?':         # rising, curious
        a = u(3300, 3700) * s
        return _syl(r, [(0, a), (0.6, a * 1.2), (1, a * 1.55)], u(0.11, 0.15), att=0.2, rel=0.3)
    raise ValueError(kind)


SPECIES = {
    # name: (phrase recipe generator, pitch scale)
    'robin': 1.0, 'wren': 1.0, 'tit': 1.0, 'finch': 1.0, 'blackbird': 1.0, 'small': 1.08,
}


def bird_phrase(r, species, s=1.0):
    """One song phrase for a 'species' (mono)."""
    parts = []

    def gap(a, b):
        parts.append(np.zeros(n_of(r.uniform(a, b))))

    if species == 'robin':
        for _ in range(int(r.integers(4, 8))):
            parts.append(bird_syllable(r, r.choice(['whistle', 'seee', 'trill', 'warble', 'tsip', 'tweet'],
                                                   p=[.28, .14, .16, .2, .1, .12]), s))
            gap(0.03, 0.09)
    elif species == 'wren':
        for _ in range(3):
            parts.append(bird_syllable(r, 'chip', s)); gap(0.03, 0.05)
        parts.append(bird_syllable(r, 'trill', s * 1.05)); gap(0.02, 0.05)
        parts.append(bird_syllable(r, 'churr', s)); gap(0.03, 0.06)
        parts.append(bird_syllable(r, 'trill', s * 0.95))
    elif species == 'tit':
        for _ in range(int(r.integers(3, 5))):
            parts.append(bird_syllable(r, 'teacher', s)); gap(0.06, 0.09)
    elif species == 'finch':
        k = int(r.integers(8, 13))
        for j in range(k):
            hi = 6000 * s * (1 - 0.3 * j / k)
            parts.append(_syl(r, [(0, hi), (1, hi * 0.72)], 0.045 - 0.015 * j / k, att=0.1, rel=0.5))
            gap(0.025 - 0.012 * j / k, 0.035 - 0.015 * j / k)
        parts.append(bird_syllable(r, 'tweet', s * 1.1)); gap(0.02, 0.03)
        parts.append(bird_syllable(r, 'chip', s))
    elif species == 'blackbird':
        for _ in range(int(r.integers(3, 5))):
            parts.append(bird_syllable(r, 'flute', s)); gap(0.05, 0.12)
        parts.append(bird_syllable(r, 'trill', s * 1.2) * 0.5)
    elif species == 'small':
        for _ in range(int(r.integers(2, 4))):
            parts.append(bird_syllable(r, r.choice(['tsip', 'tweet', 'seee'], p=[.5, .35, .15]), s)); gap(0.05, 0.12)
    else:
        raise ValueError(species)
    return np.concatenate(parts)


def wing_flaps(r, dur, rate=15.0, level=1.0, size=1.0, glide=0.12, env=None):
    """Bird wing beats: each flap a soft feathery noise burst ('fff')."""
    n = n_of(dur)
    y = np.zeros(n)
    tt = r.uniform(0, 0.02)
    while tt < dur - 0.05:
        if r.random() < glide:
            tt += r.uniform(0.08, 0.2)
            continue
        m = n_of(0.6 / rate)
        tb = tvec(m)
        e = np.clip(tb / 0.006, 0, 1) * np.exp(-tb / (0.25 / rate))
        s = bp(noise(m, r), 280 / size, 3400 / size) * e
        s += 0.3 * bp(noise(m, r), 3000, 8000) * e ** 2
        add_at(y, s * r.uniform(0.7, 1.0), tt)
        tt += (1.0 / rate) * r.uniform(0.88, 1.12)
    if env is not None:
        y *= env[:n]
    return level * fades(y, 0.002, 0.03)


# =============================================================================
# picture-synced composites (v2: after the picture landed)
# =============================================================================
def bloom_hit(r, level=1.0, root=None):
    """The heart flares out of the chest (one frame of light): a warm, soft, rich hit.
    Onset at sample 0 (<3 ms attack): low 'whoom' + warm band body with a small swell as
    the light floods the frame (~0.3 s) + an open-fifth glow on D + air + rivet glints."""
    root = HUM_HZ if root is None else root                  # D2
    n = n_of(1.8)
    t = tvec(n)
    att = np.clip(t / 0.003, 0, 1)
    f = root * 1.2 * (1 + 0.5 * np.exp(-t / 0.04))          # ~88 -> 66 Hz
    low = np.sin(phase_of(f)) * np.exp(-t / 0.28) * att
    body = unit(bp(noise(n, r, 'pink'), 140, 1300, 2))
    benv = att * (0.75 * np.exp(-t / 0.09) + 0.35 * np.exp(-0.5 * ((t - 0.2) / 0.1) ** 2)) * np.exp(-t / 0.9)
    body = tvf(body * benv, 2600 * np.exp(-t / 0.5) + 350, 0.7, 'lp')
    glow = np.zeros(n)
    for k, (mult, a) in enumerate(((2, 1.0), (3, 0.7), (4, 0.6), (6, 0.35), (8, 0.22))):   # D3 A3 D4 A4 D5
        fk = root * mult * (1 + 0.0025 * (smooth_rand(n, 3, r) - 0.5))
        glow += a * np.sin(phase_of(fk) + r.random() * TAU)
    glow = glow / 2.2 * att * np.exp(-t / 0.75) * (1 + 0.25 * np.exp(-0.5 * ((t - 0.22) / 0.12) ** 2))
    air = unit(hp(noise(n, r), 3000)) * np.clip(t / 0.006, 0, 1) * np.exp(-t / 0.16)
    glints = np.zeros(n)
    for k in range(5):
        i = n_of(r.uniform(0.015, 0.14))
        g = modal(0.5, np.array([note('D7'), note('A7')]) * r.uniform(0.99, 1.01), [0.35, 0.2],
                  [1.0, 0.5], r) * r.uniform(0.3, 0.8)
        m = min(len(g), n - i)
        glints[i:i + m] += g[:m]
    y = 1.0 * low + 0.55 * body + 0.5 * glow + 0.12 * air + 0.05 * glints
    return level * fades(y, 0.0004, 0.2)


def heart_press(r, level=1.0):
    """Hands press the heart into dry cracked soil: crust crunch + warm sub pulse + a soft
    'whump' as the glow drains into the cracks (~0.13 s race) + soil grains hopping.
    Contact at sample 0."""
    n = n_of(1.3)
    t = tvec(n)
    y = np.zeros(n)
    # the crust gives: dense crunch (grains) + a few sharper crust cracks + low crumble
    cr = crackle(n, 26000 * np.exp(-t / 0.035), r, band=(600, 6500), grain_ms=0.6, amp_sigma=0.9)
    y += 0.45 * cr / (np.std(cr[:n_of(0.05)]) + 1e-9) * np.exp(-t / 0.05)
    for k in range(3):
        i = n_of(r.uniform(0.0, 0.03))
        c = bp(noise(n_of(0.02), r), 1500, 7000) * np.exp(-tvec(n_of(0.02)) / 0.002) * r.uniform(0.6, 1.2)
        y[i:i + len(c)] += c
    y += 0.5 * unit(bp(noise(n, r, 'pink'), 180, 900)) * np.exp(-t / 0.03) * np.clip(t / 0.002, 0, 1)
    # warm sub pulse (the heart meets the earth)
    y += 1.3 * fit(sub_thump(r, 68.0, 40.0, 1.2, knock=0.3, body=0.45, mid=0.3), n)
    # the whump: warm low-mid bloom that sinks (LP closes) as the light drains into the cracks
    w = unit(bp(noise(n, r, 'pink'), 110, 800))
    wenv = np.clip(t / 0.02, 0, 1) * np.exp(-t / 0.2)
    w = tvf(w * wenv, 1800 * np.exp(-t / 0.12) + 220, 0.7, 'lp')
    y += 0.8 * w
    # soil grains hop (f6-f10)
    x = poisson_impulses(n, 900 * np.exp(-t / 0.05), r, 0.8)
    y += 0.25 * resonate(x, [2400.0, 3900.0, 5600.0], [0.006, 0.004, 0.003], [1, .8, .6])
    return level * fades(y, 0.0003, 0.1)


def power_down_v2(r, dur, dropouts=(), steps=(), breath=None, level=1.0, f_start=1150.0, f_end=48.0):
    """Descending power-down whine whose stutters follow the eye's drawn flickers.
    dropouts: [(t0, t1, gain)] relative to the start (gain 0 = out, 0.5 = half);
    steps: [(t, gain)] level after each flicker ('weaker each time');
    breath: (t0, t1) the last brightening just before the click."""
    n = n_of(dur)
    t = tvec(n)
    x = t / dur
    f = f_start * (f_end / f_start) ** (x ** 0.85)
    gate = np.ones(n)
    hic = np.zeros(n)
    for (a, b, g) in dropouts:
        gate[(t >= a) & (t < b)] = max(g, 0.03)
        hic += 0.1 * np.exp(-((t - b - 0.01) / 0.015) ** 2)             # pitch hiccup as it catches again
    lvl = np.ones(n)
    for (a, g) in steps:
        lvl[t >= a] *= g
    if breath is not None:
        a, b = breath
        bump = np.exp(-((t - 0.5 * (a + b)) / max(b - a, 0.01)) ** 2)
        lvl *= 1 + 0.5 * bump
        hic += 0.06 * bump
    f = f * (1 + hic)
    gate = onepole(gate, 400)
    amps = [1.0, 0.0, 0.33, 0.0, 0.2, 0.05, 0.12, 0.04, 0.08]
    y = unit(additive(f, amps, rng_=r))
    y += 0.4 * unit(servo(r, dur, 90, 260, 'down', whine=0.0, grit=0.05)) * (1 - x)
    y = tvf(y, np.maximum(f * 5, 250), 0.7, 'lp')
    e = (1 - 0.6 * x ** 1.5) * ar(n, 0.01, 0.06, hold=dur - 0.07)
    y = y * e * gate * lvl
    return level * fades(normalize(y), 0.003, 0.02)
