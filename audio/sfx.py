"""
THE LAST SEED - sound design stem.   audio/sfx.py  ->  audio/sfx.wav

    python audio/sfx.py            # render audio/sfx.wav (+ stems audio/sfx_amb|robot|fx.wav)
    python audio/sfx.py --review   # ...and verification plots / level report in review/sfx_*

48 kHz, stereo, float32, exactly 60.000 s. Every event time is derived from
cues.CUES (edit cues.py and re-run to retime). Derived sub-timings (step
cadences, offsets inside a gesture) are written relative to their cue.

Signal flow:  event -> (pan, level) -> stem dry bus  +  send -> per-space
convolution reverb (desert / storm / glass / valley) -> stem.
Stems: amb (ambience beds + weather), robot (steps, servos, voice, core),
fx (foley & story events). Global: storm hard-cut, TOTAL SILENCE gate,
start/end fades, look-ahead limiter (same gain applied to every stem, so
amb + robot + fx == sfx.wav exactly).
"""
import argparse
import os
import sys
import time

sys.dont_write_bytecode = True          # keep audio/ free of __pycache__
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
for _p in (PROJ, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from scipy.io import wavfile

from cues import CUES
from sfx_dsp import *          # noqa: F401,F403  (SR, n_of, filters, reverb, limiter ...)
from sfx_lib import *          # noqa: F401,F403  (the sound library)
import sfx_lib as L
import sfx_sync
from sfx_sync import pan_of_x

T = dict(CUES)
DUR = 60.0                     # contractual file length (2,880,000 samples)
TE = min(T.get('end', DUR), DUR)   # end of picture (fades land here)
N = int(round(DUR * SR))       # 2,880,000
assert N == 2_880_000

OUT_WAV = os.path.join(HERE, 'sfx.wav')
STEM_WAV = {s: os.path.join(HERE, f'sfx_{s}.wav') for s in ('amb', 'robot', 'fx')}
REVIEW = os.path.join(PROJ, 'review')

CEILING_DB = -1.2              # true-peak ceiling of the safety limiter (4x oversampled detection)
HUM = L.HUM_HZ                 # D2. Pitch plan + the single transpose knob (PITCH_SHIFT) live in sfx_lib.py:
note = L.note                  # every pitched SFX below is written as a note name.

# ---------------------------------------------------------------------------------------------
# Screen positions -> pan (-1 left .. +1 right). Where the picture exports a position (walks, the
# kneel, the storm crest, the bird, the eye) it is read from the shot via SY (sfx_sync); these
# are the remaining hand-placed ones.
# ---------------------------------------------------------------------------------------------
SY = {}                        # picture sync (sfx_sync.picture_sync), rebuilt at every render()
PAN = {
    's1_robot_a': -0.55, 's1_robot_b': -0.15,   # tiny robot crossing the wide shot
    's2_robot': -0.25, 's2_stalk': 0.05,
    's3a_enter': -0.8, 's3a_stop': -0.42, 'glint': 0.44,
    's3b_robot': 0.12,
    's4_robot': -0.35,
    's5_robot': -0.1,
    's6_robot': -0.32, 's6_can': -0.18,
    's7_robot': 0.1, 's7_planter': 0.35,
    'center': 0.0,
}


# =============================================================================================
# MIXER
# =============================================================================================
class Mixer:
    SPACES = ('desert', 'storm', 'glass', 'valley')

    def __init__(self):
        self.dry = {s: np.zeros((2, N)) for s in ('amb', 'robot', 'fx')}
        self.send = {}
        self.log = []

    def add(self, name, sig, t, stem='fx', space='desert', send=0.2, pan_=0.0, peak_db=None, rms_db=None,
            gain_db=0.0, cue=None, cut_at=None, cut_fade=0.07, kind=None):
        """Place a sound. mono -> panned; stereo -> balanced toward pan_ (if nonzero).
        peak_db / rms_db set the absolute dry level (dBFS); gain_db trims."""
        sig = np.asarray(sig, float)
        if sig.ndim == 1:
            st = pan(sig, pan_)
        else:
            st = balance(sig, pan_) if (np.ndim(pan_) > 0 or pan_ != 0) else sig.copy()
        if peak_db is not None:
            st = st * (undb(peak_db) / (np.max(np.abs(st)) + 1e-12))
        elif rms_db is not None:
            st = st * (undb(rms_db) / (np.sqrt(np.mean(st ** 2) * 2) + 1e-12))
        st = st * undb(gain_db)
        st = fades(st, 0.0008, 0.004)
        i0 = n_of(t)
        if cut_at is not None:                              # hard-cut the tail at a picture cut
            k = n_of(cut_at) - i0
            if k <= 0:
                return
            if st.shape[1] > k:
                st = st[:, :k]
                f = min(n_of(cut_fade), k)
                st[:, k - f:] *= np.linspace(1, 0, f) ** 2
        add_at(self.dry[stem], st, t)
        if space and send > 0:
            key = (stem, space)
            if key not in self.send:
                self.send[key] = np.zeros((2, N))
            add_at(self.send[key], st * send, t)
        pk = float(np.max(np.abs(st))) if st.size else 0.0
        a = np.max(np.abs(st), axis=0)
        if kind is None:
            kind = 'bed' if name.startswith('bed:') else 'hit'
        thr = {'hit': -24, 'swell': -40, 'bed': -40}[kind]
        on = int(np.argmax(a > pk * undb(thr))) if pk > 0 else 0
        pki = int(np.argmax(a))
        self.log.append(dict(name=name, t=float(t), cue=cue, stem=stem, peak_db=float(db(pk)), kind=kind,
                             onset=float(t + on / SR), peak_t=float(t + pki / SR), dur=st.shape[1] / SR))


def db_curve(pts):
    """Gain automation in dB through (t, dB) points (smooth) -> linear (N,)."""
    return undb(env_pts(N, pts, smooth=True))


def bump(t0, w, a=1.0):
    """Gaussian bump on the film timeline (for gust curves)."""
    t = tvec(N)
    return a * np.exp(-0.5 * ((t - t0) / w) ** 2)


def seg(t0, t1, fin=0.05, fout=0.05):
    """1 inside [t0, t1] with raised-cosine edges (N,)."""
    t = tvec(N)
    a = np.clip((t - t0) / max(fin, 1e-4), 0, 1)
    b = np.clip((t1 - t) / max(fout, 1e-4), 0, 1)
    return (0.5 - 0.5 * np.cos(np.pi * a)) * (0.5 - 0.5 * np.cos(np.pi * b))


# =============================================================================================
# 1. AMBIENCE BEDS
# =============================================================================================
def build_ambience(mx):
    t = tvec(N)
    r = rng('amb')
    s5, s6, s7 = T['cut_s5'], T['cut_s6'], T['cut_s7']

    # ---------------- dead world wind: 0 -> storm hit ------------------------------------------
    g = 0.30 + 0.22 * smooth_rand(N, 0.12, r) + 0.08 * smooth_rand(N, 0.9, r)
    # S1: the wind follows the picture's own gusts (s01.gust: the sand streams and cloth ride it)
    tg = np.arange(0.0, T['cut_s2'] + 1.0, 0.01)
    gpic = np.interp(t, tg, np.array([SY['s1_gust'](x) for x in tg]))
    w1 = 1 - smoothstep((t - (T['cut_s2'] - 0.4)) / 0.8)
    g = w1 * (0.22 + 0.46 * gpic + 0.05 * smooth_rand(N, 0.9, r)) + (1 - w1) * g
    g += bump(T['leaf_breaks'] + 0.35, 0.45, 0.28)                      # the leaf is torn away
    g += bump(T['gust'] + 0.35, 0.35, 0.55)                             # THE gust (plus whoosh event)
    g += bump(T['cut_s4'] + 1.6, 0.9, 0.15)
    g *= 1 - 0.45 * np.exp(-0.5 * ((t - (T['drop_fall'] + 0.4)) / 0.55) ** 2)   # wind holds its breath
    g += 0.95 * smoothstep((t - T['storm_rise']) / (s5 - T['storm_rise'])) ** 1.6
    win = wind_bed(N, 'deadwind', g, lag_s=0.35, howl=0.18, hiss=0.2, rumble=0.45, body_f=(170.0, 600.0),
                   body_lp=2600.0, hiss_floor=0.03, hiss_pow=2.6, howl_f=(330.0, 700.0))
    suck = 1 - 0.75 * seg(s5 - 0.15, s5 - 0.012, 0.07, 0.012)           # breath before the storm hits
    auto = db_curve([(0.0, -60), (T['fade_in'] + 0.3, -30), (T['fade_in'] + 2.4, 0), (T['cut_s2'] - 0.05, 0),
                     (T['cut_s2'] + 0.15, -3), (T['drop_fall'] - 0.3, -3), (T['drop_fall'] - 0.05, -7.5),
                     (T['drop_hit'] + 0.7, -7.5), (T['droop'] + 0.3, -3),          # the wind holds its breath
                     (T['cut_s3'], -3), (T['cut_s3'] + 0.5, -1),              # the 0.5 s dissolve into S3
                     (T['cut_c01_eye'] - 0.05, -1),
                     (T['cut_c01_eye'] + 0.12, -5), (T['cut_c02_sprout'] - 0.04, -5),
                     (T['cut_c02_sprout'] + 0.25, -13), (T['cut_s3b'] - 0.05, -13), (T['cut_s3b'] + 0.3, -2.5),
                     (T['cut_s4'], -2.0), (T['storm_rise'], -2.0), (s5, 3.0)])
    win *= auto * suck * seg(-1, s5 + 0.06, 0.1, 0.12)
    mx.add('bed: dead-world wind', win, 0.0, 'amb', 'desert', 0.12, rms_db=-33.5, cue='fade_in')

    # whistles through the ruins (narrowband, riding the gusts)
    for (t0, t1, f0, p, lv) in [(T['fade_in'] + 1.6, T['cut_s2'] - 0.4, note('E6'), 0.35, -44),
                                (T['cut_s2'] + 0.3, T['cut_s3'] - 0.4, note('D6'), -0.5, -53),
                                (T['cut_s4'] + 0.4, T['storm_rise'] + 0.5, note('C6'), 0.55, -45),
                                (T['look_down_can'] + 0.1, s5, note('A5'), -0.3, -47)]:
        n = n_of(t1 - t0)
        w = whistle(n, rng(f'wh{t0}'), f0, g=g[n_of(t0):n_of(t0) + n], q=50)
        w *= ar(n, (t1 - t0) * 0.35, (t1 - t0) * 0.45, hold=(t1 - t0) * 0.2) * g[n_of(t0):n_of(t0) + n] ** 1.2
        w *= auto[n_of(t0):n_of(t0) + n]
        mx.add(f'whistle {f0:.0f}', w, t0, 'amb', 'desert', 0.35, pan_=p, rms_db=lv, cut_at=s5 + 0.05)

    # distant metal creaks (toppled turbine, pylons, the fence post)
    # S1: the toppled turbine's loose blade rocks in 8-fps stick-slip jerks (s01.turbine_rotor);
    # every drawn jerk creaks, harder when the blade moves faster
    t_a, t_b = T['fade_in'], T['cut_s2']
    fps = SY['s1_rotor_fps']
    ks = np.arange(int(np.floor(t_a * fps)), int(np.ceil(t_b * fps)) + 1)
    angs = np.array([SY['s1_rotor_deg']((k + 1e-3) / fps) for k in ks])
    dang = np.abs(np.diff(angs, prepend=angs[0]))
    nT = n_of(t_b - t_a + 1.2)
    rate = np.zeros(nT)
    envr = np.zeros(nT)
    for k, dA in zip(ks, dang):
        if dA < 0.2 * dang.max():
            continue
        i0 = n_of(k / fps - t_a)
        i1 = min(nT, i0 + n_of(0.09))
        if i0 >= nT or i1 <= i0:
            continue
        rate[i0:i1] = 18 + 85 * dA / dang.max()
        envr[i0:i1] = np.maximum(envr[i0:i1], (dA / dang.max()) ** 1.4 *
                                 np.sin(np.pi * np.clip(tvec(i1 - i0) / 0.09, 0, 1)) ** 0.5)
    fr_ = np.array([118.0, 205.0, 331.0, 489.0, 702.0, 980.0, 1350.0])
    rot = friction(nT / SR, rate, (fr_, [1.0, 0.9, 0.7, 0.55, 0.45, 0.35, 0.3], [0.5, 1, .9, .7, .5, .35, .2]),
                   rng('rotor'), jitter=0.3, grit=0.5, env=envr)
    rot = lp(hp(rot, 90), 2400) * np.clip(tvec(nT) / 1.6, 0, 1)            # rises with the picture's fade-in
    mx.add('turbine blade creaks (8-fps jerks)', rot, t_a, 'amb', 'desert', 0.6,
           pan_=pan_of_x(SY['s1_turbine_x'](0.5 * (t_a + t_b))), peak_db=-37)
    for (t0, d, base, p, lv, rate) in [(T['cut_s2'] + 0.7, 0.9, 1.35, 0.45, -44, (15, 55, 25)),
                                       (T['cut_s4'] + 1.0, 1.3, 0.9, 0.7, -40, (9, 34, 14)),
                                       (T['cut_s4'] + 3.3, 1.0, 1.2, -0.6, -41, (14, 48, 20))]:
        c = metal_creak(rng(f'mc{t0}'), d, base, rate, bright=0.7)
        c = lp(c, 2600, 2)
        mx.add(f'distant creak', c, t0, 'amb', 'desert', 0.6, pan_=p, peak_db=lv)

    # dry stalk rattling in the gusts (S2, medium shot)
    n = n_of(T['cut_s3'] - T['cut_s2'])
    gs = g[n_of(T['cut_s2']):n_of(T['cut_s2']) + n]
    x = poisson_impulses(n, 4 + 60 * np.clip(gs - 0.35, 0, 1) ** 2, rng('stalk'), 0.7)
    rat = resonate(x, [1150.0, 1930.0, 3100.0, 4450.0], [0.02, 0.016, 0.012, 0.008], [1, .7, .5, .35])
    rat *= ar(n, 0.3, 0.2, hold=n / SR - 0.5)
    mx.add('stalk rattle', rat, T['cut_s2'], 'amb', 'desert', 0.2, pan_=PAN['s2_stalk'], peak_db=-40)

    # ---------------- storm approach (distant rumble from the right) 24.0 -> 26.0 ---------------
    d = s5 - T['storm_rise'] + 0.3
    n = n_of(d)
    tt = tvec(n)
    rr = rng('rumble')
    rum = np.stack([unit(lp(noise(n, rng(f'rum{c}'), 'brown'), 95, 4)) for c in range(2)])
    roar = np.stack([unit(bp(noise(n, rng(f'roar{c}'), 'pink'), 160, 700)) for c in range(2)])
    e = (0.3 * smoothstep(tt / 0.35) + 0.7 * smoothstep(tt / (s5 - T['storm_rise'])) ** 1.4) * \
        (0.85 + 0.15 * smooth_rand(n, 2.0, rr))
    st = (rum + 0.35 * roar * e) * e
    st = balance(st, 0.45)
    st *= seg(-1, d - 0.02, 0.1, 0.25)[:n] * suck[n_of(T['storm_rise']):n_of(T['storm_rise']) + n]
    mx.add('storm rumble swell', st, T['storm_rise'], 'amb', 'desert', 0.25, rms_db=-27, cue='storm_rise',
           kind='swell')
    roll = lp(boom(rng('roll'), 2.2, f=(48.0, 30.0)), 300, 2) + 0.4 * lp(unit(noise(n_of(2.2), rng('roll2'), 'brown'))
                                                                        * ar(n_of(2.2), 0.25, 1.6, 0.2), 200, 2)
    mx.add('distant storm roll (horizon)', roll, T['storm_rise'], 'amb', 'desert', 0.5, pan_=0.55, peak_db=-24)

    # ---------------- THE STORM 26 -> 32 --------------------------------------------------------
    t0 = s5 - 0.25
    n = n_of(s6 - t0)
    tt = tvec(n) + t0
    gs = (0.95 + 0.12 * smooth_rand(n, 1.5, rng('sg')) + 0.1 * smooth_rand(n, 5.0, rng('sg2')))
    def sb(tc, w, a):
        return a * np.exp(-0.5 * ((tt - tc) / w) ** 2)
    gs += sb(s5 + 0.15, 0.25, 0.45) + sb(T['shield'] + 0.3, 0.4, 0.6) + sb(T['antenna_rip'] - 0.1, 0.3, 0.25) \
        + sb(T['core_flicker'] - 0.4, 0.4, 0.2) + sb(T['trudge'] + 0.6, 0.5, 0.25)
    gs *= 1 - 0.25 * np.exp(-0.5 * ((tt - (T['core_flicker'] + 0.15)) / 0.25) ** 2)   # let the core stutter through
    gs *= 1 - 0.4 * np.exp(-0.5 * ((tt - (T['antenna_rip'] + 0.12)) / 0.16) ** 2)     # pocket for the clang
    gs += 0.6 * smoothstep((tt - T['trudge']) / (s6 - T['trudge']))                   # -> whiteout
    gs *= smoothstep((tt - t0) / 0.25)
    gs *= suck[n_of(t0):n_of(t0) + n]
    storm = wind_bed(n, 'storm', gs, lag_s=0.12, body=1.0, howl=0.45, hiss=0.75, rumble=0.9,
                     body_f=(300.0, 1250.0), body_q=0.65, howl_f=(430.0, 1000.0), howl_q=6.0,
                     hiss_band=(2200.0, 13000.0), hiss_rate=(4000.0, 30000.0))
    # a tearing upper roar layer
    tear = np.stack([unit(tvf(noise(n, rng(f'tear{c}'), 'pink'), 1400 + 900 * np.clip(gs, 0, 2), 1.4, 'bp'))
                     for c in range(2)]) * np.clip(gs, 0, 2) ** 2 * 0.35
    storm = balance(storm + tear, 0.28)
    mx.add('bed: STORM', storm, t0, 'amb', 'storm', 0.35, rms_db=-17.5, cue='cut_s5', cut_at=s6, cut_fade=0.075)

    # sand pelting the robot's shell (tiny metallic ticks), follows gusts
    x = poisson_impulses(n, 25 + 260 * np.clip(gs - 0.6, 0, 2) ** 1.5, rng('pelt'), 0.8)
    pel = resonate(x, L.BODY_MODES[3:] * 1.4, L.BODY_T60[3:] * 0.35, [1, .8, .6, .5, .4])
    mx.add('sand pelting shell', pel, t0, 'amb', 'storm', 0.2, pan_=pan_of_x(SY['s5_x']), peak_db=-30, cut_at=s6)

    # ---------------- NIGHT IN THE GREENHOUSE 32 -> 38 -----------------------------------------
    n = n_of(s7 - s6 + 0.05)
    tt = tvec(n)
    gg = 0.35 + 0.35 * smooth_rand(n, 0.25, rng('ng'))
    night = wind_bed(n, 'night', gg, lag_s=0.2, body=1.0, howl=0.0, hiss=0.05, rumble=0.5,
                     body_f=(160.0, 420.0), body_q=0.8)
    night = lp(night, 900, 2)
    panes = np.stack([whistle(n, rng('pane1'), note('A6'), gg, q=70) * 0.5 * smooth_rand(n, 0.4, rng('pa')),
                      whistle(n, rng('pane2'), note('E7'), gg, q=80) * 0.4 * smooth_rand(n, 0.35, rng('pb'))])
    night = night + 0.22 * panes
    f0_, f1_ = SY['c05']['fade']                                          # the picture fades to black
    night *= smoothstep(tt / 0.8) * (1 - smoothstep((tt + s6 - f0_) / (f1_ - f0_)))
    mx.add('bed: greenhouse night', night, s6, 'amb', 'glass', 0.5, rms_db=-46, cue='cut_s6')

    # ---------------- GREY DAWN (cold thin air) 38 -> 44.5 --------------------------------------
    t_off = T['silence'] - 0.08
    n = n_of(t_off - s7)
    tt = tvec(n) + s7
    air = np.stack([unit(bp(noise(n, rng(f'air{c}'), 'pink'), 1800, 9000)) for c in range(2)])
    air *= 0.8 + 0.2 * smooth_rand(n, 0.3, rng('am'))
    low = np.stack([unit(lp(noise(n, rng(f'dl{c}'), 'brown'), 240, 2)) for c in range(2)]) * 0.9
    ring = np.stack([whistle(n, rng('dr1'), note('D7'), q=90), whistle(n, rng('dr2'), note('A7'), q=90)]) * 0.12
    dawn = air + low + ring
    ea = db_curve([(0, 0), (T['cut_c03_core'] - 0.02, 0), (T['cut_c03_core'] + 0.1, -4), (T['cut_s7b'] - 0.02, -4),
                   (T['cut_s7b'] + 0.1, -1), (T['cut_c08_heart'] - 0.02, -1), (T['cut_c08_heart'] + 0.08, -4),
                   (T['power_down'], -4), (T['eye_off'], -13), (t_off, -30)])
    dawn *= ea[n_of(s7):n_of(s7) + n]
    dawn *= smoothstep((tt - s7) / 0.6) * np.clip((t_off - tt) / 0.06, 0, 1)   # S7 fades in from black (0.6 s)
    mx.add('bed: grey dawn air', dawn, s7, 'amb', 'glass', 0.35, rms_db=-44, cue='cut_s7')

    # ---------------- THE ALIVE WORLD 48 -> 60 --------------------------------------------------
    t0 = T['cut_s8b_climax']
    n = n_of(DUR - t0)
    tt = tvec(n) + t0
    ga = 0.5 + 0.18 * smooth_rand(n, 0.35, rng('ga')) + 0.08 * smooth_rand(n, 1.3, rng('ga2'))
    ga += 0.55 * np.exp(-np.clip(tt - t0, 0, None) / 1.2) * smoothstep((tt - t0) / 0.15)   # canopy settles
    leaves = rustle(n, 'leaves', ga, band=(1300.0, 9500.0), dens=(1200.0, 14000.0), body=0.35)
    breeze = np.stack([unit(lp(noise(n, rng(f'br{c}'), 'pink'), 650, 2)) for c in range(2)]) * \
        (0.4 + 0.6 * lag(ga, n_of(0.3))) * 0.55
    alive = leaves + breeze
    ea = db_curve([(0, -60), (t0 - 0.01, -60), (t0 + 0.05, -3), (T['color_wave'], -2), (T['sun_break'], 0),
                   (T['cut_s9a'], 0), (T['cut_c04'] - 0.05, 0), (T['cut_c04'] + 0.2, 1.5), (T['title'], 1.0),
                   (TE - 1.8, -1), (TE - 0.6, -14), (TE, -60)])
    alive *= ea[n_of(t0):]
    mx.add('bed: warm breeze in leaves', alive, t0, 'amb', 'valley', 0.25, rms_db=-33, cue='cut_s8b_climax')

    # meadow grass rippling (valley) - slow travelling swells
    t9 = T['cut_s9a']
    n = n_of(DUR - t9)
    tt = tvec(n)
    gm = 0.45 + 0.4 * smooth_rand(n, 0.5, rng('gm'))
    grass = rustle(n, 'grass', gm, band=(900.0, 7000.0), dens=(4000.0, 30000.0), body=0.2, flutter=3.0)
    grass = balance(grass, 0.0) * db_curve([(0, -60), (t9, -30), (t9 + 0.6, 0), (T['cut_c04'] - 0.05, 0),
                                              (T['cut_c04'] + 0.2, -7), (TE - 1.7, -9), (TE - 0.4, -30),
                                              (TE, -60)])[n_of(t9):]
    mx.add('bed: meadow grass', grass, t9, 'amb', 'valley', 0.2, rms_db=-40, cue='cut_s9a')

    # the stream (from the valley shot; off-screen & closer-quieter in the close-up)
    st = stream(n, 'stream', rate=240.0)
    st = lp(st, 9000, 2)
    st *= db_curve([(0, -60), (t9, -40), (t9 + 0.6, 0), (T['cut_c04'] - 0.05, 0), (T['cut_c04'] + 0.25, -7),
                    (TE - 1.7, -9), (TE - 0.3, -40), (TE, -60)])[n_of(t9):]
    mx.add('bed: stream', st, t9, 'amb', 'valley', 0.3, pan_=0.25, rms_db=-35, cue='cut_s9a')


# =============================================================================================
# 2. THE ROBOT (steps, servos, voice, core) - locked to the drawn frames via sfx_sync (SY)
# =============================================================================================
def step_at(mx, name, tc, pan_, peak_db, surface='dirt', space='desert', send=0.2, weight=1.0, can=0.0,
            drag=0.0, glass=0.0, lp_hz=None, cue=None, cut_at=None, seed=None, clank=None):
    """One footfall whose CONTACT lands exactly at tc (the drawn contact frame)."""
    r = rng(seed or f'{name}-{tc:.3f}')
    s = footstep(r, surface, weight=weight, clank=r.uniform(0.7, 1.2) if clank is None else clank,
                 grit=r.uniform(0.8, 1.2), can=can, drag=drag, glass=glass)
    lead = 0.0
    if drag > 0:                     # footstep() puts the contact after its drag lead-in: find it
        lead = float(np.argmax(np.abs(s) > 0.5 * np.max(np.abs(s)))) / SR - 0.001
    if lp_hz:
        s = lp(s, lp_hz, 2)
    mx.add(name, s, tc - lead, 'robot', space, send, pan_=pan_, peak_db=peak_db, cue=cue, cut_at=cut_at)


def lub_dub(mx, name, t_lub, t_dub, peak_db, pan_=0.0, space='glass', send=0.2, seed='hb', muffle=None, mid=0.35,
            cue=None):
    """One heartbeat with explicit lub / dub times (the picture's heart clock)."""
    r = rng(seed)
    y = np.zeros(n_of(t_dub - t_lub + 0.6))
    add_at(y, sub_thump(r, 68.0, 44.0, 0.55, knock=0.35, body=0.55, mid=mid), 0.0)
    add_at(y, 0.62 * sub_thump(r, 75.0, 47.5, 0.5, knock=0.25, body=0.45, mid=mid * 0.8), t_dub - t_lub)
    if muffle:
        y = lp(y, muffle, 2)
    mx.add(name, y, t_lub, 'robot', space, send, pan_=pan_, peak_db=peak_db, cue=cue)


def build_robot(mx):
    r = rng('robot')
    # ---------------- S1: tiny, far away. The drawn wide walk: contacts 0.4 + 0.5k (s01 CONTACTS) --
    for k, tc in enumerate(SY['s1_contacts']):
        vis = min(1.0, max(0.06, (tc - T['fade_in']) / 1.6)) ** 0.8        # the picture fades up 0-1.6 s
        step_at(mx, 'steps S1 (far)', tc, pan_of_x(SY['s1_x'](tc)),
                -31.0 + float(db(vis)) + rng(f's1lv{k}').uniform(-1.5, 1.5), 'dirt', 'desert', 0.55,
                weight=0.8 * (1.06 if k % 2 else 0.94), can=0.6, lp_hz=3500, seed=f's1step{k}')

    # ---------------- S2: the empty can -------------------------------------------------------
    p = PAN['s2_robot']
    mx.add('servo: idle head tilt', servo(rng('idle'), 0.3, 150, 260, 'bell', gear=0.4, bright=0.8),
           T['cut_s2'] + 0.55, 'robot', 'desert', 0.1, pan_=p, peak_db=-33)
    mx.add('servo: can raise', servo(rng('cr'), 0.62, 130, 320, 'bell', gear=0.6), T['can_raise'], 'robot',
           'desert', 0.12, pan_=p, peak_db=-23, cue='can_raise')
    mx.add('can handle creak', hinge_creak(rng('chc'), 0.3), T['can_raise'] + 0.1, 'fx', 'desert', 0.15,
           pan_=p + 0.1, peak_db=-38)
    mx.add('servo: can tilt', servo(rng('ct'), 0.5, 170, 410, 'up', gear=0.5), T['can_tilt'], 'robot', 'desert',
           0.12, pan_=p + 0.12, peak_db=-24, cue='can_tilt')
    C7 = SY['c07']
    for k, tj in enumerate(C7['jiggles'] + [C7['flick']]):                 # c07: the coaxing jiggles / flick
        mx.add('servo: coaxing jiggle', servo(rng(f'cj{k}'), 0.09, 200, 380, 'halt', gear=0.6, whine=0.3), tj,
               'robot', 'desert', 0.08, pan_=-0.35, peak_db=-34 + 2 * (k == 2))
    mx.add('servo: the arm gives up (sag)', servo(rng('sag'), C7['sag'][1] - C7['sag'][0] + 0.15, 110, 230, 'down',
                                                  gear=0.4, bright=0.7), C7['sag'][0], 'robot', 'desert', 0.1,
           pan_=-0.35, peak_db=-33)
    mx.add('servo: droop (sad, descending)', servo(rng('dr'), 0.85, 300, 880, 'down', gear=0.35, whine=0.2,
                                                   bright=0.8), T['droop'], 'robot', 'desert', 0.15, pan_=p,
           peak_db=-22, cue='droop')
    mx.add('droop settle', body_clank(rng('dsc'), 1.0, bright=0.7, dur=0.4, rattle=0.3), T['droop'] + 0.78, 'robot',
           'desert', 0.15, pan_=p, peak_db=-34)
    fl = zap(rng('fl1'), 0.07, f0=96) * 0.5 + 0.5 * core_hum(rng('fl1h'), 0.07, f0=HUM, air=0.0)
    mx.add('core weak flicker', fl, T['droop'] + 0.2, 'robot', 'desert', 0.1, pan_=p, peak_db=-42)

    # ---------------- S3a: the drawn walk contacts (walk anim @10 fps + WALK_PH), then 'startle' ---
    for k, (tc, x, leg) in enumerate(SY['s3_contacts']):
        vis = min(1.0, (tc - T['cut_s3']) / 0.5)                            # 0.5 s dissolve from S2
        lv = -23.0 + (0.8 if leg == 'near' else -0.8) - 2.5 * (1.0 - vis)
        step_at(mx, 'steps S3a', tc, pan_of_x(x), lv, 'dirt', 'desert', 0.25, can=1.0,
                weight=1.05 if leg == 'near' else 0.95, seed=f's3step{k}')
    sq, hop, land, settle = SY['s3_startle']
    p = pan_of_x(SY['s3_halt_x'])
    mx.add('startle: squash', servo(rng('sq'), 0.1, 150, 280, 'halt', gear=0.7, whine=0.2) +
           fit(0.4 * body_clank(rng('sqc'), 1.0, 0.7, 0.1, 0.2), n_of(0.1)), sq, 'robot', 'desert', 0.12, pan_=p,
           peak_db=-28, cue='robot_stop')
    mx.add('startle: head snaps (servo zip)', servo(rng('hs'), 0.14, 240, 620, 'halt', gear=0.6, whine=0.45),
           hop - 0.02, 'robot', 'desert', 0.15, pan_=p, peak_db=-21)
    mx.add('startle: hop lands', footstep(rng('hl'), 'dirt', weight=0.9, clank=1.4, can=1.0), land, 'robot',
           'desert', 0.2, pan_=p, peak_db=-24)
    mx.add('startle: settle', clicks(rng('hsc'), [0.0, 0.06], f=(1900, 3300, 5200), t60=0.03, body=0.4), settle,
           'robot', 'desert', 0.15, pan_=p, peak_db=-31)

    # ---------------- c01: the eye (close, intimate) ------------------------------------------
    t0 = T['cut_c01_eye']
    lens = servo(rng('lens'), T['eye_focus'] - t0 - 0.08, 620, 1350, 'bell', gear=0.7, whine=0.5, formant=3600,
                 bright=1.3)
    mx.add('lens focus whirr', lens, t0 + 0.02, 'robot', 'desert', 0.05, pan_=0.05, peak_db=-24, cue='cut_c01_eye',
           kind='swell')
    tk = np.arange(0.0, T['eye_focus'] - t0 - 0.15, 1 / 33.0)
    mx.add('focus ring detents', clicks(rng('det'), list(tk + rng('dj').uniform(0, 0.006, len(tk))),
                                        f=(4200, 6900, 9800), t60=0.008, body=0.0), t0 + 0.08, 'robot', 'desert',
           0.03, pan_=0.1, peak_db=-38)
    mx.add('iris clicks', clicks(rng('iris'), [0.0, 0.062, 0.118], f=(2800, 4600, 7300), t60=0.025, body=0.25),
           T['eye_focus'], 'robot', 'desert', 0.08, pan_=0.0, peak_db=-17, cue='eye_focus')
    mx.add('iris servo shake', servo(rng('ish'), 0.22, 900, 1200, 'bell', jerk=1.0, formant=3200, whine=0.6),
           T['eye_focus'] + 0.13, 'robot', 'desert', 0.05, pan_=0.0, peak_db=-30)

    # ---------------- c02: the fingers (hush) -------------------------------------------------
    d = T['leaf_touch'] - T['hand_enter']
    fing = servo(rng('fing'), d, 210, 300, 'flat', gear=0.4, whine=0.25, bright=0.8, jerk=0.5)
    fing *= ar(len(fing), 0.12, 0.1, hold=d - 0.25)
    mx.add('fingers servo (slow)', pan(fing, np.linspace(-0.6, -0.15, len(fing))), T['hand_enter'], 'robot',
           'desert', 0.08, peak_db=-33, cue='hand_enter', kind='swell')          # enters from the LEFT

    # ---------------- S3b: scoop (lean 0.1, strokes 0.3/0.5, lift 0.75, in can 1.1), rise_hold, hop ----
    p = pan_of_x(SY['s3b_kneel_x'])
    mx.add('servo: lean in to dig', servo(rng('ka'), 0.3, 160, 300, 'bell', gear=0.5), T['scoop'], 'robot',
           'desert', 0.1, pan_=p, peak_db=-30)
    for k, ts in enumerate(SY['scoop_strokes']):
        mx.add('servo: dig stroke', servo(rng(f'dst{k}'), 0.16, 180, 340, 'bell', gear=0.6), ts - 0.04, 'robot',
               'desert', 0.1, pan_=p, peak_db=-33)
    mx.add('servo: lift clump', servo(rng('lc'), 0.35, 150, 330, 'up', gear=0.5), SY['scoop_lift'] - 0.05, 'robot',
           'desert', 0.1, pan_=p, peak_db=-31)
    R = SY['rise']
    mx.add('servo: rise hugging the can', servo(rng('hug'), R['hop'] - R['grab'] + 0.05, 140, 280, 'bell', gear=0.5,
                                                bright=0.8), R['grab'], 'robot', 'desert', 0.1, pan_=p, peak_db=-31)
    mx.add('can grabbed (soil inside)', can_clink(rng('cgr'), 1.0, damp=2.6, dur=0.35), R['grab'], 'robot', 'desert',
           0.12, pan_=p - 0.08, peak_db=-33)
    mx.add('can against chest', can_clink(rng('cac'), 1.0, damp=2.5, dur=0.4) + 0.5 * body_clank(rng('cac2'), 1.0, 0.6,
                                                                                                    0.4, 0.1)[:n_of(0.4)],
           R['up'], 'robot', 'desert', 0.15, pan_=p, peak_db=-32)
    v = voice(motif_notes('happy'), rng('v-happy'), index=1.75, bright=1.1)
    mx.add('VOICE: happy beep (motif)', v, T['happy_beep'], 'robot', 'desert', 0.1, pan_=p, peak_db=-13,
           cue='happy_beep')
    bounce = body_clank(rng('bnc'), 1.0, bright=1.2, dur=0.3, rattle=0.8) + \
        0.7 * can_clink(rng('bnc2'), 1.0, damp=1.8, dur=0.3)
    mx.add('happy hop lands (rattle)', bounce, R['hop'], 'robot', 'desert', 0.15, pan_=p, peak_db=-30)

    # ---------------- S4: walk 20.0-22.5 / stop, look down, beep / walk 23.5-24.8 / look up ------
    p = pan_of_x(SY['s4_x'])
    for k, tc in enumerate(SY['s4_contacts']):
        step_at(mx, 'steps S4', tc, p, -23.0 + rng(f's4lv{k}').uniform(-1.5, 1.5), 'dirt', 'desert', 0.25, can=0.5,
                weight=1.06 if k % 2 else 0.94, cue='cut_s4' if k == 0 else None, seed=f's4step{k}')
    mx.add('stop (to look at the sprout)', footstep(rng('st1'), 'dirt', weight=0.6, clank=0.8, can=0.4),
           SY['s4_stop1'] + 0.03, 'robot', 'desert', 0.2, pan_=p, peak_db=-30)
    mx.add('servo: look down', servo(rng('ld'), 0.32, 150, 280, 'bell', gear=0.4, bright=0.8), T['look_down_can'],
           'robot', 'desert', 0.1, pan_=p, peak_db=-29)
    v = voice(motif_notes('soft'), rng('v-soft'), index=0.8, bright=0.8)
    mx.add('VOICE: soft beep', v, T['look_down_can'], 'robot', 'desert', 0.1, pan_=p, peak_db=-19,
           cue='look_down_can')
    mx.add('servo: looks up again', servo(rng('lu1'), 0.3, 150, 300, 'bell', gear=0.4, bright=0.8),
           SY['s4_lookup1'], 'robot', 'desert', 0.1, pan_=p, peak_db=-31)
    mx.add('stop scuff', footstep(rng('ssf'), 'dirt', weight=0.8, can=0.5), SY['s4_stop2'] + 0.03, 'robot',
           'desert', 0.2, pan_=p, peak_db=-27)
    mx.add('servo: look up (the storm)', servo(rng('lu'), 0.36, 150, 330, 'bell', gear=0.5), T['robot_looks_up'],
           'robot', 'desert', 0.12, pan_=p, peak_db=-23, cue='robot_looks_up')

    # ---------------- S5: brace (heavy steps), shoved, shield, stutter, rise, trudge ------------
    p = pan_of_x(SY['s5_x'])
    cut = T['cut_s6']
    mx.add('servo: brace strain', servo(rng('bs'), 0.6, 90, 210, 'bell', gear=0.9, whine=0.2, bright=0.7),
           T['cut_s5'] + 0.15, 'robot', 'storm', 0.2, pan_=p, peak_db=-24)
    for k, tc in enumerate(SY['s5_contacts']):
        trudge = tc >= SY['s5_brace2']
        step_at(mx, 'steps S5 heavy (brace)' if not trudge else 'steps S5 heavy (trudge)', tc, p,
                -15.0 if trudge else -17.0, 'sand', 'storm', 0.2, weight=1.6, drag=0.8 if trudge else 0.5,
                cut_at=cut, seed=f's5step{k}')
    shove = np.zeros(n_of(0.6))
    add_at(shove, body_clank(rng('shv'), 1.0, 1.1, 0.5, 0.9, size=1.1), 0.0)
    add_at(shove, 0.7 * can_clink(rng('shv2'), 1.0, damp=1.5, dur=0.4), 0.012)
    add_at(shove, 0.8 * servo(rng('shv3'), 0.3, 110, 260, 'halt', gear=0.9, whine=0.2), 0.0)
    mx.add('shoved by the surge', shove, SY['s5_front'], 'robot', 'storm', 0.25, pan_=p, peak_db=-19, cut_at=cut)
    mx.add('servo: turn & shield', servo(rng('sh'), 0.5, 120, 300, 'bell', gear=0.8, whine=0.3), T['shield'],
           'robot', 'storm', 0.2, pan_=p, peak_db=-22, cue='shield')
    mx.add('shield: slams down curled', body_clank(rng('shc'), 1.0, 1.0, 0.4, 0.6) +
           0.6 * can_clink(rng('shc2'), 1.0, 2.0, 0.4), T['shield'] + 0.3, 'robot', 'storm', 0.25, pan_=p,
           peak_db=-22)
    # core stutter: three quick electrical crackles with the heart-hum dropping out
    t0 = T['core_flicker']
    d = 0.75
    n = n_of(d)
    tt = tvec(n)
    gate = np.ones(n)
    for k, off in enumerate((0.0, 0.13, 0.27)):
        gate *= 1 - (1 - 0.05) * seg(t0 + off + 0.012, t0 + off + 0.07, 0.004, 0.01)[n_of(t0):n_of(t0) + n]
    dim = 1 - 0.55 * smoothstep((tt - 0.32) / 0.15)
    hum = core_hum(rng('stut'), d, f0=HUM * 2, bright=0.9, air=0.05) * gate * dim * ar(n, 0.02, 0.3, hold=0.4)
    mx.add('core hum (stutter)', hum, t0 - 0.03, 'robot', 'storm', 0.15, pan_=p, peak_db=-17, cut_at=cut)
    for k, off in enumerate((0.0, 0.13, 0.27)):
        mx.add(f'core crackle {k + 1}', zap(rng(f'cz{k}'), 0.07, f0=110 + 20 * k), t0 + off, 'robot', 'storm', 0.2,
               pan_=p, peak_db=-12 - 1.5 * k, cue='core_flicker' if k == 0 else None)
    # rises: 'unshield' turns back into the wind (30.0-30.5), then trudges (steps above)
    rise = np.zeros(n_of(0.8))
    add_at(rise, servo(rng('uns'), 0.5, 100, 280, 'up', gear=0.9, whine=0.25, bright=0.8), 0.0)
    add_at(rise, 0.7 * body_clank(rng('unsc'), 1.0, 0.9, 0.4, 0.6), 0.3)
    mx.add('rises & turns into the wind', rise, SY['s5_rise'], 'robot', 'storm', 0.2, pan_=p, peak_db=-20,
           cue='trudge', cut_at=cut)

    # ---------------- S6: limp in (walk_tired), sit (impact on the cue) ------------------------
    p = PAN['s6_robot']
    step_at(mx, 'S6 limp: hurt foot plants (in the dark)', SY['s6_hurt_contact'] + 0.005, p - 0.2, -31, 'stone',
            'glass', 0.35, weight=0.6, seed='limp0')
    mx.add('S6 limp: painful hitch (servo)', servo(rng('hitch'), 0.2, 90, 170, 'bell', gear=1.0, whine=0.1,
                                                    bright=0.7, jerk=1.0) +
           fit(0.5 * hinge_creak(rng('hitchc'), 0.18), n_of(0.2)), SY['s6_hitch'], 'robot', 'glass', 0.3,
           pan_=p - 0.2, peak_db=-30)
    step_at(mx, 'S6 limp: good leg lands', SY['s6_good_contact'], p - 0.15, -21, 'stone', 'glass', 0.35, weight=1.0,
            glass=1.0, seed='limp1')
    d0, d1 = SY['s6_drag']
    n = n_of(d1 - d0 + 0.05)
    tt = tvec(n)
    e = np.sin(np.pi * np.clip(tt / (n / SR), 0, 1)) ** 0.8
    drag = crackle(n, 6000 * e, rng('toe'), band=(700, 6000), grain_ms=0.7) + \
        0.6 * bp(noise(n, rng('toe2')), 500, 3000)
    mx.add('S6 limp: hurt toe drags on stone', unit(drag) * e, d0, 'robot', 'glass', 0.3, pan_=p - 0.1, peak_db=-30)
    mx.add('servo: sit', servo(rng('sit'), SY['s6_impact'] - SY['s6_sit_start'], 110, 240, 'down', gear=0.7,
                               whine=0.2), SY['s6_sit_start'], 'robot', 'glass', 0.2, pan_=p, peak_db=-30)
    sit = body_clank(rng('sitc'), 1.0, bright=0.9, dur=0.9, rattle=0.8, size=1.2)
    th = sub_thump(rng('sitt'), 125, 72, 0.35, knock=0.7, body=0.35)
    sitm = np.zeros(n_of(1.4))
    add_at(sitm, th, 0.0)
    add_at(sitm, 0.9 * sit, 0.004)
    add_at(sitm, 0.55 * metal_creak(rng('rib'), 0.25, 0.9, (20, 60, 25), tail=0.9, bright=0.8), 0.07)  # the iron rib
    add_at(sitm, 0.4 * body_clank(rng('sitc2'), 1.0, 0.8, 0.5, 0.5), 0.085)                          # back hits rib
    mx.add('SIT: heavy metal clank', sitm, SY['s6_impact'], 'robot', 'glass', 0.45, pan_=p, peak_db=-11,
           cue='sit_down')
    mx.add('rib settles (creak)', lp(metal_creak(rng('ribc'), 0.8, 0.75, (6, 22, 9), bright=0.6), 3000),
           SY['s6_impact'] + 0.45, 'fx', 'glass', 0.5, pan_=p - 0.15, peak_db=-33)

    # ---------------- c05: the lean (heart stutters & dips, notice + iris, the nod, the swell) --
    C5 = SY['c05']
    s0, s1 = C5['stutter']
    for k, (u, lv) in enumerate(((0.0, -35), (0.36, -40), (0.64, -36))):           # out / half / out
        mx.add('heart stutters (flicker)', zap(rng(f'hs{k}'), 0.05, f0=90) * 0.7, s0 + u * (s1 - s0), 'robot',
               'glass', 0.3, pan_=p + 0.1, peak_db=lv)
    for k, td in enumerate(C5['dips']):
        mx.add('heart dips (flicker)', zap(rng(f'hd{k}'), 0.05, f0=85) * 0.6, td - 0.04, 'robot', 'glass', 0.3,
               pan_=p + 0.1, peak_db=-41)
    mx.add('servo: turns to the can', servo(rng('lac'), 0.35, 140, 260, 'bell', gear=0.4, bright=0.8),
           T['robot_notice'] - 0.05, 'robot', 'glass', 0.25, pan_=p, peak_db=-33)
    v = voice(motif_notes('question'), rng('v-q'), index=1.0, bright=0.95, damage=0.35)
    mx.add('VOICE: questioning beep', v, T['robot_notice'], 'robot', 'glass', 0.2, pan_=p, peak_db=-16,
           cue='robot_notice')
    mx.add('iris snap refocus (click-click)', clicks(rng('iris5'), [0.0, C5['iris'][1] - C5['iris'][0]],
                                                     f=(2800, 4600, 7300), t60=0.022, body=0.2), C5['iris'][0],
           'robot', 'glass', 0.15, pan_=p + 0.05, peak_db=-27)
    nod = np.zeros(n_of(0.7))
    add_at(nod, servo(rng('lch'), 0.45, 120, 240, 'down', gear=0.5, bright=0.75), 0.0)
    add_at(nod, 0.5 * body_clank(rng('collar'), 1.0, 0.6, 0.3, 0.2, size=1.2), 0.22)             # sinks into its collar
    mx.add('the nod (looks at its heart)', nod, C5['nod'], 'robot', 'glass', 0.25, pan_=p, peak_db=-31)
    d = T['cut_s7'] - T['look_chest']
    n = n_of(d)
    tt = tvec(n) + T['look_chest']
    a0, a1, a2, a3 = C5['swell']                                                   # the heart's realisation swell
    sw = smoothstep((tt - a0) / (a1 - a0)) * (1 - smoothstep((tt - a2) / (a3 - a2)))
    dip = 1 - 0.45 * np.exp(-0.5 * ((tt - C5['dips'][1]) / 0.03) ** 2)
    env = (0.45 * smoothstep((tt - T['look_chest']) / 0.3) + 0.55 * sw) * dip
    env *= 1 - smoothstep((tt - C5['fade'][0]) / (C5['fade'][1] - C5['fade'][0]))  # the picture fades to black
    hum = core_hum(rng('hum-real'), d, f0=HUM, bright=0.45, air=0.05, detune=0.5) * env
    mx.add('CORE HUM (realisation, swells with the heart)', hum, T['look_chest'], 'robot', 'glass', 0.3,
           pan_=p + 0.1, peak_db=-19, cue='look_chest', kind='swell')
    iw = servo(rng('iwid'), C5['widen'][1] - C5['widen'][0], 300, 420, 'flat', gear=0.3, whine=0.4, bright=1.2)
    mx.add('iris widens (slow)', iw * ar(len(iw), 0.2, 0.2, hold=0.1), C5['widen'][0], 'robot', 'glass', 0.15,
           pan_=p + 0.05, peak_db=-41)

    # ---------------- S7a: dig & plant (robot_acts DIG_SCRAPE_TS / PLANT_T / DIG_PAT_TS) ---------
    RA = __import__('chars.robot_acts', fromlist=['x'])
    p = PAN['s7_planter']
    scr = getattr(RA, 'DIG_SCRAPE_TS', (0.5, 0.75))
    pats = getattr(RA, 'DIG_PAT_TS', (1.6, 1.75))
    for k, ts in enumerate(scr):
        mx.add('soil scrape (dig)', soil_scrape(rng(f'dig{k}'), 0.2, bright=0.9), T['cut_s7'] + ts, 'fx', 'glass',
               0.25, pan_=p, peak_db=-20 + (k == 1), cue='dig' if k == 0 else None)
        mx.add('servo: dig', servo(rng(f'digs{k}'), 0.18, 170, 300, 'bell', gear=0.5), T['cut_s7'] + ts - 0.03,
               'robot', 'glass', 0.15, pan_=PAN['s7_robot'], peak_db=-35)
    tpl = T['cut_s7'] + getattr(RA, 'PLANT_T', 1.2)
    mx.add('sprout set into the hole', dirt_trickle(rng('plant'), 0.35), tpl, 'fx', 'glass', 0.25, pan_=p,
           peak_db=-30)
    mx.add('firms the soil', soil_scrape(rng('firm'), 0.14, bright=0.7), tpl + 0.15, 'fx', 'glass', 0.2, pan_=p,
           peak_db=-29)
    for k, tp in enumerate(pats):
        mx.add('pat', soil_pat(rng(f'pat{k}')), T['cut_s7'] + tp, 'fx', 'glass', 0.25, pan_=p,
               peak_db=-19 - 2 * k, cue='plant_pat' if k == 0 else None)

    # ---------------- c03: the heart (latch, spring hinge, reach, FLARE + heartbeat) ------------
    C3 = SY['c03']
    mx.add('servo: finger flicks the latch', servo(rng('fl'), 0.15, 250, 420, 'bell', gear=0.5, bright=1.1),
           T['cut_c03_core'], 'robot', 'glass', 0.1, pan_=-0.1, peak_db=-33)
    mx.add('hatch latch (unlatch)', clicks(rng('latch'), [0.0, C3['latch'][1] - C3['latch'][0]],
                                           f=(2300, 3900, 6100), t60=0.035, body=0.5),
           C3['latch'][0], 'fx', 'glass', 0.12, pan_=0.05, peak_db=-15, cue='cut_c03_core')
    ajar = micro_click(rng('ajar'), 1.0, f=(1700, 2900, 4400), t60=0.05, body=0.6)
    ajar = fit(ajar, n_of(0.25)) + 0.15 * fit(unit(bp(noise(n_of(0.25), rng('ajar2')), 800, 4000)) *
                                          np.exp(-tvec(n_of(0.25)) / 0.04), n_of(0.25))
    mx.add('door pops ajar', ajar, C3['ajar'], 'fx', 'glass', 0.15, pan_=0.0, peak_db=-27)
    mx.add('spring hinge creak (swing)', hinge_creak(rng('hinge'), C3['door_stop'] - C3['open'] + 0.03),
           C3['open'], 'fx', 'glass', 0.2, pan_=-0.1, peak_db=-20, cue='hatch_open')
    stop = np.zeros(n_of(0.6))
    add_at(stop, body_clank(rng('hst'), 1.0, 1.2, 0.4, 0.3, size=0.7), 0.0)
    tw = modal(0.5, np.array([410.0, 1130.0, 2270.0]), [0.35, 0.22, 0.12], [1.0, 0.5, 0.3], rng('spring'))
    tw *= 1 + 0.35 * np.sin(TAU * 11.0 * tvec(len(tw))) * np.exp(-tvec(len(tw)) / 0.15)       # spring 'boing'
    add_at(stop, 0.35 * tw, 0.0)
    mx.add('door hits the stop (spring overshoot)', stop, C3['door_stop'], 'fx', 'glass', 0.15, pan_=-0.3,
           peak_db=-26)
    for k, tb in enumerate(C3['door_bounce']):
        mx.add('door settles', body_clank(rng(f'dsb{k}'), 1.0, 1.2, 0.25, 0.0, size=0.7), tb, 'fx', 'glass', 0.1,
               pan_=-0.3, peak_db=-34 - 4 * k)
    r0, r1 = C3['reach']
    mx.add('servo: hands reach into the cavity', servo(rng('reach'), r1 - r0, 180, 320, 'bell', gear=0.5, bright=0.9),
           r0, 'robot', 'glass', 0.12, pan_=0.0, peak_db=-31)
    mx.add('fingers cup the orb (cradle wires)', soil_scrape(rng('wr'), 0.12, bright=1.5) +
           0.3 * zap(rng('wrz'), 0.12, f0=150)[:n_of(0.12)], r1 - 0.13, 'robot', 'glass', 0.15, pan_=0.0,
           peak_db=-29)
    # THE FLARE (one frame of light at the lift): a warm bloom hit + the first lub, exactly on the frame
    mx.add('BLOOM HIT (the heart flares out)', bloom_hit(rng('bloom')), C3['lift'], 'robot', 'glass', 0.35, pan_=0.0,
           peak_db=-11.5, cue='core_out')
    beats = [C3['beats'][0], C3['beats'][1], SY['s7b_beat']]
    for k, (tl, td) in enumerate(beats):
        t_l = C3['lift'] if k == 0 else tl                                 # first lub starts ON the flare frame
        lub_dub(mx, 'heartbeat (lub-dub)', t_l, td, -12.5 + 0.8 * (k > 0) - 0.7 * k, pan_=0.0, seed=f'hb{k}')
    # the heart's hum: muffled behind the glass from the cut, opens with the hatch, blooms at the lift,
    # pulses with every beat, then sinks into the soil (c08) and dies away under the power-down
    C8 = SY['c08']
    tin = C8['press']
    tstart = T['cut_c03_core']
    d = T['power_down'] + 0.55 - tstart
    n = n_of(d)
    tt = tvec(n) + tstart
    pulse = sum(np.exp(-np.clip(tt - b, 0, None) / 0.22) * (tt >= b) for b in [C3['lift']] + [x[0] for x in beats[1:]])
    behind = 0.2 + 0.12 * smoothstep((tt - C3['open']) / 0.4)
    e = (behind + (1 - behind) * smoothstep((tt - C3['lift']) / 0.15)) * (1 + 0.35 * pulse)
    e *= smoothstep((tt - tstart) / 0.12)
    e *= 1 - 0.95 * smoothstep((tt - tin) / (C8['gone'] - tin + 0.35))
    fdrop = HUM * (1 - 0.1 * smoothstep((tt - tin) / 0.8))
    hum = core_hum(rng('heart'), d, f0=HUM, bright=0.8, air=0.04, detune=0.8)
    hum2 = additive(fdrop * 4.0, [1.0, 0.3, 0.12], rng_=rng('h2')) * 0.25        # the 'glow' partials
    hum = (hum + hum2) * e
    open_ = smoothstep((tt - C3['open']) / 0.4) * 0.3 + smoothstep((tt - C3['lift']) / 0.1) * 0.7
    lpf = (700 + 4300 * open_) * (1 - 0.85 * smoothstep((tt - tin) / 0.45)) + 300
    hum = pan(tvf(hum, lpf, 0.7, 'lp'), 0.0)
    mx.add('core hum (behind glass -> heart out -> into the soil)', hum, tstart, 'robot', 'glass', 0.3,
           peak_db=-16, kind='swell')

    # ---------------- c08: the heart pressed into the soil ---------------------------------------
    pc = -0.1
    mx.add('servo: hands lower the heart', servo(rng('lower'), tin - C8['start'], 140, 220, 'down', gear=0.4,
                                                 bright=0.8), C8['start'], 'robot', 'glass', 0.1, pan_=pc, peak_db=-34)
    mx.add('HEART PRESSED INTO THE SOIL (crunch + sub + whump)', heart_press(rng('press')), tin, 'fx', 'glass', 0.3,
           pan_=pc, peak_db=-11, cue='core_in_soil')
    lub_dub(mx, 'the cracks pulse (heartbeat under the soil)', C8['lub'], C8['dub'], -18, pan_=pc, seed='hb-soil',
            muffle=500, mid=0.2)
    mx.add('soil settles (trickle)', dirt_trickle(rng('settle'), 0.7), C8['lub'] + 0.05, 'fx', 'glass', 0.2,
           pan_=pc + 0.1, peak_db=-35)
    mx.add('soil settles (last grains)', dirt_trickle(rng('settle2'), 0.35), C8['last'] - 0.02, 'fx', 'glass', 0.2,
           pan_=pc + 0.15, peak_db=-39)
    mx.add('servo: fingers slacken', servo(rng('slack'), C8['last'] - C8['linger'] + 0.05, 90, 150, 'down',
                                           gear=0.3, whine=0.0, bright=0.6), C8['linger'] + 0.03, 'robot', 'glass',
           0.1, pan_=pc, peak_db=-37)

    # ---------------- c06: the last light (power-down locked to the eye's flickers) ------------
    C6 = SY['c06']
    t_pd, t_eye = T['power_down'], T['eye_off']
    pdn = power_down_v2(rng('pdown'), t_eye - t_pd, dropouts=[(a - t_pd, b - t_pd, g) for (a, b, g) in C6['dropouts']],
                        steps=[(a - t_pd, g) for (a, g) in C6['weaker']],
                        breath=(C6['breath'][0] - t_pd, C6['breath'][1] - t_pd))
    mx.add('POWER-DOWN whine (stutters with the eye)', pdn, t_pd, 'robot', 'glass', 0.25, pan_=0.0, peak_db=-15,
           cue='power_down', cut_at=T['silence'] - 0.02)
    for k, (a, b, g) in enumerate(C6['dropouts']):
        if g == 0.0 and k in (0, 1):
            mx.add('eye flicker tick', zap(rng(f'ef{k}'), 0.04, f0=140), a, 'robot', 'glass', 0.15, pan_=0.05,
                   peak_db=-31 - 2 * k)
    mx.add('servo: body sags (servos letting go)', servo(rng('slump'), t_eye - t_pd - 0.1, 70, 160, 'down', gear=0.9,
                                                         whine=0.0, bright=0.6), t_pd + 0.05, 'robot', 'glass', 0.2,
           pan_=0.0, peak_db=-31, cut_at=t_eye)
    mx.add('FINAL CLICK (eye off)', micro_click(rng('final'), 1.0, f=(2100, 3600, 5400), t60=0.03, body=0.45),
           C6['off'], 'robot', 'glass', 0.12, pan_=0.05, peak_db=-19, cue='eye_off', cut_at=T['silence'] - 0.01)
    mx.add('head sags (2 px)', body_clank(rng('sagc'), 1.0, 0.7, 0.12, 0.0, size=1.2), C6['sag'][1], 'robot',
           'glass', 0.1, pan_=0.0, peak_db=-33, cut_at=T['silence'] - 0.01)

    # ---------------- c04: the eye glows again (flicker, then the motif as a whisper) -----------
    C4 = SY['c04']
    pe = pan_of_x(C4['eye_x'])
    mx.add('eye flicker (waking)', clicks(rng('wake'), [x - C4['flicker_on'][0] for x in C4['flicker_on']],
                                          f=(3000, 5100, 7700), t60=0.012, body=0.1), C4['flicker_on'][0], 'robot',
           'valley', 0.15, pan_=pe, peak_db=-35, cue='eye_glow')
    n = n_of(2.6)
    tt = tvec(n)
    u = np.clip(tt / 1.75, 0, 1)
    bright = 0.42 + 0.58 * (1 - (1 - u) ** 2)                     # c04.eye_glow: quick at first, then a slow swell
    fup = HUM * 2 ** smoothstep(tt / 1.6)                          # gentle power-UP: an octave rise (mirror of 43.4)
    up = additive(fup * 2, [1.0, 0.25, 0.08], rng_=rng('pup')) * bright * ar(n, 0.3, 0.9, hold=1.4)
    mx.add('gentle power-up glow', up, C4['steady'], 'robot', 'valley', 0.3, pan_=pe, peak_db=-36)
    v = voice(motif_notes('whisper'), rng('v-whisper'), index=0.45, bright=0.62, whisper=0.55)
    mx.add('VOICE: motif whisper (callback)', v, C4['steady'] - 0.04, 'robot', 'valley', 0.25, pan_=pe, peak_db=-19)
    mx.add('servo: head lifts 1 px toward the bird', servo(rng('hlift'), 0.12, 160, 240, 'bell', gear=0.3,
                                                             bright=0.8), C4['head_lift'] - 0.03, 'robot', 'valley',
           0.15, pan_=pe, peak_db=-38)


# =============================================================================================
# 3. FOLEY & STORY EVENTS
# =============================================================================================
def build_fx(mx):
    # ---------------- S2 ------------------------------------------------------------------------
    mx.add('water: spout drip', water_drip(rng('drip')), T['drop_fall'], 'fx', 'desert', 0.25, pan_=0.0,
           peak_db=-25, cue='drop_fall')
    mx.add('water: drop hits dry earth (tk + soak)', drop_hit(rng('dhit')), T['drop_hit'], 'fx', 'desert', 0.2,
           pan_=0.02, peak_db=-22, cue='drop_hit')
    C7 = SY['c07']                                                        # c07 close-up (frames @30 fps)
    for k, tj in enumerate(C7['jiggles'] + [C7['flick']]):
        jig = 0.7 * can_clink(rng(f'jig{k}'), 1.0, damp=1.8, dur=0.3) + \
            0.3 * fit(micro_click(rng(f'jigc{k}'), 1.0, f=(2600, 4200, 6100), t60=0.02, body=0.2), n_of(0.3))
        mx.add('can jiggles (coaxing the last drop)', jig, tj + 0.02, 'fx', 'desert', 0.15, pan_=-0.35,
               peak_db=-35 + 2 * (k == 2))
    mx.add('spout snaps back up', can_clink(rng('snap'), 1.0, damp=1.3, dur=0.5), T['drop_fall'] + 0.004, 'fx',
           'desert', 0.2, pan_=-0.3, peak_db=-34)
    sat = water_drip(rng('sat'))
    mx.add('satellite droplet', lp(sat, 9000, 2), C7['satellite'], 'fx', 'desert', 0.2, pan_=-0.05, peak_db=-37)
    for k, tcr in enumerate(C7['crown']):
        dd = micro_click(rng(f'crown{k}'), 1.0, f=(3200, 5400, 8100), t60=0.006, body=0.0)
        mx.add('crown splash droplets', dd, tcr, 'fx', 'desert', 0.15, pan_=0.02 + (-0.15 if k else 0.15),
               peak_db=-38 - 2 * k)
    jig = 0.7 * can_clink(rng('jiglast'), 1.0, damp=2.0, dur=0.3)
    mx.add('one more little shake (nothing comes)', jig, C7['last_shake'], 'fx', 'desert', 0.15, pan_=-0.35,
           peak_db=-37)
    mx.add('leaf snap', leaf_snap(rng('lsnap')), T['leaf_breaks'], 'fx', 'desert', 0.25, pan_=PAN['s2_stalk'],
           peak_db=-21, cue='leaf_breaks')
    sk = leaf_skitter(rng('lskit'), 1.35)
    sk = pan(sk, PAN['s2_stalk'] - 0.75 * smoothstep(np.linspace(0, 1, len(sk)) * 1.2))   # blown away to the left
    mx.add('leaf skitter away', sk, T['leaf_breaks'] + 0.05, 'fx', 'desert', 0.25, peak_db=-27)
    # ---------------- S3a -----------------------------------------------------------------------
    g0, g1 = SY['s3_gust']                                                # the veil of dust crosses the frame
    gd = (g1 - g0) + 0.7
    w = whoosh(rng('gust'), gd, peak=(0.12 + 0.5 * (g1 - g0)) / gd, f=(420.0, 1500.0, 520.0), q=1.0,
               pan_path=(0.95, -0.95), pass_w=0.55, hiss=0.55, low=0.35, air=0.25, w=0.15)
    mx.add('GUST whoosh R->L (dust veil)', w, g0 - 0.24, 'fx', 'desert', 0.2, peak_db=-15, cue='gust')
    gl = modal(1.6, note('D7') * np.array([1.0, 2.32, 3.95, 1.5]), [1.2, 0.6, 0.3, 0.9], [1.0, 0.35, 0.12, 0.3],
               rng('gl'))                                        # D7 + A7 (+ glassy inharmonic partials)
    tw = sparkles(rng('glt'), 0.6, 30.0, (5000, 9500), gdur=(0.01, 0.03), spread=0.2)
    glint = pan(gl, 0.0) * 0.9
    add_at(glint, tw * 0.5, 0.04)
    mx.add('GLINT: crystalline ping', glint, T['glint'], 'fx', 'desert', 0.45, pan_=pan_of_x(SY['glint_x']),
           peak_db=-19, cue='glint')
    # ---------------- c01: green reflection grows as it focuses ---------------------------------
    sp = sparkles(rng('refl'), 0.8, 60 * np.linspace(0.2, 1, n_of(0.8)) ** 2, (4500, 9000), rise=5, spread=0.3)
    sp *= ar(sp.shape[1], 0.5, 0.25, hold=0.05)
    mx.add('reflection sparkle', sp, T['eye_focus'] + 0.02, 'fx', 'desert', 0.3, peak_db=-33)
    # ---------------- c02: dust grains skitter past, the leaf tick -------------------------------
    n = n_of(1.45)
    x = poisson_impulses(n, 60, rng('grain'), 0.9)
    grains = resonate(x, [4200.0, 6100.0, 8300.0], [0.004, 0.003, 0.002], [1, .8, .6])
    mx.add('dust grains skitter', grains * ar(n, 0.3, 0.3, hold=0.8), T['cut_c02_sprout'] + 0.02, 'fx', 'desert',
           0.1, pan_=0.3, peak_db=-42)
    tick = micro_click(rng('ltick'), 1.0, f=(5200, 7900, 10400), t60=0.01, body=0.0)
    lt = np.zeros(n_of(0.35))
    add_at(lt, tick, 0.0)
    rs = crackle(n_of(0.25), 3000 * np.exp(-tvec(n_of(0.25)) / 0.06), rng('lrs'), band=(3000, 10000), grain_ms=0.3)
    add_at(lt, 0.25 * unit(rs) * np.exp(-tvec(n_of(0.25)) / 0.06) * 0.12, 0.01)
    mx.add('leaf tick (fingertip)', lt, T['leaf_touch'], 'fx', 'desert', 0.08, pan_=-0.05, peak_db=-29,
           cue='leaf_touch')
    # ---------------- S3b ----------------------------------------------------------------------
    p = pan_of_x(SY['glint_x'] - 6)
    mx.add('soil scrape (first touch)', soil_scrape(rng('scp0'), 0.12, bright=0.9), SY['scoop_lean'] - 0.1, 'fx',
           'desert', 0.2, pan_=p, peak_db=-26, cue='scoop')
    for k, ts in enumerate(SY['scoop_strokes']):
        mx.add('soil scrape (dig stroke)', soil_scrape(rng(f'scp{k + 1}'), 0.19), ts - 0.03, 'fx', 'desert', 0.2,
               pan_=p, peak_db=-21 - 0.5 * k)
        mx.add('dirt flicked back', dirt_trickle(rng(f'flk{k}'), 0.25), ts + 0.1, 'fx', 'desert', 0.2,
               pan_=p - 0.25, peak_db=-33)
    mx.add('clump tears free (soil crumbles)', soil_scrape(rng('tear'), 0.1, bright=0.8) +
           fit(dirt_trickle(rng('trk'), 0.35), n_of(0.1)), SY['scoop_lift'], 'fx', 'desert', 0.2, pan_=p,
           peak_db=-28)
    mx.add('clump crumbs trickle', dirt_trickle(rng('trk2'), 0.3), SY['scoop_lift'] + 0.08, 'fx', 'desert', 0.2,
           pan_=p - 0.1, peak_db=-33)
    mx.add('soil thunk into can', soil_thunk(rng('thunk')), SY['scoop_incan'], 'fx', 'desert', 0.25,
           pan_=pan_of_x(SY['s3b_kneel_x'] - 12), peak_db=-17, cue='into_can')

    # ---------------- S5 storm events -------------------------------------------------------------
    cut = T['cut_s6']
    t0 = T['cut_s5']
    hit = np.zeros((2, n_of(2.2)))
    add_at(hit, whoosh(rng('shw'), 1.4, peak=0.05, f=(250.0, 900.0, 500.0), q=0.8, pan_path=(0.8, -0.2), pass_w=0.3,
                       hiss=0.8, low=0.8, air=0.3), 0.0)
    add_at(hit, pan(boom(rng('shb'), 2.0, f=(60.0, 30.0)), 0.3) * 0.7, 0.045)
    add_at(hit, sparks(rng('debris0'), 1.2, 90, pan_from=0.8, pan_to=-0.5) * 0.25, 0.06)
    mx.add('STORM HIT (wall of dust)', hit, t0 - 0.045, 'fx', 'storm', 0.3, peak_db=-5, cue='cut_s5', cut_at=cut)
    # the surge (foot of the haboob) rolls over the robot: shake + shove
    surge = np.zeros((2, n_of(1.6)))
    add_at(surge, whoosh(rng('surge5'), 1.3, peak=0.08, f=(220.0, 800.0, 420.0), q=0.8, pan_path=(0.7, -0.4),
                         pass_w=0.25, hiss=0.9, low=0.9, air=0.3), 0.0)
    add_at(surge, pan(boom(rng('surgeb'), 1.4, f=(52.0, 30.0)), 0.2) * 0.55, 0.1)
    mx.add('SURGE rolls over the robot', surge, SY['s5_front'] - 0.1, 'fx', 'storm', 0.3, peak_db=-8, cut_at=cut)
    # the gust rips a branch off the dead tree on the ridge (it flies off to the left)
    br = np.zeros((2, n_of(1.2)))
    add_at(br, pan(wood_crackle(rng('branch'), 0.25, 900 * np.exp(-tvec(n_of(0.25)) / 0.05)), 0.5), 0.0)
    add_at(br, pan(0.6 * wood_groan(rng('branchg'), 0.2, rate=(40, 140, 60), base=1.4), 0.5), 0.0)
    rb = rng('branch-tumble')
    for k in range(4):
        kn = modal(0.2, np.array([420.0, 980.0, 1730.0]) * rb.uniform(0.85, 1.2), [0.05, 0.035, 0.02], [1, .6, .4],
                   rb)
        add_at(br, pan(kn * (0.6 - 0.1 * k), 0.3 - 0.3 * k), 0.2 + 0.17 * k + rb.uniform(0, 0.04))
    mx.add('branch ripped off the dead tree', br, SY['s5_branch'], 'fx', 'storm', 0.3, peak_db=-19, cut_at=cut)
    w = whoosh(rng('bigg'), 1.8, peak=0.25, f=(260.0, 1100.0, 400.0), q=0.8, pan_path=(0.9, -0.9), pass_w=0.35,
               hiss=0.8, low=0.6, air=0.35)
    mx.add('BIG GUST (blast)', w, SY['s5_gust'] - 0.45, 'fx', 'storm', 0.3, peak_db=-8, cut_at=cut)
    # debris: pebbles / stones clacking, a tumbling dead bush, a far metal sheet rattling
    r = rng('debris')
    for k in range(34):
        tt = r.uniform(t0 + 0.2, cut - 0.1)
        m = n_of(0.12)
        s = modal(0.12, np.array([1400.0, 2350.0, 3800.0, 5600.0]) * r.uniform(0.7, 1.4), [0.02, 0.015, 0.01, 0.006],
                  r.uniform(0.3, 1.0, 4), r)
        s += 0.5 * hp(noise(m, r), 1500) * np.exp(-tvec(m) / 0.003)
        mx.add('debris: stone', s, tt, 'fx', 'storm', 0.3, pan_=r.uniform(-0.8, 0.9), peak_db=r.uniform(-32, -22),
               cut_at=cut)
    n = n_of(1.3)
    bush = crackle(n, 5000 * (0.6 + 0.4 * np.sin(TAU * 4.2 * tvec(n)) ** 2), rng('bush'), band=(900, 7000), grain_ms=0.6)
    bush = unit(bush) * ar(n, 0.3, 0.5, hold=0.5)
    mx.add('tumbling dead bush R->L', pan(bush, np.linspace(0.85, -0.85, n)), t0 + 0.75, 'fx', 'storm', 0.3,
           peak_db=-24, cut_at=cut)
    n = n_of(0.9)
    x = impulse_train(n, 26 + 10 * smooth_rand(n, 3, rng('shr')), rng('shr2'), 0.4, 0.5)
    rat = resonate(x, [340.0, 610.0, 1020.0, 1680.0, 2600.0], [0.08, 0.07, 0.05, 0.04, 0.03], [1, .9, .7, .5, .3])
    mx.add('far metal sheet rattle', rat * ar(n, 0.2, 0.4, hold=0.3), t0 + 1.05, 'fx', 'storm', 0.4, pan_=0.7,
           peak_db=-28, cut_at=cut)
    # the flying metal sheet: wobbling in from the right -> CLANG -> antenna snaps -> zap -> sparks
    ta = T['antenna_rip']
    n = n_of(ta - SY['s5_sheet_in'] + 0.03)
    tt = tvec(n)
    wob = unit(tvf(noise(n, rng('sheet')), 600 + 250 * np.sin(TAU * 7.5 * tt), 3.0, 'bp')) * \
        (0.5 + 0.5 * np.sin(TAU * 7.5 * tt)) ** 2 * smoothstep(tt / (n / SR)) ** 1.5
    mx.add('metal sheet tumbles in', pan(wob, np.linspace(0.85, 0.05, n)), SY['s5_sheet_in'], 'fx', 'storm', 0.25,
           peak_db=-19, cut_at=cut)
    d = 1.8
    n = n_of(d)
    tt = tvec(n)
    sheet_f = np.array([143.0, 236.0, 389.0, 557.0, 811.0, 1063.0, 1490.0, 2011.0, 2780.0, 3710.0, 4930.0])
    rs = rng('clang')
    cl = np.zeros(n)
    for k, f in enumerate(sheet_f):
        fk = f * (1 + 0.012 * np.sin(TAU * 8.0 * tt + k) * np.exp(-tt / 0.5))       # the sheet wobbling
        cl += rs.uniform(0.4, 1.0) / (1 + 0.15 * k) * np.sin(phase_of(fk) + rs.random() * TAU) * \
            np.exp(-6.9 * tt / (1.3 - 0.09 * k))
    cl = cl * np.clip(tt / 0.0005, 0, 1)
    head = body_clank(rng('headhit'), 1.0, bright=1.3, dur=d, rattle=0.9, size=1.1)
    tr = hp(noise(n, rs), 1200) * np.exp(-tt / 0.002) * 3
    thud = np.sin(phase_of(130 * (1 + np.exp(-tt / 0.01)))) * np.exp(-tt / 0.04)
    clang = unit(cl) * 0.7 + unit(head) * 0.8 + tr + 1.2 * thud
    mx.add('ANTENNA HIT: metal clang', clang, ta, 'fx', 'storm', 0.35, pan_=-0.05, peak_db=-4, cue='antenna_rip',
           cut_at=cut)
    ping = modal(0.9, np.array([1870.0, 5150.0, 10090.0]) * (1 - 0.0), [0.7, 0.35, 0.15], [1.0, 0.45, 0.2], rng('ping'))
    ping *= 1 + 0.25 * np.sin(TAU * 19 * tvec(len(ping))) * np.exp(-tvec(len(ping)) / 0.15)
    mx.add('antenna snap: ping', ping, ta + 0.025, 'fx', 'storm', 0.35, pan_=0.05, peak_db=-12, cut_at=cut)
    mx.add('antenna snap: electrical zap', zap(rng('azap'), 0.26), ta + 0.035, 'fx', 'storm', 0.25, pan_=-0.05,
           peak_db=-10, cut_at=cut)
    mx.add('spark crackles', sparks(rng('asp'), 0.9, 55, pan_from=-0.05, pan_to=-0.8), ta + 0.05, 'fx', 'storm',
           0.2, peak_db=-13, cut_at=cut)
    r = rng('antenna-tumble')
    for k in range(5):
        tk = ta + 0.25 + 0.14 * k + r.uniform(0, 0.04)
        s = modal(0.25, np.array([2650.0, 4300.0, 6900.0]) * r.uniform(0.95, 1.05), [0.12, 0.08, 0.05], [1, .6, .4], r)
        mx.add('antenna tumbles away', s, tk, 'fx', 'storm', 0.35, pan_=-0.15 - 0.15 * k, peak_db=-24 - 2.5 * k,
               cut_at=cut)
    n = n_of(0.8)
    tt = tvec(n)
    wob2 = unit(tvf(noise(n, rng('sheet2')), 520 + 200 * np.sin(TAU * 6 * tt), 3.0, 'bp')) * \
        (0.5 + 0.5 * np.sin(TAU * 6 * tt)) ** 2 * np.exp(-tt / 0.3)
    mx.add('sheet flies off', pan(wob2, np.linspace(-0.1, -0.9, n)), ta + 0.1, 'fx', 'storm', 0.3, peak_db=-22,
           cut_at=cut)
    # a pale wall of dust rolls in from the right and swallows the robot -> whiteout
    wall = whoosh(rng('wall'), 1.7, peak=0.32, f=(200.0, 850.0, 600.0), q=0.7, pan_path=(0.95, -0.1), pass_w=0.5,
                  hiss=1.0, low=0.7, air=0.3, w=0.8)
    mx.add('pale wall of dust rolls in (whiteout)', wall, SY['s5_wall'] - 0.55, 'fx', 'storm', 0.3, peak_db=-10,
           cut_at=cut)

    # ---------------- S6 greenhouse: glass tinkles, frame creak -----------------------------------
    r = rng('tinkles')
    for tt in (T['cut_s6'] + 0.45, T['sit_down'] + 0.72, T['sprout_lean'] + 1.35, T['robot_notice'] + 0.75,
               T['look_chest'] + 1.05, T['cut_s7'] + 0.9):
        s = np.zeros(n_of(0.6))
        for j in range(int(r.integers(1, 4))):
            add_at(s, glass_tink(r, r.uniform(0.4, 1.0), dur=r.uniform(0.2, 0.5), hard=0.4), r.uniform(0, 0.08))
        mx.add('glass tinkle (panes)', s, tt, 'amb', 'glass', 0.55, pan_=r.uniform(-0.7, 0.8),
               peak_db=r.uniform(-40, -35))
    # the sprout leans toward the light (c05): rises, sags back while the heart stutters, reaches again
    C5 = SY['c05']
    d = C5['start'] + 1.55 - T['sprout_lean']
    sh = shimmer(rng('lean'), d, note('C7'), ratios=(1.0, 2.0, 2.99, 4.02, 5.1), glide=2.0,    # C7 -> D7
                 att=0.45, rel=0.55, twinkle=(4.0, 9.0), spread=0.35, grains=0.35, grain_band=(4000, 9000),
                 amp_tilt=0.7)
    tt = tvec(sh.shape[1]) + T['sprout_lean']
    s0, s1 = C5['stutter']
    sh *= 1 - 0.65 * smoothstep((tt - s0) / 0.05) * (1 - smoothstep((tt - s1) / 0.12))
    mx.add('SPROUT LEAN shimmer', sh, T['sprout_lean'], 'fx', 'glass', 0.4, pan_=PAN['s6_can'], peak_db=-23,
           cue='sprout_lean', kind='swell')

    # ---------------- S8a: bloom inside --------------------------------------------------------------
    tp = T['first_pulse']
    pul = np.zeros(n_of(1.4))
    add_at(pul, sub_thump(rng('fp1'), 58, 34, 1.2, knock=0.2, body=0.35, mid=0.3), 0.0)
    add_at(pul, 0.55 * sub_thump(rng('fp2'), 62, 36, 1.0, knock=0.15, body=0.3, mid=0.25), 0.23)
    mx.add('FIRST PULSE (deep sub heartbeat)', pul, tp, 'fx', 'glass', 0.2, pan_=PAN['s7_planter'], peak_db=-7,
           cue='first_pulse')
    tv = T['veins']
    dv = T['growth_burst'] - tv + 0.35
    n = n_of(dv)
    tt = tvec(n)
    x = tt / dv
    ris = riser(rng('vr'), dv, f=(700.0, 6500.0), q=3.0, curve=1.6, w=0.8)
    spk = sparkles(rng('vs'), dv, 20 + 260 * x ** 1.5, (2500, 7000), gdur=(0.004, 0.02), rise=12, spread=0.9)
    shm = shimmer(rng('vsh'), dv, note('D5'), ratios=(1.0, 2.0, 3.0, 4.0, 6.0), glide=12.0, att=dv * 0.8, rel=0.3,
                  spread=0.8, amp_tilt=0.75)
    roots = wood_crackle(rng('vroot'), dv, 15 + 120 * x ** 2) * 0.35
    low = lp(noise(n, rng('vlow'), 'brown'), 90, 2)
    low = unit(low) * x ** 1.5 * 0.6
    veins = 0.7 * ris + 0.9 * spk + 0.8 * shm + pan(roots + low, 0.1)
    veins = balance(veins, np.linspace(0.45, 0.0, n)) * np.clip((dv - tt) / 0.3, 0, 1) ** 0.5
    mx.add('VEINS: rising shimmer', veins, tv, 'fx', 'glass', 0.35, peak_db=-15, cue='veins', kind='swell')
    un = np.zeros(n_of(0.6))
    add_at(un, 0.5 * leaf_skitter(rng('unf'), 0.5), 0.0)
    add_at(un, 0.4 * wood_groan(rng('unfw'), 0.35, rate=(30, 90, 50), base=1.6), 0.02)
    mx.add('sprout straightens (leaves unfurl)', un, tv + 0.4, 'fx', 'glass', 0.3, pan_=PAN['s7_planter'],
           peak_db=-27)
    # GROWTH BURST 46.8 -> 48.0: a surge at the cue, then a crescendo that slams into the climax
    tg = T['growth_burst']
    tc = T['cut_s8b_climax']
    dg = tc - tg
    n = n_of(dg)
    tt = tvec(n)
    x = tt / dg
    # the surge (the sprout shoots up): fwoom + fibres cracking + leaves flaring
    surge = np.zeros((2, n))
    add_at(surge, whoosh(rng('surge'), 0.75, peak=0.2, f=(260.0, 1900.0, 800.0), q=1.0, pan_path=(0.3, 0.1),
                         pass_w=0.3, hiss=0.35, low=0.6, air=0.4, w=0.8), 0.0)
    add_at(surge, pan(wood_crackle(rng('scrk'), 0.35, 700 * np.exp(-tvec(n_of(0.35)) / 0.1)), 0.3) * 0.8, 0.0)
    add_at(surge, pan(wood_groan(rng('sgrn'), 0.55, rate=(40, 120, 60), base=1.2), 0.3) * 0.7, 0.03)
    add_at(surge, leaf_burst(rng('sleaf'), 0.7, att=0.03) * 0.5, 0.05)
    # the crescendo
    g_ris = riser(rng('gr'), dg, f=(250.0, 3800.0), q=1.2, curve=1.6, w=0.9)
    g_leaf = rustle(n, 'growleaf', 0.15 + 0.95 * x ** 1.2, band=(1100.0, 9500.0), dens=(2000.0, 26000.0), body=0.4)
    g_wood = np.zeros(n)
    for k, (dt, d, b) in enumerate(((0.3, 0.55, 0.8), (0.55, 0.5, 0.65), (0.8, 0.4, 0.55))):
        add_at(g_wood, wood_groan(rng(f'gw{k}'), d, rate=(18, 70, 40), base=b) * (0.55 + 0.25 * k), dt)
    g_crk = wood_crackle(rng('gcr'), dg, 30 + 450 * x ** 2.0)
    g_glass = np.zeros((2, n))
    rr = rng('tremble')
    tq = 0.3
    while tq < dg - 0.02:
        add_at(g_glass, pan(glass_tink(rr, rr.uniform(0.3, 1.0) * (0.3 + tq / dg), dur=rr.uniform(0.04, 0.12)),
                            rr.uniform(-0.9, 0.9)), tq)
        tq += rr.exponential(1.0 / (10 + 90 * (tq / dg) ** 2))
    g_sub = unit(lp(noise(n, rng('gsub'), 'brown'), 70, 4)) * x ** 1.6
    cres = 0.6 * g_ris + 0.55 * g_leaf * (0.35 + 0.65 * x) + pan(0.7 * g_wood + 0.5 * g_crk * x, 0.2) + \
        0.5 * g_glass + pan(0.9 * g_sub, 0.0)
    for k, dt in enumerate((0.55, 0.78, 0.95)):
        add_at(cres, whoosh(rng(f'whip{k}'), 0.35, peak=0.45, f=(700.0, 2600.0, 900.0), q=1.6,
                            pan_path=((-1) ** k * 0.6, -(-1) ** k * 0.6), pass_w=0.12, hiss=0.5, low=0.1, air=0.4)
               * (0.7 + 0.15 * k), dt)
    growth = 0.55 * surge / (np.max(np.abs(surge)) + 1e-9) + cres / (np.max(np.abs(cres)) + 1e-9)
    growth *= np.clip((dg + 0.02 - tt) / 0.035, 0, 1)                 # a hair of air before the hit
    mx.add('GROWTH BURST surge + crescendo', growth, tg, 'fx', 'glass', 0.35, peak_db=-7.5, cue='growth_burst')

    # ---------------- S8b: THE CLIMAX -------------------------------------------------------------
    mx.add('CLIMAX: glass dome bursts', glass_burst(rng('burst'), 3.4), tc, 'fx', 'valley', 0.45, peak_db=-6,
           cue='cut_s8b_climax')
    mx.add('CLIMAX: deep boom', boom(rng('boom'), 3.5), tc, 'fx', 'valley', 0.3, pan_=0.0, peak_db=-5)
    tr = np.zeros(n_of(1.6))
    add_at(tr, wood_groan(rng('trunk'), 1.1, rate=(25, 60, 20), base=0.55), 0.05)
    add_at(tr, 0.6 * wood_crackle(rng('trunkc'), 1.2, 200 * np.exp(-tvec(n_of(1.2)) / 0.4)), 0.0)
    mx.add('trunk rising (wood strain)', tr, tc + 0.02, 'fx', 'valley', 0.35, pan_=0.0, peak_db=-14)
    mx.add('canopy explodes (leaves)', leaf_burst(rng('canopy'), 2.4, att=0.12), tc + 0.08, 'fx', 'valley', 0.3,
           peak_db=-11)
    # the wave of life: long sweeping whoosh with a bright shimmer layer, widening outward
    twv = T['color_wave']
    dw = T['birds'] + 0.5 - twv
    n = n_of(dw)
    tt = tvec(n)
    x = tt / dw
    wv = whoosh(rng('wave'), dw, peak=0.32, f=(300.0, 1900.0, 700.0), q=0.9, pan_path=(0.0, 0.0), hiss=0.3,
                low=0.3, air=0.35, w=1.0, rel_curve=2.0)
    wv = width(wv, 0.4 + 1.1 * smoothstep(x * 1.5))
    grass = rustle(n, 'wavegrass', np.clip(env_pts(n, [(0, 0), (dw * 0.3, 1.0), (dw, 0.2)]), 0, 1),
                   band=(1000.0, 8000.0), dens=(500.0, 30000.0), body=0.2)
    shw = shimmer(rng('wsh'), dw, note('G5'), ratios=(1.0, 1.5, 2.0, 3.0, 4.0, 6.0), glide=0.0, att=dw * 0.3,
                  rel=dw * 0.55, spread=0.9, grains=0.8, grain_band=(4000, 10000), amp_tilt=0.8)
    wave = 1.0 * wv + 0.5 * grass + 0.55 * shw
    mx.add('WAVE OF LIFE sweep (+shimmer)', wave, twv, 'fx', 'valley', 0.3, peak_db=-10, cue='color_wave',
           kind='swell')
    r = rng('pops')
    for k in range(26):
        tt_ = twv + 0.3 + r.gamma(2.0, 0.45)
        if tt_ > T['birds'] + 0.4:
            continue
        f0 = r.uniform(900, 2200)
        m = n_of(0.05)
        s = np.sin(phase_of(f0 * (1 + 0.8 * np.exp(-tvec(m) / 0.006)))) * np.exp(-tvec(m) / 0.012)
        mx.add('flower pop', s, tt_, 'fx', 'valley', 0.25, pan_=r.uniform(-0.9, 0.9), peak_db=r.uniform(-38, -31))
    # sun breaks through: the clouds tear open + a warm bright swell
    ts = T['sun_break']
    tear = whoosh(rng('tear'), 1.6, peak=0.4, f=(180.0, 700.0, 300.0), q=0.7, pan_path=(-0.3, 0.3), pass_w=0.6,
                  hiss=0.15, low=0.5, air=0.1, w=1.0)
    mx.add('clouds tear open', tear, ts - 0.1, 'fx', 'valley', 0.35, peak_db=-20, cue='sun_break', kind='swell')
    sun = shimmer(rng('sun'), 2.6, note('G6'), ratios=(1.0, 1.5, 2.0, 3.0, 4.0, 6.0), glide=0.0, att=0.7, rel=1.5,
                  twinkle=(2.0, 7.0), spread=1.0, grains=1.0, grain_band=(5000, 11000), amp_tilt=0.85)
    air = np.stack([unit(hp(noise(n_of(2.6), rng(f'sa{c}'), 'pink'), 5000)) for c in range(2)]) * \
        ar(n_of(2.6), 0.8, 1.5, hold=0.3) * 0.25
    mx.add('SUN SHIMMER', sun + air, ts, 'fx', 'valley', 0.4, peak_db=-19, kind='swell')

    # ---------------- birds ---------------------------------------------------------------------
    build_birds(mx)


def build_birds(mx):
    tb = T['birds']
    t9 = T['cut_s9a']
    tc4 = T['cut_c04']
    # a small flock flies in: wing flutters crossing the frame + a few calls (first bird right at the cue)
    r = rng('flock')
    for k in range(6):
        t0 = tb + 0.1 * k + r.uniform(0.0, 0.06) * (k > 0)
        d = r.uniform(1.1, 1.6)
        n = n_of(d)
        tt = tvec(n)
        e = np.clip(tt / 0.12, 0, 1) * np.exp(-0.5 * (np.clip(tt - d * 0.35, 0, None) / (d * 0.3)) ** 2)
        wf = wing_flaps(r, d, rate=r.uniform(11, 16), size=r.uniform(0.9, 1.2), glide=0.15)
        wf = lp(wf * e, 7000, 2)
        side = 1 if k % 2 == 0 else -1
        mx.add('flock: wings', pan(wf, np.linspace(0.9 * side, -0.7 * side, n)), t0, 'fx', 'valley', 0.3,
               peak_db=r.uniform(-28, -24) - 1.0 * k, cue='birds' if k == 0 else None)
        if k in (0, 2, 3, 5):
            c = bird_syllable(r, ['tsip', 'chip', 'tweet'][k % 3], r.uniform(0.9, 1.1))
            mx.add('flock: call', c, t0 + 0.05 + r.uniform(0.0, d * 0.5), 'fx', 'valley', 0.35, pan_=0.4 * side,
                   peak_db=r.uniform(-30, -26))
    # birdsong - different singers at different distances; fuller in the valley
    singers = [
        # species, pan, start, dist (0 near .. 1 far), mean gap(s), level dB
        ('robin', 0.55, tb + 0.6, 0.45, 3.2, -27),
        ('finch', -0.6, tb + 1.4, 0.6, 3.8, -30),
        ('tit', -0.25, t9 + 0.3, 0.75, 4.0, -33),
        ('wren', 0.75, t9 + 0.9, 0.55, 4.5, -30),
        ('blackbird', -0.75, tb + 2.3, 0.7, 4.2, -30),
        ('robin', 0.2, t9 + 1.6, 0.85, 3.5, -34),
    ]
    for si, (sp, p, t0, dist, gap, lv) in enumerate(singers):
        r = rng(f'singer{si}')
        tt = t0
        while tt < TE - 1.4:
            ph = bird_phrase(r, sp, r.uniform(0.95, 1.05))
            if tt + len(ph) / SR > TE - 0.8 or (tt >= tc4 and si >= 4):
                break                                   # the film ends on breeze + the green eye
            hush0, hush1 = T['eye_glow'] - 0.4, T['eye_glow'] + 2.2
            if tt < hush1 and tt + len(ph) / SR > hush0:
                tt = hush1 + r.uniform(0.0, 0.5)        # the valley holds its breath for the robot & the bird
                continue
            ph = lp(ph, 11000 - 6000 * dist, 2)
            lvl = lv + r.uniform(-2.5, 1.5)
            if tt >= tc4:                       # close-up: the valley recedes a little
                lvl -= 4.0
            mx.add(f'birdsong: {sp}', ph, tt, 'amb', 'valley', 0.25 + 0.5 * dist, pan_=p + r.uniform(-0.1, 0.1),
                   peak_db=lvl)
            tt += len(ph) / SR + r.uniform(0.6, 1.6) * gap
    # the valley is alive: a bee drifts past through the flowers (L -> R, a little Doppler)
    d = 1.7
    n = n_of(d)
    tt = tvec(n)
    x = tt / d
    rb = rng('bee')
    fb = 212.0 * (1 + 0.035 * np.tanh((0.5 - x) * 5)) * (1 + 0.02 * (smooth_rand(n, 9, rb) - 0.5))
    buzz = additive(fb, [1.0 / k ** 0.8 for k in range(1, 40)], rng_=rb)
    buzz *= 1 + 0.35 * np.sin(phase_of(fb * 0.5))               # wingbeat roughness
    buzz = peak_eq(lp(hp(buzz, 180), 5000), 1800, 4, 1.2)
    ebee = np.exp(-0.5 * ((x - 0.5) / 0.22) ** 2) * ar(n, 0.05, 0.1, hold=d - 0.15)
    mx.add('bee flies past (valley)', pan(buzz * ebee, np.linspace(-0.85, 0.85, n)), T['cut_s9a'] + 0.35, 'fx',
           'valley', 0.2, peak_db=-31)

    # c04: the small songbird flutters in from the UPPER RIGHT (c04.bird_state: x 336 -> LAND_X, 55.05-55.5),
    # lands on the robot's mossy head on the cue, settles, hops once toward the lens, and chirps twice (beak)
    C4 = SY['c04']
    r = rng('lander')
    f0_, f1_ = C4['fly']
    d = f1_ - f0_ + 0.04
    n = n_of(d)
    tt = tvec(n) + f0_
    xs_ = np.array([C4['bird_x'](x) for x in tt])
    fl = wing_flaps(r, d, rate=16.0, size=0.8, glide=0.0, env=np.clip(0.45 + 0.55 * (tt - f0_) / d, 0, 1))
    for k, tds in enumerate(C4['downstrokes']):                        # the drawn downstrokes (15-fps frames)
        add_at(fl, 1.3 * wing_flaps(rng(f'ds{k}'), 0.07, rate=14.0, size=0.9, glide=0.0), tds - f0_)
    b0, b1 = C4['brake']                                               # 'LAND' pose: wings spread, braking
    add_at(fl, 1.5 * wing_flaps(rng('brake'), b1 - b0 + 0.02, rate=11.0, size=1.0, glide=0.0), b0 - f0_)
    fl = biquad(fl, 'highshelf', 4000, 0.7, 2.0)                        # from above: a little more air
    mx.add('bird flutters in (upper right -> head)', pan(fl, np.array([pan_of_x(x) for x in xs_])), f0_, 'fx',
           'valley', 0.2, peak_db=-22)
    for k, tcall in enumerate((f0_ + 0.07, f0_ + 0.25)):
        mx.add('bird calls in flight', bird_syllable(rng(f'call{k}'), 'tsip', 1.05 + 0.05 * k), tcall, 'fx',
               'valley', 0.2, pan_=pan_of_x(C4['bird_x'](tcall)), peak_db=-27 - k)
    pb = pan_of_x(C4['land_x'])
    land = np.zeros(n_of(0.3))
    add_at(land, micro_click(r, 1.0, f=(2400.0, 3800.0, 6200.0), t60=0.02, body=0.15), 0.0)
    add_at(land, 0.6 * micro_click(r, 1.0, f=(2600.0, 4100.0, 6600.0), t60=0.015, body=0.1), 0.022)
    add_at(land, 0.25 * body_clank(r, 1.0, 1.2, 0.3, 0.0, size=0.75), 0.0)
    mx.add('claws land on head (mossy metal)', lp(land, 7000, 2), C4['land'], 'fx', 'valley', 0.15, pan_=pb,
           peak_db=-25, cue='bird_land')
    ruffle = unit(bp(noise(n_of(0.12), rng('ruffle')), 2500, 8000)) * ar(n_of(0.12), 0.02, 0.09, hold=0.01)
    mx.add('settles (feather ruffle)', ruffle, C4['land'] + 0.08, 'fx', 'valley', 0.15, pan_=pb, peak_db=-36)
    h0, h1 = C4['hop']
    pb2 = pan_of_x(C4['land_x'] + C4['hop_dx'])
    mx.add('hop: wing flick', wing_flaps(rng('flick'), 0.07, rate=24, size=0.8, glide=0.0), h0, 'fx', 'valley', 0.15,
           pan_=pb, peak_db=-31)
    mx.add('hop: lands', micro_click(rng('hopl'), 1.0, f=(2500.0, 4000.0, 6400.0), t60=0.012, body=0.1), h1, 'fx',
           'valley', 0.15, pan_=pb2, peak_db=-31)
    # the beak opens twice (c04: T_CHIRP, T_CHIRP + 0.13, 0.07 s each): a small, soft double chirp
    for k, tch in enumerate(C4['chirps']):
        syl = bird_syllable(rng(f'dchirp{k}'), 'tsip' if k == 0 else 'chip', 1.08 - 0.04 * k)
        syl = fit(syl, n_of(C4['chirp_len'] + 0.01))
        mx.add('soft double chirp', fades(syl, 0.001, 0.008), tch, 'fx', 'valley', 0.2, pan_=pb2,
               peak_db=-25 - 1.5 * k)


# =============================================================================================
# RENDER
# =============================================================================================
def reverb_irs():
    return {
        # open desert: sparse, long, dark-ish; a few far reflections off ruins and dunes
        'desert': make_ir(3.6, t60=(3.3, 2.7, 1.3), xover=(300.0, 2500.0), predelay=0.035, density=320,
                          er=[(0.085, 0.55, -0.6), (0.16, 0.4, 0.7), (0.29, 0.32, -0.3), (0.47, 0.25, 0.55),
                              (0.78, 0.18, -0.7)], lp_hz=6000, onset=0.06, diffuse_gain=0.9, seed='desert'),
        # storm: dense, short, dark, smeared
        'storm': make_ir(1.7, t60=(1.5, 1.0, 0.45), xover=(300.0, 2500.0), predelay=0.004, density=None,
                         lp_hz=6500, onset=0.01, seed='storm'),
        # glass greenhouse: bright, metallic, medium; dense early reflections off panes + ringing iron/glass
        'glass': make_ir(2.6, t60=(1.1, 1.6, 1.9), xover=(350.0, 3500.0), predelay=0.009, density=None,
                         er=[(0.006, 0.6, -0.4), (0.011, 0.5, 0.5), (0.019, 0.45, -0.2), (0.031, 0.35, 0.6),
                             (0.044, 0.3, -0.6), (0.061, 0.22, 0.3)], lp_hz=15000, onset=0.012,
                         metal=(16, 1700.0, 8000.0, 0.5, 1.2, 0.1), seed='glass'),
        # open valley: soft, airy, gentle far reflection from the hills
        'valley': make_ir(2.8, t60=(1.8, 1.5, 0.8), xover=(300.0, 3000.0), predelay=0.02, density=1400,
                          er=[(0.13, 0.22, -0.7), (0.27, 0.18, 0.6)], lp_hz=8000, onset=0.03, seed='valley'),
    }


def render(verbose=True):
    global SY
    t0 = time.time()
    SY = sfx_sync.picture_sync(T)
    for note_ in SY['notes']:
        print('  [picture sync]', note_)
    if verbose:
        print(f'  sync       {time.time() - t0:5.1f}s   (walk contacts, flickers, heart clock, bird path from shots/)')
    mx = Mixer()
    build_ambience(mx)
    if verbose:
        print(f'  ambience   {time.time() - t0:5.1f}s')
    build_robot(mx)
    if verbose:
        print(f'  robot      {time.time() - t0:5.1f}s')
    build_fx(mx)
    if verbose:
        print(f'  fx         {time.time() - t0:5.1f}s')
    irs = reverb_irs()
    stems = {k: v.copy() for k, v in mx.dry.items()}
    cut_storm = np.ones(N)
    i6 = n_of(T['cut_s6'])
    f = n_of(0.075)
    cut_storm[i6 - f:i6] = np.linspace(1, 0, f) ** 2
    cut_storm[i6:] = 0.0
    for (stem, space), snd in mx.send.items():
        wet = convolve_stereo(snd, irs[space], cross=0.3)
        if space == 'storm':
            wet *= cut_storm                                   # the storm's reverb dies with it
        stems[stem] += wet
    if verbose:
        print(f'  reverb     {time.time() - t0:5.1f}s')
    for k in stems:                                            # DC / subsonic blocker (before any gating)
        stems[k] = hp(stems[k], 16.0, 2)
    # global envelope: start from black, TOTAL SILENCE gate, end fade
    tt = tvec(N)
    g = np.ones(N)
    g *= np.clip((tt - T['fade_in']) / 0.02, 0, 1)
    s0, s1 = n_of(T['silence']), n_of(T['first_pulse'])
    fo = n_of(0.06)
    g[s0 - fo:s0] *= np.linspace(1, 0, fo) ** 2
    g[s0:s1] = 0.0
    g *= np.where(tt > TE - 1.2, np.clip((TE - tt) / 1.2, 0, 1) ** 1.5, 1.0)         # ends at exactly 0
    for k in stems:
        stems[k] *= g
    mix = stems['amb'] + stems['robot'] + stems['fx']
    lg = limiter_gain(mix, CEILING_DB, look_ms=4.0, release_ms=90.0)
    for k in stems:
        stems[k] *= lg
    mix = stems['amb'] + stems['robot'] + stems['fx']
    mix[:, s0:s1] = 0.0
    for k in stems:
        stems[k][:, s0:s1] = 0.0
    if verbose:
        print(f'  master     {time.time() - t0:5.1f}s   limiter min gain {db(lg.min()):.2f} dB')
    return mix, stems, mx, lg


def write(path, st):
    x = np.ascontiguousarray(st.T.astype(np.float32))
    assert x.shape == (N, 2)
    wavfile.write(path, SR, x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--review', action='store_true', help='write verification plots to review/sfx_*')
    ap.add_argument('--no-stems', action='store_true')
    a = ap.parse_args()
    print('THE LAST SEED - SFX render')
    mix, stems, mx, lg = render()
    write(OUT_WAV, mix)
    if not a.no_stems:
        for k, p in STEM_WAV.items():
            write(p, stems[k])
    pk = db(np.max(np.abs(mix)))
    print(f'wrote {OUT_WAV}  ({mix.shape[1]} samples, {mix.shape[1] / SR:.3f} s)  sample peak {pk:.2f} dBFS')
    if a.review:
        import sfx_report
        sfx_report.report(mix, stems, mx, lg, T)


if __name__ == '__main__':
    main()
