"""
THE LAST SEED - score renderer & mix.

    python -m audio.music            # render -> audio/music.wav (+ review/music_*.png)
    python -m audio.music --noplots

48 kHz stereo float32, exactly 60.000 s. All timing comes from cues.CUES
(through audio/music_score.py).

Signal flow: events -> instrument groups (EQ) -> dry + hall/room reverb +
SNES-style echo, rendered PER SEGMENT so reverb tails can be cut hard:
  A  0.0 .. cut_s6   (storm cut off by 32.0)
  B  cut_s6 .. silence  (dry + wet gated to digital zero by 44.5)
  C  first_pulse .. end (fades to zero exactly at 60.0)
-> master: loudness normalise (~-16 LUFS), look-ahead limiter (<= -1 dBFS),
   hard digital silence 44.5-45.3.
"""
import hashlib
import os
import pickle
import sys
import time

import numpy as np
from scipy.io import wavfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cues import CUES as C
from audio.music_dsp import (SR, ns, hz, db2a, a2db, rng_for, fade, lp, hp, peq, shelf, make_ir,
                             convolve_reverb, snes_echo, limiter, lufs_integrated, loudness_curve,
                             true_peak_db, width)
from audio import music_inst as I
from audio import music_perc as P
from audio.music_score import build

N = ns(C['end'])
import tempfile
# stem cache (speeds up mix iterations); set MUSIC_CACHE='' to disable
CACHE = os.environ.get('MUSIC_CACHE', os.path.join(tempfile.gettempdir(), 'last_seed_music_cache'))
TARGET_LUFS = -16.0
CEILING_DB = -1.2

# ------------------------------------------------------------------ groups
GROUP = {
    'mbox': 'mbox', 'celesta': 'bells', 'glock': 'bells', 'piano': 'piano', 'harp': 'harp', 'pizz': 'pizz',
    'vln1': 'str_hi', 'vln2': 'str_hi', 'vla': 'str_hi', 'vln_harm': 'str_harm', 'vc': 'str_lo', 'cb': 'str_lo',
    'flute': 'winds', 'ocarina': 'winds', 'horn': 'horn',
    'choir_s': 'choir', 'choir_a': 'choir', 'choir_t': 'choir', 'choir_b': 'choir',
    'pad_cold': 'pad', 'pad_warm': 'pad',
    'taiko': 'perc', 'tom': 'perc', 'timp': 'perc', 'timp_roll': 'perc', 'sub': 'sub', 'heartbeat': 'sub',
    'cymbal': 'cym', 'cym_swell': 'cym', 'rev_cym': 'cym',
    'air': 'fx', 'rumble': 'fx', 'rev_note': 'fx',
}


def _eq_mbox(x):
    return shelf(peq(hp(x, 180, 2), 2600, -2.0, 1.0), 9000, -3.0, 'high')


def _eq_piano(x):
    return shelf(peq(hp(x, 28, 2), 250, -1.5, 1.0), 6000, -2.0, 'high')


def _eq_harp(x):
    return peq(hp(x, 45, 2), 180, -2.0, 1.0)


def _eq_pizz(x):
    return peq(peq(hp(x, 60, 2), 260, 3.0, 1.4), 1100, 2.0, 1.2)


def _eq_perc(x):     # keep the storm low-mids clear for the SFX roar
    return peq(peq(hp(x, 28, 2), 320, -4.0, 0.9), 3200, 2.0, 1.0)


def _eq_choir(x):
    return shelf(hp(x, 90, 2), 7000, -3.0, 'high')


def _eq_horn(x):
    return peq(hp(x, 70, 2), 300, -1.5, 1.0)


def _eq_cym(x):
    return hp(x, 400, 2)


def _eq_fx(x):
    return hp(x, 22, 2)


GCFG = {  # gain dB, hall send, room send, echo send, eq
    'mbox': dict(g=-4.0, hall=.26, room=0, echo=.30, eq=_eq_mbox),
    'bells': dict(g=-6.0, hall=.42, room=0, echo=.12, eq=lambda x: hp(x, 300, 2)),
    'piano': dict(g=5.0, hall=.34, room=.0, echo=0, eq=_eq_piano),
    'harp': dict(g=-7.0, hall=.30, room=0, echo=0, eq=_eq_harp),
    'pizz': dict(g=-7.0, hall=.22, room=.1, echo=0, eq=_eq_pizz),
    'str_hi': dict(g=-9.0, hall=.30, room=0, echo=0, eq=lambda x: I.string_body(x, 'vln')),
    'str_harm': dict(g=-15.0, hall=.45, room=0, echo=0, eq=lambda x: hp(x, 250, 2)),
    'str_lo': dict(g=-8.0, hall=.26, room=0, echo=0, eq=lambda x: hp(I.string_body(x, 'vc'), 38, 2)),
    'winds': dict(g=-7.0, hall=.30, room=0, echo=.08, eq=lambda x: hp(x, 180, 2)),
    'horn': dict(g=-5.0, hall=.30, room=0, echo=0, eq=_eq_horn),
    'choir': dict(g=-5.0, hall=.38, room=0, echo=0, eq=_eq_choir),
    'pad': dict(g=-22.0, hall=.30, room=0, echo=0, eq=lambda x: hp(x, 110, 2)),
    'perc': dict(g=-3.0, hall=.10, room=.28, echo=0, eq=_eq_perc),
    'sub': dict(g=-6.0, hall=.0, room=.05, echo=0, eq=lambda x: lp(hp(x, 24, 2), 180, 2)),
    'cym': dict(g=-11.0, hall=.20, room=0, echo=0, eq=_eq_cym),
    'fx': dict(g=-10.0, hall=.15, room=0, echo=0, eq=_eq_fx),
}

STR_DEF = {
    'vln1': dict(pan=-.5, spread=.45, bright=1.0),
    'vln2': dict(pan=-.2, spread=.4, bright=.9),
    'vla': dict(pan=.2, spread=.4, bright=.75),
    'vc': dict(pan=.42, spread=.35, bright=.6),
    'cb': dict(pan=.5, spread=.25, bright=.45, vib_cents=6.0),
    'vln_harm': dict(pan=.0, spread=.7, tone='harm', vib_cents=8.0),
}
HUMAN = {'mbox': .005, 'celesta': .004, 'glock': .003, 'piano': .010, 'harp': .007, 'pizz': .009}
CHOIR_PAN = {'choir_s': -.3, 'choir_a': -.1, 'choir_t': .1, 'choir_b': .3}


# ------------------------------------------------------------------ event rendering
def render_event(ev, idx):
    """-> (t_start, stereo array) or None"""
    inst, kw = ev['inst'], dict(ev['kw'])
    if ev['kind'] == 'note':
        midi = ev['midi'] + 12 * kw.pop('octave', 0)
        rng = rng_for('hum', inst, idx, round(ev['t'], 4), midi)
        t = ev['t'] + HUMAN.get(inst, 0) * float(np.clip(rng.standard_normal(), -2, 2))
        vel = float(np.clip(ev['vel'] * (1 + .04 * rng.standard_normal()), .02, 1.2))
        seed = idx
        ring = kw.get('ring')
        if inst == 'mbox':
            y = I.musicbox(midi, vel, seed=seed, ring=ring, decay=kw.get('decay', 1.0))
        elif inst == 'celesta':
            y = I.celesta(midi, vel, seed=seed, ring=ring)
        elif inst == 'glock':
            y = I.glock(midi, vel, seed=seed, ring=ring)
        elif inst == 'piano':
            y = I.piano(midi, vel, dur=ev['dur'] or 2.0, seed=seed, felt=0.85)
        elif inst == 'harp':
            y = I.harp(midi, vel, seed=seed, ring=ring, decay=kw.get('decay', 1.0))
        elif inst == 'pizz':
            y = I.pizz(midi, vel, seed=seed)
        else:
            raise ValueError(inst)
        pan = ev['pan'] if ev['pan'] is not None else float(np.clip((midi - 64) / 40.0, -.6, .6))
        from audio.music_dsp import pan_gains
        gl, gr = pan_gains(pan)
        return t, np.vstack([y * gl, y * gr])

    if ev['kind'] == 'line':
        notes = ev['notes']
        dyn = kw.pop('dyn', None)
        seed = kw.pop('seed', idx)
        if inst in STR_DEF:
            prm = dict(STR_DEF[inst])
            prm.update(kw)
            ts, y = I.strings_line(notes, seed=seed, **prm)
        elif inst in ('flute', 'ocarina'):
            ts, y = I.flute_line(notes, kind=inst, seed=seed, **kw)
            y = y * np.array([[0.93], [1.07]])        # slightly right of centre
        elif inst == 'horn':
            prm = dict(pan=.22)
            prm.update(kw)
            ts, y = I.horn_line(notes, seed=seed, **prm)
        elif inst.startswith('choir'):
            vowel = kw.pop('vowel', None)
            prm = dict(pan=CHOIR_PAN[inst], spread=.7)
            prm.update(kw)
            ts, src = I.choir_line(notes, seed=seed, **prm)
            y = I.formant(src, vowel or inst[-1])
        elif inst == 'pad_cold':
            ts, y = I.pad_line(notes, seed=seed, cutoff=1500.0, shape='tri', detune=9.0, glide=.06, **kw)
        elif inst == 'pad_warm':
            ts, y = I.pad_line(notes, seed=seed, cutoff=1100.0, shape='tri', detune=6.0, **kw)
        else:
            raise ValueError(inst)
        if dyn:
            tt = ts + np.arange(y.shape[1]) / SR
            y = y * np.interp(tt, [d[0] for d in dyn], [d[1] for d in dyn])[None, :]
        return ts, y

    # fx
    t = ev['t']
    mono = None
    if inst == 'timp':
        mono = P.timpani_hit(ev_midi(kw['pitch']), kw.get('vel', .8), seed=idx)
    elif inst == 'timp_roll':
        mono = P.timpani_roll(ev_midi(kw['pitch']), kw['dur'], kw.get('v0', .15), kw.get('v1', .9), seed=idx,
                              curve=kw.get('curve', 1.8))
    elif inst == 'taiko':
        mono = P.taiko(kw.get('vel', .9), kw.get('f0', 55.0), seed=idx)
    elif inst == 'tom':
        mono = P.tom(kw.get('vel', .8), kw.get('f0', 110.0), seed=idx)
    elif inst == 'sub':
        mono = P.sub_boom(kw.get('vel', 1.0), kw.get('f', 41.0))
    elif inst == 'heartbeat':
        mono = P.heartbeat(kw.get('vel', 1.0), kw.get('f', 46.0), seed=kw.get('seed', 0))
    elif inst == 'cymbal':
        y = P.cymbal(kw.get('vel', .8), 4.0, seed=kw.get('seed', idx), soft=kw.get('soft', False))
        return t, _spread(y, idx)
    elif inst == 'cym_swell':
        y = P.cymbal_swell(kw['dur'], kw.get('peak', .8), seed=idx)
        return t, _spread(y, idx)
    elif inst == 'rev_cym':
        y = P.reverse_cymbal(kw['dur'], kw.get('peak', .8), seed=idx)
        return t, _spread(y, idx)
    elif inst == 'air':
        y = P.air(kw['dur'], seed=idx, peak=kw.get('peak', .3))
        return t, fade(y, kw.get('fin', 1.0), kw.get('fout', 1.0))
    elif inst == 'rumble':
        return t, P.rumble(kw['dur'], seed=idx, peak=kw.get('peak', .6))
    elif inst == 'rev_note':
        src = I.harp(ev_midi(kw['pitch']), kw.get('vel', .5), seed=idx)
        d = ns(kw.get('dur', 1.0))
        r = src[::-1][-d:] if len(src) >= d else src[::-1]
        r = fade(r, .15, .01)
        from audio.music_dsp import pan_gains
        gl, gr = pan_gains(float(np.clip((ev_midi(kw['pitch']) - 64) / 30, -.6, .6)))
        return t - len(r) / SR, np.vstack([r * gl, r * gr])
    else:
        raise ValueError(inst)
    return t, np.vstack([mono, mono]) * np.sqrt(.5)


def ev_midi(p):
    from audio.music_dsp import m
    return m(p)


def _spread(y, idx):
    """decorrelated stereo for cymbals: tiny inter-channel delay."""
    d = 12 + idx % 17
    r = np.concatenate([np.zeros(d), y[:-d]])
    return np.vstack([y, r]) * np.sqrt(.5)


# ------------------------------------------------------------------ segments
SEG_GAIN = {
    'A': [(0.0, 0.0), (C['cut_s5'] - .03, 0.0), (C['cut_s5'] + .02, 2.5), (C['cut_s6'], 2.5)],
    'C': [(0.0, 0.0), (C['cut_s8b_climax'] - .03, 0.0), (C['cut_s8b_climax'] + .01, 1.5),
          (C['cut_s9a'] - .5, 1.5), (C['cut_s9a'] + .5, 0.0), (99, 0.0)],
}


def seg_of(t):
    if t < C['cut_s6'] - 1e-6:
        return 'A'
    if t < C['silence'] - 1e-6:
        return 'B'
    return 'C'


SEGS = {
    'A': (0.0, C['cut_s6']),
    'B': (C['cut_s6'], C['silence']),
    'C': (C['first_pulse'], C['end']),
}


def _key(obj):
    h = hashlib.sha1()
    h.update(pickle.dumps(obj))
    for f in ('music_inst.py', 'music_perc.py', 'music_dsp.py'):
        with open(os.path.join(ROOT, 'audio', f), 'rb') as fh:
            h.update(fh.read())
    return h.hexdigest()[:20]


def render_group(seg, group, evs):
    """dry stereo bus for one group within one segment (cached)."""
    t0, t1 = SEGS[seg]
    n = ns(t1 - t0)
    key = _key((seg, group, [(i, e['kind'], e['inst'], e['t'], e.get('midi'), e.get('vel'), e.get('dur'),
                              e.get('pan'), e.get('notes'), sorted(e['kw'].items(), key=lambda kv: kv[0]))
                             for i, e in evs]))
    path = os.path.join(CACHE, f'{seg}_{group}_{key}.npy') if CACHE else None
    if path and os.path.exists(path):
        try:
            return np.load(path)
        except Exception:
            pass
    bus = np.zeros((2, n))
    for i, e in evs:
        ts, y = render_event(e, i)
        i0 = ns(ts - t0)
        a, b = max(i0, 0), min(i0 + y.shape[1], n)
        if b > a:
            bus[:, a:b] += y[:, a - i0:b - i0]
    if path:
        try:
            os.makedirs(CACHE, exist_ok=True)
            np.save(path, bus)
        except OSError:
            pass
    return bus


_IR = {}


def irs():
    if not _IR:
        _IR['hall'] = make_ir(2.9, 0.032, damp_hz=4800.0, seed=11)
        _IR['room'] = make_ir(1.0, 0.012, damp_hz=6000.0, seed=12, lf_mult=1.0)
    return _IR


def gate_curve(n, t0, pts):
    tt = t0 + np.arange(n) / SR
    return np.interp(tt, [p[0] for p in pts], [p[1] for p in pts])


def cos_fade_pts(a, b, steps=24, shape=1.0):
    """breakpoints: 1 at a -> 0 at b (raised cosine^shape)."""
    xs = np.linspace(0, 1, steps)
    return [(a + (b - a) * x, float((0.5 + 0.5 * np.cos(np.pi * x)) ** shape)) for x in xs]


def render_segment(seg, events, verbose=True):
    t0, t1 = SEGS[seg]
    n = ns(t1 - t0)
    groups = {}
    for i, e in enumerate(events):
        if seg_of(e['t']) != seg:
            continue
        groups.setdefault(GROUP[e['inst']], []).append((i, e))
    dry = np.zeros((2, n))
    s_hall = np.zeros((2, n))
    s_room = np.zeros((2, n))
    s_echo = np.zeros((2, n))
    stems = {}
    for g, evs in groups.items():
        tt = time.time()
        bus = render_group(seg, g, evs)
        cfg = GCFG[g]
        bus = cfg['eq'](bus) * db2a(cfg['g'])
        stems[g] = bus
        dry += bus
        s_hall += cfg['hall'] * bus
        s_room += cfg['room'] * bus
        s_echo += cfg['echo'] * bus
        if verbose:
            print(f'    {seg} {g:9s} {len(evs):4d} ev  {time.time() - tt:6.1f}s  peak {a2db(np.max(np.abs(bus))):6.1f} dB')
    ir = irs()
    wet = width(convolve_reverb(s_hall, ir['hall']), 1.3) + convolve_reverb(s_room, ir['room']) * 0.8
    if np.any(s_echo):
        e = snes_echo(s_echo, .40, .60, .34, 5, 3000.0)      # synced: 8th / dotted 8th at 75 BPM
        wet += e + convolve_reverb(e, ir['hall']) * .25
    # gates (dry and wet separately) - guarantees the hard silences
    if seg == 'A':
        gd = gate_curve(n, t0, [(t0, 1.0)] + cos_fade_pts(C['storm_out'], t1 - .004, shape=0.6) + [(t1, 0.0)])
        # the water drop lands in near-silence: once the music has breathed out, the hall tail is
        # tightened (like the room holding its breath) and restored after the drop for the plink/droop
        dp, dh = C['drop_fall'] - .25, C['drop_hit']
        drop_db = gate_curve(n, t0, [(t0, 0.0), (dp, 0.0), (dh - .3, -14.0), (dh + .02, -27.0),
                                     (dh + .15, -20.0), (dh + .45, 0.0), (t1, 0.0)])
        gw = gd * db2a(drop_db)
    elif seg == 'B':
        gd = gate_curve(n, t0, [(t0, 1.0)] + cos_fade_pts(C['eye_off'] - .1, t1 - .004) + [(t1, 0.0)])
        gw = gate_curve(n, t0, [(t0, 1.0)] + cos_fade_pts(C['power_down'] + .2, t1 - .05) + [(t1, 0.0)])
    else:
        # the final chord decays on its own; the gate is only a safety net over near-silence
        gd = gate_curve(n, t0, [(t0, 1.0)] + cos_fade_pts(t1 - .45, t1 - .005) + [(t1, 0.0)])
        tt = t0 + np.arange(n) / SR
        gw = gd * db2a(-19.0 * np.clip(tt - (C['title'] + .5), 0, None))   # tail tightens after the cadence
    out = hp(dry, 18, 2) * gd[None, :] + hp(wet, 18, 2) * gw[None, :]
    if seg in SEG_GAIN:   # section level automation (dB breakpoints)
        pts = SEG_GAIN[seg]
        out = out * db2a(gate_curve(n, t0, pts))[None, :]
    if seg == 'A':   # storm: carve the low-mids (the SFX roar lives there)
        w = gate_curve(n, t0, [(t0, 0.0), (C['cut_s5'] - .15, 0.0), (C['cut_s5'] + .1, 1.0), (t1, 1.0)])
        dip = peq(peq(out, 300, -4.5, 0.8), 160, -2.0, 1.0)
        out = out * (1 - w)[None, :] + dip * w[None, :]
    stems = {k: v * gd[None, :] for k, v in stems.items()}
    return out, stems


# ------------------------------------------------------------------ master
def master(mix):
    """find gain G so that limiter(mix * G) integrates to TARGET_LUFS."""
    G = float(db2a(TARGET_LUFS - lufs_integrated(mix)))
    for it in range(6):
        x, g = limiter(mix * G, CEILING_DB, 0.006, 0.08)
        L = lufs_integrated(x)
        if abs(L - TARGET_LUFS) < 0.02:
            break
        G *= float(db2a(TARGET_LUFS - L))
    gr = -a2db(g)
    print(f'  limiter: max GR {gr.max():.1f} dB at {np.argmax(gr) / SR:.2f}s; '
          f'time with GR>1 dB: {np.sum(gr > 1) / SR:.2f}s, >3 dB: {np.sum(gr > 3) / SR:.2f}s')
    for a_, b_ in ((0, 26), (26, 32), (32, 45.3), (45.3, 48), (48, 53), (53, 60)):
        seg = gr[ns(a_):ns(b_)]
        print(f'    GR {a_:>4}-{b_:<4}: max {seg.max():4.1f} dB, mean {seg.mean():.2f} dB')
    tp = true_peak_db(x)
    if tp > -1.0:
        x *= db2a(-1.0 - tp - 0.05)
    return x


def render(plots=True, verbose=True):
    t_start = time.time()
    S = build()
    ev = S.ev
    mix = np.zeros((2, N))
    all_stems = {}
    for seg in ('A', 'B', 'C'):
        if verbose:
            print(f'  segment {seg}')
        out, stems = render_segment(seg, ev, verbose)
        t0 = SEGS[seg][0]
        i0 = ns(t0)
        k = min(out.shape[1], N - i0)
        mix[:, i0:i0 + k] += out[:, :k]
        for g, s in stems.items():
            if g not in all_stems:
                all_stems[g] = np.zeros((2, N))
            all_stems[g][:, i0:i0 + k] += s[:, :k]
    x = master(mix)
    # absolute digital silence between 'silence' and 'first_pulse'
    x[:, ns(C['silence']):ns(C['first_pulse'])] = 0.0
    x[:, 0] = 0.0
    x[:, -1] = 0.0
    x = x[:, :N].astype(np.float32)
    assert x.shape == (2, 2880000), x.shape
    wavfile.write(os.path.join(ROOT, 'audio', 'music.wav'), SR, x.T)
    if verbose:
        print(f'  wrote audio/music.wav  ({time.time() - t_start:.1f}s)')
    if plots:
        from audio import music_analyze as A
        A.report(x.astype(np.float64), all_stems, C)
    return x


if __name__ == '__main__':
    render(plots='--noplots' not in sys.argv)
