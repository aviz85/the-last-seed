"""Instrument auditions -> audio/music_audition.wav + review/music_audition_*.png
run:  python -m audio.music_audition [group]"""
import sys
import time
import numpy as np
from scipy.io import wavfile

from audio.music_dsp import SR, m, ns, Bus, make_ir, convolve_reverb, a2db
from audio import music_inst as I
from audio import music_perc as P
from audio.music_analyze import spec_png, find_clicks


def timed(label, fn, *a, **k):
    t = time.time()
    r = fn(*a, **k)
    print(f'  {label:28s} {time.time() - t:6.2f}s')
    return r


def group_tonal(bus):
    t = 0.2
    # music box: robot theme head (3/4, 0.8 s beats)
    theme = [('D5', 1), ('F5', 1), ('A5', 1), ('C6', 2), ('Bb5', 1), ('A5', 1), ('G5', 1), ('F5', 1), ('E5', 3)]
    tt = t
    for nm, b in theme:
        bus.add(timed('musicbox ' + nm, I.musicbox, m(nm), 0.7), tt, -0.1)
        tt += b * 0.8
    t = tt + 0.5
    for i, nm in enumerate(['F5', 'A5', 'C6', 'E6', 'F6']):
        bus.add(I.celesta(m(nm), 0.6), t + i * 0.12, 0.2)
    bus.add(I.glock(m('C7'), 0.5), t + 0.8, 0.3)
    t += 2.5
    # piano: Dm chord + melody
    for nm, dt in [('D2', 0), ('A2', 0.15), ('F3', 0.3), ('D4', 0.45)]:
        bus.add(timed('piano ' + nm, I.piano, m(nm), 0.45, 3.0), t + dt, -0.2)
    for i, nm in enumerate(['D5', 'F5', 'A5', 'C6']):
        bus.add(I.piano(m(nm), 0.4, 1.6), t + 0.8 + i * 0.6, 0.15)
    t += 4.2
    # harp arpeggio + pizz
    for i, nm in enumerate(['F2', 'C3', 'A3', 'C4', 'F4', 'G4', 'A4', 'C5', 'F5']):
        bus.add(I.harp(m(nm), 0.6), t + i * 0.11, -0.4 + i * 0.1)
    for i, nm in enumerate(['D3', 'A3', 'F3', 'A3']):
        bus.add(I.pizz(m(nm), 0.7), t + 1.6 + i * 0.5, 0.1)
    t += 4.0
    return t


def group_lines(bus, t):
    # strings chord progression SATB (Bb -> F/A -> Gm7 -> D)
    parts = {
        'vln1': [('D5', 1.2), ('C5', 1.2), ('Bb4', 1.2), ('A4', 2.0)],
        'vln2': [('F4', 1.2), ('F4', 1.2), ('F4', 1.2), ('F#4', 2.0)],
        'vla': [('Bb3', 1.2), ('A3', 1.2), ('D4', 1.2), ('D4', 2.0)],
        'vc': [('Bb2', 1.2), ('A2', 1.2), ('G2', 1.2), ('D2', 2.0)],
    }
    for i, (nm, seq) in enumerate(parts.items()):
        notes, tt = [], t
        for p, d in seq:
            notes.append((tt, d, m(p), 0.65))
            tt += d
        ts, y = timed('strings ' + nm, I.strings_line, notes, voices=6, att=0.5, rel=0.8, seed=i,
                      pan=[-0.5, -0.2, 0.2, 0.45][i], spread=0.3)
        bus.add(y, ts)
    t += 6.2
    # flute life theme (F major) then ocarina
    life = [('C5', 0.5), ('A5', 1.0), ('G5', 0.25), ('F5', 0.25), ('G5', 0.75), ('F5', 0.25), ('D5', 1.0)]
    for kind, off in (('flute', 0.0), ('ocarina', 4.4)):
        notes, tt = [], t + off
        for p, d in life:
            notes.append((tt, d, m(p), 0.7))
            tt += d
        ts, y = timed(kind, I.flute_line, notes, kind=kind)
        bus.add(y, ts, 0.1)
    t += 9.0
    # choir chord G major (SATB) 'aah'
    ch = {'s': ['D5', 'B4'], 'a': ['G4', 'G4'], 't': ['B3', 'D4'], 'b': ['G2', 'G2']}
    for reg, seq in ch.items():
        notes = [(t, 1.5, m(seq[0]), 0.7), (t + 1.5, 1.5, m(seq[1]), 0.8)]
        ts, src = timed('choir ' + reg, I.choir_line, notes, voices=6, seed=hash(reg) % 100)
        bus.add(I.formant(src, reg), ts)
    t += 4.5
    # horn phrase
    notes = [(t, 0.6, m('G3'), 0.6), (t + 0.6, 0.6, m('B3'), 0.7), (t + 1.2, 0.6, m('D4'), 0.8),
             (t + 1.8, 1.2, m('F#4'), 0.85)]
    ts, y = timed('horn', I.horn_line, notes)
    bus.add(y, ts)
    t += 3.8
    # pad
    ts, y = timed('pad', I.pad_line, [(t, 3.0, m('D3'), 0.6)] + [], cutoff=800)
    bus.add(y, ts)
    ts, y = I.pad_line([(t, 3.0, m('A3'), 0.5)], cutoff=800, seed=1)
    bus.add(y, ts)
    t += 5.0
    return t


def group_perc(bus, t):
    bus.add(timed('timpani', P.timpani_hit, m('G2'), 0.9), t, 0.0)
    bus.add(timed('timpani roll', P.timpani_roll, m('D2'), 2.0, 0.1, 0.9), t + 1.2, 0.0)
    t += 5.0
    for i in range(4):
        bus.add(P.taiko(0.9 - 0.15 * (i % 2), 55.0, i), t + i * 0.5, 0.0)
    bus.add(P.tom(0.8, 110, 1), t + 2.0, -0.3)
    bus.add(P.tom(0.8, 150, 2), t + 2.25, 0.3)
    t += 3.5
    bus.add(timed('cymbal', P.cymbal, 0.8), t, 0.0)
    t += 3.0
    bus.add(timed('cymbal swell', P.cymbal_swell, 2.0, 0.8), t, 0.0)
    t += 3.8
    bus.add(P.reverse_cymbal(1.5, 0.8), t, 0.0)
    bus.add(P.timpani_hit(m('G2'), 0.9), t + 1.5)
    t += 4.0
    bus.add(P.heartbeat(1.0), t)
    t += 2.0
    bus.add(P.air(4.0, 0, 0.8), t)
    bus.add(P.rumble(4.0, 0, 0.8), t)
    t += 4.5
    return t


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    bus = Bus(ns(110))
    t0 = time.time()
    t = 0.2
    if which in ('all', 'tonal'):
        t = group_tonal(bus)
    if which in ('all', 'lines'):
        t = group_lines(bus, t)
    if which in ('all', 'perc'):
        t = group_perc(bus, t)
    n = ns(t + 1.0)
    dry = bus.x[:, :n]
    ir = make_ir(2.6, 0.03)
    wet = convolve_reverb(dry, ir)
    mix = dry + 0.25 * wet
    mix *= 0.5 / (np.max(np.abs(mix)) + 1e-9)
    print(f'render {time.time() - t0:.1f}s, dur {t:.1f}s')
    wavfile.write(f'audio/music_audition_{which}.wav', SR, mix.T.astype(np.float32))
    # plots in 12 s chunks
    k = 0
    for s in np.arange(0, t, 12.0):
        a, b = ns(s), min(ns(s + 12.0), n)
        spec_png(mix[:, a:b], f'review/music_audition_{which}_{k}.png', t0=s, title=f'audition {which} {s:.0f}s',
                 fmax=16000, width=14, height=6)
        k += 1
    cl = find_clicks(dry)
    print('clicks (t, ratio, dB):', [(round(a, 3), round(b, 1), round(c, 1)) for a, b, c in cl[:40]])
