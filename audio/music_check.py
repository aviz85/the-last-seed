"""Symbolic harmony check of the score: flags simultaneous minor 2nds / major
7ths / minor 9ths / tritones between notes that actually overlap for a while.
    python -m audio.music_check"""
import numpy as np

from cues import CUES as C
from audio.music_score import build

RING = {'mbox': 1.2, 'celesta': .6, 'glock': .8, 'harp': .8, 'pizz': .35}   # audible overlap window (s)
INTENTIONAL = [(C['cut_s5'], C['cut_s6'], 'storm'), (C['robot_looks_up'], C['cut_s5'], 'phrygian warning'),
               (C['core_out'] - .4, C['core_out'], 'harp glissando into core_out')]
NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def nm(mi):
    return f'{NAMES[mi % 12]}{mi // 12 - 1}'


def spans(S):
    out = []
    for e in S.ev:
        if e['kind'] == 'note':
            mi = e['midi'] + 12 * e['kw'].get('octave', 0)
            d = RING.get(e['inst'], 1.0) if e['inst'] != 'piano' else min(e['dur'] or 2.0, 3.0)
            if e['kw'].get('ring'):
                d = min(d, e['kw']['ring'] - 0.1)
            out.append((e['t'], e['t'] + d, mi, e['inst']))
        elif e['kind'] == 'line':
            for t, d, mi, v in e['notes']:
                out.append((t, t + d, mi, e['inst']))
    return out


def check(step=0.05, min_overlap=0.15):
    S = build()
    sp = spans(S)
    flagged = {}
    for t in np.arange(0, C['end'], step):
        if any(a <= t < b for a, b, _ in INTENTIONAL):
            continue
        cur = [s for s in sp if s[0] <= t < s[1]]
        for i in range(len(cur)):
            for j in range(i + 1, len(cur)):
                a, b = sorted((cur[i], cur[j]), key=lambda s: s[2])
                iv = abs(a[2] - b[2])
                if iv % 12 in (1, 11, 6) and iv > 0:
                    ov = min(a[1], b[1]) - max(a[0], b[0])
                    if ov < min_overlap:
                        continue
                    key = (round(max(a[0], b[0]), 2), nm(min(a[2], b[2])), nm(max(a[2], b[2])))
                    if key not in flagged:
                        flagged[key] = (a[3], b[3], iv, round(ov, 2))
    for k in sorted(flagged):
        a, b, iv, ov = flagged[k]
        kind = {1: 'm2', 11: 'M7', 6: 'TT'}[iv % 12]
        print(f'  t={k[0]:6.2f}  {k[1]:>4s} ({a:8s}) vs {k[2]:>4s} ({b:8s})  {kind:3s} int={iv:2d} overlap={ov}s')
    print(f'{len(flagged)} flagged')


if __name__ == '__main__':
    check()
