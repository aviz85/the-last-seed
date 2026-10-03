"""Verify musical hits against cues.CUES: spectral-flux onset peaks near each
cue in the rendered audio/music.wav.   python -m audio.music_sync"""
import numpy as np
from scipy import signal
from scipy.io import wavfile

from cues import CUES as C
from audio.music_dsp import SR, ns

# cue -> (expected offset of the musical event from the cue, band lo, band hi, what)
HITS = {
    'droop': (0.0, 500, 1200, 'music box A5 (droop phrase)'),
    'cut_s3': (0.0, 500, 1200, 'music box D5 lands'),
    'glint': (0.0, 1500, 4000, 'celesta glint'),
    'cut_c02_sprout': (0.0, 80, 700, 'LIFE theme + harp'),
    'cut_s4': (0.0, 80, 500, 'journey pizz/harp'),
    'look_down_can': (0.0, 850, 1000, 'flute Bb5'),
    'cut_s5': (0.0, 30, 200, 'storm hit'),
    'shield': (0.0, 250, 1200, 'robot motif (horn/vln)'),
    'antenna_rip': (0.0, 30, 8000, 'STAB 1'),
    'core_flicker': (0.0, 30, 8000, 'STAB 2'),
    'trudge': (0.0, 30, 200, 'trudge taiko'),
    'sprout_lean': (0.0, 100, 1000, 'piano LIFE answer'),
    'robot_notice': (0.0, 80, 800, 'piano question'),
    'look_chest': (0.0, 25, 120, 'realization low note'),
    'cut_s7': (0.0, 90, 600, 'gift strings'),
    'hatch_open': (0.0, 700, 900, 'rising sixth G5'),
    'core_out': (0.0, 60, 3000, 'LUMINOUS D major'),
    'first_pulse': (0.0, 25, 120, 'heartbeat'),
    'veins': (0.0, 150, 700, 'rising arpeggios'),
    'growth_burst': (0.0, 60, 2000, 'growth burst'),
    'cut_s8b_climax': (0.0, 25, 8000, 'CLIMAX hit'),
    'color_wave': (0.025, 950, 1050, 'LIFE leap to B5'),
    'sun_break': (-0.0125, 30, 8000, 'sun break wave'),
    'cut_s9a': (0.0, 90, 1000, 'resolution'),
    'eye_glow': (0.0, 1000, 3200, 'music box D6 + celesta sparkle'),
    'title': (0.4, 40, 1000, 'final tonic (title + 0.4)'),
}


def flux(x, lo, hi, hop=0.004, win=0.023):
    mono = x.mean(axis=0)
    f, t, Z = signal.stft(mono, SR, nperseg=ns(win) // 2 * 2, noverlap=ns(win) // 2 * 2 - ns(hop))
    sel = (f >= lo) & (f <= hi)
    mag = np.log1p(1000 * np.abs(Z[sel]))
    d = np.diff(mag, axis=1)
    fl = np.sum(np.maximum(d, 0), axis=0)
    return t[1:], fl


def main():
    sr, x = wavfile.read('audio/music.wav')
    x = x.T.astype(np.float64)
    print(f'{"cue":16s} {"t":>6s}  {"onset":>7s}  {"err ms":>7s}  what')
    worst = 0
    for cue, (off, lo, hi, what) in HITS.items():
        tc = C[cue] + off
        a, b = ns(max(tc - 0.6, 0)), ns(min(tc + 0.6, 60))
        t, fl = flux(x[:, a:b], lo, hi)
        t = t + a / SR
        sel = (t >= tc - 0.1) & (t <= tc + 0.1)
        if not sel.any():
            continue
        i = np.argmax(np.where(sel, fl, -1))
        err = (t[i] - tc) * 1000
        # strength relative to the local median flux
        rel = fl[i] / (np.median(fl) + 1e-9)
        worst = max(worst, abs(err))
        print(f'{cue:16s} {tc:6.2f}  {t[i]:7.3f}  {err:+7.1f}  {what} (x{rel:.1f})')
    print(f'worst |err| = {worst:.1f} ms  (+/-2 frames = 66.7 ms)')


if __name__ == '__main__':
    main()
