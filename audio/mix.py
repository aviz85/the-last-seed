"""
Final mix: music stem + SFX stem -> audio/mix.wav (48 kHz stereo float32, 60.000 s).

  python audio/mix.py [--music_db 0] [--sfx_db 0]

Bus chain: per-stem gain -> section automation -> sum -> gentle glue compression
-> lookahead peak limiter (ceiling -1.0 dBFS) -> exact silence window -> fades.
"""
import argparse
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import maximum_filter1d, uniform_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cues import CUES

SR = 48000
N = SR * 60


def load(p):
    sr, x = wavfile.read(p)
    assert sr == SR, (p, sr)
    x = x.astype(np.float32)
    if x.dtype != np.float32 or np.abs(x).max() > 2.0:
        x = x / 32768.0
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    if len(x) < N:
        x = np.concatenate([x, np.zeros((N - len(x), 2), np.float32)])
    return x[:N]


def db(g):
    return 10 ** (g / 20.0)


def env_points(points):
    """Piecewise-linear gain envelope (in dB) from [(t, dB), ...]."""
    t = np.arange(N) / SR
    ts = [p[0] for p in points]
    gs = [p[1] for p in points]
    return db(np.interp(t, ts, gs)).astype(np.float32)[:, None]


def limiter(x, ceiling_db=-1.0, look_ms=3.0, rel_ms=120.0):
    c = db(ceiling_db)
    look = int(SR * look_ms / 1000)
    peak = np.abs(x).max(1)
    need = np.maximum(peak / c, 1.0)
    need = maximum_filter1d(need, size=2 * look + 1)
    g = 1.0 / need
    # smooth release (one-pole on the gain, attack instant thanks to lookahead max)
    a = np.exp(-1.0 / (SR * rel_ms / 1000))
    out = np.empty_like(g)
    cur = 1.0
    for i in range(0, len(g), 64):          # block-wise for speed
        blk = g[i:i + 64]
        m = blk.min()
        cur = min(m, cur * a ** 64 + (1 - a ** 64) * 1.0) if m < cur else cur * a ** 64 + (1 - a ** 64) * 1.0
        cur = min(cur, m)
        out[i:i + 64] = cur
    out = uniform_filter1d(out, size=look)  # de-zipper
    out = np.minimum(out, g * 1.0 + 0)      # never exceed required reduction
    return x * out[:, None].astype(np.float32), 20 * np.log10(out.min() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--music_db', type=float, default=-1.0)
    ap.add_argument('--sfx_db', type=float, default=0.0)
    a = ap.parse_args()
    mus = load(os.path.join(ROOT, 'audio', 'music.wav')) * db(a.music_db)
    sfx = load(os.path.join(ROOT, 'audio', 'sfx.wav')) * db(a.sfx_db)

    C = CUES
    # music automation: let key foley/voice moments breathe
    mus_env = env_points([
        (0.0, 0.0),
        (C['drop_fall'] - 0.2, 0.0), (C['drop_fall'], -3.0), (C['drop_hit'] + 0.5, -3.0), (C['droop'], 0.0),
        (C['cut_c02_sprout'] - 0.1, 0.0), (C['cut_c02_sprout'] + 0.1, -1.5), (C['cut_s3b'], 0.0),
        (C['happy_beep'] - 0.1, 0.0), (C['happy_beep'], -2.5), (C['happy_beep'] + 0.6, 0.0),
        (C['cut_s5'] - 0.05, 0.0), (C['cut_s5'], -1.0), (C['storm_out'], -1.0), (C['cut_s6'], 0.0),
        (C['robot_notice'] - 0.1, 0.0), (C['robot_notice'], -2.0), (C['robot_notice'] + 0.6, 0.0),
        (C['eye_glow'] - 0.1, 0.0), (C['eye_glow'], -2.0), (C['eye_glow'] + 0.8, 0.0),
        (60.0, 0.0),
    ])
    mix = mus * mus_env + sfx

    # gentle glue: slow RMS compressor on the sum (2:1 above -18 dBFS RMS)
    rms = np.sqrt(uniform_filter1d((mix ** 2).mean(1), size=int(SR * 0.3)) + 1e-12)
    lvl = 20 * np.log10(rms)
    over = np.maximum(lvl - (-18.0), 0)
    gr = -over * (1 - 1 / 2.0)
    gr = uniform_filter1d(gr, size=int(SR * 0.05))
    mix = mix * db(gr)[:, None].astype(np.float32)

    # makeup so the program sits around -14..-15 LUFS-ish before limiting
    mix = mix * db(2.0)
    mix, red = limiter(mix, -1.6)

    t = np.arange(N) / SR
    sil = (t >= C['silence']) & (t < C['first_pulse'])
    mix[sil] = 0.0
    fade = int(SR * 0.01)
    mix[:fade] *= np.linspace(0, 1, fade)[:, None]
    mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
    mix = np.clip(mix, -0.95, 0.95)
    out = os.path.join(ROOT, 'audio', 'mix.wav')
    wavfile.write(out, SR, mix.astype(np.float32))
    pk = 20 * np.log10(np.abs(mix).max())
    print(f'wrote {out}: peak {pk:.2f} dBFS, max limiter reduction {red:.2f} dB')


if __name__ == '__main__':
    main()
