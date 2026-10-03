"""critic_audio — onset / level analysis of the stems vs the cue sheet. Outputs to review/critic/."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
from cues import CUES

SR = 48000
OUT = os.path.join(ROOT, 'review', 'critic')


def load(p):
    sr, x = wavfile.read(os.path.join(ROOT, 'audio', p))
    return x.astype(np.float64).mean(1)


def onset_env(x, hop=240, nfft=1024):
    f, t, Z = stft(x, SR, nperseg=nfft, noverlap=nfft - hop, boundary=None, padded=False)
    M = np.log1p(1000 * np.abs(Z))
    flux = np.maximum(np.diff(M, axis=1), 0).sum(0)
    tt = t[1:]
    return tt, flux, f, t, M


def peaks(tt, env, thr_rel=0.15, min_gap=0.04):
    thr = thr_rel * np.percentile(env, 99.5)
    pk = []
    for i in range(1, len(env) - 1):
        if env[i] > thr and env[i] >= env[i - 1] and env[i] >= env[i + 1]:
            if pk and tt[i] - pk[-1][0] < min_gap:
                if env[i] > pk[-1][1]:
                    pk[-1] = (tt[i], env[i])
                continue
            pk.append((tt[i], env[i]))
    return pk


if __name__ == '__main__':
    res = {}
    for name in ('sfx.wav', 'music.wav', 'mix.wav', 'sfx_fx.wav', 'sfx_robot.wav', 'sfx_amb.wav'):
        x = load(name)
        tt, env, f, t, M = onset_env(x)
        res[name] = (tt, env)
        np.savez(os.path.join(OUT, 'onset_' + name.replace('.wav', '.npz')), tt=tt, env=env)
    tt, env = res['sfx.wav']
    pk = peaks(tt, env)
    print('number of sfx onsets', len(pk))
    for cue, ct in sorted(CUES.items(), key=lambda kv: kv[1]):
        near = [(round(p[0] - ct, 3), round(p[1] / np.percentile(env, 99.5), 2)) for p in pk if abs(p[0] - ct) < 0.25]
        print(f'{cue:16s} {ct:6.2f}  sfx onsets within 250ms (dt, strength): {near}')
