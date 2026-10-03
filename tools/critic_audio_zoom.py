import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SR = 48000
def load(p):
    sr, x = wavfile.read(os.path.join(ROOT, 'audio', p)); return x.astype(np.float64).mean(1)
stems = {n: load(n) for n in ('sfx_fx.wav', 'sfx_robot.wav', 'music.wav', 'mix.wav')}
# windows: (t0, t1, picture event time, label)
wins = [(9.4, 9.9, 9.633, 'drop hit (crown visible 9.633; blank 9.600)'),
        (33.05, 33.45, 33.2, 'sit impact (picture 33.200)'),
        (40.95, 41.6, 41.2, 'core out flare (picture 41.200)'),
        (46.6, 47.2, 46.833, 'growth burst flash (picture 46.833)'),
        (45.1, 45.6, 45.3, 'first pulse'),
        (15.1, 15.45, 15.2, 'iris snap (picture 15.200)')]
fig, axes = plt.subplots(len(wins), 1, figsize=(14, 3 * len(wins)))
for ax, (t0, t1, te, lab) in zip(axes, wins):
    i0, i1 = int(t0 * SR), int(t1 * SR)
    tt = np.arange(i0, i1) / SR
    for n, c in (('music.wav', 'tab:blue'), ('sfx_fx.wav', 'tab:orange'), ('sfx_robot.wav', 'tab:green')):
        x = stems[n][i0:i1]
        # 2 ms RMS envelope in dB
        w = int(SR * 0.002)
        k = len(x) // w
        e = 20 * np.log10(np.sqrt((x[:k * w] ** 2).reshape(k, w).mean(1)) + 1e-9)
        ax.plot(t0 + np.arange(k) * w / SR, e, c, lw=0.9, label=n)
    ax.axvline(te, color='r', lw=1.2)
    for f in np.arange(np.ceil(t0 * 30), t1 * 30):
        ax.axvline(f / 30, color='grey', lw=0.3, alpha=0.5)
    ax.set_ylim(-80, 0); ax.set_xlim(t0, t1); ax.set_title(lab, fontsize=10); ax.legend(fontsize=7, loc='upper right')
plt.tight_layout()
p = os.path.join(ROOT, 'review', 'critic', 'audio_sync_zoom.png')
plt.savefig(p, dpi=70); print(p)
