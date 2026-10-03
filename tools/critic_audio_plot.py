import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cues import CUES
import film

SR = 48000
def load(p):
    sr, x = wavfile.read(os.path.join(ROOT, 'audio', p)); return x.astype(np.float64)
mix, mus, sfx = load('mix.wav'), load('music.wav'), load('sfx.wav')
def env_db(x, win=0.05):
    n = int(SR * win); m = (x ** 2).mean(1)
    k = len(m) // n
    r = np.sqrt(m[:k * n].reshape(k, n).mean(1))
    return np.arange(k) * win, 20 * np.log10(r + 1e-9)
cuts = [s.start for s in film.TIMELINE[1:]]
t0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
t1 = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
tag = sys.argv[3] if len(sys.argv) > 3 else 'audio_overview'
fig, ax = plt.subplots(3, 1, figsize=(18, 10), sharex=True, gridspec_kw={'height_ratios': [2, 2, 2]})
for x, lab, c in ((mix, 'mix', 'k'), (mus, 'music', 'tab:blue'), (sfx, 'sfx', 'tab:orange')):
    t, e = env_db(x, 0.02 if t1 - t0 < 10 else 0.05)
    ax[0].plot(t, e, c, lw=0.8, label=lab)
ax[0].set_ylim(-70, 0); ax[0].legend(loc='lower left'); ax[0].set_ylabel('RMS dBFS')
ss = mix.mean(1)[int(t0 * SR):int(t1 * SR)]
f, tt, Z = stft(ss, SR, nperseg=2048, noverlap=2048 - 480)
ax[1].pcolormesh(tt + t0, f, 20 * np.log10(np.abs(Z) + 1e-7), vmin=-110, vmax=-20, shading='auto', cmap='magma')
ax[1].set_yscale('symlog', linthresh=200); ax[1].set_ylim(40, 20000); ax[1].set_ylabel('mix Hz')
ss = sfx.mean(1)[int(t0 * SR):int(t1 * SR)]
f, tt, Z = stft(ss, SR, nperseg=1024, noverlap=1024 - 240)
ax[2].pcolormesh(tt + t0, f, 20 * np.log10(np.abs(Z) + 1e-7), vmin=-110, vmax=-20, shading='auto', cmap='magma')
ax[2].set_yscale('symlog', linthresh=200); ax[2].set_ylim(40, 20000); ax[2].set_ylabel('sfx Hz')
for a in ax:
    for c in cuts:
        if t0 <= c <= t1: a.axvline(c, color='cyan', lw=0.8, ls='--')
    for k, v in CUES.items():
        if t0 <= v <= t1 and not k.startswith('cut'):
            a.axvline(v, color='lime', lw=0.6, alpha=0.7)
for k, v in CUES.items():
    if t0 <= v <= t1 and not k.startswith('cut'):
        ax[0].text(v, -2, k, rotation=90, fontsize=7, va='top', color='green')
ax[2].set_xlim(t0, t1); ax[2].set_xlabel('s (cyan dashed = picture cuts, green = cues)')
plt.tight_layout()
p = os.path.join(ROOT, 'review', 'critic', f'{tag}.png')
plt.savefig(p, dpi=70)
print(p)
