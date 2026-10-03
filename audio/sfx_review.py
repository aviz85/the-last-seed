"""
Review / verification plots for the SFX stem (I can't listen, so I look).

  spec_png(path, sig, title)          waveform + log-frequency spectrogram of one sound
  grid_png(path, [(title, sig), ...]) audition sheet of many sounds
Full-film plots (overview, cue alignment, levels) live in sfx.py --review.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

from sfx_dsp import SR, db

plt.rcParams.update({'font.size': 7, 'axes.titlesize': 8})


def _mono(sig):
    sig = np.asarray(sig, float)
    return sig.mean(axis=0) if sig.ndim == 2 else sig


def spectro(ax, sig, t0=0.0, fmin=40, fmax=20000, dyn=85, nfft=1024, cmap='magma', vmax=None):
    x = _mono(sig)
    if len(x) < nfft:
        x = np.pad(x, (0, nfft - len(x)))
    f, t, S = signal.spectrogram(x, SR, window='hann', nperseg=nfft, noverlap=nfft * 3 // 4, mode='magnitude')
    S = 20 * np.log10(S + 1e-12)
    top = S.max() if vmax is None else vmax
    m = (f >= fmin) & (f <= fmax)
    ax.pcolormesh(t + t0, f[m], S[m], shading='auto', cmap=cmap, vmin=top - dyn, vmax=top)
    ax.set_yscale('log')
    ax.set_ylim(fmin, fmax)
    ax.set_yticks([50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    ax.set_yticklabels(['50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k'])
    return top


def wave(ax, sig, t0=0.0, color='#2a6f97'):
    sig = np.asarray(sig, float)
    if sig.ndim == 1:
        sig = sig[None]
    n = sig.shape[-1]
    t = t0 + np.arange(n) / SR
    step = max(1, n // 4000)
    for ch, c in zip(sig, [color, '#c0392b']):
        # min/max decimation so transients survive the plot
        k = n // step
        mx = ch[:k * step].reshape(k, step).max(1)
        mn = ch[:k * step].reshape(k, step).min(1)
        tt = t[:k * step:step]
        ax.fill_between(tt, mn, mx, color=c, alpha=0.55, lw=0)
    ax.set_xlim(t[0], t[-1])
    pk = np.max(np.abs(sig)) if sig.size else 0
    ax.set_title(ax.get_title() + f'  peak {db(pk):.1f} dBFS', loc='right')


def spec_png(path, sig, title='', fmax=20000):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5), gridspec_kw={'height_ratios': [1, 2.2]}, sharex=True)
    a1.set_title(title, loc='left')
    wave(a1, sig)
    spectro(a2, sig, fmax=fmax)
    a2.set_xlabel('s')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def grid_png(path, items, cols=3, fmax=20000, dyn=80):
    rows = (len(items) + cols - 1) // cols
    fig = plt.figure(figsize=(4.2 * cols, 3.0 * rows))
    gs = fig.add_gridspec(rows * 3, cols, hspace=0.9, wspace=0.25)
    for k, (title, sig) in enumerate(items):
        r, c = divmod(k, cols)
        a1 = fig.add_subplot(gs[r * 3, c])
        a2 = fig.add_subplot(gs[r * 3 + 1:r * 3 + 3, c], sharex=a1)
        a1.set_title(title, loc='left')
        wave(a1, sig)
        a1.set_xticklabels([])
        spectro(a2, sig, fmax=fmax, dyn=dyn)
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return path
