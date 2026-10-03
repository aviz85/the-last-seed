"""
THE LAST SEED - score analysis: spectrograms, loudness curves, click / DC /
clipping / silence checks.  (I can't listen, so I look.)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import median_filter, maximum_filter1d

from audio.music_dsp import SR, ns, a2db, hp, loudness_curve, lufs_integrated, true_peak_db


def spec_png(x, path, t0=0.0, cues=None, fmax=12000, title='', fmin=30, db_range=90, height=7, width=None,
             loud=True):
    """log-frequency spectrogram (+ optional short-term loudness strip)."""
    mono = x.mean(axis=0) if x.ndim == 2 else x
    dur = len(mono) / SR
    width = width or max(10, min(60, dur * 0.9))
    nper = 4096 if dur > 8 else 2048
    f, tt, S = signal.spectrogram(mono, SR, nperseg=nper, noverlap=nper * 3 // 4, window='hann', mode='magnitude')
    S = 20 * np.log10(S + 1e-10)
    vmax = S.max()
    rows = 2 if loud else 1
    fig, axs = plt.subplots(rows, 1, figsize=(width, height), sharex=True,
                            gridspec_kw={'height_ratios': [3, 1] if loud else [1]})
    ax = axs[0] if loud else axs
    sel = (f >= fmin) & (f <= fmax)
    ax.pcolormesh(tt + t0, f[sel], S[sel], vmin=vmax - db_range, vmax=vmax, cmap='magma', shading='auto')
    ax.set_yscale('log')
    ax.set_ylim(fmin, fmax)
    ax.set_ylabel('Hz')
    ax.set_title(title)
    if cues:
        for name, tc in cues.items():
            if t0 <= tc <= t0 + dur:
                ax.axvline(tc, color='cyan', lw=0.6, alpha=0.7)
                ax.text(tc, fmax * 0.9, name, color='cyan', fontsize=6, rotation=90, va='top')
    if loud:
        a2 = axs[1]
        st = x if x.ndim == 2 else np.vstack([x, x])
        if st.shape[1] > ns(3.2):
            tc, L = loudness_curve(st, 3.0, 0.05)
            a2.plot(tc + t0, L, color='k', lw=1, label='short-term (3 s)')
        tc, L = loudness_curve(st, 0.4, 0.02)
        a2.plot(tc + t0, L, color='tab:orange', lw=0.6, label='momentary (0.4 s)')
        env = maximum_filter1d(np.max(np.abs(st), axis=0), ns(0.02))[::ns(0.01)]
        a2.plot(np.arange(len(env)) * 0.01 + t0, a2db(env), color='tab:blue', lw=0.5, alpha=0.6, label='peak dBFS')
        a2.set_ylim(-60, 0)
        a2.grid(alpha=0.3)
        a2.legend(fontsize=6, loc='lower right')
        a2.set_ylabel('LUFS / dBFS')
        if cues:
            for name, tc in cues.items():
                if t0 <= tc <= t0 + dur:
                    a2.axvline(tc, color='c', lw=0.5, alpha=0.6)
        a2.set_xlabel('s')
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def find_clicks(x, thresh=25.0, min_db=-70.0):
    """flag broadband impulses: HF (>9 kHz) frame peaks far above their
    local median. Returns list of (time, ratio, level_db)."""
    mono = x.mean(axis=0) if x.ndim == 2 else x
    y = np.abs(hp(mono, 9000, order=4))
    fr = ns(0.0005)
    k = len(y) // fr
    pk = y[:k * fr].reshape(k, fr).max(axis=1)
    # isolated spike: compare with the loudest frame 3-12 ms before AND after
    before = np.concatenate([np.zeros(24), maximum_filter1d(pk, 18)[:-24]])[:k]
    after = np.concatenate([maximum_filter1d(pk, 18)[24:], np.zeros(24)])[:k]
    med = np.maximum(np.maximum(before, after), median_filter(pk, size=101, mode='nearest')) + 1e-9
    ratio = pk / med
    lvl = a2db(pk)
    idx = np.nonzero((ratio > thresh) & (lvl > min_db))[0]
    out = []
    last = -1
    for i in idx:
        if last >= 0 and i - last < 20:
            continue
        out.append((i * fr / SR, float(ratio[i]), float(lvl[i])))
        last = i
    return out


SECTIONS = [('I robot', 0.0, 12.0), ('suspense', 12.0, 16.0), ('life', 16.0, 20.0), ('journey', 20.0, 26.0),
            ('storm', 26.0, 32.0), ('night', 32.0, 38.0), ('gift', 38.0, 44.5), ('build', 45.3, 48.0),
            ('CLIMAX', 48.0, 53.0), ('resolve', 53.0, 57.0), ('final', 57.0, 60.0)]


def report(x, stems, cues, prefix='review/music'):
    st = stats(x, (cues['silence'], cues['first_pulse']))
    print('  --- master stats')
    for k, v in st.items():
        print(f'    {k:16s} {v}')
    tc, L = loudness_curve(x, 3.0, 0.05)
    tm, Lm = loudness_curve(x, 0.4, 0.02)
    i = int(np.argmax(L))
    print(f'    max short-term  {L[i]:.1f} LUFS at {tc[i]:.2f}s ; max momentary {Lm.max():.1f} at {tm[np.argmax(Lm)]:.2f}s')
    print('  --- sections (momentary max / mean, short-term max)')
    for name, a, b in SECTIONS:
        sel = (tm >= a) & (tm < b)
        sel2 = (tc >= a) & (tc < b)
        pk = a2db(np.max(np.abs(x[:, ns(a):ns(b)])) + 1e-12)
        print(f'    {name:9s} {a:5.1f}-{b:5.1f}  M max {Lm[sel].max():6.1f}  M mean {np.mean(Lm[sel]):6.1f}  '
              f'S max {L[sel2].max() if sel2.any() else float("nan"):6.1f}  peak {pk:6.1f}')
    cl = find_clicks(x)
    print(f'  --- click candidates: {len(cl)}', [(round(a, 3), round(b, 1), round(c, 1)) for a, b, c in cl[:25]])
    # stem balance per section (K-weighted mean-square level, pre-master dB)
    names = sorted(stems)
    print('  --- stem levels per section (dB, K-weighted RMS, pre-master; blank < -60)')
    print('    ' + 'section   ' + ''.join(f'{g[:7]:>8s}' for g in names))
    from audio.music_dsp import k_weight
    kst = {g: k_weight(stems[g]) for g in names}
    for name, a, b in SECTIONS:
        row = ''
        for g in names:
            seg = kst[g][:, ns(a):ns(b)]
            lv = 10 * np.log10(np.mean(np.sum(seg ** 2, axis=0)) + 1e-20) - 0.691
            row += f'{lv:8.1f}' if lv > -60 else '        '
        print(f'    {name:9s} {row}')
    # overview figure
    fig, axs = plt.subplots(3, 1, figsize=(26, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1.3, 1.6]})
    mono = x.mean(axis=0)
    f, tt, S = signal.spectrogram(mono, SR, nperseg=4096, noverlap=3072, mode='magnitude')
    S = 20 * np.log10(S + 1e-10)
    sel = (f >= 30) & (f <= 14000)
    axs[0].pcolormesh(tt, f[sel], S[sel], vmin=S.max() - 85, vmax=S.max(), cmap='magma', shading='auto')
    axs[0].set_yscale('log')
    axs[0].set_ylim(30, 14000)
    axs[1].plot(tc, L, 'k', lw=1.2, label='short-term 3 s')
    axs[1].plot(tm, Lm, color='tab:orange', lw=.6, label='momentary')
    axs[1].axhline(-16, color='g', lw=.5, ls='--')
    axs[1].set_ylim(-50, -4)
    axs[1].legend(fontsize=7, loc='lower left')
    axs[1].grid(alpha=.3)
    cols = plt.cm.tab20(np.linspace(0, 1, max(len(stems), 1)))
    for (g, s), c in zip(sorted(stems.items()), cols):
        if np.max(np.abs(s)) < 1e-6:
            continue
        tcs, Ls = loudness_curve(s, 0.4, 0.05)
        axs[2].plot(tcs, Ls, lw=.8, color=c, label=g)
    axs[2].set_ylim(-60, -5)
    axs[2].legend(fontsize=7, ncol=8, loc='lower left')
    axs[2].grid(alpha=.3)
    axs[2].set_ylabel('stem (pre-master) LUFS-M')
    for ax in axs:
        for name, tcue in cues.items():
            ax.axvline(tcue, color='c' if ax is not axs[0] else 'cyan', lw=.5, alpha=.6)
    for name, tcue in cues.items():
        axs[0].text(tcue, 13000, name, color='cyan', fontsize=6, rotation=90, va='top')
    axs[2].set_xlim(0, 60)
    fig.tight_layout()
    fig.savefig(prefix + '_overview.png', dpi=60)
    plt.close(fig)
    # detailed 15-s spectrogram pages
    for k in range(4):
        a, b = 15 * k, 15 * (k + 1)
        spec_png(x[:, ns(a):ns(b)], f'{prefix}_spec_{k}.png', t0=a, cues=cues, fmax=14000,
                 title=f'THE LAST SEED score {a}-{b}s', width=22, height=8)
    return st


def stats(x, silence=None):
    d = {}
    d['samples'] = x.shape[1]
    d['peak_dbfs'] = float(a2db(np.max(np.abs(x))))
    d['true_peak_dbtp'] = true_peak_db(x)
    d['lufs_i'] = float(lufs_integrated(x))
    d['dc'] = [float(np.mean(c)) for c in x]
    d['clipped'] = int(np.sum(np.abs(x) >= 0.999))
    if silence:
        a, b = ns(silence[0]), ns(silence[1])
        d['silence_max'] = float(np.max(np.abs(x[:, a:b])))
    return d
