"""
Verification for the SFX stem (called by `python audio/sfx.py --review`).

Writes review/sfx_overview.png   waveform + short-term level per stem + spectrogram, cue lines
       review/sfx_cues.png       zoom on every cue: mix envelope, cue line, logged event onset
       review/sfx_zoom_*.png     spectrogram close-ups of key passages
       review/sfx_ltas.png       long-term spectra of each section (tonal balance / headroom)
       review/sfx_levels.txt     section levels, true peak, silence check, discontinuity scan,
                                 event list with cue alignment errors
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import uniform_filter1d, maximum_filter1d

from sfx_dsp import SR, db, n_of, true_peak_db
from sfx_review import spectro, wave

HERE = os.path.dirname(os.path.abspath(__file__))
REVIEW = os.path.join(os.path.dirname(HERE), 'review')

def sections(T):
    return [('S1 dead world', T['fade_in'], T['cut_s2']), ('S2 empty can', T['cut_s2'], T['cut_s3']),
            ('S3/c01/c02 discovery', T['cut_s3'], T['cut_s4']), ('S4 journey', T['cut_s4'], T['cut_s5']),
            ('S5 STORM', T['cut_s5'], T['cut_s6']), ('S6 night', T['cut_s6'], T['cut_s7']),
            ('S7 dawn / gift', T['cut_s7'], T['silence']), ('SILENCE', T['silence'], T['first_pulse']),
            ('S8a bloom inside', T['first_pulse'], T['cut_s8b_climax']),
            ('S8b CLIMAX', T['cut_s8b_climax'], T['cut_s9a']), ('S9a valley', T['cut_s9a'], T['cut_c04']),
            ('c04 rest / end', T['cut_c04'], T['end'])]


def shots(T):
    return [0.0] + [T[k] for k in ('cut_s2', 'cut_s3', 'cut_c01_eye', 'cut_c02_sprout', 'cut_s3b', 'cut_s4', 'cut_s5',
                                   'cut_s6', 'cut_s7', 'cut_c03_core', 'cut_s7b', 'cut_s8b_climax', 'cut_s9a',
                                   'cut_c04', 'end')]


def st_rms_db(x, win=0.4):
    m = x.mean(0) if x.ndim == 2 else x
    p = uniform_filter1d(np.square(x).mean(0) if x.ndim == 2 else np.square(x), n_of(win))
    return 10 * np.log10(np.maximum(p, 1e-14))


def env_db(x, win=0.003):
    a = np.max(np.abs(x), axis=0) if x.ndim == 2 else np.abs(x)
    return db(maximum_filter1d(a, n_of(win)))


def report(mix, stems, mx, lg, T):
    os.makedirs(REVIEW, exist_ok=True)
    N = mix.shape[1]
    t = np.arange(N) / SR
    lines = []
    P = lines.append

    # ------------------------------------------------------------------ overview
    fig, ax = plt.subplots(3, 1, figsize=(22, 11), sharex=True, gridspec_kw={'height_ratios': [1.1, 1.3, 1.8]})
    wave(ax[0], mix)
    ax[0].set_ylim(-1, 1)
    ax[0].set_title('THE LAST SEED - SFX stem (L blue / R red)', loc='left')
    dec = 480
    tt = t[::dec]
    cols = {'amb': '#6a994e', 'robot': '#bc4749', 'fx': '#386fa4'}
    for k, c in cols.items():
        ax[1].plot(tt, st_rms_db(stems[k])[::dec], color=c, lw=1.0, label=f'{k} (400 ms RMS)')
    ax[1].plot(tt, st_rms_db(mix)[::dec], color='k', lw=1.2, label='mix RMS')
    ax[1].plot(tt, env_db(mix, 0.02)[::dec], color='#999', lw=0.6, label='mix peak')
    ax[1].plot(tt, db(lg[::dec]), color='m', lw=0.8, label='limiter gain')
    ax[1].set_ylim(-75, 2)
    ax[1].axhline(-1.0, color='r', lw=0.6, ls='--')
    ax[1].legend(loc='lower left', ncol=6, fontsize=7)
    ax[1].set_ylabel('dBFS')
    spectro(ax[2], mix, nfft=2048, dyn=90)
    for a in ax:
        for s in shots(T):
            a.axvline(s, color='#444', lw=0.8, alpha=0.5)
    for name, tc in T.items():
        for a in ax:
            a.axvline(tc, color='#ff9f1c', lw=0.6, alpha=0.8, ls=':')
        ax[0].text(tc, 1.0, name, rotation=90, fontsize=6, va='bottom', ha='center', color='#b5651d')
    ax[2].set_xlim(0, N / SR)
    ax[2].set_xticks(np.arange(0, 61, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(REVIEW, 'sfx_overview.png'), dpi=80)
    plt.close(fig)

    # ------------------------------------------------------------------ cue alignment
    cue_events = {}
    for ev in mx.log:
        if ev['cue'] and ev.get('kind') != 'bed':
            cue_events.setdefault(ev['cue'], []).append(ev)
    keys = [k for k in T if k in cue_events]
    cols_ = 6
    rows = (len(keys) + cols_ - 1) // cols_
    fig, axs = plt.subplots(rows, cols_, figsize=(3.2 * cols_, 2.1 * rows))
    axs = axs.ravel()
    e = env_db(mix, 0.004)
    align = []
    for i, k in enumerate(keys):
        a = axs[i]
        tc = T[k]
        i0, i1 = n_of(max(tc - 0.35, 0)), n_of(min(tc + 0.55, N / SR))
        a.plot(t[i0:i1:24], e[i0:i1:24], color='k', lw=0.8)
        a.axvline(tc, color='#ff9f1c', lw=1.5)
        a.axvspan(tc - 2 / 30, tc + 2 / 30, color='#ff9f1c', alpha=0.15)
        for ev in cue_events[k]:
            a.axvline(ev['onset'], color='#386fa4', lw=1.0, ls='--')
            align.append((k, tc, ev['name'], ev['onset'], ev['peak_t'], ev['peak_db'], ev.get('kind', 'hit')))
        a.set_ylim(-80, 0)
        a.set_title(f'{k} {tc:.2f}s', fontsize=7, loc='left')
        a.tick_params(labelsize=6)
    for j in range(len(keys), len(axs)):
        axs[j].axis('off')
    fig.suptitle('Cue alignment: orange = cue (+/-2 frames band), blue dashed = event onset (-24 dB of its peak); '
                 'black = mix peak envelope (dBFS)', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(REVIEW, 'sfx_cues.png'), dpi=80)
    plt.close(fig)

    # ------------------------------------------------------------------ zooms
    zooms = [('s2_drop', T['can_raise'] - 0.2, T['cut_s3'] + 0.2), ('s3_c01_c02', T['cut_s3'], T['cut_s4'] + 0.2),
             ('storm', T['storm_rise'] - 0.5, T['cut_s6'] + 0.6), ('night', T['cut_s6'] - 0.2, T['cut_s7'] + 0.4),
             ('gift_silence', T['cut_c03_core'] - 0.2, T['veins'] + 0.4),
             ('bloom_climax', T['first_pulse'] - 0.3, T['cut_s9a'] + 0.5), ('end', T['cut_s9a'], T['end'])]
    for name, a0, a1 in zooms:
        i0, i1 = n_of(a0), n_of(a1)
        fig, (x1, x2) = plt.subplots(2, 1, figsize=(16, 6), sharex=True, gridspec_kw={'height_ratios': [1, 2.4]})
        wave(x1, mix[:, i0:i1], t0=a0)
        x1.set_ylim(-1, 1)
        spectro(x2, mix[:, i0:i1], t0=a0, nfft=1024, dyn=85)
        for kk, tc in T.items():
            if a0 <= tc <= a1:
                for a in (x1, x2):
                    a.axvline(tc, color='#ff9f1c', lw=0.8, ls=':')
                x1.text(tc, 1.0, kk, rotation=90, fontsize=7, va='bottom', ha='center', color='#b5651d')
        fig.tight_layout()
        fig.savefig(os.path.join(REVIEW, f'sfx_zoom_{name}.png'), dpi=80)
        plt.close(fig)

    # ------------------------------------------------------------------ LTAS per section
    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    for (name, a0, a1) in sections(T):
        if name == 'SILENCE':
            continue
        x = mix[:, n_of(a0):n_of(a1)].mean(0)
        f, pxx = signal.welch(x, SR, nperseg=8192)
        ax.semilogx(f[1:], 10 * np.log10(pxx[1:] + 1e-20), lw=1, label=name)
    ax.set_xlim(20, 20000)
    ax.set_ylim(-150, -40)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title('Long-term average spectrum per section (mix)')
    fig.tight_layout()
    fig.savefig(os.path.join(REVIEW, 'sfx_ltas.png'), dpi=90)
    plt.close(fig)

    # ------------------------------------------------------------------ numbers
    P('THE LAST SEED - SFX stem report')
    P(f'samples {N}  ({N / SR:.3f} s @ {SR} Hz, stereo, float32)')
    P(f'sample peak {db(np.max(np.abs(mix))):.2f} dBFS   true peak (4x) {true_peak_db(mix):.2f} dBTP')
    P(f'limiter: min gain {db(lg.min()):.2f} dB, active (< -0.1 dB) for {np.mean(lg < undb_(-0.1)) * 100:.2f}% '
      f'of samples')
    s0, s1 = n_of(T['silence']), n_of(T['first_pulse'])
    P(f'TOTAL SILENCE {T["silence"]:.2f}-{T["first_pulse"]:.2f}: max |x| = {np.max(np.abs(mix[:, s0:s1])):.3e} '
      f'(must be 0), nonzero samples = {int(np.count_nonzero(mix[:, s0:s1]))}')
    P(f'first sample {mix[:, 0]}, last sample {mix[:, -1]}')
    # spike / click scan: 2nd difference vs its local RMS (10 ms). Designed transients also show up here;
    # each flag is listed with the events sounding at that moment so it can be checked.
    d2 = np.abs(np.diff(mix, 2, axis=1)).max(0)
    loc = np.sqrt(uniform_filter1d(d2 ** 2, n_of(0.01)) + 1e-12)
    flag = np.nonzero((d2 > 12 * loc) & (d2 > 0.02))[0]
    times = []
    for i in flag:
        tt_ = i / SR
        if not times or tt_ - times[-1] > 0.02:
            times.append(tt_)
    P(f'isolated spikes (|d2| > 12x local RMS and > 0.02): {len(times)}')
    for tt_ in times[:60]:
        near = [ev['name'] for ev in mx.log if ev['t'] - 0.005 <= tt_ <= ev['t'] + ev['dur'] and ev.get('kind') != 'bed']
        P(f'   {tt_:8.4f}s  events: {", ".join(near[:4]) if near else "(beds only)"}')
    # DC / stereo
    P(f'DC offset L {np.mean(mix[0]):.2e}  R {np.mean(mix[1]):.2e}')
    P('')
    P(f'{"section":24s} {"mix RMS":>8s} {"mix pk":>7s} | {"amb RMS":>8s} {"robot pk":>8s} {"fx pk":>7s} | '
      f'{"L-R dB":>6s} {"corr":>5s}')
    for (name, a0, a1) in sections(T):
        i0, i1 = n_of(a0), n_of(a1)
        seg = mix[:, i0:i1]

        def r_(x):
            return 10 * np.log10(np.mean(np.square(x)) + 1e-20)
        lr = r_(seg[0]) - r_(seg[1])
        c = np.corrcoef(seg[0], seg[1])[0, 1] if np.std(seg) > 0 else 0.0
        P(f'{name:24s} {r_(seg):8.1f} {db(np.max(np.abs(seg))):7.1f} | {r_(stems["amb"][:, i0:i1]):8.1f} '
          f'{db(np.max(np.abs(stems["robot"][:, i0:i1]))):8.1f} {db(np.max(np.abs(stems["fx"][:, i0:i1]))):7.1f} | '
          f'{lr:6.1f} {c:5.2f}')
    P('')
    P('CUE ALIGNMENT (onset = first sample above -24 dB of the event peak; swells: -40 dB)')
    P(f'{"cue":16s} {"cue t":>6s} {"onset":>7s} {"err ms":>7s} {"peak t":>7s} {"pk dBFS":>7s}  event')
    worst = 0
    for (k, tc, name, on, pt, pk, kind) in align:
        err = (on - tc) * 1000
        worst = max(worst, abs(err))
        flag = '' if abs(err) <= 67 else '  <-- outside +/-2 frames'
        P(f'{k:16s} {tc:6.2f} {on:7.3f} {err:7.1f} {pt:7.3f} {pk:7.1f}  {name}' + (' (swell)' if kind == 'swell'
                                                                                   else '') + flag)
    P(f'worst onset error: {worst:.1f} ms (+/-2 frames = 66.7 ms)')
    P('beds (start = when the bed begins; they fade in by design):')
    for ev in mx.log:
        if ev.get('kind') == 'bed':
            P(f'   {ev["name"]:34s} starts {ev["t"]:6.2f}  audible (-40 dB) {ev["onset"]:6.2f}  cue {ev["cue"]}')
    P('')
    P('FULL EVENT LIST')
    for ev in sorted(mx.log, key=lambda e: e['t']):
        P(f'{ev["t"]:7.3f}  onset {ev["onset"]:7.3f}  {ev["peak_db"]:6.1f} dBFS  {ev["stem"]:5s}  '
          f'{ev["dur"]:5.2f}s  {ev["name"]}' + (f'   [cue {ev["cue"]}]' if ev['cue'] else ''))
    with open(os.path.join(REVIEW, 'sfx_levels.txt'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines[:40]))
    print(f'... full report: {os.path.join(REVIEW, "sfx_levels.txt")}')


def undb_(d):
    return 10 ** (d / 20)
