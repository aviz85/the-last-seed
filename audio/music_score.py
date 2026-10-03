"""
THE LAST SEED - the score.  Every time is derived from cues.CUES so the
director can retime.

Themes
  ROBOT  (music box, D minor, 3/4):  1 3 5 | 7 . 6 | 5 4 3 | 2 . .
         D F A | C . Bb | A G F | E . .      (wind-up arpeggio, then a slow slide down)
  LIFE   (ocarina/flute, F major):   5 | 3' . . 2' 1' | 2' . . 1' | 6 . .
         C | A . . G F | G . . F | D . .     (the reaching rising sixth)
Key plan: D minor -> F major (life) -> D minor (storm, night) -> Bb -> D MAJOR
(the heart, luminous) -> silence -> G major (rebirth; LIFE lifted a whole step,
ROBOT transformed into major as its counter-melody) -> G major ending.
"""
import numpy as np

from cues import CUES as C
from audio.music_dsp import m


class Score:
    def __init__(self):
        self.ev = []

    def n(self, inst, t, p, vel, dur=None, pan=None, **kw):
        self.ev.append(dict(kind='note', inst=inst, t=float(t), midi=m(p), vel=float(vel), dur=dur, pan=pan, kw=kw))

    def line(self, inst, notes, **kw):
        nts = [(float(a), float(b), m(c), float(d)) for a, b, c, d in notes]
        self.ev.append(dict(kind='line', inst=inst, t=min(x[0] for x in nts), notes=nts, kw=kw))

    def fx(self, inst, t, **kw):
        self.ev.append(dict(kind='fx', inst=inst, t=float(t), kw=kw))


def seq(t0, unit, spec, vel=0.6):
    """'C5:1 A5:2:.7 r:1' -> [(t, dur, pitch, vel)] (durations in units)."""
    out, t = [], t0
    for tok in spec.split():
        p = tok.split(':')
        d = float(p[1]) * unit
        if p[0] != 'r':
            out.append((t, d, p[0], float(p[2]) if len(p) > 2 else vel))
        t += d
    return out


def voice(changes, end, vel=0.6):
    """[(t, pitch[, vel]), ...] -> contiguous legato notes ending at `end`
    (repeated pitches are merged = tied)."""
    out = []
    for i, ch in enumerate(changes):
        t, p = ch[0], ch[1]
        v = ch[2] if len(ch) > 2 else vel
        t2 = changes[i + 1][0] if i + 1 < len(changes) else end
        if out and m(out[-1][2]) == m(p):
            a = out[-1]
            out[-1] = (a[0], t2 - a[0], a[2], a[3])
        else:
            out.append((t, t2 - t, p, v))
    return out


# =====================================================================  ACT I
def act1(S):
    """0 - 12: lonely ROBOT theme on the music box over a cold pad and air;
    sparse; the drop; the droop (the music box winding down)."""
    b = C['cut_s3'] / 15.0                       # 0.8 s: five bars of 3/4
    T = lambda bar, beat: ((bar - 1) * 3 + (beat - 1)) * b
    mel = [(1, 1, 'D5', .60, .12), (1, 2, 'F5', .56, .05), (1, 3, 'A5', .62, .02),
           (2, 1, 'C6', .68, 0), (2, 3, 'Bb5', .55, 0),
           (3, 1, 'A5', .60, 0), (3, 2, 'G5', .54, 0), (3, 3, 'F5', .52, 0),
           (4, 1, 'E5', .60, 0)]
    mt = [T(bar, beat) + off for bar, beat, p, v, off in mel]
    quiet = C['drop_fall'] - .3                   # everything has breathed out before the drop falls
    for i, (bar, beat, p, v, off) in enumerate(mel):
        ring = mt[i + 1] - mt[i] + .35 if i + 1 < len(mel) else quiet - mt[i]
        S.n('mbox', mt[i], p, v, pan=-0.12, ring=ring)
    for bar, p in ((1, 'D4'), (2, 'G3'), (3, 'Bb3'), (4, 'A3')):     # low comb teeth
        t = T(bar, 1) + (0.12 if bar == 1 else 0) + 0.006
        S.n('mbox', t, p, .34, pan=-0.22, ring=min(4.5, quiet - t))
    # THE DROP: the music rests from drop_fall to ~drop_hit+.4 so the real 'tk' lands in near
    # silence; only a whisper of a celesta answers it, like its echo
    S.n('celesta', C['drop_hit'] + .12, 'A6', .12, pan=0.2)
    # droop: descending sigh, ritardando, ends on the tonic at the cut
    t0, t1 = C['droop'], C['cut_s3']
    w = np.array([.35, .35, .4, .5])
    w *= (t1 - t0) / w.sum()
    ts = t0 + np.concatenate([[0], np.cumsum(w)])
    for i, (t, p, v) in enumerate(zip(ts, ['A5', 'G5', 'F5', 'E5', 'D5'], [.56, .50, .47, .46, .50])):
        S.n('mbox', t, p, v, pan=-0.12, ring=(ts[i + 1] - t + .3) if i < 4 else None)
    # a singing cello a tenth below (F E D C# D)
    vc = [(ts[i], ts[i + 1] - ts[i], p, v) for i, (p, v) in enumerate(zip(['F4', 'E4', 'D4', 'C#4'], [.36, .33, .33, .35]))]
    vc.append((ts[4], 1.5, 'D4', .30))
    S.line('vc', vc, att=.35, rel=1.0, vib_cents=16, voices=3, dip=.08, spread=.2, pan=.15)
    # cold pad: open 5ths; A major (hope) as the can is raised, then it breathes out for the drop,
    # and returns on Bb (the hope sinks) for the droop
    a7 = ts[3]                                    # A7 arrives with the E5 / C#4
    hb = C['cut_c01_eye']
    g = C['glint']
    a_end = C['drop_fall'] - .25
    b_st = C['drop_hit'] + .4
    lo1 = voice([(0, 'D3'), (T(2, 1), 'G2'), (T(3, 1), 'Bb2'), (T(4, 1), 'A2')], a_end)
    mid1 = voice([(0, 'A3'), (T(2, 1), 'Bb3'), (T(3, 1), 'F3'), (T(4, 1), 'A3')], a_end)
    hi1 = voice([(0, 'D4'), (C['can_raise'], 'C#4')], a_end)
    lo2 = voice([(b_st, 'Bb2'), (a7, 'A2'), (t1, 'D3'), (g, 'C3')], hb + .05)
    mid2 = voice([(b_st, 'F3'), (a7, 'E3'), (t1, 'A3')], hb + .05)
    hi2 = voice([(b_st, 'D4'), (a7, 'C#4'), (t1, 'E4')], hb + .05)
    for k, (l1, l2) in enumerate(((lo1, lo2), (mid1, mid2), (hi1, hi2))):
        S.line('pad_cold', l1, att=1.8 if k else 2.2, rel=.3, seed=k,
               dyn=[(0, .0), (1.5, .85), (C['can_raise'], .85), (a_end, .6)])
        S.line('pad_cold', l2, att=.8, rel=.3, seed=10 + k,
               dyn=[(b_st, .75), (t1, .9), (g, 1.0), (hb - .2, .9)])
    S.fx('air', 0.0, dur=C['drop_fall'] + .1, peak=.22, fin=2.5, fout=.55)
    S.fx('air', b_st, dur=t1 + 1.2 - b_st, peak=.22, fin=.9, fout=1.2)


# =====================================================================  SUSPENSE
def suspense(S):
    t0, g, hb, L = C['cut_s3'], C['glint'], C['cut_c01_eye'], C['cut_c02_sprout']
    for k, p in enumerate(('A5', 'E6')):          # glassy harmonics, sul pont tremolo
        S.line('vln_harm', [(t0 + .15, hb - t0 - .15, p, .5)], att=.9, rel=.16, trem=.55, trem_rate=11.0,
               voices=4, seed=k, dyn=[(t0, .25), (g - .1, .8), (g + .3, .5), (hb - .15, .45)])
    S.line('cb', [(t0, hb - t0, 'D2', .3)], att=1.2, rel=.22, vib_cents=3, voices=3)
    for dt, p, v in ((-1.05, 'E7', .13), (-.62, 'A6', .15), (-.36, 'D7', .17), (-.17, 'E7', .19)):
        S.n('celesta', g + dt, p, v, pan=.35)
    S.fx('rev_cym', g - 1.1, dur=1.1, peak=.2)
    for i, p in enumerate(('A6', 'C7', 'E7', 'A7')):   # THE GLINT: first bright colour
        S.n('celesta', g + .035 * i, p, .42 - .05 * i, pan=.3)
    S.n('glock', g, 'E7', .30, pan=.35)
    S.n('glock', g + .16, 'A7', .18, pan=.42)
    # held breath: one high C (V of F) over a whisper of low C
    S.line('vln_harm', [(hb, L - hb + .06, 'C6', .5)], att=.35, rel=.35, voices=4, seed=5,
           dyn=[(hb, .35), (L - .12, .55), (L + .1, .3)])
    S.line('pad_warm', [(hb, L - hb, 'C3', .3)], att=.5, rel=.4, seed=3)
    # breath in: reversed harp chord and a faint reversed cymbal into the LIFE theme
    for p in ('F3', 'C4', 'A4', 'F5'):
        S.fx('rev_note', L, src='harp', pitch=p, vel=.42, dur=.95)
    S.fx('rev_cym', L - .8, dur=.8, peak=.1)


# =====================================================================  LIFE
def life(S):
    t0, j = C['cut_c02_sprout'], C['cut_s4']
    u = (j - t0) / 8                               # 0.5 s beats
    S.line('ocarina', seq(t0, u, 'C5:1:.68 A5:2:.74 G5:.5:.6 F5:.5:.58 G5:1.5:.66 F5:.5:.58 D5:2.5:.56'),
           dip=.22, rel=.55, dyn=[(t0, .84)])
    arps = ['F2 C3 F3 A3', 'C4 A3 F3 C3', 'E2 C3 E3 G3', 'D2 A2 D3 F3']
    chg = [t0 + 4 * u, t0 + 4 * u, t0 + 6 * u, j]         # next chord change for each group
    for i, ch in enumerate(arps):
        for k, p in enumerate(ch.split()):
            t = t0 + i * 2 * u + k * u / 2
            S.n('harp', t, p, .44 if k == 0 else .34, pan=-.3 + .08 * k, ring=chg[i] - t + .12)
    S.n('celesta', C['leaf_touch'], 'F6', .2, pan=.3)
    # first warmth: con sordino strings (continue through the journey)
    s5 = C['cut_s5']
    sr = C['storm_rise']
    x = [j + k * 2 * u for k in range(6)]           # 20 21 22 23 24 25
    S.line('vc', voice([(t0, 'F2'), (t0 + 4 * u, 'E2'), (t0 + 6 * u, 'D2')], j - .04), att=1.2, rel=.3, voices=4,
           vib_cents=10, dyn=[(t0, .23), (j, .29)])
    S.line('vc', voice([(x[0], 'Bb2'), (x[1], 'A2'), (x[2], 'G2'), (x[3], 'C3'), (x[4], 'D3')], s5),
           att=.3, rel=.15, voices=4, vib_cents=10, dyn=[(j, .33), (sr, .38), (s5, .66)])
    S.line('vla', voice([(t0, 'A3'), (t0 + 4 * u, 'G3'), (t0 + 6 * u, 'F3'), (x[3], 'E3'), (x[4], 'F3'), (x[5], 'Eb3')], s5),
           att=1.2, rel=.15, voices=4, vib_cents=10, dyn=[(t0, .3), (j, .33), (sr, .38), (s5, .66)])
    S.line('vln2', voice([(t0, 'C4'), (t0 + 6 * u, 'A3'), (x[0], 'D4'), (x[1], 'C4'), (x[2], 'Bb3'), (x[4], 'A3'),
                          (x[5], 'G3')], s5), att=1.2, rel=.15, voices=4, vib_cents=10,
           dyn=[(t0, .28), (j, .32), (sr, .38), (s5, .66)])


# =====================================================================  JOURNEY
def journey(S):
    j, s5 = C['cut_s4'], C['cut_s5']
    u = (s5 - j) / 12                               # 0.5 s beats, 3 bars of 4/4
    stop = C['robot_looks_up']
    chords = ['Bb3 D4 F4 A4', 'F3 A3 C4 E4', 'G3 Bb3 D4 F4', 'C4 E4 G4 Bb4', 'D3 F3 A3 C4', 'D3 F3 A3 C4']
    for i, ch in enumerate(chords):                 # the ROBOT arpeggio becomes the walking ostinato
        for k, p in enumerate(ch.split()):
            t = j + i * 2 * u + k * u / 2
            if t >= stop - 1e-6:
                break
            v = (.42 if k == 0 else .32) * (.85 if i >= 4 else 1.0)
            S.n('harp', t, p, v, pan=-.35 + .1 * k, ring=j + (i + 1) * 2 * u - t + .12)
    # the ROBOT theme itself on its music box as the sky darkens: D F A C . Bb (cut short)
    sr0 = C['storm_rise']
    for t, p, v, rg in ((sr0, 'D5', .46, .5), (sr0 + u / 2, 'F5', .44, .5), (sr0 + u, 'A5', .48, .5),
                        (sr0 + 1.5 * u, 'C6', .52, 1.0), (sr0 + 2.5 * u, 'Bb5', .44, 1.3)):
        S.n('mbox', t, p, v, pan=-.15, ring=rg)
    bass = ['Bb2', 'F2', 'A2', 'C3', 'G2', 'D3', 'C3', 'G2', 'D2', 'A2']
    for k, p in enumerate(bass):                    # pizzicato steps
        S.n('pizz', j + k * u, p, .62 if k % 2 == 0 else .5, pan=.25)
    # LIFE theme, 2nd phrase, reaching up to Bb at look_down_can, then sinking
    S.line('flute', seq(j + u, u, 'C5:1:.56 A5:2:.68 G5:.5:.6 F5:.5:.58 Bb5:2:.74 A5:1:.62 G5:1:.56 F5:2.5:.48'),
           dip=.3, rel=.9, dyn=[(j, .82), (C['storm_rise'], .82), (s5 - .2, .2)])
    # darkening: bass drone, rumble, timpani roll, Phrygian horn warning (D - Eb)
    sr = C['storm_rise']
    S.line('cb', [(sr, s5 - sr, 'D2', .5)], att=1.2, rel=.12, voices=4, dyn=[(sr, .3), (s5, 1.0)])
    S.fx('rumble', sr, dur=s5 - sr + .3, peak=.7)
    S.fx('timp_roll', stop - .1, pitch='D2', dur=s5 - stop + .1, v0=.08, v1=.75)
    S.fx('cym_swell', s5 - 1.3, dur=1.3, peak=.45)
    S.line('horn', [(stop, u, 'D3', .5), (stop + u, u, 'Eb3', .62)], att=.15, rel=.1)


# =====================================================================  STORM
def storm(S):
    s, s6 = C['cut_s5'], C['cut_s6']
    q = (s6 - s) / 48.0                             # 16th notes at 120 BPM
    sh, rip, fl, tr, out = C['shield'], C['antenna_rip'], C['core_flicker'], C['trudge'], C['storm_out']
    # --- impact
    S.fx('taiko', s, vel=1.0, f0=48)
    S.fx('timp', s, pitch='D2', vel=1.0)
    S.fx('cymbal', s, vel=.85)
    S.fx('sub', s, vel=.9, f=36.7)
    # --- driving taiko (bar pattern X..x..x.X.xx..x.)
    pat = [0, 3, 6, 8, 10, 11, 14]
    for bt in (s, s + 16 * q):
        for pos in pat:
            t = bt + pos * q
            if t >= rip - 1e-6:
                break
            if abs(t - s) < 1e-6:
                continue
            S.fx('taiko', t, vel=1.0 if pos in (0, 8) else .74, f0=50 + 2 * (pos % 3))
    for k, f in enumerate((150, 132, 116, 100)):    # tom fill into the shield
        S.fx('tom', sh - .5 + k * q, vel=.5 + .08 * k, f0=f)
    # 16th-note low-tom engine under the whole storm (accents on the beats)
    for a_, b_ in ((s + q, sh - 4 * q), (sh, rip - q), (tr + q, out)):
        t = a_
        while t < b_ - 1e-6:
            pos = int(round((t - s) / q)) % 4
            S.fx('tom', t, vel=(.46 if pos == 0 else .34 if pos == 2 else .26), f0=82 + (8 if pos == 0 else 0))
            t += q
    # --- strings: dissonant tremolo (Dm + Eb rub), then Bb/D when it shields
    tr_kw = dict(trem=.8, trem_rate=14.0, att=.08, rel=.12, vib_cents=6, voices=6)
    S.line('vln1', voice([(s, 'A5'), (sh, 'Bb5')], rip), **tr_kw, dyn=[(s, .6), (sh, .52), (rip, .64)])
    pat16 = ['D5', 'Eb5', 'D5', 'D5', 'D5', 'Eb5', 'D5', 'A4']
    spic = [(s + k * q, q * .55, pat16[k % 8], (.66 if k % 4 == 0 else .5)) for k in range(int(round((sh - s) / q)))]
    S.line('vln2', spic, att=.004, rel=.05, voices=6, dip=0.0, vib_cents=0.0, gap=.01)
    S.line('vla', [(s, rip - s, 'F4', .46)], **tr_kw)
    S.line('vc', [(s, rip - s, 'D3', .46)], **tr_kw)
    S.line('cb', [(s, rip - s, 'D2', .6)], att=.05, rel=.12, voices=4)
    # --- the robot's motif (D F A C = Bbmaj9 over the bass) - determination - CUT by the rip
    S.line('horn', seq(sh, 2 * q, 'D4:1:.7 F4:1:.73 A4:1:.77 C5:1.1:.82'), att=.04, rel=.08, dip=.3)
    S.line('vln2', seq(sh, 2 * q, 'D5:1:.62 F5:1:.66 A5:1:.7 C6:1.1:.76'), att=.03, rel=.08, dip=.35, voices=6)
    # --- STAB 1: antenna rip  (Eb/D - the Neapolitan over the tonic)
    stab = dict(att=.012, rel=.4, vib_cents=4, voices=6, dip=0.0)
    for inst, p in (('vln1', 'Eb5'), ('vln1', 'Bb5'), ('vln2', 'G4'), ('vla', 'Eb4'), ('vc', 'D3'), ('cb', 'D2')):
        S.line(inst, [(rip, .26, p, .95)], **stab)
    S.line('horn', [(rip, .3, 'Eb4', .9)], att=.015, rel=.3)
    S.line('horn', [(rip, .3, 'Bb3', .85)], att=.015, rel=.3, seed=2)
    S.fx('taiko', rip, vel=.85, f0=46)
    S.fx('timp', rip, pitch='D2', vel=.9)
    S.fx('cymbal', rip, vel=.9, seed=2)
    S.fx('sub', rip, vel=.6, f=36.7)
    # the antenna tumbles away: violins slide down; low tremolo creeps back
    S.line('vln1', [(rip + .07, .12, 'Bb5', .55), (rip + .19, .75, 'D5', .42)], glide=.45, att=.02, rel=.3,
           voices=5, dip=0.0)
    S.line('vc', [(rip + .3, fl - rip - .3, 'D3', .45)], trem=.8, trem_rate=15.0, att=.25, rel=.06, voices=5,
           dyn=[(rip + .3, .25), (fl, .85)])
    S.line('cb', [(rip + .3, fl - rip - .3, 'D2', .45)], att=.25, rel=.06, voices=4, dyn=[(rip + .3, .25), (fl, .85)])
    S.fx('timp_roll', rip + .55, pitch='A1', dur=fl - rip - .55, v0=.1, v1=.55)
    # --- STAB 2: core flicker (diminished 7th, stuttering x3 like the core)
    for k, (dt, v) in enumerate(((0.0, .9), (.12, .55), (.24, .38))):
        for inst, p in (('vln1', 'F5'), ('vln1', 'D6'), ('vln2', 'B4'), ('vla', 'Ab3'), ('vc', 'F3'), ('cb', 'D2')):
            S.line(inst, [(fl + dt, .075, p, v)], att=.006, rel=.12, voices=4, vib_cents=0, dip=0.0, seed=k)
    S.fx('taiko', fl, vel=.9, f0=50)
    S.fx('cymbal', fl, vel=.5, seed=3)
    S.fx('timp', fl, pitch='D2', vel=.75)
    # --- trudge: heavy half-time steps, marcato low strings, the motif rises
    for t, v in ((tr, 1.0), (tr + 4 * q, .6), (tr + 8 * q, .95), (tr + 12 * q, .7), (tr + 14 * q, .75)):
        S.fx('taiko', t, vel=v, f0=49)
    n8 = int(round((out - tr) / (2 * q))) + 1
    S.line('vc', [(tr + k * 2 * q, 2 * q * .8, 'D3', .72 if k % 2 == 0 else .56) for k in range(n8)],
           att=.02, rel=.09, dip=.5, voices=5, gap=.01)
    S.line('cb', [(tr + k * 2 * q, 2 * q * .8, 'D2', .7 if k % 2 == 0 else .55) for k in range(n8)],
           att=.02, rel=.09, dip=.5, voices=4, gap=.01)
    S.line('horn', seq(tr, 4 * q, 'D4:1:.72 F4:1:.76 A4:1:.8 C5:1.2:.86'), att=.05, rel=.1, dip=.3)
    S.line('vln2', seq(tr, 4 * q, 'D5:1:.64 F5:1:.68 A5:1:.72 C6:1.2:.8'), att=.04, rel=.1, dip=.35)
    S.line('vln1', voice([(tr, 'A5'), (tr + 4 * q, 'Bb5'), (tr + 8 * q, 'C6'), (tr + 12 * q, 'C#6')], s6),
           **tr_kw, dyn=[(tr, .6), (out, 1.0)])
    S.line('vla', voice([(tr, 'F4'), (tr + 8 * q, 'F4'), (tr + 12 * q, 'E4')], s6), **tr_kw, dyn=[(tr, .6), (out, 1.0)])
    S.fx('timp_roll', tr + 8 * q, pitch='D2', dur=s6 - tr - 8 * q, v0=.2, v1=.95)
    S.fx('cym_swell', out + .05 - 1.4, dur=1.4, peak=.6)


# =====================================================================  NIGHT
def night(S):
    n0, ln, nt, lc = C['cut_s6'], C['sprout_lean'], C['robot_notice'], C['look_chest']

    def P(t, p, v, d, pan=None):
        S.n('piano', t, p, v, dur=d, pan=pan)
    P(n0 + .35, 'D2', .30, ln - n0 - .35)          # alone in the dark: low open fifth
    P(n0 + .47, 'A2', .25, ln - n0 - .47)
    sd = C['sit_down']                             # F5 placed just after the sit-down clank
    for t, p, v in ((n0 + .60, 'D5', .33), (sd + .09, 'F5', .30), (sd + .62, 'A5', .32), (sd + 1.08, 'C6', .28)):
        P(t, p, v, ln - t)                          # ROBOT motif, halting
    P(ln - .03, 'F2', .24, nt - ln + .05)           # LIFE answers: the reaching sixth (F major)
    P(ln + .06, 'C3', .20, nt - ln)
    P(ln, 'C5', .27, nt - ln)
    P(ln + .42, 'A5', .31, nt - ln - .3)
    P(nt, 'G2', .26, lc - nt + .1)                  # the question: Gm6/9, rising, unresolved
    P(nt + .05, 'A4', .25, lc - nt)
    P(nt + .30, 'D5', .26, lc - nt - .2)
    P(nt + .60, 'E5', .30, lc - nt + .1)
    x1 = C['cut_s7'] + (C['core_out'] - C['cut_s7']) / 4
    P(lc, 'Bb0', .34, x1 - lc - .06)                # realization: the deep low note
    P(lc + .02, 'Bb1', .30, x1 - lc - .08)
    P(lc + .62, 'D5', .24, 1.6)                     # ...the answer (E -> D)
    P(lc + .66, 'F4', .16, 1.6)


# =====================================================================  THE GIFT
def gift(S):
    g0, co, pd, eo, lc = C['cut_s7'], C['core_out'], C['power_down'], C['eye_off'], C['look_chest']
    sb, ch8, cs = C['cut_s7b'], C['cut_c08_heart'], C['core_in_soil']
    st = (co - g0) / 4.0                            # 0.8 s: 38.0 38.8 39.6 40.4 41.2
    h = st / 2
    X = lambda k: g0 + k * st
    # the strings breathe IN just before core_out, the D major ARRIVES on it and blooms up to the
    # wide gold shot (cut_s7b), then thins as the heart is pressed into the soil
    dyn = [(g0 - .2, .3), (X(3), .72), (X(3) + h, .64), (co - .07, .56), (co + .03, .95), (sb, 1.08),
           (ch8, .85), (cs + .2, .55), (pd - .1, .38)]
    kw = dict(att=.55, rel=.5, vib_cents=12, voices=6, dip=0.0, dyn=dyn)
    # S: 9-8 / 4-3 / 9-10 suspensions -> the LIFE interval (rising 6th) at hatch_open ->
    #    G -> Gb (=F#) pivot: Ebm -> D major, luminous, at core_out
    S.line('vln1', voice([(X(0), 'C5'), (X(0) + h, 'Bb4'), (X(1) + h, 'A4'), (X(2) + h, 'Bb4')], X(3) - .04), **kw)
    kw2 = dict(kw, att=.09)                       # fresh bow on the leap = the hatch opening
    S.line('vln1', voice([(X(3), 'G5'), (X(3) + h, 'Gb5'), (sb, 'A5')], pd - .1), **kw2)
    S.line('vln2', voice([(X(0), 'F4'), (X(3), 'Bb4'), (co, 'A4')], cs + .25), **kw)
    S.line('vla', voice([(X(0), 'D4'), (X(1), 'C4'), (X(2), 'D4'), (X(3), 'Eb4'), (co, 'D4')], cs + .1), **kw)
    S.line('vc', voice([(X(0), 'Bb2'), (X(1), 'F2'), (X(2), 'G2'), (X(3), 'Eb2'), (co, 'D2')], cs - .05), **kw)
    S.line('cb', voice([(lc, 'Bb1'), (X(1), 'F1'), (X(2), 'G1'), (X(3), 'Eb1'), (co, 'D1')], cs - .05),
           att=1.0, rel=.5, vib_cents=6, voices=4, dyn=[(lc, .25), (g0, .35)] + dyn[1:])
    # ---- THE HEART COMES OUT: a burst of warm light that LANDS on core_out
    scale = ['D', 'E', 'F#', 'G', 'A', 'B', 'C#']
    gl, octv = [], 3
    for i in range(22):                           # D3 .. D6, the top note exactly on core_out
        nm = scale[i % 7]
        if nm == 'D' and i > 0:
            octv += 1
        gl.append(f'{nm}{octv}')
    for i, p in enumerate(gl):
        t = co - .36 + .36 * i / (len(gl) - 1)
        chord_tone = p[:-1] in ('D', 'F#', 'A')      # passing tones are damped as the chord lands
        S.n('harp', t, p, .24 + .3 * i / (len(gl) - 1), pan=-.55 + 1.1 * i / len(gl),
            ring=1.1 if chord_tone else co - t + .1)
    S.fx('rev_cym', co - .9, dur=.9, peak=.26)
    for i, p in enumerate(('D2', 'A2', 'D3')):   # warm low 'bell' of the arrival
        S.n('harp', co, p, .56, pan=-.35 + .1 * i)
    for p, v in (('D6', .42), ('F#6', .36), ('A6', .34)):
        S.n('celesta', co, p, v, pan=.3)
    S.n('glock', co, 'D7', .22, pan=.4, ring=1.8)
    S.fx('cymbal', co, vel=.3, seed=41, soft=True)
    for i in range(14):                           # celesta shimmer: the light sparkling
        S.n('celesta', co + .07 + i * .065, ('A6', 'D7', 'F#7')[i % 3], .26 * (1 - i / 16),
            pan=.15 + .3 * ((i % 3) - 1), ring=.5)
    for k, p in enumerate(('D6', 'F#6')):         # string harmonics bloom FROM core_out
        S.line('vln_harm', [(co, sb - co + .5, p, .5)], att=.07, rel=.8, voices=4, seed=10 + k,
               dyn=[(co, .6), (co + .6, .85), (sb + .4, .5)])
    for inst, p in (('choir_s', 'F#5'), ('choir_a', 'D5'), ('choir_t', 'A4')):   # a warm 'oh' of light
        S.line(inst, [(co, ch8 - co, p, .5)], att=.2, rel=.6, vowel='o', dyn=[(co, .26), (sb, .4), (ch8, .3)])
    # ...thinning to a single held note that dies with the eye
    S.line('vln_harm', [(sb + .2, eo - sb - .2, 'A5', .5)], att=.6, rel=.12, voices=2, seed=21, spread=.1,
           dyn=[(sb + .2, .0), (ch8, .55), (pd, .6), (pd + .45, .35), (eo - .02, .0)],
           bend=[(pd + .3, 0.0), (eo, -14.0)])


# =====================================================================  REBIRTH
def rebirth(S):
    fp, vn, gb, cx = C['first_pulse'], C['veins'], C['growth_burst'], C['cut_s8b_climax']
    S.fx('heartbeat', fp, vel=1.0, f=49.0)
    S.fx('heartbeat', fp + .92, vel=1.0, f=49.0, seed=1)
    # rising arpeggios (the ROBOT's 1-3-5-7 arpeggio spreading as veins of light)
    mid1 = (vn + gb) / 2
    segs = [(vn, mid1, 'E3 G3 B3 D4 E4 G4 B4 D5'), (mid1, gb, 'C4 E4 G4 B4 C5 E5 G5 B5'),
            (gb, gb + .5, 'A3 C4 E4 G4 A4 C5 E5 G5 A5 C6'), (gb + .5, cx - .45, 'D4 G4 A4 C5 D5 G5 A5 C6 D6')]
    for a, bnd, notes in segs:
        ps = notes.split()
        for i, p in enumerate(ps):
            t = a + (bnd - a) * i / len(ps)
            v = (.30 + .2 * (t - vn) / (cx - vn)) * .85
            S.n('harp', t, p, v, pan=-.5 + i / len(ps), ring=bnd - t + .12)
            S.n('celesta', t + .01, p, v * .55, pan=.4, octave=1, ring=bnd - t + .1)
    # harp glissando into the climax
    scale = ['G', 'A', 'B', 'C', 'D', 'E', 'F#']
    gl = []
    octv = 3
    for i in range(22):
        nm = scale[i % 7]
        if nm == 'C' and i > 0:
            octv += 1
        gl.append(f'{nm}{octv}')
    g_t0 = cx - .42
    for i, p in enumerate(gl):
        tg = g_t0 + .42 * i / len(gl)
        S.n('harp', tg, p, .35 + .3 * i / len(gl), pan=-.5 + i / len(gl), ring=cx - tg + .15)
    # choir enters pp and grows (Em7 -> Cmaj7 -> Am7 -> Dsus4 -> D)
    ce = cx - .1                                  # a breath before the hit
    cd = [(vn, .12), (gb, .38), (ce - .05, .72)]
    ckw = dict(att=.45, rel=.08, dyn=cd)
    S.line('choir_s', voice([(vn, 'B4'), (gb, 'C5'), (gb + .5, 'D5')], ce), **ckw)
    S.line('choir_a', voice([(vn, 'G4'), (gb + .5, 'A4'), (gb + .85, 'F#4')], ce), **ckw)
    S.line('choir_t', voice([(vn, 'E4'), (gb + .5, 'D4')], ce), **ckw)
    S.line('choir_b', voice([(vn, 'E3'), (mid1, 'C3'), (gb, 'A2'), (gb + .5, 'D3')], ce), **ckw)
    # strings: tremolo crescendo on the dominant, violins climbing
    tkw = dict(trem=.7, trem_rate=14.0, att=.3, rel=.06, voices=6)
    sd = [(gb, .25), (ce, .75)]
    S.line('cb', [(gb, ce - gb, 'D2', .7)], att=.3, rel=.06, voices=4, dyn=sd)
    S.line('vc', voice([(gb, 'A2'), (gb + .5, 'D3')], ce), **tkw, dyn=sd)
    S.line('vla', voice([(gb, 'E3'), (gb + .5, 'A3')], ce), **tkw, dyn=sd)
    S.line('vln2', voice([(gb, 'C4'), (gb + .5, 'D4'), (gb + .85, 'F#4')], ce), **tkw, dyn=sd)
    S.line('vln1', voice([(gb, 'A4'), (gb + .3, 'C5'), (gb + .6, 'D5'), (gb + .85, 'F#5'), (gb + 1.0, 'A5')], ce),
           **tkw, dyn=sd)
    S.line('horn', voice([(gb, 'D3'), (gb + .5, 'A3')], ce), att=.4, rel=.06, dyn=sd)
    S.fx('timp_roll', gb, pitch='D2', dur=cx - gb - .03, v0=.15, v1=.8)
    S.fx('cym_swell', cx - 1.2, dur=1.2, peak=.6)
    S.fx('rev_cym', cx - 1.6, dur=1.6, peak=.55)
    for t, v in ((gb + .5, .55), (gb + .75, .62), (gb + 1.0, .7), (gb + 1.1, .76)):
        S.fx('taiko', t, vel=v, f0=52)


# =====================================================================  CLIMAX
def climax(S):
    cx, r = C['cut_s8b_climax'], C['cut_s9a']
    k = (r - cx) / 8.0                              # 0.625 s beats (96 BPM)
    K = lambda b: cx + b * k
    S.fx('taiko', cx, vel=.85, f0=45)
    S.fx('timp', cx, pitch='G2', vel=1.0)
    S.fx('cymbal', cx, vel=1.0, seed=7)
    S.fx('sub', cx, vel=.72, f=49.0)
    for i, p in enumerate(('G3', 'B3', 'D4')):     # sforzando brass chord on the downbeat
        S.line('horn', [(cx, k * .9, p, .62)], att=.03, rel=.35, seed=30 + i)
    # LIFE theme, triumphant, G major:  D | B . . A G | A . . G | E . Eb | D
    mel = 'D5:1:.9 B5:2:1 A5:.5:.88 G5:.5:.9 A5:1.5:.97 G5:.5:.86 E5:1:.88 Eb5:1:.78'
    body = seq(cx, k, mel)
    S.line('choir_s', body + [(r, 1.5, 'D5', .6)], att=.05, rel=.9, dip=.15,
           dyn=[(cx, 1.0), (r, .8), (r + 1.4, .25)])
    S.line('vln2', body + [(r, .6, 'D5', .55)], att=.04, rel=.5, voices=6, dip=.2, vib_cents=16)
    up = [(t, d, p[:-1] + str(int(p[-1]) + 1), v) for t, d, p, v in body]
    S.line('vln1', up + [(r, .6, 'D6', .5)], att=.04, rel=.5, voices=6, dip=.2, vib_cents=16)
    S.line('flute', up, att=.04, rel=.4, dip=.3, bright=1.1, dyn=[(cx, .42), (r, .42)])
    # ROBOT theme transformed into major as the counter-melody: G B D | F# . E | D C B | A . G
    cm = seq(K(1), k / 2, 'G3:1 B3:1 D4:1 F#4:2 E4:1 D4:1 C4:1 B3:1 A3:2 G3:3', vel=.85)
    S.line('horn', cm, att=.04, rel=.4, dip=.3)
    S.line('vc', cm, att=.04, rel=.4, dip=.25, voices=6, vib_cents=14)
    # harmony  G | D/F# | Em | D | C | Cm | G
    ch = [K(0), K(3), K(3.5), K(4), K(5.5), K(7), r]
    hd = [(cx, 1.0), (r, .75), (r + 1.3, .2)]
    S.line('choir_a', voice(list(zip(ch, ['B4', 'A4', 'B4', 'A4', 'C5', 'C5', 'B4'])), r + 1.4), att=.05, rel=.8, dyn=hd)
    S.line('choir_t', voice(list(zip(ch, ['D4', 'D4', 'E4', 'D4', 'E4', 'Eb4', 'D4'])), r + 1.4), att=.05, rel=.8, dyn=hd)
    S.line('choir_b', voice(list(zip(ch, ['G3', 'F#3', 'E3', 'D3', 'C3', 'C3', 'G2'])), r + 1.4), att=.05, rel=.8, dyn=hd)
    S.line('vla', voice(list(zip(ch, ['D4', 'D4', 'E4', 'D4', 'E4', 'Eb4', 'D4'])), r + .4), att=.04, rel=.6, voices=6)
    S.line('cb', voice(list(zip(ch, ['G1', 'F#1', 'E1', 'D2', 'C2', 'C2', 'G1'])), r + .6), att=.04, rel=.7, voices=5)
    S.line('vc', voice(list(zip(ch, ['G2', 'F#2', 'E2', 'D2', 'C2', 'C2', 'G2'])), r + .6), att=.04, rel=.7,
           voices=4, seed=3)
    # harp: rising 16th-triplet arpeggios through each chord
    chord_notes = {0: 'G3 B3 D4 B4 D5 G5', 3: 'F#3 A3 D4 F#4 A4 D5', 3.5: 'E3 G3 B3 E4 G4 B4',
                   4: 'D3 F#3 A3 D4 F#4 A4', 5.5: 'C3 E3 G3 C4 E4 G4', 7: 'C3 Eb3 G3 C4 Eb4 G4'}
    bounds = [0, 3, 3.5, 4, 5.5, 7, 8]
    for a, b2 in zip(bounds[:-1], bounds[1:]):
        ps = chord_notes[a].split()
        n_ = int(round((b2 - a) * 6))
        for i in range(n_):
            t = K(a) + i * k / 6
            v = .42 if a < 7 else .42 - .2 * (t - K(7)) / (r - K(7))
            S.n('harp', t, ps[i % len(ps)], v, pan=-.45 + .15 * (i % len(ps)), ring=K(b2) - t + .1)
    # sparkles on the wave crests
    S.n('glock', K(1), 'B6', .3, pan=.4)
    S.n('glock', K(1) + .02, 'D7', .22, pan=.45)
    for i, p in enumerate(('G6', 'B6', 'D7', 'G7')):
        S.n('celesta', K(3.5) + .04 * i, p, .4 - .05 * i, pan=.35, ring=K(4) - K(3.5) + .08)
    S.n('glock', K(6), 'E7', .24, pan=.4, ring=K(7) - K(6) + .05)
    # percussion waves
    S.fx('timp', K(1), pitch='D2', vel=.45)
    S.fx('cym_swell', K(3.5) - 1.0, dur=1.0, peak=.5)
    S.fx('cymbal', K(3.5), vel=.7, seed=8)
    S.fx('timp', K(3.5), pitch='E2', vel=.8)
    S.fx('taiko', K(3.5), vel=.62, f0=50)
    S.fx('timp', K(5.5), pitch='C2', vel=.7)
    S.fx('cym_swell', K(6) - .9, dur=.9, peak=.38)
    S.fx('timp_roll', K(7), pitch='C2', dur=r - K(7), v0=.5, v1=.12, curve=1.0)
    S.fx('timp', r, pitch='G1', vel=.35)


# =====================================================================  RESOLUTION
def resolution(S):
    r, eg, ti, end, bl = C['cut_s9a'], C['eye_glow'], C['title'], C['end'], C['bird_land']
    ft = ti + .4                                   # the final tonic lands just after the title appears
    fend = ft + .45                                # held briefly, then a natural morendo (silent by ~59.8)
    # ROBOT theme in G major, complete at last - its opening question is answered:
    #   G B D | F# . E | D C B | A . | G   (gentle ritardando into the cadence; E after the bird lands,
    #   D exactly as the eye lights up)
    mb = [(r + .30, 'G5', .5), (r + .72, 'B5', .48), (r + 1.14, 'D6', .52), (r + 1.56, 'F#6', .55),
          (eg - .35, 'E6', .46), (eg, 'D6', .52), (eg + .52, 'C6', .46), (eg + 1.06, 'B5', .46),
          (ft - .58, 'A5', .48)]
    for i, (t, p, v) in enumerate(mb):
        nxt = mb[i + 1][0] if i + 1 < len(mb) else ft
        S.n('mbox', t, p, v, pan=-.1, ring=nxt - t + .25)
    S.n('mbox', ft, 'G5', .5, pan=-.1, decay=.52)            # the last note rings out on its own
    S.n('mbox', ft + .006, 'G4', .3, pan=-.2, decay=.38)
    # warm strings:  G | D/F# | Em7 | C(add9) (eye glow) | Am7 | G/B | D | G
    x = [r, r + 1.56, bl - .3, eg, eg + .52, eg + 1.06, ft - .58, ft]
    kw = dict(att=.8, rel=1.0, vib_cents=9, voices=5, dyn=[(r, .27), (eg, .32), (ft, .3), (fend, .24)])
    S.line('vln1', voice(list(zip(x, ['B4', 'A4', 'B4', 'C5', 'C5', 'B4', 'A4', 'B4'])), fend), **kw)
    S.line('vln2', voice(list(zip(x, ['G4', 'F#4', 'G4', 'G4', 'G4', 'G4', 'F#4', 'G4'])), fend), **kw)
    S.line('vla', voice(list(zip(x, ['D4', 'D4', 'D4', 'E4', 'E4', 'D4', 'D4', 'D4'])), fend), **kw)
    S.line('vc', voice(list(zip(x, ['G2', 'F#2', 'E2', 'C2', 'A2', 'B2', 'D2', 'G2'])), fend), **kw)
    S.line('cb', voice(list(zip(x, ['G1', 'F#1', 'E1', 'C2', 'A1', 'B1', 'D2', 'G1'])), fend), att=.8, rel=1.0,
           vib_cents=4, voices=4, dyn=[(t_, v_ * .6) for t_, v_ in kw['dyn']])
    # harp: gentle roll at the dissolve; the final roll LANDS on the tonic and dies away naturally
    for i, p in enumerate(('G2', 'D3', 'G3', 'B3', 'D4')):
        S.n('harp', r + .05 + .06 * i, p, .3, pan=-.4 + .15 * i, decay=.5, ring=3.5)
    for i, p in enumerate(('G3', 'D4', 'G4', 'B4', 'D5')):
        S.n('harp', ft - .1 + .025 * i, p, .34 - .02 * i, pan=-.4 + .18 * i, decay=.35)
    # the eye lights up green: celesta sparkle (open G-C-D colour over the Cmaj9)
    for i, p in enumerate(('G6', 'C7', 'D7', 'G7')):
        S.n('celesta', eg + .05 * i, p, .3 - .03 * i, pan=.3)
    S.n('celesta', ft + .05, 'G6', .16, pan=.25)          # a last glint on the tonic
    # soft choir 'oh' on the cadence (D -> G), releasing with the strings
    for inst, a_, b_ in (('choir_s', 'A4', 'B4'), ('choir_a', 'F#4', 'G4'), ('choir_t', 'D4', 'D4'),
                         ('choir_b', 'D3', 'G2')):
        S.line(inst, voice([(ft - .58, a_), (ft, b_)], fend - .1), att=.45, rel=.9, vowel='o',
               dyn=[(ft - .58, .2), (ft, .3), (fend, .22)])


def build():
    S = Score()
    for f in (act1, suspense, life, journey, storm, night, gift, rebirth, climax, resolution):
        f(S)
    return S
