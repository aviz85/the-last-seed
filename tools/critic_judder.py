"""Measure per-band motion between consecutive hi-res frames (half-res 960x540 = 1/3 low-res px units).
usage: python tools/critic_judder.py T0 T1 axis(x|y) band:y0-y1[,x0-x1] ... (bands in low-res px)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from critic_lib import OUT

T0, T1, AX = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
BANDS = []
for b in sys.argv[4:]:
    parts = b.split(',')
    y0, y1 = [int(v) for v in parts[0].split('-')]
    x0, x1 = (0, 320) if len(parts) == 1 else [int(v) for v in parts[1].split('-')]
    BANDS.append((y0, y1, x0, x1))

def job(k):
    import film
    f = film.frame(k / 30, hires=True)[::2, ::2].astype(np.float32).mean(-1)
    return f

def shift_est(a, b, y0, y1, x0, x1, ax, R=9):
    # find s minimizing SAD between a[region] and b[region shifted by s]
    Y0, Y1, X0, X1 = y0 * 3, y1 * 3, x0 * 3, x1 * 3
    best = None
    ref = a[Y0 + R:Y1 - R, X0 + R:X1 - R] if ax == 'y' else a[Y0 + R:Y1 - R, X0 + R:X1 - R]
    for s in range(-R, R + 1):
        if ax == 'x':
            cand = b[Y0 + R:Y1 - R, X0 + R + s:X1 - R + s]
        else:
            cand = b[Y0 + R + s:Y1 - R + s, X0 + R:X1 - R]
        if cand.shape != ref.shape:
            continue
        d = np.abs(cand - ref).mean()
        if best is None or d < best[0]:
            best = (d, s)
    return best[1] / 3.0, best[0]

if __name__ == '__main__':
    ks = list(range(int(round(T0 * 30)), int(round(T1 * 30))))
    with Pool(2) as pool:
        fr = pool.map(job, ks, chunksize=4)
    out = []
    for i in range(1, len(fr)):
        row = [ks[i] / 30]
        for (y0, y1, x0, x1) in BANDS:
            s, err = shift_est(fr[i - 1], fr[i], y0, y1, x0, x1, AX)
            row.append(s)
        out.append(row)
    out = np.array(out)
    np.save(os.path.join(OUT, f'judder_{T0:g}_{T1:g}_{AX}.npy'), out)
    for r in out:
        print(f'{r[0]:.3f} ' + ' '.join(f'{v:+.3f}' for v in r[1:]))
