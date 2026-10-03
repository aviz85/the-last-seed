"""Track the robot (mint 'paint' pixels) horizontally in hi-res film frames.
usage: python tools/critic_track.py T0 T1 x0 y0 x1 y1   (search box in low-res px)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from critic_lib import OUT

T0, T1 = float(sys.argv[1]), float(sys.argv[2])
X0, Y0, X1, Y1 = [int(v) for v in sys.argv[3:7]]
TAG = sys.argv[7] if len(sys.argv) > 7 else 'track'

def job(k):
    import film
    from engine import palette as P
    f = film.frame(k / 30, hires=True)
    c = f[Y0 * 6:Y1 * 6, X0 * 6:X1 * 6]
    names = [n for n in P.NAME_TO_IDX if n.startswith('paint')]
    m = np.zeros(c.shape[:2], bool)
    for n in names:
        m |= (c == np.array(P.C(n))).all(-1)
    cols = np.nonzero(m.any(0))[0]
    rows = np.nonzero(m.any(1))[0]
    if len(cols) == 0:
        return k, np.nan, np.nan, np.nan, np.nan
    # centroid of paint pixels + extents
    yy, xx = np.nonzero(m)
    return k, cols.min() / 6 + X0, cols.max() / 6 + X0, xx.mean() / 6 + X0, rows.min() / 6 + Y0

if __name__ == '__main__':
    ks = list(range(int(round(T0 * 30)), int(round(T1 * 30))))
    with Pool(2) as pool:
        res = pool.map(job, ks, chunksize=4)
    res = np.array(res)
    np.save(os.path.join(OUT, f'{TAG}.npy'), res)
    for k, l, r, cx, top in res:
        print(f'{k / 30:7.3f} left={l:8.3f} right={r:8.3f} cx={cx:8.3f} top={top:7.3f}')
