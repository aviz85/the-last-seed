"""Scan: render every frame at t+eps and compare with the cached frame at t (float-boundary glitches)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from critic_lib import load_lr, OUT

EPS = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-6

def job(k):
    import film
    return film.frame(k / 30 + EPS, hires=False)

if __name__ == '__main__':
    lr = load_lr()
    res = []
    with Pool(2) as pool:
        for k, img in enumerate(pool.imap(job, range(1800), chunksize=8)):
            d = (img != lr[k]).any(-1).sum()
            res.append(d)
    res = np.array(res)
    np.save(os.path.join(OUT, f'epsscan_{EPS:g}.npy'), res)
    idx = np.argsort(-res)[:40]
    for k in sorted(idx):
        if res[k] > 0:
            print(f'f{k} t={k/30:.3f} diffpx={res[k]}')
