"""critic_stats — per-frame luminance / chroma / frame-difference stats of the cached low-res film."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from critic_lib import load_lr, OUT

lr = load_lr()
N = lr.shape[0]
Y = np.zeros(N); R = np.zeros(N); G = np.zeros(N); B = np.zeros(N); D = np.zeros(N); CH = np.zeros(N)
prev = None
for k in range(N):
    f = lr[k].astype(np.float32)
    r, g, b = f[..., 0].mean(), f[..., 1].mean(), f[..., 2].mean()
    R[k], G[k], B[k] = r, g, b
    y = 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]
    Y[k] = y.mean()
    mx = f.max(-1); mn = f.min(-1)
    CH[k] = (mx - mn).mean()
    if prev is not None:
        D[k] = np.abs(f - prev).mean()
    prev = f
np.savez(os.path.join(OUT, 'stats.npz'), Y=Y, R=R, G=G, B=B, D=D, CH=CH)
print('saved')
