import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import film
from engine.core import FPS
bad = []
times = {}
prev = None
jumps = []
for k in range(int(film.DURATION * FPS)):
    t = k / FPS
    i, s = film.shot_at(t)
    t0 = time.time()
    try:
        f = film.frame(t, hires=False)
    except Exception as e:
        bad.append((t, s.module, repr(e)))
        continue
    times.setdefault(s.module, []).append(time.time() - t0)
    m = f.astype(np.float32).mean()
    if prev is not None and abs(m - prev) > 25:
        jumps.append((round(t, 3), s.module, round(prev, 1), round(m, 1)))
    prev = m
print('errors:', bad[:10], len(bad))
for mname, v in times.items():
    print(f'{mname:28s} {np.mean(v)*1000:6.1f} ms/frame  max {np.max(v)*1000:6.1f}')
print('brightness jumps (>25 mean):', jumps)
