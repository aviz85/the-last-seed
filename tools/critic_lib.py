"""critic_lib — read-only review helpers for the fresh-eyes critic.
All outputs go to review/critic/. Never touches project files.

  cache_lowres()            render all 1800 low-res composited frames -> review/critic/lr.npy
  load_lr()                 memmap of (1800,180,320,3)
  grid(frames, labels, cols, scale, path)
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from PIL import Image

OUT = os.path.join(ROOT, 'review', 'critic')
os.makedirs(OUT, exist_ok=True)
FPS = 30
NF = 1800
H, W = 180, 320


def _lr_frame(k):
    import film
    return film.frame(k / FPS, hires=False)


def _hr_frame(t):
    import film
    return film.frame(t, hires=True)


def cache_lowres(workers=2, path=None):
    from multiprocessing import Pool
    path = path or os.path.join(OUT, 'lr.npy')
    arr = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8, shape=(NF, H, W, 3))
    st = time.time()
    with Pool(workers) as pool:
        for k, img in enumerate(pool.imap(_lr_frame, range(NF), chunksize=8)):
            arr[k] = img
            if k % 200 == 0:
                print(f'frame {k}  {time.time() - st:.0f}s', flush=True)
    arr.flush()
    return path


def load_lr():
    return np.load(os.path.join(OUT, 'lr.npy'), mmap_mode='r')


def upscale(img, s):
    return np.repeat(np.repeat(img, s, axis=0), s, axis=1)


def label(img, s, x=2, y=2):
    from engine import text as T
    img = np.ascontiguousarray(img).copy()
    try:
        T.text(img, s, x, y, 'white1', shadow='ink0')
    except Exception:
        pass
    return img


def grid(frames, labels=None, cols=4, scale=2, path='x.png', pad=2, bg=(40, 40, 48), margin=False):
    frames = [np.asarray(f) for f in frames]
    if labels is not None and not margin:
        frames = [label(f, l) for f, l in zip(frames, labels)]
    frames = [upscale(f, scale) if scale > 1 else f for f in frames]
    if labels is not None and margin:
        out_ = []
        for f, l in zip(frames, labels):
            strip = np.zeros((12, f.shape[1], 3), np.uint8)
            strip[:] = (20, 20, 24)
            strip = label(strip, l, 1, 2)
            out_.append(np.concatenate([strip, f], 0))
        frames = out_
    h, w = frames[0].shape[:2]
    rows = (len(frames) + cols - 1) // cols
    out = np.zeros((rows * (h + pad) - pad, cols * (w + pad) - pad, 3), np.uint8)
    out[:] = bg
    for k, f in enumerate(frames):
        r, c = divmod(k, cols)
        out[r * (h + pad):r * (h + pad) + f.shape[0], c * (w + pad):c * (w + pad) + f.shape[1]] = f
    p = path if os.path.isabs(path) else os.path.join(OUT, path)
    Image.fromarray(out).save(p)
    return p


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'cache':
        print(cache_lowres())
