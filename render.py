"""
Render the film.

  python render.py --sheet [--every 1.0]          # contact sheet of the whole film (low-res)
  python render.py --preview [--scale 2]          # quick low-res mp4 (no audio unless present)
  python render.py --final                        # 1920x1080 H.264 + audio -> out/the_last_seed.mp4
  python render.py --range 20 26 --scale 3        # part of the film
"""
import argparse
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import film
from engine.core import W, H, FPS, SCALE, upscale
from engine import text as T

AUDIO = os.path.join(ROOT, 'audio', 'mix.wav')


def _lr(t):
    return film.frame(t, hires=False)


def _hr(t):
    return film.frame(t, hires=True)


def _scaled(args):
    t, sc = args
    if sc == SCALE:
        return film.frame(t, hires=True)
    if sc == 1:
        return film.frame(t, hires=False)
    big = film.frame(t, hires=True)
    return np.asarray(Image.fromarray(big).resize((W * sc, H * sc), Image.NEAREST))


def sheet(every, t0, t1, cols=6):
    ts = np.arange(t0, t1, every)
    rows = (len(ts) + cols - 1) // cols
    pad = 2
    out = np.zeros((rows * (H + pad), cols * (W + pad), 3), np.uint8)
    out[:] = (40, 40, 48)
    for k, t in enumerate(ts):
        f = _lr(float(t)).copy()
        T.text(f, f'{t:.1f}', 2, 2, 'white1', shadow='ink0')
        r, c = divmod(k, cols)
        out[r * (H + pad):r * (H + pad) + H, c * (W + pad):c * (W + pad) + W] = f
    p = os.path.join(ROOT, 'review', f'film_sheet_{t0:g}_{t1:g}.png')
    Image.fromarray(out).save(p)
    return p


def video(path, t0, t1, sc, audio=True, workers=2, crf=12, preset='slow'):
    n0 = int(round(t0 * FPS))
    n1 = int(round(t1 * FPS))
    ts = [k / FPS for k in range(n0, n1)]
    cmd = ['ffmpeg', '-y', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W * sc}x{H * sc}', '-r', str(FPS), '-i', '-']
    use_audio = audio and os.path.exists(AUDIO)
    if use_audio:
        cmd += ['-ss', f'{t0:.4f}', '-t', f'{t1 - t0:.4f}', '-i', AUDIO]
    cmd += ['-c:v', 'libx264', '-preset', preset, '-crf', str(crf), '-tune', 'animation',
            '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-movflags', '+faststart']
    if use_audio:
        cmd += ['-c:a', 'aac', '-b:a', '320k', '-shortest']
    cmd += [path]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    st = time.time()
    with Pool(workers) as pool:
        for k, img in enumerate(pool.imap(_scaled, [(t, sc) for t in ts], chunksize=4)):
            pr.stdin.write(np.ascontiguousarray(img).tobytes())
            if k % 150 == 0:
                print(f'  frame {k}/{len(ts)}  {time.time() - st:.0f}s', flush=True)
    pr.stdin.close()
    pr.wait()
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sheet', action='store_true')
    ap.add_argument('--every', type=float, default=1.0)
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--final', action='store_true')
    ap.add_argument('--range', type=float, nargs=2, default=None)
    ap.add_argument('--scale', type=int, default=2)
    ap.add_argument('--out', default=None)
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    t0, t1 = (0.0, film.DURATION) if a.range is None else a.range
    os.makedirs(os.path.join(ROOT, 'review'), exist_ok=True)
    os.makedirs(os.path.join(ROOT, 'out'), exist_ok=True)
    if a.sheet:
        print(sheet(a.every, t0, t1))
    if a.preview or (a.range and not a.final and not a.sheet):
        p = a.out or os.path.join(ROOT, 'review', f'film_preview_{t0:g}_{t1:g}_x{a.scale}.mp4')
        print(video(p, t0, t1, a.scale, workers=a.workers, crf=18, preset='veryfast'))
    if a.final:
        p = a.out or os.path.join(ROOT, 'out', 'the_last_seed.mp4')
        print(video(p, t0, t1, SCALE, workers=a.workers, crf=12, preset='slow'))


if __name__ == '__main__':
    main()
