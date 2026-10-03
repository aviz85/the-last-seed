"""
Preview tool — render any shot module for review. Run from the repo root:

  python -m engine.preview shots.s01_wasteland --sheet            # contact sheet (default 4x4, 1x)
  python -m engine.preview shots.s01_wasteland --sheet --cols 3 --rows 3 --scale 2
  python -m engine.preview shots.s01_wasteland --at 1.0 3.5 --scale 3   # stills (3x = 960x540)
  python -m engine.preview shots.s01_wasteland --strip 2.0 --n 8 --scale 2  # 8 consecutive frames
  python -m engine.preview shots.s01_wasteland --mp4 --scale 3          # motion preview video
  python -m engine.preview shots.s01_wasteland --bench                  # render speed

Outputs go to review/<module>_*.png|mp4 and the path is printed.
Frames go through the same palette snap + compositing path as the final film
(hi-res Stage compositing is used when --scale == 6 or with --mp4 --hires).
Open PNGs with the Read tool to look at them.
"""
import argparse
import importlib
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from .core import W, H, FPS, upscale
from .stage import to_lowres, to_hires, Stage
from . import text as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REVIEW = os.path.join(ROOT, 'review')


def load(modname):
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    return importlib.import_module(modname)


def frame_lr(mod, t):
    return to_lowres(mod.render(t))


def frame_scaled(mod, t, scale):
    f = mod.render(t)
    if scale == 6 or (isinstance(f, Stage) and scale >= 2):
        big = to_hires(f, 6)
        if scale != 6:
            im = Image.fromarray(big).resize((W * scale, H * scale), Image.NEAREST)
            return np.asarray(im)
        return big
    return upscale(to_lowres(f), scale)


def label(img, s, color='white1'):
    T.text(img, s, 2, 2, color=color, shadow='ink0')
    return img


def sheet(mod, name, cols, rows, scale, t0=None, t1=None):
    d = float(getattr(mod, 'DURATION'))
    t0 = 0.0 if t0 is None else t0
    t1 = d - 1.0 / FPS if t1 is None else t1
    n = cols * rows
    ts = np.linspace(t0, t1, n)
    pad = 2
    out = np.zeros((rows * (H + pad) - pad, cols * (W + pad) - pad, 3), np.uint8)
    out[:] = (40, 40, 48)
    for k, t in enumerate(ts):
        f = frame_lr(mod, float(t)).copy()
        label(f, f'{t:.2f}')
        r, c = divmod(k, cols)
        out[r * (H + pad):r * (H + pad) + H, c * (W + pad):c * (W + pad) + W] = f
    if scale > 1:
        out = upscale(out, scale)
    p = os.path.join(REVIEW, f'{name}_sheet.png')
    Image.fromarray(out).save(p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('module')
    ap.add_argument('--sheet', action='store_true')
    ap.add_argument('--cols', type=int, default=4)
    ap.add_argument('--rows', type=int, default=4)
    ap.add_argument('--from', dest='t0', type=float, default=None)
    ap.add_argument('--to', dest='t1', type=float, default=None)
    ap.add_argument('--at', type=float, nargs='*')
    ap.add_argument('--strip', type=float, default=None)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--every', type=int, default=1, help='frame step for --strip')
    ap.add_argument('--scale', type=int, default=1)
    ap.add_argument('--mp4', action='store_true')
    ap.add_argument('--bench', action='store_true')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()

    os.makedirs(REVIEW, exist_ok=True)
    mod = load(a.module)
    name = a.module.split('.')[-1] + (('_' + a.tag) if a.tag else '')
    dur = float(mod.DURATION)

    if a.bench:
        ts = np.linspace(0, dur - 1 / FPS, 12)
        t_start = time.time()
        for t in ts:
            frame_lr(mod, float(t))
        dt = (time.time() - t_start) / len(ts)
        print(f'{a.module}: {dt * 1000:.1f} ms/frame (low-res)  ->  {dt * dur * FPS:.1f} s for the shot')

    if a.sheet:
        print(sheet(mod, name, a.cols, a.rows, a.scale, a.t0, a.t1))

    if a.at:
        for t in a.at:
            img = frame_scaled(mod, t, a.scale)
            p = os.path.join(REVIEW, f'{name}_t{t:05.2f}_x{a.scale}.png')
            Image.fromarray(img).save(p)
            print(p)

    if a.strip is not None:
        frames = []
        for k in range(a.n):
            t = a.strip + k * a.every / FPS
            f = frame_lr(mod, t).copy()
            label(f, f'{t:.3f}')
            frames.append(f)
        cols = min(4, a.n)
        rows = (a.n + cols - 1) // cols
        pad = 2
        out = np.zeros((rows * (H + pad) - pad, cols * (W + pad) - pad, 3), np.uint8)
        out[:] = (40, 40, 48)
        for k, f in enumerate(frames):
            r, c = divmod(k, cols)
            out[r * (H + pad):r * (H + pad) + H, c * (W + pad):c * (W + pad) + W] = f
        if a.scale > 1:
            out = upscale(out, a.scale)
        p = os.path.join(REVIEW, f'{name}_strip{a.strip:05.2f}.png')
        Image.fromarray(out).save(p)
        print(p)

    if a.mp4:
        sc = max(1, a.scale)
        p = os.path.join(REVIEW, f'{name}.mp4')
        cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{W * sc}x{H * sc}', '-r', str(FPS), '-i', '-',
               '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', p]
        pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        nf = int(round(dur * FPS))
        for k in range(nf):
            img = frame_scaled(mod, k / FPS, sc)
            pr.stdin.write(np.ascontiguousarray(img).tobytes())
        pr.stdin.close()
        pr.wait()
        print(p)


if __name__ == '__main__':
    main()
