"""Review stills / sheets / mp4 of the end title.

  python tools/title_stills.py film  --at 0.2 0.6 1.0 1.6 2.4        # film.frame(57+t) stills, 3x
  python tools/title_stills.py stand --at 0.2 0.6 1.0 1.6 2.4 [--eye 200 72]
  python tools/title_stills.py stand --sheet [--eye 150 125]          # contact sheet 0..3 s
  python tools/title_stills.py stand --mp4 [--from -1.5]              # motion check (3x)
  python tools/title_stills.py film --sheet
Outputs -> review/title_*.png|mp4
"""
import argparse
import importlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from PIL import Image
from engine import *
from engine.core import BAYER8
from engine import text as T
import film
import shots.title as TT
from tools.title_standin import standin

REVIEW = os.path.join(ROOT, 'review')
T0 = [o.start for o in film.OVERLAYS if o.module == 'shots.title'][0]
END = film.DURATION


def composite_standin(t, eye):
    """Mimics film.frame (low-res) with the stand-in close-up underneath
    (also used as the picture the title's eye-light pool re-lights).
    t = title local time (may be < 0: before the title)."""
    TT.BASE_OVERRIDE = lambda tt: snap(standin(2.0 + tt, eye))
    img = snap(standin(2.0 + t, eye))
    if 0 <= t < TT.DURATION:
        res = TT.render(t)
        fade = res.get('under_fade', 0.0)
        if fade > 0:
            img = film._darken(img, min(fade, 1.0) * film.FADE_STEPS)
        lay = snap(res['layer'])
        m = lay[..., 3] > 127
        img[m] = lay[..., :3][m]
    gt = T0 + t
    if gt > END - 0.5:
        a = (END - gt) / 0.5
        img = film._darken(img, (1.0 - max(a, 0.0)) * film.FADE_STEPS)
    return img


def get(mode, t, eye):
    if mode == 'film':
        TT.BASE_OVERRIDE = None
        return film.frame(T0 + t, hires=False)
    return composite_standin(t, eye)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['film', 'stand'])
    ap.add_argument('--at', type=float, nargs='*')
    ap.add_argument('--sheet', action='store_true')
    ap.add_argument('--mp4', action='store_true')
    ap.add_argument('--strip', type=float, default=None)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--every', type=int, default=1, help='frame step for strips')
    ap.add_argument('--scale', type=int, default=3)
    ap.add_argument('--eye', type=float, nargs=2, default=None)
    ap.add_argument('--from', dest='t0', type=float, default=-1.0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--crop', type=int, nargs=4, default=None, help='x y w h (low-res px), saved at 6x')
    a = ap.parse_args()
    eye = tuple(a.eye) if a.eye else (200, 72)
    if a.mode == 'stand':
        TT.EYE_OVERRIDE = eye
    tag = a.mode + (('_' + a.tag) if a.tag else '')
    if a.at:
        for t in a.at:
            img = get(a.mode, t, eye)
            if a.crop:
                x, y, w, h = a.crop
                img = upscale(img[y:y + h, x:x + w], 6)
                p = os.path.join(REVIEW, f'title_{tag}_crop_t{t:04.2f}.png')
                Image.fromarray(img).save(p)
                print(p)
                continue
            img = upscale(img, a.scale)
            p = os.path.join(REVIEW, f'title_{tag}_t{t:04.2f}.png')
            Image.fromarray(img).save(p)
            print(p)
    if a.sheet:
        ts = np.linspace(0.0, TT.DURATION - 0.05, 12)
        cols = 3
        rows = 4
        pad = 2
        out = np.zeros((rows * (H + pad) - pad, cols * (W + pad) - pad, 3), np.uint8)
        out[:] = (40, 40, 48)
        for k, t in enumerate(ts):
            f = get(a.mode, float(t), eye).copy()
            T.text(f, f'{t:.2f}', 2, 2, 'dust7')
            r, c = divmod(k, cols)
            out[r * (H + pad):r * (H + pad) + H, c * (W + pad):c * (W + pad) + W] = f
        p = os.path.join(REVIEW, f'title_{tag}_sheet.png')
        Image.fromarray(upscale(out, 2)).save(p)
        print(p)
    if a.strip is not None and a.crop:
        # consecutive frames of a cropped region, stacked (for motion detail)
        x, y, w, h = a.crop
        frames = []
        for k in range(a.n):
            t = a.strip + k * a.every / FPS
            f = get(a.mode, t, eye)[y:y + h, x:x + w].copy()
            lab = canvas('ink1', h=h, w=24)
            T.text(lab, f'{t:.2f}', 1, 2, 'dust7')
            frames.append(np.concatenate([lab, f], 1))
        cols = 2
        rows = (len(frames) + cols - 1) // cols
        fw = w + 24
        out = np.zeros((rows * (h + 2), cols * (fw + 2), 3), np.uint8)
        out[:] = (40, 40, 48)
        for k, f in enumerate(frames):
            c, r = divmod(k, rows)
            out[r * (h + 2):r * (h + 2) + h, c * (fw + 2):c * (fw + 2) + fw] = f
        p = os.path.join(REVIEW, f'title_{tag}_cstrip{a.strip:04.2f}.png')
        Image.fromarray(upscale(out, a.scale)).save(p)
        print(p)
        a.strip = None
    if a.strip is not None:
        frames = []
        for k in range(a.n):
            t = a.strip + k / FPS
            f = get(a.mode, t, eye).copy()
            T.text(f, f'{t:.3f}', 2, 2, 'dust7')
            frames.append(f)
        cols = min(4, a.n)
        rows = (a.n + cols - 1) // cols
        pad = 2
        out = np.zeros((rows * (H + pad) - pad, cols * (W + pad) - pad, 3), np.uint8)
        out[:] = (40, 40, 48)
        for k, f in enumerate(frames):
            r, c = divmod(k, cols)
            out[r * (H + pad):r * (H + pad) + H, c * (W + pad):c * (W + pad) + W] = f
        p = os.path.join(REVIEW, f'title_{tag}_strip{a.strip:04.2f}.png')
        Image.fromarray(upscale(out, 2)).save(p)
        print(p)
    if a.mp4:
        sc = a.scale
        p = os.path.join(REVIEW, f'title_{tag}.mp4')
        cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{W * sc}x{H * sc}', '-r', str(FPS), '-i', '-',
               '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '16', '-pix_fmt', 'yuv420p', p]
        pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        n0 = int(round(a.t0 * FPS))
        for k in range(n0, int(round(TT.DURATION * FPS))):
            img = upscale(get(a.mode, k / FPS, eye), sc)
            pr.stdin.write(np.ascontiguousarray(img).tobytes())
        pr.stdin.close()
        pr.wait()
        print(p)


if __name__ == '__main__':
    main()
