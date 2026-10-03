"""Regenerate the end-title review set that the other title tools don't cover.
python tools/title_review.py  ->
  review/title_keyframes.png     key beats over the stand-in close-up (2x)
  review/title_eye_series.png    the eye light over time (stand-in; EYE_R 3 / 5)
  review/title_heartbeat.png     eye + seedling sharing one heartbeat, title receding
  review/title_film_hires.png    a native 1080p crop from film.frame(hires=True)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from PIL import Image
from engine import *
from engine import text as T
import film
import shots.title as TT
from tools.title_stills import composite_standin

REV = os.path.join(ROOT, 'review')
EYE = (200, 72)


def pad(a):
    return np.pad(a, ((1, 1), (1, 1), (0, 0)), constant_values=40)


def keyframes():
    fr = []
    for t in (-0.2, 0.3, 0.6, 1.0, 1.35, 1.7):
        f = composite_standin(t, EYE).copy()
        T.text(f, f'{58 + t:.2f}', 2, 2, 'dust6')
        fr.append(f)
    rows = [np.concatenate(fr[i:i + 2], 1) for i in (0, 2, 4)]
    Image.fromarray(upscale(np.concatenate(rows, 0), 2)).save(os.path.join(REV, 'title_keyframes.png'))


def eye_series():
    rows = []
    for er in (3, 5):
        TT.EYE_R_OVERRIDE = er
        cells = [pad(composite_standin(t, EYE)[52:92, 170:230])
                 for t in (0.1, 0.3, 0.6, 0.9, 1.2, 1.45, 1.6, 1.85)]
        rows.append(np.concatenate(cells, 1))
    TT.EYE_R_OVERRIDE = None
    Image.fromarray(upscale(np.concatenate(rows, 0), 5)).save(os.path.join(REV, 'title_eye_series.png'))


def heartbeat():
    cells = []
    for t in (0.85, 0.95, 1.05, 1.15, 1.3, 1.4, 1.5, 1.6, 1.75, 1.9):
        f = composite_standin(t, EYE)
        c = np.concatenate([f[50:95, 170:232], f[112:152, 170:232]], 0).copy()
        T.text(c, f'{t:.2f}', 1, 1, 'dust6')
        cells.append(pad(c))
    out = np.concatenate([np.concatenate(cells[:5], 1), np.concatenate(cells[5:], 1)], 0)
    Image.fromarray(upscale(out, 4)).save(os.path.join(REV, 'title_heartbeat.png'))


def hires():
    TT.BASE_OVERRIDE = None
    f = film.frame(59.1, hires=True)
    Image.fromarray(f[700:960, 400:1520]).save(os.path.join(REV, 'title_film_hires.png'))


if __name__ == '__main__':
    TT.EYE_OVERRIDE = EYE
    keyframes()
    eye_series()
    heartbeat()
    TT.EYE_OVERRIDE = None
    hires()
    print('ok')
