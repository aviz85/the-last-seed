import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
from engine import *
from engine import palette as P, text as T

t = time.time(); core._build_lut(); print('lut', round(time.time() - t, 2))
rows = list(P.RAMPS.keys())
cell = 14
img = canvas('ink0', h=len(rows) * cell + 4, w=11 * cell + 60)
for r, name in enumerate(rows):
    T.text(img, name[:6], 2, r * cell + 4, 'dust8')
    for i, c in enumerate(P.RAMPS[name]):
        draw.rect(img, 44 + i * cell, r * cell + 2, cell - 1, cell - 1, c)
Image.fromarray(upscale(img, 3)).save('review/palette.png')

im = canvas('ash3')
draw.vgradient(im, 0, 110, ['ash2', 'ash3', 'ash4', 'dust4', 'dust5'])
draw.rect(im, 0, 110, 320, 70, 'dust3')
for x in range(0, 320, 7):
    draw.rect(im, x, 115 + (x * 7) % 20, 5, 2, 'dust2')
draw.rect(im, 60, 60, 40, 50, 'paint3'); draw.rect(im, 64, 64, 10, 10, 'rust3'); draw.rect(im, 80, 64, 10, 10, 'steel4')
draw.rect(im, 200, 70, 40, 40, 'bark3')
t = time.time()
a = light.light_pool(im, 120, 120, 90, peak=4, mode='warm', sy=0.7)
print('light ms', round((time.time() - t) * 1000, 1))
b = light.light_pool(im, 120, 120, 90, peak=4, mode='green', sy=0.7)
c = light.apply_light(im, -np.linspace(0, 4, 320)[None, :].repeat(180, 0), mode='neutral')
d = light.vignette(light.light_pool(im, 250, 60, 120, peak=3, mode='sun'), 1.5)
grid = np.zeros((362, 642, 3), np.uint8)
grid[:180, :320] = a; grid[:180, 322:] = b; grid[182:, :320] = c; grid[182:, 322:] = d
Image.fromarray(upscale(grid, 2)).save('review/light_test.png')
print('ok')
