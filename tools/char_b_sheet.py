"""CHAR-B review sheets for chars/robot_acts.py.

    python tools/char_b_sheet.py strips            # every anim as frame strips (8x)
    python tools/char_b_sheet.py strip sit         # one anim
    python tools/char_b_sheet.py context           # in-context S6 / S7 tests at 1x and 3x
    python tools/char_b_sheet.py wide              # wide-scale sprites
    python tools/char_b_sheet.py all

Outputs: review/char_b_*.png
The in-context scene re-creates the greenhouse staging (camera/planter numbers
from shots/lib_greenhouse.py: sitting feet (92,158), kneeling feet (201,153)).
"""
import os
import sys
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from PIL import Image
from engine import palette as P, draw, light, noise, text as T
from engine.core import upscale, bayer
from engine.sprite import blit, anchor_pos
from chars import robot_acts as RA
from chars import props

REV = os.path.join(ROOT, 'review')
DUR = {'sit': 1.2, 'sit_idle': 3.0, 'sit_look_can': 1.6, 'sit_look_chest': 2.2, 'dig_plant': 2.1,
       'open_chest': 1.3, 'place_core': 1.1, 'power_down': 1.7, 'dead_sit': 0.2}


def _frames(anim, t0, t1, fps=60, dedup=True, **kw):
    out = []
    last = None
    n = int(round((t1 - t0) * fps)) + 1
    for k in range(n):
        t = t0 + k / fps
        s = RA.frame(anim, t, **kw)
        key = (s.img.tobytes(), tuple(sorted(s.anchors.items())))
        if dedup and key == last:
            out[-1][2] += 1.0 / fps
            continue
        last = key
        out.append([t, s, 1.0 / fps])
    return out


def strip(anim, t0=0.0, t1=None, scale=8, cols=6, bg='night3', ground='night1', **kw):
    t1 = DUR[anim] if t1 is None else t1
    kw.setdefault('can', 'sprout_wilt')
    kw.setdefault('core', 0.6)
    kw.setdefault('eye', 1.0)
    fr = _frames(anim, t0, t1, **kw)
    cw, ch = 44, 34              # cell (low-res px)
    fx, fy = 16, 27              # feet position inside the cell
    rows = (len(fr) + cols - 1) // cols
    img = np.zeros((rows * ch, cols * cw, 3), np.uint8)
    img[:] = P.C(bg)
    for i, (t, s, hold) in enumerate(fr):
        r, c = divmod(i, cols)
        x0, y0 = c * cw, r * ch
        cell = img[y0:y0 + ch, x0:x0 + cw]
        cell[fy + 1:, :] = P.C(ground)
        # planter hint for kneeling anims: soil level y=-11 from x=+5
        if anim in ('dig_plant', 'open_chest', 'place_core', 'power_down', 'dead_sit'):
            cell[fy - 10:fy + 1, fx + 6:] = P.C('dust2')
            cell[fy - 11, fx + 5:] = P.C('ash4')
            cell[fy - 12, fx + 6:] = P.C('ochre1')
        # shot-drawn props (behind the robot, as recommended to the shot artists)
        if anim in ('dig_plant', 'open_chest', 'place_core', 'power_down', 'dead_sit') and (anim != 'dig_plant' or t >= RA.PLANT_T):
            sp = props.sprout('planted')
            blit(cell, sp, fx + RA.PLANT_OFFSET[0], fy + RA.PLANT_OFFSET[1], anchor='base')
        if anim in ('open_chest', 'place_core', 'power_down', 'dead_sit'):
            cn = props.can_pose('up_l', 'empty')
            blit(cell, cn, fx + RA.DIG_CAN_OFFSET[0], fy + RA.DIG_CAN_OFFSET[1], anchor='base')
        if (anim == 'place_core' and t >= RA.CORE_PLACE_T) or anim in ('power_down', 'dead_sit'):
            o = RA.core_orb(t, 0.8 if anim == 'place_core' else 0.5)
            blit(cell, o, fx + RA.CORE_PLACE_OFFSET[0], fy + RA.CORE_PLACE_OFFSET[1], anchor='center')
        if anim in ('sit_idle', 'sit_look_can', 'sit_look_chest') or (anim == 'sit' and t >= RA.SIT_CAN_DOWN_T):
            cn = props.can('sprout_wilt')
            blit(cell, cn, fx + RA.CAN_SET_OFFSET[0], fy + RA.CAN_SET_OFFSET[1], anchor='base')
        blit(cell, s, fx, fy, anchor='feet')
    big = upscale(img, scale)
    for i, (t, s, hold) in enumerate(fr):
        r, c = divmod(i, cols)
        lab = f'{t:.2f}' + (f' ({hold:.2f}s)' if hold > 0.1 else '')
        T.text(big, lab, c * cw * scale + 3, r * ch * scale + 3, color='white1', shadow='ink0')
    # grid lines between cells
    for c in range(1, cols):
        big[:, c * cw * scale] = (20, 20, 28)
    for r in range(1, rows):
        big[r * ch * scale, :] = (20, 20, 28)
    return big


# ---------------------------------------------------------------- context scene
CX, HZ, F, CH = 160.0, 96.0, 208.0, 2.2
XP, ZP, RP, RS, HP, HS = 2.98, 8.6, 1.36, 1.16, 0.44, 0.40
S6_SIT = (92, 158)
KNEEL = (201, 153)


def _set(kind):
    H, W = 180, 320
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    dx = (xs + 0.5 - CX) / F
    dy = -(ys + 0.5 - HZ) / F
    img = np.zeros((H, W, 3), np.uint8)
    night = kind == 'night'
    # back wall / sky
    sky = ['night1', 'night2', 'night2', 'night3'] if night else ['ash3', 'ash4', 'ash5', 'ash6']
    draw.vgradient(img, 0, 110, sky, band=0.5)
    # glazing bars (suggestion)
    for x in range(8, 320, 26):
        img[0:112, x] = P.C('night0' if night else 'ash2')
    img[40, :] = P.C('night0' if night else 'ash2')
    # floor
    fl = ['night2', 'night1'] if night else ['ash4', 'ash3']
    tf = np.where(dy < 0, CH / -np.minimum(dy, -1e-6), 1e9)
    floor = (dy < 0) & (tf < 30)
    q = ((tf < 11) & floor)
    img[floor] = P.C(fl[0])
    img[q & (bayer(H, W) < np.clip((11 - tf) / 3, 0, 1))] = P.C(fl[1])
    # planter: wall (cylinder outside), coping, soil
    a = dx * dx + 1.0
    b2 = dx * XP + ZP
    c2 = XP ** 2 + ZP ** 2 - RP * RP
    d2 = b2 * b2 - a * c2
    tp = (b2 - np.sqrt(np.maximum(d2, 0))) / a
    Yp = CH + tp * dy
    wall = (d2 > 0) & (Yp >= 0) & (Yp <= HP)
    ang = np.arctan2(tp * dx - XP, tp - ZP)
    stone = ['night3', 'night4'] if night else ['ash5', 'ash6']
    img[wall] = P.C(stone[0])
    brick = wall & ((np.floor(Yp / 0.11) % 2 == 0) ^ (np.floor(ang / 0.18) % 2 == 0)) & (noise.rand(xs.astype(int), ys.astype(int), 3) < 0.35)
    img[brick] = P.C(stone[1])
    tt = np.where(dy < 0, (CH - HP) / -np.minimum(dy, -1e-6), 1e9)
    rr = np.sqrt((tt * dx - XP) ** 2 + (tt - ZP) ** 2)
    cop = (dy < 0) & (rr <= RP) & (rr > RS) & ~(wall & (tp < tt))
    img[cop] = P.C('night4' if night else 'ash7')
    ts = np.where(dy < 0, (CH - HS) / -np.minimum(dy, -1e-6), 1e9)
    rs = np.sqrt((ts * dx - XP) ** 2 + (ts - ZP) ** 2)
    soil = (dy < 0) & (rs <= RS) & ~cop & ~(wall & (tp < ts))
    img[soil] = P.C('ochre0' if night else 'dust2')
    img[soil & (noise.rand(xs.astype(int), ys.astype(int), 5) < 0.25)] = P.C('ochre1' if night else 'dust3')
    # rubble / fallen iron rib on the left (the sitting robot leans on it)
    rub = np.zeros((H, W), bool)
    rub |= ((xs - 66) / 18.0) ** 2 + ((ys - 160) / 13.0) ** 2 < 1
    rub |= ((xs - 76) / 7.0) ** 2 + ((ys - 150) / 10.0) ** 2 < 1
    img[rub] = P.C('night3' if night else 'ash4')
    img[rub & (noise.rand(xs.astype(int) // 2, ys.astype(int) // 2, 9) < 0.4)] = P.C('night4' if night else 'ash5')
    draw.thick_line(img, 30, 118, 84, 150, 3, 'night0' if night else 'ash2')
    draw.line(img, 30, 117, 84, 149, 'night4' if night else 'ash6')
    return img


def scene(kind, anim, t, x, y, core=0.4, eye=1.0, can_state='sprout_wilt', extra=None, scale=1):
    img = _set(kind)
    lvl = np.full(img.shape[:2], -0.6 if kind == 'night' else 0.0)
    img = light.apply_light(img, lvl, mode='neutral')
    spr = RA.frame(anim, t, can=can_state, core=core, eye=eye)
    # props drawn by the SHOT per the contract
    if anim == 'sit' and t >= RA.SIT_CAN_DOWN_T or anim in ('sit_idle', 'sit_look_can', 'sit_look_chest'):
        cn = props.can(can_state)
        blit(img, cn, x + RA.CAN_SET_OFFSET[0], y + RA.CAN_SET_OFFSET[1], anchor='base')
    if anim in ('open_chest', 'place_core', 'power_down', 'dead_sit'):
        cn = props.can_pose('up_l', 'empty')
        blit(img, cn, x + RA.DIG_CAN_OFFSET[0], y + RA.DIG_CAN_OFFSET[1], anchor='base')
    planted = (anim == 'dig_plant' and t >= RA.PLANT_T) or anim in ('open_chest', 'place_core', 'power_down', 'dead_sit')
    if planted:
        sp = props.sprout('planted')
        blit(img, sp, x + RA.PLANT_OFFSET[0], y + RA.PLANT_OFFSET[1], anchor='base')
    buried = (anim == 'place_core' and t >= RA.CORE_PLACE_T) or anim in ('power_down', 'dead_sit')
    blit(img, spr, x, y, anchor='feet')
    # lights
    cx, cy = anchor_pos(spr, x, y, 'feet', 'core')
    if anim in ('sit', 'sit_idle', 'sit_look_can', 'sit_look_chest', 'dig_plant') or (anim == 'open_chest' and t < RA.CORE_OUT_T):
        img = light.light_pool(img, cx + 1, cy + 1, 34 + 10 * core, peak=1.6 + 2.2 * core, mode='warm', sy=0.8)
    if 'core_held' in spr.anchors and (anim == 'place_core' and t < RA.CORE_PLACE_T or anim == 'open_chest' and t >= RA.CORE_OUT_T):
        hx, hy = anchor_pos(spr, x, y, 'feet', 'core_held')
        img = light.light_pool(img, hx, hy, 90, peak=4.0, mode='warm', sy=0.8, core=3)
    if buried:
        gx, gy = x + RA.CORE_PLACE_OFFSET[0], y + RA.CORE_PLACE_OFFSET[1]
        k = 1.0 if anim == 'place_core' else max(0.0, 1 - (t / 1.0 if anim == 'power_down' else 1.0))
        img = light.light_pool(img, gx, gy, 26 + 30 * k, peak=1.2 + 2.0 * k, mode='warm', sy=0.6)
        o = RA.core_orb(t, 0.3 + 0.6 * k)
        blit(img, o, gx, gy + 1, anchor='center')
    ex, ey = anchor_pos(spr, x, y, 'feet', 'eye')
    if eye > 0.3 and anim not in ('dead_sit',) and not (anim == 'power_down' and t >= RA.POWER_EYE_OFF_T):
        img = light.light_pool(img, ex, ey, 6, peak=1.0, mode='warm')
    if scale > 1:
        img = upscale(img, scale)
    return img


def context_sheet(kind, items, scale=3, crop=None, label=True):
    """items: list of (anim, t, x, y, kw) -> tiles in a row/col grid (cropped around the robot)."""
    tiles = []
    for anim, t, x, y, kw in items:
        im = scene(kind, anim, t, x, y, **kw)
        if crop:
            cw, ch = crop
            x0 = int(np.clip(x - cw // 2 + 6, 0, 320 - cw))
            y0 = int(np.clip(y - ch + 10, 0, 180 - ch))
            im = im[y0:y0 + ch, x0:x0 + cw]
        big = upscale(im, scale)
        if label:
            T.text(big, f'{anim} {t:.2f}', 3, 3, color='white1', shadow='ink0')
        tiles.append(big)
    return tiles


def grid(tiles, cols, pad=4):
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    out = np.zeros((rows * (h + pad) - pad, cols * (w + pad) - pad, 3), np.uint8)
    out[:] = (30, 30, 36)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        out[r * (h + pad):r * (h + pad) + t.shape[0], c * (w + pad):c * (w + pad) + t.shape[1]] = t
    return out


def save(img, name):
    p = os.path.join(REV, name)
    Image.fromarray(img).save(p)
    print(p)
    return p


def do_strips(names=None):
    for a in (names or RA.HANDLES):
        save(strip(a), f'char_b_strip_{a}.png')


def do_context():
    sx, sy = S6_SIT
    kx, ky = KNEEL
    s6 = [('sit', 0.0, sx, sy, dict(core=0.35)), ('sit', 0.45, sx, sy, dict(core=0.35)),
          ('sit', 0.6, sx, sy, dict(core=0.35)), ('sit_idle', 0.5, sx, sy, dict(core=0.3)),
          ('sit_look_can', 0.6, sx, sy, dict(core=0.3, can_state='sprout_lean')),
          ('sit_look_chest', 1.0, sx, sy, dict(core=0.35, can_state='sprout_lean'))]
    save(grid(context_sheet('night', s6, scale=3, crop=(96, 54)), 3), 'char_b_ctx_s6_x3.png')
    s7 = [('dig_plant', 0.5, kx, ky, dict(core=0.3)), ('dig_plant', 1.0, kx, ky, dict(core=0.3)),
          ('dig_plant', 2.0, kx, ky, dict(core=0.3)), ('open_chest', 1.2, kx, ky, dict(core=1.0)),
          ('place_core', 0.25, kx, ky, dict(core=1.0)), ('place_core', 0.5, kx, ky, dict(core=1.0)),
          ('power_down', 0.45, kx, ky, dict(core=0.0)), ('power_down', 1.2, kx, ky, dict(core=0.0)),
          ('dead_sit', 0.0, kx, ky, dict(core=0.0, eye=0.0))]
    save(grid(context_sheet('dawn', s7, scale=3, crop=(96, 54)), 3), 'char_b_ctx_s7_x3.png')
    # full frames at 1x
    tiles = [scene('night', 'sit_look_chest', 1.0, sx, sy, core=0.35, can_state='sprout_lean'),
             scene('dawn', 'place_core', 0.2, kx, ky, core=1.0),
             scene('dawn', 'dead_sit', 0.0, kx, ky, core=0.0, eye=0.0)]
    save(grid(tiles, 3), 'char_b_ctx_1x.png')


def do_wide():
    tiles = []
    for a in ('sit', 'sit_idle', 'dead_sit'):
        for t in (0.0, 0.3, 0.6, 1.0, 2.0):
            s = RA.frame(a, t, scale='wide', core=0.5)
            c = np.zeros((20, 24, 3), np.uint8)
            c[:] = P.C('dust5')
            blit(c, s, 8, 16, anchor='feet')
            tiles.append(upscale(c, 10))
    save(grid(tiles, 5), 'char_b_wide.png')


# ---------------------------------------------------------------- the real shots
def real(module, times, scale=3, crop=None):
    """Render the owning artist's actual shot module at `times` (their lighting, my acting),
    optionally cropped (x0, y0, w, h) -> list of upscaled tiles."""
    import importlib
    from engine.stage import to_lowres
    m = importlib.import_module(module)
    tiles = []
    for t in times:
        f = to_lowres(m.render(t))
        if crop:
            x0, y0, w, h = crop
            f = f[y0:y0 + h, x0:x0 + w]
        big = upscale(f, scale)
        T.text(big, f'{module.split(".")[-1]} {t:.2f}', 3, 3, color='white1', shadow='ink0')
        tiles.append(big)
    return tiles


def do_real():
    save(grid(real('shots.s06_night', [0.9, 1.5, 1.8, 2.3, 3.9, 5.6], scale=4, crop=(50, 118, 90, 50)), 3),
         'char_b_real_s06.png')
    save(grid(real('shots.s07_gift', [0.5, 1.0, 1.25, 1.95, 4.6, 4.8, 5.0, 5.5, 6.9], scale=4,
                   crop=(170, 118, 80, 44)), 3), 'char_b_real_s07.png')
    save(grid(real('shots.s08_bloom_int', [0.2, 0.9, 1.5], scale=4, crop=(170, 118, 80, 44)), 3),
         'char_b_real_s08a.png')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if cmd == 'strips':
        do_strips()
    elif cmd == 'strip':
        do_strips(sys.argv[2:])
    elif cmd == 'context':
        do_context()
    elif cmd == 'wide':
        do_wide()
    elif cmd == 'real':
        do_real()
    else:
        do_strips()
        do_context()
        do_wide()
