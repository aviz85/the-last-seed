"""CHAR-A review sheets for the Gardener's animation set.

usage (from the repo root; set CHAR_A_DEV=<dir> to review scratch copies):
  python tools/char_a_sheet.py strips [anim ...]   -> review/char_a_strip_<anim>.png
        every anim as 10-fps frame strips at 8x: rows = normal/no can, normal/can,
        damaged/can (+ low core/eye row), extra hold frames after one-shots
  python tools/char_a_sheet.py context             -> review/char_a_context.png
        walking on a dusty background with a tick-marked moving ground, 1x and 3x,
        consecutive 30-fps frames (the shot places the robot with robot.travel())
  python tools/char_a_sheet.py slide               -> planted-foot drift report
        (medium walk / walk_tired / brace, wide walk) — must print 0 px drift
  python tools/char_a_sheet.py wide                -> review/char_a_wide.png
  python tools/char_a_sheet.py props               -> review/char_a_props.png
  python tools/char_a_sheet.py all
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
from tools.char_a_lib import dev_modules, strip, stack, save, solid  # noqa: E402
from engine import palette as P  # noqa: E402
from engine.core import upscale, paste  # noqa: E402
from engine.sprite import blit  # noqa: E402
from engine import draw, text as TX  # noqa: E402

robot, props = dev_modules()
MINE = ['idle', 'walk', 'walk_tired', 'brace', 'shield', 'water', 'droop', 'look_down', 'look_up',
        'startle', 'kneel', 'scoop', 'rise_hold']
DEFAULT_CAN = {'idle': 'empty', 'walk': 'empty', 'walk_tired': 'sprout_wilt', 'brace': 'sprout',
               'shield': 'sprout', 'water': 'empty', 'droop': 'empty', 'look_down': 'sprout',
               'look_up': 'sprout', 'startle': 'empty', 'kneel': 'empty', 'scoop': 'empty',
               'rise_hold': 'sprout'}


def times_for(anim):
    loop, d, _ = robot.ANIMS[anim]
    n = int(round(d * 10))
    ts = [k / 10 for k in range(n)]
    if not loop:
        ts += [d, d + 0.3, d + 0.9, d + 1.5]
    if anim == 'scoop':
        ts = sorted(set(ts + [0.75]))
    return ts


def strips(anims):
    out = []
    for a in anims:
        ts = times_for(a)
        can = DEFAULT_CAN.get(a)
        rows = []
        cols = min(len(ts), 12)
        for (var, c, core, eye, tag) in (('normal', None, 1.0, 1.0, 'none'),
                                          ('normal', can, 0.8, 1.0, str(can)),
                                          ('damaged', can, 0.35, 0.6, 'dmg ' + str(can))):
            fr = [robot.frame(a, t, can=c, variant=var, core=core, eye=eye) for t in ts]
            rows.append(strip(fr, [f'{t:.2f}' for t in ts], scale=8, grid=True, ground='dust3', cols=cols))
            band = solid(9, rows[-1].shape[1] // 2, 'ink0')
            TX.text(band, f'{a}  {tag}  core {core} eye {eye}', 2, 1, 'core5')
            rows.insert(len(rows) - 1, upscale(band, 2))
        p = save(stack(rows), f'char_a_strip_{a}.png')
        out.append(p)
        print(p)
    return out


def _ground_layer(w, h, cam, ticks=True):
    """dusty ground with tick marks every 8 world px (world x = screen x + cam)."""
    img = solid(h, w, 'dust5')
    draw.vgradient(img, 0, h - 14, ['ash4', 'dust4', 'dust5', 'dust6'])
    img[h - 14:] = P.C('dust4')
    for x in range(w):
        wx = x + cam
        if ticks and wx % 8 == 0:
            img[h - 14:h - 11, x] = P.C('dust2')
        if (wx * 7919) % 13 == 0:
            img[h - 10 + (wx % 5), x] = P.C('dust3')
    return img


def context():
    """robot walks right; camera fixed; ground ticks fixed in world -> any
    sliding shows as feet moving relative to the ticks."""
    rows = []
    for anim, can, scale, n in (('walk', 'empty', 'medium', 12), ('walk', 'sprout', 'medium', 12),
                                ('walk_tired', 'sprout_wilt', 'medium', 12), ('brace', 'sprout', 'medium', 12),
                                ('walk', 'empty', 'wide', 12)):
        frames = []
        for k in range(n):
            t = k * 2 / 30.0           # every 2nd 30-fps frame
            g = _ground_layer(90, 48, 0)
            spr = robot.frame(anim, t, can=can, scale=scale, variant='normal', core=0.8)
            x = 22 + robot.travel(anim, t, scale)
            blit(g, spr, x, 48 - 14, anchor='feet')
            TX.text(g, f'{t:.2f}', 1, 1, 'white0')
            frames.append(g)
        big = np.concatenate([upscale(f, 3) for f in frames[:6]], axis=1)
        big2 = np.concatenate([upscale(f, 3) for f in frames[6:12]], axis=1)
        band = solid(9, big.shape[1] // 2, 'ink0')
        TX.text(band, f'{anim} {can} {scale}: x = x0 + robot.travel(anim, t)   (3x, every 2nd frame @30fps)', 2, 1, 'core5')
        rows += [upscale(band, 2), big, big2]
    # 1x strip: a full screen-width walk
    wide = solid(60, 320, 'dust5')
    wide[:] = _ground_layer(320, 60, 0)
    for k, t in enumerate(np.arange(0, 8, 0.8)):
        spr = robot.frame('walk', t, can='empty')
        blit(wide, spr, 10 + robot.travel('walk', t) * 3 + k * 0, 60 - 14, anchor='feet')
    rows.append(upscale(wide, 3))
    one = _ground_layer(320, 60, 0)
    for i, (anim, can, scale) in enumerate((('walk', 'empty', 'medium'), ('walk', 'sprout', 'medium'),
                                            ('walk_tired', 'sprout_wilt', 'medium'), ('brace', 'sprout', 'medium'),
                                            ('walk', 'empty', 'wide'), ('idle', 'empty', 'wide'))):
        spr = robot.frame(anim, 0.3, can=can, scale=scale, core=0.8)
        blit(one, spr, 20 + i * 50, 60 - 14, anchor='feet')
    rows.append(one)
    p = save(stack(rows), 'char_a_context.png')
    print(p)
    return p


def _boot_xs(spr, scale):
    """x positions (relative to the feet anchor) of the near/far boot pixels on
    the ground row (leg region only)."""
    fx, fy = spr.anchor('feet')
    img = spr.img.copy()
    img[:, :max(fx - 12, 0)] = 0
    img[:, fx + 14:] = 0
    near_c = np.array(P.C('rust4'))
    far_c = np.array(P.C('rust2'))
    row = fy - 1 if scale == 'medium' else fy
    out = {}
    for name, c in (('near', near_c), ('far', far_c)):
        m = (img[row, :, 3] > 0) & np.all(img[row, :, :3] == c, axis=-1)
        xs = np.nonzero(m)[0]
        out[name] = set((xs - fx).tolist()) if len(xs) else None
    return out


def slide_report():
    ok = True
    for anim, scale, can in (('walk', 'medium', None), ('walk', 'medium', 'empty'), ('walk', 'medium', 'sprout'),
                             ('walk_tired', 'medium', 'sprout_wilt'), ('brace', 'medium', 'sprout'),
                             ('walk', 'wide', 'empty'), ('walk_tired', 'wide', None), ('brace', 'wide', None)):
        fps = 30
        prev = None
        worst = 0
        planted_pairs = 0
        for k in range(fps * 4):
            t = k / fps
            spr = robot.frame(anim, t, can=None, scale=scale)     # legs are identical with/without can
            d = robot.travel(anim, t, scale)
            b = _boot_xs(spr, scale)
            if anim == 'walk_tired' and scale == 'medium':
                b['near'] = None          # the hurt near leg DRAGS along the ground on purpose
            cur = {n: (None if v is None else {x + d for x in v}) for n, v in b.items()}
            if prev is not None:
                for n in ('near', 'far'):
                    if cur[n] and prev[n]:
                        common = cur[n] & prev[n]
                        if common:
                            planted_pairs += 1
                            # a planted boot must keep exactly the same world pixels
                            if cur[n] != prev[n] and len(cur[n]) == len(prev[n]):
                                shift = min(abs(a - b) for a in cur[n] for b in prev[n] if a != b) if cur[n] - prev[n] else 0
                                worst = max(worst, shift)
            prev = cur
        spd = robot.travel(anim, 4.0, scale) / 4.0
        print(f'{anim:11s} {scale:6s} can={str(can):12s} speed {spd:5.2f} px/s   planted-pair checks {planted_pairs:4d}   '
              f'max planted-boot drift {worst} px')
        ok &= worst == 0
    print('FOOT SLIDE CHECK:', 'PASS' if ok else 'FAIL')


def wide_sheet():
    rows = []
    for a in ('idle', 'walk', 'walk_tired', 'brace', 'kneel'):
        ts = [k / 10 for k in range(20 if a != 'kneel' else 8)]
        for var, can in (('normal', 'empty'), ('damaged', 'sprout')):
            fr = [robot.frame(a, t, can=can, variant=var, scale='wide', core=0.8) for t in ts]
            rows.append(strip(fr, [f'{a[:4]}{t:.1f}' for t in ts], scale=10, grid=True, ground='dust3', cols=20))
    p = save(stack(rows), 'char_a_wide.png')
    print(p)


def props_sheet():
    rows = []
    cans = [props.can(s) for s in ('empty', 'sprout', 'sprout_wilt', 'sprout_lean')]
    rows.append(strip(cans, ['empty', 'sprout', 'wilt', 'lean'], scale=10, grid=True, crop=False))
    poses = [props.can_pose(p, 'empty') for p in props.CAN_POSES]
    rows.append(strip(poses, list(props.CAN_POSES), scale=10, grid=True, crop=False))
    sp = [props.sprout(s, 0.0) for s in ('ground', 'wilt', 'planted')]
    sp += [props.sprout('lean', 0.0, lean=v) for v in (-1, -0.5, 0, 0.5, 1)]
    rows.append(strip(sp, ['ground', 'wilt', 'planted', 'l-1', 'l-.5', 'l0', 'l.5', 'l1'], scale=10, grid=True, crop=False))
    wind = [props.sprout('ground', k / 10.0, wind=1.0) for k in range(10)]
    rows.append(strip(wind, [f'w{k}' for k in range(10)], scale=10, grid=True, crop=False))
    ctx = _ground_layer(160, 30, 0, ticks=False)
    for i, s in enumerate(cans + [props.sprout('ground'), props.sprout('wilt'), props.sprout('lean', lean=-1)]):
        blit(ctx, s, 10 + i * 20, 30 - 14, anchor='base')
    rows += [upscale(ctx, 3), ctx]
    p = save(stack(rows), 'char_a_props.png')
    print(p)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if cmd == 'strips':
        strips(sys.argv[2:] or MINE)
    elif cmd == 'context':
        context()
    elif cmd == 'slide':
        slide_report()
    elif cmd == 'wide':
        wide_sheet()
    elif cmd == 'props':
        props_sheet()
    else:
        strips(MINE)
        context()
        slide_report()
        wide_sheet()
        props_sheet()
