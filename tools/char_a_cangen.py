"""CHAR-A dev tool: rasterise the watering can at a tilt angle as ASCII (a
drawing guide; the final art is hand-cleaned ASCII in chars/props.py)."""
import math
import sys
import numpy as np


def can_ascii(deg, pivot=(0.5, 3.5), W=18, H=14, ox=4, oy=4):
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)

    def rot(x, y):
        dx, dy = x - pivot[0], y - pivot[1]
        return pivot[0] + dx * c - dy * s + ox, pivot[1] + dx * s + dy * c + oy

    def inv(X, Y):
        X, Y = X - ox, Y - oy
        dx, dy = X - pivot[0], Y - pivot[1]
        return pivot[0] + dx * c + dy * s, pivot[1] - dx * s + dy * c
    g = [['.'] * W for _ in range(H)]
    body = np.zeros((H, W), bool)
    for Y in range(H):
        for X in range(W):
            u, v = inv(X + 0.5, Y + 0.5)
            # body interior box x in [3,7], y in [2,6] (pixel units, upright art)
            if 3.0 <= u < 7.0 and 2.0 <= v < 6.0:
                body[Y, X] = True
                if v < 2.9:
                    ch = 't' if u < 3.9 else ('S' if u > 6.1 else 'a')
                elif u < 3.9:
                    ch = 'o' if v < 4.5 else 'R'
                elif v > 5.0:
                    ch = 'r'
                else:
                    ch = 'R'
                g[Y][X] = ch
    # spout: from (7.5,4.5) to (11.5,1.5)
    x0, y0 = rot(7.5, 4.6)
    x1, y1 = rot(11.5, 1.5)
    n = 40
    sp = set()
    for i in range(n + 1):
        f = i / n
        X, Y = int(math.floor(x0 + (x1 - x0) * f)), int(math.floor(y0 + (y1 - y0) * f))
        sp.add((X, Y))
    for (X, Y) in sp:
        if 0 <= X < W and 0 <= Y < H and not body[Y, X]:
            g[Y][X] = 'o'
    tip = (int(math.floor(x1)), int(math.floor(y1)))
    # handle arc: points around (1,3.5) radius 1.5 on the left half
    hd = set()
    for i in range(40):
        a = math.pi / 2 + math.pi * i / 39
        X, Y = rot(2.0 + 1.5 * math.cos(a) * 1.0, 3.5 + 1.8 * math.sin(a) * -1)
        hd.add((int(math.floor(X)), int(math.floor(Y))))
    for (X, Y) in hd:
        if 0 <= X < W and 0 <= Y < H and g[Y][X] == '.':
            g[Y][X] = 'H'
    # outline
    occ = [[g[y][x] not in '.H' for x in range(W)] for y in range(H)]
    for Y in range(H):
        for X in range(W):
            if not occ[Y][X]:
                if any(0 <= Y + dy < H and 0 <= X + dx < W and occ[Y + dy][X + dx]
                       for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    g[Y][X] = 'K' if g[Y][X] in '.' else g[Y][X]
    for Y in range(H):
        for X in range(W):
            if g[Y][X] == 'H':
                g[Y][X] = 'K'
    return [''.join(r) for r in g], tip


if __name__ == '__main__':
    for d in [float(a) for a in sys.argv[1:]] or [0, 20, 35, 50]:
        rows, tip = can_ascii(d)
        print(f'--- {d} deg, tip {tip}')
        for r in rows:
            print('   ', r)
