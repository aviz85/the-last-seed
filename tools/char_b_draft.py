"""CHAR-B drafting helper (NOT used at render time).

RotSprite-style nearest-neighbour rotation of an ASCII part to get a starting
point for a hand-drawn rotated variant. Prints ASCII using the same legend
chars; the result is then hand-cleaned into chars/robot_acts.py.

    python tools/char_b_draft.py head 20
"""
import os
import sys
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import numpy as np


def to_grid(art):
    lines = [l for l in art.strip('\n').split('\n')]
    ind = min((len(l) - len(l.lstrip(' ')) for l in lines if l.strip()), default=0)
    lines = [l[ind:] for l in lines]
    w = max(len(l) for l in lines)
    g = np.full((len(lines), w), '.', dtype='<U1')
    for y, l in enumerate(lines):
        for x, ch in enumerate(l):
            g[y, x] = '.' if ch == ' ' else ch
    return g


def scale2x(g):
    h, w = g.shape
    out = np.empty((h * 2, w * 2), dtype=g.dtype)
    pad = np.pad(g, 1, mode='edge')
    for y in range(h):
        for x in range(w):
            P_ = pad[y + 1, x + 1]
            A = pad[y, x + 1]; B = pad[y + 1, x + 2]; C = pad[y + 1, x]; D = pad[y + 2, x + 1]
            e0 = e1 = e2 = e3 = P_
            if C == A and C != D and A != B:
                e0 = A
            if A == B and A != C and B != D:
                e1 = B
            if D == C and D != B and C != A:
                e2 = C
            if B == D and B != A and D != C:
                e3 = D
            out[2 * y, 2 * x] = e0
            out[2 * y, 2 * x + 1] = e1
            out[2 * y + 1, 2 * x] = e2
            out[2 * y + 1, 2 * x + 1] = e3
    return out


def rotate(g, deg, pivot, n=3, pad=6):
    """Rotate grid g by deg (clockwise on screen) around pivot (x,y) in grid px."""
    big = g
    for _ in range(n):
        big = scale2x(big)
    s = 2 ** n
    h, w = g.shape
    H_, W_ = h + 2 * pad, w + 2 * pad
    out = np.full((H_, W_), '.', dtype=g.dtype)
    th = math.radians(deg)
    ct, st = math.cos(th), math.sin(th)
    px, py = pivot
    for oy in range(H_):
        for ox in range(W_):
            # centre of output pixel in original coords (before rotation)
            X = ox - pad + 0.5 - px
            Y = oy - pad + 0.5 - py
            # inverse rotate (screen clockwise rotation by th): src = R(-th) * dst
            sx = X * ct + Y * st + px
            sy = -X * st + Y * ct + py
            bx, by = int(math.floor(sx * s)), int(math.floor(sy * s))
            if 0 <= bx < big.shape[1] and 0 <= by < big.shape[0]:
                out[oy, ox] = big[by, bx]
    # trim
    m = out != '.'
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    return out[y0:y1 + 1, x0:x1 + 1], (px + pad - x0, py + pad - y0)


HEAD_D = """
....r..........
..KKSKKKKK.....
.KsS544444KKK..
Ka5444443KtTSK.
K5444443KthggSK
K44TS433KSgVYsK
K44Ss333KSgYyaK
K33r33322KssaK.
.K222r2222KKK..
..KKKKKKKK.....
"""
BODY = """
KKKKKKKKKKKKK.
KSaK45544KK43K
KSaK4444KVYK3K
KaGK3444KYyK2K
KsGK33333KK32K
KssK2r3333R22K
KKKK22r22Rr21K
....KKKKKKKKK.
"""
DRAFT_SRC = {'head': (HEAD_D, (5.5, 10.0)), 'body': (BODY, (8.5, 7.5))}


def show(g):
    return '\n'.join(''.join(r) for r in g)


if __name__ == '__main__':
    which = sys.argv[1]
    deg = float(sys.argv[2])
    art, pivot = DRAFT_SRC[which]
    g = to_grid(art)
    r, pv = rotate(g, deg, pivot)
    print(f'pivot -> {pv}')
    print(show(r))


# ---------------------------------------------------------------------------
# better drafts: vector silhouette (polygon) rotated + interior sampled from the art
def rot_draft(art, deg, pivot, polys, pad=5):
    """polys: list of polygons [(x,y)...] in art pixel-edge coordinates (0..w, 0..h)
    whose union is the silhouette. Returns (grid, new_pivot)."""
    from engine.draw import polygon_mask
    g = to_grid(art)
    h, w = g.shape
    th = math.radians(deg)
    ct, st = math.cos(th), math.sin(th)
    px, py = pivot

    def R(x, y):
        X, Y = x - px, y - py
        return (X * ct - Y * st + px + pad, X * st + Y * ct + py + pad)
    H_, W_ = h + 2 * pad, w + 2 * pad
    sil = np.zeros((H_, W_), bool)
    for poly in polys:
        sil |= polygon_mask(H_, W_, [R(x, y) for x, y in poly])
    out = np.full((H_, W_), '.', dtype=g.dtype)
    for oy in range(H_):
        for ox in range(W_):
            if not sil[oy, ox]:
                continue
            X = ox + 0.5 - pad - px
            Y = oy + 0.5 - pad - py
            sx = X * ct + Y * st + px
            sy = -X * st + Y * ct + py
            ix, iy = int(math.floor(sx)), int(math.floor(sy))
            ix = min(max(ix, 0), w - 1); iy = min(max(iy, 0), h - 1)
            c = g[iy, ix]
            if c == '.':
                # nearest opaque neighbour
                best = None
                for r in range(1, 3):
                    for dy in range(-r, r + 1):
                        for dx in range(-r, r + 1):
                            jx, jy = ix + dx, iy + dy
                            if 0 <= jx < w and 0 <= jy < h and g[jy, jx] != '.':
                                d = (jx + 0.5 - sx) ** 2 + (jy + 0.5 - sy) ** 2
                                if best is None or d < best[0]:
                                    best = (d, g[jy, jx])
                    if best:
                        break
                c = best[1] if best else 'K'
            out[oy, ox] = c
    # re-outline: silhouette edge pixels -> K
    e = sil & ~(np.roll(sil, 1, 0) & np.roll(sil, -1, 0) & np.roll(sil, 1, 1) & np.roll(sil, -1, 1))
    out[e] = 'K'
    m = out != '.'
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    return out[y0:y1 + 1, x0:x1 + 1], (px + pad - x0, py + pad - y0)


def grid_png(grids, path, scale=14):
    from engine.sprite import from_ascii
    from tools.char_b_view import grid_image
    from chars import robot_acts as RA
    from PIL import Image
    sprs = [from_ascii(show(g), RA.LEG) for g in grids]
    Image.fromarray(grid_image(sprs, scale=scale, anchors=False)).save(path)
    return path


DOME = """
....r..........
..KKSKKKKK.....
.KsS5444444....
Ka54444433.....
K5444443333....
K44TS4333333...
K44Ss3333333...
K33r3332222....
.K222r2222.....
..KKKKKKKK.....
"""
LENS = """
..KKK..
.KtTSK.
KtHggSK
KSgEesK
KSgefaK
.KssaK.
..KKK..
"""


def head_draft(deg, pupil='dr'):
    """dome rotated about the neck pivot (5.5,10) + canonical lens stamped at the
    rotated lens centre (11.5, 5.5)."""
    dome_poly = [[(2, 1), (10, 1), (10, 2), (11, 2), (11, 9), (10, 9), (10, 10), (2, 10), (2, 9),
                  (1, 9), (1, 8), (0, 8), (0, 3), (1, 3), (1, 2), (2, 2)],
                 [(4, 0), (5, 0), (5, 1), (4, 1)]]
    g, pv = rot_draft(DOME, deg, (5.5, 10.0), dome_poly, pad=6)
    th = math.radians(deg)
    X, Y = 11.5 - 5.5, 5.5 - 10.0
    cx = X * math.cos(th) - Y * math.sin(th) + pv[0]
    cy = X * math.sin(th) + Y * math.cos(th) + pv[1]
    L = to_grid(LENS)
    ox, oy = int(round(cx - 3.5)), int(round(cy - 3.5))
    H_ = max(g.shape[0], oy + 7) + 1
    W_ = max(g.shape[1], ox + 7) + 1
    out = np.full((H_, W_), '.', dtype=g.dtype)
    out[:g.shape[0], :g.shape[1]] = g
    for y in range(7):
        for x in range(7):
            if L[y, x] != '.':
                out[oy + y, ox + x] = L[y, x]
    return out, pv


BODY_SHELL = """
....KKKKKKKKK.
....K45544443K
....K4444444 K
....K3444444 K
....K33333332K
....K2r3333R2K
....K22r22Rr1K
....KKKKKKKKK.
""".replace(' ', '3')
PACK = """
KKKK
KSaK
KSaK
KaGK
KsGK
KssK
KKKK
"""
PORT = """
.KK.
KCcK
KcdK
.KK.
"""


def body_draft(deg):
    """Rotated body about the hip (8.5, 8): shell + backpack drafted separately,
    porthole re-stamped as a clean ring at its rotated centre."""
    th = math.radians(deg)
    ct, st = math.cos(th), math.sin(th)
    hx, hy = 8.5, 8.0
    pad = 6
    shell_poly = [(3, 0), (13, 0), (14, 1), (14, 7), (13, 8), (4, 8), (4, 7), (3, 7)]
    pack_poly = [(0, 0), (4, 0), (4, 7), (0, 7)]
    from engine.draw import polygon_mask
    W_, H_ = 14 + 2 * pad, 8 + 2 * pad

    def R(x, y):
        X, Y = x - hx, y - hy
        return (X * ct - Y * st + hx + pad, X * st + Y * ct + hy + pad)
    sh_m = polygon_mask(H_, W_, [R(x, y) for x, y in shell_poly])
    pk_m = polygon_mask(H_, W_, [R(x, y) for x, y in pack_poly]) & ~sh_m
    shell = to_grid(BODY_SHELL)
    pack = to_grid(PACK)
    out = np.full((H_, W_), '.', dtype=shell.dtype)

    def sample(grid, ox_, oy_, x0):
        X = ox_ + 0.5 - pad - hx
        Y = oy_ + 0.5 - pad - hy
        sx = X * ct + Y * st + hx - x0
        sy = -X * st + Y * ct + hy
        ix = min(max(int(math.floor(sx)), 0), grid.shape[1] - 1)
        iy = min(max(int(math.floor(sy)), 0), grid.shape[0] - 1)
        c = grid[iy, ix]
        return 'K' if c == '.' else c
    for oy in range(H_):
        for ox in range(W_):
            if sh_m[oy, ox]:
                out[oy, ox] = sample(shell, ox, oy, 0)
            elif pk_m[oy, ox]:
                out[oy, ox] = sample(pack, ox, oy, 0)
    sil = sh_m | pk_m
    e = sil & ~(np.roll(sil, 1, 0) & np.roll(sil, -1, 0) & np.roll(sil, 1, 1) & np.roll(sil, -1, 1))
    out[e] = 'K'
    # separator between pack and shell
    sep = sh_m & (np.roll(pk_m, 1, 1) | np.roll(pk_m, -1, 1) | np.roll(pk_m, 1, 0) | np.roll(pk_m, -1, 0))
    out[sep] = 'K'
    # porthole
    px_, py_ = R(10.0, 3.0)
    P_ = to_grid(PORT)
    ox0, oy0 = int(round(px_ - 2)), int(round(py_ - 2))
    for y in range(4):
        for x in range(4):
            if P_[y, x] != '.':
                out[oy0 + y, ox0 + x] = P_[y, x]
    nk = R(7.5, 0.0)
    shd = R(6.5, 2.5)
    m = out != '.'
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    info = {'neck': (nk[0] - x0, nk[1] - y0), 'shoulder': (shd[0] - x0, shd[1] - y0),
            'hip': (hx + pad - x0, hy + pad - y0), 'port': (ox0 + 1 - x0, oy0 + 1 - y0)}
    return out[y0:y1 + 1, x0:x1 + 1], info
