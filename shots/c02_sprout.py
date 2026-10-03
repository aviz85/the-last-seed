"""
c02 · THE SPROUT  (global 16.0–17.5, local 0–1.5)

Ground-level macro. Dry, cracked mud plates in steep perspective; soft,
low-contrast horizon bands; the last living sprout (≈55 px) growing from a
crack, the split seed husk at its base; two cotyledon leaves, the left one
drooping; it trembles in the wind; dust grains skitter right -> left.

0.60  (cue hand_enter 16.6)  three steel fingers enter from the LEFT, slowly
1.10  (cue leaf_touch 17.1)  the fingertip touches the drooping leaf's tip ->
      the leaf dips, springs back (overshoot) and settles.
"""
import math
import numpy as np
from engine import *
from engine import noise as NZ
from shots.lib_closeup import (ci, cis, shift, dshift, quantize, clean, ell, disc, edge_in, edge_out,
                               edge_side, shifted, dilate, erode, dist_to, normalize, puts, put,
                               curve_px, bezier, field, step, damped, smooth, blit_idx, PAL)

DURATION = 1.5
HH, WW = 180, 320
ys, xs = np.mgrid[0:HH, 0:WW].astype(np.float64)
TH4 = bayer(HH, WW, 4)
TH2 = bayer(HH, WW, 2)

# ------------------------------------------------------------------ camera (ground level)
Y_H = 116.0            # horizon line
FH = 38.0              # focal * camera height  -> Z = FH / (y - Y_H)
FOC = 100.0            # px per world unit at Z = 1
X_C = 160.0


def ground_xz(x, y):
    Z = FH / np.maximum(y - Y_H, 1e-3)
    X = (x - X_C) * Z / FOC
    return X, Z


def to_screen(X, Z):
    return X_C + FOC * X / Z, Y_H + FH / Z


# ------------------------------------------------------------------ sky + far bands
def _sky():
    img = np.zeros((HH, WW), np.uint8)
    # overcast: darker high up, luminous haze toward the horizon; narrow dithered seams
    v = (ys / Y_H) + 0.04 * np.sin(xs / 53.0)
    cols = cis(['dust5', 'dust6', 'dust7', 'dust8'])
    lev = np.zeros((HH, WW), int)
    for c in (0.30, 0.64, 0.90):
        a = np.clip((v - (c - 0.025)) / 0.05, 0, 1)
        lev += TH4 < a
    img[:] = cols[np.clip(lev, 0, 3)]
    # the pale sun lost in the dust (same disc as the wide shots), upper left
    sd = np.sqrt((xs - 62) ** 2 + (ys - 30) ** 2)
    img = np.where(sd <= 9.4, ci('dust8'), img)
    img = np.where(sd <= 6.4, ci('dust9'), img)
    img = np.where(sd <= 3.6, ci('dust10'), img)
    # the haze right above the far dunes is the brightest thing in the sky;
    # it swells a little behind the sprout (the leaves read against it)
    hb = Y_H - 26 - 7 * np.exp(-((xs - 186) / 60.0) ** 2)
    img = np.where(ys > hb + 2.0 * (TH4 - 0.5), ci('dust9'), img)
    img = np.where((ys > hb - 3 + 2.0 * (TH4 - 0.5)) & (ys <= hb + 2.0 * (TH4 - 0.5)) & (TH2 < 0.5), ci('dust9'), img)
    # far dunes (soft silhouettes, lower contrast than anything near)
    x_ = xs
    d1 = Y_H - 7 - 3 * np.sin(x_ / 47.0 + 0.5) - 1.5 * np.sin(x_ / 17.0) - 5 * np.exp(-((x_ - 262) / 34) ** 2)
    d2 = Y_H - 2.5 - 2.0 * np.sin(x_ / 29.0 + 2.0) - 1.0 * np.sin(x_ / 9.0 + 1)
    img = np.where((ys > d1) & (ys <= Y_H), ci('dust7'), img)
    # distant ruined tower + leaning pylon: flat silhouettes, two steps off the haze
    tw = ((np.abs(xs - 70) <= 3) & (ys > Y_H - 27 + 3 * (xs > 71))) | \
         ((np.abs(xs - 66) <= 1) & (ys > Y_H - 19)) | \
         ((np.abs(xs - 75) <= 1) & (ys > Y_H - 14)) | \
         ((xs > 62) & (xs < 79) & (ys > Y_H - 10))
    img = np.where(tw & (ys < d2), ci('dust7'), img)
    py = (np.abs((xs - 236) - 0.35 * (ys - (Y_H - 12))) <= 0.7) & (ys > Y_H - 20) & (ys < d2)
    img = np.where(py, ci('dust7'), img)
    img = np.where((ys > d2) & (ys <= Y_H + 1), ci('dust6'), img)
    return img


# ------------------------------------------------------------------ cracked ground
CELL = 0.78


def _voronoi(X, Z, cell, seed, jit=0.8):
    gx = X / cell
    gz = Z / cell
    ix = np.floor(gx).astype(int)
    iz = np.floor(gz).astype(int)
    F1 = np.full(X.shape, 1e9)
    F2 = np.full(X.shape, 1e9)
    ID = np.zeros(X.shape, np.int64)
    P1x = np.zeros(X.shape)
    P1z = np.zeros(X.shape)
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            cx = ix + dx
            cz = iz + dz
            px = cx + 0.5 + (NZ.rand(cx, cz, seed) - 0.5) * jit
            pz = cz + 0.5 + (NZ.rand(cx, cz, seed + 1) - 0.5) * jit
            d = np.sqrt((gx - px) ** 2 + ((gz - pz) * 1.0) ** 2)
            closer = d < F1
            F2 = np.where(closer, F1, np.minimum(F2, d))
            ID = np.where(closer, cx * 7919 + cz * 104729, ID)
            P1x = np.where(closer, px, P1x)
            P1z = np.where(closer, pz, P1z)
            F1 = np.where(closer, d, F1)
    return F1 * cell, F2 * cell, ID, P1x * cell, P1z * cell


def _ground():
    img = np.zeros((HH, WW), np.uint8)
    gm = ys > Y_H + 0.5
    X, Z = ground_xz(xs, ys)
    F1, F2, ID, PX, PZ = _voronoi(X, Z, CELL, 11)
    edge = (F2 - F1) / 2.0                              # world distance to the cell boundary
    cw = 0.020 + 0.018 * NZ.rand(ID, 3, 5)              # crack half-width per plate (world)
    crack = gm & (edge < cw)
    plate = gm & ~crack
    # plate top shading: per-plate tilt + edge curl toward the far-left light
    tilt = (NZ.rand(ID, 7, 5) - 0.5)
    ox, oz = X - PX, Z - PZ                              # from the plate centre outward
    on = np.sqrt(ox ** 2 + oz ** 2) + 1e-6
    toward_light = (-ox * 0.75 + oz * 0.66) / on         # outward normal . light (far-left)
    curl = np.clip(1 - edge / 0.09, 0, 1)
    v = 0.52 + 0.16 * tilt + 0.30 * curl * toward_light - 0.12 * curl * (toward_light < -0.2)
    # atmospheric haze: far plates flatten toward the horizon tone
    far = np.clip((Z - 1.6) / 4.0, 0, 1)
    v = v * (1 - far) + 0.62 * far
    # near foreground slightly darker (out of the light, very near)
    v = v - 0.10 * np.clip((ys - 160) / 20, 0, 1)
    top = quantize(v, ['dust4', 'dust5', 'dust6', 'dust7', 'dust8'], cuts=[0.34, 0.47, 0.62, 0.78])
    img = np.where(plate, top, img)
    img = clean(img, plate, 2)
    # crack interior: the far plate's wall (facing us), then deep shadow
    wall_px = np.maximum(1, np.round(0.028 * FOC / Z)).astype(int)
    above = np.zeros((HH, WW), bool)
    for k in range(1, 7):
        sh = shifted(plate, 0, k)                      # plate k px above this pixel
        above |= sh & (wall_px >= k)
    wall = crack & above
    img = np.where(crack, ci('dust1'), img)
    img = np.where(crack & (Z > 1.9), ci('dust3'), img)
    img = np.where(crack & (Z > 3.2), ci('dust5'), img)
    img = np.where(crack & (Z > 5.0), top, img)
    iso = crack & (Z > 1.5)
    nb = shifted(iso, 1, 0).astype(int) + shifted(iso, -1, 0) + shifted(iso, 0, 1) + shifted(iso, 0, -1) + \
         shifted(iso, 1, 1) + shifted(iso, -1, -1) + shifted(iso, 1, -1) + shifted(iso, -1, 1)
    img = np.where(iso & (nb < 2), top, img)
    img = np.where(wall, ci('dust3'), img)
    img = np.where(wall & shifted(plate, 0, 1), ci('dust4'), img)
    # lit lip on the near plate's top edge (curled edge catching light)
    lip = plate & edge_side(plate, 0, -1) & (Z < 2.4) & (toward_light > -0.1)
    img = np.where(lip, shift(img, 1), img)
    img = np.where(lip & (Z < 1.3) & (toward_light > 0.45), ci('dust9'), img)
    # fine secondary cracks on the plate tops (thin, sparse)
    f1, f2, id2, _, _ = _voronoi(X, Z, CELL * 0.33, 23)
    e2 = (f2 - f1) / 2.0
    fine = plate & (e2 < 0.0045 * np.maximum(Z, 0.6)) & (NZ.rand(id2, 1, 3) > 0.62) & (Z < 1.6) & (edge > 0.06)
    img = np.where(fine, shift(img, -1), img)
    # haze near the horizon (far ground dissolves into dust)
    hz = gm & (ys < Y_H + 5)
    img = np.where(hz, ci('dust6'), img)
    return img, plate, crack, X, Z, ID, edge


SKY = _sky()
GROUND, PLATE_M, CRACK_M, GX, GZ, GID, GEDGE = _ground()

# ------------------------------------------------------------------ sprout placement
def _find_crack_near(x0, y0, r=10):
    best = None
    for y in range(int(y0 - r), int(y0 + r)):
        for x in range(int(x0 - r), int(x0 + r)):
            if 0 <= x < WW and 0 <= y < HH and CRACK_M[y, x] and PLATE_M[min(y + 2, HH - 1), x]:
                d = (x - x0) ** 2 + (y - y0) ** 2
                if best is None or d < best[0]:
                    best = (d, x, y)
    return (best[1], best[2]) if best else (int(x0), int(y0))


BASE_X, BASE_Y = _find_crack_near(176, 150, 14)

# ------------------------------------------------------------------ pebbles + grains (static)
def _pebbles(img):
    rng = 0
    for k in range(70):
        X = (NZ.rand(k, 1, 91) - 0.5) * 3.2
        Z = 0.62 + NZ.rand(k, 2, 91) ** 1.6 * 2.2
        x, y = to_screen(X, Z)
        x, y = int(round(x)), int(round(y))
        if not (2 < x < WW - 3 and Y_H + 3 < y < HH - 2):
            continue
        if not PLATE_M[y, x]:
            continue
        big = NZ.rand(k, 3, 91) > 0.7 and Z < 1.3
        base = img[y, x]
        lo = shift(np.array([[base]]), -2)[0, 0]
        hi = shift(np.array([[base]]), 1)[0, 0]
        if big:
            pat = [(0, 0, hi), (1, 0, base), (-1, 1, base), (0, 1, base), (1, 1, lo), (2, 1, lo), (0, 2, lo), (1, 2, lo)]
        else:
            pat = [(0, 0, hi), (1, 0, lo), (0, 1, lo)]
        for dx, dy, c in pat:
            if 0 <= x + dx < WW and 0 <= y + dy < HH and PLATE_M[y + dy, x + dx]:
                img[y + dy, x + dx] = c
    return img


GROUND = _pebbles(GROUND)
BG = np.where(ys > Y_H + 0.5, GROUND, SKY).astype(np.uint8)

# ------------------------------------------------------------------ the sprout
STEM_H = 42


def stem_pts(sway):
    """stem centre per row, base -> node (float x, int y)."""
    pts = []
    for i in range(STEM_H + 1):
        s = i / STEM_H
        x = BASE_X + 1.6 * math.sin(s * math.pi * 1.15) - 1.2 * s + sway * s ** 1.8
        pts.append((x, BASE_Y - i))
    return pts


def _draw_stem(img, sway):
    pts = stem_pts(sway)
    n = len(pts)
    for i, (x, y) in enumerate(pts):
        s = i / (n - 1)
        if s < 0.12:
            cols = ['leaf8', 'leaf7', 'leaf5']          # pale, just out of the dark
        elif s < 0.40:
            cols = ['leaf7', 'leaf6', 'leaf4']
        elif s < 0.80:
            cols = ['leaf7', 'leaf5', 'leaf3']
        else:
            cols = ['leaf6', 'leaf4']
        wdt = len(cols)
        x0 = int(math.floor(x - (wdt - 1) / 2 + 0.5))
        for k in range(wdt):
            put(img, x0 + k, int(y), cols[k])
    return pts[-1]


LEAF_R = dict(L=22.0, W=6.2, droop=0.10, dark=0)        # fresh leaf, reaching up-right
LEAF_L = dict(L=21.0, W=5.4, droop=0.62, dark=1)        # tired leaf, drooping to the left


def _raster_leaf(ang, L, W, droop, dark):
    """Rasterise a cotyledon directly at angle `ang` (0 = right, +90deg = down).
    Its base is at local (0,0); returns (idx, mask, (ox, oy)) where (ox,oy) is
    the base pixel inside the sprite, and the tip offset."""
    R = int(L + W + 6)
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float64)
    ux, uy = math.cos(ang), math.sin(ang)
    nx, ny = -uy, ux                                   # 'below' side for a rightward leaf
    ss = np.linspace(0.0, 1.0, 70)
    best = np.full(xx.shape, 1e9)
    bs = np.zeros(xx.shape)
    bt = np.zeros(xx.shape)
    for s_ in ss:
        mx = ux * s_ * L
        my = uy * s_ * L + droop * L * s_ * s_          # droop always bends toward screen-down
        d2 = (xx - mx) ** 2 + (yy - my) ** 2
        better = d2 < best
        best = np.where(better, d2, best)
        bs = np.where(better, s_, bs)
        # signed distance: + on the screen-lower side of the midrib
        tx, ty = ux, uy + 2 * droop * s_
        tl = math.hypot(tx, ty)
        side = ((xx - mx) * (-ty / tl) + (yy - my) * (tx / tl))
        bt = np.where(better, side, bt)
    if ux < 0:
        bt = -bt                                       # keep '+' = lower side for leftward leaves
    dist = np.sqrt(best)
    s_ = bs
    # broad, rounded cotyledon with a short petiole and a soft point
    blade = np.clip((s_ - 0.12) / 0.88, 0, 1)
    wprof = W * np.sin(np.pi * blade ** 0.85) ** 0.62
    wprof = np.where(s_ < 0.12, 0.6, wprof)
    t_ = np.sign(bt) * dist
    inside = dist <= wprof + 0.35
    inside &= ~((s_ >= 0.999) & (dist > 0.6))
    rel = t_ / np.maximum(wprof, 0.6)                  # -1 top edge .. +1 bottom edge
    v = 0.66 - 0.36 * rel - 0.10 * (blade > 0.82) - 0.14 * dark
    col = quantize(v, ['leaf3', 'leaf4', 'leaf5', 'leaf6', 'leaf7'], cuts=[0.30, 0.45, 0.62, 0.84])
    img = np.where(inside, col, 0).astype(np.uint8)
    img = clean(img, inside, 2)
    # midrib: a thin darker groove with a lit pixel row above it
    rib = inside & (np.abs(t_ - 0.2) < 0.5) & (blade > 0.05) & (blade < 0.72)
    img = np.where(rib, ci('leaf4' if not dark else 'leaf3'), img)
    # selective outline
    ol = edge_in(inside)
    img = np.where(ol & (rel > -0.15), ci('leaf2'), img)
    img = np.where(ol & (rel <= -0.15) & (blade > 0.1), ci('leaf6' if dark else 'leaf7'), img)
    img = np.where(ol & (blade > 0.93), ci('leaf3'), img)
    # crisp highlight on the upper surface
    hl = inside & ~ol & (rel < -0.40) & (blade > 0.25) & (blade < 0.52)
    img = np.where(hl, ci('leaf8' if not dark else 'leaf7'), img)
    tip = (ux * L, uy * L + droop * L)
    return img, inside, (R, R), tip


_LEAF_CACHE = {}


def leaf_at(which, ang):
    q = int(round(math.degrees(ang) / 2.0))
    key = (which, q)
    if key not in _LEAF_CACHE:
        P_ = LEAF_R if which == 'R' else LEAF_L
        _LEAF_CACHE[key] = _raster_leaf(math.radians(q * 2.0), P_['L'], P_['W'], P_['droop'], P_['dark'])
    return _LEAF_CACHE[key]


ANG_R0 = math.radians(-26)                     # right leaf points up-right
ANG_L0 = math.radians(180 - 14)                 # left leaf points left (droop bends it down)

for _a in range(-30, 31, 2):
    leaf_at('R', ANG_R0 + math.radians(_a))
    leaf_at('L', ANG_L0 + math.radians(_a))


def leaf_tip(which, ang, node):
    img, m, base, tip = leaf_at(which, ang)
    return node[0] + tip[0], node[1] + tip[1]


# ------------------------------------------------------------------ seed husk (hand-drawn)
# the two halves of the split seed, left one still cupping the stem
# left half: stands cupped against the stem (outer shell lit on the left,
#            pale inner lining facing the stem);  right half: fallen on its
#            back, open side up (pale lining as a crescent), shell underneath.
HUSK_L = ["..ab...",
          ".abcx..",
          "abbcxx.",
          "abccxx.",
          "abcsxx.",
          "abccx..",
          ".bccd..",
          "..dd..."]
HUSK_R = ["..........",
          "...xxxxx..",
          ".axxxxxxxc",
          "abbbsbbccd",
          ".ddcccdd..",
          "...ddd...."]
HUSK_COL = {'a': 'ochre6', 'b': 'ochre4', 'c': 'ochre3', 'd': 'ochre1', 'x': 'dust9', 's': 'ochre2'}


def _husk(img):
    bx, by = BASE_X, BASE_Y
    for j, row in enumerate(HUSK_L):
        for i, ch in enumerate(row):
            if ch != '.':
                put(img, bx - 6 + i, by - 7 + j, HUSK_COL[ch])
    for j, row in enumerate(HUSK_R):
        for i, ch in enumerate(row):
            if ch != '.':
                put(img, bx + 2 + i, by - 4 + j, HUSK_COL[ch])
    # the lining is not flat white: shade its far edge
    put(img, bx - 2, by - 3, 'dust8')
    put(img, bx + 9, by - 2, 'dust8')
    put(img, bx + 3, by - 2, 'dust8')
    # soft contact shadow on the plate
    for (dx, dy) in ((12, 1), (11, 1), (10, 2), (9, 2), (8, 2), (-7, 0), (-6, 1), (-5, 1)):
        x, y = bx + dx, by + dy
        if 0 <= x < WW and 0 <= y < HH and PLATE_M[y, x]:
            img[y, x] = shift(img[y:y + 1, x:x + 1], -1)[0, 0]
    # crumbs of soil pushed up by the sprout, and root hairs in the crack
    for (dx, dy, c) in ((-8, 0, 'dust3'), (-9, 1, 'dust2'), (13, 1, 'dust3'), (0, 1, 'dust8'),
                        (-1, 2, 'dust7'), (1, 2, 'dust8'), (0, 3, 'dust6'), (2, 3, 'dust6')):
        put(img, bx + dx, by + dy, c)
    return img


# ------------------------------------------------------------------ the hand (pre-rendered poses)
STEEL = ['ink1', 'steel0', 'steel1', 'steel2', 'steel3', 'steel4', 'steel5', 'steel6']
HAND_H, HAND_W = 150, 250
HOX, HOY = 70, 60                     # offset of the hand rig inside its sprite
PAINTC = ['paint1', 'paint2', 'paint3', 'paint4', 'paint5']


def _seg(Hd, Wd, x0, y0, ang, L, r, ends=0.55):
    """mechanical finger segment: a rounded box along `ang` from (x0,y0), length L,
    half-thickness r. Returns mask, across (-1 top .. +1 bottom), along 0..1."""
    yy, xx = np.mgrid[0:Hd, 0:Wd].astype(np.float64)
    ux, uy = math.cos(ang), math.sin(ang)
    nx, ny = -uy, ux
    a = (xx - x0) * ux + (yy - y0) * uy
    b = (xx - x0) * nx + (yy - y0) * ny
    half = L / 2 + r * ends
    q = np.abs((a - L / 2) / half) ** 5 + np.abs(b / r) ** 2.4
    return q <= 1.0, b / r, a / L


def _hand_pose(bend):
    Hd, Wd = HAND_H, HAND_W
    img = np.zeros((Hd, Wd), np.uint8)
    m = np.zeros((Hd, Wd), bool)
    z = np.full((Hd, Wd), -1e9)

    def paint(mask, col, depth):
        sel = mask & (depth >= z)
        img[sel] = col[sel] if isinstance(col, np.ndarray) else ci(col)
        z[sel] = depth
        m[sel] = True
        return sel

    def steel_shade(across, along, dim=0.0):
        v = 0.56 - 0.40 * across - dim
        v = v + 0.35 * (np.abs(across + 0.55) < 0.16)          # specular line along the top
        return quantize(v, STEEL[2:8], cuts=[0.24, 0.38, 0.54, 0.72, 0.9])

    # chain: forearm -> wrist ring -> palm, each starting where the last ends
    fa_ang = math.radians(33)
    PX0, PY0 = HOX + 16, HOY + 34                       # palm start (knuckle side is to the right)
    WL = 9.0
    wx0 = PX0 - 4 - WL * math.cos(fa_ang)
    wy0 = PY0 - 8 - WL * math.sin(fa_ang)
    FL = 172.0
    fx0 = wx0 - (FL - 2) * math.cos(fa_ang)
    fy0 = wy0 - (FL - 2) * math.sin(fa_ang)
    fa, fac, fal = _seg(Hd, Wd, fx0, fy0, fa_ang, FL, 8.0, 0.0)
    # thin steel arm (model: steel arms), a mint elbow plate far up out of frame
    fcol = quantize(0.56 - 0.40 * fac, STEEL[2:7], cuts=[0.24, 0.38, 0.54, 0.72])
    paint(fa, fcol, 0)
    seam = fa & (np.abs(fal * FL - (FL - 22)) < 0.8)
    img[seam] = ci('steel1')
    lip = fa & (np.abs(fal * FL - (FL - 20.5)) < 0.7)
    img[lip] = shift(img, 1)[lip]
    for c_ in (-0.4, 0.35):
        L_ = FL - 30
        rx = fx0 + L_ * math.cos(fa_ang) - c_ * 10.5 * math.sin(fa_ang)
        ry = fy0 + L_ * math.sin(fa_ang) + c_ * 10.5 * math.cos(fa_ang)
        rx, ry = int(round(rx)), int(round(ry))
        img[ry, rx] = ci('steel5')
        img[ry + 1, rx + 1] = ci('steel1')
    wr, wrc, wrl = _seg(Hd, Wd, wx0, wy0, fa_ang, WL, 11.6, 0.05)
    wv = 0.56 - 0.40 * wrc - 0.12 * (np.abs(wrl - 0.5) > 0.36)
    paint(wr, quantize(wv, STEEL[2:7], cuts=[0.24, 0.38, 0.54, 0.72]), 1)
    # palm: mint back plate on top, steel underside
    pm, pmc, pml = _seg(Hd, Wd, PX0, PY0, math.radians(10), 22, 12.0, 0.6)
    pv = 0.56 - 0.40 * pmc
    pcol = np.where(pmc < 0.25, quantize(pv, PAINTC, cuts=[0.24, 0.38, 0.56, 0.78]),
                    quantize(pv - 0.05, STEEL[2:7], cuts=[0.24, 0.38, 0.54, 0.72]))
    paint(pm, pcol, 2)
    # ring finger (behind, darkest), curled under
    x, y = HOX + 38, HOY + 45
    for ang, L, r in ((20, 16, 4.6), (62, 12, 4.3), (100, 8, 4.0)):
        sm, sc, sl = _seg(Hd, Wd, x, y, math.radians(ang), L, r)
        paint(sm, steel_shade(sc, sl, 0.22), 3)
        x, y = x + L * math.cos(math.radians(ang)), y + L * math.sin(math.radians(ang))
    # middle finger, half curled
    x, y = HOX + 40, HOY + 40
    for ang, L, r in ((10, 19, 4.9), (40, 14, 4.6), (78, 9, 4.3)):
        sm, sc, sl = _seg(Hd, Wd, x, y, math.radians(ang), L, r)
        paint(sm, steel_shade(sc, sl, 0.12), 4)
        x, y = x + L * math.cos(math.radians(ang)), y + L * math.sin(math.radians(ang))
    # index finger: extended toward the leaf
    joints = []
    x, y = HOX + 40, HOY + 31
    for k, (ang, L, r) in enumerate(((6, 24, 5.2), (12, 19, 4.9), (18 + bend, 13, 4.6))):
        sm, sc, sl = _seg(Hd, Wd, x, y, math.radians(ang), L, r)
        paint(sm, steel_shade(sc, sl, 0.0), 6 + k * 0.1)
        joints.append((x, y, r))
        x, y = x + L * math.cos(math.radians(ang)), y + L * math.sin(math.radians(ang))
    a3 = math.radians(18 + bend)
    tip = (x + 4.6 * 0.55 * math.cos(a3), y + 4.6 * 0.55 * math.sin(a3))
    img = clean(img, m, 2)
    # knuckles: round joint caps with a pin (index finger front-most)
    for (jx, jy, r) in joints:
        jm = disc(Hd, Wd, jx, jy, r - 0.6, 0.2)
        img[jm] = ci('steel3')
        img[jm & ~erode(jm, 1)] = ci('steel1')
        img[jm & edge_side(jm, 0, -1)] = ci('steel4')
        jxi, jyi = int(round(jx)), int(round(jy))
        for (dx, dy, c) in ((-1, -1, 'steel6'), (0, -1, 'steel5'), (-1, 0, 'steel5'), (0, 0, 'steel2'), (1, 1, 'steel1')):
            img[jyi + dy, jxi + dx] = ci(c)
    # rivets on the back plate + a rust stain
    for (rx, ry) in ((HOX + 20, HOY + 27), (HOX + 30, HOY + 29)):
        img[ry, rx] = ci('paint5')
        img[ry + 1, rx + 1] = ci('paint1')
    for (rx, ry, c) in ((HOX + 24, HOY + 44, 'rust2'), (HOX + 25, HOY + 45, 'rust1'), (HOX + 26, HOY + 45, 'rust2'), (HOX + 14, HOY + 38, 'rust1')):
        if m[ry, rx]:
            img[ry, rx] = ci(c)
    # outline: dark steel0 around, lit sky rim along the top of the index finger
    ol = edge_out(m)
    border = np.zeros_like(m)
    border[:2, :] = True
    border[:, :2] = True
    ol &= ~border
    img[ol] = ci('steel0')
    m = m | ol
    # soft rubber pad under the fingertip
    tx, ty = int(round(tip[0])), int(round(tip[1]))
    for (dx, dy) in ((-2, 3), (-3, 3), (-1, 3), (-4, 2)):
        X_, Y_ = tx + dx, ty + dy
        if 0 <= X_ < Wd and 0 <= Y_ < Hd and m[Y_, X_] and not ol[Y_, X_]:
            img[Y_, X_] = ci('ink2')
    return img, m, tip


HAND_A = _hand_pose(0.0)
HAND_B = _hand_pose(9.0)

# ------------------------------------------------------------------ timing
T_ENTER, T_TOUCH = 0.60, 1.10


def sprout_state(t):
    """sway of the stem + leaf angles (wind tremble + touch bob)."""
    t12 = step(t, 12)
    w = NZ.fbm1(np.array([t12 * 2.2]), 3, 3)[0] - 0.5
    sway = round(2.0 * w)                                    # integer px at the node
    fl_r = math.radians(7.0 * (NZ.vnoise1(np.array([t12 * 6.0]), 5)[0] - 0.5))
    fl_l = math.radians(6.0 * (NZ.vnoise1(np.array([t12 * 5.0]), 8)[0] - 0.5))
    bob = 0.0
    if t >= T_TOUCH:
        dt = t - T_TOUCH
        push = math.radians(12) * math.exp(-((dt - 0.05) / 0.06) ** 2) if dt < 0.16 else 0.0
        spring = -math.radians(14) * damped(dt - 0.10, 2.6, 4.2) if dt > 0.10 else 0.0
        bob = push + spring
    # leftward leaf: +angle rotates the tip UP (counter-clockwise from 180) -> subtract to droop
    return sway, ANG_R0 + fl_r, ANG_L0 - fl_l - bob


def hand_pos(t, node, ang_l):
    """integer top-left of the hand sprite so that its fingertip reaches the
    drooping leaf's tip exactly at T_TOUCH (arriving from the upper-left)."""
    tipL = leaf_tip('L', ANG_L0, node)
    tx, ty = tipL[0] - 2, tipL[1] - 3
    if t < T_ENTER:
        return None, None
    pose = HAND_B if (T_TOUCH <= t < T_TOUCH + 0.22) else HAND_A
    tip = pose[2]
    ex, ey = tx - tip[0], ty - tip[1]                    # contact position
    if t < T_TOUCH:
        u = min(max((t - T_ENTER) / (T_TOUCH - T_ENTER), 0), 1)
        e = 1 - (1 - u) ** 3                                 # quick slide in, very slow arrival
        sx, sy = ex - 158, ey - 30
        x = sx + (ex - sx) * e
        y = sy + (ey - sy) * e
    else:
        dt = t - T_TOUCH
        back = smooth((dt - 0.22) / 0.25)                    # withdraws a hair after the touch
        x = ex - 1.0 * back
        y = ey - 1.0 * back
    return (int(round(x)), int(round(y))), pose


# ------------------------------------------------------------------ dust
def _grains(img, t):
    # skittering grains along the ground (right -> left), with hops
    n = 34
    i = np.arange(n)
    X0 = (NZ.rand(i, 1, 61) - 0.5) * 3.6
    Z0 = 0.7 + NZ.rand(i, 2, 61) * 1.4
    spd = 0.9 + NZ.rand(i, 3, 61) * 1.4
    Xc = ((X0 - spd * t + 2.0) % 4.0) - 2.0
    hop = np.abs(np.sin((t * (3 + 3 * NZ.rand(i, 4, 61)) + NZ.rand(i, 5, 61)) * math.pi)) * 0.03
    x = X_C + FOC * Xc / Z0
    y = Y_H + FH / Z0 - hop * FOC / Z0
    for xi, yi, zi, k in zip(x, y, Z0, i):
        xi, yi = int(round(xi)), int(round(yi))
        if 0 <= xi < WW and Y_H < yi < HH:
            img[yi, xi] = ci('dust8' if zi < 1.2 else 'dust7')
            if zi < 0.9 and xi + 1 < WW:
                img[yi, xi + 1] = ci('dust6')           # motion smear
    # airborne motes
    xm, ym, rm = field(t, 22, 71, (-10, 60, 340, 115), vel=(-34.0, 1.5), jitter=(10, 3), sway=(1.5, 0.6))
    for xi, yi, ri in zip(xm, ym, rm):
        xi, yi = int(xi), int(yi)
        if 0 <= xi < WW and 0 <= yi < HH:
            c = ci('dust9' if ri > 0.7 else 'dust8')
            img[yi, xi] = c
            if ri > 0.45 and xi + 1 < WW:
                img[yi, xi + 1] = ci('dust7' if yi < Y_H else 'dust6')
    return img


# ------------------------------------------------------------------ render
def render(t):
    t = min(max(t, 0.0), DURATION + 1.0)
    img = BG.copy()
    sway, ar, al = sprout_state(t)
    node = _draw_stem(img, sway)
    node = (int(round(node[0])), int(round(node[1])))
    # leaves (back leaf first: the right leaf sits slightly behind)
    for which, ang in (('R', ar), ('L', al)):
        spr, sm, base, _tip = leaf_at(which, ang)
        blit_idx(img, spr, sm, node[0] - base[0], node[1] - base[1])
    # node bud: two tiny pixels where the leaves meet
    put(img, node[0], node[1], 'leaf7')
    put(img, node[0], node[1] - 1, 'leaf6')
    img = _husk(img)
    # hand
    hp, pose = hand_pos(t, node, al)
    if hp is not None:
        himg, hm, tip = pose
        blit_idx(img, himg, hm, hp[0], hp[1])
    img = _grains(img, t)
    img = _touch_fx(img, t, node)
    return PAL[img]


def _touch_fx(img, t, node):
    """the touch: a tiny living sparkle at the leaf tip + a few grains of dust
    shaken off the leaf as it bobs."""
    dt = t - T_TOUCH
    if dt < 0 or dt > 0.6:
        return img
    tx, ty = leaf_tip('L', ANG_L0, node)
    tx, ty = int(round(tx)), int(round(ty))
    if dt < 0.24:
        k = int(dt * 30)                               # frame index since contact
        arm = (1, 2, 3, 3, 2, 2, 1, 1)[min(k, 7)]
        cc = 'white0' if k < 3 else 'leaf8'
        put(img, tx, ty, cc)
        for a in range(1, arm + 1):
            col = 'leaf8' if a == 1 else ('leaf7' if a == 2 else 'leaf5')
            for (dx, dy) in ((a, 0), (-a, 0), (0, a), (0, -a)):
                put(img, tx + dx, ty + dy, col)
        if 1 <= k <= 4:
            for (dx, dy) in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
                put(img, tx + dx, ty + dy, 'leaf6')
    # dust grains falling from the leaf
    for i in range(4):
        vx = (NZ.rand(i, 1, 55) - 0.5) * 16
        g = 60.0
        x = tx - 4 + i * 2 + vx * dt
        y = ty + 1 + 5 * dt + 0.5 * g * dt * dt
        if y < BASE_Y + 2:
            put(img, int(round(x)), int(round(y)), 'dust4' if i % 2 else 'dust5')
    return img
