"""
lib_drop.py — art + helpers for c07 · THE LAST DROP (the ground-level insert
cut into S2).  Owner: c07.

Works in PALETTE-INDEX space like lib_closeup: an image is an (h, w) uint8
array of palette indices; it becomes RGB only at the very end (PAL[idx]).

  camera        the same low, ground-level lens as c02 (horizon, focal, height)
  build_sky     hazy warm-grey overcast in soft bands (the sun is off-screen upper
                left: the haze glows there), far dunes, S2's pale dead city
  build_ground  cracked mud plates (c02's Voronoi construction), dark cracks, lit lips
  raster_spout  the watering can's big rusty spout for one pose: tapered tube with a
                flared, rolled (and knocked) lip, the throat's inner wall, a dent,
                the lap seam, a riveted clamp, dust settled on its back, backlit rim
  build_stalk   the dead sunflower's stalk base, backlit; hanging_leaf: its limp
                dead leaf (S2's ash greys); draw_grass: the dry tuft at its foot
  water         shade_water: a ball of water as a lens (the world inverted inside:
                dusky ground on top, the bright haze as a bowed line, pale sky below;
                dark rim, refraction crescent + caustic, the sun's glint);
                bead_sprite / drop_sprite: pendant bead (with a pinching neck),
                falling drop (round / stretched); splat_big, crown_big (a see-through
                sheet with seven points), puddle_big
"""
import math
import numpy as np
from engine import noise as NZ
from engine.core import bayer
from shots.lib_closeup import (ci, cis, shift, quantize, clean, edge_in, edge_out, edge_side, shifted, field,
                               put, blit_idx, PAL, step)

HH, WW = 180, 320
ys, xs = np.mgrid[0:HH, 0:WW].astype(np.float64)
TH4 = bayer(HH, WW, 4)
TH2 = bayer(HH, WW, 2)

# ------------------------------------------------------------------ camera (ground level, as c02)
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
def build_sky(glow=(-46.0, -40.0)):
    img = np.zeros((HH, WW), np.uint8)
    # overcast: darker high up, luminous haze toward the horizon; narrow dithered seams
    v = (ys / Y_H) + 0.035 * np.sin(xs / 47.0 + 1.3) + 0.02 * np.sin(xs / 13.0)
    # the pale sun is off-screen, upper left (S2's key light): the haze glows there
    gd = np.sqrt((xs - glow[0]) ** 2 + ((ys - glow[1]) * 1.25) ** 2)
    v = v + 0.26 * np.clip(1 - gd / 150.0, 0, 1) ** 1.6
    cols = cis(['dust5', 'dust6', 'dust7', 'dust8'])
    lev = np.zeros((HH, WW), int)
    for c in (0.28, 0.62, 0.89):
        a = np.clip((v - (c - 0.025)) / 0.05, 0, 1)
        lev += TH4 < a
    img[:] = cols[np.clip(lev, 0, 3)]
    # the haze right above the far dunes is the brightest thing in the sky
    hb = Y_H - 24 - 6 * np.exp(-((xs - 150) / 70.0) ** 2)
    img = np.where(ys > hb + 2.0 * (TH4 - 0.5), ci('dust9'), img)
    img = np.where((ys > hb - 3 + 2.0 * (TH4 - 0.5)) & (ys <= hb + 2.0 * (TH4 - 0.5)) & (TH2 < 0.5), ci('dust9'), img)
    # far dunes (soft silhouettes, lower contrast than anything near)
    d1 = Y_H - 6 - 2.5 * np.sin(xs / 41.0 + 2.1) - 1.2 * np.sin(xs / 15.0) - 4 * np.exp(-((xs - 60) / 30) ** 2)
    d2 = Y_H - 2.5 - 1.8 * np.sin(xs / 31.0 + 0.7) - 0.9 * np.sin(xs / 8.0 + 2)
    img = np.where((ys > d1) & (ys <= Y_H), ci('dust7'), img)
    # the dead city, very far right (S2's skyline): flat blocks two steps off the haze
    city = np.zeros((HH, WW), bool)
    for (x0, x1, h) in ((236, 243, 9), (245, 249, 13), (251, 259, 7), (264, 268, 16), (269, 273, 11),
                        (278, 287, 6), (291, 295, 12), (302, 310, 8)):
        city |= (xs >= x0) & (xs <= x1) & (ys > d1.min() - h + 3) & (ys < d2)
    # a broken top on the tallest tower, and a leaning mast
    city &= ~((xs >= 266) & (ys < Y_H - 20 + (xs - 266) * 1.0))
    city |= (np.abs((xs - 292) - 0.3 * (ys - (Y_H - 12))) <= 0.5) & (ys > Y_H - 22) & (ys < d2)
    # cool and pale with distance, like S2's skyline (lit edges a step lighter)
    img = np.where(city & (ys < d1), ci('ash8'), img)
    img = np.where(city & (ys < d1) & ~shifted(city, -1, 0), ci('ash7'), img)
    img = np.where((ys > d2) & (ys <= Y_H + 1), ci('dust6'), img)
    return img


# ------------------------------------------------------------------ cracked ground (c02's construction)
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
            d = np.sqrt((gx - px) ** 2 + (gz - pz) ** 2)
            closer = d < F1
            F2 = np.where(closer, F1, np.minimum(F2, d))
            ID = np.where(closer, cx * 7919 + cz * 104729, ID)
            P1x = np.where(closer, px, P1x)
            P1z = np.where(closer, pz, P1z)
            F1 = np.where(closer, d, F1)
    return F1 * cell, F2 * cell, ID, P1x * cell, P1z * cell


def build_ground(seed=31, xoff=0.0):
    img = np.zeros((HH, WW), np.uint8)
    gm = ys > Y_H + 0.5
    X, Z = ground_xz(xs, ys)
    X = X + xoff
    F1, F2, ID, PX, PZ = _voronoi(X, Z, CELL, seed)
    edge = (F2 - F1) / 2.0
    cw = 0.020 + 0.018 * NZ.rand(ID, 3, 5)
    crack = gm & (edge < cw)
    plate = gm & ~crack
    tilt = (NZ.rand(ID, 7, 5) - 0.5)
    ox, oz = X - PX, Z - PZ
    on = np.sqrt(ox ** 2 + oz ** 2) + 1e-6
    toward_light = (-ox * 0.75 + oz * 0.66) / on
    curl = np.clip(1 - edge / 0.09, 0, 1)
    v = 0.52 + 0.16 * tilt + 0.30 * curl * toward_light - 0.12 * curl * (toward_light < -0.2)
    far = np.clip((Z - 1.6) / 4.0, 0, 1)
    v = v * (1 - far) + 0.62 * far
    v = v - 0.10 * np.clip((ys - 160) / 20, 0, 1)
    top = quantize(v, ['dust4', 'dust5', 'dust6', 'dust7', 'dust8'], cuts=[0.34, 0.47, 0.62, 0.78])
    img = np.where(plate, top, img)
    img = clean(img, plate, 2)
    wall_px = np.maximum(1, np.round(0.028 * FOC / Z)).astype(int)
    above = np.zeros((HH, WW), bool)
    for k in range(1, 7):
        sh = shifted(plate, 0, k)
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
    lip = plate & edge_side(plate, 0, -1) & (Z < 2.4) & (toward_light > -0.1)
    img = np.where(lip, shift(img, 1), img)
    img = np.where(lip & (Z < 1.3) & (toward_light > 0.45), ci('dust9'), img)
    f1, f2, id2, _, _ = _voronoi(X, Z, CELL * 0.33, seed + 12)
    e2 = (f2 - f1) / 2.0
    fine = plate & (e2 < 0.0045 * np.maximum(Z, 0.6)) & (NZ.rand(id2, 1, 3) > 0.62) & (Z < 1.6) & (edge > 0.06)
    img = np.where(fine, shift(img, -1), img)
    hz = gm & (ys < Y_H + 5)
    img = np.where(hz, ci('dust6'), img)
    return dict(img=img, plate=plate, crack=crack, X=X, Z=Z, ID=ID, edge=edge, top=top)


def pebbles(img, plate, seed=91, n=70, avoid=()):
    for k in range(n):
        X = (NZ.rand(k, 1, seed) - 0.5) * 3.2
        Z = 0.62 + NZ.rand(k, 2, seed) ** 1.6 * 2.2
        x, y = to_screen(X, Z)
        x, y = int(round(x)), int(round(y))
        if not (2 < x < WW - 3 and Y_H + 3 < y < HH - 2):
            continue
        if not plate[y, x]:
            continue
        if any(a[0] <= x <= a[2] and a[1] <= y <= a[3] for a in avoid):
            continue
        big = NZ.rand(k, 3, seed) > 0.7 and Z < 1.3
        base = img[y, x]
        lo = shift(np.array([[base]]), -2)[0, 0]
        hi = shift(np.array([[base]]), 1)[0, 0]
        if big:
            pat = [(0, 0, hi), (1, 0, base), (-1, 1, base), (0, 1, base), (1, 1, lo), (2, 1, lo), (0, 2, lo), (1, 2, lo)]
        else:
            pat = [(0, 0, hi), (1, 0, lo), (0, 1, lo)]
        for dx, dy, c in pat:
            if 0 <= x + dx < WW and 0 <= y + dy < HH and plate[y + dy, x + dx]:
                img[y + dy, x + dx] = c
    return img


# ------------------------------------------------------------------ the spout
def _fbm2(u, v, seed, oct=3):
    return NZ.fbm2(u, v, seed, oct)


class SpoutGeom:
    """Screen-space geometry of the tapered spout for one pose.
    tip: centre of the mouth; ang: axis angle below horizontal (rad, pointing
    right-down); k: how much the mouth faces the camera (ellipse minor/major).
    The last FL px flare out by `flare` (a rolled, slightly trumpeted end)."""
    FL = 7.0

    def __init__(self, tip, ang, k=0.46, r_tip=7.0, grow=0.036, flare=1.8):
        self.T = np.array(tip, float)
        self.a = ang
        self.d = np.array([math.cos(ang), math.sin(ang)])          # axis, toward the mouth
        self.n = np.array([-math.sin(ang), math.cos(ang)])         # across, toward the underside
        self.k, self.r_tip, self.grow, self.flare = k, r_tip, grow, flare
        self.R = r_tip + flare                                      # outer radius of the rolled lip

    def coords(self, X, Y):
        px, py = X - self.T[0], Y - self.T[1]
        s = px * self.d[0] + py * self.d[1]
        b = px * self.n[0] + py * self.n[1]
        return s, b

    def radius(self, s):
        sd = np.maximum(-s, 0)
        fl = self.flare * np.clip(1 - sd / self.FL, 0, 1) ** 2
        return self.r_tip + self.grow * sd + fl

    def lip_low(self):
        """lowest point of the outer lip (where water gathers), float screen xy."""
        R, k = self.R - 0.3, self.k
        best = None
        for ph in np.linspace(0, 2 * math.pi, 721):
            p = self.T + R * math.cos(ph) * self.n + R * k * math.sin(ph) * self.d
            if best is None or p[1] > best[1]:
                best = p
        return best


def raster_spout(tip, ang, k=0.46, r_tip=7.0, grow=0.036, flare=1.8, box_pad=4, seed=3,
                 band_sd=(96.0, 103.0), dent=(30.0, -0.30, 4.2, 0.30)):
    """Rasterise the spout (tube + rolled lip + open mouth). Returns
    dict(idx, mask, x0, y0, geom, ring, throat, h, w) — a sprite cropped to its bbox.
    Lighting: the dusty overcast dome from above + the pale sun behind (upper
    left, off-screen) -> a backlit rim along the top silhouette, a dusty back,
    rusty flanks, a warm bounce underneath."""
    g = SpoutGeom(tip, ang, k, r_tip, grow, flare)
    x0, y0 = -12, -12
    x1 = int(math.ceil(g.T[0] + g.R + box_pad))
    y1 = int(math.ceil(g.T[1] + g.R + box_pad))
    h, w = y1 - y0, x1 - x0
    Y, X = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    s, b = g.coords(X, Y)
    r = g.radius(s)
    u = b / r                                                       # -1 top edge .. +1 underside
    sd = np.clip(-s, 0, None)                                       # distance back from the mouth
    ur = u * r                                                      # px across (isotropic noise)
    tube = (s <= 0.0) & (np.abs(b) <= r + 0.25)
    band = (sd >= band_sd[0]) & (sd <= band_sd[1]) & (np.abs(b) <= r + 1.2)
    # the rolled lip took a knock: a short stretch of it (upper right) is flattened
    phm = np.arctan2(b / g.R, s / (g.R * g.k))
    kink = 1.0 - 0.11 * np.exp(-((phm + 1.05) / 0.32) ** 2)
    E = (s / (g.R * g.k * kink)) ** 2 + (b / (g.R * kink)) ** 2     # mouth ellipse (outer lip)
    Ri = g.R - 2.5
    Ei = (s / (Ri * g.k)) ** 2 + (b / Ri) ** 2                      # inner edge of the lip
    mouth = E <= 1.0
    inner = Ei <= 1.0
    mask = tube | mouth | band
    # ---------------- clean cylinder bands, parallel to the axis
    body = np.full(u.shape, ci('rust2'), np.uint8)
    body = np.where(u < -0.05, ci('rust3'), body)
    body = np.where(u > 0.45, ci('rust1'), body)
    # its back: a skin of settled dust (the dust/rust boundary is the only ragged edge)
    edge_u = -0.40 + 0.22 * (_fbm2(sd / 7.0, 7.0 + 0 * sd, seed + 21) - 0.5) + \
        0.16 * (_fbm2(sd / 2.4, 3.0 + 0 * sd, seed + 23) - 0.5)
    dusty = u < edge_u
    body = np.where(dusty, ci('dust5'), body)
    body = np.where(dusty & (u < -0.66), ci('dust6'), body)
    # rust blotches: a few broad patches of darker scale, sparse paler flakes
    n1 = NZ.fbm2(sd / 13.0, ur / 5.0 + 3.0, seed, 4) + 0.35 * (NZ.fbm2(sd / 3.0, ur / 3.0, seed + 5, 2) - 0.5)
    blot = (n1 > 0.66) & ~dusty & (u < 0.6)
    body = np.where(blot, shift(body, -1), body)
    n2 = _fbm2(sd / 4.0, ur / 3.0, seed + 7)
    flake = (n2 > 0.74) & ~dusty & (u < 0.25) & ~blot
    body = np.where(flake, shift(body, 1), body)
    # where the dust skin ends, the rust shows bright in a few chips
    chip = ~dusty & (u < edge_u + 0.14) & (_fbm2(sd / 2.5, 5.0 + 0 * sd, seed + 29) > 0.62)
    body = np.where(chip, ci('rust4'), body)
    fl = dusty & (_fbm2(sd / 2.4, ur / 2.4, seed + 27) > 0.74)
    body = np.where(fl, ci('rust3'), body)
    body = clean(body, tube, 2)
    # ---------------- the lap seam along the underside, with solder blobs
    th = 0.5 / np.maximum(r, 1)
    seam = np.abs(u - 0.30) < th
    seam_hi = np.abs(u - 0.30 + 2 * th) < th
    body = np.where(seam, ci('rust1'), body)
    body = np.where(seam_hi, shift(body, 1), body)
    blob = seam & (np.abs(((sd + 6) % 19.0) - 9.5) < 1.0) & (sd > 14)
    body = np.where(blob, ci('rust3'), body)
    # ---------------- a dent in the lit flank: a shallow crater.  Its upper-left wall
    # turns from the light (a crisp dark crescent), its floor sits in half-shade,
    # its lower-right wall faces the light (a lit crescent), a glint on the crease.
    dsd, du, dA, dB = dent
    ea = (sd - dsd) / dA                                              # + toward the mouth
    eb = (ur - du * r) / (dB * r)                                    # + toward the underside
    dr = np.sqrt(ea * ea + eb * eb)
    dm = tube & (dr < 1.0)
    facing = (-ea * 0.55 - eb * 0.83)                                  # >0: the wall facing up-back (lit)
    body = np.where(dm, ci('rust2'), body)                             # the floor, half-shaded
    body = np.where(dm & (dr > 0.45) & (facing < -0.25), ci('rust1'), body)
    body = np.where(dm & (dr > 0.62) & (facing < -0.45), ci('rust0'), body)
    body = np.where(dm & (dr > 0.50) & (facing > 0.30), ci('rust4'), body)
    body = np.where(dm & (dr > 0.72) & (facing > 0.55), ci('dust7'), body)
    rimd = tube & (dr >= 1.0) & (dr < 1.28) & (facing < -0.35)
    body = np.where(rimd, shift(body, 1), body)                        # the crease outside catches light
    # reflected ground light: a thin warm line just inside the underside
    bounce = (u > 0.80) & (u < 0.95) & (sd > 3)
    body = np.where(bounce, ci('rust2'), body)
    idx = np.where(tube, body, 0).astype(np.uint8)
    # ---------------- the riveted clamp band (steel, the robot's design language)
    bu = b / (r + 1.2)
    bcol = quantize(0.52 - 0.40 * bu, ['steel1', 'steel2', 'steel3', 'steel4', 'steel5'],
                    cuts=[0.20, 0.36, 0.56, 0.78])
    idx = np.where(band, bcol, idx)
    idx = np.where(band & ((np.abs(sd - band_sd[0]) < 0.6) | (np.abs(sd - band_sd[1]) < 0.6)), ci('steel1'), idx)
    rs = 0.5 * (band_sd[0] + band_sd[1])
    for ru in (-0.40, 0.34):
        rr = np.sqrt((sd - rs) ** 2 + ((bu - ru) * (r + 1.2)) ** 2)
        rv_ = band & (rr < 1.4)
        idx = np.where(rv_, ci('steel1'), idx)
        idx = np.where(rv_ & ((bu - ru) * (r + 1.2) < 0.3) & ((sd - rs) > -0.9), ci('steel5'), idx)
    # ---------------- the flared, rolled end: a groove where the flare starts
    fl_m = tube & (sd < g.FL) & ~mouth
    idx = np.where(fl_m & (np.abs(sd - g.FL + 0.5) < 0.55), ci('rust1'), idx)
    # ---------------- the mouth: rolled lip ring + dark throat
    ring = mouth & ~inner
    ph = np.arctan2(b / g.R, s / (g.R * g.k))                                  # around the lip
    lit = -np.sin(ph) * 0.8 - np.cos(ph) * 0.35                                 # faces up (-b) and back (-s)
    rcol = quantize(0.45 + 0.40 * lit, ['rust1', 'rust2', 'rust3', 'rust4'], cuts=[0.25, 0.45, 0.68])
    idx = np.where(ring, rcol, idx)
    # the throat: we look up into the tube — its inner wall shows as a crescent
    # (the inner ellipse minus itself pushed back along the axis), the depth is black
    Ed = ((s + 2.8 * g.k * 3.0) / (Ri * g.k)) ** 2 + ((b + 2.2) / Ri) ** 2
    deep = inner & (Ed <= 1.0)
    wallc = inner & ~deep
    idx = np.where(inner, ci('rust0'), idx)
    idx = np.where(wallc & (b / Ri < 0.1), ci('rust1'), idx)
    idx = np.where(wallc & (b / Ri < -0.55), ci('rust2'), idx)
    idx = np.where(deep, ci('ink1'), idx)
    idx = np.where(deep & (np.sqrt(Ed) < 0.55), ci('ink0'), idx)
    spec = ring & (lit > 0.78)
    idx = np.where(spec, ci('rust5'), idx)
    # ---------------- outline + backlit rim
    ol = edge_in(mask)
    top_edge = ol & (u < -0.3) & (s <= 0.0)
    idx = np.where(ol, ci('steel0'), idx)
    rim_c = np.where(sd > 64, ci('dust9'), np.where(sd > 30, ci('dust8'), np.where(sd > 10, ci('dust7'), ci('rust4'))))
    idx = np.where(top_edge & ~band, rim_c, idx)
    return dict(idx=idx, mask=mask, x0=x0, y0=y0, geom=g, ring=ring, throat=inner, h=h, w=w)


# ------------------------------------------------------------------ the dead sunflower (S2's plant)
def stalk_cx(y, base):
    """centre x of the stalk at row y (leans right toward the bowed head above)."""
    hgt = base[1] - y
    return base[0] + 7.0 * (max(hgt, 0) / 160.0) ** 1.7


def build_stalk(img, base=(296.0, 153.0), node_y=60.0, seed=5):
    """Paint the dry, woody stalk base (static) into img. Returns its mask.
    Backlit by the hazy sun (upper left): a dark fibrous silhouette with a thin
    bright rim and a few lit bristles on its left flank."""
    bx, by = base
    y0, y1 = 0, int(by) + 1
    x0, x1 = int(bx - 18), WW
    Y, X = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    hgt = by - Y
    cx = bx + 7.0 * (np.maximum(hgt, 0) / 160.0) ** 1.7 + 0.6 * np.sin(Y / 23.0)
    half = 5.4 - 1.2 * np.clip(hgt / 150.0, 0, 1) + 0.35 * np.sin(Y / 11.0 + 0.7)
    half = half + 2.6 * np.clip(1 - hgt / 8.0, 0, 1) ** 2                     # flares into the soil
    half = half + 1.0 * np.exp(-((Y - node_y) / 2.4) ** 2)                     # the node's knuckle
    a = (X - cx) / half                                                        # -1 left .. +1 right
    m = np.abs(a) <= 1.0
    # fibrous ridges running up the stalk (lit flank of each ridge, dark groove)
    wav = 0.10 * np.sin(Y / 9.0 + 1.3) + 0.06 * np.sin(Y / 3.7)
    ph = ((a + 1.0) * 1.6 + wav) % 1.0
    v = 0.44 - 0.34 * a + 0.12 * (ph < 0.30) - 0.14 * (ph > 0.78)
    v = v + 0.10 * (NZ.fbm1(Y[:, 0] / 6.0, seed)[:, None] - 0.5)
    col = quantize(v, ['dust1', 'dust2', 'dust3', 'dust4', 'dust5'], cuts=[0.16, 0.34, 0.54, 0.78])
    col = clean(col, m, 2)
    # dry splits in the rind
    for (sy, sx, ln) in ((104, -0.2, 7), (131, 0.35, 5), (38, -0.45, 6), (80, 0.1, 4)):
        spl = m & (np.abs(Y - sy - (a - sx) * 3.0) < 0.6) & (np.abs(a - sx) < ln / 12.0)
        col = np.where(spl, ci('dust1'), col)
    # the node: a darker collar ring, lit on top
    col = np.where(m & (np.abs(Y - node_y - 2.4) < 0.6), ci('dust1'), col)
    col = np.where(m & (np.abs(Y - node_y + 1.2) < 0.6) & (a < 0.4), ci('dust5'), col)
    ol = edge_in(m)
    col = np.where(ol, ci('dust1'), col)
    # backlit: the left silhouette glows (a 1-px rim), fading into the dust near the ground
    rim = ol & (a <= -0.3) & (Y < by - 6)
    col = np.where(rim, np.where(Y < 100, ci('dust7'), ci('dust6')), col)
    sub = img[y0:y1, x0:x1]
    sub[m] = col[m]
    full = np.zeros((HH, WW), bool)
    full[y0:y1, x0:x1] = m
    # a few bristles on the lit flank catching the light (short, angled up)
    for yb in (18, 29, 44, 71, 88, 97, 116, 127):
        xl = stalk_cx(yb, base) + 0.6 * math.sin(yb / 23.0)
        hl = 5.4 - 1.2 * min((by - yb) / 150.0, 1) + 0.35 * math.sin(yb / 11.0 + 0.7)
        xe = int(math.floor(xl - hl)) - 1
        put(img, xe, yb, 'dust6')
        if yb % 3 == 0:
            put(img, xe - 1, yb - 1, 'dust6')
        full[yb, xe] = True
    # where it enters the ground: a crumbly collar of dry soil + a contact shadow to the right
    for (dx, dy, c) in ((-9, 0, 'dust3'), (-8, 0, 'dust4'), (-8, -1, 'dust5'), (-10, 1, 'dust2'), (-7, 1, 'dust2'),
                        (7, 0, 'dust2'), (8, 0, 'dust3'), (9, 1, 'dust2'), (10, 1, 'dust3'), (6, 1, 'dust1'),
                        (-4, 1, 'dust2'), (-3, 1, 'dust1'), (3, 1, 'dust1'), (0, 1, 'dust1'), (1, 1, 'dust1')):
        put(img, int(bx) + dx, int(by) + dy, c)
    for k in range(11, 22):
        yy = int(by) + 1 + (k - 11) // 5
        xx = int(bx) + k
        if 0 <= xx < WW and 0 <= yy < HH:
            img[yy, xx] = shift(np.array([[img[yy, xx]]]), -1)[0, 0]
    return full


def hanging_leaf(sway=0.0, L=33.0, W=8.0, seed=41):
    """A limp, crumpled dead leaf hanging from its snapped-down petiole (S2's
    plant: ash greys, dust2 line). Local frame: (0,0) = where the petiole leaves
    the node; the leaf hangs down-left. sway (rad) swings it about the node.
    Returns (idx, mask, (ox, oy)) with (ox, oy) the node point in the sprite."""
    R = int(L + 14)
    Y, X = np.mgrid[-4:R, -R:8].astype(np.float64)
    # midrib: the petiole droops out-left, the blade hangs, the tip curls back right
    ss = np.linspace(0.0, 1.0, 90)
    cs, sn = math.cos(sway), math.sin(sway)
    pts = []
    for s_ in ss:
        if s_ < 0.22:                                   # petiole
            q = s_ / 0.22
            x, y = -7.5 * q, 1.0 + 3.5 * q * q
        else:
            q = (s_ - 0.22) / 0.78
            x = -7.5 - 2.2 * math.sin(q * 2.2) + 3.4 * q ** 2.2
            y = 4.5 + L * q
        pts.append((x * cs - y * sn, x * sn + y * cs))
    P_ = np.array(pts)
    best = np.full(X.shape, 1e9)
    bs = np.zeros(X.shape)
    side = np.zeros(X.shape)
    for i in range(len(P_) - 1):
        ax, ay = P_[i]
        dx, dy = P_[i + 1] - P_[i]
        l2 = dx * dx + dy * dy
        tt = np.clip(((X - ax) * dx + (Y - ay) * dy) / l2, 0, 1)
        px, py = ax + tt * dx, ay + tt * dy
        d2 = (X - px) ** 2 + (Y - py) ** 2
        bt = d2 < best
        best = np.where(bt, d2, best)
        bs = np.where(bt, (ss[i] + tt * (ss[i + 1] - ss[i])), bs)
        sg = np.sign((X - px) * dy - (Y - py) * dx)     # + = right of travel (the leaf's left/lit half faces -)
        side = np.where(bt, sg, side)
    d = np.sqrt(best)
    q = np.clip((bs - 0.22) / 0.78, 0, 1)
    blade = bs >= 0.22
    wprof = W * np.sin(np.pi * np.clip(q, 1e-3, 1) ** 0.62) ** 1.15
    # crumpled margin, the lit half broader, the other half curled behind
    wprof = wprof * (1 + 0.06 * np.sin(q * 29.0 + 1.0) + 0.26 * (NZ.vnoise1(q * 6.0, seed) - 0.5))
    wprof = np.where(side < 0, wprof, wprof * 0.60 * (1 + 0.25 * (NZ.vnoise1(q * 4.0, seed + 3) - 0.5)))
    tt_ = side * d
    inside = (blade & (d <= wprof + 0.3)) | (~blade & (d <= 0.9))
    rel = np.where(blade, tt_ / np.maximum(wprof, 0.6), 0.0)
    # backlit like the stalk: a dark leaf, a thin lit rim, a hint of its folds
    col = np.where(rel < 0.0, ci('ash3'), ci('dust2'))
    col = np.where(rel < -0.62, ci('ash4'), col)
    for qc, amp in ((0.24, 0.8), (0.45, 1.1), (0.66, 1.0)):
        yl = qc * L + amp * np.sin(rel * 2.6 + qc * 9)
        ridge = blade & inside & (np.abs(q * L - yl) < 0.55) & (rel < -0.1) & (rel > -0.75)
        col = np.where(ridge, ci('ash4'), col)
    col = np.where(blade & inside & (np.abs(tt_ - 0.2) < 0.55) & (q < 0.9) & (q > 0.05), ci('ash4'), col)
    col = np.where(~blade & inside, ci('dust3'), col)
    col = clean(col, inside, 1)
    ol = edge_in(inside) & blade
    col = np.where(ol, ci('dust1'), col)
    col = np.where(ol & (rel < -0.4) & (q < 0.88), ci('ash6'), col)      # backlit edge
    col = np.where(ol & (rel < -0.4) & (q < 0.5) & (q > 0.12), ci('ash7'), col)
    idx = np.where(inside, col, 0).astype(np.uint8)
    # outline the petiole below (dark underside)
    pet_ol = edge_out(inside & ~blade) & ~inside & (Y > 0)
    idx = np.where(pet_ol & shifted(inside & ~blade, 0, 1), ci('dust1'), idx)
    m = inside | (pet_ol & shifted(inside & ~blade, 0, 1))
    return idx, m, (R, 4)


GRASS = [  # base dx, height, lean (px at the tip, - = left, with the wind), colour, thickness
    (-8, 14, -5.0, 'dust3', 1), (-6, 19, -6.5, 'dust4', 1), (-5, 11, -3.0, 'dust2', 2),
    (-3, 23, -4.0, 'dust4', 1), (-2, 15, -2.5, 'dust3', 1), (4, 17, -3.5, 'dust4', 1),
    (6, 12, -4.5, 'dust3', 1), (8, 20, -7.0, 'dust5', 1), (10, 9, -3.0, 'dust2', 1), (-10, 8, -3.5, 'dust4', 1)]


def draw_grass(img, t, base=(296.0, 153.0), wind=1.0, seed=17):
    """dry grass blades at the stalk's foot, bending with the wind (12 fps)."""
    t12 = step(t, 12)
    bx, by = base
    for i, (dx, hgt, lean, c, th) in enumerate(GRASS):
        g = NZ.fbm1(np.array([t12 * 2.4 + i * 0.37]), seed, 2)[0] - 0.5
        tipx = lean * (1.0 + 0.25 * wind * g * 2)
        x0_ = bx + dx
        for k in range(hgt + 1):
            f = k / hgt
            x = x0_ + tipx * f * f
            y = by - k
            xi, yi = int(round(x)), int(round(y))
            col = c if f < 0.75 else ('dust6' if c != 'dust2' else 'dust4')
            put(img, xi, yi, col)
            if th > 1 and f < 0.6:
                put(img, xi + 1, yi, 'dust1')
    return img


# ================================================================== water, procedural (macro scale)
def _mask_from_profile(wfun, h, cx):
    """rows 0..h-1; wfun(yc) -> half-width at row centre yc (px). Column centre cx
    (float, e.g. 6.0 means the axis runs through the middle of column 6)."""
    W_ = int(math.ceil(2 * cx + 1))
    m = np.zeros((h, W_), bool)
    for y in range(h):
        w = wfun(y + 0.5)
        if w <= 0:
            continue
        for x in range(W_):
            if abs(x + 0.5 - (cx + 0.5)) <= w + 0.02:
                m[y, x] = True
    return m


def shade_water(m, cx, cy, rx, ry, spec=True, horizon=True, heavy=True):
    """Clear water inside mask m.  A ball of water is a lens: it shows the
    world upside down — the dusky ground along its top, the bright haze as a
    thin line that bows with the lens, the pale sky filling its belly — ringed
    by a dark rim; the light it gathers pools in a bright crescent and a
    caustic bottom-right; the hazy sun glints top-left.
    (cx, cy, rx, ry): the lens ellipse in px (pixel centres at +0.5)."""
    h, w = m.shape
    Y, X = np.mgrid[0:h, 0:w].astype(np.float64) + 0.5
    nx = (X - cx) / rx
    ny = (Y - cy) / ry
    rr = np.sqrt(nx * nx + ny * ny)
    idx = np.full((h, w), ci('glass3'), np.uint8)
    top = (ny < -0.28 + 0.30 * nx * nx)                               # the inverted ground
    idx = np.where(top, ci('glass2'), idx)
    idx = np.where(top & (rr > 0.78), ci('ash5'), idx)
    if horizon:
        hy = -0.28 + 0.30 * nx * nx                                    # the haze line bows with the lens
        hz = (ny >= hy) & (ny < hy + 1.25 / ry) & (np.abs(nx) < 0.80)
        idx = np.where(hz, ci('glass4'), idx)
    cres = (ny > 0.34) & (rr > 0.60) & (nx > -0.50)
    idx = np.where(cres, ci('glass4'), idx)                           # refraction crescent
    ca = ((nx - 0.26) / 0.24) ** 2 + ((ny - 0.60) / 0.17) ** 2 < 1.0
    idx = np.where(ca, ci('white0'), idx)                             # caustic
    ol = edge_in(m)
    idx = np.where(ol, ci('ash4'), idx)
    if heavy:
        idx = np.where(ol & (ny > 0.55) & (np.abs(nx) < 0.7), ci('ash3'), idx)
    if spec:
        sp = ((nx + 0.42) / 0.24) ** 2 + ((ny + 0.44) / 0.19) ** 2 < 1.0
        idx = np.where(sp & ~ol, ci('white1'), idx)
    return np.where(m, idx, 0).astype(np.uint8)


def bead_sprite(r, hgt, w0=2.2, neck=None):
    """A pendant bead hanging from the lip: contact half-width w0 at the top,
    a round belly of radius r, total height hgt.  neck=(len, w_top): the belly
    hangs `len` px lower on a 1-px thread under a small meniscus (letting go).
    Returns (idx, mask, cx) — cx = the column under the lip's lowest point."""
    cxf = math.ceil(max(r, w0)) + 0.0
    if neck is None:
        c = hgt - r

        def wf(y):
            if y >= c:
                v = r * r - (y - c) ** 2
                return math.sqrt(v) if v > 0 else 0.0
            q = y / max(c, 1e-3)
            return w0 + (r - w0) * math.sin(0.5 * math.pi * q) ** 1.3

        m = _mask_from_profile(wf, int(math.ceil(hgt)), cxf)
        idx = shade_water(m, cxf + 0.5, c, r, r * 1.02)
        return idx, m, int(cxf)
    ln, wt = neck
    body_h = hgt - ln - 2
    c = 2 + ln + (body_h - r)

    def wf(y):
        if y < 2.0:                                     # the meniscus still on the lip
            return wt * (1.0 - 0.35 * (y / 2.0))
        if y < 2 + ln:                                  # the thread
            return 0.45
        yy = y - (2 + ln)
        if y >= c:
            v = r * r - (y - c) ** 2
            return math.sqrt(v) if v > 0 else 0.0
        q = yy / max(c - (2 + ln), 1e-3)
        return 0.9 + (r - 0.9) * math.sin(0.5 * math.pi * q) ** 0.9

    m = _mask_from_profile(wf, int(math.ceil(hgt)), cxf)
    idx = shade_water(m, cxf + 0.5, c, r, r * 1.02)
    upper = np.arange(m.shape[0])[:, None] < 2 + ln
    idx = np.where(m & upper, ci('glass3'), idx)
    idx = np.where(m & upper & edge_in(m), ci('ash4'), idx)
    idx = np.where(m & upper & (np.arange(m.shape[0])[:, None] >= 2), ci('ash5'), idx)
    return idx, m, int(cxf)


def drop_sprite(r, stretch=1.0):
    """The falling drop: round (stretch 1) or drawn out along its fall (an
    ellipse, a little narrower). Returns (idx, mask, cx, bottom_row)."""
    ry = r * stretch
    rx = r / math.sqrt(stretch)
    hgt = int(math.ceil(2 * ry))
    cy = hgt - ry
    cxf = math.ceil(rx) + 0.0

    def wf(y):
        v = 1 - ((y - cy) / ry) ** 2
        # a touch more pointed on top when stretched
        k = 1.0 + 0.6 * (stretch - 1.0) * (y < cy)
        return rx * (max(v, 0.0) ** (0.5 * k))

    m = _mask_from_profile(wf, hgt, cxf)
    idx = shade_water(m, cxf + 0.5, cy + (0.1 * ry if stretch > 1 else 0), rx, ry)
    return idx, m, int(cxf), hgt - 1



# ------------------------------------------------------------------ splash at macro scale
def _canvas(Wd, Hd):
    return np.zeros((Hd, Wd), np.uint8), np.zeros((Hd, Wd), bool)


def splat_big():
    """contact: the drop slaps flat, a flash at its heart, a thin sheet racing out."""
    Wd, Hd, ox, oy = 31, 9, 15, 7
    idx, m = _canvas(Wd, Hd)
    Y, X = np.mgrid[0:Hd, 0:Wd].astype(np.float64)
    blob = ((X - ox) / 8.6) ** 2 + ((Y - (oy - 2.2)) / 3.0) ** 2 <= 1.0
    sheet = (np.abs(X - ox) <= 11) & (np.abs(Y - oy) <= 0.6)
    m = blob | sheet
    idx = np.where(m, ci('glass3'), 0).astype(np.uint8)
    idx = np.where(blob & (Y < oy - 3.2), ci('glass2'), idx)
    idx = np.where(blob & (np.abs(Y - (oy - 2.0)) < 0.6) & (np.abs(X - ox) < 6), ci('glass4'), idx)
    flash = ((X - ox + 1) / 3.2) ** 2 + ((Y - (oy - 2.5)) / 1.2) ** 2 <= 1.0
    idx = np.where(flash, ci('white1'), idx)
    idx = np.where(sheet & (np.abs(X - ox) > 8), ci('glass4'), idx)
    ol = edge_out(blob) & ~m
    idx = np.where(ol, ci('ash4'), idx)
    m = m | ol
    return idx, m, (ox, oy)


def crown_big(phase):
    """The crown, seen from ground level, at the drop's scale.
    phase 0: a thin, see-through sheet standing round the impact, seven points
    flung up and out, a bead riding each tip.  phase 1: slumped to a low,
    broken ring.  Returns (idx, mask, (ox, oy), clear) — `clear` marks sheet
    pixels drawn as 'the ground seen through water' (brightened at blit)."""
    Wd, Hd, ox, oy = 41, 24, 20, 20
    idx, m = _canvas(Wd, Hd)
    clear = np.zeros((Hd, Wd), bool)
    A, B = (12.5, 4.8) if phase == 0 else (15.5, 5.6)
    h0 = 6.0 if phase == 0 else 1.6

    def P(x, y, c, see=False):
        xi, yi = int(round(x)) + ox, int(round(y)) + oy
        if 0 <= xi < Wd and 0 <= yi < Hd:
            idx[yi, xi] = ci(c)
            m[yi, xi] = True
            clear[yi, xi] = see

    for half in (1, 0):
        for k in range(361):
            th = math.pi * k / 360.0 + (math.pi if half else 0.0)
            if phase == 1 and (0.5 + 0.5 * math.sin(th * 5 + 1.0)) < 0.28:
                continue                                    # the ring breaks up
            ex, ey = A * math.cos(th), -B * 0.35 + B * math.sin(th)
            front = math.sin(th) > 0
            hh = h0 * (0.75 + 0.25 * abs(math.cos(th))) + (1.2 if not front and phase == 0 else 0.0)
            n = int(math.ceil(hh))
            for j in range(n + 1):
                f = j / max(hh, 1e-3)
                x = ex * (1 + 0.10 * f)
                y = ey - j
                rimrow = j >= n - 1
                side = abs(math.cos(th)) > 0.86                 # seen edge-on: dense water
                if front:
                    if rimrow:
                        P(x, y, 'glass4')
                    elif side:
                        P(x, y, 'glass3')
                    else:
                        P(x, y, 'glass3', see=True)
                else:
                    P(x, y, 'glass3' if rimrow else 'glass2', see=not rimrow and not side)
    if phase == 0:
        for th, hs in ((0.0, 5.0), (0.78, 6.5), (2.36, 6.5), (math.pi, 5.0), (3.93, 7.5), (4.71, 8.5), (5.50, 7.5)):
            ex, ey = A * math.cos(th), -B * 0.35 + B * math.sin(th)
            front = math.sin(th) > 0
            base_h = h0 * (0.75 + 0.25 * abs(math.cos(th))) + (0 if front else 1.2)
            lean = 3.2 * math.cos(th)
            for j in range(int(hs) + 1):
                f = j / hs
                x = ex * 1.10 + lean * f * f
                y = ey - base_h - j
                P(x, y, 'glass4' if f < 0.8 else 'white0')
                if f < 0.35:
                    P(x + (1 if math.cos(th) >= 0 else -1), y, 'glass3')
            tx = ex * 1.10 + lean * 1.25
            ty = ey - base_h - hs - 2.2
            P(tx, ty, 'white0')
            P(tx + 1, ty, 'glass4')
            P(tx, ty - 1, 'glass4')
        P(-2, -3, 'white1')
        P(-1, -3, 'white0')
    else:
        P(-3, -1, 'white0')
        P(4, 0, 'glass4')
    Y, X = np.mgrid[0:Hd, 0:Wd]
    ol = edge_out(m) & ~m & (Y >= oy - 3)
    idx = np.where(ol, ci('ash4'), idx)
    m = m | ol
    return idx, m, (ox, oy), clear


def puddle_big(a, b):
    """the little pool left behind, a flat lens of water soaking away."""
    Wd, Hd = int(2 * a + 4), int(2 * b + 4)
    ox, oy = Wd // 2, Hd // 2
    Y, X = np.mgrid[0:Hd, 0:Wd].astype(np.float64) + 0.5
    m = ((X - ox - 0.5) / a) ** 2 + ((Y - oy - 0.5) / b) ** 2 <= 1.0
    idx = np.where(m, ci('glass2'), 0).astype(np.uint8)
    idx = np.where(m & (Y > oy + 0.5), ci('glass3'), idx)                  # reflects the bright haze
    idx = np.where(m & (np.abs(Y - oy - 1.0) < 0.6) & (np.abs(X - ox - 1.5) < a * 0.45), ci('glass4'), idx)
    idx = np.where(m & (np.abs(Y - oy - 1.0) < 0.6) & (np.abs(X - ox - 2.5) < max(a * 0.12, 0.6)), ci('white0'), idx)
    ol = edge_in(m)
    idx = np.where(ol, ci('ash4'), idx)
    return idx, m, (ox, oy)


SPLASH_BIG = [splat_big() + (None,), crown_big(0), crown_big(1)]
PUDDLES = [puddle_big(a, b) for a, b in ((11.0, 3.2), (9.0, 2.6), (7.0, 2.1), (5.0, 1.6), (3.0, 1.1))]


# ======================================================================================
# c08 · THE HEART INTO THE SOIL — the planter at dawn, seen from above-front (~35 deg)
# ======================================================================================
from shots.lib_closeup import RampLight, level_of, mech_hand, thick_curve, bezier, raster_leaf, disc, ell, curve_px

# the camera looks down at the soil at ~35 deg: world 'up' in screen-3D (x right, y down,
# z toward the viewer) and the soil's forward (away) direction
SIN35, COS35 = 0.574, 0.819
UP3 = np.array([0.0, -COS35, SIN35])
DAWN3 = np.array([0.58, -0.78, 0.22]) / np.linalg.norm([0.58, -0.78, 0.22])   # c06's grey dawn, from above RIGHT
Y_HZ = -260.0                     # virtual horizon of the soil plane (perspective)
ZOOM = 1.95                       # close-up: 1 soil unit = ZOOM px at the reference row


def soil_world(x, y):
    """screen -> soil-plane coordinates (X lateral, D depth away from us) in soil
    units: a circle on the soil reads as a flattened ellipse, smaller far away."""
    g = (150.0 - Y_HZ) / (y - Y_HZ) / ZOOM
    X = (x - 160.0) * g
    D = (1.60 * (150.0 - Y_HZ) ** 2 / (y - Y_HZ) - 1.60 * (150.0 - Y_HZ)) / ZOOM
    return X, D


def soil_z(y):
    """screen-3D depth of the soil plane at row y (orthographic approximation)."""
    return (y - 110.0) * (COS35 / SIN35)


# the planter's near rim: a stone coping along the bottom, curving up toward the
# left (the planter is round and the robot kneels at its left side)
def rim_y(x, far=False):
    x = np.asarray(x, float)
    return 150.0 + 16.0 * (x / 320.0) - 20.0 * np.clip((140.0 - x) / 140.0, 0, 1) ** 2.2


# c06's material ramps (same planter, same dawn) — the soil, stone, steel and paint
# read exactly as in the shot that follows
C08_RL = RampLight({
    'paint': dict(
        neutral=['ink0', 'paint0', 'paint1', 'paint2', 'paint3', 'paint3', 'paint4', 'paint4'],
        warm=['ink0', 'paint0', 'bark1', 'ochre2', 'ochre3', 'rust3', 'ochre4', 'rust4', 'ochre5', 'core3',
              'ochre6', 'core4'],
        cold=['ink0', 'paint0', 'paint1', 'paint2', 'paint3', 'paint4', 'paint5', 'paint6', 'dust10', 'white0'],
        members={'paint0': -2, 'paint1': -1, 'paint2': -1, 'paint3': 0, 'paint4': 1, 'paint5': 2, 'paint6': 2}),
    'steel': dict(
        neutral=['ink0', 'steel0', 'steel1', 'steel1', 'steel2', 'steel3', 'steel3', 'steel4'],
        warm=['ink0', 'steel0', 'bark1', 'rust1', 'bark2', 'rust2', 'ochre3', 'rust3', 'ochre4', 'rust4', 'core3',
              'core4', 'core5'],
        cold=['ink0', 'steel0', 'steel1', 'steel1', 'steel2', 'steel3', 'steel4', 'steel5', 'steel6', 'white0'],
        members={'steel0': -2, 'steel1': -1, 'steel2': 0, 'steel3': 1, 'steel4': 2, 'steel5': 2, 'steel6': 3}),
    'stone': dict(
        neutral=['ink0', 'dust1', 'dust2', 'dust3', 'dust4', 'dust5', 'dust5', 'dust6'],
        warm=['ink0', 'dust1', 'dust2', 'dust3', 'ochre3', 'ochre4', 'ochre5', 'ochre6', 'core4', 'core5'],
        cold=['ink0', 'dust1', 'dust2', 'dust3', 'ash4', 'ash5', 'ash6', 'ash7', 'dust9', 'ash8'],
        members={'dust0': -3, 'dust1': -2, 'dust2': -1, 'dust3': 0, 'dust4': 1, 'dust5': 2}),
    'brick': dict(
        neutral=['ink0', 'bark0', 'bark1', 'bark2', 'bark3', 'bark4', 'bark4'],
        warm=['ink0', 'bark0', 'rust1', 'bark2', 'rust2', 'ochre3', 'rust3', 'ochre4', 'rust4', 'core3'],
        cold=['ink0', 'bark0', 'bark1', 'bark2', 'bark3', 'dust5', 'bark4', 'dust6'],
        members={'bark1': -2, 'bark2': -1, 'bark3': 0, 'bark4': 1, 'bark5': 2}),
    'soil': dict(
        neutral=['ink0', 'ink1', 'bark0', 'bark1', 'bark2', 'bark2', 'bark3'],
        warm=['ink0', 'ink1', 'bark0', 'bark1', 'bark2', 'bark3', 'rust3', 'rust4', 'core3', 'core4', 'core5',
              'core6'],
        cold=['ink0', 'ink1', 'bark0', 'bark1', 'bark2', 'bark3', 'dust4', 'dust5'],
        members={'ochre0': -1, 'ochre1': 0, 'ochre2': 1}),
    'floor': dict(
        neutral=['ink0', 'ink1', 'night1', 'night2', 'ash1', 'ash2', 'ash3'],
        warm=['ink0', 'ink1', 'bark0', 'rust0', 'bark1', 'rust1', 'ochre2', 'rust2', 'ochre3'],
        cold=['ink0', 'ink1', 'night1', 'night2', 'ash2', 'ash3', 'ash4', 'ash5', 'ash6'],
        members={'night1': -1, 'night2': 0, 'night3': 1}),
})


def _nz3(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def c08_planter(heart=(150.0, 118.0), sprout=(190.0, 114.0), seed=7):
    """static albedo / normals / masks of the planter scene. Returns a dict."""
    alb = np.zeros((HH, WW), np.uint8)
    nrm = np.zeros((HH, WW, 3))
    ynear = rim_y(xs)
    yfar = np.full((HH, WW), -1.0)
    bg = np.zeros((HH, WW), bool)
    ftop = np.zeros((HH, WW), bool)
    fface = np.zeros((HH, WW), bool)
    # ---- soil: a dry crust broken into plates, crisp cracks, a few crumbs
    soil = (ys >= yfar) & (ys < ynear)
    Xs, Ds = soil_world(xs, np.clip(ys, yfar, ynear + 1))
    # the crack network: Voronoi in soil space, its input warped so the cracks wander
    wx_ = Xs + 5.0 * (NZ.fbm2(Xs / 14.0, Ds / 14.0, seed + 41, 3) - 0.5)
    wz_ = Ds + 5.0 * (NZ.fbm2(Xs / 14.0, Ds / 14.0, seed + 43, 3) - 0.5)
    F1, F2, ID, PX, PZ = _voronoi(wx_, wz_, 34.0, seed + 11, 1.0)
    edge = (F2 - F1) / 2.0
    hx, hy = heart
    Xh, Dh = soil_world(np.array(hx), np.array(hy))
    rad = np.sqrt((Xs - Xh) ** 2 + (Ds - Dh) ** 2)
    ang = np.arctan2(Ds - Dh, Xs - Xh)
    # the crust: dark and granular — crumbs catch a little light on top, shadow beneath
    tone = NZ.rand(ID, 7, 3)
    ox, oz = wx_ - PX, wz_ - PZ
    on = np.sqrt(ox * ox + oz * oz) + 1e-6
    face = (ox * 0.62 + oz * 0.78) / on
    curl = np.clip(1 - edge / 5.0, 0, 1)
    v = 0.50 + 0.10 * (tone - 0.5) + 0.12 * curl * face
    grain = NZ.fbm2(Xs / 2.2, Ds / 2.2, seed + 51, 2)
    v = v + 0.22 * (grain - 0.5)
    alb = np.where(soil, quantize(v, ['ochre0', 'ochre1', 'ochre2'], cuts=[0.40, 0.62]), alb)
    alb = clean(alb, soil, 2)
    crumb = soil & (grain > 0.66)
    alb = np.where(soil & ~crumb & shifted(crumb, 0, 1) & (grain < 0.55), ci('ochre0'), alb)
    ntilt = np.stack([0.18 * curl * (ox / on), np.zeros_like(Xs), 0.10 * curl * (oz / on)], -1)
    nrm = np.where(soil[..., None], _nz3(UP3[None, None, :] + ntilt), nrm)
    # the disturbed ground where the robot dug and patted (S7a): rougher, crumbly
    Xsp, Dsp = soil_world(np.array(sprout[0]), np.array(sprout[1]))
    dug = soil & (np.sqrt(((Xs - Xsp) / 15.0) ** 2 + ((Ds - Dsp) / 10.0) ** 2) < 1.0)
    crumbly = dug & (NZ.fbm2(Xs / 2.6, Ds / 2.6, seed + 21, 2) > 0.60)
    alb = np.where(crumbly, ci('ochre2'), alb)
    alb = np.where(dug & ~crumbly & shifted(crumbly, 0, 1), ci('ochre0'), alb)
    # cracks: the network + fractures radiating from the burial spot (they will carry the light)
    net = soil & (edge < 0.55 + 0.55 * NZ.rand(ID, 3, 9) + 0.25 * NZ.fbm2(Xs / 6.0, Ds / 6.0, seed + 61, 2)) & ~dug
    radial = np.zeros((HH, WW), bool)
    for k, (a0, L_) in enumerate(((0.55, 17.0), (2.60, 15.0), (4.45, 13.0))):
        wob = 0.55 * (NZ.vnoise1(rad / 6.0, k + 3) - 0.5) + 0.25 * (NZ.vnoise1(rad / 2.5, k + 9) - 0.5)
        da = np.angle(np.exp(1j * (ang - a0 - wob)))
        w_ = 0.40 + 0.40 * np.clip(1 - rad / L_, 0, 1)
        radial |= soil & (np.abs(da) * rad < w_) & (rad < L_) & (rad > 4.0)
    cracks = net | radial
    alb = np.where(cracks, ci('ochre0'), alb)
    deep = cracks & ~shifted(~cracks, 0, -1)              # the crack's shadowed wall under its lip
    lipm = soil & ~cracks & shifted(cracks, 0, -1)        # the near plate's lip over a crack catches light
    alb = np.where(lipm, ci('ochre2'), alb)
    # ---- the near coping: worn stone slabs (top, rounded inner lip) and their front face
    CT = 15.0                                                  # coping top depth on screen
    ctop = (ys >= ynear) & (ys < ynear + CT)
    cface = ys >= ynear + CT
    joint = np.abs(((xs - 0.30 * (ys - ynear) + 23.0) % 71.0) - 35.5) > 34.9
    alb = np.where(ctop, ci('dust3'), alb)
    alb = np.where(ctop & (ys < ynear + 3.0), ci('dust4'), alb)                 # the rounded lip
    alb = np.where(ctop & (NZ.fbm2(xs / 5.0, ys / 2.5, seed + 31, 2) > 0.64), ci('dust2'), alb)
    alb = np.where(ctop & (NZ.fbm2(xs / 3.0, ys / 2.0, seed + 33, 2) > 0.70), ci('dust4'), alb)
    alb = np.where(ctop & joint, ci('dust1'), alb)
    alb = np.where(ctop & (ys >= ynear + CT - 1.5), ci('dust2'), alb)           # the arris
    alb = np.where(cface, ci('bark3'), alb)                                      # old brick below
    brick = (np.abs(((ys - ynear - CT) % 7.0) - 3.5) > 3.0) | \
        (np.abs(((xs + 13.0 * (np.floor((ys - ynear - CT) / 7.0) % 2)) % 26.0) - 13.0) > 12.4)
    alb = np.where(cface & brick, ci('bark1'), alb)
    nrm = np.where(ctop[..., None], UP3, nrm)
    nrm = np.where((ctop & (ys < ynear + 3.0))[..., None], _nz3([0.0, -0.4, 0.92]), nrm)
    nrm = np.where(cface[..., None], _nz3([0.0, SIN35, COS35]), nrm)
    return dict(alb=alb, nrm=nrm, soil=soil, bg=bg, ftop=ftop, fface=fface, cracks=cracks, deep=deep,
                ynear=ynear, yfar=yfar, Xs=Xs, Ds=Ds, rad=rad, dug=dug, ctop=ctop, cface=cface)


# ------------------------------------------------------------------ the heart (c03's orb, this scale)
def c08_orb(r=9.0, power=1.0):
    """The glowing amber orb exactly as c03 draws it (lit from within, a hot
    filament, a darker glass rim, the white specular crescent upper left),
    at radius r. Returns (idx, mask, R) centred at (R, R)."""
    R = int(r + 2)
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float64)
    d = np.sqrt(xx ** 2 + (yy * 1.02) ** 2)
    m = d <= r + 0.35
    dc = np.sqrt((xx + 1.0) ** 2 + (yy + 0.7) ** 2)
    v = (1 - dc / r) * 0.8 + 0.22 * (power - 1.0) + 0.06
    col = quantize(v, ['core1', 'core2', 'core3', 'core4', 'core5', 'core6', 'core7'],
                   cuts=[-0.12, 0.02, 0.18, 0.36, 0.56, 0.76])
    img = np.where(m, col, 0).astype(np.uint8)
    img = clean(img, m, 1)
    rim = m & (d > r - 1.1)
    img = np.where(rim, ci('core2' if power > 0.7 else 'core1'), img)
    img = np.where(rim & (xx + yy > r * 0.9), ci('core3' if power > 0.7 else 'core2'), img)
    sp = m & (np.sqrt((xx + r * 0.42) ** 2 + (yy + r * 0.42) ** 2) < r * 0.36) & \
        (np.sqrt((xx + r * 0.25) ** 2 + (yy + r * 0.25) ** 2) > r * 0.32)
    img = np.where(sp, ci('white1' if power > 0.6 else 'core6'), img)
    for j, row in enumerate(['#.#', '.#.']):                         # the filament
        for i, ch in enumerate(row):
            if ch == '#':
                img[R - 1 + j, R - 1 + i] = ci('white1' if power > 0.5 else 'core7')
    return img, m, R


def c08_cradle(img, ox, oy, r, hot=True, clip=None):
    """c03's copper cradle wires over the orb's lower front: dark copper, hot glints.
    clip(x, y) -> bool: skip pixels (e.g. below the soil)."""
    dark, hi = ci('rust1' if hot else 'rust0'), ci('core4' if hot else 'rust3')
    # one latitude band across the lower front, two meridians rising from the bottom:
    # solid dark copper, a single hot glint where each catches the light
    lines = []
    lat, tilt = 0.46, 0.30
    cy_ = oy + r * lat
    ax_ = r * math.sqrt(max(1 - lat * lat, 0)) + 0.4
    lines.append([(ox + ax_ * math.cos(a), cy_ + ax_ * tilt * math.sin(a)) for a in np.linspace(0.1, math.pi - 0.1, 24)])
    for k, pts in enumerate(lines):
        px_ = curve_px(pts)
        for i, (x, y) in enumerate(px_):
            if 0 <= x < WW and 0 <= y < HH and not (clip and clip(x, y)):
                img[y, x] = dark
        g = px_[len(px_) // 3] if k == 0 else px_[len(px_) // 2]
        if 0 <= g[0] < WW and 0 <= g[1] < HH and not (clip and clip(g[0], g[1])):
            img[g[1], g[0]] = hi
    return img



# ------------------------------------------------------------------ the arms (thin steel, c02/c06 design)
def c08_arm(shoulder, elbow, wrist, r_up=4.6, r_fore=4.0, mint=True):
    """upper arm + elbow joint + forearm as lit-able albedo/normals/mask (full frame,
    computed inside the arm's on-screen box). Steel tubes; a mint band on the upper
    arm (the robot's paint), a seam near the wrist."""
    pts_all = np.array([shoulder, elbow, wrist], float)
    pad = int(math.ceil(max(r_up, r_fore))) + 3
    bx0 = int(max(math.floor(pts_all[:, 0].min()) - pad, 0))
    bx1 = int(min(math.ceil(pts_all[:, 0].max()) + pad + 1, WW))
    by0 = int(max(math.floor(pts_all[:, 1].min()) - pad, 0))
    by1 = int(min(math.ceil(pts_all[:, 1].max()) + pad + 1, HH))
    col_f = np.zeros((HH, WW), np.uint8)
    m_f = np.zeros((HH, WW), bool)
    n_f = np.zeros((HH, WW, 3))
    if bx0 >= bx1 or by0 >= by1:
        return col_f, m_f, n_f
    h, w = by1 - by0, bx1 - bx0
    sh = (shoulder[0] - bx0, shoulder[1] - by0)
    el = (elbow[0] - bx0, elbow[1] - by0)
    wr = (wrist[0] - bx0, wrist[1] - by0)
    Y, X = np.mgrid[0:h, 0:w].astype(np.float64)
    col = np.zeros((h, w), np.uint8)
    m = np.zeros((h, w), bool)
    nrm = np.zeros((h, w, 3))
    for k, (a_, b_, r0, r1) in enumerate(((sh, el, r_up, r_up * 0.95), (el, wr, r_fore, r_fore * 0.9))):
        mm, al, ac, d = thick_curve(h, w, [a_, b_], r0, r1)
        mm &= ~m
        dx, dy = b_[0] - a_[0], b_[1] - a_[1]
        L_ = math.hypot(dx, dy)
        px_, py_ = -dy / L_, dx / L_
        if py_ > 0:                                       # 'across' +1 = the underside
            px_, py_, ac = -px_, -py_, -ac
        sel = mm
        acs = ac[sel]
        nn = _nz3(np.stack([px_ * acs, py_ * acs, np.sqrt(np.clip(1 - acs ** 2, 0, 1)) + 0.15], -1))
        c = np.where(acs < -0.25, ci('steel3'), ci('steel2')).astype(np.uint8)
        als = al[sel]
        if k == 0 and mint:
            band = (als > 0.45) & (als < 0.80)
            c = np.where(band, np.where(acs < -0.2, ci('paint4'), ci('paint3')), c)
            c = np.where((np.abs(als - 0.45) < 0.03) | (np.abs(als - 0.80) < 0.03), ci('paint1'), c)
        if k == 1:
            c = np.where(np.abs(als - 0.86) < 0.025, ci('steel1'), c)      # the seam near the wrist
        col[sel] = c
        nrm[sel] = nn
        m |= mm
    ej = disc(h, w, el[0], el[1], r_up * 1.05, 0.2)
    col[ej] = ci('steel2')
    col[ej & disc(h, w, el[0] - 1.2, el[1] - 1.4, r_up * 0.45, 0.1)] = ci('steel4')
    nrm[ej] = _nz3(np.stack([(X[ej] - el[0]) / r_up, (Y[ej] - el[1]) / r_up, np.full(int(ej.sum()), 0.9)], -1))
    m |= ej
    ol = edge_out(m)
    col[ol] = ci('steel0')
    nrm[ol] = np.array([0.0, 0.0, 1.0])
    m |= ol
    col_f[by0:by1, bx0:bx1] = col
    m_f[by0:by1, bx0:bx1] = m
    n_f[by0:by1, bx0:bx1] = nrm
    return col_f, m_f, n_f


# ------------------------------------------------------------------ the tiny planted sprout (c02 / c06 design)
C08_SPROUT_PALS = {
    # stem (lit, shade), right leaf pal / edge, left leaf pal / edge
    'dawn': (('leaf5', 'leaf3'), ('leaf2', 'leaf3', 'leaf4', 'leaf5', 'leaf6'), ('leaf1', 'leaf6', 'leaf3', 'leaf7'),
             ('leaf1', 'leaf2', 'leaf3', 'leaf4', 'leaf5'), ('leaf1', 'leaf5', 'leaf2', 'leaf6')),
    'warm': (('leaf6', 'leaf4'), ('leaf3', 'leaf4', 'leaf5', 'leaf6', 'leaf7'), ('meadow2', 'leaf7', 'leaf4', 'leaf8'),
             ('leaf2', 'leaf3', 'leaf4', 'leaf5', 'leaf6'), ('meadow2', 'leaf6', 'leaf3', 'leaf7')),
    'hot': (('leaf7', 'leaf5'), ('leaf4', 'leaf5', 'leaf6', 'leaf7', 'leaf8'), ('meadow3', 'leaf8', 'leaf5', 'white0'),
            ('leaf3', 'leaf4', 'leaf5', 'leaf6', 'leaf7'), ('meadow3', 'leaf7', 'leaf4', 'leaf8')),
}


def c08_sprout(pal='dawn', lean=0.0, dl=0.0, dr=0.0, H_=35.0):
    """the wilting planted sprout: a pale stem bowing a little to the right under
    its two cotyledons (the left one drooping). lean: extra bow of the tip (px);
    dl / dr: extra leaf angles (rad, + = tip down) for the tremble. Returns
    (idx, mask, (bx, by)) with (bx, by) = the stem base inside the sprite."""
    stem, palR, edgeR, palL, edgeL = C08_SPROUT_PALS[pal]
    S = 72
    idx = np.zeros((S, S), np.uint8)
    m = np.zeros((S, S), bool)
    bx, by = 30.0, 66.0
    pts = [(bx, by), (bx + 0.5, by - H_ * 0.40), (bx + 2.5 + 0.3 * lean, by - H_ * 0.78),
           (bx + 4.0 + lean, by - H_)]
    curve = bezier(*pts, n=28)
    mm, al, ac, d = thick_curve(S, S, curve, 1.8, 1.1)
    idx = np.where(mm, np.where(ac < -0.1, ci(stem[0]), ci(stem[1])), idx)
    m |= mm
    tip = pts[-1]
    for (ang, L, W, drp, pal_, edge) in ((math.radians(-18) + dr, 14.5, 4.6, 0.28, palR, edgeR),
                                         (math.radians(198) - dl, 13.5, 4.2, 0.55, palL, edgeL)):
        li, lm, (R, _), tp = raster_leaf(ang, L, W, drp, 0, pal=pal_, edge=edge)
        x0 = int(round(tip[0])) - R
        y0 = int(round(tip[1])) - R
        h, w = lm.shape
        ya, yb = max(y0, 0), min(y0 + h, S)
        xa, xb = max(x0, 0), min(x0 + w, S)
        sub = idx[ya:yb, xa:xb]
        lmm = lm[ya - y0:yb - y0, xa - x0:xb - x0]
        sub[lmm] = li[ya - y0:yb - y0, xa - x0:xb - x0][lmm]
        m[ya:yb, xa:xb] |= lmm
    # the node where the leaves meet, and the stem's foot in its little mound
    for (dx, dy, c) in ((0, 0, stem[0]), (0, -1, stem[1])):
        xi, yi = int(round(tip[0])) + dx, int(round(tip[1])) + dy
        idx[yi, xi] = ci(c)
        m[yi, xi] = True
    return idx, m, (int(bx), int(by))


def crack_geodesic(cracks, src, maxd=80.0):
    """distance (px) along the crack network from the source pixels (8-connected
    walk; diagonal steps cost 1.41). inf off the network or beyond maxd."""
    import heapq
    h, w = cracks.shape
    dist = np.full((h, w), np.inf)
    heap = []
    for (y, x) in zip(*np.nonzero(src & cracks)):
        dist[y, x] = 0.0
        heap.append((0.0, int(y), int(x)))
    heapq.heapify(heap)
    steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    while heap:
        d0, y, x = heapq.heappop(heap)
        if d0 > dist[y, x] or d0 > maxd:
            continue
        for dy, dx, c in steps:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and cracks[yy, xx]:
                nd = d0 + c
                if nd < dist[yy, xx]:
                    dist[yy, xx] = nd
                    heapq.heappush(heap, (nd, yy, xx))
    return dist
