"""
c03 · THE HEART  (global 40.0–42.1 in the cut; module 0–2.5)

Close-up of the Gardener's chest (damaged variant) in the cold grey dawn: the
round porthole hatch (~70 px) with rivets, seams, chipped mint paint, the core
glowing dimly behind cracked glass.

Lit as a real scene: every surface has an albedo, a normal and a depth. A cool
dawn key from the upper right and the heart itself — a point light with a
body — light it; each pixel is then snapped only to the colours its material
may take (mint turns bronze and gold in the heart's light, steel turns copper,
the frame corners stay cold), with narrow ordered-dither seams between steps.

0.00–0.07  (cut, latch click) the robot's left hand — c02's hand: steel fingers,
           mint back plate — flicks the latch lever up with its index finger
0.14–0.42  the hand withdraws to the lower right
0.30       the door pops ajar (a hairline of warm light)
0.40       (cue hatch_open) spring hinge: the door swings open toward us,
           overshoots, settles (hand-made perspective frames at 15 fps); the
           heart's light fills the cavity, the frame's inner bevel, the door
0.80–1.20  both hands rise into the cavity and cradle the orb; it brightens
1.20       (cue core_out) the heart is lifted out: a FLARE of light — the orb
           blazes, bloom rings, a star glint, rim light on every finger, glints
           on the rivets — then it beats (~1.2 Hz) in INTENSITY (never in size):
           graded warm light across the chest, the corners staying cold
1.55–2.10  hold on the blazing orb in the hands (cut at 2.1)
"""
import math
import numpy as np
from engine import *
from engine import noise as NZ
from shots.lib_closeup import (ci, cis, quantize, clean, ell, disc, edge_in, edge_out, edge_side, shifted,
                               dilate, erode, raycast_superellipsoid, normalize, curve_px, bezier, field,
                               smooth, blit_idx, thick_curve, PAL)
from shots.lib_closeup2 import RampSnap, HandRig, LUM, tone1

DURATION = 2.5
HH, WW = 180, 320
ys, xs = np.mgrid[0:HH, 0:WW].astype(np.float64)
TH4 = bayer(HH, WW, 4)
TH2 = bayer(HH, WW, 2)

# ------------------------------------------------------------------ layout
CX, CY = 160.0, 86.0          # porthole centre
R_DOOR = 35.0                 # hatch door outer radius
R_GLASS = 25.0                # glass window radius
R_HOLE = 33.5                 # chest opening radius (under the door)
R_FRAME = 40.0                # steel port frame (on the chest) outer radius
HINGE_X = CX - R_DOOR - 3.0   # vertical hinge axis
DAWN = normalize([0.45, -0.70, 0.55])      # cold dawn: upper right
RR = np.sqrt((xs - CX) ** 2 + (ys - CY) ** 2)
ANG = np.arctan2(ys - CY, xs - CX)
RX, RY = np.cos(ANG), np.sin(ANG)
ZV = np.zeros((HH, WW))


def _n3(x, y, z):
    v = np.stack(np.broadcast_arrays(x, y, z), -1).astype(np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


# ------------------------------------------------------------------ materials -> hand-picked ramps
F_PAINT, F_STEEL, F_RUST, F_DARK, F_BG, F_DUST = 1, 2, 3, 4, 5, 6
RSNAP = RampSnap({
    F_PAINT: dict(cold='ink1 paint0-6 glass4', warm='ink1 bark1 ochre1-6 core4-7'),
    F_STEEL: dict(cold='ink0 steel0-6 white0', warm='ink0 ink1 bark1 rust1-4 core3-7'),
    F_RUST: dict(cold='ink1 rust0-6', warm='ink1 rust0-4 core3-6'),
    F_DARK: dict(cold='ink0-2 night3 steel1-2', warm='ink0 rust0 core0-5'),
    F_DUST: dict(cold='dust3-9', warm='dust3 ochre3-7 core5-6'),
}, {'paint': F_PAINT, 'steel': F_STEEL, 'rust': F_RUST, 'ochre': F_RUST, 'bark': F_RUST, 'ink': F_DARK,
    'night': F_DARK, 'ash': F_BG, 'dust': F_DUST, 'glass': F_STEEL})
AMB, KEY = 0.15, 0.29            # dawn: ambient sky fill + cool key (front face ~ paint2-3)


# ------------------------------------------------------------------ background (greenhouse, dawn; pre-lit, far)
def _background():
    v = 1.0 - ys / 180.0
    img = quantize(v + 0.05 * np.sin(xs / 40.0), ['ash2', 'ash3', 'ash4'], cuts=[0.45, 0.78], dith=0.06)
    rib = np.abs((xs - 40) - 0.0022 * (ys - 200) ** 2 + 30) < 3.0
    img = np.where(rib, ci('ash1'), img)
    rib2 = np.abs((xs - 300) + 0.002 * (ys - 190) ** 2 - 40) < 2.2
    img = np.where(rib2, ci('ash1'), img)
    return img.astype(np.uint8)


# ------------------------------------------------------------------ the scene buffers
class Scene:
    """albedo (palette idx), normal, depth, ambient occlusion, shininess."""

    def __init__(self):
        self.alb = np.zeros((HH, WW), np.uint8)
        self.nrm = np.zeros((HH, WW, 3))
        self.nrm[..., 2] = 1.0
        self.z = np.full((HH, WW), -250.0)
        self.ao = np.ones((HH, WW))
        self.sh = np.zeros((HH, WW))          # specular weight
        self.wo = np.ones((HH, WW))           # how much of the heart's light reaches it

    def put(self, m, alb, nrm=None, z=None, ao=None, sh=None, wo=None):
        self.alb = np.where(m, alb, self.alb).astype(np.uint8)
        if nrm is not None:
            self.nrm = np.where(m[..., None], nrm, self.nrm)
        if z is not None:
            self.z = np.where(m, z, self.z)
        if ao is not None:
            self.ao = np.where(m, ao, self.ao)
        if sh is not None:
            self.sh = np.where(m, sh, self.sh)
        if wo is not None:
            self.wo = np.where(m, wo, self.wo)

    def copy(self):
        s = Scene.__new__(Scene)
        s.alb, s.nrm, s.z, s.ao, s.sh, s.wo = (self.alb.copy(), self.nrm.copy(), self.z.copy(), self.ao.copy(),
                                               self.sh.copy(), self.wo.copy())
        return s


# ------------------------------------------------------------------ body (frontal rounded box)
F0 = np.array([0.0, 0.0, 1.0])
U0 = np.array([0.0, -1.0, 0.0])
S0 = np.array([-1.0, 0.0, 0.0])
BODY_C = np.array([160.0, 150.0, -150.0])
BODY_A = (150.0, 142.0, 128.0)      # (front, up, side)
BODY_M, BODY_Z, BODY_N, BODY_L = raycast_superellipsoid(HH, WW, BODY_C, F0, U0, S0, BODY_A, n=6.0,
                                                        zrange=(-10, 170), steps=160)
BODY_Z = np.nan_to_num(BODY_Z, nan=-250.0)


def _seam_x(sx_, y):
    return sx_ + 0.0006 * (y - 90) ** 2 * np.sign(sx_ - 160)


RIVETS = []
for _sx in (72, 249):
    for _yy in range(46, 180, 30):
        RIVETS.append((int(round(_seam_x(_sx, _yy))), _yy))
FRAME_BOLTS = [(CX + 37.0 * math.cos(a), CY + 37.0 * math.sin(a))
               for a in np.radians(np.arange(8) * 45.0 + 22.5)]


def _dome(sc, cx, cy, r, alb, z0, region=None, ao_rim=0.72):
    """a domed rivet / bolt head: dome normals, its own albedo, a contact
    shadow ring on the lower-left (away from the dawn)."""
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    m = d <= r + 0.3
    if region is not None:
        m &= region
    dx, dy = (xs - cx) / (r + 0.5), (ys - cy) / (r + 0.5)
    nz = np.sqrt(np.clip(1 - dx * dx - dy * dy, 0, 1)) + 0.2
    sc.put(m, ci(alb), _n3(dx, dy, nz), z0 + 1.0, 1.0, 0.8)
    ring = (d > r + 0.3) & (d <= r + 1.4) & ((xs - cx) + (ys - cy) * 1.3 > -0.5)
    if region is not None:
        ring &= region
    sc.ao = np.where(ring, np.minimum(sc.ao, ao_rim), sc.ao)
    return m


def _body_scene():
    sc = Scene()
    bg = _background()
    sc.put(~BODY_M, bg, _n3(0.0, 0.0, 1.0), -250.0, 1.0, 0.0)
    # mint paint; grime settles low (dithered transition)
    grime = 1.0 - 0.22 * np.clip((ys - 120) / 60.0, 0, 1)
    # the panels are not perfectly flat: each (between the seams) sits at its own
    # slight tilt, with a gentle waviness, so the heart's light rings bend and
    # step at the seams instead of drawing a perfect bullseye
    panel = np.where(xs < _seam_x(78, ys), 0, np.where(xs < _seam_x(243, ys), 1, 2))
    tilt = np.array([[0.05, 0.02], [-0.03, 0.035], [-0.05, -0.02]])[panel]
    wav_x = 0.05 * (NZ.fbm2(xs / 38.0, ys / 30.0, 41, 2) - 0.5)
    wav_y = 0.05 * (NZ.fbm2(xs / 34.0, ys / 40.0, 42, 2) - 0.5)
    bn = BODY_N + np.stack([tilt[..., 0] + wav_x, tilt[..., 1] + wav_y, np.zeros_like(xs)], -1)
    bn = bn / np.linalg.norm(bn, axis=-1, keepdims=True)
    sc.put(BODY_M, ci('paint5'), bn, BODY_Z, grime, 0.12)
    m = BODY_M
    # panel seams (curved verticals left & right of the port): a V-cut groove
    for sx_ in (78, 243):
        cxs = _seam_x(sx_, ys)
        dd = xs - cxs
        g = m & (np.abs(dd) < 0.6) & (ys > 22)
        sc.alb = np.where(g, ci('paint1'), sc.alb)
        lw = m & (dd >= -1.6) & (dd < -0.6) & (ys > 22)          # left wall faces right: catches the dawn
        rw = m & (dd > 0.6) & (dd <= 1.6) & (ys > 22)
        sc.nrm = np.where(lw[..., None], _n3(0.55, 0.0, 0.83), sc.nrm)
        sc.nrm = np.where(rw[..., None], _n3(-0.55, 0.0, 0.83), sc.nrm)
    # collar seam under the neck
    cy_ = 24 + 0.0009 * (xs - 160) ** 2
    g = m & (np.abs(ys - cy_) < 0.6)
    sc.alb = np.where(g, ci('paint1'), sc.alb)
    sc.nrm = np.where((m & (ys - cy_ > 0.6) & (ys - cy_ < 1.7))[..., None], _n3(0.0, -0.55, 0.83), sc.nrm)
    # rivets along the seams
    for (x, y) in RIVETS:
        if 2 <= x < WW - 3 and 2 <= y < HH - 3 and m[y, x]:
            _dome(sc, x, y, 2.2, 'paint5', float(BODY_Z[y, x]), m)
    # storm scratches: a bright gouge with a dark lip below
    def scratch(pts):
        px = curve_px(pts)
        for (x, y) in px:
            if m[y + 1, x]:
                sc.alb[y + 1, x] = ci('paint2')
        for (x, y) in px:
            if m[y, x]:
                sc.alb[y, x] = ci('paint6')
    scratch([(92, 44), (110, 37), (126, 33)])
    scratch([(96, 50), (112, 44), (121, 41)])
    scratch([(99, 55), (107, 52)])
    scratch([(204, 142), (221, 150), (232, 153)])
    scratch([(214, 34), (226, 30)])
    # chipped paint: bare steel islands, the broken paint edge dark above
    for (cx_, cy2, rx_, ry_) in ((118, 128, 6, 3), (216, 64, 4, 2.5), (96, 96, 3, 2)):
        ch = ell(HH, WW, cx_, cy2, rx_, ry_, 0.2) & m
        ch &= ~(np.sqrt((xs - cx_ - rx_ * 0.6) ** 2 + (ys - cy2 + ry_) ** 2) < ry_ * 0.9)
        sc.alb = np.where(ch, ci('steel4'), sc.alb)
        sc.sh = np.where(ch, 0.7, sc.sh)
        sc.alb = np.where(edge_out(ch) & m & (ys < cy2), ci('paint1'), sc.alb)
        sc.alb = np.where(edge_out(ch) & m & (ys >= cy2), ci('paint6'), sc.alb)
    # dent (lower right): the rim tilts
    dd = ((xs - 233) / 10.0) ** 2 + ((ys - 126) / 6.0) ** 2
    rim = m & (dd > 0.5) & (dd <= 1.1)
    sc.nrm = np.where(rim[..., None], _n3(-(xs - 233) / 10.0 * 0.5, -(ys - 126) / 6.0 * 0.5, 0.85), sc.nrm)
    # faded sprout emblem stencil
    emb = ['.##..##', '####.##', '.###.#.', '...##..', '...#...', '...#...', '.#####.']
    for j, row in enumerate(emb):
        for i, ch_ in enumerate(row):
            if ch_ == '#' and (i, j) not in ((1, 1), (5, 6), (2, 6), (3, 3)):
                sc.alb[58 + j, 100 + i] = ci('dust7')
    return sc


def _neck_head(sc):
    """corrugated steel neck at the top centre, the head's bottom rim above."""
    nk = (np.abs(xs - 160) < 20) & (ys < 20) & ~BODY_M
    dx = (xs - 160) / 21.0
    rib = (ys.astype(int) % 4)
    ty = np.where(rib == 0, 0.5, np.where(rib == 1, -0.4, 0.0))
    sc.put(nk, np.where(rib == 0, ci('steel3'), ci('steel4')).astype(np.uint8),
           _n3(dx, ty, np.sqrt(np.clip(1 - dx * dx, 0.05, 1))), -20.0, 0.9, 0.6, 0.5)
    hd = ys < 5 + 0.0016 * (xs - 160) ** 2 - 6 * (np.abs(xs - 160) > 90)
    hd &= np.abs(xs - 160) < 118
    sc.put(hd, ci('paint4'), _n3((xs - 160) / 300.0, 0.8, 0.45), -10.0, 0.8, 0.1, 0.35)
    sc.alb = np.where(hd & edge_side(hd, 0, 1), ci('paint1'), sc.alb)


def _port(sc):
    """steel frame ring on the chest (bevelled), frame bolts, hinge, latch catch."""
    fr = (RR <= R_FRAME + 0.35) & (RR > R_HOLE)
    outer = RR > R_FRAME - 2.6
    inner = RR < R_HOLE + 2.1
    n_out = _n3(RX * 0.75, RY * 0.75, 0.66)
    n_in = _n3(-RX * 0.8, -RY * 0.8, 0.6)
    n = np.where(outer[..., None], n_out, np.where(inner[..., None], n_in, _n3(0.0, 0.0, 1.0)))
    sc.put(fr, ci('steel4'), n, 1.5, 1.0, 1.0)
    sc.alb = np.where(fr & (RR > R_FRAME - 0.6), ci('steel1'), sc.alb)
    sc.alb = np.where(fr & (np.abs(RR - (R_FRAME - 2.6)) < 0.5), ci('steel3'), sc.alb)
    # contact shadow of the frame on the paint (lower left)
    shd = (np.sqrt((xs - CX + 2) ** 2 + (ys - CY - 2) ** 2) <= R_FRAME + 1.6) & ~fr & (RR > R_FRAME)
    sc.ao = np.where(shd & BODY_M, 0.6, sc.ao)
    for (bx, by) in FRAME_BOLTS:
        _dome(sc, bx, by, 1.4, 'steel5', 2.0, fr)
    # hinge (left): two barrel knuckles + pin
    hx = HINGE_X
    for (y0, y1) in ((CY - 13, CY - 3), (CY + 3, CY + 13)):
        kn = (np.abs(xs - hx) <= 3.2) & (ys >= y0) & (ys <= y1)
        dx = (xs - hx) / 3.8
        sc.put(kn, ci('steel4'), _n3(dx, 0.0, np.sqrt(np.clip(1 - dx * dx, 0.05, 1))), 4.0, 1.0, 1.0)
        sc.alb = np.where(kn & ((ys == y0) | (ys == y1)), ci('steel2'), sc.alb)
        sc.ao = np.where(BODY_M & (np.abs(xs - hx + 4) <= 1) & (ys >= y0 + 1) & (ys <= y1 + 1), 0.6, sc.ao)
    pin = (np.abs(xs - hx) <= 0.5) & (np.abs(ys - CY) <= 15) & ((ys < CY - 13) | (ys > CY + 13))
    sc.put(pin, ci('steel5'), _n3(0.0, 0.0, 1.0), 4.0, 1.0, 1.0)
    # latch catch on the frame (right)
    lc = (xs >= CX + R_FRAME - 3) & (xs <= CX + R_FRAME + 4) & (np.abs(ys - CY) <= 4)
    ty = np.clip((ys - CY) / 4.0, -1, 1)
    sc.put(lc, ci('steel3'), _n3(0.0, ty * 0.6, 0.8), 3.0, 1.0, 0.9)
    sc.alb = np.where(lc & edge_in(lc), ci('steel1'), sc.alb)
    return fr


def _rust(sc):
    """rust only where water sits: the foot of the seams, rivet feet, the
    frame's lower lip and the hinge knuckles' bottoms (never streaks)."""
    def fleck(x, y, c):
        x, y = int(x), int(y)
        if 0 <= x < WW and 0 <= y < HH and BODY_M[y, x]:
            sc.alb[y, x] = ci(c)
    for i, (x, y) in enumerate(RIVETS):
        if y > 60 and NZ.rand(i, 1, 33) > 0.3:
            for k, (dx, dy) in enumerate(((1, 3), (0, 3), (2, 2), (-1, 3), (3, 1))):
                v = float(NZ.rand(i, k, 34))
                if v > 0.25:
                    fleck(x + dx, y + dy, 'rust2' if v > 0.6 else 'rust1')
    for sx_ in (78, 243):
        for y in range(120, 180):
            x = int(round(_seam_x(sx_, y)))
            v = float(NZ.vnoise1(np.array([y / 4.0]), sx_)[0])
            if v > 0.5:
                fleck(x, y, 'rust2' if v > 0.72 else 'rust1')
    # the frame's lower lip holds water: rust flecks along its outer edge (bottom arc)
    lip = (np.abs(RR - (R_FRAME + 0.9)) < 0.7) & (np.sin(ANG) > 0.55) & BODY_M
    ys_, xs_ = np.nonzero(lip)
    for x, y in zip(xs_, ys_):
        v = float(NZ.vnoise2(np.array(x / 3.0), np.array(y / 3.0), 36))
        if v > 0.48:
            sc.alb[y, x] = ci('rust2' if v > 0.7 else 'rust1')
    for (x, y) in ((HINGE_X, CY - 2), (HINGE_X, CY + 14)):
        for dx in (-3, -2, -1, 0, 1, 2):
            if NZ.rand(int(x) + dx, int(y), 37) > 0.3:
                fleck(x + dx, y, 'rust1')
    # the body's lower edge (where the chest curves under)
    be = BODY_M & ~shifted(BODY_M, 0, -2) & (ys > 150) & (ys < HH - 2)
    ys_, xs_ = np.nonzero(be)
    for x, y in zip(xs_, ys_):
        v = float(NZ.vnoise2(np.array(x / 4.0), np.array(y / 4.0), 38))
        if v > 0.5:
            sc.alb[y, x] = ci('rust2' if v > 0.7 else 'rust1')


STATIC = _body_scene()
_neck_head(STATIC)
FRAME_M = _port(STATIC)
_rust(STATIC)
BG_IDX = _background()

# ------------------------------------------------------------------ the closed door
GLASS_M = RR <= R_GLASS + 0.35
DOOR_M = RR <= R_DOOR + 0.35
DOOR_RIVETS = [(CX + 30.2 * math.cos(a), CY + 30.2 * math.sin(a)) for a in np.arange(10) * 2 * math.pi / 10 + 0.3]


def _door_closed(sc):
    ring = DOOR_M & ~GLASS_M
    outer = RR > R_DOOR - 3
    inner = RR < R_GLASS + 2.5
    n = np.where(outer[..., None], _n3(RX * 0.7, RY * 0.7, 0.72),
                 np.where(inner[..., None], _n3(-RX * 0.7, -RY * 0.7, 0.72), _n3(0.0, 0.0, 1.0)))
    sc.put(ring, ci('steel4'), n, 3.0, 1.0, 1.0)
    sc.alb = np.where(ring & (RR > R_DOOR - 0.6), ci('steel1'), sc.alb)
    for (x, y) in DOOR_RIVETS:
        _dome(sc, x, y, 1.5, 'steel5', 3.5, ring)


def _glass(power):
    """the porthole glass (closed): dark, amber glow of the core behind it,
    cold dawn reflections, grime and the storm crack (emissive)."""
    d = RR
    img = quantize(-d + 3.0 * power, ['night2', 'rust0', 'core0', 'core1', 'core2', 'core3', 'core4'],
                   cuts=[-22, -18, -14.5, -11, -7.5, -4], dith=2.2)
    img = np.where(np.abs(d - 12.5) < 0.6, np.where(img == ci('core4'), ci('core3'), img), img)
    sheen = (np.sqrt((xs - CX - 9) ** 2 + (ys - CY + 9) ** 2) < 17) & (np.sqrt((xs - CX - 3) ** 2 + (ys - CY + 3) ** 2) > 16)
    img = np.where(sheen & GLASS_M & (TH2 < 0.5), ci('glass2'), img)
    arc = (np.abs(d - 22.5) < 0.6) & (np.cos(ANG - 2.3) > 0.55)
    img = np.where(arc, ci('glass2'), img)
    hi = (np.abs(d - 21.0) < 0.6) & (np.cos(ANG + 0.9) > 0.8)
    img = np.where(hi, ci('glass4'), img)
    grime = GLASS_M & (ys > CY + 14) & (TH4 < np.clip((ys - CY - 14) / 10, 0, 0.8))
    img = np.where(grime, ci('dust3'), img)
    crack = [(CX - 18, CY - 13), (CX - 11, CY - 7), (CX - 8, CY - 8), (CX - 2, CY - 1), (CX + 1, CY + 6)]
    for (x, y) in curve_px(crack):
        if GLASS_M[y, x]:
            img[y, x] = ci('glass4')
            if GLASS_M[y + 1, x]:
                img[y + 1, x] = ci('ink1')
    for (x, y) in curve_px([(CX - 8, CY - 8), (CX - 7, CY - 14), (CX - 4, CY - 18)]):
        if GLASS_M[y, x]:
            img[y, x] = ci('glass3')
    img = np.where(GLASS_M & (d > R_GLASS - 1.2) & (np.cos(ANG + 0.95) > -0.2), ci('ink1'), img)
    return img.astype(np.uint8)


# ------------------------------------------------------------------ cavity (behind the door)
HOLE_M = RR <= R_HOLE + 0.35
BACK_C = (CX + 2.0, CY + 3.0)
RB = np.sqrt((xs - BACK_C[0]) ** 2 + (ys - BACK_C[1]) ** 2)


def _cavity(sc):
    back = HOLE_M & (RB <= R_HOLE - 5)
    wall = HOLE_M & ~back
    sc.put(wall, ci('steel2'), _n3(-RX * 0.9, -RY * 0.9, 0.42), -9.0, 0.9, 0.5)
    sc.put(back, ci('ink2'), _n3(0.0, 0.0, 1.0), -18.0, 0.9, 0.2)
    ring = back & (np.abs(RB - 21.5) < 1.6)
    sc.alb = np.where(ring, ci('steel1'), sc.alb)
    for k in range(12):
        a = k * math.pi / 6 + 0.26
        x, y = BACK_C[0] + 21.5 * math.cos(a), BACK_C[1] + 21.5 * math.sin(a)
        _dome(sc, x, y, 1.2, 'steel3', -18.0, back)
    for k in range(4):
        a = k * math.pi / 2 + math.pi / 4
        pts = [(BACK_C[0] + 14 * math.cos(a), BACK_C[1] + 14 * math.sin(a)),
               (BACK_C[0] + 26 * math.cos(a), BACK_C[1] + 26 * math.sin(a))]
        for (x, y) in curve_px(pts):
            if back[y, x]:
                sc.alb[y, x] = ci('steel1')
    lip = HOLE_M & ~erode(HOLE_M, 1)
    sc.put(lip, ci('steel3'), _n3(RX * 0.3, RY * 0.3, 0.95), 0.5, 1.0, 0.8)


# ------------------------------------------------------------------ perspective door frames (albedo + normals)
CAM = np.array([160.0, 90.0, 300.0])
DOOR_T = 3.0


def _door_frame(theta_deg):
    """the hatch door rotated theta about its hinge (opening toward the camera,
    swinging left): albedo, normals, depth, mask + emissive glass pixels."""
    th = math.radians(theta_deg)
    n = np.array([-math.sin(th), 0.0, math.cos(th)])         # outer normal
    u = np.array([math.cos(th), 0.0, math.sin(th)])          # along the door, away from the hinge
    Hp = np.array([HINGE_X, CY, 1.5])
    dirx, diry, dirz = xs - CAM[0], ys - CAM[1], -CAM[2] * np.ones_like(xs)
    dn = dirx * n[0] + diry * n[1] + dirz * n[2]
    a0 = R_DOOR + 3.0

    def slice_(o):
        t = ((Hp - CAM) @ n + o) / np.where(np.abs(dn) < 1e-9, 1e-9, dn)
        Px, Py, Pz = CAM[0] + t * dirx, CAM[1] + t * diry, CAM[2] + t * dirz
        a = (Px - Hp[0]) * u[0] + (Pz - Hp[2]) * u[2]
        b = Py - CY
        return np.sqrt((a - a0) ** 2 + b ** 2), np.arctan2(b, a - a0), Pz

    outer_vis = (CAM - Hp) @ n > 0
    rr_o, an_o, z_o = slice_(+DOOR_T / 2)
    rr_i, an_i, z_i = slice_(-DOOR_T / 2)
    union = np.zeros((HH, WW), bool)
    for o in np.linspace(-DOOR_T / 2, DOOR_T / 2, 7):
        rr_, _, _ = slice_(o)
        union |= rr_ <= R_DOOR + 0.35
    if outer_vis:
        face, rr_f, an_f, zf, nf = rr_o <= R_DOOR + 0.35, rr_o, an_o, z_o, n
    else:
        face, rr_f, an_f, zf, nf = rr_i <= R_DOOR + 0.35, rr_i, an_i, z_i, -n
    rim = union & ~face
    alb = np.zeros((HH, WW), np.uint8)
    nrm = np.zeros((HH, WW, 3))
    ca, sa = np.cos(an_f), np.sin(an_f)
    radial = u[None, None, :] * ca[..., None] + np.array([0.0, 1.0, 0.0])[None, None, :] * sa[..., None]
    alb = np.where(rim, ci('steel3'), alb)
    nrm = np.where(rim[..., None], radial, nrm)
    alb = np.where(rim & edge_in(union), ci('steel1'), alb)
    emit = np.zeros((HH, WW), np.uint8)
    em = np.zeros((HH, WW), bool)
    fn = np.broadcast_to(nf, (HH, WW, 3))
    if outer_vis:
        ring = face & (rr_f > R_GLASS)
        tilt = np.where(rr_f > R_DOOR - 3, 0.7, np.where(rr_f < R_GLASS + 2.5, -0.7, 0.0))
        nn = fn + tilt[..., None] * radial
        nn = nn / np.linalg.norm(nn, axis=-1, keepdims=True)
        alb = np.where(ring, ci('steel4'), alb)
        nrm = np.where(ring[..., None], nn, nrm)
        for k in range(10):
            aa = k * 2 * math.pi / 10 + 0.3
            rv = face & (np.abs(rr_f - 30.2) < 1.3) & (np.abs(((an_f - aa + math.pi) % (2 * math.pi)) - math.pi) < 0.045)
            alb = np.where(rv, ci('steel5'), alb)
        alb = np.where(face & (rr_f > R_DOOR - 0.7), ci('steel1'), alb)
        gl = face & (rr_f <= R_GLASS)
        gv = 0.3 + 0.25 * np.cos(an_f + 0.6)
        g = quantize(gv, ['night1', 'night2', 'glass0', 'glass1'], cuts=[0.22, 0.38, 0.5])
        streak = gl & (np.abs((an_f % (2 * math.pi)) - (4.0 + 0.8 * th)) < 0.18) & (rr_f > 8)
        g = np.where(streak, ci('glass3'), g)
        g = np.where(gl & (np.abs(np.sin(an_f - 2.4)) < 0.05) & (rr_f < 20), ci('glass3'), g)
        emit = np.where(gl, g, emit)
        em |= gl
    else:
        ring = face & (rr_f > R_GLASS)
        alb = np.where(ring, ci('steel3'), alb)
        nrm = np.where(ring[..., None], fn, nrm)
        gk = face & (np.abs(rr_f - (R_DOOR - 3.5)) < 1.6)
        alb = np.where(gk, ci('ink2'), alb)
        for k in range(6):
            aa = k * math.pi / 3
            scw = face & (np.abs(rr_f - 28.5) < 1.1) & (np.abs(((an_f - aa + math.pi) % (2 * math.pi)) - math.pi) < 0.06)
            alb = np.where(scw, ci('steel5'), alb)
        alb = np.where(face & (rr_f > R_DOOR - 0.7), ci('steel1'), alb)
        gl = face & (rr_f <= R_GLASS)
        # through the glass: the mint chest behind the door (lit like the chest), a little dimmer
        alb = np.where(gl, ci('paint4'), alb)
        nrm = np.where(gl[..., None], np.array([0.0, 0.0, 1.0]), nrm)
        # the dirty glass: a soft sheen band and a thin arc (emissive, in front)
        sheen = gl & (np.abs(np.sin(an_f - 2.2)) < 0.22) & (rr_f > 6) & (rr_f < 21) & (TH2 < 0.5)
        arc = gl & (np.abs(rr_f - 20) < 0.6) & (np.cos(an_f) > 0.3)
        emit = np.where(sheen, ci('glass2'), emit)
        emit = np.where(arc, ci('glass3'), emit)
        em |= sheen | arc
        zglass = gl
    z = np.where(face, zf, np.maximum(z_o, z_i))
    if not outer_vis:
        z = np.where(zglass, 0.0, z)
    nrm = np.where((union & ~em)[..., None] & (np.linalg.norm(nrm, axis=-1, keepdims=True) < 0.5), fn, nrm)
    return dict(alb=alb, nrm=nrm, z=z, m=union, emit=emit.astype(np.uint8), em=em)


DOOR_SEQ = [30, 64, 98, 126, 112, 120]
DOOR_FRAMES = {a: _door_frame(a) for a in [5] + DOOR_SEQ}

# ------------------------------------------------------------------ latch lever (on the frame)
LATCH_P = (CX + R_DOOR + 2.0, CY)
LEVER_L = 15.0
LEVER_ANG = (92.0, 44.0, -6.0)                    # closed (down) -> half -> flicked open


def lever_end(k):
    a = math.radians(LEVER_ANG[k])
    return LATCH_P[0] + LEVER_L * math.cos(a), LATCH_P[1] + LEVER_L * math.sin(a)


def _lever(sc, k):
    px, py = LATCH_P
    ex, ey = lever_end(k)
    m, al, ac, d = thick_curve(HH, WW, [(px, py), (ex, ey)], 2.5, 2.1)
    dx, dy = ex - px, ey - py
    L = math.hypot(dx, dy)
    pxn, pyn = -dy / L, dx / L
    sc.put(m, ci('steel4'), _n3(pxn * ac * 0.8, pyn * ac * 0.8, 0.7), 6.0, 1.0, 1.0)
    sc.alb = np.where(edge_in(m) & (al > 0.2), ci('steel2'), sc.alb)
    # the pivot boss
    _dome(sc, px, py, 3.1, 'steel4', 6.5)
    sc.alb[int(py), int(px)] = ci('steel1')


# ------------------------------------------------------------------ the hands (c02's design)
def _finger(hr, pts, radii, z, alb='steel4', cap=True):
    """one of c02's segmented steel fingers through joint points (knuckle ->
    tip): rounded boxes, a 1-px seam at each joint, a round cap on the knuckle."""
    for i in range(len(pts) - 1):
        hr.seg(pts[i], pts[i + 1], radii[i], z + 0.01 * i, alb, bulge=0.95, prof=1.0)
    for i in range(1, len(pts) - 1):
        ang = math.atan2(pts[i + 1][1] - pts[i - 1][1], pts[i + 1][0] - pts[i - 1][0])
        hr.joint_line(pts[i], ang, radii[i] - 0.4)
    if cap:
        hr.flat_cap(pts[0], radii[0] * 0.78)


def _hand_latch(k):
    """the robot's LEFT hand seen from the back, reaching up from the lower right:
    steel forearm, wrist ring, the mint back plate, three steel fingers — the
    index hooked under the lever's tip, middle and ring curled. After the flick
    (k=2) the index relaxes. Built in a local box with the index tip at (TX, TY)."""
    Hd, Wd = 170, 170
    TX, TY = 24.0, 20.0
    a = math.radians((-126.0, -118.0, -112.0)[k])
    ux, uy = math.cos(a), math.sin(a)
    px, py = -uy, ux                                   # +p: the index side (upper right)
    segs = [(12.0, 4.7, 0.0), (9.5 if k < 2 else 7.0, 4.5, (-6.0, -2.0, 24.0)[k]), (7.5 if k < 2 else 4.2, 4.2, (-16.0, -8.0, 70.0)[k])]
    angs = [a + math.radians(sg[2]) for sg in segs]
    x, y = TX, TY
    pts = [(x, y)]
    for (L_, r_, _), an in zip(reversed(segs), reversed(angs)):
        x, y = x - L_ * math.cos(an), y - L_ * math.sin(an)
        pts.append((x, y))
    pts = pts[::-1]                                   # knuckle -> tip
    kx, ky = pts[0]
    W = (kx - ux * 20.0 - px * 5.2, ky - uy * 20.0 - py * 5.2)          # wrist centre
    kc = (W[0] + ux * 20.0, W[1] + uy * 20.0)
    hr = HandRig(Hd, Wd)
    hr.seg((W[0] - ux * 110, W[1] - uy * 110), (W[0] - ux * 2.0, W[1] - uy * 2.0), 6.6, 0.0, 'steel4')
    hr.joint_line((W[0] - ux * 24.0, W[1] - uy * 24.0), a, 6.2, 'steel2')
    hr.seg((W[0] - ux * 4.8, W[1] - uy * 4.8), (W[0] + ux * 3.4, W[1] + uy * 3.4), 9.6, 1.0, 'steel3')
    hr.joint_line((W[0] - ux * 4.0, W[1] - uy * 4.0), a, 9.0, 'steel1')
    hr.joint_line((W[0] + ux * 2.6, W[1] + uy * 2.6), a, 9.0, 'steel1')
    hr.seg((W[0] + ux * 2.6, W[1] + uy * 2.6), kc, 9.0, 2.0, 'steel4')
    hr.seg((W[0] + ux * 4.8, W[1] + uy * 4.8), (W[0] + ux * 17.4, W[1] + uy * 17.4), 7.0, 2.1, 'paint5')
    for (s_, t_) in ((0.28, -3.2), (0.72, 3.0)):
        cx_ = W[0] + ux * (4.8 + 12.6 * s_) + px * t_
        cy_ = W[1] + uy * (4.8 + 12.6 * s_) + py * t_
        hr.alb[int(round(cy_)), int(round(cx_))] = ci('paint6')
        hr.alb[int(round(cy_)) + 1, int(round(cx_)) + 1] = ci('paint2')
    # ring + middle, curled under (short, foreshortened)
    for (off, a1, a2, L1, L2, r1, z0) in ((-5.8, 18.0, 86.0, 8.0, 4.8, 4.0, 3.0), (-0.5, 9.0, 72.0, 9.4, 5.4, 4.3, 3.4)):
        bx, by = kc[0] + px * off, kc[1] + py * off
        an1 = a + math.radians(a1)
        jx, jy = bx + L1 * math.cos(an1), by + L1 * math.sin(an1)
        an2 = a + math.radians(a2)
        _finger(hr, [(bx, by), (jx, jy), (jx + L2 * math.cos(an2), jy + L2 * math.sin(an2))], [r1, r1 * 0.92], z0,
                'steel4')
    _finger(hr, pts, [sg[1] for sg in segs], 4.0, 'steel4')
    out = hr.finish()
    out['tip'] = (TX, TY)
    return out


LATCH_TIPS = [(-3.6, 3.0), (-3.0, 4.0), (-1.5, 5.0)]   # fingertip relative to the lever end
LATCH_HANDS = [_hand_latch(k) for k in range(3)]


def _hands_cup(r):
    """both hands cradling the orb (radius r): wrists low and outside, the mint
    back plates facing out, palms under the orb, three steel fingers each
    wrapping up its side — the index across the front of the orb (a dark
    silhouette with burning rims), the middle on its edge, the ring behind."""
    Hd, Wd = int(8.6 * r), int(8.0 * r)
    ox, oy = Wd / 2.0, 1.6 * r
    hr = HandRig(Hd, Wd)
    for side in (-1, 1):
        def P(x, y):
            return (ox + side * x * r, oy + y * r)

        def A(th, R):                       # around the orb: th 0 = bottom, 90 = its outer side
            t_ = math.radians(th)
            return (ox + side * math.sin(t_) * R * r, oy + math.cos(t_) * R * r)
        hr.seg(P(3.5, 7.2), P(1.26, 1.60), 0.33 * r, 0.3 * r, 'steel4')
        hr.seg(P(1.46, 1.86), P(1.12, 1.42), 0.44 * r, 0.5 * r, 'steel3')
        pa, pb = P(1.16, 1.46), P(0.60, 1.02)
        hr.seg(pa, pb, 0.40 * r, 0.6 * r, 'steel4', plate=1 if side < 0 else -1)
        fingers = [  # (angles, radii from the orb centre, finger radius, z) - back to front
            ((44, 64, 86, 104), (1.40, 1.34, 1.28, 1.22), 0.23, -0.7),
            ((37, 58, 80, 97), (1.16, 1.11, 1.07, 1.04), 0.25, 0.15),
            ((29, 50, 70, 85), (0.93, 0.88, 0.85, 0.84), 0.26, 0.85),
        ]
        for (ths, Rs, fr, zf) in fingers:
            pts = [A(th, R_) for th, R_ in zip(ths, Rs)]
            _finger(hr, pts, [fr * r, fr * r * 0.95, fr * r * 0.9], zf * r, 'steel4')
    out = hr.finish()
    out['o'] = (ox, oy)
    return out


HAND_SIZES = list(range(15, 22))
CUP_HANDS = {r_: _hands_cup(float(r_)) for r_ in HAND_SIZES}

# ------------------------------------------------------------------ the core (orb in its wire cradle)
ORB_R = 15.0
FILAMENT = ['#...#', '.#.#.', '..#..']
_ORB_CACHE = {}


def orb_sprite(power, radius=ORB_R):
    """glowing amber orb; power ~0.2 (dim, inside) .. 2 (blazing). The glass
    body keeps its size — only the light inside it changes."""
    q = int(round(power * 10))
    rq = int(round(radius))
    if (q, rq) in _ORB_CACHE:
        return _ORB_CACHE[(q, rq)]
    pw = q / 10.0
    r = float(rq)
    R = int(r + 2)
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float64)
    d = np.sqrt(xx ** 2 + (yy * 1.02) ** 2)
    m = d <= r + 0.35
    dc = np.sqrt((xx + 1.5) ** 2 + (yy + 1.0) ** 2)
    v = (1 - dc / r) * 0.8 + 0.22 * (pw - 1.0) + 0.06
    col = quantize(v, ['core1', 'core2', 'core3', 'core4', 'core5', 'core6', 'core7'],
                   cuts=[-0.12, 0.02, 0.18, 0.36, 0.56, 0.76])
    if pw > 1.25:
        col = np.where(v > 0.92, ci('white1'), col)
    img = np.where(m, col, 0).astype(np.uint8)
    img = clean(img, m, 1)
    rim = m & (d > r - 1.2)
    img = np.where(rim, ci('core3' if pw > 1.1 else ('core2' if pw > 0.7 else 'core1')), img)
    img = np.where(rim & (xx + yy > r * 0.9), ci('core4' if pw > 1.1 else ('core3' if pw > 0.7 else 'core2')), img)
    sp = m & (np.sqrt((xx + r * 0.42) ** 2 + (yy + r * 0.42) ** 2) < r * 0.34) & \
        (np.sqrt((xx + r * 0.26) ** 2 + (yy + r * 0.26) ** 2) > r * 0.30)
    img = np.where(sp, ci('white1' if pw > 0.6 else 'core6'), img)
    fil = FILAMENT if r < 18 else ['#.....#', '.#...#.', '..#.#..', '...#...']
    for j, row in enumerate(fil):
        for i, ch in enumerate(row):
            if ch == '#':
                img[R - 2 + j, R - len(row) // 2 - 1 + i] = ci('white1' if pw > 0.5 else 'core7')
    _ORB_CACHE[(q, rq)] = (img, m, R)
    return _ORB_CACHE[(q, rq)]


CABLE_ANCHORS = [(CX - 27.0, CY - 12.0), (CX - 20.0, CY + 22.0), (CX + 21.0, CY + 21.0), (CX + 27.0, CY - 10.0)]
BACK_ANCHORS = [(CX - 15.0, CY - 9.0), (CX + 13.0, CY - 8.0)]


def _cables(img, ox, oy, r, hot, lifted, hide=None):
    """the core's cables. Inside the chest: four short spokes to the cavity wall.
    Lifted out: two cords from the back of the orb, sagging back into the
    cavity (still plugged in), lit from below by the heart."""
    if not lifted:
        for k, (ax, ay) in enumerate(CABLE_ANCHORS):
            ang = math.atan2(ay - oy, ax - ox)
            px, py = ox + math.cos(ang) * r * 0.55, oy + math.sin(ang) * r * 0.55
            mx, my = (px + ax) / 2, (py + ay) / 2 + 2.0
            pts = bezier((ax, ay), (mx, my), (px, py), n=24)
            m, al, ac, d = thick_curve(HH, WW, pts, 1.0, 1.3)
            if hide is not None:
                m &= ~hide
            img = np.where(m, ci('ink1'), img)
            up = m & (ac * (1 if ax > ox else -1) < -0.25)
            img = np.where(up, ci('rust3' if hot else 'rust1'), img)
        return img
    for k, (ax, ay, rad, sx_, sag) in enumerate(((CX + 5.0, CY - 1.0, 2.6, 0.22, -2.0),
                                                   (CX - 7.0, CY + 1.0, 1.5, -0.26, 2.0))):
        px, py = ox + sx_ * r, oy - r * 0.50
        mx, my = (px + ax) / 2 + sag, (py + ay) / 2 + 3.0
        pts = bezier((ax, ay), (mx, my), (px, py), n=24)
        m, al, ac, d = thick_curve(HH, WW, pts, rad * 0.9, rad)
        if hide is not None:
            m &= ~hide
        img = np.where(m, ci('ink1'), img)
        # lit from below / the orb's side: a hot edge; a faint cold top edge
        lo = m & (edge_side(m, 0, 1) | edge_side(m, -1 if px > ax else 1, 0))
        img = np.where(lo, ci('core3' if hot else 'rust2'), img)
        if rad > 2:
            img = np.where(m & ~lo & shifted(lo, 0, -1), ci('rust2' if hot else 'rust1'), img)
            img = np.where(m & (np.abs(al - 0.5) < 0.03), ci('steel1'), img)       # a clamp band
    return img


def _cradle(img, ox, oy, r, hot, hide=None):
    """copper cradle wires over the orb's lower front: one latitude band and
    two meridians — dark copper, hot glints."""
    dark = ci('rust1' if hot else 'rust0')
    hi = ci('core4' if hot else 'rust3')
    lat, tilt = 0.46, 0.30
    cy_ = oy + r * lat
    ax_ = r * math.sqrt(max(1 - lat * lat, 0)) + 0.5
    lines = [[(ox + ax_ * math.cos(a), cy_ + ax_ * tilt * math.sin(a)) for a in np.linspace(0.08, math.pi - 0.08, 30)]]
    for sgn in (-1, 1):
        lines.append([(ox + sgn * r * 0.55 * math.sin(a), oy + r * math.cos(a)) for a in np.linspace(0.0, 1.2, 16)])
    for k, pts in enumerate(lines):
        for i, (x, y) in enumerate(curve_px(pts)):
            if 0 <= x < WW and 0 <= y < HH and not (hide is not None and hide[y, x]):
                img[y, x] = dark if (i + k) % 5 else hi
    return img


# ------------------------------------------------------------------ precomputed scene variants (speed)
def _scene_closed(k):
    sc = STATIC.copy()
    _door_closed(sc)
    _lever(sc, k)
    return sc


SC_CLOSED = [_scene_closed(k) for k in range(3)]
SC_OPEN = STATIC.copy()
_cavity(SC_OPEN)
_lever(SC_OPEN, 2)
SC_DOOR = {}
for _a, _D in DOOR_FRAMES.items():
    _sc = SC_OPEN.copy()
    _sc.put(_D['m'] & ~_D['em'], _D['alb'], _D['nrm'], _D['z'], 1.0, 0.9)
    SC_DOOR[_a] = _sc
_GLASS_CACHE = {}


def glass_q(pw):
    q = int(round(pw * 20))
    if q not in _GLASS_CACHE:
        _GLASS_CACHE[q] = _glass(q / 20.0)
    return _GLASS_CACHE[q]


# ------------------------------------------------------------------ timeline
T_LATCH1, T_LATCH2 = 0.033, 0.067            # lever flicked open 1-2 frames after the cut (latch click)
T_AJAR, T_OPEN = 0.30, 0.40
T_REACH, T_LIFT, T_HOLD = 0.80, 1.20, 1.55
BEAT_HZ = 1.2


def door_angle(t):
    if t < T_AJAR:
        return 0
    if t < T_OPEN:
        return 5
    k = int((t - T_OPEN) * 15 + 1e-6)             # hand-made frames at 15 fps
    return DOOR_SEQ[min(k, len(DOOR_SEQ) - 1)]


def heartbeat(t):
    """lub-dub (0..1) at ~1.2 Hz, the first beat right at the lift."""
    if t < T_LIFT:
        return 0.0
    ph = ((t - T_LIFT) * BEAT_HZ) % 1.0
    return math.exp(-((ph - 0.03) / 0.06) ** 2) + 0.6 * math.exp(-((ph - 0.25) / 0.07) ** 2)


def flare(t):
    """the lift's flare: 1.0 on the first frame, decaying over ~5 frames."""
    dt = t - T_LIFT
    if dt < 0:
        return 0.0
    k = int(dt * 30 + 1e-6)
    return (1.0, 0.55, 0.30, 0.16, 0.08, 0.03)[k] if k < 6 else 0.0


def orb_state(t):
    """orb centre (x, y, z), radius, emissive power (sprite), light power."""
    if t < T_LIFT:
        flick = float(NZ.vnoise1(np.array([t * 13.0]), 3)[0])
        opened = smooth((t - T_OPEN) / 0.6)
        wake = smooth((t - T_REACH) / (T_LIFT - T_REACH))        # it brightens as the hands close on it
        pw = 0.20 + 0.18 * flick + 0.30 * opened + 0.35 * wake
        return (CX, CY + 1.0, -6.0), ORB_R, pw, 1.2 * pw
    u = min(max((t - T_LIFT) / (T_HOLD - T_LIFT), 0), 1)
    u = 1 - (1 - u) ** 3                            # quick lift, gentle arrival
    oz = 10.0 + 32.0 * u
    r = ORB_R + 6.0 * u
    hb = heartbeat(t)
    f = flare(t)
    pw = 1.15 + 0.35 * hb + 0.8 * f
    lp = 3.4 * (1.0 + 0.50 * hb) * (1.0 + 1.6 * f)
    return (CX, CY + 1.0 + 20.0 * u, oz), r, pw, lp


# ------------------------------------------------------------------ light
D0 = 30.0


def shade(sc, orb, bloom, hands_m=None):
    """light the scene: dawn (ambient + key) + the heart (a point light with a
    body) + its bloom veil, as luminance + warm share -> hand-picked ramps."""
    (ox, oy, oz), orr, _, lp = orb
    nrm, z = sc.nrm, sc.z
    kd = np.clip(nrm[..., 0] * DAWN[0] + nrm[..., 1] * DAWN[1] + nrm[..., 2] * DAWN[2], 0, 1)
    Ec = sc.ao * (AMB + KEY * kd)
    Lx, Ly, Lz = ox - xs, oy - ys, oz - z
    d = np.sqrt(Lx * Lx + Ly * Ly + Lz * Lz) + 1e-6
    nl = (nrm[..., 0] * Lx + nrm[..., 1] * Ly + nrm[..., 2] * Lz) / d
    lam = np.clip((nl + 0.12) / 1.12, 0, 1)
    de = np.maximum(d - 0.55 * orr, 1.0)
    W = sc.wo * lp * lam / (1.0 + (de / D0) ** 2)
    if hands_m is not None:
        W = np.where(hands_m, W * 1.8, W)            # the hands hold it: their rims burn
    # specular glints (steel, rivets): the heart's and the dawn's
    hx, hy, hz = Lx / d, Ly / d, Lz / d + 1.0
    hn = np.sqrt(hx * hx + hy * hy + hz * hz)
    sp = np.clip((nrm[..., 0] * hx + nrm[..., 1] * hy + nrm[..., 2] * hz) / hn, 0, 1) ** 22
    Ew = W + sc.sh * sp * W * 0.9
    spd = np.clip(nrm[..., 0] * HD[0] + nrm[..., 1] * HD[1] + nrm[..., 2] * HD[2], 0, 1) ** 26
    Ec = Ec + sc.sh * spd * 0.45
    ya = LUM[sc.alb]
    V = 0.0
    if bloom > 0:
        d2 = np.sqrt((xs - ox) ** 2 + (ys - oy) ** 2)
        rr = np.maximum(d2, orr * 0.9)
        V = bloom * ((orr / rr) ** 2.6 + 0.12 * (orr / rr) ** 1.1)
        if hands_m is not None:
            V = np.where(hands_m, 0.0, V)
    yc = ya * Ec
    yw = ya * Ew + V
    Y = yc + yw
    wshare = yw / np.maximum(Y, 1e-6)
    L = np.cbrt(tone1(Y))
    fam = RSNAP.family(sc.alb)
    img = RSNAP.snap(fam, L, wshare, band=0.30, wband=0.09, keep=sc.alb)
    # the far greenhouse keeps its own pre-lit colours
    return img, W


HD = (DAWN + np.array([0.0, 0.0, 1.0])) / np.linalg.norm(DAWN + np.array([0.0, 0.0, 1.0]))


# ------------------------------------------------------------------ render
def _blit_scene(sc, part, x0, y0, zoff=0.0):
    """paste a HandRig result (local box) into the scene at integer (x0, y0)."""
    m = part['m']
    h, w = m.shape
    X0, Y0 = int(round(x0)), int(round(y0))
    xa, ya = max(X0, 0), max(Y0, 0)
    xb, yb = min(X0 + w, WW), min(Y0 + h, HH)
    if xa >= xb or ya >= yb:
        return np.zeros((HH, WW), bool), np.zeros((HH, WW), bool)
    sl_s = (slice(ya - Y0, yb - Y0), slice(xa - X0, xb - X0))
    sl_d = (slice(ya, yb), slice(xa, xb))
    mm = m[sl_s]
    full = np.zeros((HH, WW), bool)
    full[sl_d] = mm
    alb = sc.alb[sl_d]
    alb[mm] = part['alb'][sl_s][mm]
    sep = part['sep'][sl_s]
    alb[sep & mm] = ci('steel1')
    sc.nrm[sl_d][mm] = part['nrm'][sl_s][mm]
    sc.z[sl_d][mm] = part['z'][sl_s][mm] + zoff
    sc.ao[sl_d][mm] = 1.0
    sc.sh[sl_d][mm] = 0.9
    ol = np.zeros((HH, WW), bool)
    ol[sl_d] = part['ol'][sl_s]
    ol &= ~full
    return full, ol


def _outline(img, ol, W, hands_m, hot_col='rust2'):
    """hands' outline: dark steel, but burning amber where the heart's light
    grazes the adjacent edge of the hand (sel-out)."""
    Wh = np.where(hands_m, W, 0.0)
    nb = np.maximum.reduce([np.roll(Wh, 1, 0), np.roll(Wh, -1, 0), np.roll(Wh, 1, 1), np.roll(Wh, -1, 1)])
    img = np.where(ol, ci('steel0'), img)
    img = np.where(ol & (nb > 1.6), ci(hot_col), img)
    img = np.where(ol & (nb > 3.2), ci('core3'), img)
    return img


def render(t):
    t = min(max(t, 0.0), DURATION + 1.0)
    orb = orb_state(t)
    (ox, oy, oz), orr, pw, lp = orb
    da = door_angle(t)
    lifted = t >= T_LIFT
    lk = 0 if t < T_LATCH1 else (1 if t < T_LATCH2 else 2)
    if da <= 5:
        sc = SC_CLOSED[lk].copy()
        emit, em = glass_q(pw), GLASS_M
    else:
        sc = SC_DOOR[da].copy()
        emit, em = DOOR_FRAMES[da]['emit'], DOOR_FRAMES[da]['em']
    # ---- hands
    hands_m = np.zeros((HH, WW), bool)
    hands_ol = np.zeros((HH, WW), bool)
    front_m = np.zeros((HH, WW), bool)
    if t < 0.42:
        part = LATCH_HANDS[lk]
        ex, ey = lever_end(lk)
        dx, dy = LATCH_TIPS[lk]
        out = smooth((t - 0.14) / 0.26)
        tx, ty = part['tip']
        hx = ex + dx - tx + out * 44
        hy = ey + dy - ty + out * 80
        hm, hol = _blit_scene(sc, part, hx, hy, 10.0)
        hands_m |= hm
        hands_ol |= hol
    if t >= T_REACH:
        rkey = min(HAND_SIZES, key=lambda q_: abs(q_ - orr))
        part = CUP_HANDS[rkey]
        rise = 1 - smooth((t - T_REACH) / (T_LIFT - T_REACH))
        hx = ox - part['o'][0]
        hy = oy - part['o'][1] + rise * 80
        hm, hol = _blit_scene(sc, part, hx, hy, oz)
        hands_m |= hm
        hands_ol |= hol
        front_m = hm & (sc.z > oz + 0.3 * orr)
    em = em & ~hands_m
    # ---- light
    bloom = 0.0
    if da > 5 or lifted:
        bloom = (0.10 + 0.25 * pw) if not lifted else (0.55 + 0.30 * heartbeat(t) + 1.4 * flare(t))
    img, W = shade(sc, orb, bloom, hands_m)
    img = np.where(em, emit, img)
    img = _outline(img, hands_ol & ~em, W, hands_m)
    if da == 5:
        # the door pops ajar: a hairline of warm light along its free (right) edge
        gap = (np.abs(RR - R_DOOR - 0.6) < 0.6) & (np.cos(ANG) > 0.45) & ~hands_m
        img = np.where(gap, ci('core4'), img)
        img = np.where(gap & (np.cos(ANG) > 0.8), ci('core6'), img)
    # ---- the orb, its cables and cradle
    if da > 5 or lifted:
        hot = pw > 0.8
        if not lifted:                      # once lifted, the cords run straight back, hidden by the orb
            img = _cables(img, ox, oy, orr, hot, lifted, hide=hands_m)
        orb_i, om, R = orb_sprite(pw, orr)
        ocx, ocy = int(round(ox)), int(round(oy))
        full = np.zeros((HH, WW), bool)
        ya, xa = ocy - R, ocx - R
        full[max(ya, 0):ya + 2 * R + 1, max(xa, 0):xa + 2 * R + 1] = \
            om[max(-ya, 0):, max(-xa, 0):][:HH - max(ya, 0), :WW - max(xa, 0)]
        vis = full & ~front_m
        tmp = img.copy()
        blit_idx(tmp, orb_i, om, xa, ya)
        img = np.where(vis, tmp, img)
        img = _cradle(img, ox, oy, orr, hot, hide=front_m)
    # ---- the flare's star glint: a tapered cross, solid near the heart, breaking into dashes
    f = flare(t)
    if f > 0.25:
        oxi, oyi = int(round(ox)), int(round(oy))
        for (dx_, dy_, L) in ((1, 0, int(18 + 80 * f)), (0, 1, int(8 + 36 * f))):
            for k in range(-L, L + 1):
                a = 1 - abs(k) / L
                x, y = oxi + dx_ * k, oyi + dy_ * k
                if not (0 <= x < WW and 0 <= y < HH):
                    continue
                if a < 0.35 and TH4[y, x] > a * 2.2:
                    continue
                img[y, x] = ci('white1' if a > 0.7 else ('core7' if a > 0.45 else ('core6' if a > 0.2 else 'core5')))
                if a > 0.8:                    # the core of the glint is 3 px thick
                    img[y + dx_, x + dy_] = ci('core7')
                    img[y - dx_, x - dy_] = ci('core7')
    # ---- dust in the cold air; golden motes in the heart's light
    if not lifted:
        x, y, r = field(t, 18, 202, (0, 0, 320, 180), vel=(-2.5, 1.5), jitter=(2, 2), sway=(2.0, 0.3))
        for xi, yi, ri in zip(x, y, r):
            xi, yi = int(xi), int(yi)
            if 0 <= xi < WW and 0 <= yi < HH and not hands_m[yi, xi]:
                img[yi, xi] = ci('ash6' if ri > 0.5 else 'ash5') if W[yi, xi] < 1.0 else ci('core4')
    else:
        x, y, r = field(t, 30, 303, (40, 0, 240, 180), vel=(0.0, -9.0), jitter=(3, 4), sway=(2.0, 0.5))
        for xi, yi, ri in zip(x, y, r):
            xi, yi = int(xi), int(yi)
            if 0 <= xi < WW and 0 <= yi < HH and W[yi, xi] > 0.6 and not hands_m[yi, xi]:
                img[yi, xi] = ci('core6' if (ri > 0.6 or W[yi, xi] > 2.0) else 'core5')
    return PAL[img]
