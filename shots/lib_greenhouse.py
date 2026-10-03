"""
shots/lib_greenhouse.py -- THE GREENHOUSE (S6 NIGHT, S7 THE GIFT, S8a BLOOM INSIDE).
Owner: GREENHOUSE (lead environment & lighting).

One fixed medium framing inside the ruined Victorian palm-house. The set is
BUILT ONCE at import as material/albedo layers plus per-pixel 3D maps
(position, normal) from a tiny ray-caster (floor, brick knee wall, glass drum,
hemispherical dome, round planter), hand-composed props (rubble heap + fallen
I-beam, potting bench + tools, dry dipping tank + tap, seed trays, hanging pots,
dead potted palm, floor shards) and RELIT per shot in ENERGY space with
palette-true stepped LUTs (albedo x light tint x GAIN**level, snapped to the
shot's colour-script ramps):

    albedo -> energy per light (falloff, wrap-Lambert, cones, 3D card/cylinder
    shadows, sky transmission through the broken dome) -> level -> LUT per pixel
    (dithered mix between light tints) -> emissive layers (sky through the panes,
    glows, veins, blossoms) on top.

Camera: pinhole, horizontal, horizon y=96, camera 2.2 m high, F=208 px
(1 m = 26 px at Z = 8 m, the robot's depth). All public screen coordinates are
FRAME coordinates (0..320, 0..180); internally the set has an overscan margin M
on every side (S8a shake); crop() cuts the frame out.

STAGING (frame px) -- for the close-up / exterior artists
  S6 sit (feet)        (92, 158)   limp starts at x=88 (rig travel, zero slide)
  S6 can base          (105, 158)  = sit feet + CAN_SET_OFFSET; sprout at its mouth
  rubble heap          x -6..101, y 120..163; fallen iron rib from (-10,82) to (88,156)
  S7/S8 kneel (feet)   (201, 153)  depth 7.96 m; planter on its right
  planter              bbox (199,133)-(268,158); soil ellipse centre (232, 139.5)
  sprout (planted)     (216, 142)  = kneel + PLANT_OFFSET (hand-off, S8a); S7 draws it
                                   5 px right, at (221, 142), clear of the kneeling
                                   robot's head (s07_gift.SPROUT_NUDGE)
  heart in the soil    (212, 142)  = kneel + CORE_PLACE_OFFSET; resting hand (211,144)
  dig can base         (193, 153)  = kneel + DIG_CAN_OFFSET (robot_acts.dig_can)
  tree trunk (S8a)     base (217,143) twisting up to (232,2) and out of frame toward
                       the dome apex (x=232, far above the frame); ~24 px wide at the
                       base, ~15 px at the top edge; canopy fills x~110..320, y<75.
  light directions     moon + dawn from the upper right (behind the dome);
                       the heart's gold from the planter; veins from below.
"""
import math
import numpy as np

from engine import W, H, C, RAMPS, canvas, layer, paste, bayer, dither_mask
from engine import draw, light, noise, ease, fx
from engine import palette as P
from engine.core import to_index, from_index
from engine.sprite import Sprite, blit, anchor_pos, from_ascii, flip

# =============================================================== camera
M = 8                                   # overscan margin (px) on every side
SW, SH = W + 2 * M, H + 2 * M
CX, HZ = 160.0, 96.0                    # principal point x / horizon y (frame coords)
F = 208.0                               # focal length (px)
CH = 2.2                                # camera height (m)

# dome / drum / knee wall (a Victorian palm-house on a low brick wall)
DXc, DZc = 2.98, 8.6                    # dome axis on the floor (m)
R = 10.0                                # dome & drum radius
KW = 1.0                                # brick knee-wall height
VW = 1.9                                # vertical glass drum height
SY = KW + VW                            # springing line of the dome

# round raised planter bed, centred under the dome apex
XP, ZP = DXc, DZc
RP = 1.36                               # outer radius (stone wall)
RS = 1.16                               # soil radius (inside the coping)
HP = 0.44                               # coping top height
HS = 0.40                               # soil surface height (11 px above floor at Z=8)

# ---------------------------------------------------------------- staging (frame coords)
# S6: the robot limps in and sits against the rubble on the left.
S6_SIT = (92, 158)                      # sitting feet
S6_WALK_FROM = 85.0                     # x at t=0 (walk_tired ~6 px/s -> arrives 1.2 s)
# S7/S8: kneeling at the planter's left side (planter on its right).
KNEEL = (201, 153)                      # kneeling feet
PLANTER_C = None                        # filled below: screen centre of the soil ellipse


def proj(X, Y, Z):
    """World (m) -> frame coords (float)."""
    return CX + F * X / Z, HZ - F * (Y - CH) / Z


def floor_Z(y):
    """Depth of the floor seen at frame row y."""
    return F * CH / max(y - HZ, 1e-3)


def unproj_floor(x, y):
    Z = floor_Z(y)
    return (x - CX) * Z / F, Z


def S(x, y):
    """frame -> set-canvas coords."""
    return int(round(x)) + M, int(round(y)) + M


# ================================================================ ray caster
ys_, xs_ = np.mgrid[0:SH, 0:SW].astype(np.float64)
_dx = (xs_ + 0.5 - M - CX) / F
_dy = -(ys_ + 0.5 - M - HZ) / F

SID_EXT, SID_FLOOR, SID_KNEE, SID_DRUM, SID_DOME, SID_PSIDE, SID_PRIM, SID_SOIL = range(8)
INF = 1e9


def _raycast():
    t = np.full((SH, SW), INF)
    sid = np.zeros((SH, SW), np.int8)
    # drum / knee wall: inside of the cylinder (camera is inside)
    a = _dx * _dx + 1.0
    b = _dx * DXc + DZc
    c = DXc ** 2 + DZc ** 2 - R * R
    tc = (b + np.sqrt(np.maximum(b * b - a * c, 0))) / a
    Yc = CH + tc * _dy
    m = (Yc >= 0) & (Yc < KW)
    t[m] = tc[m]; sid[m] = SID_KNEE
    m = (Yc >= KW) & (Yc < SY)
    t[m] = tc[m]; sid[m] = SID_DRUM
    # dome sphere, centre (DXc, SY, DZc)
    ox, oy, oz = -DXc, CH - SY, -DZc
    A = _dx * _dx + _dy * _dy + 1.0
    B = _dx * ox + _dy * oy + oz
    Cc = ox * ox + oy * oy + oz * oz - R * R
    ts = (-B + np.sqrt(np.maximum(B * B - A * Cc, 0))) / A
    m = Yc >= SY
    t[m] = ts[m]; sid[m] = SID_DOME
    # floor
    with np.errstate(divide='ignore', invalid='ignore'):
        tf = np.where(_dy < 0, CH / -_dy, INF)
    Xf = tf * _dx
    m = (_dy < 0) & ((Xf - DXc) ** 2 + (tf - DZc) ** 2 < R * R) & (tf < t)
    t[m] = tf[m]; sid[m] = SID_FLOOR
    # planter wall (outside of a cylinder -> near root)
    b2 = _dx * XP + ZP
    c2 = XP ** 2 + ZP ** 2 - RP * RP
    d2 = b2 * b2 - a * c2
    tp = (b2 - np.sqrt(np.maximum(d2, 0))) / a
    Yp = CH + tp * _dy
    m = (d2 > 0) & (Yp >= 0) & (Yp <= HP) & (tp < t)
    t[m] = tp[m]; sid[m] = SID_PSIDE
    # coping ring top
    with np.errstate(divide='ignore', invalid='ignore'):
        tt = np.where(_dy < 0, (CH - HP) / -_dy, INF)
    rr = np.sqrt((tt * _dx - XP) ** 2 + (tt - ZP) ** 2)
    m = (_dy < 0) & (rr <= RP) & (rr > RS) & (tt < t)
    t[m] = tt[m]; sid[m] = SID_PRIM
    # soil (slightly lower than the coping)
    with np.errstate(divide='ignore', invalid='ignore'):
        tso = np.where(_dy < 0, (CH - HS) / -_dy, INF)
    rs = np.sqrt((tso * _dx - XP) ** 2 + (tso - ZP) ** 2)
    m = (_dy < 0) & (rs <= RS) & (tso < t + 0.35) & (sid != SID_PRIM)
    m &= ~((sid == SID_PSIDE) & (tp < tso))
    t[m] = tso[m]; sid[m] = SID_SOIL
    PX = t * _dx
    PY = CH + t * _dy
    PZ = t
    # normals
    N = np.zeros((SH, SW, 3))
    N[..., 1] = 1.0                                   # floor, coping, soil: up
    wm = (sid == SID_KNEE) | (sid == SID_DRUM)
    N[..., 0] = np.where(wm, -(PX - DXc) / R, N[..., 0])
    N[..., 2] = np.where(wm, -(PZ - DZc) / R, N[..., 2])
    N[..., 1] = np.where(wm, 0.0, N[..., 1])
    dm = sid == SID_DOME
    for k, (v, c0) in enumerate(((PX, DXc), (PY, SY), (PZ, DZc))):
        N[..., k] = np.where(dm, -(v - c0) / R, N[..., k])
    pm = sid == SID_PSIDE
    N[..., 0] = np.where(pm, (PX - XP) / RP, N[..., 0])
    N[..., 2] = np.where(pm, (PZ - ZP) / RP, N[..., 2])
    N[..., 1] = np.where(pm, 0.0, N[..., 1])
    return sid, np.stack([PX, PY, PZ], -1), N


SID, POS, NRM = _raycast()
PHI = np.arctan2(POS[..., 0] - DXc, POS[..., 2] - DZc)        # azimuth around the dome axis
THETA = np.arcsin(np.clip((POS[..., 1] - SY) / R, -1, 1))       # dome elevation
PPHI = np.arctan2(POS[..., 0] - XP, POS[..., 2] - ZP)          # azimuth around the planter

_sx, _sy = proj(XP, HS, ZP)
PLANTER_C = (_sx, _sy)
SPROUT_XY = (KNEEL[0] + 15, KNEEL[1] - 11)       # PLANT_OFFSET hand-off (frame coords)


# ================================================================ helpers
def col(name):
    return np.array(C(name), np.uint8)


def put(img, mask, name):
    img[mask] = C(name)


def h2(a, b, seed):
    return noise.rand(a, b, seed)


def shift(m, dx, dy):
    """Shift a 2D array by (dx, dy) with zero fill."""
    o = np.zeros_like(m)
    h, w = m.shape[:2]
    xs0, xs1 = max(dx, 0), min(w + dx, w)
    ys0, ys1 = max(dy, 0), min(h + dy, h)
    o[ys0:ys1, xs0:xs1] = m[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return o


def edge_of(m):
    """Boundary pixels of a mask (inside the mask)."""
    inner = m & shift(m, 1, 0) & shift(m, -1, 0) & shift(m, 0, 1) & shift(m, 0, -1)
    return m & ~inner


# ================================================================ exterior (seen through the glass)
FY = ys_ - M                      # frame-y of every set pixel
FX = xs_ - M


def _bands(img, t, cols, band=0.5, n=4, mask=None):
    """Quantise a 0..1 field t into palette bands with ordered-dither seams."""
    k = len(cols)
    lv = np.clip(t, 0, 1) * (k - 1)
    fl = np.floor(lv)
    fr = lv - fl
    fr = np.clip((fr - 0.5) / max(band, 1e-3) + 0.5, 0, 1)
    q = np.clip((fl + (bayer(img.shape[0], img.shape[1], n) < fr)).astype(int), 0, k - 1)
    for i, cn in enumerate(cols):
        m = q == i
        if mask is not None:
            m &= mask
        img[m] = C(cn)
    return q


def _exterior(kind):
    img = np.zeros((SH, SW, 3), np.uint8)
    # sky: vertical gradient + slow dust-cloud banding + a hidden light source
    fx_ = FX / 90.0
    clouds = noise.fbm2(fx_ * 0.9, FY / 14.0, 17, 4)
    if kind == 'night':
        sky_cols = ['night1', 'night2', 'night2', 'night3', 'night3', 'night4']
        tsky = np.clip((FY + 10) / (HZ + 10), 0, 1) ** 1.35
        tsky = tsky + (clouds - 0.5) * 0.30
        # the moon, drowned in dust: a faint glow upper right
        g = np.exp(-(((FX - 286) / 70.0) ** 2 + ((FY - 18) / 42.0) ** 2))
        tsky = tsky + g * 0.55
        _bands(img, tsky, sky_cols, band=0.55)
        land = ['ink0', 'ink1', 'night1', 'night2']
        ruin = 'night2'
        haze = 'night3'
    else:   # 'dawn'
        sky_cols = ['ash3', 'ash4', 'ash5', 'ash5', 'ash6', 'ash7', 'dust8']
        tsky = np.clip((FY + 10) / (HZ + 10), 0, 1) ** 1.2
        tsky = tsky + (clouds - 0.5) * 0.28
        g = np.exp(-(((FX - 300) / 90.0) ** 2 + ((FY - 30) / 60.0) ** 2))
        tsky = tsky + g * 0.45
        _bands(img, tsky, sky_cols, band=0.55)
        land = ['ash2', 'ash3', 'ash4', 'ash5']
        ruin = 'ash5'
        haze = 'ash6'
    # far ruined skyline on the horizon (continuity with S1), low contrast
    hx = np.arange(SW) - M
    sky_line = np.full(SW, HZ + M + 1.0)
    for (x0, w_, h_, kind_) in [(18, 6, 9, 0), (27, 4, 5, 1), (34, 9, 13, 2), (47, 5, 7, 0), (58, 3, 11, 1),
                                (66, 8, 6, 0), (80, 4, 8, 1), (112, 7, 4, 0), (250, 5, 6, 0), (262, 3, 9, 1),
                                (270, 8, 5, 0), (300, 4, 7, 2), (309, 6, 4, 0)]:
        for i in range(w_):
            x = x0 + i
            hh = h_
            if kind_ == 1 and i == w_ - 1:
                hh = h_ - 3                       # snapped top
            if kind_ == 2:
                hh = h_ - (i % 3 == 1) * 2 - (i > w_ // 2) * 3
            if 0 <= x + M < SW:
                sky_line[x + M] = min(sky_line[x + M], HZ + M + 1 - hh)
    ruins_m = ys_ >= sky_line[None, :]
    img[ruins_m] = C(ruin)
    # dune bands of the dead valley below the horizon (the greenhouse is on a hill)
    d1 = HZ + M + 1 + 2.5 * noise.fbm1(hx / 38.0, 5, 3) - 1.0
    d2 = HZ + M + 4 + 4.0 * noise.fbm1(hx / 26.0 + 9, 6, 3)
    d3 = HZ + M + 9 + 6.0 * noise.fbm1(hx / 19.0 + 3, 7, 3)
    for dl, cn in ((d1, land[3]), (d2, land[2]), (d3, land[1])):
        img[ys_ >= dl[None, :]] = C(cn)
    img[ys_ >= (d3 + 9)[None, :]] = C(land[0])
    # horizon haze line (thin, dithered)
    hm = (np.abs(ys_ - (HZ + M + 0.5)) < 1.0) & (bayer(SH, SW) < 0.5) & ~ruins_m
    img[hm] = C(haze)
    # where the pixel shows open sky (clouds can drift there)
    EXT_SKYM[kind] = ~(ruins_m | (ys_ >= d1[None, :]) | hm)
    return img


EXT_SKYM = {}
EXT = {'night': _exterior('night'), 'dawn': _exterior('dawn')}


# ================================================================ glazing
GLASS = (SID == SID_DRUM) | (SID == SID_DOME)
BARW = math.radians(5.0)                   # glazing-bar spacing (one thin bar between ribs)
RIBW = math.radians(10.0)                  # main-rib spacing
ROWH = math.radians(7.0)                   # dome purlin spacing
YT = KW + VW * 0.54                        # drum transom height

IPHI = np.floor(PHI / BARW).astype(np.int64)
_row = np.where(SID == SID_DRUM, np.where(POS[..., 1] < YT, -2, -1), np.floor(THETA / ROWH).astype(np.int64))
IROW = np.where(GLASS, _row, -9)
PANE = np.where(GLASS, (IPHI + 300) * 64 + (IROW + 16), -1)

# fractional position inside the pane: u across (0 left..1 right), v up (0 bottom..1 top)
PU = PHI / BARW - IPHI
_vd = np.where(POS[..., 1] < YT, (POS[..., 1] - KW) / (YT - KW), (POS[..., 1] - YT) / (SY - YT))
PV = np.where(SID == SID_DRUM, _vd, THETA / ROWH - np.floor(THETA / ROWH))


def dome_pt(phi, th):
    return (DXc + R * math.cos(th) * math.sin(phi), SY + R * math.sin(th), DZc + R * math.cos(th) * math.cos(phi))


def drum_pt(phi, y):
    return (DXc + R * math.sin(phi), y, DZc + R * math.cos(phi))


def _pp_path(pts):
    """Clean an ordered pixel path: drop duplicates and L-corners (pixel-perfect)."""
    out = []
    for p in pts:
        if out and out[-1] == p:
            continue
        out.append(p)
    res = []
    for i, p in enumerate(out):
        if 0 < i < len(out) - 1 and res:
            a, b = res[-1], out[i + 1]
            if abs(a[0] - b[0]) == 1 and abs(a[1] - b[1]) == 1 and \
                    (p[0] == a[0] or p[1] == a[1]) and (p[0] == b[0] or p[1] == b[1]):
                continue
        res.append(p)
    return res


def curve_path(world_pts):
    """World points -> clean 8-connected pixel path in SET coords."""
    pix = []
    prev = None
    for (X, Y, Z) in world_pts:
        if Z < 0.4:
            prev = None
            continue
        x, y = proj(X, Y, Z)
        x, y = x + M, y + M
        if not (-40 < x < SW + 40 and -40 < y < SH + 40):
            prev = None
            continue
        if prev is not None:
            xs, ys = draw.line_points(prev[0], prev[1], x, y)
            pix.extend(zip(xs, ys))
        prev = (x, y)
    return _pp_path(pix)


def path_mask(path, mask_into, allow=None):
    for (x, y) in path:
        if 0 <= x < SW and 0 <= y < SH and (allow is None or allow[y, x]):
            mask_into[y, x] = True


BAR = np.zeros((SH, SW), bool)
_DOMEM = SID == SID_DOME
_DRUMM = SID == SID_DRUM
for k in range(-40, 40):
    phi = (2 * k + 1) * BARW                   # odd multiples of 5 deg: thin bars
    path_mask(curve_path([dome_pt(phi, math.radians(a)) for a in np.linspace(0, 80, 500)]), BAR, _DOMEM)
    path_mask(curve_path([drum_pt(phi, y) for y in np.linspace(KW, SY, 60)]), BAR, _DRUMM)
for j in range(1, 12):
    th = j * ROWH
    path_mask(curve_path([dome_pt(math.radians(a), th) for a in np.linspace(-178, 178, 2400)]), BAR, _DOMEM)
TRANSOM = np.zeros((SH, SW), bool)
path_mask(curve_path([drum_pt(math.radians(a), YT) for a in np.linspace(-178, 178, 2400)]), TRANSOM, _DRUMM)
BAR |= TRANSOM

# main ribs: world-width bands every 10 deg (drum pilasters continue up as dome ribs)
_dphi = (PHI + RIBW / 2) % RIBW - RIBW / 2
_cos = np.where(SID == SID_DOME, np.cos(THETA), 1.0)
_dist = np.abs(_dphi) * R * _cos
_fp = POS[..., 2] / F                      # metres per pixel at that depth
RIBW_M = 0.19
RIB = GLASS & (_dist < RIBW_M / 2 + _fp * 0.40)
IRIB = np.round(PHI / RIBW).astype(np.int64)
# ring beam at the springing line + drum sill
RING = GLASS & (np.abs(POS[..., 1] - SY) < 0.07 + _fp * 0.5)
SILL = GLASS & (POS[..., 1] - KW < 0.05 + _fp * 0.6)

# --- the collapse: a hole in the upper-left of the dome where the fallen rib tore out
_hole_c = (math.radians(-62), math.radians(22))
HOLE_D = ((PHI - _hole_c[0]) / math.radians(17)) ** 2 + ((THETA - _hole_c[1]) / math.radians(15)) ** 2
_hn = noise.fbm2(PHI * 6, THETA * 6, 41, 3)
HOLE = GLASS & _DOMEM & (HOLE_D + (_hn - 0.5) * 0.9 < 1.0)
FALLEN_RIB_IDX = int(round(math.radians(-60) / RIBW))     # the rib that fell
_stub_th = math.radians(30)
RIB_MISSING = RIB & (IRIB == FALLEN_RIB_IDX) & (THETA < _stub_th) & _DOMEM
RIB_MISSING |= RIB & (IRIB == FALLEN_RIB_IDX) & _DRUMM & (POS[..., 1] > YT + 0.25)
# bars/purlins torn out inside the hole; twisted stubs survive near its rim
BAR &= ~(HOLE & (HOLE_D + (noise.vnoise2(FX / 3.0, FY / 3.0, 45) - 0.5) * 0.5 < 0.86))

# --- per pane state: 0 intact, 1 broken (open), 2 cracked, 3 half-broken
_pr = noise.rand(IPHI, IROW, 101)
_pr2 = noise.rand(IPHI, IROW, 102)
_near_hole = np.clip(1.8 - HOLE_D, 0, 1)
_pbreak = 0.12 + 0.40 * _near_hole + 0.08 * (FX < 110) + 0.05 * _DOMEM
PSTATE = np.where(_pr < _pbreak, 1, np.where(_pr < _pbreak + 0.10, 3, np.where(_pr2 < 0.16, 2, 0)))
PSTATE = np.where(HOLE, 1, PSTATE)
PSTATE = np.where(GLASS, PSTATE, -1)
# half-broken panes: a slanted break line through the pane (the lower part survives)
_ang = noise.rand(IPHI, IROW, 103)
_jag = noise.vnoise2(FX / 1.7, FY / 1.7, 107)
_half_keep = (PV + (PU - 0.5) * (_ang - 0.5) * 1.6 + (_jag - 0.5) * 0.12) < (0.30 + 0.35 * _pr2)
# jagged shards left in the frame of broken panes
_edge_d = np.minimum(np.minimum(PU, 1 - PU) * 2.0, np.minimum(PV, 1 - PV) * 1.2)
SHARD = GLASS & (PSTATE == 1) & ~HOLE & (_edge_d < 0.02 + 0.20 * _jag * (noise.rand(IPHI, IROW, 109) < 0.7))
SHARD |= GLASS & HOLE & (_edge_d < 0.10 * _jag) & (HOLE_D > 0.75)
OPEN = GLASS & (((PSTATE == 1) & ~SHARD) | ((PSTATE == 3) & ~_half_keep))
# dust caked along the bottom of every pane (a clean band with a dithered edge)
_dr = noise.rand(IPHI, IROW, 111)
_dh = np.where(_dr < 0.45, 0.04, np.where(_dr < 0.85, 0.14, 0.34)) + 0.08 * (noise.vnoise2(FX / 4.0, 0.0, 112) - 0.5)
DUSTG = np.clip((_dh - PV) / 0.10 + 0.5, 0, 1)
DUSTG = np.where(GLASS & ~OPEN, DUSTG, 0.0)
# reflection shine (two short diagonal strokes near the upper-left of some panes)
_sh = noise.rand(IPHI, IROW, 113)
_diag = (FX + FY + np.floor(_sh * 13)).astype(np.int64)
SHINE = GLASS & ~OPEN & (_sh < 0.16) & ((_diag % 19 == 0) | (_diag % 19 == 2)) & (PV > 0.45) & (PU < 0.75) & (DUSTG < 0.5)

FRAME_IRON = GLASS & (BAR | RIB | RING | SILL) & ~RIB_MISSING
FRAME_IRON &= ~(HOLE & ~RIB & ~BAR)
# rib shading classes (lit edge on the right: sky light comes from the upper right)
RIB_HI = RIB & ~shift(RIB, -1, 0)
RIB_LO = RIB & ~shift(RIB, 1, 0) & ~RIB_HI

# cracks in cracked panes: zig-zag lines across the pane
CRACK = np.zeros((SH, SW), bool)
_cracked = np.unique(PANE[(PSTATE == 2) & GLASS])
for pid in _cracked[:400]:
    ip, ir = pid // 64 - 300, pid % 64 - 16
    r1 = float(noise.rand(ip, ir, 121)); r2 = float(noise.rand(ip, ir, 122)); r3 = float(noise.rand(ip, ir, 123))
    pts = []
    for s in np.linspace(0, 1, 7):
        u = 0.1 + 0.8 * (r1 + (r2 - r1) * s) + (float(noise.rand(int(s * 9), ip, 124 + ir)) - 0.5) * 0.18
        v = 0.95 - 0.9 * s
        phi = (ip + u) * BARW
        if ir >= 0:
            pts.append(dome_pt(phi, (ir + v) * ROWH))
        else:
            y0, y1 = (KW, YT) if ir == -2 else (YT, SY)
            pts.append(drum_pt(phi, y0 + (y1 - y0) * v))
    path_mask(curve_path(pts), CRACK, GLASS & (PANE == pid))
CRACK &= ~FRAME_IRON


# ================================================================ albedo: walls, floor, planter
def _boundary_h(idx, mask):
    """1-px lines where an integer field changes vertically (near-horizontal boundaries)."""
    return mask & shift(mask, 0, -1) & (idx != shift(idx, 0, -1))


def _boundary_v(idx, mask):
    """1-px lines where an integer field changes horizontally (near-vertical boundaries)."""
    return mask & shift(mask, -1, 0) & (idx != shift(idx, -1, 0))


def _build_albedo():
    alb = np.zeros((SH, SW, 3), np.uint8)
    PX, PY, PZ = POS[..., 0], POS[..., 1], POS[..., 2]
    th = bayer(SH, SW)

    # ---------------- knee wall: dusty brick courses with a stone coping + heating pipes
    km = SID == SID_KNEE
    course = np.floor(PY / 0.25).astype(np.int64)
    arc = PHI * R
    stag = np.where(course % 2 == 0, 0.0, 0.25)
    brick = np.floor((arc + stag) / 0.5).astype(np.int64)
    bv = noise.rand(brick, course, 201)
    alb[km] = C('dust2')
    alb[km & (bv < 0.45)] = C('ochre1')
    alb[km & (bv > 0.86)] = C('rust1')
    # weathering: the lower courses are caked with dust
    wdust = (noise.fbm2(FX / 7.0, FY / 4.0, 203, 3) + (PY < 0.3) * 0.3) > 0.70
    alb[km & wdust] = C('dust3')
    mort = _boundary_h(course, km) | _boundary_v(brick * 1000 + course, km)
    alb[mort] = C('dust1')
    # coping stone on top of the knee wall
    cop = km & (PY > KW - 0.09 - _fp * 0.5)
    alb[cop] = C('dust5')
    cop_top = cop & ~shift(cop, 0, 1)
    alb[cop_top] = C('dust6')
    alb[km & ~cop & shift(cop, 0, 1)] = C('ink2')          # shadow line under the coping
    # two cast-iron heating pipes along the base of the wall
    for (py0, pw) in ((0.16, 0.07), (0.30, 0.07)):
        pm = km & (np.abs(PY - py0) < pw / 2 + _fp * 0.35)
        alb[pm] = C('steel2')
        alb[pm & ~shift(pm, 0, 1)] = C('steel3')
        alb[pm & ~shift(pm, 0, -1)] = C('steel1')
        rs_ = pm & (noise.fbm2(FX / 3.0, py0 * 50, 205, 2) > 0.66)
        alb[rs_] = C('rust1')

    # ---------------- floor: dusty quarry tiles (checker), grout, cracks, missing tiles, drifts
    fm = SID == SID_FLOOR
    TS = 0.5
    ti = np.floor(PX / TS).astype(np.int64)
    tj = np.floor(PZ / TS).astype(np.int64)
    chk = (ti + tj) % 2 == 0
    tv = noise.rand(ti, tj, 211)
    alb[fm & chk] = C('dust4')
    alb[fm & ~chk] = C('ochre2')
    alb[fm & ~chk & (tv < 0.22)] = C('dust3')
    alb[fm & chk & (tv < 0.18)] = C('dust3')
    near = PZ < 13.5
    grout = (_boundary_h(tj, fm) | _boundary_v(ti, fm)) & near
    alb[grout] = C('dust2')
    # missing / broken tiles: dirt showing through
    miss = fm & (noise.rand(ti, tj, 213) < 0.06)
    alb[miss] = C('dust2')
    alb[miss & (noise.vnoise2(FX / 2.0, FY / 1.5, 214) > 0.6)] = C('ochre1')
    # hairline cracks across some tiles
    tu = PX / TS - ti
    tw = PZ / TS - tj
    ca = noise.rand(ti, tj, 215)
    crk = fm & (ca < 0.26) & (np.abs(tu - (0.2 + 0.6 * noise.rand(ti, tj, 216)) - (tw - 0.5) * (ca * 3 - 0.45)) < 0.03 + _fp * 0.22)
    alb[crk & near] = C('dust2')
    # dust: 2 solid levels (film / drift) with narrow dithered seams -- clusters, not noise
    rr_ = np.sqrt((PX - DXc) ** 2 + (PZ - DZc) ** 2)
    dust = noise.fbm2(PX * 0.9, PZ * 1.9, 217, 4) + 0.40 * np.clip((rr_ - 7.8) / 1.6, 0, 1) \
        + 0.28 * np.clip((-1.2 - PX) / 3.0, 0, 1)
    lv = np.clip((dust - 0.50) * 6.0, 0, 2.0)
    q = np.floor(lv) + (th < np.clip((lv - np.floor(lv) - 0.5) / 0.35 + 0.5, 0, 1))
    film = fm & (q >= 1) & ~grout
    alb[film & chk] = C('dust5')
    alb[film & ~chk] = C('dust4')
    drift = fm & (q >= 2)
    alb[drift] = C('dust5')
    # wind ripples on the drifts
    rip = drift & (np.sin(PX * 8.0 + PZ * 1.6 + noise.vnoise2(PX * 1.5, PZ * 1.5, 219) * 3) > 0.86)
    alb[rip] = C('dust6')

    # ---------------- planter: stone wall, coping ring, dry cracked soil
    pm = SID == SID_PSIDE
    pc = np.floor(PY / (HP / 2)).astype(np.int64)
    pa = PPHI * RP
    pst = np.where(pc % 2 == 0, 0.0, 0.22)
    pb = np.floor((pa + pst) / 0.44).astype(np.int64)
    pv_ = noise.rand(pb, pc, 221)
    alb[pm] = C('dust5')
    alb[pm & (pv_ < 0.30)] = C('dust4')
    alb[pm & (pv_ > 0.82)] = C('dust6')
    alb[_boundary_h(pc, pm) | _boundary_v(pb * 100 + pc, pm)] = C('dust3')
    # grime at the foot of the bed
    alb[pm & (PY < 0.07 + _fp * 0.5)] = C('dust3')
    # chipped block corners
    chip = pm & (noise.vnoise2(FX / 2.5, FY / 2.5, 223) > 0.80)
    alb[chip] = C('dust4')
    # coping ring
    cm = SID == SID_PRIM
    ca_ = np.floor(PPHI / math.radians(24)).astype(np.int64)
    alb[cm] = C('dust6')
    alb[cm & (noise.rand(ca_, 0, 225) < 0.35)] = C('dust5')
    front_edge = cm & ~shift(cm, 0, -1)                   # the rounded front lip catches light
    alb[front_edge] = C('dust7')
    alb[_boundary_v(ca_, cm)] = C('dust4')
    inner_edge = cm & shift(SID == SID_SOIL, 0, 1)
    alb[inner_edge] = C('dust4')
    # soil: dry, cracked into plates (voronoi on the soil plane)
    sm = SID == SID_SOIL
    npts = 46
    k = np.arange(npts)
    ang = noise.rand(k, 1, 231) * 2 * math.pi
    rad = np.sqrt(noise.rand(k, 2, 231)) * RS
    vx = XP + rad * np.cos(ang)
    vz = ZP + rad * np.sin(ang)
    sx, sz = PX[sm], PZ[sm]
    d = (sx[:, None] - vx[None, :]) ** 2 + ((sz[:, None] - vz[None, :]) * 1.0) ** 2
    near_i = np.full((SH, SW), -1, np.int64)
    near_i[sm] = d.argmin(1)
    d.sort(1)
    edge_w = np.zeros((SH, SW))
    edge_w[sm] = np.sqrt(d[:, 1]) - np.sqrt(d[:, 0])
    plate = noise.rand(near_i, 3, 233)
    alb[sm] = C('dust3')
    alb[sm & (plate < 0.35)] = C('dust2')
    alb[sm & (plate > 0.85)] = C('dust4')
    crack_s = (_boundary_h(near_i, sm) | _boundary_v(near_i, sm))
    alb[crack_s] = C('dust1')
    # curled plate edges catch a little light
    alb[sm & ~crack_s & shift(crack_s, 0, 1) & (plate > 0.4)] = C('dust4')
    # the soil sits a little below the coping: a shadow crescent at the back
    back_sh = sm & shift(cm, 0, 1) & ~shift(cm, 0, -1)
    alb[back_sh] = C('dust1')
    return alb, near_i


ALB, SOIL_PLATE = _build_albedo()
# glazing iron: painted wrought iron gone to rust (dark steel, a lit edge, rust flakes)
ALB[FRAME_IRON] = C('steel2')
ALB[FRAME_IRON & RIB_HI] = C('steel3')
ALB[FRAME_IRON & RIB_LO] = C('steel1')
ALB[FRAME_IRON & RING] = C('steel2')
ALB[FRAME_IRON & RING & ~shift(RING, 0, 1)] = C('steel3')
_ir = FRAME_IRON & (noise.fbm2(FX / 5.0, FY / 9.0, 191, 3) > 0.72) & (FY > 70)
ALB[_ir] = C('rust1')
ALB[_ir & RIB_HI] = C('rust3')


# ================================================================ props (z-aware, relightable)
# Every prop pixel writes albedo + a world position (from its depth Z) + a normal,
# so the relighting treats props exactly like the ray-cast architecture.
MAT = np.zeros((SH, SW), np.int8)        # 0 plain, 1 glass shard (sparkles), 2 iron, 3 soil, 4 emissive-ish
MAT[FRAME_IRON] = 2
MAT[SID == SID_SOIL] = 3
PROPM = np.zeros((SH, SW), bool)         # prop coverage (for shadows / occlusion queries)
FGM = np.zeros((SH, SW), bool)           # foreground (drawn over the robot)


def _world_at(xs, ys, Z):
    X = (xs - M + 0.5 - CX) * Z / F
    Y = CH - (ys - M + 0.5 - HZ) * Z / F
    return X, Y


def poly_mask(pts):
    """Scanline polygon fill (even-odd), bbox-limited, pixel centres. pts in SET coords."""
    pts = np.asarray(pts, np.float64)
    m = np.zeros((SH, SW), bool)
    y0 = max(int(math.floor(pts[:, 1].min())), 0)
    y1 = min(int(math.ceil(pts[:, 1].max())) + 1, SH)
    xs = np.arange(SW) + 0.5
    n = len(pts)
    for yi in range(y0, y1):
        yc = yi + 0.5
        xi = []
        for i in range(n):
            xa, ya = pts[i]
            xb, yb = pts[(i + 1) % n]
            if (ya <= yc < yb) or (yb <= yc < ya):
                xi.append(xa + (yc - ya) * (xb - xa) / (yb - ya))
        xi.sort()
        for k in range(0, len(xi) - 1, 2):
            m[yi] |= (xs >= xi[k]) & (xs < xi[k + 1])
    return m


def stamp(mask, color, Z, normal=(0.0, 0.0, -1.0), mat=0, fg=False, alb=None, pos=None, nrm=None):
    """Write prop pixels: albedo, world position at depth Z (scalar or per-pixel), normal."""
    alb = ALB if alb is None else alb
    pos = POS if pos is None else pos
    nrm = NRM if nrm is None else nrm
    if not mask.any():
        return
    yy, xx = np.nonzero(mask)
    Zv = Z[yy, xx] if isinstance(Z, np.ndarray) and Z.ndim == 2 else Z
    X, Y = _world_at(xx, yy, Zv)
    alb[yy, xx] = C(color) if isinstance(color, str) else color
    pos[yy, xx, 0] = X
    pos[yy, xx, 1] = Y
    pos[yy, xx, 2] = Zv
    nv = np.asarray(normal, np.float64)
    if nv.ndim == 1:
        nrm[yy, xx] = nv
    else:
        nrm[yy, xx] = nv[yy, xx]
    MAT[yy, xx] = mat
    PROPM[yy, xx] = True
    if fg:
        FGM[yy, xx] = True


def auto_normals(mask, blur=2.0, face=0.9):
    """Pseudo-3D normals from a silhouette: edges face outward, interior faces camera."""
    from scipy.ndimage import gaussian_filter
    f = gaussian_filter(mask.astype(np.float64), blur)
    gy, gx = np.gradient(f)
    nx = -gx
    ny = gy                       # screen y down -> world y up
    nz = -np.full_like(f, face) * np.clip(f * 1.2, 0.2, 1.0)
    n = np.stack([nx * 3.0, ny * 3.0, nz], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    return n


def SPt(x, y):
    return (x + M, y + M)


def _bezier(p0, p1, p2, n):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = map(np.asarray, (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


# ---------------------------------------------------------------- rubble + fallen rib (left)
RIB_P0, RIB_P1, RIB_P2 = (-10.0, 82.0), (34.0, 101.0), (88.0, 156.5)   # frame coords
RIB_Z0, RIB_Z2 = 9.2, 7.55


def _floor_Z_at(y):
    return F * CH / max(y - HZ, 1.0)


def box(x, y, w, h, d, tilt, cols, Z, brickwork=False, fg=False, seed=0):
    """A little masonry block seen from the front-above: lit top face, mid front
    face, dark right side face; rotated by `tilt` (rad) around its centre.
    (x, y) = front face top-left in FRAME coords. cols = (top, front, side, dark)."""
    cx, cy = x + w / 2.0, y + h / 2.0
    ca, sa = math.cos(tilt), math.sin(tilt)

    def rot(px, py):
        dx, dy = px - cx, py - cy
        return (cx + dx * ca - dy * sa + M, cy + dx * sa + dy * ca + M)
    sk = d * 0.8
    front = [rot(x, y), rot(x + w, y), rot(x + w, y + h), rot(x, y + h)]
    top = [rot(x + sk, y - d), rot(x + w + sk, y - d), rot(x + w, y), rot(x, y)]
    side = [rot(x + w, y), rot(x + w + sk, y - d), rot(x + w + sk, y + h - d), rot(x + w, y + h)]
    mf = poly_mask(front)
    mt = poly_mask(top) & ~mf if d > 0 else np.zeros_like(mf)
    ms = poly_mask(side) & ~mf & ~mt if d > 0 else np.zeros_like(mf)
    stamp(mf, cols[1], Z, (0.0, 0.1, -1.0), fg=fg)
    stamp(mt, cols[0], Z, (-0.1, 1.0, -0.35), fg=fg)
    stamp(ms, cols[2], Z, (1.0, 0.1, -0.35), fg=fg)
    allm = mf | mt | ms
    if brickwork:
        # courses of small bricks on the front face
        yy = (ys_ - (y + M)) * ca - (xs_ - (x + M)) * sa
        xx = (xs_ - (x + M)) * ca + (ys_ - (y + M)) * sa
        crs = np.floor(yy / 3.0).astype(np.int64)
        bk = np.floor((xx + (crs % 2) * 3) / 6.0).astype(np.int64)
        joint = mf & ((_boundary_h(crs, mf)) | _boundary_v(bk * 100 + crs, mf))
        ALB[joint] = C(cols[3])
        bvv = noise.rand(bk, crs, 331 + seed)
        ALB[mf & ~joint & (bvv < 0.3)] = C(cols[2])
    # chipped corner: nibble a random pixel off the top-left
    return allm


def _rubble_and_rib():
    rnd = lambda i, k: float(noise.rand(i, k, 301))
    xs = np.arange(SW)
    fx = xs - M
    # --- mound (crushed brick, sand, earth): silhouette + base contour (frame coords)
    top_pts = [(-4, 126), (4, 125), (12, 122), (20, 124), (30, 121), (40, 119), (48, 122),
               (56, 127), (64, 133), (72, 140), (80, 147), (88, 153), (95, 158), (101, 162)]
    base_pts = [(-4, 152), (10, 157), (30, 161), (55, 163), (80, 162), (101, 162)]
    top = np.interp(fx, [p[0] for p in top_pts], [p[1] for p in top_pts])
    top += (noise.fbm1(fx / 4.0, 303, 3) - 0.5) * 3.0
    base = np.interp(fx, [p[0] for p in base_pts], [p[1] for p in base_pts])
    valid = (fx >= -6) & (fx <= 101)
    mound = (ys_ >= (top + M)[None, :]) & (ys_ <= (base + M)[None, :]) & valid[None, :]
    Zm = np.clip(F * CH / np.maximum(base[None, :] - 4 - HZ, 1), 7.2, 11.0) * np.ones((SH, 1))
    nm = auto_normals(mound, 3.0)
    stamp(mound, 'dust3', Zm, nm)
    th_ = bayer(SH, SW)
    # crushed brick & grit clusters, lighter sand on the upper slopes, dark crevices
    grit = mound & (noise.fbm2(FX / 3.0, FY / 2.5, 305, 3) > 0.56)
    ALB[grit] = C('ochre1')
    sand = mound & (noise.fbm2(FX / 6.0, FY / 3.0, 306, 3) > 0.50) & (nm[..., 1] > 0.05)
    ALB[sand] = C('dust4')
    topm = mound & ~shift(mound, 0, 1)
    ALB[topm] = C('dust5')
    ALB[mound & shift(topm, 0, 1) & (th_ < 0.5)] = C('dust4')
    # --- big pieces (hand-placed), back to front
    B = []
    B.append(box(2, 126, 22, 11, 2, -0.16, ('dust4', 'rust1', 'ink2', 'dust1'), 10.0, brickwork=True, seed=1))
    B.append(box(24, 124, 22, 4, 2, 0.20, ('dust7', 'dust5', 'dust3', 'dust2'), 9.8))       # coping slab
    B.append(box(44, 126, 9, 8, 2, 0.08, ('dust6', 'dust4', 'dust2', 'dust1'), 9.6))        # stone block
    B.append(box(14, 136, 7, 4, 1, 0.3, ('ash6', 'ash4', 'ash2', 'ash1'), 9.4))
    B.append(box(32, 131, 12, 7, 2, -0.10, ('ochre3', 'rust1', 'ink2', 'dust1'), 9.4, brickwork=True, seed=2))
    # broken window frame leaning in the pile, a big shard still in it
    fr = poly_mask([SPt(56, 118), SPt(70, 122), SPt(66, 142), SPt(52, 138)])
    fr_in = poly_mask([SPt(58, 121), SPt(67, 124), SPt(64, 139), SPt(55, 136)])
    frame_m = fr & ~fr_in
    stamp(frame_m, 'steel2', 9.2, (0.2, 0.3, -0.9), mat=2)
    ALB[frame_m & ~shift(fr, 0, 1)] = C('steel4')
    shard = fr_in & poly_mask([SPt(55, 136), SPt(64, 139), SPt(66, 126), SPt(60, 129)])
    stamp(shard, 'glass1', 9.25, (-0.2, 0.4, -0.9), mat=1)
    ALB[edge_of(shard) & ~shift(shard, 1, 0)] = C('glass3')
    ALB[shard & ((FX + FY) % 7 == 0)] = C('glass2')
    B.append(box(50, 139, 8, 5, 2, -0.25, ('dust6', 'dust4', 'dust2', 'dust1'), 8.9))
    B.append(box(62, 143, 6, 3, 1, 0.15, ('rust3', 'rust2', 'rust1', 'ink2'), 8.6))
    B.append(box(22, 145, 8, 4, 2, 0.05, ('ochre3', 'ochre2', 'ochre1', 'ink2'), 8.8))
    B.append(box(40, 148, 6, 3, 1, -0.2, ('rust3', 'rust2', 'rust1', 'ink2'), 8.4))
    B.append(box(71, 150, 7, 4, 2, 0.12, ('dust6', 'dust4', 'dust2', 'dust1'), 8.2))
    B.append(box(6, 148, 9, 5, 2, -0.08, ('ash5', 'ash4', 'ash2', 'ash1'), 8.9))
    B.append(box(56, 153, 5, 3, 1, 0.3, ('rust3', 'rust2', 'rust1', 'ink2'), 7.9))
    B.append(box(30, 155, 6, 3, 1, -0.1, ('dust5', 'dust4', 'dust2', 'dust1'), 7.8))
    # small grit & brick crumbs scattered on the slopes (tiny 2-3 px clusters)
    for i in range(40):
        cx = -2 + rnd(i, 41) * 100
        t0 = float(np.interp(cx, fx, top))
        b0 = float(np.interp(cx, fx, base))
        cy = t0 + 2 + rnd(i, 42) * max(b0 - t0 - 3, 1)
        c3 = [('rust2', 'rust1'), ('dust5', 'dust3'), ('ochre2', 'ochre1'), ('ash4', 'ash2')][int(rnd(i, 43) * 4)]
        m = np.zeros((SH, SW), bool)
        xi, yi = int(cx) + M, int(cy) + M
        m[yi, xi:xi + 2] = True
        if rnd(i, 44) < 0.5:
            m[yi - 1, xi] = True
        stamp(m & mound, c3[0], 9.0, (0.0, 0.6, -0.8))
        ALB[shift(m, 0, 1) & mound & ~m] = C(c3[1])
    # twisted glazing bars poking out of the pile
    for i, (x0, y0, x1, y1, bend) in enumerate([(26, 128, 18, 110, 3), (47, 125, 55, 108, -4),
                                                 (80, 148, 92, 138, 2), (10, 132, 2, 117, -2)]):
        pts = _bezier((x0, y0), ((x0 + x1) / 2 + bend, (y0 + y1) / 2), (x1, y1), 40)
        m = np.zeros((SH, SW), bool)
        path_mask(_pp_path([(int(round(p[0] + M)), int(round(p[1] + M))) for p in pts]), m)
        stamp(m, 'steel3', 9.3, (0.4, 0.5, -0.7), mat=2)
        ALB[m & (noise.rand(xs_.astype(int), ys_.astype(int), 35 + i) < 0.35)] = C('steel1')

    # --- the fallen iron rib: a heavy I-beam arching from the upper left into the pile
    n = 500
    path = _bezier(RIB_P0, RIB_P1, RIB_P2, n)
    tang = np.gradient(path, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    nor = np.stack([tang[:, 1], -tang[:, 0]], 1)
    nor[nor[:, 1] > 0] *= -1                                # "up" side of the beam
    Zs = np.linspace(RIB_Z0, RIB_Z2, n)
    thick = F * 0.27 / Zs
    ribm = np.zeros((SH, SW), bool)
    sgrid = np.zeros((SH, SW))
    Zrib = np.full((SH, SW), 8.0)
    for k in range(n):
        for s_ in np.linspace(-0.5, 0.5, 15):
            xi = int(round(path[k, 0] + nor[k, 0] * s_ * thick[k] + M))
            yi = int(round(path[k, 1] + nor[k, 1] * s_ * thick[k] + M))
            if 0 <= xi < SW and 0 <= yi < SH:
                ribm[yi, xi] = True
                Zrib[yi, xi] = Zs[k]
                sgrid[yi, xi] = s_
    # snapped, twisted end buried in the pile
    endm = poly_mask([SPt(83, 149), SPt(93, 152), SPt(91, 159), SPt(85, 161), SPt(80, 156)])
    ribm &= ~(endm & (noise.vnoise2(FX / 1.5, FY / 1.5, 311) > 0.42))
    upper = ribm & (sgrid > 0.28)
    lower = ribm & (sgrid < -0.30)
    stamp(ribm, 'steel1', Zrib, (0.0, 0.1, -1.0), mat=2)
    stamp(upper, 'steel3', Zrib, (-0.2, 1.0, -0.3), mat=2)
    stamp(lower, 'steel0', Zrib, (0.1, -1.0, -0.3), mat=2)
    top_edge = ribm & ~shift(ribm, 0, 1)
    ALB[top_edge] = C('steel4')
    # rivets along the web
    for k in range(8, n, 13):
        xi = int(round(path[k, 0] + M))
        yi = int(round(path[k, 1] + M))
        if 0 <= xi < SW and 0 <= yi < SH and ribm[yi, xi] and not upper[yi, xi]:
            ALB[yi, xi] = C('steel4')
            if yi + 1 < SH and ribm[yi + 1, xi]:
                ALB[yi + 1, xi] = C('steel0')
    # sparse rust: streaks running down from the rivets and flaking on the flange
    rust = ribm & ~top_edge & ~upper & (noise.fbm2(FX / 5.0, FY / 2.0, 313, 3) > 0.70)
    ALB[rust] = C('rust1')
    fl_ = upper & (noise.fbm2(FX / 3.0, FY / 3.0, 314, 2) > 0.66)
    ALB[fl_] = C('rust2')
    # sand piled against the beam where it lies in the mound
    drift = shift(ribm, 0, -1) & ~ribm & mound
    ALB[drift] = C('dust5')
    return mound, ribm


RUBBLE_M, FALLEN_RIB_M = _rubble_and_rib()


RUBBLE_M, FALLEN_RIB_M = _rubble_and_rib()


# ---------------------------------------------------------------- the gardener's workplace
def _line_mask(pts):
    m = np.zeros((SH, SW), bool)
    path_mask(_pp_path([(int(round(p[0] + M)), int(round(p[1] + M))) for p in pts]), m)
    return m


def _seg(x0, y0, x1, y1):
    xs, ys = draw.line_points(x0 + M, y0 + M, x1 + M, y1 + M)
    m = np.zeros((SH, SW), bool)
    for x, y in zip(xs, ys):
        if 0 <= x < SW and 0 <= y < SH:
            m[y, x] = True
    return m


def _rect(x, y, w, h):
    m = np.zeros((SH, SW), bool)
    x0, y0 = max(int(x) + M, 0), max(int(y) + M, 0)
    m[y0:max(int(y + h) + M, 0), x0:max(int(x + w) + M, 0)] = True
    return m


def pot_mask(cx, top, w, h, rim=1):
    """Terracotta flower pot (tapered), frame coords of rim centre/top."""
    wb = w * 0.72
    body = poly_mask([SPt(cx - w / 2, top + rim), SPt(cx + w / 2, top + rim),
                      SPt(cx + wb / 2, top + h), SPt(cx - wb / 2, top + h)])
    rimm = _rect(cx - w / 2 - 0.5, top, w + 1, rim)
    return body, rimm


PROP_CARDS = {}


def _workplace():
    _before = PROPM.copy()
    # ---- potting bench (Z 12.5): top plank, legs, lower shelf
    Zb = 12.5
    bx0, bx1 = 122, 155
    ytop = 118
    legs = [_rect(bx0 + 1, ytop + 2, 1, 13), _rect(bx1 - 2, ytop + 2, 1, 13),
            _rect(bx0 + 3, ytop + 1, 1, 12), _rect(bx1 - 4, ytop + 1, 1, 12)]
    for i, lm in enumerate(legs):
        stamp(lm, 'ochre1' if i < 2 else 'dust1', Zb + (0.4 if i >= 2 else 0), (0.3, 0, -1))
    shelf = _rect(bx0 + 1, 127, bx1 - bx0 - 2, 1)
    stamp(shelf, 'ochre2', Zb, (0, 1, -0.4))
    plank_top = _rect(bx0, ytop, bx1 - bx0, 1)
    plank_front = _rect(bx0, ytop + 1, bx1 - bx0, 2)
    stamp(plank_front, 'ochre2', Zb, (0, 0, -1))
    stamp(plank_top, 'ochre4', Zb, (0, 1, -0.3))
    ALB[_rect(bx0, ytop + 2, bx1 - bx0, 1)] = C('ochre1')
    for xg in (bx0 + 9, bx0 + 21):                           # plank joints
        ALB[_rect(xg, ytop + 1, 1, 2)] = C('ochre1')
    # items on the bench: seed trays, pots, a stack of pots
    t1 = _rect(bx0 + 2, ytop - 2, 8, 2); stamp(t1, 'ochre3', Zb, (0, 0.4, -1))
    ALB[_rect(bx0 + 2, ytop - 2, 8, 1)] = C('ochre4')
    ALB[_rect(bx0 + 3, ytop - 1, 6, 1)] = C('ochre1')
    t2 = _rect(bx0 + 4, ytop - 4, 7, 2); stamp(t2, 'dust4', Zb, (0, 0.4, -1))
    ALB[_rect(bx0 + 4, ytop - 4, 7, 1)] = C('dust5')
    for (cx, w, h) in ((bx0 + 15, 4, 3), (bx0 + 19, 3, 3), (bx0 + 27, 4, 4)):
        body, rimm = pot_mask(cx, ytop - h, w, h)
        stamp(body, 'rust2', Zb, (0, 0, -1)); stamp(rimm, 'rust3', Zb, (0, 0.6, -0.8))
        ALB[body & ~shift(body, -1, 0)] = C('rust1')
    for k in range(4):                                       # stacked pots
        body, rimm = pot_mask(bx0 + 23, ytop - 3 - k * 2, 4, 3)
        stamp(body | rimm, 'rust2', Zb, (0, 0, -1))
        ALB[rimm] = C('rust3')
    # under the bench: a sack slumped against a leg + more pots
    sack = poly_mask([SPt(bx0 + 4, 131), SPt(bx0 + 5, 125), SPt(bx0 + 9, 124), SPt(bx0 + 12, 126), SPt(bx0 + 12, 131)])
    stamp(sack, 'dust4', Zb + 0.2, auto_normals(sack, 1.2))
    ALB[sack & ~shift(sack, 0, 1)] = C('dust5')
    ALB[_seg(bx0 + 6, 127, bx0 + 10, 126) & sack] = C('dust2')
    for cx in (bx0 + 20, bx0 + 25):
        body, rimm = pot_mask(cx, 124, 4, 3)
        stamp(body | rimm, 'rust1', Zb + 0.2, (0, 0, -1))
        ALB[rimm] = C('rust2')

    # ---- tools leaning against the bench's left end (rake, spade, hoe)
    Zt = 12.3
    rake = _seg(118, 131, 112, 101)
    stamp(rake, 'ochre3', Zt, (-0.3, 0.2, -1))
    rh = _seg(109, 100, 115, 101) | _seg(109, 100, 109, 101)
    for tx in range(109, 116, 2):
        rh |= _rect(tx, 101, 1, 2)
    stamp(rh, 'steel3', Zt, (0, 0.5, -1), mat=2)
    spade = _seg(121, 131, 118, 106)
    stamp(spade, 'ochre2', Zt - 0.1, (-0.3, 0.2, -1))
    grip = _rect(117, 104, 3, 1) | _rect(117, 105, 1, 1) | _rect(119, 105, 1, 1)
    stamp(grip, 'ochre3', Zt - 0.1, (0, 1, -0.5))
    blade = poly_mask([SPt(119.5, 125), SPt(123.5, 125.5), SPt(123, 131), SPt(121, 132), SPt(119, 130.5)])
    stamp(blade, 'steel3', Zt - 0.1, (0.2, 0.3, -1), mat=2)
    ALB[blade & ~shift(blade, -1, 0)] = C('steel2')
    ALB[blade & (noise.rand(xs_.astype(int), ys_.astype(int), 341) < 0.3)] = C('rust2')
    hoe = _seg(124, 131, 128, 104)
    stamp(hoe, 'dust3', Zt + 0.2, (0.3, 0.2, -1))
    stamp(_rect(126, 103, 4, 2), 'steel2', Zt + 0.2, (0, 0.5, -1), mat=2)

    PROP_CARDS['bench'] = (PROPM & ~_before, 12.4)
    _before = PROPM.copy()
    # ---- dipping tank (stone trough) + rusted standpipe and tap
    Zk = 11.0
    tx0, tx1, ttop, tbot = 162, 186, 126, 137
    tf = _rect(tx0, ttop + 2, tx1 - tx0, tbot - ttop - 2)
    stamp(tf, 'dust5', Zk, (0, 0, -1))
    tt = _rect(tx0, ttop, tx1 - tx0, 2)
    stamp(tt, 'dust6', Zk + 0.3, (0, 1, -0.3))
    inner = _rect(tx0 + 2, ttop, tx1 - tx0 - 4, 1)
    stamp(inner, 'dust1', Zk + 0.4, (0, 1, -0.3))
    ALB[_rect(tx0, ttop + 2, tx1 - tx0, 1)] = C('dust4')
    ALB[_rect(tx0, tbot - 1, tx1 - tx0, 1)] = C('dust3')
    for xj in (tx0 + 8, tx0 + 16):
        ALB[_rect(xj, ttop + 3, 1, tbot - ttop - 4)] = C('dust4')
    ALB[_seg(tx0 + 11, ttop + 3, tx0 + 13, ttop + 8) | _seg(tx0 + 13, ttop + 8, tx0 + 12, tbot - 2)] = C('dust3')
    # old water stain (a darker tide line) -- the tank is bone dry
    ALB[_rect(tx0 + 1, ttop + 4, tx1 - tx0 - 2, 1) & (bayer(SH, SW) < 0.6)] = C('dust4')
    pipe = _rect(tx1 - 5, 110, 2, ttop - 110)
    stamp(pipe, 'steel2', Zk + 0.6, (0.4, 0, -1), mat=2)
    ALB[_rect(tx1 - 5, 110, 1, ttop - 110)] = C('steel3')
    spout = _seg(tx1 - 5, 110, tx1 - 10, 110) | _seg(tx1 - 10, 110, tx1 - 12, 112) | _rect(tx1 - 12, 112, 1, 2)
    stamp(spout, 'steel2', Zk + 0.6, (0, 1, -0.5), mat=2)
    valve = _rect(tx1 - 6, 107, 4, 1) | _rect(tx1 - 5, 106, 2, 1) | _rect(tx1 - 4, 108, 1, 2)
    stamp(valve, 'rust3', Zk + 0.6, (0, 1, -0.5), mat=2)
    ALB[_rect(tx1 - 5, 113, 2, 3)] = C('rust2')
    ALB[_rect(tx1 - 5, 118, 1, 2)] = C('rust1')
    # a pipe running from the tap along the floor to the back wall
    fp_ = _rect(tx1 - 4, 124, 30, 1)
    ALB[fp_ & ~PROPM & (SID == SID_FLOOR)] = C('steel1')

    PROP_CARDS['tank'] = (PROPM & ~_before, 11.1)
    _before = PROPM.copy()
    # ---- seed trays stacked on the floor right of the planter
    Zs = 9.4
    sx0, sy0 = 271, 139
    for k in range(3):
        y = sy0 + 5 - k * 3
        m = _rect(sx0 + (k == 2), y, 11, 2)
        stamp(m, ('ochre2', 'dust4', 'ochre3')[k], Zs, (0, 0, -1))
        ALB[_rect(sx0 + (k == 2), y, 11, 1)] = C(('ochre3', 'dust5', 'ochre4')[k])
        ALB[_rect(sx0 + (k == 2) + 10, y, 1, 2)] = C('ochre1')
    lean = poly_mask([SPt(sx0 + 1, sy0 - 3), SPt(sx0 + 11, sy0 - 6), SPt(sx0 + 12, sy0 - 4), SPt(sx0 + 2, sy0 - 1)])
    stamp(lean, 'dust4', Zs, (0.2, 0.6, -0.8))
    ALB[lean & ~shift(lean, 0, 1)] = C('dust5')

    PROP_CARDS['trays'] = (PROPM & ~_before, 9.4)
    # ---- floor shards of fallen glass (sparkle when light touches them): a few clusters
    clusters = [(22, 166, 9), (58, 170, 7), (104, 162, 6), (120, 146, 5), (70, 131, 4), (150, 170, 5),
                (178, 150, 4), (262, 170, 5), (246, 128, 3), (8, 172, 5)]
    for ci, (cxs, cys, nn) in enumerate(clusters):
        for i in range(nn):
            x = cxs + (float(noise.rand(ci, i, 351)) - 0.5) * 14
            y = cys + (float(noise.rand(ci, i, 352)) - 0.5) * 5
            xi, yi = int(x) + M, int(y) + M
            if not (0 <= xi < SW - 2 and 0 <= yi < SH - 1):
                continue
            if SID[yi, xi] != SID_FLOOR or PROPM[yi, xi]:
                continue
            Zf = POS[yi, xi, 2]
            m = np.zeros((SH, SW), bool)
            m[yi, xi] = True
            big = noise.rand(ci, i, 353) < 0.4
            if big:
                m[yi, xi + 1] = True
            stamp(m, 'glass2', Zf, (0.0, 1.0, -0.2), mat=1)
            if big:
                ALB[yi, xi + 1] = C('glass1')


_workplace()


# ---------------------------------------------------------------- hanging pots + dead palm (drawn later, FG)
HANG = [(131, 57, 13.0, 0), (152, 40, 14.0, 1), (296, 64, 12.0, 2)]   # (x, y of pot rim, Z, seed)


def hanging_pot(img_alb, x, y, Z, seed, swing=0.0):
    """Chain from the top of frame down to an empty hanging pot. Returns mask."""
    xb = x + swing
    ch = _seg(int(round(x + swing * 0.2)), -M, int(round(xb)), y - 4)
    br = _seg(int(round(xb)), y - 4, int(round(xb - 2)), y) | _seg(int(round(xb)), y - 4, int(round(xb + 2)), y)
    body, rimm = pot_mask(xb, y, 5, 4)
    return ch, br, body, rimm


def _dead_palm():
    """Big terracotta pot in the right foreground with a long-dead palm: framing FG."""
    Zp = 6.2
    cx, top = 306, 156
    body = poly_mask([SPt(cx - 14, top + 3), SPt(cx + 14, top + 3), SPt(cx + 10, top + 25), SPt(cx - 10, top + 25)])
    rimm = poly_mask([SPt(cx - 15.5, top), SPt(cx + 15.5, top), SPt(cx + 15, top + 4), SPt(cx - 15, top + 4)])
    nb = auto_normals(body | rimm, 3.0)
    stamp(body, 'rust2', Zp, nb, fg=True)
    stamp(rimm, 'rust3', Zp, nb, fg=True)
    ALB[rimm & ~shift(rimm, 0, 1)] = C('rust4')
    ALB[rimm & ~shift(rimm, 0, -1)] = C('rust1')
    ALB[body & ~shift(body, -1, 0)] = C('rust1')
    ALB[body & shift(~body, 1, 0) & ~rimm] = C('rust3')
    # dust caked on the pot and a crack
    ALB[body & (noise.fbm2(FX / 4.0, FY / 3.0, 361, 3) > 0.60)] = C('dust4')
    ALB[_seg(cx - 4, top + 4, cx - 2, top + 11) | _seg(cx - 2, top + 11, cx - 5, top + 17)] = C('rust1')
    # soil + trunk stump
    soil = _rect(cx - 13, top, 26, 1)
    stamp(soil, 'dust2', Zp + 0.3, (0, 1, 0), fg=True)
    trunk = poly_mask([SPt(cx - 4, top), SPt(cx + 3, top), SPt(cx + 2, top - 14), SPt(cx - 2, top - 15)])
    stamp(trunk, 'dust3', Zp + 0.2, auto_normals(trunk, 1.5), fg=True)
    for yy in range(top - 14, top, 3):                     # fibrous rings
        ALB[_rect(cx - 3, yy, 6, 1) & trunk] = C('ochre1')
    ALB[trunk & ~shift(trunk, 1, 0)] = C('dust4')
    # dead fronds: arching rachis with fishbone leaflets hanging from it (dry, grey-brown)
    fronds = [((cx - 1, top - 14), (cx - 22, top - 58), (cx - 50, top - 44), 1.0),
              ((cx, top - 15), (cx - 6, top - 70), (cx - 28, top - 74), 0.9),
              ((cx + 1, top - 15), (cx + 10, top - 56), (cx + 30, top - 58), 0.8),
              ((cx - 2, top - 13), (cx - 24, top - 26), (cx - 36, top - 2), 0.8),   # broken, hanging
              ((cx + 1, top - 13), (cx + 16, top - 26), (cx + 20, top + 2), 0.7)]
    for fi, (p0, p1, p2, sc) in enumerate(fronds):
        pts = _bezier(p0, p1, p2, 120)
        rach = _line_mask(pts)
        cr = 'dust4' if fi % 2 == 0 else 'ochre2'
        stamp(rach, cr, Zp + 0.1, (0, 0.5, -0.8), fg=True)
        leaf = np.zeros((SH, SW), bool)
        for j in range(10, 118, 3):
            px, py = pts[j]
            tx, ty = pts[j + 1] - pts[j - 1]
            tl = math.hypot(tx, ty) + 1e-9
            tx, ty = tx / tl, ty / tl
            u = j / 120.0
            ln = sc * (7.0 * (1 - u) ** 0.5 + 1.5)
            if noise.rand(fi, j, 363) < 0.18 + 0.25 * u:
                continue                                      # broken-off leaflets
            for side in (-1, 1):
                # leaflet direction: out from the rachis, drooping under gravity
                ox, oy = -ty * side, tx * side
                dx, dy = ox * 0.55 + tx * 0.35, oy * 0.55 + ty * 0.35 + 0.75
                dl = math.hypot(dx, dy)
                ex, ey = px + dx / dl * ln, py + dy / dl * ln
                leaf |= _seg(int(round(px)), int(round(py)), int(round(ex)), int(round(ey)))
        leaf &= ~rach
        stamp(leaf & ~(FGM & ~leaf), 'dust3' if fi % 2 == 0 else 'ochre1', Zp + 0.15, (0, 0.2, -0.9), fg=True)
        tips = leaf & ~shift(leaf, 0, -1)
        ALB[tips & (noise.rand(xs_.astype(int), ys_.astype(int), 365 + fi) < 0.5)] = C('dust2')
    return body | rimm | trunk


PALM_M = _dead_palm()


# ================================================================ glass layers per time of day
def _lum(img):
    f = img.astype(np.float64)
    return (0.30 * f[..., 0] + 0.55 * f[..., 1] + 0.15 * f[..., 2]) / 255.0


def _glass_layer(kind):
    ext = EXT[kind]
    out = ext.copy()
    th = bayer(SH, SW)
    lum = _lum(ext)
    intact = GLASS & ~OPEN
    if kind == 'night':
        ramp_ = ['ink1', 'night1', 'night1', 'night2', 'night2', 'night3']
        t = np.clip((lum - 0.035) / 0.16, 0, 1)
        dust_c, dust_c2, shine_c, crack_c = 'night1', 'ink2', 'night3', 'night3'
    else:
        ramp_ = ['glass1', 'ash4', 'glass2', 'ash6', 'glass3']
        t = np.clip((lum - 0.30) / 0.36, 0, 1)
        dust_c, dust_c2, shine_c, crack_c = 'ash6', 'dust6', 'glass4', 'glass3'
    tmp = np.zeros_like(out)
    _bands(tmp, t, ramp_, band=0.5)
    out[intact] = tmp[intact]
    dm = intact & (th < DUSTG)
    out[dm] = C(dust_c)
    out[intact & (DUSTG > 0.95) & (PV < 0.06)] = C(dust_c2)
    out[SHINE] = C(shine_c)
    out[CRACK] = C(crack_c)
    return out


GLASS_RGB = {k: _glass_layer(k) for k in ('night', 'dawn')}

# S6 read on normal screens: the night glass + sky one ramp step brighter overall,
# two on the moon side of the dome (upper right) -- still deep blue night.
_NIGHT_UP = {'ink0': 'ink1', 'ink1': 'ink2', 'ink2': 'night3', 'night0': 'night1', 'night1': 'night2',
             'night2': 'night3', 'night3': 'night4', 'night4': 'night5', 'night5': 'night5'}


def _step_up(img, mask):
    """Raise every night/ink colour under mask one step along the night chain."""
    k = ((img[..., 0].astype(np.int64) << 16) | (img[..., 1].astype(np.int64) << 8) | img[..., 2])
    out = img.copy()
    for a, b in _NIGHT_UP.items():
        ca = C(a)
        m = mask & (k == ((ca[0] << 16) | (ca[1] << 8) | ca[2]))
        out[m] = C(b)
    return out


_moon_g = np.exp(-(((FX - 286) / 95.0) ** 2 + ((FY - 18) / 60.0) ** 2))
GLASS_RGB['night'] = _step_up(GLASS_RGB['night'], GLASS)
GLASS_RGB['night'] = _step_up(GLASS_RGB['night'], GLASS & (bayer(SH, SW) < np.clip((_moon_g - 0.30) / 0.35, 0, 1)))


# ---------------------------------------------------------------- S7: dawn clouds drifting beyond the glass
# The dawn glass layer rebuilt per frame (only the glass pixels) from a cloud field
# that slides sideways; everything else about the glass (tint bands, dust, cracks,
# shine) is identical to GLASS_RGB['dawn'] (which is the drift at s = 0).
CLOUD_SPEED = 6.0                         # px/s, drifting to the right
_CL_PAD = 64
_GY, _GX = np.nonzero(GLASS)
_xe = (np.arange(-_CL_PAD, SW) - M) / 90.0 * 0.9
CLOUD_EXT = noise.fbm2(_xe[None, :], (FY[:, :1] / 14.0), 17, 4)       # (SH, SW + PAD)
_TB = (np.clip((FY + 10) / (HZ + 10), 0, 1) ** 1.2 +
       0.45 * np.exp(-(((FX - 300) / 90.0) ** 2 + ((FY - 30) / 60.0) ** 2)))[_GY, _GX]
_B4 = bayer(SH, SW, 4)[_GY, _GX]
_SKYM = EXT_SKYM['dawn'][_GY, _GX]
_EXTS = EXT['dawn'][_GY, _GX]
_INT = (GLASS & ~OPEN)[_GY, _GX]
_DM = (_INT & (bayer(SH, SW)[_GY, _GX] < DUSTG[_GY, _GX]))
_DM2 = _INT & (DUSTG[_GY, _GX] > 0.95) & (PV[_GY, _GX] < 0.06)
_SHN, _CRK = SHINE[_GY, _GX], CRACK[_GY, _GX]
_SKYC = np.array([C(n) for n in ('ash3', 'ash4', 'ash5', 'ash5', 'ash6', 'ash7', 'dust8')], np.uint8)
_GLC = np.array([C(n) for n in ('glass1', 'ash4', 'glass2', 'ash6', 'glass3')], np.uint8)


def _band_idx(tv, k, band, th):
    lv_ = np.clip(tv, 0, 1) * (k - 1)
    fl = np.floor(lv_)
    fr = np.clip((lv_ - fl - 0.5) / band + 0.5, 0, 1)
    return np.clip((fl + (th < fr)).astype(np.int64), 0, k - 1)


def cloud_field(t, speed=CLOUD_SPEED):
    """Cloud density at the glass pixels (flat) after drifting speed*t px."""
    s = min(max(speed * t, 0.0), _CL_PAD - 2.0)
    i0 = int(math.floor(s))
    fr = s - i0
    a = CLOUD_EXT[_GY, _GX + _CL_PAD - i0]
    b = CLOUD_EXT[_GY, _GX + _CL_PAD - i0 - 1]
    return a * (1 - fr) + b * fr


def dawn_glass(t, speed=CLOUD_SPEED):
    """GLASS_RGB['dawn'] with the clouds drifted (full-size image, glass pixels valid)."""
    ts = _TB + (cloud_field(t, speed) - 0.5) * 0.28
    ext = np.where(_SKYM[:, None], _SKYC[_band_idx(ts, 7, 0.55, _B4)], _EXTS)
    lum = (0.30 * ext[:, 0] + 0.55 * ext[:, 1] + 0.15 * ext[:, 2]) / 255.0
    gl = _GLC[_band_idx(np.clip((lum - 0.30) / 0.36, 0, 1), 5, 0.5, _B4)]
    out = np.where(_INT[:, None], gl, ext)
    out[_DM] = C('ash6')
    out[_DM2] = C('dust6')
    out[_SHN] = C('glass4')
    out[_CRK] = C('glass3')
    img = GLASS_RGB['dawn'].copy()
    img[_GY, _GX] = out
    return img


# ================================================================ relighting engine
MAXL = light.MAXL


BAY = bayer(SH, SW)


def quantize(level, band=0.45, th=None):
    th = BAY if th is None else th
    fl = np.floor(level)
    fr = level - fl
    if band < 1.0:
        fr = np.clip((fr - 0.5) / max(band, 1e-3) + 0.5, 0, 1)
    return np.clip((fl + (th < fr)).astype(np.int32), -MAXL, MAXL)


def card_shadow(P, lp, card_mask, card_Z, off=(0, 0)):
    """True where the segment P -> light crosses the plane Z=card_Z inside card_mask
    (card_mask in SET coords, as drawn on screen at depth card_Z)."""
    lx, ly, lz = lp
    Pz = P[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (card_Z - Pz) / (lz - Pz)
    ok = (t > 0.0) & (t < 1.0) & (Pz > card_Z + 0.02)
    Ix = P[..., 0] + t * (lx - P[..., 0])
    Iy = P[..., 1] + t * (ly - P[..., 1])
    sx = np.round(CX + F * Ix / card_Z + M - 0.5).astype(np.int64)
    sy = np.round(HZ - F * (Iy - CH) / card_Z + M - 0.5).astype(np.int64)
    ok &= (sx >= 0) & (sx < SW) & (sy >= 0) & (sy < SH)
    res = np.zeros(P.shape[:2], bool)
    res[ok] = card_mask[sy[ok], sx[ok]]
    return res


def planter_shadow(P, lp, height=HP, radius=RP):
    """True where the segment P -> light passes through the planter cylinder
    (below its coping height) -- excluding the planter's own lit surfaces."""
    lx, ly, lz = lp
    dx = lx - P[..., 0]
    dz = lz - P[..., 2]
    fx_ = P[..., 0] - XP
    fz_ = P[..., 2] - ZP
    a = dx * dx + dz * dz + 1e-9
    b = 2 * (fx_ * dx + fz_ * dz)
    c = fx_ * fx_ + fz_ * fz_ - radius * radius
    disc = b * b - 4 * a * c
    sq = np.sqrt(np.maximum(disc, 0))
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)
    hit = np.zeros(P.shape[:2], bool)
    for t_ in (t1, t2):
        yh = P[..., 1] + t_ * (ly - P[..., 1])
        hit |= (disc > 0) & (t_ > 0.02) & (t_ < 0.98) & (yh < height) & (yh > -0.05)
    return hit


ALB_IDX = to_index(ALB)


# ================================================================ per-frame composition
GLASSPIX = GLASS & ~PROPM & ~FRAME_IRON


def _base(kind):
    b = ALB.copy()
    b[GLASSPIX] = GLASS_RGB[kind][GLASSPIX]
    return b


BASE = {k: _base(k) for k in ('night', 'dawn')}
BASE_IDX = {k: to_index(BASE[k]) for k in BASE}
_CORE_KEYS = set()
for _c in RAMPS['core']:
    _h = _c.lstrip('#')
    _CORE_KEYS.add((int(_h[0:2], 16) << 16) | (int(_h[2:4], 16) << 8) | int(_h[4:6], 16))
_LEAF_KEYS = set()
for _c in RAMPS['leaf']:
    _h = _c.lstrip('#')
    _LEAF_KEYS.add((int(_h[0:2], 16) << 16) | (int(_h[2:4], 16) << 8) | int(_h[4:6], 16))


def depth_at_feet(y):
    return F * CH / max(y + 0.5 - HZ, 1e-3)


def _keys(rgb):
    r = rgb[..., 0].astype(np.int64)
    g = rgb[..., 1].astype(np.int64)
    b = rgb[..., 2].astype(np.int64)
    return (r << 16) | (g << 8) | b


class Scene:
    """Mutable per-frame working copy of the set (albedo idx + 3D maps)."""

    def __init__(self, kind):
        self.kind = kind
        self.img = BASE[kind].copy()
        self.pos = POS.copy()
        self.nrm = NRM.copy()
        self.emis = np.zeros((SH, SW), bool)       # pixels not relit (glows, core, eye)
        self.robot = np.zeros((SH, SW), bool)
        self.actor = np.zeros((SH, SW), bool)      # robot + held/placed props (for shadows)
        self.dirty = []                            # bboxes to re-index

    def place(self, spr, x, y, anchor='feet', Z=None, emissive_core=True, is_robot=False,
              normals_blur=1.4, under_fg=True, emissive_leaf=False):
        """Blit a Sprite with `anchor` at frame (x, y) at depth Z (default: floor
        depth under (x, y)). Returns (mask, top-left set coords)."""
        ax, ay = spr.anchor(anchor)
        px = int(round(x)) - ax + M
        py = int(round(y)) - ay + M
        h, w = spr.h, spr.w
        x0, y0 = max(px, 0), max(py, 0)
        x1, y1 = min(px + w, SW), min(py + h, SH)
        if x0 >= x1 or y0 >= y1:
            return np.zeros((SH, SW), bool), (px, py)
        sub = spr.img[y0 - py:y1 - py, x0 - px:x1 - px]
        a = sub[..., 3] > 127
        if under_fg:
            a &= ~FGM[y0:y1, x0:x1]
        if Z is None:
            Z = depth_at_feet(y)
        region = self.img[y0:y1, x0:x1]
        region[a] = sub[..., :3][a]
        yy, xx = np.nonzero(a)
        X, Y = _world_at(xx + x0, yy + y0, Z)
        self.pos[yy + y0, xx + x0, 0] = X
        self.pos[yy + y0, xx + x0, 1] = Y
        self.pos[yy + y0, xx + x0, 2] = Z
        # silhouette normals (computed on a padded local window)
        pad = 4
        loc = np.zeros((y1 - y0 + 2 * pad, x1 - x0 + 2 * pad), bool)
        loc[pad:-pad, pad:-pad] = a
        from scipy.ndimage import gaussian_filter
        f = gaussian_filter(loc.astype(np.float64), normals_blur)
        gy, gx = np.gradient(f)
        n = np.stack([-gx * 3.0, gy * 3.0, -np.clip(f * 1.2, 0.25, 1.0)], -1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
        n = n[pad:-pad, pad:-pad]
        self.nrm[yy + y0, xx + x0] = n[yy, xx]
        m = np.zeros((SH, SW), bool)
        m[y0:y1, x0:x1] = a
        if emissive_core or emissive_leaf:
            k = _keys(sub[..., :3])
            keys = (list(_CORE_KEYS) if emissive_core else []) + (list(_LEAF_KEYS) if emissive_leaf else [])
            em = a & np.isin(k, keys)
            self.emis[y0:y1, x0:x1] |= em
        if is_robot:
            self.robot |= m
        self.actor |= m
        return m, (px, py)


def crop(img, dx=0, dy=0):
    return img[M + dy:M + dy + H, M + dx:M + dx + W]


def world_of_frame(x, y, Z):
    X = (x + 0.5 - CX) * Z / F
    Y = CH - (y + 0.5 - HZ) * Z / F
    return np.array([X, Y, Z])


# rim masks: solid pixels whose upper/right neighbour is open glass or sky (moon/dawn side)
_SOLID = ~GLASSPIX
RIM_MOON = _SOLID & (shift(GLASSPIX, -1, 0) | shift(GLASSPIX, 0, 1)) & (FY < 125)


def depth_rims(z, solid, jump=0.45):
    """Silhouette edges of nearer objects: top edges (pixel above is farther/sky)
    and right edges (pixel to the right is farther/sky)."""
    zu = shift(z, 0, 1)            # value of the pixel above, at this position
    zr = shift(z, -1, 0)           # value of the pixel to the right
    su = shift(solid, 0, 1)
    sr = shift(solid, -1, 0)
    top = solid & ((~su) | (zu - z > jump))
    right = solid & ((~sr) | (zr - z > jump))
    return top, right


_Zs = POS[..., 2].copy()
RIM_TOP, RIM_RIGHT = depth_rims(_Zs, _SOLID & (FY > 60))
RIM_TOP &= ~(SID == SID_FLOOR) | PROPM


# ================================================================ colored-light LUTs (GREENHOUSE)
# out = albedo * tint * GAIN**k  (linear light), snapped to allowed ramps.
# Level k: 0 = albedo seen under that light at 'daylight' exposure.
GAIN = 1.55
DEAD = [i for i, r in enumerate(P.RAMP_OF) if r in ('ink', 'dust', 'ash', 'ochre', 'rust', 'steel', 'paint',
                                                     'night', 'glass', 'core', 'leaf')]
ALIVE = list(range(P.N))
ENV = [i for i, r in enumerate(P.RAMP_OF) if r in ('ink', 'dust', 'ash', 'ochre', 'rust', 'steel', 'night', 'core')]
ENV_ALIVE = [i for i, r in enumerate(P.RAMP_OF) if r not in ('paint', 'glass')]
TINTS = {
    'moon':  (0.46, 0.60, 1.00),
    'amber': (1.00, 0.50, 0.14),
    'gold':  (1.00, 0.74, 0.34),
    'green': (0.40, 1.00, 0.28),
    'dawn':  (0.70, 0.84, 1.00),
    'neutral': (1.0, 1.0, 1.0),
    'vein':  (0.86, 1.00, 0.42),
    'sun':   (1.00, 0.86, 0.55),
    'glassgold': (1.00, 0.60, 0.16),
}
_MYLUT = {}


def tint_lut(tint, allowed='dead', gain=GAIN):
    """(2*MAXL+1, N) palette-index LUT for a colored light. tint: name or rgb."""
    key = (tint if isinstance(tint, str) else tuple(np.round(tint, 3)), allowed, gain)
    if key in _MYLUT:
        return _MYLUT[key]
    tv = np.asarray(TINTS[tint] if isinstance(tint, str) else tint, np.float64)
    al = {'dead': DEAD, 'env': ENV, 'alive': ALIVE, 'env_alive': ENV_ALIVE}[allowed]
    base = P.PAL_LIN
    out = np.zeros((2 * MAXL + 1, P.N), np.uint8)
    for k in range(-MAXL, MAXL + 1):
        c = base * tv[None, :] * gain ** k
        # overexposure bleeds toward white (keeps highlights luminous, not clipped hue)
        over = np.clip(c.max(-1, keepdims=True) - 1.0, 0, None)
        c = c + over * 0.35
        out[k + MAXL] = P.nearest_index(np.clip(c, 0, 1), allowed=al)
    _MYLUT[key] = out
    return out


def mix_tints(a, b, n=5):
    ta = np.asarray(TINTS[a] if isinstance(a, str) else a, np.float64)
    tb = np.asarray(TINTS[b] if isinstance(b, str) else b, np.float64)
    return [tuple(ta + (tb - ta) * i / (n - 1)) for i in range(n)]


def lv(E):
    """energy -> level (float)."""
    return np.log(np.maximum(E, 1e-6)) / math.log(GAIN)


def en(L):
    return GAIN ** np.asarray(L, np.float64)


def relight_mix(idx, level, luts, which, band=0.45, th=None):
    """Per-pixel choice of LUT (which: int map into luts) at quantized level."""
    q = quantize(level, band, th)
    out_idx = np.empty_like(idx)
    for k, lt in enumerate(luts):
        m = which == k
        if m.any():
            out_idx[m] = lt[q[m] + MAXL, idx[m]]
    return from_index(out_idx)


def ratio_index(Ea, Eb, n=5, th=None):
    """Dithered index 0..n-1 of Eb/(Ea+Eb) (0 = all A, n-1 = all B)."""
    th = BAY if th is None else th
    r = Eb / np.maximum(Ea + Eb, 1e-9) * (n - 1)
    fl = np.floor(r)
    return np.clip((fl + (th < (r - fl))).astype(np.int64), 0, n - 1)


def light_level(P, N, lp, peak, slope, reach, wrap=0.5, cone=None):
    """Point light as ENERGY with a level-linear falloff: level = peak - slope*d,
    zero beyond `reach` metres; modulated by wrap-Lambert and an optional cone.
    Small lights are evaluated only inside their screen-space bounding box."""
    if reach < 3.0:
        lp_ = np.asarray(lp, np.float64)
        zc = max(lp_[2] - reach, 0.6)
        sx, sy = proj(lp_[0], lp_[1], lp_[2])
        rad = F * reach / zc + 4
        x0, x1 = int(max(sx + M - rad, 0)), int(min(sx + M + rad + 1, SW))
        y0, y1 = int(max(sy + M - rad, 0)), int(min(sy + M + rad * 1.6 + 1, SH))
        out = np.zeros(P.shape[:2])
        if x0 < x1 and y0 < y1:
            out[y0:y1, x0:x1] = _light_level_full(P[y0:y1, x0:x1], N[y0:y1, x0:x1], lp_, peak, slope,
                                                  reach, wrap, cone)
        return out
    return _light_level_full(P, N, lp, peak, slope, reach, wrap, cone)


def _light_level_full(P, N, lp, peak, slope, reach, wrap=0.5, cone=None):
    d = P - np.asarray(lp, np.float64)
    dist = np.sqrt((d * d).sum(-1)) + 1e-6
    ldir = -d / dist[..., None]
    lam = np.clip(((N * ldir).sum(-1) + wrap) / (1.0 + wrap), 0, 1)
    E = en(peak - slope * dist) * lam
    E *= np.clip((reach - dist) / (0.25 * reach), 0, 1)
    if cone is not None:
        cd, spill = cone
        cd = np.asarray(cd, np.float64)
        cd = cd / np.linalg.norm(cd)
        c = ((d / dist[..., None]) * cd).sum(-1)
        E *= spill + (1 - spill) * np.clip((c + 0.35) / 1.35, 0, 1)
    return E


# ================================================================ sky transmission (light through the dome)
def _pane_at(phi, y_or_th, on_dome):
    """Material of the glazing at (phi, theta|Y): returns transmission 0..1
    (iron 0, dusty glass ~0.4, open 1)."""
    iphi = np.floor(phi / BARW).astype(np.int64)
    pu = phi / BARW - iphi
    if on_dome:
        th = y_or_th
        irow = np.floor(th / ROWH).astype(np.int64)
        pv = th / ROWH - irow
        cosv = np.cos(th)
        hd = ((phi - _hole_c[0]) / math.radians(17)) ** 2 + ((th - _hole_c[1]) / math.radians(15)) ** 2
        hn = noise.fbm2(phi * 6, th * 6, 41, 3)
        hole = hd + (hn - 0.5) * 0.9 < 1.0
        dist_rib = np.abs((phi + RIBW / 2) % RIBW - RIBW / 2) * R * cosv
        iron = (dist_rib < RIBW_M * 0.8) | (th < 0.012)
        iron &= ~(hole & (dist_rib > RIBW_M / 2))
        near_hole = np.clip(1.8 - hd, 0, 1)
        pbreak = 0.12 + 0.40 * near_hole + 0.05
    else:
        yy = y_or_th
        irow = np.where(yy < YT, -2, -1)
        dist_rib = np.abs((phi + RIBW / 2) % RIBW - RIBW / 2) * R
        iron = (dist_rib < RIBW_M * 0.8) | (yy > SY - 0.10)
        hole = np.zeros_like(iron)
        pbreak = 0.12 + 0.06
    pr = noise.rand(iphi, irow, 101)
    opened = (pr < pbreak) | hole
    half = (pr >= pbreak) & (pr < pbreak + 0.10)
    tr = np.where(opened, 1.0, np.where(half, 0.7, 0.38))
    return np.where(iron, 0.0, tr)


def sky_transmission(Pw, D):
    """For world points Pw (...,3) inside the greenhouse, transmission of light
    arriving from direction D (unit, pointing TO the light)."""
    D = np.asarray(D, np.float64)
    D = D / np.linalg.norm(D)
    px, py, pz = Pw[..., 0], Pw[..., 1], Pw[..., 2]
    fx_, fz_ = px - DXc, pz - DZc
    a = D[0] ** 2 + D[2] ** 2 + 1e-12
    b = 2 * (fx_ * D[0] + fz_ * D[2])
    c = fx_ ** 2 + fz_ ** 2 - R * R
    s_c = (-b + np.sqrt(np.maximum(b * b - 4 * a * c, 0))) / (2 * a)
    yc = py + s_c * D[1]
    ox, oy, oz = px - DXc, py - SY, pz - DZc
    B = ox * D[0] + oy * D[1] + oz * D[2]
    Cs = ox * ox + oy * oy + oz * oz - R * R
    s_s = -B + np.sqrt(np.maximum(B * B - Cs, 0))
    hx, hy, hz = px + s_s * D[0], py + s_s * D[1], pz + s_s * D[2]
    phi_s = np.arctan2(hx - DXc, hz - DZc)
    th_s = np.arcsin(np.clip((hy - SY) / R, -1, 1))
    phi_c = np.arctan2(px + s_c * D[0] - DXc, pz + s_c * D[2] - DZc)
    t_dome = _pane_at(phi_s, th_s, True)
    t_drum = _pane_at(phi_c, yc, False)
    out = np.where(yc >= SY, t_dome, np.where(yc >= KW, t_drum, 0.0))
    return out


MOON_DIR = np.array([0.30, 0.90, 0.32]) / np.linalg.norm([0.30, 0.90, 0.32])
DAWN_DIR = np.array([0.55, 0.78, 0.30]) / np.linalg.norm([0.55, 0.78, 0.30])


def _soft(m, s=0.8):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(m, s)


_solid_pts = _SOLID & ~(SID == SID_EXT)
SKY_T = {}
for _k, _D in (('moon', MOON_DIR), ('dawn', DAWN_DIR)):
    _t = sky_transmission(POS, _D)
    _t[GLASSPIX] = 0.0
    SKY_T[_k] = _soft(_t, 1.3)
# broad version (diffuse overcast dawn: only the big openings matter)
SKY_T['dawn_soft'] = _soft(SKY_T['dawn'], 5.0)


# ================================================================ shared FX
def screen_glow(cx, cy, peak, falloff, radius):
    """Energy of a screen-space bloom around frame point (cx, cy)."""
    x0, x1 = int(cx + M - radius - 1), int(cx + M + radius + 2)
    y0, y1 = int(cy + M - radius - 1), int(cy + M + radius + 2)
    E = np.zeros((SH, SW))
    x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, SW), min(y1, SH)
    if x0c >= x1c or y0c >= y1c:
        return E
    yy, xx = np.mgrid[y0c:y1c, x0c:x1c]
    d = np.sqrt((xx + 0.5 - (cx + M)) ** 2 + (yy + 0.5 - (cy + M)) ** 2)
    E[y0c:y1c, x0c:x1c] = np.where(d < radius, en(peak - falloff * d), 0.0)
    return E


def motes(t, n, seed, area, vel, sway, life_mask_fn=None):
    """Deterministic drifting dust motes -> (x, y, r) frame coords."""
    x, y, r = fx.field(t, n, seed, area, vel=vel, jitter=(vel[0] * 0.6 + 0.8, vel[1] * 0.6 + 0.8),
                       sway=sway)
    return x, y, r


def hanging_pots_draw(sc, t, sway_amp=0.0, shake=0.0, sway=None):
    """Draw the three hanging empty pots on chains into a Scene (albedo + depth).
    sway: optional per-pot (amplitude px, angular speed) -- None keeps sway_amp."""
    ms = []
    for i, (x, y, Z, seed) in enumerate(HANG):
        ph = seed * 1.7
        if sway is not None and sway[i] is not None:
            sw = sway[i][0] * math.sin(t * sway[i][1] + ph)
        else:
            sw = sway_amp * math.sin(t * 1.3 + ph)
        sw += shake * math.sin(t * 17 + ph * 3)
        ch, br, body, rimm = hanging_pot(None, x, y, Z, seed, swing=sw)
        for m_, colr in ((ch, 'steel1'), (br, 'steel2'), (body, 'rust2'), (rimm, 'rust3')):
            yy, xx = np.nonzero(m_)
            if len(yy) == 0:
                continue
            X, Y = _world_at(xx, yy, Z)
            sc.img[yy, xx] = C(colr)
            sc.pos[yy, xx, 0] = X
            sc.pos[yy, xx, 1] = Y
            sc.pos[yy, xx, 2] = Z
            sc.nrm[yy, xx] = (0.0, -0.3, -1.0)
            sc.actor[yy, xx] = True           # in front of the glass: never overwritten by it
        # dead trailing stems hanging over the rim
        xb = int(round(x + sw))
        for k, (dx_, ln) in enumerate(((-2, 5), (2, 7), (0, 3))):
            yy = np.arange(y + 1, y + 1 + ln) + M
            xx = np.full_like(yy, xb + dx_ + M) + (np.arange(ln) // 3) * (1 if dx_ > 0 else -1)
            ok = (yy >= 0) & (yy < SH) & (xx >= 0) & (xx < SW)
            sc.img[yy[ok], xx[ok]] = C('dust3')
            sc.actor[yy[ok], xx[ok]] = True
        ms.append(body | rimm | ch)
    return ms


# ================================================================ dawn light (S7 / S8a)
_DAWN_CACHE = {}


def _dawn_static(strength, amb):
    key = (round(strength, 3), amb)
    if key in _DAWN_CACHE:
        return _DAWN_CACHE[key]
    N = NRM
    up = N[..., 1]
    right = N[..., 0]
    La = np.where(up > 0.6, amb, np.where(up > -0.2, amb - 1.0, amb - 2.0))
    La = np.where((up <= 0.6) & (right > 0.45), La + 1.0, La)
    Ea = en(La)
    lam = np.clip(((N * DAWN_DIR).sum(-1) + 0.3) / 1.3, 0, 1)
    T = np.clip((SKY_T['dawn_soft'] - 0.3) / 0.5, 0, 1)
    Ed = en(-0.6) * T * lam * strength
    E = Ea * (0.6 + 0.4 * strength) + Ed
    rim = RIM_TOP | RIM_RIGHT
    E = np.where(rim, np.maximum(E, en(amb + 1.0) * (0.5 + 0.5 * strength)), E)
    if len(_DAWN_CACHE) > 24:
        _DAWN_CACHE.clear()
    _DAWN_CACHE[key] = E
    return E


def dawn_energy(sc, strength=1.0, amb=-2.0):
    """Cold diffuse dawn: integer ambient by facing + broad soft light through the
    dome's openings + silhouette rims (light from the upper right). Energy map.
    The static environment part is cached; actors are lit per frame."""
    E = _dawn_static(strength, amb).copy()
    if not sc.actor.any():
        return E
    yy, xx = np.nonzero(sc.actor)
    Nn = sc.nrm[yy, xx]
    up = Nn[:, 1]
    right = Nn[:, 0]
    La = np.where(up > 0.6, amb, np.where(up > -0.2, amb - 1.0, amb - 2.0))
    La = np.where((up <= 0.6) & (right > 0.45), La + 1.0, La)
    la = np.clip(((Nn * DAWN_DIR).sum(-1) + 0.3) / 1.3, 0, 1)
    rb = sc.robot[yy, xx]
    Ta = np.full(len(yy), 0.6)
    if rb.any():
        Ta[rb] = np.clip((sky_transmission(sc.pos[yy[rb], xx[rb]], DAWN_DIR) - 0.3) / 0.5, 0, 1) * 0.6 + 0.2
    E[yy, xx] = en(La) * (0.6 + 0.4 * strength) + en(-0.6) * Ta * la * strength
    rt, rr = depth_rims(sc.pos[..., 2], sc.actor | _SOLID, 0.45)
    ar = sc.actor & (rt | rr)
    E = np.where(ar, np.maximum(E, en(amb + 1.6) * (0.4 + 0.6 * strength)), E)
    return E


def shaft_level(x0, y0, width, dirx, diry, length, soft=3.0):
    """Screen-space light shaft (band) from frame point (x0,y0) along (dirx,diry)."""
    d = np.array([dirx, diry], np.float64)
    d /= np.linalg.norm(d)
    px = FX - x0
    py = FY - y0
    along = px * d[0] + py * d[1]
    across = np.abs(-px * d[1] + py * d[0])
    w = width * (0.7 + 0.6 * np.clip(along / length, 0, 1))
    a = np.clip((w - across) / soft, 0, 1) * np.clip(along / 12.0, 0, 1) * np.clip((length - along) / (0.3 * length), 0, 1)
    return a


_ADD = {}


def add_lut(tint, allowed='dead'):
    """Additive-light LUT (identity at level 0, brighter & tinted above): like the
    engine's light modes but snapped only to `allowed` ramps (colour script)."""
    key = (tint, allowed)
    if key in _ADD:
        return _ADD[key]
    tv = np.asarray(TINTS[tint] if isinstance(tint, str) else tint, np.float64)
    tn = tv / tv.max()
    al = {'dead': DEAD, 'env': ENV, 'alive': ALIVE, 'env_alive': ENV_ALIVE}[allowed]
    base = P.PAL_LIN
    out = np.zeros((MAXL + 1, P.N), np.uint8)
    out[0] = np.arange(P.N)
    for k in range(1, MAXL + 1):
        c = base * (1 + 0.42 * k * tn) + 0.05 * k * tn
        out[k] = P.nearest_index(np.clip(c, 0, 1), allowed=al)
    _ADD[key] = out
    return out


def glass_reflect(img, sc_glass_mask, E_refl, mode='gold', scale=1.0, allowed='dead'):
    """Brighten glass pixels (final colours) by reflected interior light: an
    identity-at-zero additive LUT restricted to the shot's colour script."""
    m = sc_glass_mask & (E_refl > 0.02) & ~OPEN
    if not m.any():
        return img
    Lr = np.clip(np.log1p(E_refl * scale) / math.log(GAIN), 0, MAXL)
    q = quantize(Lr)
    idx = to_index(np.ascontiguousarray(img))
    lt = add_lut({'sun': 'gold', 'warm': 'amber', 'green': 'vein', 'moon': 'moon'}.get(mode, mode), allowed)
    out_idx = idx.copy()
    out_idx[m] = lt[np.clip(q[m], 0, MAXL), idx[m]]
    img[m] = from_index(out_idx[m])
    return img


def core_orb(img, x, y, power, t, flare=True, occ=None, clip_below=None):
    """A blazing orb drawn by the shot (used for the heart sinking into the soil):
    all radii scale with `power`; clip_below = frame y below which it is hidden
    (inside the soil)."""
    X, Y = x + M, y + M
    yy, xx = np.ogrid[:SH, :SW]
    d = np.sqrt((xx + 0.5 - X - 0.5) ** 2 + (yy + 0.5 - Y - 0.5) ** 2)
    keep = np.ones((SH, SW), bool) if occ is None else ~occ
    if clip_below is not None:
        keep = keep & (yy < clip_below + M)
    r0 = 0.6 + 1.9 * power
    img[(d < r0 + 1.6) & (d >= r0) & keep & (BAY < 0.6 * power + 0.2)] = C('core5')
    img[(d < r0) & keep] = C('core6')
    img[(d < r0 * 0.55) & keep] = C('core7')
    return img


_soil_m = SID == SID_SOIL
SOIL_CRACK = (_boundary_h(SOIL_PLATE, _soil_m) | _boundary_v(SOIL_PLATE, _soil_m)) & ~PROPM
CORE_SPOT = (KNEEL[0] + 11, KNEEL[1] - 11)       # CORE_PLACE_OFFSET hand-off (frame)


def buried_glow(img, hx, hy, dt, strength=1.0, occ=None):
    """The heart under the soil: a small warm ember breathing slowly through the
    soil (a soft 2-ring glow + the nearest cracks). dt = s since it was buried."""
    a = strength * (0.55 + 0.45 * math.sin(dt * 2.2) ** 2) * min(1.0, 0.3 + dt * 2.0)
    d = np.hypot(FX - hx, (FY - hy + 0.5) * 2.0)
    soil = (SID == SID_SOIL) | (SID == SID_PRIM)
    if occ is not None:
        soil = soil & ~occ
    inner = soil & (d < 1.6) & (BAY < 0.35 + 0.65 * a)
    outer = soil & (d >= 1.6) & (d < 3.2) & (BAY < 0.45 * a)
    cr = SOIL_CRACK & soil & (d >= 1.6) & (d < 5.0) & (BAY < 0.7 * a)
    img[outer] = C('core1')
    img[cr] = C('core2')
    img[inner] = C('core3') if a > 0.5 else C('core2')
    return img


def soft_shadow_factor(P, shadow_mask, occ_Z, hard=0.06, soft=0.75, spread=9.0):
    """Shadow multiplier that softens (lightens) with distance behind the occluder
    (a finite-size light makes a wide penumbra far away)."""
    d = np.clip((P[..., 2] - occ_Z) / spread, 0, 1)
    f = hard + (soft - hard) * d
    return np.where(shadow_mask, f, 1.0)


# ================================================================ atmosphere & framing
_vx = (FX - 196.0) / 205.0
_vy = (FY - 118.0) / 125.0
VIGN = np.clip(np.sqrt(_vx ** 2 + _vy ** 2) - 0.55, 0, 1) / 0.45   # 0 centre .. 1 corners


def vignette_level(strength=1.4):
    """Level offset (<=0) darkening the frame edges (planter/robot area untouched)."""
    return -strength * VIGN ** 1.6


def haze_lift(L, amount, far_from=11.0, far_to=19.0, pos=None):
    """Aerial perspective: lift dark far surfaces toward `amount` levels."""
    Z = (POS if pos is None else pos)[..., 2]
    f = np.clip((Z - far_from) / (far_to - far_from), 0, 1)
    return np.maximum(L, L + (amount - L) * f * 0.6)


def relight_scene(sc, L, tints, which, band=0.35, env='env', actor='dead', first_dead=False):
    """Relight a Scene: environment pixels use `env`-restricted LUTs (no teal/green
    drift on stone & dust), actors (robot, can, sprout) keep their full palette.
    first_dead: the first tint (pure ambient) keeps the dead-world palette even if
    env/actor allow the living ramps (continuity across the S7 -> S8a cut)."""
    luts = [tint_lut(tn, 'env' if (first_dead and i == 0) else env) for i, tn in enumerate(tints)] + \
           [tint_lut(tn, 'dead' if (first_dead and i == 0) else actor) for i, tn in enumerate(tints)]
    w = np.where(sc.actor, which + len(tints), which)
    return relight_mix(to_index(sc.img), L, luts, w, band=band)


# ================================================================ S8a: veins of light (precomputed network)
def _floor_scale_y(y):
    """Foreshortening of the floor at frame row y (vertical px per horizontal px)."""
    return np.clip((y - HZ) / 60.0, 0.25, 1.0)


def _core_floor_xz():
    x, y = CORE_SPOT
    Z = F * (CH - HS) / (y + 0.5 - HZ)
    X = (x + 0.5 - CX) * Z / F
    return X, Z


def _surface_Y(X, Z):
    """Height of the ground surface at (X, Z): soil / coping / floor."""
    r = math.hypot(X - XP, Z - ZP)
    if r <= RS:
        return HS
    if r <= RP:
        return HP
    return 0.0


def _gen_veins(seed=801):
    """Root-light network in WORLD space (XZ, metres) from the buried heart.
    Returns list of (xz (n,2), arc0 metres, order)."""
    X0, Z0 = _core_floor_xz()
    rnd = lambda a, b, c=0: float(noise.rand(a, b, seed + c))
    # main roots: (heading deg: 0=+X right, 90=+Z away), length m
    mains = [(180, 4.8), (205, 3.6), (150, 3.2), (240, 3.4), (275, 3.0), (315, 3.2),
             (345, 4.6), (15, 4.2), (60, 5.2), (100, 5.0), (125, 4.4)]
    out = []
    todo = [(X0, Z0, math.radians(h + (rnd(i, 1) - 0.5) * 14), ln, 0.0, 0, i) for i, (h, ln) in enumerate(mains)]
    k = 0
    while todo and k < 400:
        x, z, hd, ln, arc0, order, sid = todo.pop()
        k += 1
        pts = [(x, z)]
        st = 0.045
        arc = 0.0
        ph = rnd(sid, k, 3) * 100
        hd0 = hd
        amp_ = 0.55 if order == 0 else 0.75
        while arc < ln:
            # bounded meander around the root's heading (no accumulating curl)
            m1 = float(noise.vnoise1(np.array([arc * 1.1 + ph]), seed + 11)[0]) - 0.5
            m2 = float(noise.vnoise1(np.array([arc * 4.3 + ph * 1.7]), seed + 12)[0]) - 0.5
            hd = hd0 + amp_ * m1 * 1.6 + 0.45 * m2
            x += math.cos(hd) * st
            z += math.sin(hd) * st
            arc += st
            if math.hypot(x - DXc, z - DZc) > R - 0.4 or z < 4.6:
                break
            pts.append((x, z))
            # branch off
            pb = (0.022 if order == 0 else 0.03) if arc > 0.25 else 0.0
            if order < 3 and rnd(k, int(arc * 1000), 5) < pb:
                side = 1 if rnd(k, int(arc * 1000), 6) < 0.5 else -1
                bl = (ln - arc) * (0.25 + 0.35 * rnd(k, int(arc * 1000), 7)) + 0.15
                todo.append((x, z, hd + side * (0.45 + 0.5 * rnd(k, int(arc * 1000), 8)), bl, arc0 + arc,
                             order + 1, sid * 100 + k))
        if len(pts) > 3:
            out.append((np.array(pts), arc0, order))
    return out


VEINS = _gen_veins()
VEIN_SPEED = 3.4                             # metres of arc per second (before easing)


VEIN_W0 = (0.12, 0.075, 0.05, 0.04)          # world width (m) at the root of each order
VEIN_WTIP = 0.022                             # ... tapering to this at the tip


def _raster_veins():
    """Per-pixel arc distance (metres) from the heart, width class, depth.
    Also: tapering on-screen width (px) along each root, its tip points."""
    global VEIN_WPX, VEIN_TIPS
    D = np.full((SH, SW), np.inf)
    WD = np.zeros((SH, SW), np.int8)
    WPX = np.zeros((SH, SW))
    tips = []
    for xz, arc0, order in VEINS:
        seg = np.diff(xz, axis=0)
        arcs = np.concatenate([[0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))]) + arc0
        a_end = arcs[-1]
        ln_ = max(a_end - arc0, 1e-3)
        last = None
        prevY = None
        path = []
        for (X, Z), a in zip(xz, arcs):
            Y = _surface_Y(X, Z)
            if prevY is not None and abs(Y - prevY) > 0.2:
                # climbing over the rim: vertical run down/up the planter wall
                for yy in np.linspace(prevY, Y, 6):
                    sx, sy = proj(X, yy, Z)
                    path.append((sx, sy, a, Z))
            sx, sy = proj(X, Y, Z)
            path.append((sx, sy, a, Z))
            prevY = Y
        pix = []
        for i in range(len(path) - 1):
            x0, y0, a0, z0 = path[i]
            x1, y1, a1, z1 = path[i + 1]
            xs, ys = draw.line_points(x0 - 0.5 + M, y0 - 0.5 + M, x1 - 0.5 + M, y1 - 0.5 + M)
            for x, y in zip(xs, ys):
                pix.append((x, y, a0, z0))
        clean = _pp_path([(p_[0], p_[1]) for p_ in pix])
        dmap = {(p_[0], p_[1]): (p_[2], p_[3]) for p_ in pix}
        for (x, y) in clean:
            if 0 <= x < SW and 0 <= y < SH:
                d, z = dmap.get((x, y), (arcs[-1], xz[-1][1]))
                # visible only where the ground surface at that pixel is this root
                if abs(POS[y, x, 2] - z) > 0.6 and SID[y, x] not in (SID_PSIDE,):
                    continue
                if d < D[y, x]:
                    D[y, x] = d
                    WD[y, x] = 2 if (order == 0 and d < 1.4) else 1
                    # tapering: thick at the root of the branch, fine at its tip
                    u = min(max((d - arc0) / ln_, 0.0), 1.0) ** 0.8
                    w_m = VEIN_W0[min(order, 3)] * (1 - u) + VEIN_WTIP * u
                    WPX[y, x] = max(WPX[y, x], w_m * F / max(POS[y, x, 2], 1.0))
                last = (x, y, d)
        if last is not None:
            tips.append(last)
    VEIN_WPX = WPX
    VEIN_TIPS = tips
    thick = (WD == 2)
    below = shift(thick, 0, 1) & ~np.isfinite(D)
    Db = shift(np.where(thick, D, np.inf), 0, 1)
    D = np.where(below, Db + 0.01, D)
    WD = np.where(below, 1, WD)
    return D, WD


VEIN_WPX = None
VEIN_TIPS = []
VEIN_D, VEIN_W = _raster_veins()
VEIN_LINE = VEIN_WPX > 0                     # the 1-px centre lines (arc owner pixels)


# ================================================================ S8a: the tree (precomputed, revealed by time)
T_BURST = 1.8                                # local S8a time of the growth burst (growth_burst 46.8)


def _catmull(pts, per=10):
    P_ = np.asarray(pts, np.float64)
    P_ = np.vstack([P_[0] * 2 - P_[1], P_, P_[-1] * 2 - P_[-2]])
    out = []
    for i in range(1, len(P_) - 2):
        p0, p1, p2, p3 = P_[i - 1], P_[i], P_[i + 1], P_[i + 2]
        for s in np.linspace(0, 1, per, endpoint=False):
            s2, s3 = s * s, s * s * s
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s2 + (-p0 + 3 * p1 - 3 * p2 + p3) * s3))
    out.append(P_[-2])
    return np.array(out)


def _resample(pts, step=0.5):
    seg = np.diff(pts, axis=0)
    ln = np.hypot(seg[:, 0], seg[:, 1])
    arc = np.concatenate([[0], np.cumsum(ln)])
    n = max(int(arc[-1] / step), 2)
    a = np.linspace(0, arc[-1], n)
    return np.interp(a, arc, pts[:, 0]), np.interp(a, arc, pts[:, 1]), a


BARK_R = ['bark1', 'bark2', 'bark3', 'bark4', 'bark5', 'bark6']


def _stroke_fields(pts, w0, w1, twist=2.0, flare=None):
    """Per-pixel fields of a tapered wooden stroke: arc (px from start), across
    (-1..1), width, shade index (0..5). Returns dict of arrays (only covered px)."""
    xs, ys, arc = _resample(_catmull(pts, 12), 0.4)
    L_ = arc[-1]
    best = np.full((SH, SW), 9.0)
    A = np.full((SH, SW), np.inf)
    ACR = np.zeros((SH, SW))
    WID = np.zeros((SH, SW))
    SHD = np.zeros((SH, SW), np.int8)
    for n in range(len(xs)):
        s = arc[n] / max(L_, 1e-6)
        wd = w0 + (w1 - w0) * s
        if flare is not None:
            wd += flare * max(0.0, 1 - arc[n] / 14.0) ** 2
        r = wd / 2.0
        cx, cy = xs[n] + M, ys[n] + M
        X0, X1 = int(max(cx - r - 1, 0)), int(min(cx + r + 2, SW))
        Y0, Y1 = int(max(cy - r - 1, 0)), int(min(cy + r + 2, SH))
        if X0 >= X1 or Y0 >= Y1:
            continue
        yy = np.arange(Y0, Y1)[:, None] + 0.5
        xx = np.arange(X0, X1)[None, :] + 0.5
        ox_, oy_ = xx - cx, yy - cy
        d = np.sqrt(ox_ ** 2 + oy_ ** 2) / max(r + 0.35, 0.6)
        inside = d <= 1.0
        sub_b = best[Y0:Y1, X0:X1]
        upd = inside & (d < sub_b + 0.2)
        if not upd.any():
            continue
        if n + 1 < len(xs):
            tx_, ty_ = xs[n + 1] - xs[n], ys[n + 1] - ys[n]
        else:
            tx_, ty_ = xs[n] - xs[n - 1], ys[n] - ys[n - 1]
        tl = math.hypot(tx_, ty_) + 1e-6
        across = (ox_ * (-ty_) + oy_ * tx_) / tl / max(r, 0.6)
        # cylinder shading lit from the right (dawn) + twisting grooves
        ld = -across
        ph = arc[n] * 0.11 + across * twist
        groove = (np.sin(ph * math.pi) > 0.72) & (r > 1.6)
        ridge = (np.sin(ph * math.pi) < -0.9) & (r > 2.4) & (ld > -0.2)
        lv = 2.0 + 1.7 * ld - 0.9 * groove + 0.7 * ridge
        lv = np.where(d > 0.84, lv - 0.9, lv)
        q = np.clip(np.floor(lv + 0.5), 0, 5).astype(np.int8)
        sub_b[upd] = np.minimum(sub_b[upd], d[upd])
        A[Y0:Y1, X0:X1][upd] = arc[n]
        ACR[Y0:Y1, X0:X1][upd] = np.abs(across)[upd]
        WID[Y0:Y1, X0:X1][upd] = wd
        SHD[Y0:Y1, X0:X1][upd] = q[upd]
    return A, ACR, WID, SHD, L_


# trunk: from the sprout, twisting up toward the dome's apex (above the planter centre)
_sx, _sy = SPROUT_XY
TRUNK_PTS = [(_sx + 1, _sy + 1), (_sx + 3, _sy - 20), (_sx - 1, _sy - 48), (_sx + 6, _sy - 80),
             (_sx + 14, _sy - 112), (_sx + 16, _sy - 140), (_sx + 20, _sy - 175)]
BRANCHES = [
    # control points, w0, w1, start (s after burst), grow duration
    ([(_sx + 4, 70), (_sx - 20, 58), (_sx - 52, 52), (_sx - 88, 36), (_sx - 120, 30)], 7.0, 1.6, 0.26, 0.42),
    ([(_sx + 9, 54), (_sx + 30, 44), (_sx + 58, 42), (_sx + 92, 26), (_sx + 118, 22)], 6.5, 1.6, 0.30, 0.40),
    ([(_sx + 8, 38), (_sx - 10, 24), (_sx - 38, 16), (_sx - 70, -2)], 6.0, 1.8, 0.34, 0.36),
    ([(_sx + 13, 26), (_sx + 36, 14), (_sx + 60, 4), (_sx + 80, -12)], 5.5, 1.8, 0.37, 0.34),
    ([(_sx + 2, 96), (_sx - 18, 90), (_sx - 38, 80), (_sx - 56, 72)], 4.5, 1.4, 0.22, 0.34),
    ([(_sx + 10, 88), (_sx + 30, 80), (_sx + 48, 76), (_sx + 66, 64)], 4.0, 1.4, 0.24, 0.32),
    ([(_sx + 15, 12), (_sx + 6, -2), (_sx - 4, -14)], 5.0, 2.5, 0.40, 0.25),
]
ROOTS = [
    ([(_sx - 2, _sy + 1), (_sx - 7, _sy + 3), (_sx - 11, _sy + 6), (_sx - 15, _sy + 11)], 4.5, 1.6, 0.05, 0.30),   # toward the robot
    ([(_sx + 2, _sy + 1), (_sx + 5, _sy + 5), (_sx + 6, _sy + 10), (_sx + 9, _sy + 15)], 5.0, 1.8, 0.06, 0.32),   # over the front rim
    ([(_sx + 4, _sy), (_sx + 12, _sy - 1), (_sx + 22, _sy + 1), (_sx + 30, _sy + 4)], 4.0, 1.4, 0.08, 0.34),     # across the soil
    ([(_sx - 1, _sy), (_sx - 4, _sy - 3), (_sx - 7, _sy - 5)], 3.5, 1.4, 0.10, 0.22),
    ([(_sx + 6, _sy + 2), (_sx + 16, _sy + 7), (_sx + 24, _sy + 12), (_sx + 32, _sy + 17)], 3.5, 1.2, 0.12, 0.36),
]


def _build_tree():
    """Returns dict with per-pixel reveal time (s after burst), albedo idx, depth."""
    appear = np.full((SH, SW), np.inf)
    tidx = np.zeros((SH, SW), np.uint8)
    barkc = np.array([P.NAME_TO_IDX[n] for n in BARK_R], np.uint8)
    # trunk: height reveal over 0.45 s (out-cubic), width thickens over 0.8 s
    A, ACR, WID, SHD, L_ = _stroke_fields(TRUNK_PTS, 24.0, 15.0, twist=2.4, flare=8.0)
    m = np.isfinite(A)
    s_h = A / L_
    # invert the easing: time when the growth front reaches this arc
    th = 0.50 * (1 - np.cbrt(np.clip(1 - s_h, 0, 1)))
    # time when the width has thickened enough to cover |across| (a thick column
    # erupting; the last swelling of the flanks trails the front a little)
    tw = th + 0.10 + 0.30 * np.clip(ACR - 0.45, 0, 1) ** 1.2
    tt = np.where(m, np.where(ACR < 0.45, th, tw), np.inf)
    upd = tt < appear
    appear[upd] = tt[upd]
    tidx[upd] = barkc[SHD[upd]]
    trunk_m = m.copy()
    for pts, w0, w1, st, du in BRANCHES + ROOTS:
        A, ACR, WID, SHD, L_ = _stroke_fields(pts, w0 * 1.35, max(w1 * 1.3, 2.2), twist=1.6)
        m = np.isfinite(A)
        s_h = A / L_
        tt = st + du * (1 - np.cbrt(np.clip(1 - s_h, 0, 1))) + 0.25 * np.clip(ACR - 0.4, 0, 1)
        tt = np.where(m, tt, np.inf)
        upd = tt < appear
        appear[upd] = tt[upd]
        tidx[upd] = barkc[SHD[upd]]
    return appear, tidx, trunk_m


TREE_T, TREE_IDX, TRUNK_M = _build_tree()
TREE_Z = 8.1


def _build_leaves(seed=851):
    """Leaf clumps: masses on the outer 2/3 of every branch (bigger toward the
    tips, lifted above the wood) + the crown pressing down from the top of the
    frame. (cx, cy, r, birth_s_after_burst)."""
    cl = []
    rnd = lambda a, b, c: float(noise.rand(a, b, seed + c))
    for bi, (pts, w0, w1, st, du) in enumerate(BRANCHES):
        xs, ys, arc = _resample(_catmull(pts, 12), 1.0)
        n = len(xs)
        j = int(n * 0.30)
        while j < n:
            s_ = j / n
            rr = 7.0 + 9.0 * s_ + 4 * rnd(bi, j, 0)
            ox = (rnd(bi, j, 1) - 0.5) * 9
            oy = -rr * (0.35 + 0.25 * rnd(bi, j, 2))
            birth = st + du * (1 - (1 - s_) ** 3) + 0.03 + 0.07 * rnd(bi, j, 3)
            cl.append((xs[j] + ox, ys[j] + oy, rr, birth))
            # a second, lower clump hanging below some branch stretches
            if rnd(bi, j, 4) < 0.45:
                cl.append((xs[j] + ox * 0.5 + 4, ys[j] + rr * 0.35, rr * 0.7, birth + 0.05))
            j += max(4, int(rr * 0.75))
    # the crown: a mass pressing down from above the frame around the trunk
    for j in range(16):
        a_ = j / 15.0
        x = _sx - 70 + a_ * 170 + (rnd(j, 7, 5) - 0.5) * 14
        y = -8 + 16 * rnd(j, 8, 5) + 10 * math.sin(a_ * math.pi) ** 2
        cl.append((x, y, 14 + 8 * rnd(j, 9, 5), 0.30 + 0.16 * abs(a_ - 0.45)))
    return np.array(cl)


LEAVES = _build_leaves()
_LEAF_TONES = np.array([P.NAME_TO_IDX[n] for n in ('leaf1', 'leaf2', 'leaf3', 'leaf4', 'leaf5', 'leaf6', 'leaf7')], np.uint8)


# ---------------------------------------------------------------- S8a renderers
_SOIL_RICH = np.array([P.NAME_TO_IDX[n] for n in ('bark0', 'bark1', 'bark2', 'bark3')], np.uint8)
_MOSS = [C('meadow0'), C('meadow1'), C('meadow2'), C('leaf2'), C('meadow3')]
_dxz_core = None


def _core_xz():
    global _dxz_core
    if _dxz_core is None:
        lp = world_of_frame(CORE_SPOT[0], CORE_SPOT[1], depth_at_feet(KNEEL[1]))
        _dxz_core = np.hypot(POS[..., 0] - lp[0], (POS[..., 2] - lp[2]) * 1.0)
    return _dxz_core


_VEIN_NEAR = None


def _vein_near():
    """For every pixel: min vein arc-distance within 3 px (for moss / grass growth)."""
    global _VEIN_NEAR
    if _VEIN_NEAR is None:
        from scipy.ndimage import minimum_filter
        D = np.where(np.isfinite(VEIN_D), VEIN_D, 1e6)
        _VEIN_NEAR = minimum_filter(D, size=7)
    return _VEIN_NEAR


def revive_soil(sc, t, front, t_veins=0.8):
    """Colour returns: rich dark soil spreading from the heart, moss creeping over
    the planter stones and grass tufts along the veins."""
    if front <= 0:
        return
    u = t - t_veins
    rad = 1.9 * ease.out_cubic(min(u / 1.6, 1.0))
    d = _core_xz()
    soil = (SID == SID_SOIL) & ~sc.actor & (d < rad + (NT_SOIL - 0.5) * 0.5)
    if soil.any():
        pl = PL_RAND
        k = np.where(pl < 0.3, 1, np.where(pl < 0.8, 2, 3))
        k = np.where(SOIL_CRACK, 0, k)
        sc.img[soil] = P.PAL[_SOIL_RICH[k[soil]]]
    vn = _vein_near()
    reach = front - vn
    # when did the nearest vein pass here? (inverse of the S8a front curve)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_pass = t_veins - 0.45 * np.log(np.clip(1 - vn / 5.6, 1e-3, 1))
    # moss creeps over the planter's stones behind the veins (slowly, in patches)
    stone = ((SID == SID_PSIDE) | (SID == SID_PRIM)) & ~sc.actor & ~PROPM
    age = t - (t_pass + 0.45 + 0.9 * NT_MOSS)
    moss = stone & (age > 0) & (NT_MOSS2 > 0.45) & (vn < 1e5)
    if moss.any():
        lvl = np.clip((age / 0.5 + NT_MOSSL * 0.9), 0, 3.99).astype(int)
        cols = np.array(_MOSS, np.uint8)
        sc.img[moss] = cols[lvl[moss]]
    # tiny grass blades popping along the veins on the floor and soil (later)
    if t > 1.9:
        g = ((SID == SID_FLOOR) | (SID == SID_SOIL)) & ~sc.actor & ~PROPM & (reach > 0.9) & (vn < 1e5)
        g &= (NT_GRASS < 0.05 * min(1.0, (t - 1.9) / 0.6)) & (NT_MOSS2 > 0.5)
        up = shift(g, 0, -1) & ~sc.actor
        up2 = shift(up, 0, -1) & ~sc.actor & (NT_GRASS2 < 0.5)
        sc.img[g] = C('meadow2')
        sc.img[up] = C('meadow3')
        sc.img[up2] = C('meadow4')


def draw_tree(sc, tb, t, amp):
    """Reveal the bark (trunk, branches, roots) and pop the leaf clumps."""
    m = (TREE_T <= tb) & ~sc.robot
    if m.any():
        yy, xx = np.nonzero(m)
        sc.img[yy, xx] = P.PAL[TREE_IDX[yy, xx]]
        X, Y = _world_at(xx, yy, TREE_Z)
        sc.pos[yy, xx, 0] = X
        sc.pos[yy, xx, 1] = Y
        sc.pos[yy, xx, 2] = TREE_Z
        sc.nrm[yy, xx] = (0.35, 0.0, -0.94)
        sc.actor |= m
        sc.robot &= ~m
    # the growth front of the trunk glows white-gold (sap of light)
    front_px = m & ~shift(m, 0, -1) & (TREE_T > tb - 0.12) & TRUNK_M
    sc.img[front_px] = C('gold4')
    sc.emis |= front_px
    return draw_foliage(sc, tb, t, amp)


_VEIN_COLS = {'tip': C('core7'), 'hot': C('core6'), 'warm': C('gold3'), 'mid': C('leaf6'),
              'old': C('leaf5'), 'dim': C('leaf4')}


VEIN_SURF = ((SID == SID_FLOOR) | (SID == SID_SOIL) | (SID == SID_PSIDE) | (SID == SID_PRIM)) & ~PROPM
VEIN_SURF |= RUBBLE_M & ~FALLEN_RIB_M


def _vein_fields():
    """Static geometry of the drawn veins around the 1-px centre lines: every pixel
    within reach takes the arc / width of its nearest centre pixel. Cores are
    1, 2 or 3 px wide (tapering along each root); a soft halo surrounds them.
    Returns flat per-candidate arrays (set coords)."""
    from scipy.ndimage import distance_transform_edt
    dist, (iy, ix) = distance_transform_edt(~VEIN_LINE, return_indices=True)
    wc = VEIN_WPX[iy, ix]
    wcls = np.where(wc < 1.6, 1, np.where(wc < 2.5, 2, 3))
    oy = ys_.astype(np.int64) - iy
    ox = xs_.astype(np.int64) - ix
    core = (dist == 0) | ((wcls == 3) & (dist <= 1.0)) | \
           ((wcls == 2) & (((oy == 1) & (ox == 0)) | ((oy == 0) & (ox == 1))))
    rc = np.where(wcls == 1, 0.0, np.where(wcls == 2, 0.6, 1.0))
    reach = 1.2 + 0.7 * wcls
    h0 = np.clip(1.0 - (dist - rc) / reach, 0.0, 1.0) ** 1.4 * ~core
    same = np.abs(POS[..., 2] - POS[iy, ix, 2]) < 0.8          # no glow across depth jumps
    cand = VEIN_SURF & same & (core | (h0 > 0.02))
    yy, xx = np.nonzero(cand)
    j = (iy[yy, xx], ix[yy, xx])
    return dict(yy=yy, xx=xx, dc=VEIN_D[j], core=core[yy, xx], centre=VEIN_LINE[yy, xx],
                edge=core[yy, xx] & ~VEIN_LINE[yy, xx], h0=h0[yy, xx], w=wcls[yy, xx])


VF = _vein_fields()
_VT = [(x, y, d) for (x, y, d) in VEIN_TIPS if VEIN_SURF[y, x]]
VT_X = np.array([p_[0] for p_ in _VT], np.int64)
VT_Y = np.array([p_[1] for p_ in _VT], np.int64)
VT_D = np.array([p_[2] for p_ in _VT])
# core colours by age: (centre, edge) -- white-hot racing tips, gold, then living green
_VC = {k: (np.array(C(a), np.uint8), np.array(C(b), np.uint8)) for k, (a, b) in {
    'tip': ('white1', 'core7'), 'hot': ('core7', 'core6'), 'warm': ('gold4', 'gold3'),
    'mid': ('leaf7', 'leaf6'), 'old': ('leaf6', 'leaf5'), 'dim': ('leaf5', 'leaf4')}.items()}


def _add_glow(img, yy, xx, level, tint, allowed):
    """Additive dithered glow on final colours at flat pixel lists (levels >= 0)."""
    q = quantize(level, band=1.0, th=BAY[yy, xx])
    k = q > 0
    if not k.any():
        return
    yy, xx, q = yy[k], xx[k], np.minimum(q[k], MAXL)
    idx = to_index(img[yy, xx][:, None, :])[:, 0]
    lt = add_lut(tint, allowed)
    img[yy, xx] = P.PAL[lt[q, idx]]


def draw_veins(img, t, front, vm, occ=None, allowed='alive'):
    """Emissive roots of light with body: 1-3 px luminous cores that taper toward
    the tips (centre brighter than the edges), a soft dithered glow halo on the
    ground around them, white-hot racing fronts, bright buds at the root tips.
    White-hot at the racing tips, gold behind, settling to living green; a slow
    pulse travels outward along them. Drawn only on the ground surfaces (floor,
    soil, planter, rubble) and never over anything standing in front (occ)."""
    if front <= 0 or not vm.any():
        return
    f = VF
    rev = f['dc'] <= front
    if occ is not None:
        rev &= ~occ[f['yy'], f['xx']]
    if not rev.any():
        return
    yy, xx = f['yy'][rev], f['xx'][rev]
    dc = f['dc'][rev]
    age = front - dc
    wave = np.sin((dc - t * 2.2) * 5.0)
    core, edge, h0, w = f['core'][rev], f['edge'][rev], f['h0'][rev], f['w'][rev]
    # ---- halo: light spilling onto the ground (gold while young, green later)
    hot = np.clip(1.0 - age / 1.2, 0.0, 1.0)
    kmax = 1.5 + 2.0 * hot + 0.8 * (wave > 0.6) * (age >= 1.2)
    lvl = h0 * kmax
    hm = ~core & (lvl > 0.05)
    young = hm & (age < 0.9)
    old = hm & (age >= 0.9)
    if young.any():
        _add_glow(img, yy[young], xx[young], lvl[young], 'gold', allowed)
    if old.any():
        _add_glow(img, yy[old], xx[old], lvl[old], 'green', allowed)
    # ---- cores
    cm = core
    TIP, HOT, WARM, MID, OLD, DIM = range(6)
    key = np.full(len(yy), OLD, np.int8)
    key[age < 2.2] = MID
    key[(age >= 2.2) & (wave > 0.55)] = MID
    key[(age >= 1.2) & (wave > 0.82)] = WARM            # gold pulses running outward
    key[age < 1.2] = WARM
    key[age < 0.35] = HOT
    key[age < 0.10] = TIP
    key[(age >= 2.2) & (w == 1) & (wave < -0.5)] = DIM
    for k_, name in enumerate(('tip', 'hot', 'warm', 'mid', 'old', 'dim')):
        cc, ce = _VC[name]
        m = cm & (key == k_)
        if m.any():
            img[yy[m & ~edge], xx[m & ~edge]] = cc
            img[yy[m & edge], xx[m & edge]] = ce
    # ---- bright buds at the tips of the roots once the light has reached them
    if len(VT_X):
        on = VT_D <= front
        if occ is not None:
            on &= ~occ[VT_Y, VT_X]
        for x, y, d in zip(VT_X[on], VT_Y[on], VT_D[on]):
            a_ = front - d
            tw = 0.5 + 0.5 * math.sin(t * 7.0 + d * 13.0)
            cc = 'white1' if a_ < 0.25 else ('leaf8' if tw > 0.35 else 'leaf7')
            img[y, x] = C(cc)
            if a_ < 0.6 or tw > 0.7:
                arm = 'core6' if a_ < 0.25 else 'leaf7'
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    xi, yi = x + dx, y + dy
                    if 0 <= xi < SW and 0 <= yi < SH and VEIN_SURF[yi, xi] and \
                            (occ is None or not occ[yi, xi]):
                        img[yi, xi] = C(arm)


def life_motes(img, t, tb, E_life):
    """Motes of light rising from the veins; after the burst they swirl up
    around the trunk. Falling dust + glass grit from the trembling dome."""
    if t > 0.9:
        n = 45 if tb < 0 else 80
        i = np.arange(n)
        ph = noise.rand(i, 1, 891)
        spd = 14 + 30 * noise.rand(i, 2, 891) + (40 if tb >= 0 else 0)
        age = (t * (0.6 + 0.8 * noise.rand(i, 3, 891)) + ph) % 1.0
        if tb < 0:
            x0 = CORE_SPOT[0] + (noise.rand(i, 4, 891) - 0.5) * 150
            y0 = CORE_SPOT[1] + 5 + (noise.rand(i, 5, 891) - 0.3) * 25
            x = x0 + np.sin(t * 2 + ph * 6) * 3
            y = y0 - age * spd
        else:
            ang = ph * 6.283 + t * (2.0 + 1.5 * noise.rand(i, 6, 891))
            rad = 8 + 38 * noise.rand(i, 7, 891) * (1 - 0.3 * age)
            x = SPROUT_XY[0] + 8 + np.cos(ang) * rad
            y = SPROUT_XY[1] - age * (spd * 3.2) + np.sin(ang) * rad * 0.25
        vis = age < 0.9
        c = np.where(noise.rand(i, 8, 891) < 0.35, 1, 0)
        for k in np.nonzero(vis)[0]:
            xi, yi = int(x[k]) + M, int(y[k]) + M
            if 0 <= xi < SW and 0 <= yi < SH:
                img[yi, xi] = C('gold4') if c[k] else C('leaf7')
                if age[k] < 0.25 and xi + 1 < SW:
                    img[yi, xi + 1] = C('white1') if c[k] else C('leaf8')
    if tb > 0.1:
        # dust and glass grit shaken loose from the dome
        n = 60
        i = np.arange(n)
        t0 = 0.1 + noise.rand(i, 1, 893) * 1.0
        a = tb - t0
        ok = a > 0
        x = noise.rand(i, 2, 893) * W
        y = -5 + 0.5 * 170 * a ** 2 + 20 * a
        for k in np.nonzero(ok & (y < H))[0]:
            xi, yi = int(x[k]) + M, int(y[k]) + M
            if 0 <= xi < SW and 0 <= yi < SH:
                img[yi, xi] = C('glass4') if k % 3 == 0 else C('dust6')


def flash_frame(img, x, y, tb):
    """1-2 impact frames: a white-gold star at the base of the tree."""
    X, Y = int(x) + M, int(y) + M
    r = 16 if tb < 0.035 else 10
    yy, xx = np.ogrid[:SH, :SW]
    d = np.sqrt((xx - X) ** 2 + ((yy - Y) * 1.3) ** 2)
    img[d < r * 0.45] = C('white1')
    img[(d < r) & (d >= r * 0.45) & (BAY < 0.6)] = C('gold4')
    for k in range(8):
        a = k * math.pi / 4 + 0.2
        ln = r * (2.2 if k % 2 == 0 else 1.4)
        xs, ys = draw.line_points(X, Y, X + math.cos(a) * ln, Y + math.sin(a) * ln * 0.8)
        for px, py in zip(xs, ys):
            if 0 <= px < SW and 0 <= py < SH:
                img[py, px] = C('white1')


_SHARD_PIX = np.argwhere((MAT == 1) & (SID == SID_FLOOR))


def tremble_shards(img, t, amp):
    """Floor shards rattle: they flick sideways and glint."""
    f = int(t * 30)
    for k, (y, x) in enumerate(_SHARD_PIX):
        r = noise.rand(k, f, 897)
        if r < 0.35 * min(amp / 3.0, 1.0):
            dx = 1 if r < 0.17 else -1
            if 0 <= x + dx < SW:
                img[y, x + dx] = img[y, x]
                img[y, x] = C('glass4') if r < 0.08 else img[y, x]


# ---------------------------------------------------------------- static noise textures (speed)
NT_JAG = noise.vnoise2(FX / 1.7, FY / 1.7, 891)
NT_TEX = noise.vnoise2(FX / 1.3, FY / 1.3, 893)
NT_SOIL = noise.vnoise2(FX / 3.0, FY / 2.0, 871)
NT_MOSS = noise.fbm2(FX / 4.0, FY / 3.0, 875, 3)
NT_MOSS2 = noise.fbm2(FX / 3.0, FY / 2.0, 876, 2)
NT_MOSSL = noise.vnoise2(FX / 2.0, FY / 2.0, 877)
NT_GRASS = noise.rand(xs_.astype(int), ys_.astype(int), 879)
NT_GRASS2 = noise.rand(xs_.astype(int), ys_.astype(int), 880)
PL_RAND = noise.rand(SOIL_PLATE, 5, 873)


# ---------------------------------------------------------------- foliage: clumped, shaded spheres
def _build_spheres(seed=861):
    """Sub-spheres of every leaf clump: (cx, cy, cz, r, birth, host_x, host_y)."""
    S = []
    for c, (cx, cy, R, birth) in enumerate(LEAVES):
        n = 4 + int(R / 3.2)
        for k in range(n):
            a = float(noise.rand(c * 17 + k, 1, seed)) * 2 * math.pi
            dd = R * 0.58 * math.sqrt(float(noise.rand(c * 17 + k, 2, seed)))
            ox, oy = math.cos(a) * dd, math.sin(a) * dd * 0.75 - R * 0.1
            r = R * (0.38 + 0.24 * float(noise.rand(c * 17 + k, 3, seed)))
            cz = (1 - dd / max(R, 1)) * R * 0.6 + float(noise.rand(c * 17 + k, 4, seed)) * 2 - oy * 0.3
            S.append((cx + ox, cy + oy, cz, r, birth + 0.06 * float(noise.rand(c * 17 + k, 5, seed)), cx, cy))
    return np.array(S)


LEAF_S = _build_spheres()
_ks = np.arange(len(LEAF_S))
LEAF_R = np.stack([noise.rand(_ks, j, 895 + (j > 1) * (j - 1)) for j in range(2)] +
                  [noise.rand(_ks, b, 896) for b in range(5)] +
                  [noise.rand(_ks, b, 897) for b in range(5)] +
                  [noise.rand(_ks, b, 898) for b in range(5)], 1)
_LT = np.array([P.NAME_TO_IDX[n] for n in ('leaf1', 'leaf2', 'leaf3', 'leaf4', 'leaf5', 'leaf6', 'leaf7')], np.uint8)
_NT_LEAF = None


def draw_foliage(sc, tb, t, amp):
    """Z-buffered clumps of leaf spheres: top-lit (dawn from above-right),
    under-lit by the veins' glow, texture + contact shadows, pink/white blossoms."""
    S = LEAF_S
    age = tb - S[:, 4]
    live = age > 0
    if not live.any():
        return np.zeros((SH, SW), bool)
    g = np.clip(age / 0.26, 0, 1)
    s_ = 1.70158 * 1.4
    gg = g - 1
    pop = np.where(age <= 0, 0.0, gg * gg * ((s_ + 1) * gg + s_) + 1)
    pe = 1 - (1 - np.clip(age / 0.32, 0, 1)) ** 3
    cx = S[:, 5] + (S[:, 0] - S[:, 5]) * pe + amp * 0.25 * np.sin(t * 9 + S[:, 0] * 0.1)
    cy = S[:, 6] + (S[:, 1] - S[:, 6]) * pe
    r = S[:, 3] * pop
    idx = np.nonzero(live & (r > 0.6))[0]
    zb = np.full((SH, SW), -1e9)
    sid = np.full((SH, SW), -1, np.int32)
    sh = np.zeros((SH, SW))
    for k in idx:
        X, Y, Z, rr = cx[k] + M, cy[k] + M, S[k, 2], r[k]
        x0, x1 = int(max(X - rr - 2, 0)), int(min(X + rr + 3, SW))
        y0, y1 = int(max(Y - rr - 2, 0)), int(min(Y + rr + 3, SH))
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx_, dy_ = (xx + 0.5 - X) / rr, (yy + 0.5 - Y) / rr
        jag = (NT_JAG[yy, (xx + k * 37) % SW] - 0.5) * 0.5
        d2 = dx_ * dx_ + dy_ * dy_
        inside = np.sqrt(d2) + jag * 0.6 < 1.0
        nz = np.sqrt(np.clip(1 - d2, 0, 1))
        z = Z + nz * rr
        sub_z = zb[y0:y1, x0:x1]
        upd = inside & (z > sub_z)
        if not upd.any():
            continue
        sub_z[upd] = z[upd]
        sid[y0:y1, x0:x1][upd] = k
        lit = 0.45 * dx_ - 0.75 * dy_ + 0.45 * nz            # sky from above (right)
        under = np.clip(dy_, 0, 1) * 0.9                    # the veins' glow from below
        tex = NT_TEX[yy, (xx + k * 53) % SW] - 0.5
        v = 1.9 + 2.4 * lit + 0.6 * under + 1.1 * tex - 1.3 * (np.sqrt(d2) > 0.86)
        sh[y0:y1, x0:x1][upd] = v[upd]
    m = sid >= 0
    m &= ~sc.robot
    if not m.any():
        return m
    # contact shadow: a sphere overlapped by a nearer one up-left of it
    zp = np.pad(zb, ((2, 0), (2, 0)), constant_values=-1e9)[:-2, :-2]
    occ = (zp - zb) > 2.0
    sh = sh - 1.2 * occ
    yy, xx = np.nonzero(m)
    q = np.clip(np.floor(sh[yy, xx] + (BAY[yy, xx] - 0.5) * 0.5 + 0.5), 0, 6).astype(int)
    sc.img[yy, xx] = P.PAL[_LT[q]]
    # dark underside outline where the canopy ends below
    bot = m & ~shift(m, 0, -1)
    sc.img[bot] = C('leaf1')
    X_, Y_ = _world_at(xx, yy, TREE_Z - 0.3)
    sc.pos[yy, xx, 0] = X_
    sc.pos[yy, xx, 1] = Y_
    sc.pos[yy, xx, 2] = TREE_Z - 0.3
    sc.nrm[yy, xx] = (0.3, 0.5, -0.8)
    sc.actor |= m
    sc.emis |= m                   # foliage carries its own baked light model
    # blossoms: pink & white, on the lit upper parts, arriving a beat after the leaves
    for k in idx:
        if age[k] < 0.16 or LEAF_R[k, 0] > 0.62:
            continue
        nb = 1 + int(LEAF_R[k, 1] * 4)
        for b in range(nb):
            a = -2.2 + (LEAF_R[k, 2 + b] - 0.5) * 2.4
            rr = r[k] * (0.2 + 0.6 * LEAF_R[k, 7 + b])
            bx = int(round(cx[k] + math.cos(a) * rr)) + M
            by = int(round(cy[k] + math.sin(a) * rr)) + M
            if not (1 <= bx < SW - 1 and 1 <= by < SH - 1) or sid[by, bx] != k:
                continue
            pa = age[k] - 0.16 - 0.04 * b
            if pa < 0:
                continue
            v = LEAF_R[k, 12 + b]
            c0 = 'white1' if pa < 0.05 else ('bloom5' if v < 0.3 else ('white0' if v < 0.45 else 'bloom4'))
            c1 = 'bloom3' if v >= 0.45 else 'bloom4'
            sc.img[by, bx] = C(c0)
            sc.emis[by, bx] = True
            if sh[by, bx] > 2.2:
                for (ox, oy) in ((1, 0), (0, 1), (-1, 0), (0, -1)) if v < 0.7 else ((1, 0), (0, 1), (1, 1)):
                    if sid[by + oy, bx + ox] == k:
                        sc.img[by + oy, bx + ox] = C(c1)
                        sc.emis[by + oy, bx + ox] = True
    return m


# ---------------------------------------------------------------- S7b: gold on the iron
_IRON_ALL = FRAME_IRON | FALLEN_RIB_M
_IR_L = _IRON_ALL & ~shift(_IRON_ALL, 1, 0)          # left edge pixel of iron runs
_IR_R = _IRON_ALL & ~shift(_IRON_ALL, -1, 0)         # right edge
_IR_B = _IRON_ALL & ~shift(_IRON_ALL, 0, -1)         # bottom edge
_IR_T = _IRON_ALL & ~shift(_IRON_ALL, 0, 1)          # top edge


def iron_glint(lx, ly, E):
    """Level bonus for iron edges that face a light at frame (lx, ly): the sides of
    the ribs turned toward the heart and the undersides of bars catch it."""
    face = (_IR_R & (FX < lx - 1)) | (_IR_L & (FX > lx + 1)) | (_IR_B & (FY < ly - 1)) | (_IR_T & (FY > ly + 1))
    return np.where(face, E, 0.0)


def orb_halo(img, x, y, power, t, occ=None):
    """Glow + short anamorphic flare around the heart held in the robot's hands
    (the orb itself is drawn by the robot sprite). Clean concentric rings with a
    sparse dithered rim -- not a checkerboard."""
    X, Y = int(round(x)) + M, int(round(y)) + M
    beat = 0.5 + 0.5 * math.sin(t * 2 * math.pi * 1.2) ** 8
    r1 = 2.6 + 0.6 * power
    r2 = 3.6 + 1.2 * power + 0.6 * beat
    yy, xx = np.ogrid[:SH, :SW]
    d = np.sqrt((xx - X) ** 2 + ((yy - Y) * 1.2) ** 2)
    keep = np.ones((SH, SW), bool) if occ is None else ~occ
    ring1 = (d >= 2.2) & (d < r1) & keep
    ring2 = (d >= r1) & (d < r2) & keep & ((BAY < 0.5) | (d < r2 - 1.0))
    img[ring2] = C('core5')
    img[ring1] = C('core6')
    if power > 0.4:
        ln = int(3 + 5 * power)
        for dx in range(-ln, ln + 1):
            if abs(dx) <= 2 or not (0 <= X + dx < SW):
                continue
            fade = 1 - abs(dx) / (ln + 1)
            if fade > 0.5 or BAY[Y, X + dx] < fade * 1.2:
                img[Y, X + dx] = C('core7') if fade > 0.5 else C('core6')
    return img


def god_rays(x, y, t, n=19, seed=951, length=260.0, spin=0.05):
    """Volumetric rays radiating from frame point (x, y) through dusty air:
    returns 0..1 intensity map (soft bands, slowly turning, flickering)."""
    ang = (np.arange(n) / n) * 2 * math.pi + spin * t + noise.rand(np.arange(n), 1, seed) * 0.5
    wid = 0.012 + 0.022 * noise.rand(np.arange(n), 2, seed)
    fl = 0.6 + 0.4 * np.sin(t * (2 + 3 * noise.rand(np.arange(n), 3, seed)) + np.arange(n))
    dx = FX - x
    dy = (FY - y) * 1.0
    a = np.arctan2(dy, dx)
    dist = np.sqrt(dx * dx + dy * dy)
    lv = np.zeros((SH, SW))
    for k in range(n):
        da = np.abs((a - ang[k] + math.pi) % (2 * math.pi) - math.pi)
        lv = np.maximum(lv, np.clip(1 - da / wid[k], 0, 1) * fl[k])
    fall = np.clip(1 - dist / length, 0, 1) ** 1.3 * np.clip(dist / 10.0, 0, 1)
    return lv * fall


def shaft_motes(img, t7, sh, E_gold=None, fade=1.0):
    """Dust motes glittering in the pale dawn shafts (and in the gold light).
    t7 = S7's clock (S8a passes 7.0 + t) so the motes drift on across the cut."""
    mx_, my_, mr_ = motes(t7, 140, 721, (-10, -10, 340, 190), vel=(-1.2, 1.6), sway=(1.5, 0.25))
    for i in range(len(mx_)):
        xi, yi = int(mx_[i]) + M, int(my_[i]) + M
        if not (0 <= xi < SW and 0 <= yi < SH):
            continue
        if E_gold is not None and E_gold[yi, xi] > en(1.0) and mr_[i] < 0.6:
            img[yi, xi] = C('core6') if E_gold[yi, xi] > en(2.5) else C('core4')
        elif sh[yi, xi] > 0.4 and mr_[i] < 0.5 * fade:
            img[yi, xi] = C('ash7') if mr_[i] < 0.25 else C('glass3')


DAWN_SHAFTS = [(282, -12, 13, -0.44, 1.0, 185, 1.0), (244, -12, 5, -0.44, 1.0, 150, 0.6)]


def dawn_shafts():
    """0..1 map of the pale dawn shafts falling through the broken dome (S7 / S8a)."""
    a = [shaft_level(x0, y0, w, dx, dy, ln) * st for (x0, y0, w, dx, dy, ln, st) in DAWN_SHAFTS]
    return np.maximum(a[0], a[1])


def dawn_motes(img, t, sh, occ=None, n=90, seed=741):
    """Dust motes drifting slowly down through the dawn shafts and glittering as
    they turn in the light (dim ash specks, now and then a white glint). Visible
    only inside the shafts (sh); never drawn over occ / the foreground."""
    for si, (x0, y0, wd, dx, dy, ln, st) in enumerate(DAWN_SHAFTS):
        d = np.array([dx, dy], np.float64)
        d /= np.linalg.norm(d)
        k = int(n * (1.0 if si == 0 else 0.4))
        i = np.arange(k)
        r = [noise.rand(i, j, seed + 10 * si) for j in range(7)]
        along = (r[0] * ln + t * (2.5 + 3.5 * r[1])) % ln              # sinking down the beam
        w = wd * (0.7 + 0.6 * along / ln)
        across = (r[2] - 0.5) * 1.7 * w + np.sin(t * (0.7 + 0.8 * r[3]) + r[4] * 6.283) * 1.6
        x = x0 + d[0] * along - d[1] * across
        y = y0 + d[1] * along + d[0] * across
        tw = np.sin(t * (1.6 + 2.6 * r[5]) + r[6] * 6.283 * 4)         # turning facets
        xi = np.floor(x).astype(np.int64) + M
        yi = np.floor(y).astype(np.int64) + M
        ok = (xi >= 0) & (xi < SW) & (yi >= 0) & (yi < SH) & (tw > -0.35)
        xi, yi, tw = xi[ok], yi[ok], tw[ok]
        vis = (sh[yi, xi] > 0.22) & ~FGM[yi, xi]
        if occ is not None:
            vis &= ~occ[yi, xi]
        for x_, y_, tw_, s_ in zip(xi[vis], yi[vis], tw[vis], sh[yi[vis], xi[vis]]):
            if tw_ > 0.93 and s_ > 0.45:
                img[y_, x_] = C('white1' if tw_ > 0.985 else 'white0')      # a glint
                if tw_ > 0.985:                                               # ... a tiny star
                    for ax_, ay_ in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        xa, ya = x_ + ax_, y_ + ay_
                        if 0 <= xa < SW and 0 <= ya < SH and not FGM[ya, xa] and \
                                (occ is None or not occ[ya, xa]):
                            img[ya, xa] = C('ash8')
            elif tw_ > 0.55 or s_ > 0.7:
                img[y_, x_] = C('ash8')
            else:
                img[y_, x_] = C('ash7')


def glass_glint(img, t, t0, x, y, dur=0.4, size=2, occ=None):
    """A star-shaped glint on the glass (a shard catching the light): the arms
    grow to `size` px and shrink again over `dur` s."""
    u = (t - t0) / dur
    if not (0.0 <= u < 1.0):
        return
    e = math.sin(math.pi * u)
    ln = int(round(e * size + 0.2))
    X, Y = int(x) + M, int(y) + M
    pix = [(X, Y, 'white1' if e > 0.45 else 'white0')]
    for k in range(1, ln + 1):
        cc = 'white0' if k < ln or ln == 1 else 'glass4'
        pix += [(X + k, Y, cc), (X - k, Y, cc), (X, Y + k, cc), (X, Y - k, cc)]
    if e > 0.7:
        pix += [(X + 1, Y + 1, 'glass4'), (X - 1, Y - 1, 'glass4'), (X + 1, Y - 1, 'glass4'), (X - 1, Y + 1, 'glass4')]
    for (xi, yi, cc) in pix:
        if 0 <= xi < SW and 0 <= yi < SH and GLASS[yi, xi] and (occ is None or not occ[yi, xi]):
            img[yi, xi] = C(cc)


def sprite_mask(spr, x, y, anchor='feet'):
    """Footprint (set coords) a sprite would cover if placed at frame (x, y)."""
    ax, ay = spr.anchor(anchor)
    px, py = int(round(x)) - ax + M, int(round(y)) - ay + M
    m = np.zeros((SH, SW), bool)
    x0, y0 = max(px, 0), max(py, 0)
    x1, y1 = min(px + spr.w, SW), min(py + spr.h, SH)
    if x0 < x1 and y0 < y1:
        m[y0:y1, x0:x1] = spr.img[y0 - py:y1 - py, x0 - px:x1 - px, 3] > 127
    return m


def buried_breath(sc, dt, Zr):
    """Faint light of the buried heart breathing under the soil (energy map).
    dt = seconds since burial. Shared by the end of S7 and S8a (continuity)."""
    hx, hy = CORE_SPOT
    lp = world_of_frame(hx, hy + 1, Zr - 0.02)
    lp[1] = HS - 0.05
    a = min(1.0, max(0.0, (dt - 0.6) / 0.8))
    return a * light_level(sc.pos, sc.nrm, lp, peak=-0.5 + 0.4 * math.sin(dt * 2.2) ** 2,
                           slope=6.0, reach=0.7, wrap=0.9)


def grief_dim(t_global_s7):
    """0..1 how far the room has sunk toward darkness after the heart is gone.
    Driven by S7's clock (S8a passes 7.0 + t): starts when the eye dies (6.3),
    deepest by 7.3, then held until the life light returns."""
    return ease.smoothstep(ease.prog(t_global_s7, 6.25, 7.35))


def grief_grade(img, gd, keep=None):
    """Whole-frame grade toward darkness (glass included), deeper at the edges.
    keep: mask of pixels left untouched (glows)."""
    if gd <= 0.01:
        return img
    lvmap = -gd * (0.7 + 1.1 * VIGN)
    q = quantize(lvmap, band=0.12)
    idx = to_index(np.ascontiguousarray(img))
    lt = tint_lut((0.92, 0.96, 1.0), 'dead')          # palette-true, hue-keeping darkening
    out = from_index(lt[q + MAXL, idx])
    if keep is not None:
        out[keep] = img[keep]
    return out


def dust_puff(img, x, y, t, t0, n=10, seed=0, spread=7.0, rise=4.0, life=0.55,
              cols=('dust5', 'dust4'), lit=None):
    """Little ground-dust puff (clusters that swell, drift up and thin out)."""
    u = (t - t0) / life
    if not (0 <= u < 1):
        return
    i = np.arange(n)
    ang = (noise.rand(i, 1, seed) - 0.5) * math.pi * 0.9
    sp = 0.4 + 0.6 * noise.rand(i, 2, seed)
    e = 1 - (1 - u) ** 2
    px = x + np.sin(ang) * spread * sp * e
    py = y - rise * sp * e - 0.5
    keep = noise.rand(i, 3, seed) > u * 0.9
    for k in np.nonzero(keep)[0]:
        xi, yi = int(round(px[k])) + M, int(round(py[k])) + M
        if 0 <= xi < SW - 1 and 0 <= yi < SH:
            c = C(cols[0] if (k + int(u * 4)) % 3 else cols[1])
            if lit is not None:
                c = C(lit[0] if k % 2 else lit[1])
            img[yi, xi] = c
            if u < 0.45 and noise.rand(k, 4, seed) < 0.5:
                img[yi, xi + 1] = c


def spark_burst(img, x, y, t, t0, n=16, seed=0, speed=(20, 55), life=(0.25, 0.5), gravity=90.0,
                spread=(-math.pi, 0.0), cols=('core7', 'core6', 'core5')):
    """Hot sparks flying from a point (gold/white), arcs under gravity."""
    if t < t0 or t > t0 + life[1]:
        return
    xs_, ys_, af, alive, r_ = fx.bursts(t, n, seed, (x, y), speed=speed, life=life,
                                        gravity=gravity, spread=spread, t0=t0)
    for k in np.nonzero(alive)[0]:
        xi, yi = int(xs_[k]) + M, int(ys_[k]) + M
        if 0 <= xi < SW and 0 <= yi < SH:
            img[yi, xi] = C(cols[0] if af[k] < 0.3 else (cols[1] if af[k] < 0.65 else cols[2]))
