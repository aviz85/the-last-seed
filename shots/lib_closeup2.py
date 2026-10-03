"""
lib_closeup2.py — helpers for the polished close-ups c01, c03, c04 (owner:
CLOSEUPS polish). Palette-index space like lib_closeup (imported read-only).

  rust_bloom / seam_rust / rivet_rust   rust ONLY where water sits: in seams,
                                        around rivet feet, along bottom edges
                                        (never streaks running down a face)
  RampSnap                              physical light (lightness + warm share)
                                        -> hand-picked cold / warm ramps per
                                        material, narrow dithered seams
  HandRig                               c02's hand (steel 3-finger hand, mint
                                        back plate / cuff) as articulated
                                        capsule chains, any pose, with normals
"""
import math
import numpy as np
from engine.core import bayer
from engine import noise as NZ
from shots.lib_closeup import (ci, cis, shift, quantize, clean, edge_in, edge_out, edge_side, shifted,
                               dilate, erode, thick_curve, PAL, NPAL)

# ------------------------------------------------------------------ rust (seams, rivets, bottom edges)
RUST_DARK = ('rust1', 'ochre2', 'rust2')          # crevice rust: brown, not blood-red
RUST_CRUST = 'rust3'                              # the odd crusty fleck on a lit edge


def rust_bloom(img, x, y, rx, ry, seed=0, mask=None, lit=(-0.6, -0.8), crust=True):
    """A small, clustered patch of surface rust (elliptical, wider than tall:
    rust spreads ALONG a seam, it does not run down). Dark brown core, a warmer
    crusty fleck on the lit side, the paint around it a touch darker."""
    h, w = img.shape
    x0, x1 = int(max(x - rx - 2, 0)), int(min(x + rx + 3, w))
    y0, y1 = int(max(y - ry - 2, 0)), int(min(y + ry + 3, h))
    if x0 >= x1 or y0 >= y1:
        return img
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    n_ = NZ.vnoise2(xx / 1.7, yy / 1.7, seed)
    d = np.sqrt(((xx - x) / max(rx, 0.5)) ** 2 + ((yy - y) / max(ry, 0.5)) ** 2)
    core = d < 0.62 + 0.55 * n_
    halo = (d < 1.05 + 0.45 * n_) & ~core
    if mask is not None:
        mk = mask[y0:y1, x0:x1]
        core &= mk
        halo &= mk
    sub = img[y0:y1, x0:x1]
    # stain: the paint around the rust is one step dirtier (checker thinned)
    th = bayer(y1 - y0, x1 - x0, 2, x0, y0)
    stain = halo & (th < 0.5)
    sub[stain] = shift(sub, -1)[stain]
    col = np.where(n_ > 0.62, ci('rust2'), np.where(n_ > 0.35, ci('ochre2'), ci('rust1')))
    sub[core] = col[core]
    if crust:
        # one or two crusty flecks on the lit side of the patch
        side = (xx - x) * lit[0] + (yy - y) * lit[1]
        fl = core & (side > 0.25 * max(rx, ry)) & (n_ > 0.55)
        sub[fl] = ci(RUST_CRUST)
    return img


def seam_rust(img, seam, mask=None, seed=0, density=0.5, scale=5.0):
    """Rust living IN a seam line (bool mask of the crevice pixels): broken
    dashes (runs of several px) of dark rust whose colour drifts slowly along
    the seam; a crusty fleck only at the heart of the longest runs."""
    h, w = img.shape
    ys_, xs_ = np.nonzero(seam)
    if len(xs_) == 0:
        return img
    n1 = NZ.vnoise2(xs_ / scale, ys_ / scale, seed)
    n3 = NZ.vnoise2(xs_ / 3.0, ys_ / 3.0, seed + 1)
    on = n1 > 1.0 - density
    col = np.where(n3 > 0.68, ci('rust2'), np.where(n3 > 0.40, ci('ochre2'), ci('rust1')))
    core = on & (n1 > 1.0 - 0.3 * density)
    col = np.where(core & (n3 > 0.62), ci('rust2'), col)
    col = np.where(core & (n1 > 1.0 - 0.12 * density) & (n3 > 0.78), ci(RUST_CRUST), col)
    ok = on if mask is None else on & mask[ys_, xs_]
    img[ys_[ok], xs_[ok]] = col[ok]
    return img


def rivet_rust(img, x, y, r=2.0, seed=0, mask=None, sides=((1, 1), (0, 1), (1, 0), (-1, 1), (2, 1), (1, 2))):
    """rust gathered at the FOOT of a domed rivet (below / lower-right: water
    sits there), a few pixels, no run-off."""
    h, w = img.shape
    for k, (dx, dy) in enumerate(sides):
        px, py = int(round(x + dx * r)), int(round(y + dy * r))
        if not (0 <= px < w and 0 <= py < h):
            continue
        if mask is not None and not mask[py, px]:
            continue
        v = float(NZ.rand(k, px * 7 + py, seed))
        if v < 0.15:
            continue
        img[py, px] = ci('rust2' if v > 0.62 else ('rust1' if v > 0.35 else 'ochre2'))
    return img


def edge_rust(img, region, depth=2, seed=0, density=0.55, scale=3.0, side=(0, 1)):
    """rust along a bottom (or given side) edge of `region`: a broken band
    `depth` px deep just inside the edge, heavier at the very edge."""
    e1 = edge_side(region, side[0], side[1])
    band = e1.copy()
    cur = e1
    for d in range(1, depth):
        cur = shifted(cur, -side[0], -side[1]) & region
        band |= cur
    ys_, xs_ = np.nonzero(band)
    if len(xs_) == 0:
        return img
    n1 = NZ.vnoise2(xs_ / scale, ys_ / scale, seed)
    n3 = NZ.vnoise2(xs_ / 3.0, ys_ / 3.0, seed + 3)
    at_edge = e1[ys_, xs_]
    # clusters: the inner row only where the edge row is already heavy
    on = n1 > (1.0 - density) + 0.22 * (~at_edge)
    col = np.where(at_edge, np.where(n3 > 0.55, ci('rust1'), ci('rust0')),
                   np.where(n3 > 0.66, ci('rust2'), ci('ochre2')))
    col = np.where(on & ~at_edge & (n1 > 0.93) & (n3 > 0.7), ci(RUST_CRUST), col)
    img[ys_[on], xs_[on]] = col[on]
    return img


# ================================================================== LIT RENDER (linear light -> family palette)
from engine import palette as _P

LIN = _P.PAL_LIN                     # (N,3) linear rgb of every palette colour
LAB = _P.PAL_LAB                     # (N,3) OKLab
_RAMP_NAME = list(_P.RAMP_OF)


def names_of(spec):
    """'paint0-6 ink1 core3-7' -> list of palette names."""
    out = []
    for tok in spec.split():
        if '-' in tok:
            base = tok.rstrip('0123456789-')
            a, b = tok[len(base):].split('-')
            out += [f'{base}{k}' for k in range(int(a), int(b) + 1)]
        else:
            out.append(tok)
    return out


# ================================================================== THE GARDENER'S HAND (c02's design, articulated)
def _box_seg(xx, yy, x0, y0, x1, y1, r, ends=0.55, p=5.0, q=2.4):
    """c02's mechanical segment: a rounded box from (x0,y0) to (x1,y1), half
    thickness r. Returns (mask, across -1..1, along 0..1)."""
    dx, dy = x1 - x0, y1 - y0
    L = max(math.hypot(dx, dy), 1e-6)
    ux, uy = dx / L, dy / L
    a = (xx - x0) * ux + (yy - y0) * uy
    b = (xx - x0) * (-uy) + (yy - y0) * ux
    half = L / 2 + r * ends
    f = np.abs((a - L / 2) / half) ** p + np.abs(b / r) ** q
    return f <= 1.0, np.clip(b / r, -1, 1), np.clip(a / L, 0, 1), (ux, uy)


class HandRig:
    """Rasterise c02's hand (steel wrist ring, MINT back plate over the palm,
    three segmented steel fingers with round knuckle caps and pins) from a
    list of parts, painter-ordered by z, into albedo / normal / depth maps.

    part = ('seg', (x0,y0), (x1,y1), r, z, albedo, plate)  plate: None or +1/-1
                  (the side of the across-axis that carries the mint plate)
         | ('cap', (x,y), r, z)                            knuckle cap + pin
    All in screen px of a local box (h, w)."""

    def __init__(self, h, w):
        self.h, self.w = h, w
        self.yy, self.xx = np.mgrid[0:h, 0:w].astype(np.float64)
        self.alb = np.zeros((h, w), np.uint8)
        self.nrm = np.zeros((h, w, 3))
        self.nrm[..., 2] = 1.0
        self.z = np.full((h, w), -1e9)
        self.m = np.zeros((h, w), bool)
        self.sep = np.zeros((h, w), bool)
        self.pid = np.full((h, w), -1, np.int32)
        self.k = 0

    def _put(self, mask, alb, nrm, z):
        sel = mask & (z >= self.z)
        # a dark separation line where this part overlaps an earlier (behind) part
        ring = edge_out(mask) & self.m & ~mask & (self.z < z)
        self.sep |= ring
        self.sep &= ~sel
        self.alb[sel] = alb[sel] if isinstance(alb, np.ndarray) else ci(alb)
        self.nrm[sel] = nrm[sel]
        self.z[sel] = z if np.isscalar(z) else z[sel]
        self.m |= sel
        self.pid[sel] = self.k
        self.k += 1

    def seg(self, p0, p1, r, z, alb='steel4', plate=None, bulge=0.8, prof=1.5):
        m, ac, al, (ux, uy) = _box_seg(self.xx, self.yy, p0[0], p0[1], p1[0], p1[1], r)
        px, py = -uy, ux
        # cross-section normal: flat-ish top, rounded sides (c02's boxy segments)
        s = np.sign(ac) * np.abs(ac) ** prof
        nz = np.sqrt(np.clip(1 - s * s, 0, 1)) + 0.15
        nrm = np.stack([px * s * bulge, py * s * bulge, nz], -1)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        a = np.full((self.h, self.w), ci(alb), np.uint8)
        if plate is not None:
            pm = (ac * plate) > -0.2
            a = np.where(pm, ci('paint5'), a)
            # the plate's edge: a dark seam line where mint meets steel
            edge = pm & (np.abs(ac * plate + 0.2) < 0.22)
            a = np.where(edge, ci('paint1'), a)
        self._put(m, a, nrm, z)
        return m

    def cap(self, c, r, z, alb='steel5'):
        m = (self.xx - c[0]) ** 2 + (self.yy - c[1]) ** 2 <= (r + 0.3) ** 2
        dx, dy = (self.xx - c[0]) / max(r, 0.5), (self.yy - c[1]) / max(r, 0.5)
        nz = np.sqrt(np.clip(1 - dx * dx - dy * dy, 0, 1)) + 0.25
        nrm = np.stack([dx * 0.9, dy * 0.9, nz], -1)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        a = np.full((self.h, self.w), ci(alb), np.uint8)
        # the pin: a dark dot in the middle (the lighting adds its glint)
        pin = (np.abs(self.xx - c[0]) < 0.8) & (np.abs(self.yy - c[1]) < 0.8)
        a = np.where(pin, ci('steel1'), a)
        self._put(m, a, nrm, z)
        return m

    def flat_cap(self, c, r, fill='steel5', ring='steel2', pin='steel1'):
        """c02's round knuckle cap as a PAINTED detail on the part beneath it
        (keeps that part's normals, so it lights with the finger): a lighter
        disc, a darker rim on its lower-right, a pin in the middle."""
        d = np.sqrt((self.xx - c[0]) ** 2 + (self.yy - c[1]) ** 2)
        m = (d <= r + 0.3) & self.m
        rim = m & (d > r - 0.7) & ((self.xx - c[0]) + (self.yy - c[1]) > -0.3 * r)
        self.alb[m] = ci(fill)
        self.alb[rim] = ci(ring)
        pn = m & (np.abs(self.xx - c[0]) < 0.7) & (np.abs(self.yy - c[1]) < 0.7)
        self.alb[pn] = ci(pin)

    def joint_line(self, p, ang, r, col='steel2'):
        """a 1-px joint seam across a finger at p (perpendicular to ang)."""
        ux, uy = math.cos(ang), math.sin(ang)
        a = (self.xx - p[0]) * ux + (self.yy - p[1]) * uy
        b = (self.xx - p[0]) * (-uy) + (self.yy - p[1]) * ux
        m = (np.abs(a) < 0.55) & (np.abs(b) <= r) & self.m
        self.alb[m] = ci(col)

    def finish(self):
        """outline (steel0) around the whole hand; separation lines inside."""
        ol = edge_out(self.m)
        return dict(alb=self.alb, nrm=self.nrm, z=self.z, m=self.m, sep=self.sep & self.m, ol=ol)


# ================================================================== RAMP SNAP (physical light -> hand-picked ramps)
LUM = LIN @ np.array([0.2126, 0.7152, 0.0722])          # linear luminance of every palette colour


class RampSnap:
    """Map a physically computed light (luminance + the share of it that is
    WARM) to hand-picked palette ramps per material family:
        cold ramp  - the material in ambient / dawn / sky light
        warm ramp  - the same material in the heart's (or the sun's) light
    Each ramp is a list of names; entries are sorted by OKLab lightness and a
    pixel's lightness picks its position, with an ordered-dither seam only in
    a narrow band between neighbours. The warm/cold choice is dithered in a
    narrow band around warm share 0.5."""

    def __init__(self, fams, fam_of):
        self.r = {}
        for fid, spec in fams.items():
            d = {}
            for key in ('cold', 'warm'):
                idx = np.array([ci(n) for n in names_of(spec[key])])
                Ls = LAB[idx, 0]
                o = np.argsort(Ls, kind='stable')
                d[key] = (idx[o], Ls[o])
            self.r[fid] = d
        self.fam_lut = np.zeros(NPAL, np.int32)
        for i in range(NPAL):
            self.fam_lut[i] = fam_of.get(_RAMP_NAME[i], 0)

    def family(self, alb):
        return self.fam_lut[alb]

    def snap(self, fam, L, w, band=0.34, wband=0.25, th=None, thw=None, keep=None):
        h, wd = fam.shape
        out = np.zeros((h, wd), np.uint8) if keep is None else keep.copy()
        if th is None:
            th = bayer(h, wd, 4)
        if thw is None:
            thw = bayer(h, wd, 4, 2, 1)
        for fid, d in self.r.items():
            m = fam == fid
            if not m.any():
                continue
            bd = band.get(fid, band.get(0, 0.34)) if isinstance(band, dict) else band
            Lm, wm, tm = L[m], w[m], th[m]
            warm = np.clip((wm - 0.5) / max(wband, 1e-3) + 0.5, 0, 1) > thw[m]
            res = np.zeros(len(Lm), np.uint8)
            for key, sel in (('cold', ~warm), ('warm', warm)):
                if not sel.any():
                    continue
                idx, Ls = d[key]
                p = np.interp(Lm[sel], Ls, np.arange(len(Ls)))
                fl = np.floor(p)
                fr = np.clip((p - fl - 0.5) / max(bd, 1e-3) + 0.5, 0, 1)
                q = np.clip((fl + (tm[sel] < fr)).astype(np.int32), 0, len(idx) - 1)
                res[sel] = idx[q]
            out[m] = res
        return out


def tone1(y, knee=0.62):
    """scalar luminance shoulder: linear below the knee, rolling toward 1."""
    return np.where(y < knee, y, knee + (1 - knee) * (1 - np.exp(-(y - knee) / (1 - knee))))
