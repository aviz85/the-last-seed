"""
S8a · BLOOM, INSIDE (global 45.0-48.0).  Owner: GREENHOUSE.

Same framing as S6/S7. The robot sits dead ('dead_sit') at the planter.
  0.0-0.3  NEAR DARKNESS (matches the end of close-up c06, ~2.8 light steps below
           the dawn grade): deep night/ash tones, robot and set barely visible,
           only c06's faint ember under the soil
  0.3      one heartbeat: a ring of warm light pulses out through the soil -- the
           first light we see (first_pulse 45.3)
  0.8      veins of green-gold light race out through the soil, over the rim,
           across the floor and under the robot (veins 45.8) -- roots with body:
           1-3 px luminous cores tapering toward the tips, a soft dithered glow on
           the ground around them, white-hot racing fronts, bright buds at the
           root tips; their spreading glow lifts the room back up (near the veins
           first, the dome last) ...
  1.8      ... to full brightness at the burst
  1.2      the sprout straightens, glows, its leaves unfurl
  1.8      GROWTH BURST (growth_burst 46.8): flash; the stem becomes a trunk that
           shoots out of frame, branches whip out, leaves pop, blossoms; motes
           swirl, the glass trembles, the camera shake builds to the cut at 3.0.
           Colour returns locally: rich dark soil, moss creeping over the stones.
"""
import math
import numpy as np

from engine import *
from engine import ease, noise, fx
from engine.sprite import anchor_pos
from engine.stage import Stage
from chars import robot, props
from chars import robot_acts as RA
from shots import lib_greenhouse as G

DURATION = 3.0

T_PULSE = 0.3
T_VEINS = 0.8
T_SPROUT = 1.2
T_BURST = G.T_BURST                       # 1.8
DIG_CAN = getattr(RA, 'DIG_CAN_OFFSET', (-8, 0))
HX, HY = G.CORE_SPOT                      # the buried heart
SX, SY = G.SPROUT_XY


# warm the LUT cache at import (steady-state speed from the first frame)
G.tint_lut((0.92, 0.96, 1.0), 'dead')
for _tn in G.mix_tints('dawn', 'vein', 5):
    G.tint_lut(_tn, 'env_alive')
    G.tint_lut(_tn, 'alive')
G.tint_lut(G.mix_tints('dawn', 'vein', 5)[0], 'env')
G.tint_lut(G.mix_tints('dawn', 'vein', 5)[0], 'dead')
for _tn in G.mix_tints('dawn', 'gold', 5):
    G.tint_lut(_tn, 'env')
    G.tint_lut(_tn, 'dead')


D_OPEN = -4.2                             # opening darkness (my light steps; calibrated to c06's end)
GLASS_MUL = 1.8                           # the unlit glass/sky needs a deeper grade to match
NIGHT_T = (0.62, 0.74, 1.0)               # tint of the darkness grade (deep night / ash)
G.tint_lut(NIGHT_T, 'dead')


def darkness(t, vm):
    """Per-pixel darkness offset (steps, <= 0). Flat D_OPEN until the veins; from 0.8
    the veins' spreading glow lifts it -- first around them, then the whole room --
    reaching 0 (full brightness) at the burst (1.8)."""
    g = ease.smoothstep(ease.prog(t, T_VEINS, T_BURST))
    if g >= 1.0:
        return 0.0
    if t < T_VEINS or not vm.any():
        return np.full((G.SH, G.SW), D_OPEN * (1.0 - g))
    from scipy.ndimage import gaussian_filter
    near = gaussian_filter((vm & G.VEIN_SURF).astype(np.float64), 16.0)
    near = np.clip(near * 40.0, 0.0, 1.0)
    local = ease.smoothstep(ease.prog(t, T_VEINS, T_VEINS + 0.6))
    return D_OPEN * (1.0 - g) * (1.0 - 0.7 * local * near)


def dark_grade(img, mask, D):
    """Palette-true darkening of final-colour pixels (the glass / sky seen through
    the dome) by D steps toward deep night -- ordered-dither fade between steps."""
    if np.isscalar(D):
        if D > -0.02:
            return img
        D = np.full(mask.shape, float(D))
    m = mask & (D < -0.02)
    if not m.any():
        return img
    q = G.quantize(D, band=1.0)
    idx = G.to_index(np.ascontiguousarray(img))
    lt = G.tint_lut(NIGHT_T, 'dead')
    out = idx.copy()
    out[m] = lt[q[m] + G.MAXL, idx[m]]
    img[m] = G.from_index(out[m])
    return img


def vein_front(t):
    """Arc length (metres) the root-light network has reached at local time t."""
    u = t - T_VEINS
    if u <= 0:
        return -1.0
    return 5.6 * (1 - math.exp(-u / 0.45))


def shake_amp(t):
    if t < T_BURST:
        return 0.0
    u = (t - T_BURST) / (DURATION - T_BURST)
    return 0.6 + 3.2 * u ** 1.6


def sprout_sprite(t):
    """The sprout waking (1.2-1.8): straightens, glows, two leaves unfurl."""
    u = ease.out_back(ease.prog(t, T_SPROUT, T_SPROUT + 0.5), 1.4)
    h = int(round(6 + 5 * u))
    spr = layer(16, 13)
    cx = 6
    for y in range(h):
        yy = 15 - y
        spr[yy, cx] = (*C('leaf5' if y < h - 2 else 'leaf7'), 255)
    ly = 15 - h + 2
    ln = int(round(1 + 3 * u))
    for k in range(1, ln + 1):
        droop = 0 if u > 0.6 else (1 if k > 1 else 0)
        for side in (-1, 1):
            x = cx + side * k
            y = ly - (k // 2) + droop
            if 0 <= x < 13 and 0 <= y < 16:
                spr[y, x] = (*C('leaf6' if k < ln else 'leaf7'), 255)
                if k > 1 and y + 1 < 16:
                    spr[y + 1, x] = (*C('leaf4'), 255)
    from engine.sprite import Sprite
    return Sprite(spr, {'base': (cx, 15)})


def render(t):
    sc = G.Scene('dawn')
    kx, ky = G.KNEEL
    Zr = G.depth_at_feet(ky)
    spr = robot.frame('dead_sit', t, variant='damaged', scale='medium', core=0.0, eye=0.0)
    can = RA.dig_can('empty') if hasattr(RA, 'dig_can') else props.can('empty')
    sc.place(can, kx + DIG_CAN[0], ky + DIG_CAN[1], 'base', Z=Zr + 0.05)
    amp = shake_amp(t)
    swing = amp * 0.8
    G.hanging_pots_draw(sc, t, sway_amp=swing * 0.6, shake=swing * 0.5)

    # ---------------------------------------------------------------- colour returns
    front = vein_front(t)
    vm = np.isfinite(G.VEIN_D) & (G.VEIN_D <= front)
    alive = ease.prog(t, T_VEINS + 0.3, DURATION)
    G.revive_soil(sc, t, front)

    # ---------------------------------------------------------------- the tree
    tb = t - T_BURST
    if t < T_BURST + 0.05:
        sp = sprout_sprite(t) if t >= T_SPROUT else props.sprout('planted', t, wind=0.02)
        spm, _ = sc.place(sp, SX, SY, 'base', Z=Zr + 0.05, emissive_core=False)
        if t >= T_VEINS + 0.15:                 # the veins reach it: it lights up
            sc.emis |= spm
    sc.place(spr, kx, ky, 'feet', Z=Zr, is_robot=True)
    newborn = np.zeros((G.SH, G.SW), bool)
    if tb >= 0:
        leaves = G.draw_tree(sc, tb, t, amp)
        newborn = ((G.TREE_T <= tb) & ~sc.robot) | leaves

    # ---------------------------------------------------------------- lighting
    P, N = sc.pos, sc.nrm
    # near darkness after c06; the veins' light lifts the room back up by the burst
    D = darkness(t, vm)
    dark = G.en(D)
    dfrac = float(np.mean(D)) / D_OPEN if np.ndim(D) else 0.0     # 1 = fully dark .. 0
    E_dawn = G.dawn_energy(sc, strength=1.0 - 0.35 * alive, amb=-2.0) * dark
    # the pale dawn shaft, only as the room comes back (drowned by the life light later)
    sh = np.maximum(G.shaft_level(282, -12, 13, -0.44, 1.0, 185), G.shaft_level(244, -12, 5, -0.44, 1.0, 150) * 0.6)
    E_dawn = E_dawn + G.en(0.5) * sh * dark * (1.0 - ease.smoothstep(ease.prog(t, T_VEINS, T_BURST)))
    E_life = np.zeros_like(E_dawn)
    # the buried heart's faint breath, then the heartbeat ring
    lp = G.world_of_frame(HX, HY + 1, Zr - 0.02)
    lp[1] = G.HS - 0.05
    E_life += G.buried_breath(sc, 2.0 + t, Zr) * (0.25 + 0.75 * ease.smoothstep(ease.prog(t, T_PULSE - 0.05, T_PULSE + 0.2)))
    if t >= T_PULSE:
        u = t - T_PULSE
        rad = 0.2 + 4.5 * ease.out_cubic(min(u / 0.7, 1.0))
        fade = max(0.0, 1 - u / 0.75)
        dxz = np.hypot(P[..., 0] - lp[0], (P[..., 2] - lp[2]))
        ring = np.exp(-((dxz - rad) / 0.35) ** 2) * (P[..., 1] < 1.2)
        E_life += G.en(2.2) * ring * fade
        E_life += G.en(1.5) * max(0.0, 1 - u / 0.25) * np.exp(-(dxz / 0.6) ** 2)
    # the veins light the floor, the planter and the robot from below
    if front > 0:
        from scipy.ndimage import gaussian_filter
        vms = (vm & G.VEIN_SURF).astype(np.float64)
        vf = np.minimum(gaussian_filter(vms, 2.2) * 3.2, 0.9)
        vf += np.minimum(gaussian_filter(vms, 7.0) * 4.5, 0.9)
        vf = np.where(G.SID == G.SID_SOIL, np.minimum(vf, 0.55), vf)
        E_life += G.en(0.4) * vf * (0.8 + 0.4 * alive)
    # the sprout's glow as it wakes
    if T_SPROUT <= t < T_BURST + 0.3:
        u = ease.prog(t, T_SPROUT, T_BURST)
        ls = G.world_of_frame(SX, SY - 6, Zr)
        E_life += G.light_level(P, N, ls, peak=0.3 + 1.6 * u, slope=6.0, reach=0.7, wrap=0.8)
    # the burst: a flash from the base of the tree, then the living light
    if tb >= 0:
        flash = G.en(4.0) * math.exp(-tb / 0.10) + G.en(1.5) * math.exp(-tb / 0.6)
        lb = G.world_of_frame(SX + 2, SY - 10, Zr)
        E_life += flash * G.light_level(P, N, lb, peak=0.0, slope=0.35, reach=30, wrap=0.8)
        E_life += G.light_level(P, N, lb, peak=1.2 + 0.8 * ease.prog(tb, 0, 1.0), slope=0.55, reach=25, wrap=0.6)
    # the dead robot: keep its uplight gentle so its paint never turns neon
    E_life = np.where(sc.robot, np.minimum(E_life, G.en(0.3)), E_life)
    E = E_dawn + E_life
    L = G.lv(E)
    # newborn wood & leaves glow with life (a lift so they read vivid, not murky)
    L = np.where(newborn, np.maximum(L, 0.4), L)
    L = G.haze_lift(L, 0.6 + (2.0 * np.minimum(D, 0.0) if np.ndim(D) else 0.0), pos=sc.pos)
    L = L + G.vignette_level(1.3 * (1 - 0.5 * alive) + 0.6 * dfrac)
    L[sc.emis] = 0.0
    which = G.ratio_index(E_dawn, E_life, 5)
    # warm heartbeat first (same tints as the end of S7), green-gold life from the veins on
    if t < T_VEINS:
        img = G.relight_scene(sc, L, G.mix_tints('dawn', 'gold', 5), which, band=0.3)
    else:
        img = G.relight_scene(sc, L, G.mix_tints('dawn', 'vein', 5), which, band=0.3,
                              env='env_alive', actor='alive', first_dead=True)
    gm = G.GLASSPIX & ~sc.actor
    img[gm] = G.GLASS_RGB['dawn'][gm]
    dark_grade(img, gm, D * GLASS_MUL if np.ndim(D) else D)   # the dome and sky sink into deep night too
    G.glass_reflect(img, gm, E_life * (0.35 + 0.9 * G.DUSTG), mode='green', scale=1.5,
                    allowed='dead' if t < T_VEINS else 'alive')
    img[sc.emis] = sc.img[sc.emis]

    # ---------------------------------------------------------------- emissive FX
    G.draw_veins(img, t, front, vm, occ=sc.actor)
    if t < T_VEINS + 0.4:
        # c06's faint ember until the heartbeat flares it, then it breathes
        st_ = 0.3 if t < T_PULSE - 0.03 else 1.0 + 1.5 * max(0.0, 1 - abs(t - T_PULSE) / 0.2)
        G.buried_glow(img, HX, HY, 2.0 + t, strength=st_, occ=sc.actor)
    g_lift = ease.smoothstep(ease.prog(t, T_VEINS, T_BURST))
    G.shaft_motes(img, 7.0 + t, sh, None, fade=g_lift * (1.0 - g_lift) * 2.0)
    G.life_motes(img, t, tb, E_life)
    if 0 <= tb < 0.9:
        G.spark_burst(img, SX, SY - 1, t, T_BURST, n=34, seed=871, speed=(30, 95), life=(0.4, 0.85),
                      gravity=160.0, spread=(-math.pi * 0.95, -math.pi * 0.05),
                      cols=('bark3', 'dust4', 'bark2'))
        G.spark_burst(img, SX, SY - 3, t, T_BURST, n=22, seed=872, speed=(40, 120), life=(0.3, 0.6),
                      gravity=40.0, spread=(-math.pi, 0.0), cols=('white1', 'gold4', 'leaf7'))
    if tb >= 0 and tb < 0.07:
        # impact frames: white-gold flash over the tree base
        G.flash_frame(img, SX, SY - 8, tb)
    # shards on the floor jitter as the ground trembles
    if amp > 0.5:
        G.tremble_shards(img, t, amp)
    st = Stage(bg='ink0')
    st.add(img, -G.M, -G.M)
    if amp > 0:
        dx, dy = fx.shake(t, amp, seed=881, freq=22.0)
        st.shake = (dx, dy * 0.8)
    return st
