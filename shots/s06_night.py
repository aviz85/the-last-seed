"""
S6 · NIGHT IN THE GREENHOUSE (clock 0-6.0 = global 32.0-38.0; film.py shows
0-2.3 = 32.0-34.3, fading in 32.0-32.8; c05 covers the rest). Owner: GREENHOUSE.

Night inside the ruined palm-house, exposed to read on normal screens (frame
median ~27/255, p95 ~57): the robot's dim, flickering chest core is the only warm
light (its pool lights the sit); the dying sprout adds a whisper of green; the
iron dome is a dark cage against a deep-blue dust-night sky, its ribs rimmed by
the drowned moon (upper right, brightest there).

Local timeline (cues.py, global = local + 32.0):
  0.0-0.62 limps in (walk_tired, can='sprout_wilt'); 'sit' starts at 1.2 - SIT_IMPACT_T so
           the butt hits the floor ON the sit_down cue (33.2, metal clank + dust)
  +SIT_CAN_DOWN_T  can set down on its right; the shot draws can + sprout
  2.5-3.7  the sprout leans toward the glowing chest (sprout_lean 34.5) + shimmer
  3.5      'sit_look_can'   (robot_notice 35.5)
  4.6      'sit_look_chest' (look_chest 36.6) -> hold
"""
import math
import numpy as np

from engine import *
from engine import ease, noise
from engine.sprite import anchor_pos
from chars import robot, props
from chars import robot_acts as RA
from shots import lib_greenhouse as G

DURATION = 6.0

T_SIT_CUE = 1.2                                   # sit_down 33.2: the clank (butt hits the floor)
T_SIT = T_SIT_CUE - getattr(RA, 'SIT_IMPACT_T', 7 / 12)   # 'sit' starts so the impact lands on the cue
T_LEAN0, T_LEAN1 = 2.5, 3.7
T_LOOK_CAN = 3.5
T_LOOK_CHEST = 4.6
WALK_IN = robot.travel('walk_tired', T_SIT)      # px the limp covers before the sit (8)

# night exposure (light steps): readable on normal screens, still night
AMB = -5.0                    # moon ambient (integer: flat areas never dither)
RIM_SET = -2.0                # moon rims on the rubble, props, planter
RIM_IRON = -2.0               # moon rims on the dome iron (+1 on the moon side)
RIM_ROBOT = -2.0              # moon rim on the robot's head / shoulders
POOL_PEAK = 5.3               # the heart's warm pool
POOL_REACH = 2.4
BLOOM = 2.0
MOON_SIDE = G._moon_g > 0.30


# warm the LUT cache at import (steady-state speed from the first frame)
for _tn in G.mix_tints('moon', 'amber', 4) + ['green']:
    G.tint_lut(_tn, 'env')
    G.tint_lut(_tn, 'dead')


def robot_pose(t):
    """(anim, t_anim, x, y) for the robot at local time t. The limp uses the rig's
    stepped ground travel (zero foot slide) and ends exactly on the sit spot."""
    sx, sy = G.S6_SIT
    if t < T_SIT:
        x = sx - WALK_IN + robot.travel('walk_tired', t)
        return 'walk_tired', t, x, sy
    ts = t - T_SIT
    d_sit = robot.duration('sit')
    if t < T_LOOK_CAN:
        if ts < d_sit:
            return 'sit', ts, sx, sy
        return 'sit_idle', ts - d_sit, sx, sy
    if t < T_LOOK_CHEST:
        return 'sit_look_can', t - T_LOOK_CAN, sx, sy
    return 'sit_look_chest', t - T_LOOK_CHEST, sx, sy


def core_glow(t):
    """Dim, unsteady heart (0.3 nominal) -- flickers like a failing bulb."""
    f = ease.flicker(t, seed=61, rate=9.0)
    f2 = ease.flicker(t, seed=67, rate=23.0)
    g = 0.30 * (0.80 + 0.30 * f) * (0.93 + 0.12 * f2)
    # two brief dips (a stutter of the damaged core)
    for tc, w in ((0.9, 0.05), (4.15, 0.07)):
        if abs(t - tc) < w:
            g *= 0.55
    # realisation: as it looks down at its heart, the heart swells once, softly
    sw = ease.smoothstep(ease.prog(t, 4.95, 5.45)) * (1 - ease.smoothstep(ease.prog(t, 5.55, 5.98)))
    return g * (1.0 + 0.30 * sw)


def render(t):
    sc = G.Scene('night')
    anim, ta, x, y = robot_pose(t)
    g = core_glow(t)
    can_state = 'sprout_wilt'
    spr = robot.frame(anim, ta, can=can_state, variant='damaged', scale='medium', core=g / 0.3 * 0.3, eye=0.45)
    Zr = G.depth_at_feet(y)
    rm, _ = sc.place(spr, x, y, 'feet', Z=Zr, is_robot=True)
    cx, cy = anchor_pos(spr, x, y, 'feet', 'core')

    # can + sprout on the ground (after the sit anim sets it down): the same
    # can-with-sprout sprite the robot carried, so the hand-off has no pop
    sprout_xy = None
    if t >= T_SIT + RA.SIT_CAN_DOWN_T:
        kx, ky = G.S6_SIT[0] + RA.CAN_SET_OFFSET[0], G.S6_SIT[1] + RA.CAN_SET_OFFSET[1]
        lean = -ease.in_out_sine(ease.prog(t, T_LEAN0, T_LEAN1))
        if t < T_LEAN0:
            can = props.can('sprout_wilt', t=t, wind=0.1)
        else:
            can = props.can('sprout_lean', t=t, wind=0.1, lean=lean)
        sc.place(can, kx, ky, 'base', Z=Zr - 0.05, emissive_leaf=True)
        mx, my = anchor_pos(can, kx, ky, 'base', 'mouth')
        sprout_xy = (mx, my - 2)

    G.hanging_pots_draw(sc, t, sway_amp=0.0)
    # ------------------------------------------------------------ lighting (energy space)
    P, N = sc.pos, sc.nrm
    # moonlight through the dust: integer levels (flat areas never dither)
    Ls = np.array([0.45, 0.62, 0.64])
    Ls /= np.linalg.norm(Ls)
    sky = np.clip((N * Ls).sum(-1), 0, 1)
    Lm = np.full((G.SH, G.SW), AMB)
    # faint moon patches where the light falls through broken panes
    Lm += 2.6 * np.clip((G.SKY_T['moon'] - 0.5) / 0.4, 0, 1)
    # a thin cold moonbeam through the dome onto the dead planter (where the tree will be)
    mb = G.shaft_level(262, -12, 8, -0.19, 1.0, 215, soft=2.5)
    Lm = np.maximum(Lm, AMB + 4.6 * mb)
    # moonlit rims: silhouette tops/right sides, the iron of the dome (two steps
    # brighter on the moon side, upper right), the fallen rib and the robot
    Lm = np.where(G.RIM_TOP | G.RIM_RIGHT, np.maximum(Lm, RIM_SET), Lm)
    Lm = np.where(G.RIM_MOON, np.where(MOON_SIDE, RIM_IRON + 1.0, RIM_IRON), Lm)
    Lm = np.where(G.FALLEN_RIB_M & G.RIM_TOP, RIM_IRON + 1.0, Lm)
    rt, rr = G.depth_rims(sc.pos[..., 2], sc.robot | G._SOLID, 0.45)
    Lm = np.where(sc.robot & (rt | rr), RIM_ROBOT, Lm)
    Em = G.en(Lm)
    # the core's warm pool (emitted forward through the chest porthole)
    lp = G.world_of_frame(cx, cy, Zr - 0.15)
    k = g / 0.3
    Ew = G.light_level(P, N, lp, peak=POOL_PEAK + 1.6 * math.log(k + 1e-3), slope=5.6, reach=POOL_REACH,
                       wrap=0.65, cone=((1.0, 0.0, -0.6), 0.45))
    Ew[G.GLASSPIX] *= 0.15 + 0.6 * G.DUSTG[G.GLASSPIX]
    # the porthole faces forward: its own body only catches light below/right of it
    rb = sc.robot
    if rb.any():
        yy, xx = np.nonzero(rb)
        fx_ = (xx - G.M) - cx
        fy_ = (yy - G.M) - cy
        own = np.clip(0.25 + 0.18 * fx_ + 0.10 * fy_, 0.12, 1.0)
        Ew[yy, xx] *= own
    # bloom: the heart glows through the dark air around the porthole
    Ew += G.screen_glow(cx, cy, BLOOM + 1.2 * math.log(k + 1e-3), 0.62, 7)
    # the sprout's tiny green life-glow
    Eg = np.zeros_like(Ew)
    if sprout_xy is not None:
        gl = 0.2 + 0.8 * ease.smoothstep(ease.prog(t, T_LEAN0, T_LEAN1))
        gl += 0.15 * math.sin(t * 5.1)
        lg = G.world_of_frame(sprout_xy[0], sprout_xy[1], Zr - 0.1)
        Eg = G.light_level(P, N, lg, peak=gl, slope=14.0, reach=0.38, wrap=0.8)
    E = Em + Ew + Eg
    L = G.lv(E)
    L[G.GLASSPIX & (Ew < 0.02) & ~sc.actor] = 0.0   # glass keeps its own night colours
    L = L + G.vignette_level(1.5)
    L[sc.emis] = 0.0
    which = G.ratio_index(Em, Ew, 4)
    # the robot keeps a little of its mint: its own warm tint is capped
    which = np.where(sc.robot, np.minimum(which, 2), which)
    which = np.where(Eg > np.maximum(Ew, Em) * 0.8, 4, which)
    img = G.relight_scene(sc, L, G.mix_tints('moon', 'amber', 4) + ['green'], which, band=0.4)
    # glass: untouched night colours where no warm light reaches
    gm = G.GLASSPIX & (Ew < 0.02) & ~sc.actor
    img[gm] = G.GLASS_RGB['night'][gm]
    # the moonbeam is visible in the dusty air in front of the glass
    G.glass_reflect(img, gm & (mb > 0.05), mb * 0.55, mode='moon', scale=1.0)
    em = sc.emis
    img[em] = sc.img[em]
    # footfalls of the limp and the clank of the sit (dust, lit only by the core)
    sx0, sy0 = G.S6_SIT
    for j, tf in enumerate((0.22,)):
        fxp = sx0 - WALK_IN + robot.travel('walk_tired', tf) + (3 if j % 2 else -1)
        G.dust_puff(img, fxp, sy0, t, tf, n=6, seed=640 + j, spread=4, rise=2, life=0.45,
                    lit=('rust3', 'ochre3'))
    t_imp = T_SIT_CUE
    G.dust_puff(img, sx0 - 3, sy0, t, t_imp, n=14, seed=650, spread=9, rise=4, life=0.7,
                lit=('rust3', 'ochre4'))
    # dust motes drifting through the warm light (only visible where it's lit)
    mx_, my_, mr_ = G.motes(t, 70, 611, (cx - 45, cy - 40, 90, 70), vel=(1.5, -0.8), sway=(1.2, 0.3))
    for i in range(len(mx_)):
        px, py = int(mx_[i]) + G.M, int(my_[i]) + G.M
        if 0 <= px < G.SW and 0 <= py < G.SH:
            e = Ew[py, px]
            if e > G.en(-1.0):
                img[py, px] = C('core5') if e > G.en(1.0) else (C('core3') if e > G.en(0.0) else C('rust3'))
    # a few motes turning in the moonbeam
    bx_, by_, br_ = G.motes(t, 40, 612, (180, -10, 110, 170), vel=(0.6, 1.1), sway=(1.0, 0.2))
    for i in range(len(bx_)):
        px, py = int(bx_[i]) + G.M, int(by_[i]) + G.M
        if 0 <= px < G.SW and 0 <= py < G.SH and mb[py, px] > 0.45 and br_[i] < 0.5:
            img[py, px] = C('night5') if br_[i] < 0.2 else C('night4')
    # green shimmer as the sprout reaches for the light
    if sprout_xy is not None and T_LEAN0 - 0.1 < t < T_LEAN1 + 0.9:
        a_ = ease.prog(t, T_LEAN0 - 0.1, T_LEAN0 + 0.3) * (1 - ease.prog(t, T_LEAN1 + 0.3, T_LEAN1 + 0.9))
        for i in range(9):
            ph = float(noise.rand(i, 1, 631))
            cyc = (t * 0.9 + ph) % 1.0
            px = sprout_xy[0] + (float(noise.rand(i, 2, 631)) - 0.5) * 8 - 3 * cyc * ease.prog(t, T_LEAN0, T_LEAN1)
            py = sprout_xy[1] + 2 - cyc * 11
            if noise.rand(i, int(t * 12), 632) < 0.75 * a_ and cyc < 0.85:
                xi, yi = int(px) + G.M, int(py) + G.M
                if 0 <= xi < G.SW and 0 <= yi < G.SH:
                    img[yi, xi] = C('leaf7') if cyc < 0.4 else C('leaf5')
    return G.crop(img)
