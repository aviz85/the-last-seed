"""
S7 · THE GIFT. Owner: GREENHOUSE. One clock 0-7.0 (global = local + 38.0); film.py
shows local 0-2.0 (S7a, 38.0-40.0, fading in from black 38.0-38.6: 'next morning')
and 4.1-4.8 (S7b, 42.1-42.8); c03, c08 and c06 cover the rest.

Same framing as S6. GREY DAWN: cold pale light through the broken dome.
  A (0-2.0)  kneeling at the planter: 'dig_plant' (scrapes 0.5-1.2, sprout in
             the soil at PLANT_T=1.2 -> props.sprout('planted'), pats ~1.6).
             Life: pale shafts through the dome (breathing as thin cloud veils
             pass the sun) with dust motes drifting and glittering in them, the
             clouds drifting beyond the glass, two glints on the dome's shards
             (0.85, 1.5), the right hanging pot swaying on its chain.
             The robot's head hides the freshly planted sprout until it sits
             back (~1.83); the sprout stands SPROUT_NUDGE px right of the
             planting spot, clear of the head's silhouette (S7a and S7b).
  hidden     'open_chest' 2.4-3.6 -> holds the blazing core (anchor core_held)
  B (4.1-4.8) the heart in its hands floods the greenhouse with gold, long
             shadows; 'place_core' 4.6 (touches soil 5.0 = core_in_soil 43.0), the
             light sinks into the soil; 'power_down' 5.4 (eye off 6.3);
             6.3-7.0 stillness, cold dawn only.
"""
import math
import numpy as np

from engine import *
from engine import ease, noise, fx
from engine.sprite import anchor_pos
from chars import robot, props
from chars import robot_acts as RA
from shots import lib_greenhouse as G

DURATION = 7.0

T_DIG = 0.0
T_OPEN = 2.4
T_PLACE = 4.6
T_TOUCH = T_PLACE + RA.CORE_PLACE_T          # 5.0
T_POWER = 5.4
T_EYE_OFF = T_POWER + RA.POWER_EYE_OFF_T     # 6.3
SCRAPES = tuple(getattr(RA, 'DIG_SCRAPE_TS', (0.5, 0.75)))
PATS = tuple(getattr(RA, 'DIG_PAT_TS', (1.6, 1.75)))
DIG_CAN = getattr(RA, 'DIG_CAN_OFFSET', (-8, 0))
CORE_OUT_T = getattr(RA, 'CORE_OUT_T', 0.75)
D_DIG = robot.duration('dig_plant')

# staging / life (S7a is 'the next morning': the air and the sky move)
SPROUT_NUDGE = 5              # px right of the planting spot (PLANT_OFFSET unchanged): the
                              # sprout never touches the robot's head silhouette
POT_SWAY = (1.5, 2.6)         # the right hanging pot: amplitude px, rad/s
GLINTS = ((0.85, 283, 20, 0.40, 2), (1.50, 196, 45, 0.35, 2))     # (t0, x, y, dur, size)
SHAFT_AIR = 0.45              # light steps the shafts add to the air in front of the set
SHAFT_GLASS = 0.8             # ... and in front of the glass (energy)


# warm the LUT cache at import (steady-state speed from the first frame)
G.tint_lut((0.92, 0.96, 1.0), 'dead')
for _tn in G.mix_tints('dawn', 'gold', 5):
    G.tint_lut(_tn, 'env')
    G.tint_lut(_tn, 'dead')


def robot_pose(t):
    if t < T_OPEN:
        return 'dig_plant', t - T_DIG
    if t < T_PLACE:
        return 'open_chest', t - T_OPEN
    if t < T_POWER:
        return 'place_core', t - T_PLACE
    return 'power_down', t - T_POWER


def eye_level(t):
    # power_down drives its own flicker/death (x this); dim & tired before
    return 0.0 if t >= T_EYE_OFF else 0.5


def core_power(t):
    """(room brightness 0..1, buried?) of the removed heart."""
    if t < T_OPEN + CORE_OUT_T - 0.1:
        return 0.0, False
    if t < T_TOUCH:
        return 1.0, False
    u = t - T_TOUCH
    return max(0.0, 1.0 - u / 1.1), True


def heart_beat(t):
    """Light pulse locked to the orb sprite's lub-dub (robot_acts HEARTBEAT_HZ, on the
    running animation's clock)."""
    hz = getattr(RA, 'HEARTBEAT_HZ', 1.2)
    ta = t - (T_OPEN if t < T_PLACE else T_PLACE)
    ta = math.floor(max(ta, 0.0) * 12.0 + 1e-6) / 12.0        # the rig's 12 fps steps
    p = (ta * hz) % 1.0
    if p < 0.1 or 0.2 <= p < 0.3:
        return 0.35
    if p < 0.42:
        return 0.12
    return 0.0


def pw_now(t):
    """0..1 how much the heart dominates the room (for fading other lights)."""
    if t < T_OPEN + CORE_OUT_T - 0.1:
        return 0.0
    if t < T_TOUCH:
        return ease.smoothstep(ease.prog(t, T_OPEN + CORE_OUT_T - 0.1, T_OPEN + CORE_OUT_T + 0.2))
    return 1.0 - ease.smoothstep(ease.prog(t, T_TOUCH, T_TOUCH + 0.7))


def sink_state(t):
    """After the touch: light height offset (m, below the resting height), level
    drop (steps) -- the light sinks into the soil, the room follows it down."""
    u = max(0.0, t - T_TOUCH)
    dy = 0.30 * ease.in_out_sine(min(u / 0.5, 1.0))
    drop = 2.2 * ease.in_quad(min(u / 0.45, 1.0)) + 5.0 * ease.smoothstep(ease.prog(u, 0.45, 1.1))
    return dy, drop


def render(t):
    sc = G.Scene('dawn')
    kx, ky = G.KNEEL
    anim, ta = robot_pose(t)
    chest_core = 0.3
    spr = robot.frame(anim, ta, can=None, variant='damaged', scale='medium',
                      core=chest_core, eye=eye_level(t))
    Zr = G.depth_at_feet(ky)
    # drawn BEHIND the robot: the can (inside the sprite during dig_plant, then the
    # shot's dig_can at the same spot) and the planted sprout
    if anim != 'dig_plant' or ta >= D_DIG:
        can = RA.dig_can('empty') if hasattr(RA, 'dig_can') else props.can('empty')
        sc.place(can, kx + DIG_CAN[0], ky + DIG_CAN[1], 'base', Z=Zr + 0.05)
    sx, sy = kx + RA.PLANT_OFFSET[0], ky + RA.PLANT_OFFSET[1]
    if t >= RA.PLANT_T:
        sp = props.sprout('planted', t, wind=0.05)
        # while the robot is bent over the hole its head hides the sprout (no leaf
        # tip peeking out of its silhouette); once it sits back the sprout stands a
        # few px right of the planting spot -- clear negative space from the head
        spm0 = G.sprite_mask(sp, sx, sy, 'base')
        cover = (spm0 & G.sprite_mask(spr, kx, ky, 'feet')).sum() / max(spm0.sum(), 1)
        if cover < 0.5:
            sc.place(sp, sx + SPROUT_NUDGE, sy, 'base', Z=Zr + 0.05, emissive_core=False, emissive_leaf=True)
    sc.place(spr, kx, ky, 'feet', Z=Zr, is_robot=True)
    # where is the heart?
    pw, buried = core_power(t)
    if t < T_TOUCH:
        hx, hy = anchor_pos(spr, kx, ky, 'feet', 'core_held')
    else:
        hx, hy = kx + RA.CORE_PLACE_OFFSET[0], ky + RA.CORE_PLACE_OFFSET[1]
    # one of the hanging pots sways gently on its chain (a breath of morning air)
    G.hanging_pots_draw(sc, t, sway=(None, None, POT_SWAY))

    # ------------------------------------------------------------ lighting
    P, N = sc.pos, sc.nrm
    gd = G.grief_dim(t)                        # the room sinks toward darkness after the eye dies
    E_dawn = G.dawn_energy(sc, strength=1.0, amb=-2.0) * G.en(-1.3 * gd)
    # pale shafts of dawn falling through the broken dome onto the planter
    sh = G.dawn_shafts()
    # thin veils of cloud passing the low sun: the shafts breathe gently
    sh = sh * (0.88 + 0.24 * float(noise.vnoise1(np.array([t * 0.8]), 747)[0]))
    # the pale shaft is drowned out while the heart blazes
    sh = sh * (1.0 - 0.85 * pw_now(t))
    E_dawn = E_dawn + G.en(0.5 - 1.3 * gd) * sh
    E_gold = np.zeros_like(E_dawn)
    E_glass = E_gold
    ry = None
    if pw > 0:
        depth_off = 0.35
        lp = G.world_of_frame(hx, hy, Zr - depth_off)
        drop = 0.0
        if buried:
            dy, drop = sink_state(t)
            lp[1] = G.HS + 0.12 - dy
        peak = 5.0 + heart_beat(t) - drop
        E_gold = G.light_level(P, N, lp, peak=peak, slope=0.46, reach=30.0, wrap=0.35)
        # shadows: the robot's own body, the planter (grows as the light sinks
        # below the rim), the tank and the bench
        sh_r = G.card_shadow(P, lp, sc.robot, Zr)
        sh_p = G.planter_shadow(P, lp) & ~((G.SID >= G.SID_PSIDE) & ~G.PROPM)
        sh_t = G.card_shadow(P, lp, *G.PROP_CARDS['tank'])
        sh_b = G.card_shadow(P, lp, *G.PROP_CARDS['bench'])
        E_glass = E_gold.copy()          # the glass mirrors the whole lit room: no cast shadows
        E_gold = E_gold * G.soft_shadow_factor(P, sh_r, Zr, hard=0.05, soft=1.0, spread=9.0)
        E_gold = np.where(sh_p | sh_t | sh_b, E_gold * 0.08, E_gold)
        if buried:
            # once under the soil surface only the bed around it keeps a glow
            under = np.clip((G.HS - lp[1]) / 0.12, 0, 1)
            soil = (G.SID == G.SID_SOIL) & ~sc.actor
            E_soil = G.light_level(P, N, lp, peak=2.8 - 0.6 * drop, slope=8.0, reach=1.2, wrap=0.9)
            E_gold = np.where(soil, np.maximum(E_gold, E_soil), E_gold * (1 - 0.92 * under))
    if t > T_TOUCH:
        E_gold = E_gold + G.buried_breath(sc, t - T_TOUCH, Zr)
    if pw > 0 and not buried:
        # the iron's edges turned toward the heart catch it (gold tracery)
        E_gold = E_gold + G.iron_glint(hx, hy, E_gold * 2.5)
        # god rays: the blazing heart lights the dust in the air (volumetric)
        ry = G.god_rays(hx, hy, t, length=230.0) * (G.FY < hy + 6) * min(1.0, pw)
    # the robot's own body: lit warm in front, never blown out (the heart must
    # stay the brightest thing in the frame)
    if pw > 0:
        rb = sc.robot & ~sc.emis
        if rb.any():
            yy, xx = np.nonzero(rb)
            fx_ = (xx - G.M) - hx
            front = np.clip(0.35 + 0.12 * fx_ + 0.05 * ((yy - G.M) - hy), 0.15, 1.0)
            E_gold[yy, xx] = np.minimum(E_gold[yy, xx], G.en(1.3)) * front
    E = E_dawn + E_gold
    L = G.lv(E)
    L = G.haze_lift(L, 0.6, pos=sc.pos)
    L = L + SHAFT_AIR * sh * ~G.GLASSPIX           # the lit dust in the air of the shafts
    if ry is not None:
        L = L + 1.3 * ry
    L = L + G.vignette_level(1.3 + 0.8 * gd)
    L[sc.emis] = 0.0
    which = G.ratio_index(E_dawn, E_gold, 5)
    img = G.relight_scene(sc, L, G.mix_tints('dawn', 'gold', 5), which, band=0.3)
    # glass keeps its dawn colours (the clouds beyond it drift slowly) + reflects the gold
    gm = G.GLASSPIX & ~sc.actor
    img[gm] = G.dawn_glass(t)[gm]
    # the shafts are visible in the dusty air in front of the glass too
    G.glass_reflect(img, gm & (sh > 0.05), SHAFT_GLASS * sh, mode='dawn', scale=1.0)
    if pw > 0:
        Eg_ = E_glass * (0.55 + 1.2 * G.DUSTG)
        if buried:
            Eg_ = Eg_ * (1.0 - pw_now(t) * 0 - (1 - pw_now(t)))
        if ry is not None:
            Eg_ = Eg_ + G.en(1.4) * ry
        G.glass_reflect(img, gm, Eg_, mode='glassgold', scale=3.0)
    img[sc.emis] = sc.img[sc.emis]
    img = G.grief_grade(img, gd, keep=sc.emis)

    # ------------------------------------------------------------ FX
    # soil clods flung from the hole at every scrape
    for k, ts in enumerate(SCRAPES):
        if ts <= t < ts + 0.6:
            x_, y_, af, alive, r_ = fx.bursts(t, 7, 700 + k, (sx - 1, sy - 1), speed=(14, 34), life=(0.35, 0.55),
                                              gravity=150.0, spread=(-math.pi * 0.95, -math.pi * 0.05), t0=ts)
            for i in np.nonzero(alive)[0]:
                xi, yi = int(x_[i]) + G.M, int(y_[i]) + G.M
                if 0 <= xi < G.SW and 0 <= yi < G.SH:
                    img[yi, xi] = C('dust2') if r_[i] < 0.5 else C('dust4')
    for k, tp in enumerate(PATS):
        if tp <= t < tp + 0.3:
            u = (t - tp) / 0.3
            for dx in (-3, -2, 2, 3):
                xi = int(sx + dx * (1 + u)) + G.M
                yi = int(sy - u * 2) + G.M
                if noise.rand(k, dx, 710) < 1 - u:
                    img[yi, xi] = C('dust5')
    # the heart meets the soil: a spray of gold sparks + a puff of soil
    cxs, cys = kx + RA.CORE_PLACE_OFFSET[0], ky + RA.CORE_PLACE_OFFSET[1]
    G.spark_burst(img, cxs, cys, t, T_TOUCH, n=18, seed=731, speed=(18, 50), life=(0.25, 0.55))
    G.dust_puff(img, cxs, cys + 1, t, T_TOUCH, n=10, seed=732, spread=6, rise=3, life=0.5,
                cols=('dust3', 'dust4'))
    # the lifeless body settles: a last puff of dust
    t_still = T_POWER + getattr(RA, 'POWER_STILL_T', 1.0)
    G.dust_puff(img, kx - 2, ky, t, t_still - 0.35, n=12, seed=733, spread=8, rise=3, life=0.8)
    # dust motes drifting and glittering in the dawn shafts, gold motes in the heart's light
    G.dawn_motes(img, t, sh, occ=sc.robot)
    G.shaft_motes(img, t, np.zeros_like(sh), E_gold)
    # the low sun catches a shard or two of the dome
    for (t0, gx, gy, gd, gs) in GLINTS:
        G.glass_glint(img, t, t0, gx, gy, dur=gd, size=gs, occ=sc.actor)
    # the heart itself (the orb is in the robot sprite until it touches the soil)
    if pw > 0.02:
        if not buried:
            G.orb_halo(img, hx, hy, min(pw, 1.2), t)
        else:
            # sinking: the heart's glow shrinks into the soil
            u = ease.prog(t, T_TOUCH, T_TOUCH + 0.35)
            if u < 1:
                G.core_orb(img, hx, hy + 2.5 * u, (1 - u), t, flare=False, occ=sc.robot,
                           clip_below=hy + 1)
    if buried:
        # cracks in the soil glowing from below as the light sinks, then fading
        a_ = max(0.0, 1.0 - (t - T_TOUCH) / 1.2)
        glow_r = 3 + 9 * ease.out_cubic(ease.prog(t, T_TOUCH, T_TOUCH + 0.4))
        dd = np.hypot(G.FX - hx, (G.FY - hy) * 1.8)
        cm = G.SOIL_CRACK & (dd < glow_r) & ~sc.actor & (G.BAY < a_ * 1.2 - dd / glow_r * 0.5)
        img[cm & (dd < glow_r * 0.45)] = C('core5')
        img[cm & (dd >= glow_r * 0.45)] = C('core3')
        # the faint buried glow persisting into S8a
        G.buried_glow(img, hx, hy, t - T_TOUCH, occ=sc.actor)
    return G.crop(img)
