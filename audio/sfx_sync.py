"""
Picture sync for the SFX stem.

Reads the shot modules' own timing constants (walk phases and contact frames,
startle/scoop/rise beats, flicker frames, the heart's clock, the bird's flight
path...) so footsteps and foley land ON the drawn frames. Every event is ANCHORED
ON ITS CUE (cues.CUES: the cut cues for walk contacts, the story cue for each
beat); only the offsets inside a gesture come from the shots. So editing cues.py
alone still retimes the whole stem, and when cues and picture agree (the normal
workflow) the sounds sit on the drawn frames.

If a shot module can't be imported (broken WIP, missing file) the documented
defaults below are used (verified against the shots on 2026-09-30) and a note
is printed - the render never fails because of picture code.

    SY = picture_sync(T)   ->  dict of global times (s) / screen positions / callables
"""
import importlib
import math

W_SCREEN = 320.0


def pan_of_x(x):
    """screen x (0..320) -> pan (-1 .. +1)."""
    return max(-1.0, min(1.0, float(x) / W_SCREEN * 2.0 - 1.0))


_NOTES = []


def _mod(name):
    try:
        return importlib.import_module(name)
    except Exception as e:                                     # pragma: no cover - fallback path
        _NOTES.append(f'{name}: not importable ({e.__class__.__name__}: {e}); using documented defaults')
        return None


def _get(mod, attr, default):
    if mod is None:
        return default
    return getattr(mod, attr, default)


def _contacts_from_legs(legs_near, legs_far_fn, nfr):
    """Contact frames of a cyclic walk: frames where a foot's lift goes >0 -> 0."""
    out = []
    for k in range(nfr):
        for leg in (legs_near, [legs_far_fn(j) for j in range(nfr)]):
            if leg[(k - 1) % nfr][1] > 0 and leg[k][1] == 0:
                out.append(k)
    return sorted(set(out))


def shot_starts():
    film = _mod('film')
    starts = {}
    if film is not None:
        for s in film.TIMELINE:
            starts.setdefault(s.module.split('.')[-1], s.start)
    return starts


def picture_sync(T):
    _NOTES.clear()
    robot = _mod('chars.robot')
    RA = _mod('chars.robot_acts')
    ST = shot_starts()
    SY = {}

    # ------------------------------------------------------------------ S1 (wide walk)
    s01 = _mod('shots.s01_wasteland')
    t0 = T['fade_in']
    phases = [c[0] for c in _get(s01, 'CONTACTS', ((0.4, 0), (0.9, 2)))]
    cyc = 2 * float(_get(s01, 'STEP_PERIOD', 0.5))
    SY['s1_contacts'] = sorted(t0 + n * cyc + ph for n in range(40) for ph in phases
                               if t0 + n * cyc + ph < T['cut_s2'])
    if s01 is not None:
        SY['s1_x'] = lambda t: s01.robot_x(t - t0) - s01.cam(t - t0) * s01.F['mid']
        SY['s1_gust'] = lambda t: s01.gust(t - t0)
        hub = s01.TURB.get('hub', (60.0, 70.0))
        SY['s1_turbine_x'] = lambda t: hub[0] - s01.cam(t - t0) * s01.F['fmid']
    else:
        SY['s1_x'] = lambda t: 92.0 + 2.0 * (t - t0)
        SY['s1_gust'] = lambda t: min(1.0, max(0.0, 0.12 + 0.95 * math.exp(-((t - t0 - 2.4) / 0.95) ** 2)
                                               + 0.85 * math.exp(-((t - t0 - 5.7) / 0.85) ** 2)
                                               + 0.08 * math.sin((t - t0) * 5.1)))
        SY['s1_turbine_x'] = lambda t: 60.0

    def rotor_deg(t):
        """the loose turbine blade's rock (deg), stepped at 8 fps (s01.turbine_rotor)."""
        tq = math.floor((t - t0) * 8.0 + 1e-6) / 8.0
        ph = tq / 3.3 * 2 * math.pi
        return 1.5 * math.sin(ph) + 0.45 * math.sin(2.7 * ph + 1.0)
    SY['s1_rotor_deg'] = rotor_deg
    SY['s1_rotor_fps'] = 8.0

    # ------------------------------------------------------------------ medium walk contact frames
    if robot is not None and hasattr(robot, '_WK_N'):
        wk = _contacts_from_legs(robot._WK_N, lambda k: robot._WK_N[(k + 4) % 8], 8)
    else:
        wk = [0, 4]
    SY['walk_contact_frames'] = wk                    # 10 fps, 8-frame cycle

    # ------------------------------------------------------------------ S3 (discovery)
    s03 = _mod('shots.s03_discovery')
    b = T['cut_s3']
    WALK_PH = float(_get(s03, 'WALK_PH', 0.1))
    T_STOP = float(_get(s03, 'T_STOP', 1.8))
    X0 = float(_get(s03, 'X0', 116.0))
    trav = (lambda a: robot.travel('walk', a)) if robot is not None else (lambda a: math.floor(a * 10 + 1e-6))
    c3 = []
    for k in range(0, 60):
        if k % 8 in wk:
            tl = k / 10.0 - WALK_PH
            if 0.0 <= tl < T_STOP:
                x = X0 + trav(tl + WALK_PH) - trav(WALK_PH)
                c3.append((b + tl, x, 'near' if k % 8 == wk[0] else 'far'))
    SY['s3_contacts'] = c3
    SY['s3_startle'] = [T['robot_stop'] + d for d in (0.0, 0.1, 0.3, 0.4)]   # squash, hop+snap, land, settle
    SY['s3_halt_x'] = float(_get(s03, 'X_HALT', 134.0))
    SY['s3_gust'] = (T['gust'], T['gust'] + float(_get(s03, 'T_GUST1', 1.2)) - float(_get(s03, 'T_GUST0', 0.4)))
    SY['glint_x'] = float(_get(s03, 'SPROUT_X', 230.0))
    SY['s3b_kneel_x'] = float(_get(s03, 'KNEEL_X', 217.0))
    sc = T['scoop']
    SY['scoop_lean'] = sc + 0.1                                            # robot.py 'scoop' reference
    SY['scoop_strokes'] = [sc + 0.3, sc + 0.5]
    SY['scoop_lift'] = sc + float(_get(robot, 'SCOOP_LIFT_T', 0.75))
    SY['scoop_incan'] = T['into_can']
    tr = T['happy_beep'] - 0.4                         # rise_hold's happy hop (t=0.4) lands on the beep
    SY['rise'] = dict(start=tr, grab=tr + 0.1, swing=tr + 0.2, up=tr + 0.3, hop=tr + 0.4)

    # ------------------------------------------------------------------ S4 (journey)
    s04 = _mod('shots.s04_journey')
    b = T['cut_s4']
    ST1, LU1 = float(_get(s04, 'T_STOP1', 2.5)), float(_get(s04, 'T_LOOK_UP1', 3.2))
    GO2, ST2 = float(_get(s04, 'T_GO2', 3.5)), float(_get(s04, 'T_STOP2', 4.8))
    LUP = float(_get(s04, 'T_LOOKUP', 5.0))
    ph1 = float(_get(s04, 'WALK_PH1', 0.4)) % 0.4
    stop1 = T['look_down_can']                         # walk 1 stops as it looks down at the sprout
    go2 = stop1 + (GO2 - ST1)
    stop2 = T['robot_looks_up'] - (LUP - ST2)          # walk 2 stops just before it looks up
    c4 = [b + 0.4 * k - ph1 for k in range(40) if 0.0 <= 0.4 * k - ph1 and b + 0.4 * k - ph1 < stop1]
    c4 += [go2 + 0.4 * k for k in range(40) if go2 + 0.4 * k < stop2]
    SY['s4_contacts'] = c4
    SY['s4_x'] = float(_get(s04, 'SCREEN_X', 100.0))
    SY['s4_stop1'] = stop1
    SY['s4_lookup1'] = stop1 + (LU1 - ST1)
    SY['s4_go2'] = go2
    SY['s4_stop2'] = stop2

    # ------------------------------------------------------------------ S5 (storm)
    s05 = _mod('shots.s05_storm')
    b = T['cut_s5']
    if robot is not None and hasattr(robot, '_BR_N'):
        br = _contacts_from_legs(robot._BR_N, lambda k: robot._BR_F[k], 18)
    else:
        br = [7, 16]
    T_SHIELD = T['shield'] - b
    T_BRACE2 = T['trudge'] - b + float(_get(s05, 'T_BRACE2', 4.5)) - float(_get(s05, 'T_RISE', 4.0))
    c5 = []
    for seg0, seg1 in ((0.0, T_SHIELD), (T_BRACE2, T['cut_s6'] - b)):
        for n in range(8):
            for k in br:
                tl = seg0 + n * 1.5 + k / 12.0
                if seg0 <= tl < seg1:
                    c5.append(b + tl)
    SY['s5_contacts'] = sorted(c5)
    SY['s5_x'] = float(_get(s05, 'RX', 112)) - float(_get(s05, 'M', 8))
    SY['s5_front'] = b + float(_get(s05, 'T_FRONT', 0.33))                 # the surge rolls over the robot
    SY['s5_branch'] = b + 1.28                                             # s05 docstring: branch ripped off
    SY['s5_gust'] = b + float(_get(s05, 'T_GUST', 1.42))
    SY['s5_sheet_in'] = T['antenna_rip'] - (float(_get(s05, 'T_IMPACT', 2.5)) - float(_get(s05, 'T_SHEET_IN', 2.10)))
    SY['s5_rise'] = T['trudge']
    SY['s5_brace2'] = b + T_BRACE2
    SY['s5_wall'] = b + 5.1                                                # s05 docstring: pale wall swallows it

    # ------------------------------------------------------------------ S6 (night: limp + sit)
    s06 = _mod('shots.s06_night')
    b = T['cut_s6']
    T_SIT = T['sit_down'] - b - float(_get(RA, 'SIT_IMPACT_T', 7 / 12))   # 'sit' starts so the impact hits the cue
    SY['s6_hurt_contact'] = b + 0.0                   # walk_tired k=0: near (hurt) foot planted
    SY['s6_hitch'] = b + 0.1                          # 0.1-0.3 painful hitch onto the hurt leg
    SY['s6_good_contact'] = b + 0.3                   # k=3: the good (far) leg lands
    SY['s6_drag'] = (b + 0.4, b + T_SIT)              # the hurt toe drags (until the sit takes over)
    SY['s6_sit_start'] = b + T_SIT
    SY['s6_can_touch'] = b + T_SIT + 1.0 / 3.0        # robot_acts: 'sets the can down (touch 1/3 s)'
    SY['s6_can_release'] = b + T_SIT + float(_get(RA, 'SIT_CAN_DOWN_T', 0.45))
    SY['s6_impact'] = b + T_SIT + float(_get(RA, 'SIT_IMPACT_T', 7 / 12))

    # ------------------------------------------------------------------ c07 (the last drop) - frames @30 fps
    f = 1 / 30.0
    df, dh = T['drop_fall'], T['drop_hit']
    SY['c07'] = dict(start=df - 0.3, jiggles=[df - 9 * f, df - 3 * f], flick=df - f,
                     satellite=df + 2 * f, crown=[dh + f, dh + 2 * f], last_shake=dh + 3 * f,
                     bump=(dh + 5 * f, dh + 8 * f), sag=(dh + 11 * f, dh + 14 * f))

    # ------------------------------------------------------------------ c05 (the lean close-up)
    c05 = _mod('shots.c05_lean')
    b = T['sprout_lean'] - float(_get(c05, 'T_LEAN0', 0.20))
    SY['c05'] = dict(start=b, stutter=(b + 0.56, b + 0.74), dips=[b + 1.93, b + 3.30],
                     swell=(b + 2.62, b + 2.92, b + 3.0, b + 3.35), iris=[b + 1.60, b + 1.667],
                     widen=(b + 2.9, b + 3.4), nod=T['look_chest'],
                     fade=(b + 3.4, b + 3.7))

    # ------------------------------------------------------------------ c03 (the heart)
    c03 = _mod('shots.c03_core')
    b = T['cut_c03_core']
    t_open = T['hatch_open']
    seq = list(_get(c03, 'DOOR_SEQ', [30, 64, 98, 126, 112, 120]))
    frames = [t_open + k / 15.0 for k in range(len(seq))]
    imax = max(range(len(seq)), key=lambda k: seq[k])
    hz = float(_get(c03, 'BEAT_HZ', 1.2))
    lift = T['core_out']
    T_LIFT = float(_get(c03, 'T_LIFT', 1.20))
    SY['c03'] = dict(latch=[b + float(_get(c03, 'T_LATCH1', 0.033)), b + float(_get(c03, 'T_LATCH2', 0.067))],
                     ajar=t_open - (float(_get(c03, 'T_OPEN', 0.40)) - float(_get(c03, 'T_AJAR', 0.30))),
                     open=t_open, door_frames=frames,
                     door_stop=frames[imax], door_bounce=frames[imax + 1:] if imax + 1 < len(frames) else [],
                     reach=(lift - (T_LIFT - float(_get(c03, 'T_REACH', 0.80))), lift), lift=lift, beat_hz=hz,
                     beats=[(lift + (n + 0.03) / hz, lift + (n + 0.25) / hz) for n in range(3)])
    # S7b (wide) re-phases the heart on 'place_core' (robot_acts clock, 12 fps rig)
    s07 = _mod('shots.s07_gift')
    t_place = T['core_in_soil'] - float(_get(RA, 'CORE_PLACE_T', 0.4))    # 'place_core' starts (S7b re-phases)
    hz7 = float(_get(RA, 'HEARTBEAT_HZ', 1.2))
    SY['s7b_beat'] = (t_place, t_place + math.ceil(0.2 / hz7 * 12.0 - 1e-9) / 12.0)

    # ------------------------------------------------------------------ c08 (heart into the soil) @30 fps
    c08 = _mod('shots.c08_heart')
    fps8 = float(_get(c08, 'FPS_', 30))
    b = T['core_in_soil'] - float(_get(c08, 'F_PRESS', 6)) / fps8          # c08 local 0 (the press is f6)
    fr = lambda name, d: b + float(_get(c08, name, d)) / fps8
    SY['c08'] = dict(start=b, press=fr('F_PRESS', 6), gone=fr('F_GONE', 16), lub=fr('F_LUB', 12),
                     dub=fr('F_DUB', 14), linger=fr('F_LINGER', 18), last=fr('F_LAST', 23),
                     crack_race=(fr('F_PRESS', 6), fr('F_PRESS', 6) + 4 / fps8))

    # ------------------------------------------------------------------ c06 (the last light)
    c06 = _mod('shots.c06_fade')
    b = T['power_down']                               # c06 local 0 == power_down (film shows it from 0.2)
    f1, f2 = float(_get(c06, 'T_FLICK1', 0.25)), float(_get(c06, 'T_FLICK2', 0.50))
    toff = float(_get(c06, 'T_OFF', 0.90))
    f = 1 / 30.0
    SY['c06'] = dict(dropouts=[(b + f1, b + f1 + 2 * f, 0.0), (b + f2, b + f2 + f, 0.0),
                               (b + f2 + f, b + f2 + 2 * f, 0.5), (b + f2 + 2 * f, b + f2 + 4 * f, 0.0)],
                     weaker=[(b + f1 + 2 * f, 0.85), (b + f2 + 4 * f, 0.75)],
                     breath=(T['eye_off'] - f, T['eye_off']), off=T['eye_off'], sag=[T['eye_off'], T['eye_off'] + f])

    # ------------------------------------------------------------------ c04 (rest: the bird, the eye)
    c04 = _mod('shots.c04_rest')
    T_FLY0, T_LAND = float(_get(c04, 'T_FLY0', 0.05)), float(_get(c04, 'T_LAND', 0.50))
    b = T['bird_land'] - T_LAND                        # c04 local 0, anchored on the landing cue
    LAND_X = float(_get(c04, 'LAND_X', 217))

    def bird_x(t):
        tl = t - b
        if tl < T_FLY0:
            return 336.0
        if tl < T_LAND:
            u = (tl - T_FLY0) / (T_LAND - T_FLY0)
            e = 1 - (1 - u) ** 2
            return 336.0 + (LAND_X - 336.0) * e
        return LAND_X
    T_GLOW = float(_get(c04, 'T_GLOW', 1.20))
    bg = T['eye_glow'] - T_GLOW                        # the hold's beats (hop, tilt, lift, chirp) follow the eye
    T_HOP = float(_get(c04, 'T_HOP', 1.58))
    T_CHIRP = float(_get(c04, 'T_CHIRP', 2.30))
    eye = _get(c04, 'EYE_POS', (206, 112))
    flick_on = [T['eye_glow'] + k / 30.0 for k, v in enumerate((0.30, 0.30, 0.0, 0.42, 0.0, 0.0))
                if v > 0 and (k == 0 or not (0.30, 0.30, 0.0, 0.42, 0.0, 0.0)[k - 1] > 0)]
    SY['c04'] = dict(fly=(b + T_FLY0, b + T_LAND), land=b + T_LAND, land_x=LAND_X, bird_x=bird_x,
                     downstrokes=[b + k / 15.0 for k in range(1, 40, 2) if T_FLY0 <= k / 15.0 < T_LAND - 0.08],
                     brake=(b + T_FLY0 + 0.82 * (T_LAND - T_FLY0), b + T_LAND),
                     hop=(bg + T_HOP + 0.05, bg + T_HOP + 0.17), hop_dx=4.0,
                     chirps=[bg + T_CHIRP, bg + T_CHIRP + 0.13], chirp_len=0.07,
                     head_lift=bg + float(_get(c04, 'T_LIFT', 2.05)),
                     glow=T['eye_glow'], flicker_on=flick_on,
                     steady=T['eye_glow'] + float(_get(c04, 'T_STEADY', T_GLOW + 0.2)) - T_GLOW,
                     eye_x=float(eye[0]))
    # ------------------------------------------------------------------ picture <-> cue sheet consistency
    checks = [('s03_discovery start', ST.get('s03_discovery'), T['cut_s3']),
              ('s04_journey start', ST.get('s04_journey'), T['cut_s4']),
              ('s05_storm start', ST.get('s05_storm'), T['cut_s5']),
              ('s06_night start', ST.get('s06_night'), T['cut_s6']),
              ('c03_core start', ST.get('c03_core'), T['cut_c03_core']),
              ('c08_heart start', ST.get('c08_heart'), T.get('cut_c08_heart')),
              ('c04_rest start', ST.get('c04_rest'), T['cut_c04'])]
    if s03 is not None:
        checks.append(('s03.T_STOP', T['cut_s3'] + T_STOP, T['robot_stop']))
    if s06 is not None:
        checks.append(('s06 sit impact', T['cut_s6'] + float(_get(s06, 'T_SIT_CUE', 1.2)), T['sit_down']))
    if c03 is not None:
        checks.append(('c03.T_LIFT', T['cut_c03_core'] + float(_get(c03, 'T_LIFT', 1.2)), T['core_out']))
    if c08 is not None and T.get('cut_c08_heart') is not None:
        checks.append(('c08.F_PRESS', T['cut_c08_heart'] + float(_get(c08, 'F_PRESS', 6)) / fps8, T['core_in_soil']))
    if c06 is not None:
        checks.append(('c06.T_OFF', T['power_down'] + toff, T['eye_off']))
    if c04 is not None:
        checks.append(('c04.T_LAND', T['cut_c04'] + T_LAND, T['bird_land']))
        checks.append(('c04.T_GLOW', T['cut_c04'] + T_GLOW, T['eye_glow']))
    for name, pic, cue in checks:
        if pic is not None and cue is not None and abs(pic - cue) > 1.0 / 30.0 + 1e-6:
            _NOTES.append(f'picture/cue mismatch: {name} = {pic:.3f} s but cues.py says {cue:.3f} s '
                          f'(SFX follow cues.py)')
    SY['notes'] = list(_NOTES)
    return SY


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '..')
    from cues import CUES
    SY = picture_sync(dict(CUES))
    for k, v in SY.items():
        if not callable(v):
            print(f'{k:20s} {v}')
