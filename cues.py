"""
THE LAST SEED — master cue sheet (GLOBAL film time, seconds).

These are CONTRACTUAL sync points. Picture must hit them within +/-2 frames;
music and sound design are built on them. If a shot owner has to move an
event, they report the new time and the director updates this file.
"""

CUES = {
    # ---------------- S1  THE DEAD WORLD (0.0 - 7.0)
    'fade_in':          0.0,    # dither fade from black (0.0-1.6)
    'robot_steps':      0.8,    # tiny robot walking in the wide shot (soft clanky steps, ~2.2 steps/s)
    # ---------------- S2  THE EMPTY CAN (7.0 - 12.0)
    'cut_s2':           7.0,
    'can_raise':        8.2,    # robot lifts the watering can
    'can_tilt':         8.8,    # tilts it over the dead stalk
    'drop_fall':        9.2,    # ONE drop leaves the spout
    'drop_hit':         9.6,    # ...hits the dry ground (tiny 'tk'), soaks away
    'droop':            10.4,   # robot lowers can, head droops (sad servo whine, descending)
    'leaf_breaks':      11.0,   # a dead leaf snaps off the stalk and tumbles away
    # ---------------- S3  DISCOVERY (12.0 - 20.0, with close-up inserts)
    'cut_s3':           12.0,
    'gust':             12.4,   # wind gust sweeps dust across frame (right -> left)
    'glint':            13.4,   # a GREEN glint appears ahead (sparkle)
    'robot_stop':       13.8,   # robot halts, head snaps toward it (servo jolt)
    'cut_c01_eye':      14.5,   # CLOSE-UP eye: lens servo whirr...
    'eye_focus':        15.2,   # ...iris click-click as it focuses; green reflection grows
    'cut_c02_sprout':   16.0,   # CLOSE-UP sprout: hush; first warm 'life' motif
    'hand_enter':       16.6,   # robot fingers enter frame, slowly
    'leaf_touch':       17.1,   # fingertip touches the leaf tip; leaf bobs
    'cut_s3b':          17.5,   # back to medium: robot kneeling by the sprout
    'scoop':            17.9,   # digs around it (soil scrape)
    'into_can':         19.0,   # soil + sprout set into the can (soft thunk)
    'happy_beep':       19.5,   # robot's tender little beep-boop (its 'voice' motif)
    # ---------------- S4  JOURNEY (20.0 - 26.0)
    'cut_s4':           20.0,   # dusk tracking shot; walking rhythm
    'look_down_can':    22.5,   # robot looks down at the sprout (soft beep)
    'storm_rise':       24.0,   # wind rises; dark dust-wall on the horizon swallows the sun; low rumble
    'robot_looks_up':   25.0,   # robot stops, looks up at the storm
    # ---------------- S5  STORM (26.0 - 32.0)
    'cut_s5':           26.0,   # STORM HITS (roar)
    'shield':           27.5,   # robot turns its back to the wind, curls around the can
    'antenna_rip':      28.5,   # flying metal sheet strikes: antenna snaps off (clang + zap + sparks)
    'core_flicker':     29.6,   # chest core stutters (3 quick electrical crackles)
    'trudge':           30.0,   # robot rises and trudges on into the wind
    'storm_out':        31.65,  # fade to black begins
    # ---------------- S6  NIGHT IN THE GREENHOUSE (32.0 - 38.0)
    'cut_s6':           32.0,   # sudden quiet: faint wind, glass tinkles (fade in 32.0-32.8)
    'sit_down':         33.2,   # robot sits (metal clank)
    'sprout_lean':      34.5,   # the sprout slowly bends toward the robot's glowing chest (shimmer)
    'robot_notice':     35.5,   # robot turns to the can (soft questioning beep)
    'look_chest':       36.6,   # robot looks down at its own core (low hum) — realisation
    # ---------------- S7  THE GIFT (38.0 - 45.0, with close-up insert)
    'cut_s7':           38.0,   # grey dawn; cold air
    'dig':              38.5,   # digs a hole in the planter (3 scrapes 38.5-39.2)
    'plant_pat':        39.6,   # pats the soil (pat pat)
    'cut_c03_core':     40.0,   # CLOSE-UP chest: hatch latch click
    'hatch_open':       40.4,   # hinge creak, hatch swings open
    'core_out':         41.2,   # core lifted out: resonant warm hum swells, heartbeat pulse
    'cut_s7b':          42.1,   # wide: the whole greenhouse lit gold by the heart in its hands
    'cut_c08_heart':    42.8,   # CLOSE-UP: hands press the heart into the soil beside the sprout
    'core_in_soil':     43.0,   # presses core into the soil (soft thump; light sinks in)
    'power_down':       43.4,   # descending power-down whine (0.9 s), eye flickers
    'cut_c06_fade':     43.6,   # CLOSE-UP: the robot's face as its eye dies
    'eye_off':          44.3,   # final click; eye dark; body settles
    'silence':          44.5,   # TOTAL SILENCE until first_pulse
    # ---------------- S8  BLOOM (45.0 - 53.0)
    'first_pulse':      45.3,   # one deep heartbeat of light in the soil
    'veins':            45.8,   # glowing veins race through the ground (rising shimmer)
    'growth_burst':     46.8,   # sprout surges upward (whoosh, creaking wood) -> crescendo
    'cut_s8b_climax':   48.0,   # EXTERIOR: tree bursts through the dome — CLIMAX HIT (glass burst + choir)
    'color_wave':       48.6,   # wave of life sweeps across the valley (48.6-51.5)
    'sun_break':        50.2,   # clouds tear open, sun rays break through
    'birds':            51.0,   # birds fly in
    # ---------------- S9  REST (53.0 - 60.0)
    'cut_s9a':          53.0,   # lush valley (dissolve 53.0-53.6): birdsong, breeze, stream
    'cut_c04':          55.0,   # CLOSE-UP overgrown robot
    'bird_land':        55.5,   # a small bird lands on its head (flutter + chirp)
    'eye_glow':         56.2,   # the robot's eye lights up soft GREEN (its beep motif, gentle)
    'title':            58.0,   # title card fades in over the picture (2.0 s)
    'end':              60.0,
}
