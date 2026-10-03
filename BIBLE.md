# THE LAST SEED — Production Bible

> A 60-second, wordless, pixel-art short film. 320×180 native pixels, shown at
> 6× (1920×1080), 30 fps, with an original score and full sound design.
>
> **The bar:** the client asked for something so impressive that *no one could
> imagine it was made by an AI rather than a professional team working for
> months.* Every frame must look intentional, hand-crafted and alive. When in
> doubt, do the harder, more beautiful thing.

Project root: the repository root (run everything from there).

---

## 1. The story

**Logline.** In a dead, grey world, a small rusty gardening robot finds the last
living sprout, carries it across the wasteland through a sandstorm to its old
greenhouse — and gives up its own glowing heart to make it bloom.

**Theme.** Giving your light away is how the world gets its color back.

**The through-line of color (the film's backbone).** The first half of the film
is almost colorless: warm greys, dust, ash, rust. The ONLY saturated things are
the robot's amber chest core (its heart) and — from 13.4 s — the sprout's vivid
green. When the robot gives its heart (43 s), color floods back into the whole
world (48–53 s). The ending is lush and saturated.

**Setup / payoff the audience must feel:**
1. The robot is a *gardener* with an *empty* watering can → loneliness, devotion.
2. The sprout is the only green in the world → hope.
3. At night the dying sprout **leans toward the robot's glowing chest** → it needs light; the robot's core IS light. (Setup.)
4. The robot takes out its core and plants it with the sprout → sacrifice. (Payoff.)
5. The world is reborn; later, a bird lands on the moss-covered robot and its eye glows **green** → what you give comes back.

**Characters.** Only the robot (and, at the very end, a small songbird). No humans.

---

## 2. The robot — "the Gardener"

Canonical model: `chars/model.py` (`IDLE`, 18×26 px, faces right). Look at
`review/model_v2.png`. Every depiction — wide, medium, close-up — must match it.

* Big head (≈ 40 % of height), faded retro **mint paint** (`paint` ramp), rust
  streaks near the bottom edges, a round **ear-bolt** on the side.
* ONE big round **camera-lens eye** on the front-right of the head: dark steel
  barrel ring (lighter on top-left), dark glass, a small **amber pupil**
  (`core3`–`core6`), a white glint top-left. (Close-ups: iris aperture blades
  inside the glass, focus rings on the barrel.)
* Thin **antenna** from the top-back of the head with a tiny amber bulb.
  Ripped off in the storm (28.5 s) → from then on only a 1-px stub.
* Short steel neck. Compact rounded **body**, mint paint, rusty lower rim.
* A round **porthole on the chest** (front-right) showing the glowing amber
  **power core** — its heart. The porthole is a hatch that can open (close-up
  c03). After 43 s it is open and empty (later a flower grows from it).
* Cracked **solar-panel backpack** on its back (dark steel frame, ash-blue cells).
* Thin steel **arms**, simple 3-finger hands. Stubby steel legs, rust-colored boots.
* Outline: `steel0`. Light comes from the **upper left** in neutral lighting.
* Personality: gentle, earnest, a little clumsy; moves with small servo
  jolts and settles (overshoot + settle). Its "voice" is tiny melodic beeps.

**Continuity table**

| Shot | Variant | Can | Core glow | Eye | Antenna |
|---|---|---|---|---|---|
| S1 wide | normal | empty | ~0.8, steady | on | yes |
| S2 | normal | empty | 0.7 (one weak flicker at 10.4) | on | yes |
| S3 | normal | empty → **sprout** from 19.0 | 0.7 | on | yes |
| c01 | normal | – | – | on, amber | yes |
| S4 | normal | sprout | 0.6 | on | yes |
| S5 | normal → **damaged at 28.5** | sprout (hugged) | 0.6 → stutters 29.6 → 0.35 | on | ripped 28.5 |
| S6 | damaged | sprout_wilt → sprout_lean (34.5) | 0.3, flickery | dim | stub |
| S7a | damaged | sprout moved into soil; can empty on ground | 0.3 | dim | stub |
| c03 | damaged | – | core OUT: **blazing** (full, pulsing) | – | – |
| S7b | damaged | can empty on ground | core in hands → soil → **0** | off at 44.3 | stub |
| S8a | damaged, `dead_sit` | – | 0, chest hatch open & empty | off | stub |
| c04 | damaged + overgrown (moss, flowers, a flower in the chest) | – | – | off → **green** at 56.8 | stub |

**Screen direction.** The journey always moves **left → right**. The greenhouse
dome is always toward the **right** (ahead). The storm wind blows from the
**right** (against the robot, dust streams right → left).

---

## 3. Shot list (exact timings)

Global time = film clock. Each shot module has its own clock `t` starting at 0.
Some medium shots are interrupted by close-ups: the medium shot's clock keeps
running "off-screen" during the cutaway (see *hidden* ranges).

| # | Module (owner) | Global | Module `t` | Scale / camera |
|---|---|---|---|---|
| S1 | `shots/s01_wasteland.py` (WASTELAND) | 0.0–7.0 | 0–7 | ultra-wide, slow parallax truck right |
| S2 | `shots/s02_empty_can.py` (DISCOVERY) | 7.0–12.0 | 0–5 | medium, static (optional 1–2 px drift) |
| S3a | `shots/s03_discovery.py` (DISCOVERY) | 12.0–14.5 | 0–2.5 | medium |
| c01 | `shots/c01_eye.py` (CLOSEUPS) | 14.5–16.0 | 0–1.5 | extreme close-up |
| c02 | `shots/c02_sprout.py` (CLOSEUPS) | 16.0–17.5 | 0–1.5 | extreme close-up, ground level |
| S3b | `shots/s03_discovery.py` | 17.5–20.0 | 5.5–8.0 (*2.5–5.5 hidden*) | medium |
| S4 | `shots/s04_journey.py` (WASTELAND) | 20.0–26.0 | 0–6 | medium tracking, deep parallax |
| S5 | `shots/s05_storm.py` (STORM) | 26.0–32.0 | 0–6 | medium, violent shake; fades out 31.65–32.0 |
| S6 | `shots/s06_night.py` (GREENHOUSE) | 32.0–38.0 | 0–6 | medium interior, static; fades in 32.0–32.8 |
| S7a | `shots/s07_gift.py` (GREENHOUSE) | 38.0–40.0 | 0–2.0 | same framing as S6 |
| c03 | `shots/c03_core.py` (CLOSEUPS) | 40.0–42.5 | 0–2.5 | close-up of chest |
| S7b | `shots/s07_gift.py` | 42.5–45.0 | 4.5–7.0 (*2.0–4.5 hidden*) | same framing |
| S8a | `shots/s08_bloom_int.py` (GREENHOUSE) | 45.0–48.0 | 0–3 | same framing, builds to shake |
| S8b | `shots/s08_bloom_ext.py` (BLOOM) | 48.0–53.0 | 0–5 | exterior wide, slow pull-back |
| S9a | `shots/s09_valley.py` (BLOOM) | 53.0–55.0 | 0–2 (dissolve-in 0.6 s) | wide, gentle drift |
| c04 | `shots/c04_rest.py` (CLOSEUPS) | 55.0–60.0 | 0–5 | close-up overgrown robot |
| T | `shots/title.py` (TITLE) overlay | 57.0–60.0 | 0–3 | title over c04 |

Exact event times are in `cues.py` (global seconds). **Hit them ±2 frames.**
Local time = global − shot start (+ offset for the second half of S3/S7).

### S1 · THE DEAD WORLD (0–7)
Ultra-wide establishing. A dust-choked sky: heavy overcast of warm greys, the
sun only a pale smudge through the haze. Far horizon: the skeletal ruins of a
city (broken towers, a snapped bridge) in cool `ash` greys, low contrast. Mid:
endless dunes, dead tree skeletons, half-buried rusted machines (a toppled
wind turbine, pylons, a car husk). Near: cracked earth, stones. Foreground:
dark silhouetted debris/dead branches sliding past fast (parallax depth). Dust
motes float; sand streams skim the ground in the wind. **The tiny robot (wide
scale)** walks left→right across the mid-ground — its chest core the only warm
point of light in the frame. **Far right horizon: a tiny dome silhouette on a
hill** (the greenhouse — subtle). Camera trucks right slowly, eases out at the end.

### S2 · THE EMPTY CAN (7–12)
Medium. The robot stands (left of center, facing right) by a dead plant — a
brittle, dry, leafless stalk with two last dead grey leaves — in cracked earth.
Hazy wasteland behind (leaning fence post, distant ruins, lower contrast).
Local timeline: idle looking at the stalk → `water` anim (raise 1.2, tilt 1.8)
→ ONE drop falls from the spout (2.2) → hits the dry ground (2.6): a 1–2 px
dark wet spot that fades away in ~0.6 s → the robot lowers the can and
**droops** (3.4) → hold. A dead leaf snaps off the stalk (4.0) and tumbles
away in the wind. Its core flickers weakly once around 3.6.

### S3 · DISCOVERY (12–20, module 8 s)
**Part A (t 0–2.5):** the robot walks in from the left (can empty). At 0.4 a
wind gust sweeps a veil of dust across frame (right→left); as it clears, it has
uncovered something ahead (~x 230, on the ground line). At 1.4 a **green glint**
twinkles there — the first saturated green of the film. At 1.8 the robot halts
with a `startle` (head snaps toward it). Hold.
*Hidden (2.5–5.5): it walks up and kneels (plan the positions so part B matches).*
**Part B (t 5.5–8.0):** the robot kneels on the LEFT of the sprout, facing
right. `scoop` from 5.9: digs around it, lifts the soil clump with the sprout,
sets it into the watering can (7.0). Then it hugs the can to its chest (7.3+);
its pupil glints; a tiny happy bounce with the beep at 7.5. The sprout gives
off a faint green glow (`light` mode `green`, very small radius).

### c01 · THE EYE (14.5–16)
Extreme close-up of the robot's lens (the lens ≈ 110–130 px across), 3/4 view
facing right, mint paint and rust around it, dust drifting in front. Inside the
glass: iris aperture blades, focus rings on the barrel. At 0.7 the iris
**contracts and refocuses** (blades rotate, rings shift, tiny servo shake). A
curved reflection in the glass shows the grey sky — and a small **green**
point (the sprout) that grows brighter as it focuses.

### c02 · THE SPROUT (16–17.5)
Ground-level extreme close-up. A tiny sprout (≈ 50–60 px tall) growing from a
crack in dry, cracked earth; the split **seed husk** still at its base (the
"last seed"). Pale-green stem, two small leaves, one drooping; it trembles in
the wind; dust grains skitter past. Background: soft low-contrast bands of
horizon. At 0.6 the robot's steel fingers enter from the LEFT, slowly; at 1.1 a
fingertip touches a leaf tip and the leaf bobs. The green here is the most
saturated color in the film so far.

### S4 · THE JOURNEY (20–26)
Medium tracking shot at dusk. The robot (medium scale) walks left→right
holding the can with the sprout; the camera tracks so it stays in the left
third while the world scrolls: foreground rocks sliding fast, dunes, the ruins,
and ahead on the far horizon the **dome on its hill**. The dust sky glows a dim
rust-ochre sunset; the sun is a dull red disc low on the right. Long shadows.
At 2.5 it looks down at the sprout (tender). At 4.0 the wind rises; on the
horizon ahead a **towering dark wall of dust** rises and swallows the sun; the
light dims. At 5.0 the robot stops and looks up at it. Cut.

### S5 · THE STORM (26–32)
The storm hits. Medium, violent camera shake, visibility collapsing into ochre
murk; 3 depths of dust (far haze, mid streaks, near fast debris: stones, a
tumbling dead bush, a metal sheet). Wind from the RIGHT. 0–1.5 the robot
`brace`s, leaning into the wind, can hugged. 1.5 a huge gust: it turns its back
to the wind and **curls around the can** (`shield`), protecting the sprout. 2.5
a flying metal sheet strikes its head: **antenna snaps off** and tumbles away,
a spray of sparks (variant `damaged` from here). 3.6 the **core stutters**
(three quick flickers, then dimmer). 4.0–6.0 it rises and trudges forward into
the wind (`brace`, damaged) while the dust thickens toward a whiteout. The
amber core pool and the sprout's green are the only color.

### S6 · NIGHT IN THE GREENHOUSE (32–38)
Inside the ruined Victorian greenhouse at night. Darkness; the dome's iron ribs
faintly silhouetted against a starless dust-night sky through broken panes;
broken glass on the floor. A round **stone planter bed** of dry soil on the
RIGHT third. 0–1.2 the robot (damaged, `walk_tired`) limps in from the left,
sets the can down on the ground on its right, and **sits** (1.2) with its back
against a fallen iron rib/rubble on the left, facing right. Its dim flickering
core is the only light (warm pool) plus the faint green of the wilting sprout.
2.5: **the sprout slowly leans toward the robot's glowing chest** (≈ 1.2 s).
3.5: the robot notices (`sit_look_can`). 4.6: it looks down at its own chest
(`sit_look_chest`) — realization. Hold. Quiet, intimate, heartbreaking.

### S7 · THE GIFT (38–45, module 7 s)
Same framing, **grey dawn**: cold pale light through the broken dome.
**Part A (0–2.0):** the robot kneels at the planter (on its right): `dig_plant`
— digs (0.5), moves the sprout from the can into the hole, pats the soil
(1.6). The sprout stands in the dead soil, wilting.
*Hidden (2.0–4.5): it opens its chest and lifts out the core (seen in c03).*
**Part B (4.5–7.0):** the robot kneels holding its **blazing core** in both
hands — the whole greenhouse lit warm gold (huge contrast with the cold dawn).
5.0: it **presses the core into the soil** beside the sprout; the light sinks
into the ground and dims. 5.4: `power_down` — the eye flickers and dies (6.3),
it slumps into a seated heap. 6.3–7.0 stillness, cold light only.

### c03 · THE HEART (40–42.5)
Close-up of the robot's chest: the round porthole hatch (≈ 70 px), rivets, rust,
mint paint, the core glowing behind the glass. Steel fingers unlatch it (0.0–0.2),
it swings open (0.4), fingers reach in and **lift the core out** (1.2): a
luminous amber orb in a cradle of wires; its light floods the frame and pulses
like a heartbeat. Hold on the orb in the hands.

### S8a · BLOOM, INSIDE (45–48)
Same framing; the slumped robot, dark. 0.3: the soil **pulses** once (a ring of
warm light — a heartbeat). 0.8: **glowing veins** (green-gold) race outward
through the soil and across the floor, branching like roots, climbing the
planter and reaching under the robot. 1.2: the sprout straightens and glows;
leaves unfurl. 1.8: **GROWTH BURST** — the stem thickens into a trunk and
shoots up out of frame, branches whip outward, leaves pop, light flares, motes
swirl, the dome's glass trembles, shake builds to the cut. Color begins to
return locally (rich dark soil, moss spreading).

### S8b · BLOOM, OUTSIDE (48–53) — the climax
Exterior wide: the hill with the ruined greenhouse; the whole dead valley
beyond, ruins on the horizon, dust sky. 0.0: **the tree bursts up through the
dome's apex** — glass shards explode outward sparkling; trunk rising, canopy
exploding outward (leaf clusters + blossoms). 0.6: a **wave of life** races
outward from the tree across the land: behind its glowing wavefront, grass
sweeps over the dunes, flowers pop, dead trees leaf out, vines climb ruins.
2.2: the **dust clouds tear open**, golden god-rays break through, blue sky
spreads, the sun appears. 3.0: birds. 3–5: slow pull-back / tilt reveals the
reborn world.

### S9a · THE VALLEY (53–55)
Time has passed. Lush valley in soft morning light; the giant tree over the
overgrown dome; meadows of flowers rippling in the wind; a sparkling stream;
birds, butterflies. Gentle drift. Serene.

### c04 · REST (55–60)
Close-up of the robot's head and shoulders where it sat — overgrown with moss,
tiny flowers in its seams, a flower growing from its open chest porthole,
dappled sunlight through leaves moving across it, petals drifting. Lens dark
and dusty. 0.8: a **small songbird** flutters in and lands on its head, hops,
chirps. 1.8: the lens flickers… and lights up **soft green** (leaf ramp),
slowly brightening; the bird tilts its head at it. 2.0–5.0 hold (title
overlays from 2.0). Export `EYE_POS = (x, y)` (the pupil center in screen
pixels) for the title overlay.

### Title (57–60)
"THE LAST SEED" in a custom pixel display font, over c04 as it fades to near-black,
the green eye glow staying alive in the dark. The film ends with a global fade.

---

## 4. Look & color

**Palette:** `engine/palette.py` (`review/palette.png`). Use ONLY these colors,
by name: `C('dust5')`, `RAMPS['leaf'][3]`… The compositor snaps every frame to
this palette; off-palette colors will be forced to their nearest neighbour, so
author in palette colors to stay in control.

| Section | Ramps | Mood / light |
|---|---|---|
| S1–S3, c01–c02 | dust, ash, ochre (muted), rust, steel, paint + core (robot) + **leaf only for the sprout** | flat dusty overcast, top-left soft light, haze |
| S4 | + warmer ochre/rust dusk, `core0-3` for the dull sun | low sun from the right, long shadows |
| S5 | ochre (dark end), dust, ash | murk; the core pool is the key light |
| S6 | night, ink, glass, steel | darkness; the dim amber core pool; faint green |
| S7a | ash, glass, dust (cold) | pale cold dawn from above-right |
| S7b/c03 | + core full range | the removed core blazes gold |
| S8a | cold → green/gold veins (`leaf`, `gold`, `core`) | light from below (the soil) |
| S8b/S9a/c04 | everything: meadow, leaf, sky, bloom, gold, violet, red, bark, white | warm sunlight, blue sky, saturated |

---

## 5. Pixel-art commandments (non-negotiable)

1. **No anti-aliasing, no blur, no soft alpha, no scaling of art.** One pixel size for the whole film. Never resample a sprite/image — draw it at the size you need.
2. **Palette colors only.** Transparency/fades = ordered dithering or palette steps (`engine.light`, `dither_mask`).
3. **Clusters, not noise.** Every pixel is placed with intent. No random single-pixel speckle over large areas; no "pillow shading"; no banding by accident. Dithering sparingly: in transitions, skies, glows (use `band` ≈ 0.3–0.6).
4. **Readable silhouettes & values.** Squint test: the story beat must read in a 1-bit silhouette. Keep the robot readable against every background (value contrast, rim light).
5. **Hue-shifted ramps.** Shadows cooler/purpler, highlights warmer. Light direction consistent within a shot.
6. **Depth:** 3–6 parallax layers where the camera moves; atmospheric perspective (far = lighter, lower contrast, bluer); foreground elements for scale.
7. **Life in every frame.** Something always moves: dust motes, sand streams, flicker, cloth/grass sway, breathing idle, blinking lens. No dead frames — even "holds" breathe.
8. **Animation principles:** anticipation, overshoot & settle, ease-in/out, follow-through, secondary motion. Characters animate at 10–12 fps (hold frames); camera, particles and FX at 30 fps.
9. **Camera:** slow moves use `Stage` for sub-pixel glide; ease in/out; shakes decay; no jitter.
10. **Juice:** impact frames (1–2 frame flashes), sparks, debris, dust puffs on footsteps/kneels, light that reacts.

---

## 6. Technical contract

**Engine** (`engine/`, owned by the director — do not edit; if you need a helper, put it in your own module):
`W, H = 320, 180`, `FPS = 30`. Canvases are `uint8` numpy arrays: `(H, W, 3)` RGB or `(h, w, 4)` RGBA with binary alpha.
* `engine.canvas(color)`, `engine.layer(h, w)`, `C(name)`, `RAMPS`, `paste`, `fill`, `fill_dither`, `dither_mask`, `bayer`, `snap`
* `engine.draw`: `pset, points, rect, hline, vline, line, thick_line, disc, ring, ellipse, polygon, polygon_mask, fill_below, vgradient, radial_bands, outline_mask`
* `engine.light`: `apply_light(img, level_map, mode, band)`, `light_pool(img, cx, cy, radius, peak, mode, sy, core)`, `radial_level`, `darken`, `vignette`, `grade` — modes: `warm, sun, cool, moon, green, flash, neutral`; negative levels darken toward cool shadow.
* `engine.noise`: `rand, vnoise1, fbm1, vnoise2, fbm2, grid` (deterministic).
* `engine.fx`: `field, draw_field, streaks, bursts, rays_level, shake, wind_gust`.
* `engine.ease`: `prog, keyframes, smoothstep, in_out_sine, out_back, out_elastic, …, frame_index, step_time, flicker`.
* `engine.sprite`: `from_ascii, blit(dst, spr, x, y, anchor, flipx), anchor_pos, flip, outline, silhouette, recolor, pick`.
* `engine.Stage`: layered frame with sub-pixel positions (see `engine/stage.py`).

**Shot module contract** (`shots/<name>.py`):
```python
from engine import *
DURATION = 6.0                 # must match the shot list
def render(t):                 # t in seconds, 0 <= t < DURATION (+ up to 1 s beyond: handle gracefully)
    ...
    return img                 # (180, 320, 3) uint8   OR   a Stage
```
* `render(t)` must be a **pure function of t** (frames render out of order and in parallel). Cache static art at import time — fine; never keep per-frame state.
* Performance budget: **≤ 150 ms / frame** at low-res (check with `--bench`).
* Use `engine.noise` / fixed seeds — never `np.random` without a fixed seed per call.

**Characters** — `chars/robot.py` `frame(anim, t, can, variant, scale, core, eye, face_left)` and `chars/props.py` (`sprout(...)`, `can(...)`). See their docstrings for the animation list. They are being built in parallel; until they land, the stubs return the model pose so you can block out positions and timing. Anchor with `blit(img, spr, x, ground_y, anchor='feet')`; light sources from `anchor_pos(..., 'core')` / `'eye'`.
* The medium robot is **26 px** tall; wide scale **≈ 11 px**. The close-ups (CLOSEUPS) are separate large illustrations of the same design.
* The shot draws glows/lights (core pool, eye glint) itself using the anchors.

**Review tools**
```bash
python -m engine.preview shots.s05_storm --sheet                 # 4x4 contact sheet  -> review/…_sheet.png
python -m engine.preview shots.s05_storm --at 1.0 2.5 --scale 3  # stills at 960x540
python -m engine.preview shots.s05_storm --strip 2.4 --n 8 --scale 2   # consecutive frames (animation check)
python -m engine.preview shots.s05_storm --bench
python -m engine.preview shots.s05_storm --mp4 --scale 2         # motion preview
python render.py --sheet --every 1.0                             # whole film so far
python tools/spriteview.py chars.model.IDLE --scale 12 --out review/x.png
```
**Look at your frames with the Read tool** after every significant change. Review
at 2–3× for detail and 1× for readability. Iterate until it's stunning.

**File ownership.** Only edit files you own. Put helpers in your own
`shots/lib_<yourname>.py`. Don't edit `engine/`, `film.py`, `render.py`,
`cues.py`, other agents' files, or this bible. Do not spawn sub-agents.

---

## 7. Sound (for the audio team)

48 kHz stereo WAV, exactly 60.000 s. Cues in `cues.py`. Music: `audio/music.py → audio/music.wav`.
Sound design: `audio/sfx.py → audio/sfx.wav`. The director mixes them in `audio/mix.py`.
Style: a 16-bit-era-inspired but emotionally modern score (think the warmth of
SNES-era RPG scores and modern pixel-art game soundtracks), fully synthesized
in numpy/scipy — instruments designed by hand (music box, soft piano, strings,
flute/ocarina, choir, timpani), with lush reverb. Sound design is real-feeling
foley and ambience synthesized from noise and oscillators.
