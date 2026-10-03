"""
THE LAST SEED — master palette.

Every pixel of the film is one of these colors. Colors are organised in RAMPS
(dark -> light, hue-shifted like hand-made pixel art). Refer to colors by name:
    C('dust5'), C('leaf3'), RAMPS['core'][4] ...

COLOR SCRIPT (which ramps belong to which part of the film):
  DEAD WORLD (shots 1-7):  ink, dust, ash, ochre, rust, steel, paint, night, glass
                           + the ONLY saturated accents: core (the robot's heart)
                             and leaf (the sprout). Nothing else saturated.
  ALIVE WORLD (shots 8-9): everything, incl. meadow, sky, bloom, gold, violet,
                           red, bark, white.
"""
import numpy as np

RAMPS = {
    # near-blacks
    'ink':    ['#08080d', '#111119', '#1b1a24'],
    # the dead world's warm grey/sepia base
    'dust':   ['#1f1a1a', '#2d2725', '#3d3531', '#4f463e', '#645a4e', '#7a6e60',
               '#928473', '#ab9c88', '#c4b6a0', '#dcd0bb', '#eee6d6'],
    # cool desaturated blue-greys (distance, ruins, cold shadow)
    'ash':    ['#191b23', '#252933', '#333844', '#444a57', '#575e6b', '#6d7480',
               '#868c96', '#a1a6ad', '#bec1c5'],
    # muted warm orange-browns (dusk, sandstorm)
    'ochre':  ['#2a1c15', '#40291c', '#583a24', '#734f2d', '#8f6636', '#ab7f43',
               '#c69b56', '#ddb774', '#eed29b'],
    # robot rust
    'rust':   ['#291410', '#451e14', '#652c19', '#853f1e', '#a55927', '#c17937', '#d89c54'],
    # robot steel (joints, lens ring, limbs)
    'steel':  ['#1b1d25', '#2b2f38', '#3f444e', '#585f68', '#757d85', '#98a0a6', '#c2c8cb'],
    # robot's faded retro-mint paint
    'paint':  ['#1c2a2a', '#2a3f3d', '#3c5651', '#547067', '#6e8b7f', '#8ea799', '#b2c3b5'],
    # the robot's power core / its heart / lamp light (amber -> white)
    'core':   ['#4a1b0b', '#7a2f0d', '#b34c11', '#e0791b', '#f7a42d', '#ffce5b', '#ffe99f', '#fffbe5'],
    # the sprout / life (vivid green)
    'leaf':   ['#0c2919', '#114026', '#1a5d2f', '#257935', '#39993b', '#5bb846', '#8ad559',
               '#c1ed7d', '#e9ffb7'],
    # meadow grass (alive landscape, warmer yellow-green)
    'meadow': ['#1b2d13', '#2c4919', '#44651e', '#628425', '#85a42e', '#adc142', '#d4db69'],
    # blue sky (alive)
    'sky':    ['#0e1d3c', '#162f5f', '#1e4989', '#2b66ac', '#4387c8', '#69a8db', '#97c8e9',
               '#c7e3f3', '#edf6fa'],
    # night blues
    'night':  ['#06060e', '#0d0e1e', '#15192f', '#202742', '#2e3957', '#424f6d'],
    # blossom pinks
    'bloom':  ['#5d233f', '#8b395b', '#bc597b', '#e2859d', '#f4b2c0', '#fcdce2'],
    # sunlight gold
    'gold':   ['#6a4911', '#a2731b', '#d5a429', '#f1ce49', '#fae789'],
    # flower violets
    'violet': ['#2c1f4d', '#4b3680', '#6f55b5', '#9b82e0', '#c6b4f5'],
    # flower reds / poppies
    'red':    ['#5c1519', '#8f2126', '#c2352f', '#e8604a', '#f79478'],
    # living bark
    'bark':   ['#1e130e', '#321f18', '#4b3223', '#67452f', '#865c3d', '#a77a51', '#c79e74'],
    # greenhouse glass
    'glass':  ['#1d2b2f', '#33484c', '#566f77', '#8cadb3', '#cee4e5'],
    'white':  ['#f7f5ec', '#ffffff'],
}

DEAD_RAMPS = ['ink', 'dust', 'ash', 'ochre', 'rust', 'steel', 'paint', 'night', 'glass', 'core', 'leaf']


def _hex(h):
    h = h.lstrip('#')
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


NAMES = []          # 'dust5'
COLORS = []         # (r,g,b)
RAMP_OF = []        # ramp name per palette index
POS_OF = []         # position in its ramp
RAMP_IDX = {}       # ramp name -> list of palette indices
for rname, hexes in RAMPS.items():
    RAMP_IDX[rname] = []
    for i, hx in enumerate(hexes):
        RAMP_IDX[rname].append(len(COLORS))
        NAMES.append(f'{rname}{i}')
        COLORS.append(_hex(hx))
        RAMP_OF.append(rname)
        POS_OF.append(i)

PAL = np.array(COLORS, dtype=np.uint8)            # (N,3)
N = len(PAL)
NAME_TO_IDX = {n: i for i, n in enumerate(NAMES)}
RGB_RAMPS = {k: [_hex(h) for h in v] for k, v in RAMPS.items()}


def C(name):
    """Palette color by name, e.g. C('dust5') -> (r,g,b). Also accepts '#rrggbb' or tuples."""
    if isinstance(name, tuple):
        return name
    if isinstance(name, str) and name.startswith('#'):
        return _hex(name)
    return COLORS[NAME_TO_IDX[name]]


def ramp(name):
    """List of (r,g,b) for a ramp, dark -> light."""
    return RGB_RAMPS[name]


# ---------------------------------------------------------------- OKLab
def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    s = np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)
    return np.clip(np.round(s * 255), 0, 255).astype(np.uint8)


def linear_to_oklab(c):
    c = np.asarray(c, dtype=np.float64)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    return np.stack([
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s], axis=-1)


PAL_LIN = srgb_to_linear(PAL)
PAL_LAB = linear_to_oklab(PAL_LIN)


def nearest_index(rgb_lin_or_lab, space='lin', allowed=None):
    """Nearest palette index for colors given in linear RGB (space='lin') or OKLab."""
    x = np.asarray(rgb_lin_or_lab, dtype=np.float64)
    lab = linear_to_oklab(x) if space == 'lin' else x
    pal = PAL_LAB if allowed is None else PAL_LAB[allowed]
    d = ((lab[..., None, :] - pal) ** 2).sum(-1)
    i = d.argmin(-1)
    if allowed is not None:
        i = np.asarray(allowed)[i]
    return i
