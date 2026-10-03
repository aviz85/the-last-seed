"""Create placeholder shot modules (only if missing)."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOTS = {
    's01_wasteland': (7.0, 'dust4'), 's02_empty_can': (5.0, 'dust5'), 's03_discovery': (8.0, 'dust5'),
    'c01_eye': (1.5, 'steel2'), 'c02_sprout': (1.5, 'dust3'), 's04_journey': (6.0, 'ochre4'),
    's05_storm': (6.0, 'ochre3'), 's06_night': (6.0, 'night2'), 's07_gift': (7.0, 'ash4'),
    'c03_core': (2.5, 'rust1'), 's08_bloom_int': (3.0, 'leaf2'), 's08_bloom_ext': (5.0, 'sky4'),
    's09_valley': (2.0, 'meadow4'), 'c04_rest': (5.0, 'leaf3'), 'c08_heart': (0.8, 'core2'),
}
TPL = '''"""PLACEHOLDER — replaced by the shot's owner agent."""
from engine import *
from engine import text as T, draw
DURATION = {dur}


def render(t):
    img = canvas('{col}')
    T.text(img, '{name}', 160, 80, 'white0', align='center', shadow='ink0')
    T.text(img, f'{{t:.2f}}', 160, 92, 'white0', align='center', shadow='ink0')
    draw.rect(img, 0, 176, int(320 * min(t / DURATION, 1)), 4, 'core5')
    return img
'''
TITLE = '''"""PLACEHOLDER title overlay — replaced by the TITLE agent."""
from engine import *
from engine import text as T
DURATION = 3.0


def render(t):
    lay = layer()
    if t > 0.5:
        T.text(lay, 'THE LAST SEED', 160, 86, 'core5', align='center')
    return {{'layer': lay, 'under_fade': min(1.0, t / 1.0) * 0.7}}
'''
for name, (dur, col) in SHOTS.items():
    p = os.path.join(ROOT, 'shots', name + '.py')
    if not os.path.exists(p):
        open(p, 'w').write(TPL.format(dur=dur, col=col, name=name.upper()))
p = os.path.join(ROOT, 'shots', 'title.py')
if not os.path.exists(p):
    open(p, 'w').write(TITLE.format())
print('stubs ok')
