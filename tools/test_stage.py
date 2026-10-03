import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from engine import *
from engine import draw
from chars import robot
from engine.sprite import blit
DURATION = 2.0
SKY = canvas('ash4'); draw.vgradient(SKY, 0, 180, ['ash3','ash4','dust5','dust6'])
HILLS = layer(180, 700)
hs = 100 + 20*noise.fbm1(np.arange(700)/60.0, 3)
m = draw.fill_below(HILLS, hs, 'dust3')
def render(t):
    st = Stage()
    cam = 40*t
    st.add(SKY, 0, 0)
    st.add(HILLS, -cam*0.5, 0)
    g = layer(40, 400); draw.rect(g, 0, 0, 400, 40, 'dust2')
    blit(g, robot.frame('walk', t, scale='medium'), 60 + 0*t, 20, anchor='feet')
    st.add(g, -cam + 40*t, 140)
    return st
