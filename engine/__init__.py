"""
THE LAST SEED — pixel-art film engine.

    from engine import *          # W, H, FPS, canvas, layer, C, RAMPS, ...
    from engine import draw, light, noise, ease, sprite, fx, text
"""
from .core import (W, H, FPS, SCALE, canvas, layer, bayer, dither_mask, dither_levels,
                   snap, to_index, from_index, paste, over, fill, fill_dither, upscale)
from .palette import C, RAMPS, ramp
from .stage import Stage, to_lowres, to_hires
from . import palette, core, draw, light, noise, ease, sprite, fx, text, stage
