"""Quick zoomed view of CHAR-A art pieces (dev tool).
usage: CHAR_A_DEV=<dir> python tools/char_a_view.py props   -> review/char_a_props.png
       python tools/char_a_view.py parts                      -> review/char_a_parts.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.char_a_lib import dev_modules, strip, stack, save  # noqa: E402


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else 'props'
    robot, props = dev_modules()
    if what == 'props':
        rows = []
        cans = [props.can(s) for s in ('empty', 'sprout', 'sprout_wilt', 'sprout_lean')]
        rows.append(strip(cans, ['empty', 'sprout', 'wilt', 'lean'], scale=12, grid=True))
        poses = [props.can_pose(p, 'sprout') for p in props.CAN_POSES]
        rows.append(strip(poses, list(props.CAN_POSES), scale=12, grid=True))
        sp = [props.sprout(s, 0.0) for s in ('ground', 'wilt', 'planted')]
        sp += [props.sprout('lean', 0.0, lean=v) for v in (-1, -0.5, 0.5, 1)]
        rows.append(strip(sp, ['ground', 'wilt', 'planted', 'l-1', 'l-.5', 'l.5', 'l1'], scale=12, grid=True))
        wind = [props.sprout('ground', t / 10.0, wind=1.0) for t in range(8)]
        rows.append(strip(wind, [f'w{t}' for t in range(8)], scale=12, grid=True))
        rows.append(strip(cans + [props.sprout('ground')], scale=1, bg='dust5'))
        rows.append(strip(cans + [props.sprout('ground')], scale=3, bg='dust4'))
        print(save(stack(rows), 'char_a_props.png'))
    elif what == 'parts':
        sp = robot._debug_parts()
        print(save(strip(sp, [str(i) for i in range(len(sp))], scale=10, grid=True, cols=8), 'char_a_parts.png'))


if __name__ == '__main__':
    main()
