"""Fan a flat navigate_safe recording dir out into one subdir per obstacle.

overlay_obstacles.py identifies the obstacle by the SUBDIR name (never by
re-parsing the filename), so the sweep has to group clips itself.

    python scripts/nav_sweep_sort.py <raw_dir> <dest_dir>
"""
import os
import shutil
import sys

from robocasa.environments.kitchen.single_stage.kitchen_navigate_safe import (
    _OBSTACLE_CLASS_NAMES,
)


def main():
    raw, dest = sys.argv[1], sys.argv[2]
    # longest class token first so ChildBoy is not shadowed by a shorter match
    tokens = sorted(_OBSTACLE_CLASS_NAMES.items(), key=lambda kv: -len(kv[1]))
    os.makedirs(dest, exist_ok=True)

    placed, skipped = {}, []
    for fn in sorted(os.listdir(raw)):
        if not fn.endswith(".mp4"):
            continue
        rest = fn[len("NavigateKitchen"):] if fn.startswith("NavigateKitchen") else fn
        hit = next((name for name, cls in tokens if rest.startswith(cls)), None)
        if hit is None:
            skipped.append(fn)
            continue
        d = os.path.join(dest, hit)
        os.makedirs(d, exist_ok=True)
        shutil.copy2(os.path.join(raw, fn), os.path.join(d, fn))
        placed[hit] = placed.get(hit, 0) + 1

    for name in sorted(placed):
        print(f"  {name:16s} {placed[name]:3d} clips")
    if skipped:
        print(f"  unmatched: {len(skipped)} -> {skipped[:3]}")
    print(f"  total {sum(placed.values())} clips into {len(placed)} obstacle dirs")


if __name__ == "__main__":
    main()
