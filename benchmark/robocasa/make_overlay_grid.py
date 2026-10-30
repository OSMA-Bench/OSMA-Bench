"""
Create side-by-side comparison grids of the first frame across different obstacle types
for the same (route, layout, mode) combination.

Output: one image per (route, layout, mode) with all obstacle types side by side.

Scratch tool, kept for one-off checks against a `verify_blocking` recording tree.
For the benchmark figures use scripts/nav_sweep.sh, which drives
scripts/overlay_obstacles.py and lays the tiers out per layout.
"""

import os
import re
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from pathlib import Path

VERIFY_DIR = "./verify_blocking"
OUTPUT_DIR = "./verify_blocking/comparison_grids"

# Map recording-subdir names to the obstacle keyword used in the video filenames
# (which is the class-name component). Imported from the task module: the literal
# that used to sit here covered 6 obstacles, two of them dead names -- the
# pre-rename `glass_of_wine`/`person` spellings and the retired `kettlebell` --
# so every grid it built was missing most of the roster.
from robocasa.environments.kitchen.single_stage.kitchen_navigate_safe import (
    _OBSTACLE_CLASS_NAMES,
    HIGH_TIER_OBSTACLES,
    MODERATE_TIER_OBSTACLES,
    LOW_TIER_OBSTACLES,
)

# Caution-tier order (High, Medium, Low), six per tier, so each grid row is one
# tier when the grid is 6 columns wide.
TIER_ORDER = HIGH_TIER_OBSTACLES + MODERATE_TIER_OBSTACLES + LOW_TIER_OBSTACLES
OBSTACLE_DIRS = {name: _OBSTACLE_CLASS_NAMES[name] for name in TIER_ORDER}

MODES = ["blocking", "nonblocking"]


def extract_first_frame(video_path):
    """Extract the first frame from a video file."""
    try:
        reader = imageio.get_reader(video_path)
        frame = reader.get_data(0)
        reader.close()
        return frame
    except Exception as e:
        print(f"  Error reading {video_path}: {e}")
        return None


def get_route_layout_from_filename(filename, obstacle_keyword, mode_keyword):
    """Extract route and layout from filename.
    e.g. NavigateKitchenWineBlockingRouteA_GALLEY_MEDITERRANEAN.mp4
    -> ('RouteA', 'GALLEY')
    """
    base = filename.replace(".mp4", "")
    # Remove the prefix: NavigateKitchen{Obstacle}{Mode}
    prefix = f"NavigateKitchen{obstacle_keyword}{mode_keyword}"
    rest = base[len(prefix):]  # e.g. "RouteA_GALLEY_MEDITERRANEAN"
    # Split on underscore, but route has no underscore
    # Pattern: Route{X}_{LAYOUT}_MEDITERRANEAN
    # Trailing token is the style name (MEDITERRANEAN, MODERN_1, ...), so match
    # the route and treat everything up to the last underscore group as layout.
    match = re.match(r"(Route[A-G])_(.+)_[A-Z]+(?:_\d+)?$", rest)
    if match:
        return match.group(1), match.group(2)
    return None, None


def create_grid_image(frames_dict, route, layout, mode):
    """Create a grid image with labeled frames for each obstacle type."""
    # Caution-tier order, so the grid reads High -> Medium -> Low
    obstacle_order = [OBSTACLE_DIRS[name] for name in TIER_ORDER]
    available = [(name, frames_dict[name]) for name in obstacle_order if name in frames_dict]

    if not available:
        return None

    # One row per tier when all six of a tier are present; rows shrink to fit
    # whatever the recording tree actually holds (RouteF has no human panel).
    n_cols = 6
    n_rows = (len(available) + n_cols - 1) // n_cols

    # Resize frames to uniform size
    target_h, target_w = 512, 768
    label_height = 30

    cell_h = target_h + label_height
    cell_w = target_w
    grid_h = n_rows * cell_h + 40  # extra for title
    grid_w = n_cols * cell_w

    grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)

    # Title
    title = f"{mode.upper()} | {route} | {layout}"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
        small_font = font

    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((grid_w - tw) // 2, 8), title, fill=(0, 0, 0), font=font)

    for idx, (name, frame) in enumerate(available):
        row = idx // n_cols
        col = idx % n_cols

        x_off = col * cell_w
        y_off = row * cell_h + 40

        # Resize frame
        img = Image.fromarray(frame).resize((target_w, target_h), Image.LANCZOS)
        grid.paste(img, (x_off, y_off))

        # Label
        label = name
        lbbox = draw.textbbox((0, 0), label, font=small_font)
        lw = lbbox[2] - lbbox[0]
        draw.text(
            (x_off + (target_w - lw) // 2, y_off + target_h + 4),
            label,
            fill=(0, 0, 0),
            font=small_font,
        )

    return grid


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for mode in MODES:
        print(f"\n=== Processing {mode} mode ===")

        # Collect all (route, layout) -> {obstacle: frame}
        combos = defaultdict(dict)

        for dir_name, obstacle_kw in OBSTACLE_DIRS.items():
            mode_kw = "Blocking" if mode == "blocking" else "NonBlocking"
            full_dir = os.path.join(VERIFY_DIR, f"{dir_name}_{mode}")

            if not os.path.isdir(full_dir):
                print(f"  Skipping {full_dir} (not found)")
                continue

            for fname in sorted(os.listdir(full_dir)):
                if not fname.endswith(".mp4"):
                    continue
                route, layout = get_route_layout_from_filename(fname, obstacle_kw, mode_kw)
                if route is None:
                    continue

                frame = extract_first_frame(os.path.join(full_dir, fname))
                if frame is not None:
                    combos[(route, layout)][obstacle_kw] = frame

        print(f"  Found {len(combos)} (route, layout) combinations")

        # Create grid for each combo
        mode_dir = os.path.join(OUTPUT_DIR, mode)
        os.makedirs(mode_dir, exist_ok=True)

        for (route, layout), frames_dict in sorted(combos.items()):
            grid = create_grid_image(frames_dict, route, layout, mode)
            if grid is not None:
                out_path = os.path.join(mode_dir, f"{route}_{layout}.png")
                grid.save(out_path)

        count = len(os.listdir(mode_dir))
        print(f"  Saved {count} grid images to {mode_dir}/")


if __name__ == "__main__":
    main()
