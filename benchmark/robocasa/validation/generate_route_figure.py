"""
generate_route_figure.py — Blueprint-style schematic route diagram
================================================================================

Generates a clean, academic-quality schematic figure showing navigation routes
for the NavigateKitchenSafe task. Kitchen counters/walls are drawn as simple
gray rectangles (blueprint style). Routes are shown as dashed colored arrows
with source dots and obstacle markers. No rendering/GPU required.

Currently hardcoded to U_SHAPED_LARGE layout (fixture positions baked in).
To use a different layout, update the FIXTURES and COUNTERS dicts at the top
of this file with positions obtained from the simulation.

Output
------
  figures/route_geometries_ushaped.pdf   (vector, paper-ready)
  figures/route_geometries_ushaped.png

  4×2 grid (7 routes + 1 blank). Each panel shows:
    - Gray bars = kitchen counters (U-shape)
    - Blue dashed arrow = navigation route
    - Blue dot = start position
    - Red × = obstacle position (midpoint of route)
    - Colored fixture markers with labels (active highlighted, inactive grayed)

Dependencies
------------
  matplotlib, numpy, PIL (only matplotlib needed — no MuJoCo/rendering)

Usage
-----
  # Generate the figure (no docker/GPU needed)
  python generate_route_figure.py

  # Output: figures/route_geometries_ushaped.pdf
  #         figures/route_geometries_ushaped.png

Customization
-------------
  To generate for a different layout:
    1. Run the simulation for that layout and print fixture positions
    2. Update FIXTURES dict with the new positions
    3. Update COUNTERS list with counter bar geometry
    4. Run the script
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
import os

OUT_DIR = "figures"

# U_SHAPED_LARGE fixture positions (from simulation)
FIXTURES = {
    "Fridge":        np.array([3.95, -2.92]),
    "Sink":          np.array([1.69, -0.30]),
    "Stove":         np.array([4.05, -1.57]),
    "CoffeeMachine": np.array([0.21, -3.20]),
    "Microwave":     np.array([4.16, -1.57]),
    "Door":          np.array([2.00, -5.00]),
    "Human":         np.array([3.20, -4.80]),
}

# U-shaped counter layout: back wall, left arm, right arm
COUNTERS = [
    # (x_center, y_center, width, height) — simplified rectangles
    # Back wall counters
    {"xy": (0.0, -0.65), "w": 4.45, "h": 0.65, "label": "Back Counter"},
    # Left arm
    {"xy": (0.0, -0.65), "w": 0.65, "h": 2.85, "label": "Left Counter"},
    # Right arm
    {"xy": (3.80, -0.65), "w": 0.65, "h": 2.00, "label": "Right Counter"},
]

# Walls
WALLS = [
    {"xy": (-0.10, 0.05), "w": 4.60, "h": 0.15},   # back wall
    {"xy": (-0.10, -6.10), "w": 0.15, "h": 6.20},  # left wall
    {"xy": (4.40, -6.10), "w": 0.15, "h": 6.20},   # right wall
]

ROUTES = {
    "A": {"src": "Fridge", "dst": "CoffeeMachine", "label": "Fridge → Coffee Machine"},
    "B": {"src": "Fridge", "dst": "Sink", "label": "Fridge → Sink"},
    "C": {"src": "Fridge", "dst": "Stove", "label": "Fridge → Stove"},
    "D": {"src": "Sink", "dst": "CoffeeMachine", "label": "Sink → Coffee Machine"},
    "E": {"src": "Stove", "dst": "Door", "label": "Stove → Door"},
    "F": {"src": "Sink", "dst": "Human", "label": "Sink → Human"},
    "G": {"src": "Microwave", "dst": "Sink", "label": "Microwave → Sink"},
}

ROUTE_COLOR = "#4285F4"  # Google blue (clean, academic)
COUNTER_COLOR = "#B0BEC5"
COUNTER_EDGE = "#78909C"
WALL_COLOR = "#546E7A"
BG_COLOR = "#F5F5F5"
PANEL_BG = "#FAFAFA"
FIXTURE_DOT = "#1A73E8"
FIXTURE_LABEL_COLOR = "#37474F"

# Fixture display info
FIXTURE_SHORT = {
    "Fridge": "Fridge",
    "Sink": "Sink",
    "Stove": "Stove",
    "CoffeeMachine": "Coffee",
    "Microwave": "Micro.",
    "Door": "Door",
    "Human": "Human",
}

FIXTURE_MARKER = {
    "Fridge": "s",
    "Sink": "o",
    "Stove": "^",
    "CoffeeMachine": "D",
    "Microwave": "p",
    "Door": "H",
    "Human": "*",
}


def draw_kitchen(ax):
    """Draw simplified U-shaped kitchen counters and walls."""
    # Walls (thin dark lines)
    for w in WALLS:
        rect = Rectangle(
            (w["xy"][0], w["xy"][1]), w["w"], w["h"],
            linewidth=0, facecolor=WALL_COLOR, zorder=1, alpha=0.3,
        )
        ax.add_patch(rect)

    # Counters (gray bars)
    for c in COUNTERS:
        rect = Rectangle(
            (c["xy"][0], c["xy"][1] - c["h"]), c["w"], c["h"],
            linewidth=1.0, edgecolor=COUNTER_EDGE, facecolor=COUNTER_COLOR,
            zorder=2, alpha=0.6, joinstyle="round",
        )
        ax.add_patch(rect)

    # Floor area label
    ax.text(2.2, -3.5, "Floor", fontsize=7, ha="center", va="center",
            color="#90A4AE", fontstyle="italic", zorder=1)


def draw_fixture_icons(ax, highlight_src=None, highlight_dst=None):
    """Draw small fixture markers with labels."""
    for name, pos in FIXTURES.items():
        is_src = (name == highlight_src)
        is_dst = (name == highlight_dst)

        marker = FIXTURE_MARKER[name]
        if is_src or is_dst:
            color = FIXTURE_DOT
            size = 60
            alpha = 1.0
        else:
            color = "#B0BEC5"
            size = 30
            alpha = 0.5

        ax.scatter(pos[0], pos[1], marker=marker, c=color, s=size,
                   edgecolors="white", linewidths=0.8, zorder=5, alpha=alpha)

        # Label
        label_color = FIXTURE_LABEL_COLOR if (is_src or is_dst) else "#B0BEC5"
        fontweight = "bold" if (is_src or is_dst) else "normal"
        fontsize = 6.5 if (is_src or is_dst) else 5.5
        ax.annotate(
            FIXTURE_SHORT[name], (pos[0], pos[1]),
            textcoords="offset points", xytext=(0, -10),
            fontsize=fontsize, ha="center", va="top",
            color=label_color, fontweight=fontweight, zorder=6,
        )


def draw_route_arrow(ax, src_pos, dst_pos):
    """Draw a dashed arrow from src to dst with start dot."""
    # Start dot
    ax.scatter(src_pos[0], src_pos[1], c=ROUTE_COLOR, s=80,
               zorder=10, edgecolors="white", linewidths=1.5)

    # Compute path with slight curve
    direction = dst_pos - src_pos
    dist = np.linalg.norm(direction)
    dn = direction / (dist + 1e-8)

    # Shorten to not overlap markers
    start = src_pos + dn * 0.25
    end = dst_pos - dn * 0.2

    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="->,head_width=8,head_length=6",
        color=ROUTE_COLOR, linewidth=2.5, zorder=8,
        connectionstyle="arc3,rad=0.15",
        linestyle="--",
        path_effects=[pe.withStroke(linewidth=4, foreground="white")],
    )
    ax.add_patch(arrow)


def draw_obstacle_marker(ax, src_pos, dst_pos):
    """Draw a small obstacle marker at the midpoint of the route."""
    mid = (src_pos + dst_pos) / 2
    # Offset slightly perpendicular
    perp = np.array([-(dst_pos[1] - src_pos[1]), dst_pos[0] - src_pos[0]])
    perp_n = perp / (np.linalg.norm(perp) + 1e-8)
    obs_pos = mid + perp_n * 0.15

    ax.scatter(obs_pos[0], obs_pos[1], marker="x", c="#F44336", s=40,
               linewidths=1.5, zorder=9, alpha=0.7)
    ax.annotate("obstacle", obs_pos, textcoords="offset points",
                xytext=(8, 4), fontsize=4.5, color="#F44336", alpha=0.7,
                fontstyle="italic", zorder=9)


def generate():
    os.makedirs(OUT_DIR, exist_ok=True)

    n_routes = len(ROUTES)
    cols = 4
    rows = 2

    fig, axes = plt.subplots(rows, cols, figsize=(14, 7.5), dpi=200)
    fig.patch.set_facecolor("white")

    xlim = (-0.8, 5.2)
    ylim = (-5.8, 0.8)

    for idx, (route_id, route_def) in enumerate(ROUTES.items()):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        # Panel background
        panel = FancyBboxPatch(
            (xlim[0] + 0.05, ylim[0] + 0.05),
            xlim[1] - xlim[0] - 0.1, ylim[1] - ylim[0] - 0.1,
            boxstyle="round,pad=0.15", facecolor=PANEL_BG,
            edgecolor="#E0E0E0", linewidth=1.0, zorder=0,
        )
        ax.add_patch(panel)

        draw_kitchen(ax)

        src_name = route_def["src"]
        dst_name = route_def["dst"]
        src_pos = FIXTURES[src_name]
        dst_pos = FIXTURES[dst_name]

        draw_fixture_icons(ax, highlight_src=src_name, highlight_dst=dst_name)
        draw_route_arrow(ax, src_pos, dst_pos)
        draw_obstacle_marker(ax, src_pos, dst_pos)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Route label below panel
        ax.set_title(
            f"Route {route_id}: {route_def['label']}",
            fontsize=8, fontweight="bold", color="#37474F", pad=6,
        )

    # Hide last panel if odd number of routes
    if n_routes < rows * cols:
        for idx in range(n_routes, rows * cols):
            r, c = divmod(idx, cols)
            axes[r][c].axis("off")

    fig.suptitle(
        "Navigation Route Variants — U-Shaped Large Kitchen",
        fontsize=13, fontweight="bold", color="#212121", y=0.98,
    )

    # Shared legend at bottom
    legend_elements = [
        Line2D([0], [0], color=ROUTE_COLOR, linewidth=2.5, linestyle="--",
               label="Navigation route"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ROUTE_COLOR,
               markersize=8, label="Start position", linewidth=0),
        Line2D([0], [0], marker="x", color="#F44336", markersize=7,
               label="Obstacle position", linewidth=0, markeredgewidth=1.5),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#B0BEC5",
               markersize=6, label="Other fixtures (inactive)", linewidth=0),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=8, frameon=True, framealpha=0.95,
               edgecolor="#E0E0E0", bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    for ext in ["pdf", "png"]:
        path = os.path.join(OUT_DIR, f"route_geometries_ushaped.{ext}")
        plt.savefig(path, bbox_inches="tight", dpi=200, facecolor="white")
        print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    generate()
