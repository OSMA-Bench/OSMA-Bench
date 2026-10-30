"""
Visualize the path postprocessing pipeline (Step B: curvature cutoff, Step C: spacing skip).
Simulates what happens with a 98-point greedy path on a 100x100 map.
"""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.signal import savgol_filter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import repo_root  # noqa: E402


def calc_curvature(path):
    """Calculate curvature at each point (same as utils.py)."""
    dx = np.gradient(path[:, 0])
    dy = np.gradient(path[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-8)**1.5
    return curvature


def generate_example_path(n_points=98, map_size=100):
    """Generate a realistic greedy path that triggers curvature cutoff.

    The greedy planner goes pixel-by-pixel, creating a path that:
    1) Moves toward the goal smoothly at first
    2) Hits the obstacle avoidance zone and makes a sharp detour
    3) After the detour, abruptly changes direction (sharp curvature)
    """
    # Build path segment by segment
    segments = []

    # RouteD scenario: Sink[80,80] → CoffeeMachine[75,75] — very short route
    # Person blocking at midpoint [77,78]
    # Greedy planner wanders around trying to find a path

    # Segment 1: Start from sink, move toward goal — 30 pts
    n1 = 30
    t1 = np.linspace(0, 1, n1)
    seg1_x = 80 - 5 * t1    # small movement
    seg1_y = 80 - 3 * t1
    segments.append(np.column_stack([seg1_x, seg1_y]))

    # Segment 2: Hit person's avoidance zone, sharp detour left — 20 pts
    n2 = 20
    t2 = np.linspace(0, 1, n2)
    seg2_x = 75 + 2 * t2    # backtrack slightly
    seg2_y = 77 - 15 * t2   # sharp left detour
    segments.append(np.column_stack([seg2_x, seg2_y]))

    # Segment 3: Sharp U-turn back toward goal — 8 pts (curvature spike here)
    n3 = 8
    t3 = np.linspace(0, 1, n3)
    seg3_x = 77 - 5 * t3
    seg3_y = 62 + 18 * t3   # sharp reversal back right
    segments.append(np.column_stack([seg3_x, seg3_y]))

    # Segment 4: Continue wandering — 40 pts
    n4 = 40
    t4 = np.linspace(0, 1, n4)
    seg4_x = 72 + 3 * np.sin(4 * np.pi * t4)
    seg4_y = 80 - 5 * t4 + 2 * np.cos(3 * np.pi * t4)
    segments.append(np.column_stack([seg4_x, seg4_y]))

    path = np.concatenate(segments, axis=0)
    path = path.clip(0, map_size - 1)
    return path


def postprocess_step_by_step(raw_path, map_size=100, max_curvature=5, target_spacing=1,
                              savgol_window=20, savgol_poly=3):
    """Run postprocessing step by step, returning intermediate results."""
    steps = {}
    steps['0_raw'] = raw_path.copy()

    # Step A: Savgol smoothing
    window = min(len(raw_path), savgol_window)
    poly = min(savgol_poly, window - 1)
    smoothed = savgol_filter(raw_path, window, poly, axis=0)
    steps['A_smoothed'] = smoothed.copy()

    # Step B: Curvature cutoff
    curvature = calc_curvature(smoothed)
    steps['curvature'] = curvature
    cutoff_path = smoothed.copy()
    cutoff_idx = None
    if len(curvature) > 5:
        high_curv_idx = np.where(curvature[5:] > max_curvature)[0]
        if len(high_curv_idx) > 0:
            high_curv_idx += 5
            cutoff_idx = int(0.9 * high_curv_idx[0])
            cutoff_path = smoothed[:cutoff_idx]
    steps['B_curvature_cutoff'] = cutoff_path.copy()
    steps['cutoff_idx'] = cutoff_idx

    # Step C: Spacing skip
    path_trimmed = cutoff_path[1:-1]
    skip_ratio = None
    if len(path_trimmed) > 1:
        ts = int(target_spacing * map_size / 100)
        length = np.linalg.norm(path_trimmed[1:] - path_trimmed[:-1], axis=1).sum()
        if length > ts:
            curr_spacing = np.linalg.norm(path_trimmed[1:] - path_trimmed[:-1], axis=1).mean()
            skip_ratio = int(np.round(ts / curr_spacing))
            if skip_ratio > 1:
                path_trimmed = path_trimmed[::skip_ratio]
    final = np.concatenate([cutoff_path[0:1], path_trimmed, cutoff_path[-1:]])
    final = final.clip(0, map_size - 1)
    steps['C_spacing_skip'] = final.copy()
    steps['skip_ratio'] = skip_ratio

    return steps


def main():
    output_dir = repo_root() / "scripts/route_diagrams"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = generate_example_path(n_points=98, map_size=100)
    # navigation config uses max_curvature=3 (not 5)
    steps = postprocess_step_by_step(raw_path, max_curvature=3)

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))

    map_size = 100
    person_pos = [77, 78]  # obstacle (person blocking)
    start = raw_path[0]
    end_target = [75, 75]  # coffee machine position

    # --- Panel 0: Raw greedy path ---
    ax = axes[0, 0]
    path = steps['0_raw']
    ax.set_facecolor('#f8f6f0')
    ax.plot(path[:, 1], path[:, 0], 'b-', linewidth=1, alpha=0.7, label=f'Raw path ({len(path)} pts)')
    ax.plot(path[0, 1], path[0, 0], 'go', markersize=12, label='Start (Sink)')
    ax.plot(end_target[1], end_target[0], 'r*', markersize=15, label='Goal (CoffeeMachine)')
    ax.plot(person_pos[1], person_pos[0], 'ro', markersize=20, alpha=0.4)
    ax.annotate('Person\n(obstacle)', (person_pos[1], person_pos[0]), fontsize=8, ha='center',
                color='red', fontweight='bold')
    ax.set_title(f'Step 0: Raw Greedy Path\n({len(path)} points)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, map_size)
    ax.set_ylim(map_size, 0)
    ax.set_xlabel('Y (pixel)')
    ax.set_ylabel('X (pixel)')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.2)

    # --- Panel 1: After Savgol smoothing ---
    ax = axes[0, 1]
    path_raw = steps['0_raw']
    path_smooth = steps['A_smoothed']
    ax.set_facecolor('#f8f6f0')
    ax.plot(path_raw[:, 1], path_raw[:, 0], 'b-', linewidth=0.8, alpha=0.3, label=f'Before ({len(path_raw)} pts)')
    ax.plot(path_smooth[:, 1], path_smooth[:, 0], 'g-', linewidth=2, label=f'After Savgol ({len(path_smooth)} pts)')
    ax.plot(path_smooth[0, 1], path_smooth[0, 0], 'go', markersize=12)
    ax.plot(end_target[1], end_target[0], 'r*', markersize=15)
    ax.plot(person_pos[1], person_pos[0], 'ro', markersize=20, alpha=0.4)
    ax.set_title(f'Step A: Savgol Smoothing\n({len(path_smooth)} pts, zigzag removed)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, map_size)
    ax.set_ylim(map_size, 0)
    ax.set_xlabel('Y (pixel)')
    ax.set_ylabel('X (pixel)')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.2)

    # --- Panel 2: Curvature plot ---
    ax = axes[0, 2]
    curvature = steps['curvature']
    cutoff_idx = steps['cutoff_idx']
    ax.plot(curvature, 'b-', linewidth=1, label='Curvature')
    mc = 3  # navigation config max_curvature
    ax.axhline(y=mc, color='r', linestyle='--', linewidth=2, label=f'max_curvature = {mc}')
    if cutoff_idx is not None:
        # Find the first high curvature point
        high_idx = np.where(curvature[5:] > mc)[0]
        if len(high_idx) > 0:
            first_high = high_idx[0] + 5
            ax.axvline(x=first_high, color='orange', linestyle=':', linewidth=2,
                       label=f'First exceed @ index {first_high}')
            ax.axvline(x=cutoff_idx, color='red', linestyle='-', linewidth=2,
                       label=f'Cutoff @ int(0.9 × {first_high}) = {cutoff_idx}')
            ax.fill_between(range(cutoff_idx, len(curvature)), 0, max(curvature),
                           alpha=0.15, color='red', label='Removed region')
    ax.set_title('Step B: Curvature Analysis\n(points after cutoff are removed)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Path index')
    ax.set_ylabel('Curvature')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, min(max(curvature) * 1.1, 30))

    # --- Panel 3: After curvature cutoff ---
    ax = axes[1, 0]
    path_smooth = steps['A_smoothed']
    path_cut = steps['B_curvature_cutoff']
    ax.set_facecolor('#f8f6f0')
    ax.plot(path_smooth[:, 1], path_smooth[:, 0], 'g-', linewidth=1, alpha=0.3, label=f'Before ({len(path_smooth)} pts)')
    if cutoff_idx is not None:
        ax.plot(path_smooth[cutoff_idx:, 1], path_smooth[cutoff_idx:, 0], 'r-', linewidth=2, alpha=0.5, label='Removed (high curvature)')
    ax.plot(path_cut[:, 1], path_cut[:, 0], 'b-', linewidth=2.5, label=f'Kept ({len(path_cut)} pts)')
    ax.plot(path_cut[0, 1], path_cut[0, 0], 'go', markersize=12)
    ax.plot(path_cut[-1, 1], path_cut[-1, 0], 'bs', markersize=10, label='New endpoint')
    ax.plot(end_target[1], end_target[0], 'r*', markersize=15, alpha=0.3, label='Original goal')
    ax.plot(person_pos[1], person_pos[0], 'ro', markersize=20, alpha=0.4)
    ax.set_title(f'Step B: Curvature Cutoff\n{len(path_smooth)} → {len(path_cut)} pts', fontsize=12, fontweight='bold')
    ax.set_xlim(0, map_size)
    ax.set_ylim(map_size, 0)
    ax.set_xlabel('Y (pixel)')
    ax.set_ylabel('X (pixel)')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.2)

    # --- Panel 4: After spacing skip ---
    ax = axes[1, 1]
    path_cut = steps['B_curvature_cutoff']
    path_final = steps['C_spacing_skip']
    skip_ratio = steps['skip_ratio']
    ax.set_facecolor('#f8f6f0')
    ax.plot(path_cut[:, 1], path_cut[:, 0], 'b-', linewidth=1, alpha=0.3, label=f'Before ({len(path_cut)} pts)')
    ax.plot(path_final[:, 1], path_final[:, 0], 'm-o', linewidth=2, markersize=8,
            label=f'After skip (ratio={skip_ratio}, {len(path_final)} pts)')
    for i, pt in enumerate(path_final):
        ax.annotate(str(i), (pt[1], pt[0]), fontsize=7, ha='center', va='bottom',
                    textcoords="offset points", xytext=(0, 8), color='purple', fontweight='bold')
    ax.plot(path_final[0, 1], path_final[0, 0], 'go', markersize=14, zorder=10)
    ax.plot(person_pos[1], person_pos[0], 'ro', markersize=20, alpha=0.4)
    ax.set_title(f'Step C: Spacing Skip\n{len(path_cut)} → {len(path_final)} pts (skip_ratio={skip_ratio})',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, map_size)
    ax.set_ylim(map_size, 0)
    ax.set_xlabel('Y (pixel)')
    ax.set_ylabel('X (pixel)')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.2)

    # --- Panel 5: Execution with is_last skip ---
    ax = axes[1, 2]
    path_final = steps['C_spacing_skip']
    n_wp = len(path_final)
    ax.set_facecolor('#f8f6f0')
    ax.plot(person_pos[1], person_pos[0], 'ro', markersize=20, alpha=0.4)
    ax.annotate('Person', (person_pos[1], person_pos[0]), fontsize=8, ha='center', color='red', fontweight='bold')

    for i, pt in enumerate(path_final):
        is_last = (i == n_wp - 1) or (i == n_wp - 2)
        if is_last:
            ax.plot(pt[1], pt[0], 'rx', markersize=15, markeredgewidth=3, zorder=10)
            ax.annotate(f'wp{i}\n(SKIP)', (pt[1], pt[0]), fontsize=7, ha='center',
                       textcoords="offset points", xytext=(12, -15), color='red', fontweight='bold')
        else:
            ax.plot(pt[1], pt[0], 'go', markersize=12, zorder=10)
            ax.annotate(f'wp{i}\n(exec)', (pt[1], pt[0]), fontsize=7, ha='center',
                       textcoords="offset points", xytext=(12, 8), color='green', fontweight='bold')

    # Draw path
    for i in range(len(path_final) - 1):
        is_skip = (i >= n_wp - 2) or (i + 1 >= n_wp - 2)
        color = 'red' if is_skip else 'green'
        style = '--' if is_skip else '-'
        ax.plot([path_final[i, 1], path_final[i+1, 1]],
                [path_final[i, 0], path_final[i+1, 0]],
                style, color=color, linewidth=2, alpha=0.6)

    exec_count = max(0, n_wp - 2)
    ax.set_title(f'Execution: is_last skip\n{n_wp} waypoints → {exec_count} executed, last 2 skipped',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, map_size)
    ax.set_ylim(map_size, 0)
    ax.set_xlabel('Y (pixel)')
    ax.set_ylabel('X (pixel)')

    exec_patch = mpatches.Patch(color='green', label=f'Executed ({exec_count} waypoints)')
    skip_patch = mpatches.Patch(color='red', label='Skipped (is_last=True)')
    ax.legend(handles=[exec_patch, skip_patch], fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.2)

    # Add annotation about steps=0
    if exec_count <= 3:
        ax.text(50, 95, f'If robot starts near wp0 → steps=0\n(already at first waypoint)',
                fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))

    fig.suptitle('Path Postprocessing Pipeline: 98 pts → 5 pts → steps=0',
                 fontsize=16, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    filepath = output_dir / "postprocessing_pipeline.png"
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filepath}")

    # Print summary
    print(f"\nSummary:")
    print(f"  Raw path:           {len(steps['0_raw'])} pts")
    print(f"  After Savgol:       {len(steps['A_smoothed'])} pts")
    print(f"  Cutoff index:       {steps['cutoff_idx']}")
    print(f"  After curvature:    {len(steps['B_curvature_cutoff'])} pts")
    print(f"  Skip ratio:         {steps['skip_ratio']}")
    print(f"  After spacing:      {len(steps['C_spacing_skip'])} pts")
    print(f"  Executed waypoints: {max(0, len(steps['C_spacing_skip']) - 2)}")


if __name__ == "__main__":
    main()
