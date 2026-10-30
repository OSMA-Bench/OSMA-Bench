"""
Extract final frame from topview video and annotate with robot/goal positions.
Uses MuJoCo camera projection with Y-flip correction (video is saved with [::-1]).
"""
import cv2
import json
import math
import numpy as np
import os
import shutil


# Topview camera params (same across all layouts)
CAM_POS = np.array([1.453, -4.122, 9.475])
CAM_MAT = np.array([
    [-7.99974401e-03,  9.95820527e-01, -9.09806677e-02],
    [-9.99967993e-01, -7.99974401e-03,  3.37003578e-07],
    [-7.19942632e-04,  9.09777558e-02,  9.95852518e-01],
])
CAM_FOVY = 45.0


def world_to_pixel(world_xy, img_w=640, img_h=480):
    """Project world XY (z=0) to pixel in the flipped topview image."""
    d = np.array([world_xy[0], world_xy[1], 0.0]) - CAM_POS
    x = np.dot(d, CAM_MAT[:, 0])
    y = np.dot(d, CAM_MAT[:, 1])
    z = -np.dot(d, CAM_MAT[:, 2])
    if z <= 0:
        return None
    f = img_h / (2.0 * np.tan(np.radians(CAM_FOVY) / 2.0))
    px = int(img_w / 2 + f * x / z)
    py = int(img_h / 2 + f * y / z)
    # Video is saved with [::-1] vertical flip
    py = img_h - py
    return (px, py)


def get_last_frame(video_path):
    """Extract last frame from video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def draw_marker(img, px, label, color, radius=12):
    """Draw a labeled circle marker with background."""
    if px is None:
        return
    cv2.circle(img, px, radius, color, 2)
    cv2.circle(img, px, 4, color, -1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    tx, ty = px[0] + radius + 4, px[1] + 5
    cv2.rectangle(img, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 4), (0, 0, 0), -1)
    cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def compute_ori_error(ev, route):
    """Recompute orientation error matching _check_success logic."""
    robot_yaw = ev.get("final_yaw_rad")
    goal_yaw = ev.get("goal_yaw_rad")
    final_xy = ev.get("final_pos_xy")
    goal_xy = ev.get("goal_pos_xy")

    if robot_yaw is None or final_xy is None or goal_xy is None:
        return None

    # RouteF (dst=Human): dot(robot_fwd, dir_to_human)
    if route == "F":
        robot_fwd = np.array([math.cos(robot_yaw), math.sin(robot_yaw)])
        dir_to_goal = np.array(goal_xy) - np.array(final_xy)
        d = np.linalg.norm(dir_to_goal)
        if d > 1e-3:
            return float(np.dot(robot_fwd, dir_to_goal / d))
        return 1.0
    else:
        # General: cos(goal_yaw - robot_yaw)
        if goal_yaw is None:
            return None
        return float(math.cos(goal_yaw - robot_yaw))


def annotate_frame(frame, ev, task_name, instruction, route=None):
    """Annotate frame with start/final/goal positions, distance, and instruction."""
    h, w = frame.shape[:2]
    annotated = frame.copy()

    start_xy = ev.get("start_pos_xy")
    final_xy = ev.get("final_pos_xy")
    goal_xy = ev.get("goal_pos_xy")
    dist = ev.get("dist_to_goal_m", 0)
    success = ev.get("task_success", False)

    # Recompute ori_cos with correct logic
    ori_cos = compute_ori_error(ev, route)

    # Convert ori_cos to degrees
    ori_deg = math.degrees(math.acos(max(-1, min(1, ori_cos)))) if ori_cos is not None else None

    # Project to pixels
    start_px = world_to_pixel(start_xy, w, h) if start_xy else None
    final_px = world_to_pixel(final_xy, w, h) if final_xy else None
    goal_px = world_to_pixel(goal_xy, w, h) if goal_xy else None

    # Draw start (blue dotted trail to final)
    if start_px and final_px:
        for i in range(0, 100, 3):
            t = i / 100.0
            px = int(start_px[0] * (1 - t) + final_px[0] * t)
            py = int(start_px[1] * (1 - t) + final_px[1] * t)
            cv2.circle(annotated, (px, py), 1, (255, 150, 0), -1)
    draw_marker(annotated, start_px, "START", (255, 150, 0), radius=10)

    # Draw goal (cyan)
    draw_marker(annotated, goal_px, "GOAL", (255, 255, 0), radius=15)

    # Determine status: SUCCESS / PARTIAL SUCCESS / FAILURE
    dist_thresh = 0.8 if route == "F" else 0.5
    ori_thresh_cos = 0.8    # ~37 degrees
    pos_ok = dist <= dist_thresh
    ori_ok = ori_cos is not None and ori_cos >= ori_thresh_cos

    if pos_ok and ori_ok:
        status = "SUCCESS"
        status_color = (0, 255, 0)     # green
        robot_color = (0, 255, 0)
    elif pos_ok:
        status = "PARTIAL SUCCESS"
        status_color = (0, 200, 255)   # orange
        robot_color = (0, 200, 255)
    else:
        status = "FAILURE"
        status_color = (0, 0, 255)     # red
        robot_color = (0, 0, 255)

    draw_marker(annotated, final_px, "ROBOT", robot_color, radius=12)

    # Draw distance line: final -> goal
    if final_px and goal_px:
        cv2.line(annotated, final_px, goal_px, (255, 255, 255), 1, cv2.LINE_AA)
        mid = ((final_px[0] + goal_px[0]) // 2, (final_px[1] + goal_px[1]) // 2)
        dist_text = f"{dist:.3f}m"
        (tw, th), _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (mid[0] - tw // 2 - 4, mid[1] - th - 6),
                      (mid[0] + tw // 2 + 4, mid[1] + 6), (0, 0, 0), -1)
        cv2.putText(annotated, dist_text, (mid[0] - tw // 2, mid[1] + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Header
    ori_str = f"{ori_deg:.1f}deg" if ori_deg is not None else "-"
    cv2.rectangle(annotated, (0, 0), (w, 65), (0, 0, 0), -1)
    cv2.putText(annotated, task_name, (10, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    instr_display = instruction[:80] if instruction else ""
    cv2.putText(annotated, f'"{instr_display}"', (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(annotated, f"{status} | dist={dist:.3f}m | ori_err={ori_str}",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

    return annotated


def generate_annotated_frames(base_dir):
    """Generate annotated final frames for all tasks in base_dir."""
    results_path = os.path.join(base_dir, "results.json")
    if not os.path.exists(results_path):
        results_path = os.path.join(base_dir, "results_latest.json")
    if not os.path.exists(results_path):
        print(f"No results file found in {base_dir}")
        return
    output_dir = os.path.join(base_dir, "visualizations")

    # Clean old
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    with open(results_path) as f:
        data = json.load(f)
    results = data["results"]

    processed = 0
    skipped = 0

    for r in results:
        ev = r["evaluation"]
        task_name = r["task_info"]["task_name"]
        instruction = r["task_info"].get("instruction", "")

        if "error" in ev or ev.get("final_pos_xy") is None:
            skipped += 1
            continue

        # Find task directory
        task_dir = None
        for d in os.listdir(base_dir):
            if d.startswith(task_name) and os.path.isdir(os.path.join(base_dir, d)):
                task_dir = os.path.join(base_dir, d)
                break
        if task_dir is None:
            skipped += 1
            continue

        video_path = os.path.join(task_dir, "topview_image.mp4")
        if not os.path.exists(video_path):
            skipped += 1
            continue

        frame = get_last_frame(video_path)
        if frame is None:
            skipped += 1
            continue

        route = r["task_info"].get("route")
        annotated = annotate_frame(frame, ev, task_name, instruction, route=route)
        out_path = os.path.join(output_dir, f"{task_name}_final.png")
        cv2.imwrite(out_path, annotated)
        processed += 1

    print(f"Processed: {processed}, Skipped: {skipped}")

    # Per-route grids
    routes = ["A", "B", "C", "D", "E", "F", "G"]
    for route in routes:
        route_results = [r for r in results
                         if r["task_info"].get("route") == route
                         and "error" not in r["evaluation"]
                         and r["evaluation"].get("final_pos_xy") is not None]
        if not route_results:
            continue

        n = len(route_results)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        cell_w, cell_h = 640, 480
        grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

        for idx, r in enumerate(route_results):
            tn = r["task_info"]["task_name"]
            img_path = os.path.join(output_dir, f"{tn}_final.png")
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (cell_w, cell_h))
            row, col = idx // cols, idx % cols
            grid[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w] = img

        grid_path = os.path.join(output_dir, f"route_{route}_grid.png")
        cv2.imwrite(grid_path, grid)
        print(f"Route {route}: {n} tasks -> {grid_path}")


def main():
    import sys
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        # Find the latest navigation output directory
        import glob
        dirs = sorted(glob.glob("/workspace/policy/Voxposer/outputs/navigation_*"))
        if not dirs:
            print("No navigation output directories found")
            return
        base_dir = dirs[-1]
        print(f"Using latest output: {base_dir}")
    generate_annotated_frames(base_dir)


if __name__ == "__main__":
    main()
