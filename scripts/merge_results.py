"""DEPRECATED — use scripts/merge_workers.py instead.

This script targets the pre-2026-05 schema (jerk_rms / blocking_mode / error
/ etc.). merge_workers.py uses the canonical SSI_SRL/SSI_OCT pipeline,
auto-detects missing tasks, and cleans up per-worker files. This file is
kept only to keep legacy archived runs analysable.
"""
import json
import os
import sys
import glob


def merge_worker_results(run_dir):
    """Merge results_w*.json files in a single run directory into results.json."""
    pattern = os.path.join(run_dir, "results_w*.json")
    result_files = sorted(glob.glob(pattern))

    # Fallback: also check for plain results.json (single-worker run)
    if not result_files:
        single = os.path.join(run_dir, "results.json")
        if os.path.exists(single):
            print(f"Single results.json found in {run_dir}, nothing to merge.")
            return single
        print(f"No results_w*.json files found in {run_dir}")
        return None

    print(f"Found {len(result_files)} worker result files:")
    all_results = []
    for f in result_files:
        print(f"  {f}")
        with open(f) as fh:
            data = json.load(fh)
        results = data.get("results", [])
        all_results.extend(results)
        print(f"    -> {len(results)} tasks")

    # Deduplicate by task_name (in case of reruns)
    seen = {}
    for r in all_results:
        key = r["task_info"]["task_name"]
        seen[key] = r  # later results overwrite earlier
    merged = list(seen.values())

    # Compute summary
    total = len(merged)
    success = sum(1 for r in merged if r["evaluation"].get("success"))
    valid = [r["evaluation"] for r in merged if "error" not in r["evaluation"]]

    summary = {
        "total_tasks": total,
        "success_count": success,
        "success_rate": success / total if total > 0 else 0.0,
    }

    if valid:
        jerks = [e["jerk_rms"] for e in valid if e.get("jerk_rms") is not None]
        summary["avg_jerk_rms"] = sum(jerks) / len(jerks) if jerks else None
        violations = [e["violation_ratio"] for e in valid if e.get("violation_ratio") is not None]
        summary["avg_violation_ratio"] = sum(violations) / len(violations) if violations else None
        dists = [e["dist_to_goal_m"] for e in valid if e.get("dist_to_goal_m") is not None]
        summary["avg_dist_to_goal_m"] = sum(dists) / len(dists) if dists else None

    # Save merged results.json
    output_path = os.path.join(run_dir, "results.json")
    with open(output_path, "w") as fh:
        json.dump({"summary": summary, "results": merged}, fh, indent=2)

    print(f"\nMerged: {total} tasks -> {output_path}")
    print(f"Success: {success}/{total} ({summary['success_rate']:.1%})")
    if summary.get("avg_dist_to_goal_m"):
        print(f"Avg dist: {summary['avg_dist_to_goal_m']:.3f}m")

    return output_path


def main():
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        # Auto-find latest navigation output directory
        base = "/workspace/policy/Voxposer/outputs"
        dirs = sorted(glob.glob(os.path.join(base, "navigation_*")))
        if not dirs:
            print("No navigation output directories found")
            return
        run_dir = dirs[-1]
        print(f"Using latest output: {run_dir}")

    output_path = merge_worker_results(run_dir)

    # Generate visualizations on merged results
    if output_path:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmark", "robocasa", "robocasa"))
            from utils.visualization_utils import generate_annotated_frames
            print("\nGenerating visualizations...")
            generate_annotated_frames(run_dir)
        except Exception as e:
            print(f"Visualization failed: {e}")
            print("Run manually: python3 scripts/annotate_final_frames.py " + run_dir)


if __name__ == "__main__":
    main()
