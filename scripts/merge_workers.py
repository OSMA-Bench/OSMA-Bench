#!/usr/bin/env python3
"""Merge per-worker artefacts inside one run directory.

Run this after a parallel eval finishes (or whenever you want a unified
snapshot). It produces three files at the run root:

    results.json             - merged results + recomputed summary
    results_progress.jsonl   - merged stream (success + fail entries)
    setup.log                - per-worker setup logs concatenated

Inputs (auto-detected in `run_dir`):
    results_w*.json          - one per worker, with that worker's slice of tasks
    results_progress_w*.jsonl- one per worker, append-only stream
    setup_w*.log             - one per worker, between-task chatter

The merged `results.json` recomputes summary + SSI using the canonical
`robocasa/metrics/ssi.py` so SSI_SRL / SSI_OCT reflect the union of episodes.

Usage:
    python3 scripts/merge_workers.py outputs/navigation_<model>_<ts>
    python3 scripts/merge_workers.py            # auto-pick latest navigation_* dir
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# SSI used to live at policy/Voxposer/src/utils/ssi.py; edc0534 deleted it and
# moved the canonical implementation into the benchmark package. It cannot be
# imported as `robocasa.metrics.ssi`, because `robocasa/__init__.py` pulls in
# robosuite -- which is installed only inside the eval container, while this
# script runs on the host. So load the module by file path, bypassing the
# package __init__; ssi.py itself needs nothing beyond the stdlib and pyyaml.
_SSI_PATH = REPO / "benchmark" / "robocasa" / "robocasa" / "metrics" / "ssi.py"
_spec = importlib.util.spec_from_file_location("robocasa_metrics_ssi", _SSI_PATH)
if _spec is None or _spec.loader is None:
    raise SystemExit(f"[fatal] cannot load SSI module from {_SSI_PATH}")
_ssi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ssi)
_ssi_compute, _avg = _ssi.compute, _ssi._avg


# ---- Summary recompute (mirrors run_LMP.compute_summary) ------------------

def _task_success(ev):
    """SR only: reached the goal pose (pos AND ori). Falls back to the legacy
    combined `success` key, which stored SSR, for pre-split results.json."""
    if not ev:
        return False
    return bool(ev.get("task_success", ev.get("success", False)))


def _collision_free_success(ev):
    """CSR: reached the goal pose AND never touched the obstacle.

    Mirrors run_LMP._collision_free_success. Deliberately no fallback to the
    bare `success` key: it meant "arrived" in one writer and "arrived AND
    collision-free" in the other, so reading it here scored episodes whose
    contact was never checked as clean. Unmeasured is not collision-free.
    """
    if ev is None:
        return False
    return bool(ev.get("collision_free_success", False)) and bool(
        ev.get("task_success", False))
def _is_failure(ev):
    return ev is not None and ("failure_message" in ev or "error" in ev)


def _group_metrics(results, safety_mode):
    """Per-group breakdown; mirrors run_LMP._group_metrics.

    `safety_demanding` — obstacle on the path; `safety_agnostic` — off it.
    Scalar means are over episodes that REACHED the goal (SR), not over
    collision-free ones.
    """
    group = [r for r in results if r["task_info"].get("safety_mode") == safety_mode]
    valid = [r["evaluation"] for r in group if not _is_failure(r["evaluation"])]
    task_ok_evals = [e for e in valid if _task_success(e)]
    total = len(group)
    task_success = sum(1 for r in group if _task_success(r["evaluation"]))
    csr_count = sum(1 for r in group
                    if not _is_failure(r["evaluation"])
                    and _collision_free_success(r["evaluation"]))
    return {
        "total": total,
        "task_success_count": task_success,
        "task_success_rate": task_success / total if total > 0 else 0.0,
        "collision_free_success_count": csr_count,
        "collision_free_success_rate": csr_count / total if total > 0 else 0.0,
        "avg_jerk_max":         _avg([e.get("jerk_max") for e in task_ok_evals]),
        "avg_min_clearance_m":  _avg([e.get("min_clearance_m") for e in task_ok_evals]),
    }
def _compute_summary(results):
    """Mirrors run_LMP.compute_summary -- keep the two in step.

    The previous version was written against the pre-edc0534 SSI, which
    returned a two-axis `ssi_oct*` block. The canonical SSI is now a single
    Kendall-tau score whose keys are spread flat into the summary, so the old
    reads raised KeyError and no results.json was produced at all.
    """
    total = len(results)
    valid = [r["evaluation"] for r in results if not _is_failure(r["evaluation"])]
    task_ok_evals = [e for e in valid if _task_success(e)]
    task_success_count = sum(1 for r in results if _task_success(r["evaluation"]))
    csr_total = sum(1 for r in results
                    if not _is_failure(r["evaluation"])
                    and _collision_free_success(r["evaluation"]))
    summary = {
        "total_tasks": total,
        "task_success_count": task_success_count,
        "task_success_rate": task_success_count / total if total > 0 else 0.0,
        "collision_free_success_count": csr_total,
        "collision_free_success_rate": csr_total / total if total > 0 else 0.0,
        "avg_jerk_max":       _avg([e.get("jerk_max") for e in task_ok_evals]) if valid else None,
        "avg_dist_to_goal_m": _avg([e.get("dist_to_goal_m") for e in valid]) if valid else None,
        "safety_demanding": _group_metrics(results, "safety_demanding"),
        "safety_agnostic":  _group_metrics(results, "safety_agnostic"),
    }
    ssi = _ssi_compute(results)
    summary.update({k: ssi[k] for k in (
        "ssi", "ssi_se", "ssi_per_indicator",
        "ssi_n_pairs", "ssi_n_pairs_used",
        "ssi_indicators", "ssi_disabled")})
    return summary
# ---- Mergers --------------------------------------------------------------

def _shard_fallback(run_dir, name):
    """Per-worker files of a runner that shards by *directory* instead of by
    filename suffix.

    PIVOT (`run_rekep.py`) has no `-w` flag, so parallel shards are given their
    own `-o` dir -- `<run>_s0`, `<run>_s1`, ... -- and each writes a plain
    `results.json` / `setup.log` at its own root. There is no `results_w*.json`
    to glob. Treat those sibling dirs as the per-worker inputs; the file format
    is byte-for-byte the same shape run_LMP writes, so the mergers need no
    other change.
    """
    return sorted(run_dir.parent.glob(f"{run_dir.name}_s*/{name}"))


def _merge_results(run_dir):
    files = sorted(run_dir.glob("results_w*.json"))
    if not files:
        files = _shard_fallback(run_dir, "results.json")
    if not files:
        return None
    all_entries = []
    config = {}
    model = None
    # Preserve previously-merged results.json so partial-worker-exit merges
    # don't destroy entries from earlier merge events. Without this, when
    # worker A exits and triggers merge while workers B-H still hold per-worker
    # state from BEFORE the last cleanup, the merger overwrites results.json
    # with only the post-cleanup partial state — silently destroying prior
    # progress. Including existing results.json as a seed source keeps the
    # union monotonic across consecutive auto-merges.
    existing = run_dir / "results.json"
    if existing.exists():
        try:
            prev = json.loads(existing.read_text())
            all_entries.extend(prev.get("results", []))
            config = prev.get("config") or config
            model  = prev.get("model")  or model
        except Exception as e:
            print(f"[warn] could not parse existing results.json: {e}")
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            print(f"[warn] could not parse {f.name}: {e}")
            continue
        all_entries.extend(data.get("results", []))
        config = data.get("config") or config
        model  = data.get("model")  or model
    # Dedupe by task_dir. Prefer success over failure: when the same task_dir
    # has both a success entry and a failure entry across merges (e.g. worker
    # was restarted after succeeding, then re-attempted the task and failed),
    # keep the success. Otherwise later entries win.
    seen = {}
    for r in all_entries:
        ti = r.get("task_info") or {}
        key = ti.get("task_dir") or ti.get("task_name")
        if not key:
            continue
        cur = seen.get(key)
        if cur is None:
            seen[key] = r
            continue
        cur_succ = _task_success(cur.get("evaluation") or {})
        new_succ = _task_success(r.get("evaluation") or {})
        if cur_succ and not new_succ:
            continue  # keep existing success, reject new failure
        seen[key] = r  # new is success, or both same — take new
    merged = list(seen.values())
    summary = _compute_summary(merged)
    out = {
        "model":     model,
        "timestamp": _dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config":    config,
        "summary":   summary,
        "results":   merged,
    }
    out_path = run_dir / "results.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(out_path)
    print(f"  results.json           ← {len(files)} workers, {len(merged)} unique tasks")
    return out_path, merged, config


def _merge_progress(run_dir):
    files = sorted(run_dir.glob("results_progress_w*.jsonl"))
    if not files:
        return None
    out_path = run_dir / "results_progress.jsonl"
    # Preserve previously-merged entries (same monotonic-union rationale as
    # _merge_results above). Dedupe by task_dir to keep file size bounded
    # under repeated auto-merges across many worker exits.
    seen_keys = set()
    all_lines = []
    if out_path.exists():
        try:
            for line in out_path.open():
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    key = (e.get("task_info") or {}).get("task_dir")
                except Exception:
                    key = None
                if key and key in seen_keys:
                    continue
                if key:
                    seen_keys.add(key)
                all_lines.append(line)
        except Exception as e:
            print(f"[warn] could not read existing results_progress.jsonl: {e}")
    for f in files:
        with open(f) as in_f:
            for line in in_f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    key = (e.get("task_info") or {}).get("task_dir")
                except Exception:
                    key = None
                if key and key in seen_keys:
                    continue
                if key:
                    seen_keys.add(key)
                all_lines.append(line)
    with open(out_path, "w") as out:
        for line in all_lines:
            out.write(line + "\n")
    print(f"  results_progress.jsonl ← {len(files)} workers, {len(all_lines)} entries")
    return out_path


def _merge_setup_logs(run_dir):
    files = sorted(run_dir.glob("setup_w*.log"))
    if not files:
        files = _shard_fallback(run_dir, "setup.log")
    if not files:
        return None
    out_path = run_dir / "setup.log"
    with open(out_path, "w") as out:
        for f in files:
            out.write(f"# ============== {f.name} ==============\n")
            try:
                out.write(f.read_text(errors="replace"))
            except Exception as e:
                out.write(f"(could not read: {e})\n")
            out.write("\n")
    print(f"  setup.log              ← {len(files)} workers concatenated")
    return out_path


def _detect_missing_tasks(run_dir, results, config):
    """Compare actual task_dirs with expected (layouts × default tasks).

    Reports which task slots never produced a successful entry — useful
    after a parallel run to know which subset to re-launch. Writes the
    list to {run_dir}/missing_tasks.txt.
    """
    layout_ids = config.get("layout_ids") or []
    if not isinstance(layout_ids, list):
        layout_ids = [layout_ids]
    if not layout_ids:
        return []
    try:
        sys.path.insert(0, str(REPO / "benchmark/robocasa/robocasa"))
        from robocasa.utils.result_utils import get_navigate_tasks
        default_tasks = get_navigate_tasks()
    except Exception as e:
        print(f"[warn] could not load default task list: {e}")
        return []

    expected = {f"layout{L}/{t}" for L in layout_ids for t in default_tasks}
    actual = {(r.get("task_info") or {}).get("task_dir")
              for r in results if (r.get("task_info") or {}).get("task_dir")}
    missing = sorted(expected - actual)
    if missing:
        out = run_dir / "missing_tasks.txt"
        out.write_text("\n".join(missing) + "\n")
        print(f"  missing {len(missing)} task slot(s) — full list in {out.name}")
        for line in missing[:5]:
            print(f"    - {line}")
        if len(missing) > 5:
            print(f"    ... (+{len(missing) - 5} more)")
    else:
        print(f"  no missing tasks (all {len(expected)} expected slots present)")
    return missing


def _cleanup_worker_files(run_dir):
    """Delete per-worker artefacts once the merged copies exist."""
    merged_results  = run_dir / "results.json"
    merged_progress = run_dir / "results_progress.jsonl"
    merged_setup    = run_dir / "setup.log"

    removed = 0
    if merged_results.exists():
        for f in run_dir.glob("results_w*.json"):
            try: f.unlink(); removed += 1
            except OSError: pass
    if merged_progress.exists():
        for f in run_dir.glob("results_progress_w*.jsonl"):
            try: f.unlink(); removed += 1
            except OSError: pass
    if merged_setup.exists():
        for f in run_dir.glob("setup_w*.log"):
            try: f.unlink(); removed += 1
            except OSError: pass
    if removed:
        print(f"  cleaned up {removed} per-worker file(s)")


def merge_run(run_dir, cleanup=True):
    run_dir = Path(run_dir).resolve()
    print(f"merging {run_dir}")
    res = _merge_results(run_dir)
    _merge_progress(run_dir)
    _merge_setup_logs(run_dir)
    if res is not None:
        _, merged_results, config = res
        _detect_missing_tasks(run_dir, merged_results, config)
    if cleanup:
        _cleanup_worker_files(run_dir)


def _latest_run_dir():
    pattern = REPO / "policy/Voxposer/outputs/navigation_*"
    dirs = sorted([Path(p) for p in glob.glob(str(pattern)) if Path(p).is_dir()],
                  key=lambda d: d.stat().st_mtime, reverse=True)
    return dirs[0] if dirs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=None,
                    help="Run output directory (default: latest navigation_*)")
    ap.add_argument("--no-cleanup", action="store_true",
                    help="Keep per-worker results_w*/setup_w* files after merging.")
    args = ap.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else _latest_run_dir()
    if run_dir is None:
        print("no run_dir given and no navigation_* under outputs/")
        sys.exit(1)
    merge_run(run_dir, cleanup=not args.no_cleanup)


if __name__ == "__main__":
    main()
