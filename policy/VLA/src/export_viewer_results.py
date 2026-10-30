#!/usr/bin/env python3
"""Export OSMA-Bench VLA sweep(s) into the team viewer layout (Viewer_Guide.md).

    results/VLA/<model>/layout<code>/<TaskClassName>/run_seed<S>.mp4
                                                     tracked_seed<S>.npz

Derived artefact only — everything is regenerated from the sweep's
``results.json`` (episode list + ``evaluation``), the per-task
``trajectory_log.json`` (via the canonical v2 ``ssi.enrich_with_boundary_stats``)
and ``vla_dump.npz``.  The mp4 is ``_runs/<run_dir>/rollout.mp4`` re-encoded to
the guide's browser-safe profile.

Conventions fixed by anon on 2026-09-02 (see anon/vla_viewer_export_0902.md; procedure = /viewer-export skill).
Lived in anon/vla_viewer_export_0902.py until 2026-09-04 (policy/VLA was frozen by running sweeps).
  layout   = RoboCasa LayoutType *code* ID (layout0/2/5/7/8), not the paper label
  obstacle = full task class name (NavigateKitchenDogBlockingRouteB)
  model    = XR-1 · pi0.5-OpenTau · pi0.5-Human300-r50
  seed     = env seed (== policy seed) → run_seed<S>

Usage
-----
    python3 policy/VLA/src/export_viewer_results.py \\
        --sweep ~/vla/sweeps/20260831T055800Z__xr1__osmabench1250__seed0__h500__r16 \\
        [--sweep ...] --out-root ~/vla/viewer_export [--jobs 4]
        [--skip-video] [--dry-run] [--verify-only] [--model-name NAME] [--expect-n N] [--force]

Never run with more than a few --jobs on the shared server: ffmpeg CPU load has
made concurrent EGL initialisation fail for other people's sweeps.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # never drop __pycache__ into watched trees

import argparse
import datetime as _dt
import functools
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent  # layout-agnostic: inline osmabench / submodule / VLA-nav clone
DEFAULT_V2 = "~/vla/src/robocasa_v2"
DEFAULT_OUT = "~/vla/viewer_export"
CATEGORY = "VLA"

#: (model_profile, replan_steps) -> viewer model folder.  Anything else fails
#: loudly so a stray sweep (H300 native r5, GR00T, 690/v1) cannot slip in.
MODEL_NAMES = {
    ("xr1", 16): "XR-1",
    ("pi05_opentau", 20): "pi0.5-OpenTau",
    ("pi05_human300", 50): "pi0.5-Human300-r50",
    # GR00T N1.5 target post-training (atomic_seen) — a full v2 panel model since 2026-09-03
    # (README "정식 패널 모델"); viewer name chosen by anon 2026-09-04.
    ("groot_n1_5_atomic_seen", 16): "GR00T-N1.5-atomic_seen",
}
EXPECTED_N = 1250
BASELINE_INSTRUCTION_MODE = "environment"
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")  # viewer: no spaces/한글/# ? %
EXPECTED_LAYOUTS = [0, 2, 5, 7, 8]

FFMPEG_ARGS = [
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", "-preset", "veryfast",
    "-movflags", "+faststart", "-vf", "scale=640:-2,fps=15",
]
MAX_MP4_BYTES = 10 * 1024 * 1024
REQUIRED_KEYS = ("success", "obstacle_name", "min_clearance_m", "v_b", "a_b_mean", "jerk_b_mean")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{_dt.datetime.now(_dt.timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_v2_ssi():
    """Load the v2 ssi.py exactly like policy/VLA/src/compute_ssi_v2.py does
    (importlib from OSMA_BENCH_ROBOCASA_V2) — the inline benchmark tree's ssi.py
    has the pre-v2 roster and drops the tier for 9/18 obstacles."""
    root = Path(os.environ.get("OSMA_BENCH_ROBOCASA_V2", DEFAULT_V2))
    path = root / "robocasa" / "utils" / "ssi.py"
    if not path.is_file():
        raise SystemExit(f"v2 ssi.py not found: {path} (set OSMA_BENCH_ROBOCASA_V2)")
    spec = importlib.util.spec_from_file_location("_robocasa_ssi_v2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    for name in ("enrich_with_boundary_stats", "ep_v_b", "ep_accel_b_mean",
                 "ep_jerk_b_mean", "ep_min_clearance", "_task_success", "TIER_OF", "TIER_R_B", "DT"):
        if not hasattr(mod, name):
            raise SystemExit(f"v2 ssi.py lacks {name}")
    return mod, path


@functools.lru_cache(maxsize=1)
def load_control_log_interval_steps() -> int:
    """Read the rollout cadence from the same Robocasa tree as the sweep."""
    root = Path(os.environ.get(
        "OSMA_BENCH_ROBOCASA_ROOT",
        os.environ.get("OSMA_BENCH_ROBOCASA_V2", DEFAULT_V2),
    ))
    path = root / "robocasa" / "control.py"
    if not path.is_file():
        raise SystemExit(f"control cadence module not found: {path} (set OSMA_BENCH_ROBOCASA_ROOT)")
    spec = importlib.util.spec_from_file_location("_robocasa_control", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load control cadence module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    steps = int(module.CONTROL_LOG_INTERVAL_STEPS)
    if steps < 1:
        raise SystemExit(f"CONTROL_LOG_INTERVAL_STEPS must be positive, got {steps}")
    return steps


def f32(x) -> np.ndarray:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.array(np.nan, dtype=np.float32)
    return np.array(float(x), dtype=np.float32)


def git_head() -> dict:
    """Provenance of the tree this script lives in (works in all three layouts)."""
    def _run(*args):
        try:
            return subprocess.run(["git", "-C", str(SCRIPT_DIR), *args], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None
    top = _run("rev-parse", "--show-toplevel")
    rel = str(Path(__file__).resolve().relative_to(top)) if top else None
    return {
        "repo": top,
        "commit": _run("rev-parse", "HEAD"),
        "script_status": _run("status", "--porcelain", "--", rel) if rel else None,
    }


# ----------------------------------------------------------------------------
# sweep handling
# ----------------------------------------------------------------------------

class Sweep:
    def __init__(self, path: Path, model_name: str | None = None, expect_n: int = EXPECTED_N):
        self.path = path.resolve()
        self.id = self.path.name
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        self.results_path = self.path / "results.json"
        data = json.loads(self.results_path.read_text())
        if data.get("schema_version") != 1:
            raise SystemExit(f"{self.id}: unexpected results schema {data.get('schema_version')}")
        self.results = data["results"]
        m = self.manifest
        self.profile = m["model_profile"]
        self.replan = int(m["replan_steps"])
        self.seed = int(m["seed"])
        if int(m.get("policy_seed", self.seed)) != self.seed:
            raise SystemExit(f"{self.id}: env seed {self.seed} != policy seed {m.get('policy_seed')}")
        self.instruction_mode = str(m.get("instruction_mode", ""))
        key = (self.profile, self.replan)
        if model_name is not None:
            if not MODEL_NAME_RE.match(model_name):
                raise SystemExit(f"--model-name {model_name!r}: letters/digits/._- only (viewer path rule)")
            if model_name in MODEL_NAMES.values() and (key not in MODEL_NAMES or MODEL_NAMES[key] != model_name
                                                       or self.instruction_mode != BASELINE_INSTRUCTION_MODE):
                raise SystemExit(f"--model-name {model_name!r} is a baseline panel name; a different "
                                 f"experiment must not overwrite it — pick e.g. '{model_name}-<condition>'")
            self.model, self.model_source = model_name, "cli"
        elif key in MODEL_NAMES and self.instruction_mode == BASELINE_INSTRUCTION_MODE:
            self.model, self.model_source = MODEL_NAMES[key], "auto"
        elif key in MODEL_NAMES:
            raise SystemExit(f"{self.id}: instruction_mode={self.instruction_mode!r} is not the baseline "
                             f"({BASELINE_INSTRUCTION_MODE!r}) — this is a separate experiment; pass "
                             f"--model-name '{MODEL_NAMES[key]}-<condition>' (one --sweep per call)")
        else:
            raise SystemExit(f"{self.id}: no viewer model name for profile={self.profile!r} replan={self.replan} — "
                             f"add it to MODEL_NAMES (panel model) or pass --model-name (experiment)")
        # sanity: directory name must agree with manifest
        if f"__seed{self.seed}__" not in self.id or f"__r{self.replan}" not in self.id \
                or f"__{self.profile}__" not in self.id:
            raise SystemExit(f"{self.id}: directory name disagrees with manifest "
                             f"({self.profile}, seed {self.seed}, r{self.replan})")
        if m.get("robocasa_root_source") != "OSMA_BENCH_ROBOCASA_ROOT" \
                or m.get("robosuite_root_source") != "OSMA_BENCH_ROBOSUITE_ROOT":
            raise SystemExit(f"{self.id}: not a v2-stack sweep (robocasa/robosuite root source)")
        if sorted(m["layout_ids_code"]) != EXPECTED_LAYOUTS:
            raise SystemExit(f"{self.id}: layouts {m['layout_ids_code']} != {EXPECTED_LAYOUTS}")
        if len(self.results) != expect_n:
            raise SystemExit(f"{self.id}: {len(self.results)} entries != {expect_n} (use --expect-n for a "
                             f"different task set; never export a partial sweep)")
        errs = [r for r in self.results if "error" in r["evaluation"] or "failure_message" in r["evaluation"]]
        if errs:
            raise SystemExit(f"{self.id}: {len(errs)} error entries — export refused (rerun the sweep)")
        tasks = {r["task_info"]["task_dir"] for r in self.results}
        if len(tasks) != expect_n:
            raise SystemExit(f"{self.id}: duplicate task_dir entries")

    # -- per-entry paths -----------------------------------------------------
    def leaf(self, out_results: Path, ti: dict) -> Path:
        return out_results / CATEGORY / self.model / f"layout{int(ti['layout_id'])}" / ti["task_name"]

    def src_mp4(self, ev: dict) -> Path:
        return self.path / ev["run_dir"] / "rollout.mp4"

    def src_dump(self, ti: dict) -> Path:
        return self.path / ti["task_dir"] / "vla_dump.npz"


def build_npz_payload(sweep: Sweep, entry: dict, ssi) -> dict:
    ti, ev = entry["task_info"], entry["evaluation"]
    obstacle = ti["obstacle"]
    if obstacle not in ssi.TIER_OF or obstacle not in ssi.TIER_R_B:
        raise RuntimeError(f"{sweep.id} {ti['task_name']}: obstacle {obstacle!r} not in v2 roster")
    if ti["task_name"] != Path(ti["task_dir"]).name:
        raise RuntimeError(f"task_name/task_dir mismatch: {ti}")

    z = np.load(sweep.src_dump(ti), allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    if meta["environment"]["name"] != ti["task_name"] or int(meta["environment"]["seed"]) != sweep.seed:
        raise RuntimeError(f"vla_dump meta disagrees with task_info: {ti['task_name']} / seed {sweep.seed}")
    control_freq = float(z["control_freq_hz"])
    force = np.asarray(z["obstacle_contact_force_peak_n"], dtype=np.float32)  # (T, K)
    contact_ever = np.asarray(z["obstacle_contact_ever"], dtype=bool)
    min_obs = np.asarray(z["min_obstacle_distance_m"], dtype=np.float32)
    goal_dist = np.asarray(z["goal_distance_m"], dtype=np.float32)

    # canonical accessors (after enrich_with_boundary_stats has run on the list)
    v_b = ssi.ep_v_b(ev)
    a_b = ssi.ep_accel_b_mean(ev)
    j_b = ssi.ep_jerk_b_mean(ev)
    d_min = ssi.ep_min_clearance(ev)
    if d_min is None:
        raise RuntimeError(f"{ti['task_name']}: min_clearance_m missing")
    if min_obs.size and not math.isclose(float(min_obs.min()), float(d_min), abs_tol=1e-5):
        raise RuntimeError(f"{ti['task_name']}: dump min distance {min_obs.min()} != evaluation {d_min}")

    return {
        # --- required by the viewer ---
        "success": np.array(bool(ev["success"])),
        "obstacle_name": np.array(str(obstacle)),
        "min_clearance_m": f32(d_min),
        "v_b": f32(v_b),
        "a_b_mean": f32(a_b),
        "jerk_b_mean": f32(j_b),
        # --- success variants / safety ---
        "task_success_obstacle_rule": np.array(bool(ssi._task_success(ev, obstacle))),
        "safe_success": np.array(bool(ev.get("safe_success"))),
        "violation_ratio": f32(ev.get("violation_ratio")),
        "dist_to_goal_m": f32(ev.get("dist_to_goal_m")),
        "ori_cos": f32(ev.get("ori_cos")),
        "path_length_m": f32(ev.get("path_length_m")),
        "jerk_max": f32(ev.get("jerk_max")),
        "n_steps": np.array(int(ev["num_steps"])),
        "termination_reason": np.array(str(meta.get("termination_reason", ""))),
        # --- tier / boundary geometry ---
        "tier": np.array(str(ssi.TIER_OF[obstacle])),
        "r_b": f32(ssi.TIER_R_B[obstacle]),
        "distance_cap_m": f32(float(z["distance_cap_m"])),
        # --- time steps (three different dt's are in play; see the .md) ---
        "dt": f32(1.0 / control_freq),
        "dt_traj_log": f32(load_control_log_interval_steps() / control_freq),
        "a_b_dt_canonical": f32(ssi.DT),
        # --- contact (robocasa v2 substep force accumulation) ---
        "contact_force": f32(float(np.nanmax(force)) if force.size else None),
        "contact": np.array(bool(contact_ever[-1]) if contact_ever.size else False),
        "contact_steps": np.array(int((force > 0).any(axis=1).sum()) if force.size else 0),
        # --- provenance / filters ---
        "safety_mode": np.array(str(ti["safety_mode"])),
        "route": np.array(str(ti["route"])),
        "layout_code_id": np.array(int(ti["layout_id"])),
        "task_name": np.array(str(ti["task_name"])),
        "model_profile": np.array(str(sweep.profile)),
        "replan_steps": np.array(int(sweep.replan)),
        "seed": np.array(int(sweep.seed)),
        "sweep_id": np.array(str(sweep.id)),
        "run_dir": np.array(str(ev["run_dir"])),
        # --- time series (viewer shows min/max/mean summaries only) ---
        "min_obs_dist": min_obs,
        "goal_dist": goal_dist,
    }


#: Fallbacks when the guide's default (crf 28 @ 640) exceeds 10 MB — measured on the
#: worst layout5 episode (17.3 MB): crf 34 @ 640 → 7.5 MB, crf 34 @ 480 → 2.7 MB.
FALLBACK_PASSES = (("34", "640"), ("34", "480"))


def _ffmpeg(src: Path, dst: Path, crf: str, width: str) -> None:
    args = list(FFMPEG_ARGS)
    args[args.index("-crf") + 1] = crf
    args[args.index("-vf") + 1] = f"scale={width}:-2,fps=15"
    cmd = ["nice", "-n", "19", "ffmpeg", "-y", "-loglevel", "error", "-threads", "2",
           "-i", str(src), *args, str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src}: {proc.stderr.strip()[-500:]}")


def encode_video(src: Path, dst: Path) -> tuple[Path, float, str]:
    """Re-encode per Viewer_Guide §3; escalate crf/scale until ≤ MAX_MP4_BYTES.

    Idempotent: an existing dst newer than src and within the size cap is kept.
    Returns (dst, seconds, pass_label) with seconds == 0.0 when skipped."""
    if dst.exists() and dst.stat().st_size > 0 and dst.stat().st_mtime >= src.stat().st_mtime \
            and dst.stat().st_size <= MAX_MP4_BYTES:
        return dst, 0.0, "skip"
    tmp = dst.parent / f".{dst.stem}.part.mp4"  # keep .mp4 so ffmpeg picks the muxer
    t0 = time.time()
    label = "crf28@640"
    _ffmpeg(src, tmp, "28", "640")
    for crf, width in FALLBACK_PASSES:
        if tmp.stat().st_size <= MAX_MP4_BYTES:
            break
        label = f"crf{crf}@{width}"
        _ffmpeg(src, tmp, crf, width)
    if tmp.stat().st_size > MAX_MP4_BYTES:
        raise RuntimeError(f"{src}: still {tmp.stat().st_size} bytes after {label}")
    os.replace(tmp, dst)
    return dst, time.time() - t0, label


def ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    st["size"] = path.stat().st_size
    return st


# ----------------------------------------------------------------------------
# verification
# ----------------------------------------------------------------------------

def _rates(rows):
    sd = [r for r in rows if r["mode"] == "safety_demanding"]
    sa = [r for r in rows if r["mode"] == "safety_agnostic"]
    rate = lambda g, k: (sum(1 for r in g if r[k]) / len(g)) if g else None
    return {
        "n": len(rows), "n_sd": len(sd), "n_sa": len(sa),
        "SR_SD": rate(sd, "success"), "SR_SA": rate(sa, "success"), "SR_all": rate(rows, "success"),
        "SSR_SD": rate(sd, "safe"), "SSR_SA": rate(sa, "safe"),
        "SR_SD_obstacle_rule": rate(sd, "success_obs"), "SR_SA_obstacle_rule": rate(sa, "success_obs"),
    }


def verify_sweep(sweep: Sweep, out_results: Path, ssi, jobs: int, check_video: bool) -> dict:
    """Re-read what was written and compare against results.json."""
    from_npz, from_json = [], []
    probs: list[str] = []
    npz_paths, mp4_paths = [], []
    for entry in sweep.results:
        ti, ev = entry["task_info"], entry["evaluation"]
        leaf = sweep.leaf(out_results, ti)
        npz_p = leaf / f"tracked_seed{sweep.seed}.npz"
        mp4_p = leaf / f"run_seed{sweep.seed}.mp4"
        npz_paths.append(npz_p)
        mp4_paths.append(mp4_p)
        if not npz_p.is_file():
            probs.append(f"missing npz {npz_p}")
            continue
        if check_video and not mp4_p.is_file():
            probs.append(f"missing mp4 {mp4_p}")
        try:
            z = np.load(npz_p, allow_pickle=False)
        except Exception as exc:  # noqa: BLE001
            probs.append(f"unreadable npz {npz_p}: {exc}")
            continue
        for k in REQUIRED_KEYS:
            if k not in z.files:
                probs.append(f"{npz_p}: missing key {k}")
        if z["success"].dtype != np.bool_ or z["success"].ndim != 0:
            probs.append(f"{npz_p}: success must be 0-d bool")
        for k in ("min_clearance_m", "v_b", "a_b_mean", "jerk_b_mean"):
            if z[k].dtype != np.float32 or z[k].ndim != 0:
                probs.append(f"{npz_p}: {k} must be 0-d float32")
        if str(z["obstacle_name"]) != ti["obstacle"]:
            probs.append(f"{npz_p}: obstacle_name {z['obstacle_name']} != {ti['obstacle']}")
        # boundary primitives must mirror the enriched evaluation (None <-> NaN)
        for k, getter in (("a_b_mean", ssi.ep_accel_b_mean), ("jerk_b_mean", ssi.ep_jerk_b_mean), ("v_b", ssi.ep_v_b)):
            want = getter(ev)
            got = float(z[k])
            if (want is None) != math.isnan(got) or (want is not None and not math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-7)):
                probs.append(f"{npz_p}: {k} {got} != canonical {want}")
        from_npz.append({"mode": str(z["safety_mode"]), "success": bool(z["success"]),
                         "safe": bool(z["safe_success"]), "success_obs": bool(z["task_success_obstacle_rule"])})
        from_json.append({"mode": ti["safety_mode"], "success": bool(ev["success"]),
                          "safe": bool(ev.get("safe_success")), "success_obs": bool(ssi._task_success(ev, ti["obstacle"]))})

    # exactly one pair per leaf, nothing else matching the viewer regex
    leaves = {p.parent for p in npz_paths}
    for leaf in leaves:
        names = sorted(q.name for q in leaf.iterdir())
        extra = [n for n in names if n.startswith(("run_seed", "tracked_seed"))
                 and n not in (f"run_seed{sweep.seed}.mp4", f"tracked_seed{sweep.seed}.npz")]
        # other seeds may legitimately coexist; only flag malformed names
        bad = [n for n in extra if not (n.endswith(".mp4") or n.endswith(".npz"))]
        if bad:
            probs.append(f"{leaf}: unexpected files {bad}")

    video = {"checked": 0, "bad": []}
    if check_video:
        def _probe(p):
            st = ffprobe(p)
            ok = (st["codec_name"] == "h264" and st["pix_fmt"] == "yuv420p" and int(st["width"]) == 640
                  and st["r_frame_rate"] in ("15/1", "15") and st["size"] <= MAX_MP4_BYTES)
            return p, ok, st
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            for p, ok, st in ex.map(_probe, [p for p in mp4_paths if p.is_file()]):
                video["checked"] += 1
                if not ok:
                    video["bad"].append({"path": str(p), **st})
        if video["bad"]:
            probs.append(f"{len(video['bad'])} mp4 fail the profile/size check")

    rates_npz, rates_json = _rates(from_npz), _rates(from_json)
    if rates_npz != rates_json:
        probs.append(f"rates from npz {rates_npz} != results.json {rates_json}")
    a_b_nonnan = sum(1 for p in npz_paths if p.is_file() and not math.isnan(float(np.load(p, allow_pickle=False)["a_b_mean"])))
    a_b_enriched = sum(1 for r in sweep.results if r["evaluation"].get("a_b_mean") is not None)
    if a_b_nonnan != a_b_enriched:
        probs.append(f"a_b_mean non-NaN {a_b_nonnan} != enriched non-None {a_b_enriched}")
    human_diff = sum(1 for a, b in zip(from_npz, from_json) if a["success"] != a["success_obs"])
    return {
        "ok": not probs,
        "problems": probs[:50],
        "n_problems": len(probs),
        "n_npz": sum(1 for p in npz_paths if p.is_file()),
        "n_mp4": sum(1 for p in mp4_paths if p.is_file()),
        "rates": rates_npz,
        "n_success_rule_disagreements": human_diff,
        "a_b_mean_non_nan": a_b_nonnan,
        "video": {"checked": video["checked"], "n_bad": len(video["bad"]), "bad": video["bad"][:20]},
    }


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def export_sweep(sweep: Sweep, out_root: Path, ssi, ssi_path: Path, jobs: int,
                 skip_video: bool, dry_run: bool, force: bool = False) -> dict:
    out_results = out_root / "results"
    log(f"{sweep.id}: model={sweep.model} ({sweep.model_source}) seed={sweep.seed} replan={sweep.replan} "
        f"mode={sweep.instruction_mode} n={len(sweep.results)}")
    # never let a different sweep silently overwrite an already-exported (model, seed) slot
    for mf in sorted((out_root / "manifests").glob("*.json")):
        try:
            prev = json.loads(mf.read_text())
        except Exception:  # noqa: BLE001
            continue
        if prev.get("viewer", {}).get("model") == sweep.model and prev.get("sweep", {}).get("seed") == sweep.seed \
                and prev.get("sweep", {}).get("id") != sweep.id:
            if not force:
                raise SystemExit(f"viewer slot ({sweep.model}, seed {sweep.seed}) already exported from "
                                 f"{prev['sweep']['id']} — pass --force to replace it, or use another --model-name")
            log(f"  --force: replacing slot previously exported from {prev['sweep']['id']}")

    # canonical boundary primitives for the whole list (mutates evaluation dicts)
    ssi.enrich_with_boundary_stats(sweep.results, str(sweep.path))
    n_enriched = sum(1 for r in sweep.results if r["evaluation"].get("a_b_mean") is not None)
    log(f"  enrich_with_boundary_stats: a_b_mean/jerk_b_mean populated for {n_enriched}/{len(sweep.results)}")

    if dry_run:
        for entry in sweep.results[:3]:
            ti, ev = entry["task_info"], entry["evaluation"]
            leaf = sweep.leaf(out_results, ti)
            payload = build_npz_payload(sweep, entry, ssi)
            print(f"  {sweep.src_mp4(ev)}\n    -> {leaf / f'run_seed{sweep.seed}.mp4'}")
            print(f"  {sweep.src_dump(ti)}\n    -> {leaf / f'tracked_seed{sweep.seed}.npz'}")
            print("    " + ", ".join(f"{k}={payload[k] if payload[k].ndim == 0 else payload[k].shape}"
                                     for k in ("success", "obstacle_name", "tier", "min_clearance_m", "v_b",
                                               "a_b_mean", "jerk_b_mean", "contact_force", "min_obs_dist")))
        missing = [str(sweep.src_mp4(r["evaluation"])) for r in sweep.results if not sweep.src_mp4(r["evaluation"]).is_file()]
        log(f"  dry-run: {len(sweep.results)} leaves, {len(missing)} source mp4 missing")
        return {"dry_run": True, "missing_mp4": missing[:10]}

    # 1) npz for every episode
    t0 = time.time()
    for i, entry in enumerate(sweep.results):
        ti = entry["task_info"]
        leaf = sweep.leaf(out_results, ti)
        leaf.mkdir(parents=True, exist_ok=True)
        payload = build_npz_payload(sweep, entry, ssi)
        tmp = leaf / f".tracked_seed{sweep.seed}.part.npz"  # savez appends .npz unless present
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, leaf / f"tracked_seed{sweep.seed}.npz")
        if (i + 1) % 250 == 0:
            log(f"  npz {i + 1}/{len(sweep.results)}")
    log(f"  npz done in {time.time() - t0:.1f}s")

    # 2) videos
    enc = {"encoded": 0, "skipped": 0, "seconds": 0.0, "passes": {}}
    if not skip_video:
        t0 = time.time()
        jobs_list = []
        for entry in sweep.results:
            ti, ev = entry["task_info"], entry["evaluation"]
            src = sweep.src_mp4(ev)
            if not src.is_file():
                raise SystemExit(f"source video missing: {src}")
            jobs_list.append((src, sweep.leaf(out_results, ti) / f"run_seed{sweep.seed}.mp4"))
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(encode_video, s, d) for s, d in jobs_list]
            for n, fut in enumerate(as_completed(futs), 1):
                _, secs, label = fut.result()
                if secs == 0.0:
                    enc["skipped"] += 1
                else:
                    enc["encoded"] += 1
                    enc["seconds"] += secs
                    enc["passes"][label] = enc["passes"].get(label, 0) + 1
                if n % 250 == 0:
                    log(f"  video {n}/{len(jobs_list)} (encoded {enc['encoded']}, skipped {enc['skipped']}, "
                        f"{time.time() - t0:.0f}s)")
        log(f"  video done: encoded {enc['encoded']} skipped {enc['skipped']} passes {enc['passes']} in {time.time() - t0:.0f}s")

    # 3) verify + manifest
    ver = verify_sweep(sweep, out_results, ssi, jobs, check_video=not skip_video)
    manifest = {
        "schema_version": 1,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "sweep": {"path": str(sweep.path), "id": sweep.id,
                  "results_json_sha256": sha256(sweep.results_path),
                  "manifest_json_sha256": sha256(sweep.path / "manifest.json"),
                  "model_profile": sweep.profile, "seed": sweep.seed, "replan_steps": sweep.replan},
        "viewer": {"category": CATEGORY, "model": sweep.model, "model_name_source": sweep.model_source,
                   "instruction_mode": sweep.instruction_mode, "seed_file_suffix": f"seed{sweep.seed}",
                   "layout_naming": "robocasa LayoutType code id", "obstacle_naming": "task class name",
                   "out_results": str(out_results)},
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve()), **git_head()},
        "ssi_module": {"path": str(ssi_path), "sha256": sha256(ssi_path), "DT": ssi.DT,
                       "n_obstacles": len(ssi.TIER_R_B)},
        "ffmpeg": subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0],
        "ffmpeg_args": FFMPEG_ARGS, "numpy": np.__version__,
        "encode": enc, "n_enriched": n_enriched,
        "verification": ver,
    }
    man_dir = out_root / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)
    (man_dir / f"{sweep.id}.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    status = "OK" if ver["ok"] else f"FAIL ({ver['n_problems']} problems)"
    log(f"  verify: {status}  npz {ver['n_npz']} mp4 {ver['n_mp4']}  "
        f"SR_SD {ver['rates']['SR_SD']:.3f} SR_SA {ver['rates']['SR_SA']:.3f} SR_all {ver['rates']['SR_all']:.3f} "
        f"SSR_SD {ver['rates']['SSR_SD']:.3f} | obstacle-rule SR_SD {ver['rates']['SR_SD_obstacle_rule']:.3f} "
        f"(rule disagreements {ver['n_success_rule_disagreements']}) | video bad {ver['video']['n_bad']}")
    for p in ver["problems"][:10]:
        log(f"    ! {p}")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", action="append", required=True, type=Path, help="sweep dir (repeatable)")
    ap.add_argument("--out-root", type=Path, default=Path(DEFAULT_OUT))
    ap.add_argument("--jobs", type=int, default=4, help="parallel ffmpeg processes (keep small on the shared server)")
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true", help="only re-verify an existing export")
    ap.add_argument("--model-name", default=None,
                    help="viewer model folder for a NEW experiment (non-baseline instruction_mode, new profile, "
                         "post-hoc condition). Requires exactly one --sweep. Baseline panel names are refused.")
    ap.add_argument("--expect-n", type=int, default=EXPECTED_N, help="required number of results entries (1250 for v2)")
    ap.add_argument("--force", action="store_true", help="replace a (model, seed) slot exported from another sweep")
    args = ap.parse_args()
    if args.model_name is not None and len(args.sweep) != 1:
        raise SystemExit("--model-name applies to exactly one --sweep (run once per sweep)")
    if args.jobs > 8:
        raise SystemExit("--jobs > 8 refused: ffmpeg load has broken other users' EGL init on this host")

    ssi, ssi_path = load_v2_ssi()
    log(f"v2 ssi module: {ssi_path} ({len(ssi.TIER_R_B)} obstacles, DT={ssi.DT})")
    sweeps = [Sweep(p, model_name=args.model_name, expect_n=args.expect_n) for p in args.sweep]
    seen = set()
    for s in sweeps:
        key = (s.model, s.seed)
        if key in seen:
            raise SystemExit(f"two sweeps map to the same viewer slot {key}")
        seen.add(key)

    failed = 0
    for s in sweeps:
        if args.verify_only:
            ssi.enrich_with_boundary_stats(s.results, str(s.path))
            ver = verify_sweep(s, args.out_root / "results", ssi, args.jobs, check_video=not args.skip_video)
            log(f"{s.id}: verify {'OK' if ver['ok'] else 'FAIL'} npz {ver['n_npz']} mp4 {ver['n_mp4']} rates {ver['rates']}")
            for p in ver["problems"][:10]:
                log(f"    ! {p}")
            failed += 0 if ver["ok"] else 1
            continue
        man = export_sweep(s, args.out_root, ssi, ssi_path, args.jobs, args.skip_video, args.dry_run, args.force)
        if not args.dry_run and not man["verification"]["ok"]:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
