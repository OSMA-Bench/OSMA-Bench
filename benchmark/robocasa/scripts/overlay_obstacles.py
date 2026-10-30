"""
Overlay the per-obstacle navigation videos that share an identical
(blocking-mode, route, layout, style) into a single translucent composite.

test_video/ is laid out as one subdir per obstacle:

    test_video/<obstacle>/NavigateKitchen<Obstacle><Mode>Route<R>_<LAYOUT>_<STYLE>.mp4

For a fixed scene (layout+route+mode+style) only the obstacle differs and the
camera/robot are identical, so combining the frames produces one clip showing
where every obstacle type lands in the same kitchen. Two combine modes:

  --mode mean (default): equal-weight average (ffmpeg `mix`); each obstacle
      is a faint 1/N ghost.
  --mode max: per-pixel maximum (chained `blend=lighten`); the shared
      background stays at normal brightness and every obstacle stays sharp.
      A literal sum is intentionally not offered: summing N near-identical
      bright frames clips to white and destroys the image.

Only dependency is ffmpeg. No numpy / OpenCV / imageio needed.

Examples:
    # Ghost-average every group, output to test_video_overlay/
    python overlay_obstacles.py --root test_video

    # Colour-coded difference (clearest): each obstacle a distinct hue
    python overlay_obstacles.py --root test_video --mode diff

    # Only L_SHAPED_LARGE RouteA, blocking, and print the commands only
    python overlay_obstacles.py --root test_video --filter BlockingRouteA_L_SHAPED_LARGE --dry-run

    # Restrict to a subset of obstacles, 4 parallel encodes
    python overlay_obstacles.py --root test_video --obstacles dog cat vase --jobs 4
"""

import argparse
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# A scene clip is NavigateKitchen<Obstacle><Mode>Route<R>_<LAYOUT>_<STYLE>.
# Capture everything from the mode onward as the group key; the obstacle
# token (any casing, any word-joining) is consumed by the non-greedy .+?,
# so the obstacle is identified by its SUBDIR name, never re-derived from
# the filename (which previously dropped crawlingbaby/glassofwater/
# hotchocolate because the subdir lacked the filename's internal capitals).
GROUP_KEY_RE = re.compile(r"^NavigateKitchen.+?((?:Non)?BlockingRoute[A-G]_.+)$")

# Distinct hues for --mode diff, one per obstacle (assigned in sorted order
# so the obstacle->colour mapping is stable across every group).
DIFF_PALETTE = [
    (1.0, 0.0, 0.0),   # red
    (0.0, 1.0, 0.0),   # green
    (0.3, 0.5, 1.0),   # blue
    (1.0, 1.0, 0.0),   # yellow
    (1.0, 0.0, 1.0),   # magenta
    (0.0, 1.0, 1.0),   # cyan
    (1.0, 0.55, 0.0),  # orange
    (0.7, 0.0, 1.0),   # purple
    (1.0, 1.0, 1.0),   # white
    (0.6, 1.0, 0.0),   # lime
]

def discover_groups(root, only_obstacles=None, key_filter=None):
    """
    Returns {group_key: {obstacle: Path}} where obstacle is the subdir name
    and group_key is <Mode>Route<R>_<LAYOUT>_<STYLE> parsed from the
    filename (independent of how the subdir is spelled).
    """
    groups = defaultdict(dict)
    if not root.is_dir():
        raise SystemExit(f"[error] root not found: {root}")

    subdirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name != "overlay")
    for sub in subdirs:
        obstacle = sub.name
        if only_obstacles and obstacle not in only_obstacles:
            continue
        for mp4 in sorted(sub.glob("*.mp4")):
            m = GROUP_KEY_RE.match(mp4.stem)
            if not m:
                continue  # not a NavigateKitchen scene clip
            group_key = m.group(1)
            if key_filter and key_filter not in group_key:
                continue
            groups[group_key][obstacle] = mp4
    return groups


def probe_size(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    w, h = out.stdout.strip().split(",")
    return int(w), int(h)


def build_cmd(inputs, out_path, width, height, mode="mean"):
    """ffmpeg command: scale every input to a common size then combine.

    mode="mean" (default): mix auto-normalizes by 1/sum(weights) -> the
        equal-weight mean; every obstacle shows as a faint 1/N ghost.
    mode="max": per-pixel maximum across all inputs (chained
        blend=lighten; max is associative so the pairwise chain is the true
        N-way max). Shared background at normal brightness, each obstacle
        sharp wherever it is the brightest layer.
    mode="diff": background-subtracted, colour-coded difference. The mean
        of all inputs approximates the static kitchen (each obstacle sits
        elsewhere, so it averages out). For each input i,
        |input_i - mean| isolates obstacle i's changed pixels; that
        magnitude is tinted with a distinct palette colour, all tints are
        screen-combined, then laid over a dimmed copy of the kitchen.
        Result: every obstacle is an explicitly different colour, so the
        difference between the overlaid scenes is unmistakable.
    """
    n = len(inputs)
    cmd = ["ffmpeg", "-y"]
    for p in inputs:
        cmd += ["-i", str(p)]
    chains = []

    if mode == "grid":
        # Small-multiples contact sheet: one panel per obstacle of the same
        # scene, near-square grid. The only unambiguous way to compare
        # obstacles that occupy the same spot (an overlay would collapse
        # them). Panel order = sorted(obstacle), printed by render_one.
        import math
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        # cw, ch = width // cols, height // rows
        cw, ch = width, height  # no resizing; just show the center of each frame
        for i in range(n):
            chains.append(
                f"[{i}:v]scale={cw}:{ch}:force_original_aspect_ratio=decrease,"
                f"pad={cw}:{ch}:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"setsar=1,format=rgb24[t{i}]"
            )
        layout = "|".join(
            f"{(k % cols) * cw}_{(k // cols) * ch}" for k in range(n)
        )
        labels = "".join(f"[t{i}]" for i in range(n))
        # xstack the per-obstacle panels, then pad to even dimensions and
        # convert to yuv420p so libx264 accepts the (often large) canvas.
        # Unlike diff/the old grid still, this keeps the full clip duration
        # so the result is a small-multiples *video*, one panel per obstacle.
        chains.append(
            f"{labels}xstack=inputs={n}:layout={layout}:fill=black,"
            f"pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p[out]"
        )
        cmd += [
            "-filter_complex", ";".join(chains),
            "-map", "[out]",
            "-shortest",                 # stop at the shortest input
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            "-an",
            str(out_path),
        ]
        return cmd

    if mode == "diff" and n == 1:
        # degenerate: nothing to diff against, just emit the frame
        chains.append(f"[0:v]scale={width}:{height},setsar=1,format=rgb24[out]")
        cmd += ["-filter_complex", ";".join(chains), "-map", "[out]",
                "-frames:v", "1", "-update", "1", str(out_path)]
        return cmd

    if mode == "diff" and n >= 2:
        # one copy of each input for the clean plate, one for its own diff
        for i in range(n):
            chains.append(
                f"[{i}:v]scale={width}:{height},setsar=1,format=gbrp,"
                f"split=2[m{i}][d{i}]"
            )
        # Clean background = per-pixel MEDIAN across all inputs. At any pixel
        # only one clip has its obstacle there, the rest show bare floor, so
        # the median is the obstacle-free kitchen. (A mean would instead
        # carry a 1/N ghost of every obstacle, which then bleeds every
        # palette colour into every blob and washes the result to white.)
        chains.append(
            "".join(f"[m{i}]" for i in range(n)) + f"xmedian=inputs={n}[bg]"
        )
        # background reused once per input + once for the final backdrop
        chains.append(f"[bg]split={n + 1}" + "".join(f"[g{k}]" for k in range(n + 1)))
        # |input_i - mean| has a strong, localised spike at obstacle i and a
        # weak broad floor from render noise (antialiasing/lighting differs
        # slightly everywhere). A plain linear gain amplifies that floor too,
        # and screen-blending the result across N layers sums to white/
        # yellow. Instead threshold with colorlevels: rimin clips the noise
        # floor to black, rimax boosts what survives -> each layer is black
        # except its obstacle, which becomes the pure palette colour.
        NOISE_FLOOR = 0.09   # difference below this (codec/AA noise) -> black
        PEAK = 0.22          # difference at/above this -> full palette colour
        for i in range(n):
            r, g, b = DIFF_PALETTE[i % len(DIFF_PALETTE)]
            chains.append(
                f"[d{i}][g{i}]blend=all_mode=difference,format=gray,format=gbrp,"
                f"colorlevels="
                f"rimin={NOISE_FLOOR}:rimax={PEAK}:"
                f"gimin={NOISE_FLOOR}:gimax={PEAK}:"
                f"bimin={NOISE_FLOOR}:bimax={PEAK},"
                f"colorchannelmixer="
                f"rr={r}:rg=0:rb=0:"
                f"gr={g}:gg=0:gb=0:"
                f"br={b}:bg=0:bb=0[c{i}]"
            )
        # Combine the colour layers with `lighten` (per-pixel max): the
        # obstacles sit at different places so each keeps its own pure hue;
        # where two coincide, the brighter wins (no colour-summing wash).
        prev = "c0"
        for i in range(1, n):
            lbl = f"s{i}"
            chains.append(f"[{prev}][c{i}]blend=all_mode=lighten[{lbl}]")
            prev = lbl
        chains.append(f"[g{n}]colorchannelmixer=rr=.28:gg=.28:bb=.28[bgd]")
        # lighten over the dimmed kitchen keeps the hues pure (screen would
        # tint them with the background).
        chains.append(f"[bgd][{prev}]blend=all_mode=lighten,format=rgb24[out]")
        filtergraph = ";".join(chains)
        # The clips are near-static (zero-action policy), so one
        # representative frame fully conveys the per-obstacle difference.
        # `-frames:v 1` makes ffmpeg decode/process only the first frame —
        # ~16x faster than running the heavy N-input graph over all frames
        # (which timed out). Output is a single PNG.
        cmd += [
            "-filter_complex", filtergraph, "-map", "[out]",
            "-frames:v", "1", "-update", "1", str(out_path),
        ]
        return cmd

    for i in range(n):
        chains.append(f"[{i}:v]scale={width}:{height},setsar=1,format=yuv420p[v{i}]")

    if mode == "max":
        if n == 1:
            chains.append("[v0]null[out]")
        else:
            prev = "v0"
            for i in range(1, n):
                lbl = "out" if i == n - 1 else f"b{i}"
                chains.append(f"[{prev}][v{i}]blend=all_mode=lighten[{lbl}]")
                prev = lbl
    else:  # mean (also the n==1 fallback for diff)
        labels = "".join(f"[v{i}]" for i in range(n))
        weights = " ".join("1" for _ in range(n))
        chains.append(f"{labels}mix=inputs={n}:weights={weights}[out]")
    filtergraph = ";".join(chains)
    cmd += [
        "-filter_complex", filtergraph,
        "-map", "[out]",
        "-shortest",                 # stop at the shortest input (person has fewer)
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-an",
        str(out_path),
    ]
    return cmd


def render_one(group_key, members, out_dir, dry_run, mode="mean"):
    order = sorted(members)
    inputs = [members[o] for o in order]
    # diff emits a single still; mean/max/grid emit clips.
    ext = "png" if mode == "diff" else "mp4"
    out_path = out_dir / f"overlay_{group_key}.{ext}"
    width, height = probe_size(inputs[0])
    cmd = build_cmd(inputs, out_path, width, height, mode)
    if dry_run:
        return group_key, "dry-run", " ".join(cmd)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        tail = res.stderr.strip().splitlines()[-1:] or [""]
        return group_key, "FAILED", tail[0]
    extra = f" [panels: {', '.join(order)}]" if mode == "grid" else ""
    return group_key, "ok", f"{len(inputs)} obstacles -> {out_path.name}{extra}"


def main():
    ap = argparse.ArgumentParser(description="Overlay per-obstacle videos sharing a scene.")
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).parent / "../test_video",
                    help="dir containing one subdir per obstacle (default: ./test_video)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default: <root>_overlay, a writable sibling "
                         "of root since the recording dir is often read-only)")
    ap.add_argument("--obstacles", nargs="+", default=None,
                    help="restrict to these obstacle subdirs (default: all)")
    ap.add_argument("--filter", dest="key_filter", default=None,
                    help="only group keys containing this substring "
                         "(e.g. BlockingRouteA_L_SHAPED_LARGE)")
    ap.add_argument("--min-obstacles", type=int, default=2,
                    help="skip groups with fewer than this many obstacles (default: 2)")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1),
                    help="parallel ffmpeg encodes (default: min(8, ncpu))")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the ffmpeg commands instead of running them")
    ap.add_argument("--mode", choices=["mean", "max", "diff", "grid"],
                    default="mean",
                    help="mean: 1/N ghost average clip (default). "
                         "max: per-pixel max (lighten) clip, bg normal. "
                         "diff: PNG, xmedian-subtracted, each obstacle a "
                         "distinct colour (good only when obstacles differ "
                         "in location). "
                         "grid: mp4 small-multiples video, one panel per "
                         "obstacle of the same scene (clearest when obstacles "
                         "share a location).")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("[error] ffmpeg/ffprobe not found on PATH")

    out_dir = args.out or args.root.parent / (args.root.name + "_overlay")
    groups = discover_groups(args.root, set(args.obstacles) if args.obstacles else None,
                             args.key_filter)

    todo = {k: v for k, v in groups.items() if len(v) >= args.min_obstacles}
    skipped = sorted(set(groups) - set(todo))
    if not todo:
        raise SystemExit(f"[error] no groups with >= {args.min_obstacles} obstacles "
                          f"(found {len(groups)} groups total)")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] {len(todo)} groups to overlay, {len(skipped)} skipped "
          f"(< {args.min_obstacles} obstacles), out -> {out_dir}")

    ok = fail = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(render_one, k, v, out_dir, args.dry_run, args.mode): k
                for k, v in sorted(todo.items())}
        for fut in concurrent.futures.as_completed(futs):
            key, status, detail = fut.result()
            if status == "FAILED":
                fail += 1
                print(f"  [FAIL] {key}: {detail}")
            elif status == "dry-run":
                print(f"  {key}:\n    {detail}")
            else:
                ok += 1
                print(f"  [ok] {key}: {detail}")

    print(f"\n[done] ok={ok} failed={fail} skipped={len(skipped)}")
    if skipped:
        print("[info] skipped (single obstacle / filtered): "
              + ", ".join(skipped[:10]) + (" ..." if len(skipped) > 10 else ""))
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
