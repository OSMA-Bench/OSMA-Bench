#!/usr/bin/env bash
# Sweep every navigation obstacle across a set of layouts, then build the
# group-wise (per caution tier) overlay comparisons.
#
#   scripts/nav_sweep.sh [LAYOUT ...]
#
# One clip per registered navigate_safe class per layout. The roster is 18
# obstacles x 7 routes x 2 blocking modes minus the two human-on-RouteF
# combinations (the posed_human is RouteF's target and cannot also be its
# obstacle) = 250 classes per layout. Over the five layouts the env supports
# that is the full 250 x 5 = 1250-task benchmark.
#
# Output tree (under figures/nav_sweep):
#   videos/<LAYOUT>/<obstacle>/NavigateKitchen<Obs><Mode>Route<R>_<LAYOUT>_<STYLE>.mp4
#   overlay/<LAYOUT>/<tier>/{diff,grid}/...
#   stability/validation_<LAYOUT>.csv
set -euo pipefail

cd "$(dirname "$0")/.."

LAYOUTS=("$@")
if [ ${#LAYOUTS[@]} -eq 0 ]; then
    # The evaluation set is enforced by the env itself (_SUPPORTED_LAYOUTS in
    # kitchen_navigate_safe.py), which rejects anything outside it. Read the
    # list from there rather than repeating it: a hardcoded copy silently went
    # stale once already, and every task of the dropped layout then failed
    # 250 times before the sweep aborted on the empty overlay.
    read -ra LAYOUTS <<<"$(python -c '
import contextlib, sys
with contextlib.redirect_stdout(sys.stderr):
    from robocasa.environments.kitchen.single_stage.kitchen_navigate_safe import _SUPPORTED_LAYOUTS
    from robocasa.models.scenes.scene_registry import LayoutType
print(" ".join(LayoutType(i).name for i in _SUPPORTED_LAYOUTS))')"
    echo "[layouts] ${LAYOUTS[*]}"
fi

OUT=figures/nav_sweep
HORIZON=100          # 100 frames at fps 20 -> 5 s of video
WORKERS=${WORKERS:-16}
# Intent: use the whole GPU pool. The runner narrows this to the devices that
# can actually create an EGL context, so a host whose EGL enumeration order
# disagrees with the CUDA ordinal still works without a hardcoded id here.
GPUS=${GPUS:-0 1 2 3}

# Resolve once for the whole sweep. Without this every per-layout invocation
# re-probes, and each probe costs a subprocess import of mujoco.
if [ -z "${ROBOCASA_EGL_DEVICE:-}" ]; then
    ROBOCASA_EGL_DEVICE=$(python scripts/resolve_egl.py $GPUS)
    if [ -n "$ROBOCASA_EGL_DEVICE" ]; then
        export ROBOCASA_EGL_DEVICE
        echo "[egl] using device(s): $ROBOCASA_EGL_DEVICE"
    else
        echo "[egl] no working EGL device found; rendering will fail" >&2
        exit 1
    fi
fi

# Caution tiers, 6 obstacles each. Mirrors HIGH_TIER_OBSTACLES /
# MODERATE_TIER_OBSTACLES / LOW_TIER_OBSTACLES in kitchen_navigate_safe.py
# (and TIER_OF in robocasa/utils/ssi.py). Keep all three in sync: the tiers
# are deliberately equal-sized so a per-tier mean is taken over the same
# number of obstacle types.
HIGH=(human child_boy child_girl crawling_baby cat dog)
MODERATE=(wine glass_of_water hot_chocolate vase flower_pot table_lamp)
LOW=(trashbin delivery_box cardboard_box wooden_crate floor_cushion duffel_bag)

mkdir -p "$OUT"/{videos,overlay,stability}

for L in "${LAYOUTS[@]}"; do
    echo "=============== $L ==============="
    RAW="$OUT/_raw/$L"
    mkdir -p "$RAW"

    MUJOCO_GL=egl python run_env_no_teleop_parallel.py \
        --env navigate_safe --layout "$L" \
        --record_path "$RAW" --horizon "$HORIZON" \
        --num_workers "$WORKERS" --gpu_ids $ROBOCASA_EGL_DEVICE --skip-existing 
        # --filter_env_keyword "RouteE" 

    n_clips=$(ls "$RAW"/*.mp4 2>/dev/null | wc -l)
    echo "[$L] recorded $n_clips clips"
    # An unsupported or broken layout records nothing; overlaying it would abort
    # the whole sweep (set -e) and lose the layouts still queued behind it.
    if [ "$n_clips" -eq 0 ]; then
        echo "[$L] no clips recorded -- skipping overlays for this layout" >&2
        continue
    fi

    # fan the flat recording dir out into one subdir per obstacle, which is
    # the layout overlay_obstacles.py discovers
    python scripts/nav_sweep_sort.py "$RAW" "$OUT/videos/$L"
    [ -f "$RAW/validation_report.csv" ] && \
        cp "$RAW/validation_report.csv" "$OUT/stability/validation_$L.csv"

    for tier in high moderate low; do
        case $tier in
            high)     members=("${HIGH[@]}") ;;
            moderate) members=("${MODERATE[@]}") ;;
            low)      members=("${LOW[@]}") ;;
        esac
        for mode in diff grid; do
            python scripts/overlay_obstacles.py \
                --root "$OUT/videos/$L" \
                --out  "$OUT/overlay/$L/$tier/$mode" \
                --obstacles "${members[@]}" \
                --mode "$mode" --jobs 8 >/dev/null
        done
        echo "[$L] $tier tier -> $(ls "$OUT/overlay/$L/$tier/diff" 2>/dev/null | wc -l) diff images"
    done
done

echo "sweep complete: $(find "$OUT/videos" -name '*.mp4' | wc -l) clips"

# Leave the browsable page next to the assets it points at, so inspecting a
# sweep never depends on remembering a second command.
# A page that fails to build must not report the whole sweep as failed: the
# 1250 clips are already on disk and the page is one command away.
scripts/nav_sweep_html.sh "$OUT" ||
    echo "[warn] report build failed; rerun scripts/nav_sweep_html.sh" >&2
