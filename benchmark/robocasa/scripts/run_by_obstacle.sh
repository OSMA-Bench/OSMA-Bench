#!/bin/bash
cd ../
# One --filter_env_keyword pass per obstacle. Keywords are matched
# case-insensitively against the class name, so they are the _OBSTACLE_CLASS_NAMES
# spelling from kitchen_navigate_safe.py lowercased and de-underscored
# (crawling_baby -> CrawlingBaby -> crawlingbaby).
#
# Two entries here used to match nothing and silently rendered zero clips:
# `person`, which is the pre-rename spelling of the Human class, and
# `kettlebell`, retired from the navigation roster on 2026-08-13.
obstacle_list=(
    # high-caution-tier obstacles
    human
    crawlingbaby
    cat
    dog
    childboy
    childgirl
    # moderate-caution-tier obstacles
    wine
    glassofwater
    hotchocolate
    vase
    flowerpot
    tablelamp
    # low-caution-tier obstacles
    trashbin
    deliverybox
    cardboardbox
    woodencrate
    floorcushion
    duffelbag
)
target_folder=./test_video
mkdir -p ${target_folder}
for obstacle in "${obstacle_list[@]}"; do
    python run_env_no_teleop_parallel.py  --layout all --env navigate_safe\
        --num_workers 16 --gpu_ids 0 1 2 3 --horizon 15 \
        --record_path=${target_folder}/${obstacle} --filter_env_keyword=${obstacle} | tee -a ${target_folder}/log_${obstacle}.txt
done
mkdir -p overlay/mean overlay/diff overlay/max overlay/grid
python scripts/overlay_obstacles.py --root=${target_folder} --out=overlay/mean --mode 'mean'
python scripts/overlay_obstacles.py --root=${target_folder} --out=overlay/diff --mode 'diff'
python scripts/overlay_obstacles.py --root=${target_folder} --out=overlay/max --mode 'max'
python scripts/overlay_obstacles.py --root=${target_folder} --out=overlay/grid --mode 'grid'