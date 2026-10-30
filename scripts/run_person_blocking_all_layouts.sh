#!/bin/bash
# Run PersonBlocking Route A~G across all 9 layouts (excluding EMPTY_ROOM)
# Layout IDs: 0=ONE_WALL_SMALL, 1=ONE_WALL_LARGE, 2=L_SHAPED_SMALL, 3=L_SHAPED_LARGE,
#             4=GALLEY, 5=U_SHAPED_SMALL, 6=U_SHAPED_LARGE, 7=G_SHAPED_SMALL,
#             8=G_SHAPED_LARGE, 9=WRAPAROUND

LAYOUTS=(
    ONE_WALL_SMALL
    ONE_WALL_LARGE
    L_SHAPED_SMALL
    L_SHAPED_LARGE
    GALLEY
    U_SHAPED_SMALL
    U_SHAPED_LARGE
    G_SHAPED_SMALL
    G_SHAPED_LARGE
    WRAPAROUND
)

ROUTES=(A B C D E G)
MODEL="Qwen/Qwen3-4B-Instruct-2507"
PORT=8002

TASKS=""
for layout in "${LAYOUTS[@]}"; do
    for route in "${ROUTES[@]}"; do
        TASKS="$TASKS NavigateKitchenHumanBlockingRoute${route}_${layout}"
    done
done

echo "Running $(echo $TASKS | wc -w) tasks total"
echo "Layouts: ${LAYOUTS[*]}"
echo "Routes: ${ROUTES[*]}"

docker exec -it robotics-safety-container bash -c \
    "cd /workspace/policy/Voxposer && MUJOCO_GL=egl python3 src/run_LMP.py -m $MODEL -p $PORT $TASKS"
