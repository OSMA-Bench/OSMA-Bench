#!/bin/bash

cd ../
# # ONE_WALL_SMALL / top view
# CUDA_VISIBLE_DEVICES=0,1 python run_env_no_teleop.py --env navigate_safe --layout ONE_WALL_SMALL --filter_env_keyword=BlockingRouteA --capture_initial_stage --action_mode='turn_left' --record_path="./figure/one_wall_small/top_view" --camera_view='routed_one_wall_small_topview' --horizon 50 --gpu_id 0
CUDA_VISIBLE_DEVICES=2,3 python run_env_no_teleop.py --env navigate_safe --layout ONE_WALL_SMALL_NO_WALL --filter_env_keyword=BlockingRouteA --capture_initial_stage --action_mode='turn_left' --record_path="./figure/one_wall_small/view_1024" --camera_view='routed_one_wall_small' --horizon 50 --gpu_id 3 --camera_shape 1024 1024
# # ONE_WALL_LARGE / top view
# python run_env_no_teleop_parallel.py --num_workers 16 --gpu_ids 0 1 2 3 --env navigate_safe --layout ONE_WALL_LARGE --filter_env_keyword=BlockingRouteF --capture_initial_stage --record_path="./figure/one_wall_large/top_view" --camera_view='routed_one_wall_large_topview'

# # ONE_WALL_LARGE / routed view
# python run_env_no_teleop_parallel.py --num_workers 16 --gpu_ids 0 1 2 3 --env navigate_safe --layout ONE_WALL_LARGE --filter_env_keyword=BlockingRouteF --capture_initial_stage --record_path="./figure/one_wall_large/routed_view" --camera_view='routed_one_wall_large'

# python run_env_no_teleop_parallel.py --num_workers 16 --gpu_ids 0 1 2 3 --env navigate_safe --layout U_SHAPED_LARGE --filter_env_keyword=BlockingRouteF --capture_initial_stage --record_path="./figure/u_large_routeF/view" --camera_view='routed_u_larged'

# python run_env_no_teleop_parallel.py --num_workers 16 --gpu_ids 0 1 2 3 --env navigate_safe --layout U_SHAPED_LARGE --filter_env_keyword=BlockingRouteF --capture_initial_stage --record_path="./figure/u_large_routeF/view" --camera_view='routed_u_larged_topview'
#### routeA
# U_SHAPED_LARGE / routed view
# python run_env_no_teleop_parallel.py --num_workers 16 --gpu_ids 0 1 2 3 --env navigate_safe --layout U_SHAPED_LARGE --filter_env_keyword=BlockingRouteA --capture_initial_stage --record_path="./figure/u_large_routeA/view" --camera_view='routed_u_larged_routeA'

# # U_SHAPED_LARGE / top view
# python run_env_no_teleop_parallel.py --num_workers 16 --gpu_ids 0 1 2 3 --env navigate_safe --layout U_SHAPED_LARGE --filter_env_keyword=BlockingRouteA --capture_initial_stage --record_path="./figure/u_large_routeA/top_view" --camera_view='routed_u_larged_topview'


# python run_env_no_teleop_parallel.py --num_workers 16 --gpu_ids 0 1 2 3 --env navigate_safe --layout ONE_WALL_SMALL --filter_env_keyword=BlockingRouteA --record_path="./figure/one_wall_small/view" --capture_initial_stage --camera_view='routed_one_wall_small' 

# U_SHAPED_LARGE / top view
# python run_env_no_teleop_parallel.py --num_workers 16 --gpu_ids 0 1 2 3 --env navigate_safe --layout ONE_WALL_SMALL --filter_env_keyword=BlockingRouteA --record_path="./figure/one_wall_small/top_view" --capture_initial_stage --camera_view='routed_one_wall_small_topview'



CUDA_VISIBLE_DEVICES=2,3 python run_env_no_teleop.py --env navigate_safe --layout U_SHAPED_SMALL --filter_env_keyword=BlockingRouteA --capture_initial_stage --record_path="./figure/u_shaped_small/view_1024" --camera_view='topview' --horizon 50 --gpu_id 3 --camera_shape 1024 1024