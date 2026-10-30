import os
import numpy as np
import imageio
import argparse, random
from termcolor import colored
from tqdm import tqdm
import time
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import xml_path_completion
from task_listup import task_envs_list
from robocasa.utils.env_utils import hide_gripper_marker_sites
import logging
# (옵션) robosuite 키보드 텔레옵 유틸
_HAS_RS_KEYBOARD = True
try:
    from robosuite.devices import Keyboard
    from robosuite.utils.input_utils import input2action
except Exception:
    _HAS_RS_KEYBOARD = False

from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
from robocasa.models.scenes.scene_registry import LayoutType, StyleType, LAYOUT_GROUPS_TO_IDS, STYLE_GROUPS_TO_IDS

# ====== 환경 생성 ======
def create_own_env(
    env_name,
    robots="PandaOmron",
    camera_names=[
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
        "robot0_frontview",
    ],
    camera_widths=128,
    camera_heights=128,
    seed=None,
    render_onscreen=True,           # <<< 온스크린
    render_camera="robot0_frontview",
    obj_instance_split=None,
    generative_textures=None,
    randomize_cameras=False,
    layout_and_style_ids=None,
    layout_ids=None,
    style_ids=None,
    has_human=True,
):
    controller_config = load_composite_controller_config(
        controller=None,
        robot=robots if isinstance(robots, str) else robots[0],
    )

    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_config,
        # on-screen 렌더링
        has_renderer=render_onscreen,
        has_offscreen_renderer=not render_onscreen,
        render_camera=render_camera,
        # 관측치
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,        # <<< 온스크린 조작용이라 False 권장
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        camera_depths=False,
        # 기타
        seed=seed,
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        layout_ids=layout_ids,
        style_ids=style_ids,
        translucent_robot=False,
        render_gpu_device_id=0,
    )
    if 'Door' in env_name :
        env_kwargs['has_human'] = has_human
    env = robosuite.make(**env_kwargs)
    hide_gripper_marker_sites(env)
    return env

def run_keyboard_teleop(env, horizon=2000, record_path=None, args=None):
    """
    전역 키 리스너(pynput) 기반 텔레옵 (GLFW 핸들 불필요)
    PandaOmron (DoF 12) 액션 인덱스에 맞춘 매핑

    이동:
      ↑/↓ : action[7] ± lin  (앞/뒤)
      ←/→ : action[8] ± lin  (좌/우)
      PageUp : action[10] += lin (위)
      PageDown : action[11] += lin (아래)  # 11은 '아래로 가기' → 양수 가속

    회전(부호 양방향, 한 축으로 통일):
      1/2 : action[3] yaw  (-/+)
      3/4 : action[4] pitch(-/+)
      5/6 : action[5] roll (-/+)

    그리퍼:
      [ / ] : action[6] -/+  (닫기 / 열기)

    기타:
      Backspace : reset
      Esc       : quit
    """
    import numpy as np
    import imageio
    from termcolor import colored
    # 안내
    logging.info("[teleop] 간단 키 매핑 모드 시작 (전역 키 리스너)")
    logging.info("  이동(Translation): ←/→(좌/우), ↑/↓(전/후), b(위), n(아래)")
    logging.info("  회전(Rotation): 2/3(Yaw -/+)  |  4/5(Pitch -/+)  |  6/7(Roll -/+)")
    logging.info("  그리퍼 이동: 8(앞으로), -(왼쪽), =(오른쪽), g(위), h(아래)")
    logging.info("  그리퍼 개폐: ,(닫기)  .(열기)")
    logging.info("  기타: y(좌회전/Left Turn)")
    logging.info("  Backspace: 리셋, Esc: 종료")

    writer = imageio.get_writer(record_path, fps=20) if record_path else None

    # ===== 액션 인덱스 (테이블 기준 고정) =====
    IDX_FWD_BACK = 7     # 앞으로/뒤로
    IDX_LEFT_RIGHT = 8   # 왼/오
    IDX_LEFT_TURN=9      # 왼쪽으로 돌기t
    IDX_UP = 10          # 위로 / 앞뒤로
    # IDX_DOWN = 11        # 아래로 (양수 가속)
    IDX_YAW = 3          # 시계/반시계 (부호로 양방향)
    IDX_PITCH = 4        # 앞/뒤로 돌기
    IDX_ROLL = 5         # 왼/오로 돌기
    IDX_GRIP = 6         # 집기(그리퍼)
    IDX_GRIP_FRONT = 0   # 그리퍼 앞으로
    IDX_GRIP_LEFT = 1    # 그리퍼 왼쪽으로
    IDX_GRIP_UP = 2      # 그리퍼 위로

    # 감도
    lin = 1.0
    rot = 0.5
    grip = 1.0

    low, high = env.action_spec
    action = np.zeros_like(high)
    obs = env.reset()
    hide_gripper_marker_sites(env)
    # NOTE: do not call env.reset() here — robosuite.make() already runs
    # _load_model() + _reset_internal() once, so an extra reset would
    # re-sample obstacles/objects (and log "Obstacle ... at" twice).

    if writer is not None:
        frame = env.sim.render(height=1024, width=1536, camera_name=args.camera_view)[::-1]
        writer.append_data(frame)
    else:
        env.render()
    
    # 전역 키 리스너
    try:
        from pynput import keyboard
    except Exception as e:
        logging.error(colored(f"[teleop] pynput 불러오기 실패: {e}", "red"))
        logging.info("  macOS '입력 모니터링' 권한을 IDE/Python에 허용하세요.")
        return

    pressed = set()
    def on_press(key): pressed.add(key)
    def on_release(key):
        if key in pressed:
            pressed.remove(key)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    K = keyboard.Key
    KC = keyboard.KeyCode
    def down(*keys): return any(k in pressed for k in keys)

    t = 0
    success_flag = False
    try:
        while True:
            action[:] = 0.0

            # ===== Translation =====
            if down(K.up):    action[IDX_FWD_BACK] += lin    # 앞으로
            if down(K.down):  action[IDX_FWD_BACK] -= lin    # 뒤로
            if down(K.left):  action[IDX_LEFT_RIGHT] += lin  # 왼쪽
            if down(K.right): action[IDX_LEFT_RIGHT] -= lin  # 오른쪽
            if down(KC.from_char('y')): action[IDX_LEFT_TURN] += lin       # 왼쪽회전
            if down(KC.from_char('u')): action[IDX_LEFT_TURN] -= lin       # 오른쪽 회전

            if down(KC.from_char('b')):   action[IDX_UP] += lin      # 위
            # if down(KC.from_char('n')): action[IDX_DOWN] += lin    # 아래 (양수 가속)
            if down(KC.from_char('n')): action[IDX_UP] -= lin    # 아래 (양수 가속)
            if down(KC.from_char('g')): action[IDX_GRIP_UP] += lin     # 그리퍼 위로
            if down(KC.from_char('h')): action[IDX_GRIP_UP] -= lin       # 그리퍼 아래로

            # ===== Rotation =====
            # Yaw (1/2): 반시계 / 시계
            if down(KC.from_char('2')): action[IDX_YAW] -= rot
            if down(KC.from_char('3')): action[IDX_YAW] += rot

            # Pitch (3/4): 뒤 / 앞
            if down(KC.from_char('4')): action[IDX_PITCH] -= rot
            if down(KC.from_char('5')): action[IDX_PITCH] += rot

            # Roll (5/6): 왼 / 오
            if down(KC.from_char('6')): action[IDX_ROLL] -= rot
            if down(KC.from_char('7')): action[IDX_ROLL] += rot
            
            if down(KC.from_char('8')): action[IDX_GRIP_FRONT] += lin    # 그리퍼 앞으로
            if down(KC.from_char('-')): action[IDX_GRIP_LEFT] += lin    # 그리퍼 왼쪽으로
            if down(KC.from_char('=')): action[IDX_GRIP_LEFT] -= lin    # 그리퍼 오른쪽으로

            # ===== Gripper =====
            if down(KC.from_char(',')): action[IDX_GRIP] -= grip   # 닫기
            if down(KC.from_char('.')): action[IDX_GRIP] += grip   # 열기

            # ===== Reset / Quit =====
            if down(K.backspace):
                obs = env.reset()
                hide_gripper_marker_sites(env)
                
                if writer is not None:
                    frame = env.sim.render(height=1024, width=1536, camera_name=args.camera_view)[::-1]
                    writer.append_data(frame)
                else:
                    env.render()

                t += 1
                continue
            if down(K.ctrl_l) and down(KC.from_char('c')):
                break

            # step & render
            obs, reward, done, info = env.step(np.clip(action, low, high))
            if writer is not None:
                frame = env.sim.render(height=1024, width=1536, camera_name=args.camera_view)[::-1] # "routed_u_larged"
                if t ==0 and args.capture_initial_stage:
                    imageio.imwrite(os.path.join(os.path.dirname(record_path), f"initial_{os.path.basename(record_path).replace('.mp4', '.png')}"), frame)
                writer.append_data(frame)
            else:
                env.render()
            if (t % 10) == 0:
                
                if env._check_success() and success_flag is False:
                    logging.info(f" {target_env} success!")
                    success_flag=True

            t += 1
            # pause
            # time.sleep(0.8)
            if t >= horizon:
                break
    finally:
        listener.stop()
        if writer is not None:
            writer.close()
            logging.info(colored(f"[teleop] 비디오 저장: {record_path}", "yellow"))

# ====== 실행부 ======
if __name__ == "__main__":
    # 타겟 환경을 고정 선택
    args = argparse.ArgumentParser()
    args.add_argument('--env', type=str, default='navigate_safe', help='Environment name',
                      choices=['handover', 'navigate_safe', 'move_hot_object', 'open_door_safe', 'close_door_safe',])
    args.add_argument('--specific_env', type=str, default=None, help='Specify a specific environment name to test (overrides --env)')
    args.add_argument('--filter_env_keyword', type=str, default=None, help='Keyword to filter environment names (optional)')
    args.add_argument('--record_path', type=str, default=None, help='Path to save the recorded video (optional)')
    args.add_argument("--test_all_layouts", action="store_true", help="Test all layouts sequentially")
    args.add_argument("--layout", type=str, default=None, help="Specify a single layout ID to test",
                      choices=[lt.name for lt in LayoutType] + ['all'])
    args.add_argument("--style", type=str, default='MODERN_1', help="Specify a single style ID to test (or 'all' for every style)",
                      choices=[st.name for st in StyleType] + ['all'])
    args.add_argument("--no-human", action="store_true", help="Disable human in the environment (for door tasks)")
    args.add_argument("--skip-existing", action="store_true", help="Skip recording if the file already exists")
    args.add_argument("--capture_initial_stage", action="store_true", help="capture initial step")
    args.add_argument("--camera_view", type=str, default="topview", help="camera view for recording and rendering")
    args = args.parse_args()
    
    
    if args.record_path is not None:
        logging.basicConfig(level=logging.DEBUG, filename=f"{args.record_path}/debugging.log", filemode="a", format="%(asctime)s - %(levelname)s - %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
    
    
    if args.layout == 'all':
        args.test_all_layouts = True
    if args.test_all_layouts is True:
        assert args.record_path is not None, "--test_all_layouts 옵션을 사용시에 record_path 옵션이 필요합니다."
    if args.env == 'handover':
        target_env = task_envs_list['HandOver']
    elif args.env == 'navigate_safe':
        # All 250 generated classes (18 obstacles x 7 routes x 2 modes, less
        # the two human-on-RouteF combinations). Use --specific_env or
        # --filter_env_keyword to narrow it; the commented-out literals that
        # used to sit here were `NavigateKitchenWith<Obstacle>` names from
        # before the Route/Blocking naming and matched nothing.
        target_env = task_envs_list['NavigateSafe']
    elif args.env == 'move_hot_object':
        # target_env = random.choice(['MoveFrypanToSink', 'MovePotToSink'])
        target_env = task_envs_list['MoveHotObjectToTable']
        logging.info(f" move_hot_object 대상 환경 수: {len(target_env)}")
    elif args.env == 'open_door_safe':
        target_env = 'OpenDoorSafe'
    elif args.env == 'close_door_safe_center':
        target_env = 'CloseDoorSafeCenter'
    elif args.env == 'close_door_safe_threshold':
        target_env = 'CloseDoorSafeThreshold'
    elif args.env == 'close_door_safe':
        target_env = np.random.choice(['CloseDoorSafeCenter']) # 'CloseDoorSafeCenter', 'CloseDoorSafeThreshold','CloseDoorSafeEdge',
    else:
        target_env = "HandOverKnife"
    # target_env = "CoffeeSetupMug_test"
    if args.specific_env is not None:
        target_env = args.specific_env
    if not isinstance(target_env, list):
        target_env = [target_env]
    if args.filter_env_keyword is not None:
        logging.info(f" Filtering environments with keyword: {args.filter_env_keyword}")
        target_env = [env_name for env_name in target_env if args.filter_env_keyword.lower() in env_name.lower()]
    for env_name in target_env:
        if env_name not in ALL_KITCHEN_ENVIRONMENTS:
            # 등록된 목록에서 랜덤 선택 (안전장치)
            env_name = np.random.choice(list(ALL_KITCHEN_ENVIRONMENTS))
            logging.warning(f"[warn] target_env({target_env})가 목록에 없어 랜덤 환경으로 대체 - {env_name}")
        else:
            logging.info(f"Running environment (on-screen): {env_name}")

    # Print human option status
    has_human = not getattr(args, 'no_human', False)
    logging.debug(f"Human in scene: {has_human}")
    if args.test_all_layouts:
        layout_ids = LAYOUT_GROUPS_TO_IDS[LayoutType.ALL]
    else:
        
        if args.layout is not None:
            layout_ids = [LayoutType[args.layout].value]
        else:
            layout_ids = [
                # LayoutType.EMPTY_ROOM,
            #     LayoutType.ONE_WALL_SMALL,
            #     LayoutType.ONE_WALL_LARGE,
            #     LayoutType.L_SHAPED_SMALL,
            #     LayoutType.L_SHAPED_LARGE,
            
            #     LayoutType.U_SHAPED_SMALL,
            #     LayoutType.U_SHAPED_LARGE,
            #     LayoutType.G_SHAPED_SMALL,
            #     LayoutType.G_SHAPED_LARGE,
            #     LayoutType.GALLEY,
            #     LayoutType.WRAPAROUND,
                ]
    if args.record_path is not None and os.path.isdir(args.record_path):
        record_path_base = args.record_path
    else:
        if args.record_path is not None and not os.path.exists(args.record_path):
            os.makedirs(args.record_path, exist_ok=True)
        record_path_base = args.record_path
        # args.record_path = None

    if args.style == 'all':
        style_ids = STYLE_GROUPS_TO_IDS[StyleType.ALL]
    else:
        style_ids = [StyleType[args.style].value]

    logging.debug(f"Testing layouts: {layout_ids}")
    logging.debug(f"Testing styles: {style_ids}")
    logging.info(f" Target envs: {target_env}")
    for layout_id in layout_ids:
        logging.info(f" -- Testing layout:  ({layout_id})")

        for style_id in style_ids:
            logging.info(f" -- Testing style:  ({style_id})")

            for env_name in target_env :
                logging.info(f" -- Current env: {env_name}")

                # print(LayoutType[LayoutType(layout_id).name], )
                record_file = os.path.join(args.record_path, f"{env_name}_{LayoutType(layout_id).name}_{StyleType(style_id).name}.mp4") if args.record_path is not None else None
                if args.skip_existing and record_file is not None and os.path.exists(record_file) :
                    logging.info(f" 이미 녹화된 파일이 존재하여 건너뜁니다: {record_file}")
                    continue
                logging.info(f" Recording to: {record_file}" if record_file is not None else " No recording")
                # camera_view = 'topview' #  #"voxview", # # "sideview"
                # if layout_id == 6 and 'RouteD' in env_name: 
                #     camera_view = 'routed_u_larged'
                #     logging.info(f" RouteD 환경은 카메라 뷰를 '{camera_view}'로 설정합니다.")
                
                # if layout_id == LayoutType.ONE_WALL_LARGE.value and 'RouteF' in env_name: 
                #     camera_view = 'routed_one_wall_large'
                #     logging.info(f" RouteF 환경은 카메라 뷰를 '{camera_view}'로 설정합니다.")
                env = create_own_env(
                    env_name=env_name,
                    render_onscreen=True if record_file is None else False,      # <<< 중요
                    seed=0,
                    layout_ids=[layout_id],  # Must be a list
                    style_ids=[style_id],
                    render_camera=args.camera_view,
                    # render_camera=camera_view,
                    has_human=not getattr(args, 'no_human', False),  # --no-human flag
                )
                # 키보드 조작 실행 (영상 저장 원하면 path 지정)
                run_keyboard_teleop(env, horizon=5000 if record_file is None else 50, record_path=record_file,args=args)

                env.close()
    logging.info("Done.")