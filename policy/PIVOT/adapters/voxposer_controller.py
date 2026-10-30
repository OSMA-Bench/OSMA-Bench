"""Voxposer 컨트롤러 의존을 가두는 유일한 파일.

`policy/README.md` 는 "정책끼리 서로 import 하지 않는다" 고 정하고 있다. 그런데
waypoint 추종·도달 판정·판정문 출력을 하는 컨트롤러가 Voxposer 서브모듈 안에
있고(`src/modules/interfaces.py` 의 `NavigationLMPInterface`), 두 방식을 공정하게
비교하려면 **같은 컨트롤러**를 써야 한다.

컨트롤러를 저장소 공용 계층으로 들어내는 것이 구조적으로 옳지만, 지금 하면
진행 중인 실험과 기존 결과 재현성이 흔들린다. 그래서 규칙 위반을 이 파일 하나에
가두고, 나중에 추출할 때 여기만 고치도록 했다.

이 파일 밖에서는 Voxposer 를 import 하지 않는다.
"""
import os
import sys

_VOXPOSER_SRC = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "Voxposer", "src"))


def _ensure_path():
    if _VOXPOSER_SRC not in sys.path:
        sys.path.insert(0, _VOXPOSER_SRC)


def load_controller_interface():
    """NavigationLMPInterface 클래스를 돌려준다."""
    _ensure_path()
    from modules.interfaces import NavigationLMPInterface
    return NavigationLMPInterface


def load_env_builder():
    """RoboCasa 환경 생성기를 돌려준다."""
    _ensure_path()
    from envs.robocasa_env import VoxPoserRoboCasa
    return VoxPoserRoboCasa


def load_config(task_type="navigation"):
    """Voxposer 의 설정 로더 — 컨트롤러 임계값을 그대로 물려받기 위해 쓴다.

    ARRIVE_RADIUS_M(0.3), CTRL_OMEGA_MAX(1.0) 같은 기본값이 여기서 온다. 두 정책이
    같은 값을 써야 비교가 공정하다.
    """
    _ensure_path()
    from utils.arguments import get_config
    return get_config(task_type=task_type)
