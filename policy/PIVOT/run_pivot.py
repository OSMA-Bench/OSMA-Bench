#!/usr/bin/env python3
"""PIVOT 방식 정책 실행 진입점.

환경 생성·에피소드 루프·컨트롤러·평가·판정문 기록은 Voxposer 의 ``run_tasks`` 를
**그대로 공유**한다 (`external_planner` 훅). 여기서 하는 일은 계획 단계를 PIVOT 방식
solver 로 갈아끼우는 것뿐이다 - `keypoint_nav` 와 같은 구조다.

평가 코드를 복제하지 않는 이유는 하나다: 세 정책이 같은 잣대로 채점되어야 비교가
성립한다. 복제하면 판정문 형식이나 임계가 조용히 어긋난다.

    python3 run_pivot.py -m Qwen/Qwen3-VL-8B-Instruct -p 8003 \\
        --layout-ids 0 --style-ids 3 -o outputs/smoke NavigateKitchenCatBlockingRouteA
"""
import argparse
import json
import os
import sys

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
sys.path.insert(0, _HERE)

from adapters.planner_hook import PivotPlannerHook          # noqa: E402
from adapters.voxposer_controller import _ensure_path       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*", help="과제 이름. 비우면 전체")
    ap.add_argument("-m", "--model", default=None)
    ap.add_argument("-p", "--port", type=int, default=None)
    ap.add_argument("-o", "--output-dir", default="outputs/pivot")
    ap.add_argument("-c", "--config", default=os.path.join(_HERE, "config.yaml"))
    ap.add_argument("--layout-ids", default="0")
    ap.add_argument("--style-ids", default="3")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--subgoals", choices=("solver", "stops", "pivot"),
                    default=None,
                    help="단계 끝점: solver(최적화) | stops(전체 뷰 1회 질의) "
                         "| pivot(반복 질의)")
    ap.add_argument("--dynamics", dest="dynamics", action="store_true",
                    default=None, help="v/a/J 제약을 VLM 에게 묻는다")
    ap.add_argument("--no-dynamics", dest="dynamics", action="store_false",
                    help="동역학 어휘를 빼고 기하만 (기본)")
    ap.add_argument("--max-stages", type=int, default=None,
                    help="단계 상한 (기본 4)")
    ap.add_argument("--seed", type=int, default=None,
                    help="VLM · pivot · 두 solver 가 나눠 쓰는 시드 (기본 0)")
    ap.add_argument("--maxfun", type=int, default=None,
                    help="두 solver 의 sampling_maxfun 을 한꺼번에")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.subgoals is not None:
        cfg.setdefault("main", {})["subgoals"] = args.subgoals
    if args.dynamics is not None:
        cfg.setdefault("main", {})["dynamics"] = bool(args.dynamics)
    if args.seed is not None:
        cfg.setdefault("main", {})["seed"] = args.seed
        cfg.setdefault("subgoal_solver", {})["seed"] = args.seed
        cfg.setdefault("path_solver", {})["seed"] = args.seed
    if args.maxfun is not None:
        cfg.setdefault("subgoal_solver", {})["sampling_maxfun"] = args.maxfun
        cfg.setdefault("path_solver", {})["sampling_maxfun"] = args.maxfun
    model = args.model or cfg["model"]["name"]
    base_url = cfg["model"].get("base_url", "http://localhost:8000/v1")
    if args.port:
        base_url = f"http://localhost:{args.port}/v1"
    cfg.setdefault("model", {})["name"] = model
    cfg["model"]["base_url"] = base_url

    if args.max_stages is not None:
        from src import stages as STG
        STG.MAX_STAGES = int(args.max_stages)

    out_abs = os.path.abspath(args.output_dir)
    os.makedirs(out_abs, exist_ok=True)
    hook = PivotPlannerHook(cfg, prompts_dir=os.path.join(_HERE, "prompts"),
                            out_root=out_abs)

    _ensure_path()
    from run_LMP import run_tasks                            # noqa: E402

    def _ids(s):
        return [int(x) for x in str(s).split(",") if str(x).strip()]

    with open(os.path.join(out_abs, "pivot_config.json"), "w") as f:
        json.dump({"config": cfg, "model": model, "base_url": base_url}, f,
                  ensure_ascii=False, indent=2)

    # run_LMP 은 Voxposer 디렉터리에서 실행되는 것을 전제로 설정 파일을 상대
    # 경로로 찾는다. 출력은 미리 절대 경로로 바꿔 두고 작업 디렉터리를 옮긴다.
    os.chdir(os.path.normpath(os.path.join(_HERE, "..", "Voxposer")))
    run_tasks(args.tasks or None, model=model, port=args.port or 8000,
              output_dir=out_abs, max_retries=args.max_retries,
              layout_ids=_ids(args.layout_ids), style_ids=_ids(args.style_ids),
              external_planner=hook)


if __name__ == "__main__":
    main()
