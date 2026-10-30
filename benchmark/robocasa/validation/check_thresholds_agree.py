"""The environment and the scorer must use the same arrival test.

They did not. The env scored at 0.6 m, and at 0.9 m when the target was the
human, while eval_config.yaml — which the re-scoring reads — said 0.5 m. Over a
finished 1250-episode run that gap flipped 33 verdicts and moved TSR by 2.6
points, decided entirely by which side did the scoring. Nothing reported the
disagreement, because each side was self-consistent.

The human exception was the worse half: it scored the most dangerous obstacle
in the roster most leniently. Stopping 0.9 m short of a person counted as
arrival while stopping 0.6 m short of a bin did not.

This walks real environments rather than reading the constant, so a per-task
adjustment reintroduced anywhere still fails here.
"""
import numpy as np
import robosuite

from robocasa.metrics._config import DIST_TH, ORI_TH

TASKS = [
    "NavigateKitchenHumanBlockingRouteA",      # the one that used to get +0.3
    "NavigateKitchenChildBoyBlockingRouteA",   # high tier, not the human itself
    "NavigateKitchenCatBlockingRouteB",
    "NavigateKitchenWineBlockingRouteC",
    "NavigateKitchenTrashbinBlockingRouteD",
    "NavigateKitchenDuffelBagNonBlockingRouteE",
]


def main():
    bad = []
    for task in TASKS:
        env = robosuite.make(
            env_name=task, robots="PandaOmron", layout_ids=0, style_ids=3,
            has_renderer=False, has_offscreen_renderer=False,
            use_camera_obs=False, control_freq=20, seed=0,
        )
        env.reset()
        low, _ = env.action_spec
        env.step(np.zeros_like(low))
        info = env.get_trajectory_info()
        pos, ori = info["pos_threshold"], info["ori_threshold"]
        # One cone for every route, doors included: the door aims at the panel
        # normal now, so it is scored on the same cos scale as everything else
        # and a second threshold would have nothing to mean.
        ok = abs(pos - DIST_TH) < 1e-9 and abs(ori - ORI_TH) < 1e-9
        print(f"  {task:44s} pos={pos:.3f} ori={ori:.3f} "
              f"(want {ORI_TH:.3f}) {'ok' if ok else 'MISMATCH'}")
        if not ok:
            bad.append((task, pos, ori, ORI_TH))
        env.close()

    if bad:
        print(f"\n{len(bad)} task(s) disagree with eval_config.yaml "
              f"(expected pos={DIST_TH}, ori={ORI_TH}):")
        for task, pos, ori, want in bad:
            print(f"  {task}: pos={pos}, ori={ori}, expected ori={want}")
        raise SystemExit(1)
    print(f"\nall {len(TASKS)} tasks score at pos={DIST_TH}, ori={ORI_TH}")


if __name__ == "__main__":
    main()
