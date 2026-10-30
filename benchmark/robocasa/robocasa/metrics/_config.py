"""Load eval_config.yaml, and fail on import if it disagrees with itself.

One parse, in one place. A config that nothing reads is documentation, and
documentation drifts; a config parsed in three places drifts against itself.

The SSI section is owned by ssi.py — the tier roster, the indicators and their
comparisons are re-exported from there rather than parsed again here, so the
two cannot disagree about what SSI measures.
"""
import os

import yaml

from robocasa.control import CONTROL_LOG_INTERVAL_STEPS
from robocasa.metrics.ssi import INDICATORS, TIER_OF, TIERS  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "eval_config.yaml")

with open(PATH) as _fh:
    CFG = yaml.safe_load(_fh)

SUITE = CFG["suite"]
TSR = CFG["tsr"]
CSR = CFG["csr"]
CADENCE = CFG["cadence"]
SMOOTHING = CFG["smoothing"]

DIST_TH = float(TSR["distance_threshold_m"])
ORI_TH = float(TSR["orientation_cos_threshold"])
PLANNED = int(SUITE["planned_episodes"])
LAYOUTS = list(SUITE["layouts"])

COLLISION_DISPLACEMENT_M = float(CSR["displacement_threshold_m"])
IMMOVABLE_OBSTACLES = frozenset(CSR["immovable_obstacles"])

DISTANCE_MEASURE_MAX_M = float(CFG["distance_measure_max_m"])
JERK_SMOOTHING = SMOOTHING["jerk"]
CONTROL_DT = float(CADENCE["control_dt_s"])
LOG_DT = float(CADENCE["log_dt_s"])


def _check():
    """Internal consistency, so a wrong number fails loudly instead of quietly."""
    problems = []
    n = SUITE["task_classes"] * len(LAYOUTS)
    if n != PLANNED:
        problems.append(f"planned_episodes {PLANNED} != "
                        f"{SUITE['task_classes']} x {len(LAYOUTS)} = {n}")
    if abs(LOG_DT - CADENCE["log_interval_steps"] / CADENCE["control_hz"]) > 1e-9:
        problems.append("log_dt_s does not match log_interval_steps / control_hz")
    if int(CADENCE["log_interval_steps"]) != CONTROL_LOG_INTERVAL_STEPS:
        problems.append("cadence.log_interval_steps does not match "
                        "CONTROL_LOG_INTERVAL_STEPS")
    w = JERK_SMOOTHING["window"]
    if w % 2 == 0:
        problems.append(f"savgol window {w} must be odd")
    if w <= JERK_SMOOTHING["polyorder"]:
        problems.append("savgol window must exceed polyorder")
    unknown = sorted(IMMOVABLE_OBSTACLES - {o.title().replace("_", "")
                                            for o in TIER_OF})
    if unknown:
        problems.append(f"immovable_obstacles names not in the tier roster: "
                        f"{unknown}")
    return problems


PROBLEMS = _check()
if PROBLEMS:
    raise SystemExit("eval_config.yaml is inconsistent:\n  - "
                     + "\n  - ".join(PROBLEMS))


def summary():
    return "\n".join([
        f"suite      {SUITE['task_classes']} classes x {len(LAYOUTS)} layouts "
        f"= {PLANNED} episodes",
        f"TSR        dist <= {DIST_TH} m AND ori >= {ORI_TH}, "
        f"denominator = {TSR['denominator']}",
        f"CSR        reached AND untouched; evidence "
        f"{CSR['evidence_priority']}, displacement > "
        f"{COLLISION_DISPLACEMENT_M} m, "
        f"{len(IMMOVABLE_OBSTACLES)} obstacles undecidable",
        f"cadence    statistics from {CADENCE['statistics_from']} "
        f"(dt={CONTROL_DT}s); series logged every {LOG_DT}s",
        f"smoothing  jerk only: savgol window {JERK_SMOOTHING['window']} "
        f"({JERK_SMOOTHING['window'] * CONTROL_DT:.2f}s), "
        f"poly {JERK_SMOOTHING['polyorder']}",
        f"SSI        {len(INDICATORS)} indicators over "
        f"{len(TIERS)} tiers: "
        + ", ".join(f"{n}({c})" for n, _k, _s, c in INDICATORS),
    ])


if __name__ == "__main__":
    print(summary())
