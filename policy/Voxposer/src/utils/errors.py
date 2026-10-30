"""Error taxonomy for the navigation eval pipeline.

Two axes:
  * **category**  — granular reason a task failed (e.g. ``planner_empty``)
  * **scope**     — coarse `sim` vs `llm` partition that drives retry policy

`sim` failures are retried (env init / mujoco / planner are typically
transient); `llm` failures (model produced unusable code, vLLM unreachable,
hard timeouts) are not — retrying with the same fresh inference rarely flips
the outcome and just burns budget.

`classify(exc)` returns a category name. `is_retryable(category)` returns
whether the outer task-level retry loop should attempt again.

Categories whose triggers we have not yet observed fall into ``unknown``,
which is treated as `sim` (optimistic — log and retry). New string patterns
that show up repeatedly should be promoted to a named category.
"""
from __future__ import annotations

import json
import os
from typing import Optional


# --- Exception types raised by the LMP layer -------------------------------

class LMPEmptyOutput(Exception):
    """Raised when the LLM returns no executable code at all."""


class LMPApiUnreachable(Exception):
    """Raised when the vLLM/OpenAI endpoint cannot be reached after N retries."""


class LMPNoActuation(Exception):
    """Raised when LMP exec completed but the robot took zero steps.

    Indicates the model produced code that neither called ``composer`` nor
    ``execute_navigation`` — actuation never started.
    """


class LMPUnresolvedTarget(Exception):
    """Raised when the affordance map names no reachable goal cell.

    The planner few-shots tell the model to write placeholder names ('the
    goal', 'object_1', ...) and promise that "the harness replaces these
    placeholders with the actual scene names". No such substitution exists,
    so ``parse_query_obj('goal')`` resolves to nothing. Until 2026-09-04 that
    silently returned a fallback observation positioned at the world origin,
    and the robot dutifully navigated to (0, 0) — logged as an ordinary
    goal-miss. layout8 RouteC failed this way 36/36 with no error recorded
    anywhere. Fail loudly instead: an unresolvable goal is a broken plan, not
    a navigation attempt, and must not be scored as one.
    """


# --- Pattern → category map ------------------------------------------------

# Order matters: first match wins. Keep more-specific patterns above generic
# ones (e.g. "Current sensor for observable" before "is invalid").
ERROR_PATTERNS = (
    ("render_framebuffer",  ("framebuffer",)),
    ("render_pointcloud",   ("point cloud",)),
    ("sensor_missing",      ("Current sensor for observable", "sensor", "observable")),
    ("planner_empty",       ("path_voxel is empty", "path_pixel is empty",
                             "assert len(path_voxel)", "assert len(path_pixel)")),
    ("lmp_import_banned",   ("assert phrase not in code_str",)),
    ("lmp_exec_error",      ("Error executing code",)),
    ("mujoco_runtime",      ("MjFatalError", "mjr_render", "mj_step", "MuJoCo")),
    ("task_setup",          ("ROUTE_DEFINITIONS", "target_pos", "fixtures")),
)

# Categories that imply a transient simulation issue → retryable.
SIM_CATEGORIES = frozenset({
    "render_framebuffer",
    "render_pointcloud",
    "sensor_missing",
    "planner_empty",
    "task_setup",
    "mujoco_runtime",
    "unknown",
})

# vLLM-inference-side issues. Distinct from llm-output errors below — these
# are infrastructure/connectivity failures (endpoint hung, dropped) and are
# usually transient, so we retry them like sim errors.
VLLM_CATEGORIES = frozenset({
    "timeout",
    "api_unreachable",
})

# Categories that imply a model output problem → not retryable.
# (Model produced unusable Python code or no actuation — fresh inference
#  rarely changes that, retrying would just burn budget.)
LLM_CATEGORIES = frozenset({
    "lmp_empty",
    "lmp_import_banned",
    "lmp_exec_error",
    "lmp_no_actuation",
    "lmp_unresolved_target",
})


def classify(exc: BaseException) -> str:
    """Return the category name for an exception."""
    name = type(exc).__name__
    if name == "TaskTimeout":
        return "timeout"
    if name == "LMPApiUnreachable":
        return "api_unreachable"
    if name == "LMPEmptyOutput":
        return "lmp_empty"
    if name == "LMPNoActuation":
        return "lmp_no_actuation"
    if name == "LMPUnresolvedTarget":
        return "lmp_unresolved_target"
    s = str(exc).lower()
    for cat, pats in ERROR_PATTERNS:
        for p in pats:
            if p.lower() in s:
                return cat
    return "unknown"


def is_retryable(category: str) -> bool:
    """Whether the outer retry loop should re-attempt for this category.

    Policy:
      - simulation-side errors (incl. unknown) are retried (transient sim issues)
      - vllm-inference-side errors (timeout, api_unreachable) are retried
        (transient endpoint hangs / disconnects)
      - llm-output errors (lmp_empty, lmp_no_actuation, lmp_import_banned,
        lmp_exec_error) are NOT retried — fresh inference rarely fixes them
    """
    return category in SIM_CATEGORIES or category in VLLM_CATEGORIES


def log_unknown(run_dir: str, task_info: dict, error_str: str) -> None:
    """Append an unknown-category error to {run_dir}/unknown_errors.jsonl.

    Operators can grep this file to discover patterns that should be promoted
    to a named ERROR_PATTERNS entry.
    """
    if not run_dir:
        return
    path = os.path.join(run_dir, "unknown_errors.jsonl")
    record = {
        "task_name": (task_info or {}).get("task_name"),
        "task_dir":  (task_info or {}).get("task_dir"),
        "error":     error_str,
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass


__all__ = [
    "LMPEmptyOutput",
    "LMPApiUnreachable",
    "LMPNoActuation",
    "ERROR_PATTERNS",
    "SIM_CATEGORIES",
    "VLLM_CATEGORIES",
    "LLM_CATEGORIES",
    "classify",
    "is_retryable",
    "log_unknown",
]
