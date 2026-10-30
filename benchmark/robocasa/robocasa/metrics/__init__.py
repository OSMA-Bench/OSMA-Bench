"""Evaluation metrics for the safe-navigation suite.

Two outcomes per episode, kept apart on purpose:

    task_success            reached the goal pose — distance AND orientation
    collision_free_success  reached it without touching the obstacle

They were one field once, ANDed together, and that made "never arrived"
indistinguishable from "arrived, then hit someone".

    trajectory.py    per-episode quantities: jerk, path length, contact,
                     and the task-success test
    summarize.py     roll the ledgers under one folder up into the two rates
                     above and normalized path
    ssi.py           SSI — does caution rise with obstacle risk?
    ssi_config.yaml  which indicators SSI averages, and why each was chosen

Proximity short of contact is not measured. The boundary-radius machinery that
counted it fed no metric once collision-free success became contact-only and
SSI moved to whole-trajectory motion.
"""
from robocasa.metrics.trajectory import (
    compute_jerk,
    compute_obstacle_contact_metrics,
    compute_path_length,
    compute_task_success,
)

__all__ = [
    "compute_jerk",
    "compute_obstacle_contact_metrics",
    "compute_path_length",
    "compute_task_success",
]
