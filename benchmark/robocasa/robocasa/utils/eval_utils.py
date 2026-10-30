"""Utilities for task registry queries, category parsing, and result saving."""

import os
import re
import json
import datetime

from robocasa.models.scenes.scene_registry import LayoutType


# LayoutType members with non-negative values (actual layouts, not groups)
_LAYOUT_NAME_TO_ID = {
    lt.name: lt.value for lt in LayoutType if lt.value >= 0
}


def get_navigate_tasks():
    """Fetch all NavigateKitchen* task names from the robocasa registry."""
    import robocasa  # noqa: F401 – triggers env registration
    from robocasa.environments.kitchen.kitchen import REGISTERED_KITCHEN_ENVS

    return sorted(
        name for name in REGISTERED_KITCHEN_ENVS
        # Drop the two base classes: NavigateKitchen samples a random src/dst
        # and has no obstacle, NavigateKitchenWithObstacles is abstract.
        if name.startswith("NavigateKitchen")
        and name not in ("NavigateKitchen", "NavigateKitchenWithObstacles")
    )


def parse_task_spec(task_spec):
    """Parse 'TaskName_LAYOUT' into (task_name, layout_id).

    Examples:
        'NavigateKitchenDogNonBlockingRouteC_L_SHAPED_LARGE' -> ('NavigateKitchenDogNonBlockingRouteC', 3)
        'NavigateKitchenDogNonBlockingRouteC' -> ('NavigateKitchenDogNonBlockingRouteC', None)
    """
    for name, lid in sorted(_LAYOUT_NAME_TO_ID.items(), key=lambda x: -len(x[0])):
        suffix = f"_{name}"
        if task_spec.endswith(suffix):
            return task_spec[:-len(suffix)], lid
    return task_spec, None


def parse_task_categories(task_name):
    """Extract obstacle, blocking_mode, and route from a NavigateKitchen task name.

    Examples:
        'NavigateKitchenDogNonBlockingRouteC' -> ('Dog', 'NonBlocking', 'C')
        'NavigateKitchenWineBlockingRouteA' -> ('Wine', 'Blocking', 'A')
    """
    m = re.match(
        r"NavigateKitchen(?P<obstacle>.+?)(?P<blocking>NonBlocking|Blocking)Route(?P<route>[A-Z])$",
        task_name,
    )
    if m:
        return m.group("obstacle"), m.group("blocking"), m.group("route")
    return None, None, None


def save_results(results, summary, model, output_dir="results", filename=None):
    """Save per-task results (task_info + evaluation) and summary to a JSON file.

    Args:
        results: list of dicts, each with 'task_info' and 'evaluation' keys.
        summary: dict with aggregate statistics (from compute_summary).
        model: model name string.
        output_dir: directory to save results.
        filename: optional fixed filename (default: timestamped).

    Returns:
        filepath of the saved JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if filename is None:
        model_safe = model.replace("/", "_") if model else "unknown_model"
        filename = f"{model_safe}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    output = {
        "model": model,
        "timestamp": timestamp,
        "summary": summary,
        "results": results,
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    return filepath
