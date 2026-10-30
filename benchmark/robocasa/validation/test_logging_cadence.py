"""Regression tests for the shared 20 Hz trajectory logging cadence."""

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

import yaml


ROBOCASA_ROOT = Path(__file__).parents[1] / "robocasa"


def _load_logger():
    package = types.ModuleType("robocasa")
    package.__path__ = [str(ROBOCASA_ROOT)]
    sys.modules.setdefault("robocasa", package)
    spec = importlib.util.spec_from_file_location(
        "robocasa.logging.log", ROBOCASA_ROOT / "logging" / "log.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_logger_uses_the_shared_control_interval():
    logger = _load_logger()
    assert logger.LOG_INTERVAL == 1


def test_metric_config_declares_the_control_cadence():
    config = yaml.safe_load((ROBOCASA_ROOT / "metrics" / "eval_config.yaml").read_text())
    assert config["cadence"]["log_interval_steps"] == 1


def test_live_rates_cover_all_finished_episodes():
    logger = _load_logger()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        logger.RunLog.start(root, policy="test", sim_commit="test")
        logger.EpisodeLog(root, task="first", episode_id="first").finish(
            task_success=True, collision_free_success=True
        )
        logger.EpisodeLog(root, task="second", episode_id="second").finish(
            task_success=False, collision_free_success=False
        )
        rates = logger.write_live_rates(root)
        persisted = json.loads((root / "derived" / "rates.json").read_text())

    assert rates == persisted
    assert rates["task_success_rate"] == 0.5
    assert rates["collision_free_success_rate"] == 0.5
