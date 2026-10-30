"""The VLA viewer export must use the rollout's shared control cadence."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
EXPORT = ROOT / "src" / "export_viewer_results.py"
CONTROL = ROOT.parents[1] / "benchmark" / "robocasa" / "robocasa" / "control.py"


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("export_viewer_results", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_reads_the_shared_control_interval(monkeypatch):
    module = _load_module(EXPORT)
    monkeypatch.setenv("OSMA_BENCH_ROBOCASA_ROOT", str(CONTROL.parents[1]))
    assert module.load_control_log_interval_steps() == 1


def test_export_has_no_private_trajectory_interval_constant():
    assert "TRAJECTORY_LOG_INTERVAL =" not in EXPORT.read_text()
