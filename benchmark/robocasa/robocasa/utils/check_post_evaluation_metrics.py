"""Compatibility wrapper for validating post-evaluation reports."""
import importlib.util
from pathlib import Path


_CORE = Path(__file__).resolve().parents[1] / "metrics" / "summarize.py"
_SPEC = importlib.util.spec_from_file_location("robocasa_post_evaluation", _CORE)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def check_post_evaluation(report):
    return _MOD.check_post_evaluation(report)


__all__ = ["check_post_evaluation"]
