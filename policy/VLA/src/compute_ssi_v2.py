"""Run the canonical SSI aggregation against the **robocasa_v2** tier roster.

Why this exists
---------------
`scripts/compute_ssi.py` is the canonical SSI entry point and must not be
reimplemented (README.md § "Metrics"). But it hardcodes the module it scores with::

    scripts/compute_ssi.py:38
    _SSI_PATH = _REPO / "benchmark" / "robocasa" / "robocasa" / "utils" / "ssi.py"

That inlined tree is the **pre-2026-08-26 roster** (11 obstacle types, includes
Kettlebell). The upstream v2 tree used for the 1,250-instance sweeps carries a
different roster (19 types: Kettlebell dropped; ChildBoy/ChildGirl/FlowerPot/
TableLamp/DeliveryBox/CardboardBox/WoodenCrate/FloorCushion/DuffelBag added).
Scoring a v2 sweep with the old module silently loses the tier for every new
obstacle — roughly half the task classes.

`scripts/**` is a read-only area, so instead of editing it this module loads it
and rebinds the three names it resolved from the old path to the v2 module.
Everything else — `report()`, `load_results()`, the CLI — is the original code.

The two modules expose an identical set of public functions (verified
2026-08-30), so the swap is a roster change, not a formula change.

Usage
-----
    OSMA_BENCH_ROBOCASA_V2=/path/to/robocasa_v2 \
        python3 policy/VLA/src/compute_ssi_v2.py <sweep_dir_or_results.json> ...

`OSMA_BENCH_ROBOCASA_ROOT` is accepted as a fallback for the v2 tree (it is the
variable the sweep itself ran with). There is deliberately **no default**:
scoring a v2 sweep against the inline pre-v2 roster would silently drop half
the tiers. The canonical script is located through `OSMA_BENCH_COMPUTE_SSI`,
defaulting to the parent repository's `scripts/compute_ssi.py`.

Use plain `scripts/compute_ssi.py` for the pre-v2 690 sweeps.

⚠ robocasa metric-cleanup trees (2026-09-04+) deleted ``robocasa/utils/ssi.py`` — this wrapper
cannot score sweeps run on them and exits with the path error below. SSI on those sweeps waits
for the team's redefined metric (team-internal notes on the v2 migration, §9-2).
"""

from __future__ import annotations

import importlib.util as _ilu
import os
import sys
from pathlib import Path

from repo_paths import ROBOCASA_ROOT_ENV, resolve_paths  # noqa: E402

V2_ROOT_ENV = "OSMA_BENCH_ROBOCASA_V2"
PATHS = resolve_paths()
_CANONICAL = PATHS.canonical_compute_ssi


def _load(path: Path, name: str):
    spec = _ilu.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _v2_ssi_path() -> Path:
    value = os.environ.get(V2_ROOT_ENV) or os.environ.get(ROBOCASA_ROOT_ENV)
    if not value:
        raise SystemExit(
            f"[compute_ssi_v2] set {V2_ROOT_ENV} (or {ROBOCASA_ROOT_ENV}) to the robocasa_v2 "
            "checkout root; there is no default because the inline tree carries the pre-v2 roster."
        )
    path = Path(value).expanduser().resolve() / "robocasa" / "utils" / "ssi.py"
    if not path.is_file():
        raise SystemExit(f"[compute_ssi_v2] v2 ssi.py not found: {path}")
    return path


def build():
    """Return the canonical module with its SSI implementation swapped to v2."""
    canonical = _load(PATHS.require_compute_ssi("compute_ssi_v2"), "_osmabench_compute_ssi")
    v2 = _load(_v2_ssi_path(), "_robocasa_ssi_v2")

    # Rebind exactly the names compute_ssi.py resolved from the old tree
    # (scripts/compute_ssi.py:41-43). report() reads these as module globals.
    canonical.compute = v2.compute
    canonical.enrich_with_boundary_stats = v2.enrich_with_boundary_stats
    canonical.TIERS, canonical.AXES, canonical.LEGACY_AXES = (
        v2.TIERS,
        v2.AXES,
        v2.LEGACY_AXES,
    )
    return canonical, v2


def main() -> None:
    canonical, v2 = build()
    tier_map = next(
        (val for val in vars(v2).values() if isinstance(val, dict) and "Dog" in val),
        None,
    )
    print(
        f"[compute_ssi_v2] ssi module: {_v2_ssi_path()}  "
        f"(obstacle types: {len(tier_map) if tier_map else '?'})",
        file=sys.stderr,
    )
    canonical.main()


if __name__ == "__main__":
    main()
