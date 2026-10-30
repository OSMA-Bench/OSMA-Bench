"""Write a post-evaluation metrics report from one or more ledgers."""
import argparse
import importlib.util
import json
from pathlib import Path


_CORE = Path(__file__).resolve().parents[1] / "metrics" / "summarize.py"
_SPEC = importlib.util.spec_from_file_location("robocasa_post_evaluation", _CORE)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger_dirs_legacy", nargs="*", help=argparse.SUPPRESS)
    parser.add_argument("--inputs", nargs="+", dest="ledger_dirs", required=False,
                        help="one or more ledger directories")
    parser.add_argument("--scope", nargs="+", choices=["all", "task_success", "collision_free_task_success"],
                        help="SSI scopes to include (default: all scopes)")
    parser.add_argument("--comparison", choices=["individual", "matched_intersection"],
                        default="individual")
    parser.add_argument("--aggregate-seeds", action="store_true",
                        help="group *_seedN_*_ledger inputs and report mean/std across seeds")
    parser.add_argument("--allow-partial", dest="allow_partial", action="store_true",
                        default=True,
                        help="compute pairwise tau for cells with at least two tiers (default)")
    parser.add_argument("--strict-complete", action="store_true",
                        help="require H/M/L tiers in every cell")
    parser.add_argument("--matched-intersection", action="store_true",
                        help="restrict all models to their common tasks")
    parser.add_argument("--optimal", default=_MOD.OPTIMAL)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    args.ledger_dirs = args.ledger_dirs or args.ledger_dirs_legacy
    if not args.ledger_dirs:
        parser.error("--inputs requires at least one ledger directory")
    expanded = []
    for item in args.ledger_dirs:
        p = Path(item)
        if (p / "episodes.jsonl").is_file():
            expanded.append(str(p))
        else:
            expanded.extend(str(x) for x in _MOD.find_ledgers(p))
    if not expanded:
        parser.error("no ledger found under --inputs")
    report = _MOD.summarize_post_evaluation(
        expanded, args.optimal,
        matched_intersection=args.matched_intersection,
        comparison=("matched_intersection" if args.matched_intersection else args.comparison),
        scopes=args.scope,
        allow_partial=(args.allow_partial and not args.strict_complete),
        aggregate_seeds=args.aggregate_seeds)
    _MOD.check_post_evaluation(report)
    with Path(args.out).open("w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
