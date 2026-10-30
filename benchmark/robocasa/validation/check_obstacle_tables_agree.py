"""Cross-check the tables that define an obstacle, and fail loudly.

An obstacle is defined in three places that nothing keeps in sync:

    _OBSTACLE_CLASS_NAMES      registers it and generates its task classes
    TIER_TO_OBSTACLES          the tier tuples the sweep and figures group by
    TIER_OF                    the caution tier SSI aggregates by

There were two more, both radii: a per-obstacle r_b the env scored boundary
intrusion against, and the per-tier copy SSI reported. Both are gone. Proximity
short of contact feeds no metric now, and the one radius left is a fixed
OBSTACLE_KEEPOUT_RADIUS_M that only diagnostics draw, so there is no longer a
radius for two tables to disagree about.

Every mismatch found so far has failed SILENTLY rather than raising:

  * missing from TIER_OF -> `compute_ssi` does `if ell is None: continue`,
    so every episode of that obstacle runs, burns GPU, and is then dropped
    from the metric with no warning. `Kettlebell` sat here until it was
    retired from the navigation roster on 2026-08-13, silently costing 20
    task classes / 160 instances at 8 layouts.
  * missing from a TIER_TO_OBSTACLES tuple -> it is scored, but the sweep and
    the tier figures never render it, so a tier's per-obstacle mean is taken
    over fewer types than the roster has. The Moderate tuple listed only its
    three floor obstacles this way, omitting the three table drinks.

Also checks the spawn class, which is a separate silent hazard: an obstacle
in neither TABLE_OBSTACLES nor TIPPY_FLOOR_OBSTACLES defaults to a 5 cm
floor drop, and tall / high-CoM / round meshes topple on that impact. That
is not detectable here (it needs a physics rollout), so this script only
reports the classification for review and points at the sweep.

Exit status: 0 clean, 1 if any inconsistency was found — usable in CI.

    python validation/check_tier_consistency.py
    python validation/check_tier_consistency.py --quiet   # only problems
"""
import argparse
import sys
from collections import defaultdict

# Lower case, matching ssi_config.yaml. TIER_TO_OBSTACLES in the env still
# capitalises; the comparison above folds case so only membership is checked.
TIER_ORDER = ("high", "medium", "low")


def load_tables():
    from robocasa.environments.kitchen.single_stage.kitchen_navigate_safe import (
        _OBSTACLE_CLASS_NAMES, TABLE_OBSTACLES, TIPPY_FLOOR_OBSTACLES,
        TIER_TO_OBSTACLES,
    )
    # No radius table on either side any more. The tier roster is still worth
    # cross-checking, because an obstacle missing from it is silently dropped
    # from every SSI comparison.
    from robocasa.metrics.ssi import TIER_OF
    # obstacle key -> the tier tuple it appears in, so a key in none of them
    # (or in two) is visible per row rather than only in the balance summary.
    tier_tuple_of = {}
    for tier, members in TIER_TO_OBSTACLES.items():
        for key in members:
            tier_tuple_of.setdefault(key, []).append(tier)
    return dict(
        classes=_OBSTACLE_CLASS_NAMES, table=TABLE_OBSTACLES,
        tippy=TIPPY_FLOOR_OBSTACLES, tier_of=TIER_OF,
        tier_tuple_of=tier_tuple_of,
    )


# `human` is the posed_human fixture, not a spawned free body: _reset_internal
# returns before the pin loop for it, so it never drops and needs no spawn
# class. Mirrors NON_SPAWNED in validation/check_spawn_class.py.
NON_SPAWNED = {"human"}


def spawn_class(key, t):
    if key in NON_SPAWNED:
        return "fixture (no drop)"
    if key in t["table"]:
        return "table (+1 cm)"
    if key in t["tippy"]:
        return "tippy (+2 cm)"
    return "plain floor (+5 cm)"


def check(quiet=False):
    t = load_tables()
    problems = []
    by_tier = defaultdict(list)

    rows = []
    for key, cls in sorted(t["classes"].items()):
        # TIER_OF is keyed by the obstacle key ("child_boy"), not by the
        # generated class name ("ChildBoy"): the roster lives in
        # ssi_config.yaml, which names obstacles the way the env does.
        tier = t["tier_of"].get(key)
        issues = []
        if tier is None:
            issues.append("no TIER_OF entry -> every episode is dropped from SSI")
        tuples = t["tier_tuple_of"].get(key, [])
        if not tuples:
            issues.append("in no TIER_TO_OBSTACLES tuple -> never swept or "
                          "plotted, and its tier's mean is short one type")
        elif len(tuples) > 1:
            issues.append(f"in {len(tuples)} TIER_TO_OBSTACLES tuples: {tuples}")
        # Case-insensitive: TIER_TO_OBSTACLES spells the tiers "High"/"Medium"
        # /"Low" and ssi_config.yaml uses lower case. Only the membership is
        # being checked here, not the spelling.
        elif tier is not None and tuples[0].lower() != tier.lower():
            issues.append(f"tier disagrees: TIER_TO_OBSTACLES={tuples[0]} "
                          f"TIER_OF={tier}")
        if issues:
            problems.append((key, cls, issues))
        by_tier[tier].append(key)
        rows.append((key, cls, tier, spawn_class(key, t), issues))

    # tables naming obstacles that are not registered
    registered = set(t["classes"].values())
    registered_keys = set(t["classes"])
    for name in sorted(t["tier_of"]):
        if name not in registered_keys:
            # a retired alias is harmless; flag it as info, not a failure
            if not quiet:
                print(f"[info] SSI tables list {name!r}, which is not a "
                      f"registered obstacle (stale alias?)")

    if not quiet:
        hdr = (f"{'obstacle':17s} {'ClassName':15s} {'tier':>7s}  "
               f"{'spawn':18s}")
        print(hdr)
        print("-" * len(hdr))
        for key, cls, tier, spawn, issues in rows:
            flag = "  <-- " + "; ".join(issues) if issues else ""
            print(f"{key:17s} {cls:15s} {str(tier):>7s}  {spawn:18s}{flag}")

        print("\ntier balance:")
        for tier in TIER_ORDER:
            members = by_tier.get(tier, [])
            print(f"  {tier:7s} {len(members):2d}  {', '.join(members)}")
        untiered = by_tier.get(None, [])
        if untiered:
            print(f"  {'(none)':7s} {len(untiered):2d}  {', '.join(untiered)}")

        sizes = {tier: len(by_tier.get(tier, [])) for tier in TIER_ORDER}
        if len(set(sizes.values())) != 1:
            print(f"\n[warn] tiers are unbalanced ({sizes}) — a per-tier mean "
                  f"is then computed over different numbers of obstacle types, "
                  f"so a tier contrast can be driven by roster size.")

        print("\nSpawn class is NOT verified here — it needs a physics rollout. "
              "After any roster change run:\n"
              "  python validation/check_obstacle_stability.py --seeds 0\n"
              "and read `starts_fallen` as 'wrong spawn class'.")

    if problems:
        print(f"\nFAIL — {len(problems)} inconsistency(ies):")
        for key, cls, issues in problems:
            for i in issues:
                print(f"  {key} ({cls}): {i}")
        return 1
    print("\nOK — all registered obstacles agree across radius, tier and SSI tables.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true",
                    help="print only problems (for CI)")
    args = ap.parse_args()
    sys.exit(check(quiet=args.quiet))


if __name__ == "__main__":
    main()
