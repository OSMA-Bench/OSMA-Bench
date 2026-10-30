#!/usr/bin/env bash
# Full pipeline driver: per-layout rerun -> aggregate -> filter ->
# image copy -> destination-reachability check.
#
# Usage:
#   ./rerun.sh                                 # all layouts -> initial_violations/
#   ./rerun.sh --gpus 0 1 2 3                  # parallel rerun + parallel dest check
#   ./rerun.sh --root my_run                   # custom root
#   ./rerun.sh --layouts L_SHAPED_SMALL        # subset
#   ./rerun.sh --aggregate-only                # skip rerun stage
#   ./rerun.sh --aggregate-only --skip-destination  # just merge + filter + images
#   ./rerun.sh --gpus 0 1 2 3 --dest-workers 8 # override dest-check parallelism
#
# All args are forwarded to rerun_and_aggregate.py.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOSUITE="$(cd "$HERE/../robosuite" 2>/dev/null && pwd || echo "")"

export PYTHONPATH="${ROBOSUITE}:${HERE}${PYTHONPATH:+:$PYTHONPATH}"

exec python "$HERE/rerun_and_aggregate.py" "$@"
