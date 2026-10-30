#!/usr/bin/env bash
# Build (or rebuild) the browsable HTML report over a nav_sweep output tree.
#
#   scripts/nav_sweep_html.sh                    # figures/nav_sweep -> its index.html
#   scripts/nav_sweep_html.sh --open             # ...and open it in a browser
#   scripts/nav_sweep_html.sh figures/nav_sweep_260905
#   scripts/nav_sweep_html.sh -- --no-posters    # anything after -- goes to the
#                                                # generator (--jobs, --out, --title)
#
# nav_sweep.sh calls this itself, so a finished sweep already has its page;
# run it by hand after re-rendering overlays, or to point at an older tree.
set -euo pipefail

cd "$(dirname "$0")/.."

ROOT=figures/nav_sweep
OPEN=0
FORWARD=()

while [ $# -gt 0 ]; do
    case "$1" in
        --open) OPEN=1 ;;
        --) shift; FORWARD+=("$@"); break ;;
        -h|--help)  # the comment block under the shebang, minus its markers
            awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"
            exit 0 ;;
        -*) echo "[error] unknown flag $1 (use -- to pass flags on to the generator)" >&2
            exit 2 ;;
        *)  ROOT=$1 ;;
    esac
    shift
done

[ -d "$ROOT" ] || { echo "[error] no such sweep root: $ROOT" >&2; exit 1; }

# The generator reads the caution-tier roster out of robocasa. An editable
# install resolves on its own, but a bare shell (or one whose .pth files were
# not loaded) needs the repo on the path; without it the generator still works,
# falling back to the tier arrays in nav_sweep.sh.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python scripts/nav_sweep_report.py "$ROOT" "${FORWARD[@]}"

PAGE="$(cd "$ROOT" && pwd)/index.html"
echo "[open] file://$PAGE"

if [ "$OPEN" -eq 1 ]; then
    # Whichever of these exists; none of them is guaranteed on a headless box.
    for opener in xdg-open google-chrome firefox open; do
        if command -v "$opener" >/dev/null 2>&1; then
            "$opener" "$PAGE" >/dev/null 2>&1 &
            exit 0
        fi
    done
    echo "[warn] no browser found; open the URL above yourself" >&2
fi
