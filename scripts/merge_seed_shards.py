#!/usr/bin/env python3
"""Merge the rollout shards and ledgers of one model/seed/policy run.

Shards are the numeric sweep folders ``{tag}_s<N>`` and the recovery folders
``{tag}_srecovery_<stamp>_l<layout>``.  A cell is (layout_id, task_name).  When
several shards hold a result for the same cell, the latest shard wins: numeric
shards in index order, then recoveries in stamp order.  The winning shard also
supplies the cell's ``layout<L>/<task>/`` folder, ledger row and trajectory, so
every artifact of a counted cell comes from the attempt that was counted.

Example:
  python scripts/merge_seed_shards.py --root outputs --prefix robobrain \
      --seed 3 --suffix piv --dest outputs/robobrain_seed3_pivot

Result: <dest>/{results.json,results_progress.jsonl,layout*/,ledger/,merge_manifest.json}
The merge is built in a hidden staging folder next to <dest> and renamed into
place.  An existing <dest> is never modified: the merge stops unless --replace
is given, which moves the old folder aside to <dest>.replaced-<stamp>.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

LOG_NAMES = {"run.log", "worker.log", "setup.log"}
TRAJ_NAME = re.compile(r"l(\d+)_s\d+_(.+)_seed[^_]+\.npz")


def _ordered(root: Path, tag: str, suffix: str = "") -> list[Path]:
    """Numeric shards by index, then recovery shards by stamp and layout."""
    name = re.compile(rf"{re.escape(tag)}_(?:s(\d+)|srecovery_(\d{{8}}T\d{{6}})_l(\d+)){re.escape(suffix)}")
    numeric, recovery = [], []
    for path in root.glob(f"{tag}_s*{suffix}"):
        m = name.fullmatch(path.name)
        if not m or not path.is_dir():
            continue
        if m.group(1) is not None:
            numeric.append((int(m.group(1)), path))
        else:
            recovery.append((m.group(2), int(m.group(3)), path))
    return [p for _, p in sorted(numeric)] + [p for *_, p in sorted(recovery)]


def _cell_key(cell: tuple[int, str]) -> str:
    return f"layout{cell[0]}/{cell[1]}"


def _skip_posteval(_dir: str, names: list[str]) -> list[str]:
    return [n for n in names if "posteval" in n.lower()]


def _merge_results(shards: list[Path], dest: Path, manifest: dict) -> dict:
    rows, owner, holders = {}, {}, {}
    configs = []
    raw = 0
    for shard in shards:
        for path in sorted(shard.glob("results*.json")):
            if "progress" in path.name:
                continue
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"unreadable results file {path}: {exc}") from exc
            config = {"config": data.get("config"), "model": data.get("model")}
            for entry in configs:
                if entry["value"] == config:
                    entry["shards"].append(shard.name)
                    break
            else:
                configs.append({"value": config, "shards": [shard.name]})
            for row in data.get("results", []):
                info = row.get("task_info", {})
                cell = (info.get("layout_id"), info.get("task_name"))
                if not (isinstance(cell[0], int) and isinstance(cell[1], str)):
                    raise RuntimeError(f"result row without layout_id/task_name in {path}")
                raw += 1
                rows[cell] = row  # keeps the first position, latest retry wins
                owner[cell] = shard
                holders.setdefault(cell, []).append(shard.name)
    if rows:
        out = {"results": list(rows.values())}
        if configs[0]["value"]["config"] is not None:
            out["config"] = configs[0]["value"]["config"]
        if configs[0]["value"]["model"] is not None:
            out["model"] = configs[0]["value"]["model"]
        (dest / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    counts = {}
    for shard in owner.values():
        counts[shard.name] = counts.get(shard.name, 0) + 1
    manifest.update({
        "result_rows_raw": raw,
        "cells": len(rows),
        "cells_with_several_results": {_cell_key(c): h for c, h in holders.items() if len(h) > 1},
        "cells_per_owner": counts,
        "cell_owner": {_cell_key(c): s.name for c, s in owner.items()},
    })
    if len(configs) > 1:
        manifest["results_config_variants"] = configs
    return owner


def _merge_progress(shards: list[Path], dest: Path) -> None:
    lines = []
    for shard in shards:
        for path in sorted(shard.glob("results_progress*.jsonl")):
            lines.extend(line for line in path.read_text().splitlines(True) if line.strip())
    if lines:
        (dest / "results_progress.jsonl").write_text(
            "".join(line if line.endswith("\n") else line + "\n" for line in lines))


def _merge_top_level(shards: list[Path], dest: Path, manifest: dict) -> None:
    """Logs: first shard wins.  unknown_errors.jsonl: concatenated.  Any other
    file (pivot_config.json): first copy kept, each differing copy saved as
    <stem>.variant<K><suffix> and listed in the manifest."""
    variants = {}
    for shard in shards:
        for item in sorted(shard.iterdir()):
            name = item.name
            if "posteval" in name.lower() or (item.is_dir() and name.startswith("layout")):
                continue
            if name.startswith("results") and item.suffix in {".json", ".jsonl"}:
                continue
            if item.is_dir():
                raise RuntimeError(f"unexpected folder in shard: {item}")
            if name in LOG_NAMES or name.startswith("setup_"):
                if not (dest / name).exists():
                    shutil.copy2(item, dest / name)
                continue
            data = item.read_bytes()
            if name == "unknown_errors.jsonl":
                with (dest / name).open("ab") as out:
                    out.write(data if not data or data.endswith(b"\n") else data + b"\n")
                continue
            seen = variants.setdefault(name, [])
            for entry in seen:
                if entry["bytes"] == data:
                    entry["shards"].append(shard.name)
                    break
            else:
                stem, ext = os.path.splitext(name)
                target = name if not seen else f"{stem}.variant{len(seen)}{ext}"
                shutil.copy2(item, dest / target)
                seen.append({"bytes": data, "file": target, "shards": [shard.name]})
    manifest["top_level_variants"] = {
        name: [{"file": e["file"], "shards": e["shards"]} for e in seen]
        for name, seen in variants.items() if len(seen) > 1}


def _merge_cells(shards: list[Path], owner: dict, dest: Path, manifest: dict) -> None:
    dirs = {}
    for shard in shards:
        for layout_dir in sorted(shard.glob("layout*")):
            m = re.fullmatch(r"layout(\d+)", layout_dir.name)
            if not m or not layout_dir.is_dir():
                raise RuntimeError(f"unexpected layout entry: {layout_dir}")
            for cell_dir in sorted(layout_dir.iterdir()):
                if not cell_dir.is_dir():
                    raise RuntimeError(f"unexpected file in layout folder: {cell_dir}")
                dirs.setdefault((int(m.group(1)), cell_dir.name), []).append(shard)

    def copy(shard: Path, cell: tuple[int, str]) -> None:
        rel = _cell_key(cell)
        shutil.copytree(shard / rel, dest / rel, ignore=_skip_posteval)

    missing_dir, without_result = [], {}
    for cell, shard in owner.items():
        if shard in dirs.get(cell, []):
            copy(shard, cell)
        else:
            missing_dir.append(_cell_key(cell))
    for cell, holders in dirs.items():
        if cell not in owner:
            copy(holders[-1], cell)  # latest failed attempt, kept for debugging
            without_result[_cell_key(cell)] = holders[-1].name
    manifest["owners_without_cell_dir"] = missing_dir
    manifest["cell_dirs_without_result"] = without_result


def _merge_ledgers(ledgers: list[Path], owner: dict, dest: Path) -> dict:
    """Each counted cell takes its ledger row and trajectory from the ledger of
    the shard that owns its result; rows of cells without a result are left out."""
    dest.mkdir(parents=True, exist_ok=True)
    rows, trajs, raw = {}, {}, 0
    for ledger in ledgers:
        rows[ledger], trajs[ledger] = {}, {}
        src = ledger / "episodes.jsonl"
        if src.exists():
            for line in src.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    raw += 1
                    rows[ledger][(int(row["layout"]), row["task"])] = row
        if (ledger / "traj").is_dir():
            for item in (ledger / "traj").iterdir():
                m = TRAJ_NAME.fullmatch(item.name)
                if not m:
                    raise RuntimeError(f"unexpected trajectory name: {item}")
                trajs[ledger].setdefault((int(m.group(1)), m.group(2)), []).append(item)
    by_shard = {ledger.name[: -len("_ledger")]: ledger for ledger in ledgers}
    chosen, not_from_owner, no_row, no_traj = {}, [], [], []
    for cell, shard in owner.items():
        ledger = by_shard.get(shard.name)
        if ledger is not None and cell in rows[ledger]:
            chosen[cell] = ledger
            continue
        fallback = [led for led in ledgers if cell in rows[led]]
        if fallback:
            chosen[cell] = fallback[-1]
            not_from_owner.append(_cell_key(cell))
        else:
            no_row.append(_cell_key(cell))
    with (dest / "episodes.jsonl").open("w") as out:
        for cell, ledger in chosen.items():
            out.write(json.dumps(rows[ledger][cell], ensure_ascii=False) + "\n")
    traj_dst = dest / "traj"
    traj_dst.mkdir(exist_ok=True)
    for cell, ledger in chosen.items():
        files = trajs[ledger].get(cell, [])
        if not files:
            no_traj.append(_cell_key(cell))
        for item in files:
            shutil.copy2(item, traj_dst / item.name)
    for ledger in ledgers:
        if (ledger / "run.json").exists():
            shutil.copy2(ledger / "run.json", dest / "run.json")
            break
    without_result = sorted({_cell_key(c) for r in rows.values() for c in r} - {_cell_key(c) for c in owner})
    summary = {
        "source_ledgers": [str(p) for p in ledgers], "n_ledgers": len(ledgers),
        "raw_episode_rows": raw, "deduplicated_episode_rows": len(chosen),
        "dedup_key": ["layout", "task"], "row_source": "ledger of the shard that owns the result",
        "rows_not_from_result_owner": not_from_owner, "results_without_ledger_row": no_row,
        "rows_without_trajectory": no_traj, "ledger_cells_without_result": without_result,
    }
    (dest / "merge_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _build(tag: str, shards: list[Path], ledgers: list[Path], staging: Path, dest: Path) -> dict:
    manifest = {"tag": tag, "dest": str(dest), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "shards": [s.name for s in shards]}
    owner = _merge_results(shards, staging, manifest)
    _merge_progress(shards, staging)
    _merge_top_level(shards, staging, manifest)
    _merge_cells(shards, owner, staging, manifest)
    ledger = _merge_ledgers(ledgers, owner, staging / "ledger")
    manifest["ledger"] = {k: v for k, v in ledger.items() if k != "source_ledgers"}
    (staging / "merge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("outputs"))
    ap.add_argument("--prefix", required=True, help="model prefix before _seedN")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--suffix", help="optional policy suffix placed after _seedN, e.g. vox or piv")
    ap.add_argument("--dest", type=Path)
    ap.add_argument("--replace", action="store_true",
                    help="move an existing --dest aside to <dest>.replaced-<stamp> and rebuild it")
    args = ap.parse_args()
    tag = f"{args.prefix}_seed{args.seed}"
    if args.suffix:
        tag = f"{tag}_{args.suffix}"
    shards = _ordered(args.root, tag)
    ledgers = _ordered(args.root, tag, "_ledger")
    if not shards and not ledgers:
        ap.error(f"no shard or ledger folders found for {tag}")
    dest = Path(os.path.abspath(args.dest or (args.root / tag)))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest.parent / f".{dest.name}.merge.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(f"another merge into {dest} is running")
        if dest.exists() and any(dest.iterdir()) and not args.replace:
            sys.exit(f"{dest} already exists; pass --replace to move it aside and rebuild it")
        staging = dest.with_name(f".{dest.name}.staging-{os.getpid()}")
        staging.mkdir()
        try:
            manifest = _build(tag, shards, ledgers, staging, dest)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        moved = None
        if dest.exists():
            if any(dest.iterdir()):
                moved = dest.with_name(f"{dest.name}.replaced-{time.strftime('%Y%m%dT%H%M%S')}")
                dest.rename(moved)
            else:
                dest.rmdir()
        staging.rename(dest)
    note = f"; previous folder moved to {moved}" if moved else ""
    print(f"merged {len(shards)} shards + {len(ledgers)} ledgers -> {dest}: "
          f"{manifest['cells']} cells from {manifest['result_rows_raw']} result rows{note}")


if __name__ == "__main__":
    main()
