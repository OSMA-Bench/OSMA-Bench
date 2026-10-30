"""FOREWARN-style policy steering (Wu et al., arXiv 2502.01828) for any model the VLA server serves.

When the runner needs actions, ``Steerer.decide`` snapshots the env and, for each candidate, restores
the snapshot and runs ``lookahead_chunks`` chunks of the policy closed-loop in that same env, keeping
the base path and a topview frame per chunk. The env is restored again, a VLM narrates every
candidate's filmstrip and picks one with the Fig. 9 prompt, and the runner replays the chosen
candidate's chunks. Replay is deterministic, so the next decision checks that the env reached
exactly the state the chosen candidate ended in.

``verifier.select_with`` says what the picking step gets: the descriptions, the Mode-labelled
filmstrips, or both. The Mode shuffle happens here, so the number can be written on the images.

Candidates differ only in the policy's random draw; ``chunk_noise`` says how, and the ``sampling``
block of configs/steering/forewarn.yaml sets it.
"""

from __future__ import annotations

import collections
import copy
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from episode_dump import _data_array, resolve_obstacle_groups
from forewarn import snapshot, verifier


KEYS = {
    "": {"num_candidates", "lookahead_chunks", "sampling", "verifier"},
    "sampling": {"seed", "shared_noise"},
    "sampling.shared_noise": {"chunks", "noise_scale"},
    "verifier": {
        "base_url", "model", "task_condition", "temperature", "seed", "timeout_seconds", "max_parallel",
        "narration_max_tokens", "selection_max_tokens", "select_with",
    },
}


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for section, expected in KEYS.items():
        block = config
        for name in filter(None, section.split(".")):
            block = block.get(name) if isinstance(block, dict) else None
        if not isinstance(block, dict) or set(block) != expected:
            found = sorted(block) if isinstance(block, dict) else block
            raise ValueError(f"{path}: {section or 'top level'} keys must be {sorted(expected)}, found {found}")
    shared = config["sampling"]["shared_noise"]
    first, last = shared["chunks"]
    if not (config["num_candidates"] >= 1 and 0 <= first <= last < config["lookahead_chunks"]):
        raise ValueError(f"{path}: need num_candidates >= 1 and 0 <= sampling.shared_noise.chunks[0] <= [1] < lookahead_chunks")
    if not shared["noise_scale"] > 0:
        raise ValueError(f"{path}: sampling.shared_noise.noise_scale must be positive")
    if config["verifier"]["select_with"] not in verifier.SELECT_WITH:
        raise ValueError(f"{path}: verifier.select_with must be one of {list(verifier.SELECT_WITH)}")
    return config


def chunk_noise(sampling: dict[str, Any], decision_step: int, candidate: int, chunk: int) -> dict[str, Any]:
    """The starting noise of a candidate's ``chunk``-th chunk (one policy call): which draw, its policy_seed and its scale.

    Chunks in ``shared_noise.chunks`` = [first, last] all reuse draw ``first``, times ``noise_scale``;
    every other chunk uses its own draw, times 1. policy_seed goes to /reset, where every adapter
    reseeds its RNG, so one policy_seed is one draw; noise_scale goes to /act.
    """
    first, last = sampling["shared_noise"]["chunks"]
    shared = first <= chunk <= last
    draw = first if shared else chunk
    seed = np.random.SeedSequence([sampling["seed"], decision_step, candidate, draw]).generate_state(1)[0]
    scale = float(sampling["shared_noise"]["noise_scale"]) if shared else 1.0
    return {"noise_draw": draw, "policy_seed": int(seed), "noise_scale": scale}


class Steerer:
    """``query(history, step, sampling) -> (chunk, query record)``, ``step(action) -> (transition, applied,
    robot, task)`` and ``observe(observation) -> history.append kwargs`` are the runner's own functions."""

    def __init__(
        self,
        env: Any,
        config: dict[str, Any],
        *,
        server_meta: dict[str, Any],
        client: Any,
        instruction: str,
        model_profile: str,
        query: Callable[..., tuple[np.ndarray, dict[str, Any]]],
        step: Callable[[np.ndarray], tuple[Any, Any, dict[str, Any], dict[str, Any]]],
        observe: Callable[[dict[str, Any]], dict[str, Any]],
        replan_steps: int,
        goal_xy: Any,
        control_freq: float,
        stop_on_env_success: bool,
        run_dir: Path,
    ) -> None:
        scale = config["sampling"]["shared_noise"]["noise_scale"]
        if scale != 1.0 and "noise_scale" not in server_meta.get("act_sampling_keys", []):
            raise ValueError(
                f"sampling.shared_noise.noise_scale is {scale}, but the {model_profile} server does not list noise_scale "
                "in /meta act_sampling_keys. Set it to 1.0; candidates then differ by policy_seed alone."
            )
        self.env, self.config, self.client = env, config, client
        self.instruction, self.model_profile = instruction, model_profile
        self.query, self.step, self.observe = query, step, observe
        self.replan_steps, self.control_freq = int(replan_steps), float(control_freq)
        self.goal_xy = np.asarray(goal_xy, dtype=np.float64).reshape(2)
        self.stop_on_env_success = bool(stop_on_env_success)
        self.image_dir, self.log_path = Path(run_dir) / "steering", Path(run_dir) / "steering.jsonl"
        self.camera = verifier.TopviewCamera.from_env(env)
        self.obstacle_geoms = [group["geom_ids"] for group in resolve_obstacle_groups(env)[0].values()]
        self.decision = 0
        self.expected: tuple[int, dict[str, str]] | None = None
        self.summary = {"decisions": 0, "fallbacks": 0, "commit_checks_matched": 0, "commit_checks_failed": 0}

    def _commit_check(self, step: int) -> dict[str, Any] | None:
        if self.expected is None:
            return None
        (expected_step, expected), self.expected = self.expected, None
        now = snapshot.digest(snapshot.capture_env(self.env))
        check = {"expected_step": expected_step, "step": step, "differs": sorted(k for k in now if now[k] != expected[k])}
        check["matched"] = step == expected_step and not check["differs"]
        self.summary["commit_checks_matched" if check["matched"] else "commit_checks_failed"] += 1
        if not check["matched"]:
            logging.warning("Steering commit check failed before decision %d: %s", self.decision, check)
        return check

    def _branch(self, step: int, candidate: int, history: Any, budget: int, region: Any, start: np.ndarray) -> dict[str, Any]:
        history, plan = copy.deepcopy(history), collections.deque()
        chunks: list[list[Any]] = []
        path, frames = [start], [(0, verifier.render_region(self.env, region))]
        contact_steps, clearance, task = 0, math.inf, {}
        for offset in range(budget):
            if not plan:
                noise = chunk_noise(self.config["sampling"], step, candidate, len(chunks))
                self.client.reset(instruction=self.instruction, model_profile=self.model_profile, policy_seed=noise["policy_seed"])
                scale = noise["noise_scale"]
                chunk, query = self.query(history, step + offset, {"noise_scale": scale} if scale != 1.0 else None)
                query["steering"] = {"decision": self.decision, "candidate": candidate, "branch_query_index": len(chunks), **noise}
                chunks.append([chunk, 0, query])
                plan.extend(chunk[: self.replan_steps])
            transition, _, robot, task = self.step(plan.popleft())
            chunks[-1][1] += 1
            path.append(np.asarray(robot["base_pos"], dtype=np.float64))
            contact_steps += any(task.get("obstacle_contacts", {}).values())
            clearance = min(clearance, task.get("min_obstacle_distance_m", math.inf))
            final = (self.stop_on_env_success and task["env_success"]) or offset == budget - 1
            # Only the observations a later query in this branch can sample are rendered.
            if not final and len(plan) < history.queue_length:
                history.append(step=step + offset + 1, **self.observe(transition[0]))
            if final or not plan:
                frames.append((offset + 1, verifier.render_region(self.env, region)))
            if final:
                break
        return {
            "chunks": chunks,
            "executed_steps": len(path) - 1,
            "env_success": bool(task.get("env_success", False)),
            "contact_steps": int(contact_steps),
            "min_obstacle_distance_m": None if math.isinf(clearance) else float(clearance),
            "path": np.stack(path),
            "frames": frames,
            "end_digest": snapshot.digest(snapshot.capture_env(self.env)),
        }

    def decide(self, step: int, history: Any, observation: dict[str, Any], steps_left: int) -> list[tuple[np.ndarray, int, dict[str, Any]]]:
        """Chunks to execute from ``step``, as (chunk, number of its actions to run, query record)."""
        started = time.perf_counter()
        record: dict[str, Any] = {"decision": self.decision, "step": step, "commit_check": self._commit_check(step)}
        start_state = snapshot.capture_env(self.env)
        start_digest = snapshot.digest(start_state)
        budget = min(self.config["lookahead_chunks"] * self.replan_steps, steps_left)
        start = np.asarray(observation["robot0_base_pos"], dtype=np.float64).reshape(3)
        region = self.camera.box_around(np.stack([start[:2], self.goal_xy]), verifier.REGION_MARGIN_M)
        obstacles = np.array([_data_array(self.env.sim.data, "geom_xpos")[ids].mean(axis=0) for ids in self.obstacle_geoms]).reshape(-1, 3)
        candidates = []
        for candidate in range(self.config["num_candidates"]):
            snapshot.restore_env(self.env, start_state)
            candidates.append(self._branch(step, candidate, history, budget, region, start))
        snapshot.restore_env(self.env, start_state)
        if snapshot.digest(snapshot.capture_env(self.env)) != start_digest:
            raise snapshot.SnapshotRestoreError(f"decision {self.decision}: restoring the start state did not reproduce it")

        points = np.concatenate([c["path"] for c in candidates] + [[[*self.goal_xy, 0.0]], obstacles])
        crop = verifier.intersect(self.camera.box_around(points, verifier.CROP_MARGIN_M), region)
        verifier_config = self.config["verifier"]
        select_with = verifier_config["select_with"]
        # Shuffled here, not in the verifier, so the Mode number can be written on the images it reads.
        order = verifier.mode_order(verifier_config["seed"], self.decision, len(candidates))
        mode_of = {candidate: position + 1 for position, candidate in enumerate(order)}
        # Only the strips some call will actually read are drawn: images narrates nothing, descriptions
        # sends no image.
        narrating, showing = select_with != "images", select_with != "descriptions"
        record_keys = ("executed_steps", "env_success", "contact_steps", "min_obstacle_distance_m", "end_digest")
        record_keys += ("filmstrip", "filmstrip_sha256") if narrating else ()
        record_keys += ("selection_image", "selection_image_sha256") if showing else ()
        pngs, labels, selection_pngs = [], [], []
        for index, c in enumerate(candidates):
            labels.append([verifier.time_label(offset, self.control_freq) for offset, _ in c["frames"]])
            if narrating:
                pngs.append(verifier.png_bytes(verifier.filmstrip(c["frames"], c["path"], self.camera, region, crop, self.control_freq)))
                c["filmstrip"] = str(self.image_dir / f"d{self.decision:02d}_cand{index:02d}.png")
                c["filmstrip_sha256"] = verifier.save_png(pngs[-1], c["filmstrip"])
            if showing:
                sheet = verifier.filmstrip(
                    c["frames"], c["path"], self.camera, region, crop, self.control_freq,
                    mode=mode_of[index], long_side=verifier.SELECTION_FRAME_LONG_SIDE,
                )
                selection_pngs.append(verifier.png_bytes(sheet))
                c["selection_image"] = str(self.image_dir / f"d{self.decision:02d}_mode{mode_of[index]:02d}.png")
                c["selection_image_sha256"] = verifier.save_png(selection_pngs[-1], c["selection_image"])
        chosen, selection = verifier.select(
            verifier_config, self.decision, self.instruction, pngs, labels, order, selection_pngs or None
        )
        fallback = chosen is None
        if fallback:
            logging.warning("Steering decision %d at step %d: verifier returned %s; executing candidate 0", self.decision, step, selection["status"])
            chosen = 0
        narrations = selection.get("narrations") or [None] * len(candidates)
        rows = [
            {**c, "mode": mode_of[index], "narration": narrations[index], "termination": "success" if c["env_success"] else "budget"}
            for index, c in enumerate(candidates)
        ]
        lines = [
            f"decision {self.decision} at step {step}: {len(candidates)} candidates, up to {budget} steps each",
            f"verifier {verifier_config['model']} ({select_with}): {selection['status']} -> candidate {chosen}" + (" (fallback)" if fallback else ""),
            f"instruction: {self.instruction}",
        ]
        overview_path = self.image_dir / f"d{self.decision:02d}_paths.png"
        image = verifier.overview(candidates[0]["frames"][0][1], [c["path"] for c in candidates], obstacles, self.goal_xy, self.camera, region, crop, rows, chosen, lines)
        verifier.save_png(verifier.png_bytes(image), overview_path)

        for chunk, count, query in candidates[chosen]["chunks"]:
            query["steering"].update(status=selection["status"], fallback=fallback)
        self.expected = (step + candidates[chosen]["executed_steps"], candidates[chosen]["end_digest"])
        record.update(
            budget_steps=budget,
            candidates=[
                {
                    "queries": [
                        {key: q["steering"][key] for key in ("noise_draw", "policy_seed", "noise_scale")} | {"query_step": q["input_audit"]["query_step"]}
                        for _, _, q in c["chunks"]
                    ],
                    **{key: c[key] for key in record_keys},
                    "path_xy_mm": np.rint(c["path"][:, :2] * 1000).astype(int).tolist(),
                }
                for c in candidates
            ],
            selection=selection,
            executed_candidate=chosen,
            fallback=fallback,
            overview=str(overview_path),
            seconds=round(time.perf_counter() - started, 3),
        )
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.summary["decisions"] += 1
        self.summary["fallbacks"] += fallback
        self.decision += 1
        return [(chunk, count, query) for chunk, count, query in candidates[chosen]["chunks"]]

    def finalize(self, step: int) -> dict[str, Any]:
        """Check the last replay and return the per-episode counts."""
        self.summary["final_commit_check"] = self._commit_check(step)
        return dict(self.summary)
