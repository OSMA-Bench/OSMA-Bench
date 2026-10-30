"""Skill-based execution scaffolding (multi-skill LMP, Phase 0).

General framework that runs Track A (single manip), Track B (nav+manip), and
Track C (multi-stage) through ONE executor loop with explicit per-stage state.

Design: docs/superpowers/specs/2026-07-15-multiskill-lmp-design.md

Phase 0 is intentionally TRANSPARENT: with VOX_SKILL_EXEC=1 the executor runs the
existing planner LMP as a single whole-instruction skill, so behavior is identical
to the current path. Later phases split the instruction into real skills, add
stage-boundary workspace re-init, per-stage success/retry, and state handoff.

IMPORTANT: costmaps are NOT unified. Each skill keeps its own planner + space,
selected by Skill.kind ('nav' = 2D floor, 'manip' = 3D voxel). EpisodeState is the
only thing that crosses a skill boundary.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from utils.utils import get_logger

logger = get_logger(__name__)

MAX_RETRY = int(os.environ.get('VOX_SKILL_MAX_RETRY', '2'))  # attempts per stage (grasp retries on miss)


@dataclass
class EpisodeState:
    """Explicit state handed off between skills (replaces implicit env flags).

    Only this object crosses a skill boundary — never a costmap or workspace box.
    """
    held_obj: Optional[str] = None            # object currently in the gripper
    opened_fixtures: set = field(default_factory=set)   # doors/drawers opened
    base_pose: Optional[tuple] = None         # (x, y, yaw) of the mobile base
    last_ee_world: Optional[tuple] = None     # last EE world xyz (bounds sanity)

    @classmethod
    def initial(cls, env) -> "EpisodeState":
        """Seed from the real reset pose; tolerate a headless/partial env."""
        ee = base = None
        try:
            ee = tuple(np.asarray(env.get_ee_pos_world(), dtype=float)) \
                if hasattr(env, 'get_ee_pos_world') else tuple(np.asarray(env.get_ee_pos(), dtype=float))
        except Exception:
            pass
        try:
            bid = env.env.sim.model.body_name2id('mobilebase0_base')
            bp = np.asarray(env.env.sim.data.body_xpos[bid], dtype=float)
            base = (float(bp[0]), float(bp[1]), 0.0)
        except Exception:
            pass
        return cls(held_obj=None, opened_fixtures=set(), base_pose=base, last_ee_world=ee)


@dataclass
class Skill:
    """One unit of execution. kind selects planner + coordinate space + costmap.

    - compose(state): emit + run this skill's sub-program (side-effecting).
    - success(env, state) -> bool: per-stage predicate (P3 wires real checks).
    - workspace: 'recenter' (nav skills re-init the manip box) | 'reuse'.
    """
    name: str
    kind: str                                 # 'nav' | 'manip'
    compose: Callable[["EpisodeState"], object]
    success: Callable[[object, "EpisodeState"], bool] = lambda env, st: True
    workspace: str = 'reuse'                  # 'recenter' | 'reuse'
    update: Callable[["EpisodeState", object], "EpisodeState"] = lambda st, env: st


class Executor:
    """Runs an ordered skill list with per-stage workspace/state/retry handling.

    Phase 0: the only skill is the whole instruction via the existing planner, so
    the loop is a transparent pass-through. Phases 1+ populate real skills.
    """

    def __init__(self, env, snapshot_dir: Optional[str] = None):
        self._env = env
        self._snapshot_dir = snapshot_dir       # stage snapshots (viz + resume)

    # ---- stage snapshots -----------------------------------------------------
    # One .npz (qpos/qvel) + journal.json entry per completed stage. Dual use:
    #  1. viz: render_scene_pointcloud_html restores the qpos so the rendered arm
    #     matches that stage's start marker (fixes the start-vs-gripper offset).
    #  2. resume: on an outer retry (VOX_SKILL_RESUME=1) restore the last
    #     completed stage and skip straight to the failed one instead of
    #     re-running the whole episode from reset.
    def _sim(self):
        real = getattr(self._env, '_env', self._env)
        return getattr(real, 'env', real).sim

    # logical fixture flags that live OUTSIDE qpos (python-side state a qpos
    # restore misses — e.g. the microwave's turned_on toggled by button press).
    _FIXTURE_FLAGS = ('_turned_on',)

    def _base_env(self):
        real = getattr(self._env, '_env', self._env)
        return getattr(real, 'env', real)

    def _fixture_states(self) -> dict:
        out = {}
        try:
            for name, fx in (getattr(self._base_env(), 'fixtures', {}) or {}).items():
                st = {a: bool(getattr(fx, a)) for a in self._FIXTURE_FLAGS if hasattr(fx, a)}
                if st:
                    out[name] = st
        except Exception:
            pass
        return out

    def _restore_fixture_states(self, states: dict):
        try:
            fixtures = getattr(self._base_env(), 'fixtures', {}) or {}
            for name, st in (states or {}).items():
                fx = fixtures.get(name)
                if fx is None:
                    continue
                for a, v in st.items():
                    if hasattr(fx, a):
                        setattr(fx, a, bool(v))
        except Exception as e:
            logger.debug(f'[skill-resume] fixture-state restore failed: {e}')

    def _align_base_for_place(self, skill):
        """Pre-position the base at RoboCasa's canonical use-pose for the target
        fixture BEFORE the place program runs. A diagonal approach dead-reckoned
        by navigate_to stalls against the opened door / counter (thaw place: EE
        pinned at z1.15 with the torso already maxed at 0.34 — the geometry, not
        reach, was the blocker). compute_robot_base_placement_pose is the env's
        own 'where to stand to use this fixture' answer; the in-program
        navigate_to then sees its stop_dist already satisfied and stays put."""
        try:
            tgt = _place_target(skill.name)
            base_env = self._base_env()
            fx = getattr(base_env, tgt, None) if tgt else None
            logger.info(f'[skill-align] enter: tgt={tgt} fx={"ok" if fx is not None else None} '
                        f'has_crbpp={hasattr(base_env, "compute_robot_base_placement_pose")}')
            if fx is None or not hasattr(base_env, 'compute_robot_base_placement_pose'):
                logger.warning(f'[skill-align] SKIP (fx={fx is not None}, crbpp={hasattr(base_env, "compute_robot_base_placement_pose")}) — base stays put')
                return
            pos, ori = base_env.compute_robot_base_placement_pose(fx)
            real = getattr(self._env, '_env', self._env)
            if not hasattr(real, 'navigate_base_to'):
                logger.warning('[skill-align] SKIP — env has no navigate_base_to')
                return
            _b0 = np.asarray(base_env.sim.data.body_xpos[base_env.sim.model.body_name2id('mobilebase0_base')][:2], dtype=float)
            logger.info(f'[skill-align] pre-place base {np.round(_b0,2)} -> use-pose {np.round(pos[:2], 2)} (yaw {ori[2]:.2f}) for {tgt}')
            # aggressive: tighter stop_dist + more steps so the base actually
            # reaches the 20cm use-pose stance (0.15 left it ~0.45m short — the
            # arm then can't reach the high cavity).
            real.navigate_base_to(np.asarray(pos[:2], dtype=float), stop_dist=0.06, max_steps=400)
            _b1 = np.asarray(base_env.sim.data.body_xpos[base_env.sim.model.body_name2id('mobilebase0_base')][:2], dtype=float)
            logger.info(f'[skill-align] base after nav: {np.round(_b1,2)} (moved {np.linalg.norm(_b1-_b0):.2f}m, {np.linalg.norm(_b1-pos[:2]):.2f}m from use-pose)')
        except Exception as e:
            logger.warning(f'[skill-align] failed: {e}')

    def _release_gripper(self, settle: int = 40):
        """Open the gripper and let physics settle so a held object drops onto /
        into the target. Also clears the env holding flag (which deactivates the
        opt-in weld). Used at the end of every place skill."""
        try:
            base = self._base_env()
            real = getattr(self._env, '_env', self._env)
            # DESCEND to the surface first: release too high leaves the object
            # hovering ~1cm over the counter (contact check fails). Lower the arm
            # (grip still closed) until the OBJECT stops descending = it rests on
            # the surface, then open.
            def _objz():
                try:
                    return float(base.sim.data.body_xpos[base.obj_body_id['obj']][2])
                except Exception:
                    return None
            _pz = _objz()
            for _ in range(40):
                o, _, _, _ = base.step(np.concatenate([[0, 0, -0.12], [0, 0, 0], [1.0], [0, 0], [0], [-0.3]]))
                real.latest_obs = o
                _cz = _objz()
                if _cz is not None and _pz is not None and abs(_cz - _pz) < 0.002:
                    break                              # object stopped = on surface
                _pz = _cz
            for _ in range(20):
                o, _, _, _ = base.step(np.concatenate([[0, 0, 0], [0, 0, 0], [-1.0], [0, 0], [0], [0]]))
                real.latest_obs = o
            real._holding_obj = False
            real._last_rs_grip = -1.0
            # RETRACT the arm up-and-back so the object settles onto the target
            # and gripper_obj_far() passes (place success needs the gripper >=25cm
            # from the object). Torso + arm-up, gripper held open.
            for _ in range(45):
                o, _, _, _ = base.step(np.concatenate([[0, 0, 0.4], [0, 0, 0], [-1.0], [0, 0], [0], [0.8]]))
                real.latest_obs = o
            for _ in range(settle):
                o, _, _, _ = base.step(np.concatenate([[0, 0, 0], [0, 0, 0], [-1.0], [0, 0], [0], [0]]))
                real.latest_obs = o
            logger.info('[skill-place] released gripper + retracted + settled')
        except Exception as e:
            logger.warning(f'[skill-place] release failed: {e}')

    def _force_release_near(self, skill) -> bool:
        """Last-resort place completion: the env's release gate only opens the
        gripper within 8cm of the final waypoint, but at the base's reach limit
        the EE stalls ~6-15cm short of a cavity target — the object is hovering
        at the fixture mouth yet never released (thaw stage-2: held_obj stayed
        'mushroom' through every retry). If the failed place left the EE close
        above/at the target fixture, open the gripper directly so the object
        drops in, then lift clear."""
        try:
            tgt = _place_target(skill.name)
            base_env = self._base_env()
            fx = getattr(base_env, tgt, None) if tgt else None
            if fx is None or not hasattr(fx, 'pos'):
                return False
            real = getattr(self._env, '_env', self._env)
            ee = np.asarray(real.latest_obs['robot0_eef_pos'], dtype=float)
            fxp = np.asarray(fx.pos, dtype=float)
            if np.linalg.norm(ee[:2] - fxp[:2]) > 0.35 or abs(ee[2] - fxp[2]) > 0.40:
                logger.info(f'[skill-release] EE not at {tgt} mouth (ee={ee.round(2)} fx={fxp.round(2)}) — skip')
                return False
            logger.warning(f'[skill-release] forcing release at {tgt} mouth (EE {np.linalg.norm(ee - fxp):.2f}m from fixture ref)')
            # PUSH-IN first: releasing at 0.38m from the body center (~13cm
            # OUTSIDE the opening plane) drops the payload onto the counter in
            # front (n=3 reproduced), and 0.24m still teeters it on the sill.
            # Aim at the CAVITY INTERIOR CENTER (get_int_sites box), not the
            # body center, and push until the EE is well inside its footprint.
            _aim = fxp[:2]
            try:
                _p0, _px, _py, _pz = fx.get_int_sites(relative=False)
                _ctr = (np.asarray(_px) + np.asarray(_py)) / 2.0  # box face midpoint ≈ interior center xy
                _aim = np.asarray(_ctr[:2], dtype=float)
                logger.info(f'[skill-release] aiming interior center {np.round(_aim, 2)}')
            except Exception:
                pass
            if hasattr(real, '_world_dpos_to_arm_cmd'):
                _short = True
                for _ in range(70):
                    ee = np.asarray(real.latest_obs['robot0_eef_pos'], dtype=float)
                    d = _aim - ee[:2]
                    if np.linalg.norm(d) < 0.15:
                        logger.info(f'[skill-release] pushed to {np.linalg.norm(d):.2f}m of interior center — releasing inside')
                        _short = False
                        break
                    c = real._world_dpos_to_arm_cmd(np.array([d[0], d[1], 0.0]))
                    o, _, _, _ = base_env.step(np.concatenate([
                        np.clip(c / 0.08, -0.4, 0.4), [0, 0, 0], [1.0], [0, 0], [0], [0]]))
                    real.latest_obs = o
                if _short:
                    # arm is at its kinematic limit (0.38m stall reproduced with
                    # zero push progress) — the remaining ~20cm can only come
                    # from MOMENTUM: open the gripper WHILE driving forward so
                    # the payload slides over the sill instead of dropping at it.
                    logger.warning('[skill-release] reach-limited — releasing ON THE MOVE (momentum slide)')
                    ee = np.asarray(real.latest_obs['robot0_eef_pos'], dtype=float)
                    d = _aim - ee[:2]
                    dn = d / (np.linalg.norm(d) + 1e-9)
                    for _k in range(12):
                        c = real._world_dpos_to_arm_cmd(np.array([dn[0] * 0.06, dn[1] * 0.06, 0.0]))
                        _g = 1.0 if _k < 4 else -1.0     # open mid-stroke
                        o, _, _, _ = base_env.step(np.concatenate([
                            np.clip(c / 0.08, -0.6, 0.6), [0, 0, 0], [_g], [0, 0], [0], [0]]))
                        real.latest_obs = o
            for _ in range(25):
                o, _, _, _ = base_env.step(np.concatenate([[0, 0, 0], [0, 0, 0], [-1.0], [0, 0], [0], [0]]))
                real.latest_obs = o
            real._holding_obj = False
            real._last_rs_grip = -1.0
            # lift clear so the wrist doesn't pin the object / block the door
            for _ in range(20):
                o, _, _, _ = base_env.step(np.concatenate([[0, 0, 0.4], [0, 0, 0], [-1.0], [0, 0], [0], [0]]))
                real.latest_obs = o
            return True
        except Exception as e:
            logger.warning(f'[skill-release] failed: {e}')
            return False

    def _ep_hash(self) -> str:
        """hash of the sim MODEL signature (nq + body names). This is the real
        invariant a qpos snapshot needs: same model → restore valid (placement is
        overwritten by the qpos itself). Hashing get_ep_meta() was too brittle —
        its json round-trip (default=str) re-serializes differently across
        processes and produced false 'episode changed' mismatches."""
        try:
            import hashlib
            sim = self._sim()
            names = ','.join(sorted(sim.model.body_id2name(i) or '' for i in range(sim.model.nbody)))
            sig = f'{sim.model.nq}|{names}'
            return hashlib.sha1(sig.encode()).hexdigest()[:12]
        except Exception:
            return ''

    def _save_snapshot(self, i: int, skill, state: EpisodeState, ok: bool):
        if not self._snapshot_dir:
            return
        try:
            os.makedirs(self._snapshot_dir, exist_ok=True)
            sim = self._sim()
            # ctrl/act too: qpos alone puts the fingers at the closed POSITION but
            # the position-actuator target stays at its default — the first physics
            # step then re-opens the grip and drops the payload (validated: resumed
            # mushroom fell to the counter, carry-lost xy-gap 0.25m from step 1).
            np.savez(os.path.join(self._snapshot_dir, f'stage_{i}.npz'),
                     qpos=np.asarray(sim.data.qpos), qvel=np.asarray(sim.data.qvel),
                     ctrl=np.asarray(sim.data.ctrl),
                     act=np.asarray(sim.data.act) if sim.data.act is not None else np.zeros(0))
            jf = os.path.join(self._snapshot_dir, 'journal.json')
            import json as _json
            import glob as _glob
            import re as _re
            # count plan htmls emitted so far: stage i+1's FIRST plan html is
            # htmls[n_htmls] — lets the viz pair a snapshot pose with the right
            # per-stage value maps.
            _out = os.path.dirname(self._snapshot_dir)
            _n_htmls = len([f for f in _glob.glob(os.path.join(_out, '*.html'))
                            if _re.match(r'^\d', os.path.basename(f))])
            j = _json.load(open(jf)) if os.path.exists(jf) else {'stages': {}}
            j['ep_hash'] = self._ep_hash()
            j['stages'][str(i)] = {'name': skill.name, 'ok': ok,
                                   'held_obj': state.held_obj,
                                   'workspace': skill.workspace,
                                   'n_htmls': _n_htmls,
                                   'fixture_states': self._fixture_states()}
            _json.dump(j, open(jf, 'w'), indent=1)
        except Exception as e:
            logger.debug(f'[skill-snapshot] save failed: {e}')

    def _try_resume(self, skills) -> int:
        """Return the stage index to start from (0 = fresh). Restores sim state
        of the last stage whose journal entry is ok=True and whose name matches
        the current decomposition (guards against a different plan)."""
        if os.environ.get('VOX_SKILL_RESUME') != '1' or not self._snapshot_dir:
            return 0
        jf = os.path.join(self._snapshot_dir, 'journal.json')
        if not os.path.exists(jf):
            return 0
        try:
            import json as _json
            j = _json.load(open(jf))
            # snapshots are episode-specific: a reset that re-sampled ep_meta
            # (different objects/placement) invalidates every stored qpos.
            if j.get('ep_hash') and j['ep_hash'] != self._ep_hash():
                logger.info('[skill-resume] ep_meta changed since snapshots — starting fresh')
                return 0
            # resume only past a CONTIGUOUS prefix of ok stages: an early failed
            # stage (e.g. grasp missed) leaves episode state that later "ok"
            # stages can't repair — resuming after them would skip the fix.
            best = -1
            for i in range(len(skills)):
                ent = j.get('stages', {}).get(str(i))
                if (ent and ent.get('ok') and ent.get('name') == skills[i].name
                        and os.path.exists(os.path.join(self._snapshot_dir, f'stage_{i}.npz'))):
                    best = i
                else:
                    break
            if best < 0:
                return 0
            # always re-run at least the LAST stage: skipping everything leaves
            # zero env steps in this process and trips run_LMP's no-actuation
            # guard; re-running the final stage from its entry pose is cheap.
            best = min(best, len(skills) - 2)
            # never resume INTO a holding state: a restored contact grasp does
            # not survive base motion (friction anchors live outside
            # qpos/qvel/ctrl — validated: payload dropped within the first
            # ~80 nav steps every time). Walk back to a hands-free snapshot and
            # re-run the grasp fresh instead.
            while best >= 0 and j['stages'].get(str(best), {}).get('held_obj'):
                logger.info(f'[skill-resume] stage {best} snapshot holds a payload — backing off one stage (fresh re-grasp)')
                best -= 1
            if best < 0:
                return 0
            snap = np.load(os.path.join(self._snapshot_dir, f'stage_{best}.npz'))
            sim = self._sim()
            sim.data.qpos[:] = snap['qpos']
            sim.data.qvel[:] = snap['qvel']
            if 'ctrl' in snap and snap['ctrl'].shape == np.asarray(sim.data.ctrl).shape:
                sim.data.ctrl[:] = snap['ctrl']
            if 'act' in snap and snap['act'].size and sim.data.act is not None \
                    and snap['act'].shape == np.asarray(sim.data.act).shape:
                sim.data.act[:] = snap['act']
            sim.forward()
            # settle a few frames with the restored actuator targets so grasp
            # contacts re-establish BEFORE any policy motion.
            try:
                for _ in range(10):
                    sim.step()
            except Exception:
                pass
            # python-side fixture flags (turned_on etc.) are NOT in qpos —
            # restore them from the journal or the resumed run stays logically
            # at reset (validated: mpb resume pressed-button state was lost).
            self._restore_fixture_states(j['stages'].get(str(best), {}).get('fixture_states'))
            real = getattr(self._env, '_env', self._env)
            # holding flags are python-side too: without them the next nav step
            # commands the gripper OPEN (_last_rs_grip default) and drops the
            # restored payload immediately.
            if j['stages'].get(str(best), {}).get('held_obj'):
                real._holding_obj = True
                real._last_rs_grip = 1.0
            if hasattr(real, 'update_latest_obs'):
                real.update_latest_obs()
            logger.info(f'[skill-resume] restored stage {best} snapshot — resuming at stage {best + 1}')
            return best + 1
        except Exception as e:
            logger.warning(f'[skill-resume] restore failed ({e}) — starting fresh')
            return 0

    def run(self, skills, state: EpisodeState) -> EpisodeState:
        last_grasp = None                       # for drop recovery (re-grasp then re-place)
        start_at = self._try_resume(skills)
        if start_at > 0:
            # rebuild handoff state from the journal (held_obj at that point)
            try:
                import json as _json
                j = _json.load(open(os.path.join(self._snapshot_dir, 'journal.json')))
                ent = j['stages'].get(str(start_at - 1), {})
                state.held_obj = ent.get('held_obj')
            except Exception:
                pass
        for i, skill in enumerate(skills):
            if i < start_at:
                logger.info(f"[skill-exec] stage {i}: {skill.name} — SKIPPED (resumed)")
                if _is_grasp(skill.name):
                    last_grasp = skill
                continue
            logger.info(f"[skill-exec] stage {i}: {skill.name} (kind={skill.kind}, ws={skill.workspace})")
            real_env = getattr(self._env, '_env', self._env)
            if _is_place(skill.name):
                self._align_base_for_place(skill)
                # hint for the env's place-into clamp (waypoint z pinned into the
                # fixture's cavity interior — see robocasa_env [place-clamp])
                real_env._place_fixture_name = _place_target(skill.name)
                real_env._fixture_clamp_lo = 0.05    # keep payload off the floor
            elif _is_grasp(skill.name) and _from_fixture(skill.name):
                # extraction FROM an enclosed fixture: same interior z-band —
                # approach and retreat ride at cavity height so the payload
                # clears the opening frame instead of scraping the top lip
                # (Track-B PnPCabToCounter failure mode).
                real_env._place_fixture_name = _from_fixture(skill.name)
                real_env._fixture_clamp_lo = 0.0     # EE must dip to floor-lying objects
                logger.info(f'[skill-exec] extraction clamp armed for fixture {real_env._place_fixture_name}')
            else:
                real_env._place_fixture_name = None
            if skill.workspace == 'recenter' and hasattr(self._env, 'reinit_workspace'):
                # P2 hardens this; P0 has no recenter skills so this is inert.
                self._env.reinit_workspace()
            ok = False
            for attempt in range(MAX_RETRY):
                skill.compose(state)
                # PLACE RELEASE: some place codegen (e.g. 'place on the counter')
                # never opens the gripper — without a weld the object slipped out
                # accidentally, but a firm/welded grip then just hovers it over
                # the target. Force an open + settle at the end of every place so
                # the object actually comes to rest in/on the target.
                if _is_place(skill.name) and _env_holding(self._env):
                    self._release_gripper()
                ok = bool(skill.success(self._env, state))
                if ok:
                    break
                logger.warning(f"[skill-retry] {skill.name} attempt {attempt + 1} failed")
                # RELEASE-AT-MOUTH: place finished with the object STILL gripped
                # and the EE parked at the fixture mouth → the release gate never
                # fired (reach-limited convergence). Drop it in directly.
                if _is_place(skill.name) and _env_holding(self._env):
                    if self._force_release_near(skill):
                        ok = bool(skill.success(self._env, state))
                        if ok:
                            break
                # DROP RECOVERY: a place that failed with nothing in the gripper
                # means the object fell short — re-running the place alone can't
                # recover (gripper is empty), so re-grasp first, then re-place.
                if _is_place(skill.name) and last_grasp is not None and not _env_holding(self._env):
                    logger.warning(f"[skill-recover] object dropped — re-running grasp '{last_grasp.name}' before re-place")
                    last_grasp.compose(state)
                    state = last_grasp.update(state, self._env)
            if _is_grasp(skill.name):
                last_grasp = skill
            state = skill.update(state, self._env)
            self._save_snapshot(i, skill, state, ok)
            logger.info(f"[skill-exec] stage {i} {skill.name} done ok={ok} held_obj={state.held_obj}")
        return state


def decompose(instruction: str, lmps: dict):
    """Run the planner but intercept composer('..') calls to recover the ordered
    sub-goal list WITHOUT executing them. Behaviorally transparent: the same
    sub-goals are then run by the executor via the real composer.

    Returns a list of sub-goal strings (planner stage order).
    """
    plan = lmps['plan_ui']
    vv = getattr(plan, '_variable_vars', None)
    if vv is None or 'composer' not in vv:
        return None                      # unknown planner shape → caller falls back
    real_composer = vv['composer']
    recorded = []

    def recorder(subgoal, *a, **k):
        recorded.append(str(subgoal))

    vv['composer'] = recorder
    try:
        plan(instruction)                # planner emits composer(..) → recorded
    finally:
        vv['composer'] = real_composer
    return recorded


def _classify(subgoal: str) -> tuple:
    """(kind, workspace) for a sub-goal. nav sub-steps live INSIDE the composer
    program (navigate_to), so a stage that navigates gets workspace='recenter'."""
    s = subgoal.lower()
    if 'navigate' in s or 'go to' in s or 'drive to' in s:
        return 'manip', 'recenter'       # nav+manip stage; recenter at its boundary
    return 'manip', 'reuse'


import re as _re_split
# The planner uses both "navigate" and "move" for base-only tasks.  Keep the
# target matcher deliberately narrow: modifiers such as "while keeping 40cm
# from ..." belong to the same navigation skill, not to the fixture name.
_NAV_PREFIX_RE = _re_split.compile(
    r'^(?:navigate|go|drive|move)\s+to\s+the\s+(.+?)(?=\s+(?:and|while|with|keeping|avoiding|at|by)\b|$)',
    _re_split.I,
)
_SPLIT_ACTION_RE = _re_split.compile(
    r'\s*(?:,|\band\b)\s*(?=(?:open|close|grasp|pick|press|turn|place|put|pull|push)\b)',
    _re_split.I,
)


def _split_single_skills(subgoal: str):
    """Split ONE planner subgoal into ordered SINGLE-skill parts so each stage is
    a pure nav OR a pure manip primitive (never a combined 'navigate X and do Y').

    Splits only before a *new manipulation action*:
      'navigate to the microwave and open the microwave door'
      'navigate to the microwave, open the microwave door'
        -> [('navigate to the microwave', 'nav',  None),
            ('open the microwave door',   'manip','microwave')]

    Do not split navigation constraints. In particular, the planner commonly
    emits "move to the fridge while keeping at least 40cm from the dishwasher,
    at least 30cm from the sink, ...". Splitting that list at commas creates
    nonsensical manipulation stages and prevents the navigation stage from
    completing.

    A non-composite subgoal passes through unchanged as one part. The nav part's
    fixture is threaded to any following manip part that doesn't name a fixture
    itself, so the extraction/place clamps still arm (Track-B lesson: a bare
    'grasp the hot dog' loses the 'cabinet' context needed for the z-band clamp).

    Returns list of (text, kind, nav_ctx_fixture-or-None).
    """
    raw = _SPLIT_ACTION_RE.split(subgoal.strip())
    parts = [p.strip() for p in raw if p.strip()]
    if len(parts) <= 1:
        kind = 'nav' if _NAV_PREFIX_RE.match(subgoal.strip()) else 'manip'
        return [(subgoal.strip(), kind, None)]
    out = []
    nav_fx = None
    for p in parts:
        m = _NAV_PREFIX_RE.match(p)
        if m:
            nav_fx = m.group(1).strip()          # e.g. 'microwave', 'counter'
            out.append((p, 'nav', None))
        else:
            out.append((p, 'manip', nav_fx))
    return out


def _is_grasp(subgoal: str) -> bool:
    s = subgoal.lower()
    return ('grasp' in s or 'pick' in s) and 'place' not in s and 'drawer handle' not in s


def _from_fixture(subgoal: str) -> Optional[str]:
    """source fixture of a 'grasp X from the <fixture>' stage — extraction from
    an enclosed fixture needs the same interior z-band clamp as insertion (the
    Track-B failure: payload scraped off on the cabinet opening frame while the
    arm lifted during retreat)."""
    # ONLY enclosed fixtures need the interior z-band on extraction; open
    # surfaces (counter, stove) have a degenerate cavity and must NOT be
    # clamped (would pin the grasp to the surface plane).
    _ENCLOSED = ('cabinet', 'cab', 'microwave', 'fridge')
    import re as _re
    m = _re.search(r'from the ([a-z ]+?)(?:\s+and|\s*$)', subgoal.lower())
    if m:
        frag = m.group(1)
        for fx in _ENCLOSED:
            if fx in frag:
                return 'cab' if fx == 'cabinet' else fx
    # planner often phrases it 'navigate to the cabinet and grasp X' (no
    # 'from the') — any mention of an enclosed fixture in a grasp stage means
    # the payload must come out through its opening frame.
    s = subgoal.lower()
    for fx in _ENCLOSED:
        if fx in s:
            return 'cab' if fx == 'cabinet' else fx
    return None


def _env_holding(lmp_env) -> bool:
    return bool(getattr(getattr(lmp_env, '_env', None), '_holding_obj', False))


def _grasp_object_name(subgoal: str) -> Optional[str]:
    """best-effort object name from 'grasp the X' / 'pick the X ...'."""
    import re as _re
    m = _re.search(r'(?:grasp|pick)\s+(?:the\s+|up\s+the\s+)?([a-z ]+?)(?:\s+from|\s+on|\s+in|$)', subgoal.lower())
    return m.group(1).strip() if m else None


_FIXTURES = ('microwave', 'cab', 'cabinet', 'sink', 'stove', 'counter', 'fridge')


def _is_place(subgoal: str) -> bool:
    return 'place' in subgoal.lower() or 'put' in subgoal.lower()


def _place_target(subgoal: str) -> Optional[str]:
    """fixture name after 'in/inside/into the ...' — e.g. 'microwave'."""
    s = subgoal.lower()
    for fx in _FIXTURES:
        if fx in s:
            return 'cab' if fx == 'cabinet' else fx
    return None


def _obj_inside(env, fixture_attr: str) -> bool:
    """Real placement check via robocasa object_utils (same predicate the task's
    _check_success uses). False on any error → treated as not-yet-placed."""
    try:
        from robocasa.utils import object_utils as _OU
        real = getattr(env, '_env', env)
        base = getattr(real, 'env', real)
        fx = getattr(base, fixture_attr, None)
        if fx is None:
            return False
        return bool(_OU.obj_inside_of(base, 'obj', fx))
    except Exception:
        return False


def _make_success(subgoal: str):
    """Per-stage predicate. grasp → env._holding_obj; place → obj actually inside
    the named fixture (catches 'dropped short of target'). Others default True —
    a false-negative retry costs more than a missed intermediate check, and the
    task's own _check_success still gates overall success."""
    if _is_grasp(subgoal):
        return lambda env, st: _env_holding(env)
    if _is_place(subgoal):
        tgt = _place_target(subgoal)
        if tgt:
            return lambda env, st, _t=tgt: _obj_inside(env, _t)
    return lambda env, st: True


def _make_update(subgoal: str):
    """Sync EpisodeState.held_obj FROM the env's own grasp tracking (env already
    manages _holding_obj + carry gentleness). Keeps framework state accurate for
    logging / future skill decisions without duplicating the grasp logic."""
    def _upd(st: EpisodeState, env) -> EpisodeState:
        holding = _env_holding(env)
        if holding and _is_grasp(subgoal):
            st.held_obj = _grasp_object_name(subgoal) or 'obj'
        elif not holding:
            st.held_obj = None
        return st
    return _upd


def _reorder_open_before_grasp(subgoals):
    """Deterministic plan repair: an 'open <fixture>' stage that sits AFTER a
    grasp but services a LATER 'place ... <fixture>' must run BEFORE the grasp —
    opening a door with an object in the gripper drops it (validated: thaw
    stage-1 door-open lost the mushroom, held_obj=None). The LLM mirrors the
    instruction order ('pick X ... place in microwave') even though the prompt
    canonical is door-first, and cache-off reproduces it — so we repair the
    order here instead of fighting the prompt."""
    idx_grasp = next((i for i, s in enumerate(subgoals) if _is_grasp(s)), None)
    if idx_grasp is None:
        return subgoals
    moved = []
    for i, s in enumerate(subgoals):
        sl = s.lower()
        if i > idx_grasp and 'open' in sl and 'gripper' not in sl:
            fx = _place_target(s)
            if fx and any(_is_place(t) and _place_target(t) == fx
                          for t in subgoals[i + 1:]):
                moved.append(i)
    if not moved:
        return subgoals
    reordered = [subgoals[i] for i in moved] + [s for i, s in enumerate(subgoals) if i not in moved]
    logger.info(f"[skill-exec] plan repair: moved open-stages {moved} before grasp -> {reordered}")
    return reordered


def run_instruction(instruction: str, lmps: dict, lmp_env) -> EpisodeState:
    """Entry point used by run_LMP when VOX_SKILL_EXEC=1.

    P1: decompose the planner output into an explicit skill list, then run each
    sub-goal via the real composer through the executor loop. Falls back to the
    P0 whole-instruction skill if decomposition is unavailable.
    """
    state = EpisodeState.initial(lmp_env)
    snap_dir = os.path.join(getattr(lmp_env, '_output_dir', '.'), 'stage_snapshots')
    subgoals = None
    try:
        subgoals = decompose(instruction, lmps)
    except Exception as e:
        logger.warning(f"[skill-exec] decompose failed ({e}); falling back to whole-instruction")

    if not subgoals:
        whole = Skill('whole-instruction', 'manip',
                      compose=lambda st: lmps['plan_ui'](instruction),
                      workspace='reuse')
        return Executor(lmp_env, snap_dir).run([whole], state)

    # drop 'back to default pose' cleanup stages: no success predicate depends
    # on them (place/press canonicals carry their own retreat) and they cost
    # 1-2 min per episode.
    _n0 = len(subgoals)
    subgoals = [s for s in subgoals if 'back to default' not in s.lower()]
    if len(subgoals) < _n0:
        logger.info(f'[skill-exec] dropped {_n0 - len(subgoals)} default-pose stage(s)')
    # The planner/cache can reuse the open-door canonical for a close task.
    # Trust the task instruction and repair this opposite-direction plan before
    # the composer executes it.
    _instr_lc = instruction.lower()
    if ('close' in _instr_lc and 'door' in _instr_lc
            and any('open' in s.lower() for s in subgoals)
            and not any('close' in s.lower() for s in subgoals)):
        _old = list(subgoals)
        subgoals = [s.replace('open', 'close').replace('Open', 'Close') for s in subgoals]
        logger.info(f'[skill-exec] close-task plan repair: {_old} -> {subgoals}')
    # Atomic CloseSingleDoor instructions are occasionally served a stale
    # multi-stage planner cache entry (navigate/open/place/close).  Do not run
    # those unrelated stages: they can change the initial door state and make
    # the close primitive impossible to evaluate.  Composite instructions are
    # unaffected because they contain additional action verbs beyond close.
    if (_re_split.match(r'^\s*(?:safely\s+)?close\s+(?:the\s+)?(?:[a-z ]+\s+)?door\s*$', _instr_lc)
            and len(subgoals) > 1 and any('close' in s.lower() for s in subgoals)):
        _old = list(subgoals)
        _nav = [s for s in subgoals if ('navigate' in s.lower() or 'go to' in s.lower())]
        _close = [s for s in subgoals if 'close' in s.lower()][-1:]
        if _close:
            subgoals = _nav[-1:] + _close
            logger.info(f'[skill-exec] atomic-close cache repair: {_old} -> {subgoals}')
    subgoals = _reorder_open_before_grasp(subgoals)
    real_composer = lmps['plan_ui']._variable_vars['composer']
    logger.info(f"[skill-exec] decomposed into {len(subgoals)} skills: {subgoals}")
    skills = []
    if os.environ.get('VOX_SKILL_SPLIT', '1') == '1':   # default-on; VOX_SKILL_SPLIT=0 to disable
        # SINGLE-SKILL decomposition: every combined 'navigate to the X and <rest>'
        # (or comma-joined) subgoal becomes a pure NAV skill (deterministic —
        # navigate_to directly, no LLM) followed by pure MANIP skill(s) (composer
        # on the manip fragment only). Cleaner 2D/3D boundary, per-part retry, and
        # the nav fixture context is threaded to a bare manip part so extraction/
        # place clamps still arm even though '<rest>' no longer names the fixture.
        for sg in subgoals:
            for text, kind, nav_fx in _split_single_skills(sg):
                if kind == 'nav':
                    m = _NAV_PREFIX_RE.match(text)
                    nav_query = m.group(1).strip() if m else text
                    skills.append(Skill(name=f'navigate to the {nav_query}', kind='nav',
                                        compose=(lambda st, _q=nav_query: lmp_env.navigate_to(_q)),
                                        workspace='recenter'))
                    continue
                # manip: thread the nav fixture into the text for clamp/succ parsing
                ctx = text
                if nav_fx and not any(f in text.lower() for f in _FIXTURES):
                    ctx = f'{text} from the {nav_fx}'
                skills.append(Skill(name=ctx, kind='manip',
                                    compose=(lambda st, _sg=text: real_composer(_sg)),
                                    success=_make_success(ctx), update=_make_update(ctx),
                                    workspace='reuse'))
    else:
        for sg in subgoals:
            kind, ws = _classify(sg)
            # full sub-goal text — predicates (_is_place/_from_fixture/_place_target)
            # parse the NAME, and a 40-char truncation silently ate 'from the
            # cabinet' → the extraction clamp never armed (pnpc2c_skill1).
            skills.append(Skill(name=sg, kind=kind,
                                compose=(lambda st, _sg=sg: real_composer(_sg)),
                                success=_make_success(sg),
                                update=_make_update(sg),
                                workspace=ws))
    logger.info(f"[skill-exec] final skill list ({len(skills)}): {[s.name for s in skills]}")
    return Executor(lmp_env, snap_dir).run(skills, state)
