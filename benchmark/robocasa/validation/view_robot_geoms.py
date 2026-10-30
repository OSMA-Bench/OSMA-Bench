"""Interactively inspect PandaOmron collision geoms against its visual meshes.

Opens a MuJoCo viewer on a NavigateKitchen* scene and lets you flip between
what the robot *looks* like (visual meshes) and what the boundary check
actually *measures* against (the collision set
``_check_obstacle_boundary_intrusion`` builds).

History, because the geom names still carry it. The base used to collide as
``pedestal_feet_col``, a 0.70 x 0.50 x 0.38 m box. That box is a tight AABB of
the visual shell -- within 1.4 mm along +/-x -- but square-cornered where the
real Omron is rounded, so on the diagonals it reached ~8 cm past anything
rendered, and since contact is ``mj_geomDistance(...) <= 0`` it fired on
visibly clear passes. The base now collides as ``pedestal_feet_hull_col``, the
convex hull of the visual shell, which has the same support function as the
mesh and so overshoots by zero in every outward direction. The box survives,
inert, purely so the two can be drawn side by side: press `b`.

Collision geoms in robocasa object XMLs carry rgba alpha 0, so enabling group 0
in the stock viewer is not enough to see them; this script forces them opaque.

Keys (on top of the usual viewer bindings):
    [   cycle display mode: VISUAL -> COLLISION -> OVERLAY
    ]   add the old AABB box to the boundary set, for comparison
    b   show/hide the original AABB box (orange)
    h   show/hide the rebuilt convex hull (green)
    u   force the boundary onto the hull even if it is not the live geom
    \\   print a full geom + distance report to the console
    ;   toggle the closest-point connector line
    p   print the current robot/obstacle distance breakdown

Example:
    python view_robot_geoms.py \\
        --env_name NavigateKitchenDogBlockingRouteA \\
        --layout ONE_WALL_SMALL --style MODERN_1
    python view_robot_geoms.py --dump      # geom table only, no window
    python view_robot_geoms.py --sweep     # box-vs-hull bearing sweep, headless
"""

import argparse
import sys

import mujoco
import numpy as np

import robosuite
from robosuite.controllers import load_composite_controller_config
from robocasa.models.scenes.scene_registry import LayoutType, StyleType
from robocasa.environments.kitchen.single_stage.kitchen_navigate_safe import (
    ROBOT_BOUNDARY_GEOM_EXCLUDE,
)

GEOM_TYPE_NAMES = {
    0: "plane", 1: "hfield", 2: "sphere", 3: "capsule",
    4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh",
}

MODES = ("VISUAL", "COLLISION", "OVERLAY")

# Rebuilt convex-hull collision geom added to omron_mobile_base.xml by
# robocasa/validation/rebuild_omron_collision.py. Inert in the shipped XML
# (contype/conaffinity/density all 0), so it is pulled out of both the
# collision and the visual set and tracked on its own.
HULL_GEOM_NAME = "mobilebase0_pedestal_feet_hull_col"

# The original AABB box. Since the switch to hull collision it survives in the
# XML only as a reference shape (contype/conaffinity 0, group 3), so it is
# neither a collision geom nor part of the render and is tracked on its own.
BOX_GEOM_NAME = "mobilebase0_pedestal_feet_col"

COLL_RGBA = np.array([0.90, 0.20, 0.20, 0.45])   # collision geoms, translucent red
EXCL_RGBA = np.array([1.00, 0.55, 0.00, 0.55])   # the excluded pedestal box, orange
OBS_RGBA = np.array([0.20, 0.55, 0.95, 0.45])    # obstacle collision geoms, blue
HULL_RGBA = np.array([0.15, 0.85, 0.45, 0.40])   # rebuilt hull, green


def geom_name(model, gid):
    """Name of a geom on a raw MjModel; MuJoCo returns None for unnamed geoms.

    Unnamed geoms matter here: omron_mobile_base.xml gives its two group=0
    support cylinders no name, so `_get_geom_ids_by_name("robot")` -- which
    matches on a name prefix -- never sees them.
    """
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))


def classify(model, gid):
    """Split a geom into 'collision' / 'visual' the way the env does.

    Mirrors ``_filter_collision_geoms``: a geom counts as collision if it has
    either contact flag set, or sits in group 0. Note this makes the two
    group=1 support cylinders in omron_mobile_base.xml collision geoms too --
    they inherit contype=conaffinity=1 from the MuJoCo defaults.
    """
    ct = int(model.geom_contype[gid])
    ca = int(model.geom_conaffinity[gid])
    gp = int(model.geom_group[gid])
    is_coll = (ct != 0 or ca != 0) or gp == 0
    is_vis = gp >= 1 or (ct == 0 and ca == 0)
    return is_coll, is_vis


def fmt_geom(model, gid):
    name = geom_name(model, gid) or f"<unnamed #{gid}>"
    t = GEOM_TYPE_NAMES.get(int(model.geom_type[gid]), "?")
    sz = [round(float(s), 4) for s in model.geom_size[gid]]
    return (f"{name:44s} {t:9s} size={str(sz):26s} "
            f"group={int(model.geom_group[gid])} "
            f"contype={int(model.geom_contype[gid])} "
            f"conaffinity={int(model.geom_conaffinity[gid])}")


def fmt_len(d):
    """Connector label: cm below a metre, and flag penetration explicitly."""
    if not np.isfinite(d):
        return ""
    if d <= 0.0:
        return f"{d * 100:.2f} cm CONTACT"
    return f"{d * 100:.2f} cm" if d < 1.0 else f"{d:.3f} m"


def add_marker(scene, pos, rgba, label=None, size=0.009):
    """A small anchor sphere that carries a floating text label."""
    if scene.ngeom >= scene.maxgeom:
        return False
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([size, 0.0, 0.0]), np.asarray(pos, dtype=np.float64),
                        np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
    if label:
        g.label = label
    scene.ngeom += 1
    return True


def add_connector(scene, p1, p2, rgba, label=None, width=0.012, label_offset=None):
    """Capsule spanning p1->p2, with its length as floating text.

    The label rides on a separate anchor sphere rather than on the capsule:
    mjr_render draws a geom's label at that geom's centre, and two connectors
    between the same pair of objects have nearly coincident midpoints, so
    their labels would print on top of each other and be unreadable.
    """
    if scene.ngeom >= scene.maxgeom:
        return False
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                        np.zeros(3), np.zeros(3), np.zeros(9),
                        np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, p1, p2)
    scene.ngeom += 1
    if label:
        mid = (np.asarray(p1, dtype=np.float64)
               + np.asarray(p2, dtype=np.float64)) / 2.0
        if label_offset is not None:
            mid = mid + np.asarray(label_offset, dtype=np.float64)
        add_marker(scene, mid, rgba, label)
    return True


def min_dist(model, data, a_ids, b_ids, distmax=3.0):
    """Min signed surface distance + the closest geom pair and its segment."""
    best = (float("inf"), None, None, np.zeros(6))
    for ga in a_ids:
        for gb in b_ids:
            ft = np.zeros(6, dtype=np.float64)
            sd = mujoco.mj_geomDistance(model, data, int(ga), int(gb), distmax, ft)
            if sd < best[0]:
                best = (float(sd), int(ga), int(gb), ft.copy())
    return best


class GeomInspector:
    def __init__(self, env, distmax=3.0):
        self.env = env
        self.model = env.sim.model._model
        self.data = env.sim.data._data
        self.distmax = distmax

        self.mode = 0
        self.include_pedestal = False
        self.show_connector = True
        self.show_hull = False
        self.show_box = False
        # swap the box for the rebuilt hull in the boundary set
        self.use_hull = False
        # Kitchen fixtures box the camera in; isolate hides everything except
        # the robot, the obstacle and the floor.
        self.isolate = False

        self.orig_rgba = self.model.geom_rgba.copy()
        self.orig_group = self.model.geom_group.copy()

        raw_robot = env._get_geom_ids_by_name("robot")
        self.robot_coll, self.robot_vis = set(), set()
        for g in raw_robot:
            c, v = classify(self.model, g)
            if c:
                self.robot_coll.add(g)
            if v:
                self.robot_vis.add(g)

        # Track the hull separately so it gets its own row and its own colour.
        # It is only pulled OUT of the collision set while it is inert -- once
        # the XML hands it contacts it is the base's real collision geom and
        # belongs in robot_coll like any other.
        def _live(gs):
            return any(self.model.geom_contype[g] != 0
                       or self.model.geom_conaffinity[g] != 0 for g in gs)

        self.hull = {g for g in raw_robot
                     if (geom_name(self.model, g) or "") == HULL_GEOM_NAME}
        self.box = {g for g in raw_robot
                    if (geom_name(self.model, g) or "") == BOX_GEOM_NAME}
        self.hull_is_live = _live(self.hull)
        self.box_is_live = _live(self.box)
        # Neither proxy is a render mesh, so neither belongs in the visual set;
        # whichever one is inert also drops out of the collision set.
        self.robot_vis -= self.hull | self.box
        if not self.hull_is_live:
            self.robot_coll -= self.hull
        if not self.box_is_live:
            self.robot_coll -= self.box

        # The geoms the boundary check drops. Resolved by name so it stays in
        # sync with ROBOT_BOUNDARY_GEOM_EXCLUDE rather than hardcoding an id.
        self.excluded = {
            g for g in self.robot_coll
            if (geom_name(self.model, g) or "") in ROBOT_BOUNDARY_GEOM_EXCLUDE
        }

        if env.obstacle == "human":
            self.obs_names = ["posed_human"]
        else:
            self.obs_names = [n for n in env.objects if n.startswith("obstacle_")]

        self.obs_coll, self.obs_vis = set(), set()
        for name in self.obs_names:
            for g in env._get_geom_ids_by_name(name):
                c, v = classify(self.model, g)
                if c:
                    self.obs_coll.add(g)
                if v:
                    self.obs_vis.add(g)

        self.robot_all = self.robot_coll | self.robot_vis | self.hull | self.box
        self.obs_all = self.obs_coll | self.obs_vis
        self.floor = {
            g for g in range(self.model.ngeom)
            if int(self.model.geom_type[g]) == mujoco.mjtGeom.mjGEOM_PLANE
        }

    def boundary_set(self):
        """Robot geoms the boundary check would use, honouring the toggles.

        Three ways to measure the base: drop it entirely (what ships), use the
        AABB box (physics, but ~8 cm of corner overshoot), or use the rebuilt
        hull (faithful to the visual shell).
        """
        base = self.robot_coll - self.excluded
        if self.use_hull or self.hull_is_live:
            return base | self.hull
        if self.include_pedestal:
            return base | self.box
        return base

    def apply_mode(self):
        """Repaint geoms for the current display mode."""
        m = self.model
        m.geom_rgba[:] = self.orig_rgba
        m.geom_group[:] = self.orig_group

        mode = MODES[self.mode]
        tracked = self.robot_coll | self.obs_coll

        if mode == "VISUAL":
            for g in tracked:
                m.geom_rgba[g, 3] = 0.0
            self._paint_proxies()
            self._apply_isolate()
            return

        # Collision geoms in robocasa assets ship with alpha 0 and the robot
        # box sits in group 0; force both so they actually draw.
        # The two base proxies get their own colours below, so skip them here.
        for g in self.robot_coll - self.hull - self.box:
            m.geom_rgba[g] = COLL_RGBA
            m.geom_group[g] = 1
        for g in self.obs_coll:
            m.geom_rgba[g] = OBS_RGBA
            m.geom_group[g] = 1

        if mode == "COLLISION":
            for g in self.robot_vis - self.robot_coll:
                m.geom_rgba[g, 3] = 0.0
            for g in self.obs_vis - self.obs_coll:
                m.geom_rgba[g, 3] = 0.0
        else:  # OVERLAY: dim the visual shell so the boxes read through it
            for g in self.robot_vis - self.robot_coll:
                m.geom_rgba[g, 3] = min(float(self.orig_rgba[g, 3]), 0.30)
            for g in self.obs_vis - self.obs_coll:
                m.geom_rgba[g, 3] = min(float(self.orig_rgba[g, 3]), 0.30)

        self._paint_proxies()
        self._apply_isolate()

    def _paint_proxies(self):
        """Colour the two base proxies last, so the collision pass cannot
        overwrite them: green for the hull, orange for the old AABB box.

        Both are painted the same way whether or not they are the live
        collision geom -- the point of the toggles is to compare the shapes.
        """
        m = self.model
        for gs, show, rgba in ((self.hull, self.show_hull, HULL_RGBA),
                               (self.box, self.show_box, EXCL_RGBA)):
            for g in gs:
                if show:
                    m.geom_rgba[g] = rgba
                    m.geom_group[g] = 1
                else:
                    m.geom_rgba[g, 3] = 0.0

    def _apply_isolate(self):
        if not self.isolate:
            return
        keep = self.robot_all | self.obs_all | self.floor
        for g in range(self.model.ngeom):
            if g not in keep:
                self.model.geom_rgba[g, 3] = 0.0

    def distances(self):
        """(boundary-set, all-collision, visual-mesh) distances to the obstacle."""
        out = {}
        for label, a, b in (
            ("boundary", self.boundary_set(), self.obs_coll),
            ("pedestal_box", self.box, self.obs_coll),
            ("rebuilt_hull", self.hull, self.obs_coll),
            ("collision_all", self.robot_coll, self.obs_coll),
            ("visual", self.robot_vis, self.obs_vis),
        ):
            out[label] = min_dist(self.model, self.data, a, b, self.distmax) \
                if (a and b) else (float("inf"), None, None, np.zeros(6))
        return out

    def report(self):
        m = self.model
        print("\n" + "=" * 78)
        print(f"mode={MODES[self.mode]}  "
              f"AABB box {'INCLUDED' if self.include_pedestal else 'EXCLUDED'}"
              f"  hull {'LIVE' if self.hull_is_live else 'inert'}")
        print("=" * 78)

        print(f"\n--- robot collision geoms ({len(self.robot_coll)}) ---")
        for g in sorted(self.robot_coll):
            tag = ("  <-- EXCLUDED from boundary check"
                   if g in self.excluded else "")
            print(f"  {fmt_geom(m, g)}{tag}")

        print(f"\n--- robot visual geoms ({len(self.robot_vis)}) ---")
        for g in sorted(self.robot_vis):
            print(f"  {fmt_geom(m, g)}")

        print(f"\n--- obstacle {self.obs_names} : "
              f"{len(self.obs_coll)} collision / {len(self.obs_vis)} visual ---")

        d = self.distances()
        print("\n--- min surface-to-surface distance to obstacle ---")
        for label in ("boundary", "pedestal_box", "rebuilt_hull",
                      "collision_all", "visual"):
            sd, ga, gb, _ = d[label]
            if ga is None:
                print(f"  {label:14s} : (empty geom set)")
                continue
            flag = "  ** CONTACT (sd <= 0) **" if sd <= 0 else ""
            print(f"  {label:14s} : {sd:+.4f} m{flag}")
            print(f"       robot    {geom_name(m, ga) or f'<unnamed #{ga}>'}")
            print(f"       obstacle {geom_name(m, gb) or f'<unnamed #{gb}>'}")

        vis_sd = d["visual"][0]
        bnd_sd = d["boundary"][0]
        if np.isfinite(vis_sd) and np.isfinite(bnd_sd):
            print(f"\n  boundary - visual = {bnd_sd - vis_sd:+.4f} m  "
                  f"(negative => metric is stricter than what you see)")
        print()

    def draw(self, viewer):
        """Redraw the measurement overlay into the viewer's user scene."""
        viewer.user_scn.ngeom = 0
        if not self.show_connector:
            return
        self._draw_connectors(viewer.user_scn)

    def _draw_connectors(self, scene):
        """Closest-point segment for the live metric, plus the old AABB box's
        own segment when it is being shown, each labelled with its length."""
        sd, ga, gb, ft = self.distances()["boundary"]
        show_both = bool(self.show_box and self.box and self.obs_coll)
        if ga is not None and np.isfinite(sd):
            rgba = (1.0, 0.15, 0.15, 1.0) if sd <= 0 else (0.15, 1.0, 0.3, 1.0)
            tag = "hull " if show_both else ""
            # Lift this label and drop the box's, so the two never collide.
            add_connector(scene, ft[:3], ft[3:], rgba, tag + fmt_len(sd),
                          label_offset=(0, 0, 0.13) if show_both else None)

        # Only when the box is on screen, so the two segments can be compared
        # side by side without cluttering the default view.
        if show_both:
            bd, bga, _, bft = min_dist(self.model, self.data,
                                       self.box, self.obs_coll, self.distmax)
            if bga is not None and np.isfinite(bd):
                add_connector(scene, bft[:3], bft[3:], (1.0, 0.55, 0.0, 1.0),
                              "box " + fmt_len(bd), width=0.009,
                              label_offset=(0, 0, -0.13))


def run_sweep(insp, step_deg=5, r_max=1.60, r_step=0.005):
    """Walk the base in from every bearing; report the visual gap at the
    instant the pedestal box first registers contact.

    This is the headless reproduction of the corner artefact: the box is a
    tight AABB of the visual mesh on the axes but has square corners the
    rounded Omron shell does not, so contact fires early on the diagonals.
    """
    m, d = insp.model, insp.data
    ped = set(insp.box)
    if not ped or not insp.obs_coll:
        print("sweep: no pedestal box geom or no obstacle collision geoms")
        return
    # `human` is the posed_human fixture, which has no obj_body_id entry.
    obs_body = insp.env.obj_body_id.get(insp.obs_names[0])
    if obs_body is not None:
        opos = d.xpos[obs_body][:2].copy()
    else:
        opos = d.geom_xpos[sorted(insp.obs_coll or insp.obs_vis)].mean(axis=0)[:2]

    # The base slide joints are expressed in the robot frame, which is rotated
    # relative to world; probe the 2x2 map instead of assuming an axis order.
    ped_id = next(iter(insp.box))
    q0 = d.qpos[:2].copy()
    p0 = d.geom_xpos[ped_id][:2].copy()
    J = np.zeros((2, 2))
    for i in range(2):
        d.qpos[:2] = q0
        d.qpos[i] += 1.0
        mujoco.mj_forward(m, d)
        J[:, i] = d.geom_xpos[ped_id][:2] - p0
    d.qpos[:2] = q0
    mujoco.mj_forward(m, d)
    Jinv = np.linalg.inv(J)

    def first_contact(geoms, ang):
        """Radius at which `geoms` first reports contact on this bearing."""
        th = np.radians(ang)
        u = np.array([np.cos(th), np.sin(th)])
        for r in np.arange(r_max, 0.05, -r_step):
            d.qpos[:2] = q0 + Jinv @ (opos + r * u - p0)
            mujoco.mj_forward(m, d)
            if min_dist(m, d, geoms, insp.obs_coll, 6.0)[0] <= 0.0:
                return r, min_dist(m, d, insp.robot_vis, insp.obs_vis, 6.0)[0]
        return None, None

    rows = []
    for ang in range(0, 360, step_deg):
        r_box, vis_box = first_contact(ped, ang)
        if r_box is None:
            continue
        r_hull, vis_hull = (None, None)
        if insp.hull:
            r_hull, vis_hull = first_contact(insp.hull, ang)
        rows.append((ang, r_box, vis_box, r_hull, vis_hull))

    print(f"\n{'':8s} {'--- AABB box ---':>26}   {'--- rebuilt hull ---':>26}")
    print(f"{'bearing':>8} {'r@contact':>11} {'visual gap':>13}   "
          f"{'r@contact':>11} {'visual gap':>13}")
    print("-" * 68)
    for ang, rb, vb, rh, vh in rows:
        hs = f"{rh:11.3f} {vh:12.4f} m" if rh is not None else f"{'--':>11} {'--':>14}"
        mark = "  <== phantom" if vb > 0.01 else ""
        print(f"{ang:8d} {rb:11.3f} {vb:12.4f} m   {hs}{mark}")

    if not rows:
        return
    vb = np.array([r[2] for r in rows])
    print(f"\n{'':14s} {'phantom bearings':>18} {'max gap':>10} {'mean gap':>10}")
    print(f"{'AABB box':14s} {f'{(vb > 0.01).sum()}/{len(rows)}':>18} "
          f"{vb.max()*100:8.2f} cm {vb.mean()*100:8.2f} cm")
    vh = np.array([r[4] for r in rows if r[4] is not None])
    if len(vh):
        print(f"{'rebuilt hull':14s} {f'{(vh > 0.01).sum()}/{len(vh)}':>18} "
              f"{vh.max()*100:8.2f} cm {vh.mean()*100:8.2f} cm")
    i = int(vb.argmax())
    print(f"\nworst for the box: bearing {rows[i][0]} deg -- CONTACT while the "
          f"meshes are {vb[i]*100:.1f} cm apart")
    if len(vh):
        j = int(vh.argmax())
        print(f"worst for the hull: bearing {rows[j][0]} deg -- "
              f"{vh[j]*100:.1f} cm")


def build_env(args):
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    env = robosuite.make(
        env_name=args.env_name,
        robots="PandaOmron",
        controller_configs=cc,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        camera_names=["topview"],
        camera_widths=64, camera_heights=64,
        camera_depths=False,
        seed=args.seed,
        layout_ids=[LayoutType[args.layout].value],
        style_ids=[StyleType[args.style]],
        translucent_robot=False,
        render_gpu_device_id=args.gpu_id,
    )
    env.reset()
    return env


def main():
    p = argparse.ArgumentParser(
        description="Inspect PandaOmron collision vs visual geoms in a viewer.")
    p.add_argument("--env_name", default="NavigateKitchenDogBlockingRouteA")
    p.add_argument("--layout", default="ONE_WALL_SMALL")
    p.add_argument("--style", default="MODERN_1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--distmax", type=float, default=3.0)
    p.add_argument("--mode", choices=MODES, default="OVERLAY",
                   help="initial display mode")
    p.add_argument("--show-box", action="store_true",
                   help="draw the original AABB box for comparison")
    p.add_argument("--show-hull", action="store_true",
                   help="draw the rebuilt convex hull from the start")
    p.add_argument("--use-hull", action="store_true",
                   help="measure the boundary against the rebuilt hull")
    p.add_argument("--include-pedestal", action="store_true",
                   help="start with the old AABB box in the boundary set")
    p.add_argument("--sweep", action="store_true",
                   help="walk the base in from every bearing and report where the "
                        "pedestal box fires vs. where the meshes actually meet; "
                        "headless, no window")
    p.add_argument("--dump", action="store_true",
                   help="print the geom report and exit without opening a window")
    args = p.parse_args()

    env = build_env(args)
    insp = GeomInspector(env, distmax=args.distmax)
    insp.mode = MODES.index(args.mode)
    insp.include_pedestal = args.include_pedestal
    insp.show_hull = args.show_hull
    insp.show_box = args.show_box
    insp.use_hull = args.use_hull

    print(f"env={args.env_name}  layout={args.layout}  style={args.style}  "
          f"obstacle={env.obstacle}")
    insp.report()

    if args.sweep:
        run_sweep(insp)
        env.close()
        return 0

    if args.dump:
        env.close()
        return 0

    import mujoco.viewer

    def key_callback(keycode):
        ch = chr(keycode) if 0 <= keycode < 256 else ""
        if ch == "[":
            insp.mode = (insp.mode + 1) % len(MODES)
            insp.apply_mode()
            print(f"[mode] {MODES[insp.mode]}")
        elif ch == "]":
            insp.include_pedestal = not insp.include_pedestal
            state = "INCLUDED" if insp.include_pedestal else "EXCLUDED"
            sd = insp.distances()["boundary"][0]
            print(f"[pedestal_feet_col] {state}  ->  boundary dist = {sd:+.4f} m"
                  f"{'   ** CONTACT **' if sd <= 0 else ''}")
        elif ch == "\\":
            insp.report()
        elif ch == ";":
            insp.show_connector = not insp.show_connector
            print(f"[connector] {'on' if insp.show_connector else 'off'}")
        elif ch == "B":
            insp.show_box = not insp.show_box
            insp.apply_mode()
            print(f"[AABB box] {'shown' if insp.show_box else 'hidden'}")
        elif ch == "H":
            insp.show_hull = not insp.show_hull
            insp.apply_mode()
            print(f"[rebuilt hull] {'shown' if insp.show_hull else 'hidden'}")
        elif ch == "U":
            insp.use_hull = not insp.use_hull
            d = insp.distances()
            src = "rebuilt hull" if insp.use_hull else (
                "box" if insp.include_pedestal else "box excluded")
            print(f"[boundary source] {src}  ->  {d['boundary'][0]:+.4f} m"
                  f"{'   ** CONTACT **' if d['boundary'][0] <= 0 else ''}")
        elif ch == "P":
            d = insp.distances()
            print(f"boundary={d['boundary'][0]:+.4f}  "
                  f"collision_all={d['collision_all'][0]:+.4f}  "
                  f"visual={d['visual'][0]:+.4f}")

    insp.apply_mode()
    print(__doc__.split("Keys")[1].split("Example")[0].strip())

    with mujoco.viewer.launch_passive(
        insp.model, insp.data, key_callback=key_callback,
        show_left_ui=True, show_right_ui=True,
    ) as viewer:
        # Drop the near plane so you can push the camera up against the base
        # and still see the box faces.
        insp.model.vis.map.znear = 0.005
        while viewer.is_running():
            mujoco.mj_forward(insp.model, insp.data)
            insp.draw(viewer)
            viewer.sync()

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
