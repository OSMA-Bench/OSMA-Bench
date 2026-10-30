"""Model Predictive Path Integral (MPPI) controller for navigation.

Operates in world coordinates with holonomic dynamics:
    x_{t+1} = x_t + dt * (vx*cos(θ) - vy*sin(θ))
    y_{t+1} = y_t + dt * (vx*sin(θ) + vy*cos(θ))
    θ_{t+1} = θ_t + dt * ω

Uses 2D obstacle/cost map (pixel coords) directly via lookup at each rollout step.
Verified on synthetic 2D scene: 84 steps to goal with 3 disk obstacles, ~4ms/step.

Env vars (override defaults):
    MPPI_H              horizon length (default 25)
    MPPI_K              num rollouts (default 1000)
    MPPI_DT             rollout dt (default 0.05)
    MPPI_SIGMA_V        translation sample sigma (default 0.6)
    MPPI_SIGMA_W        rotation sample sigma (default 1.0)
    MPPI_LAMBDA         softmin temperature (default 0.5)
    MPPI_W_OBS          obstacle cost weight (default 15.0)
    MPPI_W_GOAL         per-step goal-dist weight (default 8.0)
    MPPI_W_TERM         terminal goal-dist weight (default 80.0)
    MPPI_W_CTRL         control magnitude penalty (default 0.005)
    MPPI_UMAX_V         max body translation [-1, +1] per axis (default 1.0)
    MPPI_UMAX_W         max angular velocity [-1, +1] (default 1.5)
    MPPI_VEL_SCALE      empirical world-velocity per action=1.0 (default 0.40 m/s)
    MPPI_PLAN_DT        planning-time dt for rollout (default 0.20s — independent of env step 0.05s, so horizon = H * PLAN_DT covers ~few meters)
    MPPI_W_YAW_ALIGN    weight for yaw → goal_dir alignment cost (default 0 — disabled)
    MPPI_W_LATERAL      weight penalizing |v_y_body| (lateral motion, default 0 — disabled)
    MPPI_W_PATH         weight for path-tracking cost (default 0 — disabled).
                        When enabled, MPPI follows A* path waypoints (provided via step()).
                        Each rollout step penalized by distance to nearest path waypoint.
"""
import os
import numpy as np


class MPPI:
    def __init__(self, workspace_min, workspace_max, map_h, map_w,
                 dt=None, H=None, K=None, sigma_v=None, sigma_w=None,
                 lam=None, u_max_v=None, u_max_w=None,
                 w_obs=None, w_goal=None, w_term=None, w_ctrl=None,
                 vel_scale=None):
        e = os.environ.get
        self.dt   = float(e('MPPI_PLAN_DT', e('MPPI_DT', str(dt if dt is not None else 0.20))))
        self.H    = int(  e('MPPI_H',       str(H    if H    is not None else 40)))
        self.K    = int(  e('MPPI_K',       str(K    if K    is not None else 1000)))
        # Empirical world velocity per action=1.0. Robot in robocasa moves at
        # ~0.15-0.20 m/s when action=1.0 (joint velocity command going through
        # actuator scaling). Without this scale, rollout massively overestimates
        # progress per step → MPPI converges on stationary near-goal predictions.
        self.vel_scale = float(e('MPPI_VEL_SCALE', str(vel_scale if vel_scale is not None else 0.40)))
        # Same calibration for ω. omega_scale=15 was too aggressive (rollout
        # predicted 9rad rotation over horizon, mismatched reality).
        # omega_scale=5: rollout horizon rotation ≈ 3rad (180°), realistic.
        self.omega_scale = float(e('MPPI_OMEGA_SCALE', '5.0'))
        sv        = float(e('MPPI_SIGMA_V', str(sigma_v if sigma_v is not None else 0.6)))
        # ω samples narrower than v — robot ω physically capped at ~0.05 rad/step
        sw        = float(e('MPPI_SIGMA_W', str(sigma_w if sigma_w is not None else 0.03)))
        self.sigma = np.array([sv, sv, sw])
        self.lam  = float(e('MPPI_LAMBDA',  str(lam  if lam  is not None else 0.5)))
        umv       = float(e('MPPI_UMAX_V',  str(u_max_v if u_max_v is not None else 1.0)))
        # ω clip MUST match robot physical limit (holonomic baseline uses 0.05).
        # Higher values cause yaw drift → body-frame transformation errors → robot
        # circles instead of going to goal.
        umw       = float(e('MPPI_UMAX_W',  str(u_max_w if u_max_w is not None else 0.05)))
        self.u_max = np.array([umv, umv, umw])
        self.w_obs  = float(e('MPPI_W_OBS',  str(w_obs  if w_obs  is not None else 15.0)))
        self.w_goal = float(e('MPPI_W_GOAL', str(w_goal if w_goal is not None else 8.0)))
        self.w_term = float(e('MPPI_W_TERM', str(w_term if w_term is not None else 80.0)))
        self.w_ctrl = float(e('MPPI_W_CTRL', str(w_ctrl if w_ctrl is not None else 0.005)))
        # Forward-facing motion encouragement (off by default for backwards-compat)
        self.w_yaw_align = float(e('MPPI_W_YAW_ALIGN', '0.0'))
        self.w_lateral   = float(e('MPPI_W_LATERAL',   '0.0'))
        self.w_path      = float(e('MPPI_W_PATH',      '0.0'))

        self.workspace_min = np.asarray(workspace_min, dtype=float)[:2]
        self.workspace_max = np.asarray(workspace_max, dtype=float)[:2]
        self.map_h = int(map_h)
        self.map_w = int(map_w)
        self._ws_extent = self.workspace_max - self.workspace_min   # (2,)

        # warm-start buffer
        self.u_prev = np.zeros((self.H, 3))

    def reset(self):
        self.u_prev = np.zeros((self.H, 3))

    def _world_to_pix(self, x, y):
        """Vectorized world (x,y) → costmap indices following IMAGE convention.

        Convention (verified from _smoke_multi_sweep.py:w2c + disk):
            costmap.shape = (map_h, map_w) where:
              first dim  (rows, H) = y range  (y_world along this axis)
              second dim (cols, W) = x range  (x_world along this axis)
            Lookup is costmap[row, col] = costmap[y_idx, x_idx]

        Earlier MPPI code had this transposed, causing the cost-lookup to read
        the (y, x) transposed cell — MPPI saw "low cost" at wrong locations,
        which is why trajectories passed through red regions in viz.
        """
        fx = (x - self.workspace_min[0]) / max(self._ws_extent[0], 1e-6)
        fy = (y - self.workspace_min[1]) / max(self._ws_extent[1], 1e-6)
        row = np.clip(np.round(fy * (self.map_h - 1)).astype(int), 0, self.map_h - 1)
        col = np.clip(np.round(fx * (self.map_w - 1)).astype(int), 0, self.map_w - 1)
        return row, col

    def step(self, state, goal_xy, costmap, path_xy=None, tgt_dist_map=None,
             nominal_u=None):
        """Compute one MPPI control step.

        Args:
            state: (x, y, θ) world frame (np.array shape (3,))
            goal_xy: (x, y) world frame goal
            costmap: 2D np.array, shape (map_h, map_w). Higher = worse (obstacle).
            path_xy: optional (P, 2) array of waypoint world coords from A* planner.
                When provided AND self.w_path > 0, adds path-tracking cost
                (penalizes distance from nearest waypoint each rollout step).
            tgt_dist_map: optional (map_h, map_w) array of distance-to-goal in
                world meters. When provided, terminal cost uses this map at the
                rollout endpoint INSTEAD of Euclidean distance to goal_xy. This
                lets W_TERM pull rollouts along a geodesic field rather than
                straight-line, so detour rollouts aren't penalized for going
                "wrong way first". Same scale as Euclidean (meters), so W_TERM
                tuning carries over.

        Returns:
            u0: np.array shape (3,) = (vx_body, vy_body, ω) ∈ [-u_max, +u_max]
            info: dict with debug fields (sampled_min_cost, etc.)
        """
        x0, y0, th0 = float(state[0]), float(state[1]), float(state[2])
        gx, gy = float(goal_xy[0]), float(goal_xy[1])

        # 1. Sample K control sequences around nominal (or warm-start u_prev)
        # nominal_u (optional, shape (3,)): center sampling here. Used by Phase O
        # to bias rollouts toward A* next-waypoint direction (body-frame vel).
        # Critical for narrow-corridor cases where random N(0, σ) doesn't
        # generate path-traversal rollouts.
        noise = np.random.randn(self.K, self.H, 3) * self.sigma
        if nominal_u is not None:
            base = np.tile(np.asarray(nominal_u, dtype=float)[None, :], (self.H, 1))
        else:
            base = self.u_prev
        u = base[None, :, :] + noise
        u = np.clip(u, -self.u_max, self.u_max)

        # 2. Vectorized rollout via simple Euler integration
        xs = np.empty((self.K, self.H + 1))
        ys = np.empty((self.K, self.H + 1))
        ths = np.empty((self.K, self.H + 1))
        xs[:, 0] = x0; ys[:, 0] = y0; ths[:, 0] = th0
        # Scale action to realistic world-frame velocity (action ∈ [-1, 1] ≠ m/s)
        for t in range(self.H):
            vx = u[:, t, 0] * self.vel_scale
            vy = u[:, t, 1] * self.vel_scale
            w  = u[:, t, 2] * self.omega_scale
            cth = np.cos(ths[:, t]); sth = np.sin(ths[:, t])
            xs[:, t + 1] = xs[:, t] + self.dt * (vx * cth - vy * sth)
            ys[:, t + 1] = ys[:, t] + self.dt * (vx * sth + vy * cth)
            ths[:, t + 1] = ths[:, t] + self.dt * w

        # 3. Cost — sum of stage + terminal
        # Image convention: costmap[row, col] = costmap[y_idx, x_idx]
        row, col = self._world_to_pix(xs[:, 1:], ys[:, 1:])    # (K, H)
        cmap_vals = costmap[row, col]                          # standard image indexing
        cost_obs = self.w_obs * cmap_vals.sum(axis=1)
        cost_ctrl = self.w_ctrl * (u ** 2).sum(axis=(1, 2))
        dx_per = gx - xs[:, 1:]; dy_per = gy - ys[:, 1:]
        cost_goal = self.w_goal * np.sqrt(dx_per ** 2 + dy_per ** 2).sum(axis=1)
        if tgt_dist_map is not None:
            row_end, col_end = self._world_to_pix(xs[:, -1], ys[:, -1])
            cost_term = self.w_term * tgt_dist_map[row_end, col_end]
        else:
            dx = gx - xs[:, -1]; dy = gy - ys[:, -1]
            cost_term = self.w_term * np.sqrt(dx ** 2 + dy ** 2)
        costs = cost_obs + cost_ctrl + cost_goal + cost_term

        # Optional: yaw alignment with MOTION direction (world velocity), not
        # goal direction. Motion direction = atan2(world_vy, world_vx) where
        # world velocity is the body-frame action rotated by current yaw.
        # This makes robot face whichever way it is actually moving (around
        # obstacles too), not just toward goal.
        if self.w_yaw_align > 0:
            # World velocity from body-frame action (use SAME scaling as rollout)
            vx_body = u[:, :, 0] * self.vel_scale
            vy_body = u[:, :, 1] * self.vel_scale
            # (ω scaling not needed here — only used for translation alignment)
            cth_t = np.cos(ths[:, :-1]); sth_t = np.sin(ths[:, :-1])
            world_vx = vx_body * cth_t - vy_body * sth_t
            world_vy = vx_body * sth_t + vy_body * cth_t
            speed = np.hypot(world_vx, world_vy)
            # Only penalize alignment when actually moving (avoid arctan2 noise
            # on zero-velocity samples)
            motion_dir = np.arctan2(world_vy, world_vx)
            yaw_err = (motion_dir - ths[:, 1:] + np.pi) % (2 * np.pi) - np.pi
            # Weight by speed: faster motion → stronger alignment pressure
            costs = costs + self.w_yaw_align * (np.abs(yaw_err) * speed).sum(axis=1)
        # Optional: penalize lateral body motion to discourage sideways gait
        if self.w_lateral > 0:
            costs = costs + self.w_lateral * np.abs(u[:, :, 1]).sum(axis=1)

        # Optional: A* path-tracking cost — penalize distance from path
        # Rollout points (xs[:, 1:], ys[:, 1:]) shape (K, H).
        # path_xy shape (P, 2). Compute min distance per (K, H) point to path.
        if self.w_path > 0 and path_xy is not None and len(path_xy) > 0:
            px = np.asarray(path_xy, dtype=float).reshape(-1, 2)   # (P, 2)
            # dist (K, H, P) — broadcasted: rollout (K, H, 1, 2) vs path (1, 1, P, 2)
            dx = xs[:, 1:, None] - px[None, None, :, 0]
            dy = ys[:, 1:, None] - px[None, None, :, 1]
            dist_to_path = np.sqrt(dx**2 + dy**2)                  # (K, H, P)
            min_dist = dist_to_path.min(axis=2)                    # (K, H)
            costs = costs + self.w_path * min_dist.sum(axis=1)

        # 4. Soft-min weighted average (path integral)
        beta = float(costs.min())
        weights = np.exp(-(costs - beta) / max(self.lam, 1e-6))
        weights /= max(weights.sum(), 1e-8)
        u_opt = (weights[:, None, None] * u).sum(axis=0)     # (H, 3)

        # Roll u_prev forward for next call (drop applied step, append zeros)
        self.u_prev = np.vstack([u_opt[1:], np.zeros((1, 3))])

        info = {
            'cost_min': float(costs.min()),
            'cost_mean': float(costs.mean()),
            'top1_idx': int(np.argmin(costs)),
            'cost_obs_min': float(cost_obs.min()),
            'cost_goal_min': float(cost_goal.min()),
        }
        # Per-step debug: log first u_opt[0] + horizon-end predicted pos
        if os.environ.get('MPPI_DEBUG_STEP', '0') == '1':
            x_idx_r, y_idx_r = self._world_to_pix(np.array([x0]), np.array([y0]))
            top1 = int(np.argmin(costs))
            print(f'[mppi] state=({x0:.3f},{y0:.3f},yaw={th0:.3f}) '
                  f'goal=({gx:.3f},{gy:.3f}) dist={np.hypot(gx-x0, gy-y0):.3f}m '
                  f'u0=({u_opt[0,0]:+.3f},{u_opt[0,1]:+.3f},{u_opt[0,2]:+.3f}) '
                  f'top1_end=({xs[top1,-1]:.3f},{ys[top1,-1]:.3f}) '
                  f'top1_cost={costs[top1]:.2f} '
                  f'cost_pix=({x_idx_r[0]},{y_idx_r[0]})={costmap[x_idx_r[0], y_idx_r[0]]:.3f}',
                  flush=True)
        return u_opt[0].copy(), info
