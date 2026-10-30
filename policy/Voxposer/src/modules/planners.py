"""Greedy path planner."""
import os
import heapq
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
from utils.utils import get_clock_time, normalize_map, calc_curvature, get_logger

logger = get_logger(__name__)


def _astar_pixel(start_pos, costmap, target_mask, blocked_mask=None,
                 strict_goal=False):
    """A* search on a 2D cost grid to any cell in `target_mask`.

    Args:
        start_pos: (2,) array of (row, col) start (will be rounded to int)
        costmap:   (H, W) float array — node cost at each cell
        target_mask: (H, W) bool array — True for any acceptable goal cell
        blocked_mask: optional (H, W) bool array — cells the robot cannot
            physically enter (e.g. obstacle cells inflated by robot_radius).
            A* will not expand neighbors into a blocked cell unless the
            cell is the start or in target_mask. If start is itself blocked,
            we relax for that single cell (so the search can begin).

    Returns:
        path: (n, 2) np.ndarray of grid cells (float), or None if start invalid.
        If target is unreachable through unblocked cells, returns best-effort
        path to the closest reachable cell so postprocess can still force-append.
    """
    H, W = costmap.shape
    target_cells = np.argwhere(target_mask)
    if target_cells.size == 0:
        return None
    start = (int(round(start_pos[0])), int(round(start_pos[1])))
    if not (0 <= start[0] < H and 0 <= start[1] < W):
        return None

    # Heuristic: Euclidean distance from any cell to the closest target cell.
    # Precomputed via distance_transform_edt for O(HW) total cost.
    h_map = distance_transform_edt(~target_mask)

    # 8-neighbors with edge step cost (Euclidean distance)
    neighbors = (
        (-1, -1, 1.41421356), (-1, 0, 1.0), (-1, 1, 1.41421356),
        ( 0, -1, 1.0),                       ( 0, 1, 1.0),
        ( 1, -1, 1.41421356), ( 1, 0, 1.0), ( 1, 1, 1.41421356),
    )

    # Respect blocked_mask, always allowing the start cell.
    #
    # Target cells are exempt unless strict_goal is set. Without the exemption
    # a goal swallowed by an avoidance halo becomes unreachable, which is what
    # "the destination is undetermined" looks like. That happens to the goal
    # centre in 8-19% of episodes, but the goal region always keeps some cells
    # outside the halo, so the measured effect is under 1 pp. The start cell
    # stays exempt either way: blocking it would stop the search from
    # beginning at all.
    def _is_blocked(pos):
        if blocked_mask is None:
            return False
        if pos == start:
            return False
        if target_mask[pos] and not strict_goal:
            return False
        return bool(blocked_mask[pos])

    open_heap = [(float(h_map[start]), 0.0, start)]
    g_score = {start: 0.0}
    came_from = {}
    visited = set()
    closest_pos, closest_h = start, float(h_map[start])

    while open_heap:
        _f, g, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        # Track best-effort if A* fails to find target later
        ch = float(h_map[current])
        if ch < closest_h:
            closest_h = ch
            closest_pos = current
        # Goal test
        if target_mask[current]:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return np.array(path[::-1], dtype=float)
        for dy, dx, step_cost in neighbors:
            ny, nx = current[0] + dy, current[1] + dx
            if not (0 <= ny < H and 0 <= nx < W):
                continue
            np_ = (ny, nx)
            if _is_blocked(np_):
                continue
            edge_cost = step_cost + float(costmap[ny, nx])
            tentative_g = g + edge_cost
            if np_ not in g_score or tentative_g < g_score[np_]:
                g_score[np_] = tentative_g
                came_from[np_] = current
                f_new = tentative_g + float(h_map[ny, nx])
                heapq.heappush(open_heap, (f_new, tentative_g, np_))

    # Unreachable: return path to closest cell we managed to reach
    if closest_pos == start:
        return np.array([start], dtype=float)
    path = [closest_pos]
    cur = closest_pos
    while cur in came_from:
        cur = came_from[cur]
        path.append(cur)
    return np.array(path[::-1], dtype=float)


class PathPlanner:
    """
    A greedy path planner that greedily chooses the next voxel with the lowest cost.
    Then apply several postprocessing steps to the path.
    (TODO: can be improved using more principled methods, including extension to whole-arm planning)
    """
    def __init__(self, planner_config, map_size):
        self.config = planner_config
        self.map_size = map_size

    def _flag(self, name, default, env):
        """Read a boolean planner setting; the env var wins when set."""
        raw = os.environ.get(env)
        if raw is not None:
            return raw == '1'
        if hasattr(self.config, 'get'):
            return bool(self.config.get(name, default))
        return bool(getattr(self.config, name, default))

    def optimize(self, start_pos: np.ndarray, target_map: np.ndarray, obstacle_map: np.ndarray, object_centric=False):
        """3D voxel greedy-descent planner (manipulation).
        start_pos: (3,); target_map/obstacle_map: (S,S,S); returns (path[n,3], info)."""
        logger.debug(f'[{get_clock_time(milliseconds=True)}] planner start')
        info = dict()
        start_pos, raw_start_pos = start_pos.copy(), start_pos
        target_map, raw_target_map = target_map.copy(), target_map
        obstacle_map, raw_obstacle_map = obstacle_map.copy(), obstacle_map
        target_map = distance_transform_edt(1 - target_map)
        target_map = normalize_map(target_map)
        obstacle_map = gaussian_filter(obstacle_map, sigma=self.config.obstacle_map_gaussian_sigma)
        obstacle_map = normalize_map(obstacle_map)
        costmap = target_map * self.config.target_map_weight + obstacle_map * self.config.obstacle_map_weight
        costmap = normalize_map(costmap)
        _costmap = costmap.copy()
        stop_criteria = self._get_stop_criteria()
        path, current_pos = [start_pos], start_pos
        for i in range(self.config.max_steps):
            all_nearby_voxels = self._calculate_nearby_voxel(current_pos, object_centric=object_centric)
            nearby_score = _costmap[all_nearby_voxels[:, 0], all_nearby_voxels[:, 1], all_nearby_voxels[:, 2]]
            steepest_idx = np.argmin(nearby_score)
            next_pos = all_nearby_voxels[steepest_idx]
            _costmap[current_pos[0].round().astype(int),
                     current_pos[1].round().astype(int),
                     current_pos[2].round().astype(int)] += 1
            path.append(next_pos)
            current_pos = next_pos
            if stop_criteria(current_pos, _costmap, self.config.stop_threshold):
                break
        raw_path = np.array(path)
        logger.info(f'[{get_clock_time(milliseconds=True)}] path optimized: {len(raw_path)} pts')
        processed_path = self._postprocess_path(raw_path, raw_target_map, object_centric=object_centric)
        logger.info(f'[{get_clock_time(milliseconds=True)}] after postprocessing: {len(processed_path)} pts')
        info['start_pos'] = start_pos
        info['target_map'] = target_map
        info['obstacle_map'] = obstacle_map
        info['costmap'] = costmap
        info['costmap_altered'] = _costmap
        info['raw_start_pos'] = raw_start_pos
        info['raw_target_map'] = raw_target_map
        info['raw_obstacle_map'] = raw_obstacle_map
        info['planner_raw_path'] = raw_path.copy()
        info['planner_postprocessed_path'] = processed_path.copy()
        info['targets_voxel'] = np.argwhere(raw_target_map == 1)
        return processed_path, info

    def navigation_optimize(self, start_pos: np.ndarray, target_map: np.ndarray, obstacle_map: np.ndarray, object_centric=False, robot_radius_cells: int = 0):
        """
        config:
            start_pos: (2,) np.ndarray, start position
            target_map: (map_size, map_size) np.ndarray, target_map
            obstacle_map: (map_size, map_size) np.ndarray, obstacle_map
        Returns:
            path: (n, 3) np.ndarray, path
            info: dict, info
        """
        logger.debug(f'[{get_clock_time(milliseconds=True)}] planner start')
        info = dict()
        # make copies
        start_pos, raw_start_pos = start_pos.copy(), start_pos
        target_map, raw_target_map = target_map.copy(), target_map
        obstacle_map, raw_obstacle_map = obstacle_map.copy(), obstacle_map
        # smoothing
        # Centered-gradient target field: instead of EDT(1-mask) which yields a
        # uniformly-flat 0 inside the affordance (planner stops at first edge
        # cell with no pull toward the centre), use Euclidean distance from the
        # affordance centroid. Result: smooth bowl with minimum AT the centroid,
        # rising in all directions — planner is pulled deeper into the goal zone.
        yx_target = np.argwhere(target_map > 0)
        if len(yx_target) > 0:
            cy, cx = yx_target.mean(axis=0)
            yy, xx = np.meshgrid(np.arange(target_map.shape[0]),
                                 np.arange(target_map.shape[1]), indexing='ij')
            target_map = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        else:
            target_map = distance_transform_edt(1 - target_map)
        target_map = normalize_map(target_map)
        obstacle_map = gaussian_filter(obstacle_map, sigma=self.config.obstacle_map_gaussian_sigma)
        obstacle_map = normalize_map(obstacle_map)
        # combine target_map and obstacle_map
        costmap = target_map * self.config.target_map_weight + obstacle_map * self.config.obstacle_map_weight
        costmap = normalize_map(costmap)
        _costmap = costmap.copy()
        # Optionally use A* (W1) instead of greedy descent. A* is global, finds
        # paths around obstacle regions that local greedy can't see, and avoids
        # the "stops far from goal → force-append teleport" failure mode.
        use_astar = bool(self.config.get('use_astar', False)) if hasattr(self.config, 'get') else getattr(self.config, 'use_astar', False)
        # Avoidance enforcement, from config (robocasa_config.yaml planner
        # block). Env vars override so a single run can be flipped without
        # editing the config.
        hard_avoid = self._flag('hard_avoidance', True, 'VOXPOSER_HARD_AVOID')
        strict_goal = self._flag('strict_goal', False, 'VOXPOSER_STRICT_GOAL')
        if use_astar:
            # Threshold 0.5: filters out outer-halo cells (gradient affordance has
            # value < 0.5 there). A* terminates only in inner half-radius zone,
            # pulling the robot deeper than the standard "any halo cell" termination.
            target_mask = raw_target_map > 0.5
            # The avoidance halo decays from 1.0 at the mesh to 0.0 at the
            # commanded radius, so a 0.5 threshold enforces only half of what
            # was asked for -- a 160 cm command produced 0.75 m of clearance.
            # hard_avoidance lowers the threshold so the whole halo is
            # off-limits, for oracle and model-generated code alike.
            _thr = 0.05 if hard_avoid else 0.5
            obs_binary = (raw_obstacle_map > _thr)
            # Scale cost map up so the obstacle gradient is not drowned by
            # the Euclidean heuristic in the fallback (no-inflation) case.
            # Without scaling, costmap ∈ [0, 1] vs h_map ∈ [0, ~141 pixels]
            # → A* picks the shortest path through obstacles when blocked_mask
            # is None. Multiplying by COST_SCALE=30 makes a single high-cost
            # cell roughly equivalent to a 30-cell detour, which roughly
            # matches the worst-case radius in a 100x100 grid.
            COST_SCALE = 30.0
            cm_for_astar = (costmap * COST_SCALE).copy()
            cm_for_astar[target_mask] = 0.0   # ensure target reachable terminator

            # Multi-attempt A* with decreasing inflation radius.
            # Try the largest radius first (collision-safe path). If A* can't
            # find a path that reaches the target, retry with progressively
            # smaller radii — accepting some risk of brushing obstacles in
            # exchange for actually reaching the goal. Final attempt uses no
            # inflation (point robot), guaranteeing a best-effort path always.
            base_r = int(robot_radius_cells or 0)
            if base_r > 0:
                attempts = [base_r, max(1, base_r // 2), 1, 0]
                # Dedupe while preserving order
                seen = set(); attempts = [r for r in attempts if not (r in seen or seen.add(r))]
            else:
                attempts = [0]
            astar_path = None
            chosen_r = None
            for r in attempts:
                if r > 0:
                    from scipy.ndimage import binary_dilation
                    inflated = binary_dilation(obs_binary, iterations=int(r))
                elif hard_avoid:
                    # The final attempt normally drops blocking altogether to
                    # reach the goal at any cost, which is what kept the
                    # avoidance region from ever being absolute. Keep the
                    # region blocked and only give up the robot-radius
                    # inflation; if no path exists, let the episode fail.
                    inflated = obs_binary
                else:
                    inflated = None
                p = _astar_pixel(start_pos, cm_for_astar, target_mask,
                                 blocked_mask=inflated,
                                 strict_goal=strict_goal)
                if p is None or len(p) < 2:
                    continue
                _last = p[-1].astype(int)
                reached = bool(target_mask[_last[0], _last[1]])
                if reached:
                    astar_path, chosen_r = p, r
                    break
                # Best-effort fallback: keep the longest path (closest to goal)
                if astar_path is None or len(p) > len(astar_path):
                    astar_path, chosen_r = p, r
            if astar_path is not None and len(astar_path) > 1:
                raw_path = astar_path
                _last = astar_path[-1].astype(int)
                reached = bool(target_mask[_last[0], _last[1]])
                # Snap last cell to centroid of target_mask so path ends as
                # close to true goal as possible (instead of disk edge). Only
                # when reached=True; pick the disk cell closest to centroid
                # that is also reachable (not blocked at chosen inflation).
                if reached:
                    ys, xs = np.where(target_mask)
                    if ys.size > 0:
                        cy, cx = float(ys.mean()), float(xs.mean())
                        if chosen_r and chosen_r > 0:
                            from scipy.ndimage import binary_dilation as _bd
                            _blocked = _bd(obs_binary, iterations=int(chosen_r))
                        else:
                            _blocked = None
                        best_d = float('inf'); best_cell = None
                        for y, x in zip(ys, xs):
                            if _blocked is not None and _blocked[y, x]:
                                continue
                            d = (y - cy) ** 2 + (x - cx) ** 2
                            if d < best_d:
                                best_d, best_cell = d, (int(y), int(x))
                        if best_cell is not None and tuple(_last) != best_cell:
                            raw_path = np.vstack([raw_path, np.array(best_cell)])
                            _last = np.array(best_cell)
                            logger.info(f'[{get_clock_time(milliseconds=True)}] '
                                        f'snapped last cell to centroid {best_cell} '
                                        f'(was {astar_path[-1].astype(int).tolist()})')
                logger.info(f'[{get_clock_time(milliseconds=True)}] A* path: {len(raw_path)} pts '
                            f'(target reachable={reached}, inflation_used={chosen_r} cells)')
            else:
                logger.warning('A* returned no path even at 0 inflation; falling back to greedy')
                use_astar = False
        if not use_astar:
            # get stop criteria
            stop_criteria = self._get_stop_criteria_navigation()
            # initialize path
            path, current_pos = [start_pos], start_pos
            # optimize (greedy)
            logger.debug(f'[{get_clock_time(milliseconds=True)}] optimizing from {start_pos}')
            for i in range(self.config.max_steps):
                # calculate all nearby voxels around current position
                all_nearby_voxels = self._calculate_nearby_pixel(current_pos, object_centric=object_centric)
                # calculate the score of all nearby voxels
                nearby_score = _costmap[all_nearby_voxels[:, 0], all_nearby_voxels[:, 1]]
                # Find the minimum cost voxel
                steepest_idx = np.argmin(nearby_score)
                try:
                    next_pos = all_nearby_voxels[steepest_idx]
                except Exception as e:
                    logger.error(str(e))
                    breakpoint()
                # increase cost at current position to avoid going back
                _costmap[current_pos[0].round().astype(int),
                         current_pos[1].round().astype(int)] += 1
                # update path and current position
                path.append(next_pos)
                current_pos = next_pos
                # check stop criteria
                if stop_criteria(current_pos, _costmap, self.config.stop_threshold):
                    break
            raw_path = np.array(path)
            logger.info(f'[{get_clock_time(milliseconds=True)}] path optimized (greedy): {len(raw_path)} pts')
        # postprocess path. For A* output (global, no oscillation), skip the
        # high-curvature truncation that was designed for greedy planner —
        # A* paths naturally curve at the goal arrival, and truncating there
        # cuts the path well short of the actual target (verified bug:
        # L0/L2/L6/L7/L8 paths ended 16-57 cells short of goal because the
        # curvature cutoff fired on the natural arrival turn).
        processed_path = self._postprocess_path(
            raw_path, raw_target_map, object_centric=object_centric,
            skip_curvature_cutoff=use_astar)
        logger.info(f'[{get_clock_time(milliseconds=True)}] after postprocessing: {len(processed_path)} pts')
        logger.debug(f'[{get_clock_time(milliseconds=True)}] last waypoint: {processed_path[-1]}')
        # save info
        info['start_pos'] = start_pos
        info['target_map'] = target_map
        info['obstacle_map'] = obstacle_map
        info['costmap'] = costmap
        info['costmap_altered'] = _costmap
        info['raw_start_pos'] = raw_start_pos
        info['raw_target_map'] = raw_target_map
        info['raw_obstacle_map'] = raw_obstacle_map
        info['planner_raw_path'] = raw_path.copy()
        info['planner_postprocessed_path'] = processed_path.copy()
        info['targets_voxel'] = np.argwhere(raw_target_map == 1)
        return processed_path, info

    def _get_stop_criteria(self):
        def no_nearby_equal_criteria(current_pos, costmap, stop_threshold):
            """
            Do not stop if there is a nearby voxel with cost less than current cost + stop_threshold.
            """
            assert np.isnan(costmap).sum() == 0, 'costmap contains nan'
            current_pos_discrete = current_pos.round().clip(0, self.map_size - 1).astype(int)
            current_cost = costmap[current_pos_discrete[0], current_pos_discrete[1], current_pos_discrete[2]]
            nearby_locs = self._calculate_nearby_voxel(current_pos, object_centric=False)
            nearby_equal = np.any(costmap[nearby_locs[:, 0], nearby_locs[:, 1], nearby_locs[:, 2]] < current_cost + stop_threshold)
            if nearby_equal:
                return False
            return True
        return no_nearby_equal_criteria

    def _get_stop_criteria_navigation(self):
        def no_nearby_equal_criteria(current_pos, costmap, stop_threshold):
            """
            Do not stop if there is a nearby voxel with cost less than current cost + stop_threshold.
            """
            assert np.isnan(costmap).sum() == 0, 'costmap contains nan'
            # costmap shape may be rectangular (map_h, map_w) — use shape directly
            _h, _w = costmap.shape[:2]
            current_pos_discrete = current_pos.round().clip([0, 0], [_h - 1, _w - 1]).astype(int)
            current_cost = costmap[current_pos_discrete[0], current_pos_discrete[1]]
            nearby_locs = self._calculate_nearby_pixel(current_pos, object_centric=False, shape=(_h, _w))
            nearby_equal = np.any(costmap[nearby_locs[:, 0], nearby_locs[:, 1]] < current_cost + stop_threshold)
            if nearby_equal:
                return False
            return True
        return no_nearby_equal_criteria

    def _calculate_nearby_voxel(self, current_pos, object_centric=False):
        # create a grid of nearby voxels
        half_size = int(2 * self.map_size / 100)
        offsets = np.arange(-half_size, half_size + 1)
        # our heuristics-based dynamics model only supports planar pushing -> only xy path is considered
        if object_centric:
            offsets_grid = np.array(np.meshgrid(offsets, offsets, [0])).T.reshape(-1, 3)
            # Remove the [0, 0, 0] offset, which corresponds to the current position
            offsets_grid = offsets_grid[np.any(offsets_grid != [0, 0, 0], axis=1)]
        else:
            offsets_grid = np.array(np.meshgrid(offsets, offsets, offsets)).T.reshape(-1, 3)
            # Remove the [0, 0, 0] offset, which corresponds to the current position
            offsets_grid = offsets_grid[np.any(offsets_grid != [0, 0, 0], axis=1)]
        # Calculate all nearby voxel coordinates
        all_nearby_voxels = np.clip(current_pos + offsets_grid, 0, self.map_size - 1)
        # Remove duplicates, if any, caused by clipping
        all_nearby_voxels = np.unique(all_nearby_voxels, axis=0).astype(int)
        return all_nearby_voxels
    
    def _calculate_nearby_pixel(self, current_pos, object_centric=False, shape=None):
        # create a grid of nearby pixel
        # Use shape (map_h, map_w) for rectangular grid clipping; falls back
        # to legacy square self.map_size if shape not given.
        if shape is None:
            _h = _w = self.map_size
        else:
            _h, _w = shape
        # half_size scales with the smaller of the two dims to keep the
        # neighborhood radius isotropic in pixels.
        half_size = max(1, int(2 * min(_h, _w) / 100))
        offsets = np.arange(-half_size, half_size + 1)
        offsets_grid = np.array(np.meshgrid(offsets, offsets)).T.reshape(-1, 2)
        offsets_grid = offsets_grid[np.any(offsets_grid != [0, 0], axis=1)]
        all_nearby_voxels = np.clip(current_pos + offsets_grid, [0, 0], [_h - 1, _w - 1])
        all_nearby_voxels = np.unique(all_nearby_voxels, axis=0).astype(int)
        return all_nearby_voxels
    
    def _postprocess_path(self, path, raw_target_map, object_centric=False,
                          skip_curvature_cutoff=False):
        """
        Apply various postprocessing steps to the path.

        skip_curvature_cutoff: when True, skip the high-curvature truncation
            step. A* paths arrive at goal via natural curve and the cutoff
            would chop off the goal-side of the path. Greedy planner still
            benefits from cutoff (it can oscillate near goal).
        """
        # smooth the path
        savgol_window_size = min(len(path), self.config.savgol_window_size)
        savgol_polyorder = min(self.config.savgol_polyorder, savgol_window_size - 1)
        path = savgol_filter(path, savgol_window_size, savgol_polyorder, axis=0)
        # early cutoff if curvature is too high (greedy planner only)
        if not skip_curvature_cutoff:
            curvature = calc_curvature(path)
            if len(curvature) > 5:
                high_curvature_idx = np.where(curvature[5:] > self.config.max_curvature)[0]
                if len(high_curvature_idx) > 0:
                    high_curvature_idx += 5
                    path = path[:int(0.9 * high_curvature_idx[0])]
        # skip waypoints such that they reach target spacing
        path_trimmed = path[1:-1]
        skip_ratio = None
        if len(path_trimmed) > 1:
            target_spacing = int(self.config['target_spacing'] * self.map_size / 100)
            length = np.linalg.norm(path_trimmed[1:] - path_trimmed[:-1], axis=1).sum()
            if length > target_spacing:
                curr_spacing = np.linalg.norm(path_trimmed[1:] - path_trimmed[:-1], axis=1).mean()
                skip_ratio = np.round(target_spacing / curr_spacing).astype(int)
                if skip_ratio > 1:
                    path_trimmed = path_trimmed[::skip_ratio]
        path = np.concatenate([path[0:1], path_trimmed, path[-1:]])
        # force last position to be one of the target positions
        last_waypoint = path[-1].round().clip(0, self.map_size - 1).astype(int)
        if last_waypoint.shape[0] == 2:
            check_waypoint = raw_target_map[last_waypoint[0], last_waypoint[1]]
        else:
            check_waypoint = raw_target_map[last_waypoint[0], last_waypoint[1], last_waypoint[2]]
        if check_waypoint == 0:
            # find the closest target position
            target_pos = np.argwhere(raw_target_map == 1)
            if len(target_pos) == 0:
                # affordance values may be smoothed below 1; fall back to peak cells
                _peak = raw_target_map.max()
                target_pos = np.argwhere(raw_target_map >= _peak * 0.99) if _peak > 0 else np.array([last_waypoint])
            closest_target_idx = np.argmin(np.linalg.norm(target_pos - last_waypoint, axis=1))
            closest_target = target_pos[closest_target_idx]
            # for object centric motion, we assume we can only push in the xy plane
            if object_centric:
                closest_target[2] = last_waypoint[2]
            path = np.append(path, [closest_target], axis=0)
        # space out path more if task is object centric (so that we can push faster)
        if object_centric:
            k = self.config['pushing_skip_per_k']
            path = np.concatenate([path[k:-1:k], path[-1:]])
        path = path.clip(0, self.map_size-1)
        return path
