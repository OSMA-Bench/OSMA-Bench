"""Plotly-Based Visualizer"""
import plotly.graph_objects as go
import numpy as np
import os
import datetime
from utils.utils import get_logger
import concurrent.futures as _cf

logger = get_logger(__name__)

_PNG_POOL = _cf.ThreadPoolExecutor(max_workers=1)


def _safe_write_image(fig, path, scale=2, timeout=60):
    """kaleido subprocess can deadlock indefinitely (observed 3x: run hangs at
    ~97% CPU right after planning). Bound every export; on timeout skip the
    PNG and shut kaleido down so the next call restarts a fresh subprocess."""
    fut = _PNG_POOL.submit(fig.write_image, path, scale=scale)
    try:
        fut.result(timeout=timeout)
    except _cf.TimeoutError:
        logger.warning(f'[viz] write_image timed out ({timeout}s), skipping {os.path.basename(path)}')
        try:
            import plotly.io._kaleido as _pk
            if getattr(_pk, 'scope', None) is not None:
                _pk.scope._shutdown_kaleido()
        except Exception:
            pass
    except Exception as e:
        logger.warning(f'[viz] write_image failed: {e}')


class ValueMapVisualizer:
    """
    A Plotly-based visualizer for 3D value map and planned path.
    """
    def __init__(self, config):
        self.scene_points = None
        self.save_dir = config['save_dir']
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            os.chmod(self.save_dir, 0o777)
        self.quality = config['quality']
        self.update_quality(self.quality)
        self.map_size = config['map_size']
        self.render_mode = config.get('render_mode', 'scatter')
        self.voxel_scatter_threshold = config.get('voxel_scatter_threshold', 0.5)
        self.max_voxel_points = config.get('max_voxel_points', 6000)
    
    def update_bounds(self, lower, upper):
        self.workspace_bounds_min = lower
        self.workspace_bounds_max = upper
        self.plot_bounds_min = lower - 0.15 * (upper - lower)
        self.plot_bounds_max = upper + 0.15 * (upper - lower)
        xyz_ratio = 1 / (self.workspace_bounds_max - self.workspace_bounds_min)
        scene_scale = np.max(xyz_ratio) / xyz_ratio
        self.scene_scale = scene_scale

    def update_quality(self, quality):
        self.quality = quality
        if self.quality == 'low':
            self.downsample_ratio = 4
            self.max_scene_points = 150000
            self.costmap_opacity = 0.2 * 0.6
            self.costmap_surface_count = 10
        elif self.quality == 'low-full-scene':
            self.downsample_ratio = 4
            self.max_scene_points = 1000000
            self.costmap_opacity = 0.2 * 0.6
            self.costmap_surface_count = 10
        elif self.quality == 'low-half-scene':
            self.downsample_ratio = 4
            self.max_scene_points = 250000
            self.costmap_opacity = 0.2 * 0.6
            self.costmap_surface_count = 10
        elif self.quality == 'medium':
            self.downsample_ratio = 2
            self.max_scene_points = 300000
            self.costmap_opacity = 0.1 * 0.6
            self.costmap_surface_count = 30
        elif self.quality == 'medium-full-scene':
            self.downsample_ratio = 2
            self.max_scene_points = 1000000
            self.costmap_opacity = 0.1 * 0.6
            self.costmap_surface_count = 30
        elif self.quality == 'medium-half-scene':
            self.downsample_ratio = 2
            self.max_scene_points = 500000
            self.costmap_opacity = 0.1 * 0.6
            self.costmap_surface_count = 30
        elif self.quality == 'high':
            self.downsample_ratio = 1
            self.max_scene_points = 500000
            self.costmap_opacity = 0.07 * 0.6
            self.costmap_surface_count = 50
        elif self.quality == 'best':
            self.downsample_ratio = 1
            self.max_scene_points = 500000
            self.costmap_opacity = 0.05 * 0.6
            self.costmap_surface_count = 100
        else:
            raise ValueError(f'Unknown quality: {self.quality}; should be one of [low, medium, high]')

    def update_scene_points(self, points, colors=None):
        points = points.astype(np.float16)
        assert colors.dtype == np.uint8
        self.scene_points = (points, colors)

    def _grid_coords(self, shape):
        """World-frame grid centers for a (downsampled) voxel map, using the
        SAME affine as interfaces.voxel2pc (÷(map_size-1)) so costmap/affordance
        voxels land exactly where target/start markers do. (Previously used an
        mgrid step of extent/map_size, giving a ~1% scale mismatch = up to ~2cm
        offset at the far corner.)"""
        ds = self.downsample_ratio
        lo, hi = self.workspace_bounds_min, self.workspace_bounds_max
        def axis(a):
            k = np.arange(shape[a]) * ds                       # original voxel index
            return lo[a] + k / (self.map_size - 1) * (hi[a] - lo[a])
        return np.meshgrid(axis(0), axis(1), axis(2), indexing='ij')

    def _map_scatter(self, vmap, name, colorscale, threshold, opacity=0.6, size=2.5):
        """Generic voxel-map → world scatter trace (legend-toggleable)."""
        cm = vmap[::self.downsample_ratio, ::self.downsample_ratio, ::self.downsample_ratio]
        gx, gy, gz = self._grid_coords(cm.shape)
        m = cm > threshold
        xs, ys, zs, vs = gx[m], gy[m], gz[m], cm[m]
        if xs.size > self.max_voxel_points:
            idx = np.random.choice(xs.size, self.max_voxel_points, replace=False)
            xs, ys, zs, vs = xs[idx], ys[idx], zs[idx], vs[idx]
        return [go.Scatter3d(x=xs, y=ys, z=zs, mode='markers', name=name,
                             visible='legendonly',
                             marker=dict(size=size, color=vs, colorscale=colorscale, opacity=opacity))]

    def _costmap_traces(self, costmap):
        cm = costmap[::self.downsample_ratio, ::self.downsample_ratio, ::self.downsample_ratio]
        gx, gy, gz = self._grid_coords(cm.shape)
        mode = self.render_mode
        if mode == 'auto':
            mode = 'volume' if (cm > self.voxel_scatter_threshold).sum() <= self.max_voxel_points else 'scatter'
        if mode == 'volume':
            return [go.Volume(x=gx.flatten(), y=gy.flatten(), z=gz.flatten(), value=cm.flatten(),
                              isomin=0, isomax=1, opacity=self.costmap_opacity,
                              surface_count=self.costmap_surface_count, colorscale='Jet',
                              showlegend=True, name='costmap', showscale=False)]
        m = cm > self.voxel_scatter_threshold
        xs, ys, zs, vs = gx[m], gy[m], gz[m], cm[m]
        if xs.size > self.max_voxel_points:
            idx = np.random.choice(xs.size, self.max_voxel_points, replace=False)
            xs, ys, zs, vs = xs[idx], ys[idx], zs[idx], vs[idx]
        return [go.Scatter3d(x=xs, y=ys, z=zs, mode='markers', name='costmap',
                             visible='legendonly',  # off by default (declutter); toggle in HTML
                             marker=dict(size=1.5, color=vs, colorscale='Jet', opacity=0.25))]

    def visualize(self, info, show=False, save=True):
        """visualize the path and relevant info using plotly"""
        planner_info = info['planner_info']
        waypoints_world = np.array([p[0] for p in info['traj_world']])
        start_pos_world = info['start_pos_world']
        assert len(start_pos_world.shape) == 1
        waypoints_world = np.concatenate([start_pos_world[None, ...], waypoints_world], axis=0)
        
        fig_data = []
        # plot the planned path as a SINGLE line+marker trace (one legend entry,
        # not one per segment) so the legend stays clean.
        fig_data.append(go.Scatter3d(
            x=waypoints_world[:, 0], y=waypoints_world[:, 1], z=waypoints_world[:, 2],
            mode='lines+markers', name='path',
            line=dict(width=6, color='orange'),
            marker=dict(size=3, color='red')))
        if planner_info is not None:
            # plot costmap
            if 'costmap' in planner_info:
                fig_data += self._costmap_traces(planner_info['costmap'])
            # value maps as separate toggleable layers (affordance=activation,
            # avoidance=repulsion) — also rendered into latest_maps.png strip.
            if 'raw_target_map' in planner_info and planner_info['raw_target_map'] is not None:
                fig_data += self._map_scatter(np.asarray(planner_info['raw_target_map'], dtype=float),
                                              'affordance', 'Greens', 0.5, opacity=0.9, size=4)
            if 'obstacle_map' in planner_info and planner_info['obstacle_map'] is not None:
                fig_data += self._map_scatter(np.asarray(planner_info['obstacle_map'], dtype=float),
                                              'avoidance', 'Reds', 0.3)
            # plot start position (large, drawn on top of scene)
            if 'start_pos' in planner_info:
                fig_data.append(go.Scatter3d(x=[start_pos_world[0]], y=[start_pos_world[1]], z=[start_pos_world[2]], mode='markers', name='start', marker=dict(size=6, color='blue', symbol='circle')))
            # plot target as dots extracted from target_map
            if 'raw_target_map' in planner_info:
                targets_world = info['targets_world']
                if targets_world is not None and len(targets_world) > 0:
                    fig_data.append(go.Scatter3d(x=targets_world[:, 0], y=targets_world[:, 1], z=targets_world[:, 2], mode='markers', name='target', marker=dict(size=9, color='green', symbol='diamond', opacity=0.9)))

        # visualize scene points
        if self.scene_points is None:
            logger.debug('no scene points to overlay, skipping...')
            scene_points = None
        else:
            scene_points, scene_point_colors = self.scene_points
            # resample to reduce the number of points
            if scene_points.shape[0] > self.max_scene_points:
                resample_idx = np.random.choice(scene_points.shape[0], min(scene_points.shape[0], self.max_scene_points), replace=False)
                scene_points = scene_points[resample_idx]
                if scene_point_colors is not None:
                    scene_point_colors = scene_point_colors[resample_idx]
            if scene_point_colors is None:
                scene_point_colors = scene_points[:, 2]
            else:
                scene_point_colors = scene_point_colors / 255.0
            # add scene points as a dense RGB backdrop — enough points/opacity
            # that the robot arm and fixtures read as recognizable shapes.
            fig_data.append(go.Scatter3d(x=scene_points[:, 0], y=scene_points[:, 1], z=scene_points[:, 2],
                                        mode='markers', name='scene',
                                        marker=dict(size=2.5, color=scene_point_colors, opacity=0.85)))
        
        fig = go.Figure(data=fig_data)
 
        # set bounds and ratio
        fig.update_layout(scene=dict(xaxis=dict(range=[self.plot_bounds_min[0], self.plot_bounds_max[0]], autorange=False),
                                    yaxis=dict(range=[self.plot_bounds_min[1], self.plot_bounds_max[1]], autorange=False),
                                    zaxis=dict(range=[self.plot_bounds_min[2], self.plot_bounds_max[2]], autorange=False)),
                        scene_aspectmode='manual',
                        scene_aspectratio=dict(x=self.scene_scale[0], y=self.scene_scale[1], z=self.scene_scale[2]))

        # do not show grid and axes
        fig.update_layout(scene=dict(xaxis=dict(showgrid=False, showticklabels=False, title='', visible=False),
                                    yaxis=dict(showgrid=False, showticklabels=False, title='', visible=False),
                                    zaxis=dict(showgrid=False, showticklabels=False, title='', visible=False)))

        # set background color as white
        fig.update_layout(template='none')

        # save and show
        # VOX_NO_HTML=1 skips the per-plan plotly HTML dumps (~15MB each). These
        # are only needed by the offline scene_pc renderer; during batch sweeps
        # they pile up to tens of GB and filled the disk (2026-07-19). Keep the
        # small latest_fig.json for re-rendering, skip the big HTMLs.
        _no_html = os.environ.get('VOX_NO_HTML') == '1'
        if save and self.save_dir is not None:
            curr_time = datetime.datetime.now()
            log_id = f'{curr_time.hour}:{curr_time.minute}:{curr_time.second}'
            save_path = os.path.join(self.save_dir, log_id + '.html')
            latest_save_path = os.path.join(self.save_dir, 'latest.html')
            if not _no_html:
                logger.debug(f'saving visualization to {save_path}')
                fig.write_html(save_path)
                logger.debug(f'saving visualization to {latest_save_path}')
                fig.write_html(latest_save_path)
            # machine-readable figure for offline re-rendering (multi-view sheets).
            # Only overwrite when the value-map traces carry data — the episode's
            # final reset-pose plan has empty maps and was wiping the JSON
            # (offline renders then showed scene-only).
            try:
                _has_maps = any(getattr(t, 'name', '') in ('costmap', 'affordance')
                                and t.x is not None and len(t.x) > 0 for t in fig.data)
                if not _no_html and (_has_maps or not os.path.exists(os.path.join(self.save_dir, 'latest_fig.json'))):
                    fig.write_json(os.path.join(self.save_dir, 'latest_fig.json'))
            except Exception:
                pass
            logger.debug(f'saved to {save_path}')
            try:
                if _no_html:
                    raise StopIteration          # skip all PNG snapshots during sweeps
                png_dir = os.environ.get('MANIP_VIZ_PNG_DIR', self.save_dir)
                # snapshot cameras for PNG, then restore the original camera so
                # the returned figure is unchanged for downstream use.
                _orig_camera = fig.layout.scene.camera
                # Default camera = user-selected X2 (mid-diag SE): kitchen
                # structure + robot + start→target path all visible, unoccluded.
                _eye = dict(x=1.2, y=-1.2, z=0.7)
                _ctr = dict(x=0.0, y=0.0, z=-0.1)
                fig.update_layout(scene_camera=dict(eye=_eye, center=_ctr))
                _safe_write_image(fig, os.path.join(png_dir, 'latest_iso.png'))
                # per-plan costmap view: costmap trace visible, same camera —
                # timestamped so a task accumulates one per plan (진행 과정 확인용).
                # costmap view: cost field emphasized (bigger/opaque markers),
                # scene dimmed so the field + path pop instead of drowning in
                # the point cloud.
                _scene_dim = []
                for tr in fig.data:
                    _nm = getattr(tr, 'name', '')
                    if _nm == 'costmap':
                        tr.visible = True
                        try:
                            tr.marker.size = 3.0
                            tr.marker.opacity = 0.55
                        except Exception:
                            pass
                    elif _nm == 'scene':
                        try:
                            _scene_dim.append((tr, tr.marker.opacity))
                            tr.marker.opacity = 0.15
                        except Exception:
                            pass
                _safe_write_image(fig, os.path.join(png_dir, f'costmap_{log_id.replace(":", "-")}.png'))
                _safe_write_image(fig, os.path.join(png_dir, 'latest_costmap.png'))
                for tr in fig.data:
                    if getattr(tr, 'name', '') == 'costmap':
                        tr.visible = 'legendonly'
                        try:
                            tr.marker.size = 1.5
                            tr.marker.opacity = 0.25
                        except Exception:
                            pass
                for tr, _op in _scene_dim:
                    try:
                        tr.marker.opacity = _op
                    except Exception:
                        pass
                fig.update_layout(scene_camera=dict(eye=dict(x=0, y=0, z=1.7), center=dict(x=0, y=0, z=0)))
                _safe_write_image(fig, os.path.join(png_dir, 'latest_top.png'))
                fig.update_layout(scene_camera=_orig_camera)
                # Composite panel: [over-shoulder | top-down / costmap | robot camera]
                # — one image that shows plan + geometry + what the robot sees.
                try:
                    from PIL import Image, ImageDraw
                    def _tile(path, cap):
                        im = Image.open(path).convert('RGB').resize((640, 460))
                        cv = Image.new('RGB', (640, 484), (12, 12, 12))
                        cv.paste(im, (0, 24))
                        ImageDraw.Draw(cv).text((8, 5), cap, fill=(255, 255, 0))
                        return cv
                    tiles = [_tile(os.path.join(png_dir, 'latest_iso.png'), 'over-shoulder 3D (start→target)'),
                             _tile(os.path.join(png_dir, 'latest_top.png'), 'top-down'),
                             _tile(os.path.join(png_dir, 'latest_costmap.png'), 'costmap (cost field + path)')]
                    cam = info.get('camera_rgb')
                    # offscreen sim.render conflicts with the env's own buffer
                    # mid-rollout (black frames) — fall back to the task's saved
                    # initial_topview.png, which is always a valid scene photo.
                    if cam is None or (hasattr(cam, 'max') and np.asarray(cam).max() < 5):
                        _tv = os.path.join(self.save_dir, 'initial_topview.png')
                        cam = np.asarray(Image.open(_tv).convert('RGB')) if os.path.exists(_tv) else None
                    if cam is not None:
                        cam = np.ascontiguousarray(cam)  # [::-1] views have negative strides → PIL garbage
                        if cam.dtype != np.uint8:
                            _mx = float(cam.max()) if cam.size else 1.0
                            cam = (cam * (255.0 if _mx <= 1.001 else 1.0)).clip(0, 255).astype(np.uint8)
                        cim = Image.fromarray(cam[..., :3]).resize((640, 460))
                        cv = Image.new('RGB', (640, 484), (12, 12, 12)); cv.paste(cim, (0, 24))
                        ImageDraw.Draw(cv).text((8, 5), 'robot camera', fill=(255, 255, 0))
                        tiles.append(cv)
                    cols = 2
                    rows = (len(tiles) + cols - 1) // cols
                    panel = Image.new('RGB', (cols * 640, rows * 484), (0, 0, 0))
                    for i, t in enumerate(tiles):
                        panel.paste(t, ((i % cols) * 640, (i // cols) * 484))
                    panel.save(os.path.join(png_dir, f'panel_{log_id.replace(":", "-")}.png'))
                    panel.save(os.path.join(png_dir, 'latest_panel.png'))
                    # value-map strip: affordance / avoidance / costmap, same
                    # over-shoulder camera — 어떤 맵이 계획을 만들었는지 한 줄로.
                    _map_names = ('affordance', 'avoidance', 'costmap')
                    strip_tiles = []
                    fig.update_layout(scene_camera=dict(eye=_eye, center=_ctr))
                    for mn in _map_names:
                        found = False
                        for tr in fig.data:
                            if getattr(tr, 'name', '') in _map_names:
                                tr.visible = True if tr.name == mn else 'legendonly'
                                found = found or (tr.name == mn)
                        if not found:
                            continue
                        _mp = os.path.join(png_dir, f'_map_{mn}.png')
                        _safe_write_image(fig, _mp)
                        strip_tiles.append(_tile(_mp, f'{mn} map'))
                    for tr in fig.data:
                        if getattr(tr, 'name', '') in _map_names:
                            tr.visible = 'legendonly'
                    if strip_tiles:
                        strip = Image.new('RGB', (640 * len(strip_tiles), 484), (0, 0, 0))
                        for i, t in enumerate(strip_tiles):
                            strip.paste(t, (i * 640, 0))
                        strip.save(os.path.join(png_dir, f'maps_{log_id.replace(":", "-")}.png'))
                        strip.save(os.path.join(png_dir, 'latest_maps.png'))
                except Exception as _pe:
                    logger.warning(f'panel compose skipped: {_pe}')
            except Exception as _e:
                logger.warning(f'PNG export skipped (kaleido?): {_e}')
        if show:
            fig.show()

        return fig