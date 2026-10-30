import os
import logging
import numpy as np
import numpy as np
import plotly.graph_objects as go
import datetime
from transforms3d.quaternions import mat2quat

def set_lmp_objects(lmps, objects):
    if isinstance(lmps, dict):
        lmps = lmps.values()
    for lmp in lmps:
        lmp._context = f'objects = {objects}'

def set_lmp_images(lmps, images, cam_names=None):
    """Set camera images on all LMPs for VLM-based inference.

    Args:
        lmps: dict or iterable of LMP instances.
        images: list of numpy RGB arrays, or None to clear.
        cam_names: list of camera name strings matching images, or None.
    """
    if isinstance(lmps, dict):
        lmps = lmps.values()
    for lmp in lmps:
        lmp._images = images
        lmp._image_labels = cam_names

def get_clock_time(milliseconds=False):
    curr_time = datetime.datetime.now()
    if milliseconds:
        return f'{curr_time.hour}:{curr_time.minute}:{curr_time.second}.{curr_time.microsecond // 1000}'
    else:
        return f'{curr_time.hour}:{curr_time.minute}:{curr_time.second}'

class TermColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG:    TermColors.OKCYAN,
        logging.WARNING:  TermColors.WARNING,
        logging.ERROR:    TermColors.FAIL,
    }
    def format(self, record):
        color = self.COLORS.get(record.levelno, '')
        msg = super().format(record)
        return f"{color}{msg}{TermColors.ENDC}" if color else msg


_OUR_LOGGERS = []
_HANDLER = None


def setup_logging(verbose=False):
    """Configure logging. Our loggers are isolated (propagate=False).
    Default: INFO only (no third-party noise).
    -v: DEBUG from our loggers + WARNING from third-party."""
    global _HANDLER
    our_level = logging.DEBUG if verbose else logging.INFO
    _HANDLER = logging.StreamHandler()
    _HANDLER.setFormatter(ColorFormatter("%(message)s"))
    _HANDLER.setLevel(logging.DEBUG)
    # root: -v shows third-party WARNING, otherwise silent
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING if verbose else logging.CRITICAL)
    root.addHandler(_HANDLER)
    # suppress robosuite noise: remove its handlers and set level
    for name in ('robosuite', 'robosuite_logs'):
        tp = logging.getLogger(name)
        tp.handlers.clear()
        tp.setLevel(logging.CRITICAL if not verbose else logging.WARNING)
        tp.propagate = False  # don't let it reach root either
    # attach handler to our loggers
    for lgr in _OUR_LOGGERS:
        lgr.handlers.clear()
        lgr.addHandler(_HANDLER)
        lgr.setLevel(our_level)


def get_logger(name):
    """Return an isolated module logger (propagate=False, own handler)."""
    lgr = logging.getLogger(name)
    if lgr not in _OUR_LOGGERS:
        _OUR_LOGGERS.append(lgr)
        lgr.setLevel(logging.INFO)
        lgr.propagate = False
        if _HANDLER is not None:
            lgr.addHandler(_HANDLER)
    return lgr


_FILE_HANDLER = None


def add_file_handler(path):
    """Add a plain-text FileHandler to all our loggers (no ANSI codes)."""
    global _FILE_HANDLER
    import re as _re
    _ansi = _re.compile(r'\x1b\[[0-9;]*m')

    class _PlainFormatter(logging.Formatter):
        def format(self, record):
            return _ansi.sub('', super().format(record))

    _FILE_HANDLER = logging.FileHandler(path, encoding='utf-8')
    _FILE_HANDLER.setFormatter(_PlainFormatter("%(message)s"))
    _FILE_HANDLER.setLevel(logging.DEBUG)
    for lgr in _OUR_LOGGERS:
        lgr.addHandler(_FILE_HANDLER)
    return _FILE_HANDLER


def remove_file_handler():
    """Remove the file handler added by add_file_handler."""
    global _FILE_HANDLER
    if _FILE_HANDLER is None:
        return
    for lgr in _OUR_LOGGERS:
        lgr.removeHandler(_FILE_HANDLER)
    _FILE_HANDLER.close()
    _FILE_HANDLER = None

def load_prompt(prompt_fname):
    # get current directory
    curr_dir = 'src'
    # get full path to file
    if '/' in prompt_fname:
        parts = prompt_fname.split('/')
        full_path = os.path.join(curr_dir, 'prompts', *parts)
    else:
        full_path = os.path.join(curr_dir, 'prompts', prompt_fname)
    # if variant file doesn't exist, fall back to robocasa_navigation baseline
    if not os.path.exists(full_path) and '/' in prompt_fname:
        parts = prompt_fname.split('/')
        env_dir = parts[0]
        if env_dir.startswith('robocasa_navigation_'):
            fallback_path = os.path.join(curr_dir, 'prompts', 'robocasa_navigation', *parts[1:])
            if os.path.exists(fallback_path):
                full_path = fallback_path
    # read file
    with open(full_path, 'r') as f:
        contents = f.read().strip()
    return contents

def normalize_vector(x, eps=1e-6):
    """normalize a vector to unit length"""
    x = np.asarray(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        return np.zeros_like(x) if norm < eps else (x / norm)
    elif x.ndim == 2:
        norm = np.linalg.norm(x, axis=1)  # (N,)
        normalized = np.zeros_like(x)
        normalized[norm > eps] = x[norm > eps] / norm[norm > eps][:, None]
        return normalized

def normalize_map(map):
    """normalization voxel maps to [0, 1] without producing nan"""
    denom = map.max() - map.min()
    if denom == 0:
        return map
    return (map - map.min()) / denom

def calc_curvature(path):
    dx = np.gradient(path[:, 0])
    dy = np.gradient(path[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    if path.shape[1] == 3:
        dz = np.gradient(path[:, 2])
        ddz = np.gradient(dz)
        curvature = np.sqrt((ddy * dx - ddx * dy)**2 + (ddz * dx - ddx * dz)**2 + (ddz * dy - ddy * dz)**2) / np.power(dx**2 + dy**2 + dz**2, 3/2)
    else:
        curvature = np.abs(ddy * dx - ddx * dy) / np.power(dx**2 + dy**2, 3/2)
    # convert any nan to 0
    curvature[np.isnan(curvature)] = 0
    return curvature

class IterableDynamicObservation:
    """acts like a list of DynamicObservation objects, initialized with a function that evaluates to a list"""
    def __init__(self, func):
        assert callable(func), 'func must be callable'
        self.func = func
        self._validate_func_output()

    def _validate_func_output(self):
        evaluated = self.func()
        assert isinstance(evaluated, list), 'func must evaluate to a list'

    def __getitem__(self, index):
        def helper():
            evaluated = self.func()
            item = evaluated[index]
            # assert isinstance(item, Observation), f'got type {type(item)} instead of Observation'
            return item
        return DynamicObservation(helper)

    def __len__(self):
        return len(self.func())

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

    def __getattr__(self, key):
        # Delegate single-attribute access to first element so LLM code like
        # `obs = parse_query_obj('human'); obs.position` works without indexing.
        if key == 'func':
            raise AttributeError(key)
        return getattr(self[0], key)

    def __call__(self):
        static_list = self.func()
        return static_list

class DynamicObservation:
    """acts like dict observation but initialized with a function such that it uses the latest info"""
    def __init__(self, func):
        try:
            assert callable(func) and not isinstance(func, dict), 'func must be callable or cannot be a dict'
        except AssertionError as e:
            get_logger(__name__).error(str(e))
            import pdb; pdb.set_trace()
        self.func = func
    
    _FALLBACK = {'position': np.array([0.0, 0.0, 0.0]), 'normal': np.array([0.0, 0.0, 1.0]),
                 'name': '_fallback', 'aabb': np.zeros((2, 3)),
                 '_position_world': np.array([0.0, 0.0, 0.0]),
                 '_point_cloud_world': np.zeros((1, 3))}

    def __get__(self, key):
        evaluated = self.func()
        if evaluated is None:
            # Route through Observation so its missing-key fallbacks apply
            # (aabb → degenerate box, name → '', normal → +z). Returning
            # _FALLBACK.get(key, None) here leaked None into LLM map code
            # (`(mn), (mx) = obj.aabb` → TypeError) — the parse_query_obj LMP
            # wraps its code in `def ret_val()`, so a "None result" is really
            # a function returning None and every upstream None-check misses.
            evaluated = Observation(dict(self._FALLBACK))
        # LLMs commonly reach for `.handle_position` (and similar `*_position`
        # attributes) that the observation doesn't expose — fall back to the
        # object centroid `position` instead of raising KeyError.
        if key not in evaluated and key.endswith('position') and 'position' in evaluated:
            key = 'position'
        if isinstance(evaluated[key], np.ndarray):
            return evaluated[key].copy()
        return evaluated[key]

    def __getattr__(self, key):
        return self.__get__(key)

    def __getitem__(self, key):
        return self.__get__(key)

    def __call__(self):
        static_obs = self.func()
        if static_obs is None:
            static_obs = self._FALLBACK
        if not isinstance(static_obs, Observation):
            if not isinstance(static_obs, dict):
                get_logger(__name__).warning(
                    f'DynamicObservation: func returned {type(static_obs).__name__}, using fallback'
                )
                static_obs = self._FALLBACK
            static_obs = Observation(static_obs)
        return static_obs

class Observation(dict):
    def __init__(self, obs_dict):
        super().__init__(obs_dict)
        self.obs_dict = obs_dict
    
    def __getattr__(self, key):
        return self.__getitem__(key)

    def __getitem__(self, key):
        # `.handle_position` etc. → fall back to object centroid `position`.
        if key not in self.obs_dict and key.endswith('position') and 'position' in self.obs_dict:
            return self.obs_dict['position']
        # Missing-key fallbacks so LMP-generated code never crashes on a
        # partially-populated observation (e.g. env obs dicts without 'name').
        if key not in self.obs_dict:
            if key == 'name':
                return ''
            if key == 'aabb' and 'position' in self.obs_dict:
                _p = np.asarray(self.obs_dict['position'])
                return np.array([_p, _p])
            if key == 'normal':
                return np.array([0, 0, 1], dtype=np.float32)
        _val = self.obs_dict[key]
        # A stored-but-None aabb also breaks `(mn), (mx) = obj.aabb` unpacking.
        if _val is None and key == 'aabb' and 'position' in self.obs_dict:
            _p = np.asarray(self.obs_dict['position'])
            return np.array([_p, _p])
        return _val

    def __getstate__(self):
        return self.obs_dict
    
    def __setstate__(self, state):
        self.obs_dict = state

def pointat2quat(pointat):
    """
    calculate quaternion from pointat vector
    """
    up = np.array(pointat, dtype=np.float32)
    up = normalize_vector(up)
    rand_vec = np.array([1, 0, 0], dtype=np.float32)
    rand_vec = normalize_vector(rand_vec)
    # make sure that the random vector is close to desired direction
    if np.abs(np.dot(rand_vec, up)) > 0.99:
        rand_vec = np.array([0, 1, 0], dtype=np.float32)
        rand_vec = normalize_vector(rand_vec)
    left = np.cross(up, rand_vec)
    left = normalize_vector(left)
    forward = np.cross(left, up)
    forward = normalize_vector(forward)
    rotmat = np.eye(3).astype(np.float32)
    rotmat[:3, 0] = forward
    rotmat[:3, 1] = left
    rotmat[:3, 2] = up
    quat_wxyz = mat2quat(rotmat)
    return quat_wxyz

def visualize_points(point_cloud, point_colors=None, show=True):
    """visualize point clouds using plotly"""
    if point_colors is None:
        point_colors = point_cloud[:, 2]
    fig = go.Figure(data=[go.Scatter3d(x=point_cloud[:, 0], y=point_cloud[:, 1], z=point_cloud[:, 2],
                                    mode='markers', marker=dict(size=3, color=point_colors, opacity=1.0))])
    if show:
        fig.show()
    else:
        # save to html
        fig.write_html('temp_pc.html')
        get_logger(__name__).debug('Point cloud saved to temp_pc.html')

def _process_llm_index(indices, array_shape):
    """
    processing function for returned voxel maps (which are to be manipulated by LLMs)
    handles non-integer indexing
    handles negative indexing with manually designed special cases
    """
    if isinstance(indices, int) or isinstance(indices, np.int64) or isinstance(indices, np.int32) or isinstance(indices, np.int16) or isinstance(indices, np.int8):
        processed = indices if indices >= 0 or indices == -1 else 0
        assert len(array_shape) == 1, "1D array expected"
        processed = min(processed, array_shape[0] - 1)
    elif isinstance(indices, float) or isinstance(indices, np.float64) or isinstance(indices, np.float32) or isinstance(indices, np.float16):
        processed = np.round(indices).astype(int) if indices >= 0 or indices == -1 else 0
        assert len(array_shape) == 1, "1D array expected"
        processed = min(processed, array_shape[0] - 1)
    elif isinstance(indices, slice):
        start, stop, step = indices.start, indices.stop, indices.step
        if start is not None:
            start = np.round(start).astype(int)
        if stop is not None:
            stop = np.round(stop).astype(int)
        if step is not None:
            step = np.round(step).astype(int)
        # only convert the case where the start is negative and the stop is positive/negative
        if (start is not None and start < 0) and (stop is not None):
            if stop >= 0:
                processed = slice(0, stop, step)
            else:
                processed = slice(0, 0, step)
        else:
            processed = slice(start, stop, step)
    elif isinstance(indices, tuple) or isinstance(indices, list):
        processed = tuple(
            _process_llm_index(idx, (array_shape[i],)) for i, idx in enumerate(indices)
        )
    elif isinstance(indices, np.ndarray):
        get_logger(__name__).warning("[IndexingWrapper] numpy array indexing was converted to list")
        processed = _process_llm_index(indices.tolist(), array_shape)
    else:
        get_logger(__name__).error(f"[IndexingWrapper] {indices} (type: {type(indices)}) not supported")
        raise TypeError("Indexing type not supported")
    # give warning if index was negative
    if processed != indices:
        get_logger(__name__).warning(f"[IndexingWrapper] index was changed from {indices} to {processed}")
    # print(f"[IndexingWrapper] {idx} -> {processed}")
    return processed

class VoxelIndexingWrapper:
    """
    LLM indexing wrapper that uses _process_llm_index to process indexing
    behaves like a numpy array
    """
    def __init__(self, array):
        self.array = array

    def __getitem__(self, idx):
        return self.array[_process_llm_index(idx, tuple(self.array.shape))]
    
    def __setitem__(self, idx, value):
        self.array[_process_llm_index(idx, tuple(self.array.shape))] = value
    
    def __repr__(self) -> str:
        return self.array.__repr__()
    
    def __str__(self) -> str:
        return self.array.__str__()
    
    def __eq__(self, other):
        return self.array == other
    
    def __ne__(self, other):
        return self.array != other
    
    def __lt__(self, other):
        return self.array < other
    
    def __le__(self, other):
        return self.array <= other
    
    def __gt__(self, other):
        return self.array > other
    
    def __ge__(self, other):
        return self.array >= other
    
    def __add__(self, other):
        return self.array + other
    
    def __sub__(self, other):
        return self.array - other
    
    def __mul__(self, other):
        return self.array * other
    
    def __truediv__(self, other):
        return self.array / other
    
    def __floordiv__(self, other):
        return self.array // other
    
    def __mod__(self, other):
        return self.array % other
    
    def __divmod__(self, other):
        return self.array.__divmod__(other)
    
    def __pow__(self, other):
        return self.array ** other
    
    def __lshift__(self, other):
        return self.array << other
    
    def __rshift__(self, other):
        return self.array >> other
    
    def __and__(self, other):
        return self.array & other
    
    def __xor__(self, other):
        return self.array ^ other
    
    def __or__(self, other):
        return self.array | other
    
    def __radd__(self, other):
        return other + self.array
    
    def __rsub__(self, other):
        return other - self.array
    
    def __rmul__(self, other):
        return other * self.array
    
    def __rtruediv__(self, other):
        return other / self.array
    
    def __rfloordiv__(self, other):
        return other // self.array
    
    def __rmod__(self, other):
        return other % self.array
    
    def __rdivmod__(self, other):
        return other.__divmod__(self.array)
    
    def __rpow__(self, other):
        return other ** self.array
    
    def __rlshift__(self, other):
        return other << self.array
    
    def __rrshift__(self, other):
        return other >> self.array
    
    def __rand__(self, other):
        return other & self.array
    
    def __rxor__(self, other):
        return other ^ self.array
    
    def __ror__(self, other):
        return other | self.array
    
    def __getattribute__(self, name):
        if name == "array":
            return super().__getattribute__(name)
        elif name == "__getitem__":
            return super().__getitem__
        elif name == "__setitem__":
            return super().__setitem__
        else:
            # print(name)
            return super().array.__getattribute__(name)
    
    def __getattr__(self, name):
        return self.array.__getattribute__(name)