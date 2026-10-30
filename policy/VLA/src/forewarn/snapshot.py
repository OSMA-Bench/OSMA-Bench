"""Exact save and restore of a robosuite/robocasa env in place, and a digest to check two states match.

Three parts together determine the next ``env.step``:

* MuJoCo integration state (``mjSTATE_INTEGRATION``: time, qpos, qvel, act, ctrl, applied
  forces, mocap, eq_active, userdata, plugin state and the solver warm start).
* Model arrays the task writes while it runs. The navigation env turns the posed human toward
  the robot through ``model.body_quat`` on every control step.
* Python-side episode state reachable from the env object: step counters, observable buffers,
  controller goals, contact accumulators, trajectory histories and RNG streams, recorded by
  attribute path and written back into the same objects.

plus the global numpy and python RNGs. Nothing here proves a snapshot is complete; the steerer's
commit check (the replayed env's ``digest`` equals the chosen branch's) is what measures it.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


OWNED_MODULE_PREFIXES = ("robosuite.", "robocasa.")
# Bound to one simulator or GL context. Model objects are not skipped: some carry per-step
# state, e.g. PandaGripper.format_action integrates current_action on every step.
SKIPPED_MODULE_PREFIXES = ("robosuite.utils.binding_utils", "robosuite.renderers")
MODEL_ARRAYS = ("body_pos", "body_quat", "geom_pos", "geom_quat", "geom_rgba", "site_pos", "site_quat", "site_rgba")
PLAIN_SCALAR_TYPES = (bool, int, float, complex, str, bytes, type(None), np.generic)
MAX_PLAIN_DEPTH = 8


@dataclass
class Leaf:
    value: Any


@dataclass
class RngNode:
    kind: str
    state: Any


@dataclass
class ObjectNode:
    type_name: str
    attrs: dict[str, Any]
    # Every attribute the object had, captured or not. Restore deletes attributes outside this
    # set, because tasks create some lazily (``intrusion``, ``traj_info``) and test for them.
    names: frozenset[str] = frozenset()


@dataclass
class SequenceNode:
    type_name: str
    items: list[Any]


@dataclass
class MappingNode:
    type_name: str
    items: dict[Any, Any]
    key_set: frozenset[Any] = frozenset()


@dataclass
class AliasNode:
    path: str


@dataclass
class EnvSnapshot:
    mj_state: np.ndarray
    model_arrays: dict[str, np.ndarray]
    python_state: Any
    global_rng: tuple[Any, Any]
    skipped_types: dict[str, int] = field(default_factory=dict)


class SnapshotRestoreError(RuntimeError):
    pass


class _Missing:
    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _Missing()


def is_owned(value: Any) -> bool:
    module = type(value).__module__ or ""
    return (
        module.startswith(OWNED_MODULE_PREFIXES)
        and not module.startswith(SKIPPED_MODULE_PREFIXES)
        and not isinstance(value, type)
        and hasattr(value, "__dict__")
    )


def is_plain(value: Any, depth: int = 0) -> bool:
    if isinstance(value, PLAIN_SCALAR_TYPES):
        return True
    if depth >= MAX_PLAIN_DEPTH:
        return False
    if isinstance(value, np.ndarray):
        # Object arrays show up as np.array(None), e.g. robot.recent_torques after a push.
        return value.dtype != object or all(is_plain(item, depth + 1) for item in value.flat)
    if isinstance(value, dict):
        return all(is_plain(k, depth + 1) and is_plain(v, depth + 1) for k, v in value.items())
    if isinstance(value, (list, tuple, set, frozenset, collections.deque)):
        return all(is_plain(item, depth + 1) for item in value)
    return False


class _Capturer:
    def __init__(self) -> None:
        self.seen: dict[int, str] = {}
        self.keepalive: list[Any] = []  # ids in ``seen`` stay unique while the capture runs
        self.skipped: collections.Counter[str] = collections.Counter()

    def node(self, value: Any, path: str) -> Any:
        if isinstance(value, np.random.Generator):
            return RngNode("generator", copy.deepcopy(value.bit_generator.state))
        if isinstance(value, np.random.RandomState):
            return RngNode("random_state", value.get_state())
        if is_plain(value):
            return Leaf(copy.deepcopy(value))
        if is_owned(value):
            if id(value) in self.seen:
                return AliasNode(self.seen[id(value)])
            self.seen[id(value)] = path
            self.keepalive.append(value)
            members = list(vars(value).items())
            attrs = {name: self.node(child, f"{path}.{name}") for name, child in members}
            attrs = {name: node for name, node in attrs.items() if node is not None}
            return ObjectNode(type(value).__name__, attrs, frozenset(name for name, _ in members))
        if isinstance(value, (list, tuple)):
            items = [self.node(child, f"{path}[{index}]") for index, child in enumerate(value)]
            return None if all(item is None for item in items) else SequenceNode(type(value).__name__, items)
        if isinstance(value, dict):
            keys = [key for key in list(value) if is_plain(key)]
            items = {key: self.node(value[key], f"{path}[{key!r}]") for key in keys}
            items = {key: node for key, node in items.items() if node is not None}
            return MappingNode(type(value).__name__, items, frozenset(keys)) if items else None
        self.skipped[f"{type(value).__module__}.{type(value).__name__}"] += 1
        return None


def _restore_leaf(current: Any, setter: Callable[[Any], None], saved: Any) -> None:
    """Write into the live container where possible, so references held elsewhere see the restored value."""
    if (
        isinstance(saved, np.ndarray)
        and isinstance(current, np.ndarray)
        and current.shape == saved.shape
        and current.dtype == saved.dtype != object
        and current.flags.writeable
    ):
        np.copyto(current, saved)
    elif type(saved) is list and type(current) is list:
        current[:] = copy.deepcopy(saved)
    elif isinstance(saved, dict) and type(current) is type(saved):
        current.clear()
        current.update(copy.deepcopy(saved))
    elif isinstance(saved, collections.deque) and isinstance(current, collections.deque) and current.maxlen == saved.maxlen:
        current.clear()
        current.extend(copy.deepcopy(saved))
    else:
        setter(copy.deepcopy(saved))


def _restore_node(getter: Callable[[], Any], setter: Callable[[Any], None] | None, node: Any, path: str, problems: list[str]) -> None:
    if node is None or isinstance(node, AliasNode):
        return
    try:
        current = getter()
    except (AttributeError, KeyError, IndexError):
        current = _MISSING
    if isinstance(node, Leaf):
        if setter is not None:
            _restore_leaf(current, setter, node.value)
        elif digest_of(current) != digest_of(node.value):
            problems.append(f"{path}: immutable container element differs")
    elif isinstance(node, RngNode):
        if node.kind == "generator" and isinstance(current, np.random.Generator):
            current.bit_generator.state = copy.deepcopy(node.state)
        elif node.kind == "random_state" and isinstance(current, np.random.RandomState):
            current.set_state(node.state)
        else:
            problems.append(f"{path}: expected {node.kind}, found {type(current).__name__}")
    elif isinstance(node, ObjectNode):
        if type(current).__name__ != node.type_name:
            problems.append(f"{path}: expected {node.type_name}, found {type(current).__name__}")
            return
        for name in [name for name in vars(current) if name not in node.names]:
            delattr(current, name)
        for name, child in node.attrs.items():
            _restore_node(
                lambda obj=current, attr=name: getattr(obj, attr),
                lambda value, obj=current, attr=name: setattr(obj, attr, value),
                child,
                f"{path}.{name}",
                problems,
            )
    elif isinstance(node, SequenceNode):
        if not isinstance(current, (list, tuple)) or len(current) != len(node.items):
            problems.append(f"{path}: sequence shape differs")
            return
        for index, child in enumerate(node.items):
            _restore_node(
                lambda seq=current, i=index: seq[i],
                (lambda value, seq=current, i=index: seq.__setitem__(i, value)) if isinstance(current, list) else None,
                child,
                f"{path}[{index}]",
                problems,
            )
    elif isinstance(node, MappingNode):
        if not isinstance(current, dict):
            problems.append(f"{path}: expected mapping, found {type(current).__name__}")
            return
        for key in [key for key in current if is_plain(key) and key not in node.key_set]:
            del current[key]
        for key, child in node.items.items():
            _restore_node(
                lambda mapping=current, k=key: mapping[k],
                lambda value, mapping=current, k=key: mapping.__setitem__(k, value),
                child,
                f"{path}[{key!r}]",
                problems,
            )
    else:
        raise TypeError(f"Unknown snapshot node {type(node).__name__} at {path}")


def _mujoco_state_spec():
    import mujoco

    return mujoco, mujoco.mjtState.mjSTATE_INTEGRATION


def capture_env(env: Any) -> EnvSnapshot:
    mujoco, spec = _mujoco_state_spec()
    model, data = env.sim.model._model, env.sim.data._data
    mj_state = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
    mujoco.mj_getState(model, data, mj_state, spec)
    capturer = _Capturer()
    python_state = capturer.node(env, "env")
    return EnvSnapshot(
        mj_state=mj_state,
        model_arrays={name: np.array(getattr(model, name), copy=True) for name in MODEL_ARRAYS},
        python_state=python_state,
        global_rng=(np.random.get_state(), random.getstate()),
        skipped_types=dict(capturer.skipped),
    )


def restore_env(env: Any, snapshot: EnvSnapshot) -> None:
    """Apply ``snapshot`` to ``env`` in place: Python state, model arrays, MuJoCo state, global RNG."""
    problems: list[str] = []
    holder = {"env": env}
    _restore_node(lambda: holder["env"], None, snapshot.python_state, "env", problems)
    mujoco, spec = _mujoco_state_spec()
    model, data = env.sim.model._model, env.sim.data._data
    for name, saved in snapshot.model_arrays.items():
        target = getattr(model, name)
        if target.shape != saved.shape:
            problems.append(f"model.{name}: shape {target.shape} != {saved.shape}")
        else:
            target[...] = saved
    if snapshot.mj_state.shape != (mujoco.mj_stateSize(model, spec),):
        problems.append(f"mj_state: shape {snapshot.mj_state.shape} does not fit this model")
    if problems:
        raise SnapshotRestoreError("; ".join(problems[:20]))
    mujoco.mj_setState(model, data, snapshot.mj_state, spec)
    # Derived quantities for rendering. mj_fwdConstraint rewrites qacc_warmstart, so the saved
    # warm start is put back afterwards; the next mj_step1 recomputes everything else.
    mujoco.mj_forward(model, data)
    mujoco.mj_setState(model, data, snapshot.mj_state, spec)
    np.random.set_state(snapshot.global_rng[0])
    random.setstate(snapshot.global_rng[1])


def _feed(h: Any, value: Any) -> None:
    """Hash ``value`` by content: same dtype, shape and bits for arrays, same structure for containers and nodes."""
    if isinstance(value, np.generic):
        value = np.asarray(value)
    if isinstance(value, np.ndarray) and value.dtype != object:
        h.update(f"a{value.dtype.str}{value.shape}".encode())
        h.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, np.ndarray) or isinstance(value, (list, tuple, collections.deque)):
        items = value.flat if isinstance(value, np.ndarray) else value
        h.update(f"{type(value).__name__}{len(value) if not isinstance(value, np.ndarray) else value.shape}[".encode())
        for item in items:
            _feed(h, item)
        h.update(b"]")
    elif isinstance(value, dict):
        h.update(f"dict{len(value)}{{".encode())
        for key in sorted(value, key=repr):
            _feed(h, key)
            _feed(h, value[key])
        h.update(b"}")
    elif isinstance(value, (set, frozenset)):
        _feed(h, ("set", sorted(value, key=repr)))
    elif isinstance(value, Leaf):
        _feed(h, ("Leaf", value.value))
    elif isinstance(value, RngNode):
        _feed(h, ("Rng", value.kind, value.state))
    elif isinstance(value, ObjectNode):
        _feed(h, ("Object", value.type_name, sorted(value.names), value.attrs))
    elif isinstance(value, SequenceNode):
        _feed(h, ("Sequence", value.type_name, value.items))
    elif isinstance(value, MappingNode):
        _feed(h, ("Mapping", value.type_name, sorted(value.key_set, key=repr), value.items))
    elif isinstance(value, AliasNode):
        _feed(h, ("Alias", value.path))
    else:
        h.update(f"{type(value).__name__}:{value!r};".encode())


def digest_of(value: Any) -> str:
    h = hashlib.sha256()
    _feed(h, value)
    return h.hexdigest()[:16]


def digest(snapshot: EnvSnapshot) -> dict[str, str]:
    """Per-part content hash; two envs continue identically when all four parts match."""
    return {
        "mj_state": digest_of(snapshot.mj_state),
        "model_arrays": digest_of(snapshot.model_arrays),
        "python": digest_of(snapshot.python_state),
        "global_rng": digest_of(snapshot.global_rng),
    }
