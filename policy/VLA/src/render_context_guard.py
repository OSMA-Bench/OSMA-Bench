"""Guard robosuite's offscreen render-context lifecycle against cross-context GL frees.

Background (2026-09-06; team-internal migration notes §10-1, summary in README.md § "Layout"). robosuite's
``MujocoEnv.__init__`` already performs one reset (sim #1 + ``MjRenderContextOffscreen`` #1 with
its own EGL context). The rollout's ``env.reset()`` then builds sim #2 + render context #2, but
because robocasa constructs envs with ``renderer="mjviewer"`` (robosuite ``base.py`` only calls
``_destroy_sim`` for ``renderer == "mujoco"``), sim #1 <-> context #1 survive as a reference cycle
(``MjSim._render_context_offscreen`` <-> ``MjRenderContext.sim``). Whenever the cyclic GC later
collects them, ``MjRenderContext.__del__`` calls ``MjrContext.free()``, which MuJoCo documents as
"frees resources in the *current active* OpenGL context" — that is context #2's framebuffer and
textures (GL object names are per context and identical) — and every later frame (the topview
video **and the policy cameras**) is garbage. Whether the collection lands inside an episode
depends only on allocation history, which is why the corruption appeared with the metric-cleanup
robocasa trees (an extra YAML load at import moved the gen-2 collection) and not in the sampled
``0f5fbc83`` videos. Bisected and reproduced with the real runner: 8/8 corrupted → 0/8 with this
guard (scratch ``v3_rc_shim_probe_0906.sh``).

Two measures, robosuite-version agnostic and no-ops when robosuite is not importable:

* :func:`install` replaces ``MjRenderContext.__del__`` with :func:`release_render_context`, which
  makes the context's *own* EGL context current before freeing its ``MjrContext``, destroys the EGL
  context, and restores the previously current one. If its own context cannot be made current the
  ``MjrContext`` wrapper is kept alive instead of freed (its destructor would otherwise repeat the
  foreign free); destroying the EGL context releases the GL objects anyway.
* :func:`reap_stale_render_contexts` runs right after ``env.reset()``: ``gc.collect()`` so leftover
  contexts die *now*, under the guard, instead of at an arbitrary step, then re-makes the live
  context current and reports the counts for ``meta.json``.
"""

from __future__ import annotations

import gc
from typing import Any, Callable

_STATE: dict[str, Any] = {"installed": False, "original_del": None}
# MjrContext wrappers that could not be freed under their own GL context. Their C++ destructor
# calls mjr_freeContext in whatever context is current, so they must never be collected mid-process.
_KEEP_ALIVE: list[Any] = []

_RELEASED_ATTRS = ("con", "gl_ctx", "scn", "cam", "vopt", "pert")


def _address(ctx: Any) -> int:
    try:
        return int(ctx.address)
    except (AttributeError, TypeError, ValueError):
        return 0


def _egl_module() -> Any:
    from OpenGL import EGL  # type: ignore

    return EGL


def _egl_display() -> Any:
    from robosuite.renderers.context import egl_context  # type: ignore

    return egl_context.EGL_DISPLAY


def release_render_context(
    rc: Any,
    *,
    egl: Any | None = None,
    display_getter: Callable[[], Any] | None = None,
    keep_alive: list[Any] | None = None,
) -> dict[str, Any]:
    """Free ``rc``'s MjrContext under ``rc``'s own EGL context and destroy that EGL context.

    Safe to call from ``__del__``: never raises, idempotent (released attributes are removed).
    Returns an event dict describing what happened.
    """
    keep_alive = _KEEP_ALIVE if keep_alive is None else keep_alive
    event = {
        "switched": False,
        "mjr_freed": False,
        "kept_alive": False,
        "gl_freed": False,
        "restored": False,
    }
    gl_ctx = getattr(rc, "gl_ctx", None)
    con = getattr(rc, "con", None)
    own = getattr(gl_ctx, "_context", None)
    own_addr = _address(own) if own is not None else 0

    prev = None
    prev_addr = 0
    try:
        if egl is None:
            egl = _egl_module()
        prev = egl.eglGetCurrentContext()
        prev_addr = _address(prev)
    except Exception:
        egl = None

    try:
        if own_addr and egl is not None:
            if own_addr != prev_addr:
                gl_ctx.make_current()
                event["switched"] = True
            if con is not None and _address(egl.eglGetCurrentContext()) == own_addr:
                con.free()
                event["mjr_freed"] = True
    except Exception:
        pass
    if con is not None and not event["mjr_freed"]:
        keep_alive.append(con)
        event["kept_alive"] = True

    try:
        if gl_ctx is not None:
            gl_ctx.free()
            event["gl_freed"] = True
    except Exception:
        pass

    if event["switched"] and prev_addr and egl is not None:
        try:
            display = display_getter() if display_getter is not None else _egl_display()
            egl.eglMakeCurrent(display, egl.EGL_NO_SURFACE, egl.EGL_NO_SURFACE, prev)
            event["restored"] = True
        except Exception:
            pass

    for name in _RELEASED_ATTRS:
        try:
            if hasattr(rc, name):
                delattr(rc, name)
        except Exception:
            pass
    return event


def _guarded_del(self: Any) -> None:
    gl_ctx = getattr(self, "gl_ctx", None)
    if gl_ctx is None:  # already released
        return
    if not hasattr(gl_ctx, "_context") and _STATE["original_del"] is not None:
        _STATE["original_del"](self)  # non-EGL backend: robosuite's own behaviour
        return
    release_render_context(self)


def install() -> bool:
    """Patch ``robosuite.utils.binding_utils.MjRenderContext.__del__``. Returns installed state."""
    if _STATE["installed"]:
        return True
    try:
        from robosuite.utils import binding_utils  # type: ignore
    except Exception:
        return False
    cls = binding_utils.MjRenderContext
    _STATE["original_del"] = getattr(cls, "__del__", None)
    cls.__del__ = _guarded_del
    _STATE["installed"] = True
    return True


def is_installed() -> bool:
    return bool(_STATE["installed"])


def live_render_contexts() -> list[Any]:
    try:
        from robosuite.utils import binding_utils  # type: ignore
    except Exception:
        return []
    cls = binding_utils.MjRenderContext
    return [obj for obj in gc.get_objects() if isinstance(obj, cls)]


def reap_stale_render_contexts(sim: Any) -> dict[str, Any]:
    """After ``env.reset()``: collect leftover render contexts now and re-make the live one current."""
    live = getattr(sim, "_render_context_offscreen", None)
    before = len(live_render_contexts())
    collected = gc.collect()
    stale = [rc for rc in live_render_contexts() if rc is not live]
    released = 0
    for rc in stale:  # still referenced by a replaced sim — release explicitly, under its own context
        release_render_context(rc)
        released += 1
    try:
        if live is not None:
            live.gl_ctx.make_current()
    except Exception:
        pass
    return {
        "installed": is_installed(),
        "contexts_before_reap": before,
        "gc_collected": int(collected),
        "stale_released": released,
        "contexts_after_reap": len(live_render_contexts()),
        "kept_alive_total": len(_KEEP_ALIVE),
    }
