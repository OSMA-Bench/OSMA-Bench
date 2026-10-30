"""render_context_guard: a stale render context must free its GL objects under its OWN EGL context."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import render_context_guard as guard  # noqa: E402


class _Ctx:
    def __init__(self, address: int):
        self.address = address


class _FakeEGL:
    EGL_NO_SURFACE = 0

    def __init__(self, current: int):
        self.current = _Ctx(current)
        self.calls: list[tuple[str, int]] = []

    def eglGetCurrentContext(self):
        return self.current

    def eglMakeCurrent(self, display, draw, read, ctx):
        self.calls.append(("make_current", guard._address(ctx)))
        self.current = ctx
        return 1


class _FakeGL:
    def __init__(self, egl: _FakeEGL, address: int, *, fail_make_current: bool = False):
        self.egl = egl
        self._context = _Ctx(address)
        self.freed = False
        self.fail_make_current = fail_make_current

    def make_current(self):
        if self.fail_make_current:
            raise RuntimeError("eglMakeCurrent failed")
        self.egl.eglMakeCurrent(None, 0, 0, self._context)

    def free(self):
        if self._context is not None and guard._address(self.egl.current) == self._context.address:
            self.egl.current = _Ctx(0)
        self._context = None
        self.freed = True


class _FakeCon:
    def __init__(self, egl: _FakeEGL):
        self.egl = egl
        self.freed_under: int | None = None

    def free(self):
        self.freed_under = guard._address(self.egl.eglGetCurrentContext())


class _RC:
    def __init__(self, egl: _FakeEGL, address: int, **gl_kwargs):
        self.gl_ctx = _FakeGL(egl, address, **gl_kwargs)
        self.con = _FakeCon(egl)
        self.scn = self.cam = self.vopt = self.pert = object()


class ReleaseRenderContextTests(unittest.TestCase):
    def test_stale_context_is_freed_under_its_own_context_then_live_context_restored(self):
        egl = _FakeEGL(current=0x2)  # live context #2 is current
        rc = _RC(egl, address=0x1)  # stale context #1
        keep: list = []
        event = guard.release_render_context(rc, egl=egl, display_getter=lambda: None, keep_alive=keep)
        self.assertTrue(event["mjr_freed"])
        self.assertTrue(event["switched"] and event["gl_freed"] and event["restored"])
        self.assertFalse(event["kept_alive"])
        self.assertEqual(guard._address(egl.current), 0x2, "live context must be current again")
        self.assertEqual(keep, [])
        self.assertFalse(hasattr(rc, "con") or hasattr(rc, "gl_ctx"))

    def test_mjr_free_happens_under_own_context(self):
        egl = _FakeEGL(current=0x2)
        rc = _RC(egl, address=0x1)
        con = rc.con
        guard.release_render_context(rc, egl=egl, display_getter=lambda: None, keep_alive=[])
        self.assertEqual(con.freed_under, 0x1)

    def test_live_context_itself_needs_no_switch(self):
        egl = _FakeEGL(current=0x7)
        rc = _RC(egl, address=0x7)
        con = rc.con
        event = guard.release_render_context(rc, egl=egl, display_getter=lambda: None, keep_alive=[])
        self.assertFalse(event["switched"] or event["restored"])
        self.assertTrue(event["mjr_freed"] and event["gl_freed"])
        self.assertEqual(con.freed_under, 0x7)
        self.assertEqual(guard._address(egl.current), 0, "own context destroyed → no context current")

    def test_unswitchable_context_is_kept_alive_not_freed_under_foreign_context(self):
        egl = _FakeEGL(current=0x2)
        rc = _RC(egl, address=0x1, fail_make_current=True)
        con = rc.con
        keep: list = []
        event = guard.release_render_context(rc, egl=egl, display_getter=lambda: None, keep_alive=keep)
        self.assertIsNone(con.freed_under, "must not mjr_freeContext under the live context")
        self.assertTrue(event["kept_alive"] and event["gl_freed"])
        self.assertIs(keep[0], con)
        self.assertEqual(guard._address(egl.current), 0x2)

    def test_release_is_idempotent_and_never_raises(self):
        egl = _FakeEGL(current=0x2)
        rc = _RC(egl, address=0x1)
        guard.release_render_context(rc, egl=egl, display_getter=lambda: None, keep_alive=[])
        event = guard.release_render_context(rc, egl=egl, display_getter=lambda: None, keep_alive=[])
        self.assertFalse(any(event.values()))

    def test_guarded_del_falls_back_for_non_egl_backends(self):
        calls = []
        guard._STATE["original_del"] = lambda self: calls.append("orig")

        class _GLFW:  # no `_context` attribute
            pass

        class _RCother:
            gl_ctx = _GLFW()

        try:
            guard._guarded_del(_RCother())
        finally:
            guard._STATE["original_del"] = None
        self.assertEqual(calls, ["orig"])


class InstallAndReapTests(unittest.TestCase):
    def test_install_is_idempotent_and_reports_robosuite_availability(self):
        first = guard.install()
        second = guard.install()
        self.assertEqual(first, second)
        try:
            from robosuite.utils import binding_utils  # type: ignore
        except Exception:
            self.assertFalse(first)
        else:
            self.assertTrue(first)
            self.assertIs(binding_utils.MjRenderContext.__del__, guard._guarded_del)

    def test_reap_reports_counts_and_remakes_live_context_current(self):
        egl = _FakeEGL(current=0x2)
        live = _RC(egl, address=0x2)
        egl.current = _Ctx(0)  # e.g. a stale __del__ left no context current

        class _Sim:
            _render_context_offscreen = live

        report = guard.reap_stale_render_contexts(_Sim())
        for key in ("installed", "contexts_before_reap", "gc_collected", "stale_released",
                    "contexts_after_reap", "kept_alive_total"):
            self.assertIn(key, report)
        self.assertEqual(guard._address(egl.current), 0x2)


if __name__ == "__main__":
    unittest.main()
