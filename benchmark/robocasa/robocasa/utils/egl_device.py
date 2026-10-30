"""Pick an EGL device that can actually create a headless GL context.

MuJoCo's EGL backend addresses GPUs by their position in the EGL device
enumeration, which is not the CUDA ordinal. On hosts where the two disagree the
default (device 0) fails with EGL_BAD_DISPLAY even though every GPU is healthy,
and every render in the run dies. This probes candidates and reports one that
works.

Each probe runs in a subprocess: a failed eglInitialize leaves the calling
process with a half-built context that MuJoCo will not retry, so the check
cannot be done in-process.

Set ROBOCASA_EGL_DEVICE to skip probing entirely.
"""
import os
import subprocess
import sys

_PROBE = (
    "import mujoco;"
    "m=mujoco.MjModel.from_xml_string("
    "'<mujoco><worldbody><geom type=\"box\" size=\"1 1 1\"/></worldbody></mujoco>');"
    "d=mujoco.MjData(m);"
    "r=mujoco.Renderer(m,64,64);r.update_scene(d);r.render();"
    "print('EGL_OK')"
)

_ENV_OVERRIDE = "ROBOCASA_EGL_DEVICE"
_cache = {}


def probe_egl_device(device_id, timeout=120):
    """True if a headless MuJoCo context can be created on this EGL device."""
    if device_id in _cache:
        return _cache[device_id]
    env = dict(os.environ, MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID=str(device_id))
    try:
        res = subprocess.run([sys.executable, "-c", _PROBE], env=env,
                             capture_output=True, text=True, timeout=timeout)
        ok = res.returncode == 0 and "EGL_OK" in res.stdout
    except (subprocess.TimeoutExpired, OSError):
        ok = False
    _cache[device_id] = ok
    return ok


def resolve_egl_devices(requested=None, max_devices=16, verbose=True):
    """Return the subset of `requested` that works, else a scanned fallback.

    `requested` is a list of EGL device ids (what --gpu_id / --gpu_ids mean for
    rendering). Returns a non-empty list, or [] when no device works at all --
    in which case rendering was going to fail regardless and the caller should
    surface that rather than silently continue.
    """
    override = os.environ.get(_ENV_OVERRIDE)
    if override is not None and override.strip() != "":
        ids = [int(x) for x in override.replace(",", " ").split()]
        if verbose:
            print(f"[egl] using {_ENV_OVERRIDE}={override} (probe skipped)")
        return ids

    requested = list(requested) if requested else [0]
    working = [d for d in requested if probe_egl_device(d)]
    if working:
        if verbose and len(working) != len(requested):
            dead = [d for d in requested if d not in working]
            print(f"[egl] devices {dead} cannot create a context; using {working}")
        return working

    if verbose:
        print(f"[egl] requested device(s) {requested} cannot create a context; "
              f"scanning 0..{max_devices - 1}")
    for d in range(max_devices):
        if d in requested:
            continue
        if probe_egl_device(d):
            if verbose:
                print(f"[egl] falling back to EGL device {d} "
                      f"(export {_ENV_OVERRIDE}={d} to skip this probe)")
            return [d]

    if verbose:
        print(f"[egl] no working EGL device found in 0..{max_devices - 1}")
    return []


def resolve_egl_device(requested=None, **kw):
    """Single-device convenience wrapper; returns the id or None."""
    got = resolve_egl_devices([requested] if requested is not None else None, **kw)
    return got[0] if got else None
