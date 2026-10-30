"""Print the EGL device ids that can create a headless MuJoCo context.

    export ROBOCASA_EGL_DEVICE=$(python scripts/resolve_egl.py 0 1 2 3)

Prints nothing (and exits 1) when no device works, so a caller can test the
result and stop rather than launching a run that cannot render.
"""
import contextlib
import sys

# Importing robocasa prints banner/warning lines on stdout, which would end up
# inside the command substitution. Send anything the import emits to stderr so
# stdout carries only the device ids.
with contextlib.redirect_stdout(sys.stderr):
    from robocasa.utils.egl_device import resolve_egl_devices


def main():
    requested = [int(a) for a in sys.argv[1:]] or None
    with contextlib.redirect_stdout(sys.stderr):
        devices = resolve_egl_devices(requested, verbose=False)
    if not devices:
        return 1
    print(" ".join(str(d) for d in devices))
    return 0


if __name__ == "__main__":
    sys.exit(main())
