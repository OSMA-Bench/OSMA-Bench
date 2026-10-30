"""Sweep supervisor: auto-detect EGL framebuffer errors (#31) and OOM kills (#32)
across all running robocasa-eval-* containers.

Runs as a long-lived background process. Polls each container every 30s:
  - EGL detection: scan recent sweep log lines for "framebuffer is not complete"
                   → docker restart $container (the run_LMP.py loop's .done
                     marker resumes from the next pending task on relaunch
                     by the existing launcher).
  - OOM detection: docker inspect → State.OOMKilled.true → wait for host
                   memory ≥ MIN_FREE_GB before clearing the OOM flag (so
                   the launcher can relaunch when memory is available).

Auto-detects which containers exist (any name starting with `robocasa-eval-`
plus `robotics-safety-container`). Logs events to scripts/sweep_supervisor.log.

Usage:
    python3 scripts/sweep_supervisor.py [--min-free-gb 20] [--poll 30]
    nohup python3 scripts/sweep_supervisor.py > /tmp/supervisor.log 2>&1 &
"""
import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent / 'sweep_supervisor.log'


def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log(msg):
    line = f'[{now()}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def docker(*args, capture=True):
    try:
        r = subprocess.run(['docker', *args], capture_output=capture, text=True, timeout=30)
        return r.stdout.strip() if capture else ''
    except subprocess.TimeoutExpired:
        return ''
    except Exception as e:
        log(f'docker error: {e}')
        return ''


def list_eval_containers():
    """Auto-detect all robocasa-eval-* + robotics-safety-container running containers."""
    out = docker('ps', '--format', '{{.Names}}')
    return [n for n in out.split('\n')
            if n.startswith('robocasa-eval-') or n == 'robotics-safety-container']


def has_running_sweep(container):
    out = docker('exec', container, 'bash', '-c',
                 'ps -ef | grep run_LMP | grep -v grep | wc -l')
    try:
        return int(out) > 0
    except ValueError:
        return False


def get_recent_sweep_logs(container, max_files=3):
    """Return list of recent /tmp/eval_*.log file contents (last 200 lines each)."""
    files = docker('exec', container, 'bash', '-c',
                   'ls -t /tmp/eval_*.log /workspace/scripts/__*__/sweep*.log 2>/dev/null | head -{}'.format(max_files))
    contents = []
    for f in files.split('\n'):
        if not f.strip():
            continue
        c = docker('exec', container, 'tail', '-200', f)
        contents.append((f, c))
    return contents


def detect_egl_error(container):
    """Return (True, file) if framebuffer error found in recent logs."""
    for fname, content in get_recent_sweep_logs(container):
        if re.search(r'framebuffer.*not complete', content, re.IGNORECASE):
            return True, fname
        if 'MjRenderContextOffscreen' in content and 'has no attribute' in content:
            return True, fname
    return False, None


def is_oom_killed(container):
    out = docker('inspect', container, '--format', '{{.State.OOMKilled}}')
    return out.strip().lower() == 'true'


def host_free_gb():
    try:
        out = subprocess.check_output(['free', '-g']).decode()
        return int(out.splitlines()[1].split()[3])
    except Exception:
        return 0


def restart_container(container):
    log(f'  → restarting {container}')
    docker('restart', container, capture=False)
    time.sleep(15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--containers', nargs='*', default=None,
                    help='Explicit container list (default: NONE — must opt-in). '
                         'Use space-separated names like robocasa-eval-2 robotics-safety-container.')
    ap.add_argument('--enable-restart', action='store_true',
                    help='Allow auto-restart on EGL/OOM. Default OFF — log-only.')
    ap.add_argument('--min-free-gb', type=int, default=20,
                    help='OOM recovery: wait until host has ≥ this many GB free')
    ap.add_argument('--poll', type=int, default=30, help='Poll interval (sec)')
    args = ap.parse_args()

    if not args.containers:
        log('No containers specified — supervisor exiting. Pass --containers explicitly.')
        log('Example: python3 sweep_supervisor.py --containers robocasa-eval-2 robotics-safety-container --enable-restart')
        return
    mode = 'AUTO-RESTART' if args.enable_restart else 'LOG-ONLY (dry-run)'
    log(f'Sweep supervisor started — poll={args.poll}s, min_free_gb={args.min_free_gb}, mode={mode}')
    log(f'Watching: {args.containers}')

    state = {}  # container → {'last_egl_t': float, 'last_oom_t': float, 'restarts': int}

    while True:
        try:
            for c in args.containers:
                state.setdefault(c, {'last_egl_t': 0, 'last_oom_t': 0, 'restarts': 0})

                # 1. OOM check (require active sweep to be dead AND memory OK)
                if is_oom_killed(c):
                    if state[c]['last_oom_t'] == 0:
                        log(f'{c}: OOMKilled detected')
                        state[c]['last_oom_t'] = time.time()
                    if has_running_sweep(c):
                        log(f'{c}: has active sweep, deferring OOM action')
                        continue
                    free = host_free_gb()
                    if free < args.min_free_gb:
                        log(f'{c}: waiting for memory (host free={free}GB, need {args.min_free_gb}GB)')
                    else:
                        if args.enable_restart:
                            log(f'{c}: memory ok ({free}GB) — restart enabled, clearing OOM')
                            restart_container(c)
                            state[c]['last_oom_t'] = 0
                            state[c]['restarts'] += 1
                        else:
                            log(f'{c}: memory ok ({free}GB) — would restart (use --enable-restart)')
                    continue

                # 2. EGL error check — only if no active sweep (so we don't kill
                #    a healthy mid-task process due to historical errors).
                if not has_running_sweep(c):
                    egl, fname = detect_egl_error(c)
                    if egl and time.time() - state[c]['last_egl_t'] > 300:
                        if args.enable_restart:
                            log(f'{c}: EGL error in {fname} — restarting')
                            restart_container(c)
                            state[c]['restarts'] += 1
                        else:
                            log(f'{c}: EGL error in {fname} — would restart (use --enable-restart)')
                        state[c]['last_egl_t'] = time.time()

            time.sleep(args.poll)
        except KeyboardInterrupt:
            log('Supervisor stopped by user')
            break
        except Exception as e:
            log(f'Supervisor error: {e}')
            time.sleep(args.poll)


if __name__ == '__main__':
    main()
