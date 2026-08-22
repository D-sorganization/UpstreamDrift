"""Session conftest: import MuJoCo early to avoid Windows DLL collection crashes."""

import contextlib

with contextlib.suppress(ImportError):
    import mujoco  # noqa: F401

# Import MuJoCo early to avoid Windows DLL initialization conflicts (Access Violation)
# that occur when MuJoCo is loaded during pytest collection with certain plugins.


# ---------------------------------------------------------------------------
# Hang forensics for the Green-Suite unit gate (PR #8976, issue: gate hangs at
# ~95% and times out at 25 min). pytest-timeout (thread method), a per-test
# faulthandler_timeout, and a session-level faulthandler.dump_traceback_later
# were all silent across three hung runs, so the stall sits outside any test
# phase and in-process timers are being cancelled (pytest-timeout resets the
# shared faulthandler timer around every item).
#
# This variant cannot be cancelled from inside the process:
#   * faulthandler.register(SIGUSR1) installs a C-level handler that writes
#     every thread's stack straight to a dup of the REAL stderr fd (the job
#     log), needing neither the GIL nor Python signal dispatch;
#   * a detached watchdog subprocess signals its parent pytest process if it
#     lives past 8 minutes (then every 4, max 4 shots), and exits within 10 s
#     of the parent exiting, so healthy runs are unaffected.
# Armed only when UNIT_GATE_QUARANTINE=1 (the unit gate) on POSIX. Remove
# once the hang is diagnosed.
# ---------------------------------------------------------------------------
import os as _os

if _os.environ.get("UNIT_GATE_QUARANTINE") == "1":
    import signal as _signal

    if hasattr(_signal, "SIGUSR1"):
        with contextlib.suppress(Exception):
            import faulthandler as _faulthandler
            import subprocess as _subprocess
            import sys as _sys

            _hang_dump_file = _os.fdopen(_os.dup(2), "w")
            _faulthandler.register(
                _signal.SIGUSR1, file=_hang_dump_file, all_threads=True
            )
            _WATCHDOG_SRC = (
                "import os,sys,time,signal\n"
                "pid=int(sys.argv[1])\n"
                "deadline=time.monotonic()+480\n"
                "shots=0\n"
                "while shots<4:\n"
                "    time.sleep(10)\n"
                "    try: os.kill(pid,0)\n"
                "    except OSError: sys.exit(0)\n"
                "    if time.monotonic()>=deadline:\n"
                "        sys.stderr.write('[hang-watchdog] dumping stacks of pid %d'%pid+chr(10))\n"
                "        sys.stderr.flush()\n"
                "        try: os.kill(pid,signal.SIGUSR1)\n"
                "        except OSError: sys.exit(0)\n"
                "        shots+=1\n"
                "        deadline=time.monotonic()+240\n"
            )
            _subprocess.Popen(
                [_sys.executable, "-c", _WATCHDOG_SRC, str(_os.getpid())],
                stdin=_subprocess.DEVNULL,
                stdout=_subprocess.DEVNULL,
                stderr=_hang_dump_file,
                close_fds=True,
            )
