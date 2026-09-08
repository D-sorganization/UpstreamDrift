"""Session conftest: import MuJoCo early to avoid Windows DLL collection crashes."""

import contextlib

with contextlib.suppress(ImportError):
    import mujoco  # noqa: F401

# Import MuJoCo early to avoid Windows DLL initialization conflicts (Access Violation)
# that occur when MuJoCo is loaded during pytest collection with certain plugins.


# ---------------------------------------------------------------------------
# Hang forensics for the Green-Suite unit gate (PR #8976): the gate hangs at
# ~95% and is cancelled at 25 min. Four hung runs produced no diagnostics:
# pytest-timeout (thread method) and per-test/session faulthandler timers are
# either starved, cancelled around every item, or - the round-5 lesson -
# their output lands in pytest's fd-level global capture instead of the job
# log (os.dup(2) at conftest import copies the capture file, not the real
# stderr).
#
# This variant is immune to both problems:
#   * the REAL stderr fd is dup'd inside CaptureManager.global_and_fixture_
#     disabled(), so dumps bypass capture and reach the job log;
#   * faulthandler.register(SIGUSR1) installs a C-level handler needing
#     neither the GIL nor Python signal dispatch, and nothing in-process
#     cancels it;
#   * a detached watchdog subprocess signals its parent pytest process if it
#     lives past 8 minutes (then every 4, max 4 shots) and exits within 10 s
#     of the parent exiting, so healthy runs are unaffected.
# Armed only when UNIT_GATE_QUARANTINE=1 (the unit gate) on POSIX. Remove
# once the hang is diagnosed.
# ---------------------------------------------------------------------------
import os as _os

_WATCHDOG_SRC = "\n".join(
    [
        "import os,sys,time,signal,threading",
        "pid=int(sys.argv[1])",
        "def _pump():",
        "    while True:",
        "        line=sys.stdin.readline()",
        "        if not line: break",
        "        sys.stderr.write('[hang-watchdog] '+line)",
        "        sys.stderr.flush()",
        "t=threading.Thread(target=_pump,daemon=True)",
        "t.start()",
        "deadline=time.monotonic()+480",
        "shots=0",
        "while shots<4:",
        "    time.sleep(10)",
        "    try: os.kill(pid,0)",
        "    except OSError: sys.exit(0)",
        "    if time.monotonic()>=deadline:",
        "        sys.stderr.write('[hang-watchdog] dumping stacks of pid %d'%pid+chr(10))",
        "        sys.stderr.flush()",
        "        try: os.kill(pid,signal.SIGUSR1)",
        "        except OSError: sys.exit(0)",
        "        shots+=1",
        "        deadline=time.monotonic()+240",
    ]
)


def _arm_hang_forensics(config) -> None:
    import signal as _signal

    if not hasattr(_signal, "SIGUSR1"):
        return
    with contextlib.suppress(Exception):
        import faulthandler as _faulthandler
        import subprocess as _subprocess
        import sys as _sys

        capman = config.pluginmanager.getplugin("capturemanager")
        if capman is not None:
            with capman.global_and_fixture_disabled():
                real_fd = _os.dup(2)
        else:
            real_fd = _os.dup(2)
        pipe_r, pipe_w = _os.pipe()
        dump_file = _os.fdopen(pipe_w, "w")
        _faulthandler.register(_signal.SIGUSR1, file=dump_file, all_threads=True)
        dump_dest = _os.fdopen(real_fd, "w")
        _subprocess.Popen(
            [_sys.executable, "-c", _WATCHDOG_SRC, str(_os.getpid())],
            stdin=pipe_r,
            stdout=_subprocess.DEVNULL,
            stderr=dump_dest,
            close_fds=True,
        )
        _os.close(pipe_r)


def pytest_configure(config) -> None:
    if _os.environ.get("UNIT_GATE_QUARANTINE") == "1":
        _arm_hang_forensics(config)


# ---------------------------------------------------------------------------
# Plugin-owned ini keys under ``--strict-config``.
#
# ``pyproject.toml`` declares ``asyncio_mode`` (pytest-asyncio) and
# ``timeout`` / ``timeout_method`` (pytest-timeout). Several CI lanes run
# pytest from purpose-built venvs that install only ``pytest`` itself — the
# hash-locked articulated-authority lane and the rolling MuJoCo/Pinocchio lane
# in ci-optional-stack.yml. Without the plugins those keys are unknown, and
# ``--strict-config`` turns the "Unknown config option" warning into exit
# code 4 before a single test is collected. Registering the keys here only
# when their owning plugin is absent keeps the shared configuration valid
# everywhere; when a plugin is installed it registers its own options and
# this shim stays out of the way.
# ---------------------------------------------------------------------------
# Keyed by the plugin's registered (entry-point) name, which is also what
# ``-p no:<name>`` blocks. Entry-point plugins register before initial
# conftests load, so the plugin manager already knows whether each is active.
_PLUGIN_INI_KEYS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "asyncio": (("asyncio_mode", "string", "pytest-asyncio mode (plugin not active)"),),
    "timeout": (
        ("timeout", "string", "pytest-timeout seconds (plugin not active)"),
        ("timeout_method", "string", "pytest-timeout method (plugin not active)"),
    ),
}


def _missing_plugin_ini_keys(pluginmanager) -> tuple[tuple[str, str, str], ...]:
    """Return ini keys whose owning plugin is neither installed nor active."""
    keys: list[tuple[str, str, str]] = []
    for plugin, plugin_keys in _PLUGIN_INI_KEYS.items():
        if not pluginmanager.has_plugin(plugin):
            keys.extend(plugin_keys)
    return tuple(keys)


def pytest_addoption(parser, pluginmanager) -> None:
    for name, kind, help_text in _missing_plugin_ini_keys(pluginmanager):
        parser.addini(name, help_text, type=kind)
