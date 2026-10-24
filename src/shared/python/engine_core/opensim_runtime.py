"""Single production route to the OpenSim SDK (#12065).

The src-side counterpart of :func:`src.shared.python.data_io.native_dll_guard.
import_ezc3d`.  On Windows the OpenSim SDK and the Python ``ezc3d`` extension
ship ABI-incompatible ``ezc3d.dll`` builds, so whichever loads first owns the
DLL name and the other fails with ``DLL load failed``.  Every production
OpenSim import goes through :func:`import_opensim`, which keeps the semantics
of a plain ``import opensim`` statement:

* it imports through ``builtins.__import__`` exactly as the statement does, so
  ``sys.modules`` entries and patched importers behave unchanged;
* absence stays a ``ModuleNotFoundError`` and any unrelated failure the
  original ``ImportError``, so optional-dependency handling at call sites
  (graceful unavailability or a clear error) does not change;
* only a DLL load failure while a foreign ``ezc3d`` runtime is hosted becomes a
  typed, diagnosable ``NativeDllConflictError`` (still an ``ImportError``, from
  :mod:`src.shared.python.data_io.native_dll_guard`) naming the
  process-boundary remedy.

It never suppresses or retries an import.  Call it where the old ``import``
statement was, so lazy imports stay lazy.  This module imports only the
standard library; the DLL guard loads on the failure path alone, so routing an
import through here adds no import-time weight.
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from enum import Enum
import logging
import os
from pathlib import Path
import sys
from typing import Any

__all__ = [
    "CASADI_IPOPT_PLUGIN",
    "OpenSimProbe",
    "OpenSimState",
    "ensure_casadi_plugin_path",
    "import_opensim",
    "probe_opensim",
]

logger = logging.getLogger(__name__)

OPENSIM_PACKAGE = "opensim"
_DLL_FAILURE_MARKER = "DLL load failed"
CASADI_IPOPT_PLUGIN = "casadi_nlpsol_ipopt.dll"
CASADIPATH_ENV = "CASADIPATH"


def ensure_casadi_plugin_path(
    package_dir: Path | str,
    *,
    environ: MutableMapping[str, str] | None = None,
    platform: str | None = None,
) -> str | None:
    """Point CasADi at the OpenSim wheel's plugins on Windows (OSL-7, #13044).

    The OpenSim pip wheel bundles CasADi's solver plugins (``ipopt``) next to
    the package, but on Windows CasADi only searches ``CASADIPATH``/``PATH``,
    so ``MocoCasADiSolver`` fails with ``Plugin 'ipopt' is not found``.  This
    sets ``CASADIPATH`` to ``package_dir`` when it is unset and the directory
    holds the ipopt plugin.

    Returns the path set, or ``None`` when nothing changed: not Windows, the
    variable is already set (never overridden), or the plugin is absent.
    """
    env = os.environ if environ is None else environ
    if (sys.platform if platform is None else platform) != "win32":
        return None
    if env.get(CASADIPATH_ENV):
        return None
    directory = Path(package_dir)
    if not (directory / CASADI_IPOPT_PLUGIN).is_file():
        return None
    env[CASADIPATH_ENV] = str(directory)
    return str(directory)


def _configure_casadi_for_loaded_opensim() -> None:
    """Apply :func:`ensure_casadi_plugin_path` to the imported ``opensim``."""
    location = getattr(sys.modules.get(OPENSIM_PACKAGE), "__file__", None)
    if isinstance(location, str):
        ensure_casadi_plugin_path(Path(location).parent)


def _statement_import(name: str) -> Any:
    """Import ``name`` the way an ``import`` statement does and return it."""
    __import__(name)
    return sys.modules[name]


def _require_opensim_name(module: str) -> None:
    if module != OPENSIM_PACKAGE and not module.startswith(f"{OPENSIM_PACKAGE}."):
        raise ValueError(
            f"import_opensim only imports the {OPENSIM_PACKAGE!r} package or one "
            f"of its submodules, got {module!r}"
        )


def _ezc3d_dll_owner() -> str | None:
    """Return the path of the ``ezc3d.dll`` this process already loaded, if any."""
    from src.shared.python.data_io.native_dll_guard import (  # noqa: PLC0415 - failure path only
        EZC3D_DLL,
        loaded_dll_path,
    )

    return loaded_dll_path(EZC3D_DLL)


def _conflict_error(module: str, exc: ImportError, owner: str | None) -> ImportError:
    from src.shared.python.data_io.native_dll_guard import (  # noqa: PLC0415 - failure path only
        EZC3D_DLL,
        NativeDllConflictError,
    )

    hosted = (
        " (the Python ezc3d extension is imported)" if "ezc3d" in sys.modules else ""
    )
    message = (
        f"{module} cannot load in this process: {str(exc).rstrip('.')}. "
        f"{EZC3D_DLL} is already loaded from {owner or 'an unknown location'}"
        f"{hosted}. The native OpenSim and Python ezc3d builds are not "
        "ABI-compatible and cannot share a process. Read C3D in a separate "
        "process (scripts/diagnostics/frozen_capture_trc.py) and feed the "
        "source-bound TRC to a process that never imports ezc3d."
    )
    logger.error(message)
    return NativeDllConflictError(message)


def import_opensim(
    module: str = OPENSIM_PACKAGE,
    *,
    importer: Callable[[str], Any] = _statement_import,
) -> Any:
    """Import and return ``opensim`` (or an ``opensim.*`` submodule).

    Preconditions: ``module`` is ``"opensim"`` or starts with ``"opensim."``.
    Postcondition: on return the module is loaded; nothing is retried.  The
    result is typed ``Any`` because OpenSim ships no type information, so
    callers keep the static typing a bare ``import opensim`` gave them.

    Raises:
        ValueError: ``module`` is not part of the OpenSim package.
        ModuleNotFoundError: OpenSim is not installed (unchanged).
        NativeDllConflictError: the import failed with a DLL load error while
            the Python ``ezc3d`` extension (or another ``ezc3d.dll``) is loaded
            in this process.
        ImportError: any other import failure, unchanged.
    """
    _require_opensim_name(module)
    try:
        imported = importer(module)
        _configure_casadi_for_loaded_opensim()
        return imported
    except ImportError as exc:
        if isinstance(exc, ModuleNotFoundError):
            raise
        if _DLL_FAILURE_MARKER not in str(exc):
            raise
        owner = _ezc3d_dll_owner()
        if owner is None and "ezc3d" not in sys.modules:
            raise
        raise _conflict_error(module, exc, owner) from exc


class OpenSimState(Enum):
    """Outcome of probing the OpenSim SDK (#12883)."""

    INSTALLED = "installed"
    IMPORT_FAILED = "import_failed"  # present, but cannot load in this process
    MISSING = "missing"  # not installed


@dataclass(frozen=True)
class OpenSimProbe:
    """Structured probe result; ``reason`` is the real import message.

    Invariant: ``reason`` is ``None`` exactly when ``state`` is INSTALLED.
    ``dll_conflict`` is True when the failure is the ezc3d/OpenSim DLL clash.
    """

    state: OpenSimState
    reason: str | None = None
    dll_conflict: bool = False

    def __post_init__(self) -> None:
        if (self.state is OpenSimState.INSTALLED) != (self.reason is None):
            raise ValueError("reason must be set iff state is not INSTALLED")

    @property
    def available(self) -> bool:
        return self.state is OpenSimState.INSTALLED


def probe_opensim(
    importer: Callable[[], Any] = import_opensim,
) -> OpenSimProbe:
    """Probe OpenSim without hiding why it failed.

    Never raises ``ImportError``: absence is MISSING, a DLL clash or any other
    load failure is IMPORT_FAILED carrying the real message (not "not installed").
    """
    from src.shared.python.data_io.native_dll_guard import (  # noqa: PLC0415 - failure path only
        NativeDllConflictError,
    )

    try:
        importer()
    except ModuleNotFoundError as exc:
        return OpenSimProbe(OpenSimState.MISSING, str(exc))
    except ImportError as exc:
        return OpenSimProbe(
            OpenSimState.IMPORT_FAILED,
            str(exc) or type(exc).__name__,
            dll_conflict=isinstance(exc, NativeDllConflictError),
        )
    return OpenSimProbe(OpenSimState.INSTALLED)
